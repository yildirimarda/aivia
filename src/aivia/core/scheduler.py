"""Cron scheduler that enqueues jobs from schedule entries.

The scheduler reads a :class:`~aivia.core.config.schedule.ScheduleConfig`,
computes the next fire time for every enabled entry in its configured timezone
(with DST correctness), and enqueues a :class:`~aivia.core.models.job.Job`
into the :class:`~aivia.core.jobs.JobQueue` when a fire time is due.

Overlapping runs are prevented by acquiring a per-profile run lock (and a
global parallel-run slot) before enqueuing. A busy lock defers the entry to
the next tick, so a scheduled run never overlaps an in-flight run of the same
profile or another run when the global parallel limit is reached.

Cron validation happens at configuration load time
(:func:`~aivia.core.config.schedule.validate_cron`); an entry whose cron
expression failed validation is never present in a parsed
:class:`ScheduleConfig`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from aivia.core.config.schedule import ScheduleConfig, ScheduleEntryConfig
from aivia.core.jobs import JobQueue
from aivia.core.locks import RunLock, RunLockManager
from aivia.core.models import Job

__all__ = [
    "CronScheduler",
    "ScheduledJob",
    "next_fire_time",
]

_WEEKDAY_NAMES = {
    "sun": 0,
    "mon": 1,
    "tue": 2,
    "wed": 3,
    "thu": 4,
    "fri": 5,
    "sat": 6,
}

_MONTH_NAMES = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_MAX_HORIZON_DAYS = 365 * 8


def _cron_number(token: str, names: Mapping[str, int] | None) -> int:
    key = token.strip().lower()
    if names is not None and key in names:
        return names[key]
    return int(key, 10)


def _expand_field(
    field: str,
    *,
    low: int,
    high: int,
    names: Mapping[str, int] | None = None,
) -> set[int]:
    values: set[int] = set()
    for part in field.split(","):
        if not part:
            raise ValueError(f"empty cron field part: {field!r}")
        base, separator, step_text = part.partition("/")
        step = int(step_text) if separator else 1
        if step <= 0:
            raise ValueError(f"cron step must be positive: {step_text}")
        if base == "*":
            start, end = low, high
        elif "-" in base:
            start_text, end_text = base.split("-", 1)
            start = _cron_number(start_text, names)
            end = _cron_number(end_text, names)
        else:
            start = _cron_number(base, names)
            end = high if separator else start
        if start > end:
            raise ValueError(f"invalid cron range: {base}")
        if not low <= start <= high or not low <= end <= high:
            raise ValueError(f"cron field out of range {low}-{high}: {part}")
        for value in range(start, end + 1, step):
            values.add(value)
    return values


def _parse_cron(
    expression: str,
) -> tuple[set[int], set[int], set[int], set[int], set[int], bool, bool]:
    """Parse a validated cron expression into match sets.

    Returns ``(minutes, hours, doms, months, python_dows, dom_restricted,
    dow_restricted)``. Weekday values are normalised to Python's convention
    (Monday is 0, Sunday is 6); cron treats both 0 and 7 as Sunday.
    """
    fields = expression.split()
    if len(fields) == 6:
        fields = fields[1:]
    if len(fields) != 5:
        raise ValueError(f"cron expression must have 5 fields: {expression!r}")
    minute, hour, day, month, weekday = fields
    minutes = _expand_field(minute, low=0, high=59)
    hours = _expand_field(hour, low=0, high=23)
    doms = _expand_field(day, low=1, high=31)
    months = _expand_field(month, low=1, high=12, names=_MONTH_NAMES)
    cron_dows = _expand_field(weekday, low=0, high=7, names=_WEEKDAY_NAMES)
    python_dows = {(_dow - 1) % 7 for _dow in cron_dows}
    dom_restricted = doms != set(range(1, 32))
    dow_restricted = cron_dows != set(range(0, 8))
    return minutes, hours, doms, months, python_dows, dom_restricted, dow_restricted


def _day_matches(
    day: int,
    weekday: int,
    doms: set[int],
    dows: set[int],
    dom_restricted: bool,
    dow_restricted: bool,
) -> bool:
    dom_ok = (not dom_restricted) or (day in doms)
    dow_ok = (not dow_restricted) or (weekday in dows)
    if dom_restricted and dow_restricted:
        return dom_ok or dow_ok
    return dom_ok and dow_ok


def _advance(
    candidate: datetime,
    fields: tuple[set[int], set[int], set[int], set[int], set[int], bool, bool],
) -> datetime:
    minutes, hours, doms, months, dows, dom_restricted, dow_restricted = fields
    if candidate.month not in months:
        next_month = next((m for m in sorted(months) if m > candidate.month), None)
        if next_month is not None:
            return candidate.replace(month=next_month, day=1, hour=0, minute=0)
        return candidate.replace(
            year=candidate.year + 1,
            month=min(months),
            day=1,
            hour=0,
            minute=0,
        )
    if not _day_matches(
        candidate.day,
        candidate.weekday(),
        doms,
        dows,
        dom_restricted,
        dow_restricted,
    ):
        return candidate.replace(hour=0, minute=0) + timedelta(days=1)
    if candidate.hour not in hours:
        next_hour = next((h for h in sorted(hours) if h > candidate.hour), None)
        if next_hour is not None:
            return candidate.replace(hour=next_hour, minute=0)
        return candidate.replace(hour=0, minute=0) + timedelta(days=1)
    if candidate.minute not in minutes:
        next_minute = next((m for m in sorted(minutes) if m > candidate.minute), None)
        if next_minute is not None:
            return candidate.replace(minute=next_minute)
        return candidate.replace(hour=0, minute=0) + timedelta(hours=1)
    return candidate


def _matches(
    candidate: datetime,
    fields: tuple[set[int], set[int], set[int], set[int], set[int], bool, bool],
) -> bool:
    minutes, hours, doms, months, dows, dom_restricted, dow_restricted = fields
    if candidate.month not in months:
        return False
    if not _day_matches(
        candidate.day,
        candidate.weekday(),
        doms,
        dows,
        dom_restricted,
        dow_restricted,
    ):
        return False
    if candidate.hour not in hours:
        return False
    return candidate.minute in minutes


def next_fire_time(expression: str, timezone: str, after: datetime) -> datetime | None:
    """Return the next UTC fire time strictly after *after*.

    *expression* is a validated five-field cron. *timezone* is an IANA zone
    name such as ``Europe/Berlin``; fire times are computed in local time and
    returned as UTC so DST shifts move the UTC instant correctly. Returns
    ``None`` when no fire time exists within the eight-year horizon (for
    example ``30 2 * * *`` outside leap years).
    """
    tz = ZoneInfo(timezone)
    fields = _parse_cron(expression)
    local = after.astimezone(tz)
    candidate = local.replace(second=0, microsecond=0) + timedelta(minutes=1)
    limit = candidate + timedelta(days=_MAX_HORIZON_DAYS)
    while candidate <= limit:
        candidate = candidate.astimezone(UTC).astimezone(tz)
        if _matches(candidate, fields):
            return candidate.astimezone(UTC)
        previous = candidate
        candidate = _advance(candidate, fields)
        if candidate <= previous:
            candidate = previous + timedelta(minutes=1)
    return None


@dataclass(frozen=True, slots=True)
class ScheduledJob:
    """A job enqueued by the scheduler plus the run lock it holds."""

    job: Job
    profile: str
    schedule_name: str
    lock: RunLock
    _on_release: Callable[[], None] = field(repr=False, compare=False)

    def release(self) -> None:
        """Release the run lock held for this scheduled job."""
        self._on_release()


class CronScheduler:
    """Enqueue cron-scheduled jobs, honouring timezones and run locks."""

    def __init__(
        self,
        schedule: ScheduleConfig,
        queue: JobQueue,
        locks: RunLockManager,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.schedule = schedule
        self.queue = queue
        self.locks = locks
        self._clock = clock or (lambda: datetime.now(UTC))
        self._pending: dict[str, datetime | None] = {}
        self._held: dict[str, RunLock] = {}

    @property
    def clock(self) -> Callable[[], datetime]:
        return self._clock

    def now(self) -> datetime:
        return self._clock()

    def next_fire_time(
        self, entry: ScheduleEntryConfig, after: datetime | None = None
    ) -> datetime | None:
        """Return the next UTC fire time for *entry* strictly after *after*."""
        if not entry.enabled or not entry.cron:
            return None
        wanted = self.now() if after is None else after
        return next_fire_time(entry.cron, entry.timezone or "UTC", wanted)

    def tick(self, now: datetime | None = None) -> list[ScheduledJob]:
        """Enqueue jobs for every due entry that can acquire a run lock.

        Each entry's next fire time is remembered in ``_pending``. The first
        tick computes the first fire strictly after *now*; later ticks enqueue
        it once *now* reaches it and then advance to the following fire time.
        A profile whose run lock is already held is skipped (and its pending
        fire is kept) so scheduled runs never overlap.
        """
        wanted = self.now() if now is None else now
        enqueued: list[ScheduledJob] = []
        for name, entry in self.schedule.schedules.items():
            if not self.schedule.enabled or not entry.enabled or not entry.cron:
                continue
            pending = self._pending.get(name)
            if pending is None:
                pending = next_fire_time(entry.cron, entry.timezone or "UTC", wanted)
            if pending is None:
                continue
            if pending > wanted:
                self._pending[name] = pending
                continue
            profile = entry.profile or "default"
            if profile in self._held:
                continue
            lock = self.locks.try_acquire(profile, run_id=name)
            if lock is None:
                continue
            job = self.queue.enqueue(
                kind="schedule",
                payload={
                    "schedule": name,
                    "profile": profile,
                    "cron": entry.cron,
                    "timezone": entry.timezone,
                },
                not_before=pending,
                idempotency_key=f"schedule:{name}:{pending.isoformat()}",
            )
            self._held[profile] = lock
            self._pending[name] = next_fire_time(
                entry.cron, entry.timezone or "UTC", pending
            )

            def _release(profile: str = profile, lock: RunLock = lock) -> None:
                self._held.pop(profile, None)
                lock.release()

            enqueued.append(
                ScheduledJob(
                    job=job,
                    profile=profile,
                    schedule_name=name,
                    lock=lock,
                    _on_release=_release,
                )
            )
        return enqueued

    def release(self, profile: str) -> None:
        """Release the run lock held for *profile*, if any."""
        lock = self._held.pop(profile, None)
        if lock is not None:
            lock.release()

    def held_profiles(self) -> tuple[str, ...]:
        """Return the profiles that currently hold a run lock."""
        return tuple(self._held)

"""Cron scheduler: timezone/DST-correct next fire times, invalid cron
rejection, and run locking.
"""

from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import Engine

from aivia.core.config.schedule import (
    ScheduleConfig,
    ScheduleEntryConfig,
    is_valid_cron,
    validate_cron,
)
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.errors import ConfigError
from aivia.core.jobs import JobQueue
from aivia.core.locks import RunLockManager
from aivia.core.models import JobState
from aivia.core.scheduler import CronScheduler, ScheduledJob, next_fire_time

BERLIN = ZoneInfo("Europe/Berlin")


class Clock:
    """A mutable clock so tests never depend on wall time."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def engine(tmp_path: Path) -> Generator[Engine, None, None]:
    database = tmp_path / "aivia.db"
    created = create_database_engine(database)
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


@pytest.fixture
def queue(engine: Engine) -> JobQueue:
    return JobQueue(engine)


@pytest.fixture
def locks(tmp_path: Path) -> RunLockManager:
    return RunLockManager(max_parallel=1, lock_dir=tmp_path / "locks")


@pytest.fixture
def schedule() -> ScheduleConfig:
    return ScheduleConfig.from_mapping(
        {
            "timezone": "UTC",
            "schedules": {
                "daily": {"profile": "default", "cron": "0 9 * * *"},
                "berlin": {
                    "profile": "default",
                    "cron": "0 9 * * *",
                    "timezone": "Europe/Berlin",
                },
                "weekday": {"profile": "default", "cron": "0 9 * * 1-5"},
                "disabled": {
                    "profile": "default",
                    "cron": "0 9 * * *",
                    "enabled": False,
                },
            },
        }
    )


@pytest.fixture
def sched(
    schedule: ScheduleConfig, queue: JobQueue, locks: RunLockManager
) -> tuple[CronScheduler, Clock]:
    clock = Clock(datetime(2026, 1, 15, 10, 0, tzinfo=UTC))
    return CronScheduler(schedule, queue, locks, clock=clock), clock


# ---------------------------------------------------------------------------
# next_fire_time: timezone and DST correctness
# ---------------------------------------------------------------------------


def test_next_fire_time_is_utc_and_strictly_after() -> None:
    fire = next_fire_time("0 9 * * *", "UTC", datetime(2026, 1, 15, 10, 0, tzinfo=UTC))
    assert fire == datetime(2026, 1, 16, 9, 0, tzinfo=UTC)


def test_next_fire_time_respects_configured_timezone_winter() -> None:
    fire = next_fire_time(
        "0 9 * * *", "Europe/Berlin", datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    )
    assert fire == datetime(2026, 1, 16, 8, 0, tzinfo=UTC)


def test_dst_shifts_utc_instant_summer() -> None:
    fire = next_fire_time(
        "0 9 * * *", "Europe/Berlin", datetime(2026, 6, 15, 10, 0, tzinfo=UTC)
    )
    assert fire == datetime(2026, 6, 16, 7, 0, tzinfo=UTC)


def test_spring_forward_fires_at_local_nine() -> None:
    fire = next_fire_time(
        "0 9 * * *",
        "Europe/Berlin",
        datetime(2026, 3, 28, 23, 0, tzinfo=UTC),
    )
    assert fire == datetime(2026, 3, 29, 7, 0, tzinfo=UTC)


def test_weekday_cron_skips_weekend() -> None:
    fire = next_fire_time(
        "0 9 * * 1-5",
        "UTC",
        datetime(2026, 1, 16, 10, 0, tzinfo=UTC),
    )
    assert fire == datetime(2026, 1, 19, 9, 0, tzinfo=UTC)


def test_never_matching_cron_returns_none() -> None:
    assert next_fire_time("0 0 30 2 *", "UTC", datetime(2026, 1, 1, tzinfo=UTC)) is None


# ---------------------------------------------------------------------------
# invalid cron rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression",
    [
        "60 * * * *",
        "* 24 * * *",
        "* * 32 * *",
        "* * * 13 *",
        "* * * * 8",
        "*/0 * * * *",
        "5-1 * * * *",
        "1,,2 * * * *",
    ],
)
def test_invalid_cron_expressions_are_rejected(expression: str) -> None:
    with pytest.raises(ConfigError):
        ScheduleEntryConfig.from_mapping({"profile": "default", "cron": expression})
    with pytest.raises(ValueError):
        next_fire_time(expression, "UTC", datetime(2026, 1, 1, tzinfo=UTC))


def test_valid_cron_expressions_are_accepted() -> None:
    assert is_valid_cron("0 9 * * *")
    assert validate_cron("0 9 * * *")
    assert next_fire_time("0 9 * * *", "UTC", datetime(2026, 1, 1, tzinfo=UTC))


# ---------------------------------------------------------------------------
# scheduler tick: enqueues due jobs, skips disabled and locked
# ---------------------------------------------------------------------------


def test_first_tick_does_not_enqueue_a_future_fire(
    sched: tuple[CronScheduler, Clock],
    queue: JobQueue,
) -> None:
    scheduler, _clock = sched
    jobs = scheduler.tick()
    assert jobs == []
    assert queue.depth() == 0


def test_tick_enqueues_due_job_with_not_before(
    sched: tuple[CronScheduler, Clock],
    queue: JobQueue,
) -> None:
    scheduler, clock = sched
    scheduler.tick()  # first fire for "daily" is 2026-01-16 09:00 UTC
    clock.now = datetime(2026, 1, 16, 9, 0, tzinfo=UTC)
    jobs = scheduler.tick()

    assert len(jobs) == 1
    job = jobs[0].job
    assert job.state is JobState.PENDING
    assert job.not_before == datetime(2026, 1, 16, 9, 0, tzinfo=UTC)
    assert job.payload["schedule"] == "daily"
    assert job.payload["profile"] == "default"
    assert queue.get(job.id) == job


def test_no_double_enqueue_at_the_same_time(
    sched: tuple[CronScheduler, Clock],
    queue: JobQueue,
) -> None:
    scheduler, clock = sched
    scheduler.tick()
    clock.now = datetime(2026, 1, 16, 9, 0, tzinfo=UTC)
    scheduler.tick()  # enqueues daily at 09:00, advances pending to 2026-01-17

    jobs = scheduler.tick()  # pending is now in the future
    assert jobs == []
    assert queue.depth() == 1


def test_disabled_entry_is_skipped(
    sched: tuple[CronScheduler, Clock],
    queue: JobQueue,
) -> None:
    scheduler, clock = sched
    scheduler.tick()  # pending set for every enabled entry
    clock.now = datetime(2026, 1, 16, 9, 0, tzinfo=UTC)
    jobs = scheduler.tick()

    # every entry shares the "default" profile, so only the first one wins the lock
    assert len(jobs) == 1
    assert jobs[0].schedule_name == "daily"
    assert queue.depth() == 1


def test_release_frees_the_lock_for_the_next_tick(
    sched: tuple[CronScheduler, Clock],
    queue: JobQueue,
) -> None:
    scheduler, clock = sched
    scheduler.tick()
    clock.now = datetime(2026, 1, 16, 9, 0, tzinfo=UTC)
    jobs = scheduler.tick()
    assert len(jobs) == 1
    assert scheduler.held_profiles() == ("default",)

    jobs[0].release()
    assert scheduler.held_profiles() == ()

    clock.now = datetime(2026, 1, 17, 9, 0, tzinfo=UTC)
    jobs = scheduler.tick()
    assert len(jobs) == 1
    assert jobs[0].job.not_before == datetime(2026, 1, 17, 9, 0, tzinfo=UTC)


def test_global_parallel_limit_blocks_other_profile(
    schedule: ScheduleConfig,
    queue: JobQueue,
    locks: RunLockManager,
    tmp_path: Path,
) -> None:
    two_profiles = ScheduleConfig.from_mapping(
        {
            "schedules": {
                "alpha": {"profile": "alpha", "cron": "0 9 * * *"},
                "beta": {"profile": "beta", "cron": "0 9 * * *"},
            }
        }
    )
    clock = Clock(datetime(2026, 1, 15, 10, 0, tzinfo=UTC))
    scheduler = CronScheduler(two_profiles, queue, locks, clock=clock)
    scheduler.tick()  # pending for both profiles is 2026-01-16 09:00 UTC
    clock.now = datetime(2026, 1, 16, 9, 0, tzinfo=UTC)
    jobs = scheduler.tick()

    assert len(jobs) == 1
    assert jobs[0].schedule_name == "alpha"
    assert scheduler.held_profiles() == ("alpha",)


def test_next_fire_time_uses_entry_timezone_not_section_default(
    sched: tuple[CronScheduler, Clock],
) -> None:
    scheduler, _clock = sched
    berlin = scheduler.schedule.schedules["berlin"]
    fire = scheduler.next_fire_time(berlin)
    assert fire == datetime(2026, 1, 16, 8, 0, tzinfo=UTC)


def test_scheduled_job_release_is_alias_for_lock_release(
    sched: tuple[CronScheduler, Clock],
) -> None:
    scheduler, clock = sched
    scheduler.tick()
    clock.now = datetime(2026, 1, 16, 9, 0, tzinfo=UTC)
    jobs = scheduler.tick()
    assert isinstance(jobs[0], ScheduledJob)
    assert scheduler.held_profiles() == ("default",)
    jobs[0].release()
    assert scheduler.held_profiles() == ()

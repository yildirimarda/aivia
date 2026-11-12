"""Publish accounts and daily posting slots.

Resolves the profile and credential references that apply to a publishing
target identified by ``platform/language``, then lays out the day's posting
slots with bounded jitter and emits deferral notices when a slot cannot be
used and is pushed to a later time.

The module is deterministic: :func:`apply_jitter` and :meth:`SlotPlan.posting_times`
accept an injectable ``rng`` returning values in ``[0, 1)`` so a plan is fully
reproducible, and no provider or network call is ever made.
"""

# config-coverage: allow-file
from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from aivia.core.config.schedule import PublishAccountConfig, PublishConfig
from aivia.core.errors import ConfigError

__all__ = [
    "AccountCredentials",
    "AccountMap",
    "DeferralNotice",
    "PostingSlot",
    "SlotPlan",
    "apply_jitter",
]


@dataclass(frozen=True, slots=True)
class AccountCredentials:
    """Resolved publish credentials for one platform/language target."""

    platform: str
    language: str
    profile: str
    credential_refs: tuple[str, ...]
    enabled: bool = True

    @property
    def key(self) -> str:
        """The ``platform/language`` account key."""
        return f"{self.platform}/{self.language}"


class AccountMap:
    """Map profile + platform + language to publish credentials.

    Built from a :class:`~aivia.core.config.schedule.PublishConfig` so callers
    can look up, for a given platform and language, the profile and credential
    references that apply. Account keys are ``platform/language`` strings as
    accepted by the publish config.
    """

    def __init__(self, publish: PublishConfig) -> None:
        self._publish = publish
        self._by_key: dict[str, PublishAccountConfig] = dict(publish.accounts)
        self._by_platform_language: dict[tuple[str, str], PublishAccountConfig] = {}
        for key, account in publish.accounts.items():
            platform, separator, language = key.partition("/")
            if separator:
                self._by_platform_language[(platform, language)] = account

    @property
    def publish(self) -> PublishConfig:
        """The publish configuration the map was built from."""
        return self._publish

    @property
    def accounts(self) -> Mapping[str, PublishAccountConfig]:
        """The configured publish accounts keyed by ``platform/language``."""
        return self._by_key

    def names(self) -> tuple[str, ...]:
        """The configured account keys in mapping order."""
        return tuple(self._by_key)

    def resolve(
        self,
        platform: str,
        language: str,
        *,
        profile: str = "",
    ) -> AccountCredentials:
        """Return the credentials for *platform*/*language*.

        Raises :class:`ConfigError` when no account is configured for the
        target, the account is disabled, or the requested *profile* does not
        match the profile bound to the account.
        """
        platform = platform.strip()
        language = language.strip()
        account = self._by_platform_language.get((platform, language))
        if account is None:
            raise ConfigError(
                f"no publish account configured for {platform}/{language}",
                details={
                    "platform": platform,
                    "language": language,
                    "available": sorted(self.names()),
                },
            )
        if not account.enabled:
            raise ConfigError(
                f"publish account {platform}/{language} is disabled",
                details={
                    "platform": platform,
                    "language": language,
                    "account": f"{platform}/{language}",
                },
            )
        if profile and account.profile and account.profile != profile:
            raise ConfigError(
                f"account {platform}/{language} is bound to profile "
                f"{account.profile!r}, not {profile!r}",
                details={
                    "platform": platform,
                    "language": language,
                    "profile": profile,
                    "account_profile": account.profile,
                    "account": f"{platform}/{language}",
                },
            )
        return AccountCredentials(
            platform=account.platform,
            language=account.language,
            profile=account.profile,
            credential_refs=tuple(account.credential_refs),
            enabled=account.enabled,
        )

    def by_profile(self, profile: str) -> list[AccountCredentials]:
        """Return every account bound to *profile*, in key order."""
        profile = profile.strip()
        found: list[AccountCredentials] = []
        for key in self.names():
            account = self._by_key[key]
            if account.profile == profile:
                found.append(
                    AccountCredentials(
                        platform=account.platform,
                        language=account.language,
                        profile=account.profile,
                        credential_refs=tuple(account.credential_refs),
                        enabled=account.enabled,
                    )
                )
        return found


@dataclass(frozen=True, slots=True)
class PostingSlot:
    """One daily posting slot: a time of day to publish on."""

    hour: int = 9
    minute: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.hour <= 23:
            raise ValueError(f"hour out of range 0-23: {self.hour}")
        if not 0 <= self.minute <= 59:
            raise ValueError(f"minute out of range 0-59: {self.minute}")

    @classmethod
    def parse(cls, text: str) -> PostingSlot:
        """Parse ``HH:MM`` or ``H`` into a slot.

        Whitespace around the value is ignored and an absent minute defaults
        to zero.
        """
        text = text.strip()
        if not text:
            raise ValueError("posting slot must be non-empty")
        parts = text.split(":", 1)
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) == 2 and parts[1] else 0
        return cls(hour=hour, minute=minute)

    def as_time(self) -> time:
        """Return the slot as a :class:`datetime.time`."""
        return time(self.hour, self.minute)

    def __str__(self) -> str:
        return f"{self.hour:02d}:{self.minute:02d}"


@dataclass(frozen=True, slots=True)
class DeferralNotice:
    """A posting slot that could not be used and was deferred to a later time."""

    slot: PostingSlot
    deferred_to: datetime
    reason: str
    target: str

    @property
    def key(self) -> str:
        """The ``platform/language`` target the deferred slot belongs to."""
        return self.target


@dataclass(frozen=True, slots=True)
class SlotPlan:
    """A day's posting plan.

    ``slots`` are the nominal daily slots. :meth:`posting_times` applies jitter
    to each slot to produce the actual datetimes for *date*. ``notices``
    records any deferrals that occurred while building the plan.
    """

    date: date
    slots: tuple[PostingSlot, ...] = field(default_factory=tuple)
    jitter_seconds: int = 0
    notices: tuple[DeferralNotice, ...] = field(default_factory=tuple)
    timezone: str = "UTC"

    def __post_init__(self) -> None:
        if self.jitter_seconds < 0:
            raise ValueError(
                f"jitter_seconds must not be negative: {self.jitter_seconds}"
            )
        try:
            ZoneInfo(self.timezone)
        except Exception as exc:  # pragma: no cover - validated at config level
            raise ValueError(f"unknown timezone: {self.timezone}") from exc

    def posting_times(
        self,
        *,
        rng: Callable[[], float] | None = None,
    ) -> list[datetime]:
        """Return the actual posting datetimes for the day.

        Each nominal slot is shifted by a jitter offset sampled from
        ``[-jitter_seconds, +jitter_seconds]`` seconds. A deterministic
        ``rng`` returning values in ``[0, 1)`` makes the plan reproducible;
        when omitted, :func:`random.random` is used.
        """
        sample = (rng or random.random)()
        zone = ZoneInfo(self.timezone)
        times: list[datetime] = []
        for slot in self.slots:
            base = datetime(
                self.date.year,
                self.date.month,
                self.date.day,
                slot.hour,
                slot.minute,
                tzinfo=zone,
            )
            offset_seconds = 0.0
            if self.jitter_seconds > 0:
                offset_seconds = (sample * 2.0 - 1.0) * self.jitter_seconds
            times.append(base + timedelta(seconds=offset_seconds))
        return times

    def defer(
        self,
        slot: PostingSlot,
        reason: str,
        target: str,
    ) -> tuple[SlotPlan, DeferralNotice]:
        """Return a new plan with *slot* deferred to the next later slot.

        The deferred slot moves to the next slot in the plan (wrapping to the
        first slot of the following day when it is the last one) and a
        :class:`DeferralNotice` is recorded. The original plan is unchanged.
        """
        slots = list(self.slots)
        if slot not in slots:
            raise ValueError(f"slot {slot} is not part of this plan")
        index = slots.index(slot)
        next_index = index + 1
        next_date = self.date
        if next_index >= len(slots):
            next_index = 0
            next_date = self.date + timedelta(days=1)
        next_slot = slots[next_index]
        zone = ZoneInfo(self.timezone)
        deferred_to = datetime(
            next_date.year,
            next_date.month,
            next_date.day,
            next_slot.hour,
            next_slot.minute,
            tzinfo=zone,
        )
        notice = DeferralNotice(
            slot=slot,
            deferred_to=deferred_to,
            reason=reason,
            target=target,
        )
        return (
            SlotPlan(
                date=self.date,
                slots=self.slots,
                jitter_seconds=self.jitter_seconds,
                notices=(*self.notices, notice),
                timezone=self.timezone,
            ),
            notice,
        )


def apply_jitter(
    base: datetime,
    jitter_seconds: int,
    *,
    rng: Callable[[], float] | None = None,
) -> datetime:
    """Shift *base* by a jitter offset in ``[-jitter_seconds, +jitter_seconds]``.

    When *jitter_seconds* is zero or negative, *base* is returned unchanged.
    A deterministic ``rng`` returning values in ``[0, 1)`` makes the result
    reproducible; when omitted, :func:`random.random` is used.
    """
    if jitter_seconds <= 0 or base.tzinfo is None:
        return base
    sample = float((rng or random.random)())
    offset_seconds = (sample * 2.0 - 1.0) * jitter_seconds
    return base + timedelta(seconds=offset_seconds)

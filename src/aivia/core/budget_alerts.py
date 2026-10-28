"""Budget alert thresholds and the notifier fallback that delivers them.

Alert thresholds are absolute spend amounts configured in
:attr:`~aivia.core.config.budget.BudgetConfig.alerts`. This module compares the
current period's spend with those thresholds and returns one
:class:`BudgetAlert` per crossed threshold. A threshold is emitted at most once
per period, so a run that keeps spending does not repeat the same warning while
a new day or month resets it.

The period is a UTC calendar day by default and may be a UTC calendar month.
State is kept in :class:`BudgetAlertTracker`; callers that need to survive a
restart can persist :attr:`BudgetAlertTracker.emitted`. A
:class:`BudgetAlertNotifier` turns the due alerts into
:class:`~aivia.providers.protocols.notify.NotificationMessage` objects and sends
them through any notifier, falling back to
:class:`~aivia.providers.console.ConsoleNotifier` when none is supplied.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from enum import StrEnum
from threading import RLock
from typing import Any, cast

from aivia.core.config.budget import BudgetConfig
from aivia.core.errors import ConfigError
from aivia.core.models.base import utcnow
from aivia.providers.console import (
    ConsoleNotificationProvider,
    ConsoleNotifier,
    console_notifier,
    default_notifier,
)
from aivia.providers.protocols.notify import (
    NotificationLevel,
    NotificationMessage,
    NotificationReceipt,
    Notifier,
)

__all__ = [
    "AlertPeriod",
    "BudgetAlert",
    "BudgetAlertEmitter",
    "BudgetAlertManager",
    "BudgetAlertNotifier",
    "BudgetAlertPeriod",
    "BudgetAlertState",
    "BudgetAlertTracker",
    "ConsoleAlertNotifier",
    "ConsoleNotificationProvider",
    "ConsoleNotifier",
    "console_notifier",
    "default_notifier",
    "emit_budget_alerts",
]


class BudgetAlertPeriod(StrEnum):
    """The calendar period over which a threshold fires only once."""

    DAY = "day"
    MONTH = "month"


AlertPeriod = BudgetAlertPeriod
"""Alias for :class:`BudgetAlertPeriod`."""

_PERIOD_ALIASES = {
    "day": BudgetAlertPeriod.DAY,
    "daily": BudgetAlertPeriod.DAY,
    "month": BudgetAlertPeriod.MONTH,
    "monthly": BudgetAlertPeriod.MONTH,
}


def _coerce_amount(value: object, *, label: str = "spent_usd") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            f"{label} must be a finite, non-negative number",
            details={label: value},
        )
    amount = float(value)
    if not math.isfinite(amount) or amount < 0:
        raise ConfigError(
            f"{label} must be a finite, non-negative number",
            details={label: value},
        )
    return amount


def _coerce_now(value: object) -> datetime:
    if value is None:
        return utcnow()
    if not isinstance(value, datetime):
        raise ConfigError(
            "budget alert timestamp must be a datetime",
            details={"now": value},
        )
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _coerce_period(value: object) -> BudgetAlertPeriod:
    if isinstance(value, BudgetAlertPeriod):
        return value
    if isinstance(value, str):
        period = _PERIOD_ALIASES.get(value.strip().lower())
        if period is not None:
            return period
    allowed = ", ".join(period.value for period in BudgetAlertPeriod)
    raise ConfigError(
        f"unknown budget alert period: {value!r} (allowed: {allowed})",
        details={"period": value, "allowed": [p.value for p in BudgetAlertPeriod]},
    )


def _coerce_thresholds(
    config: BudgetConfig | Mapping[str, Any] | Iterable[float] | None,
) -> list[float]:
    if config is None:
        return []
    if isinstance(config, BudgetConfig):
        return list(config.alerts)
    if isinstance(config, Mapping):
        return list(BudgetConfig.from_mapping(config).alerts)
    if isinstance(config, (str, bytes)) or not isinstance(config, Iterable):
        raise ConfigError(
            "budget alert config must be a BudgetConfig, mapping or thresholds",
            details={"config": type(config).__name__},
        )
    return [_coerce_amount(value, label="alert threshold") for value in config]


def _reaches(amount: float, threshold: float) -> bool:
    """Return whether *amount* reached *threshold*, tolerating float noise."""
    return amount > threshold or math.isclose(
        amount,
        threshold,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


def _period_window(
    period: BudgetAlertPeriod,
    now: datetime,
) -> tuple[datetime, datetime, str]:
    """Return the half-open UTC window and stable key for *now*'s period."""
    if period is BudgetAlertPeriod.MONTH:
        start = datetime(now.year, now.month, 1, tzinfo=UTC)
        if now.month == 12:
            end = datetime(now.year + 1, 1, 1, tzinfo=UTC)
        else:
            end = datetime(now.year, now.month + 1, 1, tzinfo=UTC)
        return start, end, f"{now.year:04d}-{now.month:02d}"
    start = datetime.combine(now.date(), time.min, tzinfo=UTC)
    return start, start + timedelta(days=1), now.date().isoformat()


@dataclass(frozen=True, slots=True)
class BudgetAlert:
    """One threshold reached during one alert period.

    ``spent_usd`` is the spend observed when the threshold was crossed, so an
    alert records the amount that triggered it even when the caller later keeps
    spending. ``period_key`` is stable across processes and can be persisted
    with :attr:`BudgetAlertTracker.emitted`.
    """

    threshold_usd: float
    spent_usd: float
    period: BudgetAlertPeriod
    period_key: str
    period_start: datetime
    period_end: datetime

    @property
    def threshold(self) -> float:
        """Alias for :attr:`threshold_usd`."""
        return self.threshold_usd

    @property
    def spent(self) -> float:
        """Alias for :attr:`spent_usd`."""
        return self.spent_usd

    @property
    def exceeded_by_usd(self) -> float:
        """How far spend was above the threshold when the alert fired."""
        return max(self.spent_usd - self.threshold_usd, 0.0)

    @property
    def amount_over(self) -> float:
        """Alias for :attr:`exceeded_by_usd`."""
        return self.exceeded_by_usd

    @property
    def key(self) -> tuple[str, float]:
        """The identity used to emit this threshold once per period."""
        return (self.period_key, self.threshold_usd)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for logs and messages."""
        return {
            "threshold_usd": self.threshold_usd,
            "spent_usd": self.spent_usd,
            "exceeded_by_usd": self.exceeded_by_usd,
            "period": self.period.value,
            "period_key": self.period_key,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
        }


class BudgetAlertTracker:
    """Return each configured threshold once per calendar period.

    Pass a :class:`~aivia.core.config.budget.BudgetConfig`, a mapping using its
    aliases, or a plain iterable of thresholds. ``check`` marks the returned
    alerts as emitted while ``peek`` only reports what is due. Time is an
    argument rather than a clock attribute so callers can test month and day
    rollover without waiting.
    """

    def __init__(
        self,
        config: BudgetConfig | Mapping[str, Any] | Iterable[float] | None = None,
        *,
        thresholds: Iterable[float] | None = None,
        period: BudgetAlertPeriod | str = BudgetAlertPeriod.DAY,
    ) -> None:
        if config is not None and thresholds is not None:
            raise ConfigError("pass either budget alert config or thresholds")
        source = config if config is not None else thresholds
        self._thresholds = tuple(sorted(set(_coerce_thresholds(source))))
        self._period = _coerce_period(period)
        self._emitted: set[tuple[str, float]] = set()
        self._lock = RLock()

    @property
    def thresholds(self) -> tuple[float, ...]:
        """The configured thresholds in ascending order."""
        return self._thresholds

    @property
    def alerts(self) -> tuple[float, ...]:
        """Alias for :attr:`thresholds`."""
        return self._thresholds

    @property
    def period(self) -> BudgetAlertPeriod:
        """The calendar period over which a threshold fires once."""
        return self._period

    @property
    def emitted(self) -> frozenset[tuple[str, float]]:
        """Every ``(period_key, threshold)`` pair already emitted."""
        with self._lock:
            return frozenset(self._emitted)

    def reset(self) -> None:
        """Forget every emitted threshold, so the next check may fire again."""
        with self._lock:
            self._emitted.clear()

    def period_window(
        self,
        now: datetime | None = None,
    ) -> tuple[datetime, datetime, str]:
        """Return the current period's start, end and stable key."""
        return _period_window(self._period, _coerce_now(now))

    def _due(
        self,
        spent_usd: float,
        now: datetime | None,
        *,
        mark: bool,
    ) -> tuple[BudgetAlert, ...]:
        spent = _coerce_amount(spent_usd)
        start, end, key = _period_window(self._period, _coerce_now(now))
        with self._lock:
            due: list[BudgetAlert] = []
            for threshold in self._thresholds:
                token = (key, threshold)
                if token in self._emitted or not _reaches(spent, threshold):
                    continue
                due.append(
                    BudgetAlert(
                        threshold_usd=threshold,
                        spent_usd=spent,
                        period=self._period,
                        period_key=key,
                        period_start=start,
                        period_end=end,
                    )
                )
                if mark:
                    self._emitted.add(token)
            return tuple(due)

    def peek(
        self,
        spent_usd: float,
        *,
        now: datetime | None = None,
    ) -> tuple[BudgetAlert, ...]:
        """Return the thresholds due now without marking them emitted."""
        return self._due(spent_usd, now, mark=False)

    def check(
        self,
        spent_usd: float,
        *,
        now: datetime | None = None,
    ) -> tuple[BudgetAlert, ...]:
        """Return the thresholds due now and mark them emitted for the period."""
        return self._due(spent_usd, now, mark=True)

    def due(
        self,
        spent_usd: float,
        *,
        now: datetime | None = None,
    ) -> tuple[BudgetAlert, ...]:
        """Alias for :meth:`check`."""
        return self.check(spent_usd, now=now)

    def collect(
        self,
        spent_usd: float,
        *,
        now: datetime | None = None,
    ) -> tuple[BudgetAlert, ...]:
        """Alias for :meth:`check`."""
        return self.check(spent_usd, now=now)


BudgetAlertState = BudgetAlertTracker
"""Alias for :class:`BudgetAlertTracker` for callers that prefer state wording."""

BudgetAlertManager = BudgetAlertTracker
"""Alias for :class:`BudgetAlertTracker` for callers that prefer manager wording."""


class BudgetAlertNotifier:
    """Turn due budget alerts into notifications and send them.

    ``notifier`` may be any object exposing the notifier ``send`` coroutine. When
    it is omitted, the console notifier is used, so a module never has to build
    a channel just to report a threshold. The same :class:`BudgetAlertTracker`
    instance is reused across calls, which is what makes each threshold fire
    once per period.
    """

    def __init__(
        self,
        notifier: (
            Notifier | BudgetConfig | Mapping[str, Any] | Iterable[float] | None
        ) = None,
        config: BudgetConfig | Mapping[str, Any] | Iterable[float] | None = None,
        *,
        thresholds: Iterable[float] | None = None,
        period: BudgetAlertPeriod | str = BudgetAlertPeriod.DAY,
        tracker: BudgetAlertTracker | None = None,
    ) -> None:
        if tracker is not None and (config is not None or thresholds is not None):
            raise ConfigError("pass either a tracker or alert thresholds, not both")
        source = config
        selected: object = notifier
        if (
            notifier is not None
            and config is None
            and thresholds is None
            and not callable(getattr(notifier, "send", None))
            and (
                isinstance(notifier, (BudgetConfig, Mapping))
                or (
                    isinstance(notifier, Iterable)
                    and not isinstance(notifier, (str, bytes))
                )
            )
        ):
            source = cast(BudgetConfig | Mapping[str, Any] | Iterable[float], notifier)
            selected = None
        if tracker is not None:
            self._tracker = tracker
        else:
            self._tracker = BudgetAlertTracker(
                source,
                thresholds=thresholds,
                period=period,
            )
        self._notifier = (
            cast(Notifier, selected) if selected is not None else ConsoleNotifier()
        )
        send = getattr(self._notifier, "send", None)
        if not callable(send):
            raise ConfigError(
                "budget alert notifier must expose a send coroutine",
                details={"notifier": type(self._notifier).__name__},
            )

    @property
    def tracker(self) -> BudgetAlertTracker:
        """The tracker that enforces once-per-threshold-per-period."""
        return self._tracker

    @property
    def notifier(self) -> Notifier:
        """The notifier alerts are delivered through."""
        return self._notifier

    @property
    def thresholds(self) -> tuple[float, ...]:
        """Alias for :attr:`BudgetAlertTracker.thresholds`."""
        return self._tracker.thresholds

    @property
    def period(self) -> BudgetAlertPeriod:
        """Alias for :attr:`BudgetAlertTracker.period`."""
        return self._tracker.period

    @staticmethod
    def message_for(alert: BudgetAlert) -> NotificationMessage:
        """Build the warning message for one alert."""
        if not isinstance(alert, BudgetAlert):
            raise ConfigError(
                "budget alert notifier needs a BudgetAlert",
                details={"alert": type(alert).__name__},
            )
        return NotificationMessage(
            title=f"Budget alert: {alert.threshold_usd:.6g} USD",
            text=(
                f"Spend of {alert.spent_usd:.6g} USD reached the "
                f"{alert.threshold_usd:.6g} USD threshold for "
                f"{alert.period_key} ({alert.period.value})."
            ),
            level=NotificationLevel.WARNING,
            metadata=alert.to_dict(),
        )

    async def notify(
        self,
        alerts: Iterable[BudgetAlert],
    ) -> tuple[NotificationReceipt, ...]:
        """Send one message per alert and return every receipt."""
        receipts: list[NotificationReceipt] = []
        for alert in alerts:
            receipts.append(await self._notifier.send(self.message_for(alert)))
        return tuple(receipts)

    async def emit(
        self,
        spent_usd: float,
        *,
        now: datetime | None = None,
    ) -> tuple[NotificationReceipt, ...]:
        """Send every threshold due for *spent_usd* and return the receipts."""
        return await self.notify(self._tracker.check(spent_usd, now=now))

    async def emit_alerts(
        self,
        spent_usd: float,
        *,
        now: datetime | None = None,
    ) -> tuple[NotificationReceipt, ...]:
        """Alias for :meth:`emit`."""
        return await self.emit(spent_usd, now=now)

    async def send(
        self,
        spent_usd: float,
        *,
        now: datetime | None = None,
    ) -> tuple[NotificationReceipt, ...]:
        """Alias for :meth:`emit`."""
        return await self.emit(spent_usd, now=now)

    async def check_and_notify(
        self,
        spent_usd: float,
        *,
        now: datetime | None = None,
    ) -> tuple[NotificationReceipt, ...]:
        """Alias for :meth:`emit`."""
        return await self.emit(spent_usd, now=now)


BudgetAlertEmitter = BudgetAlertNotifier
"""Alias for :class:`BudgetAlertNotifier` for callers that prefer emitter wording."""

ConsoleAlertNotifier = ConsoleNotifier
"""Alias for :class:`ConsoleNotifier` in budget alert code."""


async def emit_budget_alerts(
    spent_usd: float,
    *,
    config: BudgetConfig | Mapping[str, Any] | Iterable[float] | None = None,
    thresholds: Iterable[float] | None = None,
    notifier: Notifier | None = None,
    period: BudgetAlertPeriod | str = BudgetAlertPeriod.DAY,
    tracker: BudgetAlertTracker | None = None,
    now: datetime | None = None,
) -> tuple[NotificationReceipt, ...]:
    """Emit the budget alerts due for *spent_usd* without a long-lived object."""
    return await BudgetAlertNotifier(
        notifier,
        config,
        thresholds=thresholds,
        period=period,
        tracker=tracker,
    ).emit(spent_usd, now=now)

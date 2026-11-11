"""Budget and failure alerts plus a daily summary built from stored data.

A review run needs a human told when spend crosses a configured threshold,
when a stage or run fails, and what the day's spend looks like once the ledger
has settled. This module ties the budget alert tracker, the console fallback
notifier and the cost ledger together behind a single :class:`AlertManager`.

Budget alerts fire once per threshold per calendar period through
:class:`~aivia.core.budget_alerts.BudgetAlertNotifier`, falling back to the
console notifier when no channel is configured. Failure alerts carry the error
and an optional context at
:data:`~aivia.providers.protocols.notify.NotificationLevel.ERROR`.
The daily summary aggregates stored cost events for one UTC day and is delivered
as an :data:`~aivia.providers.protocols.notify.NotificationLevel.INFO` message.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from typing import Any

from aivia.core.budget_alerts import (
    BudgetAlertNotifier,
    BudgetAlertTracker,
)
from aivia.core.cost_agg import CostAggregate, CostAggregator
from aivia.providers.console import ConsoleNotifier
from aivia.providers.protocols.notify import (
    NotificationLevel,
    NotificationMessage,
    NotificationReceipt,
    Notifier,
)

__all__ = [
    "AlertManager",
    "DailySummary",
]


def _as_utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _day_window(value: datetime) -> tuple[datetime, datetime]:
    """Return the inclusive start and end of the UTC day *value* falls in."""
    start = datetime.combine(value.date(), time.min, tzinfo=UTC)
    return start, start + timedelta(days=1) - timedelta.resolution


@dataclass(frozen=True, slots=True)
class DailySummary:
    """One UTC day's spend, built from stored cost events.

    ``by_provider`` is the per-provider breakdown returned by the cost
    aggregator, so a summary can list where the money went without scanning the
    ledger again.
    """

    date: str
    """The UTC calendar day, ``YYYY-MM-DD``."""
    total_usd: float
    estimated_usd: float
    actual_usd: float
    units: float
    count: int
    by_provider: tuple[CostAggregate, ...]

    @property
    def total(self) -> float:
        """Alias for :attr:`total_usd`."""
        return self.total_usd

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for messages and logs."""
        return {
            "date": self.date,
            "total_usd": self.total_usd,
            "estimated_usd": self.estimated_usd,
            "actual_usd": self.actual_usd,
            "units": self.units,
            "count": self.count,
            "by_provider": [
                {
                    "provider": bucket.key,
                    "total_usd": bucket.total_usd,
                    "count": bucket.count,
                }
                for bucket in self.by_provider
            ],
        }


class AlertManager:
    """Send budget and failure alerts plus a daily summary from stored data.

    *notifier* delivers every message and defaults to the console notifier, so
    a manager always has a channel even when no channel is configured. *tracker*
    enforces the once-per-threshold-per-period rule for budget alerts; without
    one each threshold can fire repeatedly. *aggregator* reads the stored cost
    ledger and is required to build a daily summary.
    """

    def __init__(
        self,
        *,
        notifier: Notifier | None = None,
        tracker: BudgetAlertTracker | None = None,
        aggregator: CostAggregator | None = None,
    ) -> None:
        self._notifier = notifier if notifier is not None else ConsoleNotifier()
        self._tracker = tracker if tracker is not None else BudgetAlertTracker()
        self._budget_notifier = BudgetAlertNotifier(
            notifier=self._notifier,
            tracker=self._tracker,
        )
        self._aggregator = aggregator

    @property
    def notifier(self) -> Notifier:
        """The notifier every alert and summary is delivered through."""
        return self._notifier

    @property
    def tracker(self) -> BudgetAlertTracker:
        """The tracker enforcing once-per-threshold-per-period budget alerts."""
        return self._tracker

    @property
    def aggregator(self) -> CostAggregator | None:
        """The cost ledger used to build daily summaries, if any."""
        return self._aggregator

    async def send_budget_alerts(
        self,
        spent_usd: float,
        *,
        now: datetime | None = None,
    ) -> tuple[NotificationReceipt, ...]:
        """Send the budget alerts due for *spent_usd* and return the receipts.

        Each configured threshold fires at most once per period, so repeating
        the same spend level does not repeat an alert.
        """
        return await self._budget_notifier.emit(spent_usd, now=now)

    async def send_failure_alert(
        self,
        error: BaseException | str,
        *,
        context: str = "",
        now: datetime | None = None,
    ) -> NotificationReceipt:
        """Send an error-level notification describing *error*.

        ``context`` describes where the failure happened (for example a stage or
        run id) and is prefixed to the message text. The moment the alert was
        raised is recorded in the message metadata when *now* is given.
        """
        text = str(error)
        if context:
            text = f"{context}: {text}"
        metadata: dict[str, Any] = {"error": str(error)}
        if context:
            metadata["context"] = context
        if now is not None:
            metadata["occurred_at"] = _as_utc(now).isoformat()
        message = NotificationMessage(
            title="Failure alert",
            text=text,
            level=NotificationLevel.ERROR,
            metadata=metadata,
        )
        return await self._notifier.send(message)

    def daily_summary(self, *, date: datetime | str | None = None) -> DailySummary:
        """Build one UTC day's summary from the stored cost events.

        *date* selects the day to summarize and defaults to today in UTC. Raises
        :class:`ValueError` when no cost aggregator was configured, because
        there is no stored data to build the summary from.
        """
        if self._aggregator is None:
            raise ValueError(
                "AlertManager needs a cost aggregator to build a daily summary"
            )
        moment = _as_utc(
            datetime.fromisoformat(str(date)) if isinstance(date, str) else date
        )
        start, end = _day_window(moment)
        # The per-provider buckets already sum to the day's total, so one query
        # gives both the overall spend and the breakdown behind it.
        buckets = tuple(self._aggregator.by_provider(start=start, end=end))
        total = sum(bucket.total_usd for bucket in buckets)
        estimated = sum(bucket.estimated_usd for bucket in buckets)
        actual = sum(bucket.actual_usd for bucket in buckets)
        units = sum(bucket.units for bucket in buckets)
        count = sum(bucket.count for bucket in buckets)
        return DailySummary(
            date=moment.date().isoformat(),
            total_usd=total,
            estimated_usd=estimated,
            actual_usd=actual,
            units=units,
            count=count,
            by_provider=buckets,
        )

    async def send_daily_summary(
        self,
        *,
        date: datetime | str | None = None,
        now: datetime | None = None,
    ) -> NotificationReceipt:
        """Build the daily summary for *date* and deliver it as one message."""
        summary = self.daily_summary(date=date)
        providers = (
            ", ".join(
                f"{bucket.key or '(none)'}: ${bucket.total_usd:.4f}"
                for bucket in summary.by_provider
            )
            or "no spend recorded"
        )
        message = NotificationMessage(
            title=f"Daily summary {summary.date}",
            text=(
                f"Spend on {summary.date}: ${summary.total_usd:.4f} "
                f"({summary.count} event{'s' if summary.count != 1 else ''}). "
                f"By provider: {providers}."
            ),
            level=NotificationLevel.INFO,
            metadata=summary.to_dict(),
        )
        return await self._notifier.send(message)


BudgetAlertManager = AlertManager
"""Alias for :class:`AlertManager` for callers that prefer budget wording."""

BudgetAlertEmitter = AlertManager
"""Alias for :class:`AlertManager` for callers that prefer emitter wording."""

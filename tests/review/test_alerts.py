"""Budget and failure alerts plus a daily summary built from stored data.

The manager sends budget alerts once per threshold per period, failure alerts
at error level, and a daily summary aggregated from the stored cost ledger.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from sqlalchemy import Engine

from aivia.core.budget_alerts import BudgetAlertTracker
from aivia.core.cost import CostTracker
from aivia.core.cost_agg import CostAggregator
from aivia.core.db import create_database_engine, upgrade_database
from aivia.providers.protocols.notify import (
    NotificationLevel,
    NotificationMessage,
    NotificationReceipt,
    Notifier,
)
from aivia.review import AlertManager, DailySummary

NOW = datetime(2026, 3, 15, 12, 0, tzinfo=UTC)
NEXT_DAY = NOW + timedelta(days=1)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


class RecordingNotifier:
    """A notifier that keeps every message it is asked to send."""

    def __init__(self) -> None:
        self.sent: list[NotificationMessage] = []

    async def send(self, message: Any) -> NotificationReceipt:
        self.sent.append(message)
        return NotificationReceipt(
            message_id="test-message",
            channel="test",
            provider="test",
            delivered=True,
        )


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


@pytest.fixture
def notifier() -> RecordingNotifier:
    return RecordingNotifier()


def _record(
    engine: Engine,
    *,
    day: str,
    provider: str = "acme",
    amount_usd: float = 0.10,
    estimated: bool = False,
) -> None:
    created_at = datetime.fromisoformat(f"{day}T10:00:00+00:00")
    CostTracker(engine).record(
        provider=provider,
        model="image-1",
        operation="image_gen",
        units=1.0,
        unit_kind="image",
        amount_usd=amount_usd,
        run_id="",
        estimated=estimated,
        created_at=created_at,
    )


def _manager(
    notifier: Any,
    *,
    tracker: BudgetAlertTracker | None = None,
    aggregator: CostAggregator | None = None,
) -> AlertManager:
    return AlertManager(
        notifier=cast(Notifier, notifier),
        tracker=tracker,
        aggregator=aggregator,
    )


def test_default_notifier_is_the_console_fallback(
    notifier: RecordingNotifier,
) -> None:
    channel = cast(Notifier, notifier)
    manager = AlertManager(notifier=channel)
    assert manager.notifier is channel


def test_budget_alert_sent_once_per_threshold_per_day(
    notifier: RecordingNotifier,
) -> None:
    tracker = BudgetAlertTracker([10.0, 20.0])
    manager = _manager(notifier, tracker=tracker)

    first = _run(manager.send_budget_alerts(12.0, now=NOW))

    assert len(first) == 1
    assert len(notifier.sent) == 1
    assert notifier.sent[0].level is NotificationLevel.WARNING
    assert "10" in notifier.sent[0].text

    # The same spend in the same period does not repeat the alert.
    assert _run(manager.send_budget_alerts(12.0, now=NOW)) == ()
    assert len(notifier.sent) == 1

    # A higher spend crosses the next threshold once.
    second = _run(manager.send_budget_alerts(25.0, now=NOW))
    assert len(second) == 1
    assert notifier.sent[1].level is NotificationLevel.WARNING
    assert "20" in notifier.sent[1].text
    assert _run(manager.send_budget_alerts(25.0, now=NOW)) == ()
    assert len(notifier.sent) == 2


def test_budget_alert_fires_again_in_a_new_period(
    notifier: RecordingNotifier,
) -> None:
    tracker = BudgetAlertTracker([10.0])
    manager = _manager(notifier, tracker=tracker)

    assert len(_run(manager.send_budget_alerts(12.0, now=NOW))) == 1
    assert len(_run(manager.send_budget_alerts(12.0, now=NEXT_DAY))) == 1
    assert len(notifier.sent) == 2
    assert notifier.sent[0].metadata["period_key"] == "2026-03-15"
    assert notifier.sent[1].metadata["period_key"] == "2026-03-16"


def test_failure_alert_sends_error_level_with_the_error_text(
    notifier: RecordingNotifier,
) -> None:
    manager = _manager(notifier)

    receipt = _run(
        manager.send_failure_alert(ValueError("boom"), context="stage:render")
    )

    assert isinstance(receipt, NotificationReceipt)
    assert len(notifier.sent) == 1
    message = notifier.sent[0]
    assert message.level is NotificationLevel.ERROR
    assert message.title == "Failure alert"
    assert "stage:render: boom" in message.text
    assert message.metadata["error"] == "boom"
    assert message.metadata["context"] == "stage:render"


def test_failure_alert_accepts_a_plain_string_error(
    notifier: RecordingNotifier,
) -> None:
    manager = _manager(notifier)

    _run(manager.send_failure_alert("connection lost"))

    assert notifier.sent[0].text == "connection lost"
    assert notifier.sent[0].metadata["error"] == "connection lost"


def test_failure_alert_records_occurred_at_when_now_is_given(
    notifier: RecordingNotifier,
) -> None:
    manager = _manager(notifier)

    _run(manager.send_failure_alert(ValueError("x"), now=NOW))

    assert notifier.sent[0].metadata["occurred_at"] == NOW.isoformat()


def test_daily_summary_built_from_stored_data(engine: Engine) -> None:
    _record(engine, day="2026-01-05", provider="acme", amount_usd=0.10)
    _record(engine, day="2026-01-05", provider="acme", amount_usd=0.20)
    _record(engine, day="2026-01-05", provider="beta", amount_usd=0.05)
    _record(engine, day="2026-01-06", provider="acme", amount_usd=999.0)

    aggregator = CostAggregator(engine)
    manager = _manager(RecordingNotifier(), aggregator=aggregator)

    summary = manager.daily_summary(date="2026-01-05")

    assert isinstance(summary, DailySummary)
    assert summary.date == "2026-01-05"
    assert summary.total_usd == pytest.approx(0.35)
    assert summary.count == 3
    assert summary.actual_usd == pytest.approx(0.35)
    assert summary.estimated_usd == 0.0
    by_provider = {bucket.key: bucket.total_usd for bucket in summary.by_provider}
    assert by_provider == {"acme": pytest.approx(0.30), "beta": pytest.approx(0.05)}


def test_daily_summary_uses_the_configured_date(engine: Engine) -> None:
    _record(engine, day="2026-01-05", amount_usd=0.10)
    _record(engine, day="2026-01-05", amount_usd=0.20)
    _record(engine, day="2026-01-07", amount_usd=0.50)

    aggregator = CostAggregator(engine)
    manager = _manager(RecordingNotifier(), aggregator=aggregator)

    summary = manager.daily_summary(date=datetime(2026, 1, 7, 3, 0, tzinfo=UTC))

    assert summary.date == "2026-01-07"
    assert summary.total_usd == pytest.approx(0.50)
    assert summary.count == 1


def test_daily_summary_empty_day_returns_zero(engine: Engine) -> None:
    _record(engine, day="2026-01-05", amount_usd=0.10)

    manager = _manager(RecordingNotifier(), aggregator=CostAggregator(engine))

    summary = manager.daily_summary(date="2026-01-06")

    assert summary.date == "2026-01-06"
    assert summary.total_usd == 0.0
    assert summary.count == 0
    assert summary.by_provider == ()


def test_send_daily_summary_delivers_an_info_message(engine: Engine) -> None:
    _record(engine, day="2026-01-05", provider="acme", amount_usd=0.25)

    notifier = RecordingNotifier()
    manager = _manager(notifier, aggregator=CostAggregator(engine))

    receipt = _run(manager.send_daily_summary(date="2026-01-05"))

    assert isinstance(receipt, NotificationReceipt)
    assert len(notifier.sent) == 1
    message = notifier.sent[0]
    assert message.level is NotificationLevel.INFO
    assert message.title == "Daily summary 2026-01-05"
    assert "$0.25" in message.text
    assert message.metadata["date"] == "2026-01-05"
    assert message.metadata["total_usd"] == pytest.approx(0.25)


def test_daily_summary_requires_a_cost_aggregator(notifier: RecordingNotifier) -> None:
    manager = _manager(notifier)

    with pytest.raises(ValueError, match="cost aggregator"):
        manager.daily_summary(date="2026-01-05")


def test_daily_summary_to_dict_is_json_serializable(engine: Engine) -> None:
    _record(engine, day="2026-01-05", provider="acme", amount_usd=0.10)

    manager = _manager(RecordingNotifier(), aggregator=CostAggregator(engine))
    summary = manager.daily_summary(date="2026-01-05")

    payload = summary.to_dict()
    assert payload["date"] == "2026-01-05"
    assert payload["total_usd"] == pytest.approx(0.10)
    assert payload["by_provider"] == [
        {
            "provider": "acme",
            "total_usd": pytest.approx(0.10),
            "count": 1,
        }
    ]


def test_shared_tracker_fires_each_threshold_once_across_managers(
    notifier: RecordingNotifier,
) -> None:
    tracker = BudgetAlertTracker([10.0])
    first = _manager(notifier, tracker=tracker)
    second = _manager(RecordingNotifier(), tracker=tracker)

    assert len(_run(first.send_budget_alerts(12.0, now=NOW))) == 1
    assert len(_run(second.send_budget_alerts(12.0, now=NOW))) == 0

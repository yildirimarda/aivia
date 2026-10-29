"""Budget alert thresholds, once-per-period emission and the console fallback."""

from __future__ import annotations

import asyncio
import io
import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from aivia.core.budget import BudgetGuard
from aivia.core.budget_alerts import (
    AlertPeriod,
    BudgetAlert,
    BudgetAlertEmitter,
    BudgetAlertManager,
    BudgetAlertNotifier,
    BudgetAlertPeriod,
    BudgetAlertState,
    BudgetAlertTracker,
    ConsoleAlertNotifier,
    ConsoleNotifier,
    emit_budget_alerts,
)
from aivia.core.config import BudgetConfig
from aivia.core.errors import ConfigError
from aivia.core.redact import clear_secrets, register_secret
from aivia.providers import (
    ConsoleNotificationProvider,
    console_notifier,
    default_notifier,
)
from aivia.providers.mock import MockNotifier
from aivia.providers.protocols import (
    NotificationLevel,
    NotificationMedia,
    NotificationMessage,
    NotificationReceipt,
    Notifier,
)

NOW = datetime(2026, 3, 15, 12, tzinfo=UTC)


def test_thresholds_are_sorted_and_deduplicated_from_config_aliases() -> None:
    tracker = BudgetAlertTracker({"thresholds": [20, 10, 20]})

    assert tracker.thresholds == (10.0, 20.0)
    assert tracker.alerts == (10.0, 20.0)
    assert tracker.period is BudgetAlertPeriod.DAY


def test_a_threshold_is_emitted_once_per_day() -> None:
    tracker = BudgetAlertTracker(BudgetConfig(alerts=[10, 20]))

    assert tracker.check(5, now=NOW) == ()
    first = tracker.check(12, now=NOW)

    assert [alert.threshold_usd for alert in first] == [10.0]
    assert first[0].spent_usd == 12.0
    assert first[0].period_key == "2026-03-15"
    assert first[0].period_start == datetime(2026, 3, 15, tzinfo=UTC)
    assert first[0].period_end == datetime(2026, 3, 16, tzinfo=UTC)
    assert tracker.check(12, now=NOW) == ()
    assert tracker.check(12, now=NOW + timedelta(hours=6)) == ()

    second = tracker.check(25, now=NOW)

    assert [alert.threshold_usd for alert in second] == [20.0]
    assert tracker.check(25, now=NOW) == ()
    assert tracker.emitted == {
        ("2026-03-15", 10.0),
        ("2026-03-15", 20.0),
    }


def test_a_jump_emits_every_crossed_threshold_in_ascending_order() -> None:
    tracker = BudgetAlertTracker([30, 10, 20])

    alerts = tracker.check(35, now=NOW)

    assert [alert.threshold_usd for alert in alerts] == [10.0, 20.0, 30.0]
    assert all(alert.spent_usd == 35.0 for alert in alerts)


def test_a_new_day_resets_every_threshold() -> None:
    tracker = BudgetAlertTracker([10, 20])

    assert len(tracker.check(25, now=NOW)) == 2
    tomorrow = NOW + timedelta(days=1)

    alerts = tracker.check(25, now=tomorrow)

    assert [alert.threshold_usd for alert in alerts] == [10.0, 20.0]
    assert {alert.period_key for alert in alerts} == {"2026-03-16"}


def test_a_monthly_tracker_resets_only_when_the_month_changes() -> None:
    tracker = BudgetAlertTracker([10, 20], period=BudgetAlertPeriod.MONTH)

    assert len(tracker.check(25, now=NOW)) == 2
    assert tracker.check(25, now=NOW + timedelta(days=15)) == ()
    next_month = datetime(2026, 4, 1, tzinfo=UTC)

    alerts = tracker.check(25, now=next_month)

    assert [alert.threshold_usd for alert in alerts] == [10.0, 20.0]
    assert {alert.period_key for alert in alerts} == {"2026-04"}
    start, end, key = tracker.period_window(now=NOW)
    assert start == datetime(2026, 3, 1, tzinfo=UTC)
    assert end == datetime(2026, 4, 1, tzinfo=UTC)
    assert key == "2026-03"


def test_peek_reports_without_consuming_and_reset_rearms() -> None:
    tracker = BudgetAlertTracker([10])

    peeked = tracker.peek(10, now=NOW)

    assert [alert.threshold_usd for alert in peeked] == [10.0]
    assert tracker.emitted == frozenset()
    assert len(tracker.check(10, now=NOW)) == 1
    assert tracker.check(10, now=NOW) == ()

    tracker.reset()

    assert len(tracker.check(10, now=NOW)) == 1


def test_float_noise_at_a_threshold_does_not_hide_an_alert() -> None:
    tracker = BudgetAlertTracker([0.3])

    alerts = tracker.check(0.1 + 0.2, now=NOW)

    assert [alert.threshold_usd for alert in alerts] == [0.3]


def test_alert_properties_and_serialization() -> None:
    alert = BudgetAlertTracker([10]).check(12.5, now=NOW)[0]

    assert isinstance(alert, BudgetAlert)
    assert alert.threshold == 10.0
    assert alert.spent == 12.5
    assert alert.exceeded_by_usd == pytest.approx(2.5)
    assert alert.amount_over == pytest.approx(2.5)
    assert alert.key == ("2026-03-15", 10.0)
    payload = alert.to_dict()
    assert payload["period"] == "day"
    assert payload["threshold_usd"] == 10.0
    assert payload["spent_usd"] == 12.5
    assert payload["exceeded_by_usd"] == pytest.approx(2.5)
    assert payload["period_key"] == "2026-03-15"


def test_tracker_aliases_and_period_aliases() -> None:
    assert AlertPeriod is BudgetAlertPeriod
    assert BudgetAlertState is BudgetAlertTracker
    assert BudgetAlertManager is BudgetAlertTracker
    assert BudgetAlertEmitter is BudgetAlertNotifier
    assert ConsoleAlertNotifier is ConsoleNotifier
    assert ConsoleNotificationProvider is ConsoleNotifier
    assert BudgetAlertTracker([10], period="daily").period is BudgetAlertPeriod.DAY
    assert BudgetAlertTracker([10], period="monthly").period is BudgetAlertPeriod.MONTH
    assert (
        BudgetAlertTracker([10], period=BudgetAlertPeriod.DAY).period_window(now=NOW)[2]
        == "2026-03-15"
    )


def test_invalid_alert_inputs_fail_clearly() -> None:
    with pytest.raises(ConfigError):
        BudgetAlertTracker([-1])
    with pytest.raises(ConfigError):
        BudgetAlertTracker([cast(float, True)])
    with pytest.raises(ConfigError):
        BudgetAlertTracker([cast(float, "10")])
    with pytest.raises(ConfigError):
        BudgetAlertTracker([10], period="week")
    with pytest.raises(ConfigError):
        BudgetAlertTracker([10]).check(-0.01, now=NOW)
    with pytest.raises(ConfigError):
        BudgetAlertTracker([10]).check(1, now=cast(datetime, "now"))
    with pytest.raises(ConfigError):
        BudgetAlertTracker([10], thresholds=[20])
    with pytest.raises(ConfigError):
        BudgetAlertTracker(cast(BudgetConfig, "alerts"))


def test_the_console_notifier_is_a_notifier_and_writes_redacted_lines() -> None:
    stream = io.StringIO()
    notifier = ConsoleNotifier(stream=stream)
    register_secret("supersecretvalue")

    try:
        assert isinstance(notifier, Notifier)
        assert notifier.supports("notify") is True
        receipt = asyncio.run(
            notifier.send(
                NotificationMessage(
                    title="Budget alert",
                    text="spend is high for supersecretvalue",
                    level=NotificationLevel.WARNING,
                )
            )
        )
    finally:
        clear_secrets()

    output = stream.getvalue()
    assert "[WARNING]" in output
    assert "Budget alert" in output
    assert "supersecretvalue" not in output
    assert "[REDACTED]" in output
    assert isinstance(receipt, NotificationReceipt)
    assert receipt.channel == "console"
    assert receipt.provider == "console"
    assert receipt.delivered is True
    assert receipt.delivered_at is not None
    assert receipt.metadata["level"] == "warning"
    assert notifier.sent[0].title == "Budget alert"


def test_console_notifier_renders_media_count() -> None:
    stream = io.StringIO()
    notifier = ConsoleNotifier(stream=stream)

    asyncio.run(
        notifier.send(
            NotificationMessage(
                title="Done",
                media=[NotificationMedia(data=b"png")],
            )
        )
    )

    assert "(1 attachment)" in stream.getvalue()


def test_the_console_helpers_return_the_fallback_notifier() -> None:
    assert isinstance(console_notifier(), ConsoleNotifier)
    assert isinstance(default_notifier(), ConsoleNotifier)


def test_alert_notifier_sends_one_message_per_due_threshold() -> None:
    target = MockNotifier()
    notifier = BudgetAlertNotifier(target, [10, 20])

    receipts = asyncio.run(notifier.emit(15, now=NOW))

    assert len(receipts) == 1
    assert len(target.calls) == 1
    message = cast(NotificationMessage, target.calls[0].request)
    assert message.title == "Budget alert: 10 USD"
    assert message.level is NotificationLevel.WARNING
    assert message.metadata["threshold_usd"] == 10.0
    assert message.metadata["period_key"] == "2026-03-15"

    assert asyncio.run(notifier.emit(15, now=NOW)) == ()
    assert len(target.calls) == 1

    receipts = asyncio.run(notifier.emit(25, now=NOW))

    assert len(receipts) == 1
    assert len(target.calls) == 2
    message = cast(NotificationMessage, target.calls[1].request)
    assert message.metadata["threshold_usd"] == 20.0


def test_alert_notifier_uses_the_console_when_none_is_configured() -> None:
    notifier = BudgetAlertNotifier([10])

    assert isinstance(notifier.notifier, ConsoleNotifier)
    assert notifier.thresholds == (10.0,)
    assert notifier.period is BudgetAlertPeriod.DAY


def test_alert_notifier_accepts_a_shared_tracker_and_rejects_conflicts() -> None:
    tracker = BudgetAlertTracker([10])
    notifier = BudgetAlertNotifier(MockNotifier(), tracker=tracker)

    assert notifier.tracker is tracker
    assert len(asyncio.run(notifier.emit(10, now=NOW))) == 1
    with pytest.raises(ConfigError):
        BudgetAlertNotifier(MockNotifier(), [10], tracker=tracker)
    with pytest.raises(ConfigError):
        BudgetAlertNotifier(cast(Notifier, object()))
    with pytest.raises(ConfigError):
        notifier.message_for(cast(BudgetAlert, "alert"))


def test_module_helper_emits_through_a_shared_tracker() -> None:
    tracker = BudgetAlertTracker([10])
    target = MockNotifier()

    first = asyncio.run(
        emit_budget_alerts(
            12,
            notifier=target,
            tracker=tracker,
            now=NOW,
        )
    )
    second = asyncio.run(
        emit_budget_alerts(
            12,
            notifier=target,
            tracker=tracker,
            now=NOW,
        )
    )

    assert len(first) == 1
    assert second == ()
    assert len(target.calls) == 1


def test_emitted_keys_survive_a_restart_with_the_same_period(
    tmp_path: Path,
) -> None:
    state = tmp_path / "alerts.json"
    first = BudgetAlertTracker([10, 20], state_file=state)

    assert len(first.check(25, now=NOW)) == 2
    assert first.state_file == state
    assert state.exists()

    restarted = BudgetAlertTracker([10, 20], state_file=state)

    assert restarted.emitted == first.emitted
    assert restarted.check(25, now=NOW) == ()

    tomorrow = NOW + timedelta(days=1)
    assert len(restarted.check(25, now=tomorrow)) == 2

    after = BudgetAlertTracker([10, 20], state_file=state)

    assert after.check(25, now=NOW) == ()
    assert after.check(25, now=tomorrow) == ()
    assert ("2026-03-16", 10.0) in after.emitted


def test_emitted_keys_from_past_days_are_pruned_from_the_state_file(
    tmp_path: Path,
) -> None:
    state = tmp_path / "alerts.json"
    tracker = BudgetAlertTracker([10, 20], state_file=state)

    for offset in range(5):
        assert len(tracker.check(25, now=NOW + timedelta(days=offset))) == 2

    current_key = (NOW + timedelta(days=4)).date().isoformat()
    reloaded = BudgetAlertTracker([10, 20], state_file=state)

    assert reloaded.emitted == {(current_key, 10.0), (current_key, 20.0)}
    payload = json.loads(state.read_text(encoding="utf-8"))
    assert {entry["period_key"] for entry in payload["emitted"]} == {current_key}


def test_emitted_keys_from_past_months_are_pruned_from_the_state_file(
    tmp_path: Path,
) -> None:
    state = tmp_path / "alerts.json"
    tracker = BudgetAlertTracker(
        [10, 20],
        period=BudgetAlertPeriod.MONTH,
        state_file=state,
    )

    for month in range(1, 5):
        now = datetime(2026, month, 15, tzinfo=UTC)
        assert len(tracker.check(25, now=now)) == 2

    reloaded = BudgetAlertTracker(
        [10, 20],
        period=BudgetAlertPeriod.MONTH,
        state_file=state,
    )

    assert reloaded.emitted == {("2026-04", 10.0), ("2026-04", 20.0)}


def test_a_check_prunes_stale_keys_already_present_in_the_state_file(
    tmp_path: Path,
) -> None:
    state = tmp_path / "alerts.json"
    state.write_text(
        json.dumps(
            {
                "version": 1,
                "emitted": [
                    {"period_key": "2026-03-14", "threshold_usd": 10.0},
                    {"period_key": "2026-03-15", "threshold_usd": 10.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    tracker = BudgetAlertTracker([10], state_file=state)

    assert tracker.check(10, now=NOW) == ()

    payload = json.loads(state.read_text(encoding="utf-8"))
    assert payload["emitted"] == [{"period_key": "2026-03-15", "threshold_usd": 10.0}]


def test_peek_does_not_write_state_and_reset_clears_it(tmp_path: Path) -> None:
    state = tmp_path / "alerts.json"
    tracker = BudgetAlertTracker([10], state_file=state)

    assert tracker.peek(10, now=NOW)
    assert not state.exists()

    assert len(tracker.check(10, now=NOW)) == 1
    assert state.exists()

    tracker.reset()

    assert tracker.emitted == frozenset()
    assert len(BudgetAlertTracker([10], state_file=state).check(10, now=NOW)) == 1


def test_persistent_tracker_uses_the_data_directory_and_path_alias(
    tmp_path: Path,
) -> None:
    tracker = BudgetAlertTracker.persistent([10], data_dir=tmp_path)

    assert tracker.state_file == tmp_path / "budget_alerts.json"
    assert len(tracker.check(10, now=NOW)) == 1
    assert tracker.state_file is not None
    assert tracker.state_file.exists()

    alias = tmp_path / "alias.json"

    assert BudgetAlertTracker([10], path=alias).state_file == alias
    with pytest.raises(ConfigError):
        BudgetAlertTracker([10], state_file=alias, path=alias)
    with pytest.raises(ConfigError):
        BudgetAlertTracker([10], state_file=cast(Path, 5))


def test_a_corrupt_state_file_fails_clearly(tmp_path: Path) -> None:
    state = tmp_path / "alerts.json"

    state.write_text("{", encoding="utf-8")
    with pytest.raises(ConfigError):
        BudgetAlertTracker([10], state_file=state)

    state.write_text(json.dumps({"version": 99}), encoding="utf-8")
    with pytest.raises(ConfigError):
        BudgetAlertTracker([10], state_file=state)

    state.write_text(
        json.dumps({"emitted": [{"threshold_usd": 10}]}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        BudgetAlertTracker([10], state_file=state)


def test_the_module_helper_keeps_state_across_calls_without_a_tracker(
    tmp_path: Path,
) -> None:
    state = tmp_path / "alerts.json"
    target = MockNotifier()

    first = asyncio.run(
        emit_budget_alerts(
            12,
            notifier=target,
            thresholds=[10],
            state_file=state,
            now=NOW,
        )
    )
    second = asyncio.run(
        emit_budget_alerts(
            12,
            notifier=target,
            thresholds=[10],
            state_file=state,
            now=NOW,
        )
    )

    assert len(first) == 1
    assert second == ()
    assert len(target.calls) == 1


def test_a_stale_tracker_merges_keys_written_by_another_worker(
    tmp_path: Path,
) -> None:
    state = tmp_path / "alerts.json"
    first = BudgetAlertTracker([10, 20], state_file=state)
    second = BudgetAlertTracker([10, 20], state_file=state)

    assert [alert.threshold_usd for alert in first.check(12, now=NOW)] == [10.0]
    assert [alert.threshold_usd for alert in second.check(22, now=NOW)] == [20.0]

    assert first.check(22, now=NOW) == ()
    assert second.check(12, now=NOW) == ()
    reloaded = BudgetAlertTracker([10, 20], state_file=state)

    assert reloaded.emitted == {
        ("2026-03-15", 10.0),
        ("2026-03-15", 20.0),
    }


def test_parallel_workers_do_not_lose_each_others_emitted_keys(
    tmp_path: Path,
) -> None:
    state = tmp_path / "alerts.json"
    thresholds = (10.0, 20.0, 30.0, 40.0)
    trackers = [
        BudgetAlertTracker([threshold], state_file=state) for threshold in thresholds
    ]
    barrier = threading.Barrier(len(trackers))
    emitted: list[int] = []
    errors: list[BaseException] = []
    results_lock = threading.Lock()

    def worker(tracker: BudgetAlertTracker, threshold: float) -> None:
        barrier.wait()
        try:
            alerts = tracker.check(threshold, now=NOW)
        except BaseException as exc:
            with results_lock:
                errors.append(exc)
        else:
            with results_lock:
                emitted.append(len(alerts))

    threads = [
        threading.Thread(target=worker, args=(tracker, threshold))
        for tracker, threshold in zip(trackers, thresholds, strict=True)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert not any(thread.is_alive() for thread in threads)
    assert not errors
    assert emitted == [1, 1, 1, 1]

    reloaded = BudgetAlertTracker(list(thresholds), state_file=state)

    assert reloaded.emitted == {("2026-03-15", threshold) for threshold in thresholds}


def test_budget_guard_reexports_the_alert_api() -> None:
    assert BudgetGuard is not None
    from aivia.core import budget as budget_module

    for name in (
        "BudgetAlert",
        "BudgetAlertNotifier",
        "BudgetAlertPeriod",
        "BudgetAlertTracker",
        "ConsoleNotifier",
        "emit_budget_alerts",
    ):
        assert name in budget_module.__all__

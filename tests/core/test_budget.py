"""Enforce per-call and per-run budget caps before a paid call."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from sqlalchemy import Engine

from aivia.core.budget import (
    BudgetAction,
    BudgetCheck,
    BudgetChecker,
    BudgetController,
    BudgetDecision,
    BudgetEnforcer,
    BudgetGuard,
    BudgetManager,
    BudgetPolicy,
    apply_budget_policy,
    apply_on_exceed,
    check_budget,
    decide_budget,
    enforce_budget,
    ensure_budget,
    resolve_budget,
    would_exceed_budget,
)
from aivia.core.config import BudgetConfig, OnExceed
from aivia.core.cost import CostTracker
from aivia.core.db import (
    CostEventRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.errors import BudgetError, ConfigError
from aivia.core.models.run import Run, RunState


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


@pytest.fixture
def tracker(engine: Engine) -> CostTracker:
    return CostTracker(engine)


def _record(tracker: CostTracker, **overrides: Any) -> Any:
    values: dict[str, Any] = {
        "provider": "acme",
        "model": "image-1",
        "operation": "image_gen",
        "units": 1.0,
        "unit_kind": "image",
        "amount_usd": 0.10,
        "run_id": "run_1",
    }
    values.update(overrides)
    return tracker.record(**values)


def test_per_call_cap_allows_the_cap_and_rejects_anything_above_it(
    tracker: CostTracker,
) -> None:
    guard = BudgetGuard(BudgetConfig(per_call=0.10), tracker)

    allowed = guard.check(0.10, run_id="run_1")

    assert allowed.allowed is True
    assert allowed.remaining_usd == pytest.approx(0.10)
    with pytest.raises(BudgetError) as caught:
        guard.check(0.11, run_id="run_1")

    error = caught.value
    assert error.code == "budget"
    assert error.retryable is False
    assert error.details["scope"] == "per_call"
    assert error.details["cap"] == pytest.approx(0.10)
    assert error.details["amount_usd"] == pytest.approx(0.11)
    assert error.details["spent_usd"] == pytest.approx(0.0)
    assert "per-call cap" in str(error)


def test_per_call_cap_does_not_need_a_cost_ledger() -> None:
    guard = BudgetGuard({"per_call": 0.05})

    assert guard.check(0.05).allowed is True
    with pytest.raises(BudgetError):
        guard.check(0.051)
    assert guard.spent() == 0.0


def test_per_run_cap_adds_the_estimate_to_spend_already_recorded(
    tracker: CostTracker,
) -> None:
    _record(tracker, run_id="run_1", amount_usd=0.30)
    _record(tracker, run_id="run_1", amount_usd=0.10)
    guard = BudgetGuard(BudgetConfig(per_run=0.50), tracker)

    allowed = guard.check(0.10, run_id="run_1")

    assert allowed.allowed is True
    assert allowed.spent_usd == pytest.approx(0.40)
    assert allowed.projected_usd == pytest.approx(0.50)
    assert allowed.remaining_usd == pytest.approx(0.10)
    with pytest.raises(BudgetError) as caught:
        guard.check(0.11, run_id="run_1")

    error = caught.value
    assert error.details["scope"] == "per_run"
    assert error.details["cap"] == pytest.approx(0.50)
    assert error.details["spent_usd"] == pytest.approx(0.40)
    assert error.details["projected_usd"] == pytest.approx(0.51)
    assert "per-run cap" in str(error)


def test_per_run_cap_is_scoped_to_one_run(tracker: CostTracker) -> None:
    _record(tracker, run_id="run_1", amount_usd=0.45)
    guard = BudgetGuard(BudgetConfig(per_run=0.50), tracker)

    assert guard.check(0.05, run_id="run_2").allowed is True
    with pytest.raises(BudgetError):
        guard.check(0.06, run_id="run_1")


def test_a_zero_cap_means_the_cap_is_not_configured(tracker: CostTracker) -> None:
    _record(tracker, run_id="run_1", amount_usd=100.0)
    guard = BudgetGuard(BudgetConfig(), tracker)

    result = guard.check(1_000.0, run_id="run_1")

    assert result.allowed is True
    assert result.remaining_usd is None
    assert guard.remaining("run_1") is None


def test_a_per_run_cap_requires_a_cost_ledger() -> None:
    guard = BudgetGuard({"per_run": 0.50})

    with pytest.raises(ConfigError):
        guard.check(0.10, run_id="run_1")


def test_evaluate_reports_a_rejection_without_raising(tracker: CostTracker) -> None:
    _record(tracker, run_id="run_1", amount_usd=0.40)
    guard = BudgetGuard(BudgetConfig(per_call=0.25, per_run=0.50), tracker)

    result = guard.evaluate(0.30, run_id="run_1")

    assert isinstance(result, BudgetCheck)
    assert result.exceeded is True
    assert result.allowed is False
    assert result.scope == "per_call"
    assert result.to_dict()["scope"] == "per_call"
    assert guard.would_exceed(0.30, run_id="run_1") is True
    assert guard.would_exceed(0.10, run_id="run_1") is False


def test_remaining_applies_the_tightest_active_cap(tracker: CostTracker) -> None:
    _record(tracker, run_id="run_1", amount_usd=0.40)
    guard = BudgetGuard(BudgetConfig(per_call=0.25, per_run=0.50), tracker)

    assert guard.remaining("run_1") == pytest.approx(0.10)

    loose_call = BudgetGuard(BudgetConfig(per_call=0.50, per_run=0.50), tracker)
    assert loose_call.remaining("run_1") == pytest.approx(0.10)

    per_call_only = BudgetGuard(BudgetConfig(per_call=0.30), tracker)
    assert per_call_only.remaining("run_1") == pytest.approx(0.30)


def test_check_does_not_write_a_cost_event(tracker: CostTracker) -> None:
    guard = BudgetGuard(BudgetConfig(per_call=0.10), tracker)

    guard.check(0.05, run_id="run_1")

    assert tracker.events() == []


def test_spend_includes_estimated_and_actual_events(tracker: CostTracker) -> None:
    _record(tracker, run_id="run_1", amount_usd=0.10, estimated=True)
    _record(tracker, run_id="run_1", amount_usd=0.20, estimated=False)
    guard = BudgetGuard(BudgetConfig(per_run=0.40), tracker)

    assert guard.spent("run_1") == pytest.approx(0.30)
    assert guard.check(0.10, run_id="run_1").allowed is True
    with pytest.raises(BudgetError):
        guard.check(0.11, run_id="run_1")


def test_an_omitted_run_reads_events_without_a_run(tracker: CostTracker) -> None:
    _record(tracker, run_id="", amount_usd=0.20)
    guard = BudgetGuard(BudgetConfig(per_run=0.30), tracker)

    assert guard.spent() == pytest.approx(0.20)
    assert guard.check(0.10).allowed is True
    with pytest.raises(BudgetError):
        guard.check(0.11)


@pytest.mark.parametrize("amount", [-0.01, float("nan"), float("inf"), True, "0.1"])
def test_the_estimate_must_be_a_finite_non_negative_number(
    tracker: CostTracker, amount: object
) -> None:
    guard = BudgetGuard(BudgetConfig(per_call=1.0), tracker)

    with pytest.raises(ConfigError):
        guard.check(amount)  # type: ignore[arg-type]


def test_a_check_requires_an_estimate(tracker: CostTracker) -> None:
    guard = BudgetGuard(BudgetConfig(per_call=1.0), tracker)

    with pytest.raises(ConfigError):
        guard.check()


def test_guard_accepts_an_engine_repository_or_tracker(engine: Engine) -> None:
    config = BudgetConfig(per_call=1.0)

    assert BudgetGuard(config, engine).tracker is not None
    assert BudgetGuard(config, CostEventRepository(engine)).tracker is not None
    assert BudgetGuard(config, CostTracker(engine)).tracker is not None
    assert BudgetGuard(config, engine=engine).tracker is not None
    repository = CostEventRepository(engine)
    assert BudgetGuard(config, repository=repository).tracker is not None
    assert BudgetGuard(config, cost=CostTracker(engine)).tracker is not None
    assert BudgetGuard(config, ledger=CostTracker(engine)).tracker is not None


def test_guard_accepts_the_tracker_before_the_config(engine: Engine) -> None:
    tracker = CostTracker(engine)
    guard = BudgetGuard(tracker, BudgetConfig(per_call=0.10))  # type: ignore[arg-type]

    assert guard.tracker is tracker
    assert guard.config.per_call == pytest.approx(0.10)


def test_guard_rejects_a_bad_config_or_ledger(engine: Engine) -> None:
    with pytest.raises(ConfigError):
        BudgetGuard("not a config")  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        BudgetGuard(BudgetConfig(), object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        BudgetGuard(BudgetConfig(), CostTracker(engine), engine=engine)


def test_budget_aliases_and_module_helpers(tracker: CostTracker) -> None:
    assert BudgetEnforcer is BudgetGuard
    assert BudgetChecker is BudgetGuard
    assert BudgetManager is BudgetGuard
    assert BudgetController is BudgetGuard

    config = BudgetConfig(per_call=0.10)
    guard = BudgetGuard(config, tracker)
    assert guard.check_budget(0.05).allowed is True
    assert guard.enforce_budget(0.05).allowed is True
    assert guard.would_exceed_budget(0.20) is True
    assert check_budget(0.05, config=config, tracker=tracker).allowed is True
    assert enforce_budget(0.05, config=config, tracker=tracker).allowed is True
    assert ensure_budget(0.05, config=config, tracker=tracker).allowed is True
    assert would_exceed_budget(0.20, config=config, tracker=tracker) is True
    with pytest.raises(BudgetError):
        enforce_budget(0.20, config=config, tracker=tracker)


def test_a_guard_can_be_reused_across_runs(tracker: CostTracker) -> None:
    guard = BudgetGuard(BudgetConfig(per_run=0.50), tracker)
    _record(tracker, run_id="run_1", amount_usd=0.45)

    with pytest.raises(BudgetError):
        guard.check(0.10, run_id="run_1")
    assert guard.check(0.10, run_id="run_2").allowed is True
    assert guard.check(0.05, run_id="run_1").allowed is True


def test_a_call_inside_the_cap_proceeds_without_a_notification(
    tracker: CostTracker,
) -> None:
    guard = BudgetGuard(BudgetConfig(per_call=0.10), tracker)

    decision = guard.decide(0.10, run_id="run_1", provider="acme")

    assert decision.action is BudgetAction.PROCEED
    assert decision.allowed is True
    assert decision.blocked is False
    assert decision.exceeded is False
    assert decision.provider == "acme"
    assert decision.notifications == ()
    assert decision.behaviour is None
    assert decision.message == "call is within budget"


def test_downgrade_picks_the_cheapest_affordable_cheaper_provider(
    tracker: CostTracker,
) -> None:
    guard = BudgetGuard(
        BudgetConfig(per_call=0.10, on_exceed=OnExceed.DOWNGRADE),
        tracker,
    )

    decision = guard.decide(
        0.20,
        run_id="run_1",
        provider="pricey",
        candidates={
            "pricey": 0.20,
            "mid": 0.08,
            "cheap": 0.04,
            "unaffordable": 0.50,
        },
    )

    assert decision.action is BudgetAction.DOWNGRADE
    assert decision.downgraded is True
    assert decision.allowed is True
    assert decision.blocked is False
    assert decision.behaviour is OnExceed.DOWNGRADE
    assert decision.provider == "cheap"
    assert decision.previous_provider == "pricey"
    assert decision.estimate_usd == pytest.approx(0.04)
    assert decision.check.amount_usd == pytest.approx(0.20)
    assert decision.check.scope == "per_call"
    assert "downgraded from pricey to cheap" in decision.message


def test_downgrade_respects_the_remaining_run_budget(tracker: CostTracker) -> None:
    _record(tracker, run_id="run_1", amount_usd=0.45)
    guard = BudgetGuard(
        BudgetConfig(per_run=0.50, on_exceed=OnExceed.DOWNGRADE),
        tracker,
    )

    decision = guard.decide(
        0.10,
        run_id="run_1",
        provider="pricey",
        candidates={"pricey": 0.10, "almost": 0.09, "fits": 0.05},
    )

    assert decision.downgraded is True
    assert decision.provider == "fits"
    assert decision.estimate_usd == pytest.approx(0.05)
    assert decision.check.remaining_usd == pytest.approx(0.05)


def test_downgrade_without_an_affordable_provider_pauses_the_run(
    tracker: CostTracker,
) -> None:
    _record(tracker, run_id="run_1", amount_usd=0.45)
    run = Run(state=RunState.RUNNING)
    guard = BudgetGuard(
        BudgetConfig(per_run=0.50, on_exceed=OnExceed.DOWNGRADE),
        tracker,
    )

    decision = guard.decide(
        0.10,
        run_id="run_1",
        provider="pricey",
        candidates={"pricey": 0.10, "almost": 0.09, "close": 0.08},
        run=run,
    )

    assert decision.action is BudgetAction.PAUSE
    assert decision.blocked is True
    assert decision.provider == ""
    assert decision.previous_provider == "pricey"
    assert decision.run_state is RunState.PAUSED_BUDGET
    assert decision.paused_run is not None
    assert decision.paused_run is not run
    assert "no cheaper provider" in decision.message


def test_pause_moves_the_run_to_paused_budget_with_a_reason(
    tracker: CostTracker,
) -> None:
    run = Run(state=RunState.RUNNING)
    now = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)
    guard = BudgetGuard(BudgetConfig(per_call=0.10), tracker)

    decision = guard.decide(
        0.20,
        run_id=run.id,
        provider="acme",
        run=run,
        now=now,
    )

    assert decision.action is BudgetAction.PAUSE
    assert decision.paused is True
    assert decision.allowed is False
    assert decision.behaviour is OnExceed.PAUSE
    assert decision.run_state is RunState.PAUSED_BUDGET
    assert decision.run is not None
    assert decision.run.is_budget_paused is True
    assert decision.run.paused_at == now
    assert decision.run.updated_at == now
    assert decision.run.pause_reason.startswith("budget:")
    assert "per-call cap" in decision.run.pause_reason
    assert run.state is RunState.RUNNING
    assert run.paused_at is None


def test_pause_without_a_run_still_reports_the_paused_state(
    tracker: CostTracker,
) -> None:
    guard = BudgetGuard(BudgetConfig(per_call=0.10), tracker)

    decision = guard.decide(0.20, provider="acme")

    assert decision.paused is True
    assert decision.run is None
    assert decision.run_state is RunState.PAUSED_BUDGET
    assert decision.paused_run is None


def test_notify_only_allows_the_call_and_returns_a_notification(
    tracker: CostTracker,
) -> None:
    run = Run(state=RunState.RUNNING)
    guard = BudgetGuard(
        BudgetConfig(per_call=0.10, on_exceed=OnExceed.NOTIFY),
        tracker,
    )

    decision = guard.decide(0.20, run_id="run_1", provider="acme", run=run)

    assert decision.action is BudgetAction.NOTIFY
    assert decision.notified is True
    assert decision.notify is True
    assert decision.allowed is True
    assert decision.blocked is False
    assert decision.behaviour is OnExceed.NOTIFY
    assert len(decision.notifications) == 1
    assert decision.alerts == decision.notifications
    assert "notifying and continuing" in decision.notifications[0]
    assert decision.run_state is RunState.RUNNING
    assert decision.run is run


def test_a_precomputed_check_can_be_passed_to_the_policy(
    tracker: CostTracker,
) -> None:
    guard = BudgetGuard(
        BudgetConfig(per_call=0.10, on_exceed=OnExceed.DOWNGRADE),
        tracker,
    )
    check = guard.evaluate(0.20, run_id="run_1", provider="pricey")

    decision = guard.decide(check=check, candidates={"cheap": 0.05})

    assert isinstance(check, BudgetCheck)
    assert decision.check is check
    assert decision.provider == "cheap"
    assert decision.action is BudgetAction.DOWNGRADE


def test_the_policy_rejects_bad_candidates(tracker: CostTracker) -> None:
    guard = BudgetGuard(
        BudgetConfig(per_call=0.10, on_exceed=OnExceed.DOWNGRADE),
        tracker,
    )

    with pytest.raises(ConfigError):
        guard.decide(0.20, candidates={"cheap": -0.01})
    with pytest.raises(ConfigError):
        guard.decide(0.20, candidates={"": 0.01})
    with pytest.raises(ConfigError):
        guard.decide(0.20, candidates=cast(Any, [("cheap", 0.01, 0.02)]))


def test_policy_aliases_and_module_helpers(tracker: CostTracker) -> None:
    assert BudgetPolicy is BudgetGuard
    config = BudgetConfig(per_call=0.10, on_exceed=OnExceed.NOTIFY)
    guard = BudgetGuard(config, tracker)

    assert guard.resolve(0.20).action is BudgetAction.NOTIFY
    assert guard.on_exceed(0.20).action is BudgetAction.NOTIFY
    assert guard.handle_exceed(0.20).action is BudgetAction.NOTIFY
    assert guard.handle_exceeded(0.20).action is BudgetAction.NOTIFY
    assert guard.apply_on_exceed(0.20).action is BudgetAction.NOTIFY
    assert guard.apply_exceed(0.20).action is BudgetAction.NOTIFY
    assert guard.apply_policy(0.20).action is BudgetAction.NOTIFY
    for decide in (
        decide_budget,
        resolve_budget,
        apply_budget_policy,
        apply_on_exceed,
    ):
        decision = decide(0.20, config=config, tracker=tracker)
        assert isinstance(decision, BudgetDecision)
        assert decision.action is BudgetAction.NOTIFY
    assert guard.decide(0.20).to_dict()["action"] == "notify"


def test_per_day_cap_sums_only_the_current_utc_day(tracker: CostTracker) -> None:
    now = datetime(2026, 3, 15, 12, tzinfo=UTC)
    _record(tracker, amount_usd=0.20, created_at=datetime(2026, 3, 15, 1, tzinfo=UTC))
    _record(
        tracker,
        amount_usd=0.50,
        created_at=datetime(2026, 3, 14, 23, 59, tzinfo=UTC),
    )
    _record(tracker, amount_usd=0.70, created_at=datetime(2026, 3, 16, tzinfo=UTC))
    guard = BudgetGuard(BudgetConfig(per_day=0.30), tracker)

    allowed = guard.check(0.10, now=now)

    assert allowed.allowed is True
    assert allowed.day_spent_usd == pytest.approx(0.20)
    assert allowed.day_projected_usd == pytest.approx(0.30)
    assert allowed.remaining_usd == pytest.approx(0.10)
    with pytest.raises(BudgetError) as caught:
        guard.check(0.11, now=now)

    error = caught.value
    assert error.details["scope"] == "per_day"
    assert error.details["cap"] == pytest.approx(0.30)
    assert error.details["day_spent_usd"] == pytest.approx(0.20)
    assert error.details["day_projected_usd"] == pytest.approx(0.31)
    assert "per-day cap" in str(error)


def test_per_month_cap_sums_only_the_current_utc_month(tracker: CostTracker) -> None:
    now = datetime(2026, 3, 15, 12, tzinfo=UTC)
    _record(tracker, amount_usd=0.20, created_at=datetime(2026, 3, 1, tzinfo=UTC))
    _record(tracker, amount_usd=0.10, created_at=datetime(2026, 3, 15, tzinfo=UTC))
    _record(tracker, amount_usd=0.40, created_at=datetime(2026, 2, 28, tzinfo=UTC))
    _record(tracker, amount_usd=0.50, created_at=datetime(2026, 4, 1, tzinfo=UTC))
    guard = BudgetGuard(BudgetConfig(per_month=0.50), tracker)

    allowed = guard.check(0.20, now=now)

    assert allowed.allowed is True
    assert allowed.month_spent_usd == pytest.approx(0.30)
    assert allowed.month_projected_usd == pytest.approx(0.50)
    assert allowed.remaining_usd == pytest.approx(0.20)
    with pytest.raises(BudgetError) as caught:
        guard.check(0.21, now=now)

    error = caught.value
    assert error.details["scope"] == "per_month"
    assert error.details["cap"] == pytest.approx(0.50)
    assert error.details["month_spent_usd"] == pytest.approx(0.30)
    assert error.details["month_projected_usd"] == pytest.approx(0.51)
    assert "per-month cap" in str(error)


def test_per_provider_monthly_cap_uses_that_providers_current_month_spend(
    tracker: CostTracker,
) -> None:
    now = datetime(2026, 3, 15, 12, tzinfo=UTC)
    _record(
        tracker,
        provider="acme",
        amount_usd=0.20,
        created_at=datetime(2026, 3, 2, tzinfo=UTC),
    )
    _record(
        tracker,
        provider="acme",
        amount_usd=0.50,
        created_at=datetime(2026, 2, 2, tzinfo=UTC),
    )
    _record(
        tracker,
        provider="other",
        amount_usd=0.60,
        created_at=datetime(2026, 3, 2, tzinfo=UTC),
    )
    guard = BudgetGuard(BudgetConfig(per_provider={"acme": 0.30}), tracker)

    allowed = guard.check(0.10, provider="acme", now=now)

    assert allowed.allowed is True
    assert allowed.per_provider_cap == pytest.approx(0.30)
    assert allowed.provider_spent_usd == pytest.approx(0.20)
    assert allowed.provider_projected_usd == pytest.approx(0.30)
    assert allowed.remaining_usd == pytest.approx(0.10)
    with pytest.raises(BudgetError) as caught:
        guard.check(0.11, provider="acme", now=now)

    error = caught.value
    assert error.details["scope"] == "per_provider"
    assert error.details["cap"] == pytest.approx(0.30)
    assert error.details["provider_spent_usd"] == pytest.approx(0.20)
    assert error.details["provider_projected_usd"] == pytest.approx(0.31)
    assert "provider monthly cap" in str(error)

    assert guard.check(0.11, provider="other", now=now).allowed is True
    assert guard.check(0.11, provider="unlimited", now=now).allowed is True


def test_remaining_applies_the_tightest_period_and_provider_cap(
    tracker: CostTracker,
) -> None:
    now = datetime(2026, 3, 15, 12, tzinfo=UTC)
    _record(
        tracker,
        provider="acme",
        amount_usd=0.20,
        created_at=datetime(2026, 3, 15, tzinfo=UTC),
    )
    _record(
        tracker,
        provider="other",
        amount_usd=0.30,
        created_at=datetime(2026, 3, 2, tzinfo=UTC),
    )
    guard = BudgetGuard(
        BudgetConfig(per_day=0.30, per_month=1.00, per_provider={"acme": 0.25}),
        tracker,
    )

    result = guard.evaluate(0.01, provider="acme", now=now)

    assert result.day_spent_usd == pytest.approx(0.20)
    assert result.month_spent_usd == pytest.approx(0.50)
    assert result.provider_spent_usd == pytest.approx(0.20)
    assert result.remaining_usd == pytest.approx(0.05)


@pytest.mark.parametrize(
    "config",
    [
        {"per_day": 1.0},
        {"per_month": 1.0},
        {"per_provider": {"acme": 1.0}},
    ],
)
def test_period_caps_require_a_cost_ledger(config: dict[str, Any]) -> None:
    guard = BudgetGuard(config)

    with pytest.raises(ConfigError):
        guard.check(0.10, provider="acme")


def test_the_policy_pauses_on_a_period_cap(tracker: CostTracker) -> None:
    now = datetime(2026, 3, 15, 12, tzinfo=UTC)
    _record(tracker, amount_usd=0.20, created_at=datetime(2026, 3, 15, tzinfo=UTC))
    run = Run(state=RunState.RUNNING)
    guard = BudgetGuard(BudgetConfig(per_day=0.30), tracker)

    decision = guard.decide(0.20, provider="acme", run=run, now=now)

    assert decision.action is BudgetAction.PAUSE
    assert decision.check.scope == "per_day"
    assert decision.run is not None
    assert "per-day cap" in decision.run.pause_reason


def test_reserve_records_the_estimate_so_the_next_call_sees_it(
    tracker: CostTracker,
) -> None:
    guard = BudgetGuard(BudgetConfig(per_run=0.50), tracker)

    first = guard.reserve(
        0.30,
        run_id="run_1",
        provider="acme",
        operation="image_gen",
    )

    assert first.allowed is True
    assert first.spent_usd == pytest.approx(0.0)
    events = tracker.events(run_id="run_1")
    assert len(events) == 1
    assert events[0].estimated is True
    assert events[0].amount_usd == pytest.approx(0.30)

    with pytest.raises(BudgetError) as caught:
        guard.reserve(0.30, run_id="run_1")

    assert caught.value.details["scope"] == "per_run"
    assert caught.value.details["spent_usd"] == pytest.approx(0.30)
    assert tracker.total(run_id="run_1") == pytest.approx(0.30)


def test_concurrent_reservations_cannot_both_pass_the_same_per_run_cap(
    tmp_path: Path,
) -> None:
    engine = create_database_engine(tmp_path / "aivia.db")
    upgrade_database(engine=engine)
    try:
        tracker = CostTracker(engine)
        guard = BudgetGuard(BudgetConfig(per_run=0.50), tracker)
        workers = 12
        barrier = threading.Barrier(workers)
        outcomes: list[str] = []
        outcomes_lock = threading.Lock()

        def attempt() -> None:
            barrier.wait()
            try:
                guard.reserve(0.10, run_id="run_1")
            except BudgetError:
                outcome = "rejected"
            else:
                outcome = "reserved"
            with outcomes_lock:
                outcomes.append(outcome)

        threads = [threading.Thread(target=attempt) for _ in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        assert not any(thread.is_alive() for thread in threads)

        assert outcomes.count("reserved") == 5
        assert outcomes.count("rejected") == 7
        assert tracker.total(run_id="run_1") == pytest.approx(0.50)
        assert len(tracker.events(run_id="run_1")) == 5
    finally:
        engine.dispose()

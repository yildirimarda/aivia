"""Enforce per-call and per-run budget caps before a paid call."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine

from aivia.core.budget import (
    BudgetCheck,
    BudgetChecker,
    BudgetController,
    BudgetEnforcer,
    BudgetGuard,
    BudgetManager,
    check_budget,
    enforce_budget,
    ensure_budget,
    would_exceed_budget,
)
from aivia.core.config import BudgetConfig
from aivia.core.cost import CostTracker
from aivia.core.db import (
    CostEventRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.errors import BudgetError, ConfigError


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

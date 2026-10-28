"""Reconciling provider-reported actual amounts with pre-call estimates."""

from __future__ import annotations

import math
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import Engine

from aivia.core.config import ReconcileConfig
from aivia.core.cost import CostTracker
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.errors import ConfigError
from aivia.core.models import CostEvent
from aivia.core.reconcile import (
    CostDriftWarning,
    CostReconciler,
    CostReconcileResult,
    CostReconciliation,
    Drift,
    DriftReport,
    Reconciler,
    ReconcileResult,
    Reconciliation,
    actual_amount,
    compare_amounts,
    compare_costs,
    reconcile,
    reconcile_amounts,
    reconcile_costs,
    reported_amount,
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
def tracker(engine: Engine) -> CostTracker:
    return CostTracker(engine)


def _estimate(**overrides: Any) -> CostEvent:
    values: dict[str, Any] = {
        "provider": "acme",
        "model": "image-1",
        "operation": "image_gen",
        "units": 2.0,
        "unit_kind": "image",
        "amount_usd": 1.0,
        "pricing_version": "2026-01",
        "run_id": "run_1",
        "estimated": True,
        "metadata": {"stage": "visuals"},
    }
    values.update(overrides)
    return CostEvent.model_validate(values)


def test_drift_above_the_threshold_is_flagged() -> None:
    result = Reconciler(threshold=0.1).compare(1.0, 1.2)

    assert isinstance(result, Reconciliation)
    assert result.estimate_usd == pytest.approx(1.0)
    assert result.actual_usd == pytest.approx(1.2)
    assert result.difference_usd == pytest.approx(0.2)
    assert result.drift_usd == pytest.approx(0.2)
    assert result.drift_ratio == pytest.approx(0.2)
    assert result.allowed_drift_usd == pytest.approx(0.1)
    assert result.direction == "over"
    assert result.flagged is True
    assert result.drifted is True
    assert result.exceeds_threshold is True
    assert result.within_threshold is False


def test_drift_below_or_at_the_threshold_is_not_flagged() -> None:
    reconciler = Reconciler(threshold=0.1)

    below = reconciler.compare(1.0, 1.05)
    at_limit = reconciler.compare(1.0, 1.1)
    under = reconciler.compare(1.0, 0.8)

    assert below.flagged is False
    assert at_limit.flagged is False
    assert at_limit.drift_usd == pytest.approx(at_limit.allowed_drift_usd)
    assert under.flagged is True
    assert under.difference_usd == pytest.approx(-0.2)
    assert under.direction == "under"


def test_an_absolute_floor_protects_small_estimates() -> None:
    reconciler = Reconciler(threshold=0.1, absolute_threshold_usd=0.5)

    small = reconciler.compare(0.01, 0.02)
    large = reconciler.compare(1.0, 1.6)

    assert small.drift_usd == pytest.approx(0.01)
    assert small.allowed_drift_usd == pytest.approx(0.5)
    assert small.flagged is False
    assert large.allowed_drift_usd == pytest.approx(0.5)
    assert large.flagged is True


def test_a_zero_estimate_has_an_infinite_ratio_when_anything_is_charged() -> None:
    free = Reconciler(threshold=0.1).compare(0.0, 0.0)
    charged = Reconciler(threshold=0.1).compare(0.0, 0.01)

    assert free.flagged is False
    assert free.drift_ratio == 0.0
    assert charged.flagged is True
    assert math.isinf(charged.drift_ratio)
    assert charged.to_dict()["drift_ratio"] is None


def test_events_keep_their_ids_on_the_reconciliation() -> None:
    estimate = _estimate()
    actual = _estimate(estimated=False, amount_usd=1.1)

    result = Reconciler(threshold=0.1).compare(estimate, actual)

    assert result.estimate_event is estimate
    assert result.actual_event is actual
    assert result.estimate_event_id == estimate.id
    assert result.actual_event_id == actual.id
    assert result.flagged is False


def test_a_flagged_comparison_can_emit_a_warning() -> None:
    with pytest.warns(CostDriftWarning, match="drifted") as caught:
        result = reconcile(1.0, 2.0, threshold=0.1, warn=True)

    assert result.flagged is True
    warning = caught[0].message
    assert isinstance(warning, CostDriftWarning)
    assert warning.reconciliation is result


def test_config_defaults_and_aliases_feed_the_reconciler() -> None:
    default = ReconcileConfig.from_mapping({})
    aliased = ReconcileConfig.from_mapping(
        {"threshold": 0.25, "absolute_tolerance_usd": 0.05}
    )

    assert default.drift_threshold == pytest.approx(0.1)
    assert default.absolute_threshold_usd == 0.0
    assert default.threshold == pytest.approx(0.1)
    assert aliased.drift_threshold == pytest.approx(0.25)
    assert aliased.absolute_threshold_usd == pytest.approx(0.05)
    assert aliased.tolerance_usd == pytest.approx(0.05)

    reconciler = Reconciler(default)
    assert reconciler.threshold == pytest.approx(0.1)
    assert reconciler.absolute_threshold_usd == 0.0


@pytest.mark.parametrize(
    "values",
    [
        {"drift_threshold": -0.01},
        {"absolute_threshold_usd": -0.01},
        {"drift_threshold": float("nan")},
        {"absolute_threshold_usd": float("inf")},
        {"unknown": True},
    ],
)
def test_invalid_config_is_rejected(values: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        ReconcileConfig.from_mapping(values)


@pytest.mark.parametrize("threshold", [-0.1, float("nan"), float("inf")])
def test_invalid_explicit_thresholds_are_rejected(threshold: float) -> None:
    with pytest.raises(ValueError):
        Reconciler(threshold=threshold)


def test_reported_amount_reads_the_usual_response_shapes() -> None:
    assert reported_amount(0.25) == pytest.approx(0.25)
    assert reported_amount({"cost_usd": 0.25}) == pytest.approx(0.25)
    assert reported_amount({"amount": {"usd": 0.25}}) == pytest.approx(0.25)
    assert reported_amount({"metadata": {"actual_cost_usd": 0.25}}) == pytest.approx(
        0.25
    )
    assert reported_amount({"usage": {"cost": 0.25}}) == pytest.approx(0.25)
    assert reported_amount(SimpleNamespace(cost_usd=0.25)) == pytest.approx(0.25)
    assert reported_amount(
        SimpleNamespace(metadata={"amount_usd": 0.25})
    ) == pytest.approx(0.25)
    assert actual_amount({"total_cost_usd": 0.25}) == pytest.approx(0.25)


def test_a_response_without_a_reported_amount_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not report"):
        reported_amount({"status": "ok"})
    with pytest.raises(ValueError, match="does not report"):
        reported_amount(SimpleNamespace(status="ok"))
    with pytest.raises(ValueError):
        reported_amount(-0.01)


def test_record_actual_writes_an_actual_event_and_flags_the_drift(
    tracker: CostTracker,
) -> None:
    estimate = _estimate(amount_usd=1.0)

    result = Reconciler(threshold=0.1).record_actual(
        tracker,
        estimate,
        {"metadata": {"cost_usd": 1.5, "request_id": "req_1"}},
        metadata={"stage": "visuals", "attempt": 1},
    )

    stored = result.actual_event
    assert stored is not None
    assert stored.id == result.actual_event_id
    assert stored.id != estimate.id
    assert stored.provider == "acme"
    assert stored.model == "image-1"
    assert stored.operation == "image_gen"
    assert stored.units == 2.0
    assert stored.unit_kind == "image"
    assert stored.pricing_version == "2026-01"
    assert stored.run_id == "run_1"
    assert stored.amount_usd == pytest.approx(1.5)
    assert stored.estimated is False
    assert stored.amount_kind == "actual"
    assert stored.metadata["stage"] == "visuals"
    assert stored.metadata["attempt"] == 1
    assert stored.metadata["estimate_usd"] == pytest.approx(1.0)
    assert stored.metadata["drift_usd"] == pytest.approx(0.5)
    assert stored.metadata["drift"] is True
    assert stored.metadata["drift_flagged"] is True
    assert tracker.repository.get(stored.id) == stored
    assert result.flagged is True


def test_record_actual_accepts_an_extracted_amount_and_overrides_fields(
    tracker: CostTracker,
) -> None:
    result = Reconciler().record_actual(
        tracker,
        _estimate(amount_usd=1.0),
        amount_usd=1.01,
        provider="other",
        operation="image_edit",
    )

    assert result.flagged is False
    assert result.actual_event is not None
    assert result.actual_event.provider == "other"
    assert result.actual_event.operation == "image_edit"
    assert result.actual_event.amount_usd == pytest.approx(1.01)


def test_record_actual_requires_an_amount(tracker: CostTracker) -> None:
    with pytest.raises(ValueError, match="reported amount"):
        Reconciler().record_actual(tracker, 1.0)
    with pytest.raises(ValueError):
        Reconciler().record_actual(tracker, 1.0, amount_usd=-0.1)


def test_reconcile_run_pairs_estimates_and_actuals_in_order(
    tracker: CostTracker,
) -> None:
    first_estimate = tracker.record_estimate(
        provider="acme",
        operation="image_gen",
        amount_usd=1.0,
        run_id="run_1",
    )
    tracker.record_actual(
        provider="acme",
        operation="image_gen",
        amount_usd=1.2,
        run_id="run_1",
    )
    second_estimate = tracker.record_estimate(
        provider="acme",
        operation="image_gen",
        amount_usd=1.0,
        run_id="run_1",
    )
    tracker.record_actual(
        provider="acme",
        operation="image_gen",
        amount_usd=1.05,
        run_id="run_1",
    )

    results = Reconciler(threshold=0.1).reconcile_run(tracker, "run_1")

    assert len(results) == 2
    assert results[0].estimate_event_id == first_estimate.id
    assert results[0].flagged is True
    assert results[1].estimate_event_id == second_estimate.id
    assert results[1].flagged is False


def test_reconcile_run_skips_unpaired_events(tracker: CostTracker) -> None:
    tracker.record_estimate(
        provider="acme",
        operation="image_gen",
        amount_usd=1.0,
        run_id="run_1",
    )
    tracker.record_actual(
        provider="acme",
        operation="video_gen",
        amount_usd=2.0,
        run_id="run_1",
    )

    assert Reconciler().reconcile_run(tracker, "run_1") == []


def test_public_aliases_point_at_the_same_types_and_functions() -> None:
    assert CostReconciler is Reconciler
    assert ReconcileResult is Reconciliation
    assert CostReconciliation is Reconciliation
    assert CostReconcileResult is Reconciliation
    assert DriftReport is Reconciliation
    assert Drift is Reconciliation
    assert reconcile_amounts is reconcile
    assert reconcile_costs is reconcile
    assert compare_amounts is reconcile
    assert compare_costs is reconcile

"""Recording a cost event for every paid call."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy import Engine

from aivia.core.cost import (
    CostEventRecorder,
    CostLedger,
    CostRecorder,
    CostTracker,
)
from aivia.core.db import (
    CostEventRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.models import CostEvent, CostKind


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


def _record(tracker: CostTracker, **overrides: Any) -> CostEvent:
    values: dict[str, Any] = {
        "provider": "acme",
        "model": "image-1",
        "operation": "image_gen",
        "units": 2.0,
        "unit_kind": "image",
        "amount_usd": 0.08,
        "run_id": "run_1",
        "pricing_version": "2026-01",
        "metadata": {"stage": "visuals"},
    }
    values.update(overrides)
    return tracker.record(**values)


def test_record_persists_every_paid_call_field(tracker: CostTracker) -> None:
    event = _record(tracker)

    assert event.id.startswith("cost_")
    assert event.provider == "acme"
    assert event.model == "image-1"
    assert event.operation == "image_gen"
    assert event.units == 2.0
    assert event.unit_kind == "image"
    assert event.amount_usd == 0.08
    assert event.pricing_version == "2026-01"
    assert event.run_id == "run_1"
    assert event.estimated is False
    assert event.amount_kind == CostKind.ACTUAL
    assert event.metadata == {"stage": "visuals"}

    stored = tracker.repository.get(event.id)
    assert stored == event


def test_every_paid_call_gets_its_own_event(tracker: CostTracker) -> None:
    first = _record(tracker)
    second = _record(tracker, operation="video_gen", units=5.0, amount_usd=0.40)

    assert first.id != second.id
    assert first.id < second.id
    assert tracker.count() == 2
    assert [event.id for event in tracker.events()] == [first.id, second.id]
    assert tracker.total() == pytest.approx(0.48)


def test_an_estimate_is_flagged_as_estimated(tracker: CostTracker) -> None:
    event = tracker.record_estimate(
        provider="acme",
        operation="image_gen",
        units=1.0,
        amount_usd=0.04,
        pricing_version="2026-01",
    )

    assert event.estimated is True
    assert event.is_estimated is True
    assert event.is_actual is False
    assert event.amount_kind == CostKind.ESTIMATED
    assert tracker.repository.get(event.id) == event


def test_an_actual_amount_is_flagged_as_actual(tracker: CostTracker) -> None:
    event = tracker.record_actual(
        provider="acme",
        operation="image_gen",
        units=1.0,
        amount_usd=0.05,
        pricing_version="2026-01",
    )

    assert event.estimated is False
    assert event.is_actual is True
    assert event.amount_kind == CostKind.ACTUAL


def test_record_paid_call_chooses_the_flag(tracker: CostTracker) -> None:
    estimate = tracker.record_paid_call(
        provider="acme",
        operation="tts",
        units=120.0,
        unit_kind="character",
        amount_usd=0.12,
        pricing_version="2026-02",
        estimated=True,
    )
    actual = tracker.record_paid_call(
        provider="acme",
        operation="tts",
        units=120.0,
        unit_kind="character",
        amount_usd=0.11,
        pricing_version="2026-02",
    )

    assert estimate.estimated is True
    assert actual.estimated is False


def test_record_accepts_model_aliases(tracker: CostTracker) -> None:
    event = tracker.record(
        provider_name=" acme ",
        capability=" image_gen ",
        quantity=3,
        unit="image",
        cost_usd=0.12,
        run="run_2",
        price_version="2026-03",
        amount_kind="estimated",
        details={"stage": "visuals"},
    )

    assert event.provider == "acme"
    assert event.operation == "image_gen"
    assert event.units == 3.0
    assert event.unit_kind == "image"
    assert event.amount_usd == 0.12
    assert event.run_id == "run_2"
    assert event.pricing_version == "2026-03"
    assert event.estimated is True
    assert event.metadata == {"stage": "visuals"}


def test_record_accepts_a_cost_event_and_keyword_overrides(
    tracker: CostTracker,
) -> None:
    original = CostEvent(
        provider="acme",
        operation="image_gen",
        units=1.0,
        amount_usd=0.04,
        pricing_version="2026-01",
    )

    stored = tracker.record(original, amount_usd=0.05, estimated=True)

    assert stored.id == original.id
    assert stored.amount_usd == 0.05
    assert stored.estimated is True
    assert tracker.repository.get(original.id) == stored


def test_events_filter_by_run_provider_operation_and_flag(tracker: CostTracker) -> None:
    first = _record(tracker, run_id="run_1", provider="acme", operation="image_gen")
    second = _record(
        tracker,
        run_id="run_1",
        provider="other",
        operation="video_gen",
        estimated=True,
    )
    third = _record(tracker, run_id="run_2", provider="acme", operation="image_gen")

    assert [event.id for event in tracker.events(run_id="run_1")] == [
        first.id,
        second.id,
    ]
    assert [event.id for event in tracker.events(provider="acme")] == [
        first.id,
        third.id,
    ]
    assert [event.id for event in tracker.events(operation="video_gen")] == [second.id]
    assert [event.id for event in tracker.events(estimated=True)] == [second.id]
    assert [event.id for event in tracker.events(estimated=False)] == [
        first.id,
        third.id,
    ]
    assert [event.id for event in tracker.for_run("run_2")] == [third.id]


def test_events_support_limit_and_offset(tracker: CostTracker) -> None:
    first = _record(tracker)
    second = _record(tracker)
    third = _record(tracker)

    assert [event.id for event in tracker.events(limit=2)] == [first.id, second.id]
    assert [event.id for event in tracker.events(limit=1, offset=2)] == [third.id]


def test_totals_split_estimates_from_actuals(tracker: CostTracker) -> None:
    _record(tracker, run_id="run_1", amount_usd=0.10, estimated=False)
    _record(tracker, run_id="run_1", amount_usd=0.20, estimated=True)
    _record(tracker, run_id="run_2", amount_usd=0.30, estimated=False)

    assert tracker.total() == pytest.approx(0.60)
    assert tracker.total(run_id="run_1") == pytest.approx(0.30)
    assert tracker.estimated_total(run_id="run_1") == pytest.approx(0.20)
    assert tracker.actual_total(run_id="run_1") == pytest.approx(0.10)
    assert tracker.count(run_id="run_1") == 2
    assert tracker.totals(run_id="run_1") == {
        "total": pytest.approx(0.30),
        "estimated": pytest.approx(0.20),
        "actual": pytest.approx(0.10),
        "count": 2,
    }


def test_an_empty_ledger_totals_to_zero(tracker: CostTracker) -> None:
    assert tracker.events() == []
    assert tracker.total() == 0.0
    assert tracker.count() == 0
    assert tracker.totals() == {
        "total": 0.0,
        "estimated": 0.0,
        "actual": 0.0,
        "count": 0,
    }


def test_an_actual_replacement_is_not_counted_twice_in_totals(
    tracker: CostTracker,
) -> None:
    estimate = tracker.record_estimate(
        provider="acme",
        model="image-1",
        operation="image_gen",
        units=2.0,
        unit_kind="image",
        amount_usd=0.20,
        run_id="run_1",
    )
    actual = tracker.record_actual(
        provider="acme",
        model="image-1",
        operation="image_gen",
        units=2.0,
        unit_kind="image",
        amount_usd=0.18,
        run_id="run_1",
    )

    assert actual.replaces == estimate.id
    assert actual.metadata["estimate_event_id"] == estimate.id
    assert tracker.count() == 2
    assert tracker.total(run_id="run_1") == pytest.approx(0.18)
    assert tracker.estimated_total(run_id="run_1") == pytest.approx(0.0)
    assert tracker.actual_total(run_id="run_1") == pytest.approx(0.18)
    assert tracker.totals(run_id="run_1") == {
        "total": pytest.approx(0.18),
        "estimated": pytest.approx(0.0),
        "actual": pytest.approx(0.18),
        "count": 1,
    }


def test_an_explicit_replacement_link_excludes_the_estimate(
    tracker: CostTracker,
) -> None:
    estimate = _record(tracker, amount_usd=0.20, estimated=True)
    actual = _record(
        tracker,
        amount_usd=0.25,
        estimated=False,
        replaces=estimate.id,
    )

    assert actual.replaces == estimate.id
    assert tracker.total() == pytest.approx(0.25)
    assert tracker.estimated_total() == pytest.approx(0.0)
    assert tracker.actual_total() == pytest.approx(0.25)


def test_replacements_pair_estimates_and_actuals_in_ledger_order(
    tracker: CostTracker,
) -> None:
    first_estimate = _record(tracker, amount_usd=0.10, estimated=True)
    second_estimate = _record(tracker, amount_usd=0.20, estimated=True)

    first_actual = tracker.record_actual(
        provider="acme",
        model="image-1",
        operation="image_gen",
        units=2.0,
        unit_kind="image",
        amount_usd=0.11,
        run_id="run_1",
    )
    second_actual = tracker.record_actual(
        provider="acme",
        model="image-1",
        operation="image_gen",
        units=2.0,
        unit_kind="image",
        amount_usd=0.19,
        run_id="run_1",
    )

    assert first_actual.replaces == first_estimate.id
    assert second_actual.replaces == second_estimate.id
    assert tracker.total() == pytest.approx(0.30)
    assert tracker.estimated_total() == pytest.approx(0.0)


def test_an_estimate_without_an_actual_still_counts(tracker: CostTracker) -> None:
    estimate = _record(tracker, operation="video_gen", amount_usd=0.30, estimated=True)
    _record(tracker, operation="image_gen", amount_usd=0.10, estimated=False)

    assert tracker.total() == pytest.approx(0.40)
    assert tracker.estimated_total() == pytest.approx(0.30)
    assert estimate.replaces == ""


def test_tracker_can_be_built_from_a_repository(engine: Engine) -> None:
    repository = CostEventRepository(engine)
    tracker = CostTracker(repository=repository)

    event = tracker.record(
        provider="acme",
        operation="image_gen",
        units=1.0,
        amount_usd=0.04,
        pricing_version="2026-01",
    )

    assert tracker.repository is repository
    assert repository.get(event.id) == event


def test_tracker_rejects_a_missing_or_conflicting_target(engine: Engine) -> None:
    repository = CostEventRepository(engine)

    with pytest.raises(TypeError):
        CostTracker()
    with pytest.raises(TypeError):
        CostTracker(engine, repository=repository)
    with pytest.raises(TypeError):
        CostTracker(object())  # type: ignore[arg-type]


def test_record_rejects_an_empty_call(tracker: CostTracker) -> None:
    with pytest.raises(ValueError):
        tracker.record()
    with pytest.raises(TypeError):
        tracker.record("acme")  # type: ignore[arg-type]


def test_record_rejects_invalid_amounts(tracker: CostTracker) -> None:
    with pytest.raises(ValidationError):
        tracker.record(provider="acme", operation="image_gen", amount_usd=-0.01)
    with pytest.raises(ValidationError):
        tracker.record(provider="acme", operation="image_gen", amount_usd=float("nan"))
    with pytest.raises(ValidationError):
        tracker.record(provider="acme", operation="image_gen", units=float("inf"))


def test_tracker_aliases_are_exported() -> None:
    assert CostLedger is CostTracker
    assert CostRecorder is CostTracker
    assert CostEventRecorder is CostTracker


def test_kinds_are_the_accepted_flag_values(tracker: CostTracker) -> None:
    assert tracker.kinds == (CostKind.ESTIMATED, CostKind.ACTUAL)

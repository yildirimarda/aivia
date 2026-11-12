"""A pricing drift job warns and prints suggested overrides without writing them.

The job reads pre-call estimates and provider-reported actuals from the cost
ledger, reconciles each pair, warns when the drift crosses the threshold and
prints a suggested per-provider override. Nothing is written to config or the
ledger.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from io import StringIO

import pytest
from sqlalchemy import Engine

from aivia.analytics.pricing_drift import (
    DriftSuggestion,
    PricingDriftJob,
    check_pricing_drift,
)
from aivia.core.cost import CostTracker
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.reconcile import CostDriftWarning

RUN_ID = "run_1"
PROVIDER = "acme"
MODEL = "image-1"
OPERATION = "image_gen"
UNIT_KIND = "image"


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


def _record_pair(
    tracker: CostTracker,
    *,
    provider: str = PROVIDER,
    estimate_usd: float = 1.0,
    actual_usd: float = 1.5,
    units: float = 1.0,
    unit_kind: str = UNIT_KIND,
    run_id: str = RUN_ID,
) -> None:
    tracker.record_estimate(
        provider=provider,
        model=MODEL,
        operation=OPERATION,
        units=units,
        unit_kind=unit_kind,
        amount_usd=estimate_usd,
        pricing_version="2026-01",
        run_id=run_id,
    )
    tracker.record_actual(
        provider=provider,
        model=MODEL,
        operation=OPERATION,
        units=units,
        unit_kind=unit_kind,
        amount_usd=actual_usd,
        pricing_version="2026-01",
        run_id=run_id,
    )


def test_drift_warns_and_prints_suggested_override_without_writing(
    tracker: CostTracker,
) -> None:
    _record_pair(tracker, estimate_usd=1.0, actual_usd=1.5, units=1.0)
    before = tracker.count(run_id=RUN_ID)

    buffer = StringIO()
    with pytest.warns(CostDriftWarning, match="drifted"):
        suggestions = PricingDriftJob(tracker, file=buffer).check_run(RUN_ID)

    after = tracker.count(run_id=RUN_ID)
    assert after == before  # nothing written

    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert isinstance(suggestion, DriftSuggestion)
    assert suggestion.provider == PROVIDER
    assert suggestion.estimate_usd == pytest.approx(1.0)
    assert suggestion.actual_usd == pytest.approx(1.5)
    assert suggestion.drift_usd == pytest.approx(0.5)
    assert suggestion.suggested_override_usd == pytest.approx(1.5)
    assert suggestion.units == pytest.approx(1.0)
    assert suggestion.unit_kind == UNIT_KIND

    output = buffer.getvalue()
    assert "suggested overrides (not written)" in output
    assert PROVIDER in output
    assert "1.500000" in output


def test_suggested_override_is_actual_per_unit(tracker: CostTracker) -> None:
    _record_pair(tracker, estimate_usd=2.0, actual_usd=3.0, units=2.0)
    buffer = StringIO()
    with pytest.warns(CostDriftWarning):
        suggestions = PricingDriftJob(tracker, file=buffer).check_run(RUN_ID)

    assert len(suggestions) == 1
    # 3.0 USD for 2.0 images -> 1.5 USD per image
    assert suggestions[0].suggested_override_usd == pytest.approx(1.5)


def test_zero_units_falls_back_to_actual_amount(tracker: CostTracker) -> None:
    _record_pair(tracker, estimate_usd=1.0, actual_usd=1.5, units=0.0)
    buffer = StringIO()
    with pytest.warns(CostDriftWarning):
        suggestions = PricingDriftJob(tracker, file=buffer).check_run(RUN_ID)

    assert len(suggestions) == 1
    assert suggestions[0].suggested_override_usd == pytest.approx(1.5)


def test_no_drift_no_warning_and_no_print(tracker: CostTracker) -> None:
    _record_pair(tracker, estimate_usd=1.0, actual_usd=1.05, units=1.0)
    buffer = StringIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        suggestions = PricingDriftJob(tracker, file=buffer).check_run(RUN_ID)

    assert suggestions == []
    assert caught == []
    assert buffer.getvalue() == ""


def test_multiple_providers_each_get_a_suggestion(tracker: CostTracker) -> None:
    _record_pair(tracker, provider="acme", estimate_usd=1.0, actual_usd=2.0, units=1.0)
    _record_pair(
        tracker,
        provider="fal",
        estimate_usd=1.0,
        actual_usd=1.3,
        units=1.0,
        run_id=RUN_ID,
    )
    buffer = StringIO()
    with pytest.warns(CostDriftWarning):
        suggestions = PricingDriftJob(tracker, file=buffer).check_run(RUN_ID)

    providers = {suggestion.provider for suggestion in suggestions}
    assert providers == {"acme", "fal"}
    assert len(suggestions) == 2
    output = buffer.getvalue()
    assert "acme" in output and "fal" in output


def test_print_disabled_still_returns_suggestions(tracker: CostTracker) -> None:
    _record_pair(tracker, estimate_usd=1.0, actual_usd=2.0, units=1.0)
    buffer = StringIO()
    with pytest.warns(CostDriftWarning):
        suggestions = PricingDriftJob(
            tracker, print_suggestions=False, file=buffer
        ).check_run(RUN_ID)

    assert len(suggestions) == 1
    assert suggestions[0].suggested_override_usd == pytest.approx(2.0)
    assert buffer.getvalue() == ""


def test_warn_disabled_emits_no_warning(tracker: CostTracker) -> None:
    _record_pair(tracker, estimate_usd=1.0, actual_usd=2.0, units=1.0)
    buffer = StringIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        suggestions = PricingDriftJob(tracker, warn=False, file=buffer).check_run(
            RUN_ID
        )

    assert len(suggestions) == 1
    assert caught == []


def test_check_pricing_drift_convenience_function(tracker: CostTracker) -> None:
    _record_pair(tracker, estimate_usd=1.0, actual_usd=2.0, units=1.0)
    with pytest.warns(CostDriftWarning):
        suggestions = check_pricing_drift(tracker, RUN_ID)

    assert len(suggestions) == 1
    assert suggestions[0].provider == PROVIDER


def test_unknown_run_returns_nothing_and_prints_nothing(tracker: CostTracker) -> None:
    buffer = StringIO()
    suggestions = PricingDriftJob(tracker, file=buffer).check_run("no-such-run")
    assert suggestions == []
    assert buffer.getvalue() == ""

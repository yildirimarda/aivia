"""SQL-level cost aggregation by run, time bucket and spend dimension."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime
from typing import Any

import pytest
from sqlalchemy import Engine, event

from aivia.core.cost import CostAggregator, CostGroup, CostTracker
from aivia.core.cost_agg import CostAggregate
from aivia.core.db import (
    CostEventRepository,
    RunRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.models import Run


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


def _run(engine: Engine, profile: str, format_name: str) -> Run:
    return RunRepository(engine).create(Run(profile=profile, format=format_name))


def _record(tracker: CostTracker, **overrides: Any) -> Any:
    values: dict[str, Any] = {
        "provider": "acme",
        "model": "image-1",
        "operation": "image_gen",
        "units": 1.0,
        "unit_kind": "image",
        "amount_usd": 0.10,
        "run_id": "",
        "estimated": False,
        "created_at": datetime(2026, 1, 5, 10, 0, tzinfo=UTC),
    }
    values.update(overrides)
    return tracker.record(**values)


def _by_key(buckets: list[CostAggregate]) -> dict[str, CostAggregate]:
    return {bucket.key: bucket for bucket in buckets}


def test_aggregate_by_run_sums_estimates_actuals_units_and_count(
    engine: Engine, tracker: CostTracker
) -> None:
    first_run = _run(engine, "daily", "short")
    second_run = _run(engine, "weekly", "long")
    _record(tracker, run_id=first_run.id, amount_usd=0.10, units=2.0, estimated=False)
    _record(tracker, run_id=first_run.id, amount_usd=0.20, units=3.0, estimated=True)
    _record(tracker, run_id=second_run.id, amount_usd=0.40, units=5.0, estimated=False)

    buckets = CostAggregator(engine).by_run()

    assert [bucket.key for bucket in buckets] == sorted([first_run.id, second_run.id])
    by_key = _by_key(buckets)
    first = by_key[first_run.id]
    assert first.group is CostGroup.RUN
    assert first.total_usd == pytest.approx(0.30)
    assert first.estimated_usd == pytest.approx(0.20)
    assert first.actual_usd == pytest.approx(0.10)
    assert first.units == pytest.approx(5.0)
    assert first.count == 2
    assert by_key[second_run.id].total_usd == pytest.approx(0.40)
    assert by_key[second_run.id].count == 1


def test_aggregate_by_provider_model_and_stage(
    engine: Engine, tracker: CostTracker
) -> None:
    _record(
        tracker,
        provider="acme",
        model="image-1",
        operation="image_gen",
        amount_usd=0.10,
        units=2.0,
    )
    _record(
        tracker,
        provider="acme",
        model="video-1",
        operation="video_gen",
        amount_usd=0.30,
        units=4.0,
    )
    _record(
        tracker,
        provider="other",
        model="image-1",
        operation="image_gen",
        amount_usd=0.20,
        units=1.0,
    )
    aggregator = CostAggregator(engine)

    providers = _by_key(aggregator.by_provider())
    assert providers["acme"].total_usd == pytest.approx(0.40)
    assert providers["acme"].units == pytest.approx(6.0)
    assert providers["acme"].count == 2
    assert providers["other"].total_usd == pytest.approx(0.20)

    models = _by_key(aggregator.by_model())
    assert models["image-1"].total_usd == pytest.approx(0.30)
    assert models["image-1"].count == 2
    assert models["video-1"].total_usd == pytest.approx(0.30)

    stages = _by_key(aggregator.by_stage())
    assert stages["image_gen"].total_usd == pytest.approx(0.30)
    assert stages["image_gen"].units == pytest.approx(3.0)
    assert stages["image_gen"].count == 2
    assert stages["video_gen"].total_usd == pytest.approx(0.30)


def test_aggregate_by_day_week_and_month(engine: Engine, tracker: CostTracker) -> None:
    _record(tracker, amount_usd=0.10, created_at=datetime(2026, 1, 5, 10, tzinfo=UTC))
    _record(tracker, amount_usd=0.20, created_at=datetime(2026, 1, 5, 15, tzinfo=UTC))
    _record(tracker, amount_usd=0.30, created_at=datetime(2026, 1, 12, 9, tzinfo=UTC))
    _record(tracker, amount_usd=0.40, created_at=datetime(2026, 2, 2, 0, tzinfo=UTC))
    aggregator = CostAggregator(engine)

    days = aggregator.by_day()
    assert [bucket.key for bucket in days] == [
        "2026-01-05",
        "2026-01-12",
        "2026-02-02",
    ]
    assert [bucket.total_usd for bucket in days] == pytest.approx([0.30, 0.30, 0.40])
    assert days[0].count == 2

    weeks = aggregator.by_week()
    assert [bucket.key for bucket in weeks] == [
        "2026-W01",
        "2026-W02",
        "2026-W05",
    ]
    assert [bucket.total_usd for bucket in weeks] == pytest.approx([0.30, 0.30, 0.40])

    months = aggregator.by_month()
    assert [bucket.key for bucket in months] == ["2026-01", "2026-02"]
    assert [bucket.total_usd for bucket in months] == pytest.approx([0.60, 0.40])
    assert months[0].count == 3


def test_aggregate_by_profile_and_format_joins_the_run(
    engine: Engine, tracker: CostTracker
) -> None:
    daily = _run(engine, "daily", "short")
    weekly = _run(engine, "weekly", "long")
    _record(tracker, run_id=daily.id, amount_usd=0.10)
    _record(tracker, run_id=daily.id, amount_usd=0.20)
    _record(tracker, run_id=weekly.id, amount_usd=0.40)
    _record(tracker, run_id="", amount_usd=0.05)
    aggregator = CostAggregator(engine)

    profiles = _by_key(aggregator.by_profile())
    assert sorted(profiles) == ["", "daily", "weekly"]
    assert profiles[""].total_usd == pytest.approx(0.05)
    assert profiles["daily"].total_usd == pytest.approx(0.30)
    assert profiles["daily"].count == 2
    assert profiles["weekly"].total_usd == pytest.approx(0.40)
    assert profiles["daily"].group is CostGroup.PROFILE

    formats = _by_key(aggregator.by_format())
    assert sorted(formats) == ["", "long", "short"]
    assert formats[""].total_usd == pytest.approx(0.05)
    assert formats["short"].total_usd == pytest.approx(0.30)
    assert formats["long"].total_usd == pytest.approx(0.40)
    assert formats["long"].group is CostGroup.FORMAT


def test_aggregate_filters_by_date_range_and_columns(
    engine: Engine, tracker: CostTracker
) -> None:
    _record(
        tracker,
        provider="acme",
        model="image-1",
        operation="image_gen",
        amount_usd=0.10,
        created_at=datetime(2026, 1, 5, 10, tzinfo=UTC),
    )
    _record(
        tracker,
        provider="acme",
        model="image-1",
        operation="image_gen",
        amount_usd=0.20,
        estimated=True,
        created_at=datetime(2026, 1, 12, 10, tzinfo=UTC),
    )
    _record(
        tracker,
        provider="other",
        model="video-1",
        operation="video_gen",
        amount_usd=0.40,
        created_at=datetime(2026, 1, 12, 12, tzinfo=UTC),
    )
    aggregator = CostAggregator(engine)

    same_day = aggregator.by_day(start=date(2026, 1, 12), end=date(2026, 1, 12))
    assert [bucket.key for bucket in same_day] == ["2026-01-12"]
    assert same_day[0].total_usd == pytest.approx(0.60)
    assert same_day[0].count == 2

    from_start = aggregator.by_day(start=datetime(2026, 1, 6, tzinfo=UTC))
    assert [bucket.key for bucket in from_start] == ["2026-01-12"]

    acme_stages = _by_key(aggregator.by_stage(provider="acme"))
    assert sorted(acme_stages) == ["image_gen"]
    assert acme_stages["image_gen"].total_usd == pytest.approx(0.30)

    estimated_only = _by_key(aggregator.by_provider(estimated=True))
    assert sorted(estimated_only) == ["acme"]
    assert estimated_only["acme"].total_usd == pytest.approx(0.20)

    video_only = _by_key(aggregator.by_model(stage="video_gen"))
    assert sorted(video_only) == ["video-1"]

    run_scoped = aggregator.by_stage(run_id="missing")
    assert run_scoped == []


def test_aggregate_returns_nothing_for_an_empty_ledger(engine: Engine) -> None:
    assert CostAggregator(engine).by_provider() == []
    assert CostAggregator(engine).by_day() == []


def test_aggregate_rejects_an_unknown_group(engine: Engine) -> None:
    aggregator = CostAggregator(engine)

    with pytest.raises(ValueError, match="unknown cost group"):
        aggregator.aggregate("asset")


def test_aggregator_can_be_built_from_a_repository(engine: Engine) -> None:
    repository = CostEventRepository(engine)
    aggregator = CostAggregator(repository=repository)

    assert aggregator.engine is engine
    assert aggregator.by_provider() == []


def test_aggregator_rejects_a_missing_or_conflicting_target(engine: Engine) -> None:
    repository = CostEventRepository(engine)

    with pytest.raises(TypeError):
        CostAggregator()
    with pytest.raises(TypeError):
        CostAggregator(engine, repository=repository)
    with pytest.raises(TypeError):
        CostAggregator(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("method", "group"),
    [
        ("by_run", CostGroup.RUN),
        ("by_day", CostGroup.DAY),
        ("by_week", CostGroup.WEEK),
        ("by_month", CostGroup.MONTH),
        ("by_provider", CostGroup.PROVIDER),
        ("by_model", CostGroup.MODEL),
        ("by_profile", CostGroup.PROFILE),
        ("by_format", CostGroup.FORMAT),
        ("by_stage", CostGroup.STAGE),
    ],
)
def test_convenience_methods_match_the_generic_query(
    engine: Engine, tracker: CostTracker, method: str, group: CostGroup
) -> None:
    _record(tracker)
    aggregator = CostAggregator(engine)

    convenience = getattr(aggregator, method)()
    generic = aggregator.aggregate(group)

    assert convenience == generic


def test_aggregation_runs_as_one_group_by_query(
    engine: Engine, tracker: CostTracker
) -> None:
    _record(tracker)
    aggregator = CostAggregator(engine)
    aggregator.by_provider()

    statements: list[str] = []

    def capture(
        connection: Any,
        cursor: Any,
        statement: str,
        parameters: Any,
        context: Any,
        executemany: bool,
    ) -> None:
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", capture)
    try:
        aggregator.by_provider()
    finally:
        event.remove(engine, "before_cursor_execute", capture)

    assert len(statements) == 1
    assert "GROUP BY" in statements[0].upper()
    assert "cost_events" in statements[0]


def test_aggregate_serializes_for_exports(engine: Engine, tracker: CostTracker) -> None:
    _record(tracker, amount_usd=0.25, units=3.0)
    bucket = CostAggregator(engine).by_provider()[0]

    assert bucket.to_dict() == {
        "group": "provider",
        "key": "acme",
        "total_usd": pytest.approx(0.25),
        "estimated_usd": pytest.approx(0.0),
        "actual_usd": pytest.approx(0.25),
        "units": pytest.approx(3.0),
        "count": 1,
    }
    assert bucket.total == bucket.total_usd
    assert bucket.estimated == bucket.estimated_usd
    assert bucket.actual == bucket.actual_usd

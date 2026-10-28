"""Repository create, get, update and list for every aggregate."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import pytest
from sqlalchemy import Engine

from aivia.core.db import (
    REPOSITORIES,
    AssetRepository,
    CompositionRepository,
    CostEventRepository,
    DeliveryRepository,
    FeedbackRepository,
    JobRepository,
    OrderRepository,
    PlanRepository,
    PostMetricRepository,
    PostRepository,
    RatingRepository,
    ReferenceRepository,
    Repository,
    RunRepository,
    StageRunRepository,
    StyleSpecRepository,
    TemplateRepository,
    WebCardHitRepository,
    create_database_engine,
    metadata,
    repository_for,
    upgrade_database,
)
from aivia.core.models import ModelBase, Run, RunState, StageResult


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


def _no_setup(_engine: Engine) -> dict[str, Any]:
    return {}


def _with_run(engine: Engine) -> dict[str, Any]:
    run = RunRepository(engine).create(Run())
    return {"run_id": run.id, "stage": "write"}


@dataclass(frozen=True)
class AggregateCase:
    name: str
    repository: type[Repository[Any]]
    create: Callable[[Engine], dict[str, Any]]
    field: str
    value: Any


AGGREGATES = [
    AggregateCase("runs", RunRepository, _no_setup, "state", "running"),
    AggregateCase("stage_runs", StageRunRepository, _with_run, "status", "succeeded"),
    AggregateCase("jobs", JobRepository, _no_setup, "state", "running"),
    AggregateCase("assets", AssetRepository, _no_setup, "label", "hero"),
    AggregateCase(
        "compositions", CompositionRepository, _no_setup, "format", "vertical"
    ),
    AggregateCase("cost_events", CostEventRepository, _no_setup, "amount_usd", 1.25),
    AggregateCase("plans", PlanRepository, _no_setup, "instruction", "make a card"),
    AggregateCase("templates", TemplateRepository, _no_setup, "name", "birthday"),
    AggregateCase("references", ReferenceRepository, _no_setup, "label", "example"),
    AggregateCase("style_specs", StyleSpecRepository, _no_setup, "name", "warm"),
    AggregateCase("feedback", FeedbackRepository, _no_setup, "comment", "lovely"),
    AggregateCase("ratings", RatingRepository, _no_setup, "score", 5.0),
    AggregateCase("orders", OrderRepository, _no_setup, "state", "paid"),
    AggregateCase("deliveries", DeliveryRepository, _no_setup, "state", "delivered"),
    AggregateCase("posts", PostRepository, _no_setup, "state", "published"),
    AggregateCase("post_metrics", PostMetricRepository, _no_setup, "likes", 7),
    AggregateCase(
        "web_card_hits", WebCardHitRepository, _no_setup, "card_id", "card_1"
    ),
]


def _key(case: AggregateCase, record: ModelBase | dict[str, Any]) -> tuple[Any, ...]:
    if case.repository is StageRunRepository:
        assert isinstance(record, StageResult)
        return (record.run_id, record.stage)
    if isinstance(record, ModelBase):
        return (record.model_dump()["id"],)
    return (record["id"],)


def _missing_key(case: AggregateCase) -> tuple[str, ...]:
    if case.repository is StageRunRepository:
        return ("run_missing", "write")
    return ("missing",)


def _value(record: ModelBase | dict[str, Any], field: str) -> Any:
    if isinstance(record, ModelBase):
        return getattr(record, field)
    return record[field]


def test_every_table_has_a_repository() -> None:
    assert set(REPOSITORIES) == set(metadata.tables)


@pytest.mark.parametrize("case", AGGREGATES, ids=lambda case: case.name)
def test_every_aggregate_supports_create_get_update_and_list(
    engine: Engine,
    case: AggregateCase,
) -> None:
    repository = case.repository(engine)

    created = repository.create(**case.create(engine))
    key = _key(case, created)

    fetched = repository.get(*key)
    assert fetched is not None
    assert _key(case, fetched) == key

    updated = repository.update(*key, **{case.field: case.value})
    assert updated is not None
    assert _value(updated, case.field) == case.value

    listed = repository.list()
    assert any(_key(case, record) == key for record in listed)
    stored = next(record for record in listed if _key(case, record) == key)
    assert _value(stored, case.field) == case.value


@pytest.mark.parametrize("case", AGGREGATES, ids=lambda case: case.name)
def test_a_missing_aggregate_is_not_found_and_not_updated(
    engine: Engine,
    case: AggregateCase,
) -> None:
    repository = case.repository(engine)

    assert repository.get(*_missing_key(case)) is None
    assert repository.update(*_missing_key(case), **{case.field: case.value}) is None


def test_repository_for_binds_the_engine_and_rejects_unknown_tables(
    engine: Engine,
) -> None:
    repository = repository_for("runs", engine)

    assert isinstance(repository, RunRepository)
    assert repository.engine is engine

    with pytest.raises(ValueError, match="no repository"):
        repository_for("missing_table", engine)


def test_model_repositories_return_validated_models(engine: Engine) -> None:
    stored = RunRepository(engine).create(Run(profile="demo"))

    assert isinstance(stored, Run)
    assert stored.state is RunState.QUEUED
    assert stored.profile == "demo"


def test_table_repositories_return_mappings_and_generate_ids(engine: Engine) -> None:
    stored = OrderRepository(engine).create(occasion="birthday")

    assert isinstance(stored, dict)
    assert stored["id"].startswith("order_")
    assert stored["occasion"] == "birthday"


def test_create_accepts_a_model_a_mapping_or_keywords(engine: Engine) -> None:
    repository = RunRepository(engine)

    from_model = repository.create(Run(profile="model"))
    from_mapping = repository.create({"profile": "mapping"})
    from_keywords = repository.create(profile="keywords")

    assert from_model.profile == "model"
    assert from_mapping.profile == "mapping"
    assert from_keywords.profile == "keywords"
    assert len({from_model.id, from_mapping.id, from_keywords.id}) == 3


def test_list_filters_orders_and_pages(engine: Engine) -> None:
    repository = RunRepository(engine)
    pipeline = repository.create(kind="pipeline")
    repository.create(kind="media")
    pipeline_second = repository.create(kind="pipeline")

    filtered = repository.list(kind="pipeline")
    assert [run.id for run in filtered] == [pipeline.id, pipeline_second.id]

    first_page = repository.list(limit=1)
    second_page = repository.list(limit=1, offset=1)
    assert len(first_page) == 1
    assert len(second_page) == 1
    assert first_page[0].id != second_page[0].id

    with pytest.raises(ValueError, match="no column"):
        repository.list(unknown="value")
    with pytest.raises(ValueError, match="limit"):
        repository.list(limit=-1)
    with pytest.raises(ValueError, match="offset"):
        repository.list(offset=-1)


def test_update_keeps_created_at_and_refreshes_updated_at(engine: Engine) -> None:
    repository = RunRepository(engine)
    created = repository.create(profile="before")

    updated = repository.update(created.id, profile="after")

    assert updated is not None
    assert updated.created_at == created.created_at
    assert updated.updated_at >= created.updated_at
    assert updated.profile == "after"


def test_a_composite_key_repository_requires_every_key_column(engine: Engine) -> None:
    with pytest.raises(ValueError, match="expects the key"):
        StageRunRepository(engine).get("run_only")

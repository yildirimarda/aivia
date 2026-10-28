"""The plan, template, reference, style spec, feedback and rating tables."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, inspect, select
from sqlalchemy.exc import IntegrityError

from aivia.core.db import (
    FEEDBACK_CREATED_AT_INDEX_NAME,
    FEEDBACK_SUBJECT_INDEX_NAME,
    PLAN_CREATED_AT_INDEX_NAME,
    PLAN_RUN_INDEX_NAME,
    PLAN_STATE_INDEX_NAME,
    RATING_CREATED_AT_INDEX_NAME,
    RATING_SUBJECT_INDEX_NAME,
    REFERENCE_CONTENT_HASH_INDEX_NAME,
    REFERENCE_CREATED_AT_INDEX_NAME,
    REFERENCE_KIND_INDEX_NAME,
    STYLE_SPEC_CREATED_AT_INDEX_NAME,
    STYLE_SPEC_NAME_INDEX_NAME,
    TEMPLATE_CREATED_AT_INDEX_NAME,
    TEMPLATE_KIND_INDEX_NAME,
    TEMPLATE_NAME_INDEX_NAME,
    create_database_engine,
    feedback,
    metadata,
    plans,
    ratings,
    references,
    runs,
    style_specs,
    templates,
    upgrade_database,
)

AUX_TABLES = {
    "plans",
    "templates",
    "references",
    "style_specs",
    "feedback",
    "ratings",
}


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    database = tmp_path / "data" / "aivia.db"
    upgrade_database(database)
    created = create_database_engine(database)
    try:
        yield created
    finally:
        created.dispose()


def _indexes(engine: Engine, table: str) -> dict[str, Any]:
    return {
        index["name"]: index
        for index in inspect(engine).get_indexes(table)
        if index["name"] is not None
    }


def _columns(engine: Engine, table: str) -> set[str]:
    return {column["name"] for column in inspect(engine).get_columns(table)}


def test_the_migration_creates_the_auxiliary_tables(engine: Engine) -> None:
    assert set(inspect(engine).get_table_names()) >= AUX_TABLES


def test_the_auxiliary_tables_are_registered_on_the_shared_metadata() -> None:
    assert set(metadata.tables) >= AUX_TABLES


def test_plans_carry_the_plan_record_columns(engine: Engine) -> None:
    assert {
        "id",
        "instruction",
        "state",
        "steps",
        "assumptions",
        "approval_mode",
        "approved",
        "approved_by",
        "approved_at",
        "estimate_usd",
        "cost_usd",
        "budget_usd",
        "run_id",
        "profile",
        "format",
        "idempotency_key",
        "error_code",
        "error_message",
        "metadata",
        "schema_version",
        "created_at",
        "updated_at",
    } <= _columns(engine, "plans")


def test_templates_carry_the_template_columns(engine: Engine) -> None:
    assert {
        "id",
        "name",
        "kind",
        "description",
        "path",
        "content_hash",
        "version",
        "variables",
        "asset_slots",
        "source_plan_id",
        "source_composition_id",
        "metadata",
        "schema_version",
        "created_at",
        "updated_at",
    } <= _columns(engine, "templates")


def test_references_carry_the_reference_columns(engine: Engine) -> None:
    assert {
        "id",
        "kind",
        "label",
        "path",
        "url",
        "content_hash",
        "mime_type",
        "size_bytes",
        "language",
        "source",
        "owned",
        "analysis",
        "metadata",
        "schema_version",
        "created_at",
        "updated_at",
    } <= _columns(engine, "references")


def test_style_specs_carry_the_style_spec_columns(engine: Engine) -> None:
    assert {
        "id",
        "name",
        "path",
        "content_hash",
        "version",
        "spec",
        "reference_ids",
        "recipe",
        "metadata",
        "schema_version",
        "created_at",
        "updated_at",
    } <= _columns(engine, "style_specs")


def test_feedback_carries_the_feedback_columns(engine: Engine) -> None:
    assert {
        "id",
        "subject_kind",
        "subject_id",
        "sentiment",
        "comment",
        "source",
        "run_id",
        "metadata",
        "schema_version",
        "created_at",
        "updated_at",
    } <= _columns(engine, "feedback")


def test_ratings_carry_the_rating_columns(engine: Engine) -> None:
    assert {
        "id",
        "subject_kind",
        "subject_id",
        "score",
        "rater",
        "comment",
        "run_id",
        "metadata",
        "schema_version",
        "created_at",
        "updated_at",
    } <= _columns(engine, "ratings")


def test_plans_have_lookup_indexes(engine: Engine) -> None:
    indexes = _indexes(engine, "plans")

    expected = {
        PLAN_RUN_INDEX_NAME: ["run_id"],
        PLAN_STATE_INDEX_NAME: ["state"],
        PLAN_CREATED_AT_INDEX_NAME: ["created_at"],
    }
    for name, columns in expected.items():
        assert name in indexes
        assert indexes[name]["column_names"] == columns
        assert not indexes[name]["unique"]


def test_templates_have_lookup_indexes(engine: Engine) -> None:
    indexes = _indexes(engine, "templates")

    expected = {
        TEMPLATE_NAME_INDEX_NAME: ["name"],
        TEMPLATE_KIND_INDEX_NAME: ["kind"],
        TEMPLATE_CREATED_AT_INDEX_NAME: ["created_at"],
    }
    for name, columns in expected.items():
        assert name in indexes
        assert indexes[name]["column_names"] == columns
        assert not indexes[name]["unique"]


def test_references_have_lookup_indexes(engine: Engine) -> None:
    indexes = _indexes(engine, "references")

    expected = {
        REFERENCE_KIND_INDEX_NAME: ["kind"],
        REFERENCE_CONTENT_HASH_INDEX_NAME: ["content_hash"],
        REFERENCE_CREATED_AT_INDEX_NAME: ["created_at"],
    }
    for name, columns in expected.items():
        assert name in indexes
        assert indexes[name]["column_names"] == columns
        assert not indexes[name]["unique"]


def test_style_specs_have_lookup_indexes(engine: Engine) -> None:
    indexes = _indexes(engine, "style_specs")

    expected = {
        STYLE_SPEC_NAME_INDEX_NAME: ["name"],
        STYLE_SPEC_CREATED_AT_INDEX_NAME: ["created_at"],
    }
    for name, columns in expected.items():
        assert name in indexes
        assert indexes[name]["column_names"] == columns
        assert not indexes[name]["unique"]


def test_feedback_has_subject_and_time_indexes(engine: Engine) -> None:
    indexes = _indexes(engine, "feedback")

    expected = {
        FEEDBACK_SUBJECT_INDEX_NAME: ["subject_kind", "subject_id"],
        FEEDBACK_CREATED_AT_INDEX_NAME: ["created_at"],
    }
    for name, columns in expected.items():
        assert name in indexes
        assert indexes[name]["column_names"] == columns
        assert not indexes[name]["unique"]


def test_ratings_have_a_unique_subject_index_and_a_time_index(
    engine: Engine,
) -> None:
    indexes = _indexes(engine, "ratings")

    assert indexes[RATING_SUBJECT_INDEX_NAME]["column_names"] == [
        "subject_kind",
        "subject_id",
    ]
    assert indexes[RATING_SUBJECT_INDEX_NAME]["unique"]
    assert indexes[RATING_CREATED_AT_INDEX_NAME]["column_names"] == ["created_at"]
    assert not indexes[RATING_CREATED_AT_INDEX_NAME]["unique"]


def test_a_minimal_plan_uses_the_record_defaults(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(plans.insert().values(id="plan_1"))

    with engine.connect() as connection:
        row = connection.execute(select(plans).where(plans.c.id == "plan_1")).one()

    assert row.instruction == ""
    assert row.state == "draft"
    assert row.steps == []
    assert row.assumptions == []
    assert row.approval_mode == "ask"
    assert row.approved is False
    assert row.estimate_usd == 0
    assert row.cost_usd == 0
    assert row.budget_usd is None
    assert row.run_id == ""
    assert row.idempotency_key == ""
    assert row.metadata == {}


def test_a_minimal_template_uses_the_record_defaults(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(templates.insert().values(id="template_1"))

    with engine.connect() as connection:
        row = connection.execute(
            select(templates).where(templates.c.id == "template_1")
        ).one()

    assert row.name == ""
    assert row.kind == "plan"
    assert row.version == 1
    assert row.variables == {}
    assert row.asset_slots == []
    assert row.source_plan_id == ""
    assert row.source_composition_id == ""
    assert row.metadata == {}


def test_a_minimal_reference_uses_the_record_defaults(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(references.insert().values(id="reference_1"))

    with engine.connect() as connection:
        row = connection.execute(
            select(references).where(references.c.id == "reference_1")
        ).one()

    assert row.kind == "image"
    assert row.content_hash == ""
    assert row.owned is False
    assert row.analysis == {}
    assert row.metadata == {}


def test_a_minimal_style_spec_uses_the_record_defaults(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(style_specs.insert().values(id="style_1"))

    with engine.connect() as connection:
        row = connection.execute(
            select(style_specs).where(style_specs.c.id == "style_1")
        ).one()

    assert row.name == ""
    assert row.version == 1
    assert row.spec == {}
    assert row.reference_ids == []
    assert row.recipe == {}
    assert row.metadata == {}


def test_a_minimal_feedback_uses_the_record_defaults(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(feedback.insert().values(id="feedback_1"))

    with engine.connect() as connection:
        row = connection.execute(
            select(feedback).where(feedback.c.id == "feedback_1")
        ).one()

    assert row.subject_kind == "topic"
    assert row.subject_id == ""
    assert row.sentiment == "neutral"
    assert row.comment == ""
    assert row.run_id == ""
    assert row.metadata == {}


def test_a_minimal_rating_uses_the_record_defaults(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(ratings.insert().values(id="rating_1"))

    with engine.connect() as connection:
        row = connection.execute(
            select(ratings).where(ratings.c.id == "rating_1")
        ).one()

    assert row.subject_kind == "provider"
    assert row.subject_id == ""
    assert row.score == 0
    assert row.rater == ""
    assert row.comment == ""
    assert row.run_id == ""
    assert row.metadata == {}


def test_a_subject_has_only_one_rating(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            ratings.insert().values(
                id="rating_1",
                subject_kind="provider",
                subject_id="alpha/image",
                score=4,
            )
        )

    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(
            ratings.insert().values(
                id="rating_2",
                subject_kind="provider",
                subject_id="alpha/image",
                score=2,
            )
        )

    with engine.connect() as connection:
        stored = connection.execute(select(ratings.c.id, ratings.c.score)).all()

    assert [(row.id, row.score) for row in stored] == [("rating_1", 4.0)]


def test_the_same_subject_id_can_be_rated_once_per_kind(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            ratings.insert().values(
                id="rating_provider",
                subject_kind="provider",
                subject_id="alpha",
            )
        )
        connection.execute(
            ratings.insert().values(
                id="rating_model",
                subject_kind="model",
                subject_id="alpha",
            )
        )

    with engine.connect() as connection:
        rated = (
            connection.execute(select(ratings.c.id).order_by(ratings.c.id))
            .scalars()
            .all()
        )

    assert rated == ["rating_model", "rating_provider"]


def test_a_subject_can_collect_many_feedback_entries(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            feedback.insert().values(
                id="feedback_like",
                subject_kind="topic",
                subject_id="topic_1",
                sentiment="like",
            )
        )
        connection.execute(
            feedback.insert().values(
                id="feedback_comment",
                subject_kind="topic",
                subject_id="topic_1",
                sentiment="neutral",
                comment="more like this",
            )
        )

    with engine.connect() as connection:
        entries = (
            connection.execute(
                select(feedback.c.id)
                .where(feedback.c.subject_id == "topic_1")
                .order_by(feedback.c.id)
            )
            .scalars()
            .all()
        )

    assert entries == ["feedback_comment", "feedback_like"]


def test_plans_can_be_grouped_by_run_and_state(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(runs.insert().values(id="run_1"))
        connection.execute(
            plans.insert().values(id="plan_a", run_id="run_1", state="running")
        )
        connection.execute(
            plans.insert().values(id="plan_b", run_id="run_1", state="succeeded")
        )
        connection.execute(
            plans.insert().values(id="plan_c", run_id="run_2", state="running")
        )

    with engine.connect() as connection:
        run_plans = (
            connection.execute(
                select(plans.c.id).where(plans.c.run_id == "run_1").order_by(plans.c.id)
            )
            .scalars()
            .all()
        )
        running = (
            connection.execute(
                select(plans.c.id)
                .where(plans.c.state == "running")
                .order_by(plans.c.id)
            )
            .scalars()
            .all()
        )

    assert run_plans == ["plan_a", "plan_b"]
    assert running == ["plan_a", "plan_c"]


def test_references_can_be_grouped_by_kind_and_content_hash(
    engine: Engine,
) -> None:
    with engine.begin() as connection:
        connection.execute(
            references.insert().values(
                id="reference_image",
                kind="image",
                content_hash="hash-1",
            )
        )
        connection.execute(
            references.insert().values(
                id="reference_audio",
                kind="audio",
                content_hash="hash-2",
            )
        )
        connection.execute(
            references.insert().values(
                id="reference_duplicate",
                kind="image",
                content_hash="hash-1",
            )
        )

    with engine.connect() as connection:
        images = (
            connection.execute(
                select(references.c.id)
                .where(references.c.kind == "image")
                .order_by(references.c.id)
            )
            .scalars()
            .all()
        )
        by_hash = (
            connection.execute(
                select(references.c.id).where(references.c.content_hash == "hash-1")
            )
            .scalars()
            .all()
        )

    assert images == ["reference_duplicate", "reference_image"]
    assert set(by_hash) == {"reference_duplicate", "reference_image"}


def test_templates_and_style_specs_can_be_found_by_name(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            templates.insert().values(id="template_1", name="quote", kind="plan")
        )
        connection.execute(
            templates.insert().values(id="template_2", name="quote", kind="composition")
        )
        connection.execute(style_specs.insert().values(id="style_1", name="retro"))

    with engine.connect() as connection:
        template = (
            connection.execute(
                select(templates.c.id).where(
                    templates.c.name == "quote",
                    templates.c.kind == "composition",
                )
            )
            .scalars()
            .one()
        )
        style = (
            connection.execute(
                select(style_specs.c.id).where(style_specs.c.name == "retro")
            )
            .scalars()
            .one()
        )

    assert template == "template_2"
    assert style == "style_1"


def test_the_downgrade_removes_the_auxiliary_tables(engine: Engine) -> None:
    from alembic import command

    from aivia.core.db import alembic_config

    with engine.begin() as connection:
        config = alembic_config(connection=connection)
        command.downgrade(config, "0003")

    remaining = set(inspect(engine).get_table_names())

    assert not (AUX_TABLES & remaining)

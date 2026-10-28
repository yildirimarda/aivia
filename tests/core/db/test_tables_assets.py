"""The asset, composition and cost event tables and their lookup indexes."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, inspect, select, text

from aivia.core.db import (
    ASSET_CREATED_AT_INDEX_NAME,
    ASSET_PINNED_INDEX_NAME,
    ASSET_PROMPT_HASH_INDEX_NAME,
    ASSET_PROVIDER_INDEX_NAME,
    ASSET_RUN_INDEX_NAME,
    COMPOSITION_CREATED_AT_INDEX_NAME,
    COMPOSITION_PINNED_INDEX_NAME,
    COMPOSITION_RUN_INDEX_NAME,
    COST_EVENT_CREATED_AT_INDEX_NAME,
    COST_EVENT_PROVIDER_INDEX_NAME,
    COST_EVENT_RUN_INDEX_NAME,
    assets,
    compositions,
    cost_events,
    create_database_engine,
    metadata,
    runs,
    upgrade_database,
)

ASSET_TABLES = {"assets", "compositions", "cost_events"}


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


def test_the_migration_creates_the_asset_tables(engine: Engine) -> None:
    assert set(inspect(engine).get_table_names()) >= ASSET_TABLES


def test_the_asset_tables_are_registered_on_the_shared_metadata() -> None:
    assert set(metadata.tables) >= ASSET_TABLES


def test_assets_carry_the_asset_record_columns(engine: Engine) -> None:
    columns = {column["name"] for column in inspect(engine).get_columns("assets")}

    assert {
        "id",
        "kind",
        "label",
        "path",
        "content_hash",
        "provider",
        "model",
        "prompt",
        "prompt_hash",
        "license",
        "cost_links",
        "analysis",
        "alt_text",
        "pinned",
        "quality_score",
        "run_id",
        "schema_version",
        "created_at",
        "updated_at",
    } <= columns


def test_compositions_carry_the_composition_reference_columns(engine: Engine) -> None:
    columns = {column["name"] for column in inspect(engine).get_columns("compositions")}

    assert {
        "id",
        "path",
        "content_hash",
        "format",
        "version",
        "renderer_version",
        "toolchain_fingerprint",
        "duration_seconds",
        "width",
        "height",
        "fps",
        "assets",
        "pinned",
        "run_id",
        "language",
        "platform",
        "metadata",
        "schema_version",
        "created_at",
        "updated_at",
    } <= columns


def test_cost_events_carry_the_cost_event_columns(engine: Engine) -> None:
    columns = {column["name"] for column in inspect(engine).get_columns("cost_events")}

    assert {
        "id",
        "provider",
        "model",
        "operation",
        "units",
        "unit_kind",
        "amount_usd",
        "estimated",
        "run_id",
        "pricing_version",
        "metadata",
        "schema_version",
        "created_at",
        "updated_at",
    } <= columns


def test_assets_have_lookup_indexes(engine: Engine) -> None:
    indexes = _indexes(engine, "assets")

    expected = {
        ASSET_PROMPT_HASH_INDEX_NAME: ["prompt_hash"],
        ASSET_RUN_INDEX_NAME: ["run_id"],
        ASSET_PROVIDER_INDEX_NAME: ["provider"],
        ASSET_CREATED_AT_INDEX_NAME: ["created_at"],
        ASSET_PINNED_INDEX_NAME: ["pinned"],
    }
    for name, columns in expected.items():
        assert name in indexes
        assert indexes[name]["column_names"] == columns
        assert not indexes[name]["unique"]


def test_compositions_have_lookup_indexes(engine: Engine) -> None:
    indexes = _indexes(engine, "compositions")

    expected = {
        COMPOSITION_RUN_INDEX_NAME: ["run_id"],
        COMPOSITION_CREATED_AT_INDEX_NAME: ["created_at"],
        COMPOSITION_PINNED_INDEX_NAME: ["pinned"],
    }
    for name, columns in expected.items():
        assert name in indexes
        assert indexes[name]["column_names"] == columns
        assert not indexes[name]["unique"]


def test_cost_events_have_lookup_indexes(engine: Engine) -> None:
    indexes = _indexes(engine, "cost_events")

    expected = {
        COST_EVENT_RUN_INDEX_NAME: ["run_id"],
        COST_EVENT_PROVIDER_INDEX_NAME: ["provider"],
        COST_EVENT_CREATED_AT_INDEX_NAME: ["created_at"],
    }
    for name, columns in expected.items():
        assert name in indexes
        assert indexes[name]["column_names"] == columns
        assert not indexes[name]["unique"]


def test_the_prompt_hash_index_serves_a_cache_lookup(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            assets.insert().values(id="asset_1", prompt_hash="hash-1", prompt="one")
        )
        connection.execute(
            assets.insert().values(id="asset_2", prompt_hash="hash-2", prompt="two")
        )

    with engine.connect() as connection:
        plan = connection.execute(
            text("EXPLAIN QUERY PLAN SELECT id FROM assets WHERE prompt_hash = :hash"),
            {"hash": "hash-1"},
        ).all()
        found = connection.execute(
            select(assets.c.id).where(assets.c.prompt_hash == "hash-1")
        ).scalar_one()

    assert found == "asset_1"
    assert any(ASSET_PROMPT_HASH_INDEX_NAME in str(row) for row in plan)


def test_a_minimal_asset_uses_the_record_defaults(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(assets.insert().values(id="asset_1"))

    with engine.connect() as connection:
        row = connection.execute(select(assets).where(assets.c.id == "asset_1")).one()

    assert row.kind == "image"
    assert row.provider == ""
    assert row.prompt_hash == ""
    assert row.license == {}
    assert row.cost_links == []
    assert row.analysis == {}
    assert row.pinned is False
    assert row.run_id == ""


def test_assets_can_be_grouped_by_run_provider_and_pinned(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(runs.insert().values(id="run_1"))
        connection.execute(
            assets.insert().values(
                id="asset_pinned",
                run_id="run_1",
                provider="alpha",
                prompt_hash="hash-1",
                pinned=True,
            )
        )
        connection.execute(
            assets.insert().values(
                id="asset_draft",
                run_id="run_1",
                provider="beta",
                prompt_hash="hash-2",
            )
        )

    with engine.connect() as connection:
        run_assets = (
            connection.execute(
                select(assets.c.id)
                .where(assets.c.run_id == "run_1")
                .order_by(assets.c.id)
            )
            .scalars()
            .all()
        )
        pinned = (
            connection.execute(select(assets.c.id).where(assets.c.pinned.is_(True)))
            .scalars()
            .all()
        )
        by_provider = (
            connection.execute(select(assets.c.id).where(assets.c.provider == "beta"))
            .scalars()
            .all()
        )

    assert run_assets == ["asset_draft", "asset_pinned"]
    assert pinned == ["asset_pinned"]
    assert by_provider == ["asset_draft"]


def test_a_minimal_composition_uses_the_reference_defaults(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(compositions.insert().values(id="composition_1"))

    with engine.connect() as connection:
        row = connection.execute(
            select(compositions).where(compositions.c.id == "composition_1")
        ).one()

    assert row.version == 1
    assert row.toolchain_fingerprint == {}
    assert row.assets == []
    assert row.pinned is False
    assert row.run_id == ""
    assert row.metadata == {}


def test_a_minimal_cost_event_uses_the_record_defaults(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(cost_events.insert().values(id="cost_1"))

    with engine.connect() as connection:
        row = connection.execute(
            select(cost_events).where(cost_events.c.id == "cost_1")
        ).one()

    assert row.provider == ""
    assert row.operation == ""
    assert row.units == 0
    assert row.unit_kind == "call"
    assert row.amount_usd == 0
    assert row.estimated is False
    assert row.run_id == ""
    assert row.pricing_version == ""
    assert row.metadata == {}


def test_cost_events_can_be_grouped_by_run_provider_and_time(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(runs.insert().values(id="run_1"))
        connection.execute(
            cost_events.insert().values(
                id="cost_a",
                run_id="run_1",
                provider="alpha",
                amount_usd=1.5,
            )
        )
        connection.execute(
            cost_events.insert().values(
                id="cost_b",
                run_id="run_1",
                provider="alpha",
                amount_usd=0.5,
            )
        )
        connection.execute(
            cost_events.insert().values(
                id="cost_c",
                run_id="run_1",
                provider="beta",
                amount_usd=2.0,
            )
        )

    with engine.connect() as connection:
        total = connection.execute(
            select(cost_events.c.amount_usd).where(cost_events.c.run_id == "run_1")
        ).scalars()
        alpha = (
            connection.execute(
                select(cost_events.c.id).where(cost_events.c.provider == "alpha")
            )
            .scalars()
            .all()
        )
        recent = (
            connection.execute(
                select(cost_events.c.id).where(cost_events.c.created_at.is_not(None))
            )
            .scalars()
            .all()
        )

    assert sum(total) == 4.0
    assert alpha == ["cost_a", "cost_b"]
    assert set(recent) == {"cost_a", "cost_b", "cost_c"}


def test_the_downgrade_removes_the_asset_tables(engine: Engine) -> None:
    from alembic import command

    from aivia.core.db import alembic_config

    with engine.begin() as connection:
        config = alembic_config(connection=connection)
        command.downgrade(config, "0002")

    remaining = set(inspect(engine).get_table_names())

    assert not (ASSET_TABLES & remaining)

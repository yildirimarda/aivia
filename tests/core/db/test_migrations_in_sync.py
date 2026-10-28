"""Guard against drift between the table models and the Alembic migrations."""

from __future__ import annotations

from typing import Any

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import Column, Engine, Integer, MetaData, Table

from aivia.core.db import create_database_engine, metadata, upgrade_database


def _schema_differences(engine: Engine, target: MetaData) -> list[Any]:
    with engine.connect() as connection:
        context = MigrationContext.configure(connection, opts={"compare_type": True})
        differences: list[Any] = compare_metadata(context, target)
    return differences


def _upgraded_engine() -> Engine:
    engine = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=engine)
    return engine


def test_models_and_migrations_are_in_sync() -> None:
    engine = _upgraded_engine()
    try:
        differences = _schema_differences(engine, metadata)
    finally:
        engine.dispose()

    assert differences == [], (
        "models and migrations diverged; add or amend a migration so the "
        f"database matches the models: {differences}"
    )


def test_guard_detects_a_model_only_table() -> None:
    engine = _upgraded_engine()
    try:
        drifted = MetaData()
        for table in metadata.tables.values():
            table.to_metadata(drifted)
        Table("model_only", drifted, Column("id", Integer, primary_key=True))

        differences = _schema_differences(engine, drifted)
    finally:
        engine.dispose()

    assert [difference[0] for difference in differences] == ["add_table"]
    assert differences[0][1].name == "model_only"


def test_guard_detects_a_model_only_column() -> None:
    engine = _upgraded_engine()
    try:
        drifted = MetaData()
        for table in metadata.tables.values():
            table.to_metadata(drifted)
        drifted.tables["runs"].append_column(Column("model_only", Integer))

        differences = _schema_differences(engine, drifted)
    finally:
        engine.dispose()

    assert [difference[0] for difference in differences] == ["add_column"]
    assert differences[0][3].name == "model_only"

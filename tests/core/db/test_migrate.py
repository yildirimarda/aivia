"""The local SQLite engine, its pragmas and the Alembic migration path."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import IntegrityError

from aivia.core.db import (
    DEFAULT_BUSY_TIMEOUT_MS,
    alembic_config,
    create_database_engine,
    current_revision,
    database_path,
    database_url,
    migrations_dir,
    upgrade_database,
)


def _pragma(engine: Engine, name: str) -> object:
    with engine.connect() as connection:
        return connection.execute(text(f"PRAGMA {name}")).scalar()


def test_database_path_lives_in_the_user_data_directory(tmp_path: Path) -> None:
    environ = {"XDG_CONFIG_HOME": str(tmp_path)}

    assert database_path(environ=environ) == (tmp_path / "aivia" / "data" / "aivia.db")


def test_database_path_accepts_an_explicit_directory(tmp_path: Path) -> None:
    assert database_path(tmp_path / "store") == tmp_path / "store" / "aivia.db"


def test_database_url_points_at_a_file_or_memory() -> None:
    assert database_url("/tmp/aivia.db") == "sqlite+pysqlite:////tmp/aivia.db"
    assert database_url(":memory:") == "sqlite+pysqlite:///:memory:"


def test_creating_the_engine_creates_the_data_directory(tmp_path: Path) -> None:
    environ = {"XDG_CONFIG_HOME": str(tmp_path)}
    engine = create_database_engine(environ=environ)
    try:
        with engine.connect():
            pass
        expected = database_path(environ=environ)
        assert expected.is_file()
    finally:
        engine.dispose()


def test_wal_and_foreign_keys_are_enabled_on_every_connection(
    tmp_path: Path,
) -> None:
    engine = create_database_engine(tmp_path / "aivia.db")
    try:
        with engine.connect() as first, engine.connect() as second:
            for connection in (first, second):
                assert connection.execute(text("PRAGMA journal_mode")).scalar() == "wal"
                assert connection.execute(text("PRAGMA foreign_keys")).scalar() == 1
                assert (
                    connection.execute(text("PRAGMA busy_timeout")).scalar()
                    == DEFAULT_BUSY_TIMEOUT_MS
                )
    finally:
        engine.dispose()


def test_busy_timeout_is_configurable(tmp_path: Path) -> None:
    engine = create_database_engine(tmp_path / "aivia.db", busy_timeout_ms=1_234)
    try:
        assert _pragma(engine, "busy_timeout") == 1_234
    finally:
        engine.dispose()


def test_negative_busy_timeout_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="busy_timeout_ms"):
        create_database_engine(tmp_path / "aivia.db", busy_timeout_ms=-1)


def test_foreign_keys_are_enforced(tmp_path: Path) -> None:
    engine = create_database_engine(tmp_path / "aivia.db")
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE parent (id INTEGER PRIMARY KEY)"))
            connection.execute(
                text(
                    "CREATE TABLE child ("
                    "id INTEGER PRIMARY KEY, "
                    "parent_id INTEGER REFERENCES parent(id))"
                )
            )

        with pytest.raises(IntegrityError), engine.begin() as connection:
            connection.execute(text("INSERT INTO child (parent_id) VALUES (99)"))
    finally:
        engine.dispose()


def test_migrations_create_the_database_and_record_the_revision(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data" / "aivia.db"

    upgrade_database(database)

    assert database.is_file()
    engine = create_database_engine(database)
    try:
        assert current_revision(engine) == "0005"
    finally:
        engine.dispose()


def test_migrations_are_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "data" / "aivia.db"

    upgrade_database(database)
    upgrade_database(database)

    engine = create_database_engine(database)
    try:
        assert current_revision(engine) == "0005"
    finally:
        engine.dispose()


def test_migrations_accept_a_configured_engine() -> None:
    engine = create_database_engine(url="sqlite+pysqlite:///:memory:")
    try:
        upgrade_database(engine=engine)

        assert current_revision(engine) == "0005"
    finally:
        engine.dispose()


def test_migrations_run_in_the_user_data_directory(tmp_path: Path) -> None:
    environ = {"XDG_CONFIG_HOME": str(tmp_path)}

    upgrade_database(environ=environ)

    database = database_path(environ=environ)
    assert database.is_file()
    engine = create_database_engine(environ=environ)
    try:
        assert current_revision(engine) == "0005"
    finally:
        engine.dispose()


def test_migrations_directory_is_discoverable(tmp_path: Path) -> None:
    location = migrations_dir()

    assert (location / "env.py").is_file()
    assert (location / "versions").is_dir()


def test_alembic_config_carries_the_script_location(tmp_path: Path) -> None:
    config = alembic_config(url=database_url(tmp_path / "aivia.db"))

    assert Path(config.get_main_option("script_location") or "").is_dir()
    assert config.get_main_option("sqlalchemy.url") == database_url(
        tmp_path / "aivia.db"
    )

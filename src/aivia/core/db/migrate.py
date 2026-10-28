"""Alembic migrations for the local SQLite database.

The migration scripts ship beside the package in ``migrations/``. This module
locates them, builds an Alembic configuration and upgrades the database in the
user data directory. Callers may pass an existing engine or connection so tests
and the application share one configured connection.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from alembic import command
from alembic.config import Config
from sqlalchemy import Connection, Engine, inspect, text

from aivia.core import paths
from aivia.core.db.engine import (
    create_database_engine,
    database_path,
    database_url,
)

if TYPE_CHECKING:
    from aivia.core.config.database import DatabaseConfig

__all__ = [
    "MIGRATIONS_ENV",
    "alembic_config",
    "current_revision",
    "migrations_dir",
    "upgrade_database",
]

MIGRATIONS_ENV = "AIVA_MIGRATIONS_DIR"
"""Optional environment override for the migration script directory."""


def migrations_dir() -> Path:
    """Return the directory holding the Alembic environment and versions."""
    override = os.environ.get(MIGRATIONS_ENV, "").strip()
    if override:
        return Path(override).expanduser().resolve()

    package_root = paths.package_root()
    candidates = (
        package_root / "data" / "migrations",
        package_root.parent.parent / "migrations",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "cannot find the database migrations directory; set " + MIGRATIONS_ENV
    )


def alembic_config(
    *,
    connection: Connection | None = None,
    url: str | None = None,
    script_location: str | Path | None = None,
) -> Config:
    """Build an Alembic configuration for the local database."""
    config = Config()
    location = migrations_dir() if script_location is None else Path(script_location)
    config.set_main_option("script_location", str(location))
    if url is not None:
        config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    elif connection is None:
        config.set_main_option(
            "sqlalchemy.url", database_url(database_path()).replace("%", "%%")
        )
    if connection is not None:
        config.attributes["connection"] = connection
    return config


def upgrade_database(
    path: str | Path | None = None,
    *,
    revision: str = "head",
    engine: Engine | None = None,
    url: str | None = None,
    data_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    config: DatabaseConfig | None = None,
    busy_timeout_ms: int | None = None,
    script_location: str | Path | None = None,
) -> None:
    """Upgrade the database to *revision*, creating the file when missing.

    Passing *engine* reuses its pragma-enabled connections. Otherwise an engine
    is built from *path*, *url*, *data_dir*, a *config* section or the user
    data directory and disposed afterwards.
    """
    owns_engine = engine is None
    target = (
        create_database_engine(
            path,
            url=url,
            data_dir=data_dir,
            environ=environ,
            config=config,
            busy_timeout_ms=busy_timeout_ms,
        )
        if engine is None
        else engine
    )
    try:
        with target.begin() as connection:
            alembic_cfg = alembic_config(
                connection=connection,
                script_location=script_location,
            )
            command.upgrade(alembic_cfg, revision)
    finally:
        if owns_engine:
            target.dispose()


def current_revision(engine: Engine) -> str | None:
    """Return the applied Alembic revision, or ``None`` before the first one."""
    if not inspect(engine).has_table("alembic_version"):
        return None
    with engine.connect() as connection:
        row = connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).first()
    return None if row is None else str(row[0])

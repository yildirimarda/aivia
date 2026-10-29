"""SQLite engine configured for concurrent local access.

The database lives in the user data directory resolved by
:mod:`aivia.core.paths`. Every connection enables write-ahead logging, a busy
timeout and foreign-key enforcement, so the CLI, the API and workers can share
one file without the default rollback journal raising ``database is locked``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import Engine, event
from sqlalchemy import create_engine as _create_engine
from sqlalchemy.pool import StaticPool

from aivia.core import paths

if TYPE_CHECKING:
    from aivia.core.config.database import DatabaseConfig

__all__ = [
    "DEFAULT_BUSY_TIMEOUT_MS",
    "DEFAULT_DATABASE_FILENAME",
    "configure_connection",
    "create_database_engine",
    "database_path",
    "database_url",
    "install_sqlite_pragmas",
]

DEFAULT_DATABASE_FILENAME = "aivia.db"
"""The database file name inside the user data directory."""

DEFAULT_BUSY_TIMEOUT_MS = 5_000
"""How long SQLite waits for a lock before raising ``database is locked``."""

_NON_NEGATIVE = 0


def _validate_busy_timeout(busy_timeout_ms: int) -> None:
    if busy_timeout_ms < _NON_NEGATIVE:
        raise ValueError("busy_timeout_ms must not be negative")


def database_path(
    data_dir: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    filename: str = DEFAULT_DATABASE_FILENAME,
) -> Path:
    """Return the database file inside the user data directory."""
    root = paths.data_dir(environ) if data_dir is None else Path(data_dir)
    return root.expanduser() / filename


def database_url(path: str | Path) -> str:
    """Return the SQLAlchemy URL for a SQLite file or ``:memory:``."""
    text = str(path)
    if text == ":memory:":
        return "sqlite+pysqlite:///:memory:"
    return f"sqlite+pysqlite:///{Path(text).expanduser().as_posix()}"


def configure_connection(
    dbapi_connection: sqlite3.Connection,
    *,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
) -> None:
    """Apply the concurrency and integrity pragmas to one DB-API connection."""
    _validate_busy_timeout(busy_timeout_ms)

    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        cursor.execute("PRAGMA journal_mode=WAL")
    finally:
        cursor.close()


def install_sqlite_pragmas(
    engine: Engine,
    *,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
) -> None:
    """Install a connect hook so every new connection carries the pragmas."""
    if engine.dialect.name != "sqlite":
        raise ValueError(
            f"SQLite pragmas only apply to a SQLite engine, not {engine.dialect.name}"
        )
    _validate_busy_timeout(busy_timeout_ms)

    @event.listens_for(engine, "connect")
    def _set_pragmas(
        dbapi_connection: sqlite3.Connection,
        _connection_record: Any,
    ) -> None:
        configure_connection(dbapi_connection, busy_timeout_ms=busy_timeout_ms)


def create_database_engine(
    path: str | Path | None = None,
    *,
    url: str | None = None,
    data_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    config: DatabaseConfig | None = None,
    busy_timeout_ms: int | None = None,
    filename: str | None = None,
    echo: bool = False,
) -> Engine:
    """Build an engine for the user database, creating its directory.

    Pass an explicit *path* or *data_dir* to override the user location, or a
    *url* such as ``sqlite+pysqlite:///:memory:`` for tests. A *config* section
    supplies the file name and busy timeout; explicit *filename* or
    *busy_timeout_ms* arguments are rejected alongside it. The pragmas are
    installed before any connection is opened.
    """
    if url is not None and (path is not None or data_dir is not None):
        raise ValueError("pass either url or path/data_dir, not both")
    if config is not None and (busy_timeout_ms is not None or filename is not None):
        raise ValueError("pass either config or busy_timeout_ms/filename, not both")

    if busy_timeout_ms is None:
        busy_timeout_ms = (
            config.busy_timeout_ms if config is not None else DEFAULT_BUSY_TIMEOUT_MS
        )
    if filename is None:
        filename = config.filename if config is not None else DEFAULT_DATABASE_FILENAME
    _validate_busy_timeout(busy_timeout_ms)

    if url is None:
        if path is not None:
            db_path = Path(path).expanduser()
        else:
            db_path = database_path(data_dir, environ=environ, filename=filename)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        url = database_url(db_path)

    if url == database_url(":memory:"):
        engine = _create_engine(
            url,
            echo=echo,
            future=True,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
    else:
        engine = _create_engine(url, echo=echo, future=True)
    install_sqlite_pragmas(engine, busy_timeout_ms=busy_timeout_ms)
    return engine

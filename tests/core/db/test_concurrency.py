"""Two writers on one file prove WAL and the busy timeout prevent lock errors.

SQLite serialises writers: while one connection holds the write lock the other
must wait. The engine enables write-ahead logging and a busy timeout on every
connection, so the second writer retries instead of raising ``database is
locked``. These tests run two real threads against a file database and also
show that removing the busy timeout brings the lock error straight back.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest
from sqlalchemy import Engine, text

from aivia.core.db import (
    DEFAULT_BUSY_TIMEOUT_MS,
    create_database_engine,
)

pytestmark = pytest.mark.slow

_CREATE_TABLE = "CREATE TABLE notes (id INTEGER PRIMARY KEY, writer TEXT NOT NULL)"
_INSERT = "INSERT INTO notes (writer) VALUES (?)"
_HOLD_SECONDS = 0.2
_WAIT_SECONDS = 5.0
_JOIN_SECONDS = 10.0


def _prepare(database: Path) -> Engine:
    engine = create_database_engine(database)
    with engine.begin() as connection:
        connection.execute(text(_CREATE_TABLE))
    return engine


def _pragma(engine: Engine, name: str) -> object:
    with engine.connect() as connection:
        return connection.execute(text(f"PRAGMA {name}")).scalar_one()


def _count(database: Path) -> int:
    engine = create_database_engine(database)
    try:
        with engine.connect() as connection:
            return int(
                connection.execute(text("SELECT COUNT(*) FROM notes")).scalar_one()
            )
    finally:
        engine.dispose()


def _run(threads: list[threading.Thread]) -> None:
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(_JOIN_SECONDS)
    assert not any(thread.is_alive() for thread in threads), "writer thread hung"


def test_two_writers_wait_instead_of_raising_database_is_locked(
    tmp_path: Path,
) -> None:
    database = tmp_path / "aivia.db"
    first = _prepare(database)
    second = create_database_engine(database)
    try:
        for engine in (first, second):
            assert _pragma(engine, "journal_mode") == "wal"
            assert _pragma(engine, "busy_timeout") == DEFAULT_BUSY_TIMEOUT_MS

        first_ready = threading.Event()
        second_attempting = threading.Event()
        results: dict[str, str] = {}

        def writer_first() -> None:
            connection = first.raw_connection()
            try:
                cursor = connection.cursor()
                cursor.execute("BEGIN IMMEDIATE")
                cursor.execute(_INSERT, ("first",))
                first_ready.set()
                second_attempting.wait(_WAIT_SECONDS)
                time.sleep(_HOLD_SECONDS)
                connection.commit()
                results["first"] = "ok"
            except sqlite3.OperationalError as exc:
                results["first"] = str(exc)
            finally:
                connection.close()

        def writer_second() -> None:
            connection = second.raw_connection()
            try:
                if not first_ready.wait(_WAIT_SECONDS):
                    results["second"] = "first writer never started"
                    return
                second_attempting.set()
                cursor = connection.cursor()
                cursor.execute("BEGIN IMMEDIATE")
                cursor.execute(_INSERT, ("second",))
                connection.commit()
                results["second"] = "ok"
            except sqlite3.OperationalError as exc:
                results["second"] = str(exc)
            finally:
                connection.close()

        _run(
            [
                threading.Thread(target=writer_first),
                threading.Thread(target=writer_second),
            ]
        )

        assert results == {"first": "ok", "second": "ok"}
        assert _count(database) == 2
    finally:
        first.dispose()
        second.dispose()


def test_without_a_busy_timeout_the_second_writer_is_locked_out(
    tmp_path: Path,
) -> None:
    database = tmp_path / "aivia.db"
    first = _prepare(database)
    second = create_database_engine(database, busy_timeout_ms=0)
    try:
        assert _pragma(second, "journal_mode") == "wal"
        assert _pragma(second, "busy_timeout") == 0

        first_ready = threading.Event()
        second_finished = threading.Event()
        results: dict[str, str] = {}

        def writer_first() -> None:
            connection = first.raw_connection()
            try:
                cursor = connection.cursor()
                cursor.execute("BEGIN IMMEDIATE")
                cursor.execute(_INSERT, ("first",))
                first_ready.set()
                second_finished.wait(_WAIT_SECONDS)
                connection.commit()
                results["first"] = "ok"
            except sqlite3.OperationalError as exc:
                results["first"] = str(exc)
            finally:
                connection.close()

        def writer_second() -> None:
            connection = second.raw_connection()
            try:
                if not first_ready.wait(_WAIT_SECONDS):
                    results["second"] = "first writer never started"
                    return
                cursor = connection.cursor()
                cursor.execute("BEGIN IMMEDIATE")
                cursor.execute(_INSERT, ("second",))
                connection.commit()
                results["second"] = "ok"
            except sqlite3.OperationalError as exc:
                results["second"] = str(exc)
            finally:
                second_finished.set()
                connection.close()

        _run(
            [
                threading.Thread(target=writer_first),
                threading.Thread(target=writer_second),
            ]
        )

        assert results["first"] == "ok"
        assert "database is locked" in results["second"]
        assert _count(database) == 1
    finally:
        first.dispose()
        second.dispose()

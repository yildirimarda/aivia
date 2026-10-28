"""The core run, stage run and job tables and their queue indexes."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, inspect, select, text
from sqlalchemy.exc import IntegrityError

from aivia.core.db import (
    CLAIM_INDEX_NAME,
    IDEMPOTENCY_INDEX_NAME,
    create_database_engine,
    jobs,
    metadata,
    runs,
    stage_runs,
    upgrade_database,
)

CORE_TABLES = {"runs", "stage_runs", "jobs"}


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    database = tmp_path / "data" / "aivia.db"
    upgrade_database(database)
    created = create_database_engine(database)
    try:
        yield created
    finally:
        created.dispose()


def test_the_migration_creates_the_core_tables(engine: Engine) -> None:
    assert set(inspect(engine).get_table_names()) >= CORE_TABLES


def test_the_core_tables_are_registered_on_the_shared_metadata() -> None:
    assert set(metadata.tables) >= CORE_TABLES


def test_runs_carry_the_run_record_columns(engine: Engine) -> None:
    columns = {column["name"] for column in inspect(engine).get_columns("runs")}

    assert {
        "id",
        "kind",
        "state",
        "idempotency_key",
        "toolchain_fingerprint",
        "config_snapshot",
        "cost_usd",
        "estimated_cost_usd",
        "started_at",
        "finished_at",
        "paused_at",
        "schema_version",
        "created_at",
        "updated_at",
    } <= columns


def test_stage_runs_carry_the_stage_result_columns(engine: Engine) -> None:
    columns = {column["name"] for column in inspect(engine).get_columns("stage_runs")}

    assert {
        "run_id",
        "stage",
        "status",
        "index",
        "input_hash",
        "output_hash",
        "artifacts",
        "cost_usd",
        "toolchain_fingerprint",
        "metadata",
        "started_at",
        "finished_at",
        "schema_version",
        "created_at",
        "updated_at",
    } <= columns


def test_jobs_carry_the_queue_columns(engine: Engine) -> None:
    columns = {column["name"] for column in inspect(engine).get_columns("jobs")}

    assert {
        "id",
        "kind",
        "state",
        "run_id",
        "attempts",
        "max_attempts",
        "not_before",
        "heartbeat_at",
        "claimed_by",
        "claimed_at",
        "payload",
        "metadata",
        "schema_version",
        "created_at",
        "updated_at",
    } <= columns


def test_jobs_have_a_claim_index(engine: Engine) -> None:
    indexes = {index["name"]: index for index in inspect(engine).get_indexes("jobs")}

    claim = indexes[CLAIM_INDEX_NAME]
    assert not claim["unique"]
    assert claim["column_names"] == ["state", "not_before"]


def test_runs_have_a_unique_idempotency_key_index(engine: Engine) -> None:
    indexes = {index["name"]: index for index in inspect(engine).get_indexes("runs")}

    idempotency = indexes[IDEMPOTENCY_INDEX_NAME]
    assert idempotency["unique"]
    assert idempotency["column_names"] == ["idempotency_key"]

    with engine.connect() as connection:
        definition = connection.execute(
            text("SELECT sql FROM sqlite_master WHERE type = 'index' AND name = :name"),
            {"name": IDEMPOTENCY_INDEX_NAME},
        ).scalar_one()
    assert "WHERE" in definition


def test_a_repeated_idempotency_key_is_rejected(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            runs.insert().values(id="run_first", idempotency_key="order-7")
        )

    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(
            runs.insert().values(id="run_second", idempotency_key="order-7")
        )


def test_runs_without_an_idempotency_key_do_not_collide(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(runs.insert().values(id="run_missing"))
        connection.execute(runs.insert().values(id="run_empty_a", idempotency_key=""))
        connection.execute(runs.insert().values(id="run_empty_b", idempotency_key=""))
        connection.execute(runs.insert().values(id="run_null_a", idempotency_key=None))
        connection.execute(runs.insert().values(id="run_null_b", idempotency_key=None))

    with engine.connect() as connection:
        assert connection.execute(
            select(runs.c.id).order_by(runs.c.id)
        ).scalars().all() == [
            "run_empty_a",
            "run_empty_b",
            "run_missing",
            "run_null_a",
            "run_null_b",
        ]


def test_stage_runs_belong_to_a_run_and_are_deleted_with_it(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(runs.insert().values(id="run_1"))
        connection.execute(
            stage_runs.insert().values(
                run_id="run_1",
                stage="write",
                status="succeeded",
                artifacts=["script.json"],
            )
        )

    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(stage_runs.insert().values(run_id="missing", stage="write"))

    with engine.begin() as connection:
        connection.execute(runs.delete().where(runs.c.id == "run_1"))

    with engine.connect() as connection:
        assert connection.execute(select(stage_runs)).all() == []


def test_jobs_reject_attempts_above_the_limit(engine: Engine) -> None:
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(jobs.insert().values(id="job_1", attempts=4, max_attempts=3))


def test_a_pending_job_can_be_selected_with_the_claim_index(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(runs.insert().values(id="run_1"))
        connection.execute(
            jobs.insert().values(id="job_pending", run_id="run_1", state="pending")
        )
        connection.execute(
            jobs.insert().values(id="job_running", run_id="run_1", state="running")
        )

    with engine.connect() as connection:
        claimable = (
            connection.execute(
                select(jobs.c.id)
                .where(jobs.c.state == "pending")
                .order_by(jobs.c.not_before, jobs.c.created_at)
            )
            .scalars()
            .all()
        )

    assert claimable == ["job_pending"]

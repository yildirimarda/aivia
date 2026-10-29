"""Submissions above the queue depth limit fail with a quota error."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import Engine

from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.errors import QuotaError
from aivia.core.jobs import JobQueue
from aivia.core.models import JobState

NOW = datetime(2026, 1, 2, 10, 0, tzinfo=UTC)


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


def test_enqueue_below_the_limit_reports_the_pending_depth(engine: Engine) -> None:
    queue = JobQueue(engine, max_queue_depth=2)

    assert queue.depth() == 0
    first = queue.enqueue(kind="run")
    second = queue.enqueue(kind="run")

    assert queue.depth() == 2
    assert {first.state, second.state} == {JobState.PENDING}


def test_enqueue_at_the_limit_raises_a_quota_error_with_the_depth(
    engine: Engine,
) -> None:
    queue = JobQueue(engine, max_queue_depth=2)
    queue.enqueue()
    queue.enqueue()

    with pytest.raises(QuotaError) as caught:
        queue.enqueue(kind="run")

    error = caught.value
    assert error.details["depth"] == 2
    assert error.details["max_queue_depth"] == 2
    assert "2" in str(error)
    assert queue.depth() == 2


def test_a_per_call_limit_overrides_the_queue_default(engine: Engine) -> None:
    queue = JobQueue(engine)

    queue.enqueue(max_queue_depth=1)
    with pytest.raises(QuotaError):
        queue.enqueue(max_queue_depth=1)


def test_the_jobs_config_supplies_the_default_limit(engine: Engine) -> None:
    queue = JobQueue(engine, config={"max_queue_depth": 1})

    queue.enqueue()
    with pytest.raises(QuotaError):
        queue.enqueue()


def test_zero_disables_backpressure(engine: Engine) -> None:
    queue = JobQueue(engine, max_queue_depth=0)

    for _ in range(5):
        queue.enqueue()

    assert queue.depth() == 5


def test_only_pending_jobs_count_toward_the_depth(engine: Engine) -> None:
    queue = JobQueue(engine, max_queue_depth=1)
    first = queue.enqueue()
    assert queue.claim("worker-1", now=NOW) is not None
    assert queue.depth() == 0

    second = queue.enqueue()

    assert queue.depth() == 1
    assert second.state is JobState.PENDING
    queue.complete(first.id, "worker-1", now=NOW)
    assert queue.claim("worker-2", now=NOW) is not None
    assert queue.depth() == 0
    assert first.id != second.id


def test_a_permanent_failure_frees_a_queue_slot(engine: Engine) -> None:
    queue = JobQueue(engine, max_queue_depth=1)
    job = queue.enqueue()
    assert queue.claim("worker-1", now=NOW) is not None
    queue.fail(job.id, "worker-1", now=NOW)

    assert queue.depth() == 0
    queue.enqueue()


def test_concurrent_submissions_never_overshoot_the_limit(tmp_path: Path) -> None:
    database = tmp_path / "aivia.db"
    setup = create_database_engine(database)
    upgrade_database(engine=setup)
    setup.dispose()

    limit = 5
    threads = 12
    barrier = threading.Barrier(threads)
    accepted: list[str] = []
    rejected: list[QuotaError] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def submit(index: int) -> None:
        engine = create_database_engine(database)
        try:
            queue = JobQueue(engine, max_queue_depth=limit)
            barrier.wait(timeout=5)
            try:
                job = queue.enqueue(kind=f"job-{index}")
            except QuotaError as exc:
                with lock:
                    rejected.append(exc)
            else:
                with lock:
                    accepted.append(job.id)
        except BaseException as exc:  # pragma: no cover - reported below
            with lock:
                errors.append(exc)
        finally:
            engine.dispose()

    workers = [
        threading.Thread(target=submit, args=(index,)) for index in range(threads)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(20)
    assert not any(worker.is_alive() for worker in workers)

    assert errors == []
    assert len(accepted) == limit
    assert len(rejected) == threads - limit
    assert all(error.details["depth"] == limit for error in rejected)

    check = create_database_engine(database)
    try:
        assert JobQueue(check).depth() == limit
    finally:
        check.dispose()

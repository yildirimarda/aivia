"""The durable job queue claims atomically and tracks ownership per worker."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import Engine

from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.jobs import (
    JobNotClaimedError,
    JobNotFoundError,
    JobQueue,
)
from aivia.core.models import Job, JobState

NOW = datetime(2026, 1, 2, 10, 0, tzinfo=UTC)
WORKERS = 6
JOBS = 24


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


@pytest.fixture
def queue(engine: Engine) -> JobQueue:
    return JobQueue(engine)


def test_enqueue_defaults_to_a_pending_job(queue: JobQueue) -> None:
    job = queue.enqueue(payload={"stage": "write"})

    assert job.state is JobState.PENDING
    assert job.attempts == 0
    assert job.claimed_by == ""
    assert job.payload == {"stage": "write"}
    assert queue.get(job.id) == job


def test_claim_records_the_worker_and_starts_the_first_attempt(
    queue: JobQueue,
) -> None:
    job = queue.enqueue(kind="run", max_attempts=2)

    claimed = queue.claim("worker-1", now=NOW)

    assert claimed is not None
    assert claimed.id == job.id
    assert claimed.state is JobState.RUNNING
    assert claimed.attempts == 1
    assert claimed.claimed_by == "worker-1"
    assert claimed.claimed_at == NOW
    assert claimed.started_at == NOW
    assert claimed.heartbeat_at == NOW
    assert claimed.finished_at is None
    assert queue.claim("worker-2", now=NOW) is None


def test_claim_only_takes_due_jobs_and_honours_not_before(queue: JobQueue) -> None:
    job = queue.enqueue(not_before=NOW + timedelta(seconds=30))

    assert queue.claim("worker-1", now=NOW) is None
    assert queue.claim("worker-1", now=NOW + timedelta(seconds=29)) is None

    claimed = queue.claim("worker-1", now=NOW + timedelta(seconds=30))

    assert claimed is not None
    assert claimed.id == job.id


def test_claim_can_select_a_kind_or_a_specific_job(queue: JobQueue) -> None:
    first = queue.enqueue(kind="stage")
    second = queue.enqueue(kind="publish")

    claimed = queue.claim("worker-1", kind="publish", now=NOW)

    assert claimed is not None
    assert claimed.id == second.id
    assert queue.claim("worker-1", job_id=first.id, now=NOW) is not None


def test_claim_skips_jobs_that_exhausted_their_attempts(queue: JobQueue) -> None:
    queue.enqueue(Job(attempts=3, max_attempts=3))

    assert queue.claim("worker-1", now=NOW) is None


def test_heartbeat_is_scoped_to_the_claiming_worker(queue: JobQueue) -> None:
    job = queue.enqueue()
    assert queue.claim("worker-1", now=NOW) is not None

    beat = queue.heartbeat(job.id, "worker-1", now=NOW + timedelta(seconds=30))

    assert beat.heartbeat_at == NOW + timedelta(seconds=30)
    with pytest.raises(JobNotClaimedError):
        queue.heartbeat(job.id, "worker-2", now=NOW)
    with pytest.raises(JobNotFoundError):
        queue.heartbeat("job_missing", "worker-1", now=NOW)


def test_complete_marks_success_and_releases_the_claim(queue: JobQueue) -> None:
    job = queue.enqueue()
    assert queue.claim("worker-1", now=NOW) is not None

    done = queue.complete(job.id, "worker-1", now=NOW + timedelta(seconds=5))

    assert done.state is JobState.SUCCEEDED
    assert done.finished_at == NOW + timedelta(seconds=5)
    assert done.heartbeat_at == NOW + timedelta(seconds=5)
    with pytest.raises(JobNotClaimedError):
        queue.complete(job.id, "worker-1", now=NOW)
    assert queue.claim("worker-1", now=NOW) is None


def test_complete_rejects_a_worker_that_does_not_own_the_job(
    queue: JobQueue,
) -> None:
    job = queue.enqueue()
    assert queue.claim("worker-1", now=NOW) is not None

    with pytest.raises(JobNotClaimedError):
        queue.complete(job.id, "worker-2", now=NOW)
    with pytest.raises(JobNotFoundError):
        queue.complete("job_missing", "worker-1", now=NOW)


def test_fail_permanent_records_the_error_and_stops_the_job(
    queue: JobQueue,
) -> None:
    job = queue.enqueue(max_attempts=3)
    assert queue.claim("worker-1", now=NOW) is not None

    failed = queue.fail(
        job.id,
        "worker-1",
        error_code="validation",
        error_message="bad output",
        now=NOW + timedelta(seconds=5),
    )

    assert failed.state is JobState.FAILED
    assert failed.error_code == "validation"
    assert failed.error_message == "bad output"
    assert failed.finished_at == NOW + timedelta(seconds=5)
    assert failed.attempts == 1
    assert queue.claim("worker-1", now=NOW) is None


def test_fail_retryable_requeues_with_backoff_then_dead_letters(
    queue: JobQueue,
) -> None:
    job = queue.enqueue(max_attempts=2)
    assert queue.claim("worker-1", now=NOW) is not None

    retried = queue.fail(
        job.id,
        "worker-1",
        error_code="timeout",
        error_message="slow provider",
        retryable=True,
        retry_after=30,
        now=NOW,
    )

    assert retried.state is JobState.PENDING
    assert retried.not_before == NOW + timedelta(seconds=30)
    assert retried.claimed_by == ""
    assert retried.claimed_at is None
    assert retried.heartbeat_at is None
    assert retried.attempts == 1
    assert retried.error_code == "timeout"
    assert queue.claim("worker-2", now=NOW + timedelta(seconds=29)) is None

    second = queue.claim("worker-2", now=NOW + timedelta(seconds=30))
    assert second is not None
    assert second.attempts == 2

    dead = queue.fail(
        job.id,
        "worker-2",
        error_code="timeout",
        error_message="still slow",
        retryable=True,
        now=NOW + timedelta(seconds=31),
    )

    assert dead.state is JobState.DEAD_LETTER
    assert dead.finished_at == NOW + timedelta(seconds=31)
    assert dead.attempts == 2
    assert queue.claim("worker-1", now=NOW + timedelta(minutes=5)) is None


def test_fail_rejects_a_worker_that_does_not_own_the_job(queue: JobQueue) -> None:
    job = queue.enqueue()
    assert queue.claim("worker-1", now=NOW) is not None

    with pytest.raises(JobNotClaimedError):
        queue.fail(job.id, "worker-2", now=NOW)
    with pytest.raises(JobNotFoundError):
        queue.fail("job_missing", "worker-1", now=NOW)


def test_two_workers_racing_for_one_job_only_one_wins(tmp_path: Path) -> None:
    database = tmp_path / "aivia.db"
    setup = create_database_engine(database)
    upgrade_database(engine=setup)
    job = JobQueue(setup).enqueue()
    setup.dispose()

    barrier = threading.Barrier(2)
    claimed: list[str | None] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker(name: str) -> None:
        engine = create_database_engine(database)
        try:
            queue = JobQueue(engine)
            barrier.wait(timeout=5)
            result = queue.claim(name)
            with lock:
                claimed.append(None if result is None else result.id)
        except BaseException as exc:  # pragma: no cover - reported below
            with lock:
                errors.append(exc)
        finally:
            engine.dispose()

    threads = [
        threading.Thread(target=worker, args=(f"worker-{index}",)) for index in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert not any(thread.is_alive() for thread in threads)

    assert errors == []
    assert claimed.count(job.id) == 1
    assert claimed.count(None) == 1


def test_concurrent_workers_never_claim_the_same_job(tmp_path: Path) -> None:
    database = tmp_path / "aivia.db"
    setup = create_database_engine(database)
    upgrade_database(engine=setup)
    queue = JobQueue(setup)
    job_ids = {queue.enqueue(kind="run").id for _ in range(JOBS)}
    setup.dispose()

    barrier = threading.Barrier(WORKERS)
    claimed: list[str] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        engine = create_database_engine(database)
        try:
            worker_queue = JobQueue(engine)
            barrier.wait(timeout=5)
            while True:
                job = worker_queue.claim(f"worker-{index}")
                if job is None:
                    break
                with lock:
                    claimed.append(job.id)
        except BaseException as exc:  # pragma: no cover - reported below
            with lock:
                errors.append(exc)
        finally:
            engine.dispose()

    threads = [
        threading.Thread(target=worker, args=(index,)) for index in range(WORKERS)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(20)
    assert not any(thread.is_alive() for thread in threads)

    assert errors == []
    assert len(claimed) == len(set(claimed)) == JOBS
    assert set(claimed) == job_ids

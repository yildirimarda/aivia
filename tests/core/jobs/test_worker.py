"""The worker loop retries, recovers stale claims and shuts down gracefully."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import Engine

from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.errors import RetryableError, ValidationError
from aivia.core.jobs import (
    STALE_HEARTBEAT_CODE,
    JobQueue,
    Worker,
    WorkerOutcome,
)
from aivia.core.models import Job, JobState

NOW = datetime(2026, 1, 2, 10, 0, tzinfo=UTC)


class Clock:
    """A mutable clock so tests never depend on wall time."""

    def __init__(self, now: datetime = NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


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


def test_worker_completes_a_job_and_calls_the_handler_once(queue: JobQueue) -> None:
    job = queue.enqueue()
    seen: list[str] = []
    clock = Clock()

    async def handler(job: Job) -> None:
        seen.append(job.id)

    worker = Worker(queue, handler, worker_id="worker-1", clock=clock)

    outcome = asyncio.run(worker.run_once(now=clock.now))

    assert outcome is not None
    assert outcome.succeeded is True
    assert outcome.job.state is JobState.SUCCEEDED
    assert outcome.job.attempts == 1
    assert outcome.job.id == job.id
    assert seen == [outcome.job.id]
    assert worker.processed == 1
    assert worker.succeeded == 1


def test_worker_retries_a_retryable_failure_with_exponential_backoff(
    queue: JobQueue,
) -> None:
    job = queue.enqueue(max_attempts=3)
    clock = Clock()
    calls = 0

    async def handler(claimed: Job) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RetryableError("temporary provider outage")

    worker = Worker(
        queue,
        handler,
        worker_id="worker-1",
        retry_delay_seconds=5,
        backoff_multiplier=2,
        max_retry_delay_seconds=12,
        clock=clock,
    )

    first = asyncio.run(worker.run_once(now=clock.now))

    assert first is not None
    assert first.retried is True
    assert first.job.state is JobState.PENDING
    assert first.job.attempts == 1
    assert first.job.not_before == NOW + timedelta(seconds=5)
    assert first.job.error_code == "retryable"

    clock.now = NOW + timedelta(seconds=4)
    assert asyncio.run(worker.run_once(now=clock.now)) is None

    clock.now = NOW + timedelta(seconds=5)
    second = asyncio.run(worker.run_once(now=clock.now))

    assert second is not None
    assert second.succeeded is True
    assert second.job.id == job.id
    assert second.job.state is JobState.SUCCEEDED
    assert second.job.attempts == 2
    assert worker.backoff_delay(1) == 5
    assert worker.backoff_delay(2) == 10
    assert worker.backoff_delay(3) == 12


def test_worker_fails_permanent_errors_without_another_attempt(queue: JobQueue) -> None:
    job = queue.enqueue(max_attempts=3)
    clock = Clock()

    async def handler(_job: Job) -> None:
        raise ValidationError("the output does not match the schema")

    worker = Worker(queue, handler, worker_id="worker-1", clock=clock)

    outcome = asyncio.run(worker.run_once(now=clock.now))

    assert outcome is not None
    assert outcome.failed is True
    assert outcome.retryable is False
    assert outcome.job.state is JobState.FAILED
    assert outcome.job.attempts == 1
    assert outcome.job.error_code == "validation"
    assert outcome.job.error_message == "the output does not match the schema"
    clock.now = NOW + timedelta(minutes=5)
    assert asyncio.run(worker.run_once(now=clock.now)) is None
    assert queue.get(job.id) == outcome.job


def test_worker_dead_letters_the_last_retryable_attempt(queue: JobQueue) -> None:
    job = queue.enqueue(max_attempts=1)
    clock = Clock()

    async def handler(_job: Job) -> None:
        raise RetryableError("still unavailable")

    worker = Worker(queue, handler, worker_id="worker-1", clock=clock)

    outcome = asyncio.run(worker.run_once(now=clock.now))

    assert outcome is not None
    assert outcome.dead_letter is True
    assert outcome.retryable is True
    assert outcome.job.state is JobState.DEAD_LETTER
    assert outcome.job.finished_at == NOW
    assert outcome.job.attempts == 1
    assert worker.dead_lettered == 1
    assert queue.claim("worker-2", now=NOW + timedelta(days=1)) is None
    assert queue.get(job.id) == outcome.job


def test_worker_recovers_a_stale_claim_before_claiming(queue: JobQueue) -> None:
    job = queue.enqueue(max_attempts=3)
    assert queue.claim("dead-worker", now=NOW) is not None
    clock = Clock(NOW + timedelta(seconds=121))

    async def handler(_job: Job) -> None:
        return None

    worker = Worker(
        queue,
        handler,
        worker_id="worker-1",
        heartbeat_seconds=10,
        stale_after_seconds=120,
        clock=clock,
    )

    outcome = asyncio.run(worker.run_once(now=clock.now))

    assert outcome is not None
    assert outcome.succeeded is True
    assert outcome.job.id == job.id
    assert outcome.job.state is JobState.SUCCEEDED
    assert outcome.job.attempts == 2


def test_recover_stale_returns_a_job_to_pending_or_dead_letters_it(
    queue: JobQueue,
) -> None:
    retryable = queue.enqueue(max_attempts=3)
    exhausted = queue.enqueue(max_attempts=1)
    assert queue.claim("dead-1", job_id=retryable.id, now=NOW) is not None
    assert queue.claim("dead-2", job_id=exhausted.id, now=NOW) is not None
    clock = Clock(NOW + timedelta(seconds=121))

    worker = Worker(
        queue,
        lambda _job: None,
        worker_id="worker-1",
        heartbeat_seconds=10,
        stale_after_seconds=120,
        clock=clock,
    )
    recovered = worker.recover_stale(now=clock.now)

    assert {job.id for job in recovered} == {retryable.id, exhausted.id}
    pending = queue.get(retryable.id)
    dead = queue.get(exhausted.id)
    assert pending is not None and dead is not None
    assert pending.state is JobState.PENDING
    assert pending.claimed_by == ""
    assert pending.claimed_at is None
    assert pending.heartbeat_at is None
    assert pending.not_before == clock.now
    assert pending.error_code == STALE_HEARTBEAT_CODE
    assert dead.state is JobState.DEAD_LETTER
    assert dead.finished_at == clock.now
    assert dead.error_code == STALE_HEARTBEAT_CODE
    assert dead.attempts == 1


def test_worker_keeps_the_heartbeat_alive_while_a_handler_runs(
    queue: JobQueue,
) -> None:
    queue.enqueue()

    async def handler(job: Job) -> None:
        for _ in range(200):
            current = queue.get(job.id)
            if (
                current is not None
                and current.heartbeat_at is not None
                and job.heartbeat_at is not None
                and current.heartbeat_at > job.heartbeat_at
            ):
                return
            await asyncio.sleep(0.005)
        raise AssertionError("the worker never refreshed the heartbeat")

    worker = Worker(
        queue,
        handler,
        worker_id="worker-1",
        heartbeat_seconds=0.01,
        stale_after_seconds=1,
    )

    outcome = asyncio.run(worker.run_once())

    assert outcome is not None
    assert outcome.succeeded is True
    assert outcome.job.heartbeat_at is not None
    assert outcome.job.claimed_at is not None
    assert outcome.job.heartbeat_at > outcome.job.claimed_at


def test_request_stop_finishes_the_current_job_and_claims_no_more(
    queue: JobQueue,
) -> None:
    first = queue.enqueue()
    second = queue.enqueue()
    clock = Clock()
    worker: Worker

    async def handler(job: Job) -> None:
        if job.id == first.id:
            worker.request_stop()

    worker = Worker(queue, handler, worker_id="worker-1", clock=clock)

    processed = asyncio.run(worker.run())

    assert processed == 1
    assert worker.stopping is True
    first_job = queue.get(first.id)
    second_job = queue.get(second.id)
    assert first_job is not None and first_job.state is JobState.SUCCEEDED
    assert second_job is not None and second_job.state is JobState.PENDING

    worker.clear_stop()
    clock.now = NOW + timedelta(seconds=1)
    assert asyncio.run(worker.run(max_jobs=1)) == 1
    second_job = queue.get(second.id)
    assert second_job is not None and second_job.state is JobState.SUCCEEDED


def test_worker_rejects_settings_that_cannot_be_stale(queue: JobQueue) -> None:
    with pytest.raises(ValueError, match="stale_after_seconds"):
        Worker(
            queue,
            lambda _job: None,
            heartbeat_seconds=30,
            stale_after_seconds=30,
        )


def test_run_once_returns_none_when_the_queue_is_empty(queue: JobQueue) -> None:
    worker = Worker(queue, lambda _job: None, clock=Clock())

    assert asyncio.run(worker.run_once(now=NOW)) is None
    assert worker.processed == 0


def test_worker_outcome_reports_a_lost_claim(queue: JobQueue) -> None:
    job = queue.enqueue()
    assert queue.claim("worker-1", now=NOW) is not None
    assert queue.fail(job.id, "worker-1", retryable=True, now=NOW) is not None
    assert queue.claim("worker-2", now=NOW) is not None

    worker = Worker(queue, lambda _job: None, worker_id="worker-1", clock=Clock())
    outcome = worker._record_success(job)

    assert isinstance(outcome, WorkerOutcome)
    assert outcome.succeeded is False
    assert outcome.claimed is False
    assert outcome.lost_claim is True

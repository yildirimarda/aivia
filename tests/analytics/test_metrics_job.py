"""Snapshotting a post's metrics at the 24h and 7d marks is idempotent and retryable.

A snapshot is stored once per (post, mark). A partially failed fetch is reported
as a retryable error so the durable job queue can attempt the snapshot again
without duplicating a row.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import Engine

from aivia.analytics.collect import CANONICAL_METRICS
from aivia.analytics.jobs import (
    MARK_SEVEN_DAYS,
    MARK_TWENTY_FOUR_HOURS,
    MARKS,
    snapshot_post_metrics,
)
from aivia.core.db import (
    PostMetricRepository,
    PostRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.errors import RetryableError
from aivia.core.jobs import JobQueue, Worker, WorkerOutcome
from aivia.core.models import Job, JobState

NOW = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)

POST_ID = "post-01"
PLATFORM = "tiktok"
PUBLISHED_AT = NOW
CAPTURED_24H = PUBLISHED_AT + timedelta(hours=24)
CAPTURED_7D = PUBLISHED_AT + timedelta(days=7)


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


def _make_post(
    engine: Engine, post_id: str = POST_ID, platform: str = PLATFORM
) -> None:
    PostRepository(engine).create(
        id=post_id,
        platform=platform,
        state="published",
        published_at=PUBLISHED_AT,
    )


def _raw(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = dict.fromkeys(CANONICAL_METRICS)
    base.update(overrides)
    return base


def test_snapshot_stores_normalized_metrics(engine: Engine) -> None:
    _make_post(engine)
    calls = 0

    def fetch() -> Mapping[str, object]:
        nonlocal calls
        calls += 1
        return _raw(views=1_000, likes=50, comments=10)

    stored = snapshot_post_metrics(
        engine,
        POST_ID,
        PLATFORM,
        MARK_TWENTY_FOUR_HOURS,
        CAPTURED_24H,
        fetch_metrics=fetch,
    )
    assert stored is not None

    assert calls == 1
    assert stored["post_id"] == POST_ID
    assert stored["platform"] == PLATFORM
    assert stored["views"] == 1_000
    assert stored["likes"] == 50
    assert stored["comments"] == 10
    assert stored["watch_time_seconds"] == 0.0
    assert stored["metadata"] == {"mark": MARK_TWENTY_FOUR_HOURS}
    assert stored["raw"]["views"] == 1_000


def test_snapshot_is_idempotent_per_post_and_mark(engine: Engine) -> None:
    _make_post(engine)
    calls = 0

    def fetch() -> Mapping[str, object]:
        nonlocal calls
        calls += 1
        return _raw(views=1_000)

    first = snapshot_post_metrics(
        engine,
        POST_ID,
        PLATFORM,
        MARK_TWENTY_FOUR_HOURS,
        CAPTURED_24H,
        fetch_metrics=fetch,
    )
    assert first is not None
    second = snapshot_post_metrics(
        engine,
        POST_ID,
        PLATFORM,
        MARK_TWENTY_FOUR_HOURS,
        CAPTURED_24H,
        fetch_metrics=fetch,
    )

    assert calls == 1
    assert second is not None
    assert second["id"] == first["id"]
    assert second["views"] == first["views"]

    rows = PostMetricRepository(engine).list(post_id=POST_ID)
    at_mark = [
        row for row in rows if row["metadata"] == {"mark": MARK_TWENTY_FOUR_HOURS}
    ]
    assert at_mark == [first]


def test_each_mark_is_snapshotted_independently(engine: Engine) -> None:
    _make_post(engine)

    snapshot_post_metrics(
        engine,
        POST_ID,
        PLATFORM,
        MARK_TWENTY_FOUR_HOURS,
        CAPTURED_24H,
        fetch_metrics=lambda: _raw(views=100),
    )
    snapshot_post_metrics(
        engine,
        POST_ID,
        PLATFORM,
        MARK_SEVEN_DAYS,
        CAPTURED_7D,
        fetch_metrics=lambda: _raw(views=200),
    )

    rows = PostMetricRepository(engine).list(post_id=POST_ID)
    assert {row["metadata"]["mark"] for row in rows} == set(MARKS)
    views = {row["metadata"]["mark"]: row["views"] for row in rows}
    assert views == {MARK_TWENTY_FOUR_HOURS: 100, MARK_SEVEN_DAYS: 200}


def test_fetch_failure_is_retryable_and_stores_nothing(engine: Engine) -> None:
    _make_post(engine)

    def fetch() -> Mapping[str, object]:
        raise ConnectionError("platform offline")

    with pytest.raises(RetryableError, match="could not fetch metrics"):
        snapshot_post_metrics(
            engine,
            POST_ID,
            PLATFORM,
            MARK_TWENTY_FOUR_HOURS,
            CAPTURED_24H,
            fetch_metrics=fetch,
        )

    assert PostMetricRepository(engine).list(post_id=POST_ID) == []


def test_unknown_mark_is_rejected(engine: Engine) -> None:
    _make_post(engine)

    with pytest.raises(ValueError, match="unknown mark"):
        snapshot_post_metrics(
            engine,
            POST_ID,
            PLATFORM,
            "midnight",
            CAPTURED_24H,
            fetch_metrics=lambda: _raw(views=1),
        )


def test_job_queue_retries_a_flaky_metrics_fetch(engine: Engine) -> None:
    _make_post(engine)
    queue = JobQueue(engine)
    attempts = {"n": 0}

    def fetch() -> Mapping[str, object]:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("offline")
        return _raw(views=42)

    job = queue.enqueue(
        kind="metrics",
        payload={
            "post_id": POST_ID,
            "platform": PLATFORM,
            "mark": MARK_TWENTY_FOUR_HOURS,
        },
        max_attempts=5,
    )

    clock = Clock()

    async def handler(job: Job) -> None:
        payload = job.payload
        snapshot_post_metrics(
            engine,
            str(payload["post_id"]),
            str(payload["platform"]),
            str(payload["mark"]),
            CAPTURED_24H,
            fetch_metrics=fetch,
        )

    worker = Worker(
        queue,
        handler,
        worker_id="w-1",
        clock=clock,
        heartbeat_seconds=1,
        stale_after_seconds=60,
        poll_seconds=0.01,
        retry_delay_seconds=0,
    )

    outcome: WorkerOutcome | None = None
    for _ in range(10):
        outcome = asyncio.run(worker.run_once(now=clock.now))
        if outcome is not None and outcome.succeeded:
            break

    assert outcome is not None and outcome.succeeded is True
    assert outcome.job.state is JobState.SUCCEEDED
    assert outcome.job.id == job.id
    assert attempts["n"] == 3

    rows = PostMetricRepository(engine).list(post_id=POST_ID)
    at_mark = [
        row for row in rows if row["metadata"] == {"mark": MARK_TWENTY_FOUR_HOURS}
    ]
    assert len(at_mark) == 1
    assert at_mark[0]["views"] == 42

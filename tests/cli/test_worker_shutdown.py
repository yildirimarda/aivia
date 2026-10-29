"""SIGINT and SIGTERM drain the current job and stop claiming new ones."""

from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine

from aivia.cli.signals import stop_on_signals
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.jobs import JobQueue, Worker
from aivia.core.models import Job, JobState


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


@pytest.mark.parametrize("stop_signal", [signal.SIGINT, signal.SIGTERM])
def test_stop_signal_drains_the_current_job(
    engine: Engine,
    stop_signal: signal.Signals,
) -> None:
    queue = JobQueue(engine)
    first = queue.enqueue()
    second = queue.enqueue()
    started = asyncio.Event()
    release = asyncio.Event()
    completed: list[str] = []

    async def handler(job: Job) -> None:
        if job.id == first.id:
            started.set()
            await release.wait()
            completed.append(job.id)

    async def scenario() -> int:
        worker = Worker(
            queue,
            handler,
            worker_id="worker-1",
            poll_seconds=0.01,
            heartbeat_seconds=1,
            stale_after_seconds=2,
        )
        with stop_on_signals(worker):
            running = asyncio.create_task(worker.run())
            await asyncio.wait_for(started.wait(), timeout=2)
            os.kill(os.getpid(), stop_signal)
            while not worker.stopping:
                await asyncio.sleep(0.01)
            release.set()
            return await asyncio.wait_for(running, timeout=2)

    processed = asyncio.run(scenario())

    assert processed == 1
    assert completed == [first.id]
    first_job = queue.get(first.id)
    second_job = queue.get(second.id)
    assert first_job is not None and first_job.state is JobState.SUCCEEDED
    assert second_job is not None and second_job.state is JobState.PENDING

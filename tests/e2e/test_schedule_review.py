"""Scheduled end to end review: a cron schedule fires a run, the run passes
through review (approval proceeds, rejection stops) and can pause on a budget
cap then resume."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine

from aivia.core.config.ops import ReviewConfig
from aivia.core.config.schedule import ScheduleConfig
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.db.repos import RunRepository
from aivia.core.jobs import JobQueue
from aivia.core.locks import RunLockManager
from aivia.core.models import RunState
from aivia.core.run_state import RunStateMachine
from aivia.core.scheduler import CronScheduler
from aivia.review.review_state import ReviewState, ReviewStateMachine


class Clock:
    """A mutable clock so tests never depend on wall time."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    database = tmp_path / "data" / "aivia.db"
    upgrade_database(database)
    created = create_database_engine(database)
    try:
        yield created
    finally:
        created.dispose()


@pytest.fixture
def clock() -> Clock:
    return Clock(datetime(2026, 1, 15, 10, 0, tzinfo=UTC))


@pytest.fixture
def scheduler_setup(
    engine: Engine, clock: Clock, tmp_path: Path
) -> tuple[CronScheduler, Clock, JobQueue]:
    queue = JobQueue(engine)
    locks = RunLockManager(max_parallel=1, lock_dir=tmp_path / "locks")
    schedule = ScheduleConfig.from_mapping(
        {
            "timezone": "UTC",
            "schedules": {
                "daily": {"profile": "default", "cron": "* * * * *"},
            },
        }
    )
    scheduler = CronScheduler(schedule, queue, locks, clock=clock)
    return scheduler, clock, queue


@pytest.fixture
def started_run(
    scheduler_setup: tuple[CronScheduler, Clock, JobQueue],
    engine: Engine,
) -> tuple[RunStateMachine, Any, ReviewStateMachine, Any, Clock]:
    """Tick the schedule into a due job, claim it, create and start a run, and
    open a pending review for the run's profile."""
    scheduler, clock, queue = scheduler_setup

    # First tick computes the first fire strictly after now; the second tick,
    # once the clock reaches it, enqueues the scheduled job.
    scheduler.tick()
    clock.now = datetime(2026, 1, 15, 10, 1, tzinfo=UTC)
    jobs = scheduler.tick()
    assert len(jobs) == 1
    job = jobs[0].job
    assert job.kind == "schedule"
    assert job.payload["profile"] == "default"

    # The scheduled job's worker creates the run and moves it to running.
    claimed = queue.claim("worker-1", job_id=job.id, now=clock.now)
    assert claimed is not None
    run_repo = RunRepository(engine)
    run = run_repo.create(kind="pipeline", profile="default", state="queued")
    run_machine = RunStateMachine(run_repo, clock=clock)
    run = run_machine.start(run, now=clock.now)
    assert run.state is RunState.RUNNING
    queue.complete(job.id, "worker-1", now=clock.now)

    review_machine = ReviewStateMachine(
        settings={
            "default": ReviewConfig(
                approval_mode="ask",
                timeout_seconds=60,
                on_timeout="reject",
                allow_regenerate=True,
                max_regenerations=3,
            )
        },
        clock=clock,
    )
    review = review_machine.create("review-1", "default")
    assert review.state is ReviewState.PENDING
    return run_machine, run, review_machine, review, clock


def test_scheduled_run_approval_proceeds(
    started_run: tuple[RunStateMachine, Any, ReviewStateMachine, Any, Clock],
) -> None:
    run_machine, run, review_machine, review, clock = started_run

    # A reviewer approves the pending review, so the run may proceed.
    approved = review_machine.approve(review, actor="alice", reason="looks good")
    assert approved.state is ReviewState.APPROVED
    assert approved.answered_by == "alice"
    stored_review = review_machine.get("review-1")
    assert stored_review is not None
    assert stored_review.state is ReviewState.APPROVED

    # The approved run runs to completion.
    finished = run_machine.succeed(run, now=clock.now)
    assert finished.state is RunState.SUCCEEDED
    assert finished.finished_at is not None
    stored_run = run_machine.get(run.id)
    assert stored_run is not None
    assert stored_run.state is RunState.SUCCEEDED


def test_scheduled_run_rejection_stops(
    started_run: tuple[RunStateMachine, Any, ReviewStateMachine, Any, Clock],
) -> None:
    run_machine, run, review_machine, review, clock = started_run

    # A reviewer rejects the pending review, so the run must not proceed.
    rejected = review_machine.reject(review, actor="bob", reason="not suitable")
    assert rejected.state is ReviewState.REJECTED
    assert rejected.answered_by == "bob"
    stored_review = review_machine.get("review-1")
    assert stored_review is not None
    assert stored_review.state is ReviewState.REJECTED

    # The rejected run ends in failure.
    failed = run_machine.fail(
        run,
        error_code="review_rejected",
        error_message="reviewer rejected the run",
        now=clock.now,
    )
    assert failed.state is RunState.FAILED
    assert failed.error_code == "review_rejected"
    stored_run = run_machine.get(run.id)
    assert stored_run is not None
    assert stored_run.state is RunState.FAILED


def test_scheduled_run_budget_pause_and_resume(
    started_run: tuple[RunStateMachine, Any, ReviewStateMachine, Any, Clock],
) -> None:
    run_machine, run, _review_machine, _review, clock = started_run

    # The run hits the budget cap and is paused.
    paused = run_machine.pause(run, reason="budget cap reached", now=clock.now)
    assert paused.state is RunState.PAUSED_BUDGET
    assert paused.pause_reason == "budget cap reached"
    assert paused.paused_at is not None
    stored_run = run_machine.get(run.id)
    assert stored_run is not None
    assert stored_run.state is RunState.PAUSED_BUDGET

    # Once the cap is raised the run resumes, recording who and why.
    resumed = run_machine.resume(
        paused,
        resumed_by="alice",
        reason="budget cap raised",
        now=clock.now,
    )
    assert resumed.state is RunState.RUNNING
    assert resumed.resumed_by == "alice"
    assert resumed.resume_reason == "budget cap raised"
    assert resumed.resumed_at is not None
    stored_run = run_machine.get(run.id)
    assert stored_run is not None
    assert stored_run.state is RunState.RUNNING

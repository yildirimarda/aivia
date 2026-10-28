"""Job queue contracts: attempts, scheduling, heartbeat and dead letter."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from aivia.core.ids import is_ulid
from aivia.core.models import (
    Job,
    JobModel,
    JobRecord,
    JobState,
    JobStatus,
    UnknownFieldWarning,
)


def test_job_defaults_to_a_pending_first_attempt_with_no_schedule() -> None:
    job = Job()

    assert job.id.startswith("job_")
    assert is_ulid(job.id.removeprefix("job_"))
    assert job.kind == "run"
    assert job.state is JobState.PENDING
    assert job.run_id == ""
    assert job.attempts == 0
    assert job.max_attempts == 3
    assert job.not_before is None
    assert job.heartbeat_at is None
    assert job.claimed_by == ""
    assert job.claimed_at is None
    assert job.started_at is None
    assert job.finished_at is None
    assert job.error_code == ""
    assert job.error_message == ""
    assert job.payload == {}
    assert job.metadata == {}
    assert job.attempts_remaining == 3
    assert job.has_attempts_left is True
    assert job.can_retry is True
    assert job.is_active is True
    assert job.is_terminal is False
    assert job.is_running is False
    assert job.is_dead_letter is False
    assert job.is_due() is True
    assert job.is_claimable() is True
    assert job.heartbeat_age_seconds() is None


def test_job_ids_are_unique_and_time_sortable() -> None:
    first = Job()
    second = Job()

    assert first.id != second.id
    assert first.id < second.id


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("pending", JobState.PENDING),
        ("queued", JobState.PENDING),
        ("scheduled", JobState.PENDING),
        ("retry", JobState.PENDING),
        ("backoff", JobState.PENDING),
        ("running", JobState.RUNNING),
        ("in_progress", JobState.RUNNING),
        ("claimed", JobState.RUNNING),
        ("succeeded", JobState.SUCCEEDED),
        ("completed", JobState.SUCCEEDED),
        ("failed", JobState.FAILED),
        ("error", JobState.FAILED),
        ("dead_letter", JobState.DEAD_LETTER),
        ("dead", JobState.DEAD_LETTER),
        ("dlq", JobState.DEAD_LETTER),
        ("exhausted", JobState.DEAD_LETTER),
        ("cancelled", JobState.CANCELLED),
        ("canceled", JobState.CANCELLED),
    ],
)
def test_job_state_accepts_aliases_and_is_normalized(
    raw: str, expected: JobState
) -> None:
    assert Job.model_validate({"state": raw}).state is expected


def test_job_rejects_an_unknown_state() -> None:
    with pytest.raises(ValidationError):
        Job.model_validate({"state": "teleporting"})


def test_job_state_helpers_mark_active_terminal_and_dead_letter_states() -> None:
    pending = Job(state=JobState.PENDING)
    running = Job(state=JobState.RUNNING)
    succeeded = Job(state=JobState.SUCCEEDED)
    failed = Job(state=JobState.FAILED)
    dead = Job(state=JobState.DEAD_LETTER)
    cancelled = Job(state=JobState.CANCELLED)

    assert pending.is_active is True
    assert running.is_active is True
    assert running.is_running is True
    assert succeeded.is_terminal is True
    assert failed.is_terminal is True
    assert dead.is_terminal is True
    assert dead.is_dead_letter is True
    assert dead.dead_lettered is True
    assert cancelled.is_terminal is True
    assert pending.is_terminal is False
    assert running.is_dead_letter is False


def test_job_attempts_must_be_non_negative_and_within_the_limit() -> None:
    with pytest.raises(ValidationError):
        Job(attempts=-1)
    with pytest.raises(ValidationError):
        Job(max_attempts=0)
    with pytest.raises(ValidationError, match="attempts"):
        Job(attempts=4, max_attempts=3)


def test_job_attempt_helpers_report_remaining_and_exhaustion() -> None:
    fresh = Job(attempts=0, max_attempts=3)
    halfway = Job(attempts=2, max_attempts=3)
    exhausted = Job(attempts=3, max_attempts=3, state=JobState.DEAD_LETTER)

    assert fresh.attempts_remaining == 3
    assert halfway.attempts_remaining == 1
    assert halfway.has_attempts_left is True
    assert halfway.can_retry is True
    assert exhausted.attempts_remaining == 0
    assert exhausted.has_attempts_left is False
    assert exhausted.can_retry is False
    assert Job(state=JobState.SUCCEEDED).can_retry is False


def test_job_not_before_controls_when_a_job_is_due_and_claimable() -> None:
    now = datetime(2026, 1, 2, 10, 0, tzinfo=UTC)
    future = Job(not_before=now + timedelta(seconds=30))
    past = Job(not_before=now - timedelta(seconds=30))

    assert future.is_due(now=now) is False
    assert future.is_claimable(now=now) is False
    assert past.is_due(now=now) is True
    assert past.is_claimable(now=now) is True
    assert Job(not_before=now).is_due(now=now) is True
    assert (
        Job(state=JobState.RUNNING, not_before=past.not_before).is_claimable(now=now)
        is False
    )
    assert Job(state=JobState.DEAD_LETTER).is_claimable(now=now) is False


def test_job_schedule_aliases_and_timestamp_normalization() -> None:
    local = timezone(timedelta(hours=2))
    supplied = datetime(2026, 1, 2, 12, 30, tzinfo=local)

    job = Job.model_validate({"available_at": supplied})

    assert job.not_before == datetime(2026, 1, 2, 10, 30, tzinfo=UTC)
    assert job.not_before.tzinfo is UTC
    assert (
        Job.model_validate({"next_attempt_at": supplied}).not_before == job.not_before
    )
    assert Job(not_before=datetime(2026, 1, 2, 10, 30)).not_before == job.not_before


def test_job_heartbeat_is_normalized_and_its_age_is_measurable() -> None:
    now = datetime(2026, 1, 2, 10, 0, tzinfo=UTC)
    local = timezone(timedelta(hours=2))
    supplied = datetime(2026, 1, 2, 12, 0, tzinfo=local)

    job = Job.model_validate({"last_heartbeat": supplied})

    assert job.heartbeat_at == datetime(2026, 1, 2, 10, 0, tzinfo=UTC)
    assert job.last_heartbeat == job.heartbeat_at
    assert job.heartbeat_age_seconds(now=now) == 0.0
    assert job.heartbeat_age_seconds(now=now + timedelta(seconds=45)) == 45.0


def test_job_is_stale_only_when_running_and_the_heartbeat_expired() -> None:
    now = datetime(2026, 1, 2, 10, 0, tzinfo=UTC)
    heartbeat = now - timedelta(seconds=121)
    running = Job(state=JobState.RUNNING, heartbeat_at=heartbeat)
    fresh = Job(
        state=JobState.RUNNING,
        heartbeat_at=now - timedelta(seconds=30),
    )
    pending = Job(state=JobState.PENDING, heartbeat_at=heartbeat)

    assert running.is_stale(now=now, stale_after_seconds=120) is True
    assert fresh.is_stale(now=now, stale_after_seconds=120) is False
    assert pending.is_stale(now=now, stale_after_seconds=120) is False
    assert (
        Job(
            state=JobState.RUNNING,
            claimed_at=now - timedelta(seconds=121),
        ).is_stale(now=now, stale_after_seconds=120)
        is True
    )
    assert (
        Job(state=JobState.RUNNING, heartbeat_at=now - timedelta(seconds=120)).is_stale(
            now=now, stale_after_seconds=120
        )
        is False
    )


def test_job_carries_claim_execution_and_error_metadata_with_aliases() -> None:
    stamp = datetime(2026, 1, 2, 10, 0, tzinfo=UTC)
    finished = stamp + timedelta(seconds=12)
    job = Job.model_validate(
        {
            "job_id": "job_explicit",
            "job_type": "  stage  ",
            "status": "completed",
            "run": "run_1",
            "attempt": 2,
            "max_tries": 5,
            "worker_id": "  worker-7  ",
            "locked_at": stamp,
            "started_at": stamp,
            "finished_at": finished,
            "failure_code": "provider_exhausted",
            "failure_message": "  no provider left  ",
            "arguments": {"stage": "write"},
            "details": {"queue": "default"},
        }
    )

    assert job.job_id == job.id == "job_explicit"
    assert job.kind == job.job_kind == "stage"
    assert job.state is JobState.SUCCEEDED
    assert job.status is JobState.SUCCEEDED
    assert job.job_state is JobState.SUCCEEDED
    assert job.run_id == "run_1"
    assert job.attempts == 2
    assert job.max_attempts == 5
    assert job.claimed_by == job.worker_id == "worker-7"
    assert job.claimed_at == stamp
    assert job.started_at == stamp
    assert job.finished_at == finished
    assert job.error_code == "provider_exhausted"
    assert job.error_message == "no provider left"
    assert job.payload == {"stage": "write"}
    assert job.metadata == {"queue": "default"}


def test_job_rejects_an_empty_kind() -> None:
    with pytest.raises(ValidationError, match="kind"):
        Job(kind="   ")


def test_job_rejects_a_finish_before_the_start() -> None:
    start = datetime(2026, 1, 2, 10, 0, tzinfo=UTC)

    with pytest.raises(ValidationError, match="finished_at"):
        Job(started_at=start, finished_at=start - timedelta(seconds=1))


def test_job_treats_missing_payload_and_metadata_as_empty() -> None:
    job = Job.model_validate({"payload": None, "metadata": None})

    assert job.payload == {}
    assert job.metadata == {}


def test_job_keeps_unknown_fields_with_a_warning() -> None:
    with pytest.warns(UnknownFieldWarning, match="future_field"):
        job = Job.model_validate({"future_field": "kept"})

    assert job.model_extra is not None
    assert job.model_extra["future_field"] == "kept"


def test_job_round_trips_through_canonical_json() -> None:
    stamp = datetime(2026, 1, 2, 10, 0, tzinfo=UTC)
    job = Job(
        kind="publish",
        run_id="run_1",
        attempts=1,
        max_attempts=3,
        not_before=stamp,
        heartbeat_at=stamp,
        claimed_by="worker-1",
        payload={"post": "post_1"},
    )

    restored = Job.model_validate_json(job.to_json())

    assert restored.to_json() == job.to_json()
    assert json.loads(job.to_json())["state"] == "pending"


def test_job_model_aliases_point_at_the_same_classes() -> None:
    assert JobModel is Job
    assert JobRecord is Job
    assert JobStatus is JobState

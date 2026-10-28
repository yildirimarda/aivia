"""Job records for the persistent worker queue.

A :class:`Job` is one unit of work waiting for a worker. It carries everything
the queue needs to be durable and safe: the attempt counter and attempt limit,
the earliest time it may be claimed, the heartbeat that proves a worker is
still alive and the dead-letter state reached when attempts are exhausted.

The model only describes data. It never claims a job, touches the database or
runs a worker; those operations belong to the job queue and worker loop.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import AliasChoices, Field, field_validator, model_validator

from aivia.core.ids import new_job_id
from aivia.core.models.base import ModelBase, utcnow

__all__ = [
    "Job",
    "JobModel",
    "JobRecord",
    "JobState",
    "JobStatus",
]


class JobState(StrEnum):
    """The lifecycle state of a queued job."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"
    CANCELLED = "cancelled"


_STATE_ALIASES = {
    "queued": JobState.PENDING,
    "waiting": JobState.PENDING,
    "scheduled": JobState.PENDING,
    "retry": JobState.PENDING,
    "retrying": JobState.PENDING,
    "backoff": JobState.PENDING,
    "deferred": JobState.PENDING,
    "active": JobState.RUNNING,
    "in_progress": JobState.RUNNING,
    "claimed": JobState.RUNNING,
    "started": JobState.RUNNING,
    "processing": JobState.RUNNING,
    "complete": JobState.SUCCEEDED,
    "completed": JobState.SUCCEEDED,
    "success": JobState.SUCCEEDED,
    "done": JobState.SUCCEEDED,
    "finished": JobState.SUCCEEDED,
    "error": JobState.FAILED,
    "failure": JobState.FAILED,
    "dead": JobState.DEAD_LETTER,
    "deadletter": JobState.DEAD_LETTER,
    "dead_lettered": JobState.DEAD_LETTER,
    "dlq": JobState.DEAD_LETTER,
    "exhausted": JobState.DEAD_LETTER,
    "canceled": JobState.CANCELLED,
    "aborted": JobState.CANCELLED,
}

_TERMINAL_JOB_STATES = frozenset(
    {
        JobState.SUCCEEDED,
        JobState.FAILED,
        JobState.DEAD_LETTER,
        JobState.CANCELLED,
    }
)

_ACTIVE_JOB_STATES = frozenset({JobState.PENDING, JobState.RUNNING})


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _normalize_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _coerce_mapping(value: Any) -> Any:
    if value is None:
        return {}
    return value


class Job(ModelBase):
    """One queued unit of work with retry, scheduling and liveness metadata.

    ``attempts`` counts the attempts already started and never exceeds
    ``max_attempts``. ``not_before`` is the earliest UTC time at which a worker
    may claim the job; ``None`` means immediately. ``heartbeat_at`` is refreshed
    by the running worker so stale claims can be recovered, and ``claimed_by``
    names the worker holding the job. ``DEAD_LETTER`` is the terminal state for
    a job whose attempts are exhausted.
    """

    id: str = Field(
        default_factory=new_job_id,
        min_length=1,
        validation_alias=AliasChoices("id", "job_id", "job"),
    )
    kind: str = Field(
        default="run",
        validation_alias=AliasChoices(
            "kind",
            "job_kind",
            "type",
            "job_type",
        ),
    )
    state: JobState = Field(
        default=JobState.PENDING,
        validation_alias=AliasChoices("state", "status", "job_state"),
    )
    run_id: str = Field(
        default="",
        validation_alias=AliasChoices("run_id", "run"),
    )
    attempts: int = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices(
            "attempts",
            "attempt",
            "attempt_count",
            "tries",
        ),
    )
    max_attempts: int = Field(
        default=3,
        ge=1,
        validation_alias=AliasChoices(
            "max_attempts",
            "max_attempt",
            "max_tries",
            "attempt_limit",
        ),
    )
    not_before: datetime | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "not_before",
            "available_at",
            "eligible_at",
            "run_after",
            "scheduled_for",
            "next_attempt_at",
            "backoff_until",
        ),
    )
    heartbeat_at: datetime | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "heartbeat_at",
            "heartbeat",
            "last_heartbeat",
            "last_heartbeat_at",
            "heartbeat_time",
        ),
    )
    claimed_by: str = Field(
        default="",
        validation_alias=AliasChoices(
            "claimed_by",
            "worker_id",
            "worker",
            "locked_by",
            "owner",
        ),
    )
    claimed_at: datetime | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "claimed_at",
            "locked_at",
            "claim_time",
        ),
    )
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_code: str = Field(
        default="",
        validation_alias=AliasChoices("error_code", "failure_code"),
    )
    error_message: str = Field(
        default="",
        validation_alias=AliasChoices("error_message", "error", "failure_message"),
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices(
            "payload",
            "data",
            "params",
            "arguments",
            "args",
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("metadata", "meta", "details"),
    )

    @field_validator(
        "id", "run_id", "claimed_by", "error_code", "error_message", mode="before"
    )
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("kind", mode="before")
    @classmethod
    def _validate_kind(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise ValueError("kind must be a non-empty name")
            return text
        return value

    @field_validator("state", mode="before")
    @classmethod
    def _validate_state(cls, value: Any) -> Any:
        if isinstance(value, JobState):
            return value
        if isinstance(value, str):
            text = value.strip().lower()
            return _STATE_ALIASES.get(text, text)
        return value

    @field_validator("payload", "metadata", mode="before")
    @classmethod
    def _validate_mapping(cls, value: Any) -> Any:
        return _coerce_mapping(value)

    @field_validator(
        "not_before",
        "heartbeat_at",
        "claimed_at",
        "started_at",
        "finished_at",
    )
    @classmethod
    def _normalize_optional_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _normalize_timestamp(value)

    @model_validator(mode="after")
    def _check_attempts(self) -> Self:
        if self.attempts > self.max_attempts:
            raise ValueError("attempts must not exceed max_attempts")
        return self

    @model_validator(mode="after")
    def _check_times(self) -> Self:
        if (
            self.started_at is not None
            and self.finished_at is not None
            and self.finished_at < self.started_at
        ):
            raise ValueError("finished_at must not be before started_at")
        return self

    @property
    def job_id(self) -> str:
        """Alias for :attr:`id`."""
        return self.id

    @property
    def job_kind(self) -> str:
        """Alias for :attr:`kind`."""
        return self.kind

    @property
    def status(self) -> JobState:
        """Alias for :attr:`state`."""
        return self.state

    @property
    def job_state(self) -> JobState:
        """Alias for :attr:`state`."""
        return self.state

    @property
    def worker_id(self) -> str:
        """Alias for :attr:`claimed_by`."""
        return self.claimed_by

    @property
    def last_heartbeat(self) -> datetime | None:
        """Alias for :attr:`heartbeat_at`."""
        return self.heartbeat_at

    @property
    def attempts_remaining(self) -> int:
        """How many attempts are still allowed, never negative."""
        return max(0, self.max_attempts - self.attempts)

    @property
    def has_attempts_left(self) -> bool:
        """Whether another attempt may be started."""
        return self.attempts < self.max_attempts

    @property
    def can_retry(self) -> bool:
        """Whether the job is not terminal and has attempts left."""
        return self.has_attempts_left and not self.is_terminal

    @property
    def is_running(self) -> bool:
        """Whether a worker currently holds the job."""
        return self.state is JobState.RUNNING

    @property
    def is_dead_letter(self) -> bool:
        """Whether the job exhausted its attempts and was dead-lettered."""
        return self.state is JobState.DEAD_LETTER

    @property
    def dead_lettered(self) -> bool:
        """Alias for :attr:`is_dead_letter`."""
        return self.is_dead_letter

    @property
    def is_terminal(self) -> bool:
        """Whether the job reached a final state."""
        return self.state in _TERMINAL_JOB_STATES

    @property
    def is_active(self) -> bool:
        """Whether the job is waiting or running."""
        return self.state in _ACTIVE_JOB_STATES

    def is_due(self, *, now: datetime | None = None) -> bool:
        """Whether :attr:`not_before` has passed, ignoring the job state."""
        if self.not_before is None:
            return True
        current = _normalize_timestamp(now) if now is not None else utcnow()
        return current >= self.not_before

    def is_claimable(self, *, now: datetime | None = None) -> bool:
        """Whether a worker may claim the job now."""
        return self.state is JobState.PENDING and self.is_due(now=now)

    def heartbeat_age_seconds(self, *, now: datetime | None = None) -> float | None:
        """Seconds since the last heartbeat, or ``None`` before the first one."""
        if self.heartbeat_at is None:
            return None
        current = _normalize_timestamp(now) if now is not None else utcnow()
        return (current - self.heartbeat_at).total_seconds()

    def is_stale(
        self,
        *,
        stale_after_seconds: float,
        now: datetime | None = None,
    ) -> bool:
        """Whether a running claim stopped heartbeating for too long.

        A job that never sent a heartbeat falls back to its claim or start time.
        Jobs that are not running are never stale.
        """
        if self.state is not JobState.RUNNING:
            return False
        current = _normalize_timestamp(now) if now is not None else utcnow()
        reference = (
            self.heartbeat_at or self.claimed_at or self.started_at or self.updated_at
        )
        return (current - reference).total_seconds() > stale_after_seconds


JobModel = Job
JobRecord = Job
JobStatus = JobState

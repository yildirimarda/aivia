"""Atomic, durable job queue on top of the SQLite ``jobs`` table.

A worker claims at most one job at a time. Claiming is a single write
transaction that takes SQLite's write lock before selecting the next due
pending job, so two workers can never claim the same row even when they poll
at the same instant. The claim records the worker, the attempt and the first
heartbeat; the worker refreshes the heartbeat while it runs and then either
completes the job or fails it.

Failing a retryable job returns it to ``pending`` with a backoff time and
keeps its attempt count, so a later claim starts the next attempt. A permanent
failure becomes ``failed``, and a job that exhausted its attempts becomes
``dead_letter``. The queue owns these transitions; a worker loop only decides
whether an error is retryable and how long to wait.

The queue never runs the work itself and never resolves configuration. Callers
pass an engine and the worker identity, which keeps the queue usable from the
CLI, the API and tests without importing any worker module.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import Connection, Engine, or_, select, update

from aivia.core.db.repos import JobRepository
from aivia.core.db.tables import jobs
from aivia.core.errors import AiviaError
from aivia.core.models.base import utcnow
from aivia.core.models.job import Job, JobState

__all__ = [
    "JobNotClaimedError",
    "JobNotFoundError",
    "JobQueue",
    "JobQueueError",
]


class JobQueueError(AiviaError):
    """Base class for job queue failures."""

    code = "job_queue"


class JobNotFoundError(JobQueueError):
    """The job id does not exist."""

    code = "job_not_found"


class JobNotClaimedError(JobQueueError):
    """The job is not running under the calling worker."""

    code = "job_not_claimed"


@contextmanager
def _write_transaction(engine: Engine) -> Iterator[Connection]:
    """Open a transaction that takes SQLite's write lock before reading.

    SQLite starts a deferred transaction by default, so two claims could both
    read the same pending row and only then try to update it. ``BEGIN
    IMMEDIATE`` serialises claims before the select, which is what makes the
    claim atomic even without relying on the update's row count.
    """
    with engine.begin() as connection:
        if engine.dialect.name == "sqlite":
            connection.exec_driver_sql("BEGIN IMMEDIATE")
        yield connection


def _as_utc(value: datetime | None) -> datetime:
    """Return *value* as an aware UTC datetime, defaulting to now."""
    if value is None:
        return utcnow()
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _coerce_worker(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("worker_id must be a string")
    text = value.strip()
    if not text:
        raise ValueError("worker_id must be a non-empty name")
    return text


def _coerce_job_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("job_id must be a string")
    text = value.strip()
    if not text:
        raise ValueError("job_id must be a non-empty name")
    return text


def _coerce_kind(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("kind must be a string")
    text = value.strip()
    if not text:
        raise ValueError("kind must be a non-empty name")
    return text


def _coerce_delay(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("retry_after must be a non-negative number of seconds")
    delay = float(value)
    if not math.isfinite(delay) or delay < 0.0:
        raise ValueError("retry_after must be a non-negative number of seconds")
    return delay


def _coerce_text(value: object, *, label: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value.strip()


class JobQueue:
    """Claim, heartbeat, complete and fail jobs stored in the local database.

    A claim selects the oldest due pending job, or the job named by ``job_id``,
    and moves it to ``running`` while incrementing its attempt count. Every
    later operation is scoped to the claiming worker, so a worker that lost its
    claim cannot overwrite another worker's result.
    """

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.repository = JobRepository(engine)

    def enqueue(
        self,
        job: Job | Mapping[str, Any] | None = None,
        **values: Any,
    ) -> Job:
        """Insert a new job and return it.

        Pass a :class:`~aivia.core.models.Job`, a mapping or keyword values. A
        job without a state starts as ``pending`` and a job without
        ``max_attempts`` uses the model default.
        """
        return self.repository.create(job, **values)

    def get(self, job_id: str) -> Job | None:
        """Return the job with *job_id*, or ``None`` when it does not exist."""
        return self.repository.get(_coerce_job_id(job_id))

    def claim(
        self,
        worker_id: str,
        *,
        job_id: str | None = None,
        kind: str | None = None,
        now: datetime | None = None,
    ) -> Job | None:
        """Claim the next due job, or the named job, for *worker_id*.

        Returns ``None`` when no job is due or the named job is not claimable.
        The update runs under the write lock, so concurrent callers see
        different pending rows and never claim the same job twice.
        """
        worker = _coerce_worker(worker_id)
        current = _as_utc(now)
        wanted = None if job_id is None else _coerce_job_id(job_id)
        wanted_kind = _coerce_kind(kind)

        conditions = [
            jobs.c.state == JobState.PENDING.value,
            or_(jobs.c.not_before.is_(None), jobs.c.not_before <= current),
            jobs.c.attempts < jobs.c.max_attempts,
        ]
        if wanted is not None:
            conditions.append(jobs.c.id == wanted)
        if wanted_kind is not None:
            conditions.append(jobs.c.kind == wanted_kind)

        candidate = (
            select(jobs.c.id)
            .where(*conditions)
            .order_by(jobs.c.created_at, jobs.c.id)
            .limit(1)
            .scalar_subquery()
        )
        statement = (
            update(jobs)
            .where(
                jobs.c.id == candidate,
                jobs.c.state == JobState.PENDING.value,
                jobs.c.attempts < jobs.c.max_attempts,
            )
            .values(
                state=JobState.RUNNING.value,
                attempts=jobs.c.attempts + 1,
                claimed_by=worker,
                claimed_at=current,
                started_at=current,
                heartbeat_at=current,
                finished_at=None,
                error_code="",
                error_message="",
                updated_at=current,
            )
            .returning(jobs.c.id)
        )

        claimed_id: str | None = None
        with _write_transaction(self.engine) as connection:
            row = connection.execute(statement).first()
            if row is not None:
                claimed_id = str(row[0])
        if claimed_id is None:
            return None
        return self.repository.get(claimed_id)

    def heartbeat(
        self,
        job_id: str,
        worker_id: str,
        *,
        now: datetime | None = None,
    ) -> Job:
        """Refresh the heartbeat of a job held by *worker_id*.

        Raises :class:`JobNotFoundError` for an unknown id and
        :class:`JobNotClaimedError` when another worker owns the job or it is
        no longer running.
        """
        wanted = _coerce_job_id(job_id)
        worker = _coerce_worker(worker_id)
        current = _as_utc(now)
        statement = (
            update(jobs)
            .where(
                jobs.c.id == wanted,
                jobs.c.state == JobState.RUNNING.value,
                jobs.c.claimed_by == worker,
            )
            .values(heartbeat_at=current, updated_at=current)
            .returning(jobs.c.id)
        )

        with _write_transaction(self.engine) as connection:
            row = connection.execute(statement).first()
            if row is None:
                self._raise_not_owned(connection, wanted, worker)
        return self._require(wanted)

    def complete(
        self,
        job_id: str,
        worker_id: str,
        *,
        now: datetime | None = None,
    ) -> Job:
        """Mark a job held by *worker_id* as succeeded.

        Raises :class:`JobNotFoundError` for an unknown id and
        :class:`JobNotClaimedError` when the job is not running under the
        calling worker.
        """
        wanted = _coerce_job_id(job_id)
        worker = _coerce_worker(worker_id)
        current = _as_utc(now)
        statement = (
            update(jobs)
            .where(
                jobs.c.id == wanted,
                jobs.c.state == JobState.RUNNING.value,
                jobs.c.claimed_by == worker,
            )
            .values(
                state=JobState.SUCCEEDED.value,
                finished_at=current,
                heartbeat_at=current,
                error_code="",
                error_message="",
                updated_at=current,
            )
            .returning(jobs.c.id)
        )

        with _write_transaction(self.engine) as connection:
            row = connection.execute(statement).first()
            if row is None:
                self._raise_not_owned(connection, wanted, worker)
        return self._require(wanted)

    def fail(
        self,
        job_id: str,
        worker_id: str,
        *,
        error_code: str = "",
        error_message: str = "",
        retryable: bool = False,
        retry_after: float = 0.0,
        now: datetime | None = None,
    ) -> Job:
        """Record a failure for a job held by *worker_id*.

        A retryable failure with attempts left returns the job to ``pending``
        and delays the next claim by *retry_after* seconds. A permanent failure
        becomes ``failed``. Either way, a job that already used its last
        attempt becomes ``dead_letter``.
        """
        wanted = _coerce_job_id(job_id)
        worker = _coerce_worker(worker_id)
        current = _as_utc(now)
        delay = _coerce_delay(retry_after)
        code = _coerce_text(error_code, label="error_code")
        message = _coerce_text(error_message, label="error_message")
        if not isinstance(retryable, bool):
            raise ValueError("retryable must be a boolean")

        with _write_transaction(self.engine) as connection:
            row = (
                connection.execute(
                    select(jobs.c.state, jobs.c.claimed_by, jobs.c.attempts).where(
                        jobs.c.id == wanted
                    )
                )
                .mappings()
                .first()
            )
            if row is None:
                raise JobNotFoundError(
                    f"job {wanted!r} does not exist",
                    details={"job_id": wanted},
                )
            if row["state"] != JobState.RUNNING.value or row["claimed_by"] != worker:
                self._raise_not_owned(connection, wanted, worker)

            attempts = int(row["attempts"])
            exhausted = attempts >= self._max_attempts(connection, wanted)
            values: dict[str, Any] = {
                "error_code": code,
                "error_message": message,
                "updated_at": current,
            }
            if exhausted:
                values.update(
                    state=JobState.DEAD_LETTER.value,
                    finished_at=current,
                )
            elif retryable:
                values.update(
                    state=JobState.PENDING.value,
                    not_before=current + timedelta(seconds=delay),
                    claimed_by="",
                    claimed_at=None,
                    started_at=None,
                    heartbeat_at=None,
                    finished_at=None,
                )
            else:
                values.update(
                    state=JobState.FAILED.value,
                    finished_at=current,
                )
            connection.execute(update(jobs).where(jobs.c.id == wanted).values(**values))
        return self._require(wanted)

    def _max_attempts(self, connection: Connection, job_id: str) -> int:
        value = connection.execute(
            select(jobs.c.max_attempts).where(jobs.c.id == job_id)
        ).scalar_one()
        return int(value)

    def _raise_not_owned(
        self,
        connection: Connection,
        job_id: str,
        worker: str,
    ) -> None:
        row = (
            connection.execute(
                select(jobs.c.state, jobs.c.claimed_by).where(jobs.c.id == job_id)
            )
            .mappings()
            .first()
        )
        if row is None:
            raise JobNotFoundError(
                f"job {job_id!r} does not exist",
                details={"job_id": job_id},
            )
        raise JobNotClaimedError(
            f"job {job_id!r} is not claimed by worker {worker!r}",
            details={
                "job_id": job_id,
                "worker_id": worker,
                "state": row["state"],
                "claimed_by": row["claimed_by"],
            },
        )

    def _require(self, job_id: str) -> Job:
        job = self.repository.get(job_id)
        if job is None:
            raise JobNotFoundError(
                f"job {job_id!r} does not exist",
                details={"job_id": job_id},
            )
        return job

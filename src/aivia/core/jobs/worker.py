"""The worker loop that turns queued jobs into completed work.

A :class:`Worker` claims one job at a time from :class:`~aivia.core.jobs.JobQueue`,
runs a caller-supplied handler and records the outcome. The queue owns every
state transition; the worker only decides whether a failure is retryable, how
long to wait before the next attempt and when a claim is stale enough to
recover.

While a handler runs, a background heartbeat keeps the claim alive. A worker
that dies stops heartbeating, so :meth:`Worker.recover_stale` (also run before
each claim) returns its job to the queue or dead-letters it once attempts are
exhausted. :meth:`Worker.request_stop` asks the loop to finish the current job
and stop claiming new ones, which is what a graceful shutdown needs.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from aivia.core.config.ops import JobsConfig
from aivia.core.jobs.queue import (
    JobNotClaimedError,
    JobNotFoundError,
    JobQueue,
)
from aivia.core.logging import get_logger
from aivia.core.models.base import utcnow
from aivia.core.models.job import Job, JobState
from aivia.core.redact import redact

__all__ = [
    "JobWorker",
    "Worker",
    "WorkerLoop",
    "WorkerOutcome",
]

Handler = Callable[[Job], Any]
Clock = Callable[[], datetime]
Sleep = Callable[[float], Awaitable[None]]


def _snake_case(name: str) -> str:
    result: list[str] = []
    for index, character in enumerate(name):
        if character.isupper() and index and not name[index - 1].isupper():
            result.append("_")
        result.append(character.lower())
    return "".join(result)


def _positive(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive number of seconds")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive number of seconds")
    return number


def _non_negative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a non-negative number")
    return number


def _multiplier(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("backoff_multiplier must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 1.0:
        raise ValueError("backoff_multiplier must be at least one")
    return number


def _error_code(error: BaseException) -> str:
    code = getattr(error, "code", "")
    if isinstance(code, str) and code.strip():
        return code.strip()
    return _snake_case(type(error).__name__)


def _error_message(error: BaseException) -> str:
    message = str(error).strip() or type(error).__name__
    return redact(message)


def _is_retryable(error: BaseException) -> bool:
    value = getattr(error, "retryable", None)
    if isinstance(value, bool):
        return value
    return isinstance(error, TimeoutError)


@dataclass(frozen=True, slots=True)
class WorkerOutcome:
    """The result of running one job.

    ``job`` is the state after the queue recorded success or failure.
    ``retryable`` says whether the handler failure was worth another attempt;
    ``dead_letter`` is true when that attempt was the last one allowed.
    """

    job: Job
    succeeded: bool
    retryable: bool = False
    dead_letter: bool = False
    error_code: str = ""
    error_message: str = ""
    claimed: bool = True

    @property
    def ok(self) -> bool:
        """Whether the handler completed successfully."""
        return self.succeeded

    @property
    def retried(self) -> bool:
        """Whether the failed job went back to ``pending``."""
        return not self.succeeded and self.job.state is JobState.PENDING

    @property
    def failed(self) -> bool:
        """Whether the failed job stopped permanently."""
        return self.job.state is JobState.FAILED

    @property
    def lost_claim(self) -> bool:
        """Whether another worker recovered the claim before it finished."""
        return not self.claimed


class Worker:
    """Run jobs from a :class:`JobQueue` until asked to stop.

    ``handler`` receives the claimed :class:`~aivia.core.models.Job` and may be
    synchronous or asynchronous. Its return value is ignored; raising an
    exception fails the job. A retryable error returns the job to ``pending``
    with exponential backoff, a permanent error marks it ``failed`` and the
    queue marks ``dead_letter`` when no attempt remains.

    ``config`` is the jobs configuration section. Explicit keyword arguments
    override it, and ``clock`` and ``sleep`` are injectable so tests never wait
    on wall-clock time.
    """

    def __init__(
        self,
        queue: JobQueue,
        handler: Handler,
        *,
        worker_id: str | None = None,
        config: JobsConfig | Mapping[str, Any] | None = None,
        kind: str | None = None,
        heartbeat_seconds: float | None = None,
        poll_seconds: float | None = None,
        stale_after_seconds: float | None = None,
        retry_delay_seconds: float | None = None,
        backoff_multiplier: float | None = None,
        max_retry_delay_seconds: float | None = None,
        clock: Clock | None = None,
        sleep: Sleep | None = None,
        logger: Any | None = None,
    ) -> None:
        if not isinstance(queue, JobQueue):
            raise TypeError("queue must be a JobQueue")
        if not callable(handler):
            raise TypeError("handler must be callable")
        if worker_id is not None and (
            not isinstance(worker_id, str) or not worker_id.strip()
        ):
            raise ValueError("worker_id must be a non-empty name")
        if kind is not None and (not isinstance(kind, str) or not kind.strip()):
            raise ValueError("kind must be a non-empty name")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        if sleep is not None and not callable(sleep):
            raise TypeError("sleep must be callable")

        settings = self._settings(config)
        self.queue = queue
        self.handler = handler
        self.worker_id = (worker_id or f"worker-{time.time_ns():x}").strip()
        self.kind = kind.strip() if kind is not None else None
        self.heartbeat_seconds = _positive(
            settings.heartbeat_seconds
            if heartbeat_seconds is None
            else heartbeat_seconds,
            name="heartbeat_seconds",
        )
        self.poll_seconds = _positive(
            settings.poll_seconds if poll_seconds is None else poll_seconds,
            name="poll_seconds",
        )
        self.stale_after_seconds = _positive(
            settings.stale_after_seconds
            if stale_after_seconds is None
            else stale_after_seconds,
            name="stale_after_seconds",
        )
        if self.stale_after_seconds <= self.heartbeat_seconds:
            raise ValueError(
                "stale_after_seconds must be greater than heartbeat_seconds"
            )
        self.retry_delay_seconds = _non_negative(
            settings.retry_delay_seconds
            if retry_delay_seconds is None
            else retry_delay_seconds,
            name="retry_delay_seconds",
        )
        self.backoff_multiplier = _multiplier(
            settings.backoff_multiplier
            if backoff_multiplier is None
            else backoff_multiplier
        )
        self.max_retry_delay_seconds = _non_negative(
            settings.max_retry_delay_seconds
            if max_retry_delay_seconds is None
            else max_retry_delay_seconds,
            name="max_retry_delay_seconds",
        )
        if self.retry_delay_seconds > self.max_retry_delay_seconds:
            raise ValueError(
                "retry_delay_seconds must not exceed max_retry_delay_seconds"
            )
        self._clock = clock or utcnow
        self._sleep = sleep or asyncio.sleep
        self.logger = logger or get_logger("aivia.jobs.worker")
        self._stop = False
        self._processed = 0
        self._succeeded = 0
        self._failed = 0
        self._dead_lettered = 0

    @staticmethod
    def _settings(config: JobsConfig | Mapping[str, Any] | None) -> JobsConfig:
        if config is None:
            return JobsConfig.from_mapping({})
        if isinstance(config, JobsConfig):
            return config
        if isinstance(config, Mapping):
            return JobsConfig.from_mapping(config)
        raise TypeError("config must be a JobsConfig, a mapping or None")

    @property
    def processed(self) -> int:
        """How many jobs this worker has finished."""
        return self._processed

    @property
    def succeeded(self) -> int:
        """How many jobs this worker completed successfully."""
        return self._succeeded

    @property
    def failed(self) -> int:
        """How many jobs this worker failed at least once."""
        return self._failed

    @property
    def dead_lettered(self) -> int:
        """How many jobs this worker moved to ``dead_letter``."""
        return self._dead_lettered

    @property
    def stopping(self) -> bool:
        """Whether a graceful stop has been requested."""
        return self._stop

    def request_stop(self) -> None:
        """Ask the loop to finish the current job and claim no more."""
        self._stop = True

    def clear_stop(self) -> None:
        """Clear a previous stop request so the worker can run again."""
        self._stop = False

    def backoff_delay(self, attempt: int) -> float:
        """Return the wait after *attempt* before the next attempt starts."""
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise ValueError("attempt must be a positive integer")
        delay = self.retry_delay_seconds * (self.backoff_multiplier ** (attempt - 1))
        return min(delay, self.max_retry_delay_seconds)

    def recover_stale(self, *, now: datetime | None = None) -> list[Job]:
        """Recover running jobs whose worker stopped heartbeating."""
        recovered = self.queue.recover_stale(
            stale_after_seconds=self.stale_after_seconds,
            now=now,
        )
        for job in recovered:
            self.logger.warning(
                "recovered stale job %s from worker %r into %s",
                job.id,
                job.claimed_by or "unknown",
                job.state.value,
            )
        return recovered

    async def run_once(
        self,
        *,
        now: datetime | None = None,
    ) -> WorkerOutcome | None:
        """Recover stale claims, claim one job and run it.

        Returns ``None`` when no job is due. The worker does not check its stop
        flag here, so callers can run a single job during shutdown.
        """
        current = now if now is not None else self._clock()
        self.recover_stale(now=current)
        job = self.queue.claim(self.worker_id, kind=self.kind, now=current)
        if job is None:
            return None
        return await self._execute(job)

    async def run(self, *, max_jobs: int | None = None) -> int:
        """Claim and run jobs until stopped or *max_jobs* is reached.

        A graceful stop lets the current handler finish; it only prevents new
        claims. Returns the number of jobs this call processed.
        """
        if max_jobs is not None and (
            isinstance(max_jobs, bool) or not isinstance(max_jobs, int) or max_jobs < 0
        ):
            raise ValueError("max_jobs must be a non-negative integer or None")
        processed = 0
        while not self._stop:
            if max_jobs is not None and processed >= max_jobs:
                break
            outcome = await self.run_once()
            if outcome is None:
                await self._sleep(self.poll_seconds)
                continue
            processed += 1
        return processed

    async def _execute(self, job: Job) -> WorkerOutcome:
        lost_claim = asyncio.Event()
        handler = asyncio.create_task(self._run_handler(job))
        heartbeat = asyncio.create_task(
            self._heartbeat_loop(job.id, handler=handler, lost_claim=lost_claim)
        )
        try:
            await handler
        except asyncio.CancelledError:
            await self._stop_heartbeat(heartbeat)
            if lost_claim.is_set():
                return self._record_lost_claim(job)
            self._requeue_cancelled(job)
            raise
        except Exception as error:
            await self._stop_heartbeat(heartbeat)
            if lost_claim.is_set():
                return self._record_lost_claim(job)
            return self._record_failure(job, error)
        else:
            await self._stop_heartbeat(heartbeat)
            if lost_claim.is_set():
                return self._record_lost_claim(job)
            return self._record_success(job)

    async def _run_handler(self, job: Job) -> None:
        result = self.handler(job)
        if inspect.isawaitable(result):
            await result

    async def _heartbeat_loop(
        self,
        job_id: str,
        *,
        handler: asyncio.Task[None] | None = None,
        lost_claim: asyncio.Event | None = None,
    ) -> None:
        while True:
            await self._sleep(self.heartbeat_seconds)
            try:
                self.queue.heartbeat(job_id, self.worker_id, now=self._clock())
            except (JobNotClaimedError, JobNotFoundError):
                self.logger.warning(
                    "lost claim on job %s; cancelling the in-flight handler", job_id
                )
                if lost_claim is not None:
                    lost_claim.set()
                if handler is not None and not handler.done():
                    handler.cancel()
                return
            except Exception:
                self.logger.exception("heartbeat failed for job %s", job_id)
                return

    @staticmethod
    async def _stop_heartbeat(task: asyncio.Task[None]) -> None:
        if task.done():
            with suppress(asyncio.CancelledError, Exception):
                task.result()
            return
        task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await task

    def _record_lost_claim(self, job: Job) -> WorkerOutcome:
        self.logger.warning(
            "job %s was recovered by another worker; cancelled the handler", job.id
        )
        return WorkerOutcome(
            job=job,
            succeeded=False,
            error_code="lost_claim",
            error_message="another worker recovered the claim",
            claimed=False,
        )

    def _record_success(self, job: Job) -> WorkerOutcome:
        try:
            updated = self.queue.complete(job.id, self.worker_id, now=self._clock())
        except (JobNotClaimedError, JobNotFoundError):
            self.logger.warning("lost claim on job %s before completion", job.id)
            return WorkerOutcome(
                job=job,
                succeeded=False,
                error_code="lost_claim",
                claimed=False,
            )
        self._processed += 1
        self._succeeded += 1
        self.logger.info("completed job %s", updated.id)
        return WorkerOutcome(job=updated, succeeded=True)

    def _record_failure(self, job: Job, error: BaseException) -> WorkerOutcome:
        retryable = _is_retryable(error)
        code = _error_code(error)
        message = _error_message(error)
        delay = self.backoff_delay(job.attempts) if retryable else 0.0
        try:
            updated = self.queue.fail(
                job.id,
                self.worker_id,
                error_code=code,
                error_message=message,
                retryable=retryable,
                retry_after=delay,
                now=self._clock(),
            )
        except (JobNotClaimedError, JobNotFoundError):
            self.logger.warning(
                "lost claim on job %s before failure was recorded", job.id
            )
            return WorkerOutcome(
                job=job,
                succeeded=False,
                retryable=retryable,
                error_code=code,
                error_message=message,
                claimed=False,
            )

        self._processed += 1
        self._failed += 1
        dead_letter = updated.state is JobState.DEAD_LETTER
        if dead_letter:
            self._dead_lettered += 1
        self.logger.warning(
            "job %s failed on attempt %d/%d with %s (%s)",
            updated.id,
            updated.attempts,
            updated.max_attempts,
            code,
            updated.state.value,
        )
        return WorkerOutcome(
            job=updated,
            succeeded=False,
            retryable=retryable,
            dead_letter=dead_letter,
            error_code=code,
            error_message=message,
        )

    def _requeue_cancelled(self, job: Job) -> None:
        try:
            self.queue.fail(
                job.id,
                self.worker_id,
                error_code="cancelled",
                error_message="worker was cancelled while running the job",
                retryable=True,
                now=self._clock(),
            )
        except (JobNotClaimedError, JobNotFoundError):
            self.logger.warning("lost claim on cancelled job %s", job.id)
        except Exception:
            self.logger.exception("could not requeue cancelled job %s", job.id)


JobWorker = Worker
"""Alias for :class:`Worker`."""

WorkerLoop = Worker
"""Alias for :class:`Worker`."""

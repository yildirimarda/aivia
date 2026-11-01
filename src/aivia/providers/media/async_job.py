"""Submit, poll, download and cancel provider-side asynchronous media jobs.

Some media providers do not return a finished image or clip from the generation
request. They accept the request, return a job identifier and expose a status
endpoint until the result is ready. This module owns that whole pattern so every
adapter behaves the same way:

* a submitted job is written to the local ``jobs`` table before the remote call,
  and the provider job identifier is written back immediately afterwards;
* repeating a submission with the same idempotency key returns the stored job
  instead of paying for a second generation, waiting for a racing submission to
  persist its remote id rather than returning an unusable pending record;
* a pending job whose submitter died before storing a remote id is resumed on
  the same local record, so a repeated key recovers it instead of hanging;
* polling is bounded by a deadline, records progress and terminal state, and
  raises :class:`AsyncJobTimeoutError` when the deadline passes;
* cancelling is idempotent and never calls the provider twice;
* downloading a result uses the shared HTTP client and enforces a size cap.

The helper is deliberately transport agnostic. A provider adapter supplies a
small backend with ``submit``, ``poll`` and ``cancel`` coroutines, and the
helper supplies durability, idempotency, timeouts, polling and downloads. The
same backend can be replaced by a fake in tests without touching the network.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import math
import os
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, Self, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import Engine
from sqlalchemy.exc import IntegrityError

from aivia.core.config.http import HttpClientConfig
from aivia.core.db.repos import JobRepository
from aivia.core.errors import (
    InvalidRequestError,
    ValidationError,
)
from aivia.core.errors import TimeoutError as AiviaTimeoutError
from aivia.core.http import HttpClient
from aivia.core.idempotency import IdempotencyConflict
from aivia.core.ids import new_job_id
from aivia.core.jobs import JobNotFoundError
from aivia.core.models import Job, JobState, canonical_json, utcnow
from aivia.providers.protocols.base import validate_http_url

__all__ = [
    "AsyncJob",
    "AsyncJobBackend",
    "AsyncJobHelper",
    "AsyncJobState",
    "AsyncJobStatus",
    "AsyncJobStore",
    "AsyncJobTimeoutError",
    "AsyncMediaJob",
    "AsyncMediaJobHelper",
    "DatabaseAsyncJobStore",
    "InMemoryAsyncJobStore",
    "MediaJob",
    "MediaJobHelper",
]

_META_KEY = "async_job"
"""The key inside a persisted job's metadata holding helper bookkeeping."""

_DEFAULT_TIMEOUT_SECONDS = 600.0
"""Seconds a submission may remain in flight before polling gives up."""

_DEFAULT_POLL_INTERVAL_SECONDS = 1.0
"""Seconds between status checks while a job is still running."""

_STATE_ALIASES: dict[str, str] = {
    "queued": "pending",
    "queue": "pending",
    "waiting": "pending",
    "submitted": "pending",
    "in_queue": "pending",
    "in-queue": "pending",
    "in_progress": "running",
    "in-progress": "running",
    "processing": "running",
    "active": "running",
    "started": "running",
    "complete": "succeeded",
    "completed": "succeeded",
    "success": "succeeded",
    "done": "succeeded",
    "finished": "succeeded",
    "error": "failed",
    "failure": "failed",
    "canceled": "cancelled",
    "aborted": "cancelled",
    "timeout": "timed_out",
    "timed-out": "timed_out",
    "expired": "timed_out",
}

_RESULT_KEYS = ("result_urls", "urls", "outputs", "results", "files", "output")
_RESULT_URL_KEYS = ("url", "uri", "href", "download_url", "result_url")
_REMOTE_ID_KEYS = ("job_id", "remote_id", "request_id", "id", "uuid", "requestId")
_STATE_KEYS = ("state", "status", "job_state", "phase")

_ACTIVE_SUBMISSIONS: set[str] = set()
"""Owner tokens of submissions in flight in this process."""


class AsyncJobState(StrEnum):
    """The lifecycle of a remote media generation job."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"

    @property
    def terminal(self) -> bool:
        """Whether no further polling is needed."""
        return self in {
            AsyncJobState.SUCCEEDED,
            AsyncJobState.FAILED,
            AsyncJobState.CANCELLED,
            AsyncJobState.TIMED_OUT,
        }

    @property
    def succeeded(self) -> bool:
        """Whether the job finished successfully."""
        return self is AsyncJobState.SUCCEEDED


def _coerce_state(value: Any) -> AsyncJobState:
    if isinstance(value, AsyncJobState):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        text = _STATE_ALIASES.get(text, text)
        try:
            return AsyncJobState(text)
        except ValueError as exc:
            raise ValueError(f"unknown async job state: {value!r}") from exc
    raise TypeError("async job state must be a string or AsyncJobState")


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _collect_urls(value: Any) -> tuple[str, ...]:
    """Extract result URLs from a provider status shape.

    Providers commonly return one URL, a list of URLs or a list of objects with
    a ``url`` field. Anything else is ignored rather than guessed at, so a
    malformed response cannot turn into a bogus download request.
    """
    found: list[str] = []

    def visit(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, str):
            text = item.strip()
            if text.startswith(("http://", "https://")):
                found.append(text)
            return
        if isinstance(item, Mapping):
            for key in _RESULT_URL_KEYS:
                if key in item:
                    visit(item[key])
                    return
            for key in _RESULT_KEYS:
                if key in item:
                    visit(item[key])
                    return
            return
        if isinstance(item, Sequence) and not isinstance(item, (bytes, bytearray)):
            for child in item:
                visit(child)

    visit(value)
    return tuple(dict.fromkeys(found))


def _remote_id(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in _REMOTE_ID_KEYS:
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
            if isinstance(candidate, Mapping):
                nested = _remote_id(candidate)
                if nested:
                    return nested
        return ""
    for key in _REMOTE_ID_KEYS:
        candidate = getattr(value, key, None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def _coerce_progress(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    progress = float(value)
    if not math.isfinite(progress):
        return None
    if progress > 1.0 and progress <= 100.0:
        progress /= 100.0
    return min(1.0, max(0.0, progress))


def _jsonable(value: Any) -> Any:
    """Return a canonical-JSON-safe representation of *value*.

    Bytes are replaced by their SHA-256 digest so a request carrying inline
    media can still be hashed and stored without keeping a second copy of the
    payload in the job table.
    """
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="python"))
    if isinstance(value, bytes | bytearray):
        return {"sha256": hashlib.sha256(bytes(value)).hexdigest(), "size": len(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value") and isinstance(value.value, str | int):
        return value.value
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _request_mapping(request: Any) -> dict[str, Any]:
    if isinstance(request, BaseModel):
        request = request.model_dump(mode="python")
    if not isinstance(request, Mapping):
        raise TypeError("request must be a mapping or a pydantic model")
    data = _jsonable(dict(request))
    if not isinstance(data, dict):
        raise TypeError("request must serialize to a JSON object")
    return data


def _request_digest(provider: str, request: Mapping[str, Any]) -> str:
    payload = {"provider": provider, "request": request}
    encoded = canonical_json(_jsonable(payload)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _error_code(error: BaseException) -> str:
    code = getattr(error, "code", None)
    if isinstance(code, str) and code.strip():
        return code.strip()
    return type(error).__name__.strip().lower() or "error"


def _error_message(error: BaseException) -> str:
    text = str(error).strip()
    return text or type(error).__name__


def _new_submission_owner() -> str:
    """Return a token identifying one in-flight submission in this process."""
    return f"{os.getpid()}:{new_job_id()}"


def _owner_pid(owner: str) -> int | None:
    head, _, _ = owner.partition(":")
    try:
        return int(head)
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class AsyncJobStatus:
    """One poll result from a provider backend.

    ``progress`` is normalized to ``0.0..1.0``; a provider that reports a
    percentage is converted. ``result_urls`` may be empty while a job runs and
    must contain at least one URL once the state is ``succeeded``.
    """

    state: AsyncJobState = AsyncJobState.RUNNING
    progress: float | None = None
    result_urls: tuple[str, ...] = ()
    error_code: str = ""
    error_message: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", _coerce_state(self.state))
        object.__setattr__(self, "progress", _coerce_progress(self.progress))
        object.__setattr__(self, "result_urls", _collect_urls(self.result_urls))
        object.__setattr__(self, "error_code", _text(self.error_code))
        object.__setattr__(self, "error_message", _text(self.error_message))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def result_url(self) -> str | None:
        """The first result URL, or ``None`` while the job is running."""
        return self.result_urls[0] if self.result_urls else None


class AsyncJob(BaseModel):
    """A durable record of one remote media job.

    The local :attr:`id` is a prefixed ULID and never changes. :attr:`remote_id`
    is the provider's identifier and is persisted as soon as the provider
    returns it, so a restarted helper can resume polling without submitting the
    request again.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(default_factory=new_job_id)
    provider: str = ""
    remote_id: str = ""
    state: AsyncJobState = AsyncJobState.PENDING
    idempotency_key: str = ""
    request_digest: str = ""
    request: dict[str, Any] = Field(default_factory=dict)
    result_urls: list[str] = Field(default_factory=list)
    progress: float | None = None
    error_code: str = ""
    error_message: str = ""
    submission_owner: str = ""
    submission_started_at: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "id",
        "provider",
        "remote_id",
        "idempotency_key",
        "submission_owner",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _text(value)

    @field_validator("state", mode="before")
    @classmethod
    def _validate_state(cls, value: Any) -> AsyncJobState:
        return _coerce_state(value)

    @field_validator("progress", mode="before")
    @classmethod
    def _validate_progress(cls, value: Any) -> float | None:
        return _coerce_progress(value)

    @field_validator("result_urls", mode="before")
    @classmethod
    def _validate_result_urls(cls, value: Any) -> list[str]:
        urls = _collect_urls(value)
        return [validate_http_url(url, field="result url") for url in urls]

    @field_validator("error_code", "error_message", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: Any) -> str:
        return _text(value)

    @field_validator("request", "metadata", mode="before")
    @classmethod
    def _validate_mapping(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="python")
        if not isinstance(value, Mapping):
            raise TypeError("must be a mapping")
        return dict(value)

    @property
    def job_id(self) -> str:
        """Alias for :attr:`id`."""
        return self.id

    @property
    def status(self) -> AsyncJobState:
        """Alias for :attr:`state`."""
        return self.state

    @property
    def terminal(self) -> bool:
        """Whether the job reached a state that needs no more polling."""
        return self.state.terminal

    @property
    def succeeded(self) -> bool:
        """Whether the job finished successfully."""
        return self.state.succeeded

    @property
    def result_url(self) -> str | None:
        """The first result URL, if any."""
        return self.result_urls[0] if self.result_urls else None


MediaJob = AsyncJob
"""Alias for callers that describe the record as a media job."""

AsyncMediaJob = AsyncJob
"""Alias matching the helper's name."""


class AsyncJobTimeoutError(AiviaTimeoutError):
    """A remote media job did not finish before its polling deadline."""

    code = "timeout"

    def __init__(
        self,
        message: str,
        *,
        job: AsyncJob | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        payload = dict(details or {})
        if job is not None:
            payload.update(
                {
                    "job_id": job.id,
                    "remote_id": job.remote_id,
                    "provider": job.provider,
                    "state": job.state.value,
                }
            )
        super().__init__(message, details=payload)
        self.job = job


@runtime_checkable
class AsyncJobBackend(Protocol):
    """The provider-side operations an asynchronous media adapter exposes."""

    async def submit(self, request: Any) -> Any:
        """Submit *request* and return a remote job id or a response mapping."""

    async def poll(self, remote_id: str) -> Any:
        """Return the current status of *remote_id*."""

    async def cancel(self, remote_id: str) -> Any:
        """Ask the provider to cancel *remote_id*."""


@runtime_checkable
class AsyncJobStore(Protocol):
    """Persistence used by the helper to survive a process restart."""

    def get(self, job_id: str) -> AsyncJob | None:
        """Return the job with *job_id*, or ``None``."""

    def find_by_idempotency_key(self, key: str) -> AsyncJob | None:
        """Return the job already stored for *key*, or ``None``."""

    def save(self, job: AsyncJob) -> AsyncJob:
        """Insert or update *job* and return the stored record."""


class InMemoryAsyncJobStore:
    """A process-local store for tests and short-lived helpers.

    It implements the same contract as the database store but loses its records
    when the process exits. Production callers should pass an engine.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, AsyncJob] = {}
        self._keys: dict[str, str] = {}

    def get(self, job_id: str) -> AsyncJob | None:
        return self._jobs.get(_text(job_id))

    def find_by_idempotency_key(self, key: str) -> AsyncJob | None:
        text = _text(key)
        if not text:
            return None
        job_id = self._keys.get(text)
        return None if job_id is None else self._jobs.get(job_id)

    def save(self, job: AsyncJob) -> AsyncJob:
        self._jobs[job.id] = job
        if job.idempotency_key:
            self._keys[job.idempotency_key] = job.id
        return job


class DatabaseAsyncJobStore:
    """Persist async media jobs in the shared SQLite ``jobs`` table.

    A job kind distinguishes helper records from generic queue work, so both
    can share the durable queue table without claiming each other. The remote
    identifier, idempotency key, request digest, progress and result URLs live
    in the job's metadata.
    """

    def __init__(self, engine: Engine, *, kind: str = "media_async") -> None:
        self.engine = engine
        self.kind = _text(kind) or "media_async"
        self.repository = JobRepository(engine)

    def get(self, job_id: str) -> AsyncJob | None:
        job = self.repository.get(_text(job_id))
        if job is None or job.kind != self.kind:
            return None
        return _from_job(job)

    def find_by_idempotency_key(self, key: str) -> AsyncJob | None:
        text = _text(key)
        if not text:
            return None
        matches = self.repository.list(kind=self.kind, idempotency_key=text)
        if matches:
            return _from_job(matches[0])
        for job in self.repository.list(kind=self.kind):
            record = _from_job(job)
            if record.idempotency_key == text:
                return record
        return None

    def save(self, job: AsyncJob) -> AsyncJob:
        values = _job_values(job, kind=self.kind)
        existing = self.repository.get(job.id)
        if existing is None or existing.kind != self.kind:
            stored = self.repository.create(
                Job(
                    id=job.id,
                    created_at=job.created_at,
                    updated_at=job.updated_at,
                    **values,
                )
            )
        else:
            updated = self.repository.update(job.id, **values)
            if updated is None:
                raise RuntimeError(f"async job {job.id!r} disappeared while saving")
            stored = updated
        return _from_job(stored)


_JOB_STATES: dict[AsyncJobState, JobState] = {
    AsyncJobState.PENDING: JobState.PENDING,
    AsyncJobState.RUNNING: JobState.RUNNING,
    AsyncJobState.SUCCEEDED: JobState.SUCCEEDED,
    AsyncJobState.FAILED: JobState.FAILED,
    AsyncJobState.CANCELLED: JobState.CANCELLED,
    AsyncJobState.TIMED_OUT: JobState.FAILED,
}

_ASYNC_STATES: dict[JobState, AsyncJobState] = {
    JobState.PENDING: AsyncJobState.PENDING,
    JobState.RUNNING: AsyncJobState.RUNNING,
    JobState.SUCCEEDED: AsyncJobState.SUCCEEDED,
    JobState.FAILED: AsyncJobState.FAILED,
    JobState.DEAD_LETTER: AsyncJobState.FAILED,
    JobState.CANCELLED: AsyncJobState.CANCELLED,
}


def _job_values(job: AsyncJob, *, kind: str) -> dict[str, Any]:
    metadata = dict(job.metadata)
    metadata[_META_KEY] = {
        "provider": job.provider,
        "remote_id": job.remote_id,
        "idempotency_key": job.idempotency_key,
        "request_digest": job.request_digest,
        "result_urls": list(job.result_urls),
        "progress": job.progress,
        "state": job.state.value,
        "submission_owner": job.submission_owner,
        "submission_started_at": (
            job.submission_started_at.isoformat()
            if job.submission_started_at is not None
            else None
        ),
    }
    return {
        "kind": kind,
        "state": _JOB_STATES[job.state],
        "run_id": "",
        "idempotency_key": job.idempotency_key,
        "payload": {"request": dict(job.request), "provider": job.provider},
        "metadata": metadata,
        "error_code": job.error_code,
        "error_message": job.error_message,
    }


def _from_job(job: Job) -> AsyncJob:
    raw_meta = job.metadata.get(_META_KEY)
    internal = dict(raw_meta) if isinstance(raw_meta, Mapping) else {}
    state = _ASYNC_STATES[job.state]
    if state is AsyncJobState.FAILED and internal.get("state") == (
        AsyncJobState.TIMED_OUT.value
    ):
        state = AsyncJobState.TIMED_OUT

    raw_payload = job.payload if isinstance(job.payload, Mapping) else {}
    raw_request = raw_payload.get("request")
    request = dict(raw_request) if isinstance(raw_request, Mapping) else {}
    user_metadata = {
        str(key): value for key, value in job.metadata.items() if key != _META_KEY
    }
    raw_urls = internal.get("result_urls")
    urls = _collect_urls(raw_urls)
    raw_started = internal.get("submission_started_at")
    submission_started_at: datetime | None = None
    if isinstance(raw_started, datetime):
        submission_started_at = raw_started
    elif isinstance(raw_started, str) and raw_started.strip():
        try:
            submission_started_at = datetime.fromisoformat(raw_started.strip())
        except ValueError:
            submission_started_at = None
    return AsyncJob(
        id=job.id,
        provider=_text(internal.get("provider") or raw_payload.get("provider")),
        remote_id=_text(internal.get("remote_id")),
        state=state,
        idempotency_key=_text(job.idempotency_key or internal.get("idempotency_key")),
        request_digest=_text(internal.get("request_digest")),
        request=request,
        result_urls=list(urls),
        progress=_coerce_progress(internal.get("progress")),
        error_code=job.error_code,
        error_message=job.error_message,
        submission_owner=_text(internal.get("submission_owner")),
        submission_started_at=submission_started_at,
        created_at=job.created_at,
        updated_at=job.updated_at,
        metadata=user_metadata,
    )


def _coerce_status(value: Any) -> AsyncJobStatus:
    if isinstance(value, AsyncJobStatus):
        return value
    if isinstance(value, AsyncJobState):
        return AsyncJobStatus(state=value)
    if isinstance(value, str):
        return AsyncJobStatus(state=_coerce_state(value))
    if isinstance(value, Mapping):
        state: Any = AsyncJobState.RUNNING
        for key in _STATE_KEYS:
            if key in value and value[key] is not None:
                state = value[key]
                break
        urls: Any = None
        for key in _RESULT_KEYS:
            if key in value and value[key] is not None:
                urls = value[key]
                break
        progress = value.get("progress")
        if progress is None:
            progress = value.get("percent")
        error = value.get("error")
        error_code = value.get("error_code") or value.get("code") or ""
        error_message = value.get("error_message") or value.get("message") or ""
        if isinstance(error, Mapping):
            error_code = error_code or error.get("code", "")
            error_message = error_message or error.get("message", "")
        elif isinstance(error, str) and not error_message:
            error_message = error
        metadata = value.get("metadata")
        return AsyncJobStatus(
            state=_coerce_state(state),
            progress=_coerce_progress(progress),
            result_urls=_collect_urls(urls),
            error_code=_text(error_code),
            error_message=_text(error_message),
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        )

    state = AsyncJobState.RUNNING
    for key in _STATE_KEYS:
        candidate = getattr(value, key, None)
        if candidate is not None:
            state = candidate
            break
    urls = None
    for key in _RESULT_KEYS:
        candidate = getattr(value, key, None)
        if candidate is not None:
            urls = candidate
            break
    return AsyncJobStatus(
        state=_coerce_state(state),
        progress=_coerce_progress(getattr(value, "progress", None)),
        result_urls=_collect_urls(urls),
        error_code=_text(getattr(value, "error_code", "")),
        error_message=_text(getattr(value, "error_message", "")),
    )


def _apply_status(job: AsyncJob, status: AsyncJobStatus) -> AsyncJob:
    metadata = dict(job.metadata)
    metadata.update(status.metadata)
    return job.model_copy(
        update={
            "state": status.state,
            "progress": status.progress
            if status.progress is not None
            else job.progress,
            "result_urls": list(status.result_urls) or list(job.result_urls),
            "error_code": status.error_code or job.error_code,
            "error_message": status.error_message or job.error_message,
            "metadata": metadata,
            "updated_at": utcnow(),
        }
    )


def _positive_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be a finite positive number")
    return number


class AsyncMediaJobHelper:
    """Drive provider-side asynchronous media jobs with durable state.

    Pass either a :class:`AsyncJobBackend`, or explicit ``submit``, ``poll`` and
    ``cancel`` callables. Supply an SQLAlchemy ``engine`` to persist records in
    SQLite; without one the helper uses an in-memory store. ``clock`` and
    ``sleep`` are injectable so timeout tests do not wait in real time.
    """

    def __init__(
        self,
        backend: AsyncJobBackend | None = None,
        *,
        engine: Engine | None = None,
        store: AsyncJobStore | None = None,
        submit: Callable[[Any], Any] | None = None,
        poll: Callable[[str], Any] | None = None,
        cancel: Callable[[str], Any] | None = None,
        provider: str = "",
        kind: str = "media_async",
        http: HttpClient | None = None,
        http_config: HttpClientConfig | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
        max_download_bytes: int | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if store is not None and engine is not None:
            raise TypeError("pass either an engine or a store, not both")
        if http is not None and transport is not None:
            raise TypeError("pass either an http client or a transport, not both")

        backend_submit = getattr(backend, "submit", None)
        backend_poll = getattr(backend, "poll", None)
        backend_cancel = getattr(backend, "cancel", None)
        self._submit_fn = submit or backend_submit
        self._poll_fn = poll or backend_poll
        self._cancel_fn = cancel or backend_cancel
        if self._submit_fn is None or self._poll_fn is None:
            raise TypeError("helper needs a backend or submit and poll callables")

        self._backend = backend
        self._store: AsyncJobStore
        if store is not None:
            self._store = store
        elif engine is not None:
            self._store = DatabaseAsyncJobStore(engine, kind=kind)
        else:
            self._store = InMemoryAsyncJobStore()
        self.provider = (
            _text(provider)
            or _text(getattr(backend, "name", ""))
            or _text(getattr(backend, "provider", ""))
        )
        self.kind = _text(kind) or "media_async"
        self.timeout_seconds = _positive_number(
            timeout_seconds, label="timeout_seconds"
        )
        self.poll_interval_seconds = _positive_number(
            poll_interval_seconds, label="poll_interval_seconds"
        )
        self.max_download_bytes = self._validate_max_bytes(max_download_bytes)
        self._clock = clock or time.monotonic
        self._sleep = sleep or asyncio.sleep
        self._http = http
        self._http_config = http_config or HttpClientConfig()
        self._transport = transport
        self._owns_http = http is None
        self._submit_lock = asyncio.Lock()

    @property
    def store(self) -> AsyncJobStore:
        """The persistence backing this helper."""
        return self._store

    @property
    def backend(self) -> AsyncJobBackend | None:
        """The provider backend supplied at construction, if any."""
        return self._backend

    @property
    def http(self) -> HttpClient:
        """The HTTP client used to download results."""
        if self._http is None:
            self._http = HttpClient(
                self._http_config,
                transport=self._transport,
            )
        return self._http

    @staticmethod
    def _validate_max_bytes(value: int | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("max_download_bytes must be an integer or None")
        if value <= 0:
            raise ValueError("max_download_bytes must be positive")
        return value

    def _timeout(self, value: float | None) -> float:
        if value is None:
            return self.timeout_seconds
        return _positive_number(value, label="timeout_seconds")

    def _interval(self, value: float | None) -> float:
        if value is None:
            return self.poll_interval_seconds
        return _positive_number(value, label="poll_interval_seconds")

    async def _call(self, function: Callable[..., Any], *args: Any) -> Any:
        result = function(*args)
        if inspect.isawaitable(result):
            return await result
        return result

    def get(self, job_id: str) -> AsyncJob | None:
        """Return a stored job by local id, or ``None``."""
        return self._store.get(_text(job_id))

    def require(self, job_id: str) -> AsyncJob:
        """Return a stored job or raise :class:`JobNotFoundError`."""
        text = _text(job_id)
        job = self.get(text)
        if job is None:
            raise JobNotFoundError(
                f"async job {text!r} does not exist",
                details={"job_id": text},
            )
        return job

    def _resolve(self, job: AsyncJob | str) -> AsyncJob:
        if isinstance(job, AsyncJob):
            stored = self._store.get(job.id)
            return stored if stored is not None else job
        return self.require(job)

    @staticmethod
    def _reuse_existing(existing: AsyncJob, *, digest: str, key: str) -> AsyncJob:
        if existing.request_digest and existing.request_digest != digest:
            raise IdempotencyConflict(
                f"idempotency key {key!r} was already used for async "
                f"job {existing.id} with a different request",
                details={
                    "idempotency_key": key,
                    "job_id": existing.id,
                    "existing_job_id": existing.id,
                },
            )
        return existing

    async def _reuse_or_wait(
        self, existing: AsyncJob, *, digest: str, key: str, request: Any = None
    ) -> AsyncJob:
        """Return *existing*, waiting while a live submission is in flight.

        A pending row with no remote id whose submitter is gone was left by a
        crash; it is resumed on the same local job id so a repeated key recovers
        it. A live submission is still awaited, and a record that never gains a
        remote id before the deadline raises :class:`AsyncJobTimeoutError`.
        """
        current = existing
        deadline = self._clock() + self.timeout_seconds
        while not current.remote_id and not current.terminal:
            if self._should_resume(current):
                return await self._resume(
                    current, digest=digest, key=key, request=request
                )
            remaining = deadline - self._clock()
            if remaining <= 0.0:
                raise AsyncJobTimeoutError(
                    f"async job {current.id!r} was still being submitted after "
                    f"{self.timeout_seconds:g} seconds",
                    job=current,
                )
            await self._sleep(min(self.poll_interval_seconds, remaining))
            refreshed = self._store.get(current.id)
            if refreshed is not None:
                current = refreshed
        return self._reuse_existing(current, digest=digest, key=key)

    def _should_resume(self, job: AsyncJob) -> bool:
        """Whether a pending job has no live submitter and can be resubmitted."""
        if job.remote_id or job.terminal:
            return False
        owner = job.submission_owner
        if not owner:
            return True
        if owner in _ACTIVE_SUBMISSIONS:
            return False
        pid = _owner_pid(owner)
        if pid is None or pid == os.getpid():
            return True
        started = job.submission_started_at
        if started is None:
            return True
        if started.tzinfo is None:
            started = started.replace(tzinfo=UTC)
        return (utcnow() - started).total_seconds() >= self.timeout_seconds

    async def _resume(
        self, job: AsyncJob, *, digest: str, key: str, request: Any = None
    ) -> AsyncJob:
        """Re-submit *job* after its submitter died before storing a remote id."""
        current = self._reuse_existing(job, digest=digest, key=key)
        submit_fn = self._submit_fn
        if submit_fn is None:
            raise InvalidRequestError("helper has no submit callable")
        payload = request if request is not None else current.request
        owner = _new_submission_owner()
        started = utcnow()
        pending = current.model_copy(
            update={
                "state": AsyncJobState.PENDING,
                "submission_owner": owner,
                "submission_started_at": started,
                "updated_at": started,
            }
        )
        _ACTIVE_SUBMISSIONS.add(owner)
        try:
            self._store.save(pending)
            try:
                response = await self._call(submit_fn, payload)
            except Exception as exc:
                failed = pending.model_copy(
                    update={
                        "state": AsyncJobState.FAILED,
                        "error_code": _error_code(exc),
                        "error_message": _error_message(exc),
                        "submission_owner": "",
                        "submission_started_at": None,
                        "updated_at": utcnow(),
                    }
                )
                self._store.save(failed)
                raise
            remote_id = _remote_id(response)
            if not remote_id:
                failed = pending.model_copy(
                    update={
                        "state": AsyncJobState.FAILED,
                        "error_code": "invalid_request",
                        "error_message": (
                            "provider submit response did not contain a job id"
                        ),
                        "submission_owner": "",
                        "submission_started_at": None,
                        "updated_at": utcnow(),
                    }
                )
                self._store.save(failed)
                raise InvalidRequestError(
                    "provider submit response did not contain a job id",
                    details={"job_id": current.id, "provider": self.provider},
                )
            resumed = pending.model_copy(
                update={
                    "remote_id": remote_id,
                    "state": AsyncJobState.RUNNING,
                    "submission_owner": "",
                    "submission_started_at": None,
                    "updated_at": utcnow(),
                }
            )
            return self._store.save(resumed)
        finally:
            _ACTIVE_SUBMISSIONS.discard(owner)

    async def submit(
        self,
        request: Any,
        *,
        idempotency_key: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> AsyncJob:
        """Submit *request*, or return the job stored for *idempotency_key*.

        The local record is written before the remote call so a crash during
        submission leaves an auditable row. The remote identifier is written
        back as soon as the backend returns it. Reusing a key with a different
        request raises :class:`IdempotencyConflict`. A key that is already being
        submitted by another process waits for its remote id instead of
        returning a pending job that cannot be polled; a key whose submitter
        died before storing a remote id resumes the stored job.
        """
        submit_fn = self._submit_fn
        if submit_fn is None:
            raise InvalidRequestError("helper has no submit callable")
        key = _text(idempotency_key)
        request_data = _request_mapping(request)
        digest = _request_digest(self.provider, request_data)

        async with self._submit_lock:
            if key:
                existing = self._store.find_by_idempotency_key(key)
                if existing is not None:
                    return await self._reuse_or_wait(
                        existing, digest=digest, key=key, request=request
                    )

            now = utcnow()
            owner = _new_submission_owner()
            job = AsyncJob(
                id=new_job_id(),
                provider=self.provider,
                state=AsyncJobState.PENDING,
                idempotency_key=key,
                request_digest=digest,
                request=request_data,
                submission_owner=owner,
                submission_started_at=now,
                metadata=dict(metadata or {}),
                created_at=now,
                updated_at=now,
            )
            _ACTIVE_SUBMISSIONS.add(owner)
            try:
                try:
                    self._store.save(job)
                except IntegrityError:
                    if not key:
                        raise
                    existing = self._store.find_by_idempotency_key(key)
                    if existing is None:
                        raise
                    return await self._reuse_or_wait(
                        existing, digest=digest, key=key, request=request
                    )
                try:
                    response = await self._call(submit_fn, request)
                except Exception as exc:
                    failed = job.model_copy(
                        update={
                            "state": AsyncJobState.FAILED,
                            "error_code": _error_code(exc),
                            "error_message": _error_message(exc),
                            "submission_owner": "",
                            "submission_started_at": None,
                            "updated_at": utcnow(),
                        }
                    )
                    self._store.save(failed)
                    raise

                remote_id = _remote_id(response)
                if not remote_id:
                    failed = job.model_copy(
                        update={
                            "state": AsyncJobState.FAILED,
                            "error_code": "invalid_request",
                            "error_message": (
                                "provider submit response did not contain a job id"
                            ),
                            "submission_owner": "",
                            "submission_started_at": None,
                            "updated_at": utcnow(),
                        }
                    )
                    self._store.save(failed)
                    raise InvalidRequestError(
                        "provider submit response did not contain a job id",
                        details={"job_id": job.id, "provider": self.provider},
                    )

                submitted = job.model_copy(
                    update={
                        "remote_id": remote_id,
                        "state": AsyncJobState.RUNNING,
                        "submission_owner": "",
                        "submission_started_at": None,
                        "updated_at": utcnow(),
                    }
                )
                return self._store.save(submitted)
            finally:
                _ACTIVE_SUBMISSIONS.discard(owner)

    async def resume(self, job: AsyncJob | str) -> AsyncJob:
        """Resume a pending job whose submitter died before storing a remote id.

        A job that already has a remote id or reached a terminal state is
        returned unchanged.
        """
        current = self._resolve(job)
        if current.remote_id or current.terminal:
            return current
        return await self._resume(
            current,
            digest=current.request_digest,
            key=current.idempotency_key,
        )

    async def poll(
        self,
        job: AsyncJob | str,
        *,
        timeout_seconds: float | None = None,
        interval_seconds: float | None = None,
    ) -> AsyncJob:
        """Poll until *job* is terminal or the deadline passes.

        Returns the final record on success, failure or cancellation. On a
        deadline it marks the local record ``timed_out`` and raises
        :class:`AsyncJobTimeoutError`; it does not cancel the remote job, so a
        caller can still choose to cancel or keep waiting explicitly.
        """
        poll_fn = self._poll_fn
        if poll_fn is None:
            raise InvalidRequestError("helper has no poll callable")
        current = self._resolve(job)
        if current.terminal:
            return current
        if not current.remote_id:
            raise InvalidRequestError(
                f"async job {current.id!r} has no remote job id to poll",
                details={"job_id": current.id},
            )

        timeout = self._timeout(timeout_seconds)
        interval = self._interval(interval_seconds)
        deadline = self._clock() + timeout

        while True:
            remaining = deadline - self._clock()
            if remaining <= 0.0:
                timed_out = current.model_copy(
                    update={
                        "state": AsyncJobState.TIMED_OUT,
                        "error_code": "timeout",
                        "error_message": (
                            f"async job did not finish within {timeout:g} seconds"
                        ),
                        "updated_at": utcnow(),
                    }
                )
                self._store.save(timed_out)
                raise AsyncJobTimeoutError(
                    f"async job {current.id!r} timed out after {timeout:g} seconds",
                    job=timed_out,
                )

            response = await self._call(poll_fn, current.remote_id)
            status = _coerce_status(response)
            current = self._store.save(_apply_status(current, status))
            if current.terminal:
                return current

            remaining = deadline - self._clock()
            if remaining <= 0.0:
                continue
            await self._sleep(min(interval, remaining))

    async def wait(
        self,
        job: AsyncJob | str,
        *,
        timeout_seconds: float | None = None,
        interval_seconds: float | None = None,
    ) -> AsyncJob:
        """Alias for :meth:`poll`."""
        return await self.poll(
            job,
            timeout_seconds=timeout_seconds,
            interval_seconds=interval_seconds,
        )

    poll_until_done = wait

    def status(self, job: AsyncJob | str) -> AsyncJob:
        """Return the stored state of *job* without polling the provider."""
        return self._resolve(job)

    async def cancel(self, job: AsyncJob | str) -> AsyncJob:
        """Cancel *job* at the provider and mark the local record cancelled.

        Cancelling a succeeded, failed or already cancelled job is a no-op, so a
        double cancel cannot call the provider twice. A job that timed out
        locally may still be running remotely, so it is still cancelled.
        """
        current = self._resolve(job)
        if current.state is not AsyncJobState.TIMED_OUT and current.terminal:
            return current
        if current.remote_id:
            cancel_fn = self._cancel_fn
            if cancel_fn is None:
                raise InvalidRequestError(
                    f"provider for async job {current.id!r} cannot cancel",
                    details={"job_id": current.id},
                )
            await self._call(cancel_fn, current.remote_id)
        cancelled = current.model_copy(
            update={
                "state": AsyncJobState.CANCELLED,
                "error_code": "",
                "error_message": "",
                "updated_at": utcnow(),
            }
        )
        return self._store.save(cancelled)

    async def download(
        self,
        source: AsyncJob | str,
        *,
        destination: str | Path | None = None,
        max_bytes: int | None = None,
        timeout_seconds: float | None = None,
    ) -> bytes:
        """Download the first result of *source*.

        *source* may be a stored job id, a job record or an absolute result
        URL. The response is capped by ``max_bytes`` (or the helper default)
        before it is returned or written to *destination*.
        """
        if isinstance(source, AsyncJob):
            url = self._first_result_url(source)
        elif isinstance(source, str) and source.strip().startswith(
            ("http://", "https://")
        ):
            url = source.strip()
        else:
            url = self._first_result_url(self._resolve(source))

        url = validate_http_url(url, field="download url")
        limit = self._validate_max_bytes(
            self.max_download_bytes if max_bytes is None else max_bytes
        )
        response = await self.http.get(
            url,
            timeout=timeout_seconds,
            raise_for_status=True,
        )
        try:
            length = response.headers.get("content-length")
            if length is not None and length.strip().isdigit():
                declared = int(length.strip())
                if limit is not None and declared > limit:
                    raise ValidationError(
                        f"download is {declared} bytes, above the {limit} byte cap",
                        details={"url": url, "size_bytes": declared, "limit": limit},
                    )
            data = response.content
            if limit is not None and len(data) > limit:
                raise ValidationError(
                    f"download is {len(data)} bytes, above the {limit} byte cap",
                    details={"url": url, "size_bytes": len(data), "limit": limit},
                )
        finally:
            await response.aclose()

        if destination is not None:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return data

    async def download_all(
        self,
        job: AsyncJob | str,
        *,
        max_bytes: int | None = None,
        timeout_seconds: float | None = None,
    ) -> list[bytes]:
        """Download every result URL of *job* in order."""
        current = self._resolve(job)
        if not current.result_urls:
            raise InvalidRequestError(
                f"async job {current.id!r} has no result to download",
                details={"job_id": current.id},
            )
        return [
            await self.download(
                url,
                max_bytes=max_bytes,
                timeout_seconds=timeout_seconds,
            )
            for url in current.result_urls
        ]

    @staticmethod
    def _first_result_url(job: AsyncJob) -> str:
        if not job.result_urls:
            raise InvalidRequestError(
                f"async job {job.id!r} has no result to download",
                details={"job_id": job.id, "state": job.state.value},
            )
        return job.result_urls[0]

    async def aclose(self) -> None:
        """Close the HTTP client when this helper created it."""
        if self._owns_http and self._http is not None:
            await self._http.aclose()
            self._http = None

    async def close(self) -> None:
        """Alias for :meth:`aclose`."""
        await self.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


AsyncJobHelper = AsyncMediaJobHelper
"""Alias for callers that do not name the media capability."""

MediaJobHelper = AsyncMediaJobHelper
"""Alias matching the record name."""

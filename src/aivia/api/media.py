"""Media toolkit routes: submit a generation job, poll it and download results.

The HTTP surface is deliberately thin. A submission validates the request with
the same typed protocol models the toolkit uses, stores a durable job row and
then runs the call through :class:`~aivia.media.MediaToolkit` in the background.
Status is read from the job queue, so a job survives an API restart, and a
finished result is served from the assets the media service already stored.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import uuid
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)
from sqlalchemy import Connection, Engine

from aivia.core.config import JobsConfig
from aivia.core.db import (
    JobRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.db import (
    database_path as default_database_path,
)
from aivia.core.errors import QuotaError
from aivia.core.jobs import (
    JobNotClaimedError,
    JobNotFoundError,
    JobQueue,
)
from aivia.core.logging import get_logger
from aivia.core.models import Job, JobState
from aivia.core.models.base import utcnow
from aivia.media.facade import MEDIA_METRICS_FILENAME, MediaServiceFacade
from aivia.media.toolkit import MediaResult, MediaToolkit
from aivia.providers.protocols.image import ImageRequest
from aivia.providers.protocols.image_edit import ImageEditRequest
from aivia.providers.protocols.music import MusicRequest
from aivia.providers.protocols.stt import STTRequest
from aivia.providers.protocols.tts import TTSRequest
from aivia.providers.protocols.video import VideoRequest

__all__ = ["MediaJobRunner", "MediaJobSubmission", "register_media_routes"]

_MEDIA_KINDS = frozenset({"image", "video", "speech", "transcription", "music", "edit"})
_KIND_ALIASES = {
    "tts": "speech",
    "text-to-speech": "speech",
    "text_to_speech": "speech",
    "transcribe": "transcription",
    "stt": "transcription",
    "speech-to-text": "transcription",
    "speech_to_text": "transcription",
    "image-edit": "edit",
    "image_edit": "edit",
}

_CALL_OPTIONS = frozenset(
    {
        "estimated",
        "key_prefix",
        "no_cache",
        "pinned",
        "record_cost",
        "record_sidecar",
        "remaining_budget",
        "run_id",
    }
)

_SUBMISSION_FIELDS = frozenset(
    {
        "id",
        "kind",
        "request",
        "payload",
        "options",
        "metadata",
        "idempotency_key",
        "max_attempts",
        "prompt",
    }
)


def _canonical_kind(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = _KIND_ALIASES.get(text, text)
    if text not in _MEDIA_KINDS:
        supported = ", ".join(sorted(_MEDIA_KINDS))
        raise ValueError(f"unsupported media kind {value!r} (supported: {supported})")
    return text


def _decode_base64(value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("media data must be valid base64") from exc


def _decode_data_url(value: str) -> tuple[bytes, str | None]:
    header, separator, payload = value.partition(",")
    if not separator:
        raise ValueError("data URL is missing its payload")
    media_type = ""
    if header.startswith("data:"):
        media_type = header[5:].split(";", maxsplit=1)[0]
    if ";base64" not in header:
        raise ValueError("data URL must be base64 encoded")
    return _decode_base64(payload), media_type or None


def _media_input(value: Any, *, default_type: str) -> Any:
    """Return a JSON-safe mapping the protocol can validate as a media input."""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("http://", "https://")):
            return {"url": text}
        if text.startswith("data:"):
            data, media_type = _decode_data_url(text)
            return {"data": data, "media_type": media_type or default_type}
        return {"data": _decode_base64(text), "media_type": default_type}
    if isinstance(value, Mapping):
        result = dict(value)
        inline = result.get("data")
        if isinstance(inline, str):
            result["data"] = _decode_base64(inline)
        return result
    return value


def _prepare_request(kind: str, values: Mapping[str, Any]) -> dict[str, Any]:
    """Copy *values* and decode the media fields a JSON request may carry."""
    request = dict(values)
    if kind == "speech":
        if not request.get("text") and isinstance(request.get("prompt"), str):
            request["text"] = request["prompt"]
        request.setdefault("voice", "default")
    elif kind == "edit":
        if "image" in request:
            request["image"] = _media_input(request["image"], default_type="image/png")
        if "mask" in request:
            request["mask"] = _media_input(request["mask"], default_type="image/png")
    elif kind == "transcription":
        if isinstance(request.get("audio"), str):
            audio = request["audio"].strip()
            if audio.startswith(("http://", "https://")):
                request["audio_url"] = audio
                request.pop("audio", None)
            else:
                request["audio"] = _media_input(audio, default_type="audio/mpeg")
        elif "audio" in request:
            request["audio"] = _media_input(request["audio"], default_type="audio/mpeg")
        if "audio_base64" in request and "audio" not in request:
            request["audio"] = _decode_base64(str(request.pop("audio_base64")))
        audio_value = request.get("audio")
        if isinstance(audio_value, Mapping):
            request["audio"] = audio_value.get("data")
            if audio_value.get("media_type"):
                request.setdefault("media_type", audio_value["media_type"])
            if audio_value.get("filename"):
                request.setdefault("filename", audio_value["filename"])
    return request


def _request_model(kind: str, values: Mapping[str, Any]) -> BaseModel:
    prepared = _prepare_request(kind, values)
    if kind == "image":
        return ImageRequest(**prepared)
    if kind == "video":
        return VideoRequest(**prepared)
    if kind == "speech":
        return TTSRequest(**prepared)
    if kind == "transcription":
        return STTRequest(**prepared)
    if kind == "music":
        return MusicRequest(**prepared)
    return ImageEditRequest.model_validate(prepared)


class MediaJobSubmission(BaseModel):
    """One media toolkit call to queue.

    ``kind`` names the capability. Request fields may live under :attr:`request`
    or at the top level; ``prompt`` is a shorthand for the prompt or text field
    of the selected capability. ``options`` may carry the small set of safe call
    options the toolkit understands, such as ``no_cache``.
    """

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    kind: str = Field(
        default="",
        validation_alias=AliasChoices("kind", "media_kind", "type", "job_kind"),
    )
    request: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = ""
    max_attempts: int = Field(default=3, ge=1)
    prompt: str | None = Field(
        default=None,
        validation_alias=AliasChoices("prompt", "text"),
    )

    @field_validator("kind", "idempotency_key", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    def request_values(self, kind: str | None = None) -> dict[str, Any]:
        """Return the request fields with top-level shorthands merged in."""
        values = dict(self.payload)
        values.update(self.request)
        for name, value in (self.model_extra or {}).items():
            if name in _CALL_OPTIONS or name in _SUBMISSION_FIELDS:
                continue
            values[name] = value
        for name, value in self.options.items():
            if name not in _CALL_OPTIONS:
                values[name] = value
        if self.prompt is not None:
            selected = _canonical_kind(kind or self.kind)
            field = "text" if selected == "speech" else "prompt"
            values.setdefault(field, self.prompt)
        return values

    def call_options(self) -> dict[str, Any]:
        """Return only the safe toolkit call options."""
        values = dict(self.options)
        for name, value in (self.model_extra or {}).items():
            if name in _CALL_OPTIONS:
                values.setdefault(name, value)
        return {name: value for name, value in values.items() if name in _CALL_OPTIONS}


def _local_path(value: str) -> Path | None:
    parsed = urlsplit(value)
    if parsed.scheme.lower() == "file":
        return Path(unquote(parsed.path))
    if not parsed.scheme:
        return Path(value)
    return None


def _media_state(job: Job) -> Mapping[str, Any]:
    media = job.metadata.get("media")
    return media if isinstance(media, Mapping) else {}


def _job_kind(job: Job) -> str:
    payload = job.payload if isinstance(job.payload, Mapping) else {}
    value = payload.get("kind")
    if value is None and job.kind.startswith("media."):
        value = job.kind.split(".", maxsplit=1)[1]
    try:
        return _canonical_kind(value)
    except ValueError:
        return str(value or job.kind)


def _download_links(job: Job, result: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not result:
        return []
    paths = result.get("paths") or result.get("urls") or []
    if not isinstance(paths, list):
        return []
    links: list[dict[str, Any]] = []
    for index, value in enumerate(paths):
        if not isinstance(value, str) or not value:
            continue
        path = _local_path(value)
        links.append(
            {
                "index": index,
                "name": path.name if path is not None else Path(value).name,
                "url": f"/api/v1/media/jobs/{job.id}/download/{index}",
            }
        )
    return links


def _find_idempotent(
    repository: JobRepository,
    connection: Connection,
    *,
    kind: str,
    idempotency_key: str,
) -> Job | None:
    """Return the media job stored for *idempotency_key*, if any."""
    for candidate in repository.list_in(connection, kind=kind):
        if candidate.metadata.get("idempotency_key") == idempotency_key:
            return candidate
    return None


def _job_payload(job: Job) -> dict[str, Any]:
    media = _media_state(job)
    result = media.get("result") if isinstance(media, Mapping) else None
    if not isinstance(result, Mapping):
        result = None
    payload: dict[str, Any] = {
        "id": job.id,
        "job_id": job.id,
        "kind": _job_kind(job),
        "media_kind": _job_kind(job),
        "state": job.state.value,
        "status": job.state.value,
        "attempts": job.attempts,
        "max_attempts": job.max_attempts,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "request": job.payload.get("request", {}) if job.payload else {},
        "metadata": job.metadata,
        "result": result,
    }
    if result is not None:
        payload["provider"] = result.get("provider", "")
        payload["model"] = result.get("model", "")
        payload["cost_usd"] = result.get("cost_usd", 0.0)
        payload["cached"] = bool(result.get("cached", False))
        payload["paths"] = list(result.get("paths") or [])
        payload["urls"] = list(result.get("urls") or [])
        if result.get("text") is not None:
            payload["text"] = result["text"]
    links = _download_links(job, result)
    payload["downloads"] = links
    payload["download_url"] = links[0]["url"] if links else None
    if job.error_code or job.error_message:
        payload["error_code"] = job.error_code
        payload["error_message"] = job.error_message
        payload["error"] = {
            "code": job.error_code,
            "message": job.error_message,
        }
    return payload


def _result_payload(job: Job) -> dict[str, Any]:
    payload = _job_payload(job)
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise HTTPException(
            status_code=404,
            detail=f"job {job.id!r} has no result yet",
        )
    body = dict(result)
    body.update(
        {
            "job_id": job.id,
            "id": job.id,
            "kind": _job_kind(job),
            "media_kind": _job_kind(job),
            "state": job.state.value,
            "status": job.state.value,
            "downloads": payload["downloads"],
            "download_url": payload["download_url"],
        }
    )
    return body


def _jobs_settings(
    config: JobsConfig | Mapping[str, Any] | None,
) -> JobsConfig:
    if config is None:
        return JobsConfig.from_mapping({})
    if isinstance(config, JobsConfig):
        return config
    if isinstance(config, Mapping):
        return JobsConfig.from_mapping(config)
    raise TypeError("jobs must be a JobsConfig, a mapping or None")


class MediaJobRunner:
    """Execute one queued media job through a toolkit instance.

    The runner claims the persisted job before doing any paid work, keeps the
    claim alive with a heartbeat, records the result or failure on the job row
    and always releases the claim. Each runner uses its own worker id, so a job
    that another live API process still owns is not mistaken for a crashed one.
    A job that another runner already claimed is left untouched, so an API
    restart or a duplicate background task cannot execute the same submission
    twice.
    """

    def __init__(
        self,
        engine: Callable[[], Engine],
        toolkit: Callable[[], MediaToolkit],
        *,
        jobs: JobsConfig | Mapping[str, Any] | None = None,
        worker_id: str | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if worker_id is not None and (
            not isinstance(worker_id, str) or not worker_id.strip()
        ):
            raise ValueError("worker_id must be a non-empty name")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        if sleep is not None and not callable(sleep):
            raise TypeError("sleep must be callable")
        settings = _jobs_settings(jobs)
        self._engine = engine
        self._toolkit = toolkit
        self._jobs = jobs
        self._worker_id = (worker_id or f"api-media-{uuid.uuid4().hex}").strip()
        self._clock = clock or utcnow
        self._sleep = sleep or asyncio.sleep
        self._heartbeat_seconds = settings.heartbeat_seconds
        self._stale_after_seconds = settings.stale_after_seconds
        self._logger = get_logger("aivia.api.media")

    @property
    def worker_id(self) -> str:
        """The unique identity this runner claims and heartbeats under."""
        return self._worker_id

    def pending_job_ids(self) -> list[str]:
        """Return the ids of media jobs left waiting by a previous process."""
        engine = self._engine()
        return [
            job.id
            for job in JobRepository(engine).list(state=JobState.PENDING.value)
            if job.kind.startswith("media.")
        ]

    def running_job_ids(self) -> list[str]:
        """Return the ids of media jobs left claimed by a previous process."""
        engine = self._engine()
        return [
            job.id
            for job in JobRepository(engine).list(state=JobState.RUNNING.value)
            if job.kind.startswith("media.")
        ]

    def recover_running(self) -> list[str]:
        """Release stale claims a dead API process left on media jobs.

        A live API process keeps refreshing its claim, so only a media job whose
        heartbeat stopped for longer than ``jobs.stale_after_seconds`` is
        recovered. Its claim is failed as retryable, which returns it to
        ``pending`` (or dead-letters it once its attempts are exhausted) so
        :meth:`recover_pending` can execute it in this process.
        """
        engine = self._engine()
        queue = JobQueue(engine, config=self._jobs)
        recovered = queue.recover_stale(
            stale_after_seconds=self._stale_after_seconds,
            kind_prefix="media.",
            now=self._clock(),
        )
        return [job.id for job in recovered]

    async def recover_pending(self) -> None:
        """Run every media job a restart left waiting or claimed."""
        self.recover_running()
        for job_id in self.pending_job_ids():
            await self.run(job_id)

    async def run(self, job_id: str) -> None:
        """Claim and execute *job_id*, recording its result or failure."""
        engine = self._engine()
        queue = JobQueue(engine, config=self._jobs)
        job = queue.get(job_id)
        if job is None or job.state is not JobState.PENDING:
            return
        claimed = queue.claim(self._worker_id, job_id=job_id, now=self._clock())
        if claimed is None:
            return
        lost_claim = asyncio.Event()
        heartbeat = asyncio.create_task(
            self._heartbeat_loop(queue, claimed.id, lost_claim)
        )
        result: MediaResult | None = None
        error: Exception | None = None
        try:
            kind = _job_kind(claimed)
            payload = claimed.payload if isinstance(claimed.payload, Mapping) else {}
            request = payload.get("request")
            values = request if isinstance(request, Mapping) else {}
            options = payload.get("options")
            call_options = options if isinstance(options, Mapping) else {}
            prepared = _prepare_request(kind, values)
            result = await self._toolkit().generate(kind, prepared, **call_options)
        except Exception as exc:
            error = exc
        finally:
            await self._stop_heartbeat(heartbeat)
        if lost_claim.is_set():
            return
        if error is not None:
            self._record_failure(engine, claimed, error)
            with suppress(JobNotFoundError, JobNotClaimedError):
                queue.fail(
                    claimed.id,
                    self._worker_id,
                    error_code=str(getattr(error, "code", "") or "media_job_failed"),
                    error_message=str(error),
                    retryable=False,
                    now=self._clock(),
                )
        else:
            if result is None:
                return
            self._record_result(engine, claimed, result)
            with suppress(JobNotFoundError, JobNotClaimedError):
                queue.complete(claimed.id, self._worker_id, now=self._clock())

    async def _heartbeat_loop(
        self,
        queue: JobQueue,
        job_id: str,
        lost_claim: asyncio.Event,
    ) -> None:
        while True:
            await self._sleep(self._heartbeat_seconds)
            try:
                queue.heartbeat(job_id, self._worker_id, now=self._clock())
            except (JobNotFoundError, JobNotClaimedError):
                self._logger.warning(
                    "lost claim on media job %s; another process recovered it",
                    job_id,
                )
                lost_claim.set()
                return
            except Exception:
                self._logger.exception("heartbeat failed for media job %s", job_id)
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

    def _record_result(self, engine: Engine, job: Job, result: MediaResult) -> None:
        metadata = dict(job.metadata)
        metadata["media"] = {
            "kind": _job_kind(job),
            "result": result.to_dict(),
        }
        JobRepository(engine).update(job.id, metadata=metadata)

    def _record_failure(self, engine: Engine, job: Job, exc: BaseException) -> None:
        metadata = dict(job.metadata)
        metadata["media"] = {
            "kind": _job_kind(job),
            "error": {
                "code": str(getattr(exc, "code", "") or "media_job_failed"),
                "message": str(exc),
            },
        }
        JobRepository(engine).update(job.id, metadata=metadata)


def register_media_routes(
    app: FastAPI,
    *,
    engine: Engine | None,
    environ: Mapping[str, str] | None,
    database_path: str | Path | None,
    data_dir: str | Path | None,
    toolkit: MediaToolkit | Callable[[], MediaToolkit] | None = None,
    jobs: JobsConfig | Mapping[str, Any] | None = None,
    environment: str = "",
    profile: str = "",
) -> None:
    """Attach the media toolkit job routes to *app*.

    A caller may inject a ready toolkit, a factory, or nothing. Without one the
    first job builds a toolkit from the layered config on the same database the
    job rows live in. *jobs* supplies the queue depth limit. *environment* and
    *profile* select the config the lazily built toolkit uses.
    """

    def database() -> Engine:
        existing = getattr(app.state, "engine", None)
        if isinstance(existing, Engine):
            return existing
        created = create_database_engine(
            path=database_path,
            data_dir=data_dir,
            environ=environ,
        )
        upgrade_database(engine=created)
        app.state.engine = created
        app.state.api_engine_owned = True
        return created

    def build_toolkit() -> MediaToolkit:
        existing = getattr(app.state, "media_toolkit", None)
        if isinstance(existing, MediaToolkit):
            return existing
        if callable(toolkit):
            built = toolkit()
        elif toolkit is not None:
            built = toolkit
        else:
            built = _toolkit_from_config(
                database(),
                environ=environ,
                environment=environment,
                profile=profile,
                data_dir=data_dir,
                database_path=database_path,
            )
        app.state.media_toolkit = built
        return built

    def queue_for(target: Engine) -> JobQueue:
        return JobQueue(target, config=jobs)

    runner = MediaJobRunner(database, build_toolkit, jobs=jobs)

    def database_exists() -> bool:
        existing = getattr(app.state, "engine", None)
        if isinstance(existing, Engine):
            return True
        if database_path is not None:
            return Path(database_path).expanduser().exists()
        return default_database_path(data_dir, environ=environ).exists()

    async def recover_pending_media_jobs() -> None:
        """Run media jobs a previous API process left waiting or claimed."""
        if database_exists():
            await runner.recover_pending()

    app.router.add_event_handler("startup", recover_pending_media_jobs)

    def submit(
        submission: MediaJobSubmission, kind_override: str | None = None
    ) -> dict[str, Any]:
        try:
            kind = _canonical_kind(kind_override or submission.kind)
            values = submission.request_values(kind)
            _request_model(kind, values)
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        target = database()
        queue = queue_for(target)
        values_for_job = {"kind": kind, "request": values}
        call_options = submission.call_options()
        if call_options:
            values_for_job["options"] = call_options
        metadata = dict(submission.metadata)
        idempotency_key = submission.idempotency_key
        if idempotency_key:
            metadata["idempotency_key"] = idempotency_key
        job_values: dict[str, Any] = {
            "kind": f"media.{kind}",
            "payload": values_for_job,
            "metadata": metadata,
            "max_attempts": submission.max_attempts,
        }
        if submission.id:
            job_values["id"] = submission.id
        try:
            if idempotency_key:
                with queue.write_transaction() as connection:
                    existing = _find_idempotent(
                        JobRepository(target),
                        connection,
                        kind=job_values["kind"],
                        idempotency_key=idempotency_key,
                    )
                    if existing is not None:
                        if existing.payload != values_for_job:
                            raise HTTPException(
                                status_code=409,
                                detail=(
                                    f"idempotency key {idempotency_key!r} "
                                    "was used for a different payload"
                                ),
                            )
                        return _job_payload(existing)
                    job = queue.enqueue(Job(**job_values), connection=connection)
            else:
                job = queue.enqueue(Job(**job_values))
        except QuotaError as exc:
            detail = {"message": exc.message, "code": exc.code, **exc.details}
            raise HTTPException(
                status_code=429,
                detail=detail,
                headers={"Retry-After": "1"},
            ) from exc
        return _job_payload(job)

    @app.post(
        "/api/v1/media/jobs",
        status_code=202,
        tags=["media"],
        summary="Submit a media generation job",
    )
    def submit_media_job(
        submission: MediaJobSubmission,
        background: BackgroundTasks,
    ) -> dict[str, Any]:
        """Queue one toolkit call and run it after the response is sent."""
        payload = submit(submission)
        background.add_task(runner.run, payload["job_id"])
        return payload

    @app.post(
        "/api/v1/media",
        status_code=202,
        tags=["media"],
        summary="Submit a media generation job",
        include_in_schema=False,
    )
    def submit_media_alias(
        submission: MediaJobSubmission,
        background: BackgroundTasks,
    ) -> dict[str, Any]:
        payload = submit(submission)
        background.add_task(runner.run, payload["job_id"])
        return payload

    @app.post(
        "/api/v1/media/generate",
        status_code=202,
        tags=["media"],
        summary="Submit a media generation job",
        include_in_schema=False,
    )
    def submit_media_generate(
        submission: MediaJobSubmission,
        background: BackgroundTasks,
    ) -> dict[str, Any]:
        payload = submit(submission)
        background.add_task(runner.run, payload["job_id"])
        return payload

    @app.post(
        "/api/v1/media/{kind}/jobs",
        status_code=202,
        tags=["media"],
        summary="Submit a media generation job for one capability",
        include_in_schema=False,
    )
    def submit_media_kind_job(
        kind: str,
        submission: MediaJobSubmission,
        background: BackgroundTasks,
    ) -> dict[str, Any]:
        payload = submit(submission, kind_override=kind)
        background.add_task(runner.run, payload["job_id"])
        return payload

    @app.post(
        "/api/v1/media/{kind}",
        status_code=202,
        tags=["media"],
        summary="Submit a media generation job for one capability",
        include_in_schema=False,
    )
    def submit_media_kind(
        kind: str,
        submission: MediaJobSubmission,
        background: BackgroundTasks,
    ) -> dict[str, Any]:
        payload = submit(submission, kind_override=kind)
        background.add_task(runner.run, payload["job_id"])
        return payload

    def status(job_id: str) -> dict[str, Any]:
        target = database()
        job = JobQueue(target).get(job_id)
        if job is None or not job.kind.startswith("media."):
            raise HTTPException(
                status_code=404,
                detail=f"media job {job_id!r} not found",
            )
        return _job_payload(job)

    @app.get(
        "/api/v1/media/jobs/{job_id}",
        tags=["media"],
        summary="Get media job status",
    )
    def media_job_status(job_id: str) -> dict[str, Any]:
        """Return the media job's state, result summary and download links."""
        return status(job_id)

    @app.get(
        "/api/v1/media/jobs/{job_id}/status",
        tags=["media"],
        summary="Get media job status",
        include_in_schema=False,
    )
    def media_job_status_alias(job_id: str) -> dict[str, Any]:
        return status(job_id)

    @app.get(
        "/api/v1/media/{job_id}",
        tags=["media"],
        summary="Get media job status",
        include_in_schema=False,
    )
    def media_job_status_short(job_id: str) -> dict[str, Any]:
        return status(job_id)

    def result(job_id: str) -> dict[str, Any]:
        target = database()
        job = JobQueue(target).get(job_id)
        if job is None or not job.kind.startswith("media."):
            raise HTTPException(
                status_code=404,
                detail=f"media job {job_id!r} not found",
            )
        return _result_payload(job)

    @app.get(
        "/api/v1/media/jobs/{job_id}/result",
        tags=["media"],
        summary="Get media job result",
    )
    def media_job_result(job_id: str) -> dict[str, Any]:
        """Return the finished result and its download links."""
        return result(job_id)

    def download(job_id: str, index: int) -> Any:
        target = database()
        job = JobQueue(target).get(job_id)
        if job is None or not job.kind.startswith("media."):
            raise HTTPException(
                status_code=404,
                detail=f"media job {job_id!r} not found",
            )
        media = _media_state(job)
        stored = media.get("result") if isinstance(media, Mapping) else None
        if not isinstance(stored, Mapping):
            raise HTTPException(
                status_code=404,
                detail=f"job {job_id!r} has no result yet",
            )
        paths = stored.get("paths") or stored.get("urls") or []
        if not isinstance(paths, list) or not paths:
            text = stored.get("text")
            if isinstance(text, str) and text:
                return PlainTextResponse(text)
            raise HTTPException(
                status_code=404,
                detail=f"job {job_id!r} has no downloadable result",
            )
        if index < 0 or index >= len(paths):
            raise HTTPException(
                status_code=404,
                detail=f"job {job_id!r} has no result at index {index}",
            )
        value = paths[index]
        if not isinstance(value, str) or not value:
            raise HTTPException(status_code=404, detail="result path is empty")
        path = _local_path(value)
        if path is None or not path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"result {index} for job {job_id!r} is not a local file",
            )
        return FileResponse(path, filename=path.name)

    @app.get(
        "/api/v1/media/jobs/{job_id}/download",
        tags=["media"],
        summary="Download the first media result",
    )
    def media_job_download(job_id: str, index: int = 0) -> Any:
        """Return the result file at *index* (the first by default)."""
        return download(job_id, index)

    @app.get(
        "/api/v1/media/jobs/{job_id}/download/{index}",
        tags=["media"],
        summary="Download one media result",
    )
    def media_job_download_index(job_id: str, index: int) -> Any:
        """Return the result file at *index*."""
        return download(job_id, index)

    @app.get(
        "/api/v1/media/jobs/{job_id}/result/{index}",
        tags=["media"],
        summary="Download one media result",
        include_in_schema=False,
    )
    def media_job_result_index(job_id: str, index: int) -> Any:
        return download(job_id, index)

    @app.get(
        "/api/v1/media/jobs/{job_id}/files/{index}",
        tags=["media"],
        summary="Download one media result",
        include_in_schema=False,
    )
    def media_job_files_index(job_id: str, index: int) -> Any:
        return download(job_id, index)


def _toolkit_from_config(
    engine: Engine,
    *,
    environ: Mapping[str, str] | None,
    environment: str = "",
    profile: str = "",
    data_dir: str | Path | None = None,
    database_path: str | Path | None = None,
) -> MediaToolkit:
    from aivia.core.config.resolve import resolve_config
    from aivia.core.paths import resolve_paths

    resolved = resolve_config(
        environ=environ,
        environment=environment or None,
        profile=profile or None,
    )
    root = resolved.to_app_config()
    if data_dir is not None:
        active_data_dir = Path(data_dir).expanduser()
    elif database_path is not None:
        active_data_dir = Path(database_path).expanduser().parent
    else:
        active_data_dir = resolve_paths(environ=environ).data_dir
    return MediaServiceFacade.from_config(
        root,
        data_dir=active_data_dir,
        engine=engine,
        metrics_path=active_data_dir / MEDIA_METRICS_FILENAME,
    )

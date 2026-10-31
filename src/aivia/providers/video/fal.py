"""fal video generation adapter using the fal queue API.

fal accepts a video generation request, returns a queue entry with status,
response and cancel URLs, and exposes the finished clip only after the queue
reports ``COMPLETED``. This adapter drives that lifecycle through
:class:`~aivia.providers.media.async_job.AsyncMediaJobHelper`, so submission,
queue state tracking, progress, timeouts, cancellation and downloads behave
like every other asynchronous media provider.

The adapter owns no routing, retry or budget policy. It uses the shared HTTP
client for transport and maps unsuccessful responses to the shared error
taxonomy. A safety filter is reported as
:class:`~aivia.core.errors.ContentPolicyError` rather than a generic invalid
request, because the router must not retry the same prompt against the same
provider.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

import httpx
from sqlalchemy import Engine

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import (
    ConfigError,
    ContentPolicyError,
    InvalidRequestError,
    RateLimitError,
    ValidationError,
)
from aivia.core.http import error_for_response
from aivia.core.models.provider import ProviderMeta
from aivia.core.secrets import SecretsLoader
from aivia.providers.base import Provider
from aivia.providers.media import (
    AsyncJob,
    AsyncJobState,
    AsyncJobStatus,
    AsyncMediaJobHelper,
)
from aivia.providers.protocols.media import VideoFormat
from aivia.providers.protocols.video import (
    CAPABILITY_VIDEO_GEN,
    VideoRequest,
    VideoResponse,
    VideoResult,
)
from aivia.providers.registry import register_provider

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_QUEUE_URL",
    "FAL_DEFAULT_BASE_URL",
    "FAL_QUEUE_URL",
    "Fal",
    "FalVideo",
    "FalVideoGen",
    "FalVideoGenProvider",
    "FalVideoProvider",
]

DEFAULT_BASE_URL = "https://queue.fal.run"
"""The fal queue API root, without a trailing slash."""

FAL_DEFAULT_BASE_URL = DEFAULT_BASE_URL
"""Alias that names the fal queue root explicitly."""

DEFAULT_QUEUE_URL = DEFAULT_BASE_URL
"""Alias for callers that describe the root as the queue URL."""

FAL_QUEUE_URL = DEFAULT_BASE_URL
"""Alias that names the fal queue URL explicitly."""

_DEFAULT_METADATA = ProviderMeta(
    name="fal",
    capabilities=[CAPABILITY_VIDEO_GEN],
)

_CONTENT_POLICY_MARKERS = (
    "content policy",
    "content_policy",
    "content filter",
    "content_filter",
    "safety filter",
    "safety check",
    "moderation",
    "nsfw",
    "filtered",
    "flagged",
    "inappropriate",
    "violates",
    "blocked by",
)

_RESULT_KEYS = ("videos", "video", "output", "outputs", "results", "data")

_NESTED_RESULT_KEYS = ("videos", "video", "output", "outputs", "results", "data")

_URL_KEYS = ("url", "file_url", "video_url", "download_url")

_THUMBNAIL_KEYS = ("thumbnail_url", "thumbnail", "poster_url")

_CONTENT_TYPE_FORMATS = {
    "video/mp4": VideoFormat.MP4,
    "video/webm": VideoFormat.WEBM,
    "video/quicktime": VideoFormat.MOV,
    "video/x-matroska": VideoFormat.MKV,
    "image/gif": VideoFormat.GIF,
}

_EXTENSION_FORMATS = {
    "mp4": VideoFormat.MP4,
    "webm": VideoFormat.WEBM,
    "mov": VideoFormat.MOV,
    "mkv": VideoFormat.MKV,
    "gif": VideoFormat.GIF,
}


def _validate_base_url(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("base_url must be a string")
    text = value.strip().rstrip("/")
    parsed = urlsplit(text)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ConfigError(
            f"base_url must be an absolute http(s) URL: {value!r}",
            details={"field": "base_url"},
        )
    if parsed.username is not None or parsed.password is not None:
        raise ConfigError(
            "base_url must not contain credentials",
            details={"field": "base_url"},
        )
    return text


def _resolve_api_key(
    api_key: str | None,
    api_key_ref: str | None,
    secrets: SecretsLoader | None,
) -> str:
    if secrets is not None and not isinstance(secrets, SecretsLoader):
        raise TypeError("secrets must be a SecretsLoader")
    if api_key is not None:
        if not isinstance(api_key, str):
            raise TypeError("api_key must be a string")
        text = api_key.strip()
        if not text:
            raise ConfigError("fal API key must be non-empty")
        return text
    if api_key_ref is None or not isinstance(api_key_ref, str):
        raise ConfigError(
            "fal video provider requires an API key or api_key_ref",
            details={"field": "api_key_ref"},
        )
    reference = api_key_ref.strip()
    if not reference:
        raise ConfigError(
            "fal video provider requires an API key or api_key_ref",
            details={"field": "api_key_ref"},
        )
    loader = secrets or SecretsLoader()
    return loader.get(reference)


def _validate_model(value: str) -> str:
    text = value.strip().strip("/")
    if not text:
        raise InvalidRequestError("fal video request needs a model id")
    if "://" in text or any(character.isspace() for character in text):
        raise InvalidRequestError(
            f"invalid fal model id: {value!r}",
            details={"field": "model"},
        )
    return text


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _error_text(body: Mapping[str, Any] | None) -> str:
    if body is None:
        return ""
    for key in ("detail", "error", "message", "error_message", "reason"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        nested = _mapping(value)
        if nested is not None:
            text = _error_text(nested)
            if text:
                return text
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                text = _error_text(_mapping(item))
                if text:
                    return text
    errors = body.get("errors")
    if isinstance(errors, Sequence) and not isinstance(errors, (str, bytes)):
        for item in errors:
            text = _error_text(_mapping(item))
            if text:
                return text
    return ""


def _error_code(body: Mapping[str, Any] | None) -> str:
    if body is None:
        return ""
    for key in ("code", "error_code", "type"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    nested = _mapping(body.get("error"))
    return _error_code(nested)


def _is_content_policy(*values: str) -> bool:
    text = " ".join(value for value in values if value).lower()
    return any(marker in text for marker in _CONTENT_POLICY_MARKERS)


def _looks_like_result(body: Mapping[str, Any]) -> bool:
    return any(key in body for key in _RESULT_KEYS)


def _has_error(body: Mapping[str, Any]) -> bool:
    if _looks_like_result(body):
        return False
    return bool(_error_text(body))


def _is_filtered_result(body: Mapping[str, Any]) -> bool:
    for source in (body, *_video_entries(body)):
        for key in ("has_nsfw_concepts", "nsfw", "filtered", "blocked"):
            value = source.get(key)
            if isinstance(value, bool) and value:
                return True
            if (
                isinstance(value, Sequence)
                and not isinstance(value, (str, bytes))
                and any(bool(item) for item in value)
            ):
                return True
    return False


def _json_body(response: httpx.Response) -> Mapping[str, Any] | None:
    try:
        parsed = response.json()
    except ValueError:
        return None
    return _mapping(parsed)


def _raise_fal_response(response: httpx.Response, provider: str) -> None:
    fallback = error_for_response(response)
    body = _json_body(response)
    if fallback is None and not (body is not None and _has_error(body)):
        return

    message = _error_text(body)
    code = _error_code(body)
    details: dict[str, Any] = {
        "provider": provider,
        "status": response.status_code,
    }
    if code:
        details["code"] = code
    request_id = response.headers.get("x-request-id")
    if request_id:
        details["request_id"] = request_id

    text = fallback.message if fallback is not None else "fal returned an error"
    if message:
        text = f"{text}: {message}"

    if _is_content_policy(message, code):
        raise ContentPolicyError(text, details=details)
    if fallback is None:
        raise InvalidRequestError(text, details=details)
    if isinstance(fallback, RateLimitError):
        raise RateLimitError(
            text,
            retry_after=fallback.retry_after,
            details=details,
        )
    raise type(fallback)(text, details=details)


def _raise_job_failure(body: Mapping[str, Any], provider: str) -> None:
    message = _error_text(body)
    code = _error_code(body)
    details: dict[str, Any] = {"provider": provider}
    if code:
        details["code"] = code
    text = message or "fal video job failed"
    if _is_content_policy(message, code):
        raise ContentPolicyError(text, details=details)
    raise InvalidRequestError(text, details=details)


def _build_payload(request: VideoRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "prompt": request.prompt,
        "num_videos": request.count,
    }
    if request.negative_prompt is not None:
        payload["negative_prompt"] = request.negative_prompt
    if request.seed is not None:
        payload["seed"] = request.seed
    if request.duration_seconds is not None:
        payload["duration"] = request.duration_seconds
    if request.aspect_ratio is not None:
        payload["aspect_ratio"] = request.aspect_ratio
    if request.width is not None and request.height is not None:
        payload["resolution"] = {
            "width": request.width,
            "height": request.height,
        }
    if request.fps is not None:
        payload["fps"] = request.fps
    payload["output_format"] = request.format.value
    if request.generate_audio:
        payload["generate_audio"] = True
    if request.audio_prompt is not None:
        payload["audio_prompt"] = request.audio_prompt
    return payload


def _entries_from_value(value: Any) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [{"url": text}] if text else []
    if isinstance(value, Mapping):
        if any(key in value for key in _URL_KEYS):
            return [value]
        entries: list[Mapping[str, Any]] = []
        for key in _NESTED_RESULT_KEYS:
            if key in value:
                entries.extend(_entries_from_value(value[key]))
        return entries
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        collected: list[Mapping[str, Any]] = []
        for item in value:
            collected.extend(_entries_from_value(item))
        return collected
    return []


def _video_entries(body: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in _RESULT_KEYS:
        if key in body:
            entries = _entries_from_value(body[key])
            if entries:
                return entries
    if "url" in body:
        return [body]
    return []


def _entry_url(entry: Mapping[str, Any]) -> str:
    for key in _URL_KEYS:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number <= 0.0:
        return None
    return number


def _entry_number(
    entry: Mapping[str, Any],
    body: Mapping[str, Any],
    key: str,
) -> float | None:
    for source in (entry, body):
        number = _number(source.get(key))
        if number is not None:
            return number
    return None


def _entry_int(
    entry: Mapping[str, Any],
    body: Mapping[str, Any],
    key: str,
) -> int | None:
    number = _entry_number(entry, body, key)
    if number is None:
        return None
    return int(number)


def _entry_seed(
    entry: Mapping[str, Any],
    body: Mapping[str, Any],
    request: VideoRequest,
) -> int | None:
    for source in (entry, body):
        value = source.get("seed")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return request.seed


def _entry_bool(
    entry: Mapping[str, Any],
    body: Mapping[str, Any],
    key: str,
) -> bool | None:
    for source in (entry, body):
        value = source.get(key)
        if isinstance(value, bool):
            return value
    return None


def _entry_format(
    entry: Mapping[str, Any],
    body: Mapping[str, Any],
    url: str,
    request: VideoRequest,
) -> VideoFormat:
    for source in (entry, body):
        for key in ("content_type", "mime_type"):
            value = source.get(key)
            if isinstance(value, str):
                found = _CONTENT_TYPE_FORMATS.get(value.strip().lower())
                if found is not None:
                    return found
    extension = urlsplit(url).path.rsplit(".", 1)[-1].lower()
    return _EXTENSION_FORMATS.get(extension, request.format)


def _entry_thumbnail(
    entry: Mapping[str, Any],
    body: Mapping[str, Any],
) -> str | None:
    for source in (entry, body):
        for key in _THUMBNAIL_KEYS:
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _videos_from_result(
    body: Mapping[str, Any],
    request: VideoRequest,
) -> list[VideoResult]:
    entries = _video_entries(body)
    if not entries:
        raise ValidationError(
            "fal returned no videos",
            details={"provider": "fal"},
        )
    videos: list[VideoResult] = []
    for entry in entries:
        url = _entry_url(entry)
        if not url:
            continue
        has_audio = _entry_bool(entry, body, "has_audio")
        videos.append(
            VideoResult(
                url=url,
                format=_entry_format(entry, body, url, request),
                width=_entry_int(entry, body, "width"),
                height=_entry_int(entry, body, "height"),
                duration_seconds=_entry_number(entry, body, "duration"),
                fps=_entry_int(entry, body, "fps"),
                has_audio=request.generate_audio if has_audio is None else has_audio,
                seed=_entry_seed(entry, body, request),
                thumbnail_url=_entry_thumbnail(entry, body),
                metadata={"content_type": _text(entry.get("content_type"))}
                if _text(entry.get("content_type"))
                else {},
            )
        )
    if not videos:
        raise ValidationError(
            "fal returned no video URLs",
            details={"provider": "fal"},
        )
    return videos


def _status_state(body: Mapping[str, Any]) -> AsyncJobState:
    raw = _text(body.get("status")) or _text(body.get("state"))
    text = raw.lower().replace("-", "_")
    if text in {"in_queue", "queued", "pending", "waiting", "submitted"}:
        return AsyncJobState.PENDING
    if text in {"in_progress", "processing", "running", "active"}:
        return AsyncJobState.RUNNING
    if text in {"completed", "complete", "success", "succeeded", "done"}:
        return AsyncJobState.SUCCEEDED
    if text in {"failed", "error", "failure"}:
        return AsyncJobState.FAILED
    if text in {"cancelled", "canceled", "aborted"}:
        return AsyncJobState.CANCELLED
    if _looks_like_result(body):
        return AsyncJobState.SUCCEEDED
    return AsyncJobState.RUNNING


def _progress(body: Mapping[str, Any]) -> float | None:
    for key in ("progress", "percent", "percentage", "percent_complete"):
        value = body.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip().rstrip("%")
            try:
                return float(text)
            except ValueError:
                continue
    return None


class FalVideoProvider(Provider):
    """A video provider backed by the fal queue API.

    ``api_key`` accepts an already resolved credential while ``api_key_ref``
    names a secret resolved through ``secrets``. ``default_model`` (or its
    ``model`` alias) supplies the fal model id when a request omits one.
    ``base_url`` may point at a local test server; ``transport`` injects an
    ``httpx`` mock transport so the offline suite never touches the network.
    """

    default_metadata = _DEFAULT_METADATA

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_ref: str | None = None,
        secrets: SecretsLoader | None = None,
        base_url: str = DEFAULT_BASE_URL,
        queue_url: str | None = None,
        default_model: str | None = None,
        model: str | None = None,
        download_results: bool = False,
        max_download_bytes: int | None = None,
        timeout_seconds: float = 600.0,
        poll_interval_seconds: float = 1.0,
        engine: Engine | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        name: str | None = None,
        health_timeout_seconds: float | None = None,
        close_timeout_seconds: float | None = None,
        http_config: HttpClientConfig | None = None,
    ) -> None:
        super().__init__(
            name=name,
            health_timeout_seconds=health_timeout_seconds,
            close_timeout_seconds=close_timeout_seconds,
            http_config=http_config,
        )
        if not isinstance(download_results, bool):
            raise TypeError("download_results must be a boolean")
        if transport is not None and not isinstance(
            transport, httpx.AsyncBaseTransport
        ):
            raise TypeError("transport must be an httpx.AsyncBaseTransport")

        selected = model if model is not None else default_model
        if selected is not None:
            if not isinstance(selected, str):
                raise TypeError("default_model must be a string or None")
            selected = _validate_model(selected)
        self._default_model = selected
        self._api_key = _resolve_api_key(api_key, api_key_ref, secrets)
        self._base_url = _validate_base_url(queue_url or base_url)
        self._download_results = download_results
        self._max_download_bytes = max_download_bytes
        if transport is not None:
            self._http = self.create_http_client(transport=transport)
        self._helper = AsyncMediaJobHelper(
            _FalVideoBackend(self),
            engine=engine,
            provider=self.name,
            kind="fal_video",
            http=self.http,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            max_download_bytes=max_download_bytes,
            clock=clock,
            sleep=sleep,
        )

    @property
    def helper(self) -> AsyncMediaJobHelper:
        """The asynchronous job helper driving this provider."""
        return self._helper

    @property
    def base_url(self) -> str:
        """The configured fal queue root, without a trailing slash."""
        return self._base_url

    @property
    def queue_url(self) -> str:
        """Alias for :attr:`base_url` that names the fal queue explicitly."""
        return self._base_url

    @property
    def default_model(self) -> str | None:
        """The model id used when a request does not name one."""
        return self._default_model

    @property
    def endpoint(self) -> str:
        """The generation endpoint for the configured default model."""
        if self._default_model is None:
            return self._base_url
        return self.endpoint_for(self._default_model)

    def endpoint_for(self, model: str) -> str:
        """Return the queue submission endpoint for *model*."""
        return f"{self._base_url}/{_validate_model(model)}"

    def _headers(self) -> dict[str, str]:
        return {
            "authorization": f"Key {self._api_key}",
            "content-type": "application/json",
            "accept": "application/json",
        }

    def _raise_response(self, response: httpx.Response) -> None:
        _raise_fal_response(response, self.name)

    def _derive_urls(
        self,
        remote_id: str,
        model: str | None = None,
    ) -> tuple[str, str, str]:
        text = remote_id.strip()
        if not text:
            return "", "", ""
        selected = model or self._default_model
        parts = selected.split("/") if selected else []
        app = "/".join(parts[:2]) if len(parts) >= 2 else (parts[0] if parts else "")
        prefix = f"{self._base_url}/{app}" if app else self._base_url
        response_url = f"{prefix}/requests/{text}"
        return (
            f"{response_url}/status",
            response_url,
            f"{response_url}/cancel",
        )

    def _result_from_job(
        self,
        job: AsyncJob,
        request: VideoRequest,
    ) -> list[VideoResult]:
        raw = job.metadata.get("videos")
        entries = _entries_from_value(raw)
        result = _mapping(job.metadata.get("result"))
        if entries and result is not None:
            return _videos_from_result({"videos": entries, **result}, request)
        if entries:
            return _videos_from_result({"videos": entries}, request)
        return [
            VideoResult(
                url=url,
                format=request.format,
                seed=request.seed,
                has_audio=request.generate_audio,
            )
            for url in job.result_urls
        ]

    async def _download(
        self,
        videos: list[VideoResult],
    ) -> list[VideoResult]:
        downloaded: list[VideoResult] = []
        for video in videos:
            if video.url is None:
                downloaded.append(video)
                continue
            data = await self._helper.download(
                video.url,
                max_bytes=self._max_download_bytes,
            )
            downloaded.append(
                video.model_copy(
                    update={
                        "data": data,
                        "url": None,
                    }
                )
            )
        return downloaded

    async def submit(self, request: VideoRequest) -> AsyncJob:
        """Submit *request* to the fal queue and return the stored job."""
        if not isinstance(request, VideoRequest):
            raise TypeError("request must be a VideoRequest")

        model = request.model or self._default_model
        if model is None:
            raise InvalidRequestError(
                "fal video request needs a model id",
                details={"provider": self.name},
            )
        model = _validate_model(model)
        payload = _build_payload(request)
        idempotency_key = _text(request.metadata.get("idempotency_key"))
        return await self._helper.submit(
            {"model": model, "payload": payload},
            idempotency_key=idempotency_key,
            metadata={"model": model},
        )

    async def poll(
        self,
        job: AsyncJob | str,
        *,
        timeout_seconds: float | None = None,
    ) -> AsyncJob:
        """Poll *job* until it finishes, reporting progress and enforcing a timeout."""
        return await self._helper.poll(job, timeout_seconds=timeout_seconds)

    def status(self, job: AsyncJob | str) -> AsyncJob:
        """Return the stored state of *job* without polling the provider."""
        return self._helper.status(job)

    async def cancel(self, job: AsyncJob | str) -> AsyncJob:
        """Cancel *job* at the provider and mark the local record cancelled."""
        return await self._helper.cancel(job)

    async def generate(self, request: VideoRequest) -> VideoResponse:
        """Generate *request* through the fal queue and return its videos."""
        job = await self.submit(request)
        job = await self.poll(
            job,
            timeout_seconds=request.timeout_seconds,
        )
        if job.state is AsyncJobState.FAILED:
            message = job.error_message or "fal video job failed"
            if _is_content_policy(message, job.error_code):
                raise ContentPolicyError(
                    message,
                    details={"provider": self.name, "job_id": job.id},
                )
            raise InvalidRequestError(
                message,
                details={"provider": self.name, "job_id": job.id},
            )
        if not job.succeeded:
            raise InvalidRequestError(
                f"fal video job ended in state {job.state.value}",
                details={"provider": self.name, "job_id": job.id},
            )

        model = _text(job.request.get("model"))
        videos = self._result_from_job(job, request)
        if self._download_results:
            videos = await self._download(videos)
        return VideoResponse(
            videos=videos,
            model=model,
            provider=self.name,
            request_id=job.remote_id or None,
            metadata={"job_id": job.id},
        )


class _FalVideoBackend:
    """The fal-specific submit, poll and cancel operations for the helper."""

    def __init__(self, provider: FalVideoProvider) -> None:
        self._provider = provider
        self._urls: dict[str, tuple[str, str, str]] = {}
        self._models: dict[str, str] = {}

    async def submit(self, request: Any) -> Any:
        provider = self._provider
        data = _mapping(request)
        if data is None:
            raise InvalidRequestError("fal submission must be a mapping")
        model = _validate_model(_text(data.get("model")))
        payload = _mapping(data.get("payload")) or {}
        response = await provider.http.post(
            provider.endpoint_for(model),
            headers=provider._headers(),
            json=dict(payload),
        )
        provider._raise_response(response)
        body = _json_body(response)
        if body is None:
            raise ValidationError(
                "fal returned a non-JSON submission response",
                details={"provider": provider.name},
            )
        remote_id = _text(body.get("request_id")) or _text(body.get("id"))
        if not remote_id:
            raise InvalidRequestError(
                "fal submission response did not contain a request id",
                details={"provider": provider.name},
            )
        derived_status, derived_response, derived_cancel = provider._derive_urls(
            remote_id,
            model,
        )
        self._models[remote_id] = model
        self._urls[remote_id] = (
            _text(body.get("status_url")) or derived_status,
            _text(body.get("response_url")) or derived_response,
            _text(body.get("cancel_url")) or derived_cancel,
        )
        return body

    def _urls_for(self, remote_id: str) -> tuple[str, str, str]:
        stored = self._urls.get(remote_id)
        if stored is not None:
            return stored
        return self._provider._derive_urls(remote_id, self._models.get(remote_id))

    async def poll(self, remote_id: str) -> AsyncJobStatus:
        provider = self._provider
        status_url, response_url, _ = self._urls_for(remote_id)
        if not status_url:
            raise InvalidRequestError(
                "fal job has no status URL",
                details={"provider": provider.name, "remote_id": remote_id},
            )
        response = await provider.http.get(status_url)
        provider._raise_response(response)
        body = _json_body(response)
        if body is None:
            raise ValidationError(
                "fal returned a non-JSON status response",
                details={"provider": provider.name},
            )
        state = _status_state(body)
        if state is AsyncJobState.FAILED:
            _raise_job_failure(body, provider.name)
        if state is AsyncJobState.CANCELLED:
            return AsyncJobStatus(
                state=state,
                progress=_progress(body),
                error_code="cancelled",
                error_message=_error_text(body) or "fal job was cancelled",
            )
        if state is not AsyncJobState.SUCCEEDED:
            return AsyncJobStatus(
                state=state,
                progress=_progress(body),
            )

        result = body
        if not _looks_like_result(result):
            if not response_url:
                raise InvalidRequestError(
                    "fal job has no response URL",
                    details={"provider": provider.name, "remote_id": remote_id},
                )
            result_response = await provider.http.get(response_url)
            provider._raise_response(result_response)
            parsed = _json_body(result_response)
            if parsed is None:
                raise ValidationError(
                    "fal returned a non-JSON result response",
                    details={"provider": provider.name},
                )
            if _has_error(parsed):
                _raise_job_failure(parsed, provider.name)
            result = parsed

        if _is_filtered_result(result):
            raise ContentPolicyError(
                "fal filtered the generated content",
                details={"provider": provider.name, "remote_id": remote_id},
            )
        entries = _video_entries(result)
        if not entries:
            _raise_job_failure(result, provider.name)
            raise ValidationError(
                "fal result contained no videos",
                details={"provider": provider.name},
            )
        urls = [url for url in (_entry_url(entry) for entry in entries) if url]
        return AsyncJobStatus(
            state=AsyncJobState.SUCCEEDED,
            progress=1.0,
            result_urls=tuple(urls),
            metadata={"videos": entries, "result": dict(result)},
        )

    async def cancel(self, remote_id: str) -> None:
        provider = self._provider
        _, _, cancel_url = self._urls_for(remote_id)
        if not cancel_url:
            return
        response = await provider.http.put(cancel_url)
        provider._raise_response(response)


FalVideo = FalVideoProvider
"""Alias for :class:`FalVideoProvider`."""

FalVideoGen = FalVideoProvider
"""Alias for callers that name the capability explicitly."""

FalVideoGenProvider = FalVideoProvider
"""Alias for callers that name the capability explicitly."""

Fal = FalVideoProvider
"""Short alias scoped to the video package."""

register_provider("fal-video", FalVideoProvider, module=__name__)
register_provider("fal_video", FalVideoProvider, module=__name__)

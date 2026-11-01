"""Replicate video generation adapter using versioned predictions.

Replicate does not accept a bare model name for an arbitrary model. A
prediction is created from a version id, usually written as
``owner/name:version`` in configuration. This adapter reads that versioned
model id from configuration (or a request override), submits a prediction to
``/v1/predictions`` and drives the prediction lifecycle through
:class:`~aivia.providers.media.async_job.AsyncMediaJobHelper`.

A first frame is forwarded through the configured prediction input, which
defaults to ``image`` because Replicate models name that input differently.

The adapter owns no routing, retry or budget policy. It uses the shared HTTP
client for transport and maps unsuccessful responses to the shared error
taxonomy. A safety filter is reported as
:class:`~aivia.core.errors.ContentPolicyError` rather than a generic invalid
request, because the router must not retry the same prompt against the same
provider.
"""

from __future__ import annotations

import base64
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
from aivia.providers.protocols.media import MediaInput, VideoFormat
from aivia.providers.protocols.video import (
    CAPABILITY_VIDEO_FIRST_FRAME,
    CAPABILITY_VIDEO_GEN,
    CAPABILITY_VIDEO_INPUT,
    CAPABILITY_VIDEO_NATIVE_AUDIO,
    VideoRequest,
    VideoResponse,
    VideoResult,
)
from aivia.providers.registry import register_provider

__all__ = [
    "DEFAULT_BASE_URL",
    "REPLICATE_API_URL",
    "REPLICATE_BASE_URL",
    "REPLICATE_DEFAULT_BASE_URL",
    "REPLICATE_VIDEO_API_URL",
    "REPLICATE_VIDEO_BASE_URL",
    "REPLICATE_VIDEO_DEFAULT_BASE_URL",
    "ReplicateVideo",
    "ReplicateVideoGen",
    "ReplicateVideoGenProvider",
    "ReplicateVideoProvider",
]

DEFAULT_BASE_URL = "https://api.replicate.com/v1"
"""The Replicate API root, without a trailing slash."""

REPLICATE_DEFAULT_BASE_URL = DEFAULT_BASE_URL
"""Alias that names the Replicate API root explicitly."""

REPLICATE_API_URL = DEFAULT_BASE_URL
"""Alias for callers that describe the root as the API URL."""

REPLICATE_BASE_URL = DEFAULT_BASE_URL
"""Alias for callers that prefer the short base URL name."""

REPLICATE_VIDEO_DEFAULT_BASE_URL = DEFAULT_BASE_URL
"""Alias that names the video adapter's default API root."""

REPLICATE_VIDEO_API_URL = DEFAULT_BASE_URL
"""Alias that names the video adapter's API URL."""

REPLICATE_VIDEO_BASE_URL = DEFAULT_BASE_URL
"""Alias that names the video adapter's base URL."""

_DEFAULT_METADATA = ProviderMeta(
    name="replicate",
    capabilities=[
        CAPABILITY_VIDEO_GEN,
        CAPABILITY_VIDEO_FIRST_FRAME,
        CAPABILITY_VIDEO_NATIVE_AUDIO,
    ],
)

DEFAULT_FIRST_FRAME_FIELD = "image"
"""The default prediction input that receives a first frame image."""

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

_RESULT_KEYS = ("output", "outputs", "videos", "video", "results", "data")

_URL_KEYS = (
    "url",
    "video",
    "video_url",
    "file",
    "file_url",
    "download_url",
    "uri",
)


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
            raise ConfigError("replicate video API key must be non-empty")
        return text
    if api_key_ref is None or not isinstance(api_key_ref, str):
        raise ConfigError(
            "replicate video provider requires an API key or api_key_ref",
            details={"field": "api_key_ref"},
        )
    reference = api_key_ref.strip()
    if not reference:
        raise ConfigError(
            "replicate video provider requires an API key or api_key_ref",
            details={"field": "api_key_ref"},
        )
    loader = secrets or SecretsLoader()
    return loader.get(reference)


def _validate_model_text(value: str) -> str:
    text = value.strip().strip("/")
    if not text:
        raise InvalidRequestError("replicate video request needs a model id")
    if "://" in text or any(character.isspace() for character in text):
        raise InvalidRequestError(
            f"invalid replicate model id: {value!r}",
            details={"field": "model"},
        )
    return text


def _validate_version(value: str) -> str:
    text = value.strip()
    if not text:
        raise InvalidRequestError(
            "replicate model id must include a version",
            details={"field": "version"},
        )
    if "://" in text or any(character.isspace() for character in text):
        raise InvalidRequestError(
            f"invalid replicate model version: {value!r}",
            details={"field": "version"},
        )
    return text


def _parse_model(value: str, version: str | None = None) -> tuple[str, str]:
    """Split a configured model id into its name and version."""
    text = _validate_model_text(value)
    explicit = _validate_version(version) if version is not None else ""
    if ":" in text:
        model, _, inline = text.rpartition(":")
        if not model:
            raise InvalidRequestError(
                f"invalid replicate model id: {value!r}",
                details={"field": "model"},
            )
        selected = explicit or _validate_version(inline)
    else:
        model = text
        selected = explicit
    if not selected:
        raise InvalidRequestError(
            "replicate model id must include a version, for example "
            "'owner/name:version'",
            details={"field": "model"},
        )
    return model, selected


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _error_text(body: Mapping[str, Any] | None) -> str:
    if body is None:
        return ""
    for key in ("error", "detail", "message", "error_message", "reason"):
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


def _json_body(response: httpx.Response) -> Mapping[str, Any] | None:
    try:
        parsed = response.json()
    except ValueError:
        return None
    return _mapping(parsed)


def _raise_replicate_response(response: httpx.Response, provider: str) -> None:
    fallback = error_for_response(response)
    body = _json_body(response)
    if fallback is None and not (body is not None and _error_text(body)):
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

    text = fallback.message if fallback is not None else "replicate returned an error"
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
    text = message or "replicate video prediction failed"
    if _is_content_policy(message, code):
        raise ContentPolicyError(text, details=details)
    raise InvalidRequestError(text, details=details)


def _media_url(media: MediaInput) -> str:
    if media.url is not None:
        return media.url
    if media.data is None:
        raise InvalidRequestError("replicate video frame needs inline data or a URL")
    media_type = media.media_type or "image/png"
    encoded = base64.b64encode(media.data).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _frame_url(media: MediaInput, field: str) -> str:
    if media.media_type is not None and not media.media_type.startswith("image/"):
        raise InvalidRequestError(
            f"replicate video {field} must be an image",
            details={"field": field, "media_type": media.media_type},
        )
    return _media_url(media)


def _build_input(
    request: VideoRequest,
    *,
    first_frame_field: str = DEFAULT_FIRST_FRAME_FIELD,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "prompt": request.prompt,
        "num_outputs": request.count,
    }
    if request.first_frame is not None:
        data[first_frame_field] = _frame_url(request.first_frame, "first_frame")
    if request.negative_prompt is not None:
        data["negative_prompt"] = request.negative_prompt
    if request.width is not None and request.height is not None:
        data["width"] = request.width
        data["height"] = request.height
    elif request.aspect_ratio is not None:
        data["aspect_ratio"] = request.aspect_ratio
    if request.duration_seconds is not None:
        data["duration"] = request.duration_seconds
    if request.fps is not None:
        data["fps"] = request.fps
    if request.seed is not None:
        data["seed"] = request.seed
    data["output_format"] = request.format.value
    if request.generate_audio:
        data["generate_audio"] = True
    if request.audio_prompt is not None:
        data["audio_prompt"] = request.audio_prompt
    return data


def _url_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in _URL_KEYS:
            if key in value:
                found = _url_value(value[key])
                if found:
                    return found
    return ""


def _entry_mappings(value: Any) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [{"url": text}] if text else []
    if isinstance(value, Mapping):
        if any(key in value for key in _URL_KEYS):
            return [value]
        entries: list[Mapping[str, Any]] = []
        for key in _RESULT_KEYS:
            if key in value:
                entries.extend(_entry_mappings(value[key]))
        if entries:
            return entries
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        collected: list[Mapping[str, Any]] = []
        for item in value:
            collected.extend(_entry_mappings(item))
        return collected
    return []


def _output_entries(body: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in _RESULT_KEYS:
        if key in body:
            entries = _entry_mappings(body[key])
            if entries:
                return entries
    if "url" in body:
        return [body]
    return []


def _entry_url(entry: Mapping[str, Any]) -> str:
    for key in _URL_KEYS:
        value = entry.get(key)
        found = _url_value(value)
        if found:
            return found
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
    for source in (entry, body, _mapping(body.get("input")) or {}):
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
    for source in (entry, body, _mapping(body.get("input")) or {}):
        value = source.get("seed")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return request.seed


def _entry_bool(
    entry: Mapping[str, Any],
    body: Mapping[str, Any],
    key: str,
) -> bool | None:
    for source in (entry, body, _mapping(body.get("input")) or {}):
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
        for key in ("thumbnail_url", "thumbnail", "poster_url"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _videos_from_result(
    body: Mapping[str, Any],
    request: VideoRequest,
) -> list[VideoResult]:
    entries = _output_entries(body)
    if not entries:
        raise ValidationError(
            "replicate returned no videos",
            details={"provider": "replicate"},
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
            "replicate returned no video URLs",
            details={"provider": "replicate"},
        )
    return videos


def _is_filtered_result(body: Mapping[str, Any]) -> bool:
    for source in (body, *_output_entries(body)):
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


def _status_state(body: Mapping[str, Any]) -> AsyncJobState:
    raw = _text(body.get("status")) or _text(body.get("state"))
    text = raw.lower().replace("-", "_")
    if text in {"starting", "queued", "pending", "waiting", "submitted"}:
        return AsyncJobState.PENDING
    if text in {"processing", "running", "active", "in_progress"}:
        return AsyncJobState.RUNNING
    if text in {"succeeded", "success", "complete", "completed", "done"}:
        return AsyncJobState.SUCCEEDED
    if text in {"failed", "error", "failure"}:
        return AsyncJobState.FAILED
    if text in {"canceled", "cancelled", "aborted"}:
        return AsyncJobState.CANCELLED
    if _output_entries(body):
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


class ReplicateVideoProvider(Provider):
    """A video provider backed by versioned Replicate predictions.

    ``api_key`` accepts an already resolved credential while ``api_key_ref``
    names a secret resolved through ``secrets``. ``model`` (or its
    ``default_model`` alias) is the versioned model id from configuration, for
    example ``owner/name:version``. ``version`` may supply the version
    separately. ``base_url`` may point at a local test server; ``transport``
    injects an ``httpx`` mock transport so the offline suite never touches the
    network.
    """

    default_metadata = _DEFAULT_METADATA

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_ref: str | None = None,
        credential_ref: str | None = None,
        secret_ref: str | None = None,
        secrets: SecretsLoader | None = None,
        base_url: str = DEFAULT_BASE_URL,
        api_url: str | None = None,
        default_model: str | None = None,
        model: str | None = None,
        version: str | None = None,
        model_version: str | None = None,
        first_frame_field: str | None = None,
        image_field: str | None = None,
        image_input_field: str | None = None,
        frame_field: str | None = None,
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
        selected_version = version if version is not None else model_version
        if selected_version is not None and not isinstance(selected_version, str):
            raise TypeError("version must be a string or None")
        selected_field = (
            first_frame_field
            or image_field
            or image_input_field
            or frame_field
            or DEFAULT_FIRST_FRAME_FIELD
        )
        if not isinstance(selected_field, str):
            raise TypeError("first_frame_field must be a string")
        selected_field = selected_field.strip()
        if not selected_field:
            raise ConfigError(
                "replicate first_frame_field must be non-empty",
                details={"field": "first_frame_field"},
            )
        self._first_frame_field = selected_field
        self._default_model_name: str | None = None
        self._default_version: str = ""
        self._default_model: str | None = None
        if selected is not None:
            if not isinstance(selected, str):
                raise TypeError("default_model must be a string or None")
            try:
                model_name, model_version = _parse_model(selected, selected_version)
            except InvalidRequestError as exc:
                raise ConfigError(str(exc), details=exc.details) from exc
            self._default_model_name = model_name
            self._default_version = model_version
            self._default_model = f"{model_name}:{model_version}"

        reference = api_key_ref or credential_ref or secret_ref
        self._api_key = _resolve_api_key(api_key, reference, secrets)
        self._base_url = _validate_base_url(api_url or base_url)
        self._download_results = download_results
        self._max_download_bytes = max_download_bytes
        if transport is not None:
            self._http = self.create_http_client(transport=transport)
        self._helper = AsyncMediaJobHelper(
            _ReplicateVideoBackend(self),
            engine=engine,
            provider=self.name,
            kind="replicate_video",
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
        """The configured Replicate API root, without a trailing slash."""
        return self._base_url

    @property
    def api_url(self) -> str:
        """Alias for :attr:`base_url` that names the API root explicitly."""
        return self._base_url

    @property
    def default_model(self) -> str | None:
        """The versioned model id used when a request does not name one."""
        return self._default_model

    @property
    def model(self) -> str | None:
        """Alias for :attr:`default_model`."""
        return self._default_model

    @property
    def version(self) -> str:
        """The version of the configured default model, or an empty string."""
        return self._default_version

    @property
    def endpoint(self) -> str:
        """The prediction submission endpoint."""
        return f"{self._base_url}/predictions"

    @property
    def first_frame_field(self) -> str:
        """The prediction input that receives a first frame image."""
        return self._first_frame_field

    def _headers(self) -> dict[str, str]:
        return {
            "authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
            "accept": "application/json",
        }

    def _raise_response(self, response: httpx.Response) -> None:
        _raise_replicate_response(response, self.name)

    def _model_and_version(self, value: str | None) -> tuple[str, str]:
        if value is None or not value.strip():
            if self._default_model_name is None:
                raise InvalidRequestError(
                    "replicate video request needs a model id",
                    details={"provider": self.name},
                )
            return self._default_model_name, self._default_version
        text = _validate_model_text(value)
        explicit = self._default_version if text == self._default_model_name else None
        return _parse_model(text, explicit)

    def _result_from_job(
        self,
        job: AsyncJob,
        request: VideoRequest,
    ) -> list[VideoResult]:
        raw = job.metadata.get("videos")
        entries = _entry_mappings(raw)
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
        """Submit *request* to Replicate and return the stored prediction job."""
        if not isinstance(request, VideoRequest):
            raise TypeError("request must be a VideoRequest")

        model_name, model_version = self._model_and_version(request.model)
        model = f"{model_name}:{model_version}"
        if request.first_frame is not None and not self.supports(
            CAPABILITY_VIDEO_FIRST_FRAME
        ):
            raise InvalidRequestError(
                "replicate video provider does not declare first frame support",
                details={
                    "provider": self.name,
                    "field": "first_frame",
                    "required_capability": CAPABILITY_VIDEO_FIRST_FRAME,
                },
            )
        if request.last_frame is not None:
            raise InvalidRequestError(
                "replicate video provider does not support a last frame",
                details={"provider": self.name, "field": "last_frame"},
            )
        if request.source_video is not None and not self.supports(
            CAPABILITY_VIDEO_INPUT
        ):
            raise InvalidRequestError(
                "replicate video provider does not declare video input support",
                details={
                    "provider": self.name,
                    "field": "source_video",
                    "required_capability": CAPABILITY_VIDEO_INPUT,
                },
            )
        first_frame_field = (
            _text(request.metadata.get("first_frame_field")) or self._first_frame_field
        )
        payload = _build_input(request, first_frame_field=first_frame_field)
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
        """Poll *job* until it finishes, enforcing the request timeout."""
        return await self._helper.poll(job, timeout_seconds=timeout_seconds)

    def status(self, job: AsyncJob | str) -> AsyncJob:
        """Return the stored state of *job* without polling the provider."""
        return self._helper.status(job)

    async def cancel(self, job: AsyncJob | str) -> AsyncJob:
        """Cancel *job* at the provider and mark the local record cancelled."""
        return await self._helper.cancel(job)

    async def generate(self, request: VideoRequest) -> VideoResponse:
        """Generate *request* through a Replicate prediction and return videos."""
        job = await self.submit(request)
        model_name, model_version = self._model_and_version(request.model)
        model = f"{model_name}:{model_version}"
        job = await self.poll(
            job,
            timeout_seconds=request.timeout_seconds,
        )
        if job.state is AsyncJobState.FAILED:
            message = job.error_message or "replicate video prediction failed"
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
                f"replicate video prediction ended in state {job.state.value}",
                details={"provider": self.name, "job_id": job.id},
            )

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


class _ReplicateVideoBackend:
    """The Replicate-specific submit, poll and cancel operations for the helper."""

    def __init__(self, provider: ReplicateVideoProvider) -> None:
        self._provider = provider
        self._urls: dict[str, tuple[str, str]] = {}

    async def submit(self, request: Any) -> Any:
        provider = self._provider
        data = _mapping(request)
        if data is None:
            raise InvalidRequestError("replicate submission must be a mapping")
        _, model_version = provider._model_and_version(_text(data.get("model")))
        payload = _mapping(data.get("payload")) or {}
        response = await provider.http.post(
            provider.endpoint,
            headers=provider._headers(),
            json={"version": model_version, "input": dict(payload)},
        )
        provider._raise_response(response)
        body = _json_body(response)
        if body is None:
            raise ValidationError(
                "replicate returned a non-JSON prediction response",
                details={"provider": provider.name},
            )
        remote_id = _text(body.get("id")) or _text(body.get("prediction_id"))
        if not remote_id:
            raise InvalidRequestError(
                "replicate prediction response did not contain an id",
                details={"provider": provider.name},
            )
        urls = _mapping(body.get("urls")) or {}
        get_url = _text(urls.get("get")) or (
            f"{provider.base_url}/predictions/{remote_id}"
        )
        cancel_url = _text(urls.get("cancel")) or (
            f"{provider.base_url}/predictions/{remote_id}/cancel"
        )
        self._urls[remote_id] = (get_url, cancel_url)
        return body

    def _urls_for(self, remote_id: str) -> tuple[str, str]:
        stored = self._urls.get(remote_id)
        if stored is not None:
            return stored
        base = self._provider.base_url
        return (
            f"{base}/predictions/{remote_id}",
            f"{base}/predictions/{remote_id}/cancel",
        )

    async def poll(self, remote_id: str) -> AsyncJobStatus:
        provider = self._provider
        get_url, _ = self._urls_for(remote_id)
        response = await provider.http.get(
            get_url,
            headers=provider._headers(),
        )
        provider._raise_response(response)
        body = _json_body(response)
        if body is None:
            raise ValidationError(
                "replicate returned a non-JSON prediction response",
                details={"provider": provider.name},
            )
        state = _status_state(body)
        if state is AsyncJobState.FAILED:
            _raise_job_failure(body, provider.name)
        if state is AsyncJobState.CANCELLED:
            return AsyncJobStatus(
                state=state,
                error_code="cancelled",
                error_message=_error_text(body) or "replicate prediction was canceled",
            )
        if state is not AsyncJobState.SUCCEEDED:
            return AsyncJobStatus(
                state=state,
                progress=_progress(body),
            )

        if _is_filtered_result(body):
            raise ContentPolicyError(
                "replicate filtered the generated content",
                details={"provider": provider.name, "remote_id": remote_id},
            )
        entries = _output_entries(body)
        if not entries:
            _raise_job_failure(body, provider.name)
            raise ValidationError(
                "replicate prediction contained no videos",
                details={"provider": provider.name},
            )
        urls = [url for url in (_entry_url(entry) for entry in entries) if url]
        return AsyncJobStatus(
            state=AsyncJobState.SUCCEEDED,
            progress=1.0,
            result_urls=tuple(urls),
            metadata={"videos": entries, "result": dict(body)},
        )

    async def cancel(self, remote_id: str) -> None:
        provider = self._provider
        _, cancel_url = self._urls_for(remote_id)
        response = await provider.http.post(
            cancel_url,
            headers=provider._headers(),
        )
        provider._raise_response(response)


ReplicateVideo = ReplicateVideoProvider
"""Alias for :class:`ReplicateVideoProvider`."""

ReplicateVideoGen = ReplicateVideoProvider
"""Alias for callers that name the capability explicitly."""

ReplicateVideoGenProvider = ReplicateVideoProvider
"""Alias for callers that name the capability explicitly."""

register_provider("replicate-video", ReplicateVideoProvider, module=__name__)
register_provider("replicate_video", ReplicateVideoProvider, module=__name__)

"""Google Veo video generation adapter using long-running operations.

Veo is exposed through the Google Generative Language API. A request is
submitted to ``models/{model}:predictLongRunning`` and returns an operation
name. The operation is polled until ``done`` is true, after which the finished
clip URLs live under the operation response.

A first frame is sent as the instance ``image``, either inline base64 bytes or
a URI when the frame already lives in storage.

The adapter authenticates with an API key supplied either directly or through a
credential reference resolved by :class:`~aivia.core.secrets.SecretsLoader`.
Model ids and the API root come from configuration; neither is hard-coded. The
adapter owns no routing, retry or budget policy and maps unsuccessful responses
to the shared error taxonomy. A safety filter is reported as
:class:`~aivia.core.errors.ContentPolicyError` so the router does not retry the
same prompt unchanged.
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
    AiviaError,
    AuthError,
    ConfigError,
    ContentPolicyError,
    InvalidRequestError,
    RateLimitError,
    RetryableError,
    TimeoutError,
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
    VideoRequest,
    VideoResponse,
    VideoResult,
)
from aivia.providers.registry import register_provider

__all__ = [
    "DEFAULT_API_VERSION",
    "DEFAULT_BASE_URL",
    "GOOGLE_API_URL",
    "GOOGLE_DEFAULT_BASE_URL",
    "GOOGLE_VEO_API_URL",
    "GOOGLE_VEO_BASE_URL",
    "VEO_API_URL",
    "VEO_DEFAULT_BASE_URL",
    "Veo",
    "VeoProvider",
    "VeoVideo",
    "VeoVideoGen",
    "VeoVideoGenProvider",
    "VeoVideoProvider",
]

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
"""The Google Generative Language API root, without a trailing slash."""

DEFAULT_API_VERSION = "v1beta"
"""The Generative Language API version used in request paths."""

GOOGLE_DEFAULT_BASE_URL = DEFAULT_BASE_URL
"""Alias that names the Google API root explicitly."""

GOOGLE_API_URL = DEFAULT_BASE_URL
"""Alias for callers that describe the root as the API URL."""

GOOGLE_VEO_BASE_URL = DEFAULT_BASE_URL
"""Alias that names the Google Veo API root explicitly."""

GOOGLE_VEO_API_URL = DEFAULT_BASE_URL
"""Alias that names the Google Veo API URL explicitly."""

VEO_DEFAULT_BASE_URL = DEFAULT_BASE_URL
"""Alias that names the Veo API root explicitly."""

VEO_API_URL = DEFAULT_BASE_URL
"""Alias that names the Veo API URL explicitly."""

_DEFAULT_METADATA = ProviderMeta(
    name="veo",
    capabilities=[CAPABILITY_VIDEO_GEN, CAPABILITY_VIDEO_FIRST_FRAME],
)

_PREDICT_SUFFIX = ":predictLongRunning"

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

_CONTENT_POLICY_MARKERS = (
    "content policy",
    "content_policy",
    "content filter",
    "content_filter",
    "safety system",
    "safety_system",
    "moderation",
    "nsfw",
    "rai",
    "filtered",
    "flagged",
    "inappropriate",
    "violates",
    "blocked",
)

_SAFETY_FINISH_REASONS = frozenset(
    {
        "SAFETY",
        "RECITATION",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_SAFETY",
    }
)

_ERROR_CLASSES: dict[str, type[AiviaError]] = {
    "UNAUTHENTICATED": AuthError,
    "PERMISSION_DENIED": AuthError,
    "INVALID_ARGUMENT": InvalidRequestError,
    "FAILED_PRECONDITION": InvalidRequestError,
    "NOT_FOUND": InvalidRequestError,
    "OUT_OF_RANGE": InvalidRequestError,
    "UNIMPLEMENTED": InvalidRequestError,
    "RESOURCE_EXHAUSTED": RateLimitError,
    "ABORTED": RetryableError,
    "CANCELLED": RetryableError,
    "DATA_LOSS": RetryableError,
    "INTERNAL": RetryableError,
    "UNAVAILABLE": RetryableError,
    "UNKNOWN": RetryableError,
    "DEADLINE_EXCEEDED": TimeoutError,
    "SAFETY": ContentPolicyError,
}

_RESULT_KEYS = (
    "videos",
    "video",
    "output",
    "outputs",
    "results",
    "data",
    "generateVideoResponse",
    "generatedSamples",
    "samples",
    "predictions",
)

_URL_KEYS = (
    "url",
    "uri",
    "video",
    "video_url",
    "videoUri",
    "gcsUri",
    "gcs_uri",
    "file_url",
    "download_url",
)

_FILTER_KEYS = (
    "raiMediaFilteredReasons",
    "rai_media_filtered_reasons",
    "raiMediaFilteredCount",
    "rai_media_filtered_count",
    "has_nsfw_concepts",
    "nsfw",
    "filtered",
    "blocked",
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
            raise ConfigError("veo API key must be non-empty")
        return text
    if api_key_ref is None or not isinstance(api_key_ref, str):
        raise ConfigError(
            "veo video provider requires an API key or api_key_ref",
            details={"field": "api_key_ref"},
        )
    reference = api_key_ref.strip()
    if not reference:
        raise ConfigError(
            "veo video provider requires an API key or api_key_ref",
            details={"field": "api_key_ref"},
        )
    loader = secrets or SecretsLoader()
    return loader.get(reference)


def _validate_model(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("model must be a string")
    text = value.strip()
    if not text:
        raise InvalidRequestError("veo video request needs a model id")
    if "://" in text or any(character.isspace() for character in text):
        raise InvalidRequestError(
            f"invalid veo model id: {value!r}",
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


def _error_status(body: Mapping[str, Any] | None) -> str:
    if body is None:
        return ""
    for key in ("status", "state"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    nested = _mapping(body.get("error"))
    return _error_status(nested)


def _is_content_policy(*values: str) -> bool:
    text = " ".join(value for value in values if value).lower()
    return any(marker in text for marker in _CONTENT_POLICY_MARKERS)


def _json_body(response: httpx.Response) -> Mapping[str, Any] | None:
    try:
        parsed = response.json()
    except ValueError:
        return None
    return _mapping(parsed)


def _raise_google_response(response: httpx.Response, provider: str) -> None:
    fallback = error_for_response(response)
    body = _json_body(response)
    if fallback is None and not (body is not None and _error_text(body)):
        return

    message = _error_text(body)
    code = _error_code(body)
    status = _error_status(body)
    details: dict[str, Any] = {
        "provider": provider,
        "http_status": response.status_code,
    }
    if code:
        details["code"] = code
    if status:
        details["status"] = status
    request_id = response.headers.get("x-goog-request-id") or response.headers.get(
        "x-request-id"
    )
    if request_id:
        details["request_id"] = request_id

    text = fallback.message if fallback is not None else "veo returned an error"
    if message:
        text = f"{text}: {message}"

    if _is_content_policy(message, code, status):
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
    status = _error_status(body)
    details: dict[str, Any] = {"provider": provider}
    if code:
        details["code"] = code
    if status:
        details["status"] = status
    text = message or "veo operation failed"
    if _is_content_policy(message, code, status):
        raise ContentPolicyError(text, details=details)
    raise InvalidRequestError(text, details=details)


def _frame_image(media: MediaInput) -> dict[str, str]:
    if media.media_type is not None and not media.media_type.startswith("image/"):
        raise InvalidRequestError(
            "veo video first_frame must be an image",
            details={"field": "first_frame", "media_type": media.media_type},
        )
    if media.data is not None:
        return {
            "bytesBase64Encoded": base64.b64encode(media.data).decode("ascii"),
            "mimeType": media.media_type or "image/png",
        }
    if media.url is not None:
        return {"gcsUri": media.url}
    raise InvalidRequestError(
        "veo video first_frame needs inline data or a URL",
        details={"field": "first_frame"},
    )


def _build_payload(request: VideoRequest) -> dict[str, Any]:
    parameters: dict[str, Any] = {"sampleCount": request.count}
    if request.aspect_ratio is not None:
        parameters["aspectRatio"] = request.aspect_ratio
    if request.negative_prompt is not None:
        parameters["negativePrompt"] = request.negative_prompt
    if request.duration_seconds is not None:
        parameters["durationSeconds"] = request.duration_seconds
    if request.seed is not None:
        parameters["seed"] = request.seed
    if request.generate_audio:
        parameters["generateAudio"] = True
    if request.width is not None and request.height is not None:
        parameters["resolution"] = (
            "1080p" if max(request.width, request.height) >= 1080 else "720p"
        )
    instance: dict[str, Any] = {"prompt": request.prompt}
    if request.first_frame is not None:
        instance["image"] = _frame_image(request.first_frame)
    return {"instances": [instance], "parameters": parameters}


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


def _video_entries(body: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in _RESULT_KEYS:
        if key in body:
            entries = _entry_mappings(body[key])
            if entries:
                return entries
    if "url" in body or "uri" in body:
        return [body]
    return []


def _entry_url(entry: Mapping[str, Any]) -> str:
    for key in _URL_KEYS:
        found = _url_value(entry.get(key))
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
    for source in (entry, body, _mapping(body.get("parameters")) or {}):
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
    for source in (entry, body, _mapping(body.get("parameters")) or {}):
        value = source.get("seed")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return request.seed


def _entry_bool(
    entry: Mapping[str, Any],
    body: Mapping[str, Any],
    key: str,
) -> bool | None:
    for source in (entry, body, _mapping(body.get("parameters")) or {}):
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
        for key in ("content_type", "mime_type", "mimeType"):
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
        for key in ("thumbnail_url", "thumbnail", "poster_url", "thumbnailUri"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _filter_reason(body: Mapping[str, Any]) -> str:
    for key in _FILTER_KEYS:
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, bool) and value:
            return key
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return f"{key}: {value}"
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                text = _text(item)
                if text:
                    return text
    nested = _mapping(body.get("response"))
    if nested is not None:
        return _filter_reason(nested)
    return ""


def _videos_from_result(
    body: Mapping[str, Any],
    request: VideoRequest,
) -> list[VideoResult]:
    entries = _video_entries(body)
    if not entries:
        raise ValidationError(
            "veo returned no videos",
            details={"provider": "veo"},
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
                metadata={"content_type": _text(entry.get("mimeType"))}
                if _text(entry.get("mimeType"))
                else {},
            )
        )
    if not videos:
        raise ValidationError(
            "veo returned no video URLs",
            details={"provider": "veo"},
        )
    return videos


def _is_done(body: Mapping[str, Any]) -> bool:
    done = body.get("done")
    if isinstance(done, bool):
        return done
    state = _text(body.get("state")) or _text(body.get("status"))
    return state.lower() in {"done", "succeeded", "success", "completed", "complete"}


def _progress(body: Mapping[str, Any]) -> float | None:
    for source in (body, _mapping(body.get("metadata")) or {}):
        for key in ("progress", "percent", "percentage", "progressPercent"):
            value = source.get(key)
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


class VeoVideoProvider(Provider):
    """A video provider backed by Google Veo long-running operations.

    ``api_key`` accepts an already resolved credential while ``api_key_ref``
    names a secret resolved through ``secrets``. ``default_model`` (or its
    ``model`` alias) supplies the Veo model id from configuration. ``base_url``
    may point at a local test server and ``transport`` injects an ``httpx`` mock
    transport so the offline suite never touches the network.
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
        api_version: str = DEFAULT_API_VERSION,
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
        if not isinstance(api_version, str) or not api_version.strip():
            raise ConfigError("veo api_version must be non-empty")

        selected = model if model is not None else default_model
        if selected is not None:
            if not isinstance(selected, str):
                raise TypeError("default_model must be a string or None")
            selected = _validate_model(selected)
        self._default_model = selected
        reference = api_key_ref or credential_ref or secret_ref
        self._api_key = _resolve_api_key(api_key, reference, secrets)
        self._base_url = _validate_base_url(api_url or base_url)
        self._api_version = api_version.strip()
        self._download_results = download_results
        self._max_download_bytes = max_download_bytes
        if transport is not None:
            self._http = self.create_http_client(transport=transport)
        self._helper = AsyncMediaJobHelper(
            _VeoBackend(self),
            engine=engine,
            provider=self.name,
            kind="veo_video",
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
        """The configured Google API root, without a trailing slash."""
        return self._base_url

    @property
    def api_url(self) -> str:
        """Alias for :attr:`base_url` that names the API root explicitly."""
        return self._base_url

    @property
    def api_version(self) -> str:
        """The Generative Language API version used in request paths."""
        return self._api_version

    @property
    def default_model(self) -> str | None:
        """The model id used when a request does not name one."""
        return self._default_model

    @property
    def model(self) -> str | None:
        """Alias for :attr:`default_model`."""
        return self._default_model

    @property
    def endpoint(self) -> str:
        """The generation endpoint for the configured default model."""
        if self._default_model is None:
            return f"{self._base_url}/{self._api_version}/models"
        return self.endpoint_for(self._default_model)

    def endpoint_for(self, model: str) -> str:
        """Return the long-running generation endpoint for *model*."""
        text = _validate_model(model)
        return f"{self._base_url}/{self._api_version}/models/{text}{_PREDICT_SUFFIX}"

    def operation_url(self, name: str) -> str:
        """Return the polling URL for an operation name or absolute URL."""
        text = name.strip()
        if text.startswith(("http://", "https://")):
            return text
        if text.startswith(f"{self._api_version}/"):
            return f"{self._base_url}/{text.lstrip('/')}"
        return f"{self._base_url}/{self._api_version}/{text.lstrip('/')}"

    def _headers(self) -> dict[str, str]:
        return {
            "x-goog-api-key": self._api_key,
            "content-type": "application/json",
            "accept": "application/json",
        }

    def _raise_response(self, response: httpx.Response) -> None:
        _raise_google_response(response, self.name)

    def _resolve_model(self, request: VideoRequest) -> str:
        model = request.model or self._default_model
        if model is None:
            raise InvalidRequestError(
                "veo video request needs a model id",
                details={"provider": self.name},
            )
        return _validate_model(model)

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
        """Submit *request* to Veo and return the stored operation job."""
        if not isinstance(request, VideoRequest):
            raise TypeError("request must be a VideoRequest")

        model = self._resolve_model(request)
        if request.first_frame is not None and not self.supports(
            CAPABILITY_VIDEO_FIRST_FRAME
        ):
            raise InvalidRequestError(
                "veo video provider does not declare first frame support",
                details={
                    "provider": self.name,
                    "field": "first_frame",
                    "required_capability": CAPABILITY_VIDEO_FIRST_FRAME,
                },
            )
        if request.last_frame is not None:
            raise InvalidRequestError(
                "veo video provider does not support a last frame",
                details={"provider": self.name, "field": "last_frame"},
            )
        if request.source_video is not None and not self.supports(
            CAPABILITY_VIDEO_INPUT
        ):
            raise InvalidRequestError(
                "veo video provider does not declare video input support",
                details={
                    "provider": self.name,
                    "field": "source_video",
                    "required_capability": CAPABILITY_VIDEO_INPUT,
                },
            )
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
        """Poll *job* until it finishes, enforcing the request timeout."""
        return await self._helper.poll(job, timeout_seconds=timeout_seconds)

    def status(self, job: AsyncJob | str) -> AsyncJob:
        """Return the stored state of *job* without polling the provider."""
        return self._helper.status(job)

    async def cancel(self, job: AsyncJob | str) -> AsyncJob:
        """Cancel *job* at the provider and mark the local record cancelled."""
        return await self._helper.cancel(job)

    async def generate(self, request: VideoRequest) -> VideoResponse:
        """Generate *request* through Veo and return its finished videos."""
        job = await self.submit(request)
        model = self._resolve_model(request)
        job = await self.poll(
            job,
            timeout_seconds=request.timeout_seconds,
        )
        if job.state is AsyncJobState.FAILED:
            message = job.error_message or "veo operation failed"
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
                f"veo operation ended in state {job.state.value}",
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


class _VeoBackend:
    """The Veo-specific submit, poll and cancel operations for the helper."""

    def __init__(self, provider: VeoVideoProvider) -> None:
        self._provider = provider

    async def submit(self, request: Any) -> Any:
        provider = self._provider
        data = _mapping(request)
        if data is None:
            raise InvalidRequestError("veo submission must be a mapping")
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
                "veo returned a non-JSON operation response",
                details={"provider": provider.name},
            )
        remote_id = (
            _text(body.get("name"))
            or _text(body.get("operation"))
            or _text(body.get("operationName"))
            or _text(body.get("id"))
        )
        if not remote_id:
            raise InvalidRequestError(
                "veo operation response did not contain a name",
                details={"provider": provider.name},
            )
        return {**body, "request_id": remote_id}

    async def poll(self, remote_id: str) -> AsyncJobStatus:
        provider = self._provider
        response = await provider.http.get(
            provider.operation_url(remote_id),
            headers=provider._headers(),
        )
        provider._raise_response(response)
        body = _json_body(response)
        if body is None:
            raise ValidationError(
                "veo returned a non-JSON operation response",
                details={"provider": provider.name},
            )
        if _error_text(body):
            _raise_job_failure(body, provider.name)
        if not _is_done(body):
            return AsyncJobStatus(
                state=AsyncJobState.RUNNING,
                progress=_progress(body),
            )

        result = _mapping(body.get("response")) or body
        reason = _filter_reason(body) or _filter_reason(result)
        if reason:
            raise ContentPolicyError(
                f"veo filtered the generated video: {reason}",
                details={"provider": provider.name, "reason": reason},
            )
        entries = _video_entries(result)
        if not entries:
            _raise_job_failure(result, provider.name)
            raise ValidationError(
                "veo operation contained no videos",
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
        response = await provider.http.post(
            provider.operation_url(remote_id) + ":cancel",
            headers=provider._headers(),
        )
        provider._raise_response(response)


Veo = VeoVideoProvider
"""Alias for :class:`VeoVideoProvider`."""

VeoVideo = VeoVideoProvider
"""Alias for :class:`VeoVideoProvider`."""

VeoVideoGen = VeoVideoProvider
"""Alias for callers that name the capability explicitly."""

VeoVideoGenProvider = VeoVideoProvider
"""Alias for callers that name the capability explicitly."""

VeoProvider = VeoVideoProvider
"""Alias for callers that prefer the shorter provider name."""

register_provider("veo", VeoVideoProvider, module=__name__)
register_provider("veo-video", VeoVideoProvider, module=__name__)
register_provider("veo_video", VeoVideoProvider, module=__name__)
register_provider("google-veo", VeoVideoProvider, module=__name__)
register_provider("google_veo", VeoVideoProvider, module=__name__)

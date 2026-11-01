"""OpenAI Images API adapter.

The adapter maps a vendor-neutral image request onto
``POST /v1/images/generations`` and maps both response shapes OpenAI returns:
an inline base64 payload (``b64_json``) or a remote URL. Inline bytes become an
:class:`~aivia.providers.protocols.image.ImageResult` with ``data`` set, while a
URL result carries ``url`` instead, so the media service never has to guess
which shape a provider prefers.

The adapter owns no routing, retry or budget policy. It uses the shared HTTP
client for transport and maps unsuccessful responses to the shared error
taxonomy. Model ids are never hard-coded here: they come from the request or
the configured default. The Images API only accepts a small set of pixel sizes,
so a requested size or aspect ratio is snapped to the closest supported size
before submission.
"""

from __future__ import annotations

import base64
import binascii
import math
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

import httpx

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
from aivia.providers.protocols.image import (
    CAPABILITY_IMAGE_GEN,
    ImageRequest,
    ImageResponse,
    ImageResult,
)
from aivia.providers.protocols.media import ImageFormat
from aivia.providers.registry import register_provider

__all__ = [
    "DEFAULT_BASE_URL",
    "OPENAI_API_URL",
    "OPENAI_BASE_URL",
    "OPENAI_DEFAULT_BASE_URL",
    "OpenAI",
    "OpenAIImage",
    "OpenAIImageProvider",
    "OpenAIImages",
    "OpenAIImagesProvider",
]

DEFAULT_BASE_URL = "https://api.openai.com"
"""The OpenAI API root, without a trailing slash."""

OPENAI_DEFAULT_BASE_URL = DEFAULT_BASE_URL
"""Alias that names the OpenAI API root explicitly."""

OPENAI_API_URL = DEFAULT_BASE_URL
"""Alias for callers that describe the root as the API URL."""

OPENAI_BASE_URL = DEFAULT_BASE_URL
"""Alias for callers that prefer the short base URL name."""

_DEFAULT_METADATA = ProviderMeta(
    name="openai-image",
    capabilities=[CAPABILITY_IMAGE_GEN],
    supported_options={
        CAPABILITY_IMAGE_GEN: [
            "prompt",
            "size",
            "aspect_ratio",
            "format",
            "quality",
            "style",
            "seed",
            "count",
            "model",
        ]
    },
)

_RESPONSE_FORMATS = frozenset({"b64_json", "url"})

_ERROR_CLASSES: dict[str, type[AiviaError]] = {
    "authentication_error": AuthError,
    "permission_error": AuthError,
    "permission_denied": AuthError,
    "invalid_api_key": AuthError,
    "invalid_request_error": InvalidRequestError,
    "invalid_request": InvalidRequestError,
    "not_found_error": InvalidRequestError,
    "request_too_large": InvalidRequestError,
    "context_length_exceeded": InvalidRequestError,
    "rate_limit_error": RateLimitError,
    "rate_limit_exceeded": RateLimitError,
    "insufficient_quota": RateLimitError,
    "api_error": RetryableError,
    "server_error": RetryableError,
    "overloaded_error": RetryableError,
    "timeout_error": TimeoutError,
    "content_policy_error": ContentPolicyError,
    "content_policy_violation": ContentPolicyError,
    "content_filter": ContentPolicyError,
    "moderation_blocked": ContentPolicyError,
    "safety_system": ContentPolicyError,
}

_CONTENT_POLICY_TYPES = frozenset(
    {
        "content_policy_error",
        "content_policy_violation",
        "content_filter",
        "moderation_blocked",
        "safety_system",
    }
)

_CONTENT_POLICY_MARKERS = (
    "content policy",
    "content_policy",
    "content filter",
    "content_filter",
    "moderation",
    "safety system",
    "safety_system",
    "nsfw",
    "flagged",
    "inappropriate",
    "violates",
    "blocked",
)

_GPT_IMAGE_SIZES = ("1024x1024", "1536x1024", "1024x1536")
_DALLE3_SIZES = ("1024x1024", "1792x1024", "1024x1792")
_DALLE2_SIZES = ("256x256", "512x512", "1024x1024")
_ALL_SIZES = tuple(dict.fromkeys((*_GPT_IMAGE_SIZES, *_DALLE3_SIZES, *_DALLE2_SIZES)))

_CONTENT_TYPE_FORMATS = {
    "image/png": ImageFormat.PNG,
    "image/jpeg": ImageFormat.JPEG,
    "image/jpg": ImageFormat.JPEG,
    "image/webp": ImageFormat.WEBP,
    "image/gif": ImageFormat.GIF,
    "image/avif": ImageFormat.AVIF,
    "image/tiff": ImageFormat.TIFF,
}

_EXTENSION_FORMATS = {
    "png": ImageFormat.PNG,
    "jpg": ImageFormat.JPEG,
    "jpeg": ImageFormat.JPEG,
    "webp": ImageFormat.WEBP,
    "gif": ImageFormat.GIF,
    "avif": ImageFormat.AVIF,
    "tif": ImageFormat.TIFF,
    "tiff": ImageFormat.TIFF,
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
            raise ConfigError("openai image API key must be non-empty")
        return text
    if api_key_ref is None or not isinstance(api_key_ref, str):
        raise ConfigError(
            "openai image provider requires an API key or api_key_ref",
            details={"field": "api_key_ref"},
        )
    reference = api_key_ref.strip()
    if not reference:
        raise ConfigError(
            "openai image provider requires an API key or api_key_ref",
            details={"field": "api_key_ref"},
        )
    loader = secrets or SecretsLoader()
    return loader.get(reference)


def _validate_response_format(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("response_format must be a string or None")
    text = value.strip().lower()
    if not text:
        return None
    if text not in _RESPONSE_FORMATS:
        raise ConfigError(
            "response_format must be 'url' or 'b64_json'",
            details={"field": "response_format", "value": value},
        )
    return text


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _error_payload(
    response: httpx.Response,
) -> tuple[str | None, str | None, str | None]:
    body = _mapping(_json_body(response))
    if body is None:
        return None, None, None
    error = _mapping(body.get("error"))
    if error is None:
        return None, None, None
    error_type = _text(error.get("type")) or None
    error_code = _text(error.get("code")) or None
    message = _text(error.get("message")) or None
    return error_type, error_code, message


def _error_class(
    status: int,
    error_type: str | None,
    error_code: str | None,
) -> type[AiviaError] | None:
    if status in {401, 403, 407}:
        return AuthError
    if status == 429:
        return RateLimitError
    if status in {408, 504}:
        return TimeoutError
    if status >= 500:
        if error_type in _CONTENT_POLICY_TYPES or error_code in _CONTENT_POLICY_TYPES:
            return ContentPolicyError
        return RetryableError
    if error_code is not None and error_code in _ERROR_CLASSES:
        return _ERROR_CLASSES[error_code]
    if error_type is not None and error_type in _ERROR_CLASSES:
        return _ERROR_CLASSES[error_type]
    return None


def _is_content_policy(*values: str) -> bool:
    text = " ".join(value for value in values if value).lower()
    return any(marker in text for marker in _CONTENT_POLICY_MARKERS)


def _json_body(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def _raise_openai_response(response: httpx.Response, provider: str) -> None:
    fallback = error_for_response(response)
    if fallback is None:
        return

    error_type, error_code, api_message = _error_payload(response)
    details: dict[str, Any] = {
        "provider": provider,
        "status": response.status_code,
    }
    if error_type is not None:
        details["type"] = error_type
    if error_code is not None:
        details["code"] = error_code
    request_id = response.headers.get("x-request-id")
    if request_id:
        details["request_id"] = request_id

    message = fallback.message
    if api_message is not None:
        message = f"{message}: {api_message}"

    error_class = _error_class(response.status_code, error_type, error_code)
    if _is_content_policy(api_message or "", error_code or ""):
        error_class = ContentPolicyError
    if error_class is RateLimitError:
        retry_after = getattr(fallback, "retry_after", None)
        raise RateLimitError(
            message,
            retry_after=retry_after,
            details=details,
        )
    if error_class is TimeoutError:
        raise TimeoutError(message, details=details)
    if error_class is not None:
        raise error_class(message, details=details)
    raise fallback


def _sizes_for_model(model: str) -> tuple[str, ...]:
    text = model.lower()
    if "dall-e-3" in text:
        return _DALLE3_SIZES
    if "dall-e-2" in text:
        return _DALLE2_SIZES
    return _GPT_IMAGE_SIZES


def _size_dimensions(value: str) -> tuple[int, int]:
    width, _, height = value.partition("x")
    return int(width), int(height)


def _parse_aspect_ratio(value: str) -> float:
    width, _, height = value.partition(":")
    return int(width) / int(height)


def _choose_size(
    model: str,
    request: ImageRequest,
) -> tuple[str, int, int] | None:
    if request.width is not None and request.height is not None:
        requested = f"{request.width}x{request.height}"
        if requested in _ALL_SIZES:
            return requested, request.width, request.height
        target_ratio = request.width / request.height
        target_pixels: int | None = request.width * request.height
    elif request.aspect_ratio is not None:
        target_ratio = _parse_aspect_ratio(request.aspect_ratio)
        target_pixels = None
    else:
        return None

    allowed = _sizes_for_model(model)

    def distance(value: str) -> tuple[float, float, int]:
        width, height = _size_dimensions(value)
        ratio_penalty = abs(math.log((width / height) / target_ratio))
        scale_penalty = (
            0.0
            if target_pixels is None
            else abs(math.log((width * height) / target_pixels))
        )
        return ratio_penalty, scale_penalty, width * height

    selected = min(allowed, key=distance)
    width, height = _size_dimensions(selected)
    return selected, width, height


def _build_payload(
    request: ImageRequest,
    model: str,
    response_format: str | None,
) -> tuple[dict[str, Any], tuple[str, int, int] | None]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": request.prompt,
        "n": request.count,
    }
    selected = _choose_size(model, request)
    if selected is not None:
        payload["size"] = selected[0]
    if response_format is not None:
        payload["response_format"] = response_format
    if request.quality is not None:
        payload["quality"] = request.quality
    if request.style is not None:
        payload["style"] = request.style
    return payload, selected


def _decode_base64(value: str) -> tuple[bytes, str | None]:
    text = value.strip()
    media_type: str | None = None
    if text.startswith("data:"):
        header, _, text = text.partition(",")
        media_type = header[5:].split(";", 1)[0].strip().lower() or None
    try:
        data = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValidationError(
            "openai image returned invalid base64 data",
            details={"provider": "openai-image"},
        ) from exc
    if not data:
        raise ValidationError(
            "openai image returned empty image data",
            details={"provider": "openai-image"},
        )
    return data, media_type


def _sniff_format(data: bytes, fallback: ImageFormat) -> ImageFormat:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ImageFormat.PNG
    if data.startswith(b"\xff\xd8\xff"):
        return ImageFormat.JPEG
    if data.startswith((b"GIF87a", b"GIF89a")):
        return ImageFormat.GIF
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ImageFormat.WEBP
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return ImageFormat.TIFF
    return fallback


def _url_format(url: str, fallback: ImageFormat) -> ImageFormat:
    extension = urlsplit(url).path.rsplit(".", 1)[-1].lower()
    return _EXTENSION_FORMATS.get(extension, fallback)


def _entries(body: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = body.get("data")
    if isinstance(raw, Mapping):
        return [raw]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        entries: list[Mapping[str, Any]] = []
        for item in raw:
            if isinstance(item, Mapping):
                entries.append(item)
            elif isinstance(item, str) and item.strip():
                entries.append({"url": item.strip()})
        return entries
    return []


def _images_from_body(
    body: Mapping[str, Any],
    request: ImageRequest,
    selected: tuple[str, int, int] | None,
) -> list[ImageResult]:
    entries = _entries(body)
    if not entries:
        raise ValidationError(
            "openai image returned no images",
            details={"provider": "openai-image"},
        )

    width = selected[1] if selected is not None else request.width
    height = selected[2] if selected is not None else request.height
    images: list[ImageResult] = []
    for entry in entries:
        revised_prompt = _text(entry.get("revised_prompt")) or None
        encoded = _text(entry.get("b64_json"))
        if encoded:
            data, media_type = _decode_base64(encoded)
            media_format = (
                _CONTENT_TYPE_FORMATS.get(media_type)
                if media_type is not None
                else None
            )
            images.append(
                ImageResult(
                    data=data,
                    format=media_format or _sniff_format(data, request.format),
                    width=width,
                    height=height,
                    seed=request.seed,
                    revised_prompt=revised_prompt,
                    metadata={"content_type": media_type}
                    if media_type is not None
                    else {},
                )
            )
            continue
        url = _text(entry.get("url"))
        if url:
            images.append(
                ImageResult(
                    url=url,
                    format=_url_format(url, request.format),
                    width=width,
                    height=height,
                    seed=request.seed,
                    revised_prompt=revised_prompt,
                )
            )
    if not images:
        raise ValidationError(
            "openai image returned no usable image data",
            details={"provider": "openai-image"},
        )
    return images


class OpenAIImageProvider(Provider):
    """An image provider backed by the OpenAI Images API.

    ``api_key`` accepts an already resolved credential while ``api_key_ref``
    names a secret resolved through ``secrets``. ``model`` (or its
    ``default_model`` alias) supplies the image model id from configuration.
    ``response_format`` optionally requests ``url`` or ``b64_json``; when it is
    omitted the API default is used and both response shapes are accepted.
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
        api_url: str | None = None,
        default_model: str | None = None,
        model: str | None = None,
        response_format: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
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
        if transport is not None and not isinstance(
            transport, httpx.AsyncBaseTransport
        ):
            raise TypeError("transport must be an httpx.AsyncBaseTransport")

        selected = model if model is not None else default_model
        if selected is not None:
            if not isinstance(selected, str):
                raise TypeError("default_model must be a string or None")
            selected = selected.strip() or None
        self._default_model = selected
        self._api_key = _resolve_api_key(api_key, api_key_ref, secrets)
        self._base_url = _validate_base_url(api_url or base_url)
        self._response_format = _validate_response_format(response_format)
        if transport is not None:
            self._http = self.create_http_client(transport=transport)

    @property
    def base_url(self) -> str:
        """The configured OpenAI API root, without a trailing slash."""
        return self._base_url

    @property
    def api_url(self) -> str:
        """Alias for :attr:`base_url` that names the API root explicitly."""
        return self._base_url

    @property
    def default_model(self) -> str | None:
        """The model id used when a request does not name one."""
        return self._default_model

    @property
    def model(self) -> str | None:
        """Alias for :attr:`default_model`."""
        return self._default_model

    @property
    def response_format(self) -> str | None:
        """The response shape requested from the API, if any."""
        return self._response_format

    @property
    def endpoint(self) -> str:
        """The image generation endpoint derived from :attr:`base_url`."""
        return f"{self._base_url}/v1/images/generations"

    def _headers(self) -> dict[str, str]:
        return {
            "authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
            "accept": "application/json",
        }

    def _raise_response(self, response: httpx.Response) -> None:
        _raise_openai_response(response, self.name)

    def _response_format_for(self, request: ImageRequest) -> str | None:
        override = request.metadata.get("response_format")
        if override is not None:
            return _validate_response_format(override)
        return self._response_format

    async def generate(self, request: ImageRequest) -> ImageResponse:
        """Generate *request* through the Images API and return its images."""
        if not isinstance(request, ImageRequest):
            raise TypeError("request must be an ImageRequest")

        model = request.model or self._default_model
        if model is None:
            raise InvalidRequestError(
                "openai image request needs a model id",
                details={"provider": self.name},
            )
        model = model.strip()
        if not model:
            raise InvalidRequestError(
                "openai image request needs a model id",
                details={"provider": self.name},
            )

        response_format = self._response_format_for(request)
        payload, selected = _build_payload(request, model, response_format)
        response = await self.http.post(
            self.endpoint,
            headers=self._headers(),
            json=payload,
            timeout=request.timeout_seconds,
        )
        self._raise_response(response)

        body = _mapping(_json_body(response))
        if body is None:
            raise ValidationError(
                "openai image returned a non-JSON response",
                details={"provider": self.name},
            )
        images = _images_from_body(body, request, selected)
        metadata: dict[str, Any] = {}
        created = body.get("created")
        if isinstance(created, int) and not isinstance(created, bool):
            metadata["created"] = created
        usage = _mapping(body.get("usage"))
        if usage is not None:
            metadata["usage"] = dict(usage)
        return ImageResponse(
            images=images,
            model=_text(body.get("model")) or model,
            provider=self.name,
            request_id=_text(body.get("id")) or response.headers.get("x-request-id"),
            metadata=metadata,
        )


OpenAI = OpenAIImageProvider
"""Alias for :class:`OpenAIImageProvider` inside the image package."""

OpenAIImage = OpenAIImageProvider
"""Alias for :class:`OpenAIImageProvider`."""

OpenAIImages = OpenAIImageProvider
"""Alias for :class:`OpenAIImageProvider`."""

OpenAIImagesProvider = OpenAIImageProvider
"""Alias for :class:`OpenAIImageProvider`."""

register_provider("openai-image", OpenAIImageProvider, module=__name__)
register_provider("openai_image", OpenAIImageProvider, module=__name__)

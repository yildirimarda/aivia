"""Google image generation adapter for Imagen and Gemini image models.

Google exposes image generation through two related APIs. Imagen models accept
a ``:predict`` request with ``instances`` and ``parameters`` and return base64
bytes in ``predictions``. Gemini image models accept the usual
``:generateContent`` request and return inline image parts. This adapter picks
the endpoint from the model id, maps a vendor-neutral
:class:`~aivia.providers.protocols.image.ImageRequest` onto the matching body
and returns the same :class:`~aivia.providers.protocols.image.ImageResponse`
for either shape.

The adapter owns no routing, retry or budget policy. It uses the shared HTTP
client for transport and maps unsuccessful responses to the shared error
taxonomy. A safety block is reported as
:class:`~aivia.core.errors.ContentPolicyError` rather than a generic invalid
request, because the router must not retry the same prompt unchanged. Model ids
are never hard-coded here: they come from the request or the configured default.
"""

from __future__ import annotations

import base64
import binascii
import math
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

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
    "DEFAULT_API_VERSION",
    "DEFAULT_BASE_URL",
    "GOOGLE_API_URL",
    "GOOGLE_BASE_URL",
    "GOOGLE_DEFAULT_BASE_URL",
    "GOOGLE_IMAGEN_API_URL",
    "Google",
    "GoogleImage",
    "GoogleImageProvider",
    "GoogleImagen",
    "GoogleImagenProvider",
    "GoogleImages",
    "GoogleImagesProvider",
    "GoogleProvider",
]

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
"""The Google Generative Language API root, without a trailing slash."""

DEFAULT_API_VERSION = "v1beta"
"""The Generative Language API version used in request paths."""

GOOGLE_DEFAULT_BASE_URL = DEFAULT_BASE_URL
"""Alias that names the Google API root explicitly."""

GOOGLE_API_URL = DEFAULT_BASE_URL
"""Alias for callers that describe the root as the API URL."""

GOOGLE_BASE_URL = DEFAULT_BASE_URL
"""Alias for callers that prefer the short base URL name."""

GOOGLE_IMAGEN_API_URL = DEFAULT_BASE_URL
"""Alias that names the Google Imagen API root explicitly."""

_DEFAULT_METADATA = ProviderMeta(
    name="google-image",
    capabilities=[CAPABILITY_IMAGE_GEN],
)

_IMAGEN_SUFFIX = ":predict"
_GEMINI_SUFFIX = ":generateContent"

_IMAGEN_ASPECT_RATIOS = ("1:1", "3:4", "4:3", "9:16", "16:9")

_CONTENT_TYPE_FORMATS = {
    "image/png": ImageFormat.PNG,
    "image/jpeg": ImageFormat.JPEG,
    "image/jpg": ImageFormat.JPEG,
    "image/webp": ImageFormat.WEBP,
    "image/gif": ImageFormat.GIF,
    "image/avif": ImageFormat.AVIF,
    "image/tiff": ImageFormat.TIFF,
}

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


class _SafetyRating(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    category: str = ""
    probability: str = ""
    blocked: bool = False


class _InlineData(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    mime_type: str = Field(default="", alias="mimeType")
    data: str = ""


class _FileData(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    mime_type: str = Field(default="", alias="mimeType")
    file_uri: str = Field(default="", alias="fileUri")


class _Part(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    text: str = ""
    inline_data: _InlineData | None = Field(default=None, alias="inlineData")
    file_data: _FileData | None = Field(default=None, alias="fileData")


class _Content(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    role: str = ""
    parts: list[_Part] = Field(default_factory=list)


class _Candidate(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    content: _Content = Field(default_factory=_Content)
    finish_reason: str | None = Field(default=None, alias="finishReason")
    safety_ratings: list[_SafetyRating] = Field(
        default_factory=list,
        alias="safetyRatings",
    )


class _PromptFeedback(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    block_reason: str | None = Field(default=None, alias="blockReason")
    safety_ratings: list[_SafetyRating] = Field(
        default_factory=list,
        alias="safetyRatings",
    )


class _Usage(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    prompt_token_count: int = Field(default=0, ge=0, alias="promptTokenCount")
    candidates_token_count: int = Field(
        default=0,
        ge=0,
        alias="candidatesTokenCount",
    )
    total_token_count: int = Field(default=0, ge=0, alias="totalTokenCount")


class _GenerateContent(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    candidates: list[_Candidate] = Field(default_factory=list)
    prompt_feedback: _PromptFeedback | None = Field(
        default=None,
        alias="promptFeedback",
    )
    usage_metadata: _Usage | None = Field(default=None, alias="usageMetadata")
    model_version: str = Field(default="", alias="modelVersion")
    response_id: str | None = Field(default=None, alias="responseId")


class _Prediction(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    bytes_base64_encoded: str = Field(default="", alias="bytesBase64Encoded")
    mime_type: str = Field(default="", alias="mimeType")
    rai_filtered_reason: str = Field(default="", alias="raiFilteredReason")
    seed: int | None = None
    width: int | None = None
    height: int | None = None


class _ImagenResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    predictions: list[_Prediction] = Field(default_factory=list)
    rai_filtered_reason: str = Field(default="", alias="raiFilteredReason")
    model_version: str = Field(default="", alias="modelVersion")
    response_id: str | None = Field(default=None, alias="responseId")


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
            raise ConfigError("google image API key must be non-empty")
        return text
    if api_key_ref is None or not isinstance(api_key_ref, str):
        raise ConfigError(
            "google image provider requires an API key or api_key_ref",
            details={"field": "api_key_ref"},
        )
    reference = api_key_ref.strip()
    if not reference:
        raise ConfigError(
            "google image provider requires an API key or api_key_ref",
            details={"field": "api_key_ref"},
        )
    loader = secrets or SecretsLoader()
    return loader.get(reference)


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _validate_model(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("model must be a string")
    text = value.strip()
    if not text:
        raise ConfigError("google image model must be non-empty")
    return text


def _is_imagen(model: str) -> bool:
    return "imagen" in model.lower()


def _error_payload(
    response: httpx.Response,
) -> tuple[str | None, str | None, tuple[str, ...]]:
    try:
        body = response.json()
    except ValueError:
        return None, None, ()
    if not isinstance(body, Mapping):
        return None, None, ()
    error = body.get("error")
    if not isinstance(error, Mapping):
        return None, None, ()
    raw_status = error.get("status")
    raw_message = error.get("message")
    error_status = _text(raw_status) or None
    message = _text(raw_message) or None
    reasons: list[str] = []
    details = error.get("details")
    if isinstance(details, list):
        for detail in details:
            if not isinstance(detail, Mapping):
                continue
            reason = detail.get("reason")
            if isinstance(reason, str) and reason.strip():
                reasons.append(reason.strip())
    return error_status, message, tuple(reasons)


def _error_class(
    status: int,
    error_status: str | None,
    reasons: tuple[str, ...],
) -> type[AiviaError] | None:
    if status in {401, 403, 407}:
        return AuthError
    if status == 429:
        return RateLimitError
    if status in {408, 504}:
        return TimeoutError
    if status >= 500:
        return RetryableError
    if any(reason in _SAFETY_FINISH_REASONS for reason in reasons):
        return ContentPolicyError
    if error_status is not None and error_status in _ERROR_CLASSES:
        return _ERROR_CLASSES[error_status]
    return None


def _is_content_policy(*values: str) -> bool:
    text = " ".join(value for value in values if value).lower()
    return any(marker in text for marker in _CONTENT_POLICY_MARKERS)


def _raise_google_response(response: httpx.Response, provider: str) -> None:
    fallback = error_for_response(response)
    if fallback is None:
        return

    error_status, api_message, reasons = _error_payload(response)
    details: dict[str, Any] = {
        "provider": provider,
        "status": response.status_code,
    }
    if error_status is not None:
        details["status"] = error_status
    if reasons:
        details["reasons"] = list(reasons)
    request_id = response.headers.get("x-goog-request-id") or response.headers.get(
        "x-request-id"
    )
    if request_id:
        details["request_id"] = request_id

    message = fallback.message
    if api_message is not None:
        message = f"{message}: {api_message}"

    error_class = _error_class(response.status_code, error_status, reasons)
    if _is_content_policy(api_message or "", *reasons):
        error_class = ContentPolicyError
    if error_class is RateLimitError:
        retry_after = getattr(fallback, "retry_after", None)
        raise RateLimitError(message, retry_after=retry_after, details=details)
    if error_class is TimeoutError:
        raise TimeoutError(message, details=details)
    if error_class is not None:
        raise error_class(message, details=details)
    raise fallback


def _parse_aspect_ratio(value: str) -> float:
    width, _, height = value.partition(":")
    return int(width) / int(height)


def _select_aspect_ratio(request: ImageRequest) -> str | None:
    if request.aspect_ratio is not None:
        requested = request.aspect_ratio
    elif request.width is not None and request.height is not None:
        requested = f"{request.width}:{request.height}"
    else:
        return None
    target = _parse_aspect_ratio(requested)
    return min(
        _IMAGEN_ASPECT_RATIOS,
        key=lambda value: abs(math.log(_parse_aspect_ratio(value) / target)),
    )


def _dimensions(request: ImageRequest) -> tuple[int | None, int | None]:
    if request.width is not None and request.height is not None:
        return request.width, request.height
    return None, None


def _imagen_payload(request: ImageRequest) -> dict[str, Any]:
    parameters: dict[str, Any] = {"sampleCount": request.count}
    aspect_ratio = _select_aspect_ratio(request)
    if aspect_ratio is not None:
        parameters["aspectRatio"] = aspect_ratio
    if request.negative_prompt is not None:
        parameters["negativePrompt"] = request.negative_prompt
    if request.seed is not None:
        parameters["seed"] = request.seed
    if request.format in {ImageFormat.PNG, ImageFormat.JPEG}:
        parameters["outputOptions"] = {
            "mimeType": f"image/{request.format.value}",
        }
    return {"instances": [{"prompt": request.prompt}], "parameters": parameters}


def _gemini_payload(request: ImageRequest) -> dict[str, Any]:
    generation_config: dict[str, Any] = {"responseModalities": ["TEXT", "IMAGE"]}
    aspect_ratio = _select_aspect_ratio(request)
    if aspect_ratio is not None:
        generation_config["imageConfig"] = {"aspectRatio": aspect_ratio}
    if request.count > 1:
        generation_config["candidateCount"] = request.count
    return {
        "contents": [{"role": "user", "parts": [{"text": request.prompt}]}],
        "generationConfig": generation_config,
    }


def _decode_base64(value: str, provider: str) -> bytes:
    text = value.strip()
    if text.startswith("data:"):
        _, _, text = text.partition(",")
    try:
        data = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValidationError(
            "google image returned invalid base64 data",
            details={"provider": provider},
        ) from exc
    if not data:
        raise ValidationError(
            "google image returned empty image data",
            details={"provider": provider},
        )
    return data


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


def _image_format(mime_type: str, data: bytes, fallback: ImageFormat) -> ImageFormat:
    found = _CONTENT_TYPE_FORMATS.get(mime_type.strip().lower())
    if found is not None:
        return found
    return _sniff_format(data, fallback)


def _blocked_rating(ratings: list[_SafetyRating]) -> str | None:
    for rating in ratings:
        if rating.blocked:
            return rating.category or "SAFETY"
    return None


def _safety_details(
    provider: str,
    reason: str,
    *,
    source: str,
    body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "provider": provider,
        "reason": reason,
        "source": source,
        "block_reason": reason,
    }
    if source == "candidate":
        details["finish_reason"] = reason
    if body is not None:
        response_id = body.get("responseId")
        if isinstance(response_id, str) and response_id.strip():
            details["request_id"] = response_id.strip()
    return details


def _images_from_imagen(
    body: Mapping[str, Any],
    request: ImageRequest,
    provider: str,
) -> list[ImageResult]:
    try:
        parsed = _ImagenResponse.model_validate(body)
    except PydanticValidationError as exc:
        raise ValidationError(
            "google image returned a malformed response",
            details={"provider": provider},
        ) from exc

    if parsed.rai_filtered_reason:
        raise ContentPolicyError(
            f"google blocked the image: {parsed.rai_filtered_reason}",
            details=_safety_details(
                provider,
                parsed.rai_filtered_reason,
                source="imagen",
                body=body,
            ),
        )
    if not parsed.predictions:
        raise ValidationError(
            "google image returned no images",
            details={"provider": provider},
        )

    width, height = _dimensions(request)
    images: list[ImageResult] = []
    for prediction in parsed.predictions:
        if prediction.rai_filtered_reason:
            raise ContentPolicyError(
                f"google blocked the image: {prediction.rai_filtered_reason}",
                details=_safety_details(
                    provider,
                    prediction.rai_filtered_reason,
                    source="imagen",
                    body=body,
                ),
            )
        encoded = prediction.bytes_base64_encoded.strip()
        if not encoded:
            continue
        data = _decode_base64(encoded, provider)
        mime_type = prediction.mime_type.strip().lower()
        images.append(
            ImageResult(
                data=data,
                format=_image_format(mime_type, data, request.format),
                width=prediction.width or width,
                height=prediction.height or height,
                seed=prediction.seed if prediction.seed is not None else request.seed,
                metadata={"content_type": mime_type} if mime_type else {},
            )
        )
    if not images:
        raise ValidationError(
            "google image returned no usable image data",
            details={"provider": provider},
        )
    return images


def _images_from_gemini(
    body: Mapping[str, Any],
    request: ImageRequest,
    provider: str,
) -> list[ImageResult]:
    try:
        parsed = _GenerateContent.model_validate(body)
    except PydanticValidationError as exc:
        raise ValidationError(
            "google image returned a malformed response",
            details={"provider": provider},
        ) from exc

    prompt_feedback = parsed.prompt_feedback
    if prompt_feedback is not None and prompt_feedback.block_reason is not None:
        reason = prompt_feedback.block_reason
        raise ContentPolicyError(
            f"google blocked the prompt: {reason}",
            details=_safety_details(
                provider,
                reason,
                source="prompt_feedback",
                body=body,
            ),
        )
    if not parsed.candidates:
        raise ValidationError(
            "google image returned no candidates",
            details={"provider": provider},
        )

    width, height = _dimensions(request)
    images: list[ImageResult] = []
    for candidate in parsed.candidates:
        if candidate.finish_reason in _SAFETY_FINISH_REASONS:
            reason = candidate.finish_reason
            raise ContentPolicyError(
                f"google blocked the response: {reason}",
                details=_safety_details(
                    provider,
                    reason,
                    source="candidate",
                    body=body,
                ),
            )
        blocked = _blocked_rating(candidate.safety_ratings)
        if blocked is not None:
            raise ContentPolicyError(
                f"google blocked the response: {blocked}",
                details=_safety_details(
                    provider,
                    blocked,
                    source="safety_rating",
                    body=body,
                ),
            )
        for part in candidate.content.parts:
            inline = part.inline_data
            if inline is not None and inline.data.strip():
                data = _decode_base64(inline.data, provider)
                mime_type = inline.mime_type.strip().lower()
                images.append(
                    ImageResult(
                        data=data,
                        format=_image_format(mime_type, data, request.format),
                        width=width,
                        height=height,
                        seed=request.seed,
                        metadata={"content_type": mime_type} if mime_type else {},
                    )
                )
                continue
            file_data = part.file_data
            if file_data is not None and file_data.file_uri.strip():
                uri = file_data.file_uri.strip()
                if urlsplit(uri).scheme.lower() in {"http", "https"}:
                    mime_type = file_data.mime_type.strip().lower()
                    images.append(
                        ImageResult(
                            url=uri,
                            format=_CONTENT_TYPE_FORMATS.get(
                                mime_type,
                                request.format,
                            ),
                            width=width,
                            height=height,
                            seed=request.seed,
                            metadata={"content_type": mime_type} if mime_type else {},
                        )
                    )
    if not images:
        raise ValidationError(
            "google image returned no usable image data",
            details={"provider": provider},
        )
    return images


class GoogleImageProvider(Provider):
    """An image provider backed by Google Imagen or Gemini image models.

    ``api_key`` accepts an already resolved credential while ``api_key_ref``
    names a secret resolved through ``secrets``. ``default_model`` (or its
    ``model`` alias) supplies the model id from configuration. A model whose id
    contains ``imagen`` uses the ``:predict`` endpoint; every other model uses
    ``:generateContent``. ``base_url`` may point at a local test server and
    ``transport`` injects an ``httpx`` mock transport so the offline suite never
    touches the network.
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
        api_version: str = DEFAULT_API_VERSION,
        default_model: str | None = None,
        model: str | None = None,
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
        if not isinstance(api_version, str) or not api_version.strip():
            raise ConfigError("google image api_version must be non-empty")

        selected = model if model is not None else default_model
        if selected is not None:
            if not isinstance(selected, str):
                raise TypeError("default_model must be a string or None")
            selected = selected.strip() or None
        self._default_model = selected
        self._api_key = _resolve_api_key(api_key, api_key_ref, secrets)
        self._base_url = _validate_base_url(api_url or base_url)
        self._api_version = api_version.strip()
        if transport is not None:
            self._http = self.create_http_client(transport=transport)

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
        """Return the generation endpoint for *model*."""
        text = _validate_model(model)
        suffix = _IMAGEN_SUFFIX if _is_imagen(text) else _GEMINI_SUFFIX
        return f"{self._base_url}/{self._api_version}/models/{text}{suffix}"

    def _headers(self) -> dict[str, str]:
        return {
            "x-goog-api-key": self._api_key,
            "content-type": "application/json",
            "accept": "application/json",
        }

    def _raise_response(self, response: httpx.Response) -> None:
        _raise_google_response(response, self.name)

    def _resolve_model(self, request: ImageRequest) -> str:
        model = request.model or self._default_model
        if model is None:
            raise InvalidRequestError(
                "google image request needs a model id",
                details={"provider": self.name},
            )
        return _validate_model(model)

    async def _generate_imagen(
        self,
        request: ImageRequest,
        model: str,
    ) -> ImageResponse:
        response = await self.http.post(
            self.endpoint_for(model),
            headers=self._headers(),
            json=_imagen_payload(request),
            timeout=request.timeout_seconds,
        )
        self._raise_response(response)
        body = _mapping(_json_body(response))
        if body is None:
            raise ValidationError(
                "google image returned a non-JSON response",
                details={"provider": self.name},
            )
        images = _images_from_imagen(body, request, self.name)
        metadata: dict[str, Any] = {}
        usage = _mapping(body.get("usageMetadata"))
        if usage is not None:
            metadata["usage"] = dict(usage)
        return ImageResponse(
            images=images,
            model=_text(body.get("modelVersion")) or model,
            provider=self.name,
            request_id=_text(body.get("responseId"))
            or response.headers.get("x-goog-request-id"),
            metadata=metadata,
        )

    async def _generate_gemini(
        self,
        request: ImageRequest,
        model: str,
    ) -> ImageResponse:
        response = await self.http.post(
            self.endpoint_for(model),
            headers=self._headers(),
            json=_gemini_payload(request),
            timeout=request.timeout_seconds,
        )
        self._raise_response(response)
        body = _mapping(_json_body(response))
        if body is None:
            raise ValidationError(
                "google image returned a non-JSON response",
                details={"provider": self.name},
            )
        images = _images_from_gemini(body, request, self.name)
        metadata: dict[str, Any] = {}
        usage = _mapping(body.get("usageMetadata"))
        if usage is not None:
            metadata["usage"] = dict(usage)
        return ImageResponse(
            images=images,
            model=_text(body.get("modelVersion")) or model,
            provider=self.name,
            request_id=_text(body.get("responseId"))
            or response.headers.get("x-goog-request-id"),
            metadata=metadata,
        )

    async def generate(self, request: ImageRequest) -> ImageResponse:
        """Generate *request* through the matching Google image endpoint."""
        if not isinstance(request, ImageRequest):
            raise TypeError("request must be an ImageRequest")

        model = self._resolve_model(request)
        if _is_imagen(model):
            return await self._generate_imagen(request, model)
        return await self._generate_gemini(request, model)


def _json_body(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


Google = GoogleImageProvider
"""Alias for :class:`GoogleImageProvider`."""

GoogleImage = GoogleImageProvider
"""Alias for :class:`GoogleImageProvider`."""

GoogleImages = GoogleImageProvider
"""Alias for :class:`GoogleImageProvider`."""

GoogleImagesProvider = GoogleImageProvider
"""Alias for :class:`GoogleImageProvider`."""

GoogleImagen = GoogleImageProvider
"""Alias for :class:`GoogleImageProvider`."""

GoogleImagenProvider = GoogleImageProvider
"""Alias for :class:`GoogleImageProvider`."""

GoogleProvider = GoogleImageProvider
"""Alias for :class:`GoogleImageProvider`."""

register_provider("google-image", GoogleImageProvider, module=__name__)
register_provider("google_image", GoogleImageProvider, module=__name__)
register_provider("google", GoogleImageProvider, module=__name__)

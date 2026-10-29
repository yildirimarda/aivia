"""OpenAI embeddings API adapter.

The adapter maps the vendor-neutral embedding request onto the OpenAI
``/v1/embeddings`` payload, including the optional reduced dimension, and maps
the reply back into one validated vector per input. The shared batching base
splits a large request into provider-sized calls, so this module only handles
one batch.

The provider owns no routing, retry or budget policy. It uses the shared HTTP
client for transport and maps unsuccessful responses to the shared error
taxonomy.
"""

from __future__ import annotations

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
from aivia.providers.embeddings.batch import BatchedEmbeddingProvider
from aivia.providers.protocols.embeddings import (
    CAPABILITY_EMBEDDINGS,
    Embedding,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)
from aivia.providers.registry import register_provider

__all__ = [
    "DEFAULT_BASE_URL",
    "OpenAIEmbeddingProvider",
    "OpenAIEmbeddings",
    "OpenAIEmbeddingsProvider",
]

DEFAULT_BASE_URL = "https://api.openai.com"
"""The OpenAI API root, without a trailing slash."""

_DEFAULT_METADATA = ProviderMeta(
    name="openai-embeddings",
    capabilities=[CAPABILITY_EMBEDDINGS],
)

_ERROR_CLASSES: dict[str, type[AiviaError]] = {
    "authentication_error": AuthError,
    "invalid_api_key": AuthError,
    "permission_error": AuthError,
    "permission_denied": AuthError,
    "invalid_request_error": InvalidRequestError,
    "invalid_request": InvalidRequestError,
    "not_found_error": InvalidRequestError,
    "rate_limit_error": RateLimitError,
    "rate_limit_exceeded": RateLimitError,
    "insufficient_quota": RateLimitError,
    "api_error": RetryableError,
    "server_error": RetryableError,
    "timeout_error": TimeoutError,
    "content_policy_error": ContentPolicyError,
    "content_policy_violation": ContentPolicyError,
}

_CONTENT_POLICY_TYPES = frozenset(
    {
        "content_policy_error",
        "content_policy_violation",
        "content_filter",
    }
)


class _Usage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class _EmbeddingData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    index: int = Field(default=0, ge=0)
    embedding: list[float] = Field(min_length=1)


class _EmbeddingsBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    model: str = ""
    data: list[_EmbeddingData] = Field(min_length=1)
    usage: _Usage = Field(default_factory=_Usage)


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
            raise ConfigError("openai embeddings API key must be non-empty")
        return text
    if api_key_ref is None or not isinstance(api_key_ref, str):
        raise ConfigError(
            "openai embeddings provider requires an API key or api_key_ref",
            details={"field": "api_key_ref"},
        )
    reference = api_key_ref.strip()
    if not reference:
        raise ConfigError(
            "openai embeddings provider requires an API key or api_key_ref",
            details={"field": "api_key_ref"},
        )
    loader = secrets or SecretsLoader()
    return loader.get(reference)


def _error_payload(
    response: httpx.Response,
) -> tuple[str | None, str | None, str | None]:
    try:
        body = response.json()
    except ValueError:
        return None, None, None
    if not isinstance(body, Mapping):
        return None, None, None
    error = body.get("error")
    if not isinstance(error, Mapping):
        return None, None, None
    raw_type = error.get("type")
    raw_code = error.get("code")
    raw_message = error.get("message")
    error_type = (
        raw_type.strip() if isinstance(raw_type, str) and raw_type.strip() else None
    )
    error_code = (
        raw_code.strip() if isinstance(raw_code, str) and raw_code.strip() else None
    )
    message = (
        raw_message.strip()
        if isinstance(raw_message, str) and raw_message.strip()
        else None
    )
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


class OpenAIEmbeddingProvider(BatchedEmbeddingProvider):
    """An embeddings provider backed by the OpenAI embeddings API.

    ``api_key`` accepts an already resolved credential while ``api_key_ref``
    names a secret resolved through ``secrets``. ``base_url`` may point at a
    local test server; ``transport`` injects an ``httpx`` mock transport so the
    offline suite never touches the network. ``dimensions`` sets the default
    vector size for models that support dimension reduction.
    """

    default_metadata = _DEFAULT_METADATA

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_ref: str | None = None,
        secrets: SecretsLoader | None = None,
        base_url: str = DEFAULT_BASE_URL,
        default_model: str | None = None,
        dimensions: int | None = None,
        batch_size: int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        name: str | None = None,
        health_timeout_seconds: float | None = None,
        close_timeout_seconds: float | None = None,
        http_config: HttpClientConfig | None = None,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            dimensions=dimensions,
            name=name,
            health_timeout_seconds=health_timeout_seconds,
            close_timeout_seconds=close_timeout_seconds,
            http_config=http_config,
        )
        if transport is not None and not isinstance(
            transport, httpx.AsyncBaseTransport
        ):
            raise TypeError("transport must be an httpx.AsyncBaseTransport")
        if default_model is not None:
            if not isinstance(default_model, str):
                raise TypeError("default_model must be a string or None")
            default_model = default_model.strip() or None
        self._api_key = _resolve_api_key(api_key, api_key_ref, secrets)
        self._base_url = _validate_base_url(base_url)
        self._default_model = default_model
        if transport is not None:
            self._http = self.create_http_client(transport=transport)

    @property
    def base_url(self) -> str:
        """The configured API root, without a trailing slash."""
        return self._base_url

    @property
    def default_model(self) -> str | None:
        """The model used when a request does not name one."""
        return self._default_model

    @property
    def endpoint(self) -> str:
        """The embeddings endpoint derived from :attr:`base_url`."""
        return f"{self._base_url}/v1/embeddings"

    def _headers(self) -> dict[str, str]:
        return {
            "authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
            "accept": "application/json",
        }

    def _build_payload(self, request: EmbeddingRequest) -> dict[str, Any]:
        model = request.model or self._default_model
        if model is None:
            raise InvalidRequestError(
                "openai embedding requests require a model id",
                details={"provider": self.name},
            )
        body: dict[str, Any] = {
            "model": model,
            "input": list(request.inputs),
            "encoding_format": "float",
        }
        dimensions = request.dimensions or self.dimensions
        if dimensions is not None:
            body["dimensions"] = dimensions
        return body

    def _raise_for_response(self, response: httpx.Response) -> None:
        fallback = error_for_response(response)
        if fallback is None:
            return

        error_type, error_code, api_message = _error_payload(response)
        details: dict[str, Any] = {
            "provider": self.name,
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

        error_class = _error_class(
            response.status_code,
            error_type,
            error_code,
        )
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

    def _to_response(
        self,
        parsed: _EmbeddingsBody,
        request: EmbeddingRequest,
        response: httpx.Response,
    ) -> EmbeddingResponse:
        try:
            embeddings = [
                Embedding(index=item.index, vector=item.embedding)
                for item in parsed.data
            ]
        except PydanticValidationError as exc:
            raise ValidationError(
                "openai embeddings returned a malformed vector",
                details={"provider": self.name},
            ) from exc

        usage = parsed.usage
        return EmbeddingResponse(
            embeddings=embeddings,
            model=parsed.model or request.model or self._default_model or "",
            provider=self.name,
            usage=EmbeddingUsage(
                input_tokens=usage.prompt_tokens or usage.total_tokens,
            ),
            request_id=parsed.id or response.headers.get("x-request-id"),
        )

    async def _embed_batch(self, request: EmbeddingRequest) -> EmbeddingResponse:
        payload = self._build_payload(request)
        response = await self.http.post(
            self.endpoint,
            headers=self._headers(),
            json=payload,
            timeout=request.timeout_seconds,
        )
        self._raise_for_response(response)

        try:
            body = response.json()
        except ValueError as exc:
            raise ValidationError(
                "openai embeddings returned a non-JSON response",
                details={"provider": self.name},
            ) from exc
        if not isinstance(body, Mapping):
            raise ValidationError(
                "openai embeddings returned a malformed response",
                details={"provider": self.name},
            )
        try:
            parsed = _EmbeddingsBody.model_validate(body)
        except PydanticValidationError as exc:
            raise ValidationError(
                "openai embeddings returned a malformed response",
                details={"provider": self.name},
            ) from exc
        return self._to_response(parsed, request, response)


OpenAIEmbeddings = OpenAIEmbeddingProvider
"""Alias for :class:`OpenAIEmbeddingProvider`."""

OpenAIEmbeddingsProvider = OpenAIEmbeddingProvider
"""Alias for :class:`OpenAIEmbeddingProvider`."""

register_provider("openai-embeddings", OpenAIEmbeddingProvider, module=__name__)

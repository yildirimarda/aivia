"""Anthropic Messages API adapter.

The adapter translates the vendor-neutral
:class:`~aivia.providers.protocols.llm.LLMRequest` into an Anthropic Messages
request and maps the reply back into an
:class:`~aivia.providers.protocols.llm.LLMResponse`. Anthropic keeps the system
prompt outside the message list, so every system message is collected into the
top-level ``system`` field while user, assistant and tool messages keep their
conversation order.

The provider owns no routing, retry or budget policy. It uses the shared HTTP
client for transport, maps unsuccessful responses to the shared error taxonomy
and reports token usage verbatim so the caller can turn it into a cost event.
Model ids are never hard-coded here: they come from the request or the
configured default.
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
from aivia.providers.base import Provider
from aivia.providers.protocols.llm import (
    CAPABILITY_LLM,
    ChatMessage,
    FinishReason,
    ImagePart,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    MessageRole,
    TextPart,
    ToolCall,
    ToolChoice,
)
from aivia.providers.registry import register_provider

__all__ = [
    "Anthropic",
    "AnthropicLLMProvider",
    "AnthropicProvider",
]

DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_API_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 1024

_DEFAULT_METADATA = ProviderMeta(
    name="anthropic",
    capabilities=[CAPABILITY_LLM],
)

_FINISH_REASONS: dict[str, FinishReason] = {
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "pause_turn": FinishReason.STOP,
    "max_tokens": FinishReason.LENGTH,
    "tool_use": FinishReason.TOOL_CALLS,
    "refusal": FinishReason.CONTENT_FILTER,
}

_ERROR_CLASSES: dict[str, type[AiviaError]] = {
    "authentication_error": AuthError,
    "permission_error": AuthError,
    "invalid_request_error": InvalidRequestError,
    "not_found_error": InvalidRequestError,
    "request_too_large": InvalidRequestError,
    "rate_limit_error": RateLimitError,
    "api_error": RetryableError,
    "overloaded_error": RetryableError,
    "timeout_error": TimeoutError,
    "content_policy_error": ContentPolicyError,
    "safety_error": ContentPolicyError,
}


class _Usage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_read_input_tokens: int = Field(default=0, ge=0)


class _ContentBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict[str, Any] = Field(default_factory=dict)


class _Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    model: str = ""
    content: list[_ContentBlock] = Field(default_factory=list)
    stop_reason: str | None = None
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


def _validate_max_tokens(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("default_max_tokens must be an integer")
    if value <= 0:
        raise ValueError("default_max_tokens must be positive")
    return value


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
            raise ConfigError("anthropic API key must be non-empty")
        return text
    if api_key_ref is None or not isinstance(api_key_ref, str):
        raise ConfigError(
            "anthropic provider requires an API key or api_key_ref",
            details={"field": "api_key_ref"},
        )
    reference = api_key_ref.strip()
    if not reference:
        raise ConfigError(
            "anthropic provider requires an API key or api_key_ref",
            details={"field": "api_key_ref"},
        )
    loader = secrets or SecretsLoader()
    return loader.get(reference)


def _text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _image_block(part: ImagePart) -> dict[str, Any]:
    if part.url is not None:
        return {
            "type": "image",
            "source": {"type": "url", "url": part.url},
        }
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": part.media_type,
            "data": part.data,
        },
    }


def _content_blocks(message: ChatMessage) -> list[dict[str, Any]]:
    if isinstance(message.content, str):
        return [_text_block(message.content)] if message.content else []
    blocks: list[dict[str, Any]] = []
    for part in message.content:
        if isinstance(part, TextPart):
            blocks.append(_text_block(part.text))
        elif isinstance(part, ImagePart):
            blocks.append(_image_block(part))
    return blocks


def _message_payload(message: ChatMessage) -> dict[str, Any] | None:
    if message.role is MessageRole.SYSTEM:
        return None
    if message.role is MessageRole.TOOL:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.text,
                }
            ],
        }
    if message.role is MessageRole.ASSISTANT and message.tool_calls:
        blocks = _content_blocks(message)
        for call in message.tool_calls:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                }
            )
        return {"role": "assistant", "content": blocks}
    if isinstance(message.content, str):
        return {"role": message.role.value, "content": message.content}
    return {
        "role": message.role.value,
        "content": _content_blocks(message),
    }


def _system_prompt(messages: list[ChatMessage]) -> str:
    return "\n\n".join(
        message.text
        for message in messages
        if message.role is MessageRole.SYSTEM and message.text
    )


def _error_payload(response: httpx.Response) -> tuple[str | None, str | None]:
    try:
        body = response.json()
    except ValueError:
        return None, None
    if not isinstance(body, Mapping):
        return None, None
    error = body.get("error")
    if not isinstance(error, Mapping):
        return None, None
    raw_type = error.get("type")
    raw_message = error.get("message")
    error_type = (
        raw_type.strip() if isinstance(raw_type, str) and raw_type.strip() else None
    )
    message = (
        raw_message.strip()
        if isinstance(raw_message, str) and raw_message.strip()
        else None
    )
    return error_type, message


class AnthropicLLMProvider(Provider):
    """An LLM provider backed by the Anthropic Messages API.

    ``api_key`` accepts an already resolved credential while ``api_key_ref``
    names a secret resolved through ``secrets`` (the environment, secrets file
    and keyring loader by default). ``base_url`` may point at a local test
    server; the Messages endpoint is appended to it. ``transport`` injects an
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
        api_version: str = DEFAULT_API_VERSION,
        default_model: str | None = None,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
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
            raise ConfigError("anthropic api_version must be non-empty")
        if default_model is not None:
            if not isinstance(default_model, str):
                raise TypeError("default_model must be a string or None")
            default_model = default_model.strip() or None

        self._api_key = _resolve_api_key(api_key, api_key_ref, secrets)
        self._base_url = _validate_base_url(base_url)
        self._api_version = api_version.strip()
        self._default_model = default_model
        self._default_max_tokens = _validate_max_tokens(default_max_tokens)
        if transport is not None:
            self._http = self.create_http_client(transport=transport)

    @property
    def base_url(self) -> str:
        """The configured API root, without a trailing slash."""
        return self._base_url

    @property
    def api_version(self) -> str:
        """The Anthropic API version sent with every request."""
        return self._api_version

    @property
    def default_model(self) -> str | None:
        """The model used when a request does not name one."""
        return self._default_model

    @property
    def endpoint(self) -> str:
        """The Messages endpoint derived from :attr:`base_url`."""
        return f"{self._base_url}/v1/messages"

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": self._api_version,
            "content-type": "application/json",
            "accept": "application/json",
        }

    def _build_payload(self, request: LLMRequest) -> dict[str, Any]:
        model = request.model or self._default_model
        if model is None:
            raise InvalidRequestError(
                "anthropic requests require a model id",
                details={"provider": self.name},
            )
        max_tokens = request.max_tokens or self._default_max_tokens

        messages: list[dict[str, Any]] = []
        for message in request.messages:
            payload = _message_payload(message)
            if payload is not None:
                messages.append(payload)
        if not messages:
            raise InvalidRequestError(
                "anthropic requests require at least one non-system message",
                details={"provider": self.name},
            )

        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        system = _system_prompt(request.messages)
        if system:
            body["system"] = system
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.stop:
            body["stop_sequences"] = list(request.stop)
        if request.tools and request.tool_choice is not ToolChoice.NONE:
            body["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters
                    or {"type": "object", "properties": {}},
                }
                for tool in request.tools
            ]
            body["tool_choice"] = {
                "type": "any" if request.tool_choice is ToolChoice.REQUIRED else "auto"
            }
        return body

    def _raise_for_response(self, response: httpx.Response) -> None:
        fallback = error_for_response(response)
        if fallback is None:
            return

        error_type, api_message = _error_payload(response)
        details: dict[str, Any] = {
            "provider": self.name,
            "status": response.status_code,
        }
        if error_type is not None:
            details["type"] = error_type
        request_id = response.headers.get("request-id")
        if request_id:
            details["request_id"] = request_id

        message = fallback.message
        if api_message is not None:
            message = f"{message}: {api_message}"

        error_class = _ERROR_CLASSES.get(error_type or "")
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

    def _parse_response(
        self,
        body: Mapping[str, Any],
        request: LLMRequest,
    ) -> LLMResponse:
        try:
            parsed = _Message.model_validate(body)
            tool_calls = [
                ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                )
                for block in parsed.content
                if block.type == "tool_use"
            ]
            content = "".join(
                block.text
                for block in parsed.content
                if block.type == "text" and block.text
            )
            finish_reason = _FINISH_REASONS.get(
                parsed.stop_reason or "",
                FinishReason.STOP,
            )
            return LLMResponse(
                content=content,
                model=parsed.model or request.model or self._default_model or "",
                provider=self.name,
                usage=LLMUsage(
                    input_tokens=parsed.usage.input_tokens,
                    output_tokens=parsed.usage.output_tokens,
                    cached_input_tokens=parsed.usage.cache_read_input_tokens,
                ),
                finish_reason=finish_reason,
                tool_calls=tool_calls,
                request_id=parsed.id,
                metadata=(
                    {"stop_reason": parsed.stop_reason}
                    if parsed.stop_reason is not None
                    else {}
                ),
            )
        except PydanticValidationError as exc:
            raise ValidationError(
                "anthropic returned a malformed response",
                details={"provider": self.name},
            ) from exc

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute *request* against the Messages API."""
        if not isinstance(request, LLMRequest):
            raise TypeError("request must be an LLMRequest")

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
                "anthropic returned a non-JSON response",
                details={"provider": self.name},
            ) from exc
        if not isinstance(body, Mapping):
            raise ValidationError(
                "anthropic returned a malformed response",
                details={"provider": self.name},
            )
        return self._parse_response(body, request)


AnthropicProvider = AnthropicLLMProvider
"""Alias for :class:`AnthropicLLMProvider`."""

Anthropic = AnthropicLLMProvider
"""Alias for :class:`AnthropicLLMProvider`."""

register_provider("anthropic", AnthropicLLMProvider, module=__name__)

"""OpenAI Chat Completions API adapter.

The adapter translates the vendor-neutral
:class:`~aivia.providers.protocols.llm.LLMRequest` into an OpenAI Chat
Completions request and maps the reply back into an
:class:`~aivia.providers.protocols.llm.LLMResponse`. Unlike the Messages API,
OpenAI keeps system messages inside the conversation, so every role is sent as
a chat message in its original order.

The provider owns no routing, retry or budget policy. It uses the shared HTTP
client for transport, maps unsuccessful responses to the shared error taxonomy
and reports token usage verbatim so the caller can turn it into a cost event.
Model ids are never hard-coded here: they come from the request or the
configured default.
"""

from __future__ import annotations

import json
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
    "OpenAI",
    "OpenAILLMProvider",
    "OpenAIProvider",
]

DEFAULT_BASE_URL = "https://api.openai.com"
DEFAULT_MAX_TOKENS = 1024

_DEFAULT_METADATA = ProviderMeta(
    name="openai",
    capabilities=[CAPABILITY_LLM],
)

_FINISH_REASONS: dict[str, FinishReason] = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
    "function_call": FinishReason.TOOL_CALLS,
    "content_filter": FinishReason.CONTENT_FILTER,
}

_TOOL_CHOICES: dict[ToolChoice, str] = {
    ToolChoice.AUTO: "auto",
    ToolChoice.NONE: "none",
    ToolChoice.REQUIRED: "required",
}

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
}

_CONTENT_POLICY_TYPES = frozenset(
    {
        "content_policy_error",
        "content_policy_violation",
        "content_filter",
    }
)


class _PromptTokensDetails(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cached_tokens: int = Field(default=0, ge=0)


class _Usage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    prompt_tokens_details: _PromptTokensDetails | None = None


class _Function(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = ""
    arguments: str = ""


class _ToolCall(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = ""
    type: str = "function"
    function: _Function = Field(default_factory=_Function)


class _Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str = "assistant"
    content: str | None = None
    tool_calls: list[_ToolCall] = Field(default_factory=list)


class _Choice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    index: int = 0
    message: _Message = Field(default_factory=_Message)
    finish_reason: str | None = None


class _ChatCompletion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    model: str = ""
    choices: list[_Choice] = Field(min_length=1)
    usage: _Usage | None = None


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
            raise ConfigError("openai API key must be non-empty")
        return text
    if api_key_ref is None or not isinstance(api_key_ref, str):
        raise ConfigError(
            "openai provider requires an API key or api_key_ref",
            details={"field": "api_key_ref"},
        )
    reference = api_key_ref.strip()
    if not reference:
        raise ConfigError(
            "openai provider requires an API key or api_key_ref",
            details={"field": "api_key_ref"},
        )
    loader = secrets or SecretsLoader()
    return loader.get(reference)


def _text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _image_block(part: ImagePart) -> dict[str, Any]:
    if part.url is not None:
        url = part.url
    else:
        url = f"data:{part.media_type};base64,{part.data}"
    return {
        "type": "image_url",
        "image_url": {"url": url, "detail": part.detail.value},
    }


def _content_parts(message: ChatMessage) -> list[dict[str, Any]]:
    if isinstance(message.content, str):
        return [_text_block(message.content)] if message.content else []
    blocks: list[dict[str, Any]] = []
    for part in message.content:
        if isinstance(part, TextPart):
            blocks.append(_text_block(part.text))
        elif isinstance(part, ImagePart):
            blocks.append(_image_block(part))
    return blocks


def _tool_call_payload(call: ToolCall) -> dict[str, Any]:
    return {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json.dumps(call.arguments),
        },
    }


def _message_payload(message: ChatMessage) -> dict[str, Any]:
    if message.role is MessageRole.TOOL:
        payload: dict[str, Any] = {
            "role": "tool",
            "content": message.text,
            "tool_call_id": message.tool_call_id,
        }
        if message.name is not None:
            payload["name"] = message.name
        return payload
    if message.role is MessageRole.ASSISTANT and message.tool_calls:
        return {
            "role": "assistant",
            "content": message.text or None,
            "tool_calls": [
                _tool_call_payload(call) for call in message.tool_calls
            ],
        }
    if isinstance(message.content, str):
        return {"role": message.role.value, "content": message.content}
    return {
        "role": message.role.value,
        "content": _content_parts(message),
    }


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
        if (
            error_type in _CONTENT_POLICY_TYPES
            or error_code in _CONTENT_POLICY_TYPES
        ):
            return ContentPolicyError
        return RetryableError
    if error_code is not None and error_code in _ERROR_CLASSES:
        return _ERROR_CLASSES[error_code]
    if error_type is not None and error_type in _ERROR_CLASSES:
        return _ERROR_CLASSES[error_type]
    return None


def _parse_tool_arguments(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if not text:
        return {}
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("tool call arguments must be a JSON object")
    return value


class OpenAILLMProvider(Provider):
    """An LLM provider backed by the OpenAI Chat Completions API.

    ``api_key`` accepts an already resolved credential while ``api_key_ref``
    names a secret resolved through ``secrets`` (the environment, secrets file
    and keyring loader by default). ``base_url`` may point at a local test
    server; the Chat Completions endpoint is appended to it. ``transport``
    injects an ``httpx`` mock transport so the offline suite never touches the
    network.
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
        if default_model is not None:
            if not isinstance(default_model, str):
                raise TypeError("default_model must be a string or None")
            default_model = default_model.strip() or None

        self._api_key = _resolve_api_key(api_key, api_key_ref, secrets)
        self._base_url = _validate_base_url(base_url)
        self._default_model = default_model
        self._default_max_tokens = _validate_max_tokens(default_max_tokens)
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
        """The Chat Completions endpoint derived from :attr:`base_url`."""
        return f"{self._base_url}/v1/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {
            "authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
            "accept": "application/json",
        }

    def _build_payload(self, request: LLMRequest) -> dict[str, Any]:
        model = request.model or self._default_model
        if model is None:
            raise InvalidRequestError(
                "openai requests require a model id",
                details={"provider": self.name},
            )
        max_tokens = request.max_tokens or self._default_max_tokens

        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                _message_payload(message) for message in request.messages
            ],
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.stop:
            body["stop"] = list(request.stop)
        if request.seed is not None:
            body["seed"] = request.seed
        if request.tools and request.tool_choice is not ToolChoice.NONE:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters
                        or {"type": "object", "properties": {}},
                    },
                }
                for tool in request.tools
            ]
            body["tool_choice"] = _TOOL_CHOICES[request.tool_choice]
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

    def _parse_response(
        self,
        body: Mapping[str, Any],
        request: LLMRequest,
    ) -> LLMResponse:
        try:
            parsed = _ChatCompletion.model_validate(body)
            choice = parsed.choices[0]
            message = choice.message
            tool_calls = [
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=_parse_tool_arguments(call.function.arguments),
                )
                for call in message.tool_calls
            ]
            usage = parsed.usage or _Usage()
            cached_tokens = (
                usage.prompt_tokens_details.cached_tokens
                if usage.prompt_tokens_details is not None
                else 0
            )
            finish_reason = _FINISH_REASONS.get(
                choice.finish_reason or "",
                FinishReason.STOP,
            )
            return LLMResponse(
                content=message.content or "",
                model=parsed.model or request.model or self._default_model or "",
                provider=self.name,
                usage=LLMUsage(
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    cached_input_tokens=cached_tokens,
                ),
                finish_reason=finish_reason,
                tool_calls=tool_calls,
                request_id=parsed.id,
                metadata=(
                    {"finish_reason": choice.finish_reason}
                    if choice.finish_reason is not None
                    else {}
                ),
            )
        except (PydanticValidationError, ValueError) as exc:
            raise ValidationError(
                "openai returned a malformed response",
                details={"provider": self.name},
            ) from exc

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute *request* against the Chat Completions API."""
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
                "openai returned a non-JSON response",
                details={"provider": self.name},
            ) from exc
        if not isinstance(body, Mapping):
            raise ValidationError(
                "openai returned a malformed response",
                details={"provider": self.name},
            )
        return self._parse_response(body, request)


OpenAIProvider = OpenAILLMProvider
"""Alias for :class:`OpenAILLMProvider`."""

OpenAI = OpenAILLMProvider
"""Alias for :class:`OpenAILLMProvider`."""

register_provider("openai", OpenAILLMProvider, module=__name__)

"""Google Gemini Generative Language API adapter.

The adapter translates the vendor-neutral
:class:`~aivia.providers.protocols.llm.LLMRequest` into a Gemini
``generateContent`` request and maps the reply back into an
:class:`~aivia.providers.protocols.llm.LLMResponse`. System messages become the
request's ``systemInstruction`` and assistant messages use Gemini's ``model``
role. Tool calls are sent as function declarations and returned as typed
:class:`~aivia.providers.protocols.llm.ToolCall` values.

Safety is part of the adapter's error contract: a prompt rejected through
``promptFeedback.blockReason``, a candidate stopped for safety or a blocked
safety rating all become a :class:`~aivia.core.errors.ContentPolicyError`
instead of a successful response. The provider owns no routing, retry or
budget policy. Model ids are never hard-coded here: they come from the request
or the configured default.
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
from aivia.providers.llm.web_search import (
    gemini_citations,
    split_web_search_tools,
)
from aivia.providers.protocols.llm import (
    CAPABILITY_LLM,
    AudioPart,
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
    "Gemini",
    "GeminiLLMProvider",
    "GeminiProvider",
]

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_API_VERSION = "v1beta"
DEFAULT_MAX_TOKENS = 1024

_DEFAULT_METADATA = ProviderMeta(
    name="gemini",
    capabilities=[CAPABILITY_LLM],
)

_FINISH_REASONS: dict[str, FinishReason] = {
    "FINISH_REASON_UNSPECIFIED": FinishReason.STOP,
    "STOP": FinishReason.STOP,
    "MAX_TOKENS": FinishReason.LENGTH,
    "MALFORMED_FUNCTION_CALL": FinishReason.ERROR,
    "OTHER": FinishReason.STOP,
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

_TOOL_MODES: dict[ToolChoice, str] = {
    ToolChoice.AUTO: "AUTO",
    ToolChoice.NONE: "NONE",
    ToolChoice.REQUIRED: "ANY",
}

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


class _FunctionCall(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str = ""
    name: str = ""
    args: dict[str, Any] = Field(default_factory=dict)


class _Part(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    text: str = ""
    function_call: _FunctionCall | None = Field(default=None, alias="functionCall")


class _Content(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    role: str = ""
    parts: list[_Part] = Field(default_factory=list)


class _GroundingWeb(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    uri: str = ""
    title: str = ""


class _GroundingChunk(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    web: _GroundingWeb | None = None


class _GroundingSegment(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    text: str = ""


class _GroundingSupport(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    segment: _GroundingSegment = Field(default_factory=_GroundingSegment)
    grounding_chunk_indices: list[int] = Field(
        default_factory=list,
        alias="groundingChunkIndices",
    )


class _GroundingMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    grounding_chunks: list[_GroundingChunk] = Field(
        default_factory=list,
        alias="groundingChunks",
    )
    grounding_supports: list[_GroundingSupport] = Field(
        default_factory=list,
        alias="groundingSupports",
    )
    web_search_queries: list[str] = Field(
        default_factory=list,
        alias="webSearchQueries",
    )


class _Candidate(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    content: _Content = Field(default_factory=_Content)
    finish_reason: str | None = Field(default=None, alias="finishReason")
    safety_ratings: list[_SafetyRating] = Field(
        default_factory=list,
        alias="safetyRatings",
    )
    grounding_metadata: _GroundingMetadata | None = Field(
        default=None,
        alias="groundingMetadata",
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
    cached_content_token_count: int = Field(
        default=0,
        ge=0,
        alias="cachedContentTokenCount",
    )
    thoughts_token_count: int = Field(default=0, ge=0, alias="thoughtsTokenCount")
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
            raise ConfigError("gemini API key must be non-empty")
        return text
    if api_key_ref is None or not isinstance(api_key_ref, str):
        raise ConfigError(
            "gemini provider requires an API key or api_key_ref",
            details={"field": "api_key_ref"},
        )
    reference = api_key_ref.strip()
    if not reference:
        raise ConfigError(
            "gemini provider requires an API key or api_key_ref",
            details={"field": "api_key_ref"},
        )
    loader = secrets or SecretsLoader()
    return loader.get(reference)


def _text_part(text: str) -> dict[str, Any]:
    return {"text": text}


def _image_part(part: ImagePart) -> dict[str, Any]:
    if part.url is not None:
        file_data: dict[str, Any] = {"fileUri": part.url}
        if part.media_type is not None:
            file_data["mimeType"] = part.media_type
        return {"fileData": file_data}
    return {
        "inlineData": {
            "mimeType": part.media_type,
            "data": part.data,
        }
    }


def _audio_part(part: AudioPart) -> dict[str, Any]:
    if part.url is not None:
        return {"fileData": {"fileUri": part.url}}
    return {
        "inlineData": {
            "mimeType": part.media_type,
            "data": part.data,
        }
    }


def _content_parts(message: ChatMessage) -> list[dict[str, Any]]:
    if isinstance(message.content, str):
        return [_text_part(message.content)] if message.content else []
    parts: list[dict[str, Any]] = []
    for part in message.content:
        if isinstance(part, TextPart):
            parts.append(_text_part(part.text))
        elif isinstance(part, ImagePart):
            parts.append(_image_part(part))
        elif isinstance(part, AudioPart):
            parts.append(_audio_part(part))
    return parts


def _tool_call_part(call: ToolCall) -> dict[str, Any]:
    function_call: dict[str, Any] = {
        "name": call.name,
        "args": call.arguments,
    }
    if call.id:
        function_call["id"] = call.id
    return {"functionCall": function_call}


def _tool_response_value(message: ChatMessage) -> dict[str, Any]:
    text = message.text.strip()
    if text:
        try:
            value = json.loads(text)
        except ValueError:
            value = None
        if isinstance(value, dict):
            return value
        return {"content": message.text}
    return {}


def _message_payload(message: ChatMessage) -> dict[str, Any] | None:
    if message.role is MessageRole.SYSTEM:
        return None
    if message.role is MessageRole.TOOL:
        return {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": message.name or message.tool_call_id or "tool",
                        "response": _tool_response_value(message),
                    }
                }
            ],
        }
    role = "model" if message.role is MessageRole.ASSISTANT else message.role.value
    parts = _content_parts(message)
    if message.role is MessageRole.ASSISTANT and message.tool_calls:
        parts.extend(_tool_call_part(call) for call in message.tool_calls)
    return {"role": role, "parts": parts}


def _system_prompt(messages: list[ChatMessage]) -> str:
    return "\n\n".join(
        message.text
        for message in messages
        if message.role is MessageRole.SYSTEM and message.text
    )


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
    error_status = (
        raw_status.strip()
        if isinstance(raw_status, str) and raw_status.strip()
        else None
    )
    message = (
        raw_message.strip()
        if isinstance(raw_message, str) and raw_message.strip()
        else None
    )
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


def _blocked_rating(ratings: list[_SafetyRating]) -> str | None:
    for rating in ratings:
        if rating.blocked:
            return rating.category or "SAFETY"
    return None


class GeminiLLMProvider(Provider):
    """An LLM provider backed by the Gemini Generative Language API.

    ``api_key`` accepts an already resolved credential while ``api_key_ref``
    names a secret resolved through ``secrets`` (the environment, secrets file
    and keyring loader by default). ``base_url`` may point at a local test
    server; the model ``generateContent`` endpoint is appended to it.
    ``transport`` injects an ``httpx`` mock transport so the offline suite
    never touches the network.
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
            raise ConfigError("gemini api_version must be non-empty")
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
        """The Generative Language API version used in the request path."""
        return self._api_version

    @property
    def default_model(self) -> str | None:
        """The model used when a request does not name one."""
        return self._default_model

    @property
    def endpoint(self) -> str:
        """The ``generateContent`` endpoint for the default model."""
        if self._default_model is None:
            return f"{self._base_url}/{self._api_version}/models"
        return self.endpoint_for(self._default_model)

    def endpoint_for(self, model: str) -> str:
        """Return the ``generateContent`` endpoint for *model*."""
        return f"{self._base_url}/{self._api_version}/models/{model}:generateContent"

    def _headers(self) -> dict[str, str]:
        return {
            "x-goog-api-key": self._api_key,
            "content-type": "application/json",
            "accept": "application/json",
        }

    def _resolve_model(self, request: LLMRequest) -> str:
        model = request.model or self._default_model
        if model is None:
            raise InvalidRequestError(
                "gemini requests require a model id",
                details={"provider": self.name},
            )
        return model

    def _build_payload(
        self,
        request: LLMRequest,
        model: str,
    ) -> dict[str, Any]:
        max_tokens = request.max_tokens or self._default_max_tokens

        contents: list[dict[str, Any]] = []
        for message in request.messages:
            payload = _message_payload(message)
            if payload is not None:
                contents.append(payload)
        if not contents:
            raise InvalidRequestError(
                "gemini requests require at least one non-system message",
                details={"provider": self.name},
            )

        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
        generation_config = body["generationConfig"]
        system = _system_prompt(request.messages)
        if system:
            body["systemInstruction"] = {"parts": [_text_part(system)]}
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        if request.top_p is not None:
            generation_config["topP"] = request.top_p
        if request.stop:
            generation_config["stopSequences"] = list(request.stop)
        if request.seed is not None:
            generation_config["seed"] = request.seed
        function_tools, native_search = split_web_search_tools(request.tools)
        if request.tools and request.tool_choice is not ToolChoice.NONE:
            tools_payload: list[dict[str, Any]] = []
            if native_search:
                tools_payload.append({"googleSearch": {}})
            if function_tools:
                tools_payload.append(
                    {
                        "functionDeclarations": [
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.parameters
                                or {"type": "object", "properties": {}},
                            }
                            for tool in function_tools
                        ]
                    }
                )
                body["toolConfig"] = {
                    "functionCallingConfig": {
                        "mode": _TOOL_MODES[request.tool_choice],
                    }
                }
            body["tools"] = tools_payload
        return body

    def _raise_for_response(self, response: httpx.Response) -> None:
        fallback = error_for_response(response)
        if fallback is None:
            return

        error_status, api_message, reasons = _error_payload(response)
        details: dict[str, Any] = {
            "provider": self.name,
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
            parsed = _GenerateContent.model_validate(body)
            prompt_feedback = parsed.prompt_feedback
            if prompt_feedback is not None and prompt_feedback.block_reason is not None:
                reason = prompt_feedback.block_reason
                raise ContentPolicyError(
                    f"gemini blocked the prompt: {reason}",
                    details=_safety_details(
                        self.name,
                        reason,
                        source="prompt_feedback",
                        body=body,
                    ),
                )
            if not parsed.candidates:
                raise ValueError("gemini returned no candidates")

            candidate = parsed.candidates[0]
            if candidate.finish_reason in _SAFETY_FINISH_REASONS:
                reason = candidate.finish_reason
                raise ContentPolicyError(
                    f"gemini blocked the response: {reason}",
                    details=_safety_details(
                        self.name,
                        reason,
                        source="candidate",
                        body=body,
                    ),
                )
            blocked = _blocked_rating(candidate.safety_ratings)
            if blocked is not None:
                raise ContentPolicyError(
                    f"gemini blocked the response: {blocked}",
                    details=_safety_details(
                        self.name,
                        blocked,
                        source="safety_rating",
                        body=body,
                    ),
                )

            tool_calls = [
                ToolCall(
                    id=part.function_call.id,
                    name=part.function_call.name,
                    arguments=part.function_call.args,
                )
                for part in candidate.content.parts
                if part.function_call is not None
            ]
            content = "".join(
                part.text for part in candidate.content.parts if part.text
            )
            usage = parsed.usage_metadata or _Usage()
            finish_reason = _FINISH_REASONS.get(
                candidate.finish_reason or "",
                FinishReason.STOP,
            )
            metadata: dict[str, Any] = {}
            if candidate.finish_reason is not None:
                metadata["finish_reason"] = candidate.finish_reason
            if candidate.grounding_metadata is not None:
                citations = gemini_citations(
                    candidate.grounding_metadata.model_dump(by_alias=True)
                )
                if citations:
                    metadata["citations"] = citations
            return LLMResponse(
                content=content,
                model=(
                    parsed.model_version or request.model or self._default_model or ""
                ),
                provider=self.name,
                usage=LLMUsage(
                    input_tokens=usage.prompt_token_count,
                    output_tokens=(
                        usage.candidates_token_count + usage.thoughts_token_count
                    ),
                    cached_input_tokens=usage.cached_content_token_count,
                    cached_input_included_in_input=True,
                ),
                finish_reason=finish_reason,
                tool_calls=tool_calls,
                request_id=parsed.response_id,
                metadata=metadata,
            )
        except ContentPolicyError:
            raise
        except (PydanticValidationError, ValueError) as exc:
            raise ValidationError(
                "gemini returned a malformed response",
                details={"provider": self.name},
            ) from exc

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute *request* against the Generative Language API."""
        if not isinstance(request, LLMRequest):
            raise TypeError("request must be an LLMRequest")

        model = self._resolve_model(request)
        payload = self._build_payload(request, model)
        response = await self.http.post(
            self.endpoint_for(model),
            headers=self._headers(),
            json=payload,
            timeout=request.timeout_seconds,
        )
        self._raise_for_response(response)

        try:
            body = response.json()
        except ValueError as exc:
            raise ValidationError(
                "gemini returned a non-JSON response",
                details={"provider": self.name},
            ) from exc
        if not isinstance(body, Mapping):
            raise ValidationError(
                "gemini returned a malformed response",
                details={"provider": self.name},
            )
        return self._parse_response(body, request)


GeminiProvider = GeminiLLMProvider
"""Alias for :class:`GeminiLLMProvider`."""

Gemini = GeminiLLMProvider
"""Alias for :class:`GeminiLLMProvider`."""

register_provider("gemini", GeminiLLMProvider, module=__name__)

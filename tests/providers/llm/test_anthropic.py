"""Anthropic LLM provider: messages, usage reporting and mapped errors."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from aivia.core.errors import (
    AiviaError,
    AuthError,
    ConfigError,
    ContentPolicyError,
    InvalidRequestError,
    RateLimitError,
    RetryableError,
    ValidationError,
)
from aivia.core.errors import (
    TimeoutError as AiviaTimeoutError,
)
from aivia.core.secrets import SecretsLoader
from aivia.providers import provider_registry
from aivia.providers.llm import (
    Anthropic,
    AnthropicLLMProvider,
    AnthropicProvider,
)
from aivia.providers.protocols import (
    CAPABILITY_LLM,
    ChatMessage,
    FinishReason,
    LLMRequest,
    MessageRole,
    TextPart,
    ToolChoice,
    ToolDefinition,
)

API_KEY = "test-key"
MODEL = "claude-test"


def _message_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [{"type": "text", "text": "Hello"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 4},
    }
    body.update(overrides)
    return body


def _request(**overrides: Any) -> LLMRequest:
    values: dict[str, Any] = {
        "messages": [
            ChatMessage(role=MessageRole.USER, content="Hello"),
        ],
        "model": MODEL,
    }
    values.update(overrides)
    return LLMRequest(**values)


def _provider(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> AnthropicLLMProvider:
    return AnthropicLLMProvider(
        api_key=API_KEY,
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _capture(
    body: dict[str, Any] | None = None,
    status: int = 200,
) -> tuple[list[httpx.Request], Callable[[httpx.Request], httpx.Response]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, json=body or _message_body())

    return requests, handler


def test_metadata_and_registration_describe_the_provider() -> None:
    provider = _provider(lambda request: httpx.Response(200, json=_message_body()))

    assert provider.name == "anthropic"
    assert provider.capabilities == (CAPABILITY_LLM,)
    assert provider.supports(CAPABILITY_LLM)
    assert provider.base_url == "https://api.anthropic.com"
    assert provider.endpoint == "https://api.anthropic.com/v1/messages"
    assert provider.api_version == "2023-06-01"
    assert AnthropicProvider is AnthropicLLMProvider
    assert Anthropic is AnthropicLLMProvider
    assert provider_registry.resolve("anthropic") is AnthropicLLMProvider


def test_sends_system_prompt_outside_the_message_list() -> None:
    requests, handler = _capture()
    provider = _provider(handler)

    asyncio.run(
        provider.complete(
            _request(
                messages=[
                    ChatMessage(role=MessageRole.SYSTEM, content="Be precise"),
                    ChatMessage(role=MessageRole.SYSTEM, content="Answer briefly"),
                    ChatMessage(role=MessageRole.USER, content="Explain tides"),
                ],
                max_tokens=64,
                temperature=0.2,
                top_p=0.9,
                stop=["END"],
            )
        )
    )

    sent = json.loads(requests[0].content)
    assert sent["system"] == "Be precise\n\nAnswer briefly"
    assert sent["messages"] == [{"role": "user", "content": "Explain tides"}]
    assert all(message["role"] != "system" for message in sent["messages"])
    assert sent["model"] == MODEL
    assert sent["max_tokens"] == 64
    assert sent["temperature"] == 0.2
    assert sent["top_p"] == 0.9
    assert sent["stop_sequences"] == ["END"]


def test_sends_the_api_key_and_version_headers() -> None:
    requests, handler = _capture()
    provider = _provider(handler)

    asyncio.run(provider.complete(_request()))

    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == API_KEY
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert request.headers["content-type"] == "application/json"


def test_maps_usage_content_and_finish_reason_from_the_response() -> None:
    body = _message_body(
        content=[
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "world"},
        ],
        usage={
            "input_tokens": 12,
            "output_tokens": 5,
            "cache_read_input_tokens": 3,
        },
    )
    provider = _provider(lambda request: httpx.Response(200, json=body))

    response = asyncio.run(provider.complete(_request()))

    assert response.content == "Hello world"
    assert response.text == "Hello world"
    assert response.model == MODEL
    assert response.provider == "anthropic"
    assert response.request_id == "msg_test"
    assert response.finish_reason is FinishReason.STOP
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 5
    assert response.usage.cached_input_tokens == 3
    assert response.usage.cached_input_included_in_input is False
    assert response.usage.total_input_tokens == 15
    assert response.usage.total_tokens == 17
    assert response.metadata == {"stop_reason": "end_turn"}
    assert response.has_tool_calls is False


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [
        ("end_turn", FinishReason.STOP),
        ("stop_sequence", FinishReason.STOP),
        ("max_tokens", FinishReason.LENGTH),
        ("tool_use", FinishReason.TOOL_CALLS),
        ("refusal", FinishReason.CONTENT_FILTER),
    ],
)
def test_maps_stop_reasons_to_finish_reasons(
    stop_reason: str,
    expected: FinishReason,
) -> None:
    body = _message_body(stop_reason=stop_reason)
    provider = _provider(lambda request: httpx.Response(200, json=body))

    response = asyncio.run(provider.complete(_request()))

    assert response.finish_reason is expected


def test_parses_tool_use_blocks_into_tool_calls() -> None:
    body = _message_body(
        content=[
            {
                "type": "tool_use",
                "id": "tool-1",
                "name": "lookup",
                "input": {"query": "tides"},
            }
        ],
        stop_reason="tool_use",
    )
    provider = _provider(lambda request: httpx.Response(200, json=body))

    response = asyncio.run(provider.complete(_request()))

    assert response.content == ""
    assert response.has_tool_calls is True
    assert response.finish_reason is FinishReason.TOOL_CALLS
    assert response.tool_calls[0].id == "tool-1"
    assert response.tool_calls[0].name == "lookup"
    assert response.tool_calls[0].arguments == {"query": "tides"}


def test_sends_tools_and_maps_required_choice_to_any() -> None:
    requests, handler = _capture()
    provider = _provider(handler)

    asyncio.run(
        provider.complete(
            _request(
                tools=[
                    ToolDefinition(
                        name="lookup",
                        description="Look something up",
                        parameters={
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                        },
                    )
                ],
                tool_choice=ToolChoice.REQUIRED,
            )
        )
    )

    sent = json.loads(requests[0].content)
    assert sent["tools"] == [
        {
            "name": "lookup",
            "description": "Look something up",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        }
    ]
    assert sent["tool_choice"] == {"type": "any"}


def test_uses_the_configured_default_model_and_token_limit() -> None:
    requests, handler = _capture()
    provider = _provider(handler, default_model=MODEL, default_max_tokens=256)

    asyncio.run(
        provider.complete(
            LLMRequest(
                messages=[ChatMessage(role=MessageRole.USER, content="Hi")],
            )
        )
    )

    sent = json.loads(requests[0].content)
    assert sent["model"] == MODEL
    assert sent["max_tokens"] == 256


def test_a_request_model_and_token_limit_override_the_defaults() -> None:
    requests, handler = _capture()
    provider = _provider(handler, default_model=MODEL, default_max_tokens=256)

    asyncio.run(provider.complete(_request(model="claude-other", max_tokens=32)))

    sent = json.loads(requests[0].content)
    assert sent["model"] == "claude-other"
    assert sent["max_tokens"] == 32


def test_a_request_without_a_model_is_rejected_before_sending() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("the provider must not send a request without a model")

    provider = _provider(handler)

    with pytest.raises(InvalidRequestError):
        asyncio.run(
            provider.complete(
                LLMRequest(messages=[ChatMessage(role=MessageRole.USER, content="Hi")])
            )
        )


def test_missing_api_key_is_a_config_error() -> None:
    with pytest.raises(ConfigError):
        AnthropicLLMProvider()


def test_api_key_reference_is_resolved_through_the_secrets_loader() -> None:
    requests, handler = _capture()
    loader = SecretsLoader(environ={"AIVA_ANTHROPIC_API_KEY": API_KEY})
    provider = AnthropicLLMProvider(
        api_key_ref="ANTHROPIC_API_KEY",
        secrets=loader,
        transport=httpx.MockTransport(handler),
    )

    asyncio.run(provider.complete(_request()))

    assert requests[0].headers["x-api-key"] == API_KEY


def test_a_custom_base_url_is_used_for_the_messages_endpoint() -> None:
    requests, handler = _capture()
    provider = _provider(handler, base_url="http://127.0.0.1:9999/api/")

    asyncio.run(provider.complete(_request()))

    assert str(requests[0].url) == "http://127.0.0.1:9999/api/v1/messages"


@pytest.mark.parametrize(
    ("status", "error_type", "expected"),
    [
        (401, "authentication_error", AuthError),
        (403, "permission_error", AuthError),
        (400, "invalid_request_error", InvalidRequestError),
        (404, "not_found_error", InvalidRequestError),
        (413, "request_too_large", InvalidRequestError),
        (429, "rate_limit_error", RateLimitError),
        (500, "api_error", RetryableError),
        (529, "overloaded_error", RetryableError),
        (400, "content_policy_error", ContentPolicyError),
    ],
)
def test_maps_anthropic_errors_to_the_shared_taxonomy(
    status: int,
    error_type: str,
    expected: type[AiviaError],
) -> None:
    body = {
        "type": "error",
        "error": {"type": error_type, "message": "request failed"},
    }
    provider = _provider(lambda request: httpx.Response(status, json=body))

    with pytest.raises(expected) as raised:
        asyncio.run(provider.complete(_request()))

    error = raised.value
    assert error.details["status"] == status
    assert error.details["type"] == error_type


def test_rate_limit_errors_keep_the_retry_after_delay() -> None:
    body = {
        "type": "error",
        "error": {"type": "rate_limit_error", "message": "slow down"},
    }
    provider = _provider(
        lambda request: httpx.Response(
            429,
            json=body,
            headers={"retry-after": "2.5"},
        )
    )

    with pytest.raises(RateLimitError) as raised:
        asyncio.run(provider.complete(_request()))

    assert raised.value.retry_after == 2.5


def test_an_error_without_a_json_body_is_mapped_by_status() -> None:
    provider = _provider(lambda request: httpx.Response(401, text="unauthorized"))

    with pytest.raises(AuthError):
        asyncio.run(provider.complete(_request()))


def test_a_timeout_is_mapped_to_the_shared_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    provider = _provider(handler)

    with pytest.raises(AiviaTimeoutError):
        asyncio.run(provider.complete(_request()))


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"content": "not-a-list"}),
        httpx.Response(200, json={"usage": {"input_tokens": -1}}),
        httpx.Response(200, text="not json"),
    ],
)
def test_malformed_success_responses_are_validation_errors(
    response: httpx.Response,
) -> None:
    provider = _provider(lambda request: response)

    with pytest.raises(ValidationError):
        asyncio.run(provider.complete(_request()))


def test_text_parts_are_sent_as_content_blocks() -> None:
    requests, handler = _capture()
    provider = _provider(handler)

    asyncio.run(
        provider.complete(
            _request(
                messages=[
                    ChatMessage(
                        role=MessageRole.USER,
                        content=[
                            TextPart(text="first"),
                            TextPart(text="second"),
                        ],
                    )
                ]
            )
        )
    )

    sent = json.loads(requests[0].content)
    assert sent["messages"][0]["content"] == [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
    ]

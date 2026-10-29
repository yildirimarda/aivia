"""OpenAI LLM provider: chat messages, usage reporting and mapped errors."""

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
    OpenAI,
    OpenAILLMProvider,
    OpenAIProvider,
)
from aivia.providers.protocols import (
    CAPABILITY_LLM,
    ChatMessage,
    FinishReason,
    ImageDetail,
    ImagePart,
    LLMRequest,
    MessageRole,
    TextPart,
    ToolCall,
    ToolChoice,
    ToolDefinition,
)

API_KEY = "test-key"
MODEL = "gpt-test"


def _completion_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "chatcmpl_test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
        },
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
) -> OpenAILLMProvider:
    return OpenAILLMProvider(
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
        return httpx.Response(status, json=body or _completion_body())

    return requests, handler


def test_metadata_and_registration_describe_the_provider() -> None:
    provider = _provider(
        lambda request: httpx.Response(200, json=_completion_body())
    )

    assert provider.name == "openai"
    assert provider.capabilities == (CAPABILITY_LLM,)
    assert provider.supports(CAPABILITY_LLM)
    assert provider.base_url == "https://api.openai.com"
    assert provider.endpoint == "https://api.openai.com/v1/chat/completions"
    assert OpenAIProvider is OpenAILLMProvider
    assert OpenAI is OpenAILLMProvider
    assert provider_registry.resolve("openai") is OpenAILLMProvider


def test_sends_system_messages_in_the_message_list() -> None:
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
                seed=7,
            )
        )
    )

    sent = json.loads(requests[0].content)
    assert sent["messages"] == [
        {"role": "system", "content": "Be precise"},
        {"role": "system", "content": "Answer briefly"},
        {"role": "user", "content": "Explain tides"},
    ]
    assert sent["model"] == MODEL
    assert sent["max_tokens"] == 64
    assert sent["temperature"] == 0.2
    assert sent["top_p"] == 0.9
    assert sent["stop"] == ["END"]
    assert sent["seed"] == 7


def test_sends_the_api_key_header() -> None:
    requests, handler = _capture()
    provider = _provider(handler)

    asyncio.run(provider.complete(_request()))

    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {API_KEY}"
    assert request.headers["content-type"] == "application/json"


def test_maps_usage_content_and_finish_reason_from_the_response() -> None:
    body = _completion_body(
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello world"},
                "finish_reason": "stop",
            }
        ],
        usage={
            "prompt_tokens": 12,
            "completion_tokens": 5,
            "total_tokens": 17,
            "prompt_tokens_details": {"cached_tokens": 3},
        },
    )
    provider = _provider(lambda request: httpx.Response(200, json=body))

    response = asyncio.run(provider.complete(_request()))

    assert response.content == "Hello world"
    assert response.text == "Hello world"
    assert response.model == MODEL
    assert response.provider == "openai"
    assert response.request_id == "chatcmpl_test"
    assert response.finish_reason is FinishReason.STOP
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 5
    assert response.usage.cached_input_tokens == 3
    assert response.usage.total_tokens == 17
    assert response.metadata == {"finish_reason": "stop"}
    assert response.has_tool_calls is False


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [
        ("stop", FinishReason.STOP),
        ("length", FinishReason.LENGTH),
        ("tool_calls", FinishReason.TOOL_CALLS),
        ("function_call", FinishReason.TOOL_CALLS),
        ("content_filter", FinishReason.CONTENT_FILTER),
    ],
)
def test_maps_finish_reasons_to_finish_reasons(
    finish_reason: str,
    expected: FinishReason,
) -> None:
    body = _completion_body(
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": finish_reason,
            }
        ]
    )
    provider = _provider(lambda request: httpx.Response(200, json=body))

    response = asyncio.run(provider.complete(_request()))

    assert response.finish_reason is expected


def test_parses_tool_calls_into_tool_calls() -> None:
    body = _completion_body(
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"query": "tides"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    )
    provider = _provider(lambda request: httpx.Response(200, json=body))

    response = asyncio.run(provider.complete(_request()))

    assert response.content == ""
    assert response.has_tool_calls is True
    assert response.finish_reason is FinishReason.TOOL_CALLS
    assert response.tool_calls[0].id == "call-1"
    assert response.tool_calls[0].name == "lookup"
    assert response.tool_calls[0].arguments == {"query": "tides"}


def test_sends_tools_and_maps_required_choice() -> None:
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
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look something up",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        }
    ]
    assert sent["tool_choice"] == "required"


def test_sends_assistant_tool_calls_and_tool_results() -> None:
    requests, handler = _capture()
    provider = _provider(handler)

    asyncio.run(
        provider.complete(
            _request(
                messages=[
                    ChatMessage(role=MessageRole.USER, content="Look it up"),
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="call-1",
                                name="lookup",
                                arguments={"query": "tides"},
                            )
                        ],
                    ),
                    ChatMessage(
                        role=MessageRole.TOOL,
                        content="The tide is high",
                        tool_call_id="call-1",
                    ),
                ]
            )
        )
    )

    sent = json.loads(requests[0].content)
    assistant = sent["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] is None
    assert assistant["tool_calls"] == [
        {
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "lookup",
                "arguments": '{"query": "tides"}',
            },
        }
    ]
    assert sent["messages"][2] == {
        "role": "tool",
        "content": "The tide is high",
        "tool_call_id": "call-1",
    }


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

    asyncio.run(provider.complete(_request(model="gpt-other", max_tokens=32)))

    sent = json.loads(requests[0].content)
    assert sent["model"] == "gpt-other"
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
        OpenAILLMProvider()


def test_api_key_reference_is_resolved_through_the_secrets_loader() -> None:
    requests, handler = _capture()
    loader = SecretsLoader(environ={"AIVA_OPENAI_API_KEY": API_KEY})
    provider = OpenAILLMProvider(
        api_key_ref="OPENAI_API_KEY",
        secrets=loader,
        transport=httpx.MockTransport(handler),
    )

    asyncio.run(provider.complete(_request()))

    assert requests[0].headers["authorization"] == f"Bearer {API_KEY}"


def test_a_custom_base_url_is_used_for_the_chat_endpoint() -> None:
    requests, handler = _capture()
    provider = _provider(handler, base_url="http://127.0.0.1:9999/api/")

    asyncio.run(provider.complete(_request()))

    assert str(requests[0].url) == "http://127.0.0.1:9999/api/v1/chat/completions"


@pytest.mark.parametrize(
    ("status", "error_type", "expected"),
    [
        (401, "authentication_error", AuthError),
        (403, "permission_error", AuthError),
        (400, "invalid_request_error", InvalidRequestError),
        (404, "not_found_error", InvalidRequestError),
        (413, "request_too_large", InvalidRequestError),
        (429, "rate_limit_error", RateLimitError),
        (500, "server_error", RetryableError),
        (503, "api_error", RetryableError),
        (400, "content_policy_error", ContentPolicyError),
    ],
)
def test_maps_openai_errors_to_the_shared_taxonomy(
    status: int,
    error_type: str,
    expected: type[AiviaError],
) -> None:
    body = {
        "error": {
            "message": "request failed",
            "type": error_type,
            "param": None,
            "code": None,
        }
    }
    provider = _provider(lambda request: httpx.Response(status, json=body))

    with pytest.raises(expected) as raised:
        asyncio.run(provider.complete(_request()))

    error = raised.value
    assert error.details["status"] == status
    assert error.details["type"] == error_type


def test_maps_a_content_policy_code_to_a_content_policy_error() -> None:
    body = {
        "error": {
            "message": "blocked",
            "type": "invalid_request_error",
            "code": "content_policy_violation",
        }
    }
    provider = _provider(lambda request: httpx.Response(400, json=body))

    with pytest.raises(ContentPolicyError) as raised:
        asyncio.run(provider.complete(_request()))

    assert raised.value.details["code"] == "content_policy_violation"


def test_an_invalid_api_key_is_still_an_auth_error() -> None:
    body = {
        "error": {
            "message": "bad key",
            "type": "invalid_request_error",
            "code": "invalid_api_key",
        }
    }
    provider = _provider(lambda request: httpx.Response(401, json=body))

    with pytest.raises(AuthError):
        asyncio.run(provider.complete(_request()))


def test_rate_limit_errors_keep_the_retry_after_delay() -> None:
    body = {
        "error": {
            "message": "slow down",
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded",
        }
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
        httpx.Response(200, json={"choices": []}),
        httpx.Response(
            200,
            json=_completion_body(
                usage={"prompt_tokens": -1, "completion_tokens": 4}
            ),
        ),
        httpx.Response(
            200,
            json=_completion_body(
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": "not json",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            ),
        ),
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


def test_image_parts_are_sent_as_image_urls() -> None:
    requests, handler = _capture()
    provider = _provider(handler)

    asyncio.run(
        provider.complete(
            _request(
                messages=[
                    ChatMessage(
                        role=MessageRole.USER,
                        content=[
                            TextPart(text="Describe this image"),
                            ImagePart(
                                url="https://example.com/image.png",
                                detail=ImageDetail.HIGH,
                            ),
                            ImagePart(
                                data="aGVsbG8=",
                                media_type="image/png",
                            ),
                        ],
                    )
                ]
            )
        )
    )

    sent = json.loads(requests[0].content)
    assert sent["messages"][0]["content"] == [
        {"type": "text", "text": "Describe this image"},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/image.png",
                "detail": "high",
            },
        },
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64,aGVsbG8=",
                "detail": "auto",
            },
        },
    ]

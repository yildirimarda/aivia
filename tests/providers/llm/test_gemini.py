"""Gemini LLM provider: content mapping, usage and safety block handling."""

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
    Gemini,
    GeminiLLMProvider,
    GeminiProvider,
)
from aivia.providers.protocols import (
    CAPABILITY_LLM,
    ChatMessage,
    FinishReason,
    ImagePart,
    LLMRequest,
    MessageRole,
    TextPart,
    ToolCall,
    ToolChoice,
    ToolDefinition,
)

API_KEY = "test-key"
MODEL = "gemini-test"


def _response_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "Hello"}],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 4,
            "totalTokenCount": 14,
        },
        "modelVersion": MODEL,
        "responseId": "resp_test",
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
) -> GeminiLLMProvider:
    return GeminiLLMProvider(
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
        return httpx.Response(status, json=body or _response_body())

    return requests, handler


def test_metadata_and_registration_describe_the_provider() -> None:
    provider = _provider(
        lambda request: httpx.Response(200, json=_response_body()),
        default_model=MODEL,
    )

    assert provider.name == "gemini"
    assert provider.capabilities == (CAPABILITY_LLM,)
    assert provider.supports(CAPABILITY_LLM)
    assert provider.base_url == "https://generativelanguage.googleapis.com"
    assert provider.api_version == "v1beta"
    assert provider.endpoint == (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{MODEL}:generateContent"
    )
    assert GeminiProvider is GeminiLLMProvider
    assert Gemini is GeminiLLMProvider
    assert provider_registry.resolve("gemini") is GeminiLLMProvider


def test_sends_system_instruction_and_maps_roles() -> None:
    requests, handler = _capture()
    provider = _provider(handler)

    asyncio.run(
        provider.complete(
            _request(
                messages=[
                    ChatMessage(role=MessageRole.SYSTEM, content="Be precise"),
                    ChatMessage(role=MessageRole.SYSTEM, content="Answer briefly"),
                    ChatMessage(role=MessageRole.USER, content="Explain tides"),
                    ChatMessage(role=MessageRole.ASSISTANT, content="Tides move"),
                    ChatMessage(role=MessageRole.USER, content="Why?"),
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
    assert sent["systemInstruction"] == {
        "parts": [{"text": "Be precise\n\nAnswer briefly"}]
    }
    assert sent["contents"] == [
        {"role": "user", "parts": [{"text": "Explain tides"}]},
        {"role": "model", "parts": [{"text": "Tides move"}]},
        {"role": "user", "parts": [{"text": "Why?"}]},
    ]
    assert all(message["role"] != "system" for message in sent["contents"])
    assert sent["generationConfig"] == {
        "maxOutputTokens": 64,
        "temperature": 0.2,
        "topP": 0.9,
        "stopSequences": ["END"],
        "seed": 7,
    }


def test_sends_the_api_key_header_and_model_endpoint() -> None:
    requests, handler = _capture()
    provider = _provider(handler)

    asyncio.run(provider.complete(_request()))

    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{MODEL}:generateContent"
    )
    assert request.headers["x-goog-api-key"] == API_KEY
    assert request.headers["content-type"] == "application/json"


def test_maps_usage_content_and_finish_reason_from_the_response() -> None:
    body = _response_body(
        candidates=[
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "Hello "}, {"text": "world"}],
                },
                "finishReason": "STOP",
            }
        ],
        usageMetadata={
            "promptTokenCount": 12,
            "candidatesTokenCount": 5,
            "cachedContentTokenCount": 3,
            "thoughtsTokenCount": 2,
            "totalTokenCount": 19,
        },
    )
    provider = _provider(lambda request: httpx.Response(200, json=body))

    response = asyncio.run(provider.complete(_request()))

    assert response.content == "Hello world"
    assert response.text == "Hello world"
    assert response.model == MODEL
    assert response.provider == "gemini"
    assert response.request_id == "resp_test"
    assert response.finish_reason is FinishReason.STOP
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 7
    assert response.usage.cached_input_tokens == 3
    assert response.usage.total_tokens == 19
    assert response.metadata == {"finish_reason": "STOP"}
    assert response.has_tool_calls is False


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [
        ("STOP", FinishReason.STOP),
        ("MAX_TOKENS", FinishReason.LENGTH),
        ("MALFORMED_FUNCTION_CALL", FinishReason.ERROR),
    ],
)
def test_maps_finish_reasons(
    finish_reason: str,
    expected: FinishReason,
) -> None:
    body = _response_body(
        candidates=[
            {
                "content": {"role": "model", "parts": [{"text": "Hello"}]},
                "finishReason": finish_reason,
            }
        ]
    )
    provider = _provider(lambda request: httpx.Response(200, json=body))

    response = asyncio.run(provider.complete(_request()))

    assert response.finish_reason is expected


@pytest.mark.parametrize(
    "finish_reason",
    ["SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"],
)
def test_candidate_safety_blocks_are_content_policy_errors(
    finish_reason: str,
) -> None:
    body = _response_body(
        candidates=[
            {
                "content": {"role": "model", "parts": [{"text": "partial"}]},
                "finishReason": finish_reason,
            }
        ]
    )
    provider = _provider(lambda request: httpx.Response(200, json=body))

    with pytest.raises(ContentPolicyError) as raised:
        asyncio.run(provider.complete(_request()))

    error = raised.value
    assert error.details["provider"] == "gemini"
    assert error.details["reason"] == finish_reason
    assert error.details["finish_reason"] == finish_reason
    assert error.details["source"] == "candidate"
    assert error.details["request_id"] == "resp_test"


@pytest.mark.parametrize(
    "block_reason",
    ["SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "OTHER"],
)
def test_a_blocked_prompt_is_a_content_policy_error(block_reason: str) -> None:
    body = {
        "promptFeedback": {"blockReason": block_reason},
        "responseId": "resp_blocked",
    }
    provider = _provider(lambda request: httpx.Response(200, json=body))

    with pytest.raises(ContentPolicyError) as raised:
        asyncio.run(provider.complete(_request()))

    error = raised.value
    assert error.details["provider"] == "gemini"
    assert error.details["reason"] == block_reason
    assert error.details["block_reason"] == block_reason
    assert error.details["source"] == "prompt_feedback"
    assert error.details["request_id"] == "resp_blocked"


def test_a_blocked_safety_rating_is_a_content_policy_error() -> None:
    body = _response_body(
        candidates=[
            {
                "content": {"role": "model", "parts": [{"text": "Hello"}]},
                "finishReason": "STOP",
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "probability": "HIGH",
                        "blocked": True,
                    }
                ],
            }
        ]
    )
    provider = _provider(lambda request: httpx.Response(200, json=body))

    with pytest.raises(ContentPolicyError) as raised:
        asyncio.run(provider.complete(_request()))

    error = raised.value
    assert error.details["reason"] == "HARM_CATEGORY_HARASSMENT"
    assert error.details["source"] == "safety_rating"


def test_parses_function_calls_into_tool_calls() -> None:
    body = _response_body(
        candidates=[
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "id": "call-1",
                                "name": "lookup",
                                "args": {"query": "tides"},
                            }
                        }
                    ],
                },
                "finishReason": "STOP",
            }
        ]
    )
    provider = _provider(lambda request: httpx.Response(200, json=body))

    response = asyncio.run(provider.complete(_request()))

    assert response.content == ""
    assert response.has_tool_calls is True
    assert response.tool_calls[0].id == "call-1"
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
            "functionDeclarations": [
                {
                    "name": "lookup",
                    "description": "Look something up",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                }
            ]
        }
    ]
    assert sent["toolConfig"] == {"functionCallingConfig": {"mode": "ANY"}}


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
                        content='{"height": 3}',
                        name="lookup",
                        tool_call_id="call-1",
                    ),
                ]
            )
        )
    )

    sent = json.loads(requests[0].content)
    assert sent["contents"][1] == {
        "role": "model",
        "parts": [
            {
                "functionCall": {
                    "name": "lookup",
                    "args": {"query": "tides"},
                    "id": "call-1",
                }
            }
        ],
    }
    assert sent["contents"][2] == {
        "role": "user",
        "parts": [
            {
                "functionResponse": {
                    "name": "lookup",
                    "response": {"height": 3},
                }
            }
        ],
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
    assert sent["generationConfig"]["maxOutputTokens"] == 256
    assert f"/models/{MODEL}:generateContent" in str(requests[0].url)


def test_a_request_model_and_token_limit_override_the_defaults() -> None:
    requests, handler = _capture()
    provider = _provider(handler, default_model=MODEL, default_max_tokens=256)

    asyncio.run(provider.complete(_request(model="gemini-other", max_tokens=32)))

    sent = json.loads(requests[0].content)
    assert sent["generationConfig"]["maxOutputTokens"] == 32
    assert "/models/gemini-other:generateContent" in str(requests[0].url)


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
        GeminiLLMProvider()


def test_api_key_reference_is_resolved_through_the_secrets_loader() -> None:
    requests, handler = _capture()
    loader = SecretsLoader(environ={"AIVA_GEMINI_API_KEY": API_KEY})
    provider = GeminiLLMProvider(
        api_key_ref="GEMINI_API_KEY",
        secrets=loader,
        transport=httpx.MockTransport(handler),
    )

    asyncio.run(provider.complete(_request()))

    assert requests[0].headers["x-goog-api-key"] == API_KEY


def test_a_custom_base_url_is_used_for_the_generate_endpoint() -> None:
    requests, handler = _capture()
    provider = _provider(handler, base_url="http://127.0.0.1:9999/api/")

    asyncio.run(provider.complete(_request()))

    assert str(requests[0].url) == (
        f"http://127.0.0.1:9999/api/v1beta/models/{MODEL}:generateContent"
    )


@pytest.mark.parametrize(
    ("status", "error_status", "expected"),
    [
        (401, "UNAUTHENTICATED", AuthError),
        (403, "PERMISSION_DENIED", AuthError),
        (400, "INVALID_ARGUMENT", InvalidRequestError),
        (404, "NOT_FOUND", InvalidRequestError),
        (429, "RESOURCE_EXHAUSTED", RateLimitError),
        (500, "INTERNAL", RetryableError),
        (503, "UNAVAILABLE", RetryableError),
        (504, "DEADLINE_EXCEEDED", AiviaTimeoutError),
        (400, "SAFETY", ContentPolicyError),
    ],
)
def test_maps_gemini_errors_to_the_shared_taxonomy(
    status: int,
    error_status: str,
    expected: type[AiviaError],
) -> None:
    body = {
        "error": {
            "code": status,
            "message": "request failed",
            "status": error_status,
        }
    }
    provider = _provider(lambda request: httpx.Response(status, json=body))

    with pytest.raises(expected) as raised:
        asyncio.run(provider.complete(_request()))

    error = raised.value
    assert error.details["provider"] == "gemini"
    assert error.details["status"] == error_status


def test_a_safety_reason_in_error_details_is_a_content_policy_error() -> None:
    body = {
        "error": {
            "code": 400,
            "message": "blocked",
            "status": "FAILED_PRECONDITION",
            "details": [{"reason": "SAFETY"}],
        }
    }
    provider = _provider(lambda request: httpx.Response(400, json=body))

    with pytest.raises(ContentPolicyError):
        asyncio.run(provider.complete(_request()))


def test_rate_limit_errors_keep_the_retry_after_delay() -> None:
    body = {
        "error": {
            "code": 429,
            "message": "slow down",
            "status": "RESOURCE_EXHAUSTED",
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
        httpx.Response(200, json={"candidates": []}),
        httpx.Response(200, json=_response_body(candidates="not-a-list")),
        httpx.Response(
            200,
            json=_response_body(
                usageMetadata={"promptTokenCount": -1},
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


def test_text_parts_are_sent_as_parts() -> None:
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
    assert sent["contents"][0]["parts"] == [
        {"text": "first"},
        {"text": "second"},
    ]


def test_image_parts_are_sent_as_inline_or_file_data() -> None:
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
                                media_type="image/png",
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
    assert sent["contents"][0]["parts"] == [
        {"text": "Describe this image"},
        {
            "fileData": {
                "fileUri": "https://example.com/image.png",
                "mimeType": "image/png",
            }
        },
        {
            "inlineData": {
                "mimeType": "image/png",
                "data": "aGVsbG8=",
            }
        },
    ]

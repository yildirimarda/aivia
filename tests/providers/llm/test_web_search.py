"""Native web search mapping for every LLM adapter.

The vendor-neutral search provider asks an LLM for results through a
``web_search`` tool. Each adapter must translate that marker into its vendor's
native search option and normalize the vendor's citations back into the flat
shape the search provider consumes. Everything runs offline through an
``httpx.MockTransport``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from aivia.providers.llm import (
    AnthropicLLMProvider,
    GeminiLLMProvider,
    OpenAICompatibleLLMProvider,
    OpenAILLMProvider,
)
from aivia.providers.protocols import (
    ChatMessage,
    LLMRequest,
    MessageRole,
    ToolChoice,
    ToolDefinition,
)
from aivia.providers.protocols.search import SearchRequest
from aivia.providers.search.llm_search import WEB_SEARCH_TOOL, LLMSearchProvider

API_KEY = "test-key"
MODEL = "test-model"
QUERY = "why do tides rise"

LOOKUP_TOOL = ToolDefinition(
    name="lookup",
    description="Look something up",
    parameters={"type": "object", "properties": {"query": {"type": "string"}}},
)


def _request(
    *,
    tools: list[ToolDefinition] | None = None,
    tool_choice: ToolChoice = ToolChoice.AUTO,
) -> LLMRequest:
    return LLMRequest(
        messages=[ChatMessage(role=MessageRole.USER, content=QUERY)],
        model=MODEL,
        tools=tools if tools is not None else [WEB_SEARCH_TOOL],
        tool_choice=tool_choice,
    )


def _capture(
    body: dict[str, Any],
) -> tuple[list[httpx.Request], httpx.MockTransport]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=body)

    return requests, httpx.MockTransport(handler)


def _openai_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "chatcmpl-test",
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Tides rise."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4},
    }
    body.update(overrides)
    return body


def _anthropic_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "msg-test",
        "model": MODEL,
        "content": [{"type": "text", "text": "Tides rise."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 4},
    }
    body.update(overrides)
    return body


def _gemini_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "Tides rise."}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 4,
        },
        "modelVersion": MODEL,
        "responseId": "resp-test",
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# OpenAI Chat Completions
# ---------------------------------------------------------------------------


def test_openai_maps_the_web_search_tool_to_native_options() -> None:
    requests, transport = _capture(_openai_body())
    provider = OpenAILLMProvider(api_key=API_KEY, transport=transport)

    asyncio.run(provider.complete(_request(tools=[WEB_SEARCH_TOOL, LOOKUP_TOOL])))

    sent = json.loads(requests[0].content)
    assert sent["web_search_options"] == {}
    assert [tool["function"]["name"] for tool in sent["tools"]] == ["lookup"]
    assert sent["tool_choice"] == "auto"


def test_openai_sends_only_native_search_when_no_function_tools_remain() -> None:
    requests, transport = _capture(_openai_body())
    provider = OpenAILLMProvider(api_key=API_KEY, transport=transport)

    asyncio.run(provider.complete(_request()))

    sent = json.loads(requests[0].content)
    assert sent["web_search_options"] == {}
    assert "tools" not in sent
    assert "tool_choice" not in sent


def test_openai_does_not_enable_search_when_tools_are_disabled() -> None:
    requests, transport = _capture(_openai_body())
    provider = OpenAILLMProvider(api_key=API_KEY, transport=transport)

    asyncio.run(provider.complete(_request(tool_choice=ToolChoice.NONE)))

    sent = json.loads(requests[0].content)
    assert "web_search_options" not in sent
    assert "tools" not in sent


def test_openai_maps_url_citation_annotations_to_citations() -> None:
    body = _openai_body(
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Tides rise.",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url_citation": {
                                "url": "https://example.com/tides",
                                "title": "Tides explained",
                                "start_index": 0,
                                "end_index": 10,
                            },
                        },
                        {
                            "type": "url_citation",
                            "url_citation": {
                                "url": "https://example.com/tides",
                                "title": "Tides explained",
                                "start_index": 11,
                                "end_index": 20,
                            },
                        },
                    ],
                },
                "finish_reason": "stop",
            }
        ]
    )
    provider = OpenAILLMProvider(
        api_key=API_KEY,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)),
    )

    response = asyncio.run(provider.complete(_request()))

    assert response.metadata["citations"] == [
        {
            "title": "Tides explained",
            "url": "https://example.com/tides",
            "snippet": "",
        }
    ]
    assert response.citations == response.metadata["citations"]


def test_openai_compatible_inherits_native_search_mapping() -> None:
    requests, transport = _capture(_openai_body())
    provider = OpenAICompatibleLLMProvider(
        base_url="https://llm.example.test",
        api_key=API_KEY,
        transport=transport,
    )

    asyncio.run(provider.complete(_request()))

    sent = json.loads(requests[0].content)
    assert sent["web_search_options"] == {}
    assert "tools" not in sent


# ---------------------------------------------------------------------------
# Anthropic Messages
# ---------------------------------------------------------------------------


def test_anthropic_maps_the_web_search_tool_to_the_server_tool() -> None:
    requests, transport = _capture(_anthropic_body())
    provider = AnthropicLLMProvider(api_key=API_KEY, transport=transport)

    asyncio.run(provider.complete(_request(tools=[WEB_SEARCH_TOOL, LOOKUP_TOOL])))

    sent = json.loads(requests[0].content)
    assert sent["tools"] == [
        {
            "name": "lookup",
            "description": "Look something up",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        },
        {"type": "web_search_20250305", "name": "web_search"},
    ]
    assert sent["tool_choice"] == {"type": "auto"}


def test_anthropic_omits_tool_choice_for_native_search_only() -> None:
    requests, transport = _capture(_anthropic_body())
    provider = AnthropicLLMProvider(api_key=API_KEY, transport=transport)

    asyncio.run(provider.complete(_request()))

    sent = json.loads(requests[0].content)
    assert sent["tools"] == [{"type": "web_search_20250305", "name": "web_search"}]
    assert "tool_choice" not in sent


def test_anthropic_maps_search_results_and_text_citations_to_citations() -> None:
    body = _anthropic_body(
        content=[
            {
                "type": "server_tool_use",
                "id": "srv-1",
                "name": "web_search",
                "input": {"query": QUERY},
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srv-1",
                "content": [
                    {
                        "type": "web_search_result",
                        "url": "https://example.com/tides",
                        "title": "Tides explained",
                        "encrypted_content": "secret",
                        "page_age": "2026-01-01T00:00:00Z",
                    },
                    {
                        "type": "web_search_result",
                        "url": "https://example.org/moon",
                        "title": "Moon and tides",
                    },
                ],
            },
            {
                "type": "text",
                "text": "Tides rise.",
                "citations": [
                    {
                        "type": "web_search_result_location",
                        "url": "https://example.com/tides",
                        "title": "Tides explained",
                        "cited_text": "A short snippet",
                    }
                ],
            },
        ]
    )
    provider = AnthropicLLMProvider(
        api_key=API_KEY,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)),
    )

    response = asyncio.run(provider.complete(_request()))

    assert response.content == "Tides rise."
    assert response.metadata["citations"] == [
        {
            "title": "Tides explained",
            "url": "https://example.com/tides",
            "snippet": "A short snippet",
        },
        {
            "title": "Moon and tides",
            "url": "https://example.org/moon",
            "snippet": "",
        },
    ]


# ---------------------------------------------------------------------------
# Gemini generateContent
# ---------------------------------------------------------------------------


def test_gemini_maps_the_web_search_tool_to_google_search() -> None:
    requests, transport = _capture(_gemini_body())
    provider = GeminiLLMProvider(api_key=API_KEY, transport=transport)

    asyncio.run(provider.complete(_request(tools=[WEB_SEARCH_TOOL, LOOKUP_TOOL])))

    sent = json.loads(requests[0].content)
    assert sent["tools"] == [
        {"googleSearch": {}},
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
        },
    ]
    assert sent["toolConfig"] == {"functionCallingConfig": {"mode": "AUTO"}}


def test_gemini_omits_tool_config_for_native_search_only() -> None:
    requests, transport = _capture(_gemini_body())
    provider = GeminiLLMProvider(api_key=API_KEY, transport=transport)

    asyncio.run(provider.complete(_request()))

    sent = json.loads(requests[0].content)
    assert sent["tools"] == [{"googleSearch": {}}]
    assert "toolConfig" not in sent


def test_gemini_maps_grounding_metadata_to_citations() -> None:
    body = _gemini_body(
        candidates=[
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "Tides rise."}],
                },
                "finishReason": "STOP",
                "groundingMetadata": {
                    "webSearchQueries": [QUERY],
                    "groundingChunks": [
                        {
                            "web": {
                                "uri": "https://example.com/tides",
                                "title": "Tides explained",
                            }
                        },
                        {
                            "web": {
                                "uri": "https://example.org/moon",
                                "title": "Moon and tides",
                            }
                        },
                    ],
                    "groundingSupports": [
                        {
                            "segment": {
                                "startIndex": 0,
                                "endIndex": 10,
                                "text": "A short snippet",
                            },
                            "groundingChunkIndices": [0],
                        }
                    ],
                },
            }
        ]
    )
    provider = GeminiLLMProvider(
        api_key=API_KEY,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)),
    )

    response = asyncio.run(provider.complete(_request()))

    assert response.metadata["citations"] == [
        {
            "title": "Tides explained",
            "url": "https://example.com/tides",
            "snippet": "A short snippet",
        },
        {
            "title": "Moon and tides",
            "url": "https://example.org/moon",
            "snippet": "",
        },
    ]


# ---------------------------------------------------------------------------
# End to end through the vendor-neutral search provider
# ---------------------------------------------------------------------------


def _openai_search_body() -> dict[str, Any]:
    return _openai_body(
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Tides rise.",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url_citation": {
                                "url": "https://example.com/tides",
                                "title": "Tides explained",
                            },
                        }
                    ],
                },
                "finish_reason": "stop",
            }
        ]
    )


def _anthropic_search_body() -> dict[str, Any]:
    return _anthropic_body(
        content=[
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srv-1",
                "content": [
                    {
                        "type": "web_search_result",
                        "url": "https://example.com/tides",
                        "title": "Tides explained",
                    }
                ],
            }
        ]
    )


def _gemini_search_body() -> dict[str, Any]:
    return _gemini_body(
        candidates=[
            {
                "content": {"role": "model", "parts": [{"text": "Tides rise."}]},
                "finishReason": "STOP",
                "groundingMetadata": {
                    "groundingChunks": [
                        {
                            "web": {
                                "uri": "https://example.com/tides",
                                "title": "Tides explained",
                            }
                        }
                    ]
                },
            }
        ]
    )


@pytest.mark.parametrize(
    ("builder", "body"),
    [
        (
            lambda transport: OpenAILLMProvider(api_key=API_KEY, transport=transport),
            _openai_search_body,
        ),
        (
            lambda transport: OpenAICompatibleLLMProvider(
                base_url="https://llm.example.test",
                api_key=API_KEY,
                transport=transport,
            ),
            _openai_search_body,
        ),
        (
            lambda transport: AnthropicLLMProvider(
                api_key=API_KEY, transport=transport
            ),
            _anthropic_search_body,
        ),
        (
            lambda transport: GeminiLLMProvider(api_key=API_KEY, transport=transport),
            _gemini_search_body,
        ),
    ],
)
def test_llm_search_provider_returns_native_citations(
    builder: Callable[[httpx.AsyncBaseTransport], Any],
    body: Callable[[], dict[str, Any]],
) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body()))
    provider = LLMSearchProvider(llm=builder(transport), model=MODEL)

    result = asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert result.urls == ["https://example.com/tides"]
    assert result.results[0].title == "Tides explained"
    assert result.total_results == 1

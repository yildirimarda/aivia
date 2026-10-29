"""SerpAPI and LLM-native search providers.

The tests cover request mapping, result mapping and the error taxonomy for
SerpAPI, and the LLM request shape plus every supported result shape for the
LLM-native provider. Everything runs offline with a mock transport or a fake
LLM.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any

import httpx
import pytest

from aivia.core.errors import (
    AiviaError,
    AuthError,
    BudgetError,
    ConfigError,
    InvalidRequestError,
    RateLimitError,
    RetryableError,
    ValidationError,
)
from aivia.core.errors import (
    TimeoutError as AiviaTimeoutError,
)
from aivia.core.secrets import SecretsLoader
from aivia.providers import (
    LLMSearch as TopLevelLLMSearch,
)
from aivia.providers import (
    SerpAPI as TopLevelSerpAPI,
)
from aivia.providers import provider_registry
from aivia.providers.protocols import (
    CAPABILITY_SEARCH,
    SafeSearch,
    SearchDepth,
    SearchRecency,
    SearchRequest,
    SearchResult,
)
from aivia.providers.protocols.llm import (
    LLMRequest,
    LLMResponse,
    LLMUsage,
    MessageRole,
    ToolCall,
    ToolChoice,
)
from aivia.providers.search import (
    DEFAULT_TOOL_NAME,
    SERPAPI_DEFAULT_BASE_URL,
    LLMNative,
    LLMNativeProvider,
    LLMNativeSearchProvider,
    LLMSearch,
    LLMSearchProvider,
    LLMWebSearchProvider,
    NativeSearch,
    NativeSearchProvider,
    SerpAPI,
    SerpAPIProvider,
    SerpAPISearchProvider,
)

API_KEY = "test-key"
QUERY = "ocean tides"


def _search_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "search_metadata": {
            "id": "req-serp-1",
            "status": "Success",
            "total_time_taken": 0.42,
        },
        "search_information": {
            "query_displayed": QUERY,
            "total_results": 1234,
            "organic_results_state": "Results for exact spelling",
        },
        "organic_results": [
            {
                "position": 1,
                "title": "Tides explained",
                "link": "https://example.com/tides",
                "displayed_link": "example.com",
                "snippet": "A short snippet",
                "date": "Jan 1, 2026",
                "source": "Example",
                "favicon": "https://example.com/favicon.ico",
            },
            {
                "position": 2,
                "title": "Moon and tides",
                "link": "https://example.org/moon",
                "snippet": "Another snippet",
            },
        ],
        "answer_box": {
            "title": "Ocean tides",
            "answer": "Tides are driven by the moon.",
            "link": "https://example.com/answer",
        },
        "knowledge_graph": {
            "title": "Tide",
            "description": "A rise and fall of sea levels.",
        },
    }
    body.update(overrides)
    return body


def _provider(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> SerpAPISearchProvider:
    return SerpAPISearchProvider(
        api_key=API_KEY,
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _capture(
    body: dict[str, Any] | None = None,
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> tuple[list[httpx.Request], Callable[[httpx.Request], httpx.Response]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status,
            json=body if body is not None else _search_body(),
            headers=headers,
        )

    return requests, handler


class _FakeLLM:
    """A minimal LLM provider that records requests and returns one response."""

    def __init__(self, response: LLMResponse) -> None:
        self.response = response
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return self.response


def _llm_response(**overrides: Any) -> LLMResponse:
    body: dict[str, Any] = {
        "content": "",
        "provider": "fake-llm",
        "model": "fake-model",
        "usage": LLMUsage(input_tokens=12, output_tokens=4),
        "metadata": {},
    }
    body.update(overrides)
    return LLMResponse.model_validate(body)


def _llm_provider(
    response: LLMResponse, **kwargs: Any
) -> tuple[LLMSearchProvider, _FakeLLM]:
    fake = _FakeLLM(response)
    return LLMSearchProvider(llm=fake, **kwargs), fake


# ---------------------------------------------------------------------------
# SerpAPI
# ---------------------------------------------------------------------------


def test_serpapi_metadata_and_registration_describe_the_provider() -> None:
    provider = _provider(lambda request: httpx.Response(200, json=_search_body()))

    assert provider.name == "serpapi"
    assert provider.capabilities == (CAPABILITY_SEARCH,)
    assert provider.supports(CAPABILITY_SEARCH)
    assert provider.base_url == SERPAPI_DEFAULT_BASE_URL
    assert provider.endpoint == "https://serpapi.com/search.json"
    assert provider.engine == "google"
    assert SerpAPIProvider is SerpAPISearchProvider
    assert SerpAPI is SerpAPISearchProvider
    assert TopLevelSerpAPI is SerpAPISearchProvider
    assert provider_registry.resolve("serpapi") is SerpAPISearchProvider


def test_serpapi_maps_the_search_request_to_query_parameters() -> None:
    requests, handler = _capture()
    provider = _provider(handler)
    request = SearchRequest(
        query=f"  {QUERY}  ",
        max_results=7,
        include_domains=["https://www.Example.com/path", "example.com"],
        exclude_domains=["news.example.org"],
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
        language="EN",
        country="de",
        safe_search=SafeSearch.STRICT,
        search_depth=SearchDepth.ADVANCED,
    )

    asyncio.run(provider.search(request))

    sent = requests[0]
    assert sent.method == "GET"
    assert str(sent.url).startswith("https://serpapi.com/search.json?")
    assert sent.headers["accept"] == "application/json"
    assert dict(sent.url.params) == {
        "engine": "google",
        "q": f"{QUERY} site:example.com -site:news.example.org",
        "api_key": API_KEY,
        "num": "7",
        "hl": "en",
        "gl": "DE",
        "safe": "active",
        "tbs": "cdr:1,cd_min:1/1/2026,cd_max:1/31/2026",
    }


def test_serpapi_omits_unset_filters_and_keeps_the_result_limit() -> None:
    requests, handler = _capture()
    provider = _provider(handler)

    asyncio.run(provider.search(SearchRequest(query=QUERY, max_results=100)))

    assert dict(requests[0].url.params) == {
        "engine": "google",
        "q": QUERY,
        "api_key": API_KEY,
        "num": "100",
        "safe": "active",
    }


def test_serpapi_maps_recency_to_a_time_filter() -> None:
    requests, handler = _capture()
    provider = _provider(handler)

    asyncio.run(provider.search(SearchRequest(query=QUERY, recency=SearchRecency.WEEK)))

    assert requests[0].url.params["tbs"] == "qdr:w"


def test_serpapi_a_date_range_wins_over_the_recency_window() -> None:
    requests, handler = _capture()
    provider = _provider(handler)

    asyncio.run(
        provider.search(
            SearchRequest(
                query=QUERY,
                recency=SearchRecency.DAY,
                start_date=date(2026, 2, 1),
                end_date=date(2026, 2, 28),
            )
        )
    )

    assert requests[0].url.params["tbs"] == "cdr:1,cd_min:2/1/2026,cd_max:2/28/2026"


def test_serpapi_can_turn_safe_search_off() -> None:
    requests, handler = _capture()
    provider = _provider(handler)

    asyncio.run(provider.search(SearchRequest(query=QUERY, safe_search=SafeSearch.OFF)))

    assert requests[0].url.params["safe"] == "off"


def test_serpapi_maps_results_answer_and_metadata_from_the_response() -> None:
    provider = _provider(lambda request: httpx.Response(200, json=_search_body()))

    response = asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert response.query == QUERY
    assert response.provider == "serpapi"
    assert response.answer == "Tides are driven by the moon."
    assert response.total_results == 1234
    assert response.took_ms == pytest.approx(420.0)
    assert response.request_id == "req-serp-1"
    assert response.metadata == {
        "engine": "google",
        "organic_results_state": "Results for exact spelling",
        "status": "Success",
    }

    first = response.results[0]
    assert first.title == "Tides explained"
    assert first.url == "https://example.com/tides"
    assert first.snippet == "A short snippet"
    assert first.content == ""
    assert first.text == "A short snippet"
    assert first.published_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert first.source == "serpapi"
    assert first.metadata == {
        "position": 1,
        "displayed_link": "example.com",
        "source": "Example",
        "favicon": "https://example.com/favicon.ico",
        "date": "Jan 1, 2026",
    }

    second = response.results[1]
    assert second.published_at is None
    assert second.metadata == {"position": 2}


def test_serpapi_falls_back_to_the_knowledge_graph_answer() -> None:
    body = _search_body()
    body.pop("answer_box")
    provider = _provider(lambda request: httpx.Response(200, json=body))

    response = asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert response.answer == "A rise and fall of sea levels."


def test_serpapi_falls_back_to_the_request_query_when_omitted() -> None:
    body = _search_body()
    body["search_information"] = {}
    provider = _provider(lambda request: httpx.Response(200, json=body))

    response = asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert response.query == QUERY


def test_serpapi_uses_a_response_request_id_header_when_the_body_omits_it() -> None:
    body = _search_body()
    body["search_metadata"] = {}
    provider = _provider(
        lambda request: httpx.Response(
            200,
            json=body,
            headers={"x-request-id": "header-request"},
        )
    )

    response = asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert response.request_id == "header-request"


def test_serpapi_resolves_the_api_key_from_a_secret_reference() -> None:
    requests, handler = _capture()
    loader = SecretsLoader(environ={"AIVA_SERPAPI_API_KEY": API_KEY})
    provider = SerpAPISearchProvider(
        api_key_ref="SERPAPI_API_KEY",
        secrets=loader,
        transport=httpx.MockTransport(handler),
    )

    asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert requests[0].url.params["api_key"] == API_KEY


def test_serpapi_requires_an_api_key() -> None:
    with pytest.raises(ConfigError):
        SerpAPISearchProvider()
    with pytest.raises(ConfigError):
        SerpAPISearchProvider(api_key="   ")


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (400, InvalidRequestError),
        (401, AuthError),
        (402, BudgetError),
        (403, AuthError),
        (408, AiviaTimeoutError),
        (429, RateLimitError),
        (500, RetryableError),
        (503, RetryableError),
        (504, AiviaTimeoutError),
    ],
)
def test_serpapi_maps_error_statuses_to_the_shared_taxonomy(
    status: int,
    expected: type[AiviaError],
) -> None:
    provider = _provider(
        lambda request: httpx.Response(
            status,
            json={"error": "request rejected"},
        )
    )

    with pytest.raises(expected) as caught:
        asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert caught.value.details["provider"] == "serpapi"
    assert caught.value.details["status"] == status


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Invalid API key. Please check your key.", AuthError),
        ("Your account has run out of searches.", RateLimitError),
        ("The search request was invalid.", InvalidRequestError),
    ],
)
def test_serpapi_maps_a_body_error_on_a_success_status(
    message: str,
    expected: type[AiviaError],
) -> None:
    provider = _provider(lambda request: httpx.Response(200, json={"error": message}))

    with pytest.raises(expected):
        asyncio.run(provider.search(SearchRequest(query=QUERY)))


def test_serpapi_rejects_a_non_json_response() -> None:
    provider = _provider(lambda request: httpx.Response(200, text="not json"))

    with pytest.raises(ValidationError):
        asyncio.run(provider.search(SearchRequest(query=QUERY)))


@pytest.mark.parametrize(
    "body",
    [
        [],
        {"organic_results": [{"title": "missing link"}]},
        {"organic_results": [{"link": "ftp://example.com/file"}]},
    ],
)
def test_serpapi_rejects_a_malformed_response(body: Any) -> None:
    provider = _provider(lambda request: httpx.Response(200, json=body))

    with pytest.raises(ValidationError):
        asyncio.run(provider.search(SearchRequest(query=QUERY)))


# ---------------------------------------------------------------------------
# LLM-native search
# ---------------------------------------------------------------------------


def test_llm_search_metadata_and_registration_describe_the_provider() -> None:
    provider, _ = _llm_provider(_llm_response())

    assert provider.name == "llm_native"
    assert provider.capabilities == (CAPABILITY_SEARCH,)
    assert provider.supports(CAPABILITY_SEARCH)
    assert provider.tool_name == DEFAULT_TOOL_NAME
    assert LLMWebSearchProvider is LLMSearchProvider
    assert LLMNativeSearchProvider is LLMSearchProvider
    assert LLMNativeProvider is LLMSearchProvider
    assert LLMNative is LLMSearchProvider
    assert NativeSearchProvider is LLMSearchProvider
    assert NativeSearch is LLMSearchProvider
    assert LLMSearch is LLMSearchProvider
    assert TopLevelLLMSearch is LLMSearchProvider
    assert provider_registry.resolve("llm_native") is LLMSearchProvider
    assert provider_registry.resolve("llm_search") is LLMSearchProvider


def test_llm_search_builds_a_native_web_search_request() -> None:
    response = _llm_response(
        metadata={
            "citations": [
                {
                    "url": "https://example.com/tides",
                    "title": "Tides explained",
                    "snippet": "A short snippet",
                }
            ]
        }
    )
    provider, fake = _llm_provider(response, model="model-x", role="factcheck")
    request = SearchRequest(
        query=f"  {QUERY}  ",
        max_results=5,
        include_domains=["example.com"],
        exclude_domains=["news.example.org"],
        recency=SearchRecency.WEEK,
        language="EN",
        country="de",
        safe_search=SafeSearch.STRICT,
    )

    asyncio.run(provider.search(request))

    sent = fake.requests[0]
    assert sent.model == "model-x"
    assert sent.role == "factcheck"
    assert sent.tool_choice is ToolChoice.AUTO
    assert [tool.name for tool in sent.tools] == [DEFAULT_TOOL_NAME]
    assert sent.metadata == {"web_search": True, "max_results": 5}
    assert sent.messages[0].role is MessageRole.SYSTEM
    assert "web search" in sent.messages[0].text.lower()
    prompt = sent.messages[1].text
    assert QUERY in prompt
    assert "example.com" in prompt
    assert "news.example.org" in prompt
    assert "week" in prompt
    assert "en" in prompt
    assert "DE" in prompt
    assert "strict" in prompt


def test_llm_search_maps_citation_metadata_to_results() -> None:
    response = _llm_response(
        request_id="req-llm-1",
        metadata={
            "citations": [
                {
                    "url": "https://example.com/tides",
                    "title": "Tides explained",
                    "snippet": "A short snippet",
                    "published_date": "2026-01-01T12:00:00Z",
                    "type": "url_citation",
                },
                {
                    "web": {
                        "uri": "https://example.org/moon",
                        "title": "Moon and tides",
                    }
                },
            ]
        },
    )
    provider, _ = _llm_provider(response)

    result = asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert result.query == QUERY
    assert result.provider == "llm_native"
    assert result.request_id == "req-llm-1"
    assert result.total_results == 2
    assert result.metadata["model"] == "fake-model"
    assert result.metadata["finish_reason"] == "stop"
    assert result.metadata["usage"] == {
        "input_tokens": 12,
        "output_tokens": 4,
        "cached_input_tokens": 0,
    }

    first = result.results[0]
    assert first.title == "Tides explained"
    assert first.url == "https://example.com/tides"
    assert first.snippet == "A short snippet"
    assert first.published_at == datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    assert first.source == "llm_native"
    assert first.metadata == {
        "published_date": "2026-01-01T12:00:00Z",
        "type": "url_citation",
    }

    second = result.results[1]
    assert second.title == "Moon and tides"
    assert second.url == "https://example.org/moon"


def test_llm_search_parses_a_fenced_json_body() -> None:
    content = (
        "```json\n"
        + json.dumps(
            {
                "results": [
                    {
                        "title": "Tides explained",
                        "url": "https://example.com/tides",
                        "snippet": "A short snippet",
                    }
                ]
            }
        )
        + "\n```"
    )
    provider, _ = _llm_provider(_llm_response(content=content))

    result = asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert [item.url for item in result.results] == ["https://example.com/tides"]
    assert result.answer is None


def test_llm_search_parses_a_bare_json_list() -> None:
    content = json.dumps(
        [
            {"title": "One", "url": "https://example.com/one"},
            {"title": "Two", "url": "https://example.com/two"},
        ]
    )
    provider, _ = _llm_provider(_llm_response(content=content))

    result = asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert [item.title for item in result.results] == ["One", "Two"]


def test_llm_search_parses_results_from_a_tool_call() -> None:
    response = _llm_response(
        tool_calls=[
            ToolCall(
                id="call-1",
                name=DEFAULT_TOOL_NAME,
                arguments={
                    "results": [
                        {
                            "title": "Tides explained",
                            "url": "https://example.com/tides",
                            "snippet": "A short snippet",
                        }
                    ]
                },
            )
        ]
    )
    provider, _ = _llm_provider(response)

    result = asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert [item.url for item in result.results] == ["https://example.com/tides"]


def test_llm_search_returns_an_empty_result_for_an_empty_answer() -> None:
    provider, _ = _llm_provider(_llm_response(content="   "))

    result = asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert result.results == []
    assert result.total_results == 0
    assert result.answer is None


def test_llm_search_uses_plain_text_as_an_answer_without_results() -> None:
    provider, _ = _llm_provider(_llm_response(content="No pages were found."))

    result = asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert result.results == []
    assert result.answer == "No pages were found."


def test_llm_search_rejects_json_without_a_result_list() -> None:
    provider, _ = _llm_provider(_llm_response(content='{"results": "not a list"}'))

    with pytest.raises(ValidationError):
        asyncio.run(provider.search(SearchRequest(query=QUERY)))


def test_llm_search_rejects_a_result_without_a_url() -> None:
    provider, _ = _llm_provider(
        _llm_response(metadata={"results": [{"title": "missing url"}]})
    )

    with pytest.raises(ValidationError):
        asyncio.run(provider.search(SearchRequest(query=QUERY)))


def test_llm_search_caps_results_by_the_request_and_the_configured_limit() -> None:
    citations = [
        {"url": f"https://example.com/{index}", "title": str(index)}
        for index in range(5)
    ]
    response = _llm_response(metadata={"citations": citations})

    provider, _ = _llm_provider(response, max_results=2)
    result = asyncio.run(provider.search(SearchRequest(query=QUERY, max_results=4)))
    assert len(result.results) == 2

    provider, _ = _llm_provider(response)
    result = asyncio.run(provider.search(SearchRequest(query=QUERY, max_results=3)))
    assert len(result.results) == 3


def test_llm_search_accepts_the_provider_keyword_alias() -> None:
    fake = _FakeLLM(_llm_response(metadata={"results": []}))
    provider = LLMSearchProvider(provider=fake)

    asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert fake.requests


def test_llm_search_can_rename_the_native_tool() -> None:
    response = _llm_response(
        tool_calls=[
            ToolCall(
                id="call-1",
                name="browse",
                arguments={"results": [{"url": "https://example.com/one"}]},
            )
        ]
    )
    provider, fake = _llm_provider(response, tool_name="browse")

    result = asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert fake.requests[0].tools[0].name == "browse"
    assert [item.url for item in result.results] == ["https://example.com/one"]


def test_llm_search_requires_a_wrapped_provider() -> None:
    with pytest.raises(ConfigError):
        LLMSearchProvider()
    with pytest.raises(TypeError):
        LLMSearchProvider(
            llm=_FakeLLM(_llm_response()),
            provider=_FakeLLM(_llm_response()),
        )


def test_llm_search_rejects_invalid_configuration() -> None:
    fake = _FakeLLM(_llm_response())

    with pytest.raises(ConfigError):
        LLMSearchProvider(llm=fake, role="  ")
    with pytest.raises(ConfigError):
        LLMSearchProvider(llm=fake, tool_name="  ")
    with pytest.raises(ConfigError):
        LLMSearchProvider(llm=fake, max_results=0)
    with pytest.raises(TypeError):
        LLMSearchProvider(llm=fake, max_results=True)


def test_llm_search_returns_a_validated_search_result_model() -> None:
    provider, _ = _llm_provider(
        _llm_response(metadata={"results": [{"url": "https://example.com/one"}]})
    )

    result = asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert isinstance(result.results[0], SearchResult)
    assert isinstance(result.results[0].url, str)

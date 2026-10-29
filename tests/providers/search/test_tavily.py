"""Tavily search provider: request mapping, result mapping and error taxonomy."""

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
from aivia.providers import provider_registry
from aivia.providers.protocols import (
    CAPABILITY_SEARCH,
    SafeSearch,
    SearchDepth,
    SearchRecency,
    SearchRequest,
)
from aivia.providers.search import (
    DEFAULT_BASE_URL,
    Tavily,
    TavilyProvider,
    TavilySearchProvider,
)

API_KEY = "test-key"
QUERY = "ocean tides"


def _search_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "query": QUERY,
        "results": [
            {
                "title": "Tides explained",
                "url": "https://example.com/tides",
                "content": "A short snippet",
                "raw_content": "The complete page text",
                "score": 0.91,
                "published_date": "2026-01-01T12:00:00Z",
            },
            {
                "title": "Moon and tides",
                "url": "https://example.org/moon",
                "content": "Another snippet",
                "score": 0.72,
            },
        ],
        "answer": "Tides are driven by the moon.",
        "response_time": 1.25,
        "request_id": "req-tavily-1",
    }
    body.update(overrides)
    return body


def _provider(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> TavilySearchProvider:
    return TavilySearchProvider(
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


def test_metadata_and_registration_describe_the_provider() -> None:
    provider = _provider(lambda request: httpx.Response(200, json=_search_body()))

    assert provider.name == "tavily"
    assert provider.capabilities == (CAPABILITY_SEARCH,)
    assert provider.supports(CAPABILITY_SEARCH)
    assert provider.base_url == DEFAULT_BASE_URL
    assert provider.endpoint == "https://api.tavily.com/search"
    assert TavilyProvider is TavilySearchProvider
    assert Tavily is TavilySearchProvider
    assert provider_registry.resolve("tavily") is TavilySearchProvider


def test_maps_the_search_request_to_the_tavily_payload() -> None:
    requests, handler = _capture()
    provider = _provider(handler)
    request = SearchRequest(
        query=f"  {QUERY}  ",
        max_results=7,
        include_domains=["https://www.Example.com/path", "example.com"],
        exclude_domains=["news.example.org"],
        recency=SearchRecency.WEEK,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
        language="EN",
        country="de",
        safe_search=SafeSearch.STRICT,
        search_depth=SearchDepth.ADVANCED,
    )

    asyncio.run(provider.search(request))

    sent = requests[0]
    assert sent.method == "POST"
    assert str(sent.url) == "https://api.tavily.com/search"
    assert sent.headers["authorization"] == f"Bearer {API_KEY}"
    assert sent.headers["content-type"] == "application/json"
    body = json.loads(sent.content)
    assert body == {
        "query": QUERY,
        "max_results": 7,
        "search_depth": "advanced",
        "include_domains": ["example.com"],
        "exclude_domains": ["news.example.org"],
        "time_range": "week",
        "start_date": "2026-01-01",
        "end_date": "2026-01-31",
        "country": "DE",
    }


def test_omits_unset_filters_and_clamps_the_result_limit() -> None:
    requests, handler = _capture()
    provider = _provider(handler)

    asyncio.run(provider.search(SearchRequest(query=QUERY, max_results=50)))

    body = json.loads(requests[0].content)
    assert body == {
        "query": QUERY,
        "max_results": 20,
        "search_depth": "basic",
    }


def test_can_request_an_answer_and_raw_content() -> None:
    requests, handler = _capture()
    provider = _provider(handler, include_answer=True, include_raw_content=True)

    asyncio.run(provider.search(SearchRequest(query=QUERY)))

    body = json.loads(requests[0].content)
    assert body["include_answer"] is True
    assert body["include_raw_content"] is True


def test_maps_results_answer_and_timing_from_the_response() -> None:
    provider = _provider(lambda request: httpx.Response(200, json=_search_body()))

    response = asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert response.query == QUERY
    assert response.provider == "tavily"
    assert response.answer == "Tides are driven by the moon."
    assert response.total_results == 2
    assert response.took_ms == 1250.0
    assert response.request_id == "req-tavily-1"

    first = response.results[0]
    assert first.title == "Tides explained"
    assert first.url == "https://example.com/tides"
    assert first.snippet == "A short snippet"
    assert first.content == "The complete page text"
    assert first.text == "The complete page text"
    assert first.score == 0.91
    assert first.published_at == datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    assert first.source == "tavily"

    second = response.results[1]
    assert second.content == ""
    assert second.text == "Another snippet"
    assert second.published_at is None


def test_parses_rfc_published_dates_and_falls_back_to_the_request_query() -> None:
    body = _search_body(
        query="",
        results=[
            {
                "title": "Dated result",
                "url": "https://example.com/dated",
                "content": "Dated",
                "published_date": "Wed, 01 Jan 2025 12:00:00 GMT",
            }
        ],
    )
    provider = _provider(lambda request: httpx.Response(200, json=body))

    response = asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert response.query == QUERY
    assert response.results[0].published_at == datetime(2025, 1, 1, 12, 0, tzinfo=UTC)


def test_uses_a_response_request_id_header_when_the_body_omits_it() -> None:
    body = _search_body()
    body.pop("request_id")
    provider = _provider(
        lambda request: httpx.Response(
            200,
            json=body,
            headers={"x-request-id": "header-request"},
        )
    )

    response = asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert response.request_id == "header-request"


def test_resolves_the_api_key_from_a_secret_reference() -> None:
    requests, handler = _capture()
    loader = SecretsLoader(environ={"AIVA_TAVILY_API_KEY": API_KEY})
    provider = TavilySearchProvider(
        api_key_ref="TAVILY_API_KEY",
        secrets=loader,
        transport=httpx.MockTransport(handler),
    )

    asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert requests[0].headers["authorization"] == f"Bearer {API_KEY}"


def test_requires_an_api_key() -> None:
    with pytest.raises(ConfigError):
        TavilySearchProvider()
    with pytest.raises(ConfigError):
        TavilySearchProvider(api_key="   ")


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (400, InvalidRequestError),
        (401, AuthError),
        (403, AuthError),
        (408, AiviaTimeoutError),
        (429, RateLimitError),
        (432, RateLimitError),
        (433, RateLimitError),
        (500, RetryableError),
        (504, AiviaTimeoutError),
    ],
)
def test_maps_error_statuses_to_the_shared_taxonomy(
    status: int,
    expected: type[AiviaError],
) -> None:
    provider = _provider(
        lambda request: httpx.Response(
            status,
            json={"detail": {"error": "request rejected"}},
        )
    )

    with pytest.raises(expected) as caught:
        asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert caught.value.details["provider"] == "tavily"
    assert caught.value.details["status"] == status


@pytest.mark.parametrize("status", [432, 433])
def test_maps_quota_statuses_to_rate_limited_with_the_api_message(
    status: int,
) -> None:
    provider = _provider(
        lambda request: httpx.Response(
            status,
            json={"detail": {"error": "Plan limit exceeded"}},
        )
    )

    with pytest.raises(RateLimitError) as caught:
        asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert "Plan limit exceeded" in str(caught.value)
    assert caught.value.details["status"] == status


def test_maps_a_quota_message_on_another_status_to_rate_limited() -> None:
    provider = _provider(
        lambda request: httpx.Response(
            403,
            json={"detail": "Your usage limit has been reached"},
        )
    )

    with pytest.raises(RateLimitError):
        asyncio.run(provider.search(SearchRequest(query=QUERY)))


def test_rate_limit_carries_the_retry_after_header() -> None:
    provider = _provider(
        lambda request: httpx.Response(
            429,
            json={"detail": "slow down"},
            headers={"Retry-After": "3.5"},
        )
    )

    with pytest.raises(RateLimitError) as caught:
        asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert caught.value.retry_after == 3.5


def test_rejects_a_non_json_response() -> None:
    provider = _provider(lambda request: httpx.Response(200, text="not json"))

    with pytest.raises(ValidationError):
        asyncio.run(provider.search(SearchRequest(query=QUERY)))


@pytest.mark.parametrize(
    "body",
    [
        [],
        {"results": [{"title": "missing url"}]},
        {"results": [{"url": "ftp://example.com/file"}]},
    ],
)
def test_rejects_a_malformed_response(body: Any) -> None:
    provider = _provider(lambda request: httpx.Response(200, json=body))

    with pytest.raises(ValidationError):
        asyncio.run(provider.search(SearchRequest(query=QUERY)))

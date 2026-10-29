"""Exa search provider: request mapping, result mapping and error taxonomy."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any

import httpx
import pytest

import aivia.providers.search.exa as exa_module
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
from aivia.providers import Exa as TopLevelExa
from aivia.providers import provider_registry
from aivia.providers.protocols import (
    CAPABILITY_SEARCH,
    SafeSearch,
    SearchDepth,
    SearchRecency,
    SearchRequest,
)
from aivia.providers.search import (
    EXA_DEFAULT_BASE_URL,
    Exa,
    ExaProvider,
    ExaSearchProvider,
)

API_KEY = "test-key"
QUERY = "ocean tides"


def _search_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "requestId": "req-exa-1",
        "resolvedSearchType": "auto",
        "results": [
            {
                "id": "doc-1",
                "title": "Tides explained",
                "url": "https://example.com/tides",
                "publishedDate": "2026-01-01",
                "author": "Ada Lovelace",
                "score": 0.91,
                "text": "The complete page text",
                "highlights": ["A short snippet", "Another line"],
                "highlightScores": [0.9, 0.8],
                "summary": "A short summary",
                "image": "https://example.com/tides.png",
                "favicon": "https://example.com/favicon.ico",
            },
            {
                "id": "doc-2",
                "title": "Moon and tides",
                "url": "https://example.org/moon",
                "highlights": ["Another snippet"],
            },
        ],
        "searchTime": 1250.0,
        "costDollars": {"total": 0.005},
    }
    body.update(overrides)
    return body


def _provider(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> ExaSearchProvider:
    return ExaSearchProvider(
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

    assert provider.name == "exa"
    assert provider.capabilities == (CAPABILITY_SEARCH,)
    assert provider.supports(CAPABILITY_SEARCH)
    assert provider.base_url == EXA_DEFAULT_BASE_URL
    assert provider.endpoint == "https://api.exa.ai/search"
    assert ExaProvider is ExaSearchProvider
    assert Exa is ExaSearchProvider
    assert TopLevelExa is ExaSearchProvider
    assert provider_registry.resolve("exa") is ExaSearchProvider


def test_maps_the_search_request_to_the_exa_payload() -> None:
    requests, handler = _capture()
    provider = _provider(handler)
    request = SearchRequest(
        query=f"  {QUERY}  ",
        max_results=7,
        include_domains=["https://www.Example.com/path", "example.com"],
        exclude_domains=["news.example.org"],
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
        country="de",
        safe_search=SafeSearch.STRICT,
        search_depth=SearchDepth.ADVANCED,
    )

    asyncio.run(provider.search(request))

    sent = requests[0]
    assert sent.method == "POST"
    assert str(sent.url) == "https://api.exa.ai/search"
    assert sent.headers["x-api-key"] == API_KEY
    assert sent.headers["content-type"] == "application/json"
    body = json.loads(sent.content)
    assert body == {
        "query": QUERY,
        "numResults": 7,
        "type": "deep",
        "includeDomains": ["example.com"],
        "excludeDomains": ["news.example.org"],
        "startPublishedDate": "2026-01-01",
        "endPublishedDate": "2026-01-31",
        "userLocation": "DE",
        "moderation": True,
    }


def test_omits_unset_filters_and_keeps_the_result_limit() -> None:
    requests, handler = _capture()
    provider = _provider(handler)

    asyncio.run(
        provider.search(
            SearchRequest(
                query=QUERY,
                max_results=100,
                safe_search=SafeSearch.OFF,
            )
        )
    )

    body = json.loads(requests[0].content)
    assert body == {
        "query": QUERY,
        "numResults": 100,
        "type": "auto",
    }


def test_maps_recency_to_a_published_date_lower_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed = datetime(2026, 3, 15, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(exa_module, "_now", lambda: fixed)
    requests, handler = _capture()
    provider = _provider(handler)

    asyncio.run(provider.search(SearchRequest(query=QUERY, recency=SearchRecency.WEEK)))

    body = json.loads(requests[0].content)
    assert body["startPublishedDate"] == "2026-03-08"


def test_can_request_text_highlights_and_summaries() -> None:
    requests, handler = _capture()
    provider = _provider(
        handler,
        include_text=True,
        include_highlights=True,
        include_summary=True,
    )

    asyncio.run(provider.search(SearchRequest(query=QUERY)))

    body = json.loads(requests[0].content)
    assert body["contents"] == {
        "text": True,
        "highlights": True,
        "summary": True,
    }


def test_a_search_type_override_wins_over_the_depth_mapping() -> None:
    requests, handler = _capture()
    provider = _provider(handler, search_type="neural")

    asyncio.run(
        provider.search(SearchRequest(query=QUERY, search_depth=SearchDepth.ADVANCED))
    )

    body = json.loads(requests[0].content)
    assert body["type"] == "neural"


def test_maps_results_and_timing_from_the_response() -> None:
    provider = _provider(lambda request: httpx.Response(200, json=_search_body()))

    response = asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert response.query == QUERY
    assert response.provider == "exa"
    assert response.answer is None
    assert response.total_results == 2
    assert response.took_ms == 1250.0
    assert response.request_id == "req-exa-1"
    assert response.metadata["resolved_search_type"] == "auto"

    first = response.results[0]
    assert first.title == "Tides explained"
    assert first.url == "https://example.com/tides"
    assert first.snippet == "A short snippet\n\nAnother line"
    assert first.content == "The complete page text"
    assert first.text == "The complete page text"
    assert first.score == 0.91
    assert first.published_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert first.source == "exa"
    assert first.metadata == {
        "id": "doc-1",
        "author": "Ada Lovelace",
        "summary": "A short summary",
        "highlight_scores": [0.9, 0.8],
        "image": "https://example.com/tides.png",
        "favicon": "https://example.com/favicon.ico",
    }

    second = response.results[1]
    assert second.content == ""
    assert second.text == "Another snippet"
    assert second.published_at is None


def test_uses_a_summary_as_the_snippet_when_no_highlights_are_returned() -> None:
    body = _search_body(
        results=[
            {
                "title": "Summarized result",
                "url": "https://example.com/summary",
                "summary": "A generated summary",
            }
        ]
    )
    provider = _provider(lambda request: httpx.Response(200, json=body))

    response = asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert response.results[0].snippet == "A generated summary"


def test_uses_a_response_request_id_header_when_the_body_omits_it() -> None:
    body = _search_body()
    body.pop("requestId")
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
    loader = SecretsLoader(environ={"AIVA_EXA_API_KEY": API_KEY})
    provider = ExaSearchProvider(
        api_key_ref="EXA_API_KEY",
        secrets=loader,
        transport=httpx.MockTransport(handler),
    )

    asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert requests[0].headers["x-api-key"] == API_KEY


def test_requires_an_api_key() -> None:
    with pytest.raises(ConfigError):
        ExaSearchProvider()
    with pytest.raises(ConfigError):
        ExaSearchProvider(api_key="   ")


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
def test_maps_error_statuses_to_the_shared_taxonomy(
    status: int,
    expected: type[AiviaError],
) -> None:
    provider = _provider(
        lambda request: httpx.Response(
            status,
            json={"error": "request rejected", "tag": "REJECTED"},
        )
    )

    with pytest.raises(expected) as caught:
        asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert caught.value.details["provider"] == "exa"
    assert caught.value.details["status"] == status


def test_error_details_carry_the_api_message_tag_and_request_id() -> None:
    provider = _provider(
        lambda request: httpx.Response(
            429,
            json={
                "requestId": "err-request",
                "error": "Rate limit exceeded",
                "tag": "RATE_LIMIT",
            },
            headers={"Retry-After": "3.5"},
        )
    )

    with pytest.raises(RateLimitError) as caught:
        asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert "Rate limit exceeded" in str(caught.value)
    assert caught.value.retry_after == 3.5
    assert caught.value.details["code"] == "RATE_LIMIT"
    assert caught.value.details["request_id"] == "err-request"


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

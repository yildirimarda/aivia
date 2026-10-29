"""Brave search provider: request mapping, result mapping and error taxonomy."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any

import httpx
import pytest

import aivia.providers.search.brave as brave_module
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
from aivia.providers import Brave as TopLevelBrave
from aivia.providers import provider_registry
from aivia.providers.protocols import (
    CAPABILITY_SEARCH,
    SafeSearch,
    SearchDepth,
    SearchRecency,
    SearchRequest,
)
from aivia.providers.search import (
    BRAVE_DEFAULT_BASE_URL,
    Brave,
    BraveProvider,
    BraveSearchProvider,
)

API_KEY = "test-key"
QUERY = "ocean tides"


def _search_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "type": "search",
        "query": {
            "original": QUERY,
            "more_results_available": True,
        },
        "web": {
            "type": "search",
            "results": [
                {
                    "title": "Tides explained",
                    "url": "https://example.com/tides",
                    "description": "A short snippet",
                    "page_age": "2026-01-01T12:00:00Z",
                    "age": "3 days ago",
                    "language": "en",
                    "profile": {
                        "name": "Example",
                        "long_name": "Example Site",
                        "img": "https://example.com/logo.png",
                    },
                    "meta_url": {
                        "hostname": "example.com",
                        "favicon": "https://example.com/favicon.ico",
                    },
                    "extra_snippets": ["More detail", "Even more detail"],
                },
                {
                    "title": "Moon and tides",
                    "url": "https://example.org/moon",
                    "description": "Another snippet",
                },
            ],
        },
    }
    body.update(overrides)
    return body


def _provider(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> BraveSearchProvider:
    return BraveSearchProvider(
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

    assert provider.name == "brave"
    assert provider.capabilities == (CAPABILITY_SEARCH,)
    assert provider.supports(CAPABILITY_SEARCH)
    assert provider.base_url == BRAVE_DEFAULT_BASE_URL
    assert provider.endpoint == "https://api.search.brave.com/res/v1/web/search"
    assert BraveProvider is BraveSearchProvider
    assert Brave is BraveSearchProvider
    assert TopLevelBrave is BraveSearchProvider
    assert provider_registry.resolve("brave") is BraveSearchProvider


def test_maps_the_search_request_to_the_brave_query() -> None:
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
    assert str(sent.url).startswith("https://api.search.brave.com/res/v1/web/search?")
    assert sent.headers["x-subscription-token"] == API_KEY
    assert sent.headers["accept"] == "application/json"
    assert sent.headers["accept-encoding"] == "gzip"
    assert dict(sent.url.params) == {
        "q": f"{QUERY} site:example.com -site:news.example.org",
        "count": "7",
        "safesearch": "strict",
        "country": "DE",
        "search_lang": "en",
        "freshness": "2026-01-01to2026-01-31",
    }


def test_omits_unset_filters_and_clamps_the_result_limit() -> None:
    requests, handler = _capture()
    provider = _provider(handler)

    asyncio.run(provider.search(SearchRequest(query=QUERY, max_results=50)))

    params = dict(requests[0].url.params)
    assert params == {
        "q": QUERY,
        "count": "20",
        "safesearch": "moderate",
    }


def test_maps_recency_to_a_brave_freshness_code() -> None:
    requests, handler = _capture()
    provider = _provider(handler)

    asyncio.run(provider.search(SearchRequest(query=QUERY, recency=SearchRecency.WEEK)))

    assert requests[0].url.params["freshness"] == "pw"


def test_a_date_range_wins_over_the_recency_window() -> None:
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

    assert requests[0].url.params["freshness"] == "2026-02-01to2026-02-28"


def test_can_request_extra_snippets() -> None:
    requests, handler = _capture()
    provider = _provider(handler, extra_snippets=True)

    asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert requests[0].url.params["extra_snippets"] == "true"


def test_maps_results_query_and_paging_metadata_from_the_response() -> None:
    provider = _provider(lambda request: httpx.Response(200, json=_search_body()))

    response = asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert response.query == QUERY
    assert response.provider == "brave"
    assert response.answer is None
    assert response.total_results == 2
    assert response.took_ms is None
    assert response.request_id is None
    assert response.metadata["more_results_available"] is True

    first = response.results[0]
    assert first.title == "Tides explained"
    assert first.url == "https://example.com/tides"
    assert first.snippet == "A short snippet"
    assert first.content == "More detail\n\nEven more detail"
    assert first.text == "More detail\n\nEven more detail"
    assert first.published_at == datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    assert first.source == "brave"
    assert first.metadata == {
        "age": "3 days ago",
        "language": "en",
        "extra_snippets": ["More detail", "Even more detail"],
        "profile": {
            "name": "Example",
            "long_name": "Example Site",
            "img": "https://example.com/logo.png",
        },
        "meta_url": {
            "hostname": "example.com",
            "favicon": "https://example.com/favicon.ico",
        },
    }

    second = response.results[1]
    assert second.content == ""
    assert second.text == "Another snippet"
    assert second.published_at is None
    assert second.metadata == {}


def test_falls_back_to_the_request_query_when_the_body_omits_it() -> None:
    body = _search_body()
    body["query"] = {}
    provider = _provider(lambda request: httpx.Response(200, json=body))

    response = asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert response.query == QUERY


def test_a_response_without_a_web_object_has_no_results() -> None:
    provider = _provider(
        lambda request: httpx.Response(
            200,
            json={"type": "search", "query": {"original": QUERY}},
        )
    )

    response = asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert response.results == []
    assert response.total_results == 0


def test_uses_a_response_request_id_header_when_present() -> None:
    provider = _provider(
        lambda request: httpx.Response(
            200,
            json=_search_body(),
            headers={"x-request-id": "header-request"},
        )
    )

    response = asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert response.request_id == "header-request"


def test_resolves_the_api_key_from_a_secret_reference() -> None:
    requests, handler = _capture()
    loader = SecretsLoader(environ={"AIVA_BRAVE_API_KEY": API_KEY})
    provider = BraveSearchProvider(
        api_key_ref="BRAVE_API_KEY",
        secrets=loader,
        transport=httpx.MockTransport(handler),
    )

    asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert requests[0].headers["x-subscription-token"] == API_KEY


def test_requires_an_api_key() -> None:
    with pytest.raises(ConfigError):
        BraveSearchProvider()
    with pytest.raises(ConfigError):
        BraveSearchProvider(api_key="   ")


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
            json={
                "type": "ErrorResponse",
                "error": {
                    "detail": "request rejected",
                    "code": "REJECTED",
                },
            },
        )
    )

    with pytest.raises(expected) as caught:
        asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert caught.value.details["provider"] == "brave"
    assert caught.value.details["status"] == status


def test_maps_a_quota_message_on_another_status_to_rate_limited() -> None:
    provider = _provider(
        lambda request: httpx.Response(
            403,
            json={
                "type": "ErrorResponse",
                "error": {
                    "detail": "Your plan limit has been reached",
                    "code": "USAGE_LIMIT",
                },
            },
        )
    )

    with pytest.raises(RateLimitError):
        asyncio.run(provider.search(SearchRequest(query=QUERY)))


def test_error_details_carry_the_api_message_code_and_request_id() -> None:
    provider = _provider(
        lambda request: httpx.Response(
            429,
            json={
                "type": "ErrorResponse",
                "error": {
                    "detail": "Rate limit exceeded",
                    "code": "RATE_LIMITED",
                },
            },
            headers={
                "Retry-After": "3.5",
                "x-request-id": "err-request",
            },
        )
    )

    with pytest.raises(RateLimitError) as caught:
        asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert "Rate limit exceeded" in str(caught.value)
    assert caught.value.retry_after == 3.5
    assert caught.value.details["code"] == "RATE_LIMITED"
    assert caught.value.details["request_id"] == "err-request"


def test_rate_limit_falls_back_to_the_ratelimit_reset_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(brave_module, "_now", lambda: 1_700_000_000.0)
    provider = _provider(
        lambda request: httpx.Response(
            429,
            json={
                "type": "ErrorResponse",
                "error": {"detail": "slow down", "code": "RATE_LIMITED"},
            },
            headers={"X-RateLimit-Reset": "1700000030"},
        )
    )

    with pytest.raises(RateLimitError) as caught:
        asyncio.run(provider.search(SearchRequest(query=QUERY)))

    assert caught.value.retry_after == 30.0


def test_rejects_a_non_json_response() -> None:
    provider = _provider(lambda request: httpx.Response(200, text="not json"))

    with pytest.raises(ValidationError):
        asyncio.run(provider.search(SearchRequest(query=QUERY)))


@pytest.mark.parametrize(
    "body",
    [
        [],
        {"web": {"results": [{"title": "missing url"}]}},
        {"web": {"results": [{"url": "ftp://example.com/file"}]}},
    ],
)
def test_rejects_a_malformed_response(body: Any) -> None:
    provider = _provider(lambda request: httpx.Response(200, json=body))

    with pytest.raises(ValidationError):
        asyncio.run(provider.search(SearchRequest(query=QUERY)))

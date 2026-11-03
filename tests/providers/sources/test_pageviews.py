"""Wikipedia pageviews source: top pages, traffic signal, dates and access."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any

import httpx
import pytest

from aivia.core.errors import (
    AuthError,
    ConfigError,
    InvalidRequestError,
    RateLimitError,
    RetryableError,
    ValidationError,
)
from aivia.providers import provider_registry
from aivia.providers.protocols import CAPABILITY_SOURCE, SourceRequest
from aivia.providers.sources import (
    ACCESS_LEVELS,
    DEFAULT_ACCESS,
    DEFAULT_PAGEVIEWS_BASE_URL,
    Pageviews,
    WikipediaPageviews,
    WikipediaPageviewsAPI,
    WikipediaPageviewsProvider,
    WikipediaPageviewsSource,
    WikipediaPageviewsSourceProvider,
    normalize_access,
    pageviews_url,
    pv_url,
    resolve_source,
)

DATE = datetime(2026, 5, 5, 12, tzinfo=UTC)


def _article(title: str, views: Any, rank: int | None = None) -> dict[str, Any]:
    article: dict[str, Any] = {
        "article": title.replace(" ", "_"),
        "views": views,
    }
    if rank is not None:
        article["rank"] = rank
    return article


def _body(*articles: dict[str, Any]) -> dict[str, Any]:
    return {
        "items": [
            {
                "project": "en.wikipedia",
                "access": "all-access",
                "year": "2026",
                "month": "05",
                "day": "05",
                "articles": list(articles),
            }
        ]
    }


def _provider(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> WikipediaPageviewsSourceProvider:
    return WikipediaPageviewsSourceProvider(
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _capture(
    body: Any = None,
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> tuple[list[httpx.Request], Callable[[httpx.Request], httpx.Response]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status,
            json=_body() if body is None else body,
            headers=headers,
        )

    return requests, handler


def test_metadata_registration_and_aliases() -> None:
    provider = _provider(lambda request: httpx.Response(200, json=_body()))

    assert provider.name == "wikipedia-pageviews"
    assert provider.capabilities == (CAPABILITY_SOURCE,)
    assert provider.supports(CAPABILITY_SOURCE)
    assert provider.language == "en"
    assert provider.base_url == DEFAULT_PAGEVIEWS_BASE_URL
    assert provider.access == DEFAULT_ACCESS == "all-access"
    assert provider.endpoint == (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/"
        "en.wikipedia/all-access"
    )
    assert provider.api_url == provider.endpoint
    assert provider_registry.resolve("wikipedia-pageviews") is (
        WikipediaPageviewsSourceProvider
    )
    assert provider_registry.resolve("wikipedia_pageviews") is (
        WikipediaPageviewsSourceProvider
    )
    assert resolve_source("wikipedia-pageviews") is (
        WikipediaPageviewsSourceProvider
    )
    assert resolve_source("wikipedia_pageviews") is (
        WikipediaPageviewsSourceProvider
    )
    assert resolve_source("pageviews") is WikipediaPageviewsSourceProvider
    assert WikipediaPageviewsSource is WikipediaPageviewsSourceProvider
    assert WikipediaPageviewsProvider is WikipediaPageviewsSourceProvider
    assert WikipediaPageviewsAPI is WikipediaPageviewsSourceProvider
    assert WikipediaPageviews is WikipediaPageviewsSourceProvider
    assert Pageviews is WikipediaPageviewsSourceProvider


def test_maps_top_articles_with_a_traffic_signal() -> None:
    body = _body(
        _article("Tide", 123456, rank=1),
        _article("Ocean", 50000, rank=2),
        _article("Moon", 25000, rank=3),
    )
    requests, handler = _capture(body)
    provider = _provider(handler)

    response = asyncio.run(
        provider.discover(
            SourceRequest(
                limit=2,
                mode="trending",
                window_start=DATE,
            )
        )
    )

    sent = requests[0]
    assert sent.method == "GET"
    assert sent.url.path == (
        "/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/2026/05/05"
    )
    assert sent.headers["user-agent"]

    assert response.provider == "wikipedia-pageviews"
    assert response.titles == ["Tide", "Ocean"]
    assert [candidate.signal for candidate in response.candidates] == [
        123456.0,
        50000.0,
    ]
    assert [candidate.rank for candidate in response.candidates] == [0, 1]
    assert [candidate.mode for candidate in response.candidates] == [
        "trending",
        "trending",
    ]
    assert response.first is not None
    assert response.first.summary == "123,456 pageviews on 2026-05-05"
    assert response.first.source_urls == ["https://en.wikipedia.org/wiki/Tide"]
    assert response.metadata == {
        "language": "en",
        "project": "en.wikipedia",
        "access": "all-access",
        "date": "2026-05-05",
        "total": 2,
    }


def test_the_default_date_is_today_in_utc() -> None:
    requests, handler = _capture(_body(_article("Tide", 10, rank=1)))
    provider = _provider(handler)
    today = datetime.now(UTC).date()

    asyncio.run(provider.discover(SourceRequest()))

    assert requests[0].url.path.endswith(
        f"/en.wikipedia/all-access/{today.year}/{today.month:02d}/{today.day:02d}"
    )


def test_request_metadata_selects_a_date_and_access_level() -> None:
    requests, handler = _capture(_body(_article("Tide", 10, rank=1)))
    provider = _provider(handler)

    response = asyncio.run(
        provider.discover(
            SourceRequest(metadata={"date": "2026-05-06", "access": "desktop"})
        )
    )

    assert requests[0].url.path == (
        "/api/rest_v1/metrics/pageviews/top/en.wikipedia/desktop/2026/05/06"
    )
    assert response.metadata["access"] == "desktop"
    assert response.metadata["date"] == "2026-05-06"


def test_year_month_and_day_metadata_select_a_date() -> None:
    requests, handler = _capture(_body(_article("Tide", 10, rank=1)))
    provider = _provider(handler)

    response = asyncio.run(
        provider.discover(
            SourceRequest(metadata={"year": 2026, "month": 7, "day": 8})
        )
    )

    assert requests[0].url.path.endswith("/2026/07/08")
    assert response.metadata["date"] == "2026-07-08"


def test_a_slash_separated_date_is_accepted() -> None:
    requests, handler = _capture(_body(_article("Tide", 10, rank=1)))
    provider = _provider(handler)

    asyncio.run(provider.discover(SourceRequest(metadata={"date": "2026/7/8"})))

    assert requests[0].url.path.endswith("/2026/07/08")


def test_the_request_language_selects_the_edition() -> None:
    requests, handler = _capture(_body(_article("Deniz", 10, rank=1)))
    provider = _provider(handler, language="en")

    response = asyncio.run(
        provider.discover(SourceRequest(language="TR", window_start=DATE))
    )

    assert requests[0].url.path == (
        "/api/rest_v1/metrics/pageviews/top/tr.wikipedia/all-access/2026/05/05"
    )
    assert response.metadata["language"] == "tr"
    assert response.metadata["project"] == "tr.wikipedia"


def test_noise_namespaces_and_pages_without_views_are_skipped() -> None:
    body = _body(
        _article("Main Page", 999999, rank=1),
        _article("Special:Search", 900000, rank=2),
        _article("Portal:Current events", 800000, rank=3),
        _article("Wikipedia:Featured pictures", 700000, rank=4),
        _article("File:Example.jpg", 600000, rank=5),
        {"article": "Missing_views", "rank": 6},
        {"views": 500000, "rank": 7},
        _article("Moon landing", 400000, rank=8),
    )
    _requests, handler = _capture(body)
    provider = _provider(handler)

    response = asyncio.run(provider.discover(SourceRequest(window_start=DATE)))

    assert response.titles == ["Moon landing"]
    assert response.first is not None
    assert response.first.signal == 400000.0
    assert response.first.rank == 7


def test_duplicate_titles_are_removed_and_the_limit_applies_after_filtering() -> None:
    body = _body(
        _article("Main Page", 999999, rank=1),
        _article("Moon landing", 900000, rank=2),
        _article("moon  landing", 800000, rank=3),
        _article("Second topic", 700000, rank=4),
        _article("Third topic", 600000, rank=5),
    )
    _requests, handler = _capture(body)
    provider = _provider(handler)

    response = asyncio.run(
        provider.discover(SourceRequest(window_start=DATE, limit=2))
    )

    assert response.titles == ["Moon landing", "Second topic"]
    assert [candidate.signal for candidate in response.candidates] == [
        900000.0,
        700000.0,
    ]


def test_numeric_string_and_float_views_are_accepted() -> None:
    body = _body(
        _article("First", "1,234", rank=1),
        _article("Second", 12.0, rank=2),
    )
    _requests, handler = _capture(body)
    provider = _provider(handler)

    response = asyncio.run(provider.discover(SourceRequest(window_start=DATE)))

    assert [candidate.signal for candidate in response.candidates] == [1234.0, 12.0]


def test_top_level_articles_are_accepted() -> None:
    body = {"articles": [_article("Tide", 10, rank=1)]}
    _requests, handler = _capture(body)
    provider = _provider(handler)

    response = asyncio.run(provider.discover(SourceRequest(window_start=DATE)))

    assert response.titles == ["Tide"]


def test_an_empty_items_list_is_an_empty_response() -> None:
    _requests, handler = _capture({"items": []})
    provider = _provider(handler)

    response = asyncio.run(provider.discover(SourceRequest(window_start=DATE)))

    assert response.count == 0
    assert response.first is None


@pytest.mark.parametrize(
    "body",
    [
        {"unexpected": True},
        {"items": "not a list"},
        {"items": [{"articles": "not a list"}]},
        {"articles": {"not": "a list"}},
    ],
)
def test_a_malformed_response_is_a_validation_error(body: Any) -> None:
    _requests, handler = _capture(body)
    provider = _provider(handler)

    with pytest.raises(ValidationError, match="malformed"):
        asyncio.run(provider.discover(SourceRequest(window_start=DATE)))


def test_a_non_json_response_is_a_validation_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    provider = _provider(handler)

    with pytest.raises(ValidationError, match="non-JSON"):
        asyncio.run(provider.discover(SourceRequest(window_start=DATE)))


def test_a_missing_edition_is_a_retryable_error() -> None:
    _requests, handler = _capture({"error": "not found"}, status=404)
    provider = _provider(handler)

    with pytest.raises(RetryableError) as error:
        asyncio.run(
            provider.discover(SourceRequest(window_start=DATE, language="zz"))
        )

    assert error.value.retryable is True
    assert "zz" in str(error.value)
    assert error.value.details["language"] == "zz"
    assert error.value.details["status"] == 404


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (400, InvalidRequestError),
        (401, AuthError),
        (403, AuthError),
        (429, RateLimitError),
        (500, RetryableError),
        (503, RetryableError),
    ],
)
def test_http_errors_map_to_the_shared_taxonomy(
    status: int,
    error_type: type[Exception],
) -> None:
    _requests, handler = _capture({"error": "rejected"}, status=status)
    provider = _provider(handler)

    with pytest.raises(error_type):
        asyncio.run(provider.discover(SourceRequest(window_start=DATE)))


def test_an_api_error_about_the_language_is_retryable() -> None:
    body = {"error": {"code": "badvalue", "info": "Unknown language edition"}}
    _requests, handler = _capture(body)
    provider = _provider(handler)

    with pytest.raises(RetryableError, match="edition not found"):
        asyncio.run(provider.discover(SourceRequest(window_start=DATE)))


def test_the_request_id_is_carried_into_the_response() -> None:
    _requests, handler = _capture(
        _body(_article("Tide", 10, rank=1)),
        headers={"x-request-id": "req-pv-1"},
    )
    provider = _provider(handler)

    response = asyncio.run(provider.discover(SourceRequest(window_start=DATE)))

    assert response.request_id == "req-pv-1"
    assert response.metadata["request_id"] == "req-pv-1"


def test_fetch_and_collect_delegate_to_discover() -> None:
    requests, handler = _capture(_body(_article("Tide", 10, rank=1)))
    provider = _provider(handler)

    fetched = asyncio.run(provider.fetch(SourceRequest(window_start=DATE)))
    collected = asyncio.run(provider.collect(SourceRequest(window_start=DATE)))

    assert fetched.titles
    assert [candidate.title for candidate in collected] == fetched.titles
    assert len(requests) == 2


def test_endpoint_for_accepts_a_template_or_a_root() -> None:
    provider = _provider(lambda request: httpx.Response(200, json=_body()))

    assert provider.endpoint_for("de") == (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/"
        "de.wikipedia/all-access"
    )
    assert provider.endpoint_for(
        "de",
        access="desktop",
        date=date(2026, 5, 6),
    ) == (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/"
        "de.wikipedia/desktop/2026/05/06"
    )
    custom = _provider(
        lambda request: httpx.Response(200, json=_body()),
        base_url="https://wiki.example/{language}",
    )
    assert custom.endpoint_for("de") == (
        "https://wiki.example/de/pageviews/top/de.wikipedia/all-access"
    )


def test_module_level_url_helper_matches_the_provider() -> None:
    assert pageviews_url("de") == (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/"
        "de.wikipedia/all-access"
    )
    assert pageviews_url("de", access="desktop", date="2026-05-06") == (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/"
        "de.wikipedia/desktop/2026/05/06"
    )
    assert pageviews_url("de", year=2026, month=5, day=6) == (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/"
        "de.wikipedia/all-access/2026/05/06"
    )
    assert pv_url is pageviews_url


def test_normalize_access_expands_aliases_and_rejects_unknown_levels() -> None:
    assert normalize_access(None) == DEFAULT_ACCESS
    assert normalize_access("") == DEFAULT_ACCESS
    assert normalize_access("all") == "all-access"
    assert normalize_access("mobile") == "mobile-web"
    assert normalize_access("mobile_app") == "mobile-app"
    assert normalize_access("DESKTOP") == "desktop"
    assert set(ACCESS_LEVELS) == {
        "all-access",
        "desktop",
        "mobile-app",
        "mobile-web",
    }
    with pytest.raises(ConfigError, match="access"):
        normalize_access("tablet")


@pytest.mark.parametrize("access", ["tablet", 1])
def test_invalid_configured_access_is_rejected(access: Any) -> None:
    with pytest.raises(ConfigError, match="access"):
        WikipediaPageviewsSourceProvider(access=access)


def test_an_invalid_request_access_is_an_invalid_request() -> None:
    _requests, handler = _capture(_body(_article("Tide", 10, rank=1)))
    provider = _provider(handler)

    with pytest.raises(InvalidRequestError, match="access"):
        asyncio.run(
            provider.discover(
                SourceRequest(
                    window_start=DATE,
                    metadata={"access": "tablet"},
                )
            )
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"date": "not-a-date"},
        {"year": 2026, "month": 5},
        {"year": 2026, "month": 13, "day": 1},
    ],
)
def test_an_invalid_request_date_is_an_invalid_request(
    metadata: dict[str, Any],
) -> None:
    _requests, handler = _capture(_body(_article("Tide", 10, rank=1)))
    provider = _provider(handler)

    with pytest.raises(InvalidRequestError, match=r"date|year"):
        asyncio.run(
            provider.discover(SourceRequest(window_start=DATE, metadata=metadata))
        )


def test_constructor_accepts_a_date_or_parts_but_not_both() -> None:
    provider = WikipediaPageviewsSourceProvider(date="2026-05-05")
    parts = WikipediaPageviewsSourceProvider(year=2026, month=5, day=5)

    assert provider.date == date(2026, 5, 5)
    assert parts.date == date(2026, 5, 5)
    with pytest.raises(ConfigError, match="date"):
        WikipediaPageviewsSourceProvider(date="2026-05-05", year=2026)
    with pytest.raises(ConfigError, match="year"):
        WikipediaPageviewsSourceProvider(year=2026, month=5)
    with pytest.raises(ConfigError, match="date"):
        WikipediaPageviewsSourceProvider(date="not-a-date")


@pytest.mark.parametrize(
    "base_url",
    ["ftp://example.com", "https://user" + ":pass" + "@example.com", " "],
)
def test_base_url_must_be_a_credential_free_http_url(base_url: str) -> None:
    with pytest.raises(ConfigError, match="base_url"):
        WikipediaPageviewsSourceProvider(base_url=base_url)


@pytest.mark.parametrize("language", ["bad language", "-en", "en!", "a" * 31])
def test_invalid_language_editions_are_rejected(language: str) -> None:
    with pytest.raises(ConfigError, match="language"):
        WikipediaPageviewsSourceProvider(language=language)


def test_transport_errors_are_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("missing edition", request=request)

    provider = _provider(handler)

    with pytest.raises(RetryableError, match="transport error"):
        asyncio.run(provider.discover(SourceRequest(window_start=DATE)))

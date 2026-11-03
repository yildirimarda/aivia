"""Wikipedia On this day source: dates, structured years, categories, dedupe."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
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
    DEFAULT_OTD_BASE_URL,
    OTD_CATEGORIES,
    OnThisDay,
    WikipediaOnThisDay,
    WikipediaOnThisDayAPI,
    WikipediaOnThisDayProvider,
    WikipediaOnThisDaySource,
    WikipediaOnThisDaySourceProvider,
    WikipediaOTD,
    WikipediaOTDSource,
    normalize_categories,
    on_this_day_url,
    otd_url,
    resolve_source,
)

DATE = datetime(2026, 5, 5, 12, tzinfo=UTC)


def _page(
    title: str,
    *,
    normalized: str | None = None,
    extract: str = "",
    url: str | None = None,
) -> dict[str, Any]:
    page: dict[str, Any] = {
        "title": title,
        "titles": {"normalized": normalized or title.replace("_", " ")},
        "extract": extract or f"{title} article.",
    }
    if url is not None:
        page["content_urls"] = {"desktop": {"page": url}}
    return page


def _entry(
    text: str,
    year: int | None,
    *pages: dict[str, Any],
) -> dict[str, Any]:
    entry: dict[str, Any] = {"text": text, "pages": list(pages)}
    if year is not None:
        entry["year"] = year
    return entry


def _body(**sections: list[dict[str, Any]]) -> dict[str, Any]:
    return sections


def _all_body() -> dict[str, Any]:
    return _body(
        selected=[
            _entry(
                "1969 - Apollo 11 lands on the Moon.",
                1969,
                _page(
                    "Apollo_11",
                    url="https://en.wikipedia.org/wiki/Apollo_11",
                ),
            )
        ],
        births=[
            _entry(
                "1942 - A famous person is born.",
                1942,
                _page(
                    "Famous_person",
                    url="https://en.wikipedia.org/wiki/Famous_person",
                ),
            )
        ],
        deaths=[
            _entry(
                "1900 - A famous person dies.",
                1900,
                _page(
                    "Another_person",
                    url="https://en.wikipedia.org/wiki/Another_person",
                ),
            )
        ],
        events=[
            _entry(
                "1969 - Apollo 11 launches.",
                1969,
                _page(
                    "Apollo_11",
                    url="https://en.wikipedia.org/wiki/Apollo_11",
                ),
            )
        ],
        holidays=[
            _entry(
                "A public holiday.",
                None,
                _page(
                    "Public_holiday",
                    url="https://en.wikipedia.org/wiki/Public_holiday",
                ),
            )
        ],
    )


def _provider(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> WikipediaOnThisDaySourceProvider:
    return WikipediaOnThisDaySourceProvider(
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
            json=_all_body() if body is None else body,
            headers=headers,
        )

    return requests, handler


def test_metadata_registration_and_aliases() -> None:
    provider = _provider(lambda request: httpx.Response(200, json=_all_body()))

    assert provider.name == "wikipedia-otd"
    assert provider.capabilities == (CAPABILITY_SOURCE,)
    assert provider.supports(CAPABILITY_SOURCE)
    assert provider.language == "en"
    assert provider.base_url == DEFAULT_OTD_BASE_URL
    assert provider.categories == OTD_CATEGORIES
    assert provider.endpoint == (
        "https://api.wikimedia.org/feed/v1/wikipedia/en/onthisday"
    )
    assert provider.api_url == provider.endpoint
    assert provider_registry.resolve("wikipedia-otd") is (
        WikipediaOnThisDaySourceProvider
    )
    assert provider_registry.resolve("wikipedia_otd") is (
        WikipediaOnThisDaySourceProvider
    )
    assert resolve_source("wikipedia-otd") is WikipediaOnThisDaySourceProvider
    assert resolve_source("wikipedia_otd") is WikipediaOnThisDaySourceProvider
    assert resolve_source("wikipedia-on-this-day") is (WikipediaOnThisDaySourceProvider)
    assert WikipediaOnThisDaySource is WikipediaOnThisDaySourceProvider
    assert WikipediaOnThisDayProvider is WikipediaOnThisDaySourceProvider
    assert WikipediaOnThisDayAPI is WikipediaOnThisDaySourceProvider
    assert WikipediaOTDSource is WikipediaOnThisDaySourceProvider
    assert WikipediaOnThisDay is WikipediaOnThisDaySourceProvider
    assert WikipediaOTD is WikipediaOnThisDaySourceProvider
    assert OnThisDay is WikipediaOnThisDaySourceProvider


def test_maps_dated_entries_with_structured_year_and_category() -> None:
    requests, handler = _capture()
    provider = _provider(handler)

    response = asyncio.run(
        provider.discover(
            SourceRequest(
                query="ignored",
                limit=10,
                mode="on_this_day",
                window_start=DATE,
            )
        )
    )

    sent = requests[0]
    assert sent.method == "GET"
    assert sent.url.path == "/feed/v1/wikipedia/en/onthisday/all/05/05"
    assert sent.headers["user-agent"]

    assert response.provider == "wikipedia-otd"
    assert response.query == "ignored"
    assert response.metadata == {"language": "en", "total": 4}
    assert [candidate.title for candidate in response.candidates] == [
        "Apollo 11",
        "Famous person",
        "Another person",
        "Public holiday",
    ]
    assert [candidate.category for candidate in response.candidates] == [
        "selected",
        "births",
        "deaths",
        "holidays",
    ]
    assert [candidate.year for candidate in response.candidates] == [
        1969,
        1942,
        1900,
        None,
    ]
    assert [candidate.rank for candidate in response.candidates] == [0, 1, 2, 3]
    assert all(candidate.mode == "on_this_day" for candidate in response.candidates)
    assert response.first is not None
    assert response.first.summary.startswith("1969")
    assert response.first.source_urls == ["https://en.wikipedia.org/wiki/Apollo_11"]
    assert response.first.to_topic().year == 1969


def test_the_default_date_is_today_in_utc() -> None:
    requests, handler = _capture()
    provider = _provider(handler)
    today = datetime.now(UTC).date()

    asyncio.run(provider.discover(SourceRequest()))

    assert requests[0].url.path.endswith(
        f"/onthisday/all/{today.month:02d}/{today.day:02d}"
    )


def test_request_metadata_selects_a_date_and_one_category() -> None:
    body = _body(
        births=[
            _entry(
                "1942 - Born.",
                1942,
                _page("Born_person", url="https://en.wikipedia.org/wiki/Born_person"),
            )
        ],
        events=[
            _entry(
                "1969 - Happened.",
                1969,
                _page("Event_page", url="https://en.wikipedia.org/wiki/Event_page"),
            )
        ],
    )
    requests, handler = _capture(body)
    provider = _provider(handler)

    response = asyncio.run(
        provider.discover(
            SourceRequest(metadata={"month": 5, "day": 5, "category": "births"})
        )
    )

    assert requests[0].url.path == "/feed/v1/wikipedia/en/onthisday/births/05/05"
    assert response.titles == ["Born person"]
    assert response.first is not None
    assert response.first.category == "births"


def test_a_date_string_and_iso_date_are_accepted() -> None:
    requests, handler = _capture()
    provider = _provider(handler)

    asyncio.run(provider.discover(SourceRequest(metadata={"date": "5/6"})))
    asyncio.run(provider.discover(SourceRequest(metadata={"date": "2026-07-08"})))

    assert requests[0].url.path.endswith("/05/06")
    assert requests[1].url.path.endswith("/07/08")


def test_multiple_requested_categories_use_all_and_filter_the_body() -> None:
    requests, handler = _capture()
    provider = _provider(handler, categories=("events", "holidays"))

    response = asyncio.run(provider.discover(SourceRequest(window_start=DATE)))

    assert requests[0].url.path == "/feed/v1/wikipedia/en/onthisday/all/05/05"
    assert response.titles == ["Apollo 11", "Public holiday"]
    assert [candidate.category for candidate in response.candidates] == [
        "events",
        "holidays",
    ]


def test_cross_year_dedupe_keeps_the_first_normalized_title() -> None:
    body = _body(
        events=[
            _entry(
                "1999 - A landing.",
                1999,
                _page("Moon_landing", normalized="Moon landing"),
            ),
            _entry(
                "1969 - The first landing.",
                1969,
                _page("moon landing", normalized="  MOON   landing "),
            ),
            _entry(
                "2000 - Another topic.",
                2000,
                _page("Another_topic", normalized="Another topic"),
            ),
        ]
    )
    requests, handler = _capture(body)
    provider = _provider(handler)

    response = asyncio.run(provider.discover(SourceRequest(window_start=DATE)))

    assert response.titles == ["Moon landing", "Another topic"]
    assert [candidate.year for candidate in response.candidates] == [1999, 2000]
    assert [candidate.rank for candidate in response.candidates] == [0, 1]
    assert len(requests) == 1


def test_dedupe_runs_across_categories() -> None:
    body = _body(
        selected=[
            _entry(
                "1969 - Apollo 11 lands.",
                1969,
                _page("Apollo_11", url="https://en.wikipedia.org/wiki/Apollo_11"),
            )
        ],
        events=[
            _entry(
                "1969 - Apollo 11 launches.",
                1969,
                _page("Apollo_11", url="https://en.wikipedia.org/wiki/Apollo_11"),
            )
        ],
    )
    _requests, handler = _capture(body)
    provider = _provider(handler)

    response = asyncio.run(provider.discover(SourceRequest(window_start=DATE)))

    assert response.count == 1
    assert response.first is not None
    assert response.first.category == "selected"


def test_the_limit_applies_after_dedupe() -> None:
    body = _body(
        events=[
            _entry("1999 - Same.", 1999, _page("Same_topic")),
            _entry("1969 - Same.", 1969, _page("Same_topic")),
            _entry("2000 - Second.", 2000, _page("Second_topic")),
            _entry("2001 - Third.", 2001, _page("Third_topic")),
        ]
    )
    _requests, handler = _capture(body)
    provider = _provider(handler)

    response = asyncio.run(provider.discover(SourceRequest(window_start=DATE, limit=2)))

    assert response.titles == ["Same topic", "Second topic"]
    assert response.count == 2


def test_items_without_pages_are_skipped() -> None:
    body = _body(
        events=[
            {"text": "No pages.", "year": 1900},
            {"text": "Empty pages.", "year": 1901, "pages": []},
            _entry(
                "1903 - Kept.",
                1903,
                _page("Kept_page", url="https://en.wikipedia.org/wiki/Kept_page"),
            ),
        ]
    )
    _requests, handler = _capture(body)
    provider = _provider(handler)

    response = asyncio.run(provider.discover(SourceRequest(window_start=DATE)))

    assert response.titles == ["Kept page"]


def test_a_page_without_a_title_is_a_validation_error() -> None:
    body = _body(
        events=[{"text": "No title.", "year": 1902, "pages": [{"extract": "x"}]}]
    )
    _requests, handler = _capture(body)
    provider = _provider(handler)

    with pytest.raises(ValidationError, match="without a title"):
        asyncio.run(provider.discover(SourceRequest(window_start=DATE)))


def test_a_page_without_a_desktop_url_falls_back_to_the_article_url() -> None:
    body = _body(
        events=[
            _entry(
                "1900 - A topic.",
                1900,
                _page("Moon_and_tides"),
            )
        ]
    )
    _requests, handler = _capture(body)
    provider = _provider(handler, language="de")

    response = asyncio.run(provider.discover(SourceRequest(window_start=DATE)))

    assert response.first is not None
    assert response.first.source_urls == [
        "https://de.wikipedia.org/wiki/Moon_and_tides"
    ]


def test_every_page_url_of_an_entry_is_kept() -> None:
    body = _body(
        events=[
            _entry(
                "1900 - A topic.",
                1900,
                _page("First", url="https://en.wikipedia.org/wiki/First"),
                _page("Second", url="https://en.wikipedia.org/wiki/Second"),
            )
        ]
    )
    _requests, handler = _capture(body)
    provider = _provider(handler)

    response = asyncio.run(provider.discover(SourceRequest(window_start=DATE)))

    assert response.first is not None
    assert response.first.title == "First"
    assert response.first.source_urls == [
        "https://en.wikipedia.org/wiki/First",
        "https://en.wikipedia.org/wiki/Second",
    ]


def test_an_empty_body_section_is_an_empty_response() -> None:
    _requests, handler = _capture({"events": []})
    provider = _provider(handler)

    response = asyncio.run(provider.discover(SourceRequest(window_start=DATE)))

    assert response.count == 0
    assert response.first is None


@pytest.mark.parametrize("body", [{"unexpected": True}, ["not", "a", "mapping"]])
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
        asyncio.run(provider.discover(SourceRequest(window_start=DATE, language="zz")))

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
        _all_body(),
        headers={"x-request-id": "req-otd-1"},
    )
    provider = _provider(handler)

    response = asyncio.run(provider.discover(SourceRequest(window_start=DATE)))

    assert response.request_id == "req-otd-1"
    assert response.metadata["request_id"] == "req-otd-1"


def test_fetch_and_collect_delegate_to_discover() -> None:
    requests, handler = _capture()
    provider = _provider(handler)

    fetched = asyncio.run(provider.fetch(SourceRequest(window_start=DATE)))
    collected = asyncio.run(provider.collect(SourceRequest(window_start=DATE)))

    assert fetched.titles
    assert [candidate.title for candidate in collected] == fetched.titles
    assert len(requests) == 2


def test_endpoint_for_accepts_a_template_or_a_root() -> None:
    provider = _provider(lambda request: httpx.Response(200, json=_all_body()))

    assert provider.endpoint_for("de") == (
        "https://api.wikimedia.org/feed/v1/wikipedia/de/onthisday"
    )
    assert provider.endpoint_for("de", category="events", month=5, day=6) == (
        "https://api.wikimedia.org/feed/v1/wikipedia/de/onthisday/events/05/06"
    )
    custom = _provider(
        lambda request: httpx.Response(200, json=_all_body()),
        base_url="https://wiki.example/{language}",
    )
    assert custom.endpoint_for("de") == "https://wiki.example/de/onthisday"


def test_module_level_url_helper_matches_the_provider() -> None:
    assert on_this_day_url("de") == (
        "https://api.wikimedia.org/feed/v1/wikipedia/de/onthisday"
    )
    assert on_this_day_url("de", category="events", month=5, day=6) == (
        "https://api.wikimedia.org/feed/v1/wikipedia/de/onthisday/events/05/06"
    )
    assert otd_url is on_this_day_url


@pytest.mark.parametrize(
    "categories",
    ["unknown", ["events", "unknown"], [1]],
)
def test_invalid_categories_are_rejected(categories: Any) -> None:
    with pytest.raises(ConfigError, match="categor"):
        WikipediaOnThisDaySourceProvider(categories=categories)


def test_normalize_categories_expands_all_and_aliases() -> None:
    assert normalize_categories(None) == OTD_CATEGORIES
    assert normalize_categories("all") == OTD_CATEGORIES
    assert normalize_categories("event, holiday") == ("events", "holidays")
    assert normalize_categories(["birth"]) == ("births",)


def test_an_invalid_request_category_is_an_invalid_request() -> None:
    _requests, handler = _capture()
    provider = _provider(handler)

    with pytest.raises(InvalidRequestError, match="categor"):
        asyncio.run(
            provider.discover(
                SourceRequest(window_start=DATE, metadata={"category": "unknown"})
            )
        )


@pytest.mark.parametrize(
    ("month", "day"),
    [(5, None), (None, 5), (13, 1)],
)
def test_month_and_day_must_be_valid_and_together(
    month: int | None,
    day: int | None,
) -> None:
    with pytest.raises(ConfigError, match="month"):
        WikipediaOnThisDaySourceProvider(month=month, day=day)


@pytest.mark.parametrize(
    "base_url",
    ["ftp://example.com", "https://user" + ":pass" + "@example.com", " "],
)
def test_base_url_must_be_a_credential_free_http_url(base_url: str) -> None:
    with pytest.raises(ConfigError, match="base_url"):
        WikipediaOnThisDaySourceProvider(base_url=base_url)


@pytest.mark.parametrize("language", ["bad language", "-en", "en!", "a" * 31])
def test_invalid_language_editions_are_rejected(language: str) -> None:
    with pytest.raises(ConfigError, match="language"):
        WikipediaOnThisDaySourceProvider(language=language)


def test_transport_errors_are_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("missing edition", request=request)

    provider = _provider(handler)

    with pytest.raises(RetryableError, match="transport error"):
        asyncio.run(provider.discover(SourceRequest(window_start=DATE)))

"""Wikipedia source: language editions, extracts, links and error taxonomy."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from aivia.core.config.http import HttpClientConfig
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
    DEFAULT_BASE_URL,
    DEFAULT_LANGUAGE,
    Wikipedia,
    WikipediaAPI,
    WikipediaProvider,
    WikipediaSource,
    WikipediaSourceProvider,
    edition_url,
    normalize_language,
    resolve_source,
)

QUERY = "ocean tides"


def _action_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "batchcomplete": True,
        "query": {
            "pages": [
                {
                    "pageid": 101,
                    "ns": 0,
                    "title": "Tide",
                    "index": 1,
                    "extract": "Tides are the periodic rise and fall of sea level.",
                    "fullurl": "https://en.wikipedia.org/wiki/Tide",
                },
                {
                    "pageid": 102,
                    "ns": 0,
                    "title": "Ocean",
                    "index": 2,
                    "extract": "The ocean is a body of salt water.",
                    "fullurl": "https://en.wikipedia.org/wiki/Ocean",
                },
            ]
        },
    }
    body.update(overrides)
    return body


def _provider(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> WikipediaSourceProvider:
    return WikipediaSourceProvider(
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
            json=_action_body() if body is None else body,
            headers=headers,
        )

    return requests, handler


def test_metadata_registration_and_aliases() -> None:
    provider = _provider(lambda request: httpx.Response(200, json=_action_body()))

    assert provider.name == "wikipedia"
    assert provider.capabilities == (CAPABILITY_SOURCE,)
    assert provider.supports(CAPABILITY_SOURCE)
    assert provider.language == DEFAULT_LANGUAGE == "en"
    assert provider.base_url == DEFAULT_BASE_URL
    assert provider.endpoint == "https://en.wikipedia.org/w/api.php"
    assert provider.api_url == provider.endpoint
    assert provider_registry.resolve("wikipedia") is WikipediaSourceProvider
    assert resolve_source("wikipedia") is WikipediaSourceProvider
    assert WikipediaSource is WikipediaSourceProvider
    assert WikipediaProvider is WikipediaSourceProvider
    assert WikipediaAPI is WikipediaSourceProvider
    assert Wikipedia is WikipediaSourceProvider


def test_maps_a_search_request_and_returns_extracts_with_links() -> None:
    requests, handler = _capture()
    provider = _provider(handler)

    response = asyncio.run(
        provider.discover(SourceRequest(query=f"  {QUERY}  ", limit=2, mode="niche"))
    )

    sent = requests[0]
    assert sent.method == "GET"
    assert str(sent.url).startswith("https://en.wikipedia.org/w/api.php?")
    params = dict(sent.url.params)
    assert params["action"] == "query"
    assert params["format"] == "json"
    assert params["formatversion"] == "2"
    assert params["generator"] == "search"
    assert params["gsrsearch"] == QUERY
    assert params["gsrlimit"] == "2"
    assert params["gsrnamespace"] == "0"
    assert params["prop"] == "extracts|info"
    assert params["exintro"] == "1"
    assert params["explaintext"] == "1"
    assert params["exlimit"] == "max"
    assert params["inprop"] == "url"
    assert params["redirects"] == "1"
    assert sent.headers["user-agent"]

    assert response.provider == "wikipedia"
    assert response.query == QUERY
    assert response.count == 2
    assert response.titles == ["Tide", "Ocean"]
    assert response.candidates[0].summary.startswith("Tides are")
    assert response.candidates[0].source_urls == ["https://en.wikipedia.org/wiki/Tide"]
    assert response.candidates[0].mode == "niche"
    assert [candidate.rank for candidate in response.candidates] == [0, 1]
    assert response.metadata == {"language": "en", "total": 2}


def test_the_request_language_selects_the_edition() -> None:
    requests, handler = _capture()
    provider = _provider(handler, language="en")

    response = asyncio.run(
        provider.discover(SourceRequest(query="deniz", language="TR"))
    )

    assert str(requests[0].url).startswith("https://tr.wikipedia.org/w/api.php?")
    assert response.metadata["language"] == "tr"


def test_an_underscore_in_the_language_becomes_a_hyphen() -> None:
    requests, handler = _capture()
    provider = _provider(handler)

    asyncio.run(provider.discover(SourceRequest(query="x", language="zh_yue")))

    assert requests[0].url.host == "zh-yue.wikipedia.org"
    assert normalize_language("EN_us") == "en-us"


def test_the_limit_is_capped_for_the_api() -> None:
    requests, handler = _capture()
    provider = _provider(handler)

    asyncio.run(provider.discover(SourceRequest(query=QUERY, limit=200)))

    assert requests[0].url.params["gsrlimit"] == "50"


def test_formatversion_one_pages_are_accepted() -> None:
    body = {
        "query": {
            "pages": {
                "101": {
                    "pageid": 101,
                    "title": "Tide",
                    "extract": "Tides rise and fall.",
                    "fullurl": "https://en.wikipedia.org/wiki/Tide",
                }
            }
        }
    }
    _requests, handler = _capture(body)
    provider = _provider(handler)

    response = asyncio.run(provider.discover(SourceRequest(query=QUERY)))

    assert response.titles == ["Tide"]
    assert response.first is not None
    assert response.first.summary == "Tides rise and fall."


def test_rest_style_pages_and_html_excerpts_are_supported() -> None:
    body = {
        "pages": [
            {
                "id": 101,
                "key": "Tide",
                "title": "Tide",
                "excerpt": 'Tides <span class="searchmatch">rise</span> and fall.',
                "description": "Periodic sea level change",
                "url": "https://en.wikipedia.org/wiki/Tide",
            }
        ]
    }
    _requests, handler = _capture(body)
    provider = _provider(handler)

    response = asyncio.run(provider.discover(SourceRequest(query=QUERY)))

    assert response.first is not None
    assert response.first.summary == "Tides rise and fall."
    assert response.first.source_urls == ["https://en.wikipedia.org/wiki/Tide"]


def test_a_missing_full_url_is_built_from_the_language_and_title() -> None:
    body = {
        "query": {
            "pages": [
                {
                    "pageid": 101,
                    "title": "Moon and tides",
                    "extract": "The moon drives the tides.",
                }
            ]
        }
    }
    _requests, handler = _capture(body)
    provider = _provider(handler, language="de")

    response = asyncio.run(provider.discover(SourceRequest(query=QUERY)))

    assert response.first is not None
    assert response.first.source_urls == [
        "https://de.wikipedia.org/wiki/Moon_and_tides"
    ]


def test_missing_and_invalid_pages_are_skipped() -> None:
    body = {
        "query": {
            "pages": [
                {"title": "Gone", "missing": True},
                {"title": "Bad", "invalid": True},
                {
                    "title": "Kept",
                    "extract": "Still here.",
                    "fullurl": "https://en.wikipedia.org/wiki/Kept",
                },
            ]
        }
    }
    _requests, handler = _capture(body)
    provider = _provider(handler)

    response = asyncio.run(provider.discover(SourceRequest(query=QUERY)))

    assert response.titles == ["Kept"]


def test_an_empty_query_returns_no_candidates_without_a_request() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_action_body())

    provider = _provider(handler)

    response = asyncio.run(provider.discover(SourceRequest()))

    assert response.count == 0
    assert response.metadata == {"language": "en", "total": 0}
    assert requests == []


def test_no_results_is_an_empty_response() -> None:
    _requests, handler = _capture({"batchcomplete": True})
    provider = _provider(handler)

    response = asyncio.run(provider.discover(SourceRequest(query="nothing at all")))

    assert response.count == 0
    assert response.first is None


def test_a_missing_edition_is_a_retryable_error() -> None:
    _requests, handler = _capture({"error": "not found"}, status=404)
    provider = _provider(handler)

    with pytest.raises(RetryableError) as error:
        asyncio.run(provider.discover(SourceRequest(query=QUERY, language="zz")))

    assert error.value.code == "retryable"
    assert error.value.retryable is True
    assert "zz" in str(error.value)
    assert error.value.details["language"] == "zz"
    assert error.value.details["status"] == 404


def test_a_redirect_to_the_incubator_is_a_missing_edition() -> None:
    _requests, handler = _capture(
        {"error": "moved"},
        status=302,
        headers={"location": "https://incubator.wikimedia.org/wiki/Wp/zz/v1/search"},
    )
    provider = _provider(
        handler,
        http_config=HttpClientConfig(follow_redirects=False),
    )

    with pytest.raises(RetryableError, match="edition not found"):
        asyncio.run(provider.discover(SourceRequest(query=QUERY, language="zz")))


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
        asyncio.run(provider.discover(SourceRequest(query=QUERY)))


def test_api_errors_are_reported_as_invalid_requests() -> None:
    body = {"error": {"code": "badvalue", "info": "Unrecognized parameter"}}
    _requests, handler = _capture(body)
    provider = _provider(handler)

    with pytest.raises(InvalidRequestError, match="Unrecognized parameter"):
        asyncio.run(provider.discover(SourceRequest(query=QUERY)))


def test_an_api_error_about_the_language_is_retryable() -> None:
    body = {"error": {"code": "badvalue", "info": "Unknown language edition"}}
    _requests, handler = _capture(body)
    provider = _provider(handler)

    with pytest.raises(RetryableError, match="edition not found"):
        asyncio.run(provider.discover(SourceRequest(query=QUERY)))


def test_a_non_json_response_is_a_validation_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    provider = _provider(handler)

    with pytest.raises(ValidationError, match="non-JSON"):
        asyncio.run(provider.discover(SourceRequest(query=QUERY)))


@pytest.mark.parametrize("body", [{"unexpected": True}, ["not", "a", "mapping"]])
def test_a_malformed_response_is_a_validation_error(body: Any) -> None:
    _requests, handler = _capture(body)
    provider = _provider(handler)

    with pytest.raises(ValidationError, match="malformed"):
        asyncio.run(provider.discover(SourceRequest(query=QUERY)))


def test_a_page_without_a_title_is_a_validation_error() -> None:
    body = {"query": {"pages": [{"extract": "No title here."}]}}
    _requests, handler = _capture(body)
    provider = _provider(handler)

    with pytest.raises(ValidationError, match="without a title"):
        asyncio.run(provider.discover(SourceRequest(query=QUERY)))


def test_the_request_id_is_carried_into_the_response() -> None:
    _requests, handler = _capture(
        _action_body(),
        headers={"x-request-id": "req-wiki-1"},
    )
    provider = _provider(handler)

    response = asyncio.run(provider.discover(SourceRequest(query=QUERY)))

    assert response.request_id == "req-wiki-1"
    assert response.metadata["request_id"] == "req-wiki-1"


def test_fetch_and_collect_delegate_to_discover() -> None:
    requests, handler = _capture()
    provider = _provider(handler)

    fetched = asyncio.run(provider.fetch(SourceRequest(query=QUERY)))
    collected = asyncio.run(provider.collect(SourceRequest(query=QUERY)))

    assert fetched.titles == ["Tide", "Ocean"]
    assert [candidate.title for candidate in collected] == ["Tide", "Ocean"]
    assert len(requests) == 2


def test_edition_url_accepts_a_template_or_a_concrete_endpoint() -> None:
    assert edition_url() == "https://en.wikipedia.org/w/api.php"
    assert edition_url("tr") == "https://tr.wikipedia.org/w/api.php"
    assert (
        edition_url("de", "https://wiki.example/{language}")
        == "https://wiki.example/de/w/api.php"
    )
    assert (
        edition_url("de", "https://wiki.example/w/api.php")
        == "https://wiki.example/w/api.php"
    )


@pytest.mark.parametrize(
    "base_url",
    ["ftp://example.com", "https://user" + ":pass" + "@example.com", " "],
)
def test_base_url_must_be_a_credential_free_http_url(base_url: str) -> None:
    with pytest.raises(ConfigError, match="base_url"):
        WikipediaSourceProvider(base_url=base_url)


@pytest.mark.parametrize("language", ["bad language", "-en", "en!", "a" * 31])
def test_invalid_language_editions_are_rejected(language: str) -> None:
    with pytest.raises(ConfigError, match="language"):
        WikipediaSourceProvider(language=language)


def test_transport_errors_are_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("missing edition", request=request)

    provider = _provider(handler)

    with pytest.raises(RetryableError, match="transport error"):
        asyncio.run(provider.discover(SourceRequest(query=QUERY)))

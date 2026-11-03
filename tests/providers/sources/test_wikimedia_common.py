"""Shared Wikimedia request and error helpers used by every Wikimedia source."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
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
from aivia.core.errors import TimeoutError as AiviaTimeoutError
from aivia.core.http import HttpClient
from aivia.providers.sources import pageviews, wikimedia, wikipedia, wikipedia_otd
from aivia.providers.sources.wikimedia import (
    DEFAULT_LANGUAGE,
    DEFAULT_USER_AGENT,
    article_url,
    error_details,
    fetch_json,
    missing_edition,
    normalize_language,
    raise_api_error,
    raise_for_response,
    strip_html,
    validate_base_url,
)

PROVIDER = "wikipedia"
LANGUAGE = "en"
URL = "https://en.wikipedia.org/w/api.php"


def _response(
    status: int = 200,
    *,
    headers: Mapping[str, str] | None = None,
    body: Any = None,
    text: str | None = None,
) -> httpx.Response:
    request = httpx.Request("GET", URL)
    if text is not None:
        return httpx.Response(
            status,
            request=request,
            headers=headers,
            text=text,
        )
    return httpx.Response(status, request=request, headers=headers, json=body)


def _fetch(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> tuple[Mapping[str, Any], httpx.Response]:
    async def run() -> tuple[Mapping[str, Any], httpx.Response]:
        client = HttpClient(
            HttpClientConfig(max_attempts=1),
            transport=httpx.MockTransport(handler),
        )
        try:
            return await fetch_json(
                client,
                URL,
                provider=PROVIDER,
                language=LANGUAGE,
                headers={"accept": "application/json"},
                **kwargs,
            )
        finally:
            await client.aclose()

    return asyncio.run(run())


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, DEFAULT_LANGUAGE),
        ("", DEFAULT_LANGUAGE),
        ("  ", DEFAULT_LANGUAGE),
        ("TR", "tr"),
        ("EN_us", "en-us"),
        ("zh-yue", "zh-yue"),
    ],
)
def test_normalize_language_defaults_and_normalizes(
    value: str | None,
    expected: str,
) -> None:
    assert normalize_language(value) == expected


@pytest.mark.parametrize("value", ["bad language", "-en", "en!", "a" * 31])
def test_normalize_language_rejects_invalid_editions(value: str) -> None:
    with pytest.raises(ConfigError, match="language"):
        normalize_language(value)


def test_normalize_language_rejects_a_non_string() -> None:
    with pytest.raises(TypeError, match="language"):
        normalize_language(42)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [
        "https://{language}.wikipedia.org/w/api.php",
        "https://wiki.example/w/api.php",
        "http://localhost:8080/api",
    ],
)
def test_validate_base_url_accepts_templates_and_concrete_urls(value: str) -> None:
    assert validate_base_url(value) == value


@pytest.mark.parametrize(
    "value",
    ["ftp://example.com", "https://user" + ":pass" + "@example.com", " ", ""],
)
def test_validate_base_url_rejects_unsafe_urls(value: str) -> None:
    with pytest.raises(ConfigError, match="base_url"):
        validate_base_url(value)


def test_validate_base_url_rejects_a_non_string() -> None:
    with pytest.raises(TypeError, match="base_url"):
        validate_base_url(42)  # type: ignore[arg-type]


def test_article_url_quotes_the_title_for_the_edition() -> None:
    assert article_url("de", "Moon and tides") == (
        "https://de.wikipedia.org/wiki/Moon_and_tides"
    )
    assert article_url("en", "C++ (language)") == (
        "https://en.wikipedia.org/wiki/C%2B%2B_%28language%29"
    )


def test_strip_html_removes_tags_and_decodes_entities() -> None:
    assert strip_html('Tides <span class="searchmatch">rise</span> &amp; fall') == (
        "Tides rise & fall"
    )


def test_error_details_carry_provider_language_status_and_request_id() -> None:
    response = _response(404, headers={"x-request-id": "req-wiki-1"})

    assert error_details(response, "zz", PROVIDER) == {
        "provider": PROVIDER,
        "language": "zz",
        "status": 404,
        "request_id": "req-wiki-1",
    }


def test_error_details_omit_a_missing_request_id() -> None:
    assert "request_id" not in error_details(_response(500), LANGUAGE, PROVIDER)


def test_missing_edition_is_retryable_and_names_the_language() -> None:
    error = missing_edition("zz", _response(404), PROVIDER)

    assert isinstance(error, RetryableError)
    assert error.retryable is True
    assert "zz" in str(error)
    assert error.details == {
        "provider": PROVIDER,
        "language": "zz",
        "status": 404,
    }


def test_raise_for_response_treats_a_404_as_a_missing_edition() -> None:
    with pytest.raises(RetryableError, match="edition not found") as error:
        raise_for_response(_response(404), "zz", PROVIDER)

    assert error.value.details["language"] == "zz"
    assert error.value.details["status"] == 404


@pytest.mark.parametrize(
    "location",
    [
        "https://incubator.wikimedia.org/wiki/Wp/zz/v1/search",
        "https://zz.wikipedia.org/missing.php?title=Main_Page",
    ],
)
def test_raise_for_response_treats_a_redirect_to_incubator_as_missing(
    location: str,
) -> None:
    response = _response(302, headers={"location": location})

    with pytest.raises(RetryableError, match="edition not found"):
        raise_for_response(response, "zz", PROVIDER)


def test_raise_for_response_ignores_an_unrelated_redirect() -> None:
    response = _response(302, headers={"location": "https://example.com/moved"})

    raise_for_response(response, LANGUAGE, PROVIDER)


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (400, InvalidRequestError),
        (401, AuthError),
        (403, AuthError),
        (408, AiviaTimeoutError),
        (429, RateLimitError),
        (500, RetryableError),
        (503, RetryableError),
        (504, AiviaTimeoutError),
    ],
)
def test_raise_for_response_maps_statuses_to_the_shared_taxonomy(
    status: int,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type) as error:
        raise_for_response(_response(status), LANGUAGE, PROVIDER)

    assert error.value.details["status"] == status  # type: ignore[attr-defined]
    assert error.value.details["language"] == LANGUAGE  # type: ignore[attr-defined]


def test_raise_for_response_keeps_the_retry_after_delay() -> None:
    response = _response(429, headers={"retry-after": "12"})

    with pytest.raises(RateLimitError) as error:
        raise_for_response(response, LANGUAGE, PROVIDER)

    assert error.value.retry_after == pytest.approx(12.0)
    assert error.value.details["status"] == 429


def test_raise_for_response_returns_none_for_success() -> None:
    raise_for_response(_response(200), LANGUAGE, PROVIDER)


def test_raise_api_error_treats_a_language_problem_as_retryable() -> None:
    with pytest.raises(RetryableError, match="edition not found") as error:
        raise_api_error(
            {"code": "badvalue", "info": "Unknown language edition"},
            "zz",
            PROVIDER,
        )

    assert error.value.details["error"] == "Unknown language edition"


def test_raise_api_error_treats_other_errors_as_invalid_requests() -> None:
    with pytest.raises(InvalidRequestError, match="Unrecognized parameter") as error:
        raise_api_error(
            {"code": "badvalue", "info": "Unrecognized parameter"},
            LANGUAGE,
            PROVIDER,
        )

    assert error.value.details == {
        "provider": PROVIDER,
        "language": LANGUAGE,
        "code": "badvalue",
    }


def test_raise_api_error_falls_back_to_the_code() -> None:
    with pytest.raises(InvalidRequestError, match="badvalue"):
        raise_api_error({"code": "badvalue"}, LANGUAGE, PROVIDER)


def test_fetch_json_requests_and_returns_the_mapping_body() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response(
            200,
            headers={"x-request-id": "req-wiki-2"},
            body={"query": {"pages": []}},
        )

    body, response = _fetch(handler, params={"action": "query"}, timeout=5.0)

    assert body == {"query": {"pages": []}}
    assert response.headers["x-request-id"] == "req-wiki-2"
    sent = requests[0]
    assert sent.method == "GET"
    assert sent.headers["accept"] == "application/json"
    assert sent.url.params["action"] == "query"


def test_fetch_json_raises_http_errors_before_parsing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _response(404, body={"error": "not found"})

    with pytest.raises(RetryableError, match="edition not found"):
        _fetch(handler)


def test_fetch_json_rejects_a_non_json_body_with_the_label() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _response(200, text="<html>not json</html>")

    with pytest.raises(ValidationError, match="on this day returned a non-JSON"):
        _fetch(handler, label="wikipedia on this day")


def test_fetch_json_rejects_a_non_object_json_body_with_the_label() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _response(200, body=["not", "a", "mapping"])

    with pytest.raises(ValidationError, match="pageviews returned a malformed"):
        _fetch(handler, label="wikipedia pageviews")


def test_every_wikimedia_source_reuses_the_shared_helpers() -> None:
    for module in (wikipedia, wikipedia_otd, pageviews):
        assert module.fetch_json is wikimedia.fetch_json
        assert module.normalize_language is wikimedia.normalize_language
        assert module._validate_base_url is wikimedia.validate_base_url
        assert module._raise_api_error is wikimedia.raise_api_error
        assert module._article_url is wikimedia.article_url
        assert module.DEFAULT_USER_AGENT == DEFAULT_USER_AGENT

    assert wikipedia._strip_html is wikimedia.strip_html
    assert wikipedia_otd._strip_html is wikimedia.strip_html

"""Shared request and error helpers for the Wikimedia source adapters.

Search (the MediaWiki Action API), On this day (the Wikimedia feed API) and
Pageviews (the Wikimedia REST metrics API) all speak JSON over the same shared
HTTP client and share one failure vocabulary. This module owns that common
layer so an adapter does not reimplement language validation, base URL checks,
article link building, HTML stripping, response-to-error mapping or the
GET/parse/malformed-body sequence.

The helpers stay free of routing, retry and budget policy: transport and retry
behaviour come from :mod:`aivia.core.http`, and failures map to the shared
:mod:`aivia.core.errors` taxonomy.
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote, urlsplit

import httpx

from aivia.core.errors import (
    ConfigError,
    InvalidRequestError,
    RateLimitError,
    RetryableError,
    TimeoutError,
    ValidationError,
)
from aivia.core.http import HttpClient, error_for_response

__all__ = [
    "DEFAULT_LANGUAGE",
    "DEFAULT_USER_AGENT",
    "article_url",
    "error_details",
    "fetch_json",
    "missing_edition",
    "normalize_language",
    "raise_api_error",
    "raise_for_response",
    "strip_html",
    "validate_base_url",
]

DEFAULT_LANGUAGE = "en"
"""The language edition used when neither the provider nor request sets one."""

DEFAULT_USER_AGENT = "aivia/0.1 (https://github.com/yildirimarda/aivia)"
"""A descriptive User-Agent, as the Wikimedia API etiquette requests."""

_LANGUAGE_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_TAG_PATTERN = re.compile(r"<[^>]+>")
_EDITION_HINTS = (
    "edition",
    "language",
    "wiki",
    "not found",
    "missing",
    "does not exist",
)


def normalize_language(value: str | None) -> str:
    """Return a lower-case Wikimedia edition code or raise :class:`ConfigError`.

    An underscore is accepted as a separator because request aliases use it,
    but the host name Wikimedia expects uses hyphens.
    """
    if value is None:
        return DEFAULT_LANGUAGE
    if not isinstance(value, str):
        raise TypeError("language must be a string")
    text = value.strip().lower().replace("_", "-")
    if not text:
        return DEFAULT_LANGUAGE
    if len(text) > 30 or not _LANGUAGE_PATTERN.fullmatch(text):
        raise ConfigError(
            f"invalid Wikipedia language edition: {value!r}",
            details={"field": "language"},
        )
    return text


def validate_base_url(value: str) -> str:
    """Return a credential-free absolute http(s) URL or raise :class:`ConfigError`.

    A ``{language}`` template is accepted because it is filled with a validated
    edition before the request is sent.
    """
    if not isinstance(value, str):
        raise TypeError("base_url must be a string")
    text = value.strip()
    if not text:
        raise ConfigError(
            "base_url must be a non-empty URL",
            details={"field": "base_url"},
        )
    # Validate the concrete form; a template is filled before this check.
    parsed = urlsplit(text.replace("{language}", DEFAULT_LANGUAGE))
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ConfigError(
            f"base_url must be an absolute http(s) URL: {value!r}",
            details={"field": "base_url"},
        )
    if parsed.username is not None or parsed.password is not None:
        raise ConfigError(
            "base_url must not contain credentials",
            details={"field": "base_url"},
        )
    return text


def article_url(language: str, title: str) -> str:
    """Return the canonical Wikipedia article URL for *title* in *language*."""
    return (
        f"https://{language}.wikipedia.org/wiki/"
        f"{quote(title.replace(' ', '_'), safe='')}"
    )


def strip_html(value: str) -> str:
    """Return *value* with tags removed and HTML entities decoded."""
    return html.unescape(_TAG_PATTERN.sub("", value)).strip()


def error_details(
    response: httpx.Response,
    language: str,
    provider: str,
) -> dict[str, Any]:
    """Return the shared error detail fields for *response*."""
    details: dict[str, Any] = {
        "provider": provider,
        "language": language,
        "status": response.status_code,
    }
    request_id = response.headers.get("x-request-id")
    if request_id:
        details["request_id"] = request_id
    return details


def missing_edition(
    language: str,
    response: httpx.Response,
    provider: str,
) -> RetryableError:
    """Return the retryable error for a language edition that does not exist."""
    return RetryableError(
        f"Wikipedia edition not found: {language}",
        details=error_details(response, language, provider),
    )


def raise_for_response(
    response: httpx.Response,
    language: str,
    provider: str,
) -> None:
    """Raise the shared error for *response*, treating a missing edition as retryable.

    A 404 or a redirect to the incubator means the language edition does not
    exist. Every other unsuccessful status maps through
    :func:`aivia.core.http.error_for_response` so the research collector sees
    one error taxonomy across sources.
    """
    status = response.status_code
    if status == 404:
        raise missing_edition(language, response, provider)
    if 300 <= status < 400:
        location = response.headers.get("location", "").lower()
        if "incubator" in location or "missing.php" in location:
            raise missing_edition(language, response, provider)

    fallback = error_for_response(response)
    if fallback is None:
        return
    details = error_details(response, language, provider)
    if isinstance(fallback, RateLimitError):
        raise RateLimitError(
            fallback.message,
            retry_after=fallback.retry_after,
            details=details,
        )
    if isinstance(fallback, TimeoutError):
        raise TimeoutError(fallback.message, details=details)
    raise type(fallback)(fallback.message, details=details)


def raise_api_error(
    error: Mapping[str, Any],
    language: str,
    provider: str,
) -> None:
    """Raise the shared error for a JSON ``error`` object from Wikimedia.

    An error that mentions the edition, language or wiki is retryable because
    the research collector can fall back to another source. Everything else is
    an invalid request.
    """
    code = str(error.get("code", "")).strip()
    info = str(error.get("info", "")).strip()
    text = info or code or "Wikipedia API error"
    lowered = text.lower()
    if any(hint in lowered for hint in _EDITION_HINTS):
        raise RetryableError(
            f"Wikipedia edition not found: {language}",
            details={"provider": provider, "language": language, "error": text},
        )
    raise InvalidRequestError(
        f"Wikipedia rejected the request: {text}",
        details={"provider": provider, "language": language, "code": code},
    )


async def fetch_json(
    client: HttpClient,
    url: str,
    *,
    provider: str,
    language: str,
    headers: Mapping[str, str],
    params: Mapping[str, Any] | None = None,
    timeout: float | None = None,
    label: str = "wikipedia",
) -> tuple[Mapping[str, Any], httpx.Response]:
    """GET *url* and return its JSON object body together with the response.

    The request goes through the shared client, so retries and transport
    failures keep their policy. The response is checked with
    :func:`raise_for_response` before the body is parsed. A non-JSON body or a
    JSON value that is not an object becomes a :class:`ValidationError` naming
    *label* so each adapter keeps its own diagnostics.
    """
    response = await client.get(
        url,
        headers=headers,
        params=params,
        timeout=timeout,
    )
    raise_for_response(response, language, provider)

    details = {"provider": provider, "language": language}
    try:
        body = response.json()
    except ValueError as exc:
        raise ValidationError(
            f"{label} returned a non-JSON response",
            details=details,
        ) from exc
    if not isinstance(body, Mapping):
        raise ValidationError(
            f"{label} returned a malformed response",
            details=details,
        )
    return body, response

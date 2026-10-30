"""Helpers shared by the stock media adapters.

Pexels and Pixabay both speak a small JSON search API and both report failures
as an HTTP status plus an optional error string. The helpers here keep base URL
validation, API key resolution, error mapping and paging in one place so the
two adapters differ only in their request and result mapping.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx

from aivia.core.errors import (
    ConfigError,
    RateLimitError,
)
from aivia.core.errors import (
    TimeoutError as AiviaTimeoutError,
)
from aivia.core.http import error_for_response
from aivia.core.secrets import SecretsLoader

__all__ = [
    "error_message",
    "next_page",
    "raise_for_response",
    "resolve_api_key",
    "validate_base_url",
]


def validate_base_url(value: str, *, label: str) -> str:
    """Validate and normalize an adapter's API root."""
    if not isinstance(value, str):
        raise TypeError("base_url must be a string")
    text = value.strip().rstrip("/")
    parsed = urlsplit(text)
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


def resolve_api_key(
    provider: str,
    api_key: str | None,
    api_key_ref: str | None,
    secrets: SecretsLoader | None,
) -> str:
    """Return an inline API key or resolve one through *secrets*."""
    if secrets is not None and not isinstance(secrets, SecretsLoader):
        raise TypeError("secrets must be a SecretsLoader")
    if api_key is not None:
        if not isinstance(api_key, str):
            raise TypeError("api_key must be a string")
        text = api_key.strip()
        if not text:
            raise ConfigError(f"{provider} API key must be non-empty")
        return text
    if api_key_ref is None or not isinstance(api_key_ref, str):
        raise ConfigError(
            f"{provider} provider requires an API key or api_key_ref",
            details={"field": "api_key_ref"},
        )
    reference = api_key_ref.strip()
    if not reference:
        raise ConfigError(
            f"{provider} provider requires an API key or api_key_ref",
            details={"field": "api_key_ref"},
        )
    loader = secrets or SecretsLoader()
    return loader.get(reference)


def error_message(response: httpx.Response) -> str | None:
    """Return a short error string from a provider response body, if any."""
    try:
        body = response.json()
    except ValueError:
        return None
    if not isinstance(body, Mapping):
        return None
    error = body.get("error")
    if isinstance(error, Mapping):
        for key in ("message", "detail", "error"):
            value = error.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for key in ("error", "message", "detail"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def raise_for_response(response: httpx.Response, *, provider: str) -> None:
    """Raise the shared error for an unsuccessful provider response."""
    fallback = error_for_response(response)
    if fallback is None:
        return

    details: dict[str, Any] = {
        "provider": provider,
        "status": response.status_code,
    }
    request_id = response.headers.get("x-request-id")
    if request_id:
        details["request_id"] = request_id

    message = error_message(response)
    text = fallback.message if message is None else f"{fallback.message}: {message}"
    if isinstance(fallback, RateLimitError):
        raise RateLimitError(
            text,
            retry_after=fallback.retry_after,
            details=details,
        )
    if isinstance(fallback, AiviaTimeoutError):
        raise AiviaTimeoutError(text, details=details)
    raise type(fallback)(text, details=details)


def next_page(value: Any, current: int) -> int | None:
    """Normalize a provider next-page value to an integer page number."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            number = int(text)
            return number if number > 0 else None
        try:
            params = parse_qs(urlsplit(text).query)
        except ValueError:
            return None
        raw = params.get("page", [None])[0]
        if raw is not None and str(raw).isdigit():
            number = int(raw)
            return number if number > 0 else None
        return current + 1
    return None

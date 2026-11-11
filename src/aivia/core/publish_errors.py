"""Per-platform error taxonomy for publishing.

Platform HTTP errors (status codes and error bodies) are normalized into the
shared error taxonomy in :mod:`aivia.core.errors` so the router can apply a
single retry/backoff policy regardless of which platform rejected a post.

Each platform may override the default status mapping through
:data:`PLATFORM_ERROR_MAP`. Unknown status codes fall back to
:class:`~aivia.core.errors.RetryableError` when the body signals a transient
failure and :class:`~aivia.core.errors.InvalidRequestError` otherwise.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aivia.core.errors import (
    AiviaError,
    AuthError,
    InvalidRequestError,
    QuotaError,
    RateLimitError,
    RetryableError,
    TimeoutError,
)

__all__ = [
    "PLATFORM_ERROR_MAP",
    "PublishError",
    "classify_error",
    "default_status_map",
]


class PublishError(AiviaError):
    """Base error raised when a publish attempt fails.

    ``platform`` names the platform that rejected the post so callers can scope
    alerts and retries per account.
    """

    code = "publish"
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        platform: str = "",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        merged: dict[str, Any] = dict(details or {})
        if platform:
            merged.setdefault("platform", platform)
        super().__init__(message, details=merged)
        self.platform = platform


def default_status_map() -> dict[int, type[AiviaError]]:
    """Return the default HTTP status to error-class mapping.

    The mapping is shared by every platform unless overridden in
    :data:`PLATFORM_ERROR_MAP`.
    """
    return {
        400: InvalidRequestError,
        401: AuthError,
        402: QuotaError,
        403: AuthError,
        404: InvalidRequestError,
        409: InvalidRequestError,
        422: InvalidRequestError,
        423: QuotaError,
        429: RateLimitError,
        500: RetryableError,
        502: RetryableError,
        503: RetryableError,
        504: TimeoutError,
    }


PLATFORM_ERROR_MAP: dict[str, dict[int, type[AiviaError]]] = {}
"""Per-platform overrides of the default status mapping.

Keys are normalised platform ids; values map HTTP status codes to error
classes from the shared taxonomy. A platform absent from this map uses
:func:`default_status_map`.
"""


def _body_message(body: Any) -> str:
    if isinstance(body, str):
        return body
    if isinstance(body, Mapping):
        for key in ("message", "error", "error_description", "reason"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _transient_signal(body: Any) -> bool:
    if isinstance(body, Mapping):
        for key in ("error_type", "type", "code"):
            value = str(body.get(key, ""))
            if any(
                token in value.lower()
                for token in ("rate", "throttle", "quota", "temporary")
            ):
                return True
    text = str(body).lower()
    return any(
        token in text for token in ("rate limit", "throttle", "try again", "temporary")
    )


def classify_error(
    platform: str,
    status: int,
    *,
    body: Any = None,
    message: str = "",
) -> AiviaError:
    """Classify a platform HTTP error into the shared taxonomy.

    *platform* is normalised like :func:`aivia.core.platforms._lookup_name`.
    *status* is the HTTP status code. *body* is the parsed response body, used
    both to refine the message and to detect transient failures for unknown
    status codes.

    Returns an instance of the matching error class (auth, rate limit, invalid
    request, quota, timeout, or a retryable/permanent fallback) scoped with the
    platform in its details.
    """
    key = platform.strip().lower().replace("-", "_").replace(" ", "_")
    overrides = PLATFORM_ERROR_MAP.get(key, {})
    error_class = overrides.get(status, default_status_map().get(status))

    if error_class is None:
        error_class = RetryableError if _transient_signal(body) else InvalidRequestError

    text = message or _body_message(body) or f"platform {platform} rejected the request"
    details: dict[str, Any] = {
        "platform": key,
        "status": status,
    }
    if body is not None:
        details["body"] = body
    return error_class(text, details=details)


def error_for_result(
    platform: str,
    status: int,
    *,
    body: Any = None,
    message: str = "",
) -> AiviaError | None:
    """Return the shared-taxonomy error for a failed result, or ``None`` for success.

    Convenience wrapper around :func:`classify_error` used by publish adapters
    that translate an HTTP response into a typed error before raising. A
    successful status (2xx) returns ``None``.
    """
    if 200 <= status < 300:
        return None
    return classify_error(platform, status, body=body, message=message)

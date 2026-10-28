"""Shared asynchronous HTTP client for every provider adapter.

Adapters must not each invent their own retry loop, timeout handling or log
format. This module owns the transport policy in one place: a request is
retried on transport failures, timeouts and a configurable set of HTTP status
codes with exponential backoff and jitter, every attempt is bounded by a
deadline, and request and response headers are logged only after redaction.

The underlying :class:`httpx.AsyncClient` is built with an injectable
``transport`` so the offline suite can use :class:`httpx.MockTransport`
without touching the network. ``sleep``, ``rng`` and ``clock`` are injectable
for the same reason: tests can advance retries without waiting in real time.

The client returns the final :class:`httpx.Response` by default, including a
response whose retryable status survived every attempt. Pass
``raise_for_status=True`` (or call :func:`raise_for_status`) to convert it to
the shared :mod:`aivia.core.errors` taxonomy.

Retries are method aware: methods that HTTP defines as idempotent are retried
by default, while ``POST`` (and any other non-idempotent method) makes exactly
one attempt unless the caller opts in with ``retry=True``. ``retry=False``
always forces a single attempt.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from types import TracebackType
from typing import Any, Self

import httpx

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import (
    AiviaError,
    AuthError,
    BudgetError,
    InvalidRequestError,
    RateLimitError,
    RetryableError,
)
from aivia.core.errors import TimeoutError as AiviaTimeoutError
from aivia.core.logging import get_logger
from aivia.core.redact import REDACTED, redact

__all__ = [
    "SENSITIVE_HEADERS",
    "AsyncHttpClient",
    "HttpClient",
    "HttpClientConfig",
    "error_for_response",
    "raise_for_status",
    "redact_headers",
]

SENSITIVE_HEADERS: frozenset[str] = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "api-key",
        "apikey",
        "x-auth-token",
        "x-access-token",
        "x-goog-api-key",
        "x-amz-security-token",
        "x-csrf-token",
        "x-xsrf-token",
    }
)

_SENSITIVE_PARTS = (
    "authorization",
    "cookie",
    "token",
    "secret",
    "password",
    "passwd",
    "api-key",
    "apikey",
    "api_key",
    "private-key",
)

_IDEMPOTENT_METHODS: frozenset[str] = frozenset(
    {"GET", "HEAD", "OPTIONS", "PUT", "DELETE", "TRACE"}
)


def _is_sensitive_header(name: str) -> bool:
    lowered = name.strip().lower()
    if lowered in SENSITIVE_HEADERS:
        return True
    normalised = lowered.replace("_", "-")
    return any(part in normalised for part in _SENSITIVE_PARTS)


def redact_headers(headers: Mapping[str, str] | httpx.Headers) -> dict[str, str]:
    """Return a copy of *headers* safe to log.

    Known credential headers are replaced entirely with ``[REDACTED]``.
    Every other value still passes through :func:`aivia.core.redact.redact`
    so a token accidentally placed in an unusual header is masked as well.
    Header names are preserved so a log reader can still tell which header was
    sent.
    """
    result: dict[str, str] = {}
    for name, value in headers.items():
        if _is_sensitive_header(name):
            result[name] = REDACTED
        else:
            result[name] = redact(value)
    return result


def _retry_after_seconds(response: httpx.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if raw is None or not raw.strip():
        return None
    value = raw.strip()
    try:
        delay_value = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        return max(0.0, (when - datetime.now(UTC)).total_seconds())
    if not math.isfinite(delay_value) or delay_value < 0:
        return None
    return delay_value


def error_for_response(response: httpx.Response) -> AiviaError | None:
    """Map an unsuccessful response to the shared error taxonomy.

    A response below ``400`` is not an error and returns ``None``. ``402`` maps
    to a budget error, ``401``/``403``/``407`` to an auth error, ``408`` and
    ``504`` to a timeout, ``429`` to a rate limit (carrying ``Retry-After``)
    and every other ``5xx`` to a retryable error. Remaining ``4xx`` responses
    are invalid requests. The message never includes the response body.
    """
    status = response.status_code
    if status < 400:
        return None

    request = response.request
    details: dict[str, Any] = {
        "status": status,
        "method": request.method,
        "url": redact(str(request.url)),
    }
    reason = response.reason_phrase or "HTTP error"
    message = f"HTTP {status} {reason}".strip()

    if status in {401, 403, 407}:
        return AuthError(message, details=details)
    if status == 402:
        return BudgetError(message, details=details)
    if status == 429:
        return RateLimitError(
            message,
            retry_after=_retry_after_seconds(response),
            details=details,
        )
    if status in {408, 504}:
        return AiviaTimeoutError(message, details=details)
    if status >= 500:
        return RetryableError(message, details=details)
    return InvalidRequestError(message, details=details)


async def raise_for_status(response: httpx.Response) -> None:
    """Raise the mapped :mod:`aivia.core.errors` error for *response*."""
    error = error_for_response(response)
    if error is not None:
        raise error


_raise_for_status = raise_for_status


class HttpClient:
    """A retrying async HTTP client shared by every provider adapter.

    ``config`` carries the retry, backoff and timeout policy. ``transport`` is
    passed straight to :class:`httpx.AsyncClient`, which is how tests inject a
    mock transport. ``headers`` and ``base_url`` are client defaults.
    ``logger`` defaults to this module's logger, and ``sleep``, ``rng`` and
    ``clock`` default to the real implementations but can be replaced in tests.
    """

    def __init__(
        self,
        config: HttpClientConfig | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        base_url: str | httpx.URL = "",
        headers: Mapping[str, str] | None = None,
        logger: logging.Logger | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        rng: Callable[[], float] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if config is not None and not isinstance(config, HttpClientConfig):
            raise TypeError("config must be an HttpClientConfig")
        if transport is not None and not isinstance(
            transport, httpx.AsyncBaseTransport
        ):
            raise TypeError("transport must be an httpx.AsyncBaseTransport")
        if logger is not None and not isinstance(logger, logging.Logger):
            raise TypeError("logger must be a logging.Logger")
        for label, value in (
            ("sleep", sleep),
            ("rng", rng),
            ("clock", clock),
        ):
            if value is not None and not callable(value):
                raise TypeError(f"{label} must be callable")

        self._config = config or HttpClientConfig()
        self._logger = logger or get_logger(__name__)
        self._sleep = sleep or asyncio.sleep
        self._rng = rng or random.random
        self._clock = clock or time.monotonic
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            transport=transport,
            timeout=self._timeout_for(None),
            follow_redirects=self._config.follow_redirects,
        )

    @property
    def config(self) -> HttpClientConfig:
        """The retry, backoff and timeout policy in use."""
        return self._config

    @property
    def client(self) -> httpx.AsyncClient:
        """The underlying client, exposed for streaming responses."""
        return self._client

    def _timeout_for(self, timeout: float | httpx.Timeout | None) -> httpx.Timeout:
        if isinstance(timeout, httpx.Timeout):
            return timeout
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise TypeError("timeout must be a number or httpx.Timeout")
            if not math.isfinite(float(timeout)) or float(timeout) <= 0:
                raise ValueError("timeout must be a finite positive number")
            return httpx.Timeout(float(timeout))
        return httpx.Timeout(
            self._config.timeout_seconds,
            connect=self._config.connect_timeout_seconds,
            read=self._config.read_timeout_seconds,
            write=self._config.write_timeout_seconds,
            pool=self._config.pool_timeout_seconds,
        )

    def _deadline_for(self, timeout: float | httpx.Timeout | None) -> float:
        if isinstance(timeout, httpx.Timeout):
            values = [
                value
                for value in (
                    timeout.connect,
                    timeout.read,
                    timeout.write,
                    timeout.pool,
                )
                if value is not None
            ]
            return max(values) if values else self._config.timeout_seconds
        if timeout is not None:
            return float(timeout)
        return self._config.timeout_seconds

    def _backoff_delay(self, attempt: int, retry_after: float | None) -> float:
        """Return the wait before the next attempt.

        ``Retry-After`` wins when the server supplied one and the config allows
        it. Otherwise the delay grows exponentially from
        ``backoff_base_seconds`` up to ``backoff_max_seconds``. Jitter removes
        a random fraction of the step, bounded by ``jitter_ratio``, so many
        clients do not retry in lockstep.
        """
        if retry_after is not None and self._config.respect_retry_after:
            return retry_after
        exponent = max(0, attempt - 1)
        ceiling: float = min(
            float(self._config.backoff_max_seconds),
            float(self._config.backoff_base_seconds) * (2**exponent),
        )
        if ceiling <= 0:
            return 0.0
        fraction: float = float(self._config.jitter_ratio)
        if fraction <= 0:
            return ceiling
        sample = max(0.0, min(1.0, float(self._rng())))
        return ceiling * (1.0 - fraction + fraction * sample)

    def _log_request(
        self,
        request: httpx.Request,
        attempt: int,
        attempts: int,
    ) -> None:
        self._logger.debug(
            "HTTP request %s %s attempt=%d/%d headers=%s",
            request.method,
            redact(str(request.url)),
            attempt,
            attempts,
            redact_headers(request.headers),
        )

    def _log_response(
        self,
        response: httpx.Response,
        attempt: int,
        attempts: int,
        *,
        retrying: bool,
    ) -> None:
        level = logging.WARNING if retrying else logging.DEBUG
        self._logger.log(
            level,
            "HTTP response %s %s status=%d attempt=%d/%d headers=%s",
            response.request.method,
            redact(str(response.request.url)),
            response.status_code,
            attempt,
            attempts,
            redact_headers(response.headers),
        )

    def _timeout_error(
        self,
        request: httpx.Request,
        attempt: int,
    ) -> AiviaTimeoutError:
        return AiviaTimeoutError(
            "HTTP request timed out",
            details={
                "method": request.method,
                "url": redact(str(request.url)),
                "attempt": attempt,
            },
        )

    def _transport_error(
        self,
        request: httpx.Request,
        exc: httpx.TransportError,
        attempt: int,
    ) -> RetryableError:
        return RetryableError(
            f"HTTP transport error: {type(exc).__name__}",
            details={
                "method": request.method,
                "url": redact(str(request.url)),
                "attempt": attempt,
                "error": type(exc).__name__,
            },
        )

    def _retries_enabled(self, method: str, retry: bool | None) -> bool:
        """Return whether *method* may be attempted more than once.

        ``retry=True`` opts any method in, ``retry=False`` forces a single
        attempt, and ``None`` uses the method's idempotency: safe and idempotent
        methods retry, while ``POST`` and other non-idempotent methods do not.
        """
        if retry is not None:
            if not isinstance(retry, bool):
                raise TypeError("retry must be a boolean or None")
            return retry
        return method.upper() in _IDEMPOTENT_METHODS

    async def request(
        self,
        method: str,
        url: str | httpx.URL,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, Any] | None = None,
        content: Any = None,
        data: Any = None,
        files: Any = None,
        json: Any = None,
        timeout: float | httpx.Timeout | None = None,
        retry: bool | None = None,
        raise_for_status: bool = False,
        follow_redirects: bool | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Send a request, retrying transient failures with jittered backoff.

        Each attempt is bounded by ``timeout`` (or the configured deadline).
        Transport errors, timeouts and responses whose status is in
        ``retry_statuses`` are retried until ``max_attempts`` is reached. The
        final response is returned; pass ``raise_for_status=True`` to raise the
        mapped error instead.

        Retries are method aware. ``retry=None`` (the default) retries only
        idempotent methods, so a ``POST`` makes exactly one attempt unless the
        caller passes ``retry=True``. ``retry=False`` always makes one attempt.
        """
        timeout_config = self._timeout_for(timeout)
        deadline = self._deadline_for(timeout)
        attempts = (
            self._config.max_attempts if self._retries_enabled(method, retry) else 1
        )
        redirects = (
            self._config.follow_redirects
            if follow_redirects is None
            else follow_redirects
        )

        for attempt in range(1, attempts + 1):
            request = self._client.build_request(
                method,
                url,
                headers=headers,
                params=params,
                content=content,
                data=data,
                files=files,
                json=json,
                timeout=timeout_config,
                **kwargs,
            )
            self._log_request(request, attempt, attempts)
            try:
                async with asyncio.timeout(deadline):
                    response = await self._client.send(
                        request,
                        follow_redirects=redirects,
                    )
            except (TimeoutError, httpx.TimeoutException):
                if attempt >= attempts:
                    raise self._timeout_error(request, attempt) from None
                delay = self._backoff_delay(attempt, None)
                self._logger.warning(
                    "HTTP request timed out %s %s attempt=%d/%d retrying_in=%.3fs",
                    request.method,
                    redact(str(request.url)),
                    attempt,
                    attempts,
                    delay,
                )
                await self._sleep(delay)
                continue
            except httpx.TransportError as exc:
                if attempt >= attempts:
                    raise self._transport_error(request, exc, attempt) from exc
                delay = self._backoff_delay(attempt, None)
                self._logger.warning(
                    "HTTP transport error %s %s attempt=%d/%d error=%s "
                    "retrying_in=%.3fs",
                    request.method,
                    redact(str(request.url)),
                    attempt,
                    attempts,
                    type(exc).__name__,
                    delay,
                )
                await self._sleep(delay)
                continue

            retryable_status = response.status_code in self._config.retry_statuses
            if retryable_status and attempt < attempts:
                retry_after = _retry_after_seconds(response)
                delay = self._backoff_delay(attempt, retry_after)
                self._log_response(
                    response,
                    attempt,
                    attempts,
                    retrying=True,
                )
                await response.aclose()
                await self._sleep(delay)
                continue

            self._log_response(
                response,
                attempt,
                attempts,
                retrying=False,
            )
            if raise_for_status:
                await _raise_for_status(response)
            return response

        raise AssertionError("HTTP retry loop exhausted without a result")

    async def get(self, url: str | httpx.URL, **kwargs: Any) -> httpx.Response:
        """Send a GET request."""
        return await self.request("GET", url, **kwargs)

    async def head(self, url: str | httpx.URL, **kwargs: Any) -> httpx.Response:
        """Send a HEAD request."""
        return await self.request("HEAD", url, **kwargs)

    async def options(self, url: str | httpx.URL, **kwargs: Any) -> httpx.Response:
        """Send an OPTIONS request."""
        return await self.request("OPTIONS", url, **kwargs)

    async def post(self, url: str | httpx.URL, **kwargs: Any) -> httpx.Response:
        """Send a POST request.

        POST is not retried by default; pass ``retry=True`` to opt in.
        """
        return await self.request("POST", url, **kwargs)

    async def put(self, url: str | httpx.URL, **kwargs: Any) -> httpx.Response:
        """Send a PUT request."""
        return await self.request("PUT", url, **kwargs)

    async def patch(self, url: str | httpx.URL, **kwargs: Any) -> httpx.Response:
        """Send a PATCH request."""
        return await self.request("PATCH", url, **kwargs)

    async def delete(self, url: str | httpx.URL, **kwargs: Any) -> httpx.Response:
        """Send a DELETE request."""
        return await self.request("DELETE", url, **kwargs)

    async def aclose(self) -> None:
        """Close the underlying client and its transport."""
        await self._client.aclose()

    async def close(self) -> None:
        """Alias for :meth:`aclose`."""
        await self.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


AsyncHttpClient = HttpClient

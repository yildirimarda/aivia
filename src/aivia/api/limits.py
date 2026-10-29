"""ASGI middleware enforcing request size, rate and authentication limits.

The middleware runs before routing, so a route added later is protected by
default. Health and the OpenAPI documentation stay public so a local client can
discover the API before it has a key.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import cast

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from aivia.api.security import api_key_matches, extract_api_key

__all__ = [
    "PUBLIC_PATHS",
    "ApiKeyAuthMiddleware",
    "BodySizeLimitMiddleware",
    "RateLimitMiddleware",
    "RequestRateLimiter",
]

PUBLIC_PATHS = frozenset(
    {
        "/health",
        "/openapi.json",
        "/docs",
        "/docs/oauth2-redirect",
        "/redoc",
    }
)

RATE_LIMIT_EXEMPT_PATHS = frozenset({"/health"})

Clock = Callable[[], float]

_MIN_WINDOW = 0.0


def _normalise_path(path: str) -> str:
    if path != "/":
        return path.rstrip("/")
    return path


def _raw_headers(scope: Scope) -> Iterable[tuple[bytes, bytes]]:
    return cast(Iterable[tuple[bytes, bytes]], scope.get("headers", ()))


def _header(scope: Scope, name: str) -> str | None:
    target = name.lower().encode("latin-1")
    for raw_name, raw_value in _raw_headers(scope):
        if raw_name.lower() == target:
            return raw_value.decode("latin-1")
    return None


def _headers(scope: Scope) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_name, raw_value in _raw_headers(scope):
        key = raw_name.decode("latin-1").lower()
        if key not in values:
            values[key] = raw_value.decode("latin-1")
    return values


async def _send_json(
    send: Send,
    status: int,
    payload: dict[str, object],
    headers: dict[str, str] | None = None,
) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    raw_headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    for name, value in (headers or {}).items():
        raw_headers.append(
            (name.encode("latin-1"), value.encode("latin-1")),
        )
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": raw_headers,
        }
    )
    await send({"type": "http.response.body", "body": body})


class BodySizeLimitMiddleware:
    """Reject requests whose declared or streamed body exceeds *max_body_bytes*.

    A declared ``Content-Length`` is checked before the application runs. For
    chunked requests the body is buffered only up to the limit, so a client
    cannot bypass the cap by omitting the length. Buffering also lets the
    middleware answer with ``413`` instead of raising into the request parser.
    """

    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        if max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be positive")
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = _header(scope, "content-length")
        if declared is not None:
            try:
                length = int(declared)
            except ValueError:
                await _send_json(
                    send,
                    400,
                    {"detail": "invalid content-length header"},
                )
                return
            if length < 0:
                await _send_json(
                    send,
                    400,
                    {"detail": "invalid content-length header"},
                )
                return
            if length > self.max_body_bytes:
                await _send_json(
                    send,
                    413,
                    {"detail": "request body too large"},
                )
                return

        buffered: list[Message] = []
        received = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                buffered.append(message)
                break
            received += len(message.get("body", b""))
            if received > self.max_body_bytes:
                await _send_json(
                    send,
                    413,
                    {"detail": "request body too large"},
                )
                return
            buffered.append(message)
            if not message.get("more_body", False):
                break

        pending = list(buffered)

        async def replay() -> Message:
            if pending:
                return pending.pop(0)
            return await receive()

        await self.app(scope, replay, send)


@dataclass(slots=True)
class _Window:
    started: float
    count: int


class RequestRateLimiter:
    """A fixed-window, per-client request limiter.

    ``per_minute`` of ``0`` or less disables limiting. The clock is injectable
    so tests can advance time without sleeping.
    """

    def __init__(
        self,
        per_minute: int,
        *,
        window_seconds: float = 60.0,
        clock: Clock | None = None,
    ) -> None:
        if per_minute < 0:
            raise ValueError("per_minute must be non-negative")
        if window_seconds <= _MIN_WINDOW:
            raise ValueError("window_seconds must be positive")
        self.per_minute = per_minute
        self.window_seconds = window_seconds
        self._clock = clock or time.monotonic
        self._windows: dict[str, _Window] = {}

    @property
    def enabled(self) -> bool:
        """Whether the limiter enforces a positive limit."""
        return self.per_minute > 0

    def hit(self, client: str) -> tuple[bool, float]:
        """Record one request and return ``(allowed, retry_after_seconds)``."""
        if not self.enabled:
            return True, 0.0

        now = self._clock()
        state = self._windows.get(client)
        if state is None or now - state.started >= self.window_seconds:
            state = _Window(started=now, count=0)
            self._windows[client] = state
            self._discard_expired(now)

        if state.count >= self.per_minute:
            remaining = self.window_seconds - (now - state.started)
            return False, max(0.0, remaining)

        state.count += 1
        return True, 0.0

    def _discard_expired(self, now: float) -> None:
        if len(self._windows) <= 1024:
            return
        expired = [
            client
            for client, state in self._windows.items()
            if now - state.started >= self.window_seconds
        ]
        for client in expired:
            del self._windows[client]


class RateLimitMiddleware:
    """Return ``429`` with ``Retry-After`` when a client exceeds its limit."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        limiter: RequestRateLimiter,
        exempt_paths: Iterable[str] = RATE_LIMIT_EXEMPT_PATHS,
    ) -> None:
        self.app = app
        self.limiter = limiter
        self.exempt_paths = frozenset(_normalise_path(path) for path in exempt_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = _normalise_path(scope.get("path", ""))
        if path not in self.exempt_paths:
            client = scope.get("client")
            key = client[0] if client else "unknown"
            allowed, retry_after = self.limiter.hit(key)
            if not allowed:
                headers = {"Retry-After": str(max(1, math.ceil(retry_after)))}
                await _send_json(
                    send,
                    429,
                    {"detail": "rate limit exceeded"},
                    headers,
                )
                return
        await self.app(scope, receive, send)


class ApiKeyAuthMiddleware:
    """Require the configured API key on every non-public route."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        expected_key: str,
        public_paths: Iterable[str] = PUBLIC_PATHS,
    ) -> None:
        self.app = app
        self.expected_key = expected_key
        self.public_paths = frozenset(_normalise_path(path) for path in public_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = _normalise_path(scope.get("path", ""))
        if path in self.public_paths:
            await self.app(scope, receive, send)
            return

        supplied = extract_api_key(_headers(scope))
        if not api_key_matches(self.expected_key, supplied):
            await _send_json(
                send,
                401,
                {"detail": "invalid or missing API key"},
                {"WWW-Authenticate": "Bearer"},
            )
            return
        await self.app(scope, receive, send)

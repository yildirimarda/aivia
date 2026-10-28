"""Shared async HTTP client: retries, backoff, timeouts and redacted logging."""

# secret-scan: allow-file -- this module intentionally contains fake tokens.

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from aivia.core.config import HttpClientConfig
from aivia.core.errors import (
    AuthError,
    BudgetError,
    InvalidRequestError,
    RateLimitError,
    RetryableError,
)
from aivia.core.errors import TimeoutError as AiviaTimeoutError
from aivia.core.http import (
    HttpClient,
    error_for_response,
    raise_for_status,
    redact_headers,
)
from aivia.core.redact import REDACTED


class FakeSleep:
    """Record retry delays without waiting in real time."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


class FakeRng:
    """Return a fixed sequence of samples, then repeat the last one."""

    def __init__(self, *values: float) -> None:
        self._values = list(values)
        self._last = values[-1] if values else 0.0

    def __call__(self) -> float:
        if self._values:
            self._last = self._values.pop(0)
        return self._last


def make_client(
    handler: Callable[[httpx.Request], Any],
    *,
    config: HttpClientConfig | None = None,
    sleep: FakeSleep | None = None,
    rng: FakeRng | None = None,
    logger: logging.Logger | None = None,
    **kwargs: Any,
) -> HttpClient:
    return HttpClient(
        config,
        transport=httpx.MockTransport(handler),
        sleep=sleep or FakeSleep(),
        rng=rng or FakeRng(0.5),
        logger=logger,
        **kwargs,
    )


def test_config_defaults_and_validation() -> None:
    config = HttpClientConfig()

    assert config.timeout_seconds == 30.0
    assert config.max_attempts == 3
    assert config.jitter_ratio == 0.5
    assert 503 in config.retry_statuses

    with pytest.raises(ValueError):
        HttpClientConfig(max_attempts=0)
    with pytest.raises(ValueError):
        HttpClientConfig(jitter_ratio=1.5)
    with pytest.raises(ValueError):
        HttpClientConfig(backoff_base_seconds=10, backoff_max_seconds=1)
    with pytest.raises(ValueError):
        HttpClientConfig(timeout_seconds=float("inf"))
    with pytest.raises(ValueError):
        HttpClientConfig(retry_statuses=(99,))


def test_retry_statuses_are_deduplicated_and_ordered() -> None:
    config = HttpClientConfig(retry_statuses=(503, 429, 503))

    assert config.retry_statuses == (503, 429)


def test_injected_transport_receives_the_request_and_defaults() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    client = make_client(
        handler,
        base_url="https://api.example.test",
        headers={"X-Client": "aivia"},
    )

    async def run() -> httpx.Response:
        async with client:
            return await client.get("/status")

    response = asyncio.run(run())

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert seen[0].url == httpx.URL("https://api.example.test/status")
    assert seen[0].headers["x-client"] == "aivia"
    assert client.client.is_closed


def test_retryable_statuses_back_off_exponentially() -> None:
    calls: list[int] = []
    sleep = FakeSleep()

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(503)
        return httpx.Response(200)

    client = make_client(
        handler,
        config=HttpClientConfig(
            max_attempts=3,
            backoff_base_seconds=1.0,
            backoff_max_seconds=10.0,
            jitter_ratio=0.0,
        ),
        sleep=sleep,
    )

    response = asyncio.run(client.get("https://example.test"))

    assert response.status_code == 200
    assert len(calls) == 3
    assert sleep.delays == [1.0, 2.0]


def test_jitter_removes_a_random_fraction_of_each_step() -> None:
    sleep = FakeSleep()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503 if calls < 3 else 200)

    client = make_client(
        handler,
        config=HttpClientConfig(
            max_attempts=3,
            backoff_base_seconds=1.0,
            backoff_max_seconds=10.0,
            jitter_ratio=0.5,
        ),
        sleep=sleep,
        rng=FakeRng(0.0, 1.0),
    )

    asyncio.run(client.get("https://example.test"))

    assert sleep.delays == [0.5, 2.0]


def test_backoff_is_capped_at_the_configured_maximum() -> None:
    sleep = FakeSleep()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503 if calls < 4 else 200)

    client = make_client(
        handler,
        config=HttpClientConfig(
            max_attempts=4,
            backoff_base_seconds=4.0,
            backoff_max_seconds=5.0,
            jitter_ratio=0.0,
        ),
        sleep=sleep,
    )

    asyncio.run(client.get("https://example.test"))

    assert sleep.delays == [4.0, 5.0, 5.0]


def test_the_final_retryable_response_is_returned_after_exhaustion() -> None:
    calls: list[int] = []
    sleep = FakeSleep()

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, text="still down")

    client = make_client(
        handler,
        config=HttpClientConfig(
            max_attempts=3,
            backoff_base_seconds=1.0,
            jitter_ratio=0.0,
        ),
        sleep=sleep,
    )

    response = asyncio.run(client.get("https://example.test"))

    assert response.status_code == 503
    assert len(calls) == 3
    assert len(sleep.delays) == 2


def test_raise_for_status_maps_the_final_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    client = make_client(
        handler,
        config=HttpClientConfig(max_attempts=2, backoff_base_seconds=0),
    )

    with pytest.raises(RetryableError) as caught:
        asyncio.run(client.get("https://example.test", raise_for_status=True))

    assert caught.value.details["status"] == 503
    assert caught.value.details["method"] == "GET"


def test_retry_after_header_overrides_computed_backoff() -> None:
    sleep = FakeSleep()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(200)

    client = make_client(
        handler,
        config=HttpClientConfig(
            max_attempts=2,
            backoff_base_seconds=1.0,
            jitter_ratio=0.0,
        ),
        sleep=sleep,
    )

    response = asyncio.run(client.get("https://example.test"))

    assert response.status_code == 200
    assert sleep.delays == [7.0]


def test_retry_after_can_be_disabled() -> None:
    sleep = FakeSleep()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(200)

    client = make_client(
        handler,
        config=HttpClientConfig(
            max_attempts=2,
            backoff_base_seconds=1.0,
            jitter_ratio=0.0,
            respect_retry_after=False,
        ),
        sleep=sleep,
    )

    asyncio.run(client.get("https://example.test"))

    assert sleep.delays == [1.0]


def test_non_retryable_status_is_returned_without_sleeping() -> None:
    calls: list[int] = []
    sleep = FakeSleep()

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(404, text="missing")

    client = make_client(handler, sleep=sleep)

    response = asyncio.run(client.get("https://example.test"))

    assert response.status_code == 404
    assert response.text == "missing"
    assert len(calls) == 1
    assert sleep.delays == []


def test_retry_false_makes_a_single_attempt() -> None:
    calls: list[int] = []
    sleep = FakeSleep()

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503)

    client = make_client(handler, sleep=sleep)

    response = asyncio.run(client.get("https://example.test", retry=False))

    assert response.status_code == 503
    assert len(calls) == 1
    assert sleep.delays == []


def test_post_is_not_retried_by_default() -> None:
    calls: list[int] = []
    sleep = FakeSleep()

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503)

    client = make_client(handler, sleep=sleep)

    response = asyncio.run(client.post("https://example.test"))

    assert response.status_code == 503
    assert len(calls) == 1
    assert sleep.delays == []


def test_post_retries_when_the_caller_opts_in() -> None:
    calls: list[int] = []
    sleep = FakeSleep()

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(503)
        return httpx.Response(200)

    client = make_client(
        handler,
        config=HttpClientConfig(
            max_attempts=3,
            backoff_base_seconds=1.0,
            jitter_ratio=0.0,
        ),
        sleep=sleep,
    )

    response = asyncio.run(client.post("https://example.test", retry=True))

    assert response.status_code == 200
    assert len(calls) == 3
    assert sleep.delays == [1.0, 2.0]


def test_post_transport_errors_are_not_retried_by_default() -> None:
    calls: list[int] = []
    sleep = FakeSleep()

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ConnectError("connection refused", request=request)

    client = make_client(
        handler,
        config=HttpClientConfig(
            max_attempts=3,
            backoff_base_seconds=1.0,
            jitter_ratio=0.0,
        ),
        sleep=sleep,
    )

    with pytest.raises(RetryableError) as caught:
        asyncio.run(client.post("https://example.test"))

    assert caught.value.details["attempt"] == 1
    assert len(calls) == 1
    assert sleep.delays == []


@pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS", "PUT", "DELETE"])
def test_idempotent_methods_retry_by_default(method: str) -> None:
    calls: list[int] = []
    sleep = FakeSleep()

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503 if len(calls) == 1 else 200)

    client = make_client(
        handler,
        config=HttpClientConfig(
            max_attempts=2,
            backoff_base_seconds=1.0,
            jitter_ratio=0.0,
        ),
        sleep=sleep,
    )

    response = asyncio.run(client.request(method, "https://example.test"))

    assert response.status_code == 200
    assert len(calls) == 2
    assert sleep.delays == [1.0]


def test_transport_errors_are_retried_then_succeed() -> None:
    calls: list[int] = []
    sleep = FakeSleep()

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200)

    client = make_client(
        handler,
        config=HttpClientConfig(
            max_attempts=2,
            backoff_base_seconds=1.0,
            jitter_ratio=0.0,
        ),
        sleep=sleep,
    )

    response = asyncio.run(client.get("https://example.test"))

    assert response.status_code == 200
    assert len(calls) == 2
    assert sleep.delays == [1.0]


def test_transport_error_exhaustion_raises_a_retryable_error() -> None:
    sleep = FakeSleep()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = make_client(
        handler,
        config=HttpClientConfig(
            max_attempts=2,
            backoff_base_seconds=1.0,
            jitter_ratio=0.0,
        ),
        sleep=sleep,
    )

    with pytest.raises(RetryableError) as caught:
        asyncio.run(client.get("https://example.test"))

    assert caught.value.details["error"] == "ConnectError"
    assert caught.value.details["attempt"] == 2
    assert sleep.delays == [1.0]


def test_timeout_is_retried_and_then_raised() -> None:
    sleep = FakeSleep()
    calls: list[int] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        await asyncio.sleep(5)
        return httpx.Response(200)

    client = make_client(
        handler,
        config=HttpClientConfig(
            timeout_seconds=0.02,
            max_attempts=2,
            backoff_base_seconds=0,
        ),
        sleep=sleep,
    )

    with pytest.raises(AiviaTimeoutError) as caught:
        asyncio.run(client.get("https://example.test"))

    assert len(calls) == 2
    assert caught.value.details["attempt"] == 2
    assert len(sleep.delays) == 1


def test_a_per_request_timeout_overrides_the_configured_deadline() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200)

    client = make_client(
        handler,
        config=HttpClientConfig(timeout_seconds=30, max_attempts=1),
    )

    with pytest.raises(AiviaTimeoutError):
        asyncio.run(client.get("https://example.test", timeout=0.02))


def test_request_and_response_headers_are_redacted_in_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="aivia.core.http")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "Set-Cookie": "session=super-secret-cookie",
                "X-Request-Id": "req-123",
            },
        )

    client = make_client(handler)

    asyncio.run(
        client.get(
            "https://example.test/path?token=query-secret",
            headers={
                "Authorization": "Bearer super-secret-token",
                "X-Api-Key": "another-secret",
                "X-Request-Id": "req-123",
            },
        )
    )

    logged = caplog.text
    assert "super-secret-token" not in logged
    assert "another-secret" not in logged
    assert "super-secret-cookie" not in logged
    assert "query-secret" not in logged
    assert REDACTED in logged
    assert "req-123" in logged
    assert "HTTP request GET" in logged
    assert "HTTP response GET" in logged


def test_redact_headers_masks_credentials_and_keeps_ordinary_headers() -> None:
    headers = {
        "Authorization": "Bearer abcdefghijkl",
        "Cookie": "sid=abcdefghijkl",
        "X-Request-Id": "req-123",
        "X-Trace": "token=abcdef123456",
    }

    assert redact_headers(headers) == {
        "Authorization": REDACTED,
        "Cookie": REDACTED,
        "X-Request-Id": "req-123",
        "X-Trace": "token=" + REDACTED,
    }


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (401, AuthError),
        (403, AuthError),
        (402, BudgetError),
        (408, AiviaTimeoutError),
        (429, RateLimitError),
        (400, InvalidRequestError),
        (404, InvalidRequestError),
        (500, RetryableError),
        (503, RetryableError),
        (504, AiviaTimeoutError),
    ],
)
def test_error_for_response_maps_statuses(
    status: int, error_type: type[Exception]
) -> None:
    request = httpx.Request("GET", "https://example.test/path")
    response = httpx.Response(status, request=request)

    error = error_for_response(response)

    assert isinstance(error, error_type)
    assert error.details["status"] == status
    assert error.details["url"] == "https://example.test/path"


def test_error_for_response_returns_none_for_success() -> None:
    request = httpx.Request("GET", "https://example.test")
    response = httpx.Response(204, request=request)

    assert error_for_response(response) is None


def test_raise_for_status_raises_the_mapped_error() -> None:
    request = httpx.Request("GET", "https://example.test")
    response = httpx.Response(401, request=request)

    with pytest.raises(AuthError):
        asyncio.run(raise_for_status(response))


def test_auth_errors_are_not_retried() -> None:
    calls: list[int] = []
    sleep = FakeSleep()

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401)

    client = make_client(handler, sleep=sleep)

    with pytest.raises(AuthError):
        asyncio.run(client.get("https://example.test", raise_for_status=True))

    assert len(calls) == 1
    assert sleep.delays == []


def test_response_retry_after_is_exposed_on_rate_limit_errors() -> None:
    request = httpx.Request("GET", "https://example.test")
    response = httpx.Response(
        429,
        headers={"Retry-After": "2.5"},
        request=request,
    )

    error = error_for_response(response)

    assert isinstance(error, RateLimitError)
    assert error.retry_after == pytest.approx(2.5)


def test_client_rejects_a_non_async_transport() -> None:
    with pytest.raises(TypeError):
        HttpClient(transport=object())  # type: ignore[arg-type]


def test_client_rejects_a_non_positive_timeout() -> None:
    client = make_client(lambda request: httpx.Response(200))

    with pytest.raises(ValueError):
        asyncio.run(client.get("https://example.test", timeout=0))

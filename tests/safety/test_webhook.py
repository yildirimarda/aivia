"""Webhook signing and verification in both directions.

Direction one (sign): the delivery provider stamps every outgoing payload with a
UTC timestamp and an HMAC-SHA256 signature that covers timestamp plus body.

Direction two (verify): a receiver recomputes the signature and enforces a
replay window, rejecting stale, future or tampered payloads.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac

import httpx
import pytest

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import InvalidRequestError
from aivia.core.http import HttpClient
from aivia.providers import SignedWebhookDeliveryProvider
from aivia.providers.delivery.webhook import (
    DEFAULT_REPLAY_WINDOW_SECONDS,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    _sign,
    _timestamp,
    verify,
)
from aivia.providers.protocols.delivery import (
    DeliveryArtifact,
    DeliveryMethod,
    DeliveryRequest,
)

_URL = "https://hooks.example.test/deliver"
_SECRET = "s3cr3t"
_ALLOWLIST = ["hooks.example.test"]


def _expected_signature(secret: str, timestamp: int, body: bytes) -> str:
    message = f"{timestamp}.".encode() + body
    digest = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _now() -> int:
    return _timestamp()


def _captured() -> tuple[list[httpx.Request], httpx.MockTransport]:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, text="delivered")

    return captured, httpx.MockTransport(handler)


def _request() -> DeliveryRequest:
    return DeliveryRequest(
        method=DeliveryMethod.WEBHOOK,
        target=_URL,
        order_id="order-1",
        run_id="run-1",
        subject="Your card",
        message="Enjoy",
        artifacts=[
            DeliveryArtifact(data=b"card", media_type="image/png", filename="card.png")
        ],
    )


def _provider(http_client: HttpClient) -> SignedWebhookDeliveryProvider:
    return SignedWebhookDeliveryProvider(
        url=_URL,
        secret=_SECRET,
        allowlist=_ALLOWLIST,
        http_client=http_client,
    )


# ---------------------------------------------------------------------------
# Direction one: signing
# ---------------------------------------------------------------------------


def test_delivery_stamps_every_request_with_a_utc_timestamp() -> None:
    captured, transport = _captured()
    client = HttpClient(transport=transport, config=HttpClientConfig())
    asyncio.run(_provider(client).deliver(_request()))

    assert len(captured) == 1
    call = captured[0]
    assert TIMESTAMP_HEADER in call.headers
    timestamp = int(call.headers[TIMESTAMP_HEADER])
    assert abs(_now() - timestamp) <= 2


def test_delivery_signs_timestamp_plus_body_not_body_alone() -> None:
    captured, transport = _captured()
    client = HttpClient(transport=transport, config=HttpClientConfig())
    asyncio.run(_provider(client).deliver(_request()))

    call = captured[0]
    timestamp = int(call.headers[TIMESTAMP_HEADER])
    assert call.headers[SIGNATURE_HEADER] == _expected_signature(
        _SECRET, timestamp, call.content
    )
    body_only = hmac.new(
        _SECRET.encode("utf-8"), call.content, hashlib.sha256
    ).hexdigest()
    assert call.headers[SIGNATURE_HEADER] != f"sha256={body_only}"


def test_delivery_and_verify_agree_in_the_sign_direction() -> None:
    captured, transport = _captured()
    client = HttpClient(transport=transport, config=HttpClientConfig())
    asyncio.run(_provider(client).deliver(_request()))

    call = captured[0]
    timestamp = int(call.headers[TIMESTAMP_HEADER])
    assert (
        verify(
            _SECRET,
            call.headers[SIGNATURE_HEADER],
            timestamp,
            call.content,
            tolerance_seconds=DEFAULT_REPLAY_WINDOW_SECONDS,
        )
        == timestamp
    )


# ---------------------------------------------------------------------------
# Direction two: verification
# ---------------------------------------------------------------------------


def test_verify_accepts_a_validly_signed_payload() -> None:
    body = b'{"order_id":"order-1"}'
    timestamp = _now()
    signature = _sign(_SECRET, timestamp, body)
    assert verify(_SECRET, signature, timestamp, body) == timestamp


def test_verify_rejects_a_tampered_body() -> None:
    body = b'{"order_id":"order-1"}'
    timestamp = _now()
    signature = _sign(_SECRET, timestamp, body)
    with pytest.raises(InvalidRequestError, match="invalid webhook signature"):
        verify(_SECRET, signature, timestamp, b'{"order_id":"tampered"}')


def test_verify_rejects_the_wrong_secret() -> None:
    body = b'{"order_id":"order-1"}'
    timestamp = _now()
    signature = _sign(_SECRET, timestamp, body)
    with pytest.raises(InvalidRequestError, match="invalid webhook signature"):
        verify("a-different-secret", signature, timestamp, body)


def test_verify_rejects_an_empty_signature() -> None:
    body = b"{}"
    timestamp = _now()
    with pytest.raises(InvalidRequestError, match="invalid webhook signature"):
        verify(_SECRET, "", timestamp, body)


def test_verify_rejects_a_timestamp_outside_the_replay_window() -> None:
    body = b"{}"
    timestamp = _now() - 2 * DEFAULT_REPLAY_WINDOW_SECONDS - 1
    signature = _sign(_SECRET, timestamp, body)
    with pytest.raises(InvalidRequestError, match="too old"):
        verify(_SECRET, signature, timestamp, body)


def test_verify_rejects_a_future_timestamp_outside_the_replay_window() -> None:
    body = b"{}"
    timestamp = _now() + 2 * DEFAULT_REPLAY_WINDOW_SECONDS + 1
    signature = _sign(_SECRET, timestamp, body)
    with pytest.raises(InvalidRequestError, match="future"):
        verify(_SECRET, signature, timestamp, body)


def test_verify_accepts_a_timestamp_at_the_window_edge() -> None:
    body = b"{}"
    timestamp = _now() - DEFAULT_REPLAY_WINDOW_SECONDS
    signature = _sign(_SECRET, timestamp, body)
    assert verify(_SECRET, signature, timestamp, body) == timestamp

"""Delivery failure handling: retry transient failures, notify via console
fallback and offer a redo that reuses cached artifact files.

Covers what happens when every provider in the fallback chain is exhausted:
the delivery is marked failed, a console notification is sent and a redo may
retry using the artifacts already on disk instead of regenerating them.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from aivia.core.config.orders import DeliveryConfig
from aivia.core.delivery import ScheduledDelivery
from aivia.core.errors import (
    InvalidRequestError,
    ProviderExhaustedError,
    RateLimitError,
)
from aivia.core.models.base import utcnow
from aivia.core.models.io import Delivery, DeliveryState
from aivia.core.models.provider import ProviderMeta
from aivia.providers import DeliveryChain
from aivia.providers.base import HealthStatus, ProviderHealth
from aivia.providers.protocols.delivery import (
    CAPABILITY_DELIVERY,
    DeliveryArtifact,
    DeliveryMethod,
    DeliveryReceipt,
    DeliveryRequest,
)
from aivia.providers.protocols.notify import (
    CAPABILITY_NOTIFY,
    NotificationMessage,
    NotificationReceipt,
)

__all__ = ["_FakeDelivery"]


def _artifact(data: bytes = b"card", *, filename: str = "card.png") -> DeliveryArtifact:
    return DeliveryArtifact(data=data, media_type="image/png", filename=filename)


def _request(
    method: DeliveryMethod = DeliveryMethod.FOLDER,
    target: str = "/tmp/deliveries",
    **overrides: object,
) -> DeliveryRequest:
    values: dict[str, object] = {
        "method": method,
        "target": target,
        "order_id": "order-1",
        "run_id": "run-1",
        "subject": "Your card",
        "message": "Enjoy",
        "artifacts": [_artifact()],
    }
    values.update(overrides)
    return DeliveryRequest(**values)  # type: ignore[arg-type]


class _VirtualClock:
    """A controllable clock so tests can drive time without real sleeps."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _advancing_sleep(clock: _VirtualClock) -> Callable[[float], Any]:
    async def _sleep(delay: float) -> None:
        clock.now += timedelta(seconds=delay)

    return _sleep


def _noop_sleep(_seconds: float) -> None:
    return None


class _FakeDelivery:
    """A structural delivery provider double recording calls and outcomes."""

    def __init__(
        self,
        name: str,
        *,
        outcome: str = "ok",
        error: BaseException | None = None,
        reference: str = "ref-1",
        fail_times: int = 0,
    ) -> None:
        self.name = name
        self._outcome = outcome
        self._error = error or RuntimeError("boom")
        self._reference = reference
        self._fail_times = fail_times
        self.calls = 0
        self.last_request: DeliveryRequest | None = None

    @property
    def metadata(self) -> ProviderMeta:
        return ProviderMeta(name=self.name, capabilities=[CAPABILITY_DELIVERY])

    def supports(self, capability: str) -> bool:
        return capability == CAPABILITY_DELIVERY

    async def health(self) -> ProviderHealth:
        return ProviderHealth(status=HealthStatus.HEALTHY, message="ok")

    async def close(self) -> None:
        return None

    async def deliver(self, request: DeliveryRequest) -> DeliveryReceipt:
        self.calls += 1
        self.last_request = request
        if self.calls <= self._fail_times or self._outcome == "fail":
            raise self._error
        return DeliveryReceipt(
            method=request.method,
            target=request.target,
            provider=self.name,
            reference=self._reference,
            delivered_at=utcnow(),
        )


class _CapturingNotifier:
    """A notifier double recording every message it is asked to send."""

    def __init__(self) -> None:
        self.sent: list[NotificationMessage] = []

    @property
    def name(self) -> str:
        return "console"

    @property
    def metadata(self) -> ProviderMeta:
        return ProviderMeta(name="console", capabilities=[CAPABILITY_NOTIFY])

    def supports(self, capability: str) -> bool:
        return capability == CAPABILITY_NOTIFY

    async def health(self) -> ProviderHealth:
        return ProviderHealth(status=HealthStatus.HEALTHY, message="ok")

    async def close(self) -> None:
        return None

    async def send(self, message: NotificationMessage) -> NotificationReceipt:
        if not isinstance(message, NotificationMessage):
            raise TypeError("message must be a NotificationMessage")
        self.sent.append(message)
        return NotificationReceipt(
            message_id="console-1",
            channel="console",
            provider="console",
            delivered_at=utcnow(),
        )


def _chain(
    config_values: dict[str, Any],
    providers: dict[str, _FakeDelivery],
) -> DeliveryChain:
    values = dict(config_values)
    values.setdefault("providers", {})
    for name in providers:
        values["providers"].setdefault(name, {})
    values.setdefault("default", next(iter(providers)))
    config = DeliveryConfig.from_mapping(values)
    return DeliveryChain(config, providers, sleep=_noop_sleep)


def _service(
    providers: dict[str, _FakeDelivery],
    config_values: dict[str, Any] | None = None,
    *,
    notifier: _CapturingNotifier | None = None,
    clock: _VirtualClock | None = None,
) -> tuple[ScheduledDelivery, _CapturingNotifier, _VirtualClock]:
    if config_values is None:
        config_values = {
            "chain": list(providers),
            "max_attempts": 3,
            "retry_seconds": 0,
        }
    chain = _chain(config_values, providers)
    clock = clock or _VirtualClock(datetime(2024, 1, 1, tzinfo=UTC))
    notifier = notifier if notifier is not None else _CapturingNotifier()
    service = ScheduledDelivery(
        chain, clock=clock, sleep=_noop_sleep, notifier=notifier
    )
    return service, notifier, clock


# ---------------------------------------------------------------------------
# Retrying transient failures
# ---------------------------------------------------------------------------


def test_chain_retries_a_transient_failure_before_succeeding() -> None:
    providers = {
        "a": _FakeDelivery("a", fail_times=2, error=RateLimitError("slow down"))
    }
    service, _, _ = _service(providers)
    delivery = Delivery(scheduled_for=None)
    request = _request()

    receipt = asyncio.run(service.deliver(delivery, request))

    assert receipt.provider == "a"
    assert providers["a"].calls == 3
    assert delivery.is_delivered is True
    assert delivery.has_confirmation is True


def test_permanent_error_falls_back_instead_of_retrying() -> None:
    providers = {
        "a": _FakeDelivery("a", outcome="fail", error=InvalidRequestError("bad")),
        "b": _FakeDelivery("b"),
    }
    service, _, _ = _service(providers, {"chain": ["a", "b"], "max_attempts": 3})
    delivery = Delivery(scheduled_for=None)
    request = _request()

    receipt = asyncio.run(service.deliver(delivery, request))

    assert receipt.provider == "b"
    assert providers["a"].calls == 1  # permanent errors are not retried
    assert delivery.is_delivered is True


# ---------------------------------------------------------------------------
# Exhaustion records the failure and notifies via the console fallback
# ---------------------------------------------------------------------------


def test_exhaustion_marks_the_delivery_failed_and_raises() -> None:
    providers = {"a": _FakeDelivery("a", outcome="fail", error=RateLimitError("down"))}
    service, notifier, _ = _service(
        providers, {"chain": ["a"], "max_attempts": 2, "retry_seconds": 0}
    )
    delivery = Delivery(max_attempts=3)
    request = _request()

    with pytest.raises(ProviderExhaustedError):
        asyncio.run(service.deliver(delivery, request))

    assert providers["a"].calls == 2
    assert delivery.attempts == 1
    assert delivery.state is DeliveryState.FAILED
    assert delivery.error_code == "provider_exhausted"
    assert delivery.error_message
    assert not delivery.is_delivered
    assert not delivery.has_confirmation

    assert len(notifier.sent) == 1
    message = notifier.sent[0]
    assert message.level.name == "ERROR"
    assert message.title == "Delivery failed"
    assert delivery.id in message.text
    assert any(button.label == "Redo" for button in message.buttons)
    assert message.metadata["delivery_id"] == delivery.id


# ---------------------------------------------------------------------------
# Redo reuses cached artifacts
# ---------------------------------------------------------------------------


def test_redo_reuses_cached_artifacts_and_succeeds(tmp_path: Path) -> None:
    cached = tmp_path / "cached.png"
    cached.write_bytes(b"cached-card")

    provider = _FakeDelivery("a", fail_times=1, error=RateLimitError("down"))
    service, notifier, _ = _service(
        {"a": provider},
        {"chain": ["a"], "max_attempts": 1, "retry_seconds": 0},
    )
    delivery = Delivery(
        max_attempts=3,
        files=[str(cached)],
    )
    request = _request()

    with pytest.raises(ProviderExhaustedError):
        asyncio.run(service.deliver(delivery, request))

    assert delivery.state is DeliveryState.FAILED
    assert delivery.attempts == 1
    assert len(notifier.sent) == 1

    receipt = asyncio.run(service.redo(delivery, request))

    assert receipt.provider == "a"
    assert provider.calls == 2
    assert delivery.is_delivered is True
    assert delivery.has_confirmation is True
    # The redo reused the cached file as a path-backed artifact.
    assert provider.last_request is not None
    assert len(provider.last_request.artifacts) == 1
    assert provider.last_request.artifacts[0].path == str(cached)
    assert provider.last_request.artifacts[0].filename == "cached.png"
    # One notification for the failed deliver; the redo succeeded silently.
    assert len(notifier.sent) == 1


def test_redo_fails_again_records_failure_and_notifies(tmp_path: Path) -> None:
    cached = tmp_path / "cached.png"
    cached.write_bytes(b"cached-card")

    failing = _FakeDelivery("a", outcome="fail", error=RateLimitError("down"))
    service, notifier, _ = _service(
        {"a": failing},
        {"chain": ["a"], "max_attempts": 1, "retry_seconds": 0},
    )
    delivery = Delivery(
        max_attempts=3,
        files=[str(cached)],
    )
    request = _request()

    with pytest.raises(ProviderExhaustedError):
        asyncio.run(service.deliver(delivery, request))
    assert delivery.attempts == 1

    with pytest.raises(ProviderExhaustedError):
        asyncio.run(service.redo(delivery, request))
    assert delivery.attempts == 2
    assert delivery.state is DeliveryState.FAILED
    assert len(notifier.sent) == 2  # one per failed attempt


def test_redo_rejects_an_already_delivered_delivery(tmp_path: Path) -> None:
    cached = tmp_path / "cached.png"
    cached.write_bytes(b"cached-card")

    providers = {"a": _FakeDelivery("a")}
    service, _, _ = _service(providers)
    delivery = Delivery(max_attempts=3, files=[str(cached)])
    request = _request()

    asyncio.run(service.deliver(delivery, request))
    assert delivery.is_delivered is True

    with pytest.raises(ValueError, match="already delivered"):
        asyncio.run(service.redo(delivery, request))


def test_redo_rejects_a_delivery_with_no_attempts_left(tmp_path: Path) -> None:
    cached = tmp_path / "cached.png"
    cached.write_bytes(b"cached-card")

    providers = {"a": _FakeDelivery("a", outcome="fail", error=RateLimitError("down"))}
    service, _, _ = _service(
        providers, {"chain": ["a"], "max_attempts": 1, "retry_seconds": 0}
    )
    delivery = Delivery(max_attempts=1, files=[str(cached)])
    request = _request()

    with pytest.raises(ProviderExhaustedError):
        asyncio.run(service.deliver(delivery, request))
    assert delivery.attempts == 1
    assert not delivery.has_attempts_left

    with pytest.raises(ValueError, match="no attempts left"):
        asyncio.run(service.redo(delivery, request))

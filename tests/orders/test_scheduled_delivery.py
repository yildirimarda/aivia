"""Scheduled delivery: wait until the due time in the order's timezone,
then deliver through the fallback chain with retries and record a confirmation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
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


# ---------------------------------------------------------------------------
# Scheduling and the order timezone
# ---------------------------------------------------------------------------


def test_unscheduled_delivery_is_due_immediately() -> None:
    chain = _chain({"chain": ["a"]}, {"a": _FakeDelivery("a")})
    service = ScheduledDelivery(chain)
    delivery = Delivery(scheduled_for=None)

    assert service.due_at(delivery) is None
    assert service.is_due(delivery) is True


def test_same_local_time_is_due_at_a_different_utc_instant_per_timezone() -> None:
    # 09:00 local on 1 June 2024 is 13:00 UTC in New York (EDT) but 08:00 UTC
    # in London (BST): the order's timezone moves the due instant.
    new_york = datetime(2024, 6, 1, 9, 0, 0)
    london = datetime(2024, 6, 1, 9, 0, 0)
    delivery_ny = Delivery.model_construct(scheduled_for=new_york)
    delivery_lon = Delivery.model_construct(scheduled_for=london)
    chain = _chain({"chain": ["a"]}, {"a": _FakeDelivery("a")})
    service = ScheduledDelivery(chain)

    due_ny = service.due_at(delivery_ny, timezone="America/New_York")
    due_lon = service.due_at(delivery_lon, timezone="Europe/London")

    assert due_ny is not None and due_lon is not None
    assert due_ny != due_lon
    assert due_ny.hour == 13
    assert due_lon.hour == 8


def test_delivery_is_not_due_before_the_due_instant() -> None:
    clock = _VirtualClock(datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC))
    due = datetime(2024, 6, 1, 13, 0, 0, tzinfo=UTC)
    delivery = Delivery.model_construct(scheduled_for=due)
    service = ScheduledDelivery(
        _chain({"chain": ["a"]}, {"a": _FakeDelivery("a")}),
        clock=clock,
        sleep=_advancing_sleep(clock),
    )

    assert service.is_due(delivery, now=clock()) is False


def test_deliver_waits_until_due_then_delivers_once() -> None:
    clock = _VirtualClock(datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC))
    due = datetime(2024, 6, 1, 12, 0, 1, tzinfo=UTC)
    provider = _FakeDelivery("a")
    chain = _chain({"chain": ["a"]}, {"a": provider})
    service = ScheduledDelivery(
        chain,
        clock=clock,
        sleep=_advancing_sleep(clock),
        poll_interval=0.5,
    )
    delivery = Delivery.model_construct(scheduled_for=due)
    request = _request()

    receipt = asyncio.run(service.deliver(delivery, request, timezone="UTC"))

    assert provider.calls == 1
    assert receipt.provider == "a"
    assert delivery.is_delivered is True


# ---------------------------------------------------------------------------
# Confirmation record
# ---------------------------------------------------------------------------


def test_deliver_records_a_confirmation_on_success() -> None:
    clock = _VirtualClock(utcnow())
    provider = _FakeDelivery("a")
    chain = _chain({"chain": ["a"]}, {"a": provider})
    service = ScheduledDelivery(chain, clock=clock, sleep=_noop_sleep)
    delivery = Delivery(scheduled_for=clock.now - timedelta(seconds=10))
    request = _request()

    receipt = asyncio.run(service.deliver(delivery, request))

    assert provider.calls == 1
    assert receipt.provider == "a"
    assert delivery.has_confirmation is True
    assert delivery.confirmation == "ref-1"
    assert delivery.delivered_at is not None
    assert delivery.is_delivered is True
    assert delivery.state is DeliveryState.DELIVERED


def test_deliver_without_a_scheduled_time_confirms_immediately() -> None:
    clock = _VirtualClock(utcnow())
    provider = _FakeDelivery("a")
    chain = _chain({"chain": ["a"]}, {"a": provider})
    service = ScheduledDelivery(chain, clock=clock, sleep=_noop_sleep)
    delivery = Delivery(scheduled_for=None)
    request = _request()

    receipt = asyncio.run(service.deliver(delivery, request))

    assert provider.calls == 1
    assert delivery.has_confirmation is True
    assert delivery.confirmation == receipt.reference


# ---------------------------------------------------------------------------
# Retries via the fallback chain
# ---------------------------------------------------------------------------


def test_retries_a_transient_failure_then_confirms() -> None:
    clock = _VirtualClock(utcnow())
    provider = _FakeDelivery("a", fail_times=2, error=RateLimitError("slow down"))
    chain = _chain(
        {"chain": ["a"], "max_attempts": 3, "retry_seconds": 0},
        {"a": provider},
    )
    service = ScheduledDelivery(chain, clock=clock, sleep=_noop_sleep)
    delivery = Delivery(scheduled_for=clock.now - timedelta(seconds=10))
    request = _request()

    receipt = asyncio.run(service.deliver(delivery, request))

    assert provider.calls == 3
    assert receipt.provider == "a"
    assert delivery.has_confirmation is True
    assert delivery.is_delivered is True


def test_permanent_error_falls_back_instead_of_retrying() -> None:
    clock = _VirtualClock(utcnow())
    failing = _FakeDelivery("a", outcome="fail", error=InvalidRequestError("bad"))
    succeeding = _FakeDelivery("b")
    chain = _chain(
        {"chain": ["a", "b"], "max_attempts": 3},
        {"a": failing, "b": succeeding},
    )
    service = ScheduledDelivery(chain, clock=clock, sleep=_noop_sleep)
    delivery = Delivery(scheduled_for=clock.now - timedelta(seconds=10))
    request = _request()

    receipt = asyncio.run(service.deliver(delivery, request))

    assert failing.calls == 1  # permanent errors are not retried
    assert succeeding.calls == 1
    assert receipt.provider == "b"
    assert delivery.has_confirmation is True


def test_exhaustion_raises_and_records_no_confirmation() -> None:
    clock = _VirtualClock(utcnow())
    provider = _FakeDelivery("a", outcome="fail", error=RateLimitError("down"))
    chain = _chain(
        {"chain": ["a"], "max_attempts": 2, "retry_seconds": 0},
        {"a": provider},
    )
    service = ScheduledDelivery(chain, clock=clock, sleep=_noop_sleep)
    delivery = Delivery(scheduled_for=clock.now - timedelta(seconds=10))
    request = _request()

    with pytest.raises(ProviderExhaustedError):
        asyncio.run(service.deliver(delivery, request))

    assert provider.calls == 2
    assert delivery.confirmation == ""
    assert not delivery.has_confirmation
    assert not delivery.is_delivered

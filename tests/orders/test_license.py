"""Music license gating for order delivery.

Delivery is blocked when an order's music is not cleared for commercial use,
and license metadata is stored on the order.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from aivia.core.db import OrderRepository, create_database_engine, upgrade_database
from aivia.core.delivery import ScheduledDelivery
from aivia.core.models.asset import AssetLicense
from aivia.core.models.base import utcnow
from aivia.core.models.io import Delivery, DeliveryState
from aivia.core.models.provider import ProviderMeta
from aivia.media.music_service import MusicLicenseError, MusicLicenseGate
from aivia.orders import order_music_licenses, store_order_music_licenses
from aivia.providers import DeliveryChain
from aivia.providers.base import HealthStatus, ProviderHealth
from aivia.providers.protocols.delivery import (
    CAPABILITY_DELIVERY,
    DeliveryArtifact,
    DeliveryMethod,
    DeliveryReceipt,
    DeliveryRequest,
)

COMMERCIAL = AssetLicense(
    name="CC BY 4.0",
    url="https://creativecommons.org/licenses/by/4.0/",
    commercial_use=True,
)
NON_COMMERCIAL = AssetLicense(
    name="Non-Commercial",
    url="https://example.test/non-commercial",
    commercial_use=False,
)
UNSTATED = AssetLicense(name="Unstated", url="https://example.test/unstated")


def _artifact(data: bytes = b"card", *, filename: str = "card.png") -> DeliveryArtifact:
    return DeliveryArtifact(data=data, media_type="image/png", filename=filename)


def _request(**overrides: Any) -> DeliveryRequest:
    values: dict[str, Any] = {
        "method": DeliveryMethod.FOLDER,
        "target": "/tmp/deliveries",
        "order_id": "order-1",
        "run_id": "run-1",
        "subject": "Your card",
        "message": "Enjoy",
        "artifacts": [_artifact()],
    }
    values.update(overrides)
    return DeliveryRequest(**values)


class _FakeDelivery:
    def __init__(self, name: str, *, outcome: str = "ok") -> None:
        self.name = name
        self._outcome = outcome
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
        if self._outcome == "fail":
            raise RuntimeError("boom")
        return DeliveryReceipt(
            method=request.method,
            target=request.target,
            provider=self.name,
            delivered_at=utcnow(),
        )


def _chain(providers: dict[str, _FakeDelivery]) -> DeliveryChain:
    from aivia.core.config.orders import DeliveryConfig

    config_values: dict[str, Any] = {
        "chain": list(providers),
        "default": next(iter(providers)),
        "providers": {name: {} for name in providers},
    }
    config = DeliveryConfig.from_mapping(config_values)
    return DeliveryChain(config, providers, sleep=lambda _: None)


@pytest.fixture
def engine(tmp_path: Path) -> Any:
    database = tmp_path / "data" / "aivia.db"
    upgrade_database(database)
    created = create_database_engine(database)
    try:
        yield created
    finally:
        created.dispose()


@pytest.fixture
def orders(engine: Any) -> OrderRepository:
    return OrderRepository(engine)


def _make_order(repository: OrderRepository, **values: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "occasion": "birthday",
        "name": "Ada",
        "message": "Happy birthday",
        "language": "en",
        "profile": "",
        "level": "standard",
        "timezone": "UTC",
        "currency": "USD",
        "amount_usd": 0.0,
        "budget_usd": None,
        "idempotency_key": "",
    }
    base.update(values)
    return repository.create(**cast_mapping(base))


def cast_mapping(values: dict[str, Any]) -> Mapping[str, Any]:
    return values


def _order_with_licenses(
    orders: OrderRepository, licenses: Sequence[AssetLicense]
) -> dict[str, Any]:
    order = _make_order(orders)
    updated = store_order_music_licenses(order, licenses)
    stored = orders.update(order["id"], metadata=updated["metadata"])
    assert stored is not None
    return stored


def test_store_order_music_licenses_records_licenses_on_the_order(
    orders: OrderRepository,
) -> None:
    order = _make_order(orders)
    updated = store_order_music_licenses(order, [COMMERCIAL, NON_COMMERCIAL])
    orders.update(order["id"], metadata=updated["metadata"])

    stored = orders.get(order["id"])
    assert stored is not None
    captured = order_music_licenses(stored)
    assert captured == (COMMERCIAL, NON_COMMERCIAL)
    assert stored["metadata"]["music_licenses"] == [
        COMMERCIAL.model_dump(mode="json"),
        NON_COMMERCIAL.model_dump(mode="json"),
    ]


def test_order_music_licenses_is_empty_without_metadata(
    orders: OrderRepository,
) -> None:
    order = _make_order(orders)
    assert order_music_licenses(order) == ()


def test_order_music_licenses_ignores_non_mapping_entries(
    orders: OrderRepository,
) -> None:
    order = _make_order(
        orders, metadata={"music_licenses": ["not-a-license", 123, None]}
    )
    assert order_music_licenses(order) == ()


def test_delivery_is_blocked_when_order_music_is_not_commercially_cleared(
    orders: OrderRepository,
) -> None:
    provider = _FakeDelivery("folder")
    service = ScheduledDelivery(_chain({"folder": provider}))
    order = _order_with_licenses(orders, [NON_COMMERCIAL])
    delivery = Delivery(scheduled_for=utcnow() - timedelta(seconds=10))
    request = _request(order_id=order["id"])

    with pytest.raises(MusicLicenseError) as caught:
        asyncio.run(service.deliver(delivery, request, order=order))

    assert caught.value.action == "deliver"
    assert caught.value.code == "music_license"
    assert provider.calls == 0
    assert not delivery.is_delivered
    assert delivery.error_code == "music_license"
    assert delivery.error_message == "music is not cleared for delivery"


def test_delivery_is_blocked_when_license_metadata_is_missing_or_unstated(
    orders: OrderRepository,
) -> None:
    provider = _FakeDelivery("folder")
    service = ScheduledDelivery(_chain({"folder": provider}))
    order = _order_with_licenses(orders, [UNSTATED])
    delivery = Delivery(scheduled_for=utcnow() - timedelta(seconds=10))
    request = _request(order_id=order["id"])

    with pytest.raises(MusicLicenseError):
        asyncio.run(service.deliver(delivery, request, order=order))

    assert provider.calls == 0
    assert not delivery.is_delivered


def test_delivery_proceeds_when_order_music_is_commercially_cleared(
    orders: OrderRepository,
) -> None:
    provider = _FakeDelivery("folder")
    service = ScheduledDelivery(_chain({"folder": provider}))
    order = _order_with_licenses(orders, [COMMERCIAL])
    delivery = Delivery(scheduled_for=utcnow() - timedelta(seconds=10))
    request = _request(order_id=order["id"])

    receipt = asyncio.run(service.deliver(delivery, request, order=order))

    assert provider.calls == 1
    assert receipt.provider == "folder"
    assert delivery.is_delivered
    assert delivery.state is DeliveryState.DELIVERED


def test_delivery_proceeds_when_order_has_no_music(
    orders: OrderRepository,
) -> None:
    provider = _FakeDelivery("folder")
    service = ScheduledDelivery(_chain({"folder": provider}))
    order = _make_order(orders)
    delivery = Delivery(scheduled_for=utcnow() - timedelta(seconds=10))
    request = _request(order_id=order["id"])

    asyncio.run(service.deliver(delivery, request, order=order))

    assert provider.calls == 1
    assert delivery.is_delivered


def test_delivery_without_an_order_is_not_license_checked() -> None:
    provider = _FakeDelivery("folder")
    service = ScheduledDelivery(_chain({"folder": provider}))
    delivery = Delivery(scheduled_for=utcnow() - timedelta(seconds=10))
    request = _request()

    asyncio.run(service.deliver(delivery, request))

    assert provider.calls == 1
    assert delivery.is_delivered


def test_default_gate_requires_commercial_use_for_delivery() -> None:
    gate = MusicLicenseGate()
    decision = gate.check(COMMERCIAL, action="deliver")
    assert decision.allowed is True

    decision = gate.check(NON_COMMERCIAL, action="deliver")
    assert decision.allowed is False
    assert "commercial_use_not_allowed:0" in decision.blockers

    decision = gate.check(UNSTATED, action="deliver")
    assert decision.allowed is False
    assert "commercial_use_unconfirmed:0" in decision.blockers

    decision = gate.check(action="deliver")
    assert decision.allowed is False
    assert "license_missing" in decision.blockers

    with pytest.raises(MusicLicenseError):
        gate.require(UNSTATED, action="deliver")

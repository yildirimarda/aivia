"""An order end to end across every packaged output plus a scheduled delivery.

The matrix creates one order per deliverable kind - png poster, mp3, mp4, gif
and web card - packages a master into that kind with an injected ffmpeg runner
so nothing touches the network, and asserts the pinned final exists with the
right container. A scheduled delivery then waits until the due instant in the
order's timezone and hands the packaged outputs to a local folder delivery
provider, recording a confirmation.
"""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from sqlalchemy import Engine

from aivia.core.config.orders import DeliveryConfig
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.db.repos import OrderRepository
from aivia.core.delivery import ScheduledDelivery
from aivia.core.models.base import utcnow
from aivia.core.models.io import Delivery, DeliveryState
from aivia.core.models.provider import ProviderMeta
from aivia.orders import order
from aivia.orders.packaging import package_outputs
from aivia.providers import DeliveryChain
from aivia.providers.base import HealthStatus, ProviderHealth
from aivia.providers.protocols.delivery import (
    CAPABILITY_DELIVERY,
    DeliveryArtifact,
    DeliveryMethod,
    DeliveryReceipt,
    DeliveryRequest,
)

MP4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 12
MP3 = b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 64

KINDS = ("mp4", "gif", "mp3", "poster", "web_card")

EXTENSION = {
    "mp4": "mp4",
    "gif": "gif",
    "mp3": "mp3",
    "poster": "png",
    "web_card": "html",
}

MEDIA_TYPE = {
    "mp4": "video/mp4",
    "gif": "image/gif",
    "mp3": "audio/mpeg",
    "poster": "image/png",
    "web_card": "text/html",
}

COMPOSITION = {
    "canvas": {"width": 1080, "height": 1920, "background": "#111111"},
    "duration_seconds": 5.0,
    "layers": [],
}


def _ffmpeg_writer(*, calls: list[list[str]] | None = None) -> Any:
    def run(argv: Sequence[str]) -> subprocess.CompletedProcess[Any]:
        if calls is not None:
            calls.append(list(argv))
        target = Path(argv[-1])
        target.write_bytes(MP3 if target.suffix == ".mp3" else MP4)
        return subprocess.CompletedProcess(list(argv), 0, stdout=b"", stderr=b"")

    return run


def _write_image(destination: Path) -> Path:
    image = Image.new("RGB", (200, 100), (255, 0, 0))
    image.save(destination, format="PNG")
    return destination


def _write_video(destination: Path) -> Path:
    destination.write_bytes(MP4)
    return destination


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    database = tmp_path / "data" / "aivia.db"
    upgrade_database(database)
    created = create_database_engine(database)
    try:
        yield created
    finally:
        created.dispose()


def _base_order_values() -> dict[str, Any]:
    return {
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


@pytest.mark.parametrize("kind", KINDS, ids=lambda kind: kind)
def test_order_packages_every_deliverable_kind(
    engine: Engine,
    tmp_path: Path,
    kind: str,
) -> None:
    orders = OrderRepository(engine)
    stored = order(orders, **_base_order_values())
    assert stored["id"]
    assert stored["state"] == "draft"

    video = tmp_path / "master.mp4"
    image = tmp_path / "background.png"
    _write_video(video)
    _write_image(image)

    outputs = package_outputs(
        tmp_path / "packaged",
        video=video,
        image=image,
        composition=COMPOSITION,
        formats=[kind],
        ffmpeg_runner=_ffmpeg_writer(),
    )

    assert set(outputs) == {kind}
    output = outputs[kind]
    assert output.path.is_file()
    assert output.path.stat().st_size > 0
    assert output.extension == EXTENSION[kind]
    assert output.format == EXTENSION[kind]
    assert output.pinned is True
    assert output.is_pinned is True


class _VirtualClock:
    """A controllable clock so tests can drive time without real sleeps."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _advancing_sleep(clock: _VirtualClock) -> Any:
    async def _sleep(delay: float) -> None:
        clock.now += timedelta(seconds=delay)

    return _sleep


async def _noop_sleep(_seconds: float) -> None:
    return None


class _FolderDelivery:
    """A delivery double that copies artifacts to a folder and confirms."""

    def __init__(self, folder: Path) -> None:
        self.name = "folder"
        self.folder = folder
        self.calls = 0
        self.last_request: DeliveryRequest | None = None

    @property
    def metadata(self) -> ProviderMeta:
        return ProviderMeta(name="folder", capabilities=[CAPABILITY_DELIVERY])

    def supports(self, capability: str) -> bool:
        return capability == CAPABILITY_DELIVERY

    async def health(self) -> ProviderHealth:
        return ProviderHealth(status=HealthStatus.HEALTHY, message="ok")

    async def close(self) -> None:
        return None

    async def deliver(self, request: DeliveryRequest) -> DeliveryReceipt:
        self.calls += 1
        self.last_request = request
        self.folder.mkdir(parents=True, exist_ok=True)
        for index, artifact in enumerate(request.artifacts):
            name = artifact.filename or f"artifact-{index}"
            (self.folder / name).write_bytes(artifact.data or b"")
        return DeliveryReceipt(
            method=request.method,
            target=str(self.folder),
            provider="folder",
            reference="ref-1",
            delivered_at=utcnow(),
        )


def test_order_scheduled_delivery_waits_until_due_in_order_timezone(
    engine: Engine,
    tmp_path: Path,
) -> None:
    orders = OrderRepository(engine)
    values = _base_order_values()
    values["timezone"] = "America/New_York"
    stored = order(orders, **values)
    assert stored["id"]

    video = tmp_path / "master.mp4"
    image = tmp_path / "background.png"
    _write_video(video)
    _write_image(image)

    outputs = package_outputs(
        tmp_path / "packaged",
        video=video,
        image=image,
        composition=COMPOSITION,
        formats=["mp4", "web_card"],
        ffmpeg_runner=_ffmpeg_writer(),
    )

    assert set(outputs) == {"mp4", "web_card"}
    for output in outputs.values():
        assert output.path.is_file()
        assert output.path.stat().st_size > 0
        assert output.pinned is True

    deliveries = tmp_path / "deliveries"
    provider = _FolderDelivery(deliveries)

    # 13:00 New York on 2024-06-01 (EDT) is 17:00 UTC; the clock starts one
    # second before the due instant so the service must wait once.
    clock = _VirtualClock(datetime(2024, 6, 1, 16, 59, 59, tzinfo=UTC))
    due = datetime(2024, 6, 1, 13, 0, 0)
    delivery = Delivery.model_construct(
        order_id=stored["id"],
        scheduled_for=due,
    )

    chain = DeliveryChain(
        DeliveryConfig.from_mapping(
            {
                "chain": ["folder"],
                "default": "folder",
                "providers": {"folder": {}},
                "retry_seconds": 0,
                "max_attempts": 1,
            }
        ),
        {"folder": provider},
        sleep=_noop_sleep,
    )
    service = ScheduledDelivery(
        chain,
        clock=clock,
        sleep=_advancing_sleep(clock),
        poll_interval=2.0,
    )

    artifacts = [
        DeliveryArtifact(
            data=output.path.read_bytes(),
            media_type=MEDIA_TYPE[output.kind],
            filename=output.path.name,
        )
        for output in outputs.values()
    ]
    request = DeliveryRequest(
        method=DeliveryMethod.FOLDER,
        target=str(deliveries),
        order_id=stored["id"],
        run_id="run-1",
        subject="Your order",
        message="Enjoy",
        artifacts=artifacts,
    )

    receipt = asyncio.run(
        service.deliver(delivery, request, timezone="America/New_York")
    )

    assert receipt.provider == "folder"
    assert delivery.is_delivered is True
    assert delivery.has_confirmation is True
    assert delivery.confirmation == "ref-1"
    assert delivery.delivered_at is not None
    assert delivery.state is DeliveryState.DELIVERED
    assert (deliveries / outputs["mp4"].path.name).is_file()
    assert (deliveries / outputs["web_card"].path.name).is_file()

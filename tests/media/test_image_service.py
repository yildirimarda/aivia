"""The image service is the single screened, cached, budgeted paid path."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import httpx
import pytest
from sqlalchemy import Engine

from aivia.core.budget import BudgetGuard
from aivia.core.config.budget import BudgetConfig
from aivia.core.config.capability import CapabilityChainConfig
from aivia.core.config.sections import SafetyConfig
from aivia.core.cost import CostTracker
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.errors import (
    ContentPolicyError,
    ProviderExhaustedError,
    ValidationError,
)
from aivia.core.errors import TimeoutError as AiviaTimeoutError
from aivia.core.media_probe import MediaDownloader
from aivia.core.models import AssetKind, ProviderMeta, ProviderPrice, RateLimit
from aivia.core.models.cost import CostEvent
from aivia.core.router import Router
from aivia.media.image_service import (
    IMAGE_OPERATION,
    IMAGE_UNIT,
    ImageOutcome,
    ImageService,
)
from aivia.media.screening import screen_prompt
from aivia.providers import Provider
from aivia.providers.image import ImageCostCalculator
from aivia.providers.mock import MockImageProvider
from aivia.providers.protocols import (
    CAPABILITY_IMAGE_GEN,
    ImageFormat,
    ImageRequest,
    ImageResponse,
    ImageResult,
)
from aivia.providers.storage_local import LocalStorageProvider

PROMPT = "A lighthouse at sunset"
PRICE = 0.02

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


@pytest.fixture
def tracker(engine: Engine) -> CostTracker:
    return CostTracker(engine)


@pytest.fixture
def storage(tmp_path: Path) -> LocalStorageProvider:
    return LocalStorageProvider(tmp_path / "media")


def _router(*providers: Provider) -> Router:
    return Router(
        {provider.name: provider for provider in providers},
        config=CapabilityChainConfig(chain=[provider.name for provider in providers]),
    )


def _service(
    provider: Provider,
    tracker: CostTracker,
    storage: LocalStorageProvider,
    **values: Any,
) -> ImageService:
    return ImageService(
        _router(provider),
        tracker=tracker,
        storage=storage,
        **values,
    )


def _token(prompt: str = PROMPT) -> str:
    return str(screen_prompt(prompt))


def _path(asset_path: str) -> Path:
    parsed = urlsplit(asset_path)
    assert parsed.scheme == "file"
    return Path(unquote(parsed.path))


class BadImageProvider(Provider):
    """A provider that returns bytes which are not an image."""

    def __init__(self) -> None:
        super().__init__(
            ProviderMeta(
                name="bad",
                capabilities=[CAPABILITY_IMAGE_GEN],
                pricing={
                    CAPABILITY_IMAGE_GEN: ProviderPrice(
                        amount_usd=PRICE,
                        unit=IMAGE_UNIT,
                    )
                },
                rate_limit=RateLimit(requests_per_minute=0),
            )
        )
        self.calls = 0

    async def generate(self, request: ImageRequest) -> ImageResponse:
        self.calls += 1
        return ImageResponse(
            images=[ImageResult(data=b"not an image", format=ImageFormat.PNG)],
            model="bad-image",
            provider=self.name,
        )


class UrlImageProvider(Provider):
    """A provider that returns a remote image URL instead of inline bytes."""

    def __init__(self, url: str) -> None:
        super().__init__(
            ProviderMeta(
                name="url",
                capabilities=[CAPABILITY_IMAGE_GEN],
                pricing={
                    CAPABILITY_IMAGE_GEN: ProviderPrice(
                        amount_usd=PRICE,
                        unit=IMAGE_UNIT,
                    )
                },
                rate_limit=RateLimit(requests_per_minute=0),
            )
        )
        self.url = url
        self.calls = 0

    async def generate(self, request: ImageRequest) -> ImageResponse:
        self.calls += 1
        return ImageResponse(
            images=[
                ImageResult(
                    url=self.url,
                    format=ImageFormat.PNG,
                    width=1,
                    height=1,
                )
            ],
            model="url-image",
            provider=self.name,
        )


def test_a_missing_or_mismatched_token_blocks_before_any_paid_step(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = MockImageProvider(name="acme")
    service = _service(provider, tracker, storage)

    with pytest.raises(ContentPolicyError) as missing:
        _run(service.generate(ImageRequest(prompt=PROMPT)))
    assert missing.value.details["reason"] == "missing_token"

    with pytest.raises(ContentPolicyError) as mismatched:
        _run(
            service.generate(
                ImageRequest(prompt=PROMPT),
                token=_token("A different prompt"),
            )
        )
    assert mismatched.value.details["reason"] == "text_mismatch"

    assert provider.calls == ()
    assert tracker.count() == 0
    assert service.cache is not None
    assert service.cache.repository.list() == []
    assert list(storage.root.rglob("*")) == []


def test_a_newly_configured_term_cannot_be_bypassed_by_an_old_token(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = MockImageProvider(name="acme")
    service = _service(
        provider,
        tracker,
        storage,
        safety=SafetyConfig(blocked_terms=["weapon"]),
    )
    token = screen_prompt("a weapon")

    with pytest.raises(ContentPolicyError):
        _run(service.generate(ImageRequest(prompt="a weapon"), token=token))

    assert provider.calls == ()
    assert tracker.count() == 0


def test_a_screened_request_routes_validates_stores_prices_and_writes_a_sidecar(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = MockImageProvider(name="acme")
    service = _service(provider, tracker, storage)

    outcome = _run(
        service.generate(
            ImageRequest(prompt=PROMPT, width=64, height=64),
            token=_token(),
            run_id="run_1",
            metadata={"stage": "visuals"},
        )
    )

    assert isinstance(outcome, ImageOutcome)
    assert outcome.cache_hit is False
    assert outcome.response.count == 1
    assert outcome.provider == "acme"
    assert outcome.model == "mock-image"
    assert len(provider.calls) == 1

    assert len(outcome.assets) == 1
    asset = outcome.asset
    assert asset is not None
    assert asset.kind is AssetKind.IMAGE
    assert asset.provider == "acme"
    assert asset.model == "mock-image"
    assert asset.prompt == PROMPT
    assert asset.run_id == "run_1"
    assert asset.pinned is True
    assert asset.content_hash == outcome.probes[0].sha256
    assert asset.analysis.width == 64
    assert asset.analysis.height == 64

    path = _path(asset.path)
    assert path.is_file()
    assert path.read_bytes() == outcome.probes[0].data

    event = outcome.event
    assert event is not None
    assert event.provider == "acme"
    assert event.operation == IMAGE_OPERATION
    assert event.unit_kind == IMAGE_UNIT
    assert event.units == 1.0
    assert event.amount_usd == pytest.approx(PRICE)
    assert event.run_id == "run_1"
    assert event.estimated is False
    assert event.metadata["stage"] == "visuals"
    assert event.metadata["image_count"] == 1
    assert asset.cost_links[0].cost_event_id == event.id
    assert tracker.total() == pytest.approx(PRICE)

    assert outcome.price is not None
    assert outcome.price.amount_usd == pytest.approx(PRICE)
    assert outcome.selection is not None
    assert outcome.selection.name == "acme"

    sidecar = outcome.sidecar
    assert sidecar is not None
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text("utf-8"))
    assert payload["output"] == str(path)
    assert payload["run_id"] == "run_1"
    assert payload["totals"]["total_usd"] == pytest.approx(PRICE)
    assert payload["items"][0]["cost_event_id"] == event.id
    assert payload["providers"] == ["acme"]

    assert service.cache is not None
    entries = service.cache.repository.list()
    assert len(entries) == 1
    assert entries[0].provider == ""
    assert entries[0].model == ""
    assert entries[0].prompt_hash == asset.prompt_hash


def test_an_identical_request_is_served_from_cache_without_a_provider_call(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = MockImageProvider(name="acme")
    service = _service(provider, tracker, storage)

    first = _run(
        service.generate(
            ImageRequest(prompt=PROMPT, width=64, height=64),
            token=_token(),
            run_id="run_1",
        )
    )
    second = _run(
        service.generate(
            ImageRequest(prompt=PROMPT, width=64, height=64),
            token=_token(),
            run_id="run_2",
        )
    )

    assert len(provider.calls) == 1
    assert second.cache_hit is True
    assert second.asset is not None
    assert first.asset is not None
    assert second.asset.id == first.asset.id
    assert second.response.count == 1
    assert second.event is not None
    assert second.event.amount_usd == 0.0
    assert second.event.metadata["cache"] == "hit"
    assert second.event.run_id == "run_2"
    assert second.amount_usd == 0.0
    assert tracker.total() == pytest.approx(PRICE)
    assert second.sidecar is not None
    assert second.sidecar.exists()

    assert service.cache is not None
    entry = service.cache.entry(
        service.cache_key(ImageRequest(prompt=PROMPT, width=64, height=64))
    )
    assert entry is not None
    assert entry.hit_count == 1


def test_an_over_budget_call_never_reaches_the_provider(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = MockImageProvider(name="pricey", cost_usd=0.10)
    service = _service(provider, tracker, storage)

    with pytest.raises(ProviderExhaustedError) as caught:
        _run(
            service.generate(
                ImageRequest(prompt=PROMPT),
                token=_token(),
                remaining_budget=0.05,
            )
        )

    assert provider.calls == ()
    assert caught.value.details["skipped"] == [
        {"provider": "pricey", "reason": "budget_exceeded"}
    ]
    assert tracker.count() == 0


def test_a_budget_guard_supplies_the_remaining_budget(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = MockImageProvider(name="pricey", cost_usd=0.10)
    guard = BudgetGuard(BudgetConfig(per_call=0.05), tracker)
    service = _service(provider, tracker, storage, budget=guard)

    with pytest.raises(ProviderExhaustedError):
        _run(service.generate(ImageRequest(prompt=PROMPT), token=_token()))

    assert provider.calls == ()
    assert tracker.count() == 0


def test_retryable_failure_falls_back_and_prices_the_provider_that_served(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    failing = MockImageProvider(name="failing", error=AiviaTimeoutError("slow"))
    serving = MockImageProvider(name="serving", cost_usd=0.03)
    service = ImageService(
        _router(failing, serving),
        tracker=tracker,
        storage=storage,
    )

    outcome = _run(service.generate(ImageRequest(prompt=PROMPT), token=_token()))

    assert failing.calls == ()
    assert len(serving.calls) == 1
    assert outcome.provider == "serving"
    assert outcome.selection is not None
    assert outcome.selection.name == "serving"
    assert outcome.event is not None
    assert outcome.event.provider == "serving"
    assert outcome.event.amount_usd == pytest.approx(0.03)
    assert tracker.total() == pytest.approx(0.03)


def test_invalid_image_bytes_are_rejected_before_storage_or_cost(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = BadImageProvider()
    service = _service(provider, tracker, storage)

    with pytest.raises(ValidationError):
        _run(service.generate(ImageRequest(prompt=PROMPT), token=_token()))

    assert provider.calls == 1
    assert tracker.count() == 0
    assert service.cache is not None
    assert service.cache.repository.list() == []
    assert list(storage.root.rglob("*")) == []


def test_a_remote_image_is_downloaded_validated_and_stored(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.com/image.png"
        return httpx.Response(
            200,
            headers={"content-type": "image/png"},
            content=PNG_1X1,
        )

    downloader = MediaDownloader(
        transport=httpx.MockTransport(handler),
    )
    provider = UrlImageProvider("https://example.com/image.png")
    service = _service(provider, tracker, storage, downloader=downloader)

    outcome = _run(service.generate(ImageRequest(prompt=PROMPT), token=_token()))

    assert provider.calls == 1
    assert outcome.probes[0].width == 1
    assert outcome.probes[0].height == 1
    assert outcome.probes[0].data == PNG_1X1
    assert outcome.asset is not None
    assert _path(outcome.asset.path).read_bytes() == PNG_1X1
    assert tracker.total() == pytest.approx(PRICE)


def test_count_stores_one_asset_and_cache_entry_per_output(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = MockImageProvider(name="acme")
    service = _service(provider, tracker, storage)
    request = ImageRequest(prompt=PROMPT, width=32, height=32, count=2)

    first = _run(service.generate(request, token=_token(), run_id="run_1"))

    assert len(provider.calls) == 1
    assert len(first.assets) == 2
    assert len(first.storage_objects) == 2
    assert len({asset.id for asset in first.assets}) == 2
    assert first.event is not None
    assert first.event.units == 2.0
    assert first.event.amount_usd == pytest.approx(PRICE * 2)
    assert service.cache is not None
    assert len(service.cache.repository.list()) == 2

    second = _run(service.generate(request, token=_token(), run_id="run_2"))

    assert len(provider.calls) == 1
    assert second.cache_hit is True
    assert second.response.count == 2
    assert [asset.id for asset in second.assets] == [asset.id for asset in first.assets]
    assert len(second.events) == 2
    assert second.amount_usd == 0.0
    assert tracker.total() == pytest.approx(PRICE * 2)


def test_no_cache_forces_every_request_through_the_provider(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = MockImageProvider(name="acme")
    service = _service(provider, tracker, storage, no_cache=True)

    _run(service.generate(ImageRequest(prompt=PROMPT), token=_token()))
    _run(service.generate(ImageRequest(prompt=PROMPT), token=_token()))

    assert len(provider.calls) == 2
    assert service.cache is not None
    assert service.cache.repository.list() == []
    assert tracker.total() == pytest.approx(PRICE * 2)


def test_the_cost_calculator_uses_the_image_unit_and_provider_override() -> None:
    provider = MockImageProvider(name="acme", cost_usd=PRICE)
    calculator = ImageCostCalculator({"acme": 0.07}, pricing_version="2026-01")

    price = calculator.resolve(
        provider,
        provider_name="acme",
        units=2,
        operation=IMAGE_OPERATION,
    )

    assert price.unit_kind == IMAGE_UNIT
    assert price.units == 2.0
    assert price.amount_usd == pytest.approx(0.14)
    assert price.pricing_version == "2026-01"
    event = calculator.cost_event(
        provider,
        provider_name="acme",
        units=2,
        price=price,
    )
    assert isinstance(event, CostEvent)
    assert event.operation == IMAGE_OPERATION
    assert event.unit_kind == IMAGE_UNIT
    assert event.amount_usd == pytest.approx(0.14)

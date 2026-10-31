"""Variation counts and the distinct seed recorded for every output."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine

from aivia.core.config.capability import CapabilityChainConfig
from aivia.core.cost import CostTracker
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.models import ProviderMeta, ProviderPrice, RateLimit
from aivia.core.router import Router
from aivia.media.image_service import (
    IMAGE_UNIT,
    ImageService,
    variation_seeds,
)
from aivia.media.screening import screen_prompt
from aivia.providers import Provider
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


def _service(
    provider: Provider,
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> ImageService:
    return ImageService(
        Router(
            {provider.name: provider},
            config=CapabilityChainConfig(chain=[provider.name]),
        ),
        tracker=tracker,
        storage=storage,
    )


def _token() -> str:
    return str(screen_prompt(PROMPT))


class SeedlessProvider(Provider):
    """Return valid images with either no seed or a fixed reported seed list."""

    def __init__(self, seeds: list[int | None] | None = None) -> None:
        super().__init__(
            ProviderMeta(
                name="seedless",
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
        self.seeds = seeds
        self.calls: list[ImageRequest] = []

    async def generate(self, request: ImageRequest) -> ImageResponse:
        self.calls.append(request)
        images: list[ImageResult] = []
        for index in range(request.count):
            seed = None if self.seeds is None else self.seeds[index % len(self.seeds)]
            images.append(
                ImageResult(
                    data=PNG_1X1,
                    format=ImageFormat.PNG,
                    width=1,
                    height=1,
                    seed=seed,
                )
            )
        return ImageResponse(
            images=images,
            model="seedless-image",
            provider=self.name,
        )


def test_an_explicit_seed_expands_to_a_distinct_seed_per_output(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = SeedlessProvider()
    service = _service(provider, tracker, storage)
    request = ImageRequest(prompt=PROMPT, width=1, height=1, count=3, seed=42)

    outcome = _run(service.generate(request, token=_token(), run_id="run_1"))

    assert len(provider.calls) == 1
    assert provider.calls[0].seed == 42
    assert [image.seed for image in outcome.images] == [42, 43, 44]
    assert [asset.seed for asset in outcome.assets] == [42, 43, 44]
    assert outcome.seeds == (42, 43, 44)
    assert outcome.variation_seeds == (42, 43, 44)
    assert outcome.seed == 42
    assert len(set(outcome.seeds)) == 3

    assert service.cache is not None
    for asset in outcome.assets:
        stored = service.cache.assets.get(asset.id)
        assert stored is not None
        assert stored.seed == asset.seed


def test_the_cache_key_uses_the_seed_of_each_output(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = SeedlessProvider()
    service = _service(provider, tracker, storage)
    request = ImageRequest(prompt=PROMPT, width=1, height=1, count=3, seed=42)

    keys = [service.cache_key(request, index=index) for index in range(3)]

    assert [key.parameters["seed"] for key in keys] == [42, 43, 44]
    assert len({key.digest for key in keys}) == 3


def test_duplicate_provider_seeds_are_replaced_with_distinct_values(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = SeedlessProvider(seeds=[7])
    service = _service(provider, tracker, storage)
    request = ImageRequest(prompt=PROMPT, width=1, height=1, count=3, seed=5)

    outcome = _run(service.generate(request, token=_token()))

    assert [image.seed for image in outcome.images] == [5, 6, 7]
    assert len(set(outcome.seeds)) == 3


def test_without_a_seed_every_output_gets_a_fresh_distinct_recorded_seed(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = SeedlessProvider()
    service = _service(provider, tracker, storage)
    request = ImageRequest(prompt=PROMPT, width=1, height=1, count=3)

    outcome = _run(service.generate(request, token=_token()))

    assert provider.calls[0].seed is not None
    assert outcome.seeds[0] == provider.calls[0].seed
    assert all(isinstance(seed, int) for seed in outcome.seeds)
    assert len(set(outcome.seeds)) == 3
    assert [asset.seed for asset in outcome.assets] == list(outcome.seeds)


def test_distinct_provider_seeds_are_kept_when_no_seed_was_requested(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = SeedlessProvider(seeds=[10, 20, 30])
    service = _service(provider, tracker, storage)
    request = ImageRequest(prompt=PROMPT, width=1, height=1, count=3)

    outcome = _run(service.generate(request, token=_token()))

    assert outcome.seeds == (10, 20, 30)
    assert [asset.seed for asset in outcome.assets] == [10, 20, 30]


def test_a_cache_hit_replays_the_recorded_seeds_without_a_provider_call(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = SeedlessProvider()
    service = _service(provider, tracker, storage)
    request = ImageRequest(prompt=PROMPT, width=1, height=1, count=2)

    first = _run(service.generate(request, token=_token(), run_id="run_1"))
    second = _run(service.generate(request, token=_token(), run_id="run_2"))

    assert len(provider.calls) == 1
    assert second.cache_hit is True
    assert second.seeds == first.seeds
    assert [asset.seed for asset in second.assets] == [
        asset.seed for asset in first.assets
    ]


def test_variation_seeds_is_deterministic_for_an_explicit_base() -> None:
    assert variation_seeds(10, 3) == (10, 11, 12)
    assert variation_seeds(10, 1) == (10,)

    with pytest.raises(ValueError, match="count"):
        variation_seeds(10, 0)

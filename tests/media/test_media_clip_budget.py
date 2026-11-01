"""The video service applies the per-clip cap before every paid submit."""

from __future__ import annotations

import asyncio
import json
import struct
import subprocess
from collections.abc import Iterator, Sequence
from typing import Any

import pytest
from sqlalchemy import Engine

from aivia.core.config.budget import BudgetConfig
from aivia.core.config.capability import CapabilityChainConfig
from aivia.core.cost import CostTracker
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.errors import BudgetError
from aivia.core.media_budget import ClipBudgetAction, PerClipBudgetPolicy
from aivia.core.models import ProviderMeta, ProviderPrice, RateLimit
from aivia.core.router import Router
from aivia.media.screening import screen_prompt
from aivia.media.video_cache import VideoCache
from aivia.media.video_pricing import VideoPricing
from aivia.media.video_service import VideoOutcome, VideoService
from aivia.providers import Provider
from aivia.providers.protocols import (
    CAPABILITY_VIDEO_GEN,
    VideoFormat,
    VideoRequest,
    VideoResponse,
    VideoResult,
)
from aivia.providers.storage_local import LocalStorageProvider

PROMPT = "A lighthouse in a storm"


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
def storage(tmp_path: Any) -> LocalStorageProvider:
    return LocalStorageProvider(tmp_path / "media")


def mp4() -> bytes:
    return struct.pack(">I", 24) + b"ftypisom" + b"\x00" * 12


def probe_json(*, duration: str = "5.0") -> str:
    return json.dumps(
        {
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1280,
                    "height": 720,
                    "duration": duration,
                    "avg_frame_rate": "30/1",
                }
            ],
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "format_long_name": "QuickTime / MOV",
                "duration": duration,
                "size": "12345",
            },
        }
    )


def runner_for(stdout: str) -> Any:
    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            list(argv),
            0,
            stdout=stdout,
            stderr="",
        )

    return runner


class FakeVideoProvider(Provider):
    def __init__(
        self,
        name: str,
        *,
        price: float,
        unit: str = "clip",
    ) -> None:
        super().__init__(
            ProviderMeta(
                name=name,
                capabilities=[CAPABILITY_VIDEO_GEN],
                pricing={
                    CAPABILITY_VIDEO_GEN: ProviderPrice(
                        amount_usd=price,
                        unit=unit,
                    )
                },
                rate_limit=RateLimit(requests_per_minute=0),
            )
        )
        self.calls: list[VideoRequest] = []

    async def generate(self, request: VideoRequest) -> VideoResponse:
        self.calls.append(request)
        return VideoResponse(
            videos=[
                VideoResult(data=mp4(), format=VideoFormat.MP4)
                for _ in range(request.count)
            ],
            model="video-1",
            provider=self.name,
        )


def _service(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
    *providers: Provider,
    **values: Any,
) -> VideoService:
    router = Router(
        {provider.name: provider for provider in providers},
        config=CapabilityChainConfig(chain=[provider.name for provider in providers]),
    )
    cache = VideoCache(
        engine=engine,
        storage=storage,
        ffprobe_runner=runner_for(probe_json()),
    )
    return VideoService(
        router,
        tracker=tracker,
        cache=cache,
        storage=storage,
        **values,
    )


def _token() -> str:
    return str(screen_prompt(PROMPT))


def test_a_clip_inside_the_cap_submits_the_original_provider(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    provider = FakeVideoProvider("acme", price=0.02)
    service = _service(
        tracker,
        storage,
        engine,
        provider,
        budget=BudgetConfig(per_video=0.05),
    )

    outcome = _run(service.generate(VideoRequest(prompt=PROMPT), token=_token()))

    assert len(provider.calls) == 1
    assert outcome.clip_budget is not None
    assert outcome.clip_budget.action is ClipBudgetAction.PROCEED
    assert outcome.cheaper_hint is None
    assert outcome.cheaper is False
    assert outcome.event is not None
    assert outcome.event.metadata["clip_budget"]["action"] == "proceed"
    assert "cheaper_hint" not in outcome.event.metadata


def test_an_over_cap_clip_chooses_and_records_the_cheaper_provider(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    pricey = FakeVideoProvider("pricey", price=0.10)
    cheap = FakeVideoProvider("cheap", price=0.02)
    service = _service(
        tracker,
        storage,
        engine,
        pricey,
        cheap,
        budget=BudgetConfig(per_video=0.05),
    )

    outcome = _run(service.generate(VideoRequest(prompt=PROMPT), token=_token()))

    assert pricey.calls == []
    assert len(cheap.calls) == 1
    assert outcome.provider == "cheap"
    assert outcome.downgraded is True
    assert outcome.cheaper_hint is not None
    assert outcome.cheaper_hint.provider == "cheap"
    assert outcome.cheaper_hint.estimate_usd == pytest.approx(0.02)
    assert outcome.hint is outcome.cheaper_hint
    assert outcome.clip_budget is not None
    assert outcome.clip_budget.action is ClipBudgetAction.DOWNGRADE
    assert outcome.to_dict()["cheaper_hint"]["provider"] == "cheap"
    assert outcome.event is not None
    assert outcome.event.metadata["cheaper_hint"]["provider"] == "cheap"
    assert tracker.total() == pytest.approx(0.02)


def test_an_over_cap_clip_without_a_cheaper_option_raises_before_submit(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    provider = FakeVideoProvider("pricey", price=0.10)
    service = _service(
        tracker,
        storage,
        engine,
        provider,
        budget=BudgetConfig(per_video=0.05),
    )

    with pytest.raises(BudgetError) as caught:
        _run(service.generate(VideoRequest(prompt=PROMPT), token=_token()))

    assert provider.calls == []
    assert tracker.count() == 0
    assert caught.value.details["scope"] == "per_clip"
    assert caught.value.details["cap"] == pytest.approx(0.05)
    assert caught.value.details["clip_budget"]["action"] == "blocked"
    assert service.failures
    assert service.failures[0].stage == "budget"
    assert service.failures[0].details["clip_budget"]["action"] == "blocked"


def test_a_pricing_estimate_uses_the_per_clip_cap_before_submit(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    provider = FakeVideoProvider("acme", price=0.0)
    service = _service(
        tracker,
        storage,
        engine,
        provider,
        pricing=VideoPricing(per_second=0.10),
        budget=BudgetConfig(per_video=0.20),
    )

    with pytest.raises(BudgetError):
        _run(
            service.generate(
                VideoRequest(prompt=PROMPT, duration_seconds=5.0),
                token=_token(),
            )
        )

    assert provider.calls == []
    assert tracker.count() == 0


def test_a_call_level_policy_and_candidates_override_the_service_defaults(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    pricey = FakeVideoProvider("pricey", price=0.10)
    cheap = FakeVideoProvider("cheap", price=0.02)
    service = _service(tracker, storage, engine, pricey, cheap)

    outcome = _run(
        service.generate(
            VideoRequest(prompt=PROMPT),
            token=_token(),
            clip_budget=PerClipBudgetPolicy(per_clip=0.05),
            clip_candidates={"pricey": 0.10, "cheap": 0.02},
        )
    )

    assert pricey.calls == []
    assert len(cheap.calls) == 1
    assert outcome.cheaper_hint is not None
    assert outcome.cheaper_hint.provider == "cheap"
    assert outcome.cheaper_hint.savings_usd == pytest.approx(0.08)


def test_a_cache_hit_skips_the_policy_and_keeps_the_recorded_hint_empty(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    provider = FakeVideoProvider("acme", price=0.02)
    service = _service(
        tracker,
        storage,
        engine,
        provider,
        budget=BudgetConfig(per_video=0.05),
    )
    request = VideoRequest(prompt=PROMPT)

    first = _run(service.generate(request, token=_token()))
    second = _run(service.generate(request, token=_token()))

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert len(provider.calls) == 1
    assert second.clip_budget is None
    assert second.cheaper_hint is None
    assert isinstance(second, VideoOutcome)

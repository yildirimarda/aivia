"""The still to video helper keeps both stages' cost events linked."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import struct
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import pytest
from sqlalchemy import Engine

from aivia.core.config.capability import CapabilityChainConfig
from aivia.core.cost import CostTracker
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.models import AssetKind, ProviderMeta, ProviderPrice, RateLimit
from aivia.core.models.asset import Asset
from aivia.core.router import Router
from aivia.media.i2v_flow import I2VFlow
from aivia.media.image_service import IMAGE_OPERATION, ImageService
from aivia.media.screening import screen_prompt
from aivia.media.video_cache import VIDEO_OPERATION, VideoCache
from aivia.providers.base import Provider
from aivia.providers.mock import MockImageProvider
from aivia.providers.protocols.image import ImageRequest
from aivia.providers.protocols.media import MediaInput, VideoFormat
from aivia.providers.protocols.video import (
    CAPABILITY_VIDEO_FIRST_FRAME,
    CAPABILITY_VIDEO_GEN,
    VideoRequest,
    VideoResponse,
    VideoResult,
)
from aivia.providers.storage_local import LocalStorageProvider

STILL_PROMPT = "A lighthouse at sunset"
VIDEO_PROMPT = "The lighthouse beam sweeps across the sea"
VIDEO_PRICE = 0.5

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


def _token(prompt: str = STILL_PROMPT) -> str:
    return str(screen_prompt(prompt))


def _path(asset_path: str) -> Path:
    parsed = urlsplit(asset_path)
    assert parsed.scheme == "file"
    return Path(unquote(parsed.path))


def mp4() -> bytes:
    return struct.pack(">I", 24) + b"ftypisom" + b"\x00" * 12


def probe_json(
    *,
    width: int = 1280,
    height: int = 720,
    duration: str = "5.0",
) -> str:
    streams: list[dict[str, Any]] = [
        {
            "index": 0,
            "codec_type": "video",
            "codec_name": "h264",
            "width": width,
            "height": height,
            "duration": duration,
            "avg_frame_rate": "30/1",
        }
    ]
    return json.dumps(
        {
            "streams": streams,
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
    """A video provider that returns a tiny valid container without a network."""

    def __init__(
        self,
        *,
        price: float = VIDEO_PRICE,
        unit: str = "clip",
    ) -> None:
        super().__init__(
            ProviderMeta(
                name="fake-video",
                capabilities=[
                    CAPABILITY_VIDEO_GEN,
                    CAPABILITY_VIDEO_FIRST_FRAME,
                ],
                pricing={
                    CAPABILITY_VIDEO_GEN: ProviderPrice(
                        amount_usd=price,
                        unit=unit,
                    )
                },
                rate_limit=RateLimit(requests_per_minute=0),
            )
        )
        self.requests: list[VideoRequest] = []

    async def generate(self, request: VideoRequest) -> VideoResponse:
        self.requests.append(request)
        return VideoResponse(
            videos=[
                VideoResult(
                    data=mp4(),
                    format=VideoFormat.MP4,
                    width=1280,
                    height=720,
                    duration_seconds=request.duration_seconds,
                    fps=request.fps,
                    seed=request.seed,
                )
            ],
            model="fake-video-1",
            provider=self.name,
        )


def _router(provider: Provider) -> Router:
    return Router(
        {provider.name: provider},
        config=CapabilityChainConfig(chain=[provider.name]),
    )


def _image_service(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> ImageService:
    provider = MockImageProvider(name="acme-image")
    return ImageService(
        _router(provider),
        tracker=tracker,
        storage=storage,
    )


def _video_cache(
    engine: Engine,
    storage: LocalStorageProvider,
) -> VideoCache:
    return VideoCache(
        engine=engine,
        storage=storage,
        ffprobe_runner=runner_for(probe_json()),
    )


def test_a_generated_still_and_its_animation_keep_both_cost_events(
    engine: Engine,
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeVideoProvider()
    flow = I2VFlow(
        videos=_video_cache(engine, storage),
        images=_image_service(tracker, storage),
        router=_router(provider),
        tracker=tracker,
    )

    outcome = _run(
        flow.animate(
            image_request=ImageRequest(
                prompt=STILL_PROMPT,
                width=64,
                height=64,
            ),
            token=_token(),
            prompt=VIDEO_PROMPT,
            duration_seconds=5.0,
            width=1280,
            height=720,
            fps=30,
            run_id="run_1",
        )
    )

    assert len(provider.requests) == 1
    sent = provider.requests[0]
    assert sent.prompt == VIDEO_PROMPT
    assert sent.first_frame is not None
    assert sent.first_frame.data is not None
    assert sent.first_frame.data.startswith(b"\x89PNG")

    assert outcome.still_outcome is not None
    assert outcome.still_asset is not None
    assert outcome.still_event is not None
    assert outcome.still_event.operation == IMAGE_OPERATION
    assert outcome.still_asset.kind is AssetKind.IMAGE
    assert outcome.still_asset.cost_links[0].cost_event_id == outcome.still_event.id

    assert outcome.video_event is not None
    assert outcome.video_event.operation == VIDEO_OPERATION
    assert outcome.video_event.amount_usd == pytest.approx(VIDEO_PRICE)
    assert outcome.video_event.run_id == "run_1"
    assert outcome.video_asset.kind is AssetKind.VIDEO
    assert outcome.video_asset.cost_links[0].cost_event_id == outcome.video_event.id
    assert outcome.video_asset.cost_links[0].amount_usd == pytest.approx(VIDEO_PRICE)

    assert outcome.events == (outcome.still_event, outcome.video_event)
    assert outcome.amount_usd == pytest.approx(0.52)
    assert tracker.total() == pytest.approx(0.52)
    assert outcome.cache_hit is False
    assert _path(outcome.path).read_bytes() == mp4()


def test_a_selected_still_is_animated_without_generating_an_image(
    engine: Engine,
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeVideoProvider()
    flow = I2VFlow(
        videos=_video_cache(engine, storage),
        router=_router(provider),
        tracker=tracker,
    )

    outcome = _run(
        flow.animate(
            selector=lambda: MediaInput(data=PNG_1X1, media_type="image/png"),
            prompt=VIDEO_PROMPT,
            duration_seconds=5.0,
            run_id="run_2",
        )
    )

    assert provider.requests[0].first_frame is not None
    assert provider.requests[0].first_frame.data == PNG_1X1
    assert outcome.still_outcome is None
    assert outcome.still_asset is None
    assert outcome.still_event is None
    assert outcome.events == (outcome.video_event,)
    assert outcome.video_event is not None
    assert outcome.video_event.amount_usd == pytest.approx(VIDEO_PRICE)
    assert outcome.video_asset.cost_links[0].cost_event_id == outcome.video_event.id
    assert tracker.total() == pytest.approx(VIDEO_PRICE)


def test_a_stored_still_asset_is_used_as_the_first_frame(
    engine: Engine,
    tracker: CostTracker,
    storage: LocalStorageProvider,
    tmp_path: Path,
) -> None:
    still_path = tmp_path / "still.png"
    still_path.write_bytes(PNG_1X1)
    still = Asset(
        kind=AssetKind.IMAGE,
        path=still_path.as_uri(),
        content_hash=hashlib.sha256(PNG_1X1).hexdigest(),
    )
    provider = FakeVideoProvider()
    flow = I2VFlow(
        videos=_video_cache(engine, storage),
        router=_router(provider),
        tracker=tracker,
    )

    outcome = _run(
        flow.animate(
            still,
            prompt=VIDEO_PROMPT,
            duration_seconds=5.0,
        )
    )

    assert provider.requests[0].first_frame is not None
    assert provider.requests[0].first_frame.data == PNG_1X1
    assert outcome.still_asset is still
    assert outcome.still_path == still.path
    assert outcome.still_event is None
    assert outcome.assets == (still, outcome.video_asset)


def test_a_repeated_animation_reuses_the_clip_and_its_original_cost_link(
    engine: Engine,
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeVideoProvider()
    flow = I2VFlow(
        videos=_video_cache(engine, storage),
        router=_router(provider),
        tracker=tracker,
    )
    still = MediaInput(data=PNG_1X1, media_type="image/png")

    first = _run(
        flow.animate(
            still,
            prompt=VIDEO_PROMPT,
            duration_seconds=5.0,
            run_id="run_1",
        )
    )
    second = _run(
        flow.animate(
            still,
            prompt=VIDEO_PROMPT,
            duration_seconds=5.0,
            run_id="run_2",
        )
    )

    assert len(provider.requests) == 1
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.video_asset.id == first.video_asset.id
    assert first.video_event is not None
    assert second.video_asset.cost_links[0].cost_event_id == first.video_event.id
    assert second.video_event is not None
    assert second.video_event.amount_usd == 0.0
    assert second.video_event.operation == VIDEO_OPERATION
    assert second.video_event.run_id == "run_2"
    assert tracker.total() == pytest.approx(VIDEO_PRICE)


def test_a_per_second_video_price_uses_the_requested_duration(
    engine: Engine,
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeVideoProvider(price=0.1, unit="second")
    flow = I2VFlow(
        videos=_video_cache(engine, storage),
        router=_router(provider),
        tracker=tracker,
    )

    outcome = _run(
        flow.animate(
            MediaInput(data=PNG_1X1, media_type="image/png"),
            prompt=VIDEO_PROMPT,
            duration_seconds=5.0,
            run_id="run_seconds",
        )
    )

    assert outcome.video_event is not None
    assert outcome.video_event.units == pytest.approx(5.0)
    assert outcome.video_event.unit_kind == "second"
    assert outcome.video_event.amount_usd == pytest.approx(0.5)
    assert outcome.video_asset.cost_links[0].cost_event_id == outcome.video_event.id
    assert tracker.total() == pytest.approx(0.5)


def test_a_callable_generator_can_report_the_animation_cost(
    engine: Engine,
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    calls: list[VideoRequest] = []

    async def generate(request: VideoRequest) -> VideoResponse:
        calls.append(request)
        return VideoResponse(
            videos=[
                VideoResult(
                    data=mp4(),
                    format=VideoFormat.MP4,
                    width=1280,
                    height=720,
                    duration_seconds=request.duration_seconds,
                )
            ],
            model="local-video",
            provider="local",
            metadata={"cost_usd": 0.25},
        )

    flow = I2VFlow(
        videos=_video_cache(engine, storage),
        generate_video=generate,
        tracker=tracker,
    )

    outcome = _run(
        flow.animate(
            MediaInput(data=PNG_1X1, media_type="image/png"),
            prompt=VIDEO_PROMPT,
            duration_seconds=5.0,
            run_id="run_3",
        )
    )

    assert len(calls) == 1
    assert outcome.video_event is not None
    assert outcome.video_event.amount_usd == pytest.approx(0.25)
    assert outcome.video_event.provider == "local"
    assert outcome.video_event.model == "local-video"
    assert outcome.video_asset.cost_links[0].cost_event_id == outcome.video_event.id
    assert tracker.total() == pytest.approx(0.25)


def test_an_explicit_cost_is_used_when_the_generator_reports_none(
    engine: Engine,
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    async def generate(request: VideoRequest) -> VideoResponse:
        return VideoResponse(
            videos=[
                VideoResult(
                    data=mp4(),
                    format=VideoFormat.MP4,
                    width=1280,
                    height=720,
                    duration_seconds=request.duration_seconds,
                )
            ],
            model="local-video",
            provider="local",
        )

    flow = I2VFlow(
        videos=_video_cache(engine, storage),
        generate_video=generate,
        tracker=tracker,
    )

    outcome = _run(
        flow.animate(
            MediaInput(data=PNG_1X1, media_type="image/png"),
            prompt=VIDEO_PROMPT,
            duration_seconds=5.0,
            cost_usd=0.4,
        )
    )

    assert outcome.video_event is not None
    assert outcome.video_event.amount_usd == pytest.approx(0.4)
    assert outcome.video_event.unit_kind == "clip"
    assert outcome.video_event.units == 1.0


def test_the_flow_requires_a_still_source(
    engine: Engine,
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeVideoProvider()
    flow = I2VFlow(
        videos=_video_cache(engine, storage),
        router=_router(provider),
        tracker=tracker,
    )

    with pytest.raises(ValueError, match="still, selector or image request"):
        _run(flow.animate(prompt=VIDEO_PROMPT, duration_seconds=5.0))

    assert provider.requests == []


def test_the_flow_animates_exactly_one_clip_per_still(
    engine: Engine,
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeVideoProvider()
    flow = I2VFlow(
        videos=_video_cache(engine, storage),
        router=_router(provider),
        tracker=tracker,
    )

    with pytest.raises(ValueError, match="count must be 1"):
        _run(
            flow.animate(
                MediaInput(data=PNG_1X1, media_type="image/png"),
                prompt=VIDEO_PROMPT,
                count=2,
            )
        )

    assert provider.requests == []


def test_the_module_function_and_aliases_are_exported(
    engine: Engine,
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    from aivia.media import (
        StillToVideoFlow,
        animate_still,
        i2v,
        still_to_video,
    )

    assert StillToVideoFlow is I2VFlow
    assert animate_still is i2v
    assert animate_still is still_to_video

    outcome = _run(
        animate_still(
            MediaInput(data=PNG_1X1, media_type="image/png"),
            videos=_video_cache(engine, storage),
            router=_router(FakeVideoProvider()),
            tracker=tracker,
            prompt=VIDEO_PROMPT,
            duration_seconds=5.0,
        )
    )

    assert outcome.video_event is not None
    assert outcome.video_event.amount_usd == pytest.approx(VIDEO_PRICE)

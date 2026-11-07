"""Reference conditioned generation paths rebuild scene by scene when unsupported."""

from __future__ import annotations

import asyncio
import base64
import json
import struct
from collections.abc import Coroutine, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine

from aivia.core.config.capability import CapabilityChainConfig
from aivia.core.cost import CostTracker
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.models import ProviderMeta, ProviderPrice, RateLimit
from aivia.core.router import Router
from aivia.media.image_service import ImageService
from aivia.media.screening import screen_prompt
from aivia.media.video_cache import VideoCache
from aivia.media.video_service import VideoService
from aivia.providers import Provider
from aivia.providers.protocols import (
    CAPABILITY_IMAGE_GEN,
    CAPABILITY_IMAGE_INPUT,
    CAPABILITY_VIDEO_FIRST_FRAME,
    CAPABILITY_VIDEO_GEN,
    CAPABILITY_VIDEO_INPUT,
    ImageFormat,
    ImageRequest,
    ImageResponse,
    ImageResult,
    MediaInput,
    VideoFormat,
    VideoRequest,
    VideoResponse,
    VideoResult,
)
from aivia.providers.storage_local import LocalStorageProvider
from aivia.reference import (
    ConditionedPathResult,
    generate_first_frame_to_video_conditioned,
    generate_image_to_image_conditioned,
    generate_text_to_image_conditioned,
    generate_video_to_video_conditioned,
    rebuild_scene_by_scene,
)
from aivia.reference.schema import (
    ReferenceAnalysis,
    SceneSpec,
    StructureSpec,
)

PROMPT = "A lighthouse at sunset"
PRICE = 0.02
VIDEO_PRICE = 0.05

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coro)


def mp4() -> bytes:
    return struct.pack(">I", 24) + b"ftypisom" + b"\x00" * 12


def probe_json(
    *,
    width: int = 1280,
    height: int = 720,
    duration: str = "5.0",
) -> str:
    streams = [
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
        {"streams": streams, "format": {"duration": duration, "size": "12345"}}
    )


def runner_for(stdout: str, *, calls: list[list[str]] | None = None) -> Any:
    def runner(argv: Sequence[str]) -> Any:
        if calls is not None:
            calls.append(list(argv))
        return type(
            "CompletedProcess", (), {"returncode": 0, "stdout": stdout, "stderr": ""}
        )()

    return runner


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


def _image_service(
    *providers: Provider,
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
    **values: Any,
) -> ImageService:
    return ImageService(
        _router(*providers),
        tracker=tracker,
        storage=storage,
        engine=engine,
        no_cache=True,
        **values,
    )


def _video_service(
    *providers: Provider,
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
    **values: Any,
) -> VideoService:
    probe_calls: list[list[str]] = []
    cache = VideoCache(
        engine=engine,
        storage=storage,
        ffprobe_runner=runner_for(probe_json(), calls=probe_calls),
    )
    return VideoService(
        _router(*providers),
        tracker=tracker,
        cache=cache,
        storage=storage,
        ffprobe_runner=runner_for(probe_json()),
        **values,
    )


def _token(prompt: str = PROMPT) -> str:
    return str(screen_prompt(prompt))


class FakeImageProvider(Provider):
    """An image provider returning valid PNG images and recording its calls."""

    def __init__(
        self,
        name: str = "acme",
        *,
        price: float = PRICE,
        capabilities: Sequence[str] = (CAPABILITY_IMAGE_GEN,),
        model: str = "image-1",
    ) -> None:
        super().__init__(
            ProviderMeta(
                name=name,
                capabilities=list(capabilities),
                pricing={
                    CAPABILITY_IMAGE_GEN: ProviderPrice(
                        amount_usd=price,
                        unit="image",
                    )
                },
                rate_limit=RateLimit(requests_per_minute=0),
            )
        )
        self.model = model
        self.calls: list[ImageRequest] = []

    async def generate(self, request: ImageRequest) -> ImageResponse:
        self.calls.append(request)
        images = [
            ImageResult(
                data=PNG_1X1,
                format=ImageFormat.PNG,
                width=1,
                height=1,
                seed=request.seed,
            )
            for _ in range(request.count)
        ]
        return ImageResponse(
            images=images,
            model=self.model,
            provider=self.name,
        )


class FakeVideoProvider(Provider):
    """A video provider returning inline MP4 bytes and recording its calls."""

    def __init__(
        self,
        name: str = "acme",
        *,
        price: float = VIDEO_PRICE,
        capabilities: Sequence[str] = (CAPABILITY_VIDEO_GEN,),
        model: str = "video-1",
    ) -> None:
        super().__init__(
            ProviderMeta(
                name=name,
                capabilities=list(capabilities),
                pricing={
                    CAPABILITY_VIDEO_GEN: ProviderPrice(
                        amount_usd=price,
                        unit="clip",
                    )
                },
                rate_limit=RateLimit(requests_per_minute=0),
            )
        )
        self.model = model
        self.calls: list[VideoRequest] = []

    async def generate(self, request: VideoRequest) -> VideoResponse:
        self.calls.append(request)
        return VideoResponse(
            videos=[
                VideoResult(
                    data=mp4(),
                    format=VideoFormat.MP4,
                    width=128,
                    height=128,
                    duration_seconds=request.duration_seconds or 1.0,
                    fps=request.fps,
                    seed=request.seed,
                )
                for _ in range(request.count)
            ],
            model=self.model,
            provider=self.name,
        )


def _reference_with_scenes(count: int) -> ReferenceAnalysis:
    scenes = [
        SceneSpec(
            index=index,
            description=f"scene {index} description",
            duration_seconds=2.0,
        )
        for index in range(count)
    ]
    return ReferenceAnalysis(structure=StructureSpec(scenes=scenes))


def _reference_media() -> list[MediaInput]:
    return [MediaInput(url="https://example.com/reference.png", media_type="image/png")]


def test_result_is_a_conditioned_path_result(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    provider = FakeImageProvider()
    service = _image_service(provider, tracker=tracker, storage=storage, engine=engine)
    result = _run(
        generate_text_to_image_conditioned(
            service,
            ImageRequest(prompt=PROMPT),
            token=_token(),
        )
    )
    assert isinstance(result, ConditionedPathResult)
    assert result.used_references is False
    assert result.rebuilt is False
    assert result.warning is None
    assert result.scene_count == 1
    assert len(result.outcomes) == 1


def test_text_to_image_forwards_references_to_capable_provider(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    text_only = FakeImageProvider("text-only", capabilities=(CAPABILITY_IMAGE_GEN,))
    capable = FakeImageProvider(
        "capable",
        capabilities=(CAPABILITY_IMAGE_GEN, CAPABILITY_IMAGE_INPUT),
    )

    result = _run(
        generate_text_to_image_conditioned(
            _image_service(
                text_only, capable, tracker=tracker, storage=storage, engine=engine
            ),
            ImageRequest(prompt=PROMPT),
            reference_media=_reference_media(),
            token=_token(),
        )
    )

    assert result.used_references is True
    assert result.rebuilt is False
    assert len(capable.calls) == 1
    assert len(text_only.calls) == 0
    assert capable.calls[0].reference_images == _reference_media()


def test_text_to_image_skips_text_only_providers(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    text_only = FakeImageProvider("text-only", capabilities=(CAPABILITY_IMAGE_GEN,))
    capable = FakeImageProvider(
        "capable",
        capabilities=(CAPABILITY_IMAGE_GEN, CAPABILITY_IMAGE_INPUT),
    )

    _run(
        generate_text_to_image_conditioned(
            _image_service(
                text_only, capable, tracker=tracker, storage=storage, engine=engine
            ),
            ImageRequest(prompt=PROMPT),
            reference_media=_reference_media(),
            token=_token(),
        )
    )

    assert len(text_only.calls) == 0
    assert len(capable.calls) == 1


def test_text_to_image_prompt_only_when_no_capable_provider(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    text_only = FakeImageProvider("text-only", capabilities=(CAPABILITY_IMAGE_GEN,))

    result = _run(
        generate_text_to_image_conditioned(
            _image_service(text_only, tracker=tracker, storage=storage, engine=engine),
            ImageRequest(prompt=PROMPT),
            reference_media=_reference_media(),
            token=_token(),
        )
    )

    assert result.used_references is False
    assert result.rebuilt is False
    assert result.warning is not None
    assert len(text_only.calls) == 1
    assert text_only.calls[0].reference_images == []


def test_text_to_image_rebuilds_scene_by_scene_when_unsupported(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    text_only = FakeImageProvider("text-only", capabilities=(CAPABILITY_IMAGE_GEN,))
    reference = _reference_with_scenes(3)

    result = _run(
        generate_text_to_image_conditioned(
            _image_service(text_only, tracker=tracker, storage=storage, engine=engine),
            ImageRequest(prompt=PROMPT),
            reference=reference,
            token=_token(),
        )
    )

    assert result.rebuilt is True
    assert result.used_references is False
    assert result.scene_count == 3
    assert len(result.outcomes) == 3
    assert len(text_only.calls) == 3
    for call in text_only.calls:
        assert call.reference_images == []
    prompts = [call.prompt for call in text_only.calls]
    expected = [
        f"{PROMPT} scene {index} description scene {index + 1}" for index in range(3)
    ]
    assert prompts == expected


def test_image_to_image_forwards_references(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    capable = FakeImageProvider(
        "capable",
        capabilities=(CAPABILITY_IMAGE_GEN, CAPABILITY_IMAGE_INPUT),
    )

    result = _run(
        generate_image_to_image_conditioned(
            _image_service(capable, tracker=tracker, storage=storage, engine=engine),
            ImageRequest(prompt=PROMPT),
            reference_media=_reference_media(),
            token=_token(),
        )
    )

    assert result.used_references is True
    assert result.rebuilt is False
    assert len(capable.calls) == 1
    assert capable.calls[0].reference_images == _reference_media()


def test_first_frame_to_video_forwards_first_frame(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    capable = FakeVideoProvider(
        "capable",
        capabilities=(CAPABILITY_VIDEO_GEN, CAPABILITY_VIDEO_FIRST_FRAME),
    )

    result = _run(
        generate_first_frame_to_video_conditioned(
            _video_service(capable, tracker=tracker, storage=storage, engine=engine),
            VideoRequest(prompt=PROMPT),
            reference_media=_reference_media(),
            token=_token(),
        )
    )

    assert result.used_references is True
    assert result.rebuilt is False
    assert len(capable.calls) == 1
    assert capable.calls[0].first_frame is not None
    assert capable.calls[0].first_frame.url == "https://example.com/reference.png"


def test_first_frame_to_video_requires_capable_provider(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    incapable = FakeVideoProvider(
        "incapable",
        capabilities=(CAPABILITY_VIDEO_GEN,),
    )

    result = _run(
        generate_first_frame_to_video_conditioned(
            _video_service(incapable, tracker=tracker, storage=storage, engine=engine),
            VideoRequest(prompt=PROMPT),
            reference_media=_reference_media(),
            token=_token(),
        )
    )

    assert result.used_references is False
    assert result.rebuilt is False
    assert result.warning is not None
    assert len(incapable.calls) == 1
    assert incapable.calls[0].first_frame is None


def test_first_frame_to_video_rebuilds_scene_by_scene_when_unsupported(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    incapable = FakeVideoProvider(
        "incapable",
        capabilities=(CAPABILITY_VIDEO_GEN,),
    )
    reference = _reference_with_scenes(2)

    result = _run(
        generate_first_frame_to_video_conditioned(
            _video_service(incapable, tracker=tracker, storage=storage, engine=engine),
            VideoRequest(prompt=PROMPT, duration_seconds=4.0),
            reference=reference,
            token=_token(),
        )
    )

    assert result.rebuilt is True
    assert result.used_references is False
    assert result.scene_count == 2
    assert len(result.outcomes) == 2
    assert len(incapable.calls) == 2
    for call in incapable.calls:
        assert call.first_frame is None
        assert call.source_video is None
    assert incapable.calls[0].prompt == f"{PROMPT} scene 0 description scene 1"
    assert incapable.calls[1].prompt == f"{PROMPT} scene 1 description scene 2"


def test_video_to_video_routes_to_input_capable_provider(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    source = MediaInput(data=mp4(), media_type="video/mp4")
    capable = FakeVideoProvider(
        "capable",
        capabilities=(CAPABILITY_VIDEO_GEN, CAPABILITY_VIDEO_INPUT),
    )

    result = _run(
        generate_video_to_video_conditioned(
            _video_service(capable, tracker=tracker, storage=storage, engine=engine),
            VideoRequest(prompt=PROMPT, source_video=source),
            token=_token(),
        )
    )

    assert result.used_references is True
    assert result.rebuilt is False
    assert result.warning is None
    assert len(capable.calls) == 1
    assert capable.calls[0].source_video is not None


def test_video_to_video_prompt_only_when_no_input_provider(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    source = MediaInput(data=mp4(), media_type="video/mp4")
    incapable = FakeVideoProvider(
        "incapable",
        capabilities=(CAPABILITY_VIDEO_GEN,),
    )

    result = _run(
        generate_video_to_video_conditioned(
            _video_service(incapable, tracker=tracker, storage=storage, engine=engine),
            VideoRequest(prompt=PROMPT, source_video=source),
            token=_token(),
        )
    )

    assert result.used_references is False
    assert result.rebuilt is False
    assert result.warning is not None
    assert len(incapable.calls) == 1
    assert incapable.calls[0].source_video is None


def test_video_to_video_rebuilds_scene_by_scene_when_unsupported(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    incapable = FakeVideoProvider(
        "incapable",
        capabilities=(CAPABILITY_VIDEO_GEN,),
    )
    reference = _reference_with_scenes(2)

    result = _run(
        generate_video_to_video_conditioned(
            _video_service(incapable, tracker=tracker, storage=storage, engine=engine),
            VideoRequest(prompt=PROMPT),
            reference=reference,
            token=_token(),
        )
    )

    assert result.rebuilt is True
    assert result.used_references is False
    assert result.scene_count == 2
    assert len(incapable.calls) == 2
    for call in incapable.calls:
        assert call.first_frame is None
        assert call.source_video is None


def test_rebuild_scene_by_scene_is_a_listable_helper(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    text_only = FakeImageProvider("text-only", capabilities=(CAPABILITY_IMAGE_GEN,))
    reference = _reference_with_scenes(2)

    service = _image_service(text_only, tracker=tracker, storage=storage, engine=engine)
    token = _token()

    async def generate_one(req: ImageRequest, used_token: Any = None) -> Any:
        effective_token = used_token if used_token is not None else token
        return await service.generate(req, token=effective_token)

    outcomes = _run(
        rebuild_scene_by_scene(
            generate_one,
            ImageRequest(prompt=PROMPT),
            _scenes(reference),
            screen_prompt=service.screen,
            default_token=token,
            run_id="run_1",
            generate_values={},
            build_scene_requests=lambda req, scenes: [
                req.model_copy(
                    update={
                        "reference_images": [],
                        "count": 1,
                        "prompt": f"{PROMPT} scene {i}",
                    }
                )
                for i, _ in enumerate(scenes)
            ],
        )
    )

    assert len(outcomes) == 2
    assert len(text_only.calls) == 2


def _scenes(reference: ReferenceAnalysis) -> list[SceneSpec]:
    return list(reference.structure.scenes)

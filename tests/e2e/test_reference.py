"""End-to-end spec review: the extracted style spec is shown before any paid
generation, and a declined review stops generation without calling a provider."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import struct
from collections.abc import Coroutine, Iterator
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
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
    CAPABILITY_VIDEO_GEN,
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
    ReferenceAnalysis,
    SpecReview,
    analyze_image_reference_offline,
    generate_text_to_image_conditioned,
    generate_video_to_video_conditioned,
    render_style_spec,
    review_style_spec,
)
from aivia.reference.schema import (
    PaletteSpec,
    SceneSpec,
    StructureSpec,
    StyleSpec,
)


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)


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


def runner_for(stdout: str) -> Any:
    def runner(argv: Any) -> Any:
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
    cache = VideoCache(
        engine=engine,
        storage=storage,
        ffprobe_runner=runner_for(probe_json()),
    )
    return VideoService(
        _router(*providers),
        tracker=tracker,
        cache=cache,
        storage=storage,
        ffprobe_runner=runner_for(probe_json()),
        **values,
    )


def _token(prompt: str = "A lighthouse at sunset") -> str:
    return str(screen_prompt(prompt))


class FakeImageProvider(Provider):
    """An image provider returning valid PNG images and recording its calls."""

    def __init__(
        self,
        name: str = "acme",
        *,
        price: float = 0.02,
        capabilities: tuple[str, ...] = (CAPABILITY_IMAGE_GEN,),
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
        price: float = 0.05,
        capabilities: tuple[str, ...] = (CAPABILITY_VIDEO_GEN,),
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


def _reference_bytes() -> bytes:
    image = Image.new("RGBA", (100, 50), (255, 0, 0, 255))
    image.paste((0, 0, 255, 255), (50, 0, 100, 50))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _learned_spec() -> ReferenceAnalysis:
    return analyze_image_reference_offline(_reference_bytes())


# --- review_style_spec unit behaviour ---


def test_review_style_spec_is_not_shown_when_disabled(
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = StyleSpec(name="Cinematic", palette=PaletteSpec(colors=["#112233"]))

    review = review_style_spec(spec, enabled=False)

    assert review.confirmed is True
    assert review.shown is False
    assert review.declined is False
    assert capsys.readouterr().out == ""


def test_review_style_spec_shows_the_spec_and_confirms(
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = StyleSpec(
        name="Cinematic", palette=PaletteSpec(colors=["#112233", "#abcdef"])
    )

    review = review_style_spec(spec, enabled=True, confirm=True)

    assert review.confirmed is True
    assert review.shown is True
    assert review.declined is False
    output = capsys.readouterr().out
    assert "review before paid generation" in output
    assert "#112233" in output
    assert "cinematic" in output.lower()


def test_review_style_spec_shows_the_spec_and_declines(
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = StyleSpec(name="Cinematic", palette=PaletteSpec(colors=["#112233"]))

    review = review_style_spec(spec, enabled=True, confirm=False)

    assert review.confirmed is False
    assert review.shown is True
    assert review.declined is True
    assert "review before paid generation" in capsys.readouterr().out


def test_review_style_spec_reads_stdin_when_confirm_is_not_given() -> None:
    spec = StyleSpec(name="Cinematic")
    stdin = io.StringIO("yes\n")
    stdout = io.StringIO()

    review = review_style_spec(spec, enabled=True, stdin=stdin, stdout=stdout)

    assert review.confirmed is True
    assert "review before paid generation" in stdout.getvalue()


def test_render_style_spec_produces_editable_yaml() -> None:
    spec = StyleSpec(name="Cinematic", palette=PaletteSpec(colors=["#112233"]))

    text = render_style_spec(spec)

    assert "name: Cinematic" in text
    assert "#112233" in text


# --- spec review gates paid generation ---


def test_spec_review_blocks_paid_generation_when_declined(
    capsys: pytest.CaptureFixture[str],
    engine: Engine,
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeImageProvider()
    service = _image_service(provider, tracker=tracker, storage=storage, engine=engine)
    reference = _learned_spec()

    result = _run(
        generate_text_to_image_conditioned(
            service,
            ImageRequest(prompt="A lighthouse at sunset"),
            reference=reference,
            token=_token(),
            review_enabled=True,
            review_confirm=False,
        )
    )

    output = capsys.readouterr().out
    assert "review before paid generation" in output
    assert "#ff0000" in output or "#0000ff" in output
    assert result.review_declined is True
    assert result.used_references is False
    assert result.outcomes == ()
    assert provider.calls == []


def test_spec_review_proceeds_when_confirmed(
    capsys: pytest.CaptureFixture[str],
    engine: Engine,
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeImageProvider()
    service = _image_service(provider, tracker=tracker, storage=storage, engine=engine)
    reference = _learned_spec()

    result = _run(
        generate_text_to_image_conditioned(
            service,
            ImageRequest(prompt="A lighthouse at sunset"),
            reference=reference,
            token=_token(),
            review_enabled=True,
            review_confirm=True,
        )
    )

    assert capsys.readouterr().out != ""
    assert result.review_declined is False
    assert len(provider.calls) == 1
    assert provider.calls[0].prompt == "A lighthouse at sunset"


def test_spec_review_is_skipped_when_disabled(
    capsys: pytest.CaptureFixture[str],
    engine: Engine,
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeImageProvider()
    service = _image_service(provider, tracker=tracker, storage=storage, engine=engine)
    reference = _learned_spec()

    result = _run(
        generate_text_to_image_conditioned(
            service,
            ImageRequest(prompt="A lighthouse at sunset"),
            reference=reference,
            token=_token(),
        )
    )

    assert capsys.readouterr().out == ""
    assert result.review_declined is False
    assert len(provider.calls) == 1


def test_spec_review_does_not_block_a_reference_without_analysis(
    engine: Engine,
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeImageProvider()
    service = _image_service(provider, tracker=tracker, storage=storage, engine=engine)
    reference = ReferenceAnalysis()  # nothing learned

    result = _run(
        generate_text_to_image_conditioned(
            service,
            ImageRequest(prompt="A lighthouse at sunset"),
            reference=reference,
            token=_token(),
            review_enabled=True,
            review_confirm=False,
        )
    )

    assert result.review_declined is False
    assert len(provider.calls) == 1


def test_spec_review_blocks_video_generation_when_declined(
    engine: Engine,
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeVideoProvider()
    service = _video_service(provider, tracker=tracker, storage=storage, engine=engine)
    reference = ReferenceAnalysis(
        structure=StructureSpec(
            scenes=[SceneSpec(duration_seconds=2.0, description="opening")]
        )
    )

    result = _run(
        generate_video_to_video_conditioned(
            service,
            VideoRequest(prompt="Restyle this", source_video=MediaInput(data=mp4())),
            reference=reference,
            token=_token(),
            review_enabled=True,
            review_confirm=False,
        )
    )

    assert result.review_declined is True
    assert result.outcomes == ()
    assert provider.calls == []


def test_spec_review_returns_a_spec_review_instance() -> None:
    spec = StyleSpec(name="Editorial", palette=PaletteSpec(colors=["#123456"]))

    review = review_style_spec(spec, enabled=True, confirm=True, stdout=io.StringIO())

    assert isinstance(review, SpecReview)
    assert review.confirmed is True
    assert review.shown is True

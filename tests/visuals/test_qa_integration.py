"""Visual quality gates log every attempt and its cost per beat."""

from __future__ import annotations

import asyncio
import base64
import json
import struct
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import pytest

from aivia.core.config.quality import QualityConfig
from aivia.core.media_probe import validate_media
from aivia.core.models.asset import Asset, AssetKind
from aivia.core.models.cost import CostEvent
from aivia.core.models.script import MediaKind, VisualType
from aivia.core.stages import StageContext
from aivia.media.image_service import ImageOutcome
from aivia.media.video_qa import VideoFrame
from aivia.media.video_service import VideoOutcome
from aivia.providers.protocols.image import ImageRequest, ImageResponse, ImageResult
from aivia.providers.protocols.media import VideoFormat
from aivia.providers.protocols.stock import StockAsset
from aivia.providers.protocols.video import VideoRequest, VideoResponse, VideoResult
from aivia.visuals import (
    DEFAULT_VISUAL_QA_FILENAME,
    VisualQA,
    VisualQAStage,
    VisualQAStageResult,
    resolve_image_qa,
    resolve_visual_qa,
)
from aivia.visuals.strategy import VisualRequest

PROMPT = "A lighthouse beam sweeping across a stormy sea"
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
MP4 = struct.pack(">I", 24) + b"ftypisom" + b"\x00" * 12


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _request(
    *,
    index: int = 0,
    visual_type: VisualType = VisualType.IMAGE,
    kind: MediaKind = MediaKind.IMAGE,
    prompt: str = PROMPT,
) -> VisualRequest:
    return VisualRequest(
        index=index,
        visual_type=visual_type,
        kind=kind,
        prompt=prompt,
        text=prompt,
    )


def _image_outcome(provider: str = "acme", *, amount: float = 0.02) -> ImageOutcome:
    probe = validate_media(PNG_1X1)
    asset = Asset(
        kind=AssetKind.IMAGE,
        provider=provider,
        model="mock-image",
        prompt=PROMPT,
        content_hash=probe.sha256,
    )
    event = CostEvent(
        provider=provider,
        model="mock-image",
        operation="image_gen",
        units=1.0,
        unit_kind="image",
        amount_usd=amount,
    )
    return ImageOutcome(
        response=ImageResponse(
            images=[ImageResult(data=PNG_1X1)],
            provider=provider,
            model="mock-image",
        ),
        assets=(asset,),
        probes=(probe,),
        events=(event,),
    )


def _video_outcome(provider: str = "acme", *, amount: float = 0.5) -> VideoOutcome:
    asset = Asset(
        kind=AssetKind.VIDEO,
        provider=provider,
        model="mock-video",
        prompt=PROMPT,
    )
    event = CostEvent(
        provider=provider,
        model="mock-video",
        operation="video_gen",
        units=1.0,
        unit_kind="clip",
        amount_usd=amount,
    )
    return VideoOutcome(
        request=VideoRequest(prompt=PROMPT),
        response=VideoResponse(
            videos=[
                VideoResult(
                    data=MP4,
                    format=VideoFormat.MP4,
                    duration_seconds=5.0,
                )
            ],
            provider=provider,
            model="mock-video",
        ),
        assets=(asset,),
        events=(event,),
    )


class ScriptedImageGenerator:
    """Return image outcomes in order and record every call."""

    def __init__(self, chain: list[str], outcomes: list[ImageOutcome]) -> None:
        self.providers = tuple(chain)
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        request: ImageRequest,
        *,
        exclude: set[str] | None = None,
        provider: str | None = None,
        no_cache: bool = True,
        run_id: str = "",
        metadata: Any = None,
        **_: Any,
    ) -> ImageOutcome:
        self.calls.append(
            {
                "request": request,
                "provider": provider,
                "exclude": set(exclude or ()),
                "no_cache": no_cache,
                "run_id": run_id,
                "metadata": metadata,
            }
        )
        return self.outcomes.pop(0)


class ScriptedVideoGenerator:
    """Return video outcomes in order and record every call."""

    def __init__(self, chain: list[str], outcomes: list[VideoOutcome]) -> None:
        self.providers = tuple(chain)
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        request: VideoRequest,
        *,
        exclude: set[str] | None = None,
        provider: str | None = None,
        no_cache: bool = True,
        run_id: str = "",
        metadata: Any = None,
        **_: Any,
    ) -> VideoOutcome:
        self.calls.append(
            {
                "provider": provider,
                "exclude": set(exclude or ()),
                "no_cache": no_cache,
                "run_id": run_id,
                "metadata": metadata,
            }
        )
        return self.outcomes.pop(0)


class ScriptedScorer:
    """Return one score per call from a fixed sequence."""

    def __init__(self, scores: list[float]) -> None:
        self.scores = list(scores)
        self.sources: list[Any] = []

    async def __call__(self, source: Any) -> float:
        self.sources.append(source)
        return self.scores.pop(0) if self.scores else 1.0


class ScriptedSampler:
    """Return one frame per call."""

    def __call__(self, source: Any, **_: Any) -> list[VideoFrame]:
        return [VideoFrame(index=0, timestamp_seconds=0.0, data=PNG_1X1)]


def test_image_quality_retries_log_every_attempt_and_its_cost() -> None:
    generator = ScriptedImageGenerator(
        ["acme"],
        [
            _image_outcome("acme", amount=0.04),
            _image_outcome("acme", amount=0.04),
        ],
    )
    scorer = ScriptedScorer([0.2, 0.9])

    result = _run(
        resolve_image_qa(
            _request(index=2),
            service=generator,
            scorer=scorer,
            config=QualityConfig(min_score=0.7, max_retries=2),
            run_id="run_qa",
        )
    )

    assert result.passed is True
    assert result.kind is MediaKind.IMAGE
    assert result.provider == "acme"
    assert result.score == pytest.approx(0.9)
    assert len(result.attempts) == 2
    assert [attempt.passed for attempt in result.attempts] == [False, True]
    assert [attempt.retry for attempt in result.attempts] == [False, True]
    assert [attempt.cost_usd for attempt in result.attempts] == [0.04, 0.04]
    assert result.attempt_costs == [0.04, 0.04]
    assert result.cost_usd == pytest.approx(0.08)
    assert result.total_cost_usd == pytest.approx(0.08)
    assert result.asset is not None
    assert result.asset.quality_score == pytest.approx(0.9)
    assert generator.calls[0]["run_id"] == "run_qa"
    assert generator.calls[0]["request"].metadata["stage"] == "image"
    assert generator.calls[0]["request"].metadata["beat_index"] == 2


def test_image_quality_falls_back_to_the_next_provider_and_keeps_all_costs() -> None:
    generator = ScriptedImageGenerator(
        ["first", "second"],
        [
            _image_outcome("first", amount=0.02),
            _image_outcome("first", amount=0.02),
            _image_outcome("second", amount=0.03),
        ],
    )
    scorer = ScriptedScorer([0.1, 0.2, 0.9])

    result = _run(
        resolve_image_qa(
            _request(),
            service=generator,
            scorer=scorer,
            config=QualityConfig(min_score=0.7, max_retries=1),
        )
    )

    assert result.passed is True
    assert result.provider == "second"
    assert result.fallback == "provider"
    assert result.used_fallback is True
    assert [attempt.provider for attempt in result.attempts] == [
        "first",
        "first",
        "second",
    ]
    assert [attempt.fallback for attempt in result.attempts] == [
        "",
        "",
        "provider",
    ]
    assert [attempt.cost_usd for attempt in result.attempts] == [0.02, 0.02, 0.03]
    assert result.cost_usd == pytest.approx(0.07)
    assert [call["provider"] for call in generator.calls] == [
        "first",
        "first",
        "second",
    ]


def test_image_quality_stock_fallback_is_logged_with_its_attempt() -> None:
    generator = ScriptedImageGenerator(
        ["acme"],
        [_image_outcome("acme", amount=0.02)],
    )
    stock = StockAsset(
        id="stock-1",
        provider="pexels",
        url="https://cdn.example.com/stock-1.jpg",
        width=1200,
        height=800,
    )

    result = _run(
        resolve_image_qa(
            _request(),
            service=generator,
            scorer=ScriptedScorer([0.1]),
            stock=stock,
            config=QualityConfig(min_score=0.7, max_retries=0),
        )
    )

    assert result.passed is True
    assert result.resolved is True
    assert result.asset is None
    assert result.stock is stock
    assert result.provider == "pexels"
    assert result.fallback == "stock"
    assert [attempt.fallback for attempt in result.attempts] == ["", "stock"]
    assert result.attempts[1].stock is stock
    assert result.cost_usd == pytest.approx(0.02)


def test_video_quality_retries_log_every_attempt_and_its_cost() -> None:
    generator = ScriptedVideoGenerator(
        ["first", "second"],
        [
            _video_outcome("first", amount=0.4),
            _video_outcome("first", amount=0.4),
            _video_outcome("second", amount=0.6),
        ],
    )
    scorer = ScriptedScorer([0.2, 0.3, 0.9])

    result = _run(
        resolve_visual_qa(
            _request(visual_type=VisualType.VIDEO, kind=MediaKind.VIDEO),
            video=generator,
            sampler=ScriptedSampler(),
            scorer=scorer,
            config=QualityConfig(min_score=0.7, max_retries=1, sample_frames=1),
        )
    )

    assert result.passed is True
    assert result.kind is MediaKind.VIDEO
    assert result.provider == "second"
    assert result.fallback == "provider"
    assert len(result.attempts) == 3
    assert [attempt.provider for attempt in result.attempts] == [
        "first",
        "first",
        "second",
    ]
    assert [attempt.cost_usd for attempt in result.attempts] == [0.4, 0.4, 0.6]
    assert result.cost_usd == pytest.approx(1.4)
    assert result.asset is not None
    assert result.asset.quality_score == pytest.approx(0.9)


def test_the_stage_writes_every_attempt_and_cost_to_the_manifest(
    tmp_path: Path,
) -> None:
    generator = ScriptedImageGenerator(
        ["acme"],
        [
            _image_outcome("acme", amount=0.04),
            _image_outcome("acme", amount=0.04),
        ],
    )
    stage = VisualQAStage(
        requests=[_request(index=0)],
        image=generator,
        scorer=ScriptedScorer([0.2, 0.9]),
        config=QualityConfig(min_score=0.7, max_retries=2),
    )

    result = _run(stage.run(StageContext(run_dir=tmp_path, config={})))

    assert isinstance(result, VisualQAStageResult)
    assert result.cost_usd == pytest.approx(0.08)
    assert result.metadata["attempts"] == 2
    assert result.metadata["retries"] == 1
    assert result.metadata["cost_usd"] == pytest.approx(0.08)
    assert len(result.attempts) == 2

    manifest = tmp_path / DEFAULT_VISUAL_QA_FILENAME
    assert manifest.exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["stage"] == "visual_qa"
    assert payload["metadata"]["attempts"] == 2
    assert payload["metadata"]["cost_usd"] == pytest.approx(0.08)
    beat = payload["beats"][0]
    assert [attempt["passed"] for attempt in beat["attempts"]] == [False, True]
    assert [attempt["cost_usd"] for attempt in beat["attempts"]] == [0.04, 0.04]
    assert beat["cost_usd"] == pytest.approx(0.08)
    assert beat["asset"]["quality_score"] == pytest.approx(0.9)


def test_the_stage_dispatches_images_and_videos_and_skips_stock(
    tmp_path: Path,
) -> None:
    image_generator = ScriptedImageGenerator(
        ["acme"],
        [_image_outcome("acme", amount=0.02)],
    )
    video_generator = ScriptedVideoGenerator(
        ["acme"],
        [_video_outcome("acme", amount=0.5)],
    )
    requests = [
        _request(index=0),
        _request(index=1, visual_type=VisualType.STOCK, kind=MediaKind.STOCK),
        _request(index=2, visual_type=VisualType.VIDEO, kind=MediaKind.VIDEO),
    ]
    stage = VisualQAStage(
        requests=requests,
        image=image_generator,
        video=video_generator,
        sampler=ScriptedSampler(),
        scorer=ScriptedScorer([0.9, 0.9]),
        config=QualityConfig(min_score=0.7, sample_frames=1),
    )

    result = _run(stage.run(StageContext(run_dir=tmp_path, config={})))

    assert isinstance(result, VisualQAStageResult)
    assert [beat.kind for beat in result.beats] == [
        MediaKind.IMAGE,
        MediaKind.VIDEO,
    ]
    assert [beat.index for beat in result.beats] == [0, 2]
    assert result.cost_usd == pytest.approx(0.52)
    assert len(image_generator.calls) == 1
    assert len(video_generator.calls) == 1


def test_an_empty_prompt_is_unresolved_without_calling_a_service() -> None:
    generator = ScriptedImageGenerator(["acme"], [])

    result = _run(
        resolve_visual_qa(
            VisualRequest(index=4, visual_type=VisualType.IMAGE, prompt=""),
            image=generator,
        )
    )

    assert result.passed is False
    assert result.resolved is False
    assert result.attempts == []
    assert result.metadata["reason"] == "no prompt"
    assert generator.calls == []


def test_stock_beats_are_left_to_the_stock_stage() -> None:
    quality = VisualQA()

    result = _run(
        quality.run(_request(visual_type=VisualType.STOCK, kind=MediaKind.STOCK))
    )

    assert result.passed is False
    assert result.kind is MediaKind.STOCK
    assert result.attempts == []
    assert result.metadata["reason"] == "stock quality is handled by the stock stage"

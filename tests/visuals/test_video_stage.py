"""Video beats resolve with durations clamped to the beat, cap and platform."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from aivia.core.config.capability import CapabilityChainConfig
from aivia.core.models.asset import Asset, AssetKind, compute_prompt_hash
from aivia.core.models.script import MediaKind, VisualType
from aivia.core.router import Router
from aivia.core.stages import StageContext
from aivia.providers.mock import MockVideoProvider
from aivia.providers.protocols.video import VideoRequest
from aivia.visuals import (
    DEFAULT_VIDEO_FILENAME,
    DEFAULT_VIDEOS_FILENAME,
    VideoStage,
    VideoStageResult,
    clamp_video_duration,
    resolve_video_request,
)
from aivia.visuals.strategy import VisualRequest


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _request(
    prompt: str,
    *,
    index: int = 0,
    visual_type: VisualType = VisualType.VIDEO,
    duration_seconds: float | None = None,
    aspect_ratio: str = "",
    metadata: dict[str, Any] | None = None,
    **values: Any,
) -> VisualRequest:
    payload: dict[str, Any] = {
        "index": index,
        "visual_type": visual_type,
        "kind": (
            MediaKind.VIDEO if visual_type is VisualType.VIDEO else MediaKind.IMAGE
        ),
        "prompt": prompt,
        "text": prompt,
        "duration_seconds": duration_seconds,
        "aspect_ratio": aspect_ratio,
        "metadata": metadata or {},
        **values,
    }
    return VisualRequest.model_validate(payload)


class FakeVideoService:
    """Return one stored asset per request and remember every typed request."""

    def __init__(self) -> None:
        self.requests: list[VideoRequest] = []
        self.runs: list[tuple[str, dict[str, Any] | None]] = []

    async def generate(
        self,
        request: VideoRequest,
        *,
        run_id: str = "",
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> SimpleNamespace:
        self.requests.append(request)
        self.runs.append((run_id, metadata))
        asset = Asset(
            kind=AssetKind.VIDEO,
            prompt=request.prompt,
            provider="fake-video",
            model=request.model or "fake-model",
            path=f"file:///tmp/{request.prompt.replace(' ', '-')}.mp4",
            run_id=run_id,
            seed=request.seed,
        )
        return SimpleNamespace(
            asset=asset,
            assets=(asset,),
            provider="fake-video",
            model=asset.model,
            cost_usd=0.5,
            cached=False,
        )


def test_a_video_beat_is_sent_through_the_service_with_its_options() -> None:
    service = FakeVideoService()
    request = _request(
        "A lighthouse at sunset",
        index=3,
        duration_seconds=4.0,
        aspect_ratio="16:9",
        negative_prompt="no text",
        seed=7,
        metadata={
            "model": "beat-model",
            "fps": 24,
            "generate_audio": True,
        },
    )

    result = _run(
        resolve_video_request(
            request,
            service=service,
            run_id="run_1",
            metadata={"position": 3},
        )
    )

    assert len(service.requests) == 1
    sent = service.requests[0]
    assert sent.prompt == "A lighthouse at sunset"
    assert sent.negative_prompt == "no text"
    assert sent.aspect_ratio == "16:9"
    assert sent.duration_seconds == 4.0
    assert sent.seed == 7
    assert sent.model == "beat-model"
    assert sent.count == 1
    assert sent.fps == 24
    assert sent.generate_audio is True
    assert sent.metadata["stage"] == "video"
    assert sent.metadata["beat_index"] == 3
    assert service.runs[0][0] == "run_1"

    assert result.resolved is True
    assert result.asset is not None
    assert result.asset.kind is AssetKind.VIDEO
    assert result.asset.prompt == "A lighthouse at sunset"
    assert result.asset.prompt_hash == compute_prompt_hash("A lighthouse at sunset")
    assert result.asset.provider == "fake-video"
    assert result.asset.model == "beat-model"
    assert result.prompt_hash == result.asset.prompt_hash
    assert result.provider == "fake-video"
    assert result.model == "beat-model"
    assert result.cost_usd == 0.5
    assert result.duration_seconds == 4.0
    assert result.clamped is False


def test_duration_is_clamped_to_the_beat_and_the_per_clip_cap(
    tmp_path: Path,
) -> None:
    service = FakeVideoService()
    requests = [
        _request("short beat", index=0, duration_seconds=4.0),
        _request("long beat", index=1, duration_seconds=20.0),
        _request("no duration", index=2),
    ]
    stage = VideoStage(requests=requests, service=service, max_clip_seconds=8.0)

    result = cast(
        VideoStageResult, _run(stage.run(StageContext(run_dir=tmp_path, config={})))
    )

    assert isinstance(result, VideoStageResult)
    assert [request.duration_seconds for request in service.requests] == [
        4.0,
        8.0,
        None,
    ]
    assert result.durations == {0: 4.0, 1: 8.0, 2: None}
    assert result.beats[0].clamped is False
    assert result.beats[1].clamped is True
    assert result.beats[1].max_clip_seconds == 8.0
    assert result.metadata["max_clip_seconds"] == 8.0


def test_a_beat_duration_caps_a_longer_requested_clip() -> None:
    service = FakeVideoService()
    request = _request(
        "long requested clip",
        duration_seconds=10.0,
        metadata={"beat_duration_seconds": 3.0},
    )

    result = _run(
        resolve_video_request(
            request,
            service=service,
            max_clip_seconds=8.0,
        )
    )

    sent = service.requests[0]
    assert sent.duration_seconds == 3.0
    assert result.requested_duration_seconds == 10.0
    assert result.beat_duration_seconds == 3.0
    assert result.duration_seconds == 3.0
    assert result.clamped is True


def test_the_platform_aspect_ratio_is_applied_to_every_video_beat(
    tmp_path: Path,
) -> None:
    service = FakeVideoService()
    stage = VideoStage(
        requests=[_request("portrait clip", aspect_ratio="16:9")],
        service=service,
        platform="youtube",
    )

    result = cast(
        VideoStageResult, _run(stage.run(StageContext(run_dir=tmp_path, config={})))
    )

    assert service.requests[0].aspect_ratio == "9:16"
    assert result.beats[0].aspect_ratio == "9:16"
    assert result.beats[0].platform == "youtube"
    assert result.metadata["platform"] == "youtube"
    assert result.metadata["aspect_ratio"] == "9:16"


def test_an_explicit_aspect_ratio_wins_over_the_platform(tmp_path: Path) -> None:
    service = FakeVideoService()
    stage = VideoStage(
        requests=[_request("square clip")],
        service=service,
        platform="youtube",
        aspect_ratio="1:1",
    )

    _run(stage.run(StageContext(run_dir=tmp_path, config={})))

    assert service.requests[0].aspect_ratio == "1:1"


def test_the_stage_resolves_only_video_beats_and_writes_the_manifest(
    tmp_path: Path,
) -> None:
    service = FakeVideoService()
    requests = [
        _request("A generated clip", index=0, duration_seconds=5.0),
        _request("a stock query", index=1, visual_type=VisualType.STOCK),
        _request("An image", index=2, visual_type=VisualType.IMAGE),
    ]
    stage = VideoStage(requests=requests, service=service)

    result = cast(
        VideoStageResult, _run(stage.run(StageContext(run_dir=tmp_path, config={})))
    )

    assert [beat.index for beat in result.beats] == [0]
    assert [request.prompt for request in service.requests] == ["A generated clip"]
    assert [asset.prompt_hash for asset in result.assets] == [
        compute_prompt_hash("A generated clip")
    ]
    assert result.metadata["beats"] == 1
    assert result.metadata["resolved"] == 1
    assert result.metadata["unresolved"] == 0
    assert result.metadata["providers"] == ["fake-video"]
    assert result.metadata["models"] == ["fake-model"]
    assert result.cost_usd == 0.5

    manifest = tmp_path / DEFAULT_VIDEO_FILENAME
    assert manifest.exists()
    assert (tmp_path / DEFAULT_VIDEOS_FILENAME).exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["stage"] == "video"
    assert [beat["index"] for beat in payload["beats"]] == [0]
    first = payload["beats"][0]
    assert first["prompt_hash"] == compute_prompt_hash("A generated clip")
    assert first["provider"] == "fake-video"
    assert first["model"] == "fake-model"
    assert first["duration_seconds"] == 5.0
    assert first["asset"]["prompt_hash"] == first["prompt_hash"]
    assert first["asset"]["provider"] == "fake-video"


def test_an_empty_prompt_is_unresolved_without_calling_the_service(
    tmp_path: Path,
) -> None:
    service = FakeVideoService()
    stage = VideoStage(requests=[_request("", index=0)], service=service)

    result = cast(
        VideoStageResult, _run(stage.run(StageContext(run_dir=tmp_path, config={})))
    )

    assert service.requests == []
    assert result.beats[0].resolved is False
    assert result.beats[0].error == ""
    assert result.beats[0].metadata["reason"] == "no prompt"
    assert result.unresolved == result.beats
    assert result.metadata["unresolved"] == 1
    assert (tmp_path / DEFAULT_VIDEO_FILENAME).exists()


def test_the_per_clip_cap_defaults_from_the_video_capability_config(
    tmp_path: Path,
) -> None:
    service = FakeVideoService()
    provider = MockVideoProvider(name="mock-video")
    router = Router(
        {provider.name: provider},
        config=CapabilityChainConfig(chain=[provider.name], max_clip_seconds=6.0),
    )
    stage = VideoStage(
        requests=[_request("long beat", duration_seconds=20.0)],
        service=service,
        router=router,
    )

    result = cast(
        VideoStageResult,
        _run(stage.run(StageContext(run_dir=tmp_path, config={}, router=router))),
    )

    assert service.requests[0].duration_seconds == 6.0
    assert result.beats[0].max_clip_seconds == 6.0


class FakeMedia:
    """A minimal media facade exposing only the video method the stage uses."""

    def __init__(self) -> None:
        self.requests: list[VideoRequest] = []

    async def video(
        self,
        request: VideoRequest,
        *,
        run_id: str = "",
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> SimpleNamespace:
        self.requests.append(request)
        asset = Asset(
            kind=AssetKind.VIDEO,
            prompt=request.prompt,
            provider="fake-media",
            model="fake-media-model",
            path="file:///tmp/media.mp4",
            run_id=run_id,
        )
        return SimpleNamespace(
            asset=asset,
            assets=(asset,),
            provider="fake-media",
            model="fake-media-model",
            cost_usd=0.2,
            cached=False,
        )


def test_the_default_path_uses_the_stage_context_media_facade(
    tmp_path: Path,
) -> None:
    media = FakeMedia()
    context = StageContext(run_dir=tmp_path, config={})
    context._media = media
    stage = VideoStage(requests=[_request("A lighthouse at sunset")])

    result = cast(VideoStageResult, _run(stage.run(context)))

    assert media.requests[0].prompt == "A lighthouse at sunset"
    assert result.beats[0].resolved is True
    assert result.beats[0].provider == "fake-media"
    assert result.beats[0].model == "fake-media-model"


def test_clamp_video_duration_returns_none_without_a_known_duration() -> None:
    assert clamp_video_duration(None, max_clip_seconds=8.0) is None
    assert clamp_video_duration(12.0, max_clip_seconds=8.0) == 8.0
    assert clamp_video_duration(12.0, beat_seconds=5.0, max_clip_seconds=8.0) == 5.0
    assert clamp_video_duration(None, beat_seconds=5.0) == 5.0
    assert clamp_video_duration(4.0, max_clip_seconds=0) == 4.0

"""The built-in AI video short format generates one clip for every beat."""

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import Coroutine
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from aivia.compose import (
    Timeline,
    VideoLayer,
    VideoRenderResult,
    render_video,
    timeline_to_compositions,
)
from aivia.core.config import VideoConfig
from aivia.core.config.formats import FormatRecipeConfig, OutputHandler, OutputType
from aivia.core.models.asset import Asset, AssetKind
from aivia.core.models.script import (
    Beat,
    MotionPrompt,
    Script,
    VisualType,
)
from aivia.core.stages import StageContext
from aivia.formats import load_packaged_formats
from aivia.providers.mock import MockVideoProvider
from aivia.visuals import (
    VisualReuseStage,
    VisualReuseStageResult,
    resolve_visual_requests,
)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _ai_video_short() -> FormatRecipeConfig:
    return load_packaged_formats().get("ai_video_short")


def _script(count: int = 5) -> Script:
    return Script(
        title="Five quick ideas",
        beats=[
            Beat(
                index=index,
                text=f"Short beat {index + 1} explains one idea.",
                media_prompts=[
                    MotionPrompt(
                        prompt=f"Generated motion shot {index + 1}.",
                        duration_seconds=1.0,
                        aspect_ratio="9:16",
                    )
                ],
            )
            for index in range(count)
        ],
    )


class _RecordingService:
    """Return a stored asset per request and count every call."""

    def __init__(self, kind: AssetKind, cost_usd: float) -> None:
        self.kind = kind
        self.cost_usd = cost_usd
        self.requests: list[Any] = []

    async def generate(self, request: Any, **_: Any) -> SimpleNamespace:
        self.requests.append(request)
        suffix = "mp4" if self.kind is AssetKind.VIDEO else "png"
        asset = Asset(
            kind=self.kind,
            prompt=request.prompt,
            provider=f"fake-{self.kind.value}",
            model=f"fake-{self.kind.value}-model",
            path=f"file:///tmp/{len(self.requests)}.{suffix}",
        )
        return SimpleNamespace(
            asset=asset,
            assets=(asset,),
            provider=asset.provider,
            model=asset.model,
            cost_usd=self.cost_usd,
            cached=False,
        )


class _RealVideoService(_RecordingService):
    """Use the mock video provider but write its bytes for the renderer."""

    def __init__(self, directory: Path) -> None:
        super().__init__(AssetKind.VIDEO, 0.0)
        self.provider = MockVideoProvider(name="mock-video", seed=7)
        self.directory = directory

    async def generate(self, request: Any, **_: Any) -> SimpleNamespace:
        response = await self.provider.generate(request)
        self.requests.append(request)
        result = response.videos[0]
        path = self.directory / f"clip-{len(self.requests)}.mp4"
        assert result.data is not None
        path.write_bytes(result.data)
        asset = Asset(
            kind=AssetKind.VIDEO,
            prompt=request.prompt,
            provider=response.provider,
            model=response.model,
            path=str(path),
        )
        return SimpleNamespace(
            asset=asset,
            assets=(asset,),
            provider=asset.provider,
            model=asset.model,
            cost_usd=0.0,
            cached=False,
        )


def test_the_packaged_ai_video_short_recipe_generates_video_for_every_beat() -> None:
    recipe = _ai_video_short()

    assert [stage.stage for stage in recipe.stages] == [
        "input_topic",
        "write",
        "voice",
        "visuals",
        "render",
    ]
    assert recipe.stages[3].options["strategy"] == "video"
    assert recipe.capabilities == ["llm", "tts", "video_gen"]
    assert recipe.output_type is OutputType.VIDEO
    assert recipe.handler is OutputHandler.SAVE


def test_ai_video_short_assigns_a_generated_clip_to_every_beat() -> None:
    recipe = _ai_video_short()

    requests = resolve_visual_requests(
        _script(5),
        **recipe.stages[3].options,
    )

    assert [request.visual_type for request in requests] == [VisualType.VIDEO] * 5
    assert all(request.strategy == "video" for request in requests)
    assert [request.duration_seconds for request in requests] == [1.0] * 5


def test_ai_video_short_visuals_stage_generates_one_clip_per_beat(
    tmp_path: Path,
) -> None:
    recipe = _ai_video_short()
    videos = _RecordingService(AssetKind.VIDEO, 0.50)
    stage = VisualReuseStage(
        scripts={"en": _script(5)},
        source_language="en",
        video=videos,
        **recipe.stages[3].options,
    )

    result = cast(
        VisualReuseStageResult,
        _run(stage.run(StageContext(run_dir=tmp_path, config={}))),
    )

    assert len(videos.requests) == 5
    assert [request.visual_type for request in result.requests] == [
        VisualType.VIDEO
    ] * 5
    assert len(result.beats) == 5
    assert result.cost_usd == pytest.approx(2.50)
    manifest = json.loads((tmp_path / "visual-reuse.json").read_text(encoding="utf-8"))
    assert [beat["asset"]["kind"] for beat in manifest["beats"]] == ["video"] * 5


@pytest.mark.slow
@pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None,
    reason="ffmpeg and ffprobe are required for a real AI video short render",
)
def test_the_ai_video_short_renders_one_clip_per_beat(tmp_path: Path) -> None:
    recipe = _ai_video_short()
    videos = _RealVideoService(tmp_path)
    stage = VisualReuseStage(
        scripts={"en": _script(4)},
        source_language="en",
        video=videos,
        **recipe.stages[3].options,
    )

    _run(stage.run(StageContext(run_dir=tmp_path, config={})))
    manifest = json.loads((tmp_path / "visual-reuse.json").read_text(encoding="utf-8"))
    beats: list[dict[str, Any]] = []
    assets: list[dict[str, Any]] = []
    for position, entry in enumerate(manifest["beats"]):
        beat_id = f"beat_{position}"
        beats.append(
            {
                "id": beat_id,
                "start_seconds": float(position),
                "end_seconds": float(position + 1),
            }
        )
        assets.append(
            {
                "id": f"asset_{position}",
                "beat": beat_id,
                "source": entry["asset"]["path"],
                "kind": "video",
            }
        )
    timeline = Timeline.model_validate(
        {"id": "ai-video-short", "beats": beats, "assets": assets}
    )
    video = VideoConfig.from_mapping({"resolution": "320x240", "fps": 12})

    composition = timeline_to_compositions(timeline, video=video)[0]

    video_layers = [
        layer for layer in composition.layers if isinstance(layer, VideoLayer)
    ]
    assert len(video_layers) == 4
    assert all(layer.muted for layer in video_layers)

    output = tmp_path / "ai-video-short.mp4"
    rendered = render_video(composition, output)

    assert isinstance(rendered, VideoRenderResult)
    assert output.is_file()
    assert output.stat().st_size > 0
    assert rendered.probe.primary_video is not None

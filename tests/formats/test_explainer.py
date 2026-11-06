"""The built-in explainer format mixes AI video and images under one recipe."""

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
    VideoRenderResult,
    render_video,
    timeline_to_compositions,
)
from aivia.core.config import VideoConfig
from aivia.core.config.formats import (
    FormatRecipeConfig,
    OutputHandler,
    OutputType,
)
from aivia.core.models.asset import Asset, AssetKind
from aivia.core.models.script import (
    Beat,
    ImagePrompt,
    MotionPrompt,
    Script,
    VisualType,
)
from aivia.core.stages import StageContext
from aivia.formats import load_packaged_formats
from aivia.providers.mock import MockImageProvider, MockVideoProvider
from aivia.visuals import (
    VisualReuseStage,
    VisualReuseStageResult,
    resolve_visual_requests,
)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _explainer() -> FormatRecipeConfig:
    return load_packaged_formats().get("explainer")


def _script(count: int = 4) -> Script:
    return Script(
        title="Why the sky is blue",
        beats=[
            Beat(
                index=index,
                text=f"Explainer beat {index + 1}.",
                media_prompts=[
                    ImagePrompt(
                        prompt=f"Still illustration {index + 1}.",
                        aspect_ratio="9:16",
                    ),
                    MotionPrompt(
                        prompt=f"Motion shot {index + 1}.",
                        duration_seconds=1.0,
                        aspect_ratio="9:16",
                    ),
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


class _RealMediaService(_RecordingService):
    """Use a mock provider but write its real bytes so the renderer can read them."""

    def __init__(self, provider: Any, kind: AssetKind, directory: Path) -> None:
        super().__init__(kind, 0.0)
        self.provider = provider
        self.directory = directory

    async def generate(self, request: Any, **_: Any) -> SimpleNamespace:
        response = await self.provider.generate(request)
        self.requests.append(request)
        if self.kind is AssetKind.VIDEO:
            result = response.videos[0]
            suffix = "mp4"
        else:
            result = response.images[0]
            suffix = "png"
        path = self.directory / f"{self.kind.value}-{len(self.requests)}.{suffix}"
        assert result.data is not None
        path.write_bytes(result.data)
        asset = Asset(
            kind=self.kind,
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


def test_the_packaged_explainer_recipe_uses_hybrid_visuals_after_narration() -> None:
    recipe = _explainer()

    assert [stage.stage for stage in recipe.stages] == [
        "input_topic",
        "write",
        "voice",
        "visuals",
        "render",
    ]
    assert recipe.stages[1].options == {"mode": "facts"}
    assert recipe.stages[3].options == {"strategy": "hybrid"}
    assert recipe.capabilities == ["llm", "tts", "image_gen", "video_gen"]
    assert recipe.output_type is OutputType.VIDEO
    assert recipe.handler is OutputHandler.SAVE


def test_explainer_visuals_assign_video_first_then_images() -> None:
    recipe = _explainer()

    requests = resolve_visual_requests(
        _script(5),
        **recipe.stages[3].options,
    )

    assert [request.visual_type for request in requests] == [
        VisualType.VIDEO,
        VisualType.VIDEO,
        VisualType.IMAGE,
        VisualType.IMAGE,
        VisualType.IMAGE,
    ]
    assert all(request.strategy == "hybrid" for request in requests)


def test_explainer_visuals_stage_generates_the_hybrid_set_once(
    tmp_path: Path,
) -> None:
    recipe = _explainer()
    images = _RecordingService(AssetKind.IMAGE, 0.04)
    videos = _RecordingService(AssetKind.VIDEO, 0.50)
    stage = VisualReuseStage(
        scripts={"en": _script(4)},
        source_language="en",
        image=images,
        video=videos,
        **recipe.stages[3].options,
    )

    result = cast(
        VisualReuseStageResult,
        _run(stage.run(StageContext(run_dir=tmp_path, config={}))),
    )

    assert len(videos.requests) == 2
    assert len(images.requests) == 2
    assert [request.visual_type for request in result.requests] == [
        VisualType.VIDEO,
        VisualType.VIDEO,
        VisualType.IMAGE,
        VisualType.IMAGE,
    ]
    assert result.cost_usd == pytest.approx(1.08)
    assert (tmp_path / "visual-reuse.json").is_file()


@pytest.mark.slow
@pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None,
    reason="ffmpeg and ffprobe are required for a real explainer render",
)
def test_the_explainer_renders_its_hybrid_visuals_to_a_video(tmp_path: Path) -> None:
    recipe = _explainer()
    images = _RealMediaService(
        MockImageProvider(name="mock-image", seed=7),
        AssetKind.IMAGE,
        tmp_path,
    )
    videos = _RealMediaService(
        MockVideoProvider(name="mock-video", seed=7),
        AssetKind.VIDEO,
        tmp_path,
    )
    stage = VisualReuseStage(
        scripts={"en": _script(4)},
        source_language="en",
        image=images,
        video=videos,
        **recipe.stages[3].options,
    )

    _run(stage.run(StageContext(run_dir=tmp_path, config={})))
    manifest = json.loads((tmp_path / "visual-reuse.json").read_text(encoding="utf-8"))
    beats: list[dict[str, Any]] = []
    assets: list[dict[str, Any]] = []
    for position, beat in enumerate(manifest["beats"]):
        beat_id = f"beat_{position}"
        asset = beat["asset"]
        visual_type = beat["request"]["visual_type"]
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
                "source": asset["path"],
                "kind": "video" if visual_type == "video" else "image",
            }
        )
    timeline = Timeline.model_validate(
        {"id": "explainer", "beats": beats, "assets": assets}
    )
    video = VideoConfig.from_mapping({"resolution": "320x240", "fps": 12})

    compositions = timeline_to_compositions(timeline, video=video)
    assert len(compositions) == 1
    assert [layer.type.value for layer in compositions[0].layers] == [
        "video",
        "video",
        "image",
        "image",
    ]

    output = tmp_path / "explainer.mp4"
    rendered = render_video(compositions[0], output)

    assert isinstance(rendered, VideoRenderResult)
    assert output.is_file()
    assert output.stat().st_size > 0
    assert rendered.probe.primary_video is not None

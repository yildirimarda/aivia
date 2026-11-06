"""The built-in narrated still format turns a few images and narration into a video."""

from __future__ import annotations

import asyncio
import json
import math
import shutil
import struct
import wave
from collections.abc import Coroutine
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from aivia.audio import SubtitleStage, SubtitleStageResult, VoiceStage, VoiceStageResult
from aivia.compose import (
    ImageLayer,
    TextLayer,
    Timeline,
    VideoRenderResult,
    render_video,
    timeline_to_compositions,
)
from aivia.core.config import VideoConfig
from aivia.core.config.formats import FormatRecipeConfig, OutputHandler, OutputType
from aivia.core.models.asset import Asset, AssetKind
from aivia.core.models.script import Beat, ImagePrompt, Script
from aivia.core.stages import StageContext
from aivia.formats import load_packaged_formats
from aivia.providers.mock import MockImageProvider
from aivia.visuals import (
    VisualReuseStage,
    VisualReuseStageResult,
    resolve_visual_requests,
)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
SAMPLE_RATE = 24_000


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _narrated_still() -> FormatRecipeConfig:
    return load_packaged_formats().get("narrated_still")


def _script(count: int = 3) -> Script:
    return Script(
        title="Why the sky is blue",
        beats=[
            Beat(
                index=index,
                text=f"Beat {index + 1} explains one part of the sky.",
                media_prompts=[
                    ImagePrompt(
                        prompt=f"Calm still illustration {index + 1}.",
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


class _RealImageService(_RecordingService):
    """Use the mock provider but write its bytes so the renderer can read them."""

    def __init__(self, directory: Path) -> None:
        super().__init__(AssetKind.IMAGE, 0.0)
        self.provider = MockImageProvider(name="mock-image", seed=7)
        self.directory = directory

    async def generate(self, request: Any, **_: Any) -> SimpleNamespace:
        response = await self.provider.generate(request)
        self.requests.append(request)
        result = response.images[0]
        path = self.directory / f"image-{len(self.requests)}.png"
        assert result.data is not None
        path.write_bytes(result.data)
        asset = Asset(
            kind=AssetKind.IMAGE,
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


class _ToneSpeech:
    """Write a real wav per request and return word timings for the text."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.requests: list[Any] = []

    def _write(self, path: Path, duration: float) -> None:
        frames = int(duration * SAMPLE_RATE)
        with wave.open(str(path), "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(SAMPLE_RATE)
            samples = bytearray()
            for index in range(frames):
                value = int(4000 * math.sin(2 * math.pi * 440 * index / SAMPLE_RATE))
                samples.extend(struct.pack("<h", value))
            target.writeframes(bytes(samples))

    async def __call__(self, request: Any) -> SimpleNamespace:
        self.requests.append(request)
        words = str(request.text).split()
        duration = max(0.5, len(words) * 0.35)
        path = self.directory / f"voice-{len(self.requests)}.wav"
        self._write(path, duration)
        step = duration / max(len(words), 1)
        timestamps = [
            {
                "word": word,
                "start_seconds": index * step,
                "end_seconds": (index + 1) * step,
            }
            for index, word in enumerate(words)
        ]
        return SimpleNamespace(
            duration_seconds=duration,
            path=str(path),
            cost_usd=0.01,
            provider="fake-tts",
            model="fake-tts-model",
            word_timestamps=timestamps,
        )


def _merge_visuals(timeline: dict[str, Any], manifest: dict[str, Any]) -> Timeline:
    assets: list[dict[str, Any]] = []
    for position, entry in enumerate(manifest["beats"]):
        beat = timeline["beats"][position]
        asset_id = f"asset_{position}"
        assets.append(
            {
                "id": asset_id,
                "beat": beat["id"],
                "source": entry["asset"]["path"],
                "kind": "image",
            }
        )
        beat["asset"] = asset_id
    timeline["assets"] = assets
    return Timeline.model_validate(timeline)


def _narrated_timeline(tmp_path: Path) -> tuple[Timeline, Script]:
    recipe = _narrated_still()
    script = _script(3)
    visuals = VisualReuseStage(
        scripts={"en": script},
        source_language="en",
        image=_RealImageService(tmp_path),
        **recipe.stages[3].options,
    )
    _run(visuals.run(StageContext(run_dir=tmp_path, config={})))

    voice = VoiceStage(
        scripts={"en": script},
        speech=_ToneSpeech(tmp_path / "audio"),
        languages=["en"],
    )
    _run(voice.run(StageContext(run_dir=tmp_path, config={})))

    manifest = json.loads((tmp_path / "visual-reuse.json").read_text(encoding="utf-8"))
    payload = json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    return _merge_visuals(payload, manifest), script


def test_the_packaged_narrated_still_recipe_uses_images_after_narration() -> None:
    recipe = _narrated_still()

    assert [stage.stage for stage in recipe.stages] == [
        "input_topic",
        "write",
        "voice",
        "visuals",
        "render",
    ]
    assert recipe.stages[3].options == {"strategy": "image"}
    assert recipe.capabilities == ["llm", "tts", "image_gen"]
    assert recipe.output_type is OutputType.VIDEO
    assert recipe.handler is OutputHandler.SAVE


def test_narrated_still_visuals_assign_an_image_to_every_beat() -> None:
    recipe = _narrated_still()

    requests = resolve_visual_requests(
        _script(4),
        **recipe.stages[3].options,
    )

    assert [request.visual_type.value for request in requests] == ["image"] * 4
    assert all(request.strategy == "image" for request in requests)


def test_narrated_still_visuals_stage_generates_one_image_per_beat(
    tmp_path: Path,
) -> None:
    recipe = _narrated_still()
    images = _RecordingService(AssetKind.IMAGE, 0.04)
    stage = VisualReuseStage(
        scripts={"en": _script(3)},
        source_language="en",
        image=images,
        **recipe.stages[3].options,
    )

    result = cast(
        VisualReuseStageResult,
        _run(stage.run(StageContext(run_dir=tmp_path, config={}))),
    )

    assert len(images.requests) == 3
    assert [request.visual_type.value for request in result.requests] == ["image"] * 3
    assert result.cost_usd == pytest.approx(0.12)
    assert (tmp_path / "visual-reuse.json").is_file()


def test_narrated_still_voice_stage_writes_text_and_word_timings(
    tmp_path: Path,
) -> None:
    speech = _ToneSpeech(tmp_path / "audio")
    stage = VoiceStage(scripts={"en": _script(3)}, speech=speech, languages=["en"])

    result = cast(
        VoiceStageResult,
        _run(stage.run(StageContext(run_dir=tmp_path, config={}))),
    )

    timeline = Timeline.model_validate(
        json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    )

    assert result.timeline == "timeline.json"
    assert [beat.text for beat in timeline.beats] == [
        beat.text for beat in _script(3).beats
    ]
    assert all(voice.words for voice in timeline.languages[0].voices)
    assert timeline.languages[0].voices[0].words[0].word == "Beat"


def test_narrated_still_composes_images_text_and_synced_subtitles(
    tmp_path: Path,
) -> None:
    timeline, script = _narrated_timeline(tmp_path)
    video = VideoConfig.from_mapping({"resolution": "320x240", "fps": 12})

    composition = timeline_to_compositions(timeline, video=video)[0]

    image_layers = [
        layer for layer in composition.layers if isinstance(layer, ImageLayer)
    ]
    text_layers = [
        layer for layer in composition.layers if isinstance(layer, TextLayer)
    ]
    assert len(image_layers) == len(script.beats)
    assert [layer.text for layer in text_layers] == [beat.text for beat in script.beats]
    assert all(
        layer.start_seconds < (layer.end_seconds or 0.0) for layer in text_layers
    )
    assert composition.subtitles is not None
    assert composition.subtitles.words
    assert composition.subtitles.text.startswith("Beat 1")
    assert composition.audio_tracks
    assert all(track.role.value == "voice" for track in composition.audio_tracks)

    subtitle_stage = SubtitleStage(timeline=timeline)
    subtitle_result = cast(
        SubtitleStageResult,
        _run(subtitle_stage.run(StageContext(run_dir=tmp_path, config={}))),
    )
    assert subtitle_result.artifacts == ["subtitle.en.ass"]
    subtitle_text = (tmp_path / "subtitle.en.ass").read_text(encoding="utf-8")
    assert "Beat" in subtitle_text
    assert "\\k" in subtitle_text


@pytest.mark.slow
@pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None,
    reason="ffmpeg and ffprobe are required for a real narrated still render",
)
def test_the_narrated_still_renders_images_text_and_synced_subtitles(
    tmp_path: Path,
) -> None:
    timeline, _ = _narrated_timeline(tmp_path)
    video = VideoConfig.from_mapping({"resolution": "320x240", "fps": 12})
    composition = timeline_to_compositions(timeline, video=video)[0]
    output = tmp_path / "narrated-still.mp4"

    rendered = render_video(composition, output)

    assert isinstance(rendered, VideoRenderResult)
    assert output.is_file()
    assert output.stat().st_size > 0
    assert rendered.probe.primary_video is not None
    assert rendered.probe.audio_streams

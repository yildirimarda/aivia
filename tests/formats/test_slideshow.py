"""The built-in slideshow format uses one voice track and transitions between slides."""

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

from aivia.audio import VoiceStage, VoiceStageResult
from aivia.compose import (
    AudioTrackRole,
    ImageLayer,
    Timeline,
    TransitionType,
    VideoRenderResult,
    outgoing_transition_end,
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
from aivia.visuals import VisualReuseStage, VisualReuseStageResult

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
SAMPLE_RATE = 24_000


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _slideshow() -> FormatRecipeConfig:
    return load_packaged_formats().get("slideshow")


def _script() -> Script:
    return Script(
        title="Three ideas worth sharing",
        beats=[
            Beat(
                index=index,
                text=f"Slide {index + 1} shares one clear idea with a calm voice.",
                transition=transition,
                media_prompts=[
                    ImagePrompt(
                        prompt=f"Calm slide illustration {index + 1}.",
                        aspect_ratio="9:16",
                    )
                ],
            )
            for index, transition in enumerate(("crossfade", "slide_left", "cut"))
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
        asset = Asset(
            kind=self.kind,
            prompt=request.prompt,
            provider="fake-image",
            model="fake-image-model",
            path=f"file:///tmp/{len(self.requests)}.png",
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
        path = self.directory / f"slide-{len(self.requests)}.png"
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
    """Write one real wav per request and return a measured duration."""

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
        return SimpleNamespace(
            duration_seconds=duration,
            path=str(path),
            cost_usd=0.01,
            provider="fake-tts",
            model="fake-tts-model",
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


def _slideshow_timeline(tmp_path: Path) -> tuple[Timeline, Script]:
    recipe = _slideshow()
    script = _script()
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
        **recipe.stages[2].options,
    )
    _run(voice.run(StageContext(run_dir=tmp_path, config={})))

    manifest = json.loads((tmp_path / "visual-reuse.json").read_text(encoding="utf-8"))
    payload = json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    return _merge_visuals(payload, manifest), script


def test_the_packaged_slideshow_recipe_uses_one_voice_track_and_images() -> None:
    recipe = _slideshow()

    assert [stage.stage for stage in recipe.stages] == [
        "input_topic",
        "write",
        "voice",
        "visuals",
        "render",
    ]
    assert recipe.stages[2].options == {"mode": "script"}
    assert recipe.stages[3].options == {"strategy": "image"}
    assert recipe.capabilities == ["llm", "tts", "image_gen"]
    assert recipe.output_type is OutputType.VIDEO
    assert recipe.handler is OutputHandler.SAVE


def test_slideshow_visuals_stage_generates_one_image_per_slide(tmp_path: Path) -> None:
    recipe = _slideshow()
    images = _RecordingService(AssetKind.IMAGE, 0.04)
    stage = VisualReuseStage(
        scripts={"en": _script()},
        source_language="en",
        image=images,
        **recipe.stages[3].options,
    )

    result = cast(
        VisualReuseStageResult,
        _run(stage.run(StageContext(run_dir=tmp_path, config={}))),
    )

    assert len(images.requests) == 3
    assert result.cost_usd == pytest.approx(0.12)
    assert (tmp_path / "visual-reuse.json").is_file()


def test_slideshow_voice_stage_makes_one_request_and_keeps_transitions(
    tmp_path: Path,
) -> None:
    speech = _ToneSpeech(tmp_path / "audio")
    recipe = _slideshow()
    stage = VoiceStage(
        scripts={"en": _script()},
        speech=speech,
        languages=["en"],
        **recipe.stages[2].options,
    )

    result = cast(
        VoiceStageResult,
        _run(stage.run(StageContext(run_dir=tmp_path, config={}))),
    )

    timeline = Timeline.model_validate_json(
        (tmp_path / "timeline.json").read_text(encoding="utf-8")
    )
    assert result.timeline == "timeline.json"
    assert len(speech.requests) == 1
    assert len(timeline.languages[0].voices) == 1
    assert [beat.transition.type for beat in timeline.beats if beat.transition] == [
        TransitionType.CROSSFADE,
        TransitionType.SLIDE,
        TransitionType.CUT,
    ]
    assert timeline.beats[1].transition is not None
    assert timeline.beats[1].transition.direction.value == "left"


def test_slideshow_composition_has_transitions_and_a_single_voice_track(
    tmp_path: Path,
) -> None:
    timeline, script = _slideshow_timeline(tmp_path)
    video = VideoConfig.from_mapping({"resolution": "320x240", "fps": 12})

    composition = timeline_to_compositions(timeline, video=video)[0]

    image_layers = [
        layer for layer in composition.layers if isinstance(layer, ImageLayer)
    ]
    assert len(image_layers) == len(script.beats)
    assert [layer.transition.type for layer in image_layers if layer.transition] == [
        TransitionType.CROSSFADE,
        TransitionType.SLIDE,
        TransitionType.CUT,
    ]
    assert image_layers[1].transition is not None
    assert image_layers[1].transition.direction.value == "left"
    voice_tracks = [
        track
        for track in composition.audio_tracks
        if track.role is AudioTrackRole.VOICE
    ]
    assert len(voice_tracks) == 1
    extended = outgoing_transition_end(image_layers[0], composition)
    assert extended is not None
    assert extended == pytest.approx(image_layers[1].start_seconds + 0.5)


def test_a_timeline_default_transition_applies_to_every_slide() -> None:
    timeline = Timeline.model_validate(
        {
            "id": "default-transition",
            "metadata": {
                "transition": "crossfade",
                "transition_duration": 0.25,
            },
            "beats": [
                {"id": "one", "start": 0, "end": 2, "asset": "one"},
                {"id": "two", "start": 2, "end": 4, "asset": "two"},
            ],
            "assets": {
                "one": "slides/one.png",
                "two": "slides/two.png",
            },
            "voices": [{"source": "audio/voice.wav", "end": 4}],
        }
    )

    composition = timeline_to_compositions(timeline)[0]

    image_layers = [
        layer for layer in composition.layers if isinstance(layer, ImageLayer)
    ]
    assert [layer.transition.type for layer in image_layers if layer.transition] == [
        TransitionType.CROSSFADE,
        TransitionType.CROSSFADE,
    ]
    assert all(
        layer.transition.duration_seconds == pytest.approx(0.25)
        for layer in image_layers
        if layer.transition is not None
    )


@pytest.mark.slow
@pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None,
    reason="ffmpeg and ffprobe are required for a real slideshow render",
)
def test_the_slideshow_renders_transitions_and_one_voice_track(
    tmp_path: Path,
) -> None:
    timeline, _ = _slideshow_timeline(tmp_path)
    video = VideoConfig.from_mapping({"resolution": "320x240", "fps": 12})
    composition = timeline_to_compositions(timeline, video=video)[0]
    output = tmp_path / "slideshow.mp4"

    rendered = render_video(composition, output)

    assert isinstance(rendered, VideoRenderResult)
    assert output.is_file()
    assert output.stat().st_size > 0
    assert rendered.probe.primary_video is not None
    assert rendered.probe.audio_streams

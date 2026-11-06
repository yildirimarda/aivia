"""The built-in song video format aligns lyrics, music, karaoke and backgrounds."""

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

from aivia.audio import (
    LyricAlignStage,
    MusicStage,
    SubtitleStage,
    SubtitleStageResult,
)
from aivia.compose import (
    AudioTrackRole,
    ImageLayer,
    Timeline,
    VideoRenderResult,
    composition_from_yaml,
    render_video,
)
from aivia.core.config import VideoConfig
from aivia.core.config.formats import FormatRecipeConfig, OutputHandler, OutputType
from aivia.core.models.asset import Asset, AssetKind
from aivia.core.stages import StageContext
from aivia.formats import (
    SongVideoStage,
    load_packaged_formats,
    song_video_timeline,
)
from aivia.providers.mock import MockImageProvider
from aivia.providers.protocols.audio import AudioFormat, WordTimestamp
from aivia.providers.protocols.music import MusicTrack
from aivia.providers.protocols.stt import STTRequest, STTResponse
from aivia.visuals import (
    VisualReuseStage,
)
from aivia.writing import build_lyrics_script

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
SAMPLE_RATE = 24_000


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _song_video() -> FormatRecipeConfig:
    return load_packaged_formats().get("song_video")


def _script() -> Any:
    return build_lyrics_script(
        {
            "title": "Neon sky",
            "sections": [
                {
                    "name": "verse",
                    "lines": [
                        "City lights are calling me",
                        "We are running wild and free",
                    ],
                },
                {
                    "name": "chorus",
                    "lines": [
                        "Sing it back to me",
                        "Neon sky forever",
                    ],
                },
            ],
        }
    )


def _write_script(directory: Path, script: Any, language: str = "en") -> Path:
    path = directory / f"script.{language}.json"
    path.write_text(script.model_dump_json(indent=2), encoding="utf-8")
    return path


def _transcript(script: Any) -> STTResponse:
    words: list[WordTimestamp] = []
    cursor = 0.0
    for beat in script.beats:
        for word in beat.text.split():
            start = cursor
            end = start + 0.28
            words.append(WordTimestamp(word=word, start_seconds=start, end_seconds=end))
            cursor = end + 0.04
        cursor += 0.12
    return STTResponse(
        text=" ".join(word.word for word in words),
        language="en",
        duration_seconds=cursor,
        words=words,
        provider="fake-stt",
        model="fake-stt-model",
    )


def _wav_bytes(duration: float) -> bytes:
    frames = int(duration * SAMPLE_RATE)
    chunks = bytearray()
    for index in range(frames):
        value = int(4000 * math.sin(2 * math.pi * 440 * index / SAMPLE_RATE))
        chunks.extend(struct.pack("<h", value))
    from io import BytesIO

    buffer = BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(SAMPLE_RATE)
        target.writeframes(bytes(chunks))
    return buffer.getvalue()


class _FakeSongGenerator:
    """Return one song track and record the request that asked for it."""

    def __init__(self, data: bytes, *, duration: float, format: AudioFormat) -> None:
        self.data = data
        self.duration = duration
        self.format = format
        self.requests: list[Any] = []

    async def generate(self, request: Any) -> MusicTrack:
        self.requests.append(request)
        return MusicTrack(
            data=self.data,
            format=self.format,
            duration_seconds=self.duration,
            lyrics=request.lyrics,
            title="Neon sky",
        )


class _FakeSTT:
    """A deterministic transcriber that records the audio it was given."""

    def __init__(self, response: STTResponse) -> None:
        self.response = response
        self.requests: list[STTRequest] = []

    async def transcribe(self, request: STTRequest) -> STTResponse:
        self.requests.append(request)
        return self.response


class _RecordingImageService:
    """Return a stored image asset per request and count every call."""

    def __init__(self, directory: Path, *, real: bool = False) -> None:
        self.directory = directory
        self.real = real
        self.requests: list[Any] = []
        self.provider = MockImageProvider(name="mock-image", seed=7)

    async def generate(self, request: Any, **_: Any) -> SimpleNamespace:
        self.requests.append(request)
        if self.real:
            response = await self.provider.generate(request)
            result = response.images[0]
            path = self.directory / f"background-{len(self.requests)}.png"
            assert result.data is not None
            path.write_bytes(result.data)
            asset = Asset(
                kind=AssetKind.IMAGE,
                prompt=request.prompt,
                provider=response.provider,
                model=response.model,
                path=str(path),
            )
        else:
            asset = Asset(
                kind=AssetKind.IMAGE,
                prompt=request.prompt,
                provider="fake-image",
                model="fake-image-model",
                path=f"/tmp/background-{len(self.requests)}.png",
            )
        return SimpleNamespace(
            asset=asset,
            assets=(asset,),
            provider=asset.provider,
            model=asset.model,
            cost_usd=0.02,
            cached=False,
        )


def _music_stage(
    tmp_path: Path,
    script: Any,
    recipe: FormatRecipeConfig,
    *,
    data: bytes = b"song-bytes",
    duration: float = 4.0,
    format: AudioFormat = AudioFormat.MP3,
) -> _FakeSongGenerator:
    _write_script(tmp_path, script)
    generator = _FakeSongGenerator(data, duration=duration, format=format)
    stage = MusicStage(
        options={
            **recipe.stages[2].options,
            "source": "provider",
            "duration_seconds": duration,
        },
        music=generator,
    )
    _run(stage.run(StageContext(run_dir=tmp_path, config={})))
    return generator


def _aligned_timeline(
    tmp_path: Path, script: Any, recipe: FormatRecipeConfig
) -> Timeline:
    _music_stage(tmp_path, script, recipe)
    stage = LyricAlignStage(transcriber=_FakeSTT(_transcript(script)))
    _run(stage.run(StageContext(run_dir=tmp_path, config={})))
    return Timeline.model_validate_json(
        (tmp_path / "timeline.json").read_text(encoding="utf-8")
    )


def _visual_manifest(
    tmp_path: Path,
    script: Any,
    recipe: FormatRecipeConfig,
    *,
    real: bool = False,
) -> dict[str, Any]:
    images = _RecordingImageService(tmp_path, real=real)
    stage = VisualReuseStage(
        scripts={"en": script},
        source_language="en",
        image=images,
        **recipe.stages[4].options,
    )
    _run(stage.run(StageContext(run_dir=tmp_path, config={})))
    return cast(
        dict[str, Any],
        json.loads((tmp_path / "visual-reuse.json").read_text(encoding="utf-8")),
    )


def test_the_packaged_song_video_recipe_uses_lyrics_music_alignment_and_visuals() -> (
    None
):
    recipe = _song_video()

    assert [stage.stage for stage in recipe.stages] == [
        "input_topic",
        "write",
        "music",
        "align",
        "visuals",
        "render",
    ]
    assert recipe.stages[1].options["mode"] == "lyrics"
    assert recipe.stages[2].options["lyrics_from_script"] is True
    assert recipe.stages[4].options == {"strategy": "image"}
    assert recipe.capabilities == ["llm", "music_gen", "stt", "image_gen"]
    assert recipe.output_type is OutputType.VIDEO
    assert recipe.handler is OutputHandler.PUBLISH


def test_the_music_stage_reads_lyrics_from_the_written_script(
    tmp_path: Path,
) -> None:
    script = _script()
    recipe = _song_video()

    generator = _music_stage(tmp_path, script, recipe)

    request = generator.requests[0]
    assert request.lyrics == (
        "City lights are calling me\n"
        "We are running wild and free\n"
        "Sing it back to me\n"
        "Neon sky forever"
    )
    assert request.instrumental is False
    timeline = Timeline.model_validate_json(
        (tmp_path / "timeline.json").read_text(encoding="utf-8")
    )
    assert len(timeline.music) == 1
    assert timeline.music[0].source.endswith("music/music.mp3")


def test_the_packaged_song_recipe_seeds_the_timeline_from_the_written_lyrics(
    tmp_path: Path,
) -> None:
    script = _script()
    recipe = _song_video()
    _write_script(tmp_path, script)
    assert "duration_seconds" not in recipe.stages[2].options
    generator = _FakeSongGenerator(
        b"song-bytes",
        duration=4.0,
        format=AudioFormat.MP3,
    )

    music = MusicStage(options=recipe.stages[2].options, music=generator)
    _run(music.run(StageContext(run_dir=tmp_path, config={})))

    timeline = Timeline.model_validate_json(
        (tmp_path / "timeline.json").read_text(encoding="utf-8")
    )
    assert [beat.text for beat in timeline.beats] == [
        beat.text for beat in script.beats
    ]
    assert all(
        beat.start_seconds < (beat.end_seconds or 0.0) for beat in timeline.beats
    )
    assert timeline.duration_seconds is not None
    assert timeline.duration_seconds > 0.0
    assert generator.requests[0].duration_seconds == pytest.approx(
        timeline.duration_seconds
    )

    align = LyricAlignStage(transcriber=_FakeSTT(_transcript(script)))
    _run(align.run(StageContext(run_dir=tmp_path, config={})))

    aligned = Timeline.model_validate_json(
        (tmp_path / "timeline.json").read_text(encoding="utf-8")
    )
    assert [beat.text for beat in aligned.beats] == [beat.text for beat in script.beats]
    assert aligned.duration_seconds == pytest.approx(
        _transcript(script).duration_seconds
    )


def test_alignment_writes_timed_beats_words_and_karaoke_cues(
    tmp_path: Path,
) -> None:
    script = _script()
    recipe = _song_video()
    _music_stage(tmp_path, script, recipe)
    transcriber = _FakeSTT(_transcript(script))
    stage = LyricAlignStage(transcriber=transcriber)

    result = _run(stage.run(StageContext(run_dir=tmp_path, config={})))

    assert result.artifacts == ["timeline.json", "lyrics-alignment.json"]
    assert transcriber.requests[0].audio == b"song-bytes"
    assert transcriber.requests[0].language == "en"

    timeline = Timeline.model_validate_json(
        (tmp_path / "timeline.json").read_text(encoding="utf-8")
    )
    assert [beat.text for beat in timeline.beats] == [
        beat.text for beat in script.beats
    ]
    assert all(
        beat.start_seconds < (beat.end_seconds or 0.0) for beat in timeline.beats
    )
    assert len(timeline.music) == 1
    assert timeline.music[0].source == str(tmp_path / "music" / "music.mp3")

    language = timeline.languages[0]
    assert language.language == "en"
    assert language.subtitles is not None
    assert len(language.subtitles.cues) == len(script.beats)
    assert language.subtitles.words
    assert language.subtitles.cues[0].text == script.beats[0].text
    assert language.subtitles.metadata["karaoke"] is True

    manifest = json.loads(
        (tmp_path / "lyrics-alignment.json").read_text(encoding="utf-8")
    )
    assert manifest["coverage"] == pytest.approx(1.0)
    assert manifest["lines"][0]["start_seconds"] == pytest.approx(0.0)
    assert manifest["lines"][0]["words"][0]["word"] == "City"
    assert manifest["mismatches"] == []
    assert manifest["timeline"] == "timeline.json"

    subtitle_result = cast(
        SubtitleStageResult,
        _run(
            SubtitleStage(timeline=timeline, highlight="karaoke").run(
                StageContext(run_dir=tmp_path, config={})
            )
        ),
    )
    assert subtitle_result.artifacts == ["subtitle.en.ass"]
    subtitle_text = (tmp_path / "subtitle.en.ass").read_text(encoding="utf-8")
    assert "City" in subtitle_text
    assert "\\k" in subtitle_text


def test_visual_backgrounds_attach_to_timed_lyric_beats(
    tmp_path: Path,
) -> None:
    script = _script()
    recipe = _song_video()
    timeline = _aligned_timeline(tmp_path, script, recipe)
    manifest = _visual_manifest(tmp_path, script, recipe)

    attached = song_video_timeline(timeline, manifest)

    assert [beat.asset for beat in attached.beats] == [
        f"visual_{index}" for index in range(len(script.beats))
    ]
    assert [asset.source for asset in attached.assets] == [
        f"/tmp/background-{index}.png" for index in range(1, len(script.beats) + 1)
    ]
    assert all(asset.kind == "image" for asset in attached.assets)


def test_the_song_video_stage_composes_backgrounds_and_karaoke_subtitles(
    tmp_path: Path,
) -> None:
    script = _script()
    recipe = _song_video()
    timeline = _aligned_timeline(tmp_path, script, recipe)
    manifest = _visual_manifest(tmp_path, script, recipe)
    video = VideoConfig.from_mapping({"resolution": "320x240", "fps": 12})
    stage = SongVideoStage(
        timeline=timeline,
        visuals=manifest,
        video=video,
        render=False,
    )

    result = _run(stage.run(StageContext(run_dir=tmp_path, config={})))

    assert result.artifacts == ["song-video.yaml"]
    composition = composition_from_yaml(
        (tmp_path / "song-video.yaml").read_text(encoding="utf-8")
    )
    image_layers = [
        layer for layer in composition.layers if isinstance(layer, ImageLayer)
    ]
    assert len(image_layers) == len(script.beats)
    assert composition.subtitles is not None
    assert composition.subtitles.words
    assert composition.subtitles.cues[0].text == script.beats[0].text
    music_tracks = [
        track
        for track in composition.audio_tracks
        if track.role is AudioTrackRole.MUSIC
    ]
    assert len(music_tracks) == 1


@pytest.mark.slow
@pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None,
    reason="ffmpeg and ffprobe are required for a real song video render",
)
def test_the_song_video_renders_backgrounds_karaoke_and_the_song(
    tmp_path: Path,
) -> None:
    script = _script()
    recipe = _song_video()
    duration = 4.0
    _music_stage(
        tmp_path,
        script,
        recipe,
        data=_wav_bytes(duration),
        duration=duration,
        format=AudioFormat.WAV,
    )
    align = LyricAlignStage(transcript=_transcript(script))
    _run(align.run(StageContext(run_dir=tmp_path, config={})))
    manifest = _visual_manifest(tmp_path, script, recipe, real=True)
    video = VideoConfig.from_mapping({"resolution": "320x240", "fps": 12})
    stage = SongVideoStage(visuals=manifest, video=video)

    result = _run(stage.run(StageContext(run_dir=tmp_path, config={})))

    assert result.artifacts == ["song-video.yaml", "song-video.mp4"]
    output = tmp_path / "song-video.mp4"
    assert output.is_file()
    assert output.stat().st_size > 0
    rendered = render_video(
        composition_from_yaml(
            (tmp_path / "song-video.yaml").read_text(encoding="utf-8")
        ),
        tmp_path / "song-video-again.mp4",
    )
    assert isinstance(rendered, VideoRenderResult)
    assert rendered.probe.primary_video is not None
    assert rendered.probe.audio_streams

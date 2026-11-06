"""The built-in audio-only format embeds cover art and metadata, never video."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from collections.abc import Callable, Coroutine, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from aivia.compose import AudioMetadata, build_audio_command, render_audio
from aivia.core.config.formats import OutputHandler, OutputType
from aivia.core.models.asset import Asset, AssetKind
from aivia.core.models.script import Script
from aivia.core.stages import StageContext
from aivia.formats import (
    AudioOnlyStage,
    CoverArtStage,
    audio_only_metadata,
    build_audio_only_stage,
    build_cover_art_stage,
    cover_art_prompt,
    load_packaged_formats,
)
from aivia.writing import build_lyrics_script

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _audio_only() -> Any:
    return load_packaged_formats().get("audio_only")


def _script() -> Script:
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
                }
            ],
        }
    )


def _write_script(directory: Path, script: Script, language: str = "en") -> Path:
    path = directory / f"script.{language}.json"
    path.write_text(script.model_dump_json(indent=2), encoding="utf-8")
    return path


def _write_cover(path: Path, color: str = "#ff0055") -> Path:
    Image.new("RGB", (64, 64), color).save(path)
    return path


def _timeline(directory: Path, *, duration: float = 4.0) -> dict[str, Any]:
    source = directory / "song.wav"
    source.write_bytes(b"RIFF")
    return {
        "id": "audio-only",
        "duration_seconds": duration,
        "music": [
            {
                "id": "song",
                "source": str(source),
                "duration_seconds": duration,
                "metadata": {"title": "Neon sky", "genre": "synthwave"},
            }
        ],
    }


def _write_timeline(directory: Path, *, duration: float = 4.0) -> Path:
    path = directory / "timeline.json"
    path.write_text(
        json.dumps(_timeline(directory, duration=duration)), encoding="utf-8"
    )
    return path


def _write_music(directory: Path) -> Path:
    path = directory / "music.json"
    path.write_text(
        json.dumps(
            {
                "selected": {
                    "title": "Neon sky",
                    "genre": "synthwave",
                    "provider": "fake-music",
                    "model": "fake-music-model",
                },
                "prompt": "Neon synthwave song",
            }
        ),
        encoding="utf-8",
    )
    return path


def _probe_json(
    *,
    duration: float = 4.0,
    tags: dict[str, str] | None = None,
    cover: bool = True,
) -> str:
    streams: list[dict[str, Any]] = [
        {
            "index": 0,
            "codec_type": "audio",
            "codec_name": "mp3",
            "duration": f"{duration:.3f}",
            "sample_rate": "48000",
            "channels": 2,
        }
    ]
    if cover:
        streams.append(
            {
                "index": 1,
                "codec_type": "video",
                "codec_name": "png",
                "width": 64,
                "height": 64,
                "disposition": {"attached_pic": 1},
            }
        )
    return json.dumps(
        {
            "streams": streams,
            "format": {
                "format_name": "mp3",
                "format_long_name": "MP2/3 (MPEG audio layer 2/3)",
                "duration": f"{duration:.3f}",
                "size": "12345",
                "tags": tags or {},
            },
        }
    )


def _ffmpeg_writer(
    calls: list[list[str]] | None = None,
) -> Callable[[Sequence[str]], subprocess.CompletedProcess[str]]:
    def run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if calls is not None:
            calls.append(list(argv))
        Path(argv[-1]).write_bytes(b"ID3\x04\x00\x00\x00\x00\x00\x00audio")
        return subprocess.CompletedProcess(list(argv), 0, stdout="", stderr="")

    return run


def _probe_runner(
    payload: str,
    calls: list[list[str]] | None = None,
) -> Callable[[Sequence[str]], subprocess.CompletedProcess[str]]:
    def run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if calls is not None:
            calls.append(list(argv))
        return subprocess.CompletedProcess(list(argv), 0, stdout=payload, stderr="")

    return run


class _FakeImageService:
    """Write a real image and return it as a stored asset."""

    def __init__(self, directory: Path, *, cost_usd: float = 0.03) -> None:
        self.directory = directory
        self.cost_usd = cost_usd
        self.requests: list[Any] = []

    async def generate(self, request: Any, **_: Any) -> SimpleNamespace:
        self.requests.append(request)
        path = self.directory / f"generated-cover-{len(self.requests)}.png"
        _write_cover(path, "#224466")
        asset = Asset(
            kind=AssetKind.IMAGE,
            prompt=request.prompt,
            provider="fake-image",
            model="fake-image-model",
            path=str(path),
        )
        return SimpleNamespace(
            asset=asset,
            assets=(asset,),
            provider=asset.provider,
            model=asset.model,
            cost_usd=self.cost_usd,
            cached=False,
        )


def test_the_packaged_audio_only_recipe_uses_cover_art_without_video() -> None:
    recipe = _audio_only()

    assert [stage.stage for stage in recipe.stages] == [
        "input_topic",
        "write",
        "music",
        "cover",
        "render",
    ]
    assert recipe.stages[1].options["mode"] == "lyrics"
    assert recipe.stages[2].options["lyrics_from_script"] is True
    assert recipe.stages[3].options == {"aspect_ratio": "1:1"}
    assert recipe.stages[4].options["output_format"] == "mp3"
    assert recipe.stages[4].options["embed_cover_art"] is True
    assert recipe.capabilities == ["llm", "music_gen", "image_gen"]
    assert "video_gen" not in recipe.capabilities
    assert recipe.output_type is OutputType.AUDIO
    assert recipe.handler is OutputHandler.SAVE


def test_the_cover_prompt_is_built_from_the_title_and_lyrics() -> None:
    script = _script()

    prompt = cover_art_prompt(script)

    assert "Neon sky" in prompt
    assert "City lights are calling me" in prompt
    assert "no typography" in prompt
    assert cover_art_prompt(script, prompt="An explicit cover") == "An explicit cover"


def test_metadata_merges_explicit_tags_over_the_script_and_song() -> None:
    metadata = audio_only_metadata(
        _script(),
        {"selected": {"title": "Neon sky", "genre": "synthwave"}},
        artist="AIVIA",
        album="Generated songs",
    )

    assert isinstance(metadata, AudioMetadata)
    assert metadata.title == "Neon sky"
    assert metadata.artist == "AIVIA"
    assert metadata.album == "Generated songs"
    assert metadata.genre == "synthwave"

    explicit = audio_only_metadata(
        _script(),
        {"selected": {"title": "Neon sky"}},
        metadata={"title": "Explicit title", "custom": "kept"},
    )
    assert explicit.title == "Explicit title"
    assert explicit.as_tags()["custom"] == "kept"


def test_the_cover_stage_generates_one_square_cover_and_manifest(
    tmp_path: Path,
) -> None:
    script = _script()
    _write_script(tmp_path, script)
    images = _FakeImageService(tmp_path)
    stage = CoverArtStage(scripts={"en": script}, image=images)

    result = _run(stage.run(StageContext(run_dir=tmp_path, config={})))

    assert result.artifacts == ["cover.png", "cover.json"]
    assert result.cost_usd == pytest.approx(0.03)
    assert len(images.requests) == 1
    assert "Neon sky" in images.requests[0].prompt
    cover = tmp_path / "cover.png"
    assert cover.is_file()
    manifest = json.loads((tmp_path / "cover.json").read_text(encoding="utf-8"))
    assert manifest["cover"] == "cover.png"
    assert manifest["provider"] == "fake-image"
    assert manifest["model"] == "fake-image-model"
    assert manifest["cost_usd"] == pytest.approx(0.03)
    assert manifest["title"] == "Neon sky"
    assert manifest["source"].endswith("generated-cover-1.png")


def test_the_audio_stage_embeds_cover_art_and_metadata_without_video_cost(
    tmp_path: Path,
) -> None:
    script = _script()
    _write_script(tmp_path, script)
    _write_timeline(tmp_path)
    _write_music(tmp_path)
    cover = _write_cover(tmp_path / "cover.png")
    calls: list[list[str]] = []
    stage = AudioOnlyStage(
        scripts={"en": script},
        cover_art=cover,
        ffmpeg_runner=_ffmpeg_writer(calls),
        ffprobe_runner=_probe_runner(
            _probe_json(
                tags={
                    "title": "Neon sky",
                    "artist": "AIVIA",
                    "album": "Neon sky",
                    "genre": "synthwave",
                    "comment": "Neon synthwave song",
                }
            )
        ),
        title="Neon sky",
        artist="AIVIA",
    )

    result = _run(stage.run(StageContext(run_dir=tmp_path, config={})))

    assert result.cost_usd == 0.0
    assert result.artifacts == [
        "audio-only.yaml",
        "audio-only.mp3",
        "audio-only.json",
    ]
    assert result.metadata["video_cost_usd"] == 0.0
    assert result.metadata["has_video"] is False
    output = tmp_path / "audio-only.mp3"
    assert output.is_file()
    assert cover.is_file()

    manifest = json.loads((tmp_path / "audio-only.json").read_text(encoding="utf-8"))
    assert manifest["output_type"] == "audio"
    assert manifest["video_cost_usd"] == 0.0
    assert manifest["cover_art"] == str(cover)
    item = manifest["outputs"][0]
    assert item["metadata"]["title"] == "Neon sky"
    assert item["metadata"]["artist"] == "AIVIA"
    assert item["metadata"]["genre"] == "synthwave"
    assert item["has_cover_art"] is True
    assert item["has_video"] is False
    assert item["video_cost_usd"] == 0.0
    assert item["cover_art"] == str(cover)

    argv = calls[0]
    assert "-disposition:v" in argv
    assert argv[argv.index("-disposition:v") + 1] == "attached_pic"
    assert "-vn" not in argv
    assert str(cover) in argv


def test_building_an_audio_command_attaches_a_cover_picture() -> None:
    composition = {
        "id": "episode-1",
        "canvas": {"width": 320, "height": 240},
        "duration_seconds": 2.0,
        "fps": 24,
        "layers": [],
        "audio_tracks": [{"id": "song", "source": "song.wav", "role": "music"}],
    }

    command = build_audio_command(
        composition,
        "out.mp3",
        cover_art="cover.png",
        metadata={"title": "Neon sky", "artist": "AIVIA"},
    )

    assert "-vn" not in command
    assert "attached_pic" in command
    assert "cover.png" in command
    assert command[command.index("-c:v") + 1] == "copy"
    assert command[command.index("-map") + 1] == "[aout]"
    maps = [command[index + 1] for index, item in enumerate(command) if item == "-map"]
    assert maps == ["[aout]", "1:v:0"]
    tags = [
        command[index + 1] for index, item in enumerate(command) if item == "-metadata"
    ]
    assert "title=Neon sky" in tags
    assert "artist=AIVIA" in tags
    assert [item.source for item in command.inputs] == ["song.wav", "cover.png"]


def test_render_audio_accepts_an_attached_cover_and_returns_it(
    tmp_path: Path,
) -> None:
    output = tmp_path / "episode.mp3"

    result = render_audio(
        {
            "id": "episode-1",
            "canvas": {"width": 320, "height": 240},
            "duration_seconds": 2.0,
            "fps": 24,
            "layers": [],
            "audio_tracks": [{"id": "song", "source": "song.wav", "role": "music"}],
        },
        output,
        cover_art="cover.png",
        metadata={"title": "Neon sky"},
        ffmpeg_runner=_ffmpeg_writer(),
        ffprobe_runner=_probe_runner(
            _probe_json(duration=2.0, tags={"title": "Neon sky"})
        ),
    )

    assert result.cover_art == Path("cover.png")
    assert result.has_cover_art is True
    assert output.is_file()


def test_builders_build_the_audio_and_cover_stages() -> None:
    audio = build_audio_only_stage(options={"title": "Neon sky"})
    cover = build_cover_art_stage(options={"aspect_ratio": "1:1"})

    assert isinstance(audio, AudioOnlyStage)
    assert audio.title == "Neon sky"
    assert isinstance(cover, CoverArtStage)
    assert cover.aspect_ratio == "1:1"


@pytest.mark.slow
@pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None,
    reason="ffmpeg and ffprobe are required for a real audio-only render",
)
def test_a_real_audio_only_render_embeds_cover_art_and_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "song.wav"
    subprocess.run(
        [
            str(FFMPEG),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=2",
            "-c:a",
            "pcm_s16le",
            str(source),
        ],
        check=True,
    )
    cover = _write_cover(tmp_path / "cover.png")
    timeline = {
        "id": "audio-only",
        "duration_seconds": 2.0,
        "music": [{"id": "song", "source": str(source), "duration_seconds": 2.0}],
    }
    stage = AudioOnlyStage(
        timeline=timeline,
        cover_art=cover,
        scripts={"en": _script()},
        metadata={"title": "Neon sky", "artist": "AIVIA"},
    )

    result = _run(stage.run(StageContext(run_dir=tmp_path, config={})))

    assert result.cost_usd == 0.0
    assert result.metadata["has_video"] is False
    output = tmp_path / "audio-only.mp3"
    assert output.is_file()
    assert output.read_bytes().startswith(b"ID3")
    from aivia.core.media_probe import run_ffprobe

    probe = run_ffprobe(output)
    assert probe.audio_streams
    assert probe.video_streams
    assert probe.tags["title"] == "Neon sky"
    assert probe.tags["artist"] == "AIVIA"

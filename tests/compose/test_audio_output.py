"""Audio-only outputs use the right codec, duration and ID3 metadata."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from aivia.compose import (
    AUDIO_RENDERER_VERSION,
    AudioMetadata,
    AudioOutputError,
    AudioRenderer,
    AudioRenderResult,
    Composition,
    audio_output_format,
    build_audio_command,
    render_audio,
    render_video,
)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

requires_ffmpeg = pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None,
    reason="ffmpeg and ffprobe are required for a real audio render",
)


def _composition(**overrides: Any) -> Composition:
    values: dict[str, Any] = {
        "id": "episode-1",
        "canvas": {"width": 320, "height": 240, "background": "#000000"},
        "duration_seconds": 2.0,
        "fps": 24,
        "layers": [],
        "audio_tracks": [
            {"id": "voice", "source": "voice.wav", "role": "voice"},
        ],
    }
    values.update(overrides)
    return Composition.model_validate(values)


def _option(argv: Sequence[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def _options(argv: Sequence[str], name: str) -> list[str]:
    return [argv[index + 1] for index, item in enumerate(argv) if item == name]


def _probe_json(
    *,
    codec: str = "mp3",
    duration: float = 2.0,
    tags: dict[str, str] | None = None,
    video: bool = False,
) -> str:
    streams: list[dict[str, Any]] = [
        {
            "index": 0,
            "codec_type": "audio",
            "codec_name": codec,
            "duration": f"{duration:.3f}",
            "sample_rate": "48000",
            "channels": 2,
        }
    ]
    if video:
        streams.append(
            {
                "index": 1,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 320,
                "height": 240,
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


@pytest.mark.parametrize(
    ("suffix", "codec"),
    [
        (".mp3", "libmp3lame"),
        (".m4a", "aac"),
        (".wav", "pcm_s16le"),
        (".flac", "flac"),
        (".ogg", "libvorbis"),
        (".oga", "libvorbis"),
        (".opus", "libopus"),
        (".aac", "aac"),
    ],
)
def test_the_extension_selects_a_container_compatible_codec(
    suffix: str, codec: str
) -> None:
    command = build_audio_command(_composition(), f"out{suffix}")

    assert _option(command, "-c:a") == codec
    assert "-vn" in command
    assert "[vout]" not in command


def test_aliases_and_explicit_formats_resolve_to_the_canonical_container() -> None:
    assert audio_output_format("out.oga") == "ogg"
    assert audio_output_format("out.wave") == "wav"
    assert audio_output_format("out.bin", "MP3") == "mp3"
    assert audio_output_format("out.mp4") == "m4a"

    with pytest.raises(AudioOutputError):
        audio_output_format("out.bin")


def test_the_command_has_no_video_and_enforces_the_composition_duration() -> None:
    command = build_audio_command(
        _composition(duration_seconds=3.5),
        "out.mp3",
    )

    assert command.filtergraph.startswith("[0:a]")
    assert "apad=whole_dur=3.500000" in command.filtergraph
    assert "atrim=duration=3.500000" in command.filtergraph
    assert _option(command, "-map") == "[aout]"
    assert _option(command, "-t") == "3.500000"
    assert [item.source for item in command.inputs] == ["voice.wav"]
    assert _option(command, "-c:a") == "libmp3lame"


def test_id3_metadata_is_written_with_the_expected_tag_aliases() -> None:
    command = build_audio_command(
        _composition(),
        "out.mp3",
        metadata={
            "title": "Episode 1",
            "artist": "AIVIA",
            "album": "Tests",
            "albumartist": "AIVIA",
            "year": "2026",
            "track_number": "2",
            "comment": "Rendered offline",
            "custom_note": "kept",
        },
    )

    assert _option(command, "-id3v2_version") == "3"
    assert _option(command, "-write_id3v1") == "1"
    assert _option(command, "-map_metadata") == "-1"
    tags = dict(value.split("=", 1) for value in _options(command, "-metadata"))
    assert tags["title"] == "Episode 1"
    assert tags["artist"] == "AIVIA"
    assert tags["album"] == "Tests"
    assert tags["album_artist"] == "AIVIA"
    assert tags["date"] == "2026"
    assert tags["track"] == "2"
    assert tags["comment"] == "Rendered offline"
    assert tags["custom_note"] == "kept"


def test_metadata_can_be_built_from_a_mapping_or_an_instance() -> None:
    metadata = AudioMetadata.from_value({"title": "  Episode 1  ", "artist": "AIVIA"})

    assert metadata.title == "Episode 1"
    assert metadata.as_tags() == {"title": "Episode 1", "artist": "AIVIA"}
    assert AudioMetadata.from_value(metadata) is metadata
    assert bool(AudioMetadata()) is False


def test_incompatible_codecs_and_bad_inputs_are_rejected() -> None:
    with pytest.raises(AudioOutputError):
        build_audio_command(_composition(), "out.mp3", audio_codec="aac")
    with pytest.raises(AudioOutputError):
        build_audio_command(_composition(), "out.mp3", audio_bitrate=0)
    with pytest.raises(AudioOutputError):
        build_audio_command(_composition(), "out.mp3", loudness_lufs=3.0)
    with pytest.raises(AudioOutputError):
        build_audio_command(
            _composition(),
            "out.mp3",
            metadata={"bad=key": "value"},
        )
    with pytest.raises(AudioOutputError):
        build_audio_command(_composition(audio_tracks=[]), "out.mp3")
    with pytest.raises(AudioOutputError):
        build_audio_command(
            _composition(
                audio_tracks=[{"id": "off", "source": "off.wav", "enabled": False}]
            ),
            "out.mp3",
        )


def test_render_audio_verifies_the_probed_codec_duration_and_tags(
    tmp_path: Path,
) -> None:
    output = tmp_path / "episode.mp3"
    calls: list[list[str]] = []

    result = render_audio(
        _composition(duration_seconds=2.0),
        output,
        metadata={"title": "Episode 1", "artist": "AIVIA", "album": "Tests"},
        ffmpeg_runner=_ffmpeg_writer(calls),
        ffprobe_runner=_probe_runner(
            _probe_json(
                duration=2.05,
                tags={"title": "Episode 1", "artist": "AIVIA", "album": "Tests"},
            )
        ),
    )

    assert isinstance(result, AudioRenderResult)
    assert result.output == output
    assert result.format == "mp3"
    assert result.codec == "libmp3lame"
    assert result.duration_seconds == pytest.approx(2.05)
    assert result.tags["title"] == "Episode 1"
    assert result.tags["artist"] == "AIVIA"
    assert output.read_bytes().startswith(b"ID3")
    assert calls and calls[0][-1].endswith(".mp3")


def test_render_audio_leaves_no_output_when_verification_fails(
    tmp_path: Path,
) -> None:
    output = tmp_path / "episode.mp3"

    with pytest.raises(AudioOutputError):
        render_audio(
            _composition(),
            output,
            metadata={"title": "Episode 1", "artist": "AIVIA"},
            ffmpeg_runner=_ffmpeg_writer(),
            ffprobe_runner=_probe_runner(_probe_json(tags={"title": "Episode 1"})),
        )

    assert not output.exists()


@pytest.mark.parametrize(
    "payload",
    [
        _probe_json(codec="aac"),
        _probe_json(duration=9.0),
        _probe_json(video=True),
        _probe_json(tags={"title": "Episode 1"}),
    ],
)
def test_render_audio_rejects_a_mismatched_probe(tmp_path: Path, payload: str) -> None:
    with pytest.raises(AudioOutputError):
        render_audio(
            _composition(),
            tmp_path / "episode.mp3",
            metadata={"title": "Episode 1", "artist": "AIVIA"},
            ffmpeg_runner=_ffmpeg_writer(),
            ffprobe_runner=_probe_runner(payload),
        )


def test_render_video_dispatches_audio_suffixes(tmp_path: Path) -> None:
    output = tmp_path / "episode.mp3"

    result = render_video(
        _composition(),
        output,
        metadata={"title": "Episode 1", "artist": "AIVIA"},
        ffmpeg_runner=_ffmpeg_writer(),
        ffprobe_runner=_probe_runner(
            _probe_json(tags={"title": "Episode 1", "artist": "AIVIA"})
        ),
    )

    assert isinstance(result, AudioRenderResult)
    assert result.output == output
    assert result.command.filtergraph == result.command.filter_complex
    assert AudioRenderer().version == AUDIO_RENDERER_VERSION


@requires_ffmpeg
@pytest.mark.slow
def test_real_mp3_render_has_codec_duration_and_id3_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "voice.wav"
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
            "sine=frequency=440:sample_rate=48000:duration=1.5",
            "-c:a",
            "pcm_s16le",
            str(source),
        ],
        check=True,
    )
    output = tmp_path / "episode.mp3"

    result = render_audio(
        _composition(
            duration_seconds=2.0,
            audio_tracks=[{"id": "voice", "source": str(source), "role": "voice"}],
        ),
        output,
        metadata={
            "title": "Episode 1",
            "artist": "AIVIA",
            "album": "Tests",
        },
    )

    stream = result.probe.audio_streams[0]
    assert stream.codec_name == "mp3"
    assert result.probe.duration_seconds == pytest.approx(2.0, abs=0.2)
    assert output.read_bytes().startswith(b"ID3")
    assert result.tags["title"] == "Episode 1"
    assert result.tags["artist"] == "AIVIA"
    assert result.tags["album"] == "Tests"

"""Mix composition audio and burn safe-area subtitles into a real video."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from aivia.compose import (
    AudioSubtitleError,
    Composition,
    FFmpegCommandError,
    VideoRenderer,
    VideoRenderError,
    audio_mix_settings,
    audio_tracks_for_mix,
    build_audio_filtergraph,
    build_ffmpeg_command,
    build_ffmpeg_filtergraph,
    render_video,
    subtitle_document,
    subtitle_filter,
    write_subtitle_document,
)
from aivia.media.audio_mix import verify_ducking
from aivia.media.subtitles import Subtitle as MediaSubtitle

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

pytestmark = pytest.mark.slow

requires_ffmpeg = pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None,
    reason="ffmpeg and ffprobe are required for a real video render",
)


def _composition(**overrides: Any) -> Composition:
    values: dict[str, Any] = {
        "canvas": {"width": 320, "height": 240, "background": "#000000"},
        "duration_seconds": 2.0,
        "fps": 24,
        "layers": [],
    }
    values.update(overrides)
    return Composition.model_validate(values)


def _audio_composition(**overrides: Any) -> Composition:
    values: dict[str, Any] = {
        "audio_tracks": [
            {
                "id": "voice",
                "source": "voice.wav",
                "role": "voice",
                "volume": -2.0,
                "start": 0.25,
                "fade_in": 0.1,
            },
            {
                "id": "music",
                "source": "music.wav",
                "role": "music",
                "volume": -12.0,
                "fade_out": 0.5,
                "duck": True,
            },
        ],
        "loudness_lufs": -16.0,
    }
    values.update(overrides)
    return _composition(**values)


def _subtitle_composition(**overrides: Any) -> Composition:
    values: dict[str, Any] = {
        "canvas": {"width": 1080, "height": 1920, "background": "#000000"},
        "subtitles": {"cues": [{"start": 0.0, "end": 1.5, "text": "Hello world"}]},
        "safe_area": {"top": 0.1, "bottom": 0.2, "left": 0.1, "right": 0.1},
    }
    values.update(overrides)
    return _composition(**values)


def test_composition_tracks_keep_gain_offsets_fades_and_roles() -> None:
    composition = _audio_composition(
        audio_tracks=[
            {
                "id": "voice",
                "source": "voice.wav",
                "role": "voice",
                "volume": -3.0,
                "start": 0.4,
                "duration": 1.2,
                "fade_in": 0.2,
                "fade_out": 0.1,
            },
            {"id": "music", "source": "music.wav", "role": "music"},
            {"id": "off", "source": "off.wav", "enabled": False},
        ]
    )

    tracks = audio_tracks_for_mix(composition)

    assert [track.source for track in tracks] == ["voice.wav", "music.wav"]
    voice = tracks[0]
    assert voice.voice is True
    assert voice.volume_db == pytest.approx(-3.0)
    assert voice.offset_seconds == pytest.approx(0.4)
    assert voice.duration_seconds == pytest.approx(1.2)
    assert voice.fade_in_seconds == pytest.approx(0.2)
    assert voice.fade_out_seconds == pytest.approx(0.1)
    assert tracks[1].should_duck is True


def test_mix_settings_use_the_composition_then_the_explicit_loudness() -> None:
    composition = _audio_composition()

    configured = audio_mix_settings(composition)
    explicit = audio_mix_settings(composition, loudness_lufs=-20.0)
    default = audio_mix_settings()

    assert configured.target_lufs == pytest.approx(-16.0)
    assert configured.normalize_loudness is True
    assert configured.duration_seconds == pytest.approx(2.0)
    assert explicit.target_lufs == pytest.approx(-20.0)
    assert default.target_lufs == pytest.approx(-14.0)


def test_audio_graph_offsets_inputs_and_verifies_ducking() -> None:
    composition = _audio_composition()

    graph = build_audio_filtergraph(composition, input_offset=3)
    settings = audio_mix_settings(composition)
    verification = verify_ducking(
        graph,
        settings.ducking,
        voice_tracks=1,
        background_tracks=1,
    )

    assert "[3:a]" in graph
    assert "[4:a]" in graph
    assert "volume=-2.000000dB" in graph
    assert "volume=-12.000000dB" in graph
    assert "afade=t=in:st=0:d=0.100000" in graph
    assert "areverse,afade=t=in:st=0:d=0.500000,areverse" in graph
    assert "adelay=250:all=1" in graph
    assert "sidechaincompress=" in graph
    assert "loudnorm=I=-16.000000" in graph
    assert "atrim=duration=2.000000" in graph
    assert verification.verified is True


def test_a_disabled_audio_track_produces_no_audio_graph() -> None:
    composition = _composition(
        audio_tracks=[{"id": "off", "source": "off.wav", "enabled": False}]
    )

    assert build_audio_filtergraph(composition) == ""
    assert "-an" in build_ffmpeg_command(composition, "out.mp4")


def test_command_mixes_audio_in_the_same_ffmpeg_invocation() -> None:
    composition = _audio_composition()

    command = build_ffmpeg_command(composition, "out.mp4")

    assert [item.source for item in command.inputs[1:]] == [
        "voice.wav",
        "music.wav",
    ]
    assert "sidechaincompress=" in command.filtergraph
    assert "loudnorm=I=-16.000000" in command.filtergraph
    assert "-an" not in command
    assert command[command.index("-map") + 1] == "[vout]"
    assert "[out]" in command
    assert command[command.index("-c:a") + 1] == "aac"
    assert command[command.index("-ar") + 1] == "48000"
    assert command[command.index("-ac") + 1] == "2"


def test_subtitles_are_margined_inside_the_safe_area() -> None:
    composition = _subtitle_composition()

    document = subtitle_document(composition)

    style = next(line for line in document.splitlines() if line.startswith("Style:"))
    fields = style.split(",")
    assert fields[19] == str(round(0.1 * 1080))
    assert fields[20] == str(round(0.1 * 1080))
    assert fields[21] == str(round(0.2 * 1920))
    assert "Dialogue: 0,0:00:00.00,0:00:01.50" in document
    assert "Hello world" in document


def test_a_media_subtitle_is_accepted_as_a_composition_track() -> None:
    track = MediaSubtitle.from_words(
        [{"word": "hi", "start": 0.0, "end": 0.5}],
        language="en",
    )

    composition = _composition(subtitles=track)

    assert composition.subtitles is not None
    assert composition.subtitles.language == "en"
    assert composition.subtitles.cues[0].text == "hi"


def test_composition_subtitles_group_words_and_export_exchange_formats() -> None:
    composition = _composition(
        subtitles={
            "words": [
                {"word": "one", "start": 0.0, "end": 0.4},
                {"word": "two", "start": 0.4, "end": 0.8},
            ]
        }
    )

    track = composition.subtitles

    assert track is not None
    assert [cue.text for cue in track.cues] == ["one two"]
    assert "00:00:00,000 --> 00:00:00,800" in track.to_srt()
    assert track.to_vtt().startswith("WEBVTT")


def test_subtitle_filter_escapes_the_ass_path() -> None:
    assert subtitle_filter("/tmp/a:b'c.ass") == "ass=filename='/tmp/a\\:b\\'c.ass'"


def test_a_written_ass_file_is_burned_after_the_final_overlay(tmp_path: Path) -> None:
    composition = _subtitle_composition()
    ass = write_subtitle_document(composition, tmp_path / "subs.ass")

    command = build_ffmpeg_command(composition, "out.mp4", subtitle_file=ass)

    assert ass.is_file()
    assert ass.read_text(encoding="utf-8").startswith("[Script Info]")
    assert f"[base0]ass=filename='{ass}'[vout]" in command.filtergraph
    assert "[vout]" in command


def test_subtitles_without_a_written_file_are_rejected_by_the_builder() -> None:
    composition = _subtitle_composition()

    with pytest.raises(FFmpegCommandError):
        build_ffmpeg_filtergraph(composition)


def test_a_subtitle_file_can_come_from_the_composition(tmp_path: Path) -> None:
    ass = write_subtitle_document(
        _subtitle_composition(),
        tmp_path / "from-composition.ass",
    )
    composition = _subtitle_composition(subtitle_file=str(ass))

    command = build_ffmpeg_command(composition, "out.mp4")

    assert f"ass=filename='{ass}'" in command.filtergraph


def test_an_srt_file_is_converted_to_ass_with_its_cue_times(tmp_path: Path) -> None:
    source = tmp_path / "captions.srt"
    source.write_text(
        "1\n"
        "00:00:00,500 --> 00:00:01,250\n"
        "Hello world\n"
        "\n"
        "2\n"
        "00:00:02,000 --> 00:00:03,000\n"
        "Second line\n",
        encoding="utf-8",
    )

    ass = write_subtitle_document(source, tmp_path / "captions.ass")

    document = ass.read_text(encoding="utf-8")
    assert document.startswith("[Script Info]")
    assert "Dialogue: 0,0:00:00.50,0:00:01.25" in document
    assert "Hello world" in document
    assert "Dialogue: 0,0:00:02.00,0:00:03.00" in document
    assert "Second line" in document


def test_a_vtt_file_is_converted_to_ass_and_ignores_cue_settings(
    tmp_path: Path,
) -> None:
    source = tmp_path / "captions.vtt"
    source.write_text(
        "WEBVTT\n"
        "\n"
        "NOTE generated\n"
        "\n"
        "cue-1\n"
        "00:01.000 --> 00:02.500 align:center\n"
        "Second &amp; line\n",
        encoding="utf-8",
    )

    ass = write_subtitle_document(source, tmp_path / "captions.ass")

    document = ass.read_text(encoding="utf-8")
    assert "Dialogue: 0,0:00:01.00,0:00:02.50" in document
    assert "Second & line" in document
    assert "align:center" not in document


def test_a_composition_srt_file_is_converted_before_burning(tmp_path: Path) -> None:
    source = tmp_path / "captions.srt"
    source.write_text(
        "1\n"
        "00:00:00,250 --> 00:00:01,750\n"
        "From composition\n",
        encoding="utf-8",
    )
    captured: dict[str, str] = {}

    def capture(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        graph = argv[argv.index("-filter_complex") + 1]
        prefix = "ass=filename='"
        start = graph.index(prefix) + len(prefix)
        path = Path(graph[start : graph.index("'", start)])
        captured["suffix"] = path.suffix
        captured["document"] = path.read_text(encoding="utf-8")
        return subprocess.CompletedProcess(list(argv), 1, stdout="", stderr="boom")

    composition = _composition(subtitle_file=str(source))

    with pytest.raises(VideoRenderError, match="could not render"):
        VideoRenderer(ffmpeg_runner=capture).render(
            composition,
            tmp_path / "out.mp4",
        )

    assert captured["suffix"] == ".ass"
    assert captured["document"].startswith("[Script Info]")
    assert "From composition" in captured["document"]
    assert "0:00:00.25,0:00:01.75" in captured["document"]



def test_mixing_without_tracks_is_rejected() -> None:
    with pytest.raises(AudioSubtitleError):
        from aivia.compose import mix_composition_audio

        mix_composition_audio(_composition(), "mix.wav")


def _run(argv: list[str]) -> None:
    subprocess.run(argv, check=True, capture_output=True, text=True)


def _tone(path: Path, frequency: int, duration: float) -> Path:
    assert FFMPEG is not None
    _run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={frequency}:duration={duration}",
            "-c:a",
            "pcm_s16le",
            str(path),
        ]
    )
    return path


def _video_frame(source: Path, time_seconds: float, destination: Path) -> Image.Image:
    assert FFMPEG is not None
    _run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{time_seconds:.6f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            str(destination),
        ]
    )
    with Image.open(destination) as image:
        return image.convert("RGB")


def _bright_pixels(image: Image.Image, top: int, bottom: int) -> int:
    count = 0
    for y in range(top, bottom):
        for x in range(image.width):
            pixel = image.getpixel((x, y))
            assert isinstance(pixel, tuple)
            red, green, blue = pixel[0], pixel[1], pixel[2]
            if red > 200 and green > 200 and blue > 200:
                count += 1
    return count


@requires_ffmpeg
def test_real_render_has_mixed_audio_and_safe_area_subtitles(
    tmp_path: Path,
) -> None:
    voice = _tone(tmp_path / "voice.wav", 440, 2.0)
    music = _tone(tmp_path / "music.wav", 220, 2.0)
    composition = _composition(
        canvas={"width": 320, "height": 240, "background": "#000000"},
        audio_tracks=[
            {"id": "voice", "source": str(voice), "role": "voice"},
            {"id": "music", "source": str(music), "role": "music", "duck": True},
        ],
        loudness_lufs=-16.0,
        subtitles={"cues": [{"start": 0.2, "end": 1.8, "text": "HELLO"}]},
        safe_area={"bottom": 0.2},
    )
    output = tmp_path / "audio-subs.mp4"

    result = render_video(composition, output, preset="ultrafast", crf=30)

    assert output.is_file()
    assert result.probe.audio_streams
    frame = _video_frame(output, 0.75, tmp_path / "frame.png")
    safe_top = round((1.0 - 0.2) * frame.height) - 60
    safe_bottom = round((1.0 - 0.2) * frame.height)
    assert _bright_pixels(frame, safe_top, safe_bottom) > 0
    assert _bright_pixels(frame, safe_bottom + 4, frame.height) == 0

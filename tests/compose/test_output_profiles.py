"""Output settings come from configuration with platform and profile overrides."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from aivia.compose import (
    Composition,
    FFmpegCommandError,
    OutputProfile,
    OutputProfileError,
    apply_output_profile,
    apply_video_config,
    build_ffmpeg_command,
    output_profile_for,
    render_video,
    resolve_output_profile,
)
from aivia.core.config import VideoConfig

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
        "duration_seconds": 4.0,
        "fps": 24,
        "layers": [],
    }
    values.update(overrides)
    return Composition.model_validate(values)


def _option(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def _video_config() -> VideoConfig:
    return VideoConfig.from_mapping(
        {
            "resolution": "1080x1920",
            "fps": 30,
            "codec": "libx264",
            "bitrate": 8000,
            "audio_codec": "aac",
            "audio_bitrate": 192,
            "loudness_lufs": -14.0,
            "duration_seconds": {"min": 20, "max": 45},
            "platforms": {
                "tiktok": {
                    "resolution": "720x1280",
                    "fps": 24,
                    "bitrate": 4000,
                    "loudness_lufs": -16.0,
                    "duration_seconds": {"max": 30},
                }
            },
            "profiles": {
                "short": {
                    "fps": 25,
                    "codec": "h264_nvenc",
                    "bitrate": 3000,
                    "audio_bitrate": 128,
                    "duration_seconds": {"min": 5, "max": 15},
                }
            },
        }
    )


def test_defaults_come_from_the_video_section() -> None:
    profile = resolve_output_profile()

    assert profile.resolution == "1080x1920"
    assert (profile.width, profile.height) == (1080, 1920)
    assert profile.fps == 30
    assert profile.codec == "libx264"
    assert profile.bitrate is None
    assert profile.audio_codec == "aac"
    assert profile.audio_bitrate is None
    assert profile.loudness_lufs == pytest.approx(-14.0)
    assert profile.duration_bounds == (20.0, 45.0)


def test_platform_and_profile_overrides_deep_merge_onto_the_base() -> None:
    profile = resolve_output_profile(
        _video_config(), platform="tiktok", profile="short"
    )

    assert (profile.width, profile.height) == (720, 1280)
    assert profile.fps == 25
    assert profile.codec == "h264_nvenc"
    assert profile.bitrate == 3000
    assert profile.audio_codec == "aac"
    assert profile.audio_bitrate == 128
    assert profile.loudness_lufs == pytest.approx(-16.0)
    assert profile.duration_bounds == (5.0, 15.0)


def test_a_root_config_mapping_is_accepted_like_a_video_section() -> None:
    config = {"video": _video_config().model_dump(exclude={"platforms", "profiles"})}

    profile = resolve_output_profile(config)
    default = resolve_output_profile({"video": None})

    assert profile.width == 1080
    assert profile.fps == 30
    assert profile.codec == "libx264"
    assert default.resolution == "1080x1920"


def test_override_aliases_use_the_long_output_names() -> None:
    video = VideoConfig.from_mapping(
        {
            "video_codec": "libvpx-vp9",
            "video_bitrate": 5000,
            "audio_bitrate_kbps": 128,
            "platforms": {"youtube": {"video_codec": "libx265", "video_bitrate": 6000}},
        }
    )

    assert video.codec == "libvpx-vp9"
    assert video.bitrate == 5000
    assert video.audio_bitrate == 128
    assert video.video_codec == "libvpx-vp9"
    assert video.video_bitrate == 5000
    assert video.bitrate_kbps == 5000
    assert video.audio_bitrate_kbps == 128
    assert video.for_platform("youtube").codec == "libx265"
    assert video.for_platform("youtube").bitrate == 6000


def test_apply_sets_canvas_frame_rate_duration_and_loudness() -> None:
    composition = _composition(loudness_lufs=-20.0)
    profile = OutputProfile(
        width=640,
        height=360,
        fps=30,
        codec="libx264",
        bitrate=2000,
        loudness_lufs=-14.0,
        min_duration_seconds=5.0,
        max_duration_seconds=8.0,
    )

    applied = apply_output_profile(composition, profile)

    assert (applied.canvas.width, applied.canvas.height) == (640, 360)
    assert applied.fps == 30
    assert applied.duration_seconds == pytest.approx(5.0)
    assert applied.loudness_lufs == pytest.approx(-14.0)
    assert composition.canvas.width == 320
    assert composition.duration_seconds == pytest.approx(4.0)


def test_duration_is_clamped_to_the_configured_bounds() -> None:
    profile = OutputProfile(
        width=320,
        height=240,
        fps=24,
        codec="libx264",
        min_duration_seconds=1.0,
        max_duration_seconds=3.0,
    )

    short = apply_output_profile(_composition(duration_seconds=0.5), profile)
    long = apply_output_profile(_composition(duration_seconds=9.0), profile)

    assert short.duration_seconds == pytest.approx(1.0)
    assert long.duration_seconds == pytest.approx(3.0)


def test_apply_video_config_returns_the_profile_it_used() -> None:
    composition, profile = apply_video_config(
        _composition(), _video_config(), platform="tiktok", profile="short"
    )

    assert composition.fps == 25
    assert composition.canvas.width == 720
    assert profile.codec == "h264_nvenc"


def test_ffmpeg_command_is_driven_by_the_resolved_profile() -> None:
    composition = _composition(
        duration_seconds=40.0,
        audio_tracks=[{"id": "voice", "source": "voice.wav", "role": "voice"}],
        loudness_lufs=-20.0,
    )

    command = build_ffmpeg_command(
        composition,
        "out.mp4",
        config=_video_config(),
        platform="tiktok",
        profile="short",
        preset="ultrafast",
    )

    assert "s=720x1280:r=25:d=15.000000[base0]" in command.filtergraph
    assert _option(command, "-t") == "15.000000"
    assert _option(command, "-r") == "25"
    assert _option(command, "-c:v") == "h264_nvenc"
    assert _option(command, "-b:v") == "3000k"
    assert "-crf" not in command
    assert _option(command, "-c:a") == "aac"
    assert _option(command, "-b:a") == "128k"
    assert "loudnorm=I=-16.000000" in command.filtergraph


def test_explicit_command_options_still_win_over_the_profile() -> None:
    command = build_ffmpeg_command(
        _composition(),
        "out.mp4",
        config=_video_config(),
        platform="tiktok",
        video_codec="libx265",
        bitrate=9000,
        loudness_lufs=-12.0,
    )

    assert _option(command, "-c:v") == "libx265"
    assert _option(command, "-b:v") == "9000k"
    assert _option(command, "-r") == "24"


def test_an_explicit_output_profile_wins_over_configuration() -> None:
    profile = OutputProfile(width=160, height=120, fps=12, codec="libvpx-vp9")

    command = build_ffmpeg_command(
        _composition(),
        "out.webm",
        config=_video_config(),
        output_profile=profile,
    )

    assert _option(command, "-c:v") == "libvpx-vp9"
    assert _option(command, "-r") == "12"
    assert "s=160x120:r=12" in command.filtergraph


def test_invalid_bitrates_are_rejected() -> None:
    composition = _composition()

    with pytest.raises(OutputProfileError):
        OutputProfile(width=1, height=1, fps=1, codec="", bitrate=100)
    with pytest.raises(OutputProfileError):
        OutputProfile(width=1, height=1, fps=1, codec="libx264", bitrate=0)
    with pytest.raises(FFmpegCommandError):
        build_ffmpeg_command(composition, bitrate=0)
    with pytest.raises(FFmpegCommandError):
        build_ffmpeg_command(composition, audio_bitrate=-1)
    with pytest.raises(FFmpegCommandError):
        build_ffmpeg_command(composition, config=_video_config(), video_codec="")


@pytest.mark.parametrize(
    ("output", "codec"),
    [
        ("out.webm", "libx264"),
        ("out.webm", "h264_nvenc"),
        ("out.mp4", "libvpx"),
        ("out.mp4", "libvpx-vp9"),
        ("out.mov", "libtheora"),
        ("out.ogv", "libx264"),
        ("out.avi", "libvpx-vp9"),
    ],
)
def test_an_output_container_that_cannot_carry_the_codec_is_rejected(
    output: str, codec: str
) -> None:
    with pytest.raises(OutputProfileError, match="container"):
        build_ffmpeg_command(_composition(), output, video_codec=codec)


def test_the_configured_codec_is_checked_against_the_output_container() -> None:
    with pytest.raises(OutputProfileError, match="libx264"):
        build_ffmpeg_command(_composition(), "out.webm", config=_video_config())


def test_an_explicit_codec_override_can_make_the_container_compatible() -> None:
    command = build_ffmpeg_command(
        _composition(),
        "out.webm",
        config=_video_config(),
        video_codec="libvpx-vp9",
    )

    assert _option(command, "-c:v") == "libvpx-vp9"


@pytest.mark.parametrize(
    ("output", "codec"),
    [
        ("out.mp4", "libx264"),
        ("out.mp4", "libx265"),
        ("out.mp4", "libaom-av1"),
        ("out.mov", "prores_ks"),
        ("out.mkv", "libx264"),
        ("out.mkv", "made-up-encoder"),
        ("out.webm", "libvpx-vp9"),
        ("out.ogv", "libtheora"),
        ("out.avi", "mpeg4"),
        ("out.ts", "libx265"),
    ],
)
def test_compatible_output_containers_are_accepted(output: str, codec: str) -> None:
    command = build_ffmpeg_command(_composition(), output, video_codec=codec)

    assert _option(command, "-c:v") == codec


def test_an_unknown_output_container_extension_is_rejected() -> None:
    with pytest.raises(OutputProfileError, match="unsupported video output container"):
        build_ffmpeg_command(_composition(), "out.xyz")


def test_a_missing_output_container_extension_is_rejected() -> None:
    with pytest.raises(OutputProfileError, match="cannot infer a video container"):
        build_ffmpeg_command(_composition(), "out")


def test_render_rejects_an_incompatible_container_before_ffmpeg_runs(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        return subprocess.CompletedProcess(list(argv), 0, stdout="", stderr="")

    with pytest.raises(OutputProfileError, match="container"):
        render_video(
            _composition(),
            tmp_path / "out.webm",
            config=_video_config(),
            ffmpeg_runner=runner,
        )

    assert calls == []


@requires_ffmpeg
def test_real_render_uses_configured_output_settings(tmp_path: Path) -> None:
    video = VideoConfig.from_mapping(
        {
            "resolution": "320x240",
            "fps": 30,
            "codec": "libx264",
            "bitrate": 500,
            "duration_seconds": {"min": 1, "max": 2},
            "platforms": {
                "tiktok": {"resolution": "160x120", "fps": 15},
            },
            "profiles": {
                "short": {"duration_seconds": {"min": 1}},
            },
        }
    )
    composition = _composition(duration_seconds=9.0)
    output = tmp_path / "profile.mp4"

    result = render_video(
        composition,
        output,
        config=video,
        platform="tiktok",
        profile="short",
        preset="ultrafast",
    )

    assert output.is_file()
    stream = result.probe.primary_video
    assert stream is not None
    assert (stream.width, stream.height) == (160, 120)
    assert stream.codec_name == "h264"
    assert result.probe.duration_seconds == pytest.approx(2.0, abs=0.2)
    assert _option(result.command.argv, "-b:v") == "500k"
    assert output_profile_for(video, platform="tiktok", profile="short").fps == 15

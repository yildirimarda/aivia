"""Rendered outputs are checked against the platform publishing limits."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from aivia.compose import (
    OutputIssues,
    OutputValidationError,
    measure_loudness,
    validate_output,
    validate_output_or_raise,
)
from aivia.core.media_probe import FFprobeInfo, FFprobeStream
from aivia.core.platforms import default_platforms_path, load_platforms

CATALOG = load_platforms(default_platforms_path())
TOLERANCE = float(
    CATALOG.require("youtube").limits.primary.loudness_tolerance_lufs or 0.0
)


@pytest.fixture(autouse=True)
def isolated_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


def _video_stream(width: int = 1080, height: int = 1920) -> FFprobeStream:
    return FFprobeStream(
        index=0,
        codec_type="video",
        codec_name="h264",
        width=width,
        height=height,
    )


def _audio_stream() -> FFprobeStream:
    return FFprobeStream(
        index=1,
        codec_type="audio",
        codec_name="aac",
        sample_rate=48000,
        channels=2,
    )


def _probe(
    *,
    duration: float | None = 30.0,
    width: int = 1080,
    height: int = 1920,
    audio: bool = True,
    size_bytes: int | None = None,
) -> FFprobeInfo:
    streams = [_video_stream(width, height)]
    if audio:
        streams.append(_audio_stream())
    return FFprobeInfo(
        format_name="mov,mp4,m4a,3gp,3g2,mj2",
        format_long_name="QuickTime / MOV",
        duration_seconds=duration,
        size_bytes=size_bytes,
        streams=tuple(streams),
    )


def _audio_probe(
    *,
    duration: float = 12.0,
    audio: bool = True,
    size_bytes: int | None = None,
) -> FFprobeInfo:
    streams = [_audio_stream()] if audio else []
    return FFprobeInfo(
        format_name="mp3",
        format_long_name="MP2/3 (MPEG audio layer 2/3)",
        duration_seconds=duration,
        size_bytes=size_bytes,
        streams=tuple(streams),
    )


def _codes(issues: OutputIssues) -> list[str]:
    return [issue.code for issue in issues]


def test_platform_data_declares_the_loudness_tolerance() -> None:
    assert TOLERANCE == 1.0


def test_a_video_that_fits_every_youtube_limit_has_no_issues() -> None:
    issues = validate_output(
        Path("short.mp4"),
        "youtube",
        probe=_probe(duration=30.0),
        size_bytes=10_000_000,
        loudness_lufs=-14.4,
    )

    assert issues == []
    assert issues.ok is True
    assert issues.media_type == "video"
    assert issues.platform is not None
    assert issues.platform.id == "youtube"


def test_one_video_reports_duration_resolution_size_audio_and_loudness() -> None:
    issues = validate_output(
        Path("short.mp4"),
        "youtube",
        probe=_probe(duration=200.0, width=1920, height=1080, audio=False),
        size_bytes=268435456001,
        loudness_lufs=-20.0,
    )

    assert _codes(issues) == [
        "duration",
        "resolution",
        "size",
        "audio",
        "loudness",
    ]
    assert all(issue.media_type == "video" for issue in issues)
    assert all("YouTube Shorts video" in issue.message for issue in issues)


def test_duration_below_the_minimum_is_reported_too() -> None:
    issues = validate_output(
        Path("clip.mp4"),
        "tiktok",
        probe=_probe(duration=1.0),
        size_bytes=1000,
        loudness_lufs=-14.0,
    )

    assert _codes(issues) == ["duration"]
    assert "below the minimum of 3s" in issues[0].message


def test_an_image_only_checks_image_limits() -> None:
    issues = validate_output(
        Path("post.png"),
        "instagram",
        probe=_probe(duration=None, width=1080, height=1080),
        size_bytes=1024,
        loudness_lufs=-2.0,
    )

    assert issues == []
    assert issues.media_type == "image"


def test_an_image_with_the_wrong_resolution_gets_only_that_issue() -> None:
    issues = validate_output(
        Path("post.png"),
        "instagram",
        probe=_probe(duration=None, width=1920, height=1080),
        size_bytes=1024,
    )

    assert _codes(issues) == ["resolution"]
    assert "Instagram Reels image resolution 1920x1080" in issues[0].message


def test_an_audio_output_checks_duration_size_audio_and_loudness() -> None:
    issues = validate_output(
        Path("voice.mp3"),
        "telegram",
        probe=_audio_probe(duration=4000.0, audio=False),
        size_bytes=60_000_000,
        loudness_lufs=-20.0,
    )

    assert _codes(issues) == ["duration", "size", "audio", "loudness"]
    assert issues.media_type == "audio"
    assert all("Telegram channel audio" in issue.message for issue in issues)


def test_an_audio_output_never_checks_resolution() -> None:
    issues = validate_output(
        Path("voice.mp3"),
        "telegram",
        probe=_audio_probe(duration=12.0),
        size_bytes=1024,
        loudness_lufs=-14.0,
    )

    assert issues == []


def test_a_web_card_only_checks_its_file_size() -> None:
    issues = validate_output(
        Path("card.html"),
        "telegram",
        size_bytes=5242880,
        loudness_lufs=0.0,
        duration_seconds=9999.0,
    )

    assert issues == []
    assert issues.media_type == "web_card"

    too_large = validate_output(
        Path("card.html"),
        "telegram",
        size_bytes=5242881,
    )

    assert _codes(too_large) == ["size"]
    assert "Telegram channel web_card" in too_large[0].message


def test_an_unsupported_media_type_is_reported_before_other_checks() -> None:
    issues = validate_output(
        Path("post.png"),
        "youtube",
        media_type="image",
        size_bytes=999999999999,
    )

    assert _codes(issues) == ["media_type"]
    assert "does not support image" in issues[0].message


def test_the_media_type_is_inferred_from_the_file_suffix() -> None:
    assert validate_output(Path("clip.mp4"), "x", size_bytes=1).media_type == "video"
    assert validate_output(Path("clip.mp3"), "telegram", size_bytes=1).media_type == (
        "audio"
    )
    assert validate_output(Path("card.htm"), "telegram", size_bytes=1).media_type == (
        "web_card"
    )


def test_a_platform_object_and_a_catalog_are_accepted() -> None:
    by_object = validate_output(
        Path("clip.mp4"),
        CATALOG.require("x"),
        probe=_probe(duration=10.0, width=1280, height=720),
        size_bytes=1000,
    )
    by_catalog = validate_output(
        Path("clip.mp4"),
        "x",
        catalog=CATALOG,
        probe=_probe(duration=10.0, width=1280, height=720),
        size_bytes=1000,
    )

    assert by_object == []
    assert by_catalog == []


def test_a_mapping_output_can_carry_its_own_measurements() -> None:
    issues = validate_output(
        {
            "path": "clip.mp4",
            "probe": _probe(duration=10.0, width=1280, height=720),
            "size_bytes": 1000,
            "loudness_lufs": -14.0,
        },
        "x",
    )

    assert issues == []


def test_a_render_result_is_read_through_its_public_fields(tmp_path: Path) -> None:
    output = tmp_path / "clip.mp4"
    output.write_bytes(b"video")

    class Result:
        path = output
        probe = _probe(duration=10.0, width=1280, height=720)
        size_bytes = 1000
        loudness_lufs = -14.0

    issues = validate_output(Result(), "x")

    assert issues == []


def test_validate_output_or_raise_raises_every_issue() -> None:
    with pytest.raises(OutputValidationError) as caught:
        validate_output_or_raise(
            Path("short.mp4"),
            "youtube",
            probe=_probe(duration=200.0, width=1920, height=1080, audio=False),
            size_bytes=268435456001,
            loudness_lufs=-20.0,
        )

    assert _codes(caught.value.issues) == [
        "duration",
        "resolution",
        "size",
        "audio",
        "loudness",
    ]
    assert caught.value.code == "output_validation"


def test_loudness_within_the_tolerance_is_accepted() -> None:
    inside = validate_output(
        Path("clip.mp4"),
        "youtube",
        probe=_probe(),
        size_bytes=1000,
        loudness_lufs=-14.0 + TOLERANCE,
    )
    outside = validate_output(
        Path("clip.mp4"),
        "youtube",
        probe=_probe(),
        size_bytes=1000,
        loudness_lufs=-14.0 - TOLERANCE - 0.01,
    )

    assert inside == []
    assert _codes(outside) == ["loudness"]


def test_an_explicit_loudness_tolerance_overrides_the_platform_value() -> None:
    issues = validate_output(
        Path("clip.mp4"),
        "youtube",
        probe=_probe(),
        size_bytes=1000,
        loudness_lufs=-18.0,
        loudness_tolerance_lufs=4.0,
    )

    assert issues == []


def test_measure_loudness_reads_the_ffmpeg_json_report(tmp_path: Path) -> None:
    target = tmp_path / "clip.mp4"
    target.write_bytes(b"video")

    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            argv,
            0,
            "",
            'frame=1\n{"input_i" : "-14.2", "input_tp" : "-1.0"}\n',
        )

    assert measure_loudness(target, runner=runner) == pytest.approx(-14.2)


def test_measure_loudness_can_feed_validation(tmp_path: Path) -> None:
    target = tmp_path / "clip.mp4"
    target.write_bytes(b"video")

    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        assert "loudnorm=print_format=json" in argv
        return subprocess.CompletedProcess(
            argv,
            0,
            "",
            '{"input_i" : "-14.1"}',
        )

    issues = validate_output(
        target,
        "youtube",
        probe=_probe(),
        size_bytes=1000,
        loudness_runner=runner,
    )

    assert issues == []


def test_a_failed_loudness_measurement_is_a_loudness_issue(tmp_path: Path) -> None:
    target = tmp_path / "clip.mp4"
    target.write_bytes(b"video")

    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 1, "", "bad audio")

    issues = validate_output(
        target,
        "youtube",
        probe=_probe(),
        size_bytes=1000,
        measure=True,
        loudness_runner=runner,
    )

    assert _codes(issues) == ["loudness"]
    assert "could not be measured" in issues[0].message


def test_missing_measurements_are_not_guessed() -> None:
    issues: list[Any] = validate_output(
        {"path": "unknown.mp4"},
        "youtube",
        media_type="video",
    )

    assert issues == []

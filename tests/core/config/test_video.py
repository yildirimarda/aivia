"""Video spec: resolution, frame rate, duration bounds and override merging."""

from __future__ import annotations

from typing import Any, cast

import pytest
import yaml

from aivia.core import paths
from aivia.core.config import (
    DurationRangeConfig,
    SectionRegistry,
    VideoConfig,
    VideoOverridesConfig,
    deep_merge,
    parse_resolution,
    register_core_sections,
    resolve_video,
)
from aivia.core.errors import ConfigError


def test_video_defaults_parse_resolution_fps_and_duration_bounds() -> None:
    video = VideoConfig.from_mapping({})

    assert video.resolution == "1080x1920"
    assert (video.width, video.height) == (1080, 1920)
    assert video.aspect_ratio == pytest.approx(1080 / 1920)
    assert video.fps == 30
    assert video.duration_seconds == DurationRangeConfig(min=20, max=45)
    assert video.loudness_lufs == -14.0


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1920x1080", (1920, 1080)),
        ("1080X1920", (1080, 1920)),
        ("1080\u00d71920", (1080, 1920)),
        ("1080*1920", (1080, 1920)),
        (" 720 x 1280 ", (720, 1280)),
    ],
)
def test_resolution_accepts_common_separators(
    value: str, expected: tuple[int, int]
) -> None:
    video = VideoConfig.from_mapping({"resolution": value})

    assert parse_resolution(value) == expected
    assert (video.width, video.height) == expected
    assert video.resolution == value.strip()


@pytest.mark.parametrize(
    "value",
    ["1080p", "1920", "1080x", "x1920", "0x1920", "1080x-1", "abcxdef"],
)
def test_invalid_resolution_names_the_video_field(value: str) -> None:
    with pytest.raises(ConfigError) as caught:
        VideoConfig.from_mapping({"resolution": value})

    assert "resolution" in str(caught.value)


def test_resolution_error_keeps_the_section_path_through_the_registry() -> None:
    sections = register_core_sections(SectionRegistry())

    with pytest.raises(ConfigError) as caught:
        sections.parse({"video": {"resolution": "not-a-resolution"}})

    assert "video.resolution" in str(caught.value)


@pytest.mark.parametrize("fps", [0, -1])
def test_fps_must_be_within_the_supported_range(fps: int) -> None:
    with pytest.raises(ConfigError) as caught:
        VideoConfig.from_mapping({"fps": fps})

    assert "fps" in str(caught.value)


@pytest.mark.parametrize("field", ["min", "max"])
@pytest.mark.parametrize("value", [0, -5])
def test_duration_bounds_must_be_positive(field: str, value: int) -> None:
    with pytest.raises(ConfigError) as caught:
        VideoConfig.from_mapping({"duration_seconds": {field: value}})

    assert f"duration_seconds.{field}" in str(caught.value)


def test_duration_min_must_not_exceed_max() -> None:
    with pytest.raises(ConfigError) as caught:
        VideoConfig.from_mapping({"duration_seconds": {"min": 50, "max": 40}})

    assert "duration min must not exceed max" in str(caught.value)


def test_equal_duration_bounds_are_allowed() -> None:
    video = VideoConfig.from_mapping({"duration_seconds": {"min": 30, "max": 30}})

    assert video.duration_seconds.min == 30
    assert video.duration_seconds.max == 30


def test_platform_override_deep_merges_nested_duration_bounds() -> None:
    video = VideoConfig.from_mapping(
        {
            "duration_seconds": {"min": 20, "max": 45},
            "platforms": {
                "youtube": {
                    "resolution": "1920x1080",
                    "fps": 60,
                    "duration_seconds": {"max": 60},
                }
            },
        }
    )

    resolved = video.for_platform("youtube")

    assert resolved.resolution == "1920x1080"
    assert resolved.fps == 60
    assert resolved.duration_seconds.min == 20
    assert resolved.duration_seconds.max == 60
    assert video.duration_seconds.max == 45


def test_profile_override_deep_merges_without_mutating_the_base() -> None:
    video = VideoConfig.from_mapping(
        {
            "platforms": {"tiktok": {"fps": 30, "duration_seconds": {"max": 30}}},
            "profiles": {
                "short": {
                    "fps": 24,
                    "duration_seconds": {"min": 5},
                    "loudness_lufs": -16.0,
                }
            },
        }
    )

    resolved = video.for_profile("short")

    assert resolved.fps == 24
    assert resolved.duration_seconds.min == 5
    assert resolved.duration_seconds.max == 45
    assert resolved.loudness_lufs == -16.0
    assert video.fps == 30
    assert video.duration_seconds.min == 20


def test_profile_wins_over_platform_when_both_are_selected() -> None:
    video = VideoConfig.from_mapping(
        {
            "platforms": {"youtube": {"fps": 60, "resolution": "1920x1080"}},
            "profiles": {"short": {"fps": 24}},
        }
    )

    resolved = video.merged(platform="youtube", profile="short")

    assert resolved.fps == 24
    assert resolved.resolution == "1920x1080"


def test_override_aliases_accept_the_long_field_names() -> None:
    video = VideoConfig.from_mapping(
        {
            "platform_overrides": {"youtube": {"fps": 60}},
            "profile_overrides": {"short": {"duration": {"max": 30}}},
        }
    )

    assert video.platform_overrides["youtube"].fps == 60
    assert video.profile_overrides["short"].duration_seconds is not None
    assert video.resolve(platform="youtube", profile="short").fps == 60
    assert video.for_profile("short").duration_seconds.max == 30


def test_unknown_platform_and_profile_report_their_available_names() -> None:
    video = VideoConfig.from_mapping(
        {
            "platforms": {"youtube": {}},
            "profiles": {"short": {}, "long": {}},
        }
    )

    with pytest.raises(ConfigError) as platform_error:
        video.for_platform("missing")
    with pytest.raises(ConfigError) as profile_error:
        video.for_profile("missing")

    assert "unknown video platform: missing" in str(platform_error.value)
    assert "available: youtube" in str(platform_error.value)
    assert "unknown video profile: missing" in str(profile_error.value)
    assert "available: long, short" in str(profile_error.value)


def test_codec_and_bitrate_defaults_are_configurable_with_aliases() -> None:
    default = VideoConfig.from_mapping({})
    configured = VideoConfig.from_mapping(
        {
            "video_codec": "libvpx-vp9",
            "video_bitrate": 5000,
            "audio_codec": "opus",
            "audio_bitrate_kbps": 128,
        }
    )

    assert default.codec == "libx264"
    assert default.bitrate is None
    assert default.audio_codec == "aac"
    assert default.audio_bitrate is None
    assert configured.codec == "libvpx-vp9"
    assert configured.bitrate == 5000
    assert configured.audio_codec == "opus"
    assert configured.audio_bitrate == 128
    assert configured.bitrate_kbps == 5000


def test_override_models_reject_unknown_keys_with_the_full_path() -> None:
    with pytest.raises(ConfigError) as caught:
        VideoConfig.from_mapping({"platforms": {"youtube": {"unknown_option": 8}}})

    assert "platforms.youtube.unknown_option" in str(caught.value)


def test_partial_override_duration_is_validated() -> None:
    with pytest.raises(ConfigError) as caught:
        VideoOverridesConfig.from_mapping({"duration_seconds": {"min": 30, "max": 10}})

    assert "duration min must not exceed max" in str(caught.value)


def test_deep_merge_replaces_lists_and_keeps_unmentioned_keys() -> None:
    base = {"video": {"fps": 30, "duration_seconds": {"min": 20, "max": 45}}}
    override = {"video": {"fps": 24, "duration_seconds": {"max": 60}, "stages": []}}

    merged = deep_merge(base, override)

    assert merged == {
        "video": {
            "fps": 24,
            "duration_seconds": {"min": 20, "max": 60},
            "stages": [],
        }
    }
    assert base["video"]["fps"] == 30


def test_resolve_video_helper_matches_the_method() -> None:
    video = VideoConfig.from_mapping({"platforms": {"youtube": {"fps": 60}}})

    assert resolve_video(video, platform="youtube") == video.for_platform("youtube")


def test_example_config_declares_video_controls() -> None:
    text = (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )
    payload = cast(dict[str, Any], yaml.safe_load(text))
    video = VideoConfig.from_mapping(payload["video"])

    assert video.resolution == "1080x1920"
    assert video.fps == 30
    assert video.duration_seconds.min == 20
    assert video.duration_seconds.max == 45

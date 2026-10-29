"""Platform data contract: packaged defaults, lookups and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from aivia.core import paths
from aivia.core.bootstrap import bootstrap
from aivia.core.errors import ConfigError
from aivia.core.platforms import (
    PLATFORMS_FILENAME,
    PLATFORMS_SCHEMA_VERSION,
    CaptionLimits,
    DisclosureRules,
    DurationLimits,
    PlatformCatalog,
    SafeArea,
    default_platforms_path,
    get_platform,
    load_platform_data,
    load_platforms,
    platform_names,
    resolve_platforms_path,
)

MINIMAL = """
schema_version: 1
platforms:
  video_site:
    name: Video site
    media_types: [video]
    duration:
      min_seconds: 2
      max_seconds: 60
    resolutions: [1080x1920]
    aspect_ratios: ["9:16"]
    size_limits:
      max_bytes: 1000000
    caption_limits:
      max_length: 500
      max_hashtags: 5
    safe_area:
      top: 0.1
      bottom: 0.2
      left: 0.05
      right: 0.05
    disclosure:
      required: true
      text: "AI generated"
      on_screen_label: "AI"
""".lstrip()


@pytest.fixture(autouse=True)
def isolated_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


def write_data(tmp_path: Path, text: str = MINIMAL) -> Path:
    path = tmp_path / PLATFORMS_FILENAME
    path.write_text(text, encoding="utf-8")
    return path


def test_the_package_ships_platform_data_with_every_platform() -> None:
    catalog = load_platforms()

    assert catalog.schema_version == PLATFORMS_SCHEMA_VERSION
    assert catalog.ids() == ("instagram", "telegram", "tiktok", "x", "youtube")
    assert catalog.source == default_platforms_path()
    assert default_platforms_path().is_file()


def test_the_default_data_covers_the_documented_limits() -> None:
    catalog = load_platforms()
    youtube = catalog.require("youtube")

    assert youtube.media_types == ("video",)
    assert youtube.min_duration_seconds == 1
    assert youtube.max_duration_seconds == 180
    assert youtube.resolutions == ("1080x1920",)
    assert youtube.aspect_ratios == ("9:16",)
    assert youtube.max_file_size_bytes == 268435456000
    assert youtube.caption_limits.max_length == 5000
    assert youtube.caption_limits.max_title_length == 100
    assert youtube.caption_limits.max_hashtags == 15
    assert youtube.safe_area.top == pytest.approx(0.10)
    assert youtube.safe_area.bottom == pytest.approx(0.20)
    assert youtube.disclosure.required is True
    assert youtube.disclosure.text
    assert youtube.disclosure.metadata_field == "containsSyntheticMedia"


def test_every_default_platform_has_the_required_sections() -> None:
    for platform in load_platforms().entries().values():
        assert platform.media_types
        assert platform.limits
        assert platform.caption_limits.max_length is not None
        assert platform.safe_area.top >= 0
        assert platform.safe_area.top <= 1
        assert platform.disclosure.text


def test_a_platform_supports_only_its_declared_media_types() -> None:
    instagram = load_platforms().require("instagram")

    assert instagram.supports("video") is True
    assert instagram.supports("image") is True
    assert instagram.supports("audio") is False
    assert instagram.limit("video") is not None
    assert instagram.limit("video").max_duration_seconds == 900  # type: ignore[union-attr]
    assert instagram.limit("audio") is None


def test_lookup_accepts_documented_aliases_and_normalises_case() -> None:
    catalog = load_platforms()

    assert catalog["youtube_shorts"].id == "youtube"
    assert catalog["YouTube Shorts"].id == "youtube"
    assert catalog["instagram-reels"].id == "instagram"
    assert catalog["twitter"].id == "x"
    assert "youtube_shorts" in catalog


def test_an_unknown_platform_reports_the_available_ids() -> None:
    with pytest.raises(ConfigError) as caught:
        load_platforms().require("missing")

    assert "unknown platform: missing" in str(caught.value)
    assert caught.value.details["available"] == [
        "instagram",
        "telegram",
        "tiktok",
        "x",
        "youtube",
    ]


def test_media_specific_limits_are_separate() -> None:
    x = load_platforms().require("x")

    video = x.limit("video")
    image = x.limit("image")

    assert video is not None and image is not None
    assert video.max_duration_seconds == 140
    assert image.max_duration_seconds is None
    assert image.max_file_size_bytes == 5242880


def test_duration_ranges_and_safe_areas_have_convenience_helpers() -> None:
    duration = DurationLimits(min_seconds=2, max_seconds=10)
    safe = SafeArea(top=0.1, bottom=0.2, left=0.05, right=0.05)

    assert 5 in duration
    assert 11 not in duration
    assert duration.min == 2
    assert duration["max"] == 10
    assert safe.to_pixels(1000, 2000) == (200, 50, 400, 50)
    assert safe.contains(500, 1000, width=1000, height=2000) is True
    assert safe.contains(0, 0, width=1000, height=2000) is False
    assert safe["horizontal"] == pytest.approx(0.1)


def test_caption_and_disclosure_fields_accept_aliases() -> None:
    captions = CaptionLimits(max_length=280, max_hashtags=10)
    disclosure = DisclosureRules(required=True, text="AI")

    assert captions["max_chars"] == 280
    assert captions.caption == 280
    assert captions.hashtags == 10
    assert disclosure.requires_disclosure is True
    assert disclosure.disclosure_text == "AI"
    assert disclosure["metadata_supported"] is False


def test_a_user_copy_wins_over_the_packaged_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    user_file = tmp_path / "aivia" / PLATFORMS_FILENAME
    user_file.parent.mkdir(parents=True)
    user_file.write_text(MINIMAL, encoding="utf-8")

    assert resolve_platforms_path() == user_file
    catalog = load_platforms()

    assert catalog.ids() == ("video_site",)
    assert catalog.source == user_file


def test_a_missing_user_copy_falls_back_to_the_packaged_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    assert resolve_platforms_path() == default_platforms_path()
    assert load_platforms().ids() == (
        "instagram",
        "telegram",
        "tiktok",
        "x",
        "youtube",
    )


def test_a_configured_path_is_relative_to_the_config_directory(
    tmp_path: Path,
) -> None:
    target = resolve_platforms_path(
        {"path": "custom/platforms.yaml"},
        config_dir=tmp_path,
    )

    assert target == tmp_path / "custom" / "platforms.yaml"


def test_loading_a_directory_uses_the_default_filename(tmp_path: Path) -> None:
    write_data(tmp_path)

    catalog = load_platforms(tmp_path)

    assert catalog.ids() == ("video_site",)


def test_defaults_are_merged_into_each_platform(tmp_path: Path) -> None:
    path = write_data(
        tmp_path,
        """
schema_version: 1
defaults:
  disclosure:
    required: true
    text: shared disclosure
  safe_area:
    top: 0.1
    bottom: 0.1
platforms:
  one:
    media_types: [video]
  two:
    media_types: [image]
    disclosure:
      text: local disclosure
""".lstrip(),
    )

    catalog = PlatformCatalog.load(path)

    assert catalog["one"].disclosure.text == "shared disclosure"
    assert catalog["two"].disclosure.text == "local disclosure"
    assert catalog["one"].safe_area.top == pytest.approx(0.1)
    assert catalog["one"].disclosure.required is True


def test_a_list_form_and_shorthand_values_are_accepted(tmp_path: Path) -> None:
    path = write_data(
        tmp_path,
        """
schema_version: 1
platforms:
  - id: video_site
    media_types: video
    duration: 60
    resolutions: 1080x1920
    aspect_ratios: "9:16"
    caption_limits:
      max_length: 500
    disclosure: true
""".lstrip(),
    )

    catalog = PlatformCatalog.load(path)
    platform = catalog["video_site"]

    assert platform.media_types == ("video",)
    assert platform.duration.max_seconds == 60
    assert platform.resolutions == ("1080x1920",)
    assert platform.aspect_ratios == ("9:16",)
    assert platform.disclosure.required is True


def test_get_platform_and_platform_names_use_the_active_data() -> None:
    assert get_platform("youtube").id == "youtube"
    assert "telegram" in platform_names()
    assert load_platform_data().ids() == platform_names()


def test_bootstrap_copies_the_platform_data(tmp_path: Path) -> None:
    result = bootstrap(config_dir=tmp_path / "config")

    copied = tmp_path / "config" / PLATFORMS_FILENAME
    assert copied in result.created
    assert copied.is_file()
    assert PlatformCatalog.load(copied).ids() == load_platforms().ids()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("", "platform data must be a mapping"),
        ("schema_version: 0\nplatforms: {}\n", "schema_version"),
        ("schema_version: 99\nplatforms: {}\n", "newer than supported"),
        ("schema_version: 1\nbogus: {}\n", "unknown platform data key"),
        (
            "schema_version: 1\nplatforms: {site: {bogus: 1}}\n",
            "unknown key",
        ),
        (
            "schema_version: 1\nplatforms: {site: {media_types: [video], "
            "duration: {min_seconds: 10, max_seconds: 5}}}\n",
            "must not exceed",
        ),
        (
            "schema_version: 1\nplatforms: {site: {media_types: [video], "
            "resolutions: [wide]}}\n",
            "1080x1920",
        ),
        (
            "schema_version: 1\nplatforms: {site: {media_types: [video], "
            "aspect_ratios: [wide]}}\n",
            "9:16",
        ),
        (
            "schema_version: 1\nplatforms: {site: {media_types: [video], "
            "safe_area: {top: 2}}}\n",
            "between 0 and 1",
        ),
        (
            "schema_version: 1\nplatforms: {site: {media_types: [video], "
            "caption_limits: {max_length: -1}}}\n",
            "must not be negative",
        ),
        (
            "schema_version: 1\nplatforms: {site: {media_types: [video], "
            "disclosure: {required: maybe}}}\n",
            "must be a boolean",
        ),
    ],
)
def test_invalid_platform_data_is_rejected(
    tmp_path: Path,
    payload: str,
    message: str,
) -> None:
    path = tmp_path / PLATFORMS_FILENAME
    path.write_text(payload, encoding="utf-8")

    with pytest.raises((ConfigError, yaml.YAMLError)) as caught:
        PlatformCatalog.load(path)

    assert message in str(caught.value)


def test_duplicate_ids_in_the_list_form_are_rejected(tmp_path: Path) -> None:
    path = write_data(
        tmp_path,
        """
schema_version: 1
platforms:
  - id: site
  - id: site
""".lstrip(),
    )

    with pytest.raises(ConfigError, match="duplicate platform id"):
        PlatformCatalog.load(path)


def test_a_missing_file_and_broken_yaml_fail_clearly(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read platform data"):
        PlatformCatalog.load(tmp_path / "missing.yaml")

    broken = tmp_path / "broken.yaml"
    broken.write_text("platforms: [\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        PlatformCatalog.load(broken)


def test_the_packaged_file_is_valid_yaml_with_a_schema_version() -> None:
    payload: Any = yaml.safe_load(default_platforms_path().read_text(encoding="utf-8"))

    assert payload["schema_version"] == PLATFORMS_SCHEMA_VERSION
    assert payload["platforms"]
    assert paths.package_defaults_dir() / PLATFORMS_FILENAME == default_platforms_path()

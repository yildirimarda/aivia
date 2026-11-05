"""Music configuration: source, library, mood, volume and ducking."""

from __future__ import annotations

from typing import Any, cast

import pytest
import yaml

from aivia.core import paths
from aivia.core.config import (
    CORE_SECTION_KEYS,
    MusicConfig,
    MusicDucking,
    MusicDuckingConfig,
    MusicSource,
    MusicSourceMode,
    SectionRegistry,
    register_core_sections,
)
from aivia.core.errors import ConfigError


def example_section(name: str) -> dict[str, Any]:
    text = (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )
    payload = cast(dict[str, Any], yaml.safe_load(text))
    return cast(dict[str, Any], payload[name])


def test_music_aliases_point_at_the_same_models() -> None:
    assert MusicSourceMode is MusicSource
    assert MusicDucking is MusicDuckingConfig


def test_music_defaults_use_auto_source_and_voice_ducking() -> None:
    music = MusicConfig.from_mapping({})

    assert music.source is MusicSource.AUTO
    assert music.library_path == ""
    assert music.library == ""
    assert music.mood == ""
    assert music.volume_db == -6.0
    assert music.volume == -6.0
    assert music.ducking == MusicDuckingConfig()
    assert music.ducking.enabled is True
    assert music.ducking.threshold_db == -25.0
    assert music.ducking.threshold == -25.0
    assert music.ducking.ratio == 8.0
    assert music.ducking.attack_ms == 20.0
    assert music.ducking.release_ms == 300.0


def test_music_accepts_an_explicit_source_library_mood_volume_and_ducking() -> None:
    music = MusicConfig.from_mapping(
        {
            "source": "library",
            "library_path": "music/library.yaml",
            "mood": "calm",
            "volume_db": -12.0,
            "ducking": {
                "enabled": False,
                "threshold_db": -18.0,
                "ratio": 4.0,
                "attack_ms": 10.0,
                "release_ms": 500.0,
            },
        }
    )

    assert music.source is MusicSource.LIBRARY
    assert music.library_path == "music/library.yaml"
    assert music.mood == "calm"
    assert music.volume_db == -12.0
    assert music.ducking.enabled is False
    assert music.ducking.threshold_db == -18.0
    assert music.ducking.ratio == 4.0
    assert music.ducking.attack_ms == 10.0
    assert music.ducking.release_ms == 500.0


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("auto", MusicSource.AUTO),
        ("fallback", MusicSource.AUTO),
        ("library", MusicSource.LIBRARY),
        ("local", MusicSource.LIBRARY),
        ("library_only", MusicSource.LIBRARY),
        ("provider", MusicSource.PROVIDER),
        ("ai", MusicSource.PROVIDER),
        ("generate", MusicSource.PROVIDER),
        (" GENERATED ", MusicSource.PROVIDER),
        (MusicSource.LIBRARY, MusicSource.LIBRARY),
    ],
)
def test_music_accepts_every_source_spelling(
    value: object, expected: MusicSource
) -> None:
    music = MusicConfig.from_mapping({"source": value})

    assert music.source is expected


def test_music_rejects_an_unknown_source() -> None:
    with pytest.raises(ConfigError) as caught:
        MusicConfig.from_mapping({"source": "radio"})

    assert "source" in str(caught.value)
    assert "radio" in str(caught.value)


def test_music_accepts_field_aliases_and_canonical_fields_win() -> None:
    aliased = MusicConfig.from_mapping(
        {
            "mode": "local",
            "library": "music/library.yaml",
            "volume": -10.0,
            "duck": {
                "threshold": -20.0,
                "duck_ratio": 3.0,
                "attack": 5.0,
                "release": 250.0,
            },
        }
    )
    canonical = MusicConfig.from_mapping(
        {
            "source": "auto",
            "mode": "provider",
            "library_path": "canonical.yaml",
            "library": "alias.yaml",
            "volume_db": -8.0,
            "volume": -20.0,
        }
    )

    assert aliased.source is MusicSource.LIBRARY
    assert aliased.library_path == "music/library.yaml"
    assert aliased.volume_db == -10.0
    assert aliased.ducking.threshold_db == -20.0
    assert aliased.ducking.ratio == 3.0
    assert aliased.ducking.attack_ms == 5.0
    assert aliased.ducking.release_ms == 250.0
    assert canonical.source is MusicSource.AUTO
    assert canonical.library_path == "canonical.yaml"
    assert canonical.volume_db == -8.0


@pytest.mark.parametrize(
    ("value", "enabled"),
    [
        (False, False),
        (True, True),
        ("off", False),
        ("on", True),
        ("disabled", False),
        ("enabled", True),
    ],
)
def test_music_accepts_a_ducking_shorthand(value: object, enabled: bool) -> None:
    music = MusicConfig.from_mapping({"ducking": value})

    assert music.ducking.enabled is enabled


def test_music_strips_library_path_and_mood() -> None:
    music = MusicConfig.from_mapping(
        {"library_path": "  music/library.yaml  ", "mood": "  calm  "}
    )

    assert music.library_path == "music/library.yaml"
    assert music.mood == "calm"


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_music_volume_must_be_finite(value: float) -> None:
    with pytest.raises(ConfigError) as caught:
        MusicConfig.from_mapping({"volume_db": value})

    assert "volume_db" in str(caught.value)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("threshold_db", 1.0, "threshold_db"),
        ("ratio", 0.5, "ratio"),
        ("attack_ms", -1.0, "attack_ms"),
        ("release_ms", -1.0, "release_ms"),
        ("threshold_db", float("inf"), "threshold_db"),
    ],
)
def test_music_ducking_values_are_validated(
    field: str, value: float, message: str
) -> None:
    with pytest.raises(ConfigError) as caught:
        MusicDuckingConfig.from_mapping({field: value})

    assert message in str(caught.value)


def test_music_nested_ducking_defaults_are_independent() -> None:
    first = MusicConfig.from_mapping({})
    second = MusicConfig.from_mapping({})

    first.ducking.threshold_db = -12.0
    first.ducking.enabled = False

    assert second.ducking.threshold_db == -25.0
    assert second.ducking.enabled is True


def test_music_is_registered_as_a_core_section() -> None:
    assert "music" in CORE_SECTION_KEYS

    sections = register_core_sections(SectionRegistry())
    parsed = sections.parse({"music": {"source": "local", "mood": "calm"}})

    assert parsed.music.source is MusicSource.LIBRARY  # type: ignore[attr-defined]
    assert parsed.music.mood == "calm"  # type: ignore[attr-defined]


def test_music_unknown_keys_report_the_full_path() -> None:
    sections = register_core_sections(SectionRegistry())

    with pytest.raises(ConfigError) as caught:
        sections.parse({"music": {"bogus": True}})
    assert "music.bogus" in str(caught.value)

    with pytest.raises(ConfigError) as caught:
        sections.parse({"music": {"ducking": {"bogus": True}}})
    assert "music.ducking.bogus" in str(caught.value)


def test_the_packaged_example_declares_music_controls() -> None:
    music = MusicConfig.from_mapping(example_section("music"))

    assert music.source is MusicSource.AUTO
    assert music.library_path == ""
    assert music.mood == ""
    assert music.volume_db == -6.0
    assert music.ducking.enabled is True
    assert music.ducking.threshold_db == -25.0
    assert music.ducking.ratio == 8.0
    assert music.ducking.attack_ms == 20.0
    assert music.ducking.release_ms == 300.0

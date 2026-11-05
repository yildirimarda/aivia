"""Audio QA configuration: thresholds, pronunciations and failure policy."""

from __future__ import annotations

from typing import Any, cast

import pytest
import yaml

from aivia.audio import AudioQualityChecker
from aivia.core import paths
from aivia.core.config import (
    CORE_SECTION_KEYS,
    AudioQA,
    AudioQa,
    AudioQAConfig,
    AudioQaConfig,
    AudioQualityConfig,
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


def test_audio_qa_aliases_point_at_the_same_model() -> None:
    assert AudioQaConfig is AudioQAConfig
    assert AudioQualityConfig is AudioQAConfig
    assert AudioQA is AudioQAConfig
    assert AudioQa is AudioQAConfig


def test_audio_qa_defaults_match_the_checker_defaults() -> None:
    audio_qa = AudioQAConfig.from_mapping({})

    assert audio_qa.enabled is True
    assert audio_qa.max_internal_silence_seconds == 1.0
    assert audio_qa.max_leading_silence_seconds == 0.75
    assert audio_qa.max_trailing_silence_seconds == 1.0
    assert audio_qa.max_silence_ratio == 0.5
    assert audio_qa.silent_ratio == 0.9
    assert audio_qa.min_word_confidence == 0.5
    assert audio_qa.max_mispronunciation_ratio == 0.5
    assert audio_qa.expected_words_per_second == 2.5
    assert audio_qa.min_duration_ratio == 0.4
    assert audio_qa.pronunciations == {}
    assert audio_qa.fail_on_error is True
    assert audio_qa.fail_on_errors is True
    assert audio_qa.strict is True


def test_audio_qa_accepts_explicit_thresholds_pronunciations_and_failure() -> None:
    audio_qa = AudioQAConfig.from_mapping(
        {
            "enabled": False,
            "max_internal_silence_seconds": 0.5,
            "max_leading_silence_seconds": 0.25,
            "max_trailing_silence_seconds": 0.75,
            "max_silence_ratio": 0.4,
            "silent_ratio": 0.95,
            "min_word_confidence": 0.6,
            "max_mispronunciation_ratio": 0.3,
            "expected_words_per_second": 3.0,
            "min_duration_ratio": 0.6,
            "pronunciations": {
                "nuclear": ["nucular"],
                "tomato": "to-mah-to",
            },
            "fail_on_error": False,
        }
    )

    assert audio_qa.enabled is False
    assert audio_qa.max_internal_silence_seconds == 0.5
    assert audio_qa.max_leading_silence_seconds == 0.25
    assert audio_qa.max_trailing_silence_seconds == 0.75
    assert audio_qa.max_silence_ratio == 0.4
    assert audio_qa.silent_ratio == 0.95
    assert audio_qa.min_word_confidence == 0.6
    assert audio_qa.max_mispronunciation_ratio == 0.3
    assert audio_qa.expected_words_per_second == 3.0
    assert audio_qa.min_duration_ratio == 0.6
    assert audio_qa.pronunciations == {
        "nuclear": ["nucular"],
        "tomato": ["to-mah-to"],
    }
    assert audio_qa.fail_on_error is False


def test_audio_qa_thresholds_property_feeds_the_checker() -> None:
    audio_qa = AudioQAConfig.from_mapping({"silent_ratio": 0.8})

    checker = AudioQualityChecker(
        **audio_qa.thresholds,
        pronunciations=audio_qa.pronunciations,
    )

    assert checker.silent_ratio == 0.8
    assert checker.max_internal_silence_seconds == 1.0
    assert set(audio_qa.thresholds) == {
        "max_internal_silence_seconds",
        "max_leading_silence_seconds",
        "max_trailing_silence_seconds",
        "max_silence_ratio",
        "silent_ratio",
        "min_word_confidence",
        "max_mispronunciation_ratio",
        "expected_words_per_second",
        "min_duration_ratio",
    }


def test_audio_qa_accepts_thresholds_under_a_nested_mapping() -> None:
    audio_qa = AudioQAConfig.from_mapping(
        {"thresholds": {"max_silence_ratio": 0.3, "min_word_confidence": 0.7}}
    )

    assert audio_qa.max_silence_ratio == 0.3
    assert audio_qa.min_word_confidence == 0.7
    assert audio_qa.silent_ratio == 0.9


def test_audio_qa_accepts_aliases_and_canonical_fields_win() -> None:
    aliased = AudioQAConfig.from_mapping(
        {
            "enable": False,
            "max_internal_silence": 0.4,
            "max_leading_silence": 0.2,
            "max_trailing_silence": 0.3,
            "silence_ratio": 0.35,
            "almost_silent_ratio": 0.95,
            "min_confidence": 0.65,
            "mispronunciation_ratio": 0.2,
            "words_per_second": 3.2,
            "duration_ratio": 0.7,
            "pronunciation": {"nuclear": "nucular"},
            "strict": False,
        }
    )
    canonical = AudioQAConfig.from_mapping(
        {
            "enabled": True,
            "enable": False,
            "fail_on_error": True,
            "strict": False,
            "pronunciations": {"nuclear": ["nucular"]},
            "pronunciation": {"nuclear": "wrong"},
        }
    )

    assert aliased.enabled is False
    assert aliased.max_internal_silence_seconds == 0.4
    assert aliased.max_leading_silence_seconds == 0.2
    assert aliased.max_trailing_silence_seconds == 0.3
    assert aliased.max_silence_ratio == 0.35
    assert aliased.silent_ratio == 0.95
    assert aliased.min_word_confidence == 0.65
    assert aliased.max_mispronunciation_ratio == 0.2
    assert aliased.expected_words_per_second == 3.2
    assert aliased.min_duration_ratio == 0.7
    assert aliased.pronunciations == {"nuclear": ["nucular"]}
    assert aliased.fail_on_error is False
    assert canonical.enabled is True
    assert canonical.fail_on_error is True
    assert canonical.pronunciations == {"nuclear": ["nucular"]}


def test_audio_qa_strips_and_deduplicates_pronunciations() -> None:
    audio_qa = AudioQAConfig.from_mapping(
        {
            "pronunciations": {
                "  nuclear  ": ["  nucular ", "", "nucular", "new-clear"],
                "tomato": [" to-mah-to ", "to-may-to"],
            }
        }
    )

    assert audio_qa.pronunciations == {
        "nuclear": ["nucular", "new-clear"],
        "tomato": ["to-mah-to", "to-may-to"],
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_internal_silence_seconds", -0.01),
        ("max_leading_silence_seconds", -0.01),
        ("max_trailing_silence_seconds", -0.01),
        ("max_silence_ratio", -0.01),
        ("max_silence_ratio", 1.01),
        ("silent_ratio", -0.01),
        ("silent_ratio", 1.01),
        ("min_word_confidence", -0.01),
        ("min_word_confidence", 1.01),
        ("max_mispronunciation_ratio", -0.01),
        ("max_mispronunciation_ratio", 1.01),
        ("expected_words_per_second", 0.0),
        ("expected_words_per_second", -1.0),
        ("min_duration_ratio", -0.01),
        ("min_duration_ratio", 1.01),
    ],
)
def test_audio_qa_threshold_bounds_are_enforced(field: str, value: float) -> None:
    with pytest.raises(ConfigError) as caught:
        AudioQAConfig.from_mapping({field: value})

    assert field in str(caught.value)


@pytest.mark.parametrize(
    "field",
    [
        "max_internal_silence_seconds",
        "max_leading_silence_seconds",
        "max_trailing_silence_seconds",
        "max_silence_ratio",
        "silent_ratio",
        "min_word_confidence",
        "max_mispronunciation_ratio",
        "expected_words_per_second",
        "min_duration_ratio",
    ],
)
@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_audio_qa_thresholds_must_be_finite(field: str, value: float) -> None:
    with pytest.raises(ConfigError) as caught:
        AudioQAConfig.from_mapping({field: value})

    assert field in str(caught.value)


@pytest.mark.parametrize(
    "value",
    [
        {"nuclear": []},
        {"nuclear": [""]},
        {"nuclear": [1]},
        {"nuclear": b"nucular"},
        {"": ["nucular"]},
        {"   ": ["nucular"]},
        {"nuclear": {"accepted": "nucular"}},
        ["nuclear"],
    ],
)
def test_audio_qa_rejects_invalid_pronunciations(value: object) -> None:
    with pytest.raises(ConfigError) as caught:
        AudioQAConfig.from_mapping({"pronunciations": value})

    assert "pronunciation" in str(caught.value).lower()


def test_audio_qa_pronunciation_defaults_are_independent() -> None:
    first = AudioQAConfig.from_mapping({})
    second = AudioQAConfig.from_mapping({})

    first.pronunciations["nuclear"] = ["nucular"]

    assert second.pronunciations == {}


def test_audio_qa_is_registered_as_a_core_section() -> None:
    assert "audio_qa" in CORE_SECTION_KEYS

    sections = register_core_sections(SectionRegistry())
    parsed = sections.parse(
        {
            "audio_qa": {
                "silent_ratio": 0.8,
                "pronunciations": {"nuclear": ["nucular"]},
                "fail_on_error": False,
            }
        }
    )

    assert parsed.audio_qa.silent_ratio == 0.8  # type: ignore[attr-defined]
    assert parsed.audio_qa.pronunciations == {  # type: ignore[attr-defined]
        "nuclear": ["nucular"]
    }
    assert parsed.audio_qa.fail_on_error is False  # type: ignore[attr-defined]


def test_audio_qa_unknown_keys_report_the_full_path() -> None:
    sections = register_core_sections(SectionRegistry())

    with pytest.raises(ConfigError) as caught:
        sections.parse({"audio_qa": {"bogus": True}})
    assert "audio_qa.bogus" in str(caught.value)

    with pytest.raises(ConfigError) as caught:
        sections.parse({"audio_qa": {"thresholds": {"bogus": True}}})
    assert "audio_qa.bogus" in str(caught.value)


def test_the_packaged_example_declares_audio_qa_controls() -> None:
    audio_qa = AudioQAConfig.from_mapping(example_section("audio_qa"))

    assert audio_qa.enabled is True
    assert audio_qa.max_internal_silence_seconds == 1.0
    assert audio_qa.max_leading_silence_seconds == 0.75
    assert audio_qa.max_trailing_silence_seconds == 1.0
    assert audio_qa.max_silence_ratio == 0.5
    assert audio_qa.silent_ratio == 0.9
    assert audio_qa.min_word_confidence == 0.5
    assert audio_qa.max_mispronunciation_ratio == 0.5
    assert audio_qa.expected_words_per_second == 2.5
    assert audio_qa.min_duration_ratio == 0.4
    assert audio_qa.pronunciations == {}
    assert audio_qa.fail_on_error is True

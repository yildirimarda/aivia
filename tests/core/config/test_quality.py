"""Quality and scoring config: weighted rubrics, thresholds and output settings."""

from __future__ import annotations

from typing import Any, cast

import pytest
import yaml

from aivia.core import paths
from aivia.core.config import (
    AccessibilityConfig,
    PrivacyConfig,
    QualityConfig,
    RenderConfig,
    ScoringConfig,
    SectionRegistry,
    register_core_sections,
)
from aivia.core.errors import ConfigError


def example_config() -> dict[str, Any]:
    text = (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )
    return cast(dict[str, Any], yaml.safe_load(text))


def test_quality_defaults_cover_thresholds_retries_and_weights() -> None:
    quality = QualityConfig.from_mapping({})

    assert quality.enabled is True
    assert quality.min_score == 0.7
    assert quality.max_retries == 2
    assert quality.sample_frames == 3
    assert quality.vision is True
    assert quality.weight_sum == pytest.approx(1.0)
    assert set(quality.dimensions) == {
        "prompt_adherence",
        "technical_quality",
        "aesthetic_quality",
    }


def test_scoring_defaults_cover_rubric_weights_and_minimum_score() -> None:
    scoring = ScoringConfig.from_mapping({})

    assert scoring.rubric_version == "1"
    assert scoring.min_score == 0.6
    assert scoring.min_dimension_score == 0.0
    assert scoring.weight_sum == pytest.approx(1.0)
    assert set(scoring.dimensions) == {
        "relevance",
        "originality",
        "accuracy",
        "engagement",
    }


def test_quality_accepts_explicit_controls() -> None:
    quality = QualityConfig.from_mapping(
        {
            "enabled": False,
            "min_score": 0.85,
            "max_retries": 4,
            "sample_frames": 8,
            "vision": False,
            "weights": {"clarity": 0.5, "safety": 0.5},
        }
    )

    assert quality.enabled is False
    assert quality.min_score == 0.85
    assert quality.max_retries == 4
    assert quality.sample_frames == 8
    assert quality.vision is False
    assert quality.weights == {"clarity": 0.5, "safety": 0.5}


def test_scoring_accepts_explicit_rubric() -> None:
    scoring = ScoringConfig.from_mapping(
        {
            "rubric_version": "2",
            "min_score": 0.75,
            "min_dimension_score": 0.4,
            "weights": {"relevance": 0.6, "novelty": 0.4},
        }
    )

    assert scoring.rubric_version == "2"
    assert scoring.min_score == 0.75
    assert scoring.min_dimension_score == 0.4
    assert scoring.weights == {"relevance": 0.6, "novelty": 0.4}


@pytest.mark.parametrize(
    "weights",
    [
        {"relevance": 0.5, "originality": 0.4},
        {"relevance": 0.7, "originality": 0.7},
    ],
)
def test_scoring_weights_must_sum_to_one(weights: dict[str, float]) -> None:
    with pytest.raises(ConfigError) as caught:
        ScoringConfig.from_mapping({"weights": weights})

    assert "weights" in str(caught.value)
    assert "must sum to 1" in str(caught.value)


def test_quality_weights_must_sum_to_one() -> None:
    with pytest.raises(ConfigError) as caught:
        QualityConfig.from_mapping({"weights": {"clarity": 0.2, "safety": 0.2}})

    assert "weights" in str(caught.value)
    assert "must sum to 1" in str(caught.value)


def test_weight_sums_accept_floating_point_rounding() -> None:
    scoring = ScoringConfig.from_mapping(
        {"weights": {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}}
    )

    assert scoring.weight_sum == pytest.approx(1.0)


@pytest.mark.parametrize(
    "weights",
    [
        {},
        {"relevance": -0.1, "originality": 1.1},
        {" ": 1.0},
        {"relevance": float("inf")},
    ],
)
def test_invalid_scoring_weights_are_rejected(weights: dict[str, float]) -> None:
    with pytest.raises(ConfigError) as caught:
        ScoringConfig.from_mapping({"weights": weights})

    assert "weights" in str(caught.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_score", -0.01),
        ("min_score", 1.01),
        ("min_dimension_score", -0.01),
        ("min_dimension_score", 1.01),
    ],
)
def test_scoring_thresholds_stay_between_zero_and_one(
    field: str, value: float
) -> None:
    with pytest.raises(ConfigError) as caught:
        ScoringConfig.from_mapping({field: value})

    assert field in str(caught.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_score", -0.01),
        ("min_score", 1.01),
        ("max_retries", -1),
        ("sample_frames", 0),
    ],
)
def test_quality_bounds_are_enforced(field: str, value: float) -> None:
    with pytest.raises(ConfigError) as caught:
        QualityConfig.from_mapping({field: value})

    assert field in str(caught.value)


def test_quality_and_scoring_accept_common_aliases() -> None:
    quality = QualityConfig.from_mapping(
        {
            "retries": 1,
            "threshold": 0.8,
            "frame_count": 5,
        }
    )
    scoring = ScoringConfig.from_mapping(
        {
            "version": "3",
            "dimension_weights": {"relevance": 1.0},
            "minimum_score": 0.5,
            "min_weakest_dimension": 0.25,
        }
    )

    assert quality.max_retries == 1
    assert quality.min_score == 0.8
    assert quality.sample_frames == 5
    assert scoring.rubric_version == "3"
    assert scoring.weights == {"relevance": 1.0}
    assert scoring.min_score == 0.5
    assert scoring.min_dimension_score == 0.25


def test_weight_mappings_are_independent_between_instances() -> None:
    first = ScoringConfig.from_mapping({})
    second = ScoringConfig.from_mapping({})

    first.weights["relevance"] = 0.0

    assert second.weights["relevance"] == 0.35


def test_render_defaults_cover_fonts_and_contrast() -> None:
    render = RenderConfig.from_mapping({})

    assert render.fonts_dir == "package"
    assert render.require_bundled_fonts is True
    assert render.default_font == "Lato"
    assert render.fallback_fonts == ["Lato"]
    assert render.font_size == 48
    assert render.min_contrast == 4.5
    assert render.contrast_strategy == "outline"


@pytest.mark.parametrize("strategy", ["outline", "shadow", "box"])
def test_render_accepts_every_contrast_strategy(strategy: str) -> None:
    render = RenderConfig.from_mapping({"contrast_strategy": strategy})

    assert render.contrast_strategy == strategy


def test_render_rejects_unknown_contrast_strategy() -> None:
    with pytest.raises(ConfigError) as caught:
        RenderConfig.from_mapping({"contrast_strategy": "sparkle"})

    assert "contrast_strategy" in str(caught.value)


@pytest.mark.parametrize("value", [0.5, 21.5])
def test_render_contrast_must_be_a_valid_ratio(value: float) -> None:
    with pytest.raises(ConfigError) as caught:
        RenderConfig.from_mapping({"min_contrast": value})

    assert "min_contrast" in str(caught.value)


def test_render_font_settings_must_not_be_empty() -> None:
    with pytest.raises(ConfigError) as caught:
        RenderConfig.from_mapping({"default_font": ""})
    assert "default_font" in str(caught.value)

    with pytest.raises(ConfigError) as caught:
        RenderConfig.from_mapping({"fallback_fonts": ["Lato", " "]})
    assert "fallback_fonts" in str(caught.value)

    with pytest.raises(ConfigError) as caught:
        RenderConfig.from_mapping({"font_size": 0})
    assert "font_size" in str(caught.value)


def test_privacy_defaults_cover_redaction_placeholders_and_records() -> None:
    privacy = PrivacyConfig.from_mapping({})

    assert privacy.redact_logs is True
    assert privacy.redact_prompts is False
    assert privacy.store_prompt_text is True
    assert privacy.allow_personal_data is False
    assert privacy.placeholder_mode is False
    assert privacy.data_processing_record is True


def test_privacy_accepts_placeholders_and_disables_records() -> None:
    privacy = PrivacyConfig.from_mapping(
        {
            "redact_logs": False,
            "redact_prompts": True,
            "store_prompt_text": False,
            "allow_personal_data": True,
            "placeholder_mode": True,
            "data_processing_record": False,
        }
    )

    assert privacy.redact_logs is False
    assert privacy.redact_prompts is True
    assert privacy.store_prompt_text is False
    assert privacy.allow_personal_data is True
    assert privacy.placeholder_mode is True
    assert privacy.data_processing_record is False


def test_accessibility_defaults_cover_contrast_and_subtitle_fonts() -> None:
    accessibility = AccessibilityConfig.from_mapping({})

    assert accessibility.enabled is True
    assert accessibility.alt_text is True
    assert accessibility.subtitles is True
    assert accessibility.min_contrast == 4.5
    assert accessibility.subtitle_font == "Lato"
    assert accessibility.subtitle_font_size == 32
    assert accessibility.subtitle_outline is True
    assert accessibility.subtitle_background is False


@pytest.mark.parametrize("value", [0.5, 21.5])
def test_accessibility_contrast_must_be_a_valid_ratio(value: float) -> None:
    with pytest.raises(ConfigError) as caught:
        AccessibilityConfig.from_mapping({"min_contrast": value})

    assert "min_contrast" in str(caught.value)


def test_accessibility_subtitle_font_and_size_are_validated() -> None:
    with pytest.raises(ConfigError) as caught:
        AccessibilityConfig.from_mapping({"subtitle_font": ""})
    assert "subtitle_font" in str(caught.value)

    with pytest.raises(ConfigError) as caught:
        AccessibilityConfig.from_mapping({"subtitle_font_size": 0})
    assert "subtitle_font_size" in str(caught.value)


def test_registry_reports_full_paths_for_the_new_sections() -> None:
    sections = register_core_sections(SectionRegistry())

    with pytest.raises(ConfigError) as caught:
        sections.parse({"scoring": {"weights": {"relevance": 0.5}}})
    assert "scoring.weights" in str(caught.value)

    with pytest.raises(ConfigError) as caught:
        sections.parse({"quality": {"max_retries": -1}})
    assert "quality.max_retries" in str(caught.value)


def test_quality_and_scoring_are_registered_core_sections() -> None:
    sections = register_core_sections(SectionRegistry())

    assert "quality" in sections
    assert "scoring" in sections
    assert sections.get("quality").model is QualityConfig
    assert sections.get("scoring").model is ScoringConfig


def test_example_config_declares_quality_scoring_and_output_controls() -> None:
    payload = example_config()

    quality = QualityConfig.from_mapping(payload["quality"])
    scoring = ScoringConfig.from_mapping(payload["scoring"])
    render = RenderConfig.from_mapping(payload["render"])
    privacy = PrivacyConfig.from_mapping(payload["privacy"])
    accessibility = AccessibilityConfig.from_mapping(payload["accessibility"])

    assert quality.max_retries == 2
    assert quality.weight_sum == pytest.approx(1.0)
    assert scoring.rubric_version == "1"
    assert scoring.weight_sum == pytest.approx(1.0)
    assert render.default_font == "Lato"
    assert render.contrast_strategy == "outline"
    assert privacy.placeholder_mode is False
    assert accessibility.subtitle_font == "Lato"

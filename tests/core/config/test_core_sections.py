"""Core configuration sections: registration, defaults and example config."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from aivia.core import paths
from aivia.core.config import (
    CORE_SECTION_KEYS,
    CORE_SECTIONS,
    AccessibilityConfig,
    ApiConfig,
    BudgetConfig,
    ConfigModel,
    JobsConfig,
    LanguageConfig,
    LanguagesConfig,
    LLMConfig,
    PipelineConfig,
    PlatformsConfig,
    PresetsConfig,
    PrivacyConfig,
    RenderConfig,
    RetentionConfig,
    SafetyConfig,
    SectionRegistry,
    StorageConfig,
    TracingConfig,
    VideoConfig,
    register_core_sections,
    registry,
)
from aivia.core.errors import ConfigError

EXPECTED_KEYS = {
    "accessibility",
    "api",
    "budget",
    "jobs",
    "languages",
    "llm",
    "pipeline",
    "platforms",
    "presets",
    "privacy",
    "render",
    "retention",
    "safety",
    "storage",
    "tracing",
    "video",
}


def section(parsed: ConfigModel, name: str) -> Any:
    return getattr(parsed, name)


def example_sections() -> dict[str, Any]:
    text = (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )
    payload = cast(dict[str, Any], yaml.safe_load(text))
    return {key: value for key, value in payload.items() if key in CORE_SECTION_KEYS}


def test_every_core_section_is_registered_in_the_global_registry() -> None:
    assert set(CORE_SECTION_KEYS) == EXPECTED_KEYS
    assert tuple(sorted(EXPECTED_KEYS)) == CORE_SECTION_KEYS
    assert set(CORE_SECTION_KEYS) <= set(registry.keys())


def test_core_sections_are_config_models_and_cover_every_key_once() -> None:
    assert len(CORE_SECTIONS) == len(EXPECTED_KEYS)
    for key, model in CORE_SECTIONS:
        assert key in EXPECTED_KEYS
        assert issubclass(model, ConfigModel)


def test_registering_core_sections_is_idempotent() -> None:
    sections = SectionRegistry()

    assert register_core_sections(sections) is sections
    assert register_core_sections(sections) is sections

    assert sections.keys() == CORE_SECTION_KEYS


def test_registering_a_core_key_with_another_model_still_fails() -> None:
    class OtherPipeline(ConfigModel):
        pass

    sections = SectionRegistry()
    sections.register("pipeline", OtherPipeline)

    with pytest.raises(ConfigError, match="already registered: pipeline"):
        register_core_sections(sections)


@pytest.mark.parametrize(
    ("model", "field", "expected"),
    [
        (PipelineConfig, "timezone", "UTC"),
        (
            PipelineConfig,
            "stages",
            ["research", "write", "voice", "visuals", "render", "review", "publish"],
        ),
        (VideoConfig, "resolution", "1080x1920"),
        (VideoConfig, "fps", 30),
        (VideoConfig, "loudness_lufs", -14.0),
        (RenderConfig, "fonts_dir", "package"),
        (RenderConfig, "require_bundled_fonts", True),
        (BudgetConfig, "on_exceed", "pause"),
        (LLMConfig, "max_repairs", 2),
        (StorageConfig, "backend", "local"),
        (ApiConfig, "host", "127.0.0.1"),
        (JobsConfig, "concurrency", 1),
        (SafetyConfig, "enabled", True),
        (RetentionConfig, "delete_final_outputs", False),
        (PrivacyConfig, "redact_logs", True),
        (TracingConfig, "enabled", False),
        (AccessibilityConfig, "alt_text", True),
    ],
)
def test_sections_validate_an_empty_mapping_with_defaults(
    model: type[ConfigModel], field: str, expected: Any
) -> None:
    parsed = model.from_mapping({})

    assert getattr(parsed, field) == expected


def test_nested_defaults_are_independent_between_instances() -> None:
    first = VideoConfig.from_mapping({})
    second = VideoConfig.from_mapping({})

    first.duration_seconds.min = 5

    assert second.duration_seconds.min == 20


def test_languages_accept_the_direct_mapping_shape() -> None:
    config = LanguagesConfig.from_mapping(
        {
            "default": "de",
            "en": {"name": "English", "locale": "en_US"},
            "de": {"name": "German", "locale": "de_DE"},
        }
    )

    assert config.default == "de"
    assert config.languages["en"] == LanguageConfig(name="English", locale="en_US")
    assert config.languages["de"].locale == "de_DE"


def test_languages_also_accept_the_explicit_languages_key() -> None:
    config = LanguagesConfig.from_mapping(
        {"languages": {"tr": {"name": "Turkish", "locale": "tr_TR"}}}
    )

    assert config.default == "en"
    assert config.languages["tr"].name == "Turkish"


def test_unknown_language_keys_report_the_full_dotted_path() -> None:
    with pytest.raises(ConfigError) as caught:
        LanguagesConfig.from_mapping({"en": {"bogus": True}})

    assert "languages.en.bogus" in str(caught.value)


def test_platforms_accept_accounts_directly_or_under_accounts() -> None:
    direct = PlatformsConfig.from_mapping(
        {"youtube/en": {"platform": "youtube", "language": "en"}}
    )
    nested = PlatformsConfig.from_mapping(
        {"accounts": {"tiktok/tr": {"platform": "tiktok", "language": "tr"}}}
    )

    assert direct.accounts["youtube/en"].platform == "youtube"
    assert nested.accounts["tiktok/tr"].language == "tr"


def test_presets_accept_named_presets_directly_or_under_presets() -> None:
    direct = PresetsConfig.from_mapping(
        {"short": {"description": "Short vertical video"}}
    )
    nested = PresetsConfig.from_mapping(
        {"presets": {"long": {"overrides": {"video": {"fps": 24}}}}}
    )

    assert direct.presets["short"].description == "Short vertical video"
    assert nested.presets["long"].overrides == {"video": {"fps": 24}}


def test_unknown_nested_keys_keep_the_section_in_the_path() -> None:
    sections = register_core_sections(SectionRegistry())

    with pytest.raises(ConfigError) as caught:
        sections.parse({"pipeline": {"bogus": True}})

    assert "pipeline.bogus" in str(caught.value)


def test_invalid_enum_values_name_the_section_and_field() -> None:
    sections = register_core_sections(SectionRegistry())

    with pytest.raises(ConfigError) as caught:
        sections.parse({"budget": {"on_exceed": "explode"}})

    assert "budget.on_exceed" in str(caught.value)


def test_the_packaged_example_config_validates_against_the_core_sections() -> None:
    sections = register_core_sections(SectionRegistry())

    parsed = sections.parse(example_sections())

    assert section(parsed, "pipeline").timezone == "Europe/Berlin"
    assert section(parsed, "languages").languages["tr"].locale == "tr_TR"
    assert section(parsed, "video").duration_seconds.max == 45
    assert section(parsed, "retention").delete_final_outputs is False
    assert section(parsed, "tracing").exporter == "none"


def test_environment_references_are_interpolated_in_core_sections() -> None:
    sections = register_core_sections(SectionRegistry())

    parsed = sections.parse(
        {"pipeline": {"timezone": "${AIVA_TIMEZONE}"}},
        environ={"AIVA_TIMEZONE": "Europe/Istanbul"},
    )

    assert section(parsed, "pipeline").timezone == "Europe/Istanbul"


def test_example_config_lives_in_package_defaults() -> None:
    assert (paths.package_defaults_dir() / "config.example.yaml").is_file()
    assert Path(paths.package_defaults_dir()).is_dir()

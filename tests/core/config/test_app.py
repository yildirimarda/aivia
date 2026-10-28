"""Root configuration model: assembly, defaults and cross-section checks."""

from __future__ import annotations

from typing import Any, cast

import pytest
import yaml

from aivia.core import paths
from aivia.core.config import (
    AppConfig,
    ConfigModel,
    SectionRegistry,
    assemble_root_config,
    build_app_config,
    parse_app_config,
    register_core_sections,
    registry,
    validate_cross_references,
)
from aivia.core.config.sections import CORE_SECTION_KEYS
from aivia.core.errors import ConfigError


def example_sections() -> dict[str, Any]:
    text = (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )
    payload = cast(dict[str, Any], yaml.safe_load(text))
    return {key: value for key, value in payload.items() if key in CORE_SECTION_KEYS}


def test_the_root_model_exposes_every_core_section() -> None:
    assert issubclass(AppConfig, ConfigModel)
    assert set(AppConfig.model_fields) == set(CORE_SECTION_KEYS) | {"schema_version"}
    assert assemble_root_config() is not None
    assert build_app_config(registry) is not None


def test_empty_configuration_is_valid_and_sections_are_optional() -> None:
    config = AppConfig.from_mapping({})

    assert config.pipeline is None  # type: ignore[attr-defined]
    assert config.platforms is None  # type: ignore[attr-defined]
    assert config.publish is None  # type: ignore[attr-defined]


def test_the_packaged_example_validates_as_one_root_configuration() -> None:
    config = AppConfig.from_mapping(example_sections())

    assert config.pipeline.timezone == "Europe/Berlin"  # type: ignore[attr-defined]
    assert config.platforms.accounts == {}  # type: ignore[attr-defined]
    assert config.profiles.get("default").format == "narrated_still"  # type: ignore[attr-defined]


def test_unknown_root_sections_fail_with_the_registered_keys() -> None:
    with pytest.raises(ConfigError) as caught:
        AppConfig.from_mapping({"research": {}})

    message = str(caught.value)
    assert "unknown config section: research" in message
    assert "registered:" in message
    assert caught.value.details["unknown"] == ["research"]
    assert "pipeline" in caught.value.details["available"]


def test_root_values_are_interpolated_from_the_environment() -> None:
    config = AppConfig.from_mapping(
        {"pipeline": {"timezone": "${AIVA_TIMEZONE}"}},
        environ={"AIVA_TIMEZONE": "Europe/Istanbul"},
    )

    assert config.pipeline.timezone == "Europe/Istanbul"  # type: ignore[attr-defined]


def test_publish_accounts_must_exist_in_the_platform_data() -> None:
    config: dict[str, Any] = {
        "platforms": {"accounts": {"youtube/en": {}}},
        "publish": {"accounts": {"youtube/en": {}, "tiktok/en": {}}},
    }

    with pytest.raises(ConfigError) as caught:
        AppConfig.from_mapping(config)

    assert "tiktok/en" in str(caught.value)
    assert caught.value.details["available"] == ["youtube/en"]

    config["publish"] = {"accounts": {"youtube/en": {}}}
    AppConfig.from_mapping(config)


def test_profile_accounts_must_exist_in_the_platform_data() -> None:
    config = {
        "platforms": {"accounts": {"youtube/en": {}}},
        "profiles": {"default": {"accounts": ["youtube/en", "tiktok/en"]}},
    }

    with pytest.raises(ConfigError) as caught:
        AppConfig.from_mapping(config)

    assert "profile default references unknown platform account: tiktok/en" in str(
        caught.value
    )
    assert caught.value.details["profile"] == "default"
    assert caught.value.details["account"] == "tiktok/en"

    config["profiles"] = {"default": {"accounts": ["youtube/en"]}}
    AppConfig.from_mapping(config)


def test_profile_formats_must_exist_in_the_format_recipes() -> None:
    config = {
        "formats": {"narrated_still": {}},
        "profiles": {"default": {"format": "quote_card"}},
    }

    with pytest.raises(ConfigError) as caught:
        AppConfig.from_mapping(config)

    assert "profile default references unknown format: quote_card" in str(caught.value)
    assert caught.value.details["available"] == ["narrated_still"]

    config["profiles"] = {"default": {"format": "narrated_still"}}
    AppConfig.from_mapping(config)


def test_profile_languages_must_exist_in_the_languages_section() -> None:
    config = {
        "languages": {"en": {}, "de": {}},
        "profiles": {"default": {"languages": ["en", "fr"]}},
    }

    with pytest.raises(ConfigError) as caught:
        AppConfig.from_mapping(config)

    assert "profile default references unknown language: fr" in str(caught.value)
    assert caught.value.details["profile"] == "default"
    assert caught.value.details["language"] == "fr"
    assert caught.value.details["available"] == ["de", "en"]

    config["profiles"] = {"default": {"languages": ["en", "de"]}}
    AppConfig.from_mapping(config)


def test_profile_languages_are_skipped_without_a_languages_section() -> None:
    config = AppConfig.from_mapping(
        {"profiles": {"default": {"languages": ["en", "fr"]}}}
    )

    assert config.profiles.get("default").languages == ["en", "fr"]  # type: ignore[attr-defined]


def test_schedule_languages_must_exist_in_the_languages_section() -> None:
    config = {
        "languages": {"default": "en", "en": {}, "de": {}},
        "schedule": {
            "daily": {
                "profile": "default",
                "cron": "0 9 * * *",
                "languages": ["en", "fr"],
            }
        },
    }

    with pytest.raises(ConfigError) as caught:
        AppConfig.from_mapping(config)

    assert "schedule daily references unknown language: fr" in str(caught.value)
    assert caught.value.details["schedule"] == "daily"
    assert caught.value.details["language"] == "fr"
    assert caught.value.details["available"] == ["de", "en"]

    config["schedule"] = {
        "daily": {
            "profile": "default",
            "cron": "0 9 * * *",
            "languages": ["en", "de"],
        }
    }
    AppConfig.from_mapping(config)


def test_cross_reference_validation_checks_languages_on_a_mapping() -> None:
    config = {
        "languages": {"default": "en", "en": {}},
        "profiles": {"default": {"languages": ["fr"]}},
    }

    with pytest.raises(ConfigError) as caught:
        validate_cross_references(config)
    assert "profile default references unknown language: fr" in str(caught.value)

    config["profiles"] = {"default": {"languages": ["en"]}}
    validate_cross_references(config)


def test_preset_accounts_and_formats_must_exist() -> None:
    config = {
        "platforms": {"accounts": {"youtube/en": {}}},
        "formats": {"narrated_still": {}},
        "presets": {
            "short": {
                "accounts": ["youtube/en", "tiktok/en"],
                "format": "quote_card",
            }
        },
    }

    with pytest.raises(ConfigError) as caught:
        AppConfig.from_mapping(config)
    assert "preset short references unknown platform account: tiktok/en" in str(
        caught.value
    )

    config["presets"] = {"short": {"accounts": ["youtube/en"], "format": "quote_card"}}
    with pytest.raises(ConfigError) as caught:
        AppConfig.from_mapping(config)
    assert "preset short references unknown format: quote_card" in str(caught.value)

    config["presets"] = {
        "short": {"accounts": ["youtube/en"], "format": "narrated_still"}
    }
    AppConfig.from_mapping(config)


def test_references_are_skipped_when_the_target_section_is_absent() -> None:
    config = AppConfig.from_mapping(
        {
            "profiles": {"default": {"accounts": ["youtube/en"], "format": "quote"}},
            "presets": {"short": {"accounts": ["youtube/en"], "format": "quote"}},
            "publish": {"accounts": {"youtube/en": {}}},
        }
    )

    assert config.profiles.get("default").accounts == ["youtube/en"]  # type: ignore[attr-defined]


def test_module_sections_join_the_root_model() -> None:
    class ResearchConfig(ConfigModel):
        max_sources: int = 5

    sections = register_core_sections(SectionRegistry())
    sections.register("research", ResearchConfig, module="research")
    model = build_app_config(sections)

    assert "research" in model.model_fields
    config = model.from_mapping({"research": {"max_sources": 3}})
    assert config.research.max_sources == 3  # type: ignore[attr-defined]

    with pytest.raises(ConfigError) as caught:
        model.from_mapping({"unknown": {}})
    assert "unknown config section: unknown" in str(caught.value)


def test_parse_app_config_uses_the_given_registry() -> None:
    class ResearchConfig(ConfigModel):
        max_sources: int = 5

    sections = SectionRegistry()
    sections.register("research", ResearchConfig)
    config = parse_app_config(
        {"research": {"max_sources": 2}},
        sections=sections,
    )

    assert config.research.max_sources == 2  # type: ignore[attr-defined]
    assert config.pipeline is None  # type: ignore[attr-defined]


def test_cross_reference_validation_accepts_a_mapping() -> None:
    config = {
        "platforms": {"accounts": {"youtube/en": {}}},
        "profiles": {"default": {"accounts": ["tiktok/en"]}},
    }

    with pytest.raises(ConfigError) as caught:
        validate_cross_references(config)
    assert "unknown platform account: tiktok/en" in str(caught.value)

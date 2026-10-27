"""Schedule, publish and preset configuration with reference checks."""

from __future__ import annotations

from typing import Any, cast

import pytest
import yaml

from aivia.core import paths
from aivia.core.config import (
    CORE_SECTION_KEYS,
    PresetConfig,
    PresetsConfig,
    ProfilesConfig,
    PublishConfig,
    ScheduleConfig,
    ScheduleEntryConfig,
    SectionRegistry,
    is_valid_cron,
    register_core_sections,
    validate_cron,
)
from aivia.core.errors import ConfigError


def example_section(name: str) -> dict[str, Any]:
    text = (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )
    payload = cast(dict[str, Any], yaml.safe_load(text))
    return cast(dict[str, Any], payload[name])


def profiles(*names: str) -> ProfilesConfig:
    return ProfilesConfig.from_mapping({name: {} for name in names})


def test_schedule_publish_and_presets_are_registered_core_sections() -> None:
    assert {"schedule", "publish", "presets"} <= set(CORE_SECTION_KEYS)

    sections = register_core_sections(SectionRegistry())
    parsed = sections.parse(
        {
            "profiles": {"default": {}},
            "schedule": {"daily": {"profile": "default", "cron": "0 9 * * *"}},
            "publish": {
                "accounts": {
                    "youtube/en": {
                        "credential_refs": ["YOUTUBE_TOKEN"],
                    }
                }
            },
            "presets": {"short": {"profile": "default"}},
        }
    )

    assert parsed.schedule.schedules["daily"].profile == "default"  # type: ignore[attr-defined]
    assert parsed.publish.accounts["youtube/en"].credential_refs == [  # type: ignore[attr-defined]
        "YOUTUBE_TOKEN"
    ]
    assert parsed.presets.presets["short"].profile == "default"  # type: ignore[attr-defined]


def test_empty_sections_are_valid_and_defaults_are_independent() -> None:
    first = ScheduleConfig.from_mapping({})
    second = ScheduleConfig.from_mapping({})

    assert first.enabled is True
    assert first.timezone == "UTC"
    assert first.default == ""
    assert first.schedules == {}
    assert first.entries is first.schedules
    assert PublishConfig.from_mapping({}).dry_run is True
    assert PresetsConfig.from_mapping({}).presets == {}

    first.schedules["daily"] = ScheduleEntryConfig(
        name="daily", profile="default", cron="0 9 * * *"
    )
    first.schedules["daily"].languages.append("en")

    assert second.schedules == {}


def test_schedule_accepts_direct_nested_and_single_entry_shapes() -> None:
    direct = ScheduleConfig.from_mapping(
        {"daily": {"profile": "default", "cron": "0 9 * * *"}}
    )
    nested = ScheduleConfig.from_mapping(
        {
            "timezone": "Europe/Berlin",
            "schedules": {"daily": {"profile": "default", "cron": "0 9 * * *"}},
        }
    )
    single = ScheduleConfig.from_mapping(
        {"profile": "default", "cron": "0 9 * * *", "timezone": "Europe/Berlin"}
    )

    assert direct.schedules["daily"].profile == "default"
    assert nested.schedules["daily"].timezone == "Europe/Berlin"
    assert single.schedules["default"].profile == "default"
    assert single.schedules["default"].cron == "0 9 * * *"


def test_schedule_accepts_a_list_of_entries_and_aliases() -> None:
    schedule = ScheduleConfig.from_mapping(
        {
            "tz": "Europe/Istanbul",
            "entries": [
                {"name": "morning", "profile_name": "default", "when": "@daily"},
                {"profile": "default", "cron_expression": "30 18 * * 1-5"},
            ],
        }
    )

    assert schedule.schedules["morning"].cron == "0 0 * * *"
    assert schedule.schedules["morning"].timezone == "Europe/Istanbul"
    assert schedule.schedules["default"].cron == "30 18 * * 1-5"


def test_schedule_builds_a_cron_from_explicit_fields() -> None:
    entry = ScheduleEntryConfig.from_mapping(
        {
            "profile": "default",
            "minute": "15",
            "hour": "8",
            "day_of_month": "1-5",
            "month": "jan",
            "day_of_week": "mon-fri",
        }
    )

    assert entry.cron == "15 8 1-5 jan mon-fri"

    nested = ScheduleEntryConfig.from_mapping(
        {"profile": "default", "cron": {"minute": "0", "hour": "9"}}
    )
    assert nested.cron == "0 9 * * *"


@pytest.mark.parametrize(
    "expression",
    [
        "* * * * *",
        "0 9 * * *",
        "*/15 8-18 * * mon-fri",
        "0 0 1,15 * *",
        "0 0 * jan,jul sun",
        "0 0 * * 7",
        "30 9 * * *",
        "0 */2 * * *",
        "@daily",
        "@weekly",
        "0 9 * * 1-5",
        "0 9 * * MON-FRI",
        "0 0 1 1 *",
        "0 0 * * 0",
        "0 0 1-31/2 * *",
    ],
)
def test_valid_cron_expressions_are_accepted(expression: str) -> None:
    assert is_valid_cron(expression)
    assert validate_cron(expression)
    entry = ScheduleEntryConfig.from_mapping({"profile": "default", "cron": expression})
    assert entry.cron


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "not a cron",
        "* * * *",
        "* * * * * * *",
        "60 * * * *",
        "* 24 * * *",
        "* * 0 * *",
        "* * 32 * *",
        "* * * 13 *",
        "* * * * 8",
        "*/0 * * * *",
        "5-1 * * * *",
        "* * * jan-foo *",
        "* * * * funday",
        "1,,2 * * * *",
    ],
)
def test_invalid_cron_expressions_are_rejected(expression: str) -> None:
    with pytest.raises(ConfigError) as caught:
        ScheduleEntryConfig.from_mapping({"cron": expression})

    assert "cron" in str(caught.value)


def test_invalid_timezone_and_unknown_schedule_keys_are_rejected() -> None:
    with pytest.raises(ConfigError) as caught:
        ScheduleConfig.from_mapping({"timezone": "Mars/Olympus"})
    assert "timezone" in str(caught.value)

    with pytest.raises(ConfigError) as caught:
        ScheduleConfig.from_mapping({"daily": {"profile": "default", "bogus": True}})
    assert "schedule" in str(caught.value)
    assert "bogus" in str(caught.value)


def test_schedule_default_must_exist_and_duplicates_are_rejected() -> None:
    with pytest.raises(ConfigError) as caught:
        ScheduleConfig.from_mapping(
            {
                "default": "evening",
                "daily": {"profile": "default", "cron": "0 9 * * *"},
            }
        )
    assert "default schedule" in str(caught.value)

    with pytest.raises(ConfigError) as caught:
        ScheduleEntryConfig.from_mapping(
            {"profile": "default", "cron": "0 9 * * *", "languages": ["en", "en"]}
        )
    assert "duplicate" in str(caught.value)


def test_schedule_profiles_are_checked_against_existing_profiles() -> None:
    config = {
        "daily": {"profile": "default", "cron": "0 9 * * *"},
    }

    ScheduleConfig.from_mapping(config, context={"profiles": profiles("default")})
    ScheduleConfig.from_mapping(config, profiles=["default"])
    ScheduleConfig.from_mapping(config).validate_profiles(profiles("default"))
    ScheduleConfig.from_mapping(config).check_profiles(["default"])

    with pytest.raises(ConfigError) as caught:
        ScheduleConfig.from_mapping(config, context={"profiles": profiles("other")})
    assert "unknown profile: default" in str(caught.value)

    with pytest.raises(ConfigError) as caught:
        ScheduleConfig.from_mapping(config).validate_profiles({"other": {}})
    assert "unknown profile" in str(caught.value)

    multi = ScheduleConfig.from_mapping(
        {"daily": {"profiles": ["default", "short"], "cron": "0 9 * * *"}}
    )
    multi.validate_profiles(["default", "short"])
    with pytest.raises(ConfigError) as caught:
        multi.validate_profiles(["default"])
    assert "short" in str(caught.value)


def test_schedule_registry_cross_checks_profiles() -> None:
    sections = register_core_sections(SectionRegistry())

    with pytest.raises(ConfigError) as caught:
        sections.parse(
            {
                "profiles": {"other": {}},
                "schedule": {"daily": {"profile": "default", "cron": "0 9 * * *"}},
            }
        )
    assert "unknown profile" in str(caught.value)


def test_publish_accounts_parse_aliases_and_credential_references() -> None:
    config = PublishConfig.from_mapping(
        {
            "accounts": {
                "youtube/en": {
                    "credentials": ["YOUTUBE_TOKEN", "YOUTUBE_REFRESH"],
                    "profile": "default",
                },
                "telegram/tr": {"credential_ref": "TELEGRAM_TOKEN"},
            },
            "profile": "default",
        }
    )

    assert config.dry_run is True
    assert config.default_profile == "default"
    youtube = config.accounts["youtube/en"]
    assert youtube.platform == "youtube"
    assert youtube.language == "en"
    assert youtube.credential_refs == ["YOUTUBE_TOKEN", "YOUTUBE_REFRESH"]
    assert youtube.credential_ref == "YOUTUBE_TOKEN"
    assert youtube.credentials is youtube.credential_refs
    assert config.accounts["telegram/tr"].credential_refs == ["TELEGRAM_TOKEN"]

    listed = PublishConfig.from_mapping({"accounts": ["youtube/en"]})
    assert listed.accounts["youtube/en"].platform == "youtube"
    assert listed.accounts["youtube/en"].language == "en"


def test_publish_rejects_duplicate_and_malformed_credentials() -> None:
    assert (
        PublishConfig.from_mapping(
            {"accounts": {"youtube/en": {"platform": "youtube", "language": "en"}}}
        )
        .accounts["youtube/en"]
        .credential_refs
        == []
    )

    with pytest.raises(ConfigError) as caught:
        PublishConfig.from_mapping(
            {"accounts": {"youtube/en": {"credential_refs": ["TOKEN", "TOKEN"]}}}
        )
    assert "duplicate credential reference" in str(caught.value)

    with pytest.raises(ConfigError) as caught:
        PublishConfig.from_mapping(
            {"accounts": {"youtube/en": {"credential_refs": ["bad token"]}}}
        )
    assert "whitespace" in str(caught.value)


def test_publish_account_key_must_match_platform_and_language() -> None:
    with pytest.raises(ConfigError) as caught:
        PublishConfig.from_mapping(
            {
                "accounts": {
                    "youtube/en": {
                        "platform": "tiktok",
                        "language": "en",
                        "credential_refs": ["TOKEN"],
                    }
                }
            }
        )
    assert "platform does not match" in str(caught.value)

    with pytest.raises(ConfigError) as caught:
        PublishConfig.from_mapping(
            {
                "accounts": {
                    "youtube/en": {
                        "platform": "youtube",
                        "language": "tr",
                        "credential_refs": ["TOKEN"],
                    }
                }
            }
        )
    assert "language does not match" in str(caught.value)


def test_publish_profiles_and_credentials_are_checked_when_known() -> None:
    config = {
        "accounts": {
            "youtube/en": {
                "profile": "default",
                "credential_refs": ["YOUTUBE_TOKEN"],
            }
        }
    }

    parsed = PublishConfig.from_mapping(
        config,
        context={
            "profiles": profiles("default"),
            "credentials": {"YOUTUBE_TOKEN"},
        },
    )
    parsed.validate_profiles(profiles("default"))
    parsed.validate_credentials(["YOUTUBE_TOKEN"])

    with pytest.raises(ConfigError) as caught:
        PublishConfig.from_mapping(config, context={"profiles": profiles("other")})
    assert "unknown profile" in str(caught.value)

    with pytest.raises(ConfigError) as caught:
        PublishConfig.from_mapping(config).validate_credentials(["OTHER"])
    assert "unknown credential" in str(caught.value)

    with pytest.raises(ConfigError) as caught:
        PublishConfig.from_mapping(config).validate_profiles(profiles("other"))
    assert "unknown profile" in str(caught.value)


def test_publish_registry_cross_checks_profiles_and_platform_credentials() -> None:
    sections = register_core_sections(SectionRegistry())

    with pytest.raises(ConfigError) as caught:
        sections.parse(
            {
                "profiles": {"other": {}},
                "publish": {
                    "accounts": {
                        "youtube/en": {
                            "profile": "default",
                            "credential_refs": ["TOKEN"],
                        }
                    }
                },
            }
        )
    assert "unknown profile" in str(caught.value)

    with pytest.raises(ConfigError) as caught:
        sections.parse(
            {
                "platforms": {
                    "youtube/en": {
                        "platform": "youtube",
                        "language": "en",
                        "credential_refs": ["PLATFORM_TOKEN"],
                    }
                },
                "publish": {"accounts": {"youtube/en": {"credential_refs": ["TOKEN"]}}},
            }
        )
    assert "unknown credential" in str(caught.value)


def test_publish_unknown_keys_report_the_full_path() -> None:
    sections = register_core_sections(SectionRegistry())

    with pytest.raises(ConfigError) as caught:
        sections.parse({"publish": {"accounts": {"youtube/en": {"bogus": 1}}}})
    assert "publish.accounts.youtube/en.bogus" in str(caught.value)

    with pytest.raises(ConfigError) as caught:
        sections.parse({"schedule": {"daily": {"bogus": 1}}})
    assert "bogus" in str(caught.value)
    assert "schedule" in str(caught.value)


def test_presets_accept_direct_and_nested_shapes_with_a_default() -> None:
    direct = PresetsConfig.from_mapping(
        {
            "default": "short",
            "short": {"description": "Short vertical video", "profile": "default"},
        }
    )
    nested = PresetsConfig.from_mapping(
        {
            "presets": {
                "long": {"overrides": {"video": {"fps": 24}}},
            }
        }
    )

    assert direct.default == "short"
    assert direct.presets["short"].name == "short"
    assert direct.presets["short"].description == "Short vertical video"
    assert direct.presets["short"].profile == "default"
    assert nested.presets["long"].overrides == {"video": {"fps": 24}}


def test_preset_accepts_aliases_and_rejects_bad_values() -> None:
    preset = PresetConfig.from_mapping(
        {
            "title": "Portrait",
            "recipe": "narrated_still",
            "profile_name": "default",
            "brand": "studio",
            "persona": "host",
            "publish_accounts": ["youtube/en"],
            "config": {"video": {"fps": 24}},
        }
    )

    assert preset.name == "Portrait"
    assert preset.format == "narrated_still"
    assert preset.profile == "default"
    assert preset.brand_kit == "studio"
    assert preset.character == "host"
    assert preset.accounts == ["youtube/en"]
    assert preset.overrides == {"video": {"fps": 24}}

    multi = PresetConfig.from_mapping({"profiles": {"default": {}, "short": {}}})
    assert multi.profiles == ["default", "short"]

    with pytest.raises(ConfigError) as caught:
        PresetConfig.from_mapping({"accounts": ["youtube/en", "youtube/en"]})
    assert "duplicate preset account" in str(caught.value)

    with pytest.raises(ConfigError) as caught:
        PresetsConfig.from_mapping(
            {"default": "missing", "short": {"description": "Short"}}
        )
    assert "default preset" in str(caught.value)


def test_presets_check_profiles_against_existing_profiles() -> None:
    config = {"short": {"profile": "default"}}

    PresetsConfig.from_mapping(config, context={"profiles": profiles("default")})
    PresetsConfig.from_mapping(config).validate_profiles(["default"])
    PresetsConfig.from_mapping(config).check_profiles(profiles("default"))

    with pytest.raises(ConfigError) as caught:
        PresetsConfig.from_mapping(config, context={"profiles": profiles("other")})
    assert "unknown profile" in str(caught.value)

    sections = register_core_sections(SectionRegistry())
    with pytest.raises(ConfigError) as caught:
        sections.parse({"profiles": {"other": {}}, "presets": config})
    assert "unknown profile" in str(caught.value)


def test_the_packaged_example_declares_schedule_publish_and_presets() -> None:
    schedule = ScheduleConfig.from_mapping(example_section("schedule"))
    publish = PublishConfig.from_mapping(example_section("publish"))
    presets = PresetsConfig.from_mapping(example_section("presets"))

    assert schedule.enabled is True
    assert schedule.timezone == "UTC"
    assert schedule.schedules == {}
    assert publish.enabled is True
    assert publish.dry_run is True
    assert publish.accounts == {}
    assert presets.presets == {}

"""Content profiles: formats, weighted topic modes, examples and bindings."""

from __future__ import annotations

from typing import Any, cast

import pytest
import yaml

from aivia.core import paths
from aivia.core.config import (
    CORE_SECTION_KEYS,
    ContentProfile,
    ContentProfileConfig,
    ContentProfilesConfig,
    ExamplesConfig,
    ProfileConfig,
    ProfilesConfig,
    SectionRegistry,
    TopicMode,
    register_core_sections,
)
from aivia.core.errors import ConfigError


def example_config() -> dict[str, Any]:
    text = (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )
    return cast(dict[str, Any], yaml.safe_load(text))


def test_topic_mode_enum_is_exactly_the_documented_set() -> None:
    assert tuple(mode.value for mode in TopicMode) == (
        "on_this_day",
        "niche",
        "keywords",
        "feeds",
        "evergreen_ideas",
        "trending",
        "manual",
    )


def test_profile_aliases_point_at_the_same_models() -> None:
    assert ContentProfile is ContentProfileConfig
    assert ProfileConfig is ContentProfileConfig
    assert ContentProfilesConfig is ProfilesConfig
    assert ExamplesConfig is not None


def test_empty_profile_is_valid_and_every_default_is_independent() -> None:
    first = ContentProfileConfig.from_mapping({})
    second = ContentProfileConfig.from_mapping({})

    assert first.format == ""
    assert first.topic_modes == {}
    assert first.brief == ""
    assert first.niche == ""
    assert first.keywords == []
    assert first.examples == ExamplesConfig()
    assert first.sources == []
    assert first.tone == ""
    assert first.style_bible == ""
    assert first.character is None
    assert first.brand_kit is None
    assert first.accounts == []
    assert first.overrides == {}
    assert first.voice == ""
    assert first.languages == []
    assert first.visual_strategy == ""

    first.keywords.append("history")
    first.examples.liked.append("A liked example")
    first.overrides["video"] = {"fps": 24}

    assert second.keywords == []
    assert second.examples.liked == []
    assert second.overrides == {}


def test_profile_parses_every_documented_field() -> None:
    profile = ContentProfileConfig.from_mapping(
        {
            "format": "narrated_still",
            "topic_modes": {"on_this_day": 0.6, "niche": 0.4},
            "brief": "Explain one idea in under a minute.",
            "niche": "history",
            "keywords": ["rome", "republic"],
            "examples": {
                "liked": ["Calm explainer"],
                "disliked": ["Clickbait"],
            },
            "sources": ["wikipedia", "reddit"],
            "tone": "warm and factual",
            "style_bible": "history-short",
            "character": "mascot",
            "brand_kit": "default",
            "accounts": ["youtube/en", "tiktok/en"],
            "overrides": {"video": {"fps": 24}},
            "voice": "narrator-en",
            "languages": ["en", "de"],
            "visual_strategy": "hybrid",
        }
    )

    assert profile.format == "narrated_still"
    assert profile.topic_modes == {"on_this_day": 0.6, "niche": 0.4}
    assert profile.modes == profile.topic_modes
    assert profile.format_name == "narrated_still"
    assert profile.brief == "Explain one idea in under a minute."
    assert profile.niche == "history"
    assert profile.keywords == ["rome", "republic"]
    assert profile.examples.liked == ["Calm explainer"]
    assert profile.examples.disliked == ["Clickbait"]
    assert profile.sources == ["wikipedia", "reddit"]
    assert profile.tone == "warm and factual"
    assert profile.style_bible == "history-short"
    assert profile.character == "mascot"
    assert profile.brand_kit == "default"
    assert profile.accounts == ["youtube/en", "tiktok/en"]
    assert profile.overrides == {"video": {"fps": 24}}
    assert profile.config_overrides == {"video": {"fps": 24}}
    assert profile.voice == "narrator-en"
    assert profile.languages == ["en", "de"]
    assert profile.visual_strategy == "hybrid"


@pytest.mark.parametrize(
    ("topic_modes", "expected"),
    [
        ("niche", {"niche": 1.0}),
        ({"niche": 0.75, "keywords": 0.25}, {"niche": 0.75, "keywords": 0.25}),
        (["niche", "keywords"], {"niche": 0.5, "keywords": 0.5}),
        (
            [{"mode": "niche", "weight": 0.6}, {"mode": "manual", "weight": 0.4}],
            {"niche": 0.6, "manual": 0.4},
        ),
        ({"niche": {"weight": 0.8}, "manual": 0.2}, {"niche": 0.8, "manual": 0.2}),
    ],
)
def test_topic_modes_accept_string_mapping_and_list_forms(
    topic_modes: Any, expected: dict[str, float]
) -> None:
    profile = ContentProfileConfig.from_mapping({"topic_modes": topic_modes})

    assert profile.topic_modes == expected


@pytest.mark.parametrize(
    "weights",
    [
        {"niche": 0.5, "keywords": 0.4},
        {"niche": 0.7, "keywords": 0.7},
        {"niche": -0.1, "keywords": 1.1},
        {" ": 1.0},
        {"niche": float("inf")},
        {"niche": "many"},
    ],
)
def test_invalid_topic_mode_weights_are_rejected(
    weights: dict[str, Any],
) -> None:
    with pytest.raises(ConfigError) as caught:
        ContentProfileConfig.from_mapping({"topic_modes": weights})

    assert "topic" in str(caught.value)


def test_topic_mode_weights_accept_floating_point_rounding() -> None:
    profile = ContentProfileConfig.from_mapping(
        {
            "topic_modes": {
                "on_this_day": 0.1,
                "niche": 0.2,
                "keywords": 0.3,
                "feeds": 0.4,
            }
        }
    )

    assert sum(profile.topic_modes.values()) == pytest.approx(1.0)


def test_unknown_topic_modes_fail_with_the_allowed_names() -> None:
    with pytest.raises(ConfigError) as caught:
        ContentProfileConfig.from_mapping({"topic_modes": {"bogus": 1.0}})

    assert "unknown topic mode: bogus" in str(caught.value)
    assert "niche" in str(caught.value)


def test_custom_topic_modes_may_use_a_dotted_path() -> None:
    profile = ContentProfileConfig.from_mapping(
        {"topic_modes": {"research.modes.custom": 1.0}}
    )

    assert profile.topic_modes == {"research.modes.custom": 1.0}


def test_topic_mode_names_are_stripped_and_duplicates_are_rejected() -> None:
    profile = ContentProfileConfig.from_mapping({"topic_modes": {" niche ": 1.0}})

    assert profile.topic_modes == {"niche": 1.0}

    with pytest.raises(ConfigError) as caught:
        ContentProfileConfig.from_mapping(
            {"topic_modes": {"niche": 0.5, " niche ": 0.5}}
        )

    assert "duplicate topic mode: niche" in str(caught.value)


def test_examples_accept_a_list_as_liked_and_aliases() -> None:
    listed = ContentProfileConfig.from_mapping({"examples": ["one", "two"]})
    aliased = ContentProfileConfig.from_mapping(
        {
            "liked_examples": ["liked"],
            "examples": {"disliked_examples": ["disliked"]},
        }
    )
    nested = ExamplesConfig.from_mapping({"positive": ["good"], "negative": ["bad"]})

    assert listed.examples.liked == ["one", "two"]
    assert listed.examples.disliked == []
    assert aliased.examples.liked == ["liked"]
    assert aliased.examples.disliked == ["disliked"]
    assert nested.liked == ["good"]
    assert nested.disliked == ["bad"]


@pytest.mark.parametrize("field", ["keywords", "sources", "accounts", "languages"])
def test_named_lists_are_stripped_and_reject_blanks_and_duplicates(field: str) -> None:
    profile = ContentProfileConfig.from_mapping({field: [" one ", "two"]})

    assert getattr(profile, field) == ["one", "two"]

    with pytest.raises(ConfigError) as caught:
        ContentProfileConfig.from_mapping({field: [" "]})
    assert field in str(caught.value)

    with pytest.raises(ConfigError) as caught:
        ContentProfileConfig.from_mapping({field: ["one", " one "]})
    assert "duplicate" in str(caught.value)


def test_examples_must_not_be_blank_or_duplicated() -> None:
    with pytest.raises(ConfigError) as caught:
        ExamplesConfig.from_mapping({"liked": [" "]})
    assert "example" in str(caught.value)

    with pytest.raises(ConfigError) as caught:
        ExamplesConfig.from_mapping({"disliked": ["same", "same"]})
    assert "duplicate example: same" in str(caught.value)


def test_sources_and_accounts_accept_mapping_keys_as_names() -> None:
    profile = ContentProfileConfig.from_mapping(
        {
            "sources": {"wikipedia": 0.7, "reddit": 0.3},
            "accounts": {"youtube/en": {}, "tiktok/en": {}},
        }
    )

    assert profile.sources == ["wikipedia", "reddit"]
    assert profile.accounts == ["youtube/en", "tiktok/en"]


def test_profile_accepts_common_field_aliases() -> None:
    profile = ContentProfileConfig.from_mapping(
        {
            "recipe": "quote_card",
            "topic_weights": {"niche": 1.0},
            "instruction": "Keep it short.",
            "subject": "science",
            "terms": ["space"],
            "topic_sources": ["wikipedia"],
            "voice_tone": "playful",
            "style_guide": "short-form",
            "persona": "host",
            "brand": "studio",
            "publish_accounts": ["youtube/en"],
            "config": {"video": {"fps": 24}},
            "visuals": "stock",
        }
    )

    assert profile.format == "quote_card"
    assert profile.topic_modes == {"niche": 1.0}
    assert profile.brief == "Keep it short."
    assert profile.niche == "science"
    assert profile.keywords == ["space"]
    assert profile.sources == ["wikipedia"]
    assert profile.tone == "playful"
    assert profile.style_bible == "short-form"
    assert profile.character == "host"
    assert profile.brand_kit == "studio"
    assert profile.accounts == ["youtube/en"]
    assert profile.overrides == {"video": {"fps": 24}}
    assert profile.visual_strategy == "stock"


def test_canonical_fields_win_over_aliases() -> None:
    profile = ContentProfileConfig.from_mapping(
        {
            "format": "quote_card",
            "recipe": "song_video",
            "topic_modes": {"manual": 1.0},
            "topic_mode": "niche",
        }
    )

    assert profile.format == "quote_card"
    assert profile.topic_modes == {"manual": 1.0}


@pytest.mark.parametrize("value", [" ", ""])
def test_an_explicit_blank_format_is_rejected(value: str) -> None:
    with pytest.raises(ConfigError) as caught:
        ContentProfileConfig.from_mapping({"format": value})

    assert "format" in str(caught.value)


def test_blank_character_and_brand_kit_become_none() -> None:
    profile = ContentProfileConfig.from_mapping({"character": " ", "brand_kit": ""})

    assert profile.character is None
    assert profile.brand_kit is None


def test_overrides_must_be_a_mapping() -> None:
    assert ContentProfileConfig.from_mapping({"overrides": None}).overrides == {}

    with pytest.raises(ConfigError) as caught:
        ContentProfileConfig.from_mapping({"overrides": ["video"]})

    assert "overrides" in str(caught.value)


def test_profiles_accept_direct_and_nested_mappings() -> None:
    direct = ProfilesConfig.from_mapping({"short": {"format": "quote_card"}})
    nested = ProfilesConfig.from_mapping(
        {"profiles": {"short": {"format": "quote_card"}}}
    )
    content = ProfilesConfig.from_mapping(
        {"content_profiles": {"short": {"format": "quote_card"}}}
    )

    assert direct.profiles["short"].format == "quote_card"
    assert nested.profiles["short"].format == "quote_card"
    assert content.profiles["short"].format == "quote_card"


def test_profile_names_are_stripped_and_blanks_or_duplicates_are_rejected() -> None:
    config = ProfilesConfig.from_mapping({" short ": {"format": "quote_card"}})

    assert config.names == ("short",)
    assert config.profiles["short"].format == "quote_card"

    with pytest.raises(ConfigError) as caught:
        ProfilesConfig.from_mapping({" ": {"format": "quote_card"}})
    assert "profile name" in str(caught.value)

    with pytest.raises(ConfigError) as caught:
        ProfilesConfig.from_mapping(
            {
                "short": {"format": "quote_card"},
                " short ": {"format": "narrated_still"},
            }
        )
    assert "duplicate profile: short" in str(caught.value)


def test_get_returns_a_profile_and_names_the_available_ones() -> None:
    config = ProfilesConfig.from_mapping({"alpha": {}, "beta": {}})

    assert config.get("alpha") == ContentProfileConfig()
    assert config.profile("beta") == ContentProfileConfig()
    assert config.for_profile("beta") == ContentProfileConfig()

    with pytest.raises(ConfigError) as caught:
        config.get("missing")

    assert "unknown profile: missing" in str(caught.value)
    assert "available: alpha, beta" in str(caught.value)
    assert caught.value.details["available"] == ["alpha", "beta"]


def test_unknown_profile_keys_report_the_full_path_through_the_registry() -> None:
    sections = register_core_sections(SectionRegistry())

    with pytest.raises(ConfigError) as caught:
        sections.parse({"profiles": {"short": {"bogus": True}}})

    assert "profiles.short.bogus" in str(caught.value)


def test_profile_fields_are_interpolated_from_the_environment() -> None:
    profile = ContentProfileConfig.from_mapping(
        {
            "brief": "${AIVA_BRIEF}",
            "overrides": {"video": {"resolution": "${AIVA_RESOLUTION}"}},
        },
        environ={
            "AIVA_BRIEF": "Explain the news.",
            "AIVA_RESOLUTION": "1080x1920",
        },
    )

    assert profile.brief == "Explain the news."
    assert profile.overrides == {"video": {"resolution": "1080x1920"}}


def test_profiles_section_is_registered_and_parsed() -> None:
    sections = register_core_sections(SectionRegistry())

    parsed = sections.parse(
        {"profiles": {"short": {"format": "quote_card", "keywords": ["space"]}}}
    )
    profiles = cast(ProfilesConfig, cast(Any, parsed).profiles)

    assert "profiles" in CORE_SECTION_KEYS
    assert profiles.get("short").keywords == ["space"]


def test_the_packaged_example_declares_a_valid_profile() -> None:
    profiles = ProfilesConfig.from_mapping(example_config()["profiles"])
    profile = profiles.get("default")

    assert profile.format == "narrated_still"
    assert profile.topic_modes == {"niche": 0.7, "keywords": 0.3}
    assert profile.sources == ["wikipedia"]
    assert profile.examples == ExamplesConfig()

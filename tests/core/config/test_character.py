"""Character and brand kit configuration: chain, QA, consent and branding."""

from __future__ import annotations

from typing import Any, cast

import pytest
import yaml

from aivia.core import paths
from aivia.core.config import (
    CORE_SECTION_KEYS,
    BrandConfig,
    BrandingConfig,
    BrandKit,
    BrandKitConfig,
    BrandsConfig,
    CharacterConfig,
    CharacterConsent,
    CharacterConsentConfig,
    CharacterConsistencyStrategy,
    CharacterQA,
    CharacterQAConfig,
    CharacterQaConfig,
    ConsentConfig,
    ConsistencyStrategy,
    ConsistencyStrategyOrder,
    QAConfig,
    SectionRegistry,
    register_core_sections,
)
from aivia.core.errors import ConfigError

DEFAULT_CHAIN = [
    ConsistencyStrategy.COMPOSITE,
    ConsistencyStrategy.REFERENCE,
    ConsistencyStrategy.ADAPTER,
    ConsistencyStrategy.PROMPT_ONLY,
]


def example_section(name: str) -> dict[str, Any]:
    text = (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )
    payload = cast(dict[str, Any], yaml.safe_load(text))
    return cast(dict[str, Any], payload[name])


def test_character_aliases_point_at_the_same_models() -> None:
    assert CharacterConsistencyStrategy is ConsistencyStrategy
    assert ConsistencyStrategyOrder is ConsistencyStrategy
    assert CharacterQaConfig is CharacterQAConfig
    assert CharacterQA is CharacterQAConfig
    assert QAConfig is CharacterQAConfig
    assert CharacterConsent is CharacterConsentConfig
    assert ConsentConfig is CharacterConsentConfig


def test_character_defaults_keep_prompt_only_last_and_consent_off() -> None:
    character = CharacterConfig.from_mapping({})

    assert character.enabled is True
    assert character.consistency_strategies == DEFAULT_CHAIN
    assert character.strategies == DEFAULT_CHAIN
    assert character.strategy_order == DEFAULT_CHAIN
    assert character.consistency_order == DEFAULT_CHAIN
    assert character.consistency_strategy_order == DEFAULT_CHAIN
    assert character.qa == CharacterQAConfig()
    assert character.qa.min_score == 0.8
    assert character.qa.threshold == 0.8
    assert character.qa.min_consistency == 0.8
    assert character.qa.min_similarity == 0.8
    assert character.qa.vision_threshold == 0.8
    assert character.qa.vision_similarity == 0.8
    assert character.qa.embedding_threshold == 0.75
    assert character.qa.embedding_similarity == 0.75
    assert character.qa.max_retries == 2
    assert character.qa.retries == 2
    assert character.consent == CharacterConsentConfig()
    assert character.consent.voice_cloning is False
    assert character.consent.voice_clone is False
    assert character.consent.allow_voice_cloning is False
    assert character.consent.real_person_likeness is False
    assert character.consent.likeness is False
    assert character.consent.allow_likeness is False
    assert character.consent.resemblance_warning is True
    assert character.consent.require_ownership is True
    assert character.consent.ownership_required is True
    assert character.max_reference_images == 4
    assert character.reference_image_limit == 4
    assert character.voice_cloning is False
    assert character.voice_clone is False
    assert character.likeness is False


def test_character_accepts_an_explicit_chain_qa_consent_and_limit() -> None:
    character = CharacterConfig.from_mapping(
        {
            "enabled": False,
            "consistency_strategies": ["reference", "composite", "prompt_only"],
            "qa": {
                "min_score": 0.9,
                "vision_threshold": 0.85,
                "embedding_threshold": 0.7,
                "max_retries": 4,
            },
            "consent": {
                "voice_cloning": True,
                "real_person_likeness": True,
                "resemblance_warning": False,
                "require_ownership": False,
            },
            "max_reference_images": 6,
        }
    )

    assert character.enabled is False
    assert character.consistency_strategies == [
        ConsistencyStrategy.REFERENCE,
        ConsistencyStrategy.COMPOSITE,
        ConsistencyStrategy.PROMPT_ONLY,
    ]
    assert character.qa.min_score == 0.9
    assert character.qa.vision_threshold == 0.85
    assert character.qa.embedding_threshold == 0.7
    assert character.qa.max_retries == 4
    assert character.consent.voice_cloning is True
    assert character.consent.real_person_likeness is True
    assert character.consent.resemblance_warning is False
    assert character.consent.require_ownership is False
    assert character.max_reference_images == 6


def test_character_accepts_the_strategy_order_alias() -> None:
    character = CharacterConfig.from_mapping(
        {"strategy_order": ["composite", "prompt_only"]}
    )

    assert character.consistency_strategies == [
        ConsistencyStrategy.COMPOSITE,
        ConsistencyStrategy.PROMPT_ONLY,
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("composite", ConsistencyStrategy.COMPOSITE),
        ("reference", ConsistencyStrategy.REFERENCE),
        ("adapter", ConsistencyStrategy.ADAPTER),
        ("prompt_only", ConsistencyStrategy.PROMPT_ONLY),
        ("prompt-only", ConsistencyStrategy.PROMPT_ONLY),
        ("reference_images", ConsistencyStrategy.REFERENCE),
        ("trainer", ConsistencyStrategy.ADAPTER),
        (ConsistencyStrategy.REFERENCE, ConsistencyStrategy.REFERENCE),
    ],
)
def test_character_accepts_every_strategy_spelling(
    value: object, expected: ConsistencyStrategy
) -> None:
    character = CharacterConfig.from_mapping({"consistency_strategies": [value]})

    assert character.consistency_strategies == [expected]


def test_character_rejects_unknown_duplicate_and_empty_chains() -> None:
    with pytest.raises(ConfigError) as caught:
        CharacterConfig.from_mapping({"consistency_strategies": ["magic"]})
    assert "consistency_strategies" in str(caught.value)
    assert "magic" in str(caught.value)

    with pytest.raises(ConfigError) as caught:
        CharacterConfig.from_mapping(
            {"consistency_strategies": ["reference", "reference"]}
        )
    assert "must not repeat" in str(caught.value)

    with pytest.raises(ConfigError) as caught:
        CharacterConfig.from_mapping({"consistency_strategies": []})
    assert "must not be empty" in str(caught.value)


def test_character_requires_prompt_only_to_be_the_last_strategy() -> None:
    with pytest.raises(ConfigError) as caught:
        CharacterConfig.from_mapping(
            {"consistency_strategies": ["prompt_only", "reference"]}
        )

    assert "prompt_only" in str(caught.value)
    assert "last" in str(caught.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_score", -0.01),
        ("min_score", 1.01),
        ("vision_threshold", -0.01),
        ("vision_threshold", 1.01),
        ("embedding_threshold", -0.01),
        ("embedding_threshold", 1.01),
        ("min_score", float("inf")),
    ],
)
def test_character_qa_thresholds_stay_between_zero_and_one(
    field: str, value: float
) -> None:
    with pytest.raises(ConfigError) as caught:
        CharacterQAConfig.from_mapping({field: value})

    assert field in str(caught.value)


def test_character_qa_retries_must_not_be_negative() -> None:
    with pytest.raises(ConfigError) as caught:
        CharacterQAConfig.from_mapping({"max_retries": -1})

    assert "max_retries" in str(caught.value)


def test_character_qa_accepts_aliases_and_threshold_alias() -> None:
    qa = CharacterQAConfig.from_mapping(
        {
            "min_consistency": 0.9,
            "vision_similarity": 0.8,
            "embedding_similarity": 0.6,
            "retry_limit": 3,
        }
    )

    assert qa.min_score == 0.9
    assert qa.vision_threshold == 0.8
    assert qa.embedding_threshold == 0.6
    assert qa.max_retries == 3


def test_character_consent_accepts_aliases_and_flags() -> None:
    consent = CharacterConsentConfig.from_mapping(
        {
            "voice_clone": True,
            "likeness": True,
            "resemblance_warnings": False,
            "ownership_required": False,
        }
    )

    assert consent.voice_cloning is True
    assert consent.real_person_likeness is True
    assert consent.resemblance_warning is False
    assert consent.require_ownership is False


def test_character_accepts_consent_flags_directly_or_nested() -> None:
    direct = CharacterConfig.from_mapping({"voice_clone": True, "likeness": True})
    nested = CharacterConfig.from_mapping(
        {"consent": {"voice_cloning": True, "real_person_likeness": True}}
    )

    assert direct.consent == nested.consent
    assert direct.voice_cloning is True
    assert direct.likeness is True


def test_character_max_reference_images_may_be_zero_but_not_negative() -> None:
    assert (
        CharacterConfig.from_mapping({"max_reference_images": 0}).max_reference_images
        == 0
    )

    with pytest.raises(ConfigError) as caught:
        CharacterConfig.from_mapping({"max_reference_images": -1})

    assert "max_reference_images" in str(caught.value)


def test_character_nested_defaults_are_independent_between_instances() -> None:
    first = CharacterConfig.from_mapping({})
    second = CharacterConfig.from_mapping({})

    first.consistency_strategies.append(ConsistencyStrategy.COMPOSITE)
    first.qa.min_score = 0.1
    first.consent.voice_cloning = True

    assert second.consistency_strategies == DEFAULT_CHAIN
    assert second.qa.min_score == 0.8
    assert second.consent.voice_cloning is False


def test_brand_aliases_point_at_the_same_model() -> None:
    assert BrandConfig is BrandKitConfig
    assert BrandKit is BrandKitConfig
    assert BrandingConfig is BrandKitConfig
    assert BrandsConfig is BrandKitConfig


def test_brand_kit_defaults_are_empty_and_independent() -> None:
    first = BrandKitConfig.from_mapping({})
    second = BrandKitConfig.from_mapping({})

    assert first.name == ""
    assert first.colors == []
    assert first.palette == []
    assert first.fonts == []
    assert first.typefaces == []
    assert first.logo == ""
    assert first.watermark == ""
    assert first.intro == ""
    assert first.outro == ""
    assert first.caption_style == ""

    first.colors.append("#112233")
    first.fonts.append("Lato")

    assert second.colors == []
    assert second.fonts == []


def test_brand_kit_parses_every_branding_field() -> None:
    brand = BrandKitConfig.from_mapping(
        {
            "name": " default ",
            "colors": ["#112233", "#abcdef"],
            "fonts": ["Lato", "Noto Sans"],
            "logo": "logo.png",
            "watermark": "watermark.png",
            "intro": "intro.mp4",
            "outro": "outro.mp4",
            "caption_style": "bold-outline",
        }
    )

    assert brand.name == "default"
    assert brand.colors == ["#112233", "#abcdef"]
    assert brand.fonts == ["Lato", "Noto Sans"]
    assert brand.logo == "logo.png"
    assert brand.watermark == "watermark.png"
    assert brand.intro == "intro.mp4"
    assert brand.outro == "outro.mp4"
    assert brand.caption_style == "bold-outline"


def test_brand_kit_accepts_named_colors_and_fonts() -> None:
    brand = BrandKitConfig.from_mapping(
        {
            "colors": {"primary": "#112233", "accent": "#abcdef"},
            "fonts": {"heading": "Lato", "body": "Noto Sans"},
        }
    )

    assert brand.colors == {"primary": "#112233", "accent": "#abcdef"}
    assert brand.fonts == {"heading": "Lato", "body": "Noto Sans"}


def test_brand_kit_accepts_branding_aliases() -> None:
    brand = BrandKitConfig.from_mapping(
        {
            "title": "channel",
            "palette": ["#000000"],
            "typography": ["Lato"],
            "logo_path": "brand/logo.png",
            "watermark_path": "brand/mark.png",
            "intro_clip": "brand/intro.mp4",
            "outro_clip": "brand/outro.mp4",
            "caption": "clean",
        }
    )

    assert brand.name == "channel"
    assert brand.colors == ["#000000"]
    assert brand.fonts == ["Lato"]
    assert brand.logo == "brand/logo.png"
    assert brand.watermark == "brand/mark.png"
    assert brand.intro == "brand/intro.mp4"
    assert brand.outro == "brand/outro.mp4"
    assert brand.caption_style == "clean"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("colors", [""], "colors entries must be non-empty"),
        ("fonts", ["Lato", " "], "fonts entries must be non-empty"),
        ("colors", ["#111111", "#111111"], "duplicate colors"),
        ("fonts", {"": "Lato"}, "fonts names must be non-empty"),
        ("colors", {"primary": ""}, "colors values must be non-empty"),
    ],
)
def test_brand_kit_rejects_empty_or_duplicate_branding_values(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(ConfigError) as caught:
        BrandKitConfig.from_mapping({field: value})

    assert message in str(caught.value)


def test_character_and_brand_are_registered_core_sections() -> None:
    assert {"character", "brand"} <= set(CORE_SECTION_KEYS)

    sections = register_core_sections(SectionRegistry())
    parsed = sections.parse(
        {
            "character": {"consistency_strategies": ["reference", "prompt_only"]},
            "brand": {"caption_style": "clean"},
        }
    )

    assert parsed.character.consistency_strategies == [  # type: ignore[attr-defined]
        ConsistencyStrategy.REFERENCE,
        ConsistencyStrategy.PROMPT_ONLY,
    ]
    assert parsed.brand.caption_style == "clean"  # type: ignore[attr-defined]


def test_character_and_brand_unknown_keys_report_the_full_path() -> None:
    sections = register_core_sections(SectionRegistry())

    with pytest.raises(ConfigError) as caught:
        sections.parse({"character": {"bogus": True}})
    assert "character.bogus" in str(caught.value)

    with pytest.raises(ConfigError) as caught:
        sections.parse({"brand": {"bogus": True}})
    assert "brand.bogus" in str(caught.value)


def test_the_packaged_example_declares_character_and_brand_controls() -> None:
    character = CharacterConfig.from_mapping(example_section("character"))
    brand = BrandKitConfig.from_mapping(example_section("brand"))

    assert character.enabled is True
    assert character.consistency_strategies == DEFAULT_CHAIN
    assert character.qa.min_score == 0.8
    assert character.qa.vision_threshold == 0.8
    assert character.qa.embedding_threshold == 0.75
    assert character.qa.max_retries == 2
    assert character.consent.voice_cloning is False
    assert character.consent.real_person_likeness is False
    assert character.consent.resemblance_warning is True
    assert character.consent.require_ownership is True
    assert character.max_reference_images == 4
    assert brand.name == "default"
    assert brand.colors == []
    assert brand.fonts == []
    assert brand.logo == ""
    assert brand.watermark == ""
    assert brand.intro == ""
    assert brand.outro == ""
    assert brand.caption_style == ""

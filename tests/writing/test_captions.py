"""Captions fit platform and language limits and carry provenance and disclosure."""

from __future__ import annotations

import pytest

from aivia.core.config.sections import LanguageConfig, LanguagesConfig
from aivia.core.errors import ConfigError, ContentPolicyError
from aivia.core.platforms import (
    CaptionLimits,
    DisclosureRules,
    Platform,
    load_platforms,
)
from aivia.writing import (
    DEFAULT_AI_DISCLOSURE,
    Beat,
    CaptionGenerator,
    CaptionSet,
    FactsPayload,
    Script,
    ScriptFact,
    apply_hashtag_style,
    build_captions,
    build_source_line,
    normalise_hashtags,
    normalise_tags,
)

LANGUAGES = LanguagesConfig(
    default="en",
    languages={
        "en": LanguageConfig(
            name="English",
            hashtag_style="lower",
            forbidden_phrases=["in conclusion", "at the end of the day"],
        ),
        "de": LanguageConfig(
            name="German",
            hashtag_style="title",
            forbidden_phrases=["am Ende des Tages"],
        ),
        "tr": LanguageConfig(
            name="Turkish",
            hashtag_style="upper",
            forbidden_phrases=["sonuç olarak"],
        ),
    },
)


def _platform(
    *,
    max_length: int = 5000,
    max_title_length: int | None = 100,
    max_description_length: int | None = 5000,
    max_hashtags: int = 15,
    required: bool = True,
    disclosure: str = "This content was generated with AI.",
) -> Platform:
    return Platform(
        id="example",
        name="Example",
        caption_limits=CaptionLimits(
            max_length=max_length,
            max_title_length=max_title_length,
            max_description_length=max_description_length,
            max_hashtags=max_hashtags,
        ),
        disclosure=DisclosureRules(required=required, text=disclosure),
    )


def _script(
    *,
    title: str = "The secret life of bees",
    language: str = "en",
    hook: str = "Bees can recognise human faces.",
) -> Script:
    facts = [
        ScriptFact(
            text="Honey never spoils.",
            source_urls=["https://example.com/honey"],
        ),
        ScriptFact(
            text="Bees can recognise human faces.",
            source_urls=["https://example.com/faces", "https://example.com/honey"],
        ),
    ]
    return Script(
        title=title,
        language=language,
        hook=hook,
        beats=[
            Beat(index=0, text="Honey never spoils."),
            Beat(index=1, text="Bees can recognise human faces."),
        ],
        outro="That is the story of bees.",
        payload=FactsPayload(
            facts=facts,
            source_urls=[
                "https://example.com/honey",
                "https://example.com/faces",
            ],
        ),
    )


def test_generates_caption_description_tags_and_hashtags() -> None:
    result = build_captions(_script(), _platform())

    assert isinstance(result, CaptionSet)
    assert result.platform == "example"
    assert result.language == "en"
    assert result.title == "The secret life of bees"
    assert "Bees can recognise human faces." in result.caption
    assert "Honey never spoils." in result.description
    assert result.source_line == (
        "Sources: https://example.com/honey, https://example.com/faces"
    )
    assert result.disclosure == "This content was generated with AI."
    assert result.disclosure_required is True
    assert result.hashtags
    assert all(tag.startswith("#") for tag in result.hashtags)
    assert result.tags == [tag.lstrip("#") for tag in result.hashtags]
    assert result.truncated is False
    assert result.warnings == []
    assert result.within_limits is True


def test_source_line_uses_the_script_payload_and_deduplicates_urls() -> None:
    result = build_captions(_script(), _platform())

    assert result.sources == [
        "https://example.com/honey",
        "https://example.com/faces",
    ]
    assert result.source_line.startswith("Sources:")
    assert result.source_line.count("https://example.com/honey") == 1
    assert result.source_line in result.caption
    assert result.source_line in result.description


def test_explicit_sources_replace_the_derived_line() -> None:
    result = build_captions(
        _script(),
        _platform(),
        sources=["https://explicit.example/one", "https://explicit.example/one"],
    )

    assert result.sources == ["https://explicit.example/one"]
    assert result.source_line == "Source: https://explicit.example/one"
    assert "example.com/honey" not in result.source_line


def test_a_script_without_sources_has_no_source_line() -> None:
    script = Script(title="No sources", hook="A short line.")

    result = build_captions(script, _platform())

    assert result.sources == []
    assert result.source_line == ""
    assert "Source:" not in result.caption


def test_caption_description_title_and_hashtags_stay_inside_platform_limits() -> None:
    platform = _platform(
        max_length=80,
        max_title_length=12,
        max_description_length=100,
        max_hashtags=2,
    )

    result = build_captions(_script(), platform)

    assert len(result.title) <= 12
    assert len(result.caption) <= 80
    assert len(result.description) <= 100
    assert len(result.hashtags) <= 2
    assert result.truncated is True
    assert result.within_limits is True
    assert "title truncated" in result.warnings
    assert "caption truncated" in result.warnings
    assert "description truncated" in result.warnings
    assert result.title.endswith("…")


def test_a_missing_description_limit_falls_back_to_the_caption_limit() -> None:
    platform = _platform(
        max_length=60,
        max_description_length=None,
    )

    result = build_captions(_script(), platform)

    assert result.limits["max_description_length"] == 60
    assert len(result.description) <= 60
    assert result.within_limits is True


def test_provenance_survives_a_tight_caption_limit() -> None:
    platform = _platform(
        max_length=60,
        max_title_length=12,
        max_description_length=60,
        max_hashtags=2,
    )

    result = build_captions(_script(), platform)

    assert len(result.caption) <= 60
    assert len(result.description) <= 60
    assert result.caption.endswith(result.disclosure)
    assert result.description.endswith(result.disclosure)
    assert "Sources:" in result.caption
    assert "Sources:" in result.description


@pytest.mark.parametrize(
    ("language", "style", "expected"),
    [
        ("en", "lower", ["#honeybees", "#bees"]),
        ("de", "title", ["#HoneyBees", "#Bees"]),
        ("tr", "upper", ["#HONEYBEES", "#BEES"]),
    ],
)
def test_hashtags_follow_the_language_style(
    language: str,
    style: str,
    expected: list[str],
) -> None:
    result = build_captions(
        _script(language=language),
        _platform(max_hashtags=2),
        languages=LANGUAGES,
        hashtags=["honey bees", "bees", "bees"],
    )

    assert result.language == language
    assert result.hashtags == expected
    assert result.tags == [tag.lstrip("#") for tag in expected]
    assert apply_hashtag_style("honey bees", style) == expected[0].lstrip("#")


def test_the_hashtag_limit_is_applied_to_explicit_and_derived_tags() -> None:
    explicit = build_captions(
        _script(),
        _platform(max_hashtags=1),
        hashtags=["one", "two", "three"],
    )
    derived = build_captions(_script(), _platform(max_hashtags=0))

    assert explicit.hashtags == ["#one"]
    assert explicit.tags == ["one"]
    assert derived.hashtags == []
    assert derived.tags == []


def test_forbidden_phrases_are_removed_from_every_text_field() -> None:
    script = _script(
        title="In conclusion bees",
        hook="In conclusion, bees can recognise human faces.",
    )

    result = build_captions(
        script,
        _platform(),
        languages=LANGUAGES,
        hashtags=["in conclusion", "bees"],
    )

    combined = " ".join(
        [result.title, result.caption, result.description, *result.tags]
    ).casefold()
    assert "in conclusion" not in combined
    assert "forbidden phrases removed from title" in result.warnings
    assert "forbidden phrases removed from caption" in result.warnings
    assert "forbidden phrases removed from description" in result.warnings
    assert result.hashtags == ["#bees"]


def test_strict_generation_refuses_a_forbidden_phrase_that_survives() -> None:
    generator = CaptionGenerator(
        _platform(),
        languages=LANGUAGES,
        disclosure="in conclusion",
    )

    with pytest.raises(ContentPolicyError, match="forbidden phrase") as refused:
        generator.build_strict(_script())

    assert refused.value.details["phrases"] == ["in conclusion"]


def test_disclosure_is_controlled_by_the_platform_and_can_be_overridden() -> None:
    required = build_captions(_script(), _platform())
    not_required = build_captions(
        _script(),
        _platform(required=False, disclosure=""),
    )
    overridden = build_captions(
        _script(),
        _platform(),
        disclosure="AI-assisted video.",
    )
    omitted = build_captions(_script(), _platform(), include_disclosure=False)

    assert required.disclosure == "This content was generated with AI."
    assert required.disclosure in required.caption
    assert not_required.disclosure == ""
    assert not_required.disclosure_required is False
    assert "generated with AI" not in not_required.caption
    assert overridden.disclosure == "AI-assisted video."
    assert omitted.disclosure == ""
    assert "This content was generated with AI." not in omitted.caption


def test_a_required_platform_without_text_uses_the_packaged_fallback() -> None:
    platform = _platform(disclosure="")

    result = build_captions(_script(), platform)

    assert result.disclosure == DEFAULT_AI_DISCLOSURE
    assert DEFAULT_AI_DISCLOSURE in result.caption


def test_the_packaged_platform_data_is_used_when_a_name_is_given() -> None:
    result = build_captions(_script(), "x", sources=["https://example.com/honey"])

    x = load_platforms().require("x")
    assert result.platform == "x"
    assert len(result.caption) <= (x.caption_limits.max_length or 0)
    assert len(result.hashtags) <= (x.caption_limits.max_hashtags or 0)
    assert result.disclosure == x.disclosure.text


def test_an_unknown_platform_fails_with_the_available_names() -> None:
    with pytest.raises(ConfigError, match="unknown platform") as refused:
        build_captions(_script(), "missing")

    assert "x" in refused.value.details["available"]


def test_generation_is_deterministic_and_does_not_mutate_the_script() -> None:
    script = _script()
    before = script.model_dump(mode="json")

    first = build_captions(script, _platform())
    second = build_captions(script, _platform())

    assert first == second
    assert script.model_dump(mode="json") == before


def test_source_lines_and_tag_normalisation_have_direct_helpers() -> None:
    assert build_source_line([]) == ""
    assert build_source_line(["https://one.example"]) == ("Source: https://one.example")
    assert build_source_line(["https://one.example", "https://two.example"]) == (
        "Sources: https://one.example, https://two.example"
    )
    assert normalise_tags(["  Honey Bees ", "honeybees", "#Bees"]) == [
        "honeybees",
        "bees",
    ]
    assert normalise_hashtags(["Honey Bees"], style="upper") == ["#HONEYBEES"]

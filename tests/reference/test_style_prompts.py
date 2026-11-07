"""A style spec becomes reusable style bible entries and prompt suffixes."""

from __future__ import annotations

from typing import Any

import pytest

from aivia.core.errors import ConfigError
from aivia.core.models.script import Beat, Script
from aivia.reference import (
    ReferenceAnalysis,
    ReferenceSpec,
    StylePromptError,
    StylePrompts,
    StyleSpec,
    build_style_prompts,
    prompt_suffix_from_spec,
    prompt_suffixes_from_spec,
    spec_to_style_bible,
    style_bible_entries_from_spec,
    style_bible_from_spec,
    translate_style_spec,
)
from aivia.writing import build_media_prompts


def _spec() -> StyleSpec:
    return StyleSpec.model_validate(
        {
            "name": "Cinematic short",
            "description": "A calm documentary look with a single payoff.",
            "prompt_suffix": "Natural light, shallow depth of field.",
            "structure": {
                "hook": "Open on a question",
                "beats": ["hook", "explain", "payoff"],
                "shot_types": ["wide", "close-up"],
                "transitions": ["cut", "fade"],
                "narrative_arc": "question to answer",
                "total_duration_seconds": 12.0,
            },
            "pacing": {
                "cuts_per_minute": 24,
                "average_shot_seconds": 2.75,
                "min_shot_seconds": 1.0,
                "max_shot_seconds": 4.0,
                "words_per_second": 2.8,
                "energy": 0.6,
                "style": "measured",
            },
            "palette": {
                "colors": ["#FFF", "#112233", "#112233"],
                "primary": "#ABCDEF",
                "background": "#000",
                "temperature": "warm",
                "saturation": 0.4,
                "contrast": 0.7,
                "scheme": "analogous",
            },
            "typography": {
                "font": "Lato",
                "heading_font": "Lato Black",
                "weight": "bold",
                "size": 48,
                "line_height": 1.2,
                "letter_spacing": 0.5,
                "alignment": "centre",
                "case": "uppercase",
                "color": "#FFF",
                "placement": "lower third",
                "caption_style": "burned-in",
            },
            "music": {
                "genre": "ambient",
                "tempo": 92,
                "moods": ["calm", "hopeful"],
                "key": "C",
                "scale": "minor",
                "instruments": ["piano", "strings"],
                "vocals": False,
                "energy": 0.3,
                "valence": 0.6,
                "loudness_lufs": -16.0,
            },
            "aspect": "9:16",
            "metadata": {
                "style": "editorial minimal",
                "avoid": ["clickbait", "logos"],
            },
        }
    )


def test_a_full_spec_becomes_entries_for_every_learned_section() -> None:
    translation = translate_style_spec(_spec())
    bible = translation.to_bible()

    assert isinstance(translation, StylePrompts)
    assert [entry.key for entry in translation.entries] == [
        "style",
        "structure",
        "pacing",
        "palette",
        "typography",
        "music",
        "aspect",
        "avoid",
    ]
    assert bible.name == "Cinematic short"
    assert bible.description == "A calm documentary look with a single payoff."

    assert (
        bible.require("style").text == "A calm documentary look with a single payoff."
    )
    assert bible.require("structure").text.count("hook -> explain -> payoff") == 1
    assert "12 seconds" in bible.require("structure").text
    assert "24 cuts per minute" in bible.require("pacing").text
    assert "2.8 words per second" in bible.require("pacing").text

    palette = bible.require("palette").text
    assert "#ffffff" in palette
    assert "#112233" in palette
    assert "#abcdef" in palette
    assert "warm" in palette
    assert "40% saturation" in palette
    assert "70% contrast" in palette

    typography = bible.require("typography").text
    assert "Lato" in typography
    assert "weight 700" in typography
    assert "uppercase" in typography
    assert "lower third" in typography
    assert "burned-in captions" in typography

    music = bible.require("music").text
    assert "ambient" in music
    assert "92 BPM" in music
    assert "piano" in music
    assert "instrumental" in music

    aspect = bible.require("aspect").text
    assert "9:16" in aspect
    assert "portrait" in aspect
    assert "clickbait" in bible.require("avoid").text
    assert "logos" in bible.require("avoid").text


def test_prompt_suffixes_are_grouped_by_prompt_kind() -> None:
    suffixes = prompt_suffixes_from_spec(_spec())

    assert "Natural light, shallow depth of field." in suffixes.visual
    assert "Palette #ffffff" in suffixes.visual
    assert "Lato typography" in suffixes.visual
    assert "Portrait 9:16 frame." in suffixes.visual

    assert "Measured pacing" in suffixes.motion
    assert "24 cuts per minute" in suffixes.motion
    assert "ambient" in suffixes.audio.casefold()
    assert "92 BPM" in suffixes.audio
    assert "piano" in suffixes.audio
    assert "One idea per beat" in suffixes.writing
    assert "24 cuts per minute" in suffixes.writing
    assert suffixes.negative == "No clickbait, logos."

    assert suffixes.for_kind("image") == suffixes.visual
    assert suffixes.for_kind("still") == suffixes.visual
    assert suffixes.for_kind("video") == suffixes.motion
    assert suffixes.for_kind("music") == suffixes.audio
    assert suffixes.for_kind("script") == suffixes.writing
    assert suffixes.for_kind("avoid") == suffixes.negative
    assert suffixes["image"] == suffixes.visual
    assert suffixes.as_dict() == {
        "visual": suffixes.visual,
        "motion": suffixes.motion,
        "audio": suffixes.audio,
        "writing": suffixes.writing,
        "negative": suffixes.negative,
    }
    assert bool(suffixes) is True

    with pytest.raises(KeyError, match="unknown prompt suffix kind"):
        suffixes.for_kind("thumbnail")


def test_entries_carry_short_suffixes_and_the_bible_joins_them() -> None:
    translation = translate_style_spec(_spec())
    bible = translation.to_bible()

    assert all(entry.prompt_suffix for entry in bible)
    assert bible.suffix == " ".join(entry.prompt_suffix for entry in bible)
    assert "Natural light, shallow depth of field." in bible.suffix
    assert "Measured pacing" in bible.suffix
    assert "ambient" in bible.suffix.casefold()
    assert "Portrait 9:16 frame." in bible.suffix
    assert "No clickbait, logos." in bible.suffix
    assert translation.prompt_suffix == translation.suffixes.text
    assert translation.prompt_suffixes == translation.suffixes.as_dict()
    assert translation.to_dict()["prompt_suffixes"] == translation.prompt_suffixes


def test_a_partial_spec_only_emits_the_sections_it_knows() -> None:
    spec = StyleSpec.model_validate(
        {"palette": {"colors": ["#123456"], "temperature": "cool"}}
    )

    translation = translate_style_spec(spec)

    assert [entry.key for entry in translation.entries] == ["palette"]
    assert translation.suffixes.visual == "Palette #123456, cool tone."
    assert translation.suffixes.motion == ""
    assert translation.suffixes.audio == ""
    assert translation.suffixes.writing == ""
    assert translation.suffixes.negative == ""
    assert translation.to_bible().name == "style-spec"


def test_an_empty_spec_translates_to_nothing_and_needs_no_bible() -> None:
    translation = translate_style_spec(StyleSpec())

    assert translation.is_empty is True
    assert translation.entries == ()
    assert translation.suffixes.values == ()
    assert translation.prompt_suffix == ""
    assert translation.prompt_suffixes == {}
    assert translation.to_dict()["entries"] == []

    with pytest.raises(StylePromptError, match="no style") as caught:
        translation.to_bible()

    assert isinstance(caught.value, ConfigError)
    assert caught.value.code == "style_prompt"

    with pytest.raises(StylePromptError):
        style_bible_from_spec(StyleSpec())


def test_a_reference_analysis_never_leaks_the_subject_or_detected_text() -> None:
    analysis = ReferenceAnalysis.model_validate(
        {
            "reference_id": "ref-1",
            "subject": "The original lighthouse",
            "detected_text": "the original words",
            "description": "The original lighthouse at dusk, warm and calm.",
            "prompt_suffix": "Warm light on the original lighthouse.",
            "structure": {
                "scenes": [
                    {
                        "description": "The original lighthouse",
                        "text": "the original words",
                    }
                ]
            },
            "palette": {"colors": ["#aabbcc"], "temperature": "warm"},
        }
    )

    translation = translate_style_spec(analysis)
    rendered = " ".join(
        [
            *(entry.text for entry in translation.entries),
            *(entry.prompt_suffix for entry in translation.entries),
            translation.description,
            translation.prompt_suffix,
            translation.to_bible().to_prompt(),
        ]
    ).casefold()

    assert "original lighthouse" not in rendered
    assert "original words" not in rendered
    assert "warm light on" in rendered
    assert "#aabbcc" in rendered


def test_translation_is_deterministic_and_ignores_provider_metadata() -> None:
    spec = _spec()
    spec.metadata.update(
        {
            "provider": "vendor-x",
            "model": "model-y",
            "price": "0.01",
        }
    )

    first = translate_style_spec(spec)
    second = translate_style_spec(spec)

    assert first.to_dict() == second.to_dict()
    rendered = first.to_bible().to_prompt().casefold()
    for forbidden in ("vendor-x", "model-y", "0.01"):
        assert forbidden not in rendered


def test_names_descriptions_and_shortcut_functions_agree() -> None:
    spec = _spec()

    assert spec_to_style_bible(spec, name="House").name == "House"
    assert style_bible_from_spec(spec).name == spec.name
    assert build_style_prompts(spec).to_dict() == translate_style_spec(spec).to_dict()

    custom = translate_style_spec(spec, description="A shorter note.")
    assert custom.to_bible().description == "A shorter note."

    assert [entry.key for entry in style_bible_entries_from_spec(spec)] == [
        "style",
        "structure",
        "pacing",
        "palette",
        "typography",
        "music",
        "aspect",
        "avoid",
    ]
    assert prompt_suffix_from_spec(spec) == translate_style_spec(spec).prompt_suffix


def test_mappings_and_reference_specs_are_accepted() -> None:
    mapping: dict[str, Any] = {
        "name": "mapped",
        "palette": {"colors": ["#abcdef"]},
    }
    assert prompt_suffix_from_spec(mapping) == "Palette #abcdef."

    reference = ReferenceSpec(
        path="examples/reference.png",
        analysis=ReferenceAnalysis.model_validate({"palette": {"colors": ["#123456"]}}),
    )
    assert [entry.key for entry in style_bible_entries_from_spec(reference)] == [
        "palette"
    ]


def test_a_translated_bible_styles_real_media_prompts() -> None:
    bible = style_bible_from_spec(_spec())
    script = Script(
        title="Bees",
        language="en",
        beats=[Beat(index=0, text="A bee lands on a flower.")],
    )

    styled = build_media_prompts(script, style_bible=bible)
    beat = styled.beats[0]

    assert beat.image_prompt is not None
    assert beat.motion_prompt is not None
    assert "#112233" in beat.image_prompt.prompt
    assert "Natural light, shallow depth of field." in beat.image_prompt.prompt
    assert beat.image_prompt.style == "Cinematic short"
    assert "No clickbait, logos." in (beat.image_prompt.negative_prompt or "")
    assert "No clickbait, logos." in (beat.motion_prompt.negative_prompt or "")

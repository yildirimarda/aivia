"""Converting a style spec into a format recipe, style bible and preset."""

from __future__ import annotations

from typing import Any

import pytest

from aivia.core.config.formats import OutputType
from aivia.core.config.sections import PresetConfig
from aivia.core.errors import ConfigError
from aivia.reference import (
    FormatConversion,
    StylePromptError,
    StyleSpec,
    convert_style_spec,
    style_spec_to_format_recipe,
    style_spec_to_preset,
    style_spec_to_style_bible,
)
from aivia.writing.style_bible import StyleBible


def _spec() -> StyleSpec:
    return StyleSpec.model_validate(
        {
            "name": "Cinematic short",
            "description": "A calm documentary look with a single payoff.",
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
                "colors": ["#FFF", "#112233"],
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
        }
    )


def _still_spec() -> StyleSpec:
    return StyleSpec.model_validate(
        {
            "name": "Brand card",
            "palette": {"colors": ["#112233"], "temperature": "cool"},
            "typography": {"font": "Inter", "weight": "bold"},
            "aspect": "1:1",
        }
    )


def test_convert_produces_recipe_bible_and_preset() -> None:
    conversion = convert_style_spec(_spec())

    assert isinstance(conversion, FormatConversion)
    assert isinstance(conversion.recipe, object)
    assert isinstance(conversion.bible, StyleBible)
    assert isinstance(conversion.preset, PresetConfig)

    assert conversion.recipe.output_type is OutputType.VIDEO
    assert conversion.recipe.handler.value == "save"
    assert [stage.stage for stage in conversion.recipe.stages] == [
        "write",
        "visuals",
        "subtitles",
        "music",
        "voice",
    ]
    assert conversion.recipe.capabilities == [
        "writing",
        "image",
        "captions",
        "music",
        "tts",
    ]

    assert conversion.bible.name == "Cinematic short"
    assert (
        conversion.bible.description == "A calm documentary look with a single payoff."
    )
    assert [entry.key for entry in conversion.bible.entries] == [
        "style",
        "structure",
        "pacing",
        "palette",
        "typography",
        "music",
        "aspect",
    ]

    assert conversion.preset.name == "Cinematic short"
    assert conversion.preset.overrides["video"]["aspect_ratio"] == "9:16"
    assert conversion.preset.overrides["subtitle"]["font"] == "Lato"
    assert conversion.preset.overrides["music"]["genre"] == "ambient"
    assert conversion.preset.overrides["voice"]["words_per_second"] == 2.8


def test_format_recipe_stages_derive_options_from_each_learned_section() -> None:
    recipe = style_spec_to_format_recipe(_spec())

    write = recipe.stages[0]
    assert write.stage == "write"
    assert write.options["beats"] == ["hook", "explain", "payoff"]
    assert write.options["hook"] == "Open on a question"
    assert write.options["narrative_arc"] == "question to answer"
    assert write.options["total_duration_seconds"] == 12.0

    visuals = recipe.stages[1]
    assert visuals.stage == "visuals"
    assert visuals.options["palette"] == ["#ffffff", "#112233"]
    assert visuals.options["primary"] == "#abcdef"
    assert visuals.options["background"] == "#000000"
    assert visuals.options["temperature"] == "warm"
    assert visuals.options["saturation"] == 0.4
    assert visuals.options["contrast"] == 0.7
    assert visuals.options["scheme"] == "analogous"

    subtitles = recipe.stages[2]
    assert subtitles.stage == "subtitles"
    assert subtitles.options["font"] == ["Lato", "Lato Black"]
    assert subtitles.options["font_weight"] == 700
    assert subtitles.options["font_size"] == 48.0
    assert subtitles.options["align"] == "center"
    assert subtitles.options["case"] == "upper"
    assert subtitles.options["placement"] == "lower third"
    assert subtitles.options["caption_style"] == "burned_in"

    music_stage = recipe.stages[3]
    assert music_stage.stage == "music"
    assert music_stage.options["genre"] == "ambient"
    assert music_stage.options["moods"] == ["calm", "hopeful"]
    assert music_stage.options["bpm"] == 92.0
    assert music_stage.options["instruments"] == ["piano", "strings"]

    voice = recipe.stages[4]
    assert voice.stage == "voice"
    assert voice.options["words_per_second"] == 2.8
    assert voice.options["energy"] == 0.6
    assert voice.options["style"] == "measured"


def test_still_spec_becomes_an_image_recipe_with_only_learned_stages() -> None:
    recipe = style_spec_to_format_recipe(_still_spec())

    assert recipe.output_type is OutputType.IMAGE
    assert recipe.handler.value == "save"
    assert [stage.stage for stage in recipe.stages] == ["visuals", "subtitles"]
    assert recipe.capabilities == ["image", "captions"]

    assert recipe.stages[0].options["palette"] == ["#112233"]
    assert recipe.stages[1].options["font"] == ["Inter"]


def test_empty_spec_raises() -> None:
    with pytest.raises(StylePromptError, match="no style to convert") as caught:
        convert_style_spec(StyleSpec())

    assert isinstance(caught.value, ConfigError)
    assert caught.value.code == "style_prompt"


def test_partial_spec_still_converts() -> None:
    spec = StyleSpec.model_validate({"palette": {"colors": ["#123456"]}})

    conversion = convert_style_spec(spec)

    assert conversion.recipe.output_type is OutputType.IMAGE
    assert [stage.stage for stage in conversion.recipe.stages] == ["visuals"]
    assert conversion.bible.name == "style-spec"
    assert [entry.key for entry in conversion.bible.entries] == ["palette"]
    assert conversion.preset.name == "style-spec"
    assert conversion.preset.overrides == {}


def test_name_and_description_override_the_spec() -> None:
    conversion = convert_style_spec(
        _spec(), name="House", description="A shorter note."
    )

    assert conversion.bible.name == "House"
    assert conversion.bible.description == "A shorter note."
    assert conversion.preset.name == "House"


def test_mapping_input_is_accepted() -> None:
    mapping: dict[str, Any] = {
        "name": "mapped",
        "palette": {"colors": ["#abcdef"]},
        "aspect": "9:16",
    }
    conversion = convert_style_spec(mapping)

    assert conversion.bible.name == "mapped"
    assert conversion.recipe.output_type is OutputType.IMAGE
    assert [stage.stage for stage in conversion.recipe.stages] == ["visuals"]
    assert conversion.preset.overrides["video"]["aspect_ratio"] == "9:16"


def test_conversion_is_deterministic() -> None:
    first = convert_style_spec(_spec())
    second = convert_style_spec(_spec())

    assert first.recipe == second.recipe
    assert first.bible == second.bible
    assert first.preset == second.preset
    assert first.recipe.model_dump_json() == second.recipe.model_dump_json()
    assert first.preset.model_dump_json() == second.preset.model_dump_json()


def test_individual_converters_match_the_combined_conversion() -> None:
    spec = _spec()

    recipe = style_spec_to_format_recipe(spec)
    bible = style_spec_to_style_bible(spec)
    preset = style_spec_to_preset(spec)

    combined = convert_style_spec(spec)

    assert combined.recipe == recipe
    assert combined.bible == bible
    assert combined.preset == preset


def test_recipe_rejects_duplicate_stages_and_invalid_output() -> None:
    recipe = style_spec_to_format_recipe(_spec())
    assert len({stage.stage for stage in recipe.stages}) == len(recipe.stages)


def test_style_bible_from_spec_raises_on_empty() -> None:
    with pytest.raises(StylePromptError):
        style_spec_to_style_bible(StyleSpec())

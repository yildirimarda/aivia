"""Format recipe configuration: ordered stages, options and output handling."""

from __future__ import annotations

from typing import Any, cast

import pytest
import yaml

from aivia.core import paths
from aivia.core.config import (
    CORE_SECTION_KEYS,
    FormatConfig,
    FormatHandler,
    FormatOutputType,
    FormatRecipeConfig,
    FormatsConfig,
    FormatStage,
    OutputHandler,
    OutputType,
    RecipeConfig,
    SectionRegistry,
    Stage,
    StageConfig,
    register_core_sections,
)
from aivia.core.errors import ConfigError
from aivia.core.stage_retry import StageRetryPolicy


def example_section(name: str) -> dict[str, Any]:
    text = (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )
    payload = cast(dict[str, Any], yaml.safe_load(text))
    return cast(dict[str, Any], payload[name])


def test_format_recipe_defaults_to_a_saved_video_with_no_stages() -> None:
    recipe = FormatRecipeConfig.from_mapping({})

    assert recipe.stages == []
    assert recipe.capabilities == []
    assert recipe.output_type == OutputType.VIDEO
    assert recipe.handler == OutputHandler.SAVE
    assert recipe.description == ""


def test_recipe_keeps_stage_order_and_per_stage_options() -> None:
    recipe = FormatRecipeConfig.from_mapping(
        {
            "description": "A narrated still",
            "stages": [
                {"stage": "input_topic"},
                {"stage": "write", "options": {"mode": "facts", "tone": "warm"}},
                {"stage": "voice"},
                {"stage": "render", "options": {"subtitles": True}},
            ],
            "capabilities": ["llm", "tts"],
            "output_type": "video",
            "handler": "publish",
        }
    )

    assert [item.stage for item in recipe.stages] == [
        "input_topic",
        "write",
        "voice",
        "render",
    ]
    assert recipe.stages[1].options == {"mode": "facts", "tone": "warm"}
    assert recipe.stages[3].options == {"subtitles": True}
    assert recipe.capabilities == ["llm", "tts"]
    assert recipe.output_type is OutputType.VIDEO
    assert recipe.handler is OutputHandler.PUBLISH


def test_bare_strings_and_single_key_shorthand_build_stages() -> None:
    recipe = FormatRecipeConfig.from_mapping(
        {"stages": ["input_topic", {"write": {"mode": "quote"}}, {"render": None}]}
    )

    assert [item.stage for item in recipe.stages] == [
        "input_topic",
        "write",
        "render",
    ]
    assert recipe.stages[1].options == {"mode": "quote"}
    assert recipe.stages[2].options == {}


@pytest.mark.parametrize(
    "entry",
    [
        {"name": "write", "params": {"mode": "facts"}},
        {"id": "write", "config": {"mode": "facts"}},
        {"step": "write", "settings": {"mode": "facts"}},
    ],
)
def test_stage_accepts_name_id_step_and_option_aliases(
    entry: dict[str, Any],
) -> None:
    stage = StageConfig.from_mapping(entry)

    assert stage.stage == "write"
    assert stage.name == "write"
    assert stage.id == "write"
    assert stage.options == {"mode": "facts"}
    assert stage.params == {"mode": "facts"}


def test_stage_accepts_timeout_and_retry_policy_overrides() -> None:
    stage = StageConfig.from_mapping(
        {
            "stage": "render",
            "timeout_seconds": 12.5,
            "retry_policy": {
                "max_attempts": 3,
                "retry_delay_seconds": 0.5,
                "retry_codes": ["rate_limit"],
            },
        }
    )

    assert stage.timeout_seconds == 12.5
    assert stage.retry_policy == StageRetryPolicy(
        max_attempts=3,
        retry_delay_seconds=0.5,
        retry_codes=("rate_limit",),
    )
    assert stage.retry_overrides == {
        "timeout_seconds": 12.5,
        "retry_policy": stage.retry_policy,
    }


def test_stage_accepts_flat_retry_knobs_and_aliases() -> None:
    stage = StageConfig.from_mapping(
        {
            "stage": "render",
            "timeout": 5,
            "max_attempts": 4,
            "retry_delay_seconds": 0.25,
            "backoff_multiplier": 2,
            "max_retry_delay_seconds": 10,
            "retry_on_timeout": False,
            "retry_codes": ["rate_limit"],
        }
    )

    assert stage.timeout_seconds == 5
    assert stage.retry_overrides == {
        "timeout_seconds": 5,
        "max_attempts": 4,
        "retry_delay_seconds": 0.25,
        "backoff_multiplier": 2,
        "max_retry_delay_seconds": 10,
        "retry_on_timeout": False,
        "retry_codes": ("rate_limit",),
    }


def test_stage_without_retry_overrides_keeps_its_class_policy() -> None:
    stage = StageConfig.from_mapping({"stage": "render", "options": {"fast": True}})

    assert stage.retry_overrides == {}
    assert stage.to_stage_kwargs() == {"options": {"fast": True}}


def test_recipe_stage_retry_overrides_reach_the_stage_instance() -> None:
    from aivia.core.orchestrator import _coerce_stage

    recipe = FormatRecipeConfig.from_mapping(
        {
            "stages": [
                {
                    "stage": "render",
                    "timeout_seconds": 9,
                    "retry_policy": {"max_attempts": 2},
                }
            ]
        }
    )

    stage = _coerce_stage(recipe.stages[0])

    assert stage.retry_policy == StageRetryPolicy(
        timeout_seconds=9,
        max_attempts=2,
    )


@pytest.mark.parametrize(
    ("values", "field"),
    [
        ({"timeout_seconds": 0}, "timeout_seconds"),
        ({"timeout_seconds": float("inf")}, "timeout_seconds"),
        ({"max_attempts": 0}, "max_attempts"),
        ({"retry_delay_seconds": -1}, "retry_delay_seconds"),
        ({"backoff_multiplier": 0.5}, "backoff_multiplier"),
        ({"max_retry_delay_seconds": -1}, "max_retry_delay_seconds"),
        ({"retry_policy": {"max_attempts": 0}}, "retry_policy"),
    ],
)
def test_invalid_stage_retry_overrides_report_the_field(
    values: dict[str, Any], field: str
) -> None:
    with pytest.raises(ConfigError) as caught:
        StageConfig.from_mapping({"stage": "render", **values})

    assert field in str(caught.value)


def test_recipe_accepts_output_and_handler_aliases() -> None:
    recipe = FormatRecipeConfig.from_mapping(
        {
            "steps": ["input_topic"],
            "requires": ["llm"],
            "output": "audio",
            "output_handler": "deliver",
        }
    )

    assert recipe.stages[0].stage == "input_topic"
    assert recipe.capabilities == ["llm"]
    assert recipe.output_type is OutputType.AUDIO
    assert recipe.output is OutputType.AUDIO
    assert recipe.handler is OutputHandler.DELIVER
    assert recipe.output_handler is OutputHandler.DELIVER


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("video", OutputType.VIDEO),
        ("audio", OutputType.AUDIO),
        ("image", OutputType.IMAGE),
        ("carousel", OutputType.CAROUSEL),
        ("web_card", OutputType.WEB_CARD),
        (OutputType.IMAGE, OutputType.IMAGE),
    ],
)
def test_every_output_type_is_accepted(
    value: str | OutputType, expected: OutputType
) -> None:
    recipe = FormatRecipeConfig.from_mapping({"output_type": value})

    assert recipe.output_type == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("save", OutputHandler.SAVE),
        ("publish", OutputHandler.PUBLISH),
        ("deliver", OutputHandler.DELIVER),
        (OutputHandler.PUBLISH, OutputHandler.PUBLISH),
    ],
)
def test_every_output_handler_is_accepted(
    value: str | OutputHandler, expected: OutputHandler
) -> None:
    recipe = FormatRecipeConfig.from_mapping({"handler": value})

    assert recipe.handler == expected


@pytest.mark.parametrize(
    ("values", "field"),
    [
        ({"output_type": "hologram"}, "output_type"),
        ({"handler": "archive"}, "handler"),
    ],
)
def test_unknown_output_type_or_handler_names_the_field(
    values: dict[str, Any], field: str
) -> None:
    with pytest.raises(ConfigError) as caught:
        FormatRecipeConfig.from_mapping(values)

    assert field in str(caught.value)


def test_duplicate_stages_are_rejected_and_name_the_stage() -> None:
    with pytest.raises(ConfigError) as caught:
        FormatRecipeConfig.from_mapping(
            {"stages": [{"stage": "write"}, {"stage": "write"}]}
        )

    assert "duplicate stage in recipe: write" in str(caught.value)


@pytest.mark.parametrize("name", ["", "   "])
def test_stage_names_must_not_be_blank(name: str) -> None:
    with pytest.raises(ConfigError) as caught:
        StageConfig.from_mapping({"stage": name})

    assert "stage" in str(caught.value)


def test_a_stage_requires_a_name() -> None:
    with pytest.raises(ConfigError) as caught:
        StageConfig.from_mapping({"options": {"mode": "facts"}})

    assert "stage" in str(caught.value)


def test_stage_names_are_stripped() -> None:
    stage = StageConfig.from_mapping({"stage": "  write  "})

    assert stage.stage == "write"


def test_unknown_stage_keys_report_the_full_recipe_path() -> None:
    with pytest.raises(ConfigError) as caught:
        FormatRecipeConfig.from_mapping({"stages": [{"stage": "write", "bogus": True}]})

    assert "stages.0.bogus" in str(caught.value)


def test_capabilities_are_stripped_and_duplicates_are_rejected() -> None:
    recipe = FormatRecipeConfig.from_mapping({"capabilities": [" llm ", "tts"]})

    assert recipe.capabilities == ["llm", "tts"]

    with pytest.raises(ConfigError) as caught:
        FormatRecipeConfig.from_mapping({"capabilities": ["llm", " llm "]})

    assert "duplicate capability: llm" in str(caught.value)


@pytest.mark.parametrize("entry", ["", "   "])
def test_capabilities_must_not_be_blank(entry: str) -> None:
    with pytest.raises(ConfigError) as caught:
        FormatRecipeConfig.from_mapping({"capabilities": [entry]})

    assert "capabilities" in str(caught.value)


def test_formats_accept_direct_and_nested_mappings() -> None:
    recipe = {"stages": ["input_topic", "render"], "output_type": "image"}
    direct = FormatsConfig.from_mapping({"quote_card": recipe})
    nested = FormatsConfig.from_mapping({"formats": {"quote_card": recipe}})
    recipes = FormatsConfig.from_mapping({"recipes": {"quote_card": recipe}})

    assert direct.formats["quote_card"].output_type is OutputType.IMAGE
    assert nested.formats["quote_card"].stages[0].stage == "input_topic"
    assert recipes.formats["quote_card"].stages[-1].stage == "render"


def test_format_names_are_stripped_and_blanks_are_rejected() -> None:
    config = FormatsConfig.from_mapping({" quote_card ": {"stages": ["render"]}})

    assert config.formats["quote_card"].stages[0].stage == "render"

    with pytest.raises(ConfigError) as caught:
        FormatsConfig.from_mapping({" ": {"stages": ["render"]}})

    assert "format names must be non-empty" in str(caught.value)


def test_duplicate_format_names_after_stripping_are_rejected() -> None:
    with pytest.raises(ConfigError) as caught:
        FormatsConfig.from_mapping(
            {
                "quote_card": {"stages": ["render"]},
                " quote_card ": {"stages": ["render"]},
            }
        )

    assert "duplicate format: quote_card" in str(caught.value)


def test_unknown_recipe_keys_report_the_format_and_recipe_path() -> None:
    with pytest.raises(ConfigError) as caught:
        FormatsConfig.from_mapping({"quote_card": {"bogus": True}})

    assert "formats.quote_card.bogus" in str(caught.value)


def test_stage_options_are_interpolated_from_the_environment() -> None:
    recipe = FormatRecipeConfig.from_mapping(
        {"stages": [{"stage": "write", "options": {"tone": "${AIVA_TONE}"}}]},
        environ={"AIVA_TONE": "warm"},
    )

    assert recipe.stages[0].options == {"tone": "warm"}


def test_defaults_are_independent_between_recipes_and_stages() -> None:
    first = FormatRecipeConfig.from_mapping({"stages": ["write"]})
    second = FormatRecipeConfig.from_mapping({"stages": ["write"]})

    first.capabilities.append("llm")
    first.stages[0].options["mode"] = "facts"

    assert second.capabilities == []
    assert second.stages[0].options == {}


def test_format_recipe_aliases_point_at_the_same_models() -> None:
    assert FormatConfig is FormatRecipeConfig
    assert RecipeConfig is FormatRecipeConfig
    assert FormatStage is StageConfig
    assert Stage is StageConfig
    assert FormatOutputType is OutputType
    assert FormatHandler is OutputHandler


def test_formats_section_is_registered_and_parsed() -> None:
    sections = register_core_sections(SectionRegistry())

    parsed = sections.parse(
        {
            "formats": {
                "quote_card": {
                    "stages": ["input_topic", "render"],
                    "output_type": "image",
                    "handler": "save",
                }
            }
        }
    )
    formats = cast(FormatsConfig, cast(Any, parsed).formats)

    assert "formats" in CORE_SECTION_KEYS
    assert formats.formats["quote_card"].output_type is OutputType.IMAGE


def test_the_packaged_example_declares_valid_format_recipes() -> None:
    formats = FormatsConfig.from_mapping(example_section("formats"))

    assert {"narrated_still", "song_video", "character_short"} <= set(formats.formats)
    assert formats.formats["narrated_still"].stages[0].stage == "input_topic"
    assert formats.formats["personalized_song_card"].handler is OutputHandler.DELIVER

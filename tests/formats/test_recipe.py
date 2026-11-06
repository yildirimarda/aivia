"""Recipe validation: every stage input comes from an earlier stage."""

from __future__ import annotations

from typing import Any

import pytest

from aivia.core.config.formats import FormatRecipeConfig
from aivia.core.errors import ConfigError
from aivia.formats import (
    RecipeIssues,
    RecipeValidationError,
    StageRegistry,
    check_recipe,
    validate_recipe,
)


class Topic:
    """A stand-in for the topic a recipe starts from."""


class Script:
    """A stand-in for a written script."""


class Audio:
    """A stand-in for a rendered voice track."""


class Video:
    """A stand-in for a rendered video."""


def stage_registry() -> StageRegistry:
    stages = StageRegistry()
    stages.register("input_topic", inputs=(), outputs=(Topic,))
    stages.register("write", inputs=(Topic,), outputs=(Script,))
    stages.register("voice", inputs=(Script,), outputs=(Audio,))
    stages.register("render", inputs=(Script, Audio), outputs=(Video,))
    return stages


def test_a_valid_chain_returns_the_parsed_recipe() -> None:
    recipe = validate_recipe(
        {"stages": ["input_topic", "write", "voice", "render"]},
        stage_registry(),
    )

    assert isinstance(recipe, FormatRecipeConfig)
    assert [stage.stage for stage in recipe.stages] == [
        "input_topic",
        "write",
        "voice",
        "render",
    ]


def test_a_stage_can_consume_an_output_from_an_earlier_stage() -> None:
    recipe = validate_recipe(
        ["write", "voice"],
        stage_registry(),
        inputs=(Topic,),
    )

    assert [stage.stage for stage in recipe.stages] == ["write", "voice"]


def test_caller_supplied_inputs_are_available_before_the_first_stage() -> None:
    stages = StageRegistry()
    stages.register("render", inputs=("timeline",), outputs=("video.mp4",))

    recipe = validate_recipe(["render"], stages, inputs=("timeline",))

    assert recipe.stages[0].stage == "render"


def test_an_out_of_order_input_names_the_offending_stage() -> None:
    with pytest.raises(RecipeValidationError) as caught:
        validate_recipe(
            ["voice", "write"],
            stage_registry(),
            inputs=(Topic,),
        )

    error = caught.value
    assert "stage 'voice'" in str(error)
    assert "'Script'" in str(error)
    assert error.details["stage"] == "voice"
    assert error.details["input"] == "Script"
    assert error.details["missing"] == ["Script"]
    assert error.details["errors"][0]["index"] == 0


def test_an_input_from_a_later_stage_does_not_satisfy_an_earlier_one() -> None:
    with pytest.raises(RecipeValidationError) as caught:
        validate_recipe(
            ["voice", "write"],
            stage_registry(),
            inputs=(Topic,),
        )

    assert caught.value.details["stage"] == "voice"
    assert caught.value.details["input"] == "Script"


def test_a_stage_cannot_consume_its_own_output() -> None:
    stages = StageRegistry()
    stages.register("loop", inputs=("loop_out",), outputs=("loop_out",))

    with pytest.raises(RecipeValidationError) as caught:
        validate_recipe(["loop"], stages)

    assert caught.value.details["stage"] == "loop"
    assert caught.value.details["input"] == "loop_out"


def test_an_unknown_stage_names_the_stage_and_the_registered_ones() -> None:
    with pytest.raises(ConfigError) as caught:
        validate_recipe(["missing"], stage_registry())

    error = caught.value
    assert "unknown stage: missing" in str(error)
    assert "input_topic" in str(error)
    assert error.details["stage"] == "missing"
    assert "write" in error.details["available"]


def test_check_recipe_returns_every_issue_in_stage_order() -> None:
    stages = StageRegistry()
    stages.register("first", inputs=("alpha",), outputs=("beta",))
    stages.register("second", inputs=("gamma",), outputs=("delta",))

    issues = check_recipe(["first", "second"], stages)

    assert isinstance(issues, RecipeIssues)
    assert [issue.stage for issue in issues] == ["first", "second"]
    assert [issue.input for issue in issues] == ["alpha", "gamma"]
    assert issues.ok is False
    assert issues.errors == list(issues)


def test_validate_reports_all_issues_at_once() -> None:
    stages = StageRegistry()
    stages.register("first", inputs=("alpha",), outputs=("beta",))
    stages.register("second", inputs=("gamma",), outputs=("delta",))

    with pytest.raises(RecipeValidationError) as caught:
        validate_recipe(["first", "second"], stages)

    details = caught.value.details["errors"]
    assert [item["stage"] for item in details] == ["first", "second"]


def test_a_stage_missing_several_inputs_lists_them_all() -> None:
    stages = StageRegistry()
    stages.register("render", inputs=("script", "audio"), outputs=("video",))

    issues = check_recipe(["render"], stages)

    assert issues[0].stage == "render"
    assert issues[0].missing == ("script", "audio")
    assert "script, audio" in issues[0].message


def test_an_earlier_output_can_feed_several_later_stages() -> None:
    stages = StageRegistry()
    stages.register("write", inputs=("topic",), outputs=("script",))
    stages.register("subtitles", inputs=("script",), outputs=("captions",))
    stages.register("voice", inputs=("script",), outputs=("audio",))

    recipe = validate_recipe(
        ["write", "subtitles", "voice"],
        stages,
        inputs=("topic",),
    )

    assert [stage.stage for stage in recipe.stages] == [
        "write",
        "subtitles",
        "voice",
    ]


def test_a_recipe_with_no_stages_is_valid() -> None:
    assert validate_recipe({}, stage_registry()).stages == []


def test_stage_options_do_not_affect_input_validation() -> None:
    recipe = validate_recipe(
        [{"stage": "write", "options": {"mode": "facts"}}, "voice"],
        stage_registry(),
        inputs=(Topic,),
    )

    assert recipe.stages[0].options == {"mode": "facts"}


def test_malformed_recipe_values_are_rejected() -> None:
    with pytest.raises(TypeError, match="mapping or a sequence"):
        validate_recipe(42, stage_registry())  # type: ignore[arg-type]

    with pytest.raises(ConfigError):
        validate_recipe(
            {"stages": [{"stage": "write", "bogus": True}]},
            stage_registry(),
        )


def test_initial_inputs_reject_malformed_contracts() -> None:
    with pytest.raises(TypeError, match="contract names or types"):
        validate_recipe(
            ["write"],
            stage_registry(),
            inputs=(42,),  # type: ignore[arg-type]
        )


def test_recipe_issue_dict_is_json_compatible() -> None:
    issues = check_recipe(["voice"], stage_registry())
    payload: dict[str, Any] = issues[0].to_dict()

    assert payload["stage"] == "voice"
    assert payload["code"] == "missing_input"
    assert payload["input"] == "Script"
    assert payload["available"] == []

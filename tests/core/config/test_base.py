"""Configuration base model contract: interpolation, defaults and key paths."""

from __future__ import annotations

import pytest
from pydantic import Field

from aivia.core.config import ConfigModel, dotted_path, interpolate_env
from aivia.core.errors import ConfigError


class PipelineConfig(ConfigModel):
    stages: list[str] = Field(default_factory=lambda: ["research", "render"])
    timezone: str = "UTC"
    retries: int = 2


class AppConfig(ConfigModel):
    schema_version: int = 1
    name: str = "untitled"
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)


def test_environment_references_are_interpolated_recursively() -> None:
    config = AppConfig.from_mapping(
        {
            "name": "${AIVA_NAME}",
            "pipeline": {
                "timezone": "${AIVA_TIMEZONE}",
                "stages": ["${AIVA_STAGE}", "render"],
            },
        },
        environ={
            "AIVA_NAME": "daily",
            "AIVA_TIMEZONE": "Europe/Berlin",
            "AIVA_STAGE": "research",
        },
    )

    assert config.name == "daily"
    assert config.pipeline.timezone == "Europe/Berlin"
    assert config.pipeline.stages == ["research", "render"]


def test_interpolation_leaves_non_strings_and_missing_references_untouched() -> None:
    value = {
        "number": 3,
        "flag": True,
        "nothing": None,
        "text": "plain",
        "items": [1, "two"],
    }

    assert interpolate_env(value, environ={}) == value


def test_default_is_used_when_the_variable_is_missing() -> None:
    config = AppConfig.from_mapping(
        {
            "name": "${AIVA_NAME:-fallback}",
            "pipeline": {"timezone": "${AIVA_TIMEZONE-fallback}"},
        },
        environ={},
    )

    assert config.name == "fallback"
    assert config.pipeline.timezone == "fallback"


def test_default_does_not_replace_a_present_value() -> None:
    config = AppConfig.from_mapping(
        {"name": "${AIVA_NAME:-fallback}"},
        environ={"AIVA_NAME": "configured"},
    )

    assert config.name == "configured"


def test_colon_default_replaces_an_empty_value_but_plain_default_does_not() -> None:
    config = AppConfig.from_mapping(
        {
            "name": "${AIVA_NAME:-fallback}",
            "pipeline": {"timezone": "${AIVA_TIMEZONE-fallback}"},
        },
        environ={"AIVA_NAME": "", "AIVA_TIMEZONE": ""},
    )

    assert config.name == "fallback"
    assert config.pipeline.timezone == ""


def test_missing_variable_without_default_names_the_variable_and_key_path() -> None:
    with pytest.raises(ConfigError) as caught:
        AppConfig.from_mapping(
            {"pipeline": {"timezone": "${AIVA_TIMEZONE}"}},
            environ={},
        )

    message = str(caught.value)
    assert "AIVA_TIMEZONE" in message
    assert "pipeline.timezone" in message
    assert caught.value.details["path"] == "pipeline.timezone"


def test_missing_variable_in_a_list_names_the_indexed_key_path() -> None:
    with pytest.raises(ConfigError) as caught:
        AppConfig.from_mapping(
            {"pipeline": {"stages": ["research", "${AIVA_STAGE}"]}},
            environ={},
        )

    assert "pipeline.stages.1" in str(caught.value)


def test_a_double_dollar_escapes_interpolation() -> None:
    config = AppConfig.from_mapping(
        {"name": "$${AIVA_NAME}", "pipeline": {"timezone": "costs $$5"}},
        environ={"AIVA_NAME": "ignored"},
    )

    assert config.name == "${AIVA_NAME}"
    assert config.pipeline.timezone == "costs $5"


def test_interpolation_does_not_mutate_the_input_mapping() -> None:
    values = {"pipeline": {"timezone": "${AIVA_TIMEZONE}"}}

    AppConfig.from_mapping(values, environ={"AIVA_TIMEZONE": "UTC"})

    assert values == {"pipeline": {"timezone": "${AIVA_TIMEZONE}"}}


def test_model_defaults_are_applied_when_values_are_omitted() -> None:
    config = AppConfig.from_mapping({})

    assert config.schema_version == 1
    assert config.name == "untitled"
    assert config.pipeline.stages == ["research", "render"]
    assert config.pipeline.timezone == "UTC"
    assert config.pipeline.retries == 2


def test_default_containers_are_not_shared_between_instances() -> None:
    first = AppConfig.from_mapping({})
    second = AppConfig.from_mapping({})

    first.pipeline.stages.append("publish")

    assert second.pipeline.stages == ["research", "render"]


def test_unknown_top_level_key_is_rejected_with_its_dotted_path() -> None:
    with pytest.raises(ConfigError) as caught:
        AppConfig.from_mapping({"unknown": True})

    assert "unknown config key: unknown" in str(caught.value)


def test_unknown_nested_key_is_rejected_with_its_dotted_path() -> None:
    with pytest.raises(ConfigError) as caught:
        AppConfig.from_mapping({"pipeline": {"bogus": 1}})

    message = str(caught.value)
    assert "unknown config key: pipeline.bogus" in message
    assert caught.value.details["errors"][0]["path"] == "pipeline.bogus"


def test_missing_required_key_is_reported_with_its_dotted_path() -> None:
    class RequiredConfig(ConfigModel):
        name: str

    with pytest.raises(ConfigError) as caught:
        RequiredConfig.from_mapping({})

    assert "missing required config key: name" in str(caught.value)


def test_invalid_value_is_reported_with_its_dotted_path() -> None:
    with pytest.raises(ConfigError) as caught:
        AppConfig.from_mapping({"pipeline": {"retries": "many"}})

    message = str(caught.value)
    assert "invalid value for pipeline.retries" in message


def test_environment_interpolation_happens_before_type_coercion() -> None:
    config = AppConfig.from_mapping(
        {"pipeline": {"retries": "${AIVA_RETRIES}"}},
        environ={"AIVA_RETRIES": "5"},
    )

    assert config.pipeline.retries == 5


def test_direct_construction_interpolates_the_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AIVA_NAME", "from-process")

    config = AppConfig(name="${AIVA_NAME}")

    assert config.name == "from-process"


def test_invalid_reference_names_the_key_path() -> None:
    with pytest.raises(ConfigError) as caught:
        AppConfig.from_mapping({"name": "${1BAD}"}, environ={})

    assert "invalid environment reference" in str(caught.value)
    assert "name" in str(caught.value)


def test_dotted_path_renders_strings_and_indices() -> None:
    assert dotted_path(("pipeline", "stages", 0)) == "pipeline.stages.0"
    assert dotted_path(()) == ""

"""Section registry contract: ownership, collisions and unknown keys."""

from __future__ import annotations

from typing import Any

import pytest

from aivia.core.config import ConfigModel, SectionRegistry
from aivia.core.errors import ConfigError


class PipelineConfig(ConfigModel):
    timezone: str = "UTC"
    retries: int = 2


class RenderConfig(ConfigModel):
    min_contrast: float = 4.5


class ResearchConfig(ConfigModel):
    max_sources: int = 5


def test_registered_sections_are_listed_in_sorted_order() -> None:
    sections = SectionRegistry()
    sections.register("render", RenderConfig)
    sections.register("pipeline", PipelineConfig)

    assert sections.keys() == ("pipeline", "render")
    assert sections.models() == {
        "pipeline": PipelineConfig,
        "render": RenderConfig,
    }
    assert [section.key for section in sections] == ["pipeline", "render"]
    assert "pipeline" in sections
    assert "missing" not in sections
    assert len(sections) == 2


def test_a_model_can_register_itself_with_the_decorator() -> None:
    sections = SectionRegistry()

    @sections.section("research", module="research")
    class RegisteredResearch(ResearchConfig):
        pass

    section = sections.get("research")

    assert section.model is RegisteredResearch
    assert section.module == "research"


def test_register_section_is_an_explicit_alias() -> None:
    sections = SectionRegistry()

    section = sections.register_section("render", RenderConfig, module="render")

    assert section == sections.get("render")


def test_duplicate_keys_fail_with_the_key_and_the_two_models() -> None:
    sections = SectionRegistry()
    sections.register("pipeline", PipelineConfig)

    with pytest.raises(ConfigError, match="already registered: pipeline") as caught:
        sections.register("pipeline", RenderConfig)

    assert caught.value.details["key"] == "pipeline"
    assert caught.value.details["existing"] == "PipelineConfig"
    assert caught.value.details["new"] == "RenderConfig"


def test_the_same_model_cannot_claim_one_key_twice() -> None:
    sections = SectionRegistry()
    sections.register("pipeline", PipelineConfig)

    with pytest.raises(ConfigError, match="already registered: pipeline"):
        sections.register("pipeline", PipelineConfig)


@pytest.mark.parametrize(
    "key",
    ["", " ", "two words", "two.keys", "_private", "1st", "class"],
)
def test_malformed_section_keys_are_rejected(key: str) -> None:
    sections = SectionRegistry()

    with pytest.raises(ConfigError, match="invalid config section key"):
        sections.register(key, PipelineConfig)


def test_section_keys_must_be_strings() -> None:
    sections = SectionRegistry()

    with pytest.raises(TypeError, match="key must be a string"):
        sections.register(1, PipelineConfig)  # type: ignore[arg-type]


def test_a_section_must_be_a_config_model_subclass() -> None:
    sections = SectionRegistry()

    with pytest.raises(TypeError, match="ConfigModel subclass"):
        sections.register("pipeline", dict)  # type: ignore[arg-type]


def test_unknown_section_lookup_fails_clearly() -> None:
    sections = SectionRegistry()
    sections.register("pipeline", PipelineConfig)

    with pytest.raises(KeyError, match="unknown config section: missing"):
        sections.get("missing")


def test_parse_returns_typed_sections_and_applies_defaults() -> None:
    sections = SectionRegistry()
    sections.register("pipeline", PipelineConfig)
    sections.register("render", RenderConfig)

    parsed = sections.parse({"pipeline": {"timezone": "Europe/Berlin"}, "render": {}})

    assert isinstance(parsed.pipeline, PipelineConfig)  # type: ignore[attr-defined]
    assert parsed.pipeline.timezone == "Europe/Berlin"  # type: ignore[attr-defined]
    assert parsed.pipeline.retries == 2  # type: ignore[attr-defined]
    assert isinstance(parsed.render, RenderConfig)  # type: ignore[attr-defined]
    assert parsed.render.min_contrast == 4.5  # type: ignore[attr-defined]


def test_sections_that_are_not_in_the_mapping_are_optional() -> None:
    sections = SectionRegistry()
    sections.register("pipeline", PipelineConfig)
    sections.register("research", ResearchConfig)

    parsed = sections.parse({"pipeline": {}})

    assert parsed.pipeline.timezone == "UTC"  # type: ignore[attr-defined]
    assert parsed.research is None  # type: ignore[attr-defined]


def test_unknown_top_level_section_fails_with_the_registered_keys() -> None:
    sections = SectionRegistry()
    sections.register("pipeline", PipelineConfig)
    sections.register("render", RenderConfig)

    with pytest.raises(ConfigError) as caught:
        sections.parse({"research": {}})

    message = str(caught.value)
    assert "unknown config section: research" in message
    assert "registered: pipeline, render" in message
    assert caught.value.details["unknown"] == ["research"]
    assert caught.value.details["available"] == ["pipeline", "render"]


def test_every_unknown_top_level_section_is_reported() -> None:
    sections = SectionRegistry()
    sections.register("pipeline", PipelineConfig)

    with pytest.raises(ConfigError) as caught:
        sections.parse({"zulu": {}, "alpha": {}})

    assert "unknown config section: alpha" in str(caught.value)
    assert "unknown config section: zulu" in str(caught.value)
    assert caught.value.details["unknown"] == ["alpha", "zulu"]


def test_unknown_nested_keys_keep_the_full_dotted_path() -> None:
    sections = SectionRegistry()
    sections.register("pipeline", PipelineConfig)

    with pytest.raises(ConfigError) as caught:
        sections.parse({"pipeline": {"bogus": 1}})

    assert "unknown config key: pipeline.bogus" in str(caught.value)
    assert caught.value.details["errors"][0]["path"] == "pipeline.bogus"


def test_parse_interpolates_environment_references() -> None:
    sections = SectionRegistry()
    sections.register("pipeline", PipelineConfig)

    parsed = sections.parse(
        {"pipeline": {"timezone": "${AIVA_TIMEZONE}"}},
        environ={"AIVA_TIMEZONE": "Europe/Berlin"},
    )

    assert parsed.pipeline.timezone == "Europe/Berlin"  # type: ignore[attr-defined]


def test_missing_environment_reference_keeps_the_dotted_path() -> None:
    sections = SectionRegistry()
    sections.register("pipeline", PipelineConfig)

    with pytest.raises(ConfigError) as caught:
        sections.parse(
            {"pipeline": {"timezone": "${AIVA_TIMEZONE}"}},
            environ={},
        )

    assert "pipeline.timezone" in str(caught.value)
    assert caught.value.details["path"] == "pipeline.timezone"


def test_config_root_must_be_a_mapping() -> None:
    sections = SectionRegistry()

    with pytest.raises(ConfigError, match="config root must be a mapping"):
        sections.parse(["pipeline"])  # type: ignore[arg-type]


def test_the_generated_root_model_exposes_registered_sections() -> None:
    sections = SectionRegistry()
    sections.register("pipeline", PipelineConfig)
    sections.register("render", RenderConfig)

    schema = sections.model().model_json_schema()
    properties: dict[str, Any] = schema["properties"]

    assert set(properties) == {"pipeline", "render"}

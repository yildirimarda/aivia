"""Optional module sections: core-only configs validate, absent modules fail clearly."""

from __future__ import annotations

from typing import Any, cast

import pytest
import yaml

from aivia.core import paths
from aivia.core.config import (
    ConfigModel,
    SectionRegistry,
    build_app_config,
    register_core_sections,
)
from aivia.core.config.sections import CORE_SECTION_KEYS
from aivia.core.errors import ConfigError

MODULE_KEYS = ("research", "analytics")


class ResearchConfig(ConfigModel):
    max_sources: int = 5


def core_only_model() -> type[ConfigModel]:
    return build_app_config(register_core_sections(SectionRegistry()))


def example_core_sections() -> dict[str, Any]:
    text = (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )
    payload = cast(dict[str, Any], yaml.safe_load(text))
    return {key: value for key, value in payload.items() if key in CORE_SECTION_KEYS}


def test_a_configuration_without_module_sections_still_validates() -> None:
    model = core_only_model()

    config = model.from_mapping(example_core_sections())

    for key in MODULE_KEYS:
        assert key not in model.model_fields
        assert getattr(config, key, None) is None
    assert config.pipeline.timezone == "Europe/Berlin"  # type: ignore[attr-defined]
    assert config.profiles.get("default").format == "narrated_still"  # type: ignore[attr-defined]


def test_an_empty_configuration_without_module_sections_is_valid() -> None:
    config = core_only_model().from_mapping({})

    for key in CORE_SECTION_KEYS:
        assert getattr(config, key) is None


@pytest.mark.parametrize("key", MODULE_KEYS)
def test_a_module_key_without_its_package_fails_clearly(key: str) -> None:
    model = core_only_model()

    with pytest.raises(ConfigError) as caught:
        model.from_mapping({key: {}})

    message = str(caught.value)
    assert f"unknown config section: {key}" in message
    assert "registered:" in message
    assert caught.value.details["unknown"] == [key]
    assert caught.value.details["available"] == sorted(CORE_SECTION_KEYS)


def test_a_module_key_is_rejected_even_next_to_valid_core_sections() -> None:
    model = core_only_model()

    with pytest.raises(ConfigError) as caught:
        model.from_mapping({"pipeline": {"timezone": "UTC"}, "research": {}})

    assert caught.value.details["unknown"] == ["research"]


def test_the_same_module_key_validates_once_the_module_registers_it() -> None:
    sections = SectionRegistry()
    model = build_app_config(sections)

    with pytest.raises(ConfigError, match="unknown config section: research"):
        model.from_mapping({"research": {"max_sources": 3}})

    sections.register("research", ResearchConfig, module="research")
    model = build_app_config(sections)

    config = model.from_mapping({"research": {"max_sources": 3}})

    assert config.research.max_sources == 3  # type: ignore[attr-defined]

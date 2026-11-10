"""Five built in occasions and support for adding one with files only."""

from __future__ import annotations

from typing import Any, cast

import pytest
import yaml

from aivia.core import paths
from aivia.core.config import (
    OccasionConfig,
    OccasionsConfig,
    VariableType,
)
from aivia.core.errors import ConfigError


def example_section(name: str) -> dict[str, Any]:
    text = (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )
    payload = cast(dict[str, Any], yaml.safe_load(text))
    return cast(dict[str, Any], payload[name])


def test_packaged_config_declares_five_built_in_occasions() -> None:
    occasions = OccasionsConfig.from_mapping(example_section("occasions"))

    assert occasions.default == "birthday"
    assert set(occasions.occasions) == {
        "birthday",
        "anniversary",
        "wedding",
        "graduation",
        "photo_collage",
    }


def test_builtin_occasions_have_expected_labels_and_variables() -> None:
    occasions = OccasionsConfig.from_mapping(example_section("occasions"))

    birthday = occasions.occasions["birthday"]
    assert birthday.label == "Birthday"
    assert birthday.variable_names == ["name", "age", "message"]
    assert birthday.variables["name"].type is VariableType.TEXT
    assert birthday.variables["name"].required is True

    anniversary = occasions.occasions["anniversary"]
    assert anniversary.label == "Anniversary"
    assert anniversary.variable_names == ["name", "partner", "years", "message"]

    wedding = occasions.occasions["wedding"]
    assert wedding.label == "Wedding"
    assert wedding.variable_names == ["name_one", "name_two", "date", "message"]
    assert wedding.variables["date"].type is VariableType.DATE

    graduation = occasions.occasions["graduation"]
    assert graduation.label == "Graduation"
    assert graduation.variable_names == ["name", "degree", "school", "message"]


def test_photo_collage_is_a_files_only_occasion() -> None:
    occasions = OccasionsConfig.from_mapping(example_section("occasions"))

    collage = occasions.occasions["photo_collage"]
    assert collage.label == "Photo Collage"
    assert collage.variables == {}
    assert collage.variable_names == []
    assert collage.files == ["collage_template.txt"]


def test_occasion_can_be_added_with_files_only() -> None:
    occasion = OccasionConfig.from_mapping(
        {
            "label": "Custom Upload",
            "files": ["template_a.txt", "template_b.txt"],
        }
    )

    assert occasion.label == "Custom Upload"
    assert occasion.variables == {}
    assert occasion.files == ["template_a.txt", "template_b.txt"]


def test_occasions_accept_a_files_only_occasion_alongside_variable_ones() -> None:
    occasions = OccasionsConfig.from_mapping(
        {
            "default": "birthday",
            "birthday": {"variables": ["name"], "files": ["birthday.txt"]},
            "photo_collage": {"files": ["collage_template.txt"]},
        }
    )

    assert set(occasions.occasions) == {"birthday", "photo_collage"}
    assert occasions.occasions["birthday"].variable_names == ["name"]
    assert occasions.occasions["photo_collage"].variables == {}
    assert occasions.occasions["photo_collage"].files == ["collage_template.txt"]


def test_files_only_occasion_rejects_an_unknown_default() -> None:
    with pytest.raises(ConfigError) as caught:
        OccasionsConfig.from_mapping(
            {
                "default": "photo_collage",
                "birthday": {"variables": ["name"]},
            }
        )

    assert "photo_collage" in str(caught.value)

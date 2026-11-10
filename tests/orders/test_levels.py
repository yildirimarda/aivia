"""Personalization levels: which steps each level runs and what each costs."""

from __future__ import annotations

from typing import Any, cast

import pytest
import yaml

from aivia.core import paths
from aivia.core.config import OrdersConfig, PersonalizationLevelConfig


def _levels() -> dict[str, dict[str, Any]]:
    return {
        "basic": {"stages": ["name_overlay"], "price": 0.0},
        "standard": {
            "stages": ["write", "music", "name_overlay", "render"],
            "price": 19.0,
            "cost_multiplier": 1.0,
        },
        "premium": {
            "stages": ["write", "music", "voice", "visuals", "name_overlay", "render"],
            "price": 39.0,
            "cost_multiplier": 1.5,
        },
    }


def _orders(values: dict[str, Any] | None = None) -> OrdersConfig:
    merged: dict[str, Any] = {}
    if values:
        merged.update(values)
    merged.setdefault("personalization_levels", _levels())
    return OrdersConfig.from_mapping(merged)


def test_level_resolves_its_stages_as_the_steps_it_runs() -> None:
    orders = _orders({"default_level": "standard"})

    assert orders.steps_for("basic") == ["name_overlay"]
    assert orders.steps_for("standard") == [
        "write",
        "music",
        "name_overlay",
        "render",
    ]
    assert orders.steps_for("premium") == [
        "write",
        "music",
        "voice",
        "visuals",
        "name_overlay",
        "render",
    ]


def test_cost_is_price_times_the_cost_multiplier() -> None:
    orders = _orders()

    assert orders.cost_for("basic") == 0.0
    assert orders.cost_for("standard") == pytest.approx(19.0)
    assert orders.cost_for("premium") == pytest.approx(39.0 * 1.5)


def test_default_multiplier_makes_cost_equal_price() -> None:
    orders = _orders(
        {
            "personalization_levels": {
                "flat": {"stages": ["name_overlay"], "price": 7.5}
            },
            "default_level": "flat",
        }
    )

    assert orders.cost_for() == 7.5


def test_default_level_is_used_when_no_name_is_given() -> None:
    orders = _orders({"default_level": "premium"})

    assert orders.level().name == "premium"
    assert orders.steps_for() == orders.steps_for("premium")
    assert orders.cost_for() == orders.cost_for("premium")


def test_level_returns_the_configured_level_object() -> None:
    orders = _orders({"default_level": "standard"})

    level = orders.level("premium")
    assert isinstance(level, PersonalizationLevelConfig)
    assert level.name == "premium"
    assert level.price == 39.0
    assert level.cost_multiplier == 1.5


def test_unknown_level_is_rejected() -> None:
    orders = _orders()

    with pytest.raises(ValueError, match="unknown personalization level"):
        orders.level("platinum")
    with pytest.raises(ValueError, match="unknown personalization level"):
        orders.steps_for("platinum")
    with pytest.raises(ValueError, match="unknown personalization level"):
        orders.cost_for("platinum")


def test_no_level_configured_is_rejected() -> None:
    orders = OrdersConfig.from_mapping({})

    with pytest.raises(ValueError, match="no personalization level is configured"):
        orders.level()
    with pytest.raises(ValueError, match="no personalization level is configured"):
        orders.steps_for()
    with pytest.raises(ValueError, match="no personalization level is configured"):
        orders.cost_for()


def test_packaged_levels_declare_steps_and_costs() -> None:
    text = (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )
    payload = cast(dict[str, Any], yaml.safe_load(text))
    orders = OrdersConfig.from_mapping(payload["orders"])

    assert orders.default_level == "standard"
    assert orders.steps_for("basic") == ["name_overlay"]
    assert orders.steps_for("standard") == [
        "write",
        "music",
        "name_overlay",
        "render",
    ]
    assert orders.steps_for("premium") == [
        "write",
        "music",
        "voice",
        "visuals",
        "name_overlay",
        "render",
    ]
    assert orders.cost_for("basic") == 0.0
    assert orders.cost_for("standard") == pytest.approx(19.0)
    assert orders.cost_for("premium") == pytest.approx(39.0)

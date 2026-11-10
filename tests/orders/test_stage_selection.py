"""Select pipeline stages from the requested outputs."""

from __future__ import annotations

from typing import Any

import pytest

from aivia.core.config import OrdersConfig


def _levels() -> dict[str, dict[str, Any]]:
    return {
        "basic": {
            "stages": ["name_overlay"],
            "outputs": ["png"],
            "price": 0.0,
        },
        "standard": {
            "stages": ["write", "music", "name_overlay", "render"],
            "outputs": ["mp4", "mp3"],
            "price": 19.0,
        },
        "premium": {
            "stages": ["write", "music", "voice", "visuals", "name_overlay", "render"],
            "outputs": ["mp4", "gif", "webp", "mp3", "png", "web_card"],
            "price": 39.0,
        },
    }


def _orders(values: dict[str, Any] | None = None) -> OrdersConfig:
    merged: dict[str, Any] = {}
    if values:
        merged.update(values)
    merged.setdefault("personalization_levels", _levels())
    return OrdersConfig.from_mapping(merged)


def test_png_selects_only_the_free_name_overlay_stage() -> None:
    orders = _orders({"default_level": "standard"})

    assert orders.stages_for(["png"]) == ["name_overlay"]
    assert orders.cost_for("basic") == 0.0


def test_stages_for_chooses_the_cheapest_level_covering_all_outputs() -> None:
    orders = _orders()

    # Only premium declares both png and mp4 among the packaged levels.
    assert orders.stages_for(["png", "mp4"]) == [
        "write",
        "music",
        "voice",
        "visuals",
        "name_overlay",
        "render",
    ]


def test_stages_for_returns_nothing_for_no_requested_outputs() -> None:
    orders = _orders()

    assert orders.stages_for([]) == []


def test_stages_for_rejects_outputs_no_level_produces() -> None:
    orders = _orders({"default_level": "standard"})

    with pytest.raises(ValueError, match="no personalization level produces"):
        orders.stages_for(["mov"])

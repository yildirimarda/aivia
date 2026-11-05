"""Visuals configuration: default strategy, AI video cap and assignment use."""

from __future__ import annotations

from typing import Any, cast

import pytest
import yaml

from aivia.core import paths
from aivia.core.config import (
    CORE_SECTION_KEYS,
    SectionRegistry,
    VisualsConfig,
    VisualStrategy,
    register_core_sections,
)
from aivia.core.errors import ConfigError
from aivia.core.models.script import Beat, Script, VisualType
from aivia.visuals.strategy import resolve_visual_requests
from aivia.writing.visual_assign import (
    DEFAULT_MAX_AI_VIDEO_BEATS,
    DEFAULT_VISUAL_STRATEGY,
    VisualAssigner,
    assign_visual_types,
    visual_types,
)


def example_section(name: str) -> dict[str, Any]:
    text = (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )
    payload = cast(dict[str, Any], yaml.safe_load(text))
    return cast(dict[str, Any], payload[name])


def _script(count: int = 5) -> Script:
    return Script(
        title="Bees",
        beats=[Beat(index=index, text=f"Fact {index + 1}.") for index in range(count)],
    )


def _values(script: Script) -> list[str]:
    return [beat.visual_type for beat in script.beats]


def test_visuals_defaults_match_the_assignment_defaults() -> None:
    visuals = VisualsConfig.from_mapping({})

    assert visuals.strategy is VisualStrategy.HYBRID
    assert visuals.strategy == DEFAULT_VISUAL_STRATEGY
    assert visuals.max_ai_video_beats == DEFAULT_MAX_AI_VIDEO_BEATS == 2
    assert visuals.default is VisualStrategy.HYBRID
    assert visuals.default_strategy is VisualStrategy.HYBRID
    assert visuals.visual_strategy is VisualStrategy.HYBRID
    assert visuals.max_video_beats == 2


def test_visuals_accepts_an_explicit_strategy_and_limit() -> None:
    visuals = VisualsConfig.from_mapping({"strategy": "video", "max_ai_video_beats": 1})

    assert visuals.strategy is VisualStrategy.VIDEO
    assert visuals.max_ai_video_beats == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("default", "image"),
        ("default_strategy", "stock"),
        ("visual_strategy", "still"),
        ("mode", "text"),
        ("source", "none"),
        ("strategy", "existing"),
        ("strategy", "ai_video"),
        ("strategy", "provided"),
        ("strategy", " VIDEO "),
    ],
)
def test_visuals_accepts_strategy_spellings(field: str, value: str) -> None:
    visuals = VisualsConfig.from_mapping({field: value})

    assert visuals.strategy == VisualsConfig.from_mapping({"strategy": value}).strategy


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_ai_video_beats", 0),
        ("max_ai_video_beats", 4),
        ("max_video_beats", 3),
        ("max_beats", 1),
        ("ai_video_beats", 2),
        ("max_ai_videos", 5),
    ],
)
def test_visuals_accepts_limit_aliases(field: str, value: int) -> None:
    visuals = VisualsConfig.from_mapping({field: value})

    assert visuals.max_ai_video_beats == value


def test_visuals_rejects_an_unknown_strategy() -> None:
    with pytest.raises(ConfigError) as caught:
        VisualsConfig.from_mapping({"strategy": "radio"})

    assert "radio" in str(caught.value)


@pytest.mark.parametrize("value", [-1, -10])
def test_visuals_rejects_a_negative_limit(value: int) -> None:
    with pytest.raises(ConfigError) as caught:
        VisualsConfig.from_mapping({"max_ai_video_beats": value})

    assert "max_ai_video_beats" in str(caught.value)


def test_visuals_is_registered_as_a_core_section() -> None:
    assert "visuals" in CORE_SECTION_KEYS

    sections = register_core_sections(SectionRegistry())
    parsed = sections.parse({"visuals": {"strategy": "image", "max_ai_video_beats": 1}})

    assert parsed.visuals.strategy is VisualStrategy.IMAGE  # type: ignore[attr-defined]
    assert parsed.visuals.max_ai_video_beats == 1  # type: ignore[attr-defined]


def test_visuals_unknown_keys_report_the_full_path() -> None:
    sections = register_core_sections(SectionRegistry())

    with pytest.raises(ConfigError) as caught:
        sections.parse({"visuals": {"bogus": True}})

    assert "visuals.bogus" in str(caught.value)


def test_the_packaged_example_declares_the_visuals_defaults() -> None:
    visuals = VisualsConfig.from_mapping(example_section("visuals"))

    assert visuals.strategy is VisualStrategy.HYBRID
    assert visuals.max_ai_video_beats == 2


def test_from_config_reads_the_visuals_section_or_defaults() -> None:
    assert VisualsConfig.from_config(None) == VisualsConfig()
    assert VisualsConfig.from_config({}) == VisualsConfig()

    config = {"visuals": {"strategy": "stock", "max_ai_video_beats": 0}}

    assert VisualsConfig.from_config(config).strategy is VisualStrategy.STOCK
    assert VisualsConfig.from_config(config).max_ai_video_beats == 0
    assert (
        VisualsConfig.from_config({"visual_strategy": {"default": "video"}}).strategy
        is VisualStrategy.VIDEO
    )


def test_visual_assignment_reads_the_config_section() -> None:
    config = {"visuals": {"strategy": "video", "max_ai_video_beats": 1}}

    result = assign_visual_types(_script(3), config=config)

    assert _values(result) == [
        VisualType.VIDEO.value,
        VisualType.IMAGE.value,
        VisualType.IMAGE.value,
    ]


def test_visual_assignment_reads_a_visuals_config_model() -> None:
    config = VisualsConfig.from_mapping({"strategy": "image", "max_ai_video_beats": 4})

    result = assign_visual_types(_script(3), config=config)

    assert _values(result) == [VisualType.IMAGE.value] * 3


def test_explicit_assignment_options_win_over_the_config() -> None:
    config = {"visuals": {"strategy": "video", "max_ai_video_beats": 4}}

    result = assign_visual_types(
        _script(3),
        strategy="stock",
        max_ai_video_beats=0,
        config=config,
    )

    assert _values(result) == [VisualType.STOCK.value] * 3


def test_visual_assigner_can_be_built_from_the_config() -> None:
    config = {"visuals": {"strategy": "still", "max_ai_video_beats": 1}}
    assigner = VisualAssigner.from_config(config)

    assert assigner.strategy is VisualStrategy.STILL
    assert assigner.max_ai_video_beats == 1
    assert visual_types(_script(2), config=config) == ["still", "still"]


def test_visual_request_resolution_reads_the_config_section() -> None:
    config = {"visuals": {"strategy": "video", "max_ai_video_beats": 1}}

    requests = resolve_visual_requests(_script(3), config=config)

    assert [request.visual_type for request in requests] == [
        VisualType.VIDEO,
        VisualType.IMAGE,
        VisualType.IMAGE,
    ]

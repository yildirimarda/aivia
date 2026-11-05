"""Sound effect configuration: rule files, default volume and placement."""

from __future__ import annotations

from typing import Any, cast

import pytest
import yaml

from aivia.core import paths
from aivia.core.config import (
    CORE_SECTION_KEYS,
    SectionRegistry,
    SfxConfig,
    SfxPlacement,
    SfxPlacementMode,
    SfxPosition,
    register_core_sections,
)
from aivia.core.errors import ConfigError


def example_section(name: str) -> dict[str, Any]:
    text = (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )
    payload = cast(dict[str, Any], yaml.safe_load(text))
    return cast(dict[str, Any], payload[name])


def test_sfx_aliases_point_at_the_same_models() -> None:
    assert SfxPlacementMode is SfxPlacement
    assert SfxPosition is SfxPlacement


def test_sfx_defaults_have_no_rules_and_start_placement() -> None:
    sfx = SfxConfig.from_mapping({})

    assert sfx.rule_files == []
    assert sfx.rules == []
    assert sfx.volume_db == -6.0
    assert sfx.volume == -6.0
    assert sfx.placement is SfxPlacement.START
    assert sfx.position is SfxPlacement.START


def test_sfx_accepts_explicit_rule_files_volume_and_placement() -> None:
    sfx = SfxConfig.from_mapping(
        {
            "rule_files": ["sfx/doors.yaml", "sfx/clicks.json"],
            "volume_db": -12.0,
            "placement": "center",
        }
    )

    assert sfx.rule_files == ["sfx/doors.yaml", "sfx/clicks.json"]
    assert sfx.volume_db == -12.0
    assert sfx.placement is SfxPlacement.CENTER


def test_sfx_accepts_a_single_rule_file_and_strips_entries() -> None:
    single = SfxConfig.from_mapping({"rule_files": "  sfx/doors.yaml  "})
    several = SfxConfig.from_mapping(
        {"rule_files": [" sfx/doors.yaml ", "", "sfx/doors.yaml", "sfx/clicks.json"]}
    )

    assert single.rule_files == ["sfx/doors.yaml"]
    assert several.rule_files == ["sfx/doors.yaml", "sfx/clicks.json"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("start", SfxPlacement.START),
        ("beginning", SfxPlacement.START),
        ("middle", SfxPlacement.CENTER),
        ("centre", SfxPlacement.CENTER),
        ("at_end", SfxPlacement.END),
        ("after", SfxPlacement.END),
        ("keyword", SfxPlacement.WORD),
        ("delay", SfxPlacement.OFFSET),
        ("timeline", SfxPlacement.ABSOLUTE),
        (" MIDDLE ", SfxPlacement.CENTER),
        ("at-end", SfxPlacement.END),
        (SfxPlacement.WORD, SfxPlacement.WORD),
    ],
)
def test_sfx_accepts_every_placement_spelling(
    value: object, expected: SfxPlacement
) -> None:
    sfx = SfxConfig.from_mapping({"placement": value})

    assert sfx.placement is expected


def test_sfx_rejects_an_unknown_placement() -> None:
    with pytest.raises(ConfigError) as caught:
        SfxConfig.from_mapping({"placement": "sideways"})

    assert "placement" in str(caught.value)
    assert "sideways" in str(caught.value)


def test_sfx_accepts_field_aliases_and_canonical_fields_win() -> None:
    aliased = SfxConfig.from_mapping(
        {
            "rules": ["sfx/aliased.yaml"],
            "gain_db": -9.0,
            "position": "end",
        }
    )
    canonical = SfxConfig.from_mapping(
        {
            "rule_files": ["sfx/canonical.yaml"],
            "rules": ["sfx/alias.yaml"],
            "volume_db": -4.0,
            "gain": -20.0,
            "placement": "start",
            "position": "end",
        }
    )

    assert aliased.rule_files == ["sfx/aliased.yaml"]
    assert aliased.volume_db == -9.0
    assert aliased.placement is SfxPlacement.END
    assert canonical.rule_files == ["sfx/canonical.yaml"]
    assert canonical.volume_db == -4.0
    assert canonical.placement is SfxPlacement.START


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_sfx_volume_must_be_finite(value: float) -> None:
    with pytest.raises(ConfigError) as caught:
        SfxConfig.from_mapping({"volume_db": value})

    assert "volume_db" in str(caught.value)


@pytest.mark.parametrize(
    "value",
    [{"path": "sfx/doors.yaml"}, b"sfx/doors.yaml", [1, 2], 42],
)
def test_sfx_rejects_invalid_rule_file_values(value: object) -> None:
    with pytest.raises(ConfigError) as caught:
        SfxConfig.from_mapping({"rule_files": value})

    assert "rule_files" in str(caught.value)


def test_sfx_rule_file_defaults_are_independent() -> None:
    first = SfxConfig.from_mapping({})
    second = SfxConfig.from_mapping({})

    first.rule_files.append("sfx/doors.yaml")

    assert second.rule_files == []


def test_sfx_is_registered_as_a_core_section() -> None:
    assert "sfx" in CORE_SECTION_KEYS

    sections = register_core_sections(SectionRegistry())
    parsed = sections.parse(
        {
            "sfx": {
                "rule_files": ["sfx/doors.yaml"],
                "volume_db": -10.0,
                "placement": "word",
            }
        }
    )

    assert parsed.sfx.rule_files == ["sfx/doors.yaml"]  # type: ignore[attr-defined]
    assert parsed.sfx.volume_db == -10.0  # type: ignore[attr-defined]
    assert parsed.sfx.placement is SfxPlacement.WORD  # type: ignore[attr-defined]


def test_sfx_unknown_keys_report_the_full_path() -> None:
    sections = register_core_sections(SectionRegistry())

    with pytest.raises(ConfigError) as caught:
        sections.parse({"sfx": {"bogus": True}})
    assert "sfx.bogus" in str(caught.value)


def test_the_packaged_example_declares_sfx_controls() -> None:
    sfx = SfxConfig.from_mapping(example_section("sfx"))

    assert sfx.rule_files == []
    assert sfx.volume_db == -6.0
    assert sfx.placement is SfxPlacement.START

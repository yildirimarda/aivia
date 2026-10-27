"""Planner configuration: limits, approval, tools, defaults and budget."""

from __future__ import annotations

from typing import Any, cast

import pytest
import yaml

from aivia.core import paths
from aivia.core.config import (
    CORE_SECTION_KEYS,
    ApprovalMode,
    BudgetConfig,
    PlannerApproval,
    PlannerApprovalMode,
    PlannerBudget,
    PlannerBudgetConfig,
    PlannerConfig,
    PlannerDefaultsConfig,
    SectionRegistry,
    register_core_sections,
)
from aivia.core.errors import ConfigError


def example_section(name: str) -> dict[str, Any]:
    text = (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )
    payload = cast(dict[str, Any], yaml.safe_load(text))
    return cast(dict[str, Any], payload[name])


def test_planner_defaults_bound_steps_iterations_and_ask_for_approval() -> None:
    planner = PlannerConfig.from_mapping({})

    assert planner.enabled is True
    assert planner.max_steps == 10
    assert planner.step_limit == 10
    assert planner.max_iterations == 3
    assert planner.iteration_limit == 3
    assert planner.approval_mode is PlannerApprovalMode.ASK
    assert planner.approval is PlannerApprovalMode.ASK
    assert planner.require_approval is False
    assert planner.allowed_tools == []
    assert planner.tools == []
    assert planner.allowlist == []
    assert planner.preview is True
    assert planner.defaults == PlannerDefaultsConfig()
    assert planner.default_arguments == planner.defaults
    assert planner.budget == BudgetConfig()
    assert planner.budget_per_instruction == 0.0


def test_planner_accepts_limits_approval_tools_preview_defaults_and_budget() -> None:
    planner = PlannerConfig.from_mapping(
        {
            "enabled": False,
            "max_steps": 24,
            "max_iterations": 5,
            "approval_mode": "always",
            "allowed_tools": [" image_gen ", "render"],
            "preview": False,
            "defaults": {
                "format": "quote_card",
                "language": "de",
                "duration_seconds": 12.5,
            },
            "budget": {"per_instruction": 2.5, "on_exceed": "downgrade"},
        }
    )

    assert planner.enabled is False
    assert planner.max_steps == 24
    assert planner.max_iterations == 5
    assert planner.approval_mode is PlannerApprovalMode.ALWAYS
    assert planner.require_approval is True
    assert planner.allowed_tools == ["image_gen", "render"]
    assert planner.preview is False
    assert planner.defaults.format == "quote_card"
    assert planner.defaults.language == "de"
    assert planner.defaults.duration_seconds == 12.5
    assert planner.budget.per_instruction == 2.5
    assert planner.budget.on_exceed.value == "downgrade"


def test_planner_approval_mode_enum_aliases_point_at_the_same_class() -> None:
    assert ApprovalMode is PlannerApprovalMode
    assert PlannerApproval is PlannerApprovalMode
    assert PlannerBudget is BudgetConfig
    assert PlannerBudgetConfig is BudgetConfig


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("always", PlannerApprovalMode.ALWAYS),
        ("ask", PlannerApprovalMode.ASK),
        ("never", PlannerApprovalMode.NEVER),
        ("manual", PlannerApprovalMode.ALWAYS),
        ("auto", PlannerApprovalMode.NEVER),
        (True, PlannerApprovalMode.ALWAYS),
        (False, PlannerApprovalMode.NEVER),
        (PlannerApprovalMode.ALWAYS, PlannerApprovalMode.ALWAYS),
    ],
)
def test_planner_accepts_every_approval_mode(
    value: object, expected: PlannerApprovalMode
) -> None:
    planner = PlannerConfig.from_mapping({"approval_mode": value})

    assert planner.approval_mode is expected


def test_planner_rejects_an_unknown_approval_mode() -> None:
    with pytest.raises(ConfigError) as caught:
        PlannerConfig.from_mapping({"approval_mode": "sometimes"})

    assert "approval_mode" in str(caught.value)


@pytest.mark.parametrize("value", [0, -1])
def test_max_steps_must_be_positive(value: int) -> None:
    with pytest.raises(ConfigError) as caught:
        PlannerConfig.from_mapping({"max_steps": value})

    assert "max_steps" in str(caught.value)


def test_max_iterations_may_be_zero_but_not_negative() -> None:
    assert PlannerConfig.from_mapping({"max_iterations": 0}).max_iterations == 0

    with pytest.raises(ConfigError) as caught:
        PlannerConfig.from_mapping({"max_iterations": -1})

    assert "max_iterations" in str(caught.value)


def test_allowed_tools_accept_a_comma_separated_string() -> None:
    planner = PlannerConfig.from_mapping({"allowed_tools": "image_gen, render"})

    assert planner.allowed_tools == ["image_gen", "render"]


@pytest.mark.parametrize("entry", ["", "   "])
def test_allowed_tools_reject_empty_names(entry: str) -> None:
    with pytest.raises(ConfigError) as caught:
        PlannerConfig.from_mapping({"allowed_tools": [entry]})

    assert "allowed_tools" in str(caught.value)


def test_allowed_tools_reject_duplicates() -> None:
    with pytest.raises(ConfigError) as caught:
        PlannerConfig.from_mapping({"allowed_tools": ["render", "render"]})

    assert "duplicate allowed tool: render" in str(caught.value)


def test_an_empty_allowlist_permits_every_tool_and_a_list_is_exclusive() -> None:
    open_planner = PlannerConfig.from_mapping({})
    restricted = PlannerConfig.from_mapping({"allowed_tools": ["render"]})

    assert open_planner.permits("image_gen") is True
    assert restricted.permits("render") is True
    assert restricted.permits("image_gen") is False


def test_planner_accepts_aliases_and_canonical_fields_win() -> None:
    aliased = PlannerConfig.from_mapping(
        {
            "step_limit": 7,
            "iteration_limit": 2,
            "approval": "never",
            "tools": ["render"],
            "show_preview": False,
            "default": {"format": "quote_card"},
            "instruction_budget": 1.5,
        }
    )
    canonical = PlannerConfig.from_mapping(
        {
            "max_steps": 9,
            "step_limit": 7,
            "approval_mode": "ask",
            "approval": "never",
        }
    )

    assert aliased.max_steps == 7
    assert aliased.max_iterations == 2
    assert aliased.approval_mode is PlannerApprovalMode.NEVER
    assert aliased.allowed_tools == ["render"]
    assert aliased.preview is False
    assert aliased.defaults.format == "quote_card"
    assert aliased.budget.per_instruction == 1.5
    assert canonical.max_steps == 9
    assert canonical.approval_mode is PlannerApprovalMode.ASK


def test_planner_budget_accepts_a_scalar_as_the_instruction_cap() -> None:
    planner = PlannerConfig.from_mapping({"budget": 4.25})

    assert planner.budget.per_instruction == 4.25
    assert planner.budget_per_instruction == 4.25


@pytest.mark.parametrize("value", [-0.01, -1])
def test_planner_budget_rejects_a_negative_instruction_cap(value: float) -> None:
    with pytest.raises(ConfigError) as caught:
        PlannerConfig.from_mapping({"budget": value})

    assert "budget.per_instruction" in str(caught.value)


def test_planner_defaults_accept_module_keys_and_expose_a_mapping() -> None:
    defaults = PlannerDefaultsConfig.from_mapping(
        {
            "format": "quote_card",
            "language": "tr",
            "voice": "warm",
            "my_module_option": {"nested": True},
        }
    )

    assert defaults.format == "quote_card"
    assert defaults.language == "tr"
    assert defaults.voice == "warm"
    assert defaults.get("my_module_option") == {"nested": True}
    assert defaults.get("missing", "fallback") == "fallback"
    assert defaults.values["my_module_option"] == {"nested": True}


def test_planner_defaults_are_independent_between_instances() -> None:
    first = PlannerConfig.from_mapping({})
    second = PlannerConfig.from_mapping({})

    first.max_steps = 1
    first.allowed_tools.append("render")
    first.defaults.format = "quote_card"
    first.budget.per_instruction = 3.0

    assert second.max_steps == 10
    assert second.allowed_tools == []
    assert second.defaults.format == ""
    assert second.budget.per_instruction == 0.0


def test_planner_unknown_keys_report_the_full_path() -> None:
    sections = register_core_sections(SectionRegistry())

    with pytest.raises(ConfigError) as caught:
        sections.parse({"planner": {"bogus": True}})

    assert "planner.bogus" in str(caught.value)


def test_planner_is_registered_as_a_core_section() -> None:
    assert "planner" in CORE_SECTION_KEYS

    sections = register_core_sections(SectionRegistry())
    parsed = sections.parse({"planner": {"max_steps": 6}})

    assert parsed.planner.max_steps == 6  # type: ignore[attr-defined]


def test_example_config_declares_the_planner_controls() -> None:
    planner = PlannerConfig.from_mapping(example_section("planner"))

    assert planner.enabled is True
    assert planner.max_steps == 10
    assert planner.max_iterations == 3
    assert planner.approval_mode is PlannerApprovalMode.ASK
    assert planner.allowed_tools == []
    assert planner.preview is True
    assert planner.defaults.format == "narrated_still"
    assert planner.defaults.language == "en"
    assert planner.budget.per_instruction == 0.0

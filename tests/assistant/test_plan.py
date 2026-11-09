"""Assistant plan schema: ordered steps, arguments, dependencies, estimate,
assumptions and acyclicity checks."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from aivia.core.models import Plan, PlanStep


def _step(
    step_id: str,
    *,
    tool: str = "tool",
    arguments: dict[str, Any] | None = None,
    depends_on: list[str] | None = None,
    estimate_usd: float = 0.0,
) -> PlanStep:
    return PlanStep(
        id=step_id,
        tool=tool,
        arguments=arguments or {},
        depends_on=depends_on or [],
        estimate_usd=estimate_usd,
    )


def test_plan_steps_are_ordered_and_addressable() -> None:
    plan = Plan(
        instruction="build",
        steps=[
            _step("first", tool="render", estimate_usd=0.1),
            _step("second", tool="caption", estimate_usd=0.2),
            _step("third", tool="mix", estimate_usd=0.3),
        ],
    )

    assert plan.step_count == 3
    assert plan.step_ids == ["first", "second", "third"]
    assert [step.tool for step in plan.steps] == ["render", "caption", "mix"]
    assert plan.step_by_id("second") is plan.steps[1]
    assert plan.step_by_id("missing") is None
    assert plan.total_estimate_usd == pytest.approx(0.6)


def test_plan_step_arguments_accept_aliases_and_coerce() -> None:
    step = PlanStep.model_validate(
        {"id": "s1", "tool": "render", "args": {"prompt": "x"}, "estimate": 0.5}
    )
    assert step.arguments == {"prompt": "x"}
    assert step.estimate_usd == 0.5
    assert step.estimate == 0.5

    for alias in ("arguments", "args", "params", "inputs", "options"):
        coerced = PlanStep.model_validate(
            {"id": "s2", "tool": "render", alias: {"key": "value"}}
        )
        assert coerced.arguments == {"key": "value"}

    assert PlanStep.model_validate({"id": "s3", "arguments": None}).arguments == {}


def test_plan_step_dependencies_accept_aliases() -> None:
    for alias in ("depends_on", "dependencies", "needs", "after"):
        step = PlanStep.model_validate(
            {"id": "s1", "tool": "render", alias: ["a", "b"]}
        )
        assert step.depends_on == ["a", "b"]
        assert step.dependencies == step.depends_on

    single = PlanStep.model_validate({"id": "s1", "tool": "render", "depends_on": "a"})
    assert single.depends_on == ["a"]


def test_plan_estimate_rejects_negative_values() -> None:
    with pytest.raises(ValidationError):
        PlanStep(id="s1", tool="render", estimate_usd=-0.1)


def test_plan_assumptions_are_normalized_and_deduplicated() -> None:
    plan = Plan.model_validate(
        {
            "id": "plan_1",
            "assumptions": ["  portrait  ", "portrait", "landscape"],
        }
    )

    assert plan.assumptions == ["portrait", "landscape"]

    aliased = Plan.model_validate({"id": "plan_2", "notes": ["  notes  ", "notes"]})
    assert aliased.assumptions == ["notes"]


def test_plan_detects_dependency_cycles() -> None:
    acyclic = Plan(
        steps=[
            _step("a"),
            _step("b", depends_on=["a"]),
            _step("c", depends_on=["a", "b"]),
        ]
    )
    assert acyclic.has_cycles() is False

    cyclic = Plan(
        steps=[
            _step("a", depends_on=["b"]),
            _step("b", depends_on=["a"]),
        ]
    )
    assert cyclic.has_cycles() is True

    self_loop = Plan(steps=[_step("a", depends_on=["a"])])
    assert self_loop.has_cycles() is True


def test_plan_defaults() -> None:
    plan = Plan()

    assert plan.id.startswith("plan_")
    assert plan.steps == []
    assert plan.assumptions == []
    assert plan.estimate_usd == 0.0
    assert plan.total_estimate_usd == 0.0
    assert plan.has_cycles() is False
    assert plan.step_count == 0

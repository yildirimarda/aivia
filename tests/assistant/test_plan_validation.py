"""Plan validation: allowlist, acyclicity, argument schemas and budget cap.

The planner rejects plans that use tools off the allowlist, contain dependency
cycles, pass arguments that fail a tool's JSON schema or exceed the
instruction budget cap.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from typing import Any

import pytest

from aivia.assistant.planner import Planner, PlannerPlanError
from aivia.assistant.tool_framework import Tool, ToolArguments, ToolRegistry
from aivia.core.config.planner import PlannerConfig
from aivia.core.models import Plan
from aivia.providers.protocols.llm import (
    LLMRequest,
    LLMResponse,
    LLMUsage,
)


class _CaptionArgs(ToolArguments):
    text: str
    length: int = 1


class CaptionTool(Tool):
    name = "caption"
    description = "Caption an asset to a given length."
    arguments_model = _CaptionArgs

    @classmethod
    def execute(cls, arguments: dict[str, Any]) -> Any:
        return {"text": str(arguments["text"])[: int(arguments.get("length", 1))]}


class _NoArgsTool(Tool):
    name = "noop"
    description = "A tool that takes no arguments."
    arguments_model = None

    @classmethod
    def execute(cls, arguments: dict[str, Any]) -> Any:
        return {"done": True}


def _registry(*tools: type[Tool]) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


VALID_PLAN = {
    "steps": [
        {
            "id": "s1",
            "tool": "caption",
            "arguments": {"text": "hello", "length": 3},
            "depends_on": [],
            "estimate_usd": 0.03,
        },
        {
            "id": "s2",
            "tool": "caption",
            "arguments": {"text": "world", "length": 5},
            "depends_on": ["s1"],
            "estimate_usd": 0.05,
        },
    ],
    "assumptions": ["the assets exist"],
}


def _plan_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload)


def _response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="model-a",
        provider="test",
        usage=LLMUsage(input_tokens=100, output_tokens=50),
    )


class _SequenceProvider:
    """An async LLM provider returning one canned response per call."""

    def __init__(self, responses: list[str] | str | None = None) -> None:
        if responses is None:
            self.responses = [_plan_json(VALID_PLAN)]
        elif isinstance(responses, str):
            self.responses = [responses]
        else:
            self.responses = list(responses)
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self.responses) - 1)
        return _response(self.responses[index])


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _planner(
    *tools: type[Tool],
    allowed_tools: list[str] | None = None,
    max_steps: int = 10,
    max_iterations: int = 3,
    provider: Any = None,
    max_repairs: int | None = None,
    budget_per_instruction: float | None = None,
) -> Planner:
    config = PlannerConfig(
        allowed_tools=allowed_tools if allowed_tools is not None else [],
        max_steps=max_steps,
        max_iterations=max_iterations,
    )
    if budget_per_instruction is not None:
        config.budget.per_instruction = budget_per_instruction
    return Planner(
        _registry(*tools),
        planner_config=config,
        provider=provider,
        max_repairs=max_repairs,
    )


def test_validation_rejects_a_tool_off_the_allowlist() -> None:
    provider = _SequenceProvider(
        _plan_json(
            {
                "steps": [
                    {
                        "id": "s1",
                        "tool": "noop",
                        "arguments": {},
                        "estimate_usd": 0.0,
                    },
                ]
            }
        )
    )
    planner = _planner(
        CaptionTool,
        _NoArgsTool,
        allowed_tools=["caption"],
        max_repairs=0,
        provider=provider,
    )

    with pytest.raises(PlannerPlanError) as raised:
        _run(planner.plan("run the noop tool"))

    assert "not on the allowlist" in str(raised.value)
    assert len(provider.requests) == 1


def test_validation_rejects_a_dependency_cycle() -> None:
    provider = _SequenceProvider(
        _plan_json(
            {
                "steps": [
                    {
                        "id": "s1",
                        "tool": "caption",
                        "arguments": {"text": "a"},
                        "depends_on": ["s2"],
                        "estimate_usd": 0.0,
                    },
                    {
                        "id": "s2",
                        "tool": "caption",
                        "arguments": {"text": "b"},
                        "depends_on": ["s1"],
                        "estimate_usd": 0.0,
                    },
                ]
            }
        )
    )
    planner = _planner(CaptionTool, max_repairs=0, provider=provider)

    with pytest.raises(PlannerPlanError) as raised:
        _run(planner.plan("caption in a circle"))

    assert "cycle" in str(raised.value).lower()
    assert len(provider.requests) == 1


def test_validation_rejects_a_self_dependency() -> None:
    provider = _SequenceProvider(
        _plan_json(
            {
                "steps": [
                    {
                        "id": "s1",
                        "tool": "caption",
                        "arguments": {"text": "a"},
                        "depends_on": ["s1"],
                        "estimate_usd": 0.0,
                    },
                ]
            }
        )
    )
    planner = _planner(CaptionTool, max_repairs=0, provider=provider)

    with pytest.raises(PlannerPlanError) as raised:
        _run(planner.plan("caption itself"))

    assert "depends on itself" in str(raised.value)


def test_validation_rejects_an_unknown_dependency() -> None:
    provider = _SequenceProvider(
        _plan_json(
            {
                "steps": [
                    {
                        "id": "s1",
                        "tool": "caption",
                        "arguments": {"text": "a"},
                        "depends_on": ["missing"],
                        "estimate_usd": 0.0,
                    },
                ]
            }
        )
    )
    planner = _planner(CaptionTool, max_repairs=0, provider=provider)

    with pytest.raises(PlannerPlanError):
        _run(planner.plan("caption a missing dependency"))


def test_validation_rejects_arguments_missing_a_required_field() -> None:
    provider = _SequenceProvider(
        _plan_json(
            {
                "steps": [
                    {
                        "id": "s1",
                        "tool": "caption",
                        "arguments": {},
                        "estimate_usd": 0.0,
                    },
                ]
            }
        )
    )
    planner = _planner(CaptionTool, max_repairs=0, provider=provider)

    with pytest.raises(PlannerPlanError) as raised:
        _run(planner.plan("caption without text"))

    assert "invalid arguments" in str(raised.value).lower()
    assert "caption" in str(raised.value)


def test_validation_rejects_an_unknown_argument_field() -> None:
    provider = _SequenceProvider(
        _plan_json(
            {
                "steps": [
                    {
                        "id": "s1",
                        "tool": "caption",
                        "arguments": {"text": "hello", "bogus": True},
                        "estimate_usd": 0.0,
                    },
                ]
            }
        )
    )
    planner = _planner(CaptionTool, max_repairs=0, provider=provider)

    with pytest.raises(PlannerPlanError) as raised:
        _run(planner.plan("caption with an extra field"))

    assert "invalid arguments" in str(raised.value).lower()


def test_validation_rejects_a_wrong_argument_type() -> None:
    provider = _SequenceProvider(
        _plan_json(
            {
                "steps": [
                    {
                        "id": "s1",
                        "tool": "caption",
                        "arguments": {"text": 42},
                        "estimate_usd": 0.0,
                    },
                ]
            }
        )
    )
    planner = _planner(CaptionTool, max_repairs=0, provider=provider)

    with pytest.raises(PlannerPlanError) as raised:
        _run(planner.plan("caption with a number for text"))

    assert "invalid arguments" in str(raised.value).lower()


def test_validation_rejects_arguments_for_a_tool_that_takes_none() -> None:
    provider = _SequenceProvider(
        _plan_json(
            {
                "steps": [
                    {
                        "id": "s1",
                        "tool": "noop",
                        "arguments": {"unwanted": True},
                        "estimate_usd": 0.0,
                    },
                ]
            }
        )
    )
    planner = _planner(
        CaptionTool,
        _NoArgsTool,
        max_repairs=0,
        provider=provider,
    )

    with pytest.raises(PlannerPlanError) as raised:
        _run(planner.plan("pass arguments to a no-arg tool"))

    assert "invalid arguments" in str(raised.value).lower()


def test_validation_rejects_a_plan_over_the_instruction_budget_cap() -> None:
    provider = _SequenceProvider(
        _plan_json(
            {
                "steps": [
                    {
                        "id": "s1",
                        "tool": "caption",
                        "arguments": {"text": "hello"},
                        "estimate_usd": 1.0,
                    },
                ]
            }
        )
    )
    planner = _planner(
        CaptionTool,
        max_repairs=0,
        provider=provider,
        budget_per_instruction=0.5,
    )

    with pytest.raises(PlannerPlanError) as raised:
        _run(planner.plan("an expensive caption"))

    assert "budget" in str(raised.value).lower()
    assert "exceeds" in str(raised.value).lower()


def test_validation_accepts_a_plan_at_or_under_the_budget_cap() -> None:
    provider = _SequenceProvider(
        _plan_json(
            {
                "steps": [
                    {
                        "id": "s1",
                        "tool": "caption",
                        "arguments": {"text": "hello"},
                        "estimate_usd": 0.5,
                    },
                ]
            }
        )
    )
    planner = _planner(
        CaptionTool,
        max_repairs=0,
        provider=provider,
        budget_per_instruction=0.5,
    )

    plan = _run(planner.plan("an affordable caption"))

    assert isinstance(plan, Plan)
    assert plan.step_count == 1


def test_validation_accepts_a_valid_plan() -> None:
    provider = _SequenceProvider(_plan_json(VALID_PLAN))
    planner = _planner(CaptionTool, provider=provider)

    plan = _run(planner.plan("caption the photo"))

    assert isinstance(plan, Plan)
    assert plan.instruction == "caption the photo"
    assert plan.step_count == 2
    assert plan.step_ids == ["s1", "s2"]
    assert plan.total_estimate_usd == pytest.approx(0.08)

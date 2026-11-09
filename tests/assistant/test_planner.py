"""Planner: turn an instruction, asset summary and tool schemas into a plan.

The planner is bounded by ``max_steps`` and repairs invalid LLM output up to
``max_iterations`` times before failing with :class:`PlannerPlanError`.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from typing import Any

import pytest

from aivia.assistant.planner import (
    PlannedPlan,
    PlannedStep,
    Planner,
    PlannerPlanError,
)
from aivia.assistant.tool_framework import Tool, ToolArguments, ToolRegistry
from aivia.core.config.planner import PlannerApprovalMode, PlannerConfig
from aivia.core.models import Plan
from aivia.core.models.asset import Asset, AssetKind
from aivia.providers.protocols.llm import (
    LLMRequest,
    LLMResponse,
    LLMUsage,
    MessageRole,
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
) -> Planner:
    config = PlannerConfig(
        allowed_tools=allowed_tools if allowed_tools is not None else [],
        max_steps=max_steps,
        max_iterations=max_iterations,
    )
    return Planner(
        _registry(*tools),
        planner_config=config,
        provider=provider,
        max_repairs=max_repairs,
    )


def test_planner_produces_a_valid_plan_from_an_instruction() -> None:
    provider = _SequenceProvider()
    planner = _planner(CaptionTool, provider=provider)

    plan = _run(planner.plan("caption the photo"))

    assert isinstance(plan, Plan)
    assert plan.instruction == "caption the photo"
    assert plan.step_count == 2
    assert plan.step_ids == ["s1", "s2"]
    step_s1 = plan.step_by_id("s1")
    assert step_s1 is not None
    assert step_s1.tool == "caption"
    step_s2 = plan.step_by_id("s2")
    assert step_s2 is not None
    assert step_s2.depends_on == ["s1"]
    assert plan.total_estimate_usd == pytest.approx(0.08)
    assert len(provider.requests) == 1


def test_planner_is_bounded_by_max_steps() -> None:
    provider = _SequenceProvider(
        [
            _plan_json(
                {
                    "steps": [
                        {
                            "id": "s1",
                            "tool": "caption",
                            "arguments": {},
                            "estimate_usd": 0.0,
                        },
                        {
                            "id": "s2",
                            "tool": "caption",
                            "arguments": {},
                            "estimate_usd": 0.0,
                        },
                        {
                            "id": "s3",
                            "tool": "caption",
                            "arguments": {},
                            "estimate_usd": 0.0,
                        },
                    ]
                }
            ),
            _plan_json(VALID_PLAN),
        ]
    )
    planner = _planner(CaptionTool, max_steps=2, max_repairs=1, provider=provider)

    plan = _run(planner.plan("caption three things"))

    assert plan.step_count == 2
    assert len(provider.requests) == 2


def test_planner_exact_step_limit_is_accepted() -> None:
    provider = _SequenceProvider(_plan_json(VALID_PLAN))
    planner = _planner(CaptionTool, max_steps=2, provider=provider)

    plan = _run(planner.plan("caption two things"))

    assert plan.step_count == 2


def test_planner_rejects_an_unknown_tool_and_fails() -> None:
    provider = _SequenceProvider(
        _plan_json(
            {
                "steps": [
                    {
                        "id": "s1",
                        "tool": "does-not-exist",
                        "arguments": {},
                        "estimate_usd": 0.0,
                    },
                ]
            }
        )
    )
    planner = _planner(CaptionTool, max_repairs=0, provider=provider)

    with pytest.raises(PlannerPlanError) as raised:
        _run(planner.plan("use a missing tool"))

    error = raised.value
    assert error.code == "planner_plan_error"
    assert "does-not-exist" in str(error)
    assert len(provider.requests) == 1


def test_planner_rejects_a_disallowed_tool_and_fails() -> None:
    provider = _SequenceProvider(
        _plan_json(
            {
                "steps": [
                    {"id": "s1", "tool": "noop", "arguments": {}, "estimate_usd": 0.0},
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

    with pytest.raises(PlannerPlanError):
        _run(planner.plan("run the noop tool"))

    assert len(provider.requests) == 1


def test_planner_rejects_a_dependency_cycle_and_fails() -> None:
    provider = _SequenceProvider(
        _plan_json(
            {
                "steps": [
                    {
                        "id": "s1",
                        "tool": "caption",
                        "arguments": {},
                        "depends_on": ["s2"],
                        "estimate_usd": 0.0,
                    },
                    {
                        "id": "s2",
                        "tool": "caption",
                        "arguments": {},
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


def test_planner_rejects_an_unknown_dependency_and_fails() -> None:
    provider = _SequenceProvider(
        _plan_json(
            {
                "steps": [
                    {
                        "id": "s1",
                        "tool": "caption",
                        "arguments": {},
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

    assert len(provider.requests) == 1


def test_planner_rejects_a_self_dependency_and_fails() -> None:
    provider = _SequenceProvider(
        _plan_json(
            {
                "steps": [
                    {
                        "id": "s1",
                        "tool": "caption",
                        "arguments": {},
                        "depends_on": ["s1"],
                        "estimate_usd": 0.0,
                    },
                ]
            }
        )
    )
    planner = _planner(CaptionTool, max_repairs=0, provider=provider)

    with pytest.raises(PlannerPlanError):
        _run(planner.plan("caption itself"))

    assert len(provider.requests) == 1


def test_planner_repairs_invalid_json_then_succeeds() -> None:
    provider = _SequenceProvider(["not json at all", _plan_json(VALID_PLAN)])
    planner = _planner(CaptionTool, max_repairs=1, provider=provider)

    plan = _run(planner.plan("caption the photo"))

    assert plan.step_count == 2
    assert len(provider.requests) == 2
    repair = provider.requests[1]
    assert repair.messages[-1].role is MessageRole.USER
    assert "not json at all" in repair.messages[-1].content


def test_planner_repairs_a_failed_plan_then_succeeds() -> None:
    provider = _SequenceProvider(
        [
            _plan_json(
                {
                    "steps": [
                        {
                            "id": "s1",
                            "tool": "does-not-exist",
                            "arguments": {},
                            "estimate_usd": 0.0,
                        },
                    ]
                }
            ),
            _plan_json(VALID_PLAN),
        ]
    )
    planner = _planner(CaptionTool, max_repairs=1, provider=provider)

    plan = _run(planner.plan("caption the photo"))

    assert plan.step_count == 2
    assert len(provider.requests) == 2


def test_planner_raises_after_exhausting_repairs() -> None:
    provider = _SequenceProvider(["not json", "still not json", "nope"])
    planner = _planner(CaptionTool, max_repairs=2, provider=provider)

    with pytest.raises(PlannerPlanError) as raised:
        _run(planner.plan("never valid"))

    error = raised.value
    assert error.code == "planner_plan_error"
    assert error.attempts == 3
    assert error.repairs == 2
    assert error.max_repairs == 2
    assert len(provider.requests) == 3


def test_planner_zero_repairs_means_the_first_failure_is_final() -> None:
    provider = _SequenceProvider(["not json", _plan_json(VALID_PLAN)])
    planner = _planner(CaptionTool, max_repairs=0, provider=provider)

    with pytest.raises(PlannerPlanError):
        _run(planner.plan("no repairs allowed"))

    assert len(provider.requests) == 1


def test_planner_provider_failure_is_not_repaired() -> None:
    class _FailingProvider:
        async def complete(self, request: LLMRequest) -> LLMResponse:
            raise RuntimeError("provider down")

    planner = _planner(CaptionTool, provider=_FailingProvider())

    with pytest.raises(RuntimeError, match="provider down"):
        _run(planner.plan("boom"))


def test_planner_without_a_provider_raises() -> None:
    planner = _planner(CaptionTool, provider=None)

    with pytest.raises(Exception) as raised:
        _run(planner.plan("nothing"))

    assert isinstance(raised.value, PlannerPlanError) is False


def test_planner_includes_the_asset_summary_in_the_prompt() -> None:
    provider = _SequenceProvider(_plan_json(VALID_PLAN))
    planner = _planner(CaptionTool, provider=provider)
    assets = {
        "photo": Asset(label="photo", kind=AssetKind.IMAGE, path="/tmp/photo.png"),
        "music": Asset(label="music", kind=AssetKind.AUDIO, path="/tmp/music.mp3"),
    }

    _run(planner.plan("caption the photo", assets=assets))

    prompt = provider.requests[0].messages[0].content
    assert "photo" in prompt
    assert "music" in prompt
    assert "image" in prompt
    assert "audio" in prompt


def test_planner_uses_an_explicit_asset_summary() -> None:
    provider = _SequenceProvider(_plan_json(VALID_PLAN))
    planner = _planner(CaptionTool, provider=provider)
    summary = [{"label": "cover", "kind": "image", "path": "/tmp/cover.jpg"}]

    _run(planner.plan("caption it", asset_summary=summary))

    prompt = provider.requests[0].messages[0].content
    assert "cover" in prompt
    assert "image" in prompt


def test_planner_exposes_the_tool_schemas() -> None:
    planner = _planner(CaptionTool)

    schemas = planner.schemas()
    assert set(schemas) == {"caption"}
    assert schemas["caption"]["type"] == "object"
    assert "text" in schemas["caption"]["properties"]


def test_planner_approval_mode_comes_from_config() -> None:
    provider = _SequenceProvider(_plan_json(VALID_PLAN))
    planner = _planner(CaptionTool, provider=provider)
    planner._planner_config = PlannerConfig(approval_mode=PlannerApprovalMode.ALWAYS)

    plan = _run(planner.plan("caption the photo"))

    assert plan.approval_mode.value == "always"


def test_planned_step_and_plan_models() -> None:
    step = PlannedStep(id="s1", tool="caption", arguments={"text": "x"})
    assert step.id == "s1"
    assert step.depends_on == []

    planned = PlannedPlan(steps=[step], assumptions=["note"])
    assert planned.steps[0].id == "s1"
    assert planned.assumptions == ["note"]


def test_planned_plan_defaults_to_empty() -> None:
    planned = PlannedPlan()
    assert planned.steps == []
    assert planned.assumptions == []

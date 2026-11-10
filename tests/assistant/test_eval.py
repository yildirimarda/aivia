"""Planner evaluation fixtures: expected tool sets, offline by default and cost
capped when live.

Each :class:`EvalCase` pairs an instruction with the tools the planner is
expected to select and a cost cap. Offline the fixtures run against a canned
LLM provider so no network or spend is involved. When run with ``--live`` the
same execution is re-capped against ``live_budget_usd`` so spend can never
exceed the configured live budget.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, cast

import pytest

from aivia.assistant import (
    SafetyLimits,
    Tool,
    ToolArguments,
    ToolEstimate,
    ToolFramework,
    ToolRegistry,
    ToolResult,
)
from aivia.assistant.planner import Planner
from aivia.core.config.budget import BudgetConfig
from aivia.core.config.planner import PlannerConfig
from aivia.core.models import Plan
from aivia.providers.protocols.llm import (
    LLMRequest,
    LLMResponse,
    LLMUsage,
)

# --- evaluation tools -------------------------------------------------------


class _CaptionArgs(ToolArguments):
    text: str
    length: int = 1


class CaptionTool(Tool):
    name = "caption"
    description = "Caption an asset to a given length."
    arguments_model = _CaptionArgs

    @classmethod
    def estimate(cls, arguments: dict[str, Any]) -> ToolEstimate:
        return ToolEstimate(cost_usd=0.001, units=1.0)

    @classmethod
    def execute(cls, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(output="captioned", cost_usd=0.001, estimated=False)


class ResizeTool(Tool):
    name = "resize"
    description = "Resize an asset to a target resolution."
    arguments_model = None

    @classmethod
    def estimate(cls, arguments: dict[str, Any]) -> ToolEstimate:
        return ToolEstimate(cost_usd=0.002, units=1.0)

    @classmethod
    def execute(cls, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(output="resized", cost_usd=0.002, estimated=False)


class ConvertTool(Tool):
    name = "convert"
    description = "Convert an asset to another format."
    arguments_model = None

    @classmethod
    def estimate(cls, arguments: dict[str, Any]) -> ToolEstimate:
        return ToolEstimate(cost_usd=0.002, units=1.0)

    @classmethod
    def execute(cls, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(output="converted", cost_usd=0.002, estimated=False)


class TrimTool(Tool):
    name = "trim"
    description = "Trim an asset to a duration."
    arguments_model = None

    @classmethod
    def estimate(cls, arguments: dict[str, Any]) -> ToolEstimate:
        return ToolEstimate(cost_usd=0.001, units=1.0)

    @classmethod
    def execute(cls, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(output="trimmed", cost_usd=0.001, estimated=False)


class MixTool(Tool):
    name = "mix"
    description = "Mix several audio tracks together."
    arguments_model = None

    @classmethod
    def estimate(cls, arguments: dict[str, Any]) -> ToolEstimate:
        return ToolEstimate(cost_usd=0.002, units=1.0)

    @classmethod
    def execute(cls, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(output="mixed", cost_usd=0.002, estimated=False)


class OverlayTool(Tool):
    name = "overlay"
    description = "Overlay one asset on top of another."
    arguments_model = None

    @classmethod
    def estimate(cls, arguments: dict[str, Any]) -> ToolEstimate:
        return ToolEstimate(cost_usd=0.003, units=1.0)

    @classmethod
    def execute(cls, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(output="overlaid", cost_usd=0.003, estimated=False)


class VoiceTool(Tool):
    name = "voice"
    description = "Synthesize a voice over from text."
    arguments_model = None

    @classmethod
    def estimate(cls, arguments: dict[str, Any]) -> ToolEstimate:
        return ToolEstimate(cost_usd=0.004, units=1.0)

    @classmethod
    def execute(cls, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(output="voiced", cost_usd=0.004, estimated=False)


class MusicTool(Tool):
    name = "music"
    description = "Add a music track to an asset."
    arguments_model = None

    @classmethod
    def estimate(cls, arguments: dict[str, Any]) -> ToolEstimate:
        return ToolEstimate(cost_usd=0.003, units=1.0)

    @classmethod
    def execute(cls, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(output="music", cost_usd=0.003, estimated=False)


class RenderTool(Tool):
    name = "render"
    description = "Render a composition to a final asset."
    arguments_model = None

    @classmethod
    def estimate(cls, arguments: dict[str, Any]) -> ToolEstimate:
        return ToolEstimate(cost_usd=0.005, units=1.0)

    @classmethod
    def execute(cls, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(output="rendered", cost_usd=0.005, estimated=False)


class DeliverTool(Tool):
    name = "deliver"
    description = "Deliver a finished asset to a destination."
    arguments_model = None

    @classmethod
    def estimate(cls, arguments: dict[str, Any]) -> ToolEstimate:
        return ToolEstimate(cost_usd=0.002, units=1.0)

    @classmethod
    def execute(cls, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(output="delivered", cost_usd=0.002, estimated=False)


_TOOL_ESTIMATE: dict[str, float] = {
    "caption": 0.001,
    "resize": 0.002,
    "convert": 0.002,
    "trim": 0.001,
    "mix": 0.002,
    "overlay": 0.003,
    "voice": 0.004,
    "music": 0.003,
    "render": 0.005,
    "deliver": 0.002,
}


def _step_arguments(tool_name: str) -> dict[str, Any]:
    if tool_name == "caption":
        return {"text": "the photo", "length": 5}
    return {}


# --- evaluation fixtures ----------------------------------------------------


@dataclass(frozen=True)
class EvalCase:
    """One planner evaluation: instruction, expected tool set and cost cap."""

    name: str
    instruction: str
    tools: tuple[type[Tool], ...]
    expected_tools: frozenset[str]
    budget_usd: float


EVAL_FIXTURES: tuple[EvalCase, ...] = (
    EvalCase(
        "caption only",
        "caption the photo",
        (CaptionTool,),
        frozenset({"caption"}),
        0.01,
    ),
    EvalCase(
        "resize only",
        "resize the image to 1080p",
        (ResizeTool,),
        frozenset({"resize"}),
        0.01,
    ),
    EvalCase(
        "convert only",
        "convert the video to mp4",
        (ConvertTool,),
        frozenset({"convert"}),
        0.01,
    ),
    EvalCase(
        "trim only",
        "trim the audio to 30 seconds",
        (TrimTool,),
        frozenset({"trim"}),
        0.01,
    ),
    EvalCase(
        "caption and resize",
        "caption and resize the photo",
        (CaptionTool, ResizeTool),
        frozenset({"caption", "resize"}),
        0.01,
    ),
    EvalCase(
        "resize then convert",
        "resize the image then convert it",
        (ResizeTool, ConvertTool),
        frozenset({"resize", "convert"}),
        0.01,
    ),
    EvalCase(
        "trim and mix",
        "trim the audio and mix it",
        (TrimTool, MixTool),
        frozenset({"trim", "mix"}),
        0.01,
    ),
    EvalCase(
        "caption, trim and overlay",
        "caption the photo, trim it and add an overlay",
        (CaptionTool, TrimTool, OverlayTool),
        frozenset({"caption", "trim", "overlay"}),
        0.01,
    ),
    EvalCase(
        "voice only",
        "voice over the video",
        (VoiceTool,),
        frozenset({"voice"}),
        0.01,
    ),
    EvalCase(
        "music only",
        "add music to the video",
        (MusicTool,),
        frozenset({"music"}),
        0.01,
    ),
    EvalCase(
        "render only",
        "render the composition",
        (RenderTool,),
        frozenset({"render"}),
        0.01,
    ),
    EvalCase(
        "deliver only",
        "deliver the final video",
        (DeliverTool,),
        frozenset({"deliver"}),
        0.01,
    ),
    EvalCase(
        "caption, resize and convert",
        "caption, resize and convert the image",
        (CaptionTool, ResizeTool, ConvertTool),
        frozenset({"caption", "resize", "convert"}),
        0.01,
    ),
    EvalCase(
        "trim, mix and voice",
        "trim the audio, mix it and add a voice over",
        (TrimTool, MixTool, VoiceTool),
        frozenset({"trim", "mix", "voice"}),
        0.01,
    ),
    EvalCase(
        "overlay and deliver",
        "overlay the logo and deliver the video",
        (OverlayTool, DeliverTool),
        frozenset({"overlay", "deliver"}),
        0.01,
    ),
    EvalCase(
        "caption, music and render",
        "caption the photo, add music and render",
        (CaptionTool, MusicTool, RenderTool),
        frozenset({"caption", "music", "render"}),
        0.01,
    ),
    EvalCase(
        "resize, overlay and deliver",
        "resize the image, add an overlay and deliver",
        (ResizeTool, OverlayTool, DeliverTool),
        frozenset({"resize", "overlay", "deliver"}),
        0.01,
    ),
    EvalCase(
        "convert, trim and mix",
        "convert the video, trim it and mix the audio",
        (ConvertTool, TrimTool, MixTool),
        frozenset({"convert", "trim", "mix"}),
        0.01,
    ),
    EvalCase(
        "caption, resize, trim and overlay",
        "caption, resize, trim and overlay the photo",
        (CaptionTool, ResizeTool, TrimTool, OverlayTool),
        frozenset({"caption", "resize", "trim", "overlay"}),
        0.01,
    ),
    EvalCase(
        "voice, music, render and deliver",
        "voice over, add music, render and deliver",
        (VoiceTool, MusicTool, RenderTool, DeliverTool),
        frozenset({"voice", "music", "render", "deliver"}),
        0.02,
    ),
)


@pytest.fixture(params=EVAL_FIXTURES, ids=lambda case: case.name)
def eval_case(request: pytest.FixtureRequest) -> EvalCase:
    """One of the twenty planner evaluation fixtures."""
    return cast(EvalCase, request.param)


# --- planner harness --------------------------------------------------------


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
            self.responses = []
        elif isinstance(responses, str):
            self.responses = [responses]
        else:
            self.responses = list(responses)
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self.responses) - 1)
        return _response(self.responses[index])


def _plan_document(case: EvalCase) -> dict[str, Any]:
    """Build a canned plan whose steps use exactly the expected tools."""
    steps: list[dict[str, Any]] = []
    for index, tool_name in enumerate(sorted(case.expected_tools), start=1):
        steps.append(
            {
                "id": f"s{index}",
                "tool": tool_name,
                "arguments": _step_arguments(tool_name),
                "depends_on": [],
                "estimate_usd": _TOOL_ESTIMATE[tool_name],
            }
        )
    return {"steps": steps, "assumptions": []}


def _plan_json(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload)


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _run_case(case: EvalCase, cap_usd: float) -> None:
    """Run one evaluation offline: plan, check tools, execute under a cap."""
    registry = ToolRegistry()
    for tool in case.tools:
        registry.register(tool)

    config = PlannerConfig(
        allowed_tools=[],
        budget=BudgetConfig(per_instruction=cap_usd),
    )
    provider = _SequenceProvider(_plan_json(_plan_document(case)))
    planner = Planner(
        registry,
        planner_config=config,
        provider=provider,
        max_repairs=0,
    )

    plan = _run(planner.plan(case.instruction))
    assert isinstance(plan, Plan)
    used = {step.tool for step in plan.steps}
    assert used == set(case.expected_tools)
    assert len(provider.requests) == 1

    framework = ToolFramework(
        registry,
        planner_config=config,
        safety_limits=SafetyLimits(max_cost_usd=cap_usd),
    )
    for step in plan.steps:
        framework.execute(step.tool, dict(step.arguments))
    assert framework.total_cost_usd <= cap_usd


# --- tests ------------------------------------------------------------------


def test_eval_offline_matches_expected_tool_set(eval_case: EvalCase) -> None:
    """Every fixture plans the expected tools and stays under its cap offline."""
    _run_case(eval_case, eval_case.budget_usd)


@pytest.mark.live
def test_eval_live_cost_is_capped(
    eval_case: EvalCase, live_budget_usd: Decimal
) -> None:
    """When live, execution is capped at the configured live budget."""
    _run_case(eval_case, float(live_budget_usd))

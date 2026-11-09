"""Incremental execution: re-run only the steps affected by a change.

When a step's input changes, dependency invalidation marks that step and every
step that transitively depends on it for re-execution. Every other
previously-succeeded step is skipped on resume, so a change only pays for the
affected part of the plan.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from aivia.assistant import (
    Executor,
    Tool,
    ToolArguments,
    ToolFramework,
    ToolRegistry,
    ToolResult,
)
from aivia.core.config.planner import PlannerConfig
from aivia.core.errors import PermanentError
from aivia.core.models import Plan, PlanStep
from aivia.core.stage_retry import StageRetryPolicy


class _CaptionArgs(ToolArguments):
    text: str
    length: int = 1


class CaptionTool(Tool):
    name = "caption"
    arguments_model = _CaptionArgs
    executions: ClassVar[list[dict[str, Any]]] = []

    @classmethod
    def execute(cls, arguments: dict[str, Any]) -> ToolResult:
        cls.executions.append(dict(arguments))
        length = int(arguments.get("length", 1))
        return ToolResult(
            output=str(arguments["text"])[:length],
            cost_usd=0.01 * length,
            estimated=False,
        )


class _FailingTool(Tool):
    name = "fails"
    arguments_model = None

    @classmethod
    def execute(cls, arguments: dict[str, Any]) -> ToolResult:
        raise PermanentError("nope")


def _run(coroutine: Any) -> Any:
    import asyncio

    return asyncio.run(coroutine)


def _framework(*tools: type[Tool]) -> ToolFramework:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return ToolFramework(registry, planner_config=PlannerConfig())


def _executor(
    *tools: type[Tool],
    retry_policy: StageRetryPolicy | None = None,
    state_path: str | None = None,
    resume: bool = False,
) -> Executor:
    framework = _framework(*tools)
    return Executor(
        framework,
        retry_policy=retry_policy,
        state_path=state_path,
        resume=resume,
    )


def _plan(*steps: PlanStep) -> Plan:
    return Plan(instruction="x", steps=list(steps))


def _step(
    step_id: str,
    *,
    tool: str = "caption",
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


def _executed_texts() -> list[str]:
    return [call["text"] for call in CaptionTool.executions]


def test_invalidated_step_and_its_transitive_dependents_rerun(
    tmp_path: Path,
) -> None:
    CaptionTool.executions = []
    state_path = str(tmp_path / "state.json")
    plan = _plan(
        _step("a", arguments={"text": "alpha", "length": 1}),
        _step("b", arguments={"text": "beta", "length": 1}, depends_on=["a"]),
        _step("c", arguments={"text": "gamma", "length": 1}, depends_on=["b"]),
        _step("d", arguments={"text": "delta", "length": 1}),
    )

    first = _run(_executor(CaptionTool, state_path=state_path).execute(plan))
    assert first.status.value == "succeeded"
    assert _executed_texts() == ["alpha", "beta", "gamma", "delta"]

    CaptionTool.executions = []
    second = _run(
        _executor(
            CaptionTool,
            state_path=state_path,
            resume=True,
        ).execute(plan, invalidated={"a"})
    )

    # a changed, so a, b and c re-run; d is unaffected and skipped.
    assert second.status.value == "succeeded"
    assert _executed_texts() == ["alpha", "beta", "gamma"]
    assert second.step_result("d").succeeded
    assert second.step_result("d").cost_usd == 0.01


def test_invalidating_a_middle_step_reruns_only_its_dependents(
    tmp_path: Path,
) -> None:
    CaptionTool.executions = []
    state_path = str(tmp_path / "state.json")
    plan = _plan(
        _step("a", arguments={"text": "alpha", "length": 1}),
        _step("b", arguments={"text": "beta", "length": 1}, depends_on=["a"]),
        _step("c", arguments={"text": "gamma", "length": 1}, depends_on=["b"]),
    )
    first = _run(_executor(CaptionTool, state_path=state_path).execute(plan))
    assert first.status.value == "succeeded"

    CaptionTool.executions = []
    second = _run(
        _executor(
            CaptionTool,
            state_path=state_path,
            resume=True,
        ).execute(plan, invalidated={"b"})
    )

    # b changed, so b and its dependent c re-run; a is unaffected and skipped.
    assert second.status.value == "succeeded"
    assert _executed_texts() == ["beta", "gamma"]
    assert second.step_result("a").succeeded


def test_invalidating_a_leaf_step_does_not_affect_unrelated_steps(
    tmp_path: Path,
) -> None:
    CaptionTool.executions = []
    state_path = str(tmp_path / "state.json")
    plan = _plan(
        _step("a", arguments={"text": "alpha", "length": 1}),
        _step("b", arguments={"text": "beta", "length": 1}, depends_on=["a"]),
        _step("c", arguments={"text": "gamma", "length": 1}),
    )
    _run(_executor(CaptionTool, state_path=state_path).execute(plan))

    CaptionTool.executions = []
    second = _run(
        _executor(
            CaptionTool,
            state_path=state_path,
            resume=True,
        ).execute(plan, invalidated={"c"})
    )

    # Only c re-runs; a and b are unaffected and skipped.
    assert second.status.value == "succeeded"
    assert _executed_texts() == ["gamma"]
    assert second.step_result("a").succeeded
    assert second.step_result("b").succeeded


def test_resume_without_invalidation_skips_every_succeeded_step(
    tmp_path: Path,
) -> None:
    CaptionTool.executions = []
    state_path = str(tmp_path / "state.json")
    plan = _plan(
        _step("a", arguments={"text": "alpha", "length": 1}),
        _step("b", arguments={"text": "beta", "length": 1}, depends_on=["a"]),
    )
    _run(_executor(CaptionTool, state_path=state_path).execute(plan))

    CaptionTool.executions = []
    second = _run(
        _executor(
            CaptionTool,
            state_path=state_path,
            resume=True,
        ).execute(plan)
    )

    assert second.status.value == "succeeded"
    assert _executed_texts() == []
    assert second.step_result("a").succeeded
    assert second.step_result("b").succeeded


def test_invalidated_step_whose_dependency_failed_is_skipped(
    tmp_path: Path,
) -> None:
    CaptionTool.executions = []
    state_path = str(tmp_path / "state.json")
    plan = _plan(
        _step("a", tool="fails"),
        _step(
            "b",
            tool="caption",
            arguments={"text": "beta", "length": 1},
            depends_on=["a"],
        ),
    )
    first = _run(
        _executor(
            _FailingTool,
            CaptionTool,
            state_path=state_path,
        ).execute(plan)
    )
    assert first.status.value == "failed"
    assert first.step_result("a").failed
    assert first.step_result("b").skipped
    assert first.step_result("b").error_code == "dependency_failed"

    CaptionTool.executions = []
    second = _run(
        _executor(
            _FailingTool,
            CaptionTool,
            state_path=state_path,
            resume=True,
        ).execute(plan, invalidated={"b"})
    )

    # a still fails, so the invalidated dependent b cannot run and is skipped.
    assert second.status.value == "failed"
    assert second.step_result("a").failed
    assert second.step_result("b").skipped
    assert second.step_result("b").error_code == "dependency_failed"


def test_unknown_invalidated_step_is_ignored(tmp_path: Path) -> None:
    CaptionTool.executions = []
    state_path = str(tmp_path / "state.json")
    plan = _plan(_step("a", arguments={"text": "alpha", "length": 1}))
    _run(_executor(CaptionTool, state_path=state_path).execute(plan))

    CaptionTool.executions = []
    second = _run(
        _executor(
            CaptionTool,
            state_path=state_path,
            resume=True,
        ).execute(plan, invalidated={"does-not-exist"})
    )

    assert second.status.value == "succeeded"
    assert _executed_texts() == []
    assert second.step_result("a").succeeded

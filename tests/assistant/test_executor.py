"""Executor: dependency-ordered execution with caching, per-step cost,
retries, fallbacks and resumable partial results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest

from aivia.assistant import (
    Executor,
    StepCache,
    Tool,
    ToolArguments,
    ToolEstimate,
    ToolFramework,
    ToolRegistry,
    ToolResult,
)
from aivia.core.config.planner import PlannerConfig
from aivia.core.errors import PermanentError, RetryableError
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
    def estimate(cls, arguments: dict[str, Any]) -> ToolEstimate:
        length = int(arguments.get("length", 1))
        return ToolEstimate(cost_usd=0.01 * length, units=float(length))

    @classmethod
    def execute(cls, arguments: dict[str, Any]) -> ToolResult:
        cls.executions.append(dict(arguments))
        length = int(arguments.get("length", 1))
        return ToolResult(
            output=str(arguments["text"])[:length],
            cost_usd=0.01 * length,
            estimated=False,
        )


class _FlakyTool(Tool):
    name = "flaky"
    arguments_model = None
    failures_remaining: int = 0

    @classmethod
    def execute(cls, arguments: dict[str, Any]) -> ToolResult:
        if cls.failures_remaining > 0:
            cls.failures_remaining -= 1
            raise RetryableError("not yet")
        return ToolResult(output="ok", cost_usd=0.02, estimated=False)


class _FailingTool(Tool):
    name = "fails"
    arguments_model = None

    @classmethod
    def execute(cls, arguments: dict[str, Any]) -> ToolResult:
        raise PermanentError("nope")


class _FallbackTool(Tool):
    name = "fallback"
    arguments_model = None

    @classmethod
    def execute(cls, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(output="via-fallback", cost_usd=0.03, estimated=False)


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
    sleep: Any = None,
) -> Executor:
    framework = _framework(*tools)
    return Executor(
        framework,
        retry_policy=retry_policy,
        state_path=state_path,
        resume=resume,
        sleep=sleep,
    )


async def _noop_sleep(delay: float) -> None:
    return None


def _plan(*steps: PlanStep) -> Plan:
    return Plan(instruction="x", steps=list(steps))


def _step(
    step_id: str,
    *,
    tool: str = "caption",
    arguments: dict[str, Any] | None = None,
    depends_on: list[str] | None = None,
    estimate_usd: float = 0.0,
    fallbacks: list[str] | None = None,
) -> PlanStep:
    return PlanStep(
        id=step_id,
        tool=tool,
        arguments=arguments or {},
        depends_on=depends_on or [],
        estimate_usd=estimate_usd,
        fallbacks=fallbacks or [],
    )


def test_executor_runs_steps_in_dependency_order() -> None:
    CaptionTool.executions = []
    plan = _plan(
        _step("a", arguments={"text": "first", "length": 1}),
        _step("b", arguments={"text": "second", "length": 1}, depends_on=["a"]),
        _step("c", arguments={"text": "third", "length": 1}, depends_on=["a", "b"]),
    )
    result = _run(_executor(CaptionTool).execute(plan))

    assert result.status.value == "succeeded"
    assert result.ok
    assert [call["text"] for call in CaptionTool.executions] == [
        "first",
        "second",
        "third",
    ]
    assert result.step_result("a").output == "f"
    assert result.step_result("b").output == "s"
    assert result.step_result("c").output == "t"
    assert result.total_cost_usd == pytest.approx(0.03)


def test_executor_skips_step_when_dependency_fails() -> None:
    plan = _plan(
        _step("a", tool="fails"),
        _step(
            "b", tool="caption", arguments={"text": "x", "length": 1}, depends_on=["a"]
        ),
    )
    result = _run(_executor(_FailingTool, CaptionTool).execute(plan))

    assert result.status.value == "failed"
    assert not result.ok
    assert result.step_result("a").failed
    assert result.step_result("b").skipped
    assert result.step_result("b").error_code == "dependency_failed"
    assert result.total_cost_usd == 0.0


def test_executor_caches_identical_calls() -> None:
    CaptionTool.executions = []
    plan = _plan(
        _step("a", arguments={"text": "same", "length": 2}),
        _step("b", arguments={"text": "same", "length": 2}),
    )
    executor = _executor(CaptionTool)
    result = _run(executor.execute(plan))

    assert len(CaptionTool.executions) == 1
    assert result.step_result("a").succeeded
    assert result.step_result("a").cost_usd == pytest.approx(0.02)
    assert result.step_result("b").succeeded
    assert result.step_result("b").cost_usd == 0.0
    assert result.step_result("b").cached is True
    assert result.total_cost_usd == pytest.approx(0.02)
    assert executor.cache.hits == 1
    assert executor.cache.misses == 1


def test_executor_records_per_step_cost_and_total() -> None:
    CaptionTool.executions = []
    plan = _plan(
        _step("a", arguments={"text": "hello", "length": 3}, estimate_usd=0.03),
        _step(
            "b",
            arguments={"text": "world", "length": 5},
            depends_on=["a"],
            estimate_usd=0.05,
        ),
    )
    result = _run(_executor(CaptionTool).execute(plan))

    assert result.step_result("a").cost_usd == pytest.approx(0.03)
    assert result.step_result("b").cost_usd == pytest.approx(0.05)
    assert result.total_cost_usd == pytest.approx(0.08)
    assert (
        result.total_cost_usd
        == result.step_result("a").cost_usd + result.step_result("b").cost_usd
    )


def test_executor_retries_transient_failure() -> None:
    _FlakyTool.failures_remaining = 2
    plan = _plan(_step("a", tool="flaky"))
    policy = StageRetryPolicy(max_attempts=3, retry_delay_seconds=0.0)
    result = _run(
        _executor(_FlakyTool, retry_policy=policy, sleep=_noop_sleep).execute(plan)
    )

    assert result.status.value == "succeeded"
    assert result.step_result("a").succeeded
    assert result.step_result("a").attempts == 3
    assert result.step_result("a").cost_usd == pytest.approx(0.02)


def test_executor_does_not_retry_permanent_error() -> None:
    plan = _plan(_step("a", tool="fails"))
    policy = StageRetryPolicy(max_attempts=3, retry_delay_seconds=0.0)
    result = _run(
        _executor(_FailingTool, retry_policy=policy, sleep=_noop_sleep).execute(plan)
    )

    assert result.status.value == "failed"
    assert result.step_result("a").failed
    assert result.step_result("a").attempts == 1
    assert result.step_result("a").error_code == "permanent"


def test_executor_tries_fallback_tool() -> None:
    plan = _plan(
        _step(
            "a",
            tool="fails",
            fallbacks=["fallback"],
        )
    )
    result = _run(_executor(_FailingTool, _FallbackTool).execute(plan))

    assert result.status.value == "succeeded"
    assert result.step_result("a").succeeded
    assert result.step_result("a").fallback_used == "fallback"
    assert result.step_result("a").output == "via-fallback"
    assert result.step_result("a").cost_usd == pytest.approx(0.03)


def test_executor_all_fallbacks_fail() -> None:
    _FlakyTool.failures_remaining = 1
    plan = _plan(
        _step("a", tool="fails", fallbacks=["flaky"]),
    )
    result = _run(_executor(_FailingTool, _FlakyTool).execute(plan))

    assert result.status.value == "failed"
    assert result.step_result("a").failed
    assert result.step_result("a").fallback_used == ""


def test_executor_exhausts_retries_then_uses_fallback() -> None:
    _FlakyTool.failures_remaining = 3
    plan = _plan(
        _step("a", tool="flaky", fallbacks=["fallback"]),
    )
    policy = StageRetryPolicy(max_attempts=3, retry_delay_seconds=0.0)
    result = _run(
        _executor(
            _FlakyTool, _FallbackTool, retry_policy=policy, sleep=_noop_sleep
        ).execute(plan)
    )

    assert result.status.value == "succeeded"
    assert result.step_result("a").fallback_used == "fallback"
    assert result.step_result("a").attempts == 4  # 3 flaky + 1 fallback


def test_executor_unknown_tool_fails_step() -> None:
    plan = _plan(_step("a", tool="does-not-exist"))
    result = _run(_executor().execute(plan))

    assert result.status.value == "failed"
    assert result.step_result("a").failed


def test_executor_resume_skips_succeeded_steps(tmp_path: Path) -> None:
    CaptionTool.executions = []
    _FlakyTool.failures_remaining = 1
    state_path = str(tmp_path / "state.json")

    plan = _plan(
        _step("a", arguments={"text": "alpha", "length": 1}),
        _step("b", tool="flaky"),
    )
    policy = StageRetryPolicy(max_attempts=1, retry_delay_seconds=0.0)
    first = _run(
        _executor(
            CaptionTool,
            _FlakyTool,
            retry_policy=policy,
            state_path=state_path,
            resume=False,
        ).execute(plan)
    )

    assert first.status.value == "failed"
    assert first.step_result("a").succeeded
    assert first.step_result("b").failed
    assert len(CaptionTool.executions) == 1

    _FlakyTool.failures_remaining = 0
    second = _run(
        _executor(
            CaptionTool,
            _FlakyTool,
            retry_policy=policy,
            state_path=state_path,
            resume=True,
        ).execute(plan)
    )

    assert second.status.value == "succeeded"
    assert second.step_result("a").succeeded
    assert second.step_result("b").succeeded
    # "a" was loaded from state and not re-executed; "b" ran once in each pass.
    assert len(CaptionTool.executions) == 1


def test_executor_detects_dependency_cycle() -> None:
    plan = _plan(
        _step("a", depends_on=["b"]),
        _step("b", depends_on=["a"]),
    )
    executor = _executor(CaptionTool)
    with pytest.raises(Exception) as raised:
        _run(executor.execute(plan))
    assert "cycle" in str(raised.value).lower()


def test_executor_cache_key_is_deterministic() -> None:
    arguments = {"text": "x", "length": 1}
    key_one = StepCache.key_for("caption", arguments)
    key_two = StepCache.key_for("caption", dict(arguments))
    assert key_one == key_two
    assert key_one != StepCache.key_for("caption", {"text": "y", "length": 1})
    assert key_one != StepCache.key_for("other", arguments)


def test_executor_result_accessors() -> None:
    CaptionTool.executions = []
    plan = _plan(
        _step("a", arguments={"text": "one", "length": 1}),
        _step("b", arguments={"text": "two", "length": 2}, depends_on=["a"]),
    )
    result = _run(_executor(CaptionTool).execute(plan))

    assert result.step_count == 2
    assert result.by_id("a") is result.step_result("a")
    assert result.by_id("missing") is None
    assert tuple(s.step_id for s in result.succeeded) == ("a", "b")
    assert result.failed == ()
    assert result.skipped == ()
    assert result.cached == ()
    payload = result.to_dict()
    assert payload["plan_id"] == plan.id
    assert payload["status"] == "succeeded"
    assert payload["total_cost_usd"] == pytest.approx(0.03)


def test_executor_step_state_aliases() -> None:
    step = PlanStep(
        id="s1",
        tool="caption",
        arguments={},
        fallbacks=["fb", "fb", ""],
    )
    assert step.fallbacks == ["fb"]
    assert step.fallback_tools == ["fb"]
    assert PlanStep.model_validate(
        {"id": "s2", "tool": "caption", "alternatives": ["x"]}
    ).fallbacks == ["x"]


def test_executor_empty_plan() -> None:
    result = _run(_executor(CaptionTool).execute(Plan(instruction="nothing")))
    assert result.status.value == "succeeded"
    assert result.step_count == 0
    assert result.total_cost_usd == 0.0
    assert result.ok

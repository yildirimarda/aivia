"""Executor: run a plan's steps in dependency order.

Steps run once every dependency has finished. Every step result is cached by
its tool and arguments so an identical call is never paid for twice, each step
records its own cost, transient failures are retried under a
:class:`~aivia.core.stages.stage_retry.StageRetryPolicy`, alternate tools are
tried as fallbacks, and results are persisted so a resumed run skips steps that
already succeeded.
"""
# config-coverage: allow-file — attempts/total_attempts are retry counters,
# not configuration thresholds; none belong in data or config files.

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from aivia.assistant.tool_framework import ToolFramework, ToolResult
from aivia.core.errors import AiviaError
from aivia.core.models.base import canonical_json, utcnow
from aivia.core.models.io import Plan, PlanState, PlanStep, PlanStepState
from aivia.core.resume import stable_value
from aivia.core.stage_retry import StageRetryPolicy

__all__ = [
    "ExecutionResult",
    "Executor",
    "ExecutorError",
    "StepCache",
    "StepExhaustedError",
    "StepResult",
]

_STATE_VERSION = 1
_MAX_COST_TOLERANCE = 1e-9


class ExecutorError(AiviaError):
    """Base class for every error raised by the executor."""

    code = "executor_error"
    retryable = False


class StepExhaustedError(ExecutorError):
    """A step and every one of its fallbacks failed after all retries."""

    code = "step_exhausted"
    retryable = False

    def __init__(
        self,
        step_id: str,
        tool: str,
        attempts: int,
        last_error: BaseException,
    ) -> None:
        self.step_id = step_id
        self.tool = tool
        self.attempts = attempts
        self.last_error = last_error
        self.error_code = getattr(last_error, "code", "") or type(last_error).__name__
        self.error_message = str(last_error).strip() or type(last_error).__name__
        super().__init__(
            f"step {step_id!r} (tool {tool!r}) exhausted after {attempts} attempt(s): "
            f"{self.error_message}",
            details={
                "step_id": step_id,
                "tool": tool,
                "attempts": attempts,
                "error_code": self.error_code,
                "error_message": self.error_message,
            },
        )


class _ToolFailed(ExecutorError):
    """Raised when a single tool and all its retries fail."""

    code = "tool_failed"
    retryable = False

    def __init__(self, *, attempts: int, error: BaseException) -> None:
        self.attempts = attempts
        self.last_error = error
        super().__init__(
            str(error) or type(error).__name__,
            details={"attempts": attempts, "error_code": getattr(error, "code", "")},
        )


class StepCache:
    """A content cache keyed by the tool name and its arguments.

    A cache hit returns the previously recorded output at zero cost, so an
    identical call is never executed or paid for twice within one run.
    """

    def __init__(self) -> None:
        self._entries: dict[str, tuple[Any, float]] = {}
        self.hits = 0
        self.misses = 0

    @classmethod
    def key_for(cls, tool: str, arguments: Mapping[str, Any]) -> str:
        """Return a stable cache key for *tool* called with *arguments*."""
        payload = canonical_json({"tool": tool, "arguments": dict(arguments)})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> tuple[Any, float] | None:
        """Return the cached ``(output, cost_usd)`` for *key*, or ``None``."""
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        return entry

    def set(self, key: str, output: Any, cost_usd: float) -> None:
        """Record *output* and its *cost_usd* under *key*."""
        self._entries[key] = (output, float(cost_usd))

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self.hits = 0
        self.misses = 0


@dataclass(frozen=True, slots=True)
class StepResult:
    """The recorded outcome of executing one plan step."""

    step_id: str
    tool: str
    status: PlanStepState
    output: Any = None
    cost_usd: float = 0.0
    estimated: bool = False
    attempts: int = 0
    cached: bool = False
    fallback_used: str = ""
    error_code: str = ""
    error_message: str = ""
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def succeeded(self) -> bool:
        """Whether the step completed successfully."""
        return self.status is PlanStepState.SUCCEEDED

    @property
    def failed(self) -> bool:
        """Whether the step failed after all retries and fallbacks."""
        return self.status is PlanStepState.FAILED

    @property
    def skipped(self) -> bool:
        """Whether the step was skipped because a dependency could not run."""
        return self.status is PlanStepState.SKIPPED

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation for persistence."""
        return {
            "step_id": self.step_id,
            "tool": self.tool,
            "status": self.status.value,
            "output": stable_value(self.output),
            "cost_usd": self.cost_usd,
            "estimated": self.estimated,
            "attempts": self.attempts,
            "cached": self.cached,
            "fallback_used": self.fallback_used,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "started_at": (
                self.started_at.isoformat() if self.started_at is not None else None
            ),
            "finished_at": (
                self.finished_at.isoformat() if self.finished_at is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StepResult:
        """Rebuild a result written by :meth:`to_dict`."""

        def _time(value: Any) -> datetime | None:
            if not isinstance(value, str) or not value:
                return None
            return datetime.fromisoformat(value)

        return cls(
            step_id=str(data.get("step_id", "")),
            tool=str(data.get("tool", "")),
            status=PlanStepState(str(data.get("status", "pending"))),
            output=data.get("output"),
            cost_usd=float(data.get("cost_usd", 0.0) or 0.0),
            estimated=bool(data.get("estimated", False)),
            attempts=int(data.get("attempts", 0) or 0),
            cached=bool(data.get("cached", False)),
            fallback_used=str(data.get("fallback_used", "") or ""),
            error_code=str(data.get("error_code", "") or ""),
            error_message=str(data.get("error_message", "") or ""),
            started_at=_time(data.get("started_at")),
            finished_at=_time(data.get("finished_at")),
        )


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """The ordered outcomes of executing a plan."""

    plan_id: str
    instruction: str
    status: PlanState
    results: dict[str, StepResult]
    step_order: tuple[str, ...]
    total_cost_usd: float
    started_at: datetime
    finished_at: datetime

    @property
    def step_count(self) -> int:
        """The number of steps in the plan."""
        return len(self.step_order)

    @property
    def ok(self) -> bool:
        """Whether no step failed."""
        return not self.failed

    @property
    def succeeded(self) -> tuple[StepResult, ...]:
        """Every step that completed successfully, in step order."""
        return tuple(
            self.results[step_id]
            for step_id in self.step_order
            if self.results[step_id].succeeded
        )

    @property
    def failed(self) -> tuple[StepResult, ...]:
        """Every step that failed after all retries and fallbacks."""
        return tuple(
            self.results[step_id]
            for step_id in self.step_order
            if self.results[step_id].failed
        )

    @property
    def skipped(self) -> tuple[StepResult, ...]:
        """Every step that was skipped because a dependency could not run."""
        return tuple(
            self.results[step_id]
            for step_id in self.step_order
            if self.results[step_id].skipped
        )

    @property
    def cached(self) -> tuple[StepResult, ...]:
        """Every step whose result came from the cache without re-execution."""
        return tuple(
            self.results[step_id]
            for step_id in self.step_order
            if self.results[step_id].cached
        )

    def step_result(self, step_id: str) -> StepResult | None:
        """Return the outcome for *step_id*, or ``None``."""
        return self.results.get(step_id)

    def by_id(self, step_id: str) -> StepResult | None:
        """Alias for :meth:`step_result`."""
        return self.results.get(step_id)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of the whole run."""
        return {
            "plan_id": self.plan_id,
            "instruction": self.instruction,
            "status": self.status.value,
            "step_order": list(self.step_order),
            "total_cost_usd": self.total_cost_usd,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "results": {
                step_id: result.to_dict() for step_id, result in self.results.items()
            },
        }


class Executor:
    """Run a :class:`~aivia.core.models.io.Plan` step by step.

    The executor respects each step's ``depends_on`` so dependencies always run
    first, records the cost of every step, retries transient failures, tries
    declared fallback tools when a step cannot succeed, caches identical calls
    and persists results so a resumed run skips steps that already finished.
    """

    def __init__(
        self,
        framework: ToolFramework,
        *,
        retry_policy: StageRetryPolicy | None = None,
        cache: StepCache | None = None,
        state_path: str | Path | None = None,
        resume: bool = False,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._framework = framework
        self._retry_policy = retry_policy or StageRetryPolicy()
        self._cache = cache or StepCache()
        self._state_path = Path(state_path) if state_path is not None else None
        self._resume = bool(resume)
        self._sleep = sleep or asyncio.sleep
        self._clock = clock
        self._results: dict[str, StepResult] = {}
        self._invalidated: set[str] = set()
        self._affected: set[str] = set()

    @property
    def framework(self) -> ToolFramework:
        """The tool framework used to validate and run each step."""
        return self._framework

    @property
    def retry_policy(self) -> StageRetryPolicy:
        """The retry policy applied to every tool call."""
        return self._retry_policy

    @property
    def cache(self) -> StepCache:
        """The content cache shared by every step in this run."""
        return self._cache

    @property
    def state_path(self) -> Path | None:
        """The file used to persist partial results, or ``None``."""
        return self._state_path

    @property
    def results(self) -> dict[str, StepResult]:
        """The step outcomes recorded so far, keyed by step id."""
        return dict(self._results)

    async def execute(
        self,
        plan: Plan,
        *,
        resume: bool | None = None,
        invalidated: set[str] | None = None,
    ) -> ExecutionResult:
        """Run *plan* and return its ordered outcomes.

        Steps run in dependency order, ties broken by plan order. When
        *resume* is set (or configured on the executor) a previously persisted
        state file is loaded and every step that already succeeded is skipped.

        *invalidated* names step ids whose inputs changed since the last run.
        Those steps and every step that (transitively) depends on them are
        re-executed; every other previously-succeeded step is skipped, so only
        the steps affected by the change run again.
        """
        if plan.has_cycles():
            raise ExecutorError(
                f"plan {plan.id!r} contains a dependency cycle",
                details={"plan_id": plan.id},
            )

        do_resume = self._resume if resume is None else bool(resume)
        if do_resume:
            self._load_state()
        else:
            self._results.clear()
        self._cache.clear()

        self._invalidated = set(invalidated or ())
        steps = self._order_steps(plan)
        self._affected = self._affected_steps(plan, self._invalidated)
        started_at = self._clock()
        for step in steps:
            await self._execute_step(plan, step)
        finished_at = self._clock()

        plan_ids = {step.id for step in steps}
        results = {
            step_id: result
            for step_id, result in self._results.items()
            if step_id in plan_ids
        }
        self._results = results
        if self._state_path is not None:
            self._save_state()

        statuses = [result.status for result in results.values()]
        if any(status is PlanStepState.FAILED for status in statuses):
            status = PlanState.FAILED
        elif all(
            status in {PlanStepState.SUCCEEDED, PlanStepState.SKIPPED}
            for status in statuses
        ):
            status = PlanState.SUCCEEDED
        else:
            status = PlanState.RUNNING

        total_cost = math.fsum(result.cost_usd for result in results.values())
        return ExecutionResult(
            plan_id=plan.id,
            instruction=plan.instruction,
            status=status,
            results=results,
            step_order=tuple(step.id for step in steps),
            total_cost_usd=total_cost,
            started_at=started_at,
            finished_at=finished_at,
        )

    # --- dependency order -------------------------------------------------

    def _order_steps(self, plan: Plan) -> list[PlanStep]:
        """Return the steps in dependency order, ties broken by plan order."""
        by_id = {step.id: step for step in plan.steps}
        order: list[PlanStep] = []
        done: set[str] = set()
        remaining = list(plan.steps)
        while remaining:
            progressed = False
            for step in list(remaining):
                dependencies = [dep for dep in step.depends_on if dep in by_id]
                if all(dep in done for dep in dependencies):
                    order.append(step)
                    done.add(step.id)
                    remaining.remove(step)
                    progressed = True
            if not progressed:
                order.extend(remaining)
                break
        return order

    def _affected_steps(self, plan: Plan, invalidated: set[str]) -> set[str]:
        """Return *invalidated* plus every step that transitively depends on it.

        Builds the reverse dependency map and floods from the directly
        invalidated step ids, so a change to one step invalidates every
        downstream step whose input could have changed as a result.
        """
        dependents: dict[str, set[str]] = {step.id: set() for step in plan.steps}
        for step in plan.steps:
            for dep in step.depends_on:
                if dep in dependents:
                    dependents[dep].add(step.id)
        affected = set(invalidated)
        stack = list(invalidated)
        while stack:
            current = stack.pop()
            for dependent in dependents.get(current, ()):
                if dependent not in affected:
                    affected.add(dependent)
                    stack.append(dependent)
        return affected

    # --- single step ------------------------------------------------------

    async def _execute_step(self, plan: Plan, step: PlanStep) -> None:
        step_id = step.id

        for dep in step.depends_on:
            dep_result = self._results.get(dep)
            if dep_result is None or not dep_result.succeeded:
                reason = (
                    "missing_dependency" if dep_result is None else "dependency_failed"
                )
                self._record(
                    step_id,
                    step.tool,
                    PlanStepState.SKIPPED,
                    error_code=reason,
                    error_message=(
                        f"dependency {dep!r} did not succeed"
                        if dep_result is None
                        else f"dependency {dep!r} failed"
                    ),
                )
                self._persist(step_id)
                return

        existing = self._results.get(step_id)
        if (
            existing is not None
            and existing.succeeded
            and step_id not in self._affected
        ):
            return

        cache_key = StepCache.key_for(step.tool, step.arguments)
        cached = self._cache.get(cache_key)
        if cached is not None:
            output, _cost = cached
            self._record(
                step_id,
                step.tool,
                PlanStepState.SUCCEEDED,
                output=output,
                cost_usd=0.0,
                cached=True,
                attempts=0,
            )
            self._persist(step_id)
            return

        await self._run_step_with_fallbacks(step, cache_key)

    async def _run_step_with_fallbacks(self, step: PlanStep, cache_key: str) -> None:
        policy = self._retry_policy
        sleep = self._sleep
        last_error: BaseException | None = None
        total_attempts = 0
        result: ToolResult | None = None
        used_fallback = ""

        for tool in [step.tool, *step.fallbacks]:
            try:
                tool_result, attempts = await self._attempt_tool(
                    tool, step.arguments, policy, sleep
                )
                result = tool_result
                total_attempts += attempts
                used_fallback = "" if tool == step.tool else tool
                break
            except _ToolFailed as error:
                last_error = error.last_error
                total_attempts += error.attempts
                continue

        if result is None:
            final_error = last_error or ExecutorError(
                f"step {step.id!r} failed without an error"
            )
            self._record(
                step.id,
                step.tool,
                PlanStepState.FAILED,
                error_code=getattr(final_error, "code", "")
                or type(final_error).__name__,
                error_message=str(final_error),
                attempts=total_attempts,
            )
        else:
            self._record(
                step.id,
                step.tool,
                PlanStepState.SUCCEEDED,
                output=result.output,
                cost_usd=result.cost_usd,
                estimated=result.estimated,
                attempts=total_attempts,
                fallback_used=used_fallback,
            )
            self._cache.set(cache_key, result.output, result.cost_usd)
        self._persist(step.id)

    async def _attempt_tool(
        self,
        tool: str,
        arguments: Mapping[str, Any],
        policy: StageRetryPolicy,
        sleep: Callable[[float], Awaitable[None]],
    ) -> tuple[ToolResult, int]:
        """Run *tool* once, retrying transient failures up to the policy limit.

        Returns the successful :class:`ToolResult` and the attempt count, or
        raises :class:`_ToolFailed` carrying the last error and its attempts.
        """
        last_error: BaseException | None = None
        attempts = 0
        for attempt in range(1, policy.max_attempts + 1):
            attempts = attempt
            try:
                return self._framework.execute(tool, dict(arguments)), attempts
            except BaseException as error:
                last_error = error
                if not policy.is_retryable(error):
                    raise _ToolFailed(attempts=attempts, error=error) from error
                if attempt < policy.max_attempts:
                    await sleep(policy.delay_for(attempt))
        assert last_error is not None
        raise _ToolFailed(attempts=attempts, error=last_error) from last_error

    # --- recording and persistence ----------------------------------------

    def _record(
        self,
        step_id: str,
        tool: str,
        status: PlanStepState,
        *,
        output: Any = None,
        cost_usd: float = 0.0,
        estimated: bool = False,
        attempts: int = 0,
        cached: bool = False,
        fallback_used: str = "",
        error_code: str = "",
        error_message: str = "",
    ) -> StepResult:
        started_at = self._clock()
        result = StepResult(
            step_id=step_id,
            tool=tool,
            status=status,
            output=output,
            cost_usd=float(cost_usd),
            estimated=estimated,
            attempts=attempts,
            cached=cached,
            fallback_used=fallback_used,
            error_code=error_code,
            error_message=error_message,
            started_at=started_at,
            finished_at=started_at,
        )
        self._results[step_id] = result
        return result

    def _persist(self, step_id: str) -> None:
        if self._state_path is None:
            return
        self._save_state()

    def _save_state(self) -> None:
        assert self._state_path is not None
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _STATE_VERSION,
            "updated_at": utcnow().isoformat(),
            "results": {
                step_id: result.to_dict() for step_id, result in self._results.items()
            },
        }
        _write_atomic(self._state_path, canonical_json(payload, indent=2) + "\n")

    def _load_state(self) -> None:
        if self._state_path is None or not self._state_path.is_file():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if not isinstance(data, Mapping):
            return
        raw_results = data.get("results", {})
        if not isinstance(raw_results, Mapping):
            return
        self._results.clear()
        for step_id, raw in raw_results.items():
            if not isinstance(raw, Mapping):
                continue
            self._results[str(step_id)] = StepResult.from_dict(raw)


def _write_atomic(path: Path, text: str) -> None:
    """Write *text* to *path* atomically via a temporary file and os.replace."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()
        raise

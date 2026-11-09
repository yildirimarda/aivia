"""Planner: turn an instruction, asset summary and tool schemas into a plan.

The planner asks an LLM for a JSON plan whose steps call registered, allowed
tools. The plan is bounded by ``max_steps`` from the planner configuration.
Invalid output is repaired through the LLM up to ``max_iterations`` times; when
the limit is reached the last error is raised as :class:`PlannerPlanError`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from pydantic import BaseModel, Field

from aivia.assistant.tool_framework import ToolRegistry, ToolValidationError
from aivia.core.config.planner import PlannerConfig
from aivia.core.errors import AiviaError
from aivia.core.models import Plan, PlanApprovalMode, PlanStep
from aivia.core.models.asset import Asset
from aivia.llm.structured import StructuredOutputError, parse_structured
from aivia.providers.protocols.llm import (
    ChatMessage,
    LLMRequest,
    LLMResponse,
    MessageRole,
)

__all__ = [
    "PlannedPlan",
    "PlannedStep",
    "Planner",
    "PlannerError",
    "PlannerPlanError",
]


class PlannerError(AiviaError):
    """Base class for every error raised by the planner."""

    code = "planner_error"


class PlannerPlanError(PlannerError):
    """The planner could not produce a valid plan within the repair limit."""

    code = "planner_plan_error"

    def __init__(
        self,
        message: str,
        *,
        content: str = "",
        errors: list[dict[str, Any]] | None = None,
        attempts: int = 1,
        repairs: int = 0,
        max_repairs: int = 0,
    ) -> None:
        super().__init__(
            message,
            details={
                "content": content,
                "errors": errors or [],
                "attempts": attempts,
                "repairs": repairs,
                "max_repairs": max_repairs,
            },
        )
        self.content = content
        self.errors = list(errors or [])
        self.attempts = attempts
        self.repairs = repairs
        self.max_repairs = max_repairs


class PlannedStep(BaseModel):
    """One step the LLM proposes for the plan."""

    id: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    estimate_usd: float = Field(default=0.0, ge=0.0)


class PlannedPlan(BaseModel):
    """The LLM's JSON plan proposal."""

    steps: list[PlannedStep] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class Planner:
    """Produce a validated :class:`Plan` from an instruction and tool schemas."""

    def __init__(
        self,
        registry: ToolRegistry,
        planner_config: PlannerConfig | None = None,
        provider: Any = None,
        *,
        max_repairs: int | None = None,
    ) -> None:
        self._registry = registry
        self._planner_config = planner_config or PlannerConfig()
        self._provider = provider
        self._max_repairs = (
            self._planner_config.max_iterations if max_repairs is None else max_repairs
        )

    @property
    def registry(self) -> ToolRegistry:
        """The tool registry supplying schemas and allowlist decisions."""
        return self._registry

    @property
    def planner_config(self) -> PlannerConfig:
        """The planner configuration supplying limits and approval."""
        return self._planner_config

    @property
    def provider(self) -> Any:
        """The LLM provider the planner calls."""
        return self._provider

    @property
    def max_repairs(self) -> int:
        """How many validation failures may be sent back to the model."""
        return self._max_repairs

    @property
    def step_limit(self) -> int:
        """The maximum number of steps a plan may contain."""
        return self._planner_config.max_steps

    def schemas(self) -> dict[str, dict[str, Any]]:
        """Return the JSON schema of every registered tool, keyed by name."""
        return self._registry.schemas()

    @staticmethod
    def asset_summary(assets: Mapping[str, Asset]) -> list[dict[str, Any]]:
        """Summarise *assets* as label/kind entries for the planner prompt."""
        summary: list[dict[str, Any]] = []
        for label, asset in assets.items():
            summary.append(
                {
                    "label": asset.label or label,
                    "kind": asset.kind.value,
                    "path": asset.path,
                }
            )
        return summary

    async def plan(
        self,
        instruction: str,
        *,
        assets: Mapping[str, Asset] | None = None,
        asset_summary: Sequence[Mapping[str, Any]] | None = None,
    ) -> Plan:
        """Turn *instruction* into a valid, step-bounded plan.

        *assets* maps labels to registered assets and is summarised into the
        prompt. *asset_summary* overrides the derived summary. The LLM is asked
        for JSON matching :class:`PlannedPlan`; invalid output and plans that
        break the step limit, the allowlist or dependency rules are repaired up
        to :attr:`max_repairs` times before :class:`PlannerPlanError` is raised.
        """
        if self._provider is None:
            raise PlannerError("planner has no LLM provider")
        if not callable(getattr(self._provider, "complete", None)):
            raise PlannerError("provider must expose an async complete method")

        summary = (
            list(asset_summary)
            if asset_summary is not None
            else self.asset_summary(assets or {})
        )

        messages: list[ChatMessage] = [self._opening_message(instruction, summary)]
        schema = PlannedPlan.model_json_schema()
        last_error: PlannerError | StructuredOutputError | None = None

        for attempt in range(1, self._max_repairs + 2):
            request = LLMRequest(
                messages=messages,
                response_schema=schema,
            )
            try:
                response: LLMResponse = await self._provider.complete(request)
            except Exception:  # provider failures are not repairable
                raise

            try:
                planned = cast(PlannedPlan, parse_structured(response, PlannedPlan))
            except StructuredOutputError as error:
                last_error = error
                if attempt > self._max_repairs:
                    break
                messages = self._repair_messages(
                    messages,
                    error,
                    "Your previous response was not a valid plan.",
                )
                continue

            problems = self._validate(planned)
            if problems:
                content = response.content or ""
                message = "plan failed validation: " + "; ".join(problems)
                last_error = PlannerPlanError(
                    message,
                    content=content,
                    errors=[{"message": problem} for problem in problems],
                    attempts=attempt,
                    repairs=attempt - 1,
                    max_repairs=self._max_repairs,
                )
                if attempt > self._max_repairs:
                    break
                messages = self._repair_messages(
                    messages,
                    last_error,
                    "Your previous plan failed validation:",
                )
                continue

            return self._to_plan(instruction, planned)

        assert last_error is not None
        if isinstance(last_error, PlannerPlanError):
            raise last_error
        raise PlannerPlanError(
            str(last_error) or "plan could not be produced",
            content=getattr(last_error, "content", ""),
            errors=getattr(last_error, "errors", []),
            attempts=getattr(last_error, "attempts", self._max_repairs + 1),
            repairs=self._max_repairs,
            max_repairs=self._max_repairs,
        ) from last_error

    # --- message building -------------------------------------------------

    def _opening_message(
        self,
        instruction: str,
        summary: Sequence[Mapping[str, Any]],
    ) -> ChatMessage:
        lines = [
            "You are a planning assistant. Turn the instruction into a JSON plan.",
            "",
            "Instruction:",
            instruction,
            "",
            "Available assets (label, kind):",
        ]
        if summary:
            for entry in summary:
                lines.append(f"- {entry.get('label', '')} ({entry.get('kind', '')})")
        else:
            lines.append("- (none)")
        lines.extend(
            [
                "",
                "Available tools (name -> JSON schema):",
            ]
        )
        schemas = self.schemas()
        if schemas:
            for name, schema in schemas.items():
                lines.append(f"- {name}: {json_dumps(schema)}")
        else:
            lines.append("- (none)")
        lines.extend(
            [
                "",
                "Rules:",
                "- Emit a single JSON object with a ``steps`` array and optional "
                "``assumptions`` array.",
                "- Each step needs an ``id``, a ``tool`` name from the available "
                "tools and optional ``arguments``, ``depends_on`` and "
                "``estimate_usd``.",
                f"- The plan may contain at most {self.step_limit} steps.",
                "- ``depends_on`` must only name step ids that appear earlier in "
                "the plan and must not create a cycle.",
                f"- Only tools on the allowlist may be used: "
                f"{self._planner_config.allowed_tools or '(all registered tools)'}.",
                "- Return only valid JSON, with no prose or markdown fences.",
            ]
        )
        return ChatMessage(role=MessageRole.USER, content="\n".join(lines))

    def _repair_messages(
        self,
        messages: list[ChatMessage],
        error: StructuredOutputError | PlannerPlanError,
        lead: str,
    ) -> list[ChatMessage]:
        repaired = list(messages)
        content = getattr(error, "content", "") or ""
        if content:
            repaired.append(ChatMessage(role=MessageRole.ASSISTANT, content=content))
        details = [lead]
        errors = getattr(error, "errors", []) or []
        if errors:
            details.append("Validation problems:")
            for entry in errors:
                message = entry.get("message") if isinstance(entry, Mapping) else None
                if message:
                    details.append(f"- {message}")
        if isinstance(error, StructuredOutputError) and error.errors:
            details.append("Schema errors: " + json_dumps(error.errors))
        if content:
            details.extend(["", "Previous response:", content])
        details.extend(
            [
                "",
                "Return only valid JSON that fixes the problems above, with no "
                "prose or markdown fences.",
            ]
        )
        repaired.append(ChatMessage(role=MessageRole.USER, content="\n".join(details)))
        return repaired

    # --- validation -------------------------------------------------------

    def _validate(self, planned: PlannedPlan) -> list[str]:
        problems: list[str] = []
        limit = self.step_limit
        if len(planned.steps) > limit:
            problems.append(f"plan has {len(planned.steps)} steps, limit is {limit}")
        seen_ids: set[str] = set()
        for step in planned.steps:
            if step.id in seen_ids:
                problems.append(f"duplicate step id {step.id!r}")
            seen_ids.add(step.id)
            name = step.tool
            try:
                tool = self._registry.get(name)
            except Exception:
                problems.append(f"unknown tool {name!r} in step {step.id!r}")
                continue
            if not self._planner_config.permits(name):
                problems.append(
                    f"tool {name!r} is not on the allowlist in step {step.id!r}"
                )
            try:
                tool.validate_arguments(dict(step.arguments))
            except ToolValidationError as error:
                problems.append(
                    f"invalid arguments for tool {name!r} in step {step.id!r}: {error}"
                )
            del tool
        step_ids = {step.id for step in planned.steps}
        for step in planned.steps:
            for dependency in step.depends_on:
                if dependency == step.id:
                    problems.append(f"step {step.id!r} depends on itself")
                elif dependency not in step_ids:
                    problems.append(
                        f"step {step.id!r} depends on unknown step {dependency!r}"
                    )
        if self._has_cycle(planned):
            problems.append("plan contains a dependency cycle")
        ceiling = self._planner_config.budget_per_instruction
        if ceiling and ceiling > 0:
            total = sum(step.estimate_usd for step in planned.steps)
            if total > ceiling:
                problems.append(
                    f"plan estimate {total:.4f} USD exceeds instruction budget "
                    f"{ceiling:.4f} USD"
                )
        return problems

    @staticmethod
    def _has_cycle(planned: PlannedPlan) -> bool:
        dependencies = {step.id: set(step.depends_on) for step in planned.steps}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            for dependency in dependencies.get(node, set()):
                if visit(dependency):
                    return True
            visiting.discard(node)
            visited.add(node)
            return False

        return any(visit(node) for node in dependencies)

    # --- conversion -------------------------------------------------------

    def _to_plan(self, instruction: str, planned: PlannedPlan) -> Plan:
        steps = [
            PlanStep(
                id=step.id,
                tool=step.tool,
                arguments=dict(step.arguments),
                depends_on=list(step.depends_on),
                estimate_usd=step.estimate_usd,
            )
            for step in planned.steps
        ]
        return Plan(
            instruction=instruction,
            steps=steps,
            assumptions=list(planned.assumptions),
            approval_mode=PlanApprovalMode(self._planner_config.approval_mode.value),
        )


def json_dumps(value: Any) -> str:
    """Return compact JSON for *value*, used inside planner prompts."""
    return json.dumps(value, sort_keys=True)

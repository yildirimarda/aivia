"""Tool framework: JSON schema, estimate, execute, safety limits and
allowlist enforcement.

Tools are an allowlisted registry with JSON schemas. Each tool declares its
input schema, reports an estimate before it runs and executes within the
configured safety limits. The allowlist comes from
:class:`~aivia.core.config.planner.PlannerConfig`; an empty allowlist permits
every registered tool, while a non-empty allowlist is exclusive.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, ClassVar, overload

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from aivia.core.config.planner import PlannerConfig
from aivia.core.errors import AiviaError, BudgetError

__all__ = [
    "SafetyLimits",
    "Tool",
    "ToolArguments",
    "ToolBudgetError",
    "ToolEstimate",
    "ToolFramework",
    "ToolNotAllowedError",
    "ToolRegistry",
    "ToolResult",
    "ToolSafetyLimitError",
    "ToolUnknownError",
    "ToolValidationError",
]


class ToolError(AiviaError):
    """Base class for every error raised by the tool framework."""

    code = "tool_error"


class ToolUnknownError(ToolError):
    """Raised when a tool name is not registered."""

    code = "tool_unknown"


class ToolNotAllowedError(ToolError):
    """Raised when a tool is not on the configured allowlist."""

    code = "tool_not_allowed"


class ToolValidationError(ToolError):
    """Raised when tool arguments fail the JSON schema validation."""

    code = "tool_validation"


class ToolSafetyLimitError(ToolError):
    """Raised when a tool call would exceed a configured safety limit."""

    code = "tool_safety_limit"


class ToolBudgetError(BudgetError):
    """Raised when a tool call would exceed the instruction budget."""

    code = "tool_budget"


@dataclass(frozen=True, slots=True)
class ToolEstimate:
    """The estimated cost of a tool call before it executes."""

    cost_usd: float = 0.0
    units: float = 1.0
    estimated: bool = True


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The result of a tool execution."""

    output: Any
    cost_usd: float = 0.0
    estimated: bool = False


class ToolArguments(BaseModel):
    """Base class for tool arguments; extra fields are rejected."""

    model_config = ConfigDict(extra="forbid")


class Tool:
    """A single allowlisted operation with a JSON schema, an estimate and an execute.

    Subclasses declare ``name`` and ``description`` and optionally an
    ``arguments_model`` (a :class:`ToolArguments` subclass) that describes the
    accepted input as a JSON schema. ``estimate`` reports the expected cost and
    ``execute`` performs the operation.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    arguments_model: ClassVar[type[ToolArguments] | None] = None

    @classmethod
    def schema(cls) -> dict[str, Any]:
        """Return the JSON schema describing this tool's arguments."""
        if cls.arguments_model is None:
            return {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
                "required": [],
            }
        return cls.arguments_model.model_json_schema()

    @classmethod
    def validate_arguments(cls, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate and coerce *arguments* against the tool's JSON schema.

        Raises :class:`ToolValidationError` when the arguments do not conform,
        for example when an unknown field is supplied or a required field is
        missing.
        """
        if cls.arguments_model is None:
            if arguments:
                raise ToolValidationError(
                    f"tool {cls.name!r} accepts no arguments",
                    details={"tool": cls.name, "arguments": arguments},
                )
            return {}
        try:
            model = cls.arguments_model.model_validate(arguments)
        except PydanticValidationError as error:
            raise ToolValidationError(
                f"invalid arguments for {cls.name!r}",
                details={"tool": cls.name, "errors": error.errors()},
            ) from error
        return model.model_dump()

    @classmethod
    def estimate(cls, arguments: dict[str, Any]) -> ToolEstimate:
        """Return the estimated cost of running this tool with *arguments*."""
        del arguments
        return ToolEstimate()

    @classmethod
    def execute(cls, arguments: dict[str, Any]) -> ToolResult:
        """Run the tool with validated *arguments* and return its result."""
        raise NotImplementedError(f"tool {cls.name!r} does not implement execute")


class ToolRegistry:
    """An ordered, collision-checked collection of tool classes."""

    def __init__(self) -> None:
        self._tools: dict[str, type[Tool]] = {}

    @overload
    def register(
        self,
        tool: None = None,
        *,
        name: str | None = None,
    ) -> Callable[[type[Tool]], type[Tool]]: ...

    @overload
    def register(
        self,
        tool: type[Tool],
        *,
        name: str | None = None,
    ) -> type[Tool]: ...

    def register(
        self,
        tool: type[Tool] | None = None,
        *,
        name: str | None = None,
    ) -> type[Tool] | Callable[[type[Tool]], type[Tool]]:
        """Register *tool* under its ``name`` (or *name* when given).

        When *tool* is omitted this returns a decorator, so it can be used as
        ``@registry.register`` or ``@registry.register(name="custom.name")``.
        A duplicate name is rejected.
        """

        def _register(cls: type[Tool], key: str) -> type[Tool]:
            if not isinstance(cls, type) or not issubclass(cls, Tool):
                raise TypeError(
                    f"tool {key!r} must be a Tool subclass, not {type(cls).__name__}"
                )
            if not key:
                raise ToolError("tool must declare a non-empty name")
            if key in self._tools:
                existing = self._tools[key]
                raise ToolError(
                    f"tool already registered: {key!r}",
                    details={
                        "tool": key,
                        "existing": existing.__name__,
                        "new": cls.__name__,
                    },
                )
            self._tools[key] = cls
            return cls

        if tool is None:

            def decorator(cls: type[Tool]) -> type[Tool]:
                resolved_name = name or cls.name
                return _register(cls, resolved_name)

            return decorator

        resolved_name = name or tool.name
        return _register(tool, resolved_name)

    def get(self, name: str) -> type[Tool]:
        """Return the tool class registered under *name*."""
        key = name.strip() if isinstance(name, str) else name
        if not isinstance(key, str) or not key:
            raise ToolUnknownError("tool name must not be empty")
        try:
            return self._tools[key]
        except KeyError:
            known = ", ".join(self.names()) or "none"
            raise ToolUnknownError(
                f"unknown tool: {key!r} (registered: {known})",
                details={"tool": key, "available": list(self.names())},
            ) from None

    def names(self) -> tuple[str, ...]:
        """Return the registered tool names in sorted order."""
        return tuple(sorted(self._tools))

    def schemas(self) -> dict[str, dict[str, Any]]:
        """Return the JSON schema of every registered tool, keyed by name."""
        return {name: self._tools[name].schema() for name in self.names()}

    def permits(self, name: str, allowlist: list[str] | None = None) -> bool:
        """Return whether *name* is registered and allowed by *allowlist*.

        When *allowlist* is omitted or empty every registered tool is
        permitted. Otherwise the name must appear in the list.
        """
        try:
            self.get(name)
        except ToolUnknownError:
            return False
        if not allowlist:
            return True
        return name in allowlist

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return name.strip() in self._tools

    def __iter__(self) -> Iterator[type[Tool]]:
        return iter(self._tools[name] for name in self.names())

    def __len__(self) -> int:
        return len(self._tools)


class SafetyLimits(BaseModel):
    """Runtime safety bounds for tool execution within one instruction.

    A value of ``None`` means the corresponding limit is disabled.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    max_steps: int = Field(default=10, ge=0)
    max_tool_calls: int = Field(default=10, ge=0)
    max_cost_usd: float | None = Field(default=None, ge=0)
    per_tool_cost_usd: float | None = Field(default=None, ge=0)


class ToolFramework:
    """Executes tools within the configured allowlist and safety limits.

    The framework enforces the planner allowlist before any call, validates
    arguments against each tool's JSON schema, checks the per-call and
    cumulative cost against the safety limits and the planner budget, and
    tracks the number of steps and tool calls executed.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        planner_config: PlannerConfig | None = None,
        safety_limits: SafetyLimits | None = None,
    ) -> None:
        self._registry = registry
        self._planner_config = planner_config or PlannerConfig()
        self._safety_limits = safety_limits or SafetyLimits()
        self._cost_usd = 0.0
        self._tool_calls = 0
        self._steps = 0

    @property
    def registry(self) -> ToolRegistry:
        """The tool registry this framework executes against."""
        return self._registry

    @property
    def planner_config(self) -> PlannerConfig:
        """The planner configuration supplying the allowlist and budget."""
        return self._planner_config

    @property
    def safety_limits(self) -> SafetyLimits:
        """The runtime safety limits in force."""
        return self._safety_limits

    @property
    def total_cost_usd(self) -> float:
        """The cumulative cost of executed tool calls so far."""
        return self._cost_usd

    @property
    def tool_calls(self) -> int:
        """The number of tool calls executed so far."""
        return self._tool_calls

    @property
    def steps(self) -> int:
        """The number of tool steps executed so far."""
        return self._steps

    def permits(self, name: str) -> bool:
        """Return whether *name* is registered and allowed by the allowlist."""
        return self._registry.permits(name, list(self._planner_config.allowed_tools))

    def schema(self, name: str) -> dict[str, Any]:
        """Return the JSON schema for *name*, raising if unknown or disallowed."""
        self._require_registered(name)
        return self._registry.get(name).schema()

    def estimate(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> ToolEstimate:
        """Return the estimate for *name* after enforcing the allowlist."""
        self._enforce_allowed(name)
        tool = self._registry.get(name)
        return tool.estimate(arguments or {})

    def execute(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """Validate, estimate, run and account for *name* within the limits."""
        self._enforce_allowed(name)
        tool = self._registry.get(name)
        arguments = tool.validate_arguments(arguments or {})
        estimate = tool.estimate(arguments)
        self._check_per_tool_cost(estimate.cost_usd)
        self._check_limits_before_call()
        result = tool.execute(arguments)
        cost_usd = result.cost_usd if result.cost_usd else estimate.cost_usd
        self._check_budget(cost_usd)
        self._cost_usd += cost_usd
        self._tool_calls += 1
        self._steps += 1
        return ToolResult(
            output=result.output,
            cost_usd=cost_usd,
            estimated=result.estimated,
        )

    def reset(self) -> None:
        """Clear the accumulated cost, step and call counters."""
        self._cost_usd = 0.0
        self._tool_calls = 0
        self._steps = 0

    # --- internal enforcement ---------------------------------------------

    def _require_registered(self, name: str) -> type[Tool]:
        try:
            return self._registry.get(name)
        except ToolUnknownError as error:
            raise ToolUnknownError(
                f"unknown tool: {name!r}",
                details={"tool": name},
            ) from error

    def _enforce_allowed(self, name: str) -> type[Tool]:
        tool = self._require_registered(name)
        if not self._planner_config.permits(name):
            raise ToolNotAllowedError(
                f"tool {name!r} is not on the allowlist",
                details={
                    "tool": name,
                    "allowlist": list(self._planner_config.allowed_tools),
                },
            )
        return tool

    def _check_limits_before_call(self) -> None:
        limits = self._safety_limits
        if limits.max_steps is not None and self._steps >= limits.max_steps:
            raise ToolSafetyLimitError(
                f"step limit reached: {self._steps} >= {limits.max_steps}",
                details={"steps": self._steps, "max_steps": limits.max_steps},
            )
        if (
            limits.max_tool_calls is not None
            and self._tool_calls >= limits.max_tool_calls
        ):
            message = (
                f"tool call limit reached: {self._tool_calls} "
                f">= {limits.max_tool_calls}"
            )
            raise ToolSafetyLimitError(
                message,
                details={
                    "tool_calls": self._tool_calls,
                    "max_tool_calls": limits.max_tool_calls,
                },
            )

    def _check_per_tool_cost(self, cost_usd: float) -> None:
        limit = self._safety_limits.per_tool_cost_usd
        if limit is not None and cost_usd > limit:
            raise ToolSafetyLimitError(
                f"tool cost {cost_usd:.4f} exceeds per-tool limit {limit:.4f} USD",
                details={"cost_usd": cost_usd, "per_tool_cost_usd": limit},
            )

    def _check_budget(self, added_cost: float) -> None:
        new_total = self._cost_usd + added_cost
        cap: float | None = None
        if self._safety_limits.max_cost_usd is not None:
            cap = self._safety_limits.max_cost_usd
        planner_ceiling = self._planner_config.budget_per_instruction
        if planner_ceiling and planner_ceiling > 0:
            cap = planner_ceiling if cap is None else min(cap, planner_ceiling)
        if cap is not None and new_total > cap:
            message = (
                f"tool call would exceed instruction budget: "
                f"{new_total:.4f} > {cap:.4f} USD"
            )
            raise ToolBudgetError(
                message,
                details={"total_usd": new_total, "cap_usd": cap},
            )

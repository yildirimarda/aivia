"""Planner configuration: limits, approval, tools, defaults and budget.

The planner turns a natural-language instruction into a validated plan of tool
calls. Its limits are configuration, never magic numbers: ``max_steps`` bounds
the number of steps a plan may contain and ``max_iterations`` bounds the
self-check repair loop.

``allowed_tools`` is the tool allowlist. An empty list means every registered
tool is permitted, which keeps a fresh install usable; a non-empty list is an
exclusive allowlist. ``approval_mode`` is ``always`` (a human must approve the
plan), ``ask`` (the default) or ``never`` (plans run without prompting).
``defaults`` holds the values applied in batch mode when the instruction leaves
a setting unspecified. ``budget`` reuses the shared budget model so the planner
honours the instruction cap and the usual overrun policy.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Any

from pydantic import ConfigDict, Field, field_validator, model_validator

from aivia.core.config.base import ConfigModel
from aivia.core.config.budget import BudgetConfig

__all__ = [
    "ApprovalMode",
    "PlannerApproval",
    "PlannerApprovalMode",
    "PlannerBudget",
    "PlannerBudgetConfig",
    "PlannerConfig",
    "PlannerDefaultsConfig",
]


class PlannerApprovalMode(StrEnum):
    """Whether a plan needs human approval before it executes."""

    ALWAYS = "always"
    ASK = "ask"
    NEVER = "never"


ApprovalMode = PlannerApprovalMode
PlannerApproval = PlannerApprovalMode

PlannerBudgetConfig = BudgetConfig
PlannerBudget = BudgetConfig

_APPROVAL_ALIASES = {
    "always": PlannerApprovalMode.ALWAYS,
    "manual": PlannerApprovalMode.ALWAYS,
    "required": PlannerApprovalMode.ALWAYS,
    "yes": PlannerApprovalMode.ALWAYS,
    "ask": PlannerApprovalMode.ASK,
    "confirm": PlannerApprovalMode.ASK,
    "prompt": PlannerApprovalMode.ASK,
    "never": PlannerApprovalMode.NEVER,
    "auto": PlannerApprovalMode.NEVER,
    "automatic": PlannerApprovalMode.NEVER,
    "no": PlannerApprovalMode.NEVER,
}

_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "max_steps": ("step_limit", "max_step_count", "steps", "step_count"),
    "max_iterations": (
        "iteration_limit",
        "max_iteration_count",
        "iterations",
        "iteration_count",
    ),
    "approval_mode": ("approval", "require_approval"),
    "allowed_tools": (
        "tools",
        "tool_allowlist",
        "allowlist",
        "allowed",
    ),
    "preview": ("show_preview", "preview_plan"),
    "defaults": ("default", "default_values"),
    "budget": ("instruction_budget", "budget_usd", "max_budget"),
}


class PlannerDefaultsConfig(ConfigModel):
    """Values applied in batch mode when the instruction omits them.

    The common plan settings are typed for discoverability. Additional
    settings are accepted and kept as attributes so a format or tool can add a
    default without a core release.
    """

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        validate_assignment=True,
    )

    format: str = ""
    language: str = ""
    output: str = ""
    output_type: str = ""
    resolution: str = ""
    aspect_ratio: str = ""
    duration_seconds: float | None = Field(default=None, ge=0)
    voice: str = ""
    style: str = ""
    quality: str = ""
    seed: int | None = None
    tool_options: dict[str, dict[str, Any]] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Return a default by name, including keys added by a module."""
        return getattr(self, key, default)

    @property
    def values(self) -> dict[str, Any]:
        """The configured defaults as a plain mapping."""
        return self.model_dump()


class PlannerConfig(ConfigModel):
    """Step, iteration, approval, tool, preview, default and budget settings."""

    enabled: bool = True
    max_steps: int = Field(default=10, ge=1)
    max_iterations: int = Field(default=3, ge=0)
    approval_mode: PlannerApprovalMode = PlannerApprovalMode.ASK
    allowed_tools: list[str] = Field(default_factory=list)
    preview: bool = True
    defaults: PlannerDefaultsConfig = Field(default_factory=PlannerDefaultsConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for canonical, aliases in _FIELD_ALIASES.items():
            for alias in aliases:
                if alias in values:
                    value = values.pop(alias)
                    values.setdefault(canonical, value)
        if values.get("defaults") is None:
            values["defaults"] = {}
        if values.get("budget") is None:
            values["budget"] = {}
        return values

    @field_validator("approval_mode", mode="before")
    @classmethod
    def _normalise_approval(cls, value: Any) -> Any:
        if isinstance(value, PlannerApprovalMode):
            return value
        if isinstance(value, bool):
            return PlannerApprovalMode.ALWAYS if value else PlannerApprovalMode.NEVER
        if isinstance(value, str):
            key = value.strip().lower()
            if key in _APPROVAL_ALIASES:
                return _APPROVAL_ALIASES[key]
        return value

    @field_validator("allowed_tools", mode="before")
    @classmethod
    def _normalise_tools(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            value = value.split(",")
        if not isinstance(value, Iterable) or isinstance(value, (bytes, Mapping)):
            return value
        tools: list[str] = []
        seen: set[str] = set()
        for entry in value:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError("allowed tool names must be non-empty strings")
            tool = entry.strip()
            if tool in seen:
                raise ValueError(f"duplicate allowed tool: {tool}")
            seen.add(tool)
            tools.append(tool)
        return tools

    @field_validator("budget", mode="before")
    @classmethod
    def _normalise_budget(cls, value: Any) -> Any:
        if value is None:
            return {}
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return {"per_instruction": value}
        return value

    @property
    def step_limit(self) -> int:
        """Alias for :attr:`max_steps`."""
        return self.max_steps

    @property
    def iteration_limit(self) -> int:
        """Alias for :attr:`max_iterations`."""
        return self.max_iterations

    @property
    def approval(self) -> PlannerApprovalMode:
        """Alias for :attr:`approval_mode`."""
        return self.approval_mode

    @property
    def require_approval(self) -> bool:
        """Whether every plan must be approved before it runs."""
        return self.approval_mode is PlannerApprovalMode.ALWAYS

    @property
    def tools(self) -> list[str]:
        """Alias for :attr:`allowed_tools`."""
        return self.allowed_tools

    @property
    def allowlist(self) -> list[str]:
        """Alias for :attr:`allowed_tools`."""
        return self.allowed_tools

    @property
    def default_arguments(self) -> PlannerDefaultsConfig:
        """Alias for :attr:`defaults`."""
        return self.defaults

    @property
    def budget_per_instruction(self) -> float:
        """The per-instruction cap from :attr:`budget`."""
        return self.budget.per_instruction

    def permits(self, tool: str) -> bool:
        """Return whether *tool* is allowed by the configured allowlist."""
        name = tool.strip()
        return not self.allowed_tools or name in self.allowed_tools

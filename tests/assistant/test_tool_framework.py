"""Tool framework: JSON schema, estimate, execute, safety limits and
allowlist enforcement."""

from __future__ import annotations

from typing import Any

import pytest

from aivia.assistant import (
    SafetyLimits,
    Tool,
    ToolArguments,
    ToolBudgetError,
    ToolError,
    ToolEstimate,
    ToolFramework,
    ToolNotAllowedError,
    ToolRegistry,
    ToolResult,
    ToolSafetyLimitError,
    ToolUnknownError,
    ToolValidationError,
)
from aivia.core.config.budget import BudgetConfig
from aivia.core.config.planner import PlannerConfig


class _CaptionArgs(ToolArguments):
    text: str
    length: int = 1


class CaptionTool(Tool):
    name = "caption"
    description = "Produce a caption of the given length."
    arguments_model = _CaptionArgs

    @classmethod
    def estimate(cls, arguments: dict[str, Any]) -> ToolEstimate:
        return ToolEstimate(
            cost_usd=0.01 * int(arguments.get("length", 1)),
            units=float(arguments.get("length", 1)),
        )

    @classmethod
    def execute(cls, arguments: dict[str, Any]) -> ToolResult:
        text = str(arguments["text"])
        length = int(arguments.get("length", 1))
        return ToolResult(output=text[:length], cost_usd=0.01 * length, estimated=False)


class _NoArgsTool(Tool):
    name = "noop"
    description = "A tool that takes no arguments."

    @classmethod
    def execute(cls, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(output="done")


class _FailingTool(Tool):
    name = "fails"
    description = "A tool whose execute raises."

    @classmethod
    def execute(cls, arguments: dict[str, Any]) -> ToolResult:
        raise RuntimeError("boom")


def _framework(
    allowed_tools: list[str] | None = None,
    safety_limits: SafetyLimits | None = None,
) -> ToolFramework:
    registry = ToolRegistry()
    registry.register(CaptionTool)
    registry.register(_NoArgsTool)
    registry.register(_FailingTool)
    config = PlannerConfig(allowed_tools=allowed_tools or [])
    return ToolFramework(registry, planner_config=config, safety_limits=safety_limits)


def test_registry_registers_and_lists_tools() -> None:
    registry = ToolRegistry()
    registry.register(CaptionTool)
    registry.register(_NoArgsTool)

    assert registry.names() == ("caption", "noop")
    assert "caption" in registry
    assert "missing" not in registry


def test_registry_get_returns_tool_class() -> None:
    registry = ToolRegistry()
    registry.register(CaptionTool)

    assert registry.get("caption") is CaptionTool


def test_registry_unknown_tool_raises() -> None:
    registry = ToolRegistry()

    with pytest.raises(ToolUnknownError):
        registry.get("does-not-exist")


def test_registry_duplicate_tool_raises() -> None:
    registry = ToolRegistry()
    registry.register(CaptionTool)

    with pytest.raises(ToolError):
        registry.register(CaptionTool)


def test_registry_register_decorator_with_custom_name() -> None:
    registry = ToolRegistry()

    @registry.register(name="custom.caption")
    class AliasCaption(CaptionTool):
        name = "alias.caption"

    assert registry.get("custom.caption") is AliasCaption
    assert "alias.caption" not in registry


def test_tool_schema_exports_json_schema() -> None:
    schema = CaptionTool.schema()

    assert schema["type"] == "object"
    assert "text" in schema["properties"]
    assert schema["properties"]["text"]["type"] == "string"
    assert schema["required"] == ["text"]
    assert schema["properties"]["length"]["default"] == 1


def test_tool_without_arguments_schema_is_empty_object() -> None:
    schema = _NoArgsTool.schema()

    assert schema == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
        "required": [],
    }


def test_validate_arguments_accepts_valid_input() -> None:
    coerced = CaptionTool.validate_arguments({"text": "hello world", "length": 5})

    assert coerced == {"text": "hello world", "length": 5}


def test_validate_arguments_coerces_defaults() -> None:
    coerced = CaptionTool.validate_arguments({"text": "hello"})

    assert coerced == {"text": "hello", "length": 1}


def test_validate_arguments_rejects_extra_fields() -> None:
    with pytest.raises(ToolValidationError) as exc_info:
        CaptionTool.validate_arguments({"text": "hello", "unknown": 1})

    assert exc_info.value.code == "tool_validation"
    assert "caption" in str(exc_info.value)


def test_validate_arguments_rejects_missing_required() -> None:
    with pytest.raises(ToolValidationError):
        CaptionTool.validate_arguments({})


def test_validate_arguments_rejects_arguments_on_noargs_tool() -> None:
    with pytest.raises(ToolValidationError):
        _NoArgsTool.validate_arguments({"text": "x"})


def test_estimate_returns_cost_and_units() -> None:
    estimate = CaptionTool.estimate({"text": "hello", "length": 3})

    assert isinstance(estimate, ToolEstimate)
    assert estimate.cost_usd == 0.03
    assert estimate.units == 3.0
    assert estimate.estimated is True


def test_execute_returns_result_and_consumes_no_limits_on_itself() -> None:
    result = CaptionTool.execute({"text": "hello world", "length": 2})

    assert isinstance(result, ToolResult)
    assert result.output == "he"
    assert result.cost_usd == 0.02
    assert result.estimated is False


def test_execute_propagates_tool_errors() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        _FailingTool.execute({})


def test_allowlist_empty_permits_every_registered_tool() -> None:
    registry = ToolRegistry()
    registry.register(CaptionTool)
    framework = ToolFramework(registry)

    assert framework.permits("caption") is True


def test_allowlist_nonempty_only_permits_listed_tools() -> None:
    registry = ToolRegistry()
    registry.register(CaptionTool)
    registry.register(_NoArgsTool)
    config = PlannerConfig(allowed_tools=["caption"])
    framework = ToolFramework(registry, planner_config=config)

    assert framework.permits("caption") is True
    assert framework.permits("noop") is False
    assert framework.permits("unknown") is False


def test_framework_enforces_allowlist_on_execute() -> None:
    framework = _framework(allowed_tools=["caption"])

    result = framework.execute("caption", {"text": "hello", "length": 1})
    assert result.output == "h"

    with pytest.raises(ToolNotAllowedError) as exc_info:
        framework.execute("noop", {})

    assert exc_info.value.code == "tool_not_allowed"
    assert "noop" in str(exc_info.value)


def test_framework_unknown_tool_rejected() -> None:
    framework = _framework()

    with pytest.raises(ToolUnknownError):
        framework.execute("does-not-exist", {})


def test_framework_estimate_enforces_allowlist() -> None:
    framework = _framework(allowed_tools=["caption"])

    assert framework.estimate("caption", {"text": "x", "length": 2}).cost_usd == 0.02

    with pytest.raises(ToolNotAllowedError):
        framework.estimate("noop", {})


def test_framework_execute_accumulates_cost_and_counts() -> None:
    framework = _framework()

    framework.execute("caption", {"text": "abc", "length": 2})
    framework.execute("caption", {"text": "abc", "length": 3})

    assert framework.tool_calls == 2
    assert framework.steps == 2
    assert framework.total_cost_usd == 0.05


def test_framework_reset_clears_counters() -> None:
    framework = _framework()
    framework.execute("caption", {"text": "abc", "length": 2})

    framework.reset()

    assert framework.total_cost_usd == 0.0
    assert framework.tool_calls == 0
    assert framework.steps == 0


def test_safety_limit_max_steps_blocks_further_calls() -> None:
    framework = _framework(safety_limits=SafetyLimits(max_steps=1))

    framework.execute("caption", {"text": "abc", "length": 1})

    with pytest.raises(ToolSafetyLimitError) as exc_info:
        framework.execute("caption", {"text": "abc", "length": 1})

    assert exc_info.value.code == "tool_safety_limit"
    assert framework.steps == 1


def test_safety_limit_max_tool_calls_blocks_further_calls() -> None:
    framework = _framework(safety_limits=SafetyLimits(max_tool_calls=2))

    framework.execute("caption", {"text": "abc", "length": 1})
    framework.execute("caption", {"text": "abc", "length": 1})

    with pytest.raises(ToolSafetyLimitError) as exc_info:
        framework.execute("caption", {"text": "abc", "length": 1})

    assert exc_info.value.code == "tool_safety_limit"
    assert framework.tool_calls == 2


def test_safety_limit_per_tool_cost_blocks_expensive_call() -> None:
    framework = _framework(safety_limits=SafetyLimits(per_tool_cost_usd=0.05))

    with pytest.raises(ToolSafetyLimitError) as exc_info:
        framework.execute("caption", {"text": "abc", "length": 10})

    assert exc_info.value.code == "tool_safety_limit"
    assert framework.tool_calls == 0


def test_safety_limit_max_cost_blocks_cumulative_spend() -> None:
    framework = _framework(safety_limits=SafetyLimits(max_cost_usd=0.05))

    framework.execute("caption", {"text": "abc", "length": 3})  # 0.03

    # The next call would take the total to 0.06, over the 0.05 cap, so it is
    # rejected before it executes and nothing is added.
    with pytest.raises(ToolBudgetError):
        framework.execute("caption", {"text": "abc", "length": 3})

    assert framework.total_cost_usd == 0.03
    assert framework.tool_calls == 1


def test_planner_budget_blocks_spend() -> None:
    registry = ToolRegistry()
    registry.register(CaptionTool)
    config = PlannerConfig(allowed_tools=[], budget=BudgetConfig(per_instruction=0.03))
    framework = ToolFramework(registry, planner_config=config)

    framework.execute("caption", {"text": "abc", "length": 3})  # 0.03

    with pytest.raises(ToolBudgetError):
        framework.execute("caption", {"text": "abc", "length": 1})


def test_invalid_arguments_raise_before_execution() -> None:
    framework = _framework()

    with pytest.raises(ToolValidationError):
        framework.execute("caption", {"text": "x", "length": "not-an-int"})

    assert framework.tool_calls == 0


def test_schemas_export_all_registered_tools() -> None:
    registry = ToolRegistry()
    registry.register(CaptionTool)
    registry.register(_NoArgsTool)

    schemas = registry.schemas()

    assert set(schemas) == {"caption", "noop"}
    assert schemas["caption"]["type"] == "object"
    assert schemas["noop"]["additionalProperties"] is False


def test_tool_errors_have_stable_codes() -> None:
    registry = ToolRegistry()

    with pytest.raises(ToolUnknownError) as exc_info:
        registry.get("missing")
    assert exc_info.value.code == "tool_unknown"

    framework = _framework(allowed_tools=["caption"])
    with pytest.raises(ToolNotAllowedError) as not_allowed_info:
        framework.execute("noop", {})
    assert not_allowed_info.value.code == "tool_not_allowed"

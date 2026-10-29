"""Repair loop for structured LLM output.

A structured call can fail validation even when the model was asked for JSON.
Instead of failing the whole stage on the first mismatch, the loop sends the
invalid response and the validation error back to the model, up to the repair
limit configured on :class:`~aivia.core.config.llm.LLMConfig`. Every attempt
still goes through :class:`~aivia.llm.structured.StructuredOutputHelper`, so a
failed attempt is priced and recorded just like a successful one.

Only :class:`~aivia.llm.structured.StructuredOutputError` is repaired. A
provider, auth or budget error is not something another prompt can fix, so it
propagates immediately. When the limit is reached the last validation error is
raised as a :class:`StructuredOutputError` carrying the attempt count and the
last response for the caller's diagnostics.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any, cast

from pydantic import BaseModel

from aivia.core.config.llm import LLMConfig, TokenPricing
from aivia.core.cost import CostTracker
from aivia.core.errors import ConfigError
from aivia.llm.cache import LLMCache
from aivia.llm.structured import (
    StructuredOutputError,
    StructuredOutputHelper,
    StructuredOutputResult,
)
from aivia.providers.llm.cost import LLM_OPERATION, LLMCostCalculator
from aivia.providers.protocols.llm import ChatMessage, LLMRequest, MessageRole

__all__ = [
    "RepairHelper",
    "RepairLoop",
    "RepairingStructuredOutput",
    "StructuredOutputRepair",
    "StructuredRepair",
    "StructuredRepairLoop",
    "build_repair_prompt",
    "complete_structured_with_repair",
    "complete_with_repair",
    "repair_loop",
    "repair_messages",
    "repair_structured",
    "repair_structured_output",
    "structured_repair",
]

_CONFIG_KEYS = frozenset(
    {
        "chain",
        "roles",
        "pricing",
        "max_repairs",
        "repair_limit",
        "max_repair_attempts",
        "repair_attempts",
        "max_repair",
        "repairs",
        "provider_chain",
        "providers",
        "order",
        "role_models",
        "models_by_role",
        "model_roles",
        "token_pricing",
        "model_pricing",
        "prices",
    }
)

PromptBuilder = Callable[[StructuredOutputError, type[BaseModel] | None], str]
"""Build the user message that asks the model to repair its response."""


def _repair_count(value: object, *, name: str = "max_repairs") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value


def _resolve_limit(
    config: LLMConfig | Mapping[str, Any] | None,
    max_repairs: int | None,
    repair_limit: int | None,
) -> tuple[int, LLMConfig | Mapping[str, Any] | None]:
    explicit: list[int] = []
    if max_repairs is not None:
        explicit.append(_repair_count(max_repairs))
    if repair_limit is not None:
        explicit.append(_repair_count(repair_limit, name="repair_limit"))
    if len(explicit) == 2 and explicit[0] != explicit[1]:
        raise ConfigError("max_repairs and repair_limit disagree")

    helper_config = config
    configured: int | None = None
    if isinstance(config, LLMConfig):
        configured = config.max_repairs
    elif isinstance(config, Mapping) and set(config) & _CONFIG_KEYS:
        llm_config = LLMConfig.from_mapping(config)
        helper_config = llm_config
        configured = llm_config.max_repairs

    if explicit:
        return explicit[0], helper_config
    if configured is not None:
        return configured, helper_config
    return LLMConfig().max_repairs, helper_config


def _schema_text(schema: type[BaseModel] | None) -> str:
    if schema is None:
        return ""
    return json.dumps(schema.model_json_schema(), indent=2, sort_keys=True)


def build_repair_prompt(
    error: StructuredOutputError,
    schema: type[BaseModel] | None = None,
) -> str:
    """Return the user message that asks for a corrected response.

    The prompt quotes the invalid response and the validation error, so the
    model can see exactly which field or JSON rule failed. The schema is
    repeated because some models lose the original response format once the
    conversation grows.
    """
    lines = [
        "Your previous response failed validation and must be repaired.",
    ]
    content = error.content.strip()
    if content:
        lines.extend(["", "Previous response:", content])
    lines.extend(["", f"Validation error: {error}"])
    if error.errors:
        details = json.dumps(error.errors, indent=2, sort_keys=True, default=str)
        lines.extend(["", "Validation details:", details])
    schema_text = _schema_text(schema)
    if schema_text:
        lines.extend(["", "Required JSON schema:", schema_text])
    lines.extend(
        [
            "",
            "Return only valid JSON that matches the schema, with no prose or "
            "markdown fences.",
        ]
    )
    return "\n".join(lines)


def repair_messages(
    messages: list[ChatMessage],
    error: StructuredOutputError,
    schema: type[BaseModel] | None = None,
    *,
    prompt_builder: PromptBuilder | None = None,
) -> list[ChatMessage]:
    """Return *messages* extended with the failed response and repair request."""
    repaired = list(messages)
    content = error.content.strip()
    if content:
        repaired.append(
            ChatMessage(role=MessageRole.ASSISTANT, content=content),
        )
    prompt = (
        build_repair_prompt(error, schema)
        if prompt_builder is None
        else prompt_builder(error, schema)
    )
    repaired.append(ChatMessage(role=MessageRole.USER, content=prompt))
    return repaired


class RepairLoop:
    """Call a structured helper and repair validation failures.

    ``config`` may be an :class:`LLMConfig`, a mapping accepted by it, or a
    mapping of model ids to :class:`TokenPricing` for a helper that only needs
    prices. The repair limit comes from ``max_repairs`` on the config unless an
    explicit ``max_repairs`` or ``repair_limit`` is passed.
    """

    def __init__(
        self,
        provider: Any,
        schema: type[BaseModel] | None = None,
        calculator: LLMCostCalculator | None = None,
        tracker: CostTracker | None = None,
        *,
        config: LLMConfig | Mapping[str, Any] | None = None,
        max_repairs: int | None = None,
        repair_limit: int | None = None,
        run_id: str = "",
        operation: str = LLM_OPERATION,
        pricing_version: str = "",
        metadata: Mapping[str, Any] | None = None,
        record_cost: bool = True,
        prompt_builder: PromptBuilder | None = None,
        cache: LLMCache | None = None,
        prompt_version: str = "",
        cache_parameters: Mapping[str, Any] | None = None,
        no_cache: bool = False,
    ) -> None:
        limit, helper_config = _resolve_limit(config, max_repairs, repair_limit)
        self._max_repairs = limit
        self._prompt_builder = prompt_builder
        self._helper = StructuredOutputHelper(
            provider,
            schema,
            calculator=calculator,
            tracker=tracker,
            config=cast("LLMConfig | Mapping[str, TokenPricing] | None", helper_config),
            run_id=run_id,
            operation=operation,
            pricing_version=pricing_version,
            metadata=metadata,
            record_cost=record_cost,
            cache=cache,
            prompt_version=prompt_version,
            cache_parameters=cache_parameters,
            no_cache=no_cache,
        )

    @property
    def helper(self) -> StructuredOutputHelper:
        """The structured helper used for every attempt."""
        return self._helper

    @property
    def provider(self) -> Any:
        """The provider this loop calls."""
        return self._helper.provider

    @property
    def schema(self) -> type[BaseModel] | None:
        """The default schema, when one was configured."""
        return self._helper.schema

    @property
    def max_repairs(self) -> int:
        """How many validation failures may be sent back to the model."""
        return self._max_repairs

    @property
    def repair_limit(self) -> int:
        """Alias for :attr:`max_repairs`."""
        return self._max_repairs

    @property
    def max_attempts(self) -> int:
        """The initial call plus every allowed repair."""
        return self._max_repairs + 1

    async def complete(
        self,
        request: LLMRequest,
        schema: type[BaseModel] | None = None,
        *,
        run_id: str | None = None,
        operation: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        tracker: CostTracker | None = None,
        record_cost: bool | None = None,
        max_repairs: int | None = None,
    ) -> StructuredOutputResult:
        """Execute *request*, repairing validation failures up to the limit."""
        if not isinstance(request, LLMRequest):
            raise TypeError("request must be an LLMRequest")
        model = self._helper.schema if schema is None else schema
        limit = self._max_repairs if max_repairs is None else _repair_count(max_repairs)
        messages = list(request.messages)
        last_error: StructuredOutputError | None = None

        for attempt in range(1, limit + 2):
            prepared = request.model_copy(update={"messages": messages})
            try:
                result = await self._helper.complete(
                    prepared,
                    model,
                    run_id=run_id,
                    operation=operation,
                    metadata=metadata,
                    tracker=tracker,
                    record_cost=record_cost,
                )
            except StructuredOutputError as error:
                last_error = error
                if attempt > limit:
                    break
                messages = repair_messages(
                    messages,
                    error,
                    model,
                    prompt_builder=self._prompt_builder,
                )
                continue
            if attempt > 1:
                return replace(result, attempts=attempt, repairs=attempt - 1)
            return result

        assert last_error is not None
        exhausted = StructuredOutputError(
            "structured output repair limit reached after "
            f"{limit + 1} attempts: {last_error}",
            content=last_error.content,
            errors=last_error.errors,
            details={
                "attempts": limit + 1,
                "repairs": limit,
                "max_repairs": limit,
                "last_error": str(last_error),
            },
        )
        exhausted.attempts = limit + 1
        exhausted.repairs = limit
        exhausted.max_repairs = limit
        exhausted.last_error = last_error  # type: ignore[attr-defined]
        exhausted.last_content = last_error.content  # type: ignore[attr-defined]
        exhausted.last_errors = last_error.errors  # type: ignore[attr-defined]
        raise exhausted from last_error

    async def run(
        self,
        request: LLMRequest,
        schema: type[BaseModel] | None = None,
        **values: Any,
    ) -> StructuredOutputResult:
        """Alias for :meth:`complete`."""
        return await self.complete(request, schema, **values)

    async def structured(
        self,
        request: LLMRequest,
        schema: type[BaseModel] | None = None,
        **values: Any,
    ) -> StructuredOutputResult:
        """Alias for :meth:`complete`."""
        return await self.complete(request, schema, **values)

    async def complete_structured(
        self,
        request: LLMRequest,
        schema: type[BaseModel] | None = None,
        **values: Any,
    ) -> StructuredOutputResult:
        """Alias for :meth:`complete`."""
        return await self.complete(request, schema, **values)

    async def __call__(
        self,
        request: LLMRequest,
        schema: type[BaseModel] | None = None,
        **values: Any,
    ) -> StructuredOutputResult:
        """Allow the loop to be awaited directly."""
        return await self.complete(request, schema, **values)


RepairHelper = RepairLoop
RepairingStructuredOutput = RepairLoop
StructuredOutputRepair = RepairLoop
StructuredRepair = RepairLoop
StructuredRepairLoop = RepairLoop


async def repair_structured(
    provider: Any,
    request: LLMRequest,
    schema: type[BaseModel] | None = None,
    calculator: LLMCostCalculator | None = None,
    tracker: CostTracker | None = None,
    *,
    config: LLMConfig | Mapping[str, Any] | None = None,
    max_repairs: int | None = None,
    repair_limit: int | None = None,
    run_id: str = "",
    operation: str = LLM_OPERATION,
    pricing_version: str = "",
    metadata: Mapping[str, Any] | None = None,
    record_cost: bool = True,
    prompt_builder: PromptBuilder | None = None,
    cache: LLMCache | None = None,
    prompt_version: str = "",
    cache_parameters: Mapping[str, Any] | None = None,
    no_cache: bool = False,
) -> StructuredOutputResult:
    """One-shot form of :class:`RepairLoop`."""
    loop = RepairLoop(
        provider,
        schema,
        calculator=calculator,
        tracker=tracker,
        config=config,
        max_repairs=max_repairs,
        repair_limit=repair_limit,
        run_id=run_id,
        operation=operation,
        pricing_version=pricing_version,
        metadata=metadata,
        record_cost=record_cost,
        prompt_builder=prompt_builder,
        cache=cache,
        prompt_version=prompt_version,
        cache_parameters=cache_parameters,
        no_cache=no_cache,
    )
    return await loop.complete(request)


repair_structured_output = repair_structured
complete_structured_with_repair = repair_structured
complete_with_repair = repair_structured
structured_repair = repair_structured
repair_loop = repair_structured

"""Validate an LLM response against a pydantic schema and record its cost.

Providers return free-form text. A structured call asks the model for JSON that
matches a pydantic schema, validates the parsed value and turns the reported
token usage into an actual cost event. Validation happens here rather than in
an adapter so every caller gets the same JSON extraction and error shape.

A paid call is recorded before the response is parsed. An invalid response is
still a call that cost money, so the ledger entry exists even when the caller
receives :class:`StructuredOutputError`; the repair loop can then ask again
without losing the cost of the failed attempt.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from aivia.core.config.llm import LLMConfig, TokenPricing
from aivia.core.cost import CostTracker
from aivia.core.errors import ConfigError, ValidationError
from aivia.core.models.cost import CostEvent
from aivia.llm.cache import LLMCache, LLMCacheKey
from aivia.providers.llm.cost import LLM_OPERATION, LLMCostCalculator, LLMTokenCost
from aivia.providers.protocols.llm import LLMRequest, LLMResponse, LLMUsage

__all__ = [
    "StructuredCompletion",
    "StructuredLLM",
    "StructuredOutput",
    "StructuredOutputError",
    "StructuredOutputHelper",
    "StructuredOutputResult",
    "StructuredResult",
    "StructuredValue",
    "complete_structured",
    "complete_with_schema",
    "generate_structured",
    "parse_structured",
    "parse_structured_response",
    "structured_complete",
    "validate_structured",
]

_FENCED_JSON = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class _LLMCompleter(Protocol):
    """The only provider surface a structured call needs."""

    async def complete(self, request: LLMRequest) -> LLMResponse: ...


class StructuredOutputError(ValidationError):
    """A response could not be parsed or did not match the pydantic schema.

    ``content`` is the raw model text and ``errors`` holds pydantic's
    machine-readable validation errors, so a repair prompt can quote exactly
    what was wrong without re-parsing the response.
    """

    code = "structured_output"

    attempts: int
    repairs: int
    max_repairs: int

    def __init__(
        self,
        message: str,
        *,
        content: str = "",
        errors: list[dict[str, Any]] | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        merged: dict[str, Any] = dict(details or {})
        if content:
            merged.setdefault("content", content)
        if errors is not None:
            merged.setdefault("errors", errors)
        super().__init__(message, details=merged)
        self.content = content
        self.errors = list(errors or ())


@dataclass(frozen=True, slots=True)
class StructuredOutputResult:
    """A validated value together with the response and its priced cost."""

    value: BaseModel
    response: LLMResponse
    cost: LLMTokenCost | None = None
    event: CostEvent | None = None
    schema: type[BaseModel] | None = None
    attempts: int = 1
    repairs: int = 0
    cache_hit: bool = False
    cache_key: str | None = None

    @property
    def output(self) -> BaseModel:
        """Alias for :attr:`value`."""
        return self.value

    @property
    def data(self) -> BaseModel:
        """Alias for :attr:`value`."""
        return self.value

    @property
    def parsed(self) -> BaseModel:
        """Alias for :attr:`value`."""
        return self.value

    @property
    def usage(self) -> LLMUsage:
        """The provider-reported token usage."""
        return self.response.usage

    @property
    def input_tokens(self) -> int:
        """The number of input tokens the provider reported."""
        return self.response.usage.input_tokens

    @property
    def output_tokens(self) -> int:
        """The number of output tokens the provider reported."""
        return self.response.usage.output_tokens

    @property
    def total_tokens(self) -> int:
        """The total number of tokens the provider reported."""
        return self.response.usage.total_tokens

    @property
    def provider(self) -> str:
        """The provider that served the call."""
        return self.response.provider

    @property
    def model(self) -> str:
        """The model that served the call."""
        return self.response.model

    @property
    def content(self) -> str:
        """The raw response text before validation."""
        return self.response.content

    @property
    def text(self) -> str:
        """Alias for :attr:`content`."""
        return self.response.content

    @property
    def cost_event(self) -> CostEvent | None:
        """The recorded cost event, when pricing was configured."""
        return self.event

    @property
    def token_cost(self) -> LLMTokenCost | None:
        """The priced token breakdown, when pricing was configured."""
        return self.cost

    @property
    def amount_usd(self) -> float:
        """The priced cost of the call, or zero when it was not priced."""
        if self.cost is not None:
            return self.cost.amount_usd
        if self.event is not None:
            return self.event.amount_usd
        return 0.0

    @property
    def cost_usd(self) -> float:
        """Alias for :attr:`amount_usd`."""
        return self.amount_usd

    @property
    def is_cache_hit(self) -> bool:
        """Whether the response was served from the LLM cache."""
        return self.cache_hit

    @property
    def cached(self) -> bool:
        """Alias for :attr:`cache_hit`."""
        return self.cache_hit

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for logs and metadata."""
        return {
            "value": self.value.model_dump(mode="json"),
            "response": self.response.model_dump(mode="json"),
            "cost": None if self.cost is None else self.cost.to_dict(),
            "event": None if self.event is None else self.event.model_dump(mode="json"),
            "cache_hit": self.cache_hit,
            "cache_key": self.cache_key,
        }


StructuredResult = StructuredOutputResult
StructuredCompletion = StructuredOutputResult
StructuredValue = StructuredOutputResult


def _ensure_schema(schema: object) -> type[BaseModel]:
    if not isinstance(schema, type) or not issubclass(schema, BaseModel):
        raise TypeError("schema must be a pydantic BaseModel subclass")
    return schema


def _schema_json(schema: type[BaseModel]) -> dict[str, Any]:
    return schema.model_json_schema()


def _load_json(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        raise json.JSONDecodeError("empty response", text, 0)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as first_error:
        fenced = _FENCED_JSON.search(text)
        if fenced is not None:
            return json.loads(fenced.group(1).strip())
        decoder = json.JSONDecoder()
        for index, character in enumerate(text):
            if character not in "{[":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            return value
        raise first_error from None


def parse_structured(
    response: LLMResponse | str | Mapping[str, Any] | list[Any],
    schema: type[BaseModel],
) -> BaseModel:
    """Parse and validate *response* against *schema*.

    *response* may be a full :class:`LLMResponse`, its raw text, or an already
    decoded JSON value. JSON wrapped in a markdown fence or surrounded by prose
    is accepted, because models often add it despite a JSON schema request.
    """
    model = _ensure_schema(schema)
    if isinstance(response, LLMResponse):
        content = response.content
        raw: object = content
    elif isinstance(response, str):
        content = response
        raw = content
    elif isinstance(response, (Mapping, list)):
        content = ""
        raw = response
    else:
        raise TypeError("response must be an LLMResponse, string or decoded JSON")

    try:
        data = _load_json(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise StructuredOutputError(
            f"response is not valid JSON for {model.__name__}: {error}",
            content=content,
        ) from error

    try:
        return model.model_validate(data)
    except PydanticValidationError as error:
        raise StructuredOutputError(
            f"response does not match {model.__name__}: {error}",
            content=content,
            errors=[dict(item) for item in error.errors(include_url=False)],
        ) from error


parse_structured_response = parse_structured
validate_structured = parse_structured


def _record_event(tracker: CostTracker, event: CostEvent) -> CostEvent:
    if isinstance(tracker, CostTracker):
        return tracker.record_actual(event)
    record_actual = getattr(tracker, "record_actual", None)
    if callable(record_actual):
        stored = record_actual(event)
    else:
        record = getattr(tracker, "record", None)
        if not callable(record):
            raise TypeError("tracker must expose record_actual or record")
        stored = record(event)
    return stored if isinstance(stored, CostEvent) else event


class StructuredOutputHelper:
    """Run one LLM call, validate its JSON and record tokens and cost.

    ``provider`` is any object with an async ``complete(request)`` method.
    ``schema`` may be given here or on :meth:`complete`. Pricing comes from an
    :class:`LLMCostCalculator` or an :class:`LLMConfig`; a ``tracker`` without
    pricing is rejected because a paid call must never be silently free. When
    no calculator, config or tracker is supplied the helper only validates and
    leaves :attr:`StructuredOutputResult.cost` empty.
    """

    def __init__(
        self,
        provider: _LLMCompleter,
        schema: type[BaseModel] | None = None,
        calculator: LLMCostCalculator | None = None,
        tracker: CostTracker | None = None,
        *,
        config: LLMConfig | Mapping[str, TokenPricing] | None = None,
        run_id: str = "",
        operation: str = LLM_OPERATION,
        pricing_version: str = "",
        metadata: Mapping[str, Any] | None = None,
        record_cost: bool = True,
        cache: LLMCache | None = None,
        prompt_version: str = "",
        cache_parameters: Mapping[str, Any] | None = None,
        no_cache: bool = False,
    ) -> None:
        if not callable(getattr(provider, "complete", None)):
            raise TypeError("provider must expose an async complete method")
        if cache is not None and not isinstance(cache, LLMCache):
            raise TypeError("cache must be an LLMCache")
        if not isinstance(prompt_version, str):
            raise TypeError("prompt_version must be a string")
        if not isinstance(no_cache, bool):
            raise TypeError("no_cache must be a boolean")

        if isinstance(schema, LLMCostCalculator):
            if calculator is None:
                calculator, schema = schema, None
        elif isinstance(schema, CostTracker) and tracker is None:
            tracker, schema = schema, None
        if isinstance(calculator, CostTracker) and tracker is None:
            tracker, calculator = calculator, None
        if isinstance(calculator, (LLMConfig, Mapping)) and config is None:
            config, calculator = calculator, None

        if calculator is not None and config is not None:
            raise TypeError("pass either calculator or config, not both")
        if tracker is not None and calculator is None and config is None:
            raise ConfigError(
                "recording LLM cost needs a calculator or pricing config",
            )
        if not isinstance(record_cost, bool):
            raise TypeError("record_cost must be a boolean")
        if not isinstance(operation, str) or not operation.strip():
            raise ConfigError("operation must be a non-empty string")

        self._provider = provider
        self._schema = None if schema is None else _ensure_schema(schema)
        self._calculator: LLMCostCalculator | None
        if calculator is not None:
            self._calculator = calculator
        elif config is not None:
            self._calculator = LLMCostCalculator(
                config,
                pricing_version=pricing_version,
            )
        else:
            self._calculator = None
        self._tracker = tracker
        self._run_id = run_id
        self._operation = operation.strip()
        self._metadata = dict(metadata or {})
        self._record_cost = record_cost
        self._cache = cache
        self._prompt_version = prompt_version.strip()
        self._cache_parameters = dict(cache_parameters or {})
        self._no_cache = no_cache

    @property
    def provider(self) -> _LLMCompleter:
        """The provider this helper calls."""
        return self._provider

    @property
    def schema(self) -> type[BaseModel] | None:
        """The default pydantic schema, when one was configured."""
        return self._schema

    @property
    def calculator(self) -> LLMCostCalculator | None:
        """The calculator used to price reported usage."""
        return self._calculator

    @property
    def tracker(self) -> CostTracker | None:
        """The tracker cost events are written through."""
        return self._tracker

    @property
    def cache(self) -> LLMCache | None:
        """The response cache used for lookups and stores, when configured."""
        return self._cache

    @property
    def prompt_version(self) -> str:
        """The prompt version included in every cache key."""
        return self._prompt_version

    def _cache_key_for(self, request: LLMRequest) -> LLMCacheKey | None:
        if self._cache is None or self._no_cache:
            return None
        return self._cache.key_for_request(
            self._provider,
            request,
            prompt_version=self._prompt_version or None,
            parameters=self._cache_parameters,
        )

    def _cost_for(
        self,
        response: LLMResponse,
        *,
        run_id: str,
        operation: str,
        metadata: Mapping[str, Any],
        record_cost: bool,
        tracker: CostTracker | None,
    ) -> tuple[LLMTokenCost | None, CostEvent | None]:
        calculator = self._calculator
        if calculator is None:
            return None, None
        cost = calculator.cost_for(response)
        event = calculator.cost_event(
            response,
            run_id=run_id,
            operation=operation,
            metadata=metadata,
        )
        if not record_cost or tracker is None:
            return cost, event
        return cost, _record_event(tracker, event)

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
    ) -> StructuredOutputResult:
        """Execute *request* and return its validated value and cost.

        When *request* has no ``response_schema`` the helper fills it from the
        pydantic schema. An explicit caller schema is left untouched so a
        caller can deliberately constrain the model more narrowly.
        """
        model = _ensure_schema(self._schema if schema is None else schema)
        if not isinstance(request, LLMRequest):
            raise TypeError("request must be an LLMRequest")

        prepared = request
        if request.response_schema is None:
            prepared = request.model_copy(
                update={"response_schema": _schema_json(model)}
            )

        cache = self._cache
        cache_key = self._cache_key_for(prepared)
        if cache is not None and cache_key is not None:
            cached = cache.lookup(
                cache_key,
                run_id=self._run_id if run_id is None else run_id,
                operation=self._operation if operation is None else operation,
            )
            if cached is not None:
                return StructuredOutputResult(
                    value=parse_structured(cached, model),
                    response=cached,
                    schema=model,
                    cache_hit=True,
                    cache_key=cache_key.digest,
                )

        response = await self._provider.complete(prepared)
        if not isinstance(response, LLMResponse):
            raise TypeError("provider complete must return an LLMResponse")

        call_metadata = dict(self._metadata)
        if metadata is not None:
            call_metadata.update(metadata)
        cost, event = self._cost_for(
            response,
            run_id=self._run_id if run_id is None else run_id,
            operation=self._operation if operation is None else operation,
            metadata=call_metadata,
            record_cost=(self._record_cost if record_cost is None else record_cost),
            tracker=self._tracker if tracker is None else tracker,
        )
        value = parse_structured(response, model)
        if cache is not None and cache_key is not None:
            cache.store(response, cache_key)
        return StructuredOutputResult(
            value=value,
            response=response,
            cost=cost,
            event=event,
            schema=model,
            cache_key=None if cache_key is None else cache_key.digest,
        )

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
        """Allow the helper to be awaited directly."""
        return await self.complete(request, schema, **values)


StructuredOutput = StructuredOutputHelper
StructuredLLM = StructuredOutputHelper


async def complete_structured(
    provider: _LLMCompleter,
    request: LLMRequest,
    schema: type[BaseModel] | None = None,
    calculator: LLMCostCalculator | None = None,
    tracker: CostTracker | None = None,
    *,
    config: LLMConfig | Mapping[str, TokenPricing] | None = None,
    run_id: str = "",
    operation: str = LLM_OPERATION,
    pricing_version: str = "",
    metadata: Mapping[str, Any] | None = None,
    record_cost: bool = True,
    cache: LLMCache | None = None,
    prompt_version: str = "",
    cache_parameters: Mapping[str, Any] | None = None,
    no_cache: bool = False,
) -> StructuredOutputResult:
    """One-shot form of :class:`StructuredOutputHelper`."""
    helper = StructuredOutputHelper(
        provider,
        schema,
        calculator=calculator,
        tracker=tracker,
        config=config,
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
    return await helper.complete(request)


structured_complete = complete_structured
complete_with_schema = complete_structured
generate_structured = complete_structured

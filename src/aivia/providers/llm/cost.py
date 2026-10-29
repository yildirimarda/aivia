"""Turn reported LLM token usage into cost events.

Adapters report token counts verbatim. This module is the only place that
applies the configured per-million input and output prices to those counts, so
an adapter never has to know a price and a caller never has to repeat the
arithmetic. The resulting :class:`~aivia.core.models.cost.CostEvent` is an
actual event: it describes a call that already happened.

The calculator never calls a provider and never enforces a budget. It only
resolves the price configured for the model and hands the event to a
:class:`~aivia.core.cost.CostTracker` when the caller asks it to record one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from aivia.core.config.llm import LLMConfig, TokenPricing
from aivia.core.cost import CostTracker
from aivia.core.errors import ConfigError
from aivia.core.models.cost import CostEvent
from aivia.providers.protocols.llm import LLMResponse, LLMUsage

__all__ = [
    "LLM_OPERATION",
    "TOKENS_PER_MILLION",
    "TOKEN_UNIT",
    "LLMCostCalculator",
    "LLMTokenCost",
    "TokenCostCalculator",
    "cost_event_for_response",
    "token_cost_for_usage",
]

TOKENS_PER_MILLION = 1_000_000
"""The denominator the configured token prices are expressed against."""

TOKEN_UNIT = "token"
"""The cost event unit kind used for reported token usage."""

LLM_OPERATION = "llm"
"""The default cost event operation for one LLM call."""


@dataclass(frozen=True, slots=True)
class LLMTokenCost:
    """The priced breakdown of one provider-reported token usage.

    ``input_cost_usd`` is the combined cost of the ordinary and cached input
    tokens, ``cached_input_cost_usd`` is the cached part alone and
    ``output_cost_usd`` is the output part; :attr:`amount_usd` is their sum.
    ``uncached_input_tokens`` is the part of ``input_tokens`` charged at the
    ordinary input price. ``cached_input_included_in_input`` records whether
    the provider already counted the cached tokens inside ``input_tokens``, so
    a caller can explain the split after the fact.
    """

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    uncached_input_tokens: int
    cached_input_included_in_input: bool
    input_cost_usd: float
    cached_input_cost_usd: float
    output_cost_usd: float
    pricing_version: str = ""

    @property
    def total_tokens(self) -> int:
        """Prompt tokens including separately reported cached input, plus output."""
        return (
            self.uncached_input_tokens + self.cached_input_tokens + self.output_tokens
        )

    @property
    def amount_usd(self) -> float:
        """The combined input and output cost."""
        return self.input_cost_usd + self.output_cost_usd

    @property
    def cost_usd(self) -> float:
        """Alias for :attr:`amount_usd`."""
        return self.amount_usd

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for logs and metadata."""
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "uncached_input_tokens": self.uncached_input_tokens,
            "cached_input_included_in_input": self.cached_input_included_in_input,
            "input_cost_usd": self.input_cost_usd,
            "cached_input_cost_usd": self.cached_input_cost_usd,
            "output_cost_usd": self.output_cost_usd,
            "amount_usd": self.amount_usd,
            "pricing_version": self.pricing_version,
        }


def _token_cost(
    usage: LLMUsage,
    pricing: TokenPricing,
    *,
    provider: str,
    model: str,
    pricing_version: str = "",
) -> LLMTokenCost:
    if usage.cached_input_included_in_input:
        cached_tokens = min(usage.cached_input_tokens, usage.input_tokens)
        uncached_tokens = usage.input_tokens - cached_tokens
    else:
        cached_tokens = usage.cached_input_tokens
        uncached_tokens = usage.input_tokens
    uncached_cost = uncached_tokens * pricing.input_per_million / TOKENS_PER_MILLION
    cached_cost = cached_tokens * pricing.cached_input_price / TOKENS_PER_MILLION
    output_cost = usage.output_tokens * pricing.output_per_million / TOKENS_PER_MILLION
    return LLMTokenCost(
        provider=provider,
        model=model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_input_tokens=cached_tokens,
        uncached_input_tokens=uncached_tokens,
        cached_input_included_in_input=usage.cached_input_included_in_input,
        input_cost_usd=uncached_cost + cached_cost,
        cached_input_cost_usd=cached_cost,
        output_cost_usd=output_cost,
        pricing_version=pricing_version,
    )


def _pricing_version(value: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(
            "pricing version must be a string",
            details={"pricing_version": value},
        )
    return value.strip()


class LLMCostCalculator:
    """Price reported LLM usage with the configured per-million token prices.

    ``config`` is either an :class:`~aivia.core.config.llm.LLMConfig` or a
    mapping of model id to :class:`~aivia.core.config.llm.TokenPricing`. A
    model without configured pricing is a :class:`ConfigError`, never a silent
    zero-cost call. ``pricing_version`` is copied onto every event so a ledger
    entry can name the price table that produced it.
    """

    def __init__(
        self,
        config: LLMConfig | Mapping[str, TokenPricing],
        *,
        pricing_version: str = "",
    ) -> None:
        if isinstance(config, LLMConfig):
            self._config = config
        elif isinstance(config, Mapping):
            self._config = LLMConfig.from_mapping({"pricing": dict(config)})
        else:
            raise TypeError("config must be an LLMConfig or a pricing mapping")
        self._pricing_version = _pricing_version(pricing_version)

    @property
    def config(self) -> LLMConfig:
        """The LLM configuration used to resolve prices."""
        return self._config

    @property
    def pricing_version(self) -> str:
        """The version copied onto every cost event."""
        return self._pricing_version

    def pricing_for(self, model: str) -> TokenPricing:
        """Return the configured token pricing for *model*."""
        return self._config.pricing_for(model)

    def cost_for(self, response: LLMResponse) -> LLMTokenCost:
        """Return the priced breakdown of *response*'s reported usage."""
        if not isinstance(response, LLMResponse):
            raise TypeError("response must be an LLMResponse")
        model = response.model.strip()
        if not model:
            raise ConfigError(
                "cannot price LLM usage without a model id",
                details={"provider": response.provider},
            )
        pricing = self.pricing_for(model)
        return _token_cost(
            response.usage,
            pricing,
            provider=response.provider,
            model=model,
            pricing_version=self._pricing_version,
        )

    def cost_event(
        self,
        response: LLMResponse,
        *,
        run_id: str = "",
        operation: str = LLM_OPERATION,
        estimated: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> CostEvent:
        """Build the actual cost event for *response*'s reported usage.

        The event's units are the total token count and its unit kind is
        ``token``. The input/output split, the cached count and both priced
        components are kept in the event metadata so a report can still explain
        the amount after the fact.
        """
        cost = self.cost_for(response)
        details: dict[str, Any] = {
            "input_tokens": cost.input_tokens,
            "output_tokens": cost.output_tokens,
            "cached_input_tokens": cost.cached_input_tokens,
            "uncached_input_tokens": cost.uncached_input_tokens,
            "cached_input_included_in_input": cost.cached_input_included_in_input,
            "input_cost_usd": cost.input_cost_usd,
            "cached_input_cost_usd": cost.cached_input_cost_usd,
            "output_cost_usd": cost.output_cost_usd,
        }
        if metadata is not None:
            details.update(metadata)
        return CostEvent(
            provider=cost.provider,
            model=cost.model,
            operation=operation,
            units=float(cost.total_tokens),
            unit_kind=TOKEN_UNIT,
            amount_usd=cost.amount_usd,
            estimated=estimated,
            run_id=run_id,
            pricing_version=cost.pricing_version,
            metadata=details,
        )

    def record(
        self,
        response: LLMResponse,
        tracker: CostTracker,
        *,
        run_id: str = "",
        operation: str = LLM_OPERATION,
        estimated: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> CostEvent:
        """Record the cost event for *response* and return the stored event."""
        if not isinstance(tracker, CostTracker):
            raise TypeError("tracker must be a CostTracker")
        event = self.cost_event(
            response,
            run_id=run_id,
            operation=operation,
            estimated=estimated,
            metadata=metadata,
        )
        if estimated:
            return tracker.record_estimate(event)
        return tracker.record_actual(event)

    def record_event(
        self,
        response: LLMResponse,
        tracker: CostTracker,
        **values: Any,
    ) -> CostEvent:
        """Alias for :meth:`record`."""
        return self.record(response, tracker, **values)


TokenCostCalculator = LLMCostCalculator
"""Alias for :class:`LLMCostCalculator`."""


def token_cost_for_usage(
    usage: LLMUsage,
    pricing: TokenPricing,
    *,
    provider: str = "",
    model: str = "",
    pricing_version: str = "",
) -> LLMTokenCost:
    """Price one reported usage against an already resolved price."""
    if not isinstance(usage, LLMUsage):
        raise TypeError("usage must be an LLMUsage")
    if not isinstance(pricing, TokenPricing):
        raise TypeError("pricing must be TokenPricing")
    return _token_cost(
        usage,
        pricing,
        provider=provider,
        model=model,
        pricing_version=_pricing_version(pricing_version),
    )


def cost_event_for_response(
    response: LLMResponse,
    config: LLMConfig | Mapping[str, TokenPricing],
    *,
    pricing_version: str = "",
    run_id: str = "",
    operation: str = LLM_OPERATION,
    estimated: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> CostEvent:
    """Build a cost event for *response* without constructing a calculator."""
    return LLMCostCalculator(
        config,
        pricing_version=pricing_version,
    ).cost_event(
        response,
        run_id=run_id,
        operation=operation,
        estimated=estimated,
        metadata=metadata,
    )

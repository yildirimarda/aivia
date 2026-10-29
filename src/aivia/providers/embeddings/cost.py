"""Turn reported embedding usage into cost events.

Embedding providers report how many input tokens they embedded. This module is
the only place that applies a per-million-token price to that count, so an
adapter never knows a price and a caller never repeats the arithmetic. A price
can come from an explicit model table or, as a fallback, from the provider's
metadata and the router's ``pricing_overrides``.

The calculator never calls a provider and never enforces a budget. It resolves
a price and, when asked, records the resulting actual
:class:`~aivia.core.models.cost.CostEvent` through a
:class:`~aivia.core.cost.CostTracker`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from aivia.core.cost import CostTracker
from aivia.core.models.cost import CostEvent
from aivia.core.pricing import PriceSource, PricingResolver, ResolvedPrice
from aivia.providers.base import Provider
from aivia.providers.protocols.embeddings import EmbeddingResponse

__all__ = [
    "EMBEDDING_OPERATION",
    "TOKENS_PER_MILLION",
    "TOKEN_UNIT",
    "EmbeddingCostCalculator",
    "EmbeddingTokenCost",
    "embedding_cost_event",
]

EMBEDDING_OPERATION = "embeddings"
"""The cost event operation for one embeddings request."""

TOKEN_UNIT = "token"
"""The cost event unit kind used for reported embedding token usage."""

TOKENS_PER_MILLION = 1_000_000
"""The denominator the configured token prices are expressed against."""


@dataclass(frozen=True, slots=True)
class EmbeddingTokenCost:
    """The priced breakdown of one embeddings response."""

    provider: str
    model: str
    input_tokens: int
    amount_usd: float
    pricing_version: str = ""

    @property
    def cost_usd(self) -> float:
        """Alias for :attr:`amount_usd`."""
        return self.amount_usd


def _prices(value: Mapping[str, float] | None) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("prices must be a mapping of model id to USD per million")
    prices: dict[str, float] = {}
    for model, price in value.items():
        if not isinstance(model, str) or not model.strip():
            raise ValueError("price model ids must be non-empty strings")
        if isinstance(price, bool) or not isinstance(price, (int, float)):
            raise TypeError(f"price for {model!r} must be a number")
        amount = float(price)
        if not math.isfinite(amount) or amount < 0:
            raise ValueError(f"price for {model!r} must be finite and non-negative")
        prices[model.strip()] = amount
    return prices


class EmbeddingCostCalculator:
    """Price embedding token usage and record actual cost events.

    ``prices`` maps a model id to USD per million input tokens. When a response
    names a model without an entry, the provider's metadata price for
    ``embeddings`` is used, honouring ``overrides`` keyed by provider name.
    """

    def __init__(
        self,
        prices: Mapping[str, float] | None = None,
        *,
        overrides: Mapping[str, float] | None = None,
        pricing_version: str = "",
    ) -> None:
        self._prices = _prices(prices)
        self._resolver = PricingResolver(
            overrides,
            pricing_version=pricing_version,
        )

    @property
    def prices(self) -> dict[str, float]:
        """A copy of the explicit model price table."""
        return dict(self._prices)

    @property
    def resolver(self) -> PricingResolver:
        """The resolver used for provider metadata and overrides."""
        return self._resolver

    @property
    def pricing_version(self) -> str:
        """The version copied onto every cost event."""
        return self._resolver.pricing_version

    def pricing_for(self, model: str) -> float | None:
        """Return the explicit per-million price for *model*, if configured."""
        return self._prices.get(model.strip())

    def resolve(
        self,
        provider: Provider,
        response: EmbeddingResponse,
        *,
        provider_name: str | None = None,
    ) -> ResolvedPrice:
        """Return the price of *response*'s reported embedding usage."""
        if not isinstance(response, EmbeddingResponse):
            raise TypeError("response must be an EmbeddingResponse")
        name = (provider_name or provider.name).strip()
        if not name:
            raise ValueError("provider name must be non-empty")
        tokens = float(response.usage.input_tokens)
        model = response.model.strip()
        price = self._prices.get(model)
        if price is not None:
            return ResolvedPrice(
                provider=name,
                operation=EMBEDDING_OPERATION,
                units=tokens,
                unit_kind=TOKEN_UNIT,
                billed_units=tokens / TOKENS_PER_MILLION,
                amount_usd=tokens * price / TOKENS_PER_MILLION,
                source=PriceSource.OVERRIDE,
                price_unit="million_tokens",
                units_per_price=TOKENS_PER_MILLION,
                pricing_version=self.pricing_version,
            )
        if tokens == 0:
            return ResolvedPrice(
                provider=name,
                operation=EMBEDDING_OPERATION,
                units=0.0,
                unit_kind=TOKEN_UNIT,
                billed_units=0.0,
                amount_usd=0.0,
                source=PriceSource.METADATA,
                price_unit=TOKEN_UNIT,
                pricing_version=self.pricing_version,
            )
        return self._resolver.resolve(
            provider,
            EMBEDDING_OPERATION,
            tokens,
            unit_kind=TOKEN_UNIT,
            provider_name=name,
        )

    def cost_for(
        self,
        provider: Provider,
        response: EmbeddingResponse,
        *,
        provider_name: str | None = None,
    ) -> EmbeddingTokenCost:
        """Return the priced breakdown of *response*."""
        price = self.resolve(
            provider,
            response,
            provider_name=provider_name,
        )
        return EmbeddingTokenCost(
            provider=price.provider,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            amount_usd=price.amount_usd,
            pricing_version=price.pricing_version,
        )

    def cost_event(
        self,
        provider: Provider,
        response: EmbeddingResponse,
        *,
        provider_name: str | None = None,
        run_id: str = "",
        operation: str = EMBEDDING_OPERATION,
        estimated: bool = False,
        metadata: Mapping[str, Any] | None = None,
        price: ResolvedPrice | None = None,
    ) -> CostEvent:
        """Build the actual cost event for *response*'s reported usage."""
        resolved = (
            price
            if price is not None
            else self.resolve(
                provider,
                response,
                provider_name=provider_name,
            )
        )
        details: dict[str, Any] = {
            "input_tokens": response.usage.input_tokens,
            "price_unit": resolved.price_unit,
            "price_source": resolved.source.value,
        }
        if resolved.missing:
            details["missing_price"] = True
        for key in ("batches", "cache", "cache_hits", "cache_misses"):
            value = response.metadata.get(key)
            if value is not None:
                details[key] = value
        if metadata is not None:
            details.update(metadata)
        return CostEvent(
            provider=resolved.provider,
            model=response.model,
            operation=operation,
            units=resolved.units,
            unit_kind=resolved.unit_kind,
            amount_usd=resolved.amount_usd,
            estimated=estimated,
            run_id=run_id,
            pricing_version=resolved.pricing_version,
            metadata=details,
        )

    def record(
        self,
        provider: Provider,
        response: EmbeddingResponse,
        tracker: CostTracker,
        *,
        provider_name: str | None = None,
        run_id: str = "",
        operation: str = EMBEDDING_OPERATION,
        estimated: bool = False,
        metadata: Mapping[str, Any] | None = None,
        price: ResolvedPrice | None = None,
    ) -> CostEvent:
        """Record the cost event for *response* and return the stored event."""
        if not isinstance(tracker, CostTracker):
            raise TypeError("tracker must be a CostTracker")
        event = self.cost_event(
            provider,
            response,
            provider_name=provider_name,
            run_id=run_id,
            operation=operation,
            estimated=estimated,
            metadata=metadata,
            price=price,
        )
        if estimated:
            return tracker.record_estimate(event)
        return tracker.record_actual(event)


def embedding_cost_event(
    provider: Provider,
    response: EmbeddingResponse,
    *,
    prices: Mapping[str, float] | None = None,
    overrides: Mapping[str, float] | None = None,
    pricing_version: str = "",
    **values: Any,
) -> CostEvent:
    """Build one embeddings cost event without constructing a calculator."""
    return EmbeddingCostCalculator(
        prices,
        overrides=overrides,
        pricing_version=pricing_version,
    ).cost_event(provider, response, **values)

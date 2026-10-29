"""Price one routed search call and record it in the cost ledger.

Search providers charge per request, not per returned page. The price comes
from the provider's :class:`~aivia.core.models.provider.ProviderMeta` metadata
for the ``search`` capability, with a configured per-provider override winning
when an operator corrected a stale or missing upstream price. A missing price
still records a zero-cost event and emits
:class:`~aivia.core.pricing.MissingPriceWarning`, so a search never disappears
from the ledger just because its price was not configured.

The calculator only resolves prices and builds events. The service calls it
after a successful routed request, so an event always names the provider that
actually served the call.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aivia.core.cost import CostTracker
from aivia.core.models.cost import CostEvent
from aivia.core.models.provider import ProviderMeta
from aivia.core.pricing import PricingResolver, ResolvedPrice
from aivia.providers.base import Provider
from aivia.providers.protocols.search import SearchResponse

__all__ = [
    "SEARCH_OPERATION",
    "SEARCH_UNIT",
    "SearchCost",
    "SearchCostCalculator",
    "SearchPrice",
    "search_cost_event",
]

SEARCH_OPERATION = "search"
"""The cost event operation for one search request."""

SEARCH_UNIT = "call"
"""The default cost unit for a search request."""

SearchPrice = ResolvedPrice
"""Alias for the resolved price of one search call."""

SearchCost = ResolvedPrice
"""Alias for the resolved cost of one search call."""


class SearchCostCalculator:
    """Resolve search prices and build or record their cost events."""

    def __init__(
        self,
        overrides: Mapping[str, float] | None = None,
        *,
        pricing_version: str = "",
    ) -> None:
        self._resolver = PricingResolver(
            overrides,
            pricing_version=pricing_version,
        )

    @property
    def resolver(self) -> PricingResolver:
        """The resolver used for every price lookup."""
        return self._resolver

    @property
    def pricing_version(self) -> str:
        """The version copied onto every cost event."""
        return self._resolver.pricing_version

    def resolve(
        self,
        provider: Provider | ProviderMeta,
        *,
        provider_name: str | None = None,
        units: float = 1.0,
        unit_kind: str | None = None,
    ) -> ResolvedPrice:
        """Return the price of one search request served by *provider*."""
        return self._resolver.resolve(
            provider,
            SEARCH_OPERATION,
            units,
            unit_kind=unit_kind,
            provider_name=provider_name,
        )

    def cost_for(
        self,
        provider: Provider | ProviderMeta,
        *,
        provider_name: str | None = None,
        units: float = 1.0,
        unit_kind: str | None = None,
    ) -> ResolvedPrice:
        """Alias for :meth:`resolve`."""
        return self.resolve(
            provider,
            provider_name=provider_name,
            units=units,
            unit_kind=unit_kind,
        )

    def cost_event(
        self,
        provider: Provider | ProviderMeta,
        *,
        provider_name: str | None = None,
        response: SearchResponse | None = None,
        run_id: str = "",
        operation: str = SEARCH_OPERATION,
        estimated: bool = False,
        units: float = 1.0,
        unit_kind: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        price: ResolvedPrice | None = None,
    ) -> CostEvent:
        """Build the cost event for one search request.

        *price* lets a caller reuse a resolution it already made, so a missing
        price warning is emitted once instead of twice. When *response* is
        given, its query, result count and request id are copied into the event
        metadata for provenance.
        """
        if response is not None and not isinstance(response, SearchResponse):
            raise TypeError("response must be a SearchResponse or None")
        resolved = (
            price
            if price is not None
            else self.resolve(
                provider,
                provider_name=provider_name,
                units=units,
                unit_kind=unit_kind,
            )
        )
        details: dict[str, Any] = {
            "billed_units": resolved.billed_units,
            "price_unit": resolved.price_unit,
            "price_source": resolved.source.value,
        }
        if resolved.missing:
            details["missing_price"] = True
        if response is not None:
            details["query"] = response.query
            details["result_count"] = len(response.results)
            if response.request_id:
                details["request_id"] = response.request_id
        if metadata is not None:
            details.update(metadata)
        return CostEvent(
            provider=resolved.provider,
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
        provider: Provider | ProviderMeta,
        tracker: CostTracker,
        *,
        provider_name: str | None = None,
        response: SearchResponse | None = None,
        run_id: str = "",
        operation: str = SEARCH_OPERATION,
        estimated: bool = False,
        units: float = 1.0,
        unit_kind: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        price: ResolvedPrice | None = None,
    ) -> CostEvent:
        """Record the cost event for one search request and return it."""
        if not isinstance(tracker, CostTracker):
            raise TypeError("tracker must be a CostTracker")
        event = self.cost_event(
            provider,
            provider_name=provider_name,
            response=response,
            run_id=run_id,
            operation=operation,
            estimated=estimated,
            units=units,
            unit_kind=unit_kind,
            metadata=metadata,
            price=price,
        )
        if estimated:
            return tracker.record_estimate(event)
        return tracker.record_actual(event)

    def record_event(
        self,
        provider: Provider | ProviderMeta,
        tracker: CostTracker,
        **values: Any,
    ) -> CostEvent:
        """Alias for :meth:`record`."""
        return self.record(provider, tracker, **values)


def search_cost_event(
    provider: Provider | ProviderMeta,
    *,
    overrides: Mapping[str, float] | None = None,
    pricing_version: str = "",
    **values: Any,
) -> CostEvent:
    """Build one search cost event without constructing a calculator."""
    return SearchCostCalculator(
        overrides,
        pricing_version=pricing_version,
    ).cost_event(provider, **values)

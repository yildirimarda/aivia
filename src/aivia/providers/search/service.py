"""Route a search request through the provider router and track its cost.

The service is the only place a search call is executed. It asks the router for
a provider, runs the typed ``search`` operation with fallback, normalizes and
de-duplicates the result, then records one actual cost event for the provider
that served the call. Callers never choose a vendor and never handle a vendor
payload directly.

Normalization happens before the cost event is built, so the event can report
both what the provider returned and how many unique pages survived de-dupe.
The last selection, price and event remain available on the service for a
caller that needs to explain the call without re-resolving anything.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from aivia.core.cost import CostTracker
from aivia.core.models.cost import CostEvent
from aivia.core.pricing import ResolvedPrice
from aivia.core.router import Router, RouteSelection
from aivia.providers.base import Provider
from aivia.providers.protocols.search import (
    CAPABILITY_SEARCH,
    SearchProvider,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from aivia.providers.search.cost import SEARCH_OPERATION, SearchCostCalculator
from aivia.providers.search.normalize import SearchNormalizer

__all__ = [
    "RoutedSearch",
    "SearchClient",
    "SearchExecutor",
    "SearchOutcome",
    "SearchService",
    "search",
]


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    """A normalized response together with the call's price and ledger event."""

    response: SearchResponse
    event: CostEvent | None = None
    price: ResolvedPrice | None = None
    selection: RouteSelection | None = None

    @property
    def provider(self) -> str:
        """The provider that served the call."""
        return self.response.provider

    @property
    def results(self) -> list[SearchResult]:
        """The normalized results."""
        return list(self.response.results)

    @property
    def amount_usd(self) -> float:
        """The recorded cost, or zero when no cost was recorded."""
        if self.event is not None:
            return self.event.amount_usd
        if self.price is not None:
            return self.price.amount_usd
        return 0.0


class SearchService:
    """Execute searches through a :class:`~aivia.core.router.Router`.

    ``tracker`` is optional so a caller can run an offline search without a
    database; when it is present every successful routed call records an actual
    cost event. A tracker may also be passed per call to override the default.
    ``pricing_overrides`` defaults to the router's configured overrides for the
    ``search`` capability, so selection and cost use the same price table.
    """

    def __init__(
        self,
        router: Router,
        *,
        tracker: CostTracker | None = None,
        cost_tracker: CostTracker | None = None,
        pricing_overrides: Mapping[str, float] | None = None,
        pricing_version: str = "",
        operation: str = SEARCH_OPERATION,
        normalizer: SearchNormalizer | None = None,
    ) -> None:
        if not isinstance(router, Router):
            raise TypeError("router must be a Router")
        if tracker is not None and cost_tracker is not None:
            raise TypeError("pass either tracker or cost_tracker, not both")
        active = tracker if tracker is not None else cost_tracker
        if active is not None and not isinstance(active, CostTracker):
            raise TypeError("tracker must be a CostTracker")
        if pricing_overrides is None:
            overrides: Mapping[str, float] = router.config_for(
                CAPABILITY_SEARCH
            ).pricing_overrides
        else:
            if not isinstance(pricing_overrides, Mapping):
                raise TypeError("pricing_overrides must be a mapping")
            overrides = pricing_overrides
        if not isinstance(operation, str) or not operation.strip():
            raise ValueError("operation must be a non-empty string")
        self._router = router
        self._tracker = active
        self._operation = operation.strip()
        self._normalizer = normalizer or SearchNormalizer()
        self._calculator = SearchCostCalculator(
            overrides,
            pricing_version=pricing_version,
        )
        self._last_selection: RouteSelection | None = None
        self._last_event: CostEvent | None = None
        self._last_price: ResolvedPrice | None = None

    @property
    def router(self) -> Router:
        """The router every call goes through."""
        return self._router

    @property
    def tracker(self) -> CostTracker | None:
        """The default cost tracker, when one was configured."""
        return self._tracker

    @property
    def calculator(self) -> SearchCostCalculator:
        """The calculator that prices and records search calls."""
        return self._calculator

    @property
    def normalizer(self) -> SearchNormalizer:
        """The normalization policy applied to every response."""
        return self._normalizer

    @property
    def last_selection(self) -> RouteSelection | None:
        """The most recent successful router selection."""
        return self._last_selection

    @property
    def last_event(self) -> CostEvent | None:
        """The most recent recorded cost event."""
        return self._last_event

    @property
    def last_price(self) -> ResolvedPrice | None:
        """The most recent resolved price."""
        return self._last_price

    async def search(
        self,
        request: SearchRequest,
        *,
        run_id: str = "",
        tracker: CostTracker | None = None,
        operation: str | None = None,
        estimated: bool = False,
        metadata: Mapping[str, Any] | None = None,
        record_cost: bool = True,
        dedupe: bool = True,
    ) -> SearchResponse:
        """Run *request* through the router and return a normalized response.

        A retryable provider failure falls back to the next candidate through
        :meth:`Router.execute`. The cost event names the provider that finally
        succeeded, never a provider that failed before it. ``dedupe=False``
        keeps the provider's result order while still validating each result.
        """
        if not isinstance(request, SearchRequest):
            raise TypeError("request must be a SearchRequest")
        if tracker is not None and not isinstance(tracker, CostTracker):
            raise TypeError("tracker must be a CostTracker")
        active = tracker if tracker is not None else self._tracker
        operation_name = self._operation if operation is None else operation.strip()
        if not operation_name:
            raise ValueError("operation must be a non-empty string")

        async def run(provider: Provider) -> SearchResponse:
            return await cast(SearchProvider, provider).search(request)

        response = await self._router.execute(CAPABILITY_SEARCH, run)
        selection = self._router.last_selection
        if selection is None:
            raise RuntimeError("router completed a search without a selection")
        provider_name = response.provider or selection.name
        normalized = self._normalizer.response(
            response,
            provider=provider_name,
            dedupe=dedupe,
        )
        self._last_selection = selection
        self._last_event = None
        self._last_price = None
        if record_cost and active is not None:
            price = self._calculator.resolve(
                selection.provider,
                provider_name=selection.name,
            )
            event = self._calculator.record(
                selection.provider,
                active,
                provider_name=selection.name,
                response=normalized,
                run_id=run_id,
                operation=operation_name,
                estimated=estimated,
                metadata=metadata,
                price=price,
            )
            self._last_price = price
            self._last_event = event
        return normalized

    async def search_with_cost(
        self,
        request: SearchRequest,
        *,
        run_id: str = "",
        tracker: CostTracker | None = None,
        operation: str | None = None,
        estimated: bool = False,
        metadata: Mapping[str, Any] | None = None,
        record_cost: bool = True,
        dedupe: bool = True,
    ) -> SearchOutcome:
        """Run :meth:`search` and return the response with its price and event."""
        response = await self.search(
            request,
            run_id=run_id,
            tracker=tracker,
            operation=operation,
            estimated=estimated,
            metadata=metadata,
            record_cost=record_cost,
            dedupe=dedupe,
        )
        return SearchOutcome(
            response=response,
            event=self._last_event,
            price=self._last_price,
            selection=self._last_selection,
        )


SearchClient = SearchService
"""Alias for :class:`SearchService`."""

SearchExecutor = SearchService
"""Alias for :class:`SearchService`."""

RoutedSearch = SearchService
"""Alias for :class:`SearchService`."""


async def search(
    request: SearchRequest,
    router: Router,
    *,
    tracker: CostTracker | None = None,
    pricing_overrides: Mapping[str, float] | None = None,
    pricing_version: str = "",
    run_id: str = "",
    operation: str | None = None,
    estimated: bool = False,
    metadata: Mapping[str, Any] | None = None,
    record_cost: bool = True,
    dedupe: bool = True,
) -> SearchResponse:
    """Execute one search without constructing a long-lived service."""
    service = SearchService(
        router,
        tracker=tracker,
        pricing_overrides=pricing_overrides,
        pricing_version=pricing_version,
    )
    return await service.search(
        request,
        run_id=run_id,
        operation=operation,
        estimated=estimated,
        metadata=metadata,
        record_cost=record_cost,
        dedupe=dedupe,
    )

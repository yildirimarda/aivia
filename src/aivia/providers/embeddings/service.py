"""Route an embedding request through the provider router.

The service is the only place an embedding call is executed. It asks the
router for a provider, optionally wraps it in the per-input cache, runs the
typed ``embed`` operation with fallback, then records one actual cost event for
the provider that served the call. Callers never choose a vendor and never
handle a vendor payload directly.
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
from aivia.providers.embeddings.cache import CachedEmbeddingProvider, EmbeddingCache
from aivia.providers.embeddings.cost import (
    EMBEDDING_OPERATION,
    EmbeddingCostCalculator,
)
from aivia.providers.protocols.embeddings import (
    CAPABILITY_EMBEDDINGS,
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResponse,
)

__all__ = [
    "EmbeddingOutcome",
    "EmbeddingService",
    "embed",
]


@dataclass(frozen=True, slots=True)
class EmbeddingOutcome:
    """A response together with the call's price, ledger event and selection."""

    response: EmbeddingResponse
    event: CostEvent | None = None
    price: ResolvedPrice | None = None
    selection: RouteSelection | None = None

    @property
    def provider(self) -> str:
        """The provider that served the call."""
        return self.response.provider

    @property
    def vectors(self) -> list[list[float]]:
        """Every vector in response order."""
        return self.response.vectors

    @property
    def amount_usd(self) -> float:
        """The recorded cost, or zero when no cost was recorded."""
        if self.event is not None:
            return self.event.amount_usd
        if self.price is not None:
            return self.price.amount_usd
        return 0.0


class EmbeddingService:
    """Execute embedding calls through a :class:`~aivia.core.router.Router`.

    ``prices`` maps a model id to USD per million input tokens. When a model is
    absent from that table, the provider's metadata and the router's
    ``pricing_overrides`` are used. ``cache`` enables per-input caching; pass
    ``None`` to always call the provider.
    """

    def __init__(
        self,
        router: Router,
        *,
        tracker: CostTracker | None = None,
        cost_tracker: CostTracker | None = None,
        prices: Mapping[str, float] | None = None,
        pricing_overrides: Mapping[str, float] | None = None,
        pricing_version: str = "",
        operation: str = EMBEDDING_OPERATION,
        cache: EmbeddingCache | None = None,
    ) -> None:
        if not isinstance(router, Router):
            raise TypeError("router must be a Router")
        if tracker is not None and cost_tracker is not None:
            raise TypeError("pass either tracker or cost_tracker, not both")
        active = tracker if tracker is not None else cost_tracker
        if active is not None and not isinstance(active, CostTracker):
            raise TypeError("tracker must be a CostTracker")
        if cache is not None and not isinstance(cache, EmbeddingCache):
            raise TypeError("cache must be an EmbeddingCache")
        if pricing_overrides is None:
            overrides: Mapping[str, float] = router.config_for(
                CAPABILITY_EMBEDDINGS
            ).pricing_overrides
        else:
            if not isinstance(pricing_overrides, Mapping):
                raise TypeError("pricing_overrides must be a mapping")
            overrides = pricing_overrides
        if not isinstance(operation, str) or not operation.strip():
            raise ValueError("operation must be a non-empty string")
        self._router = router
        self._tracker = active
        self._cache = cache
        self._operation = operation.strip()
        self._calculator = EmbeddingCostCalculator(
            prices,
            overrides=overrides,
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
    def cache(self) -> EmbeddingCache | None:
        """The per-input cache, when one was configured."""
        return self._cache

    @property
    def calculator(self) -> EmbeddingCostCalculator:
        """The calculator that prices and records embedding calls."""
        return self._calculator

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

    async def embed(
        self,
        request: EmbeddingRequest,
        *,
        run_id: str = "",
        tracker: CostTracker | None = None,
        operation: str | None = None,
        estimated: bool = False,
        metadata: Mapping[str, Any] | None = None,
        record_cost: bool = True,
        use_cache: bool = True,
    ) -> EmbeddingResponse:
        """Run *request* through the router and return validated vectors.

        A retryable provider failure falls back to the next candidate through
        :meth:`Router.execute`. The cost event names the provider that finally
        succeeded, never a provider that failed before it. ``use_cache=False``
        bypasses the configured cache for this call.
        """
        if not isinstance(request, EmbeddingRequest):
            raise TypeError("request must be an EmbeddingRequest")
        if tracker is not None and not isinstance(tracker, CostTracker):
            raise TypeError("tracker must be a CostTracker")
        active = tracker if tracker is not None else self._tracker
        operation_name = self._operation if operation is None else operation.strip()
        if not operation_name:
            raise ValueError("operation must be a non-empty string")

        async def run(provider: Provider) -> EmbeddingResponse:
            target: Provider = provider
            if self._cache is not None and use_cache:
                target = CachedEmbeddingProvider(provider, self._cache)
            return await cast(EmbeddingProvider, target).embed(request)

        response = await self._router.execute(CAPABILITY_EMBEDDINGS, run)
        selection = self._router.last_selection
        if selection is None:
            raise RuntimeError("router completed an embedding call without a selection")
        self._last_selection = selection
        self._last_event = None
        self._last_price = None
        if record_cost and active is not None:
            price = self._calculator.resolve(
                selection.provider,
                response,
                provider_name=selection.name,
            )
            event = self._calculator.record(
                selection.provider,
                response,
                active,
                provider_name=selection.name,
                run_id=run_id,
                operation=operation_name,
                estimated=estimated,
                metadata=metadata,
                price=price,
            )
            self._last_price = price
            self._last_event = event
        return response

    async def embed_with_cost(
        self,
        request: EmbeddingRequest,
        *,
        run_id: str = "",
        tracker: CostTracker | None = None,
        operation: str | None = None,
        estimated: bool = False,
        metadata: Mapping[str, Any] | None = None,
        record_cost: bool = True,
        use_cache: bool = True,
    ) -> EmbeddingOutcome:
        """Run :meth:`embed` and return the response with its price and event."""
        response = await self.embed(
            request,
            run_id=run_id,
            tracker=tracker,
            operation=operation,
            estimated=estimated,
            metadata=metadata,
            record_cost=record_cost,
            use_cache=use_cache,
        )
        return EmbeddingOutcome(
            response=response,
            event=self._last_event,
            price=self._last_price,
            selection=self._last_selection,
        )


async def embed(
    request: EmbeddingRequest,
    router: Router,
    *,
    tracker: CostTracker | None = None,
    prices: Mapping[str, float] | None = None,
    pricing_overrides: Mapping[str, float] | None = None,
    pricing_version: str = "",
    cache: EmbeddingCache | None = None,
    run_id: str = "",
    operation: str | None = None,
    estimated: bool = False,
    metadata: Mapping[str, Any] | None = None,
    record_cost: bool = True,
    use_cache: bool = True,
) -> EmbeddingResponse:
    """Execute one embedding call without constructing a long-lived service."""
    service = EmbeddingService(
        router,
        tracker=tracker,
        prices=prices,
        pricing_overrides=pricing_overrides,
        pricing_version=pricing_version,
        cache=cache,
    )
    return await service.embed(
        request,
        run_id=run_id,
        operation=operation,
        estimated=estimated,
        metadata=metadata,
        record_cost=record_cost,
        use_cache=use_cache,
    )

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

from sqlalchemy import Engine

from aivia.core.config.capability import CapabilityChainConfig
from aivia.core.config.embeddings import EmbeddingsConfig
from aivia.core.cost import CostTracker
from aivia.core.models.cost import CostEvent
from aivia.core.pricing import ResolvedPrice
from aivia.core.router import Router, RouteSelection
from aivia.providers.base import Provider
from aivia.providers.embeddings.batch import batch_inputs
from aivia.providers.embeddings.cache import CachedEmbeddingProvider, EmbeddingCache
from aivia.providers.embeddings.cost import (
    EMBEDDING_OPERATION,
    EmbeddingCostCalculator,
)
from aivia.providers.protocols.embeddings import (
    CAPABILITY_EMBEDDINGS,
    Embedding,
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)

__all__ = [
    "EmbeddingOutcome",
    "EmbeddingService",
    "embed",
    "embedding_service_from_config",
    "service_from_config",
]


def _embedding_config(
    config: EmbeddingsConfig | Mapping[str, Any] | None,
) -> EmbeddingsConfig | None:
    if config is None:
        return None
    if isinstance(config, EmbeddingsConfig):
        return config
    if isinstance(config, Mapping):
        return EmbeddingsConfig.from_mapping(config)
    raise TypeError("config must be an EmbeddingsConfig or a mapping")


def _merge_responses(
    responses: list[tuple[int, EmbeddingResponse]],
) -> EmbeddingResponse:
    embeddings: list[Embedding] = []
    input_tokens = 0
    model = ""
    provider = ""
    request_id: str | None = None
    metadata: dict[str, Any] = {}
    batches = 0
    cache_hits = 0
    cache_misses = 0
    for offset, response in responses:
        for embedding in response.embeddings:
            embeddings.append(
                Embedding(
                    index=offset + embedding.index,
                    vector=list(embedding.vector),
                )
            )
        input_tokens += response.usage.input_tokens
        model = model or response.model
        provider = provider or response.provider
        request_id = request_id or response.request_id
        batches += int(response.metadata.get("batches", 1) or 0)
        cache_hits += int(response.metadata.get("cache_hits", 0) or 0)
        cache_misses += int(response.metadata.get("cache_misses", 0) or 0)
        for key, value in response.metadata.items():
            if key not in {"batches", "cache_hits", "cache_misses", "cache"}:
                metadata[key] = value
    if batches:
        metadata["batches"] = batches
    if cache_hits or cache_misses:
        metadata["cache"] = "partial" if cache_hits else "miss"
        metadata["cache_hits"] = cache_hits
        metadata["cache_misses"] = cache_misses
    return EmbeddingResponse(
        embeddings=embeddings,
        model=model,
        provider=provider,
        usage=EmbeddingUsage(input_tokens=input_tokens),
        request_id=request_id,
        metadata=metadata,
    )


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

    ``config`` is the validated ``embeddings`` section. It supplies the ordered
    provider chain, the default vector dimensions, the maximum inputs per
    provider request, the cache size and the per-model prices. An explicit
    ``prices`` mapping wins over the section's prices, and an explicit ``cache``
    wins over the cache built from ``cache_size``. ``engine`` backs that cache
    with SQLite so vectors survive a restart.

    ``prices`` maps a model id to USD per million input tokens. When a model is
    absent from that table, the provider's metadata and the router's
    ``pricing_overrides`` are used. ``cache`` enables per-input caching; pass
    ``None`` to always call the provider when no config is supplied.
    """

    def __init__(
        self,
        router: Router,
        config: EmbeddingsConfig | Mapping[str, Any] | None = None,
        *,
        tracker: CostTracker | None = None,
        cost_tracker: CostTracker | None = None,
        prices: Mapping[str, float] | None = None,
        pricing_overrides: Mapping[str, float] | None = None,
        pricing_version: str = "",
        operation: str = EMBEDDING_OPERATION,
        cache: EmbeddingCache | None = None,
        engine: Engine | None = None,
    ) -> None:
        if not isinstance(router, Router):
            raise TypeError("router must be a Router")
        if tracker is not None and cost_tracker is not None:
            raise TypeError("pass either tracker or cost_tracker, not both")
        active = tracker if tracker is not None else cost_tracker
        if active is not None and not isinstance(active, CostTracker):
            raise TypeError("tracker must be a CostTracker")
        section = _embedding_config(config)
        if cache is not None and not isinstance(cache, EmbeddingCache):
            raise TypeError("cache must be an EmbeddingCache")
        if engine is not None and not isinstance(engine, Engine):
            raise TypeError("engine must be a SQLAlchemy Engine")
        if section is not None:
            if prices is None:
                prices = section.pricing
            if cache is None:
                cache = EmbeddingCache(max_entries=section.cache_size, engine=engine)
            if section.chain or section.pricing:
                router = self._configure_router(router, section)
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
        self._config = section
        self._dimensions = section.dimensions if section is not None else None
        self._batch_size = section.batch_size if section is not None else None
        self._operation = operation.strip()
        self._calculator = EmbeddingCostCalculator(
            prices,
            config=section,
            overrides=overrides,
            pricing_version=pricing_version,
        )
        self._last_selection: RouteSelection | None = None
        self._last_event: CostEvent | None = None
        self._last_price: ResolvedPrice | None = None

    @staticmethod
    def _configure_router(router: Router, config: EmbeddingsConfig) -> Router:
        base = router.config_for(CAPABILITY_EMBEDDINGS)
        overrides = dict(base.pricing_overrides)
        overrides.update(config.pricing)
        merged = base.model_copy(
            update={
                "chain": list(config.chain) or list(base.chain),
                "pricing_overrides": overrides,
            }
        )
        return Router(
            router.providers,
            configs={CAPABILITY_EMBEDDINGS: merged},
            seed=router.seed,
            circuit_breakers=router.circuits,
            rate_limiters=router.limiters,
        )

    @classmethod
    def from_config(
        cls,
        config: EmbeddingsConfig | Mapping[str, Any] | Router,
        router: Router | EmbeddingsConfig | Mapping[str, Any] | None = None,
        *,
        providers: Mapping[str, Provider] | None = None,
        **values: Any,
    ) -> EmbeddingService:
        """Build a service from the embeddings section and a router.

        Both ``from_config(config, router)`` and ``from_config(router, config)``
        are accepted. When no router is supplied, ``providers`` are wrapped in a
        router whose embeddings chain is the configured one.
        """
        if isinstance(config, Router):
            section_value = router
            router_value: Router | None = config
        elif isinstance(config, EmbeddingsConfig) and isinstance(router, Mapping):
            section_value = config
            router_value = None
            if providers is None:
                providers = cast(Mapping[str, Provider], router)
        else:
            section_value = config
            router_value = router if isinstance(router, Router) else None
        section = _embedding_config(
            cast(EmbeddingsConfig | Mapping[str, Any], section_value)
        )
        if router_value is None:
            if providers is None:
                raise TypeError("providers are required when no router is given")
            chain_config = (
                section.to_capability_config()
                if section is not None
                else CapabilityChainConfig()
            )
            router_value = Router(
                providers,
                configs={CAPABILITY_EMBEDDINGS: chain_config},
            )
        elif not isinstance(router_value, Router):
            raise TypeError("router must be a Router")
        return cls(router_value, config=section, **values)

    @property
    def router(self) -> Router:
        """The router every call goes through."""
        return self._router

    @property
    def config(self) -> EmbeddingsConfig | None:
        """The embeddings section this service was built from, if any."""
        return self._config

    @property
    def dimensions(self) -> int | None:
        """The configured default vector dimension, when one was given."""
        return self._dimensions

    @property
    def batch_size(self) -> int | None:
        """The configured maximum inputs per provider request, when given."""
        return self._batch_size

    @property
    def cache_size(self) -> int | None:
        """The configured cache bound, when a section or cache was supplied."""
        return self._cache.max_entries if self._cache is not None else None

    @property
    def prices(self) -> dict[str, float]:
        """A copy of the per-model price table used by the calculator."""
        return self._calculator.prices

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

    async def _embed_target(
        self,
        target: EmbeddingProvider,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse:
        if self._batch_size is None or len(request.inputs) <= self._batch_size:
            return await target.embed(request)
        responses: list[tuple[int, EmbeddingResponse]] = []
        for offset, inputs in enumerate(batch_inputs(request.inputs, self._batch_size)):
            batch = request.model_copy(update={"inputs": inputs})
            response = await target.embed(batch)
            responses.append((offset * self._batch_size, response))
        return _merge_responses(responses)

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
        remaining_budget: float | None = None,
    ) -> EmbeddingResponse:
        """Run *request* through the router and return validated vectors.

        A retryable provider failure falls back to the next candidate through
        :meth:`Router.execute`. The cost event names the provider that finally
        succeeded, never a provider that failed before it. ``use_cache=False``
        bypasses the configured cache for this call.

        ``remaining_budget`` is the amount still available for this call and is
        passed to the router, so a provider whose estimated embedding price
        exceeds it is never called. When no affordable provider remains the
        router refuses the call with
        :class:`~aivia.core.errors.ProviderExhaustedError` and the skipped
        entries carry ``budget_exceeded``. ``None`` disables the check.
        """
        if not isinstance(request, EmbeddingRequest):
            raise TypeError("request must be an EmbeddingRequest")
        if tracker is not None and not isinstance(tracker, CostTracker):
            raise TypeError("tracker must be a CostTracker")
        active = tracker if tracker is not None else self._tracker
        operation_name = self._operation if operation is None else operation.strip()
        if not operation_name:
            raise ValueError("operation must be a non-empty string")
        if self._dimensions is not None and request.dimensions is None:
            request = request.model_copy(update={"dimensions": self._dimensions})

        async def run(provider: Provider) -> EmbeddingResponse:
            target: Provider = provider
            if self._cache is not None and use_cache:
                target = CachedEmbeddingProvider(provider, self._cache)
            return await self._embed_target(
                cast(EmbeddingProvider, target),
                request,
            )

        response = await self._router.execute(
            CAPABILITY_EMBEDDINGS,
            run,
            remaining_budget=remaining_budget,
        )
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
        remaining_budget: float | None = None,
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
            remaining_budget=remaining_budget,
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
    config: EmbeddingsConfig | Mapping[str, Any] | None = None,
    prices: Mapping[str, float] | None = None,
    pricing_overrides: Mapping[str, float] | None = None,
    pricing_version: str = "",
    cache: EmbeddingCache | None = None,
    engine: Engine | None = None,
    run_id: str = "",
    operation: str | None = None,
    estimated: bool = False,
    metadata: Mapping[str, Any] | None = None,
    record_cost: bool = True,
    use_cache: bool = True,
    remaining_budget: float | None = None,
) -> EmbeddingResponse:
    """Execute one embedding call without constructing a long-lived service."""
    service = EmbeddingService(
        router,
        tracker=tracker,
        config=config,
        prices=prices,
        pricing_overrides=pricing_overrides,
        pricing_version=pricing_version,
        cache=cache,
        engine=engine,
    )
    return await service.embed(
        request,
        run_id=run_id,
        operation=operation,
        estimated=estimated,
        metadata=metadata,
        record_cost=record_cost,
        use_cache=use_cache,
        remaining_budget=remaining_budget,
    )


def embedding_service_from_config(
    config: EmbeddingsConfig | Mapping[str, Any] | Router,
    router: Router | EmbeddingsConfig | Mapping[str, Any] | None = None,
    *,
    providers: Mapping[str, Provider] | None = None,
    **values: Any,
) -> EmbeddingService:
    """Build an :class:`EmbeddingService` from an embeddings section."""
    return EmbeddingService.from_config(
        config,
        router,
        providers=providers,
        **values,
    )


service_from_config = embedding_service_from_config
"""Alias for :func:`embedding_service_from_config`."""

"""The search service passes the remaining budget to the router."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from aivia.core.config.capability import CapabilityChainConfig
from aivia.core.errors import ConfigError, ProviderExhaustedError
from aivia.core.models import ProviderMeta, ProviderPrice, RateLimit
from aivia.core.router import Router
from aivia.providers import Provider
from aivia.providers.protocols import (
    CAPABILITY_SEARCH,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from aivia.providers.search import SearchService
from aivia.providers.search import search as routed_search

PRICE = 0.004


class FakeSearchProvider(Provider):
    """A search provider with a configurable per-call price."""

    def __init__(
        self,
        name: str,
        *,
        price: float = PRICE,
        results: Sequence[SearchResult] | None = None,
    ) -> None:
        super().__init__(
            ProviderMeta(
                name=name,
                capabilities=[CAPABILITY_SEARCH],
                pricing={
                    CAPABILITY_SEARCH: ProviderPrice(
                        amount_usd=price,
                        unit="call",
                    )
                },
                rate_limit=RateLimit(requests_per_minute=0),
            )
        )
        self._results = list(results) if results is not None else None
        self.requests: list[SearchRequest] = []

    async def search(self, request: SearchRequest) -> SearchResponse:
        self.requests.append(request)
        results = self._results
        if results is None:
            results = [
                SearchResult(
                    title="Result",
                    url="https://example.com/result",
                    snippet="Snippet",
                )
            ]
        return SearchResponse(
            query=request.query,
            results=results,
            provider=self.name,
            total_results=len(results),
        )


def _service(*providers: FakeSearchProvider) -> SearchService:
    router = Router(
        {provider.name: provider for provider in providers},
        config=CapabilityChainConfig(chain=[provider.name for provider in providers]),
    )
    return SearchService(router)


def test_search_allows_a_call_that_fits_the_remaining_budget() -> None:
    provider = FakeSearchProvider("acme")
    service = _service(provider)

    response = asyncio.run(
        service.search(
            SearchRequest(query="topic"),
            remaining_budget=PRICE,
        )
    )

    assert response.provider == "acme"
    assert len(provider.requests) == 1


def test_search_refuses_an_over_budget_call_without_calling_the_provider() -> None:
    provider = FakeSearchProvider("pricey", price=0.10)
    service = _service(provider)

    with pytest.raises(ProviderExhaustedError) as caught:
        asyncio.run(
            service.search(
                SearchRequest(query="topic"),
                remaining_budget=0.05,
            )
        )

    assert provider.requests == []
    assert caught.value.details["skipped"] == [
        {"provider": "pricey", "reason": "budget_exceeded"}
    ]
    assert service.last_selection is None


def test_budget_aware_search_falls_back_to_an_affordable_provider() -> None:
    pricey = FakeSearchProvider("pricey", price=0.10)
    affordable = FakeSearchProvider("affordable", price=0.002)
    service = _service(pricey, affordable)

    response = asyncio.run(
        service.search(
            SearchRequest(query="topic"),
            remaining_budget=0.01,
        )
    )

    assert response.provider == "affordable"
    assert pricey.requests == []
    assert len(affordable.requests) == 1


def test_search_with_cost_forwards_the_remaining_budget() -> None:
    provider = FakeSearchProvider("pricey", price=0.10)
    service = _service(provider)

    with pytest.raises(ProviderExhaustedError):
        asyncio.run(
            service.search_with_cost(
                SearchRequest(query="topic"),
                remaining_budget=0.05,
            )
        )
    assert provider.requests == []


def test_module_level_search_forwards_the_remaining_budget() -> None:
    provider = FakeSearchProvider("pricey", price=0.10)
    router = Router(
        {"pricey": provider},
        config=CapabilityChainConfig(chain=["pricey"]),
    )

    with pytest.raises(ProviderExhaustedError):
        asyncio.run(
            routed_search(
                SearchRequest(query="topic"),
                router,
                remaining_budget=0.05,
            )
        )
    assert provider.requests == []


def test_search_without_a_budget_does_not_refuse_an_expensive_provider() -> None:
    provider = FakeSearchProvider("pricey", price=10.0)
    service = _service(provider)

    response = asyncio.run(service.search(SearchRequest(query="topic")))

    assert response.provider == "pricey"
    assert len(provider.requests) == 1


def test_search_rejects_an_invalid_remaining_budget_before_the_provider() -> None:
    provider = FakeSearchProvider("acme")
    service = _service(provider)

    with pytest.raises(ConfigError, match="remaining budget"):
        asyncio.run(
            service.search(
                SearchRequest(query="topic"),
                remaining_budget=float("nan"),
            )
        )
    assert provider.requests == []

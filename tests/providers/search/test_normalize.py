"""Normalize search results and route search calls through the router."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime

import pytest
from sqlalchemy import Engine

from aivia.core.config.capability import CapabilityChainConfig
from aivia.core.cost import CostTracker
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.errors import TimeoutError
from aivia.core.models import ProviderMeta, ProviderPrice, RateLimit
from aivia.core.router import Router
from aivia.providers import Provider
from aivia.providers.protocols import (
    CAPABILITY_SEARCH,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from aivia.providers.search import (
    SearchNormalizer,
    SearchService,
    canonical_url,
    dedupe_results,
    merge_search_responses,
    normalize_search_response,
    normalize_search_result,
    normalize_search_results,
)

PRICE = 0.004


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


class FakeSearchProvider(Provider):
    """A search provider with a configurable price, results and failure."""

    def __init__(
        self,
        name: str,
        *,
        results: Sequence[SearchResult] | None = None,
        error: Exception | None = None,
        price: float = PRICE,
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
        self._error = error
        self.requests: list[SearchRequest] = []

    async def search(self, request: SearchRequest) -> SearchResponse:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        results = self._results
        if results is None:
            results = [
                SearchResult(
                    title=f"Result {index}",
                    url=f"https://example.com/{index}",
                    snippet=f"Snippet {index}",
                )
                for index in range(1, 3)
            ]
        return SearchResponse(
            query=request.query,
            results=results,
            provider=self.name,
            total_results=len(results),
        )


def _result(url: str, *, title: str = "title", snippet: str = "") -> SearchResult:
    return SearchResult(title=title, url=url, snippet=snippet)


def test_canonical_url_collapses_case_ports_fragments_and_trailing_slashes() -> None:
    assert (
        canonical_url("HTTPS://Example.COM:443/Path/To/?q=1#section")
        == "https://example.com/Path/To?q=1"
    )
    assert canonical_url("http://example.com:80") == "http://example.com/"
    assert canonical_url("https://example.com/a/?q=1") == "https://example.com/a?q=1"


def test_a_mapping_is_normalized_into_the_shared_result_model() -> None:
    result = normalize_search_result(
        {
            "title": "  A title  ",
            "url": "https://example.com/a",
            "snippet": "  A snippet  ",
            "published_at": datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        },
        provider="acme",
    )

    assert isinstance(result, SearchResult)
    assert result.title == "A title"
    assert result.snippet == "A snippet"
    assert result.source == "acme"
    assert result.published_at == datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def test_dedupe_keeps_the_first_url_spelling_and_drops_equivalent_pages() -> None:
    first = _result("https://Example.com/a/#intro", title="first")
    equivalent = _result("https://example.com/a", title="second")
    distinct = _result("https://example.com/a?page=2", title="third")

    unique = dedupe_results([first, equivalent, distinct])

    assert [result.title for result in unique] == ["first", "third"]
    assert unique[0].url == "https://Example.com/a/#intro"


def test_normalize_results_and_response_dedupe_by_url() -> None:
    response = SearchResponse(
        query="topic",
        provider="acme",
        results=[
            _result("https://example.com/a"),
            _result("https://EXAMPLE.com/a/"),
            _result("https://example.com/b"),
        ],
    )

    normalized_results = normalize_search_results(response.results)
    normalized = normalize_search_response(response)

    assert len(normalized_results) == 2
    assert [result.url for result in normalized.results] == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert normalized.total_results == 2


def test_merging_responses_dedupes_across_providers_and_records_providers() -> None:
    first = SearchResponse(
        query="topic",
        provider="one",
        results=[_result("https://example.com/a"), _result("https://example.com/b")],
        total_results=2,
    )
    second = SearchResponse(
        query="topic",
        provider="two",
        results=[_result("https://EXAMPLE.com/a/"), _result("https://example.com/c")],
        total_results=2,
    )

    merged = merge_search_responses([first, second])

    assert [result.url for result in merged.results] == [
        "https://example.com/a",
        "https://example.com/b",
        "https://example.com/c",
    ]
    assert merged.query == "topic"
    assert merged.provider == "one,two"
    assert merged.metadata["providers"] == ["one", "two"]


def test_a_normalizer_can_keep_provider_order_when_dedupe_is_off() -> None:
    normalizer = SearchNormalizer(dedupe=False)
    response = SearchResponse(
        query="topic",
        results=[_result("https://example.com/a"), _result("https://example.com/a/")],
    )

    normalized = normalizer.response(response)

    assert len(normalized.results) == 2


def test_service_routes_the_call_and_records_the_actual_cost(engine: Engine) -> None:
    tracker = CostTracker(engine)
    provider = FakeSearchProvider(
        "acme",
        results=[
            _result("https://example.com/a"),
            _result("https://EXAMPLE.com/a/"),
            _result("https://example.com/b"),
        ],
    )
    router = Router(
        {"acme": provider},
        config=CapabilityChainConfig(chain=["acme"]),
    )
    service = SearchService(router, tracker=tracker)

    response = asyncio.run(
        service.search(
            SearchRequest(query="topic"),
            run_id="run_1",
            metadata={"stage": "research"},
        )
    )

    assert provider.requests[0].query == "topic"
    assert [result.url for result in response.results] == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert response.provider == "acme"
    event = service.last_event
    assert event is not None
    assert event.provider == "acme"
    assert event.operation == "search"
    assert event.units == 1.0
    assert event.unit_kind == "call"
    assert event.amount_usd == pytest.approx(PRICE)
    assert event.run_id == "run_1"
    assert event.estimated is False
    assert event.metadata["query"] == "topic"
    assert event.metadata["result_count"] == 2
    assert event.metadata["stage"] == "research"
    assert tracker.total() == pytest.approx(PRICE)


def test_service_falls_back_and_prices_the_provider_that_served_it(
    engine: Engine,
) -> None:
    tracker = CostTracker(engine)
    failing = FakeSearchProvider("failing", error=TimeoutError("slow"))
    serving = FakeSearchProvider("serving", price=0.002)
    router = Router(
        {"failing": failing, "serving": serving},
        config=CapabilityChainConfig(chain=["failing", "serving"]),
    )
    service = SearchService(router, tracker=tracker)

    response = asyncio.run(service.search(SearchRequest(query="topic")))

    assert len(failing.requests) == 1
    assert len(serving.requests) == 1
    assert response.provider == "serving"
    assert service.last_selection is not None
    assert service.last_selection.name == "serving"
    assert service.last_event is not None
    assert service.last_event.provider == "serving"
    assert service.last_event.amount_usd == pytest.approx(0.002)
    assert tracker.total() == pytest.approx(0.002)


def test_router_pricing_overrides_are_used_for_search_cost(engine: Engine) -> None:
    tracker = CostTracker(engine)
    provider = FakeSearchProvider("acme", price=PRICE)
    router = Router(
        {"acme": provider},
        config=CapabilityChainConfig(
            chain=["acme"],
            pricing_overrides={"acme": 0.01},
        ),
    )
    service = SearchService(router, tracker=tracker)

    asyncio.run(service.search(SearchRequest(query="topic")))

    assert service.last_event is not None
    assert service.last_event.amount_usd == pytest.approx(0.01)
    assert tracker.total() == pytest.approx(0.01)


def test_a_service_without_a_tracker_still_returns_normalized_results() -> None:
    provider = FakeSearchProvider(
        "acme",
        results=[_result("https://example.com/a"), _result("https://example.com/a/")],
    )
    router = Router({"acme": provider}, config=CapabilityChainConfig(chain=["acme"]))
    service = SearchService(router)

    response = asyncio.run(service.search(SearchRequest(query="topic")))

    assert len(response.results) == 1
    assert service.last_event is None
    assert service.last_price is None


def test_search_with_cost_returns_the_response_event_and_selection(
    engine: Engine,
) -> None:
    tracker = CostTracker(engine)
    provider = FakeSearchProvider("acme")
    router = Router({"acme": provider}, config=CapabilityChainConfig(chain=["acme"]))
    service = SearchService(router, tracker=tracker)

    outcome = asyncio.run(service.search_with_cost(SearchRequest(query="topic")))

    assert outcome.response.provider == "acme"
    assert outcome.event is not None
    assert outcome.price is not None
    assert outcome.selection is not None
    assert outcome.amount_usd == pytest.approx(PRICE)
    assert len(outcome.results) == 2

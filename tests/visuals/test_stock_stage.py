"""Stock beats resolve through ordered queries, top results and relevance scores."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from aivia.core.models.script import MediaKind, VisualType
from aivia.core.stages import StageContext
from aivia.media.stock_relevance import StockRelevance
from aivia.providers.protocols.stock import (
    StockAsset,
    StockMediaKind,
    StockOrientation,
    StockRequest,
    StockResponse,
)
from aivia.visuals import (
    StockStage,
    StockStageResult,
    resolve_stock_request,
    stock_queries,
)
from aivia.visuals.strategy import VisualRequest


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _asset(asset_id: str, *, query: str = "") -> StockAsset:
    return StockAsset(
        id=asset_id,
        provider="fake-stock",
        kind=StockMediaKind.IMAGE,
        url=f"https://cdn.example.com/{asset_id}.jpg",
        preview_url=f"https://cdn.example.com/{asset_id}-preview.jpg",
        width=1920,
        height=1080,
        creator="Synthetic Creator",
        license="CC0",
        attribution="Synthetic Creator / example.com",
        metadata={"alt": query} if query else {},
    )


def _request(
    query: str,
    *,
    index: int = 0,
    prompt: str = "",
    text: str = "",
    metadata: dict[str, Any] | None = None,
) -> VisualRequest:
    values: dict[str, Any] = {
        "index": index,
        "visual_type": VisualType.STOCK,
        "kind": MediaKind.STOCK,
        "prompt": prompt or query,
        "query": query,
        "text": text,
        "aspect_ratio": "16:9",
        "metadata": metadata or {},
    }
    return VisualRequest.model_validate(values)


class FakeSearch:
    """Return the canned assets for each query and remember every request."""

    def __init__(self, responses: dict[str, list[StockAsset]]) -> None:
        self.responses = responses
        self.requests: list[StockRequest] = []

    async def search(self, request: StockRequest) -> StockResponse:
        self.requests.append(request)
        assets = self.responses.get(request.query, [])
        return StockResponse(
            query=request.query,
            assets=list(assets),
            provider="fake-stock",
            metadata={"cost_usd": 0.01},
        )


class FakeRanker:
    """Score every candidate with a canned id to score mapping."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[tuple[list[StockAsset], str]] = []

    async def rank(
        self,
        assets: list[StockAsset],
        description: str,
        *,
        run_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> list[StockRelevance]:
        self.calls.append((list(assets), description))
        return [
            StockRelevance(asset=asset, score=self.scores[asset.id]) for asset in assets
        ]


def test_a_stock_beat_searches_its_query_and_keeps_the_best_top_result() -> None:
    assets = [_asset("first"), _asset("second"), _asset("third")]
    search = FakeSearch({"lighthouse sunset": assets})
    ranker = FakeRanker({"first": 0.4, "second": 0.92, "third": 0.7})
    request = _request("lighthouse sunset", prompt="A lighthouse at sunset")

    result = _run(
        resolve_stock_request(
            request,
            search=search,
            ranker=ranker,
            top_results=2,
            min_relevance=0.5,
        )
    )

    assert len(search.requests) == 1
    assert search.requests[0].query == "lighthouse sunset"
    assert search.requests[0].kind is StockMediaKind.IMAGE
    assert search.requests[0].orientation is StockOrientation.LANDSCAPE
    assert search.requests[0].per_page == 2
    assert [asset.id for asset in ranker.calls[0][0]] == ["first", "second"]
    assert ranker.calls[0][1] == "A lighthouse at sunset"
    assert result.asset is not None
    assert result.asset.id == "second"
    assert result.score == 0.92
    assert result.query == "lighthouse sunset"
    assert result.attempts[0].passed is True
    assert result.attempts[0].result_count == 2
    assert result.attempts[0].selected_id == "second"
    assert result.fallback_used is False
    assert result.cost_usd == 0.01
    assert result.id == "second"
    assert result.url == "https://cdn.example.com/second.jpg"


def test_a_rejected_first_query_falls_back_to_the_next_query() -> None:
    search = FakeSearch(
        {
            "too narrow": [_asset("weak")],
            "broader": [_asset("strong")],
        }
    )
    ranker = FakeRanker({"weak": 0.2, "strong": 0.9})
    request = _request(
        "too narrow",
        text="broader",
        metadata={"fallback_queries": ["broader"]},
    )

    result = _run(
        resolve_stock_request(
            request,
            search=search,
            ranker=ranker,
            min_relevance=0.5,
        )
    )

    assert [request.query for request in search.requests] == ["too narrow", "broader"]
    assert [attempt.query for attempt in result.attempts] == ["too narrow", "broader"]
    assert [attempt.passed for attempt in result.attempts] == [False, True]
    assert result.asset is not None
    assert result.asset.id == "strong"
    assert result.query == "broader"
    assert result.fallback_used is True
    assert result.cost_usd == 0.02


def test_an_empty_first_query_also_falls_back() -> None:
    search = FakeSearch({"second": [_asset("found")]})
    ranker = FakeRanker({"found": 0.8})
    request = _request("first", metadata={"queries": ["first", "second"]})

    result = _run(
        resolve_stock_request(
            request,
            search=search,
            ranker=ranker,
            min_relevance=0.5,
        )
    )

    assert [request.query for request in search.requests] == ["first", "second"]
    assert result.attempts[0].result_count == 0
    assert result.asset is not None
    assert result.asset.id == "found"


def test_without_a_ranker_the_provider_order_is_kept() -> None:
    search = FakeSearch({"bees": [_asset("one"), _asset("two")]})

    result = _run(resolve_stock_request(_request("bees"), search=search))

    assert result.asset is not None
    assert result.asset.id == "one"
    assert result.score == 1.0
    assert result.fallback_used is False


def test_queries_are_ordered_deduplicated_and_bounded() -> None:
    request = _request(
        "primary",
        prompt="prompt query",
        text="text query",
        metadata={
            "queries": ["metadata query", "primary"],
            "fallback_queries": ["metadata fallback", "text query"],
        },
    )

    queries = stock_queries(
        request,
        fallback_queries=["stage fallback"],
        limit=6,
    )

    assert queries == [
        "metadata query",
        "primary",
        "prompt query",
        "text query",
        "metadata fallback",
        "stage fallback",
    ]
    assert stock_queries(request, limit=2) == ["metadata query", "primary"]


def test_the_stage_resolves_stock_beats_and_writes_a_manifest(tmp_path: Path) -> None:
    search = FakeSearch(
        {
            "lighthouse sunset": [_asset("first"), _asset("best")],
            "bees": [_asset("bee")],
        }
    )
    ranker = FakeRanker({"first": 0.1, "best": 0.95, "bee": 0.9})
    requests = [
        _request("lighthouse sunset", index=0),
        VisualRequest(
            index=1,
            visual_type=VisualType.IMAGE,
            kind=MediaKind.IMAGE,
            prompt="A generated image",
        ),
        _request("bees", index=2),
    ]
    stage = StockStage(
        requests=requests,
        search=search,
        ranker=ranker,
        min_relevance=0.5,
    )
    context = StageContext(run_dir=tmp_path, config={})

    result = _run(stage.run(context))

    assert isinstance(result, StockStageResult)
    assert [beat.index for beat in result.beats] == [0, 2]
    assert [beat.asset.id for beat in result.beats if beat.asset is not None] == [
        "best",
        "bee",
    ]
    assert result.metadata["beats"] == 2
    assert result.metadata["resolved"] == 2
    assert result.metadata["unresolved"] == 0
    assert result.cost_usd == 0.02
    manifest = tmp_path / "stock.json"
    assert manifest.exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["stage"] == "stock"
    assert [beat["query"] for beat in payload["beats"]] == [
        "lighthouse sunset",
        "bees",
    ]
    assert payload["beats"][0]["asset"]["id"] == "best"


def test_an_unresolved_stage_beat_is_reported_without_failing(tmp_path: Path) -> None:
    search = FakeSearch({})
    stage = StockStage(requests=[_request("nothing")], search=search)
    context = StageContext(run_dir=tmp_path, config={})

    result = _run(stage.run(context))

    assert isinstance(result, StockStageResult)
    assert result.beats[0].asset is None
    assert result.beats[0].resolved is False
    assert result.unresolved == result.beats
    assert result.metadata["unresolved"] == 1
    assert (tmp_path / "stock.json").exists()

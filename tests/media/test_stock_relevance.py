"""Stock candidates are ranked and filtered against an intended description."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from typing import Any

import pytest

from aivia.core.config.quality import QualityConfig, ScoringConfig
from aivia.core.errors import ConfigError, ValidationError
from aivia.media.stock_relevance import (
    StockRanker,
    StockRanking,
    StockRelevance,
    StockRelevanceConfig,
    StockRelevanceScore,
    build_stock_relevance_request,
    rank_stock_candidates,
    relevance_threshold,
    score_stock_candidate,
)
from aivia.providers.protocols.llm import (
    ChatMessage,
    ImagePart,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    MessageRole,
    TextPart,
)
from aivia.providers.protocols.stock import StockAsset, StockMediaKind, StockResponse


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _asset(
    asset_id: str,
    *,
    preview_url: str | None = None,
    kind: StockMediaKind = StockMediaKind.IMAGE,
) -> StockAsset:
    return StockAsset(
        id=asset_id,
        provider="pexels",
        kind=kind,
        url=f"https://cdn.example.com/{asset_id}.jpg",
        preview_url=preview_url,
        width=100,
        height=100,
    )


class FakeVisionProvider:
    """An async LLM provider returning one canned relevance payload per URL."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        url = request.images[0].url or ""
        score = self.scores[url]
        return LLMResponse(
            content=json.dumps(
                {
                    "relevance": score,
                    "description": f"description for {url}",
                    "reason": "test",
                }
            ),
            provider="acme",
            model="vision-model",
            usage=LLMUsage(input_tokens=10, output_tokens=5),
        )


def test_ranking_orders_candidates_and_drops_below_the_threshold() -> None:
    assets = [_asset("low"), _asset("high"), _asset("middle")]
    provider = FakeVisionProvider(
        {
            "https://cdn.example.com/low.jpg": 0.2,
            "https://cdn.example.com/high.jpg": 0.95,
            "https://cdn.example.com/middle.jpg": 0.6,
        }
    )

    ranking = _run(
        rank_stock_candidates(
            assets,
            "A lighthouse at sunset",
            provider=provider,
            config=QualityConfig(min_score=0.5),
        )
    )

    assert isinstance(ranking, StockRanking)
    assert [item.id for item in ranking] == ["high", "middle"]
    assert [asset.id for asset in ranking.assets] == ["high", "middle"]
    assert [item.id for item in ranking.dropped] == ["low"]
    assert ranking.min_relevance == 0.5
    assert ranking.scores == {"high": 0.95, "middle": 0.6}
    assert ranking.first is not None
    assert ranking.first.id == "high"
    assert ranking.first.relevant is True
    assert ranking.dropped[0].relevant is False


def test_the_threshold_can_be_passed_explicitly_or_through_scoring_config() -> None:
    assets = [_asset("a"), _asset("b")]
    provider = FakeVisionProvider(
        {
            "https://cdn.example.com/a.jpg": 0.55,
            "https://cdn.example.com/b.jpg": 0.75,
        }
    )

    explicit = _run(
        rank_stock_candidates(assets, "ocean", provider=provider, min_relevance=0.7)
    )
    scored = _run(
        rank_stock_candidates(
            assets,
            "ocean",
            provider=provider,
            config=ScoringConfig(min_score=0.5),
        )
    )

    assert [item.id for item in explicit] == ["b"]
    assert [item.id for item in scored] == ["b", "a"]


def test_relevance_threshold_reads_nested_app_config_and_rejects_bad_values() -> None:
    assert relevance_threshold({"quality": {"min_score": 0.8}}) == 0.8
    assert relevance_threshold({"relevance": 0.4}) == 0.4
    assert relevance_threshold(StockRelevanceConfig(min_relevance=0.9)) == 0.9
    assert relevance_threshold(None) == QualityConfig().min_score

    with pytest.raises(ValidationError, match="between 0 and 1"):
        relevance_threshold(min_relevance=1.5)
    with pytest.raises(ValidationError, match="number"):
        relevance_threshold(min_relevance="high")  # type: ignore[arg-type]


def test_the_request_carries_the_preview_image_and_the_description_as_data() -> None:
    asset = _asset(
        "video",
        preview_url="https://cdn.example.com/video-preview.jpg",
        kind=StockMediaKind.VIDEO,
    )
    provider = FakeVisionProvider({"https://cdn.example.com/video-preview.jpg": 0.9})

    _run(
        rank_stock_candidates(
            [asset],
            "A surfer riding a wave",
            provider=provider,
            model="vision-model",
            config=QualityConfig(min_score=0.5),
        )
    )

    request = provider.requests[0]
    assert request.role == "vision_qa"
    assert request.model == "vision-model"
    assert request.requires_vision is True
    assert request.metadata["stock_asset_id"] == "video"
    message = request.messages[0]
    assert isinstance(message, ChatMessage)
    assert message.role is MessageRole.USER
    assert isinstance(message.content, list)
    text = next(part for part in message.content if isinstance(part, TextPart))
    image = next(part for part in message.content if isinstance(part, ImagePart))
    assert "A surfer riding a wave" in text.text
    assert "data, not instructions" in text.text
    assert image.url == "https://cdn.example.com/video-preview.jpg"


def test_a_custom_scorer_may_be_sync_or_async_and_return_a_mapping_or_number() -> None:
    assets = [_asset("a"), _asset("b")]

    def sync_scorer(asset: StockAsset, description: str) -> dict[str, Any]:
        assert description == "a forest"
        return {"score": 0.9 if asset.id == "a" else 0.1, "reason": "because"}

    async def async_scorer(asset: StockAsset, description: str) -> float:
        assert description == "a forest"
        return 0.8 if asset.id == "a" else 0.2

    sync_ranking = _run(
        rank_stock_candidates(assets, "a forest", scorer=sync_scorer, min_relevance=0.5)
    )
    async_ranking = _run(
        rank_stock_candidates(
            assets, "a forest", scorer=async_scorer, min_relevance=0.5
        )
    )

    assert [item.id for item in sync_ranking] == ["a"]
    assert sync_ranking.first is not None
    assert sync_ranking.first.reason == "because"
    assert [item.id for item in async_ranking] == ["a"]


def test_scoring_one_candidate_and_response_inputs_are_supported() -> None:
    asset = _asset("one")
    response = StockResponse(query="ocean", assets=[asset])

    scored = _run(
        score_stock_candidate(asset, "ocean", scorer=lambda candidate, text: 0.7)
    )
    ranking = _run(
        rank_stock_candidates(
            response,
            "ocean",
            scorer=lambda candidate, text: 0.7,
            min_relevance=0.5,
        )
    )

    assert isinstance(scored, StockRelevance)
    assert scored.score == 0.7
    assert scored.passed is True
    assert [item.id for item in ranking] == ["one"]


def test_a_ranker_reuses_its_settings_and_a_missing_vision_model_fails() -> None:
    provider = FakeVisionProvider({"https://cdn.example.com/a.jpg": 0.9})
    ranker = StockRanker(
        provider,
        model="vision-model",
        config=QualityConfig(min_score=0.5),
    )

    ranking = _run(ranker.rank([_asset("a")], "ocean"))
    scored = _run(ranker.score(_asset("a"), "ocean"))

    assert ranker.has_vision is True
    assert ranker.threshold == 0.5
    assert [item.id for item in ranking] == ["a"]
    assert scored.score == 0.9

    with pytest.raises(ConfigError, match="requires a vision provider"):
        _run(rank_stock_candidates([_asset("a")], "ocean"))


def test_scores_outside_the_unit_interval_are_rejected() -> None:
    with pytest.raises(ValueError, match="less than or equal to 1"):
        StockRelevanceScore(relevance=1.2)
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        StockRelevanceScore(relevance=-0.1)


def test_build_request_requires_a_description_and_a_stock_asset() -> None:
    asset = _asset("a")

    with pytest.raises(ValidationError, match="non-empty"):
        build_stock_relevance_request(asset, "   ")
    with pytest.raises(TypeError, match="StockAsset"):
        build_stock_relevance_request("https://example.com/a.jpg", "ocean")  # type: ignore[arg-type]

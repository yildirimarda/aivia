"""Score topics on their weakest rubric dimension as well as the weighted mean."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

import pytest

from aivia.core.config.quality import ScoringConfig
from aivia.core.errors import ConfigError
from aivia.core.models.topic import TopicCandidate
from aivia.providers.protocols.source import SourceResponse
from aivia.research import (
    CandidateScore,
    ScoringResult,
    ScoringRubric,
    WeakestLinkResult,
    WeakestLinkScore,
    WeakestLinkScorer,
    WeakestLinkStage,
    as_weakest_link_result,
    build_weakest_link_stage,
    filter_weakest_link,
    passes_weakest_link,
    score_candidates,
    score_candidates_weakest_link,
    score_topic,
    score_topics_weakest_link,
    score_weakest_link,
    score_weakest_link_async,
    weakest_dimension,
    weakest_link,
    weakest_link_stage,
    weakest_score,
)


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _candidate(title: str) -> TopicCandidate:
    return TopicCandidate(
        title=title,
        source_urls=[f"https://example.test/{title.replace(' ', '_')}"],
    )


def _rubric(**overrides: Any) -> ScoringRubric:
    values: dict[str, Any] = {
        "weights": {"relevance": 0.9, "novelty": 0.1},
        "min_score": 0.5,
        "min_dimension_score": 0.3,
    }
    values.update(overrides)
    return ScoringRubric(**values)


def test_a_score_records_its_weakest_dimension_and_that_dimensions_score() -> None:
    score = score_topic(
        _candidate("Lopsided"),
        scores={"relevance": 0.9, "novelty": 0.3},
        rubric=_rubric(),
    )

    assert isinstance(score, CandidateScore)
    assert score.overall == pytest.approx(0.84)
    assert score.weakest_dimension == "novelty"
    assert score.weakest_score == pytest.approx(0.3)
    assert score.weakest_link is not None
    weakest_name, weakest_value = score.weakest_link
    assert weakest_name == "novelty"
    assert weakest_value == pytest.approx(0.3)
    assert score.meets_min_dimension is True
    assert score.selected is True
    assert score.to_dict()["weakest_dimension"] == "novelty"
    assert score.to_dict()["weakest_score"] == pytest.approx(0.3)
    assert score.to_dict()["min_dimension_score"] == 0.3
    assert score.to_dict()["selected"] is True


def test_the_weakest_dimension_breaks_ties_by_rubric_order() -> None:
    rubric = ScoringRubric(
        weights={"relevance": 0.34, "novelty": 0.33, "accuracy": 0.33},
        min_score=0.1,
        min_dimension_score=0.1,
    )

    score = score_topic(
        _candidate("Tied"),
        scores={"relevance": 0.9, "novelty": 0.2, "accuracy": 0.2},
        rubric=rubric,
    )

    assert score.weakest_dimension == "novelty"
    assert score.weakest_score == pytest.approx(0.2)


def test_a_dead_dimension_blocks_a_candidate_whose_weighted_mean_passes() -> None:
    result = score_weakest_link(
        [_candidate("Dead"), _candidate("Healthy")],
        scores=[
            {"relevance": 0.9, "novelty": 0.0},
            {"relevance": 0.6, "novelty": 0.5},
        ],
        rubric=_rubric(),
    )

    assert isinstance(result, WeakestLinkResult)
    assert result.scores[0].overall == pytest.approx(0.81)
    assert result.scores[0].passed is True
    assert result.scores[0].meets_min_dimension is False
    assert result.scores[0].blocked_by_weakest_link is True
    assert result.scores[0].selected is False
    assert result.titles == ["Healthy"]
    assert result.blocked_titles == ["Dead"]
    assert [candidate.title for candidate in result.blocked_candidates] == ["Dead"]
    assert result.blocked[0].weakest_dimension == "novelty"
    assert [score.title for score in result.failed] == ["Dead"]
    assert result.count == 1
    assert result.input_count == 2
    assert result.dropped_count == 1
    assert result.min_dimension_score == 0.3
    assert result.metadata["mean_kept"] == 2
    assert result.metadata["kept"] == 1
    assert result.metadata["dropped"] == 1
    assert result.metadata["weakest_link_blocked"] == 1
    assert result.metadata["weakest_link_blocked_titles"] == ["Dead"]


def test_the_floor_is_inclusive_and_only_the_weakest_dimension_matters() -> None:
    result = score_weakest_link(
        [_candidate("Edge"), _candidate("Below")],
        scores=[
            {"relevance": 1.0, "novelty": 0.3},
            {"relevance": 1.0, "novelty": 0.29},
        ],
        rubric=_rubric(),
    )

    assert result.titles == ["Edge"]
    assert result.blocked_titles == ["Below"]


def test_a_zero_floor_keeps_the_mean_only_behaviour() -> None:
    result = score_weakest_link(
        [_candidate("Lopsided")],
        scores={"relevance": 1.0, "novelty": 0.0},
        rubric=_rubric(min_dimension_score=0.0),
    )

    assert result.titles == ["Lopsided"]
    assert result.blocked == []


def test_the_floor_can_come_from_the_scoring_config() -> None:
    config = ScoringConfig(
        weights={"relevance": 0.5, "novelty": 0.5},
        min_score=0.4,
        min_dimension_score=0.4,
    )

    result = score_weakest_link(
        [_candidate("Dead"), _candidate("Alive")],
        scores=[
            {"relevance": 1.0, "novelty": 0.0},
            {"relevance": 0.5, "novelty": 0.4},
        ],
        config=config,
    )

    assert result.rubric.min_dimension_score == 0.4
    assert result.titles == ["Alive"]
    assert result.blocked_titles == ["Dead"]


def test_the_stage_binds_the_rubric_scorer_and_weakest_link_floor() -> None:
    rubric = _rubric()

    def scorer(candidate: TopicCandidate) -> dict[str, float]:
        if "good" in candidate.title:
            return {"relevance": 0.9, "novelty": 0.9}
        return {"relevance": 0.9, "novelty": 0.0}

    stage = WeakestLinkStage(rubric, scorer=scorer)
    candidates = [_candidate("a good topic"), _candidate("a dead topic")]

    result = stage.run(candidates)
    filtered = stage.filter(candidates)
    response = stage.to_response(candidates, query="novelty")

    assert isinstance(stage, WeakestLinkStage)
    assert isinstance(stage, WeakestLinkScorer)
    assert stage.min_dimension_score == 0.3
    assert result.titles == ["a good topic"]
    assert result.blocked_titles == ["a dead topic"]
    assert [candidate.title for candidate in filtered] == ["a good topic"]
    assert isinstance(response, SourceResponse)
    assert response.provider == "scoring"
    assert response.query == "novelty"
    assert response.titles == ["a good topic"]


def test_the_stage_awaits_an_async_scorer() -> None:
    async def scorer(candidate: TopicCandidate) -> dict[str, float]:
        if candidate.title == "Keep":
            return {"relevance": 0.9, "novelty": 0.9}
        return {"relevance": 0.9, "novelty": 0.0}

    stage = WeakestLinkStage(_rubric())
    result = _run(
        stage.run_async(
            [_candidate("Keep"), _candidate("Drop")],
            scorer=scorer,
        )
    )
    direct = _run(
        score_weakest_link_async(
            [_candidate("Keep"), _candidate("Drop")],
            scorer=scorer,
            rubric=_rubric(),
        )
    )

    assert result.titles == ["Keep"]
    assert result.blocked_titles == ["Drop"]
    assert direct.titles == ["Keep"]
    assert direct.blocked_titles == ["Drop"]


def test_a_plain_scoring_result_can_be_filtered_by_the_weakest_link() -> None:
    result = score_candidates(
        [_candidate("Dead"), _candidate("Healthy")],
        scores=[
            {"relevance": 0.9, "novelty": 0.0},
            {"relevance": 0.6, "novelty": 0.5},
        ],
        rubric=_rubric(),
    )

    assert result.titles == ["Dead", "Healthy"]

    wrapped = as_weakest_link_result(result)
    filtered = filter_weakest_link(result)

    assert wrapped.titles == ["Healthy"]
    assert as_weakest_link_result(wrapped) is wrapped
    assert [candidate.title for candidate in filtered] == ["Healthy"]
    assert filter_weakest_link(result, min_dimension_score=0.0) == result.candidates


def test_filter_weakest_link_accepts_scores_and_explicit_limits() -> None:
    scores = [
        score_topic(
            _candidate("Dead"),
            scores={"relevance": 0.9, "novelty": 0.0},
            rubric=_rubric(),
        ),
        score_topic(
            _candidate("Healthy"),
            scores={"relevance": 0.9, "novelty": 0.5},
            rubric=_rubric(),
        ),
    ]

    assert [candidate.title for candidate in filter_weakest_link(scores)] == ["Healthy"]
    assert [
        candidate.title
        for candidate in filter_weakest_link(scores[0], min_dimension_score=0.0)
    ] == ["Dead"]
    assert [
        candidate.title
        for candidate in filter_weakest_link(
            scores,
            min_score=0.9,
            min_dimension_score=0.0,
        )
    ] == []


def test_the_weakest_link_helpers_read_scores_and_dimension_mappings() -> None:
    dimensions = {"relevance": 0.9, "novelty": 0.2, "reason": "clear"}

    assert weakest_dimension(dimensions) == "novelty"
    assert weakest_score(dimensions) == pytest.approx(0.2)
    assert passes_weakest_link(dimensions, 0.3) is False
    assert passes_weakest_link(dimensions, 0.2) is True
    assert weakest_dimension({"relevance": 0.9}) == "relevance"

    score = score_topic(
        _candidate("One"),
        scores={"relevance": 0.9, "novelty": 0.2},
        rubric=_rubric(),
    )
    assert weakest_dimension(score) == "novelty"
    assert weakest_score(score) == pytest.approx(0.2)
    assert passes_weakest_link(score) is False
    assert passes_weakest_link(score, 0.2) is True


def test_invalid_floors_are_rejected() -> None:
    with pytest.raises(ConfigError, match="between zero and one"):
        filter_weakest_link([], min_dimension_score=1.5)
    bad_floor: Any = "high"
    with pytest.raises(ConfigError, match="must be a number"):
        filter_weakest_link([], min_dimension_score=bad_floor)
    with pytest.raises(ConfigError, match="cannot read dimensions"):
        weakest_dimension("not a score")


def test_the_weakest_link_aliases_share_one_implementation() -> None:
    assert WeakestLinkScore is CandidateScore
    assert WeakestLinkScorer is WeakestLinkStage
    assert weakest_link is score_weakest_link
    assert score_candidates_weakest_link is score_weakest_link
    assert score_topics_weakest_link is score_weakest_link
    assert build_weakest_link_stage is weakest_link_stage
    assert isinstance(weakest_link_stage(_rubric()), WeakestLinkStage)
    assert isinstance(as_weakest_link_result(ScoringResult()), WeakestLinkResult)

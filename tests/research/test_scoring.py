"""Scoring candidates with a versioned rubric, weights and a minimum score."""

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
    DEFAULT_RUBRIC_VERSION,
    DEFAULT_SCORING_MIN_SCORE,
    CandidateScore,
    RubricDimension,
    ScoredCandidate,
    ScoreResult,
    ScoringResult,
    ScoringRubric,
    ScoringStage,
    TopicScore,
    build_scoring_rubric,
    filter_scored,
    score_candidate,
    score_candidates,
    score_candidates_async,
    score_one,
    score_topic,
    score_topics,
)


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _candidate(title: str, score: float = 0.0) -> TopicCandidate:
    return TopicCandidate(
        title=title,
        score=score,
        source_urls=[f"https://example.test/{title.replace(' ', '_')}"],
    )


def _scores(relevance: float, novelty: float) -> dict[str, float]:
    return {"relevance": relevance, "novelty": novelty}


def test_the_default_rubric_is_versioned_and_uses_the_configured_weights() -> None:
    rubric = ScoringRubric()

    assert rubric.version == DEFAULT_RUBRIC_VERSION == "1"
    assert rubric.rubric_version == "1"
    assert rubric.min_score == DEFAULT_SCORING_MIN_SCORE == 0.6
    assert rubric.min_dimension_score == 0.0
    assert rubric.weights == {
        "relevance": 0.35,
        "originality": 0.25,
        "accuracy": 0.25,
        "engagement": 0.15,
    }
    assert rubric.dimensions == (
        "relevance",
        "originality",
        "accuracy",
        "engagement",
    )
    assert [criterion.name for criterion in rubric.criteria] == list(rubric.dimensions)


def test_a_rubric_can_be_built_from_scoring_config_or_an_app_mapping() -> None:
    config = ScoringConfig(
        rubric_version="2",
        weights={"relevance": 0.6, "novelty": 0.4},
        min_score=0.75,
        min_dimension_score=0.4,
    )

    from_config = ScoringRubric.from_config(config)
    from_mapping = build_scoring_rubric(
        {
            "scoring": {
                "rubric_version": "3",
                "dimension_weights": {"relevance": 0.5, "novelty": 0.5},
                "threshold": 0.8,
            }
        }
    )

    assert from_config.version == "2"
    assert from_config.weights == {"relevance": 0.6, "novelty": 0.4}
    assert from_config.min_score == 0.75
    assert from_config.min_dimension_score == 0.4
    assert from_mapping.version == "3"
    assert from_mapping.weights == {"relevance": 0.5, "novelty": 0.5}
    assert from_mapping.min_score == 0.8


def test_explicit_dimensions_and_weights_drive_the_weighted_mean() -> None:
    rubric = ScoringRubric(
        version="2",
        criteria=[
            RubricDimension(name="relevance", description="Fits the brief."),
            RubricDimension(name="novelty", description="Adds a new angle."),
        ],
        weights={"relevance": 0.5, "novelty": 0.5},
        min_score=0.6,
    )

    score = score_topic(
        _candidate("Solar storage"),
        scores=_scores(0.8, 0.4),
        rubric=rubric,
    )

    assert isinstance(score, CandidateScore)
    assert score.dimension_scores == {"relevance": 0.8, "novelty": 0.4}
    assert score.overall == pytest.approx(0.6)
    assert score.rubric_version == "2"
    assert score.version == "2"
    assert score.weights == {"relevance": 0.5, "novelty": 0.5}
    assert score.min_score == 0.6
    assert score.passed is True
    assert score.relevant is True


def test_the_minimum_score_filters_inclusively_and_can_be_disabled() -> None:
    rubric = ScoringRubric(
        weights={"relevance": 0.5, "novelty": 0.5},
        min_score=0.6,
    )
    candidates = [_candidate("Pass"), _candidate("Edge"), _candidate("Fail")]

    result = score_candidates(
        candidates,
        scores=[_scores(1.0, 1.0), _scores(0.6, 0.6), _scores(0.59, 0.59)],
        rubric=rubric,
    )

    assert isinstance(result, ScoringResult)
    assert isinstance(result, ScoreResult)
    assert result.titles == ["Pass", "Edge"]
    assert [score.title for score in result.failed] == ["Fail"]
    assert [candidate.title for candidate in result.dropped] == ["Fail"]
    assert result.count == 2
    assert result.input_count == 3
    assert result.dropped_count == 1
    assert result.min_score == 0.6
    assert result.version == "1"
    assert result.weights == {"relevance": 0.5, "novelty": 0.5}
    assert result.dimensions == ("relevance", "novelty")

    unfiltered = score_candidates(
        candidates,
        scores=[_scores(1.0, 1.0), _scores(0.6, 0.6), _scores(0.59, 0.59)],
        rubric=rubric,
        min_score=None,
    )
    assert unfiltered.titles == ["Pass", "Edge", "Fail"]


def test_every_result_records_the_rubric_version_and_per_dimension_scores() -> None:
    rubric = ScoringRubric(
        version="2026-11",
        weights={"relevance": 0.7, "novelty": 0.3},
        min_score=0.5,
    )

    result = score_candidates(
        [_candidate("One")],
        scores={"One": {"relevance": 0.9, "novelty": 0.2, "reason": "clear"}},
        rubric=rubric,
    )

    score = result.scores[0]
    assert result.version == "2026-11"
    assert result.metadata["rubric_version"] == "2026-11"
    assert result.metadata["weights"] == {"relevance": 0.7, "novelty": 0.3}
    assert result.metadata["min_score"] == 0.5
    assert score.dimensions == {"relevance": 0.9, "novelty": 0.2}
    assert score.overall == pytest.approx(0.69)
    assert score.reason == "clear"
    assert score.to_dict()["title"] == "One"
    assert score.to_dict()["rubric_version"] == "2026-11"
    assert score.to_dict()["dimensions"] == {"relevance": 0.9, "novelty": 0.2}
    assert result.to_dict()["scores"][0]["passed"] is True
    assert result.to_dict()["rubric"]["version"] == "2026-11"


def test_a_single_dimension_mapping_is_applied_to_every_candidate() -> None:
    rubric = ScoringRubric(
        weights={"relevance": 0.5, "novelty": 0.5},
        min_score=0.5,
    )

    result = score_candidates(
        [_candidate("One"), _candidate("Two")],
        scores={"relevance": 0.8, "novelty": 0.6},
        rubric=rubric,
    )

    assert result.titles == ["One", "Two"]
    assert all(score.overall == pytest.approx(0.7) for score in result.scores)


def test_explicit_scores_may_be_keyed_by_title_or_aligned_by_position() -> None:
    rubric = ScoringRubric(
        weights={"relevance": 0.5, "novelty": 0.5},
        min_score=0.5,
    )
    candidates = [_candidate("First"), _candidate("Second")]

    keyed = score_candidates(
        candidates,
        scores={"First": _scores(1.0, 1.0), "Second": _scores(0.0, 0.0)},
        rubric=rubric,
    )
    aligned = score_candidates(
        candidates,
        scores=[_scores(0.0, 0.0), _scores(1.0, 1.0)],
        rubric=rubric,
    )

    assert keyed.titles == ["First"]
    assert aligned.titles == ["Second"]


def test_a_scorer_can_return_a_mapping_a_number_or_a_sequence() -> None:
    rubric = ScoringRubric(
        weights={"relevance": 0.5, "novelty": 0.5},
        min_score=0.5,
    )

    def mapping_scorer(candidate: TopicCandidate) -> dict[str, float]:
        return _scores(1.0, 1.0) if "good" in candidate.title else _scores(0.0, 0.0)

    def number_scorer(candidate: TopicCandidate) -> float:
        return 0.9 if "good" in candidate.title else 0.1

    def sequence_scorer(candidate: TopicCandidate) -> list[float]:
        return [1.0, 1.0] if "good" in candidate.title else [0.0, 0.0]

    candidates = [_candidate("A good topic"), _candidate("A bad topic")]

    assert score_candidates(
        candidates,
        scorer=mapping_scorer,
        rubric=rubric,
    ).titles == ["A good topic"]
    assert score_candidates(
        candidates,
        scorer=number_scorer,
        rubric=rubric,
    ).titles == ["A good topic"]
    assert score_candidates(
        candidates, scorer=sequence_scorer, rubric=rubric
    ).titles == ["A good topic"]


def test_an_async_scorer_is_awaited_by_the_async_entry_point() -> None:
    rubric = ScoringRubric(
        weights={"relevance": 0.5, "novelty": 0.5},
        min_score=0.5,
    )

    async def scorer(candidate: TopicCandidate) -> dict[str, float]:
        return _scores(1.0, 1.0) if candidate.title == "Keep" else _scores(0.0, 0.0)

    candidates = [_candidate("Keep"), _candidate("Drop")]

    result = _run(score_candidates_async(candidates, scorer=scorer, rubric=rubric))

    assert result.titles == ["Keep"]
    assert result.scores[1].passed is False


def test_a_sync_entry_point_rejects_an_async_scorer_cleanly() -> None:
    async def scorer(candidate: TopicCandidate) -> dict[str, float]:
        return _scores(1.0, 1.0)

    with pytest.raises(ConfigError, match="asynchronous"):
        score_candidates([_candidate("One")], scorer=scorer)


def test_a_candidate_without_scores_uses_its_own_score_as_a_fallback() -> None:
    rubric = ScoringRubric(
        weights={"relevance": 0.5, "novelty": 0.5},
        min_score=0.5,
    )

    result = score_candidates(
        [_candidate("Strong", 0.8), _candidate("Weak", 0.2)],
        rubric=rubric,
    )

    assert result.titles == ["Strong"]
    assert result.scores[0].dimensions == {"relevance": 0.8, "novelty": 0.8}
    assert result.scores[1].overall == pytest.approx(0.2)


def test_the_stage_binds_the_rubric_scorer_and_minimum_score() -> None:
    rubric = ScoringRubric(
        version="7",
        weights={"relevance": 0.5, "novelty": 0.5},
        min_score=0.7,
    )

    def scorer(candidate: TopicCandidate) -> dict[str, float]:
        return _scores(0.9, 0.9) if "on" in candidate.title else _scores(0.1, 0.1)

    stage = ScoringStage(rubric, scorer=scorer)
    candidates = [_candidate("on topic"), _candidate("off topic")]

    result = stage.run(candidates)
    filtered = stage.filter(candidates)
    response = stage.to_response(candidates, query="solar")

    assert isinstance(stage, ScoringStage)
    assert stage.rubric.version == "7"
    assert stage.version == "7"
    assert stage.min_score == 0.7
    assert stage.threshold == 0.7
    assert stage.dimensions == ("relevance", "novelty")
    assert result.titles == ["on topic"]
    assert [candidate.title for candidate in filtered] == ["on topic"]
    assert isinstance(response, SourceResponse)
    assert response.provider == "scoring"
    assert response.query == "solar"
    assert response.titles == ["on topic"]


def test_the_stage_accepts_config_and_overrides_and_an_async_scorer() -> None:
    config = ScoringConfig(
        rubric_version="4",
        weights={"relevance": 0.5, "novelty": 0.5},
        min_score=0.5,
    )
    stage = ScoringStage(config=config, min_score=0.8)

    async def scorer(candidate: TopicCandidate) -> dict[str, float]:
        return _scores(0.9, 0.9)

    result = _run(stage.run_async([_candidate("One")], scorer=scorer))

    assert stage.rubric.version == "4"
    assert stage.min_score == 0.8
    assert result.titles == ["One"]


def test_score_helpers_and_aliases_share_one_implementation() -> None:
    rubric = ScoringRubric(
        weights={"relevance": 0.5, "novelty": 0.5},
        min_score=0.5,
    )
    candidate = _candidate("One")

    assert score_candidate(candidate, scores=_scores(1.0, 1.0), rubric=rubric).passed
    assert score_one(candidate, scores=_scores(1.0, 1.0), rubric=rubric).passed
    assert score_topics(
        [candidate], scores=_scores(1.0, 1.0), rubric=rubric
    ).titles == ["One"]
    assert ScoredCandidate is CandidateScore
    assert TopicScore is CandidateScore
    assert filter_scored(
        score_candidates(
            [candidate, _candidate("Two")],
            scores=[_scores(1.0, 1.0), _scores(0.0, 0.0)],
            rubric=rubric,
        )
    ) == [candidate]


def test_the_minimum_dimension_floor_is_carried_but_not_enforced_here() -> None:
    rubric = ScoringRubric(
        weights={"relevance": 0.5, "novelty": 0.5},
        min_score=0.5,
        min_dimension_score=0.8,
    )

    result = score_candidates(
        [_candidate("Lopsided")],
        scores=_scores(1.0, 0.0),
        rubric=rubric,
    )

    assert result.titles == ["Lopsided"]
    assert result.scores[0].overall == pytest.approx(0.5)
    assert result.metadata["min_dimension_score"] == 0.8
    assert rubric.min_weakest_dimension == 0.8


def test_a_brief_derivation_supplies_the_rubric_dimensions() -> None:
    from aivia.research import BriefDerivation, RelevanceRubric, RubricCriterion

    derivation = BriefDerivation(
        rubric=RelevanceRubric(
            summary="Prefer explanatory topics.",
            criteria=[
                RubricCriterion(name="relevance", description="Fits the brief."),
                RubricCriterion(name="novelty", description="Adds a new angle."),
            ],
        )
    )
    config = ScoringConfig(
        rubric_version="5",
        weights={"relevance": 0.6, "novelty": 0.4},
        min_score=0.5,
    )

    rubric = ScoringRubric.from_derivation(derivation, config)
    built = build_scoring_rubric(config, derivation=derivation)

    assert rubric.version == "5"
    assert rubric.dimensions == ("relevance", "novelty")
    assert rubric.weights == {"relevance": 0.6, "novelty": 0.4}
    assert [criterion.description for criterion in rubric.criteria] == [
        "Fits the brief.",
        "Adds a new angle.",
    ]
    assert built.dimensions == ("relevance", "novelty")


def test_invalid_weights_and_thresholds_are_rejected() -> None:
    with pytest.raises(ConfigError, match="sum to 1"):
        ScoringRubric.from_config({"weights": {"relevance": 0.5, "novelty": 0.4}})
    with pytest.raises(ConfigError, match="between zero and one"):
        ScoringRubric.from_config({"min_score": 1.5})
    with pytest.raises(ConfigError, match="between zero and one"):
        build_scoring_rubric(threshold=-0.1)
    with pytest.raises(ConfigError, match="dimension score"):
        score_candidates(
            [_candidate("One")],
            scores={"One": {"relevance": 1.5}},
        )

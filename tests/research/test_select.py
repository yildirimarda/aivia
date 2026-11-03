"""Select topics per run above the minimum score with category diversity."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

import pytest

from aivia.core.errors import ConfigError
from aivia.core.models.topic import TopicCandidate
from aivia.providers.protocols.source import SourceResponse
from aivia.research import (
    DEFAULT_CATEGORY,
    DEFAULT_LIMIT,
    DEFAULT_MAX_PER_CATEGORY,
    CandidateScore,
    ScoringRubric,
    SelectedTopic,
    Selection,
    SelectionDecision,
    SelectionResult,
    SelectionStage,
    SelectionStatus,
    Selector,
    SelectResult,
    SelectStage,
    TopicSelector,
    build_selection_stage,
    category_counts,
    category_of,
    choose_topics,
    filter_selected,
    normalise_category,
    passes_selection,
    score_candidates,
    select,
    select_candidate,
    select_candidates,
    select_candidates_async,
    select_topics,
    selected_topics,
    selection_stage,
)


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _candidate(title: str, *, score: float = 0.0, category: str = "") -> TopicCandidate:
    return TopicCandidate(
        title=title,
        score=score,
        category=category,
        source_urls=[f"https://example.test/{title.replace(' ', '_')}"],
    )


def _rubric(**overrides: Any) -> ScoringRubric:
    values: dict[str, Any] = {
        "weights": {"relevance": 0.5, "novelty": 0.5},
        "min_score": 0.6,
    }
    values.update(overrides)
    return ScoringRubric(**values)


def _dimensions(relevance: float, novelty: float) -> dict[str, float]:
    return {"relevance": relevance, "novelty": novelty}


def test_the_defaults_keep_one_topic_per_category_without_a_run_limit() -> None:
    assert DEFAULT_LIMIT is None
    assert DEFAULT_MAX_PER_CATEGORY == 1
    assert DEFAULT_CATEGORY == "uncategorized"


def test_only_topics_above_the_minimum_score_are_selected_in_score_order() -> None:
    candidates = [
        _candidate("Low", score=0.4, category="science"),
        _candidate("High", score=0.9, category="science"),
        _candidate("Edge", score=0.6, category="science"),
    ]

    result = select_topics(candidates, min_score=0.6, max_per_category=None)

    assert isinstance(result, SelectionResult)
    assert isinstance(result, SelectResult)
    assert isinstance(result, Selection)
    assert result.titles == ["High", "Edge"]
    assert result.all_titles == ["High", "Edge", "Low"]
    assert result.rejected_titles == ["Low"]
    assert result.count == 2
    assert result.selected_count == 2
    assert result.input_count == 3
    assert result.rejected_count == 1
    assert result.dropped_count == 1
    assert result.min_score == 0.6
    assert result.max_per_category is None
    assert [decision.rank for decision in result.selected_decisions] == [1, 2]
    assert result.decisions[2].status is SelectionStatus.BELOW_MIN_SCORE
    assert result.decisions[2].reason == "below minimum score"
    assert result.decisions[2].below_min_score is True
    assert result.decisions[2].passed is False


def test_equal_scores_keep_their_input_order() -> None:
    candidates = [
        _candidate("First", score=0.8, category="a"),
        _candidate("Second", score=0.8, category="b"),
        _candidate("Third", score=0.8, category="c"),
    ]

    result = select_topics(candidates, max_per_category=None)

    assert result.titles == ["First", "Second", "Third"]
    assert [decision.rank for decision in result.selected_decisions] == [1, 2, 3]


def test_the_category_limit_is_case_insensitive_and_groups_blanks() -> None:
    candidates = [
        _candidate("Science one", score=0.9, category="Science"),
        _candidate("Science two", score=0.8, category="science"),
        _candidate("History", score=0.7, category="History"),
        _candidate("Unlabelled one", score=0.6),
        _candidate("Unlabelled two", score=0.5),
    ]

    result = select_topics(candidates, max_per_category=1)

    assert result.titles == ["Science one", "History", "Unlabelled one"]
    assert result.rejected_titles == ["Science two", "Unlabelled two"]
    assert result.category_counts == {
        "science": 1,
        "history": 1,
        "uncategorized": 1,
    }
    assert result.categories == ["science", "history", "uncategorized"]
    assert result.by_category["science"] == [candidates[0]]
    assert result.blocked_by_category[0].reason == "category limit reached: science"
    assert result.blocked_by_category[0].status is SelectionStatus.CATEGORY_LIMIT


def test_the_per_run_limit_caps_the_total_after_the_category_check() -> None:
    candidates = [
        _candidate("Science", score=0.9, category="science"),
        _candidate("History", score=0.8, category="history"),
        _candidate("Art", score=0.7, category="art"),
    ]

    result = select_topics(candidates, limit=2, max_per_category=1)

    assert result.titles == ["Science", "History"]
    assert result.rejected_titles == ["Art"]
    assert result.limit == 2
    assert result.blocked_by_limit[0].reason == "run limit reached"
    assert result.blocked_by_limit[0].status is SelectionStatus.RUN_LIMIT
    assert result.metadata["run_limited"] == 1


def test_a_zero_or_missing_category_limit_disables_diversity() -> None:
    candidates = [
        _candidate("One", score=0.9, category="science"),
        _candidate("Two", score=0.8, category="science"),
    ]

    disabled_with_none = select_topics(candidates, max_per_category=None)
    disabled_with_zero = select_topics(candidates, max_per_category=0)

    assert disabled_with_none.titles == ["One", "Two"]
    assert disabled_with_zero.titles == ["One", "Two"]
    assert disabled_with_none.max_per_category is None
    assert disabled_with_zero.max_per_category is None


def test_a_zero_run_limit_selects_nothing() -> None:
    result = select_topics(
        [_candidate("One", score=0.9, category="science")],
        limit=0,
    )

    assert result.titles == []
    assert result.rejected_titles == ["One"]
    assert result.blocked_by_limit[0].status is SelectionStatus.RUN_LIMIT


def test_a_scoring_result_supplies_its_rubric_minimum_and_scores() -> None:
    candidates = [
        _candidate("Strong", category="science"),
        _candidate("Weak", category="history"),
    ]
    scored = score_candidates(
        candidates,
        scores=[_dimensions(0.9, 0.9), _dimensions(0.4, 0.4)],
        rubric=_rubric(),
    )

    result = select_topics(scored)

    assert result.min_score == 0.6
    assert result.titles == ["Strong"]
    assert result.rejected_titles == ["Weak"]
    assert result.decisions[0].candidate_score is scored.scores[0]
    assert result.scores == list(scored.scores)


def test_an_explicit_none_minimum_disables_the_rubric_minimum() -> None:
    candidates = [
        _candidate("Strong", category="science"),
        _candidate("Weak", category="history"),
    ]
    scored = score_candidates(
        candidates,
        scores=[_dimensions(0.9, 0.9), _dimensions(0.4, 0.4)],
        rubric=_rubric(),
    )

    result = select_topics(scored, min_score=None, max_per_category=None)

    assert result.min_score is None
    assert result.titles == ["Strong", "Weak"]


def test_a_list_of_candidate_scores_is_accepted_directly() -> None:
    scores = [
        CandidateScore(
            candidate=_candidate("Strong", category="science"),
            dimensions=_dimensions(0.9, 0.9),
            min_score=0.6,
        ),
        CandidateScore(
            candidate=_candidate("Weak", category="history"),
            dimensions=_dimensions(0.4, 0.4),
            min_score=0.6,
        ),
    ]

    result = select_candidates(scores)

    assert result.titles == ["Strong"]
    assert result.decisions[1].below_min_score is True


def test_raw_candidates_can_be_scored_and_selected_in_one_call() -> None:
    candidates = [
        _candidate("Strong", category="science"),
        _candidate("Weak", category="history"),
    ]

    result = select(
        candidates,
        scores=[_dimensions(0.9, 0.9), _dimensions(0.4, 0.4)],
        rubric=_rubric(),
        max_per_category=1,
    )

    assert result.titles == ["Strong"]
    assert result.min_score == 0.6


def test_option_aliases_resolve_to_the_same_limits() -> None:
    candidates = [
        _candidate("One", score=0.9, category="science"),
        _candidate("Two", score=0.8, category="history"),
        _candidate("Three", score=0.7, category="art"),
    ]

    result = choose_topics(
        candidates,
        max_topics=2,
        per_category=1,
        threshold=0.5,
    )

    assert result.titles == ["One", "Two"]
    assert result.limit == 2
    assert result.max_per_category == 1
    assert result.min_score == 0.5


def test_conflicting_aliases_fail_clearly() -> None:
    with pytest.raises(ConfigError, match="conflicting limit values"):
        select_topics([_candidate("One", score=0.9)], limit=1, max_topics=2)
    with pytest.raises(ConfigError, match="conflicting max_per_category"):
        select_topics(
            [_candidate("One", score=0.9)],
            max_per_category=1,
            per_category=2,
        )
    with pytest.raises(ConfigError, match="conflicting min_score"):
        select_topics(
            [_candidate("One", score=0.9)],
            min_score=0.5,
            threshold=0.6,
        )


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("limit", -1, "selection limit must not be negative"),
        ("limit", 1.5, "selection limit must be an integer"),
        ("max_per_category", -1, "max per category must not be negative"),
        ("max_per_category", "one", "max per category must be an integer"),
        ("min_score", 2, "minimum score must be between zero and one"),
        ("min_score", "high", "minimum score must be a number"),
    ],
)
def test_invalid_limits_are_rejected(option: str, value: Any, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        select_topics([_candidate("One", score=0.9)], **{option: value})


def test_category_helpers_normalise_and_count() -> None:
    assert normalise_category("  Science  ") == "science"
    assert normalise_category("") == DEFAULT_CATEGORY
    assert category_of(_candidate("One", category="History")) == "history"
    decisions = select_topics(
        [
            _candidate("One", score=0.9, category="Science"),
            _candidate("Two", score=0.8, category="History"),
        ],
        max_per_category=None,
    ).selected_decisions
    assert category_counts(decisions) == {"science": 1, "history": 1}
    assert category_counts(
        [_candidate("Three", category="Science"), _candidate("Four")]
    ) == {"science": 1, "uncategorized": 1}


def test_result_reads_like_the_topic_list_and_serialises() -> None:
    candidate = _candidate("One", score=0.9, category="science")
    result = select_topics([candidate])

    assert list(result) == [candidate]
    assert len(result) == 1
    assert result[0] is candidate
    assert result[:] == [candidate]
    assert result == [candidate]
    assert result.filter() == [candidate]
    assert result.kept == [candidate]
    assert result.topics == [candidate]
    assert result.selected == [candidate]
    assert result.chosen == [candidate]
    assert result.picked == [candidate]
    assert result.all_candidates == [candidate]
    assert result.to_dict()["selected"][0]["title"] == "One"
    assert result.to_dict()["category_counts"] == {"science": 1}
    response = result.to_response()
    assert isinstance(response, SourceResponse)
    assert response.provider == "select"
    assert response.candidates == [candidate]


def test_selection_decision_exposes_status_and_serialises() -> None:
    candidate = _candidate("One", score=0.9, category="science")
    decision = select_candidate(candidate)

    assert isinstance(decision, SelectionDecision)
    assert isinstance(decision, SelectedTopic)
    assert decision.selected is True
    assert decision.rejected is False
    assert decision.status is SelectionStatus.SELECTED
    assert decision.reason == "selected"
    assert decision.title == "One"
    assert decision.overall == 0.9
    assert decision.category_key == "science"
    assert decision.to_dict()["status"] == "selected"
    assert decision.to_dict()["candidate"]["title"] == "One"


def test_one_candidate_helpers_and_filters() -> None:
    passing = _candidate("Pass", score=0.9)
    failing = _candidate("Fail", score=0.1)

    assert passes_selection(passing, min_score=0.5) is True
    assert passes_selection(failing, min_score=0.5) is False
    result = select_topics([passing, failing], min_score=0.5, max_per_category=None)
    assert selected_topics(result) == [passing]
    assert filter_selected(result) == [passing]
    assert selected_topics([passing, failing], max_per_category=None) == [
        passing,
        failing,
    ]


def test_the_selection_stage_binds_its_limits_and_accepts_overrides() -> None:
    stage = SelectionStage(limit=2, max_per_category=1, min_score=0.5)
    candidates = [
        _candidate("Science one", score=0.9, category="science"),
        _candidate("Science two", score=0.8, category="science"),
        _candidate("History", score=0.7, category="history"),
    ]

    result = stage.select(candidates)

    assert isinstance(stage, (Selector, TopicSelector, SelectStage))
    assert isinstance(stage, selection_stage)
    assert stage.name == "select"
    assert stage.limit == 2
    assert stage.category_limit == 1
    assert stage.threshold == 0.5
    assert result.titles == ["Science one", "History"]
    assert stage.run(candidates, max_per_category=None).titles == [
        "Science one",
        "Science two",
    ]
    assert stage.filter(candidates) == result.candidates
    assert stage.to_response(candidates).candidates == result.candidates


def test_the_stage_uses_the_rubric_minimum_when_none_is_bound() -> None:
    candidates = [
        _candidate("Strong", category="science"),
        _candidate("Weak", category="history"),
    ]
    scored = score_candidates(
        candidates,
        scores=[_dimensions(0.9, 0.9), _dimensions(0.4, 0.4)],
        rubric=_rubric(),
    )
    stage = SelectionStage(max_per_category=None)

    result = stage.select(scored)

    assert stage.min_score is None
    assert result.min_score == 0.6
    assert result.titles == ["Strong"]


def test_the_stage_builder_accepts_the_same_options() -> None:
    stage = build_selection_stage(limit=1, per_category=1, threshold=0.5)

    assert isinstance(stage, SelectionStage)
    assert stage.limit == 1
    assert stage.max_per_category == 1
    assert stage.min_score == 0.5


def test_an_async_scorer_is_awaited_before_selection() -> None:
    candidates = [
        _candidate("Strong", category="science"),
        _candidate("Weak", category="history"),
    ]

    async def scorer(candidate: TopicCandidate) -> dict[str, float]:
        if candidate.title == "Strong":
            return _dimensions(0.9, 0.9)
        return _dimensions(0.4, 0.4)

    result = _run(
        select_candidates_async(
            candidates,
            scorer=scorer,
            rubric=_rubric(),
            max_per_category=1,
        )
    )

    assert result.titles == ["Strong"]
    assert result.rejected_titles == ["Weak"]


def test_the_stage_async_variant_uses_its_bound_rubric() -> None:
    candidates = [
        _candidate("Strong", category="science"),
        _candidate("Weak", category="history"),
    ]

    async def scorer(candidate: TopicCandidate) -> dict[str, float]:
        if candidate.title == "Strong":
            return _dimensions(0.9, 0.9)
        return _dimensions(0.4, 0.4)

    stage = SelectionStage(max_per_category=1, scorer=scorer, rubric=_rubric())

    result = _run(stage.run_async(candidates))

    assert result.titles == ["Strong"]

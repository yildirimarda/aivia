"""Hook candidates are pre-scored by a deterministic weakest-property heuristic."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError as PydanticValidationError

from aivia.core.errors import ConfigError
from aivia.writing import (
    DEFAULT_HOOK_MIN_SCORE,
    DEFAULT_HOOK_PROPERTY_WEIGHTS,
    DEFAULT_HOOK_WEAKEST_WEIGHT,
    HOOK_PROPERTIES,
    HookCandidate,
    HookCandidates,
    HookCandidateScore,
    HookPreFilter,
    HookPrefilter,
    HookPreFilterConfig,
    HookPrefilterConfig,
    HookPreFilterResult,
    HookPrefilterResult,
    HookPreFilterScore,
    HookSelection,
    best_hook_candidate,
    choose_hook_by_prefilter,
    filter_hook_candidates,
    filter_hooks,
    hook_property_scores,
    pre_score_hook_candidate,
    pre_score_hook_candidates,
    prefilter_hook_candidate,
    prefilter_hook_candidates,
    prefilter_hooks,
    prescore_hook_candidate,
    prescore_hook_candidates,
    rank_hook_candidates,
    rank_hooks,
    score_hook,
    score_hook_candidate,
    score_hook_candidates,
    score_hooks,
    weakest_hook_property,
)


def test_every_property_is_scored_between_zero_and_one() -> None:
    scores = hook_property_scores(
        "In 1969, why did one phone call change the Roman plan?"
    )

    assert set(scores) == set(HOOK_PROPERTIES)
    assert all(0.0 <= value <= 1.0 for value in scores.values())
    assert scores["specificity"] >= 0.6
    assert scores["curiosity"] >= 0.6


def test_the_length_property_penalises_lines_outside_the_ideal_band() -> None:
    short = hook_property_scores("Stop.")["length"]
    ideal = hook_property_scores("Most people get this wrong.")["length"]
    long = hook_property_scores(
        "This is a very long hook that keeps going and going and explains every "
        "single detail before the viewer has any reason to care about the ending."
    )["length"]

    assert short < ideal
    assert ideal == 1.0
    assert long < ideal


def test_the_clarity_property_penalises_long_words_and_many_clauses() -> None:
    clear = hook_property_scores("Most people get this wrong.")["clarity"]
    unclear = hook_property_scores(
        "Notwithstanding the aforementioned circumstances, the "
        "incomprehensible concatenation remains disproportionately "
        "unintelligible, counterproductive and unnecessarily complicated."
    )["clarity"]

    assert unclear < clear


def test_specificity_rewards_numbers_names_times_and_quotations() -> None:
    vague = hook_property_scores("Something happened and it changed everything.")
    concrete = hook_property_scores(
        'In 1969, "Apollo" changed the plan for 3 million people.'
    )

    assert concrete["specificity"] > vague["specificity"]
    assert concrete["specificity"] >= 0.8


def test_curiosity_rewards_a_question_or_an_open_loop() -> None:
    plain = hook_property_scores("The road was built a long time ago.")
    question = hook_property_scores("Why do we still build roads the Roman way?")
    open_loop = hook_property_scores("Nobody tells you what happened next.")

    assert question["curiosity"] > plain["curiosity"]
    assert open_loop["curiosity"] > plain["curiosity"]


def test_naturalness_penalises_links_shouting_and_repeated_punctuation() -> None:
    clean = hook_property_scores("Most people get this wrong.")["naturalness"]
    noisy = hook_property_scores("STOP!!! VISIT https://example.com NOW!!!")[
        "naturalness"
    ]

    assert clean == 1.0
    assert noisy < 0.5


def test_the_final_score_is_weighted_by_the_weakest_property() -> None:
    config = HookPrefilterConfig(
        weights={"length": 1.0, "clarity": 1.0, "specificity": 1.0},
        min_score=0.0,
    )
    higher_mean_but_weaker = score_hook_candidate(
        "Nobody tells you what happened next.", config=config
    )
    lower_mean_but_balanced = score_hook_candidate(
        "STOP!!! VISIT https://example.com NOW!!!", config=config
    )

    assert higher_mean_but_weaker.weighted_mean > lower_mean_but_balanced.weighted_mean
    assert higher_mean_but_weaker.weakest_score < lower_mean_but_balanced.weakest_score
    assert higher_mean_but_weaker.score < lower_mean_but_balanced.score
    assert higher_mean_but_weaker.score == pytest.approx(
        higher_mean_but_weaker.weighted_mean * (1 - config.weakest_weight)
        + higher_mean_but_weaker.weakest_score * config.weakest_weight
    )


def test_the_weakest_property_share_is_configurable() -> None:
    text = "Nine in ten people miss this simple pattern."
    mean_only = score_hook_candidate(
        text, config=HookPrefilterConfig(weakest_weight=0.0, min_score=0.0)
    )
    weakest_only = score_hook_candidate(
        text, config=HookPrefilterConfig(weakest_weight=1.0, min_score=0.0)
    )

    assert mean_only.score == pytest.approx(mean_only.weighted_mean)
    assert weakest_only.score == pytest.approx(weakest_only.weakest_score)
    assert weakest_only.score < mean_only.score


def test_a_partial_weight_mapping_restricts_the_scored_properties() -> None:
    text = "Nine in ten people miss this simple pattern."
    score = score_hook_candidate(text, weights={"specificity": 1.0})

    assert score.weights == {"specificity": 1.0}
    assert score.weakest_property == "specificity"
    assert score.score == pytest.approx(score.properties["specificity"])


def test_the_prefilter_ranks_candidates_and_records_the_best_index() -> None:
    result = prefilter_hook_candidates(
        [
            "STOP!!! VISIT https://example.com NOW!!!",
            "Why do we still build roads the Roman way?",
            "This is a very long hook that keeps going and going and explains "
            "every single detail before the viewer has any reason to care.",
            "Nine in ten people miss this simple pattern.",
        ]
    )

    assert result.best_index == 1
    assert result.chosen == HookCandidate(
        text="Why do we still build roads the Roman way?"
    )
    assert result.hook == "Why do we still build roads the Roman way?"
    assert [score.index for score in result.ranked] == [1, 3, 2, 0]
    assert [score.index for score in result.accepted] == [1, 2, 3]
    assert [score.index for score in result.rejected] == [0]
    assert result.best_index == result.chosen_index == result.selected_index


def test_the_acceptance_floor_can_reject_every_candidate() -> None:
    config = HookPrefilterConfig(
        weights={"length": 1.0},
        min_score=0.5,
    )
    result = prefilter_hook_candidates(
        [
            "This is a very long hook that keeps going and going and explains "
            "every single detail before the viewer has any reason to care."
        ],
        config=config,
    )

    assert result.ok is False
    assert result.best is not None
    assert result.best.score < config.min_score
    assert result.accepted == ()
    assert "length" in result.reason
    with pytest.raises(ConfigError, match="no hook candidate passed"):
        result.require()
    with pytest.raises(ConfigError, match="no hook candidate passed"):
        choose_hook_by_prefilter(
            [
                "This is a very long hook that keeps going and going and explains "
                "every single detail before the viewer has any reason to care."
            ],
            config=config,
        )


def test_a_failed_score_records_the_property_that_dragged_it_down() -> None:
    score = score_hook_candidate("STOP!!! VISIT https://example.com NOW!!!")

    assert score.weakest_property == "naturalness"
    assert score.rejected is True
    assert any("naturalness" in reason for reason in score.reasons)
    assert score.weakest_link == (score.weakest_property, score.weakest_score)


def test_candidates_can_be_text_mappings_candidate_models_or_a_full_selection() -> None:
    candidates = [
        {"text": "Why does this keep happening?", "formula": "question"},
        HookCandidate(text="Most people get this wrong.", formula="contrarian"),
    ]
    from_models = score_hook_candidates(candidates)
    from_strings = score_hook_candidates(
        ["Why does this keep happening?", "Most people get this wrong."]
    )
    selection = HookSelection(
        candidates=[
            HookCandidate(text="Why does this keep happening?", formula="question"),
            HookCandidate(text="Most people get this wrong.", formula="contrarian"),
        ],
        selected_index=0,
        reason="The question opens a loop.",
    )
    from_selection = score_hook_candidates(selection)

    assert [candidate.text for candidate in from_models.candidates] == [
        candidate.text for candidate in from_strings.candidates
    ]
    assert from_selection.candidates == from_models.candidates
    assert from_models[0].formula == "question"
    assert from_models[1].formula == "contrarian"


def test_the_same_candidates_always_produce_the_same_scores() -> None:
    candidates = [
        "Why do we still build roads the Roman way?",
        "Nine in ten people miss this simple pattern.",
        "Most people get this wrong.",
    ]

    first = score_hook_candidates(candidates)
    second = score_hook_candidates(candidates)

    assert first.to_dict() == second.to_dict()


def test_the_prompt_context_shows_the_pre_score_to_the_model() -> None:
    result = prefilter_hook_candidates(
        [
            "Nine in ten people miss this simple pattern.",
            "Why do we still build roads the Roman way?",
        ]
    )
    context = result.to_prompt_context()

    assert [item["index"] for item in context] == [1, 0]
    assert result.best is not None
    assert context[0]["pre_score"] == pytest.approx(result.best.score)
    assert context[0]["weakest_property"] in HOOK_PROPERTIES
    assert set(context[0]["properties"]) == set(HOOK_PROPERTIES)


def test_the_reusable_filter_binds_the_config_and_exposes_the_helpers() -> None:
    prefilter = HookPrefilter(
        HookPrefilterConfig(weights={"curiosity": 1.0}, min_score=0.5)
    )
    candidates = [
        "The road was built a long time ago.",
        "Why do we still build roads the Roman way?",
    ]

    assert prefilter.weights == {"curiosity": 1.0}
    assert prefilter.min_score == 0.5
    assert prefilter.weakest_weight == DEFAULT_HOOK_WEAKEST_WEIGHT
    assert prefilter.score(candidates[0]).weakest_property == "curiosity"
    assert prefilter.score_all(candidates).best_index == 1
    assert prefilter.prefilter(candidates).best_index == 1
    assert [score.index for score in prefilter.rank(candidates)] == [1, 0]
    assert prefilter.filter(candidates) == (
        HookCandidate(text="Why do we still build roads the Roman way?"),
    )
    best = prefilter.best(candidates)
    assert best is not None
    assert best.index == 1
    assert prefilter.choose(candidates).text == (
        "Why do we still build roads the Roman way?"
    )
    assert prefilter.prompt_context(candidates)[0]["index"] == 1


def test_module_aliases_and_defaults_are_stable() -> None:
    assert DEFAULT_HOOK_PROPERTY_WEIGHTS == {
        "length": 0.20,
        "clarity": 0.20,
        "specificity": 0.25,
        "curiosity": 0.25,
        "naturalness": 0.10,
    }
    assert sum(DEFAULT_HOOK_PROPERTY_WEIGHTS.values()) == pytest.approx(1.0)
    assert DEFAULT_HOOK_WEAKEST_WEIGHT == 0.5
    assert DEFAULT_HOOK_MIN_SCORE == 0.40
    assert HookPreFilter is HookPrefilter
    assert HookPreFilterConfig is HookPrefilterConfig
    assert HookPreFilterResult is HookPrefilterResult
    assert HookPreFilterScore is HookCandidateScore
    assert prefilter_hook_candidate is score_hook_candidate
    assert pre_score_hook_candidate is score_hook_candidate
    assert prescore_hook_candidate is score_hook_candidate
    assert pre_score_hook_candidates is score_hook_candidates
    assert prescore_hook_candidates is score_hook_candidates
    assert prefilter_hooks is prefilter_hook_candidates
    assert filter_hooks is filter_hook_candidates
    assert rank_hooks is rank_hook_candidates
    assert score_hook is score_hook_candidate
    assert score_hooks is score_hook_candidates
    best = best_hook_candidate(["One.", "Two."])
    assert best is not None
    assert best.index == 0
    weakest_name, weakest_value = weakest_hook_property("One.")
    assert weakest_name in HOOK_PROPERTIES
    assert 0.0 <= weakest_value <= 1.0


def test_invalid_config_and_candidates_fail_clearly() -> None:
    with pytest.raises(PydanticValidationError, match="unknown hook property"):
        HookPrefilterConfig(weights={"charisma": 1.0})
    with pytest.raises(PydanticValidationError, match="non-negative"):
        HookPrefilterConfig(weights={"length": -1.0})
    with pytest.raises(PydanticValidationError, match="at least one"):
        HookPrefilterConfig(weights={})
    with pytest.raises(PydanticValidationError, match="must not all be zero"):
        HookPrefilterConfig(weights={"length": 0.0})
    with pytest.raises(PydanticValidationError, match="ideal_min_words"):
        HookPrefilterConfig(ideal_min_words=10, ideal_max_words=5)
    with pytest.raises(ConfigError, match="config"):
        score_hook_candidate("One.", config={"weights": {"charisma": 1.0}})
    with pytest.raises(ConfigError, match="config"):
        score_hook_candidate("One.", config=object())  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="candidate must be"):
        score_hook_candidate(3)
    with pytest.raises(ConfigError, match="non-negative integer"):
        score_hook_candidate("One.", index=-1)


def test_the_result_serializes_every_score_and_the_chosen_index() -> None:
    result = score_hook_candidates(
        [
            "Most people get this wrong.",
            "Why do we still build roads the Roman way?",
        ]
    )
    payload: dict[str, Any] = result.to_dict()

    assert payload["best_index"] == 1
    assert len(payload["scores"]) == 2
    assert payload["scores"][0]["text"] == "Most people get this wrong."
    assert payload["scores"][0]["weakest_property"] in HOOK_PROPERTIES
    assert payload["config"]["weights"] == dict(DEFAULT_HOOK_PROPERTY_WEIGHTS)
    assert isinstance(result.candidates, tuple)
    assert len(result) == 2
    assert list(result) == list(result.scores)


def test_the_full_candidate_set_is_accepted_without_losing_the_selection() -> None:
    output = HookCandidates.model_validate(
        {
            "candidates": [
                {"text": "Why does this keep happening?", "formula": "question"},
                {"text": "Most people get this wrong.", "formula": "contrarian"},
            ],
            "selected": 0,
            "reason": "The question opens a loop.",
        }
    )

    result = score_hook_candidates(output)

    assert result.best_index == 1
    assert result.hook == output.candidates[1].text
    assert result.candidates == tuple(output.candidates)

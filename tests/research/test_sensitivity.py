"""Filter sensitive topics into blocked or manual review with the rule recorded."""

from __future__ import annotations

from typing import Any

import pytest

from aivia.core.errors import ConfigError
from aivia.core.models.topic import TopicCandidate
from aivia.providers.protocols.source import SourceResponse
from aivia.research import (
    ALLOWED,
    BLOCKED,
    DEFAULT_RULES,
    DEFAULT_SENSITIVITY_RULES,
    MANUAL_REVIEW,
    SensitiveTopicFilter,
    SensitivityAction,
    SensitivityDecision,
    SensitivityFilter,
    SensitivityResult,
    SensitivityRule,
    SensitivityStage,
    build_sensitivity_stage,
    classify_candidate,
    filter_sensitive,
    filter_sensitive_topics,
    is_sensitive,
    screen_candidate,
    screen_candidates,
    sensitive_topics,
    sensitivity_stage,
)


def _candidate(
    title: str,
    *,
    summary: str = "",
    category: str = "",
    keywords: list[str] | None = None,
) -> TopicCandidate:
    return TopicCandidate(
        title=title,
        summary=summary,
        category=category,
        keywords=list(keywords or []),
        source_urls=[f"https://example.test/{title.replace(' ', '_')}"],
    )


def _rule(
    name: str,
    terms: list[str],
    action: str,
    **values: Any,
) -> SensitivityRule:
    return SensitivityRule(
        name=name,
        terms=terms,
        action=SensitivityAction.parse(action),
        **values,
    )


def test_a_default_block_rule_removes_a_topic_and_records_the_rule() -> None:
    result = screen_candidates(
        [
            _candidate("School shooting survivor speaks"),
            _candidate("Solar panel maintenance"),
        ]
    )

    assert isinstance(result, SensitivityResult)
    assert result.titles == ["Solar panel maintenance"]
    assert result.blocked_titles == ["School shooting survivor speaks"]
    assert result.review_titles == []
    decision = result.blocked_decisions[0]
    assert decision.action is SensitivityAction.BLOCK
    assert decision.blocked is True
    assert decision.allowed is False
    assert decision.rule == "graphic-violence"
    assert decision.matched_rule == "graphic-violence"
    assert decision.category == "violence"
    assert decision.matched_terms == ("school shooting",)
    assert "violence" in decision.reason
    assert result.metadata["input_count"] == 2
    assert result.metadata["allowed"] == 1
    assert result.metadata["blocked"] == 1
    assert result.metadata["manual_review"] == 0


def test_a_default_review_rule_holds_a_topic_without_blocking_it() -> None:
    result = screen_candidates([_candidate("Suicide prevention helpline")])

    assert result.candidates == []
    assert result.review_titles == ["Suicide prevention helpline"]
    assert result.blocked_titles == []
    decision = result.review_decisions[0]
    assert decision.action is SensitivityAction.REVIEW
    assert decision.review is True
    assert decision.manual_review is True
    assert decision.blocked is False
    assert decision.rule == "self-harm"
    assert decision.matched_terms == ("suicide",)
    assert result.manual_review == [decision.candidate]
    assert result.manual_review_count == 1
    assert result.review_count == 1
    assert result.blocked_count == 0


def test_an_ordinary_topic_is_allowed_with_no_rule_recorded() -> None:
    decision = screen_candidate(_candidate("A guide to software updates"))

    assert isinstance(decision, SensitivityDecision)
    assert decision.action is SensitivityAction.ALLOW
    assert decision.allowed is True
    assert decision.blocked is False
    assert decision.review is False
    assert decision.rule == ""
    assert decision.matched_rules == ()
    assert decision.matched_terms == ()
    assert not is_sensitive(_candidate("A guide to software updates"))


def test_matching_is_case_insensitive_whole_word_and_covers_every_field() -> None:
    rule = _rule("war", ["war"], "blocked")

    assert screen_candidate(_candidate("WAR in history"), rules=[rule]).blocked
    assert screen_candidate(
        _candidate("A topic", summary="The war ended"),
        rules=[rule],
    ).blocked
    assert screen_candidate(
        _candidate("A topic", category="War"),
        rules=[rule],
    ).blocked
    assert screen_candidate(
        _candidate("A topic", keywords=["war"]),
        rules=[rule],
    ).blocked
    assert screen_candidate(_candidate("Software release"), rules=[rule]).allowed
    assert screen_candidate(_candidate("A warning sign"), rules=[rule]).allowed


def test_punctuation_and_hyphen_variants_match_the_same_term() -> None:
    rule = _rule("self-harm", ["self-harm"], "manual_review")

    assert screen_candidate(_candidate("Self harm support"), rules=[rule]).review
    assert screen_candidate(_candidate("Self-harm support"), rules=[rule]).review


def test_a_block_rule_wins_over_a_review_rule_whichever_order_they_are_in() -> None:
    review = _rule("violence-review", ["violence"], "review")
    block = _rule("violence-block", ["massacre"], "blocked")

    decision = screen_candidate(
        _candidate("Massacre and violence"),
        rules=[review, block],
    )

    assert decision.action is SensitivityAction.BLOCK
    assert decision.rule == "violence-block"
    assert decision.matched_terms == ("massacre",)
    assert decision.matched_rules == ("violence-review", "violence-block")

    reversed_decision = screen_candidate(
        _candidate("Massacre and violence"),
        rules=[block, review],
    )
    assert reversed_decision.rule == "violence-block"
    assert reversed_decision.matched_rules == ("violence-block", "violence-review")


def test_the_first_rule_wins_a_tie_at_the_same_severity() -> None:
    first = _rule("first-block", ["massacre"], "blocked")
    second = _rule("second-block", ["violence"], "blocked")

    decision = screen_candidate(
        _candidate("Massacre and violence"),
        rules=[first, second],
    )

    assert decision.rule == "first-block"
    assert decision.matched_terms == ("massacre",)
    assert decision.matched_rules == ("first-block", "second-block")


def test_rules_can_be_supplied_as_mappings_with_action_aliases() -> None:
    decision = screen_candidate(
        _candidate("Election results"),
        rules=[
            {
                "id": "politics",
                "matches": ["election", "politics"],
                "decision": "manual",
                "category": "politics",
                "reason": "Needs a human check.",
            }
        ],
    )

    assert decision.action is SensitivityAction.REVIEW
    assert decision.rule == "politics"
    assert decision.category == "politics"
    assert decision.reason == "Needs a human check."
    assert decision.matched_terms == ("election",)


def test_explicit_blocked_and_review_terms_add_configured_rules() -> None:
    decision = screen_candidate(
        _candidate("Lottery winner"),
        blocked_terms=["lottery"],
    )
    review = screen_candidate(
        _candidate("Gambling regulation"),
        review_terms=["gambling"],
    )

    assert decision.action is SensitivityAction.BLOCK
    assert decision.rule == "blocked-terms"
    assert review.action is SensitivityAction.REVIEW
    assert review.rule == "review-terms"


def test_a_disabled_stage_allows_every_topic_and_records_it() -> None:
    stage = SensitivityStage(enabled=False)
    result = stage.run([_candidate("School shooting survivor speaks")])

    assert stage.enabled is False
    assert result.titles == ["School shooting survivor speaks"]
    assert result.blocked_titles == []
    assert result.review_titles == []
    assert result.metadata["enabled"] is False
    assert all(
        decision.action is SensitivityAction.ALLOW for decision in result.decisions
    )


def test_the_stage_filters_and_returns_a_source_response() -> None:
    stage = SensitivityStage()
    candidates = [
        _candidate("Solar panel maintenance"),
        _candidate("Suicide prevention helpline"),
        _candidate("School shooting survivor speaks"),
    ]

    filtered = stage.filter(candidates)
    response = stage.to_response(candidates, query="solar")

    assert isinstance(stage, SensitivityFilter)
    assert isinstance(stage, SensitiveTopicFilter)
    assert filtered == [candidates[0]]
    assert isinstance(response, SourceResponse)
    assert response.provider == "sensitivity"
    assert response.query == "solar"
    assert response.titles == ["Solar panel maintenance"]
    assert response.metadata["blocked"] == 1
    assert response.metadata["manual_review"] == 1


def test_module_helpers_and_builder_expose_the_same_behaviour() -> None:
    candidates = [
        _candidate("Solar panel maintenance"),
        _candidate("School shooting survivor speaks"),
    ]

    assert filter_sensitive(candidates) == [candidates[0]]
    assert filter_sensitive_topics(candidates) == [candidates[0]]
    assert sensitive_topics(candidates) == [candidates[0]]
    assert classify_candidate(candidates[1]).blocked
    assert screen_candidate(candidates[1]).blocked
    assert build_sensitivity_stage().rules == DEFAULT_SENSITIVITY_RULES
    assert sensitivity_stage is SensitivityStage


def test_the_result_serializes_the_matched_rule_for_every_decision() -> None:
    result = screen_candidates(
        [
            _candidate("Solar panel maintenance"),
            _candidate("Suicide prevention helpline"),
            _candidate("School shooting survivor speaks"),
        ]
    )

    payload = result.to_dict()

    assert payload["blocked"] == ["School shooting survivor speaks"]
    assert payload["manual_review"] == ["Suicide prevention helpline"]
    expected_titles = [
        *result.titles,
        "Suicide prevention helpline",
        "School shooting survivor speaks",
    ]
    assert [decision["title"] for decision in payload["decisions"]] == expected_titles
    by_title = {item["title"]: item for item in payload["decisions"]}
    assert by_title["School shooting survivor speaks"]["rule"] == "graphic-violence"
    assert by_title["School shooting survivor speaks"]["matched_rule"] == (
        "graphic-violence"
    )
    assert by_title["Suicide prevention helpline"]["action"] == "manual_review"
    assert by_title["Solar panel maintenance"]["action"] == "allow"
    assert by_title["Solar panel maintenance"]["rule"] == ""


def test_invalid_rules_fail_clearly() -> None:
    with pytest.raises(ConfigError, match="at least one term"):
        screen_candidates(
            [_candidate("x")],
            rules=[{"name": "empty", "terms": [], "action": "block"}],
        )
    with pytest.raises(ConfigError, match="unknown sensitivity action"):
        screen_candidates(
            [_candidate("x")],
            rules=[{"name": "bad", "terms": ["x"], "action": "explode"}],
        )
    with pytest.raises(ConfigError, match="duplicate sensitivity rule"):
        screen_candidates(
            [_candidate("x")],
            rules=[
                _rule("same", ["a"], "block"),
                _rule("SAME", ["b"], "review"),
            ],
        )
    with pytest.raises(ConfigError, match="rule must be a mapping"):
        screen_candidates([_candidate("x")], rules=[object()])


def test_default_rules_cover_both_blocked_and_review_actions() -> None:
    actions = {rule.action for rule in DEFAULT_RULES}

    assert DEFAULT_RULES is DEFAULT_SENSITIVITY_RULES
    assert SensitivityAction.BLOCK in actions
    assert SensitivityAction.REVIEW in actions
    assert all(rule.terms for rule in DEFAULT_RULES)
    assert len({rule.name for rule in DEFAULT_RULES}) == len(DEFAULT_RULES)
    assert ALLOWED is SensitivityAction.ALLOW
    assert BLOCKED is SensitivityAction.BLOCK
    assert MANUAL_REVIEW is SensitivityAction.REVIEW

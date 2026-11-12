"""Content safety gates: block lists, mandatory-review escalation and refusal.

The gate screens text against the configured ``blocked_terms`` and
``block_categories``. Clean content passes through. Blocked content is either
escalated to a mandatory (``pending``) review a human must answer, or refused
into the distinct ``refused`` terminal state, which is separate from a human
``rejected`` review.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from aivia.core.config.ops import ReviewConfig
from aivia.review.review_state import (
    IllegalReviewTransition,
    Review,
    ReviewState,
    ReviewStateMachine,
    _review_state,
)
from aivia.review.safety_gate import SafetyGate, SafetyGateResult

NOW = datetime(2026, 1, 2, 10, 0, tzinfo=UTC)


class Clock:
    """A mutable clock so tests never depend on wall time."""

    def __init__(self, now: datetime = NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def review_settings(**overrides: object) -> ReviewConfig:
    data: dict[str, object] = {
        "approval_mode": "ask",
        "timeout_seconds": 60,
        "on_timeout": "reject",
        "allow_regenerate": True,
        "max_regenerations": 3,
    }
    data.update(overrides)
    return ReviewConfig(**data)  # type: ignore[arg-type]


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def machine(clock: Clock) -> ReviewStateMachine:
    return ReviewStateMachine(
        settings={
            "default": review_settings(),
            "noregen": review_settings(allow_regenerate=False),
        },
        clock=clock,
    )


def safety_config(**overrides: object) -> dict[str, Any]:
    data: dict[str, Any] = {
        "enabled": True,
        "blocked_terms": ["badword", "forbidden phrase"],
        "block_categories": ["violence"],
    }
    data.update(overrides)
    return data


def make_gate(
    machine: ReviewStateMachine,
    *,
    profile: str = "default",
    config: Any = None,
) -> SafetyGate:
    return SafetyGate(reviews=machine, config=config, profile=profile)


def stored(machine: ReviewStateMachine, review_id: str) -> Review:
    review = machine.get(review_id)
    assert review is not None
    return review


def test_clean_content_passes_without_a_review(machine: ReviewStateMachine) -> None:
    result = make_gate(machine).screen("hello world")
    assert result.allowed is True
    assert result.blocked is False
    assert result.blocked_terms == ()
    assert result.review is None
    assert result.state is None
    assert result.reason == ""


def test_blocked_content_is_escalated_to_a_mandatory_review(
    machine: ReviewStateMachine,
) -> None:
    result = make_gate(machine, config=safety_config()).screen("this contains badword")
    assert result.allowed is False
    assert result.blocked is True
    assert result.blocked_terms == ("badword",)
    assert result.review is not None
    assert result.state is ReviewState.PENDING
    assert result.state == ReviewState.PENDING
    assert "mandatory review" in result.reason
    review = stored(machine, result.review.id)
    assert review.state is ReviewState.PENDING
    assert review.profile == "default"


def test_blocked_content_is_refused_when_escalation_is_disabled(
    machine: ReviewStateMachine,
) -> None:
    result = make_gate(machine, config=safety_config()).screen(
        "this contains badword", escalate=False
    )
    assert result.allowed is False
    assert result.blocked is True
    assert result.blocked_terms == ("badword",)
    assert result.review is not None
    assert result.state is ReviewState.REFUSED
    assert "refused" in result.reason
    review = stored(machine, result.review.id)
    assert review.state is ReviewState.REFUSED
    assert review.answered_at is not None
    assert review.answer_reason == "blocked by screening: badword"


def test_refused_is_a_distinct_terminal_state(machine: ReviewStateMachine) -> None:
    assert machine.can_transition(ReviewState.PENDING, ReviewState.REFUSED) is True
    assert machine.legal_transitions(ReviewState.REFUSED) == frozenset()
    assert machine.can_transition(ReviewState.REFUSED, ReviewState.APPROVED) is False
    assert machine.can_transition(ReviewState.REFUSED, ReviewState.PENDING) is False
    assert ReviewState("refused") is ReviewState.REFUSED
    assert _review_state("refuse") is ReviewState.REFUSED


def test_refused_review_cannot_be_transitioned_away(
    machine: ReviewStateMachine,
) -> None:
    result = make_gate(machine, config=safety_config()).screen(
        "badword", escalate=False
    )
    assert result.review is not None
    with pytest.raises(IllegalReviewTransition):
        machine.transition(result.review, ReviewState.APPROVED)
    with pytest.raises(IllegalReviewTransition):
        machine.transition(result.review, ReviewState.PENDING)
    stored_review = stored(machine, result.review.id)
    assert stored_review.state is ReviewState.REFUSED


def test_a_mandatory_review_can_be_approved_or_refused(
    machine: ReviewStateMachine,
) -> None:
    result = make_gate(machine, config=safety_config()).screen("badword")
    assert result.state is ReviewState.PENDING
    assert result.review is not None

    approved = machine.approve(result.review, actor="alice", reason="cleared")
    assert approved.state is ReviewState.APPROVED

    review = stored(machine, result.review.id)
    assert review.state is ReviewState.APPROVED
    assert review.answered_by == "alice"

    other = make_gate(machine, config=safety_config()).screen("forbidden phrase")
    assert other.review is not None
    refused = machine.refuse(other.review, actor="system", reason="not cleared")
    assert refused.state is ReviewState.REFUSED
    assert refused.answered_by == "system"


def test_blocked_terms_are_reported(machine: ReviewStateMachine) -> None:
    result = make_gate(machine, config=safety_config()).screen(
        "badword and forbidden phrase"
    )
    assert result.blocked_terms == ("badword", "forbidden phrase")


def test_blocked_term_matching_is_case_insensitive_and_whole_word(
    machine: ReviewStateMachine,
) -> None:
    clean = make_gate(machine, config=safety_config(blocked_terms=["ass"])).screen(
        "a classic example"
    )
    assert clean.allowed is True
    assert clean.blocked_terms == ()

    blocked = make_gate(machine, config=safety_config(blocked_terms=["ass"])).screen(
        "an ass in the room"
    )
    assert blocked.blocked_terms == ("ass",)

    blocked_upper = make_gate(
        machine, config=safety_config(blocked_terms=["ASS"])
    ).screen("an ASS in the room")
    assert blocked_upper.blocked_terms == ("ASS",)


def test_block_categories_are_treated_as_terms(machine: ReviewStateMachine) -> None:
    result = make_gate(machine, config=safety_config()).screen("an act of violence")
    assert result.blocked is True
    assert "violence" in result.blocked_terms


def test_disabled_safety_passes_content(machine: ReviewStateMachine) -> None:
    result = make_gate(machine, config=safety_config(enabled=False)).screen("badword")
    assert result.allowed is True
    assert result.blocked is False
    assert result.review is None


def test_screen_content_kind_is_screened_like_content(
    machine: ReviewStateMachine,
) -> None:
    result = make_gate(machine, config=safety_config()).screen(
        "badword", kind="content"
    )
    assert result.blocked is True
    assert result.blocked_terms == ("badword",)

    clean = make_gate(machine, config=safety_config()).screen(
        "safe caption", kind="caption"
    )
    assert clean.allowed is True


def test_custom_terms_and_categories_are_checked(machine: ReviewStateMachine) -> None:
    result = make_gate(machine, config=safety_config()).screen(
        "spammy scam",
        terms=["spammy"],
        categories=["scam"],
    )
    assert result.blocked is True
    assert result.blocked_terms == ("spammy", "scam")


def test_gate_result_is_a_frozen_dataclass(machine: ReviewStateMachine) -> None:
    result = make_gate(machine, config=safety_config()).screen(
        "badword", escalate=False
    )
    with pytest.raises(AttributeError):
        result.allowed = True  # type: ignore[misc]
    assert isinstance(result, SafetyGateResult)

"""The review state machine: per-profile approval settings, legal transitions,
persistence on every change and clock-driven timeout behaviour."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aivia.core.config.ops import ReviewConfig
from aivia.core.errors import AiviaError
from aivia.review.review_state import (
    LEGAL_REVIEW_TRANSITIONS,
    REVIEW_STATE_TRANSITIONS,
    IllegalReviewTransition,
    InvalidReviewTransition,
    InvalidReviewTransitionError,
    Review,
    ReviewNotFoundError,
    ReviewState,
    ReviewStateMachine,
    ReviewStateStore,
    can_transition,
    is_legal_transition,
    legal_transitions,
    transition_review,
)

NOW = datetime(2026, 1, 2, 10, 0, tzinfo=UTC)
PLUS_5 = NOW + timedelta(minutes=5)
PLUS_10 = NOW + timedelta(minutes=10)
PLUS_60_SECONDS = NOW + timedelta(seconds=60)
PLUS_30_SECONDS = NOW + timedelta(seconds=30)

ALL_STATES = tuple(ReviewState)
LEGAL_TARGETS = LEGAL_REVIEW_TRANSITIONS


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
            "daily": review_settings(),
            "auto": review_settings(approval_mode="never"),
            "notify": review_settings(on_timeout="notify"),
            "approve": review_settings(on_timeout="approve"),
            "noregen": review_settings(allow_regenerate=False),
            "oneregen": review_settings(max_regenerations=1),
        },
        clock=clock,
    )


def stored(machine: ReviewStateMachine, review_id: str) -> Review:
    review = machine.get(review_id)
    assert review is not None
    return review


@pytest.mark.parametrize("source", ALL_STATES)
@pytest.mark.parametrize("target", ALL_STATES)
def test_only_the_declared_transitions_are_legal(
    source: ReviewState, target: ReviewState
) -> None:
    expected = target in LEGAL_TARGETS[source]

    assert can_transition(source, target) is expected
    assert is_legal_transition(source, target) is expected
    assert (target in legal_transitions(source)) is expected


def test_terminal_states_never_leave_themselves() -> None:
    for state in (
        ReviewState.APPROVED,
        ReviewState.REJECTED,
        ReviewState.TIMED_OUT,
    ):
        assert legal_transitions(state) == frozenset()


def test_regenerated_can_be_resubmitted() -> None:
    assert legal_transitions(ReviewState.REGENERATED) == {
        ReviewState.PENDING,
    }


def test_pending_offers_approve_reject_regenerate_and_timeout() -> None:
    assert legal_transitions(ReviewState.PENDING) == {
        ReviewState.APPROVED,
        ReviewState.REJECTED,
        ReviewState.REGENERATED,
        ReviewState.TIMED_OUT,
    }


def test_an_illegal_transition_raises_and_does_not_mutate() -> None:
    review = Review(id="r", profile="daily", state=ReviewState.PENDING, created_at=NOW)

    with pytest.raises(IllegalReviewTransition, match="pending to pending"):
        transition_review(review, ReviewState.PENDING)

    assert review.state is ReviewState.PENDING
    assert review.answered_at is None
    assert review.timeout_at is None


def test_the_transition_error_is_also_a_value_error_and_an_aivia_error() -> None:
    with pytest.raises(ValueError):
        transition_review(
            Review(id="r", profile="daily", state=ReviewState.APPROVED, created_at=NOW),
            ReviewState.REJECTED,
        )
    with pytest.raises(AiviaError):
        transition_review(
            Review(id="r", profile="daily", state=ReviewState.APPROVED, created_at=NOW),
            ReviewState.REJECTED,
        )


def test_error_aliases_point_at_the_same_class() -> None:
    assert InvalidReviewTransition is IllegalReviewTransition
    assert InvalidReviewTransitionError is IllegalReviewTransition
    assert ReviewStateStore is ReviewStateMachine
    assert REVIEW_STATE_TRANSITIONS is LEGAL_REVIEW_TRANSITIONS


def test_approve_stamps_answered_fields(machine: ReviewStateMachine) -> None:
    review = machine.create("r1", "daily", now=NOW)
    assert review.state is ReviewState.PENDING

    approved = machine.approve(
        review, now=PLUS_5, actor=" alice ", reason=" looks good "
    )
    assert approved.state is ReviewState.APPROVED
    assert approved.answered_at == PLUS_5
    assert approved.answered_by == "alice"
    assert approved.answer_reason == "looks good"
    assert stored(machine, "r1").state is ReviewState.APPROVED
    assert stored(machine, "r1").answered_by == "alice"


def test_reject_stamps_answered_fields(machine: ReviewStateMachine) -> None:
    review = machine.create("r1", "daily", now=NOW)

    rejected = machine.reject(review, now=PLUS_5, actor="bob", reason="no")
    assert rejected.state is ReviewState.REJECTED
    assert rejected.answered_at == PLUS_5
    assert rejected.answered_by == "bob"
    assert rejected.answer_reason == "no"


def test_regenerate_increments_count(machine: ReviewStateMachine) -> None:
    review = machine.create("r1", "daily", now=NOW)

    regenerated = machine.regenerate(review, now=PLUS_5, actor="bob", reason="again")
    assert regenerated.state is ReviewState.REGENERATED
    assert regenerated.regenerated_count == 1

    resubmitted = machine.resubmit(regenerated, now=PLUS_10)
    assert resubmitted.state is ReviewState.PENDING
    assert resubmitted.regenerated_count == 1

    regenerated_again = machine.regenerate(resubmitted, now=PLUS_10)
    assert regenerated_again.regenerated_count == 2


def test_regeneration_respects_profile_settings(machine: ReviewStateMachine) -> None:
    review = machine.create("r1", "noregen", now=NOW)
    with pytest.raises(IllegalReviewTransition):
        machine.regenerate(review, now=PLUS_5)

    review = machine.create("r2", "oneregen", now=NOW)
    machine.regenerate(review, now=PLUS_5)
    resubmitted = machine.resubmit(review, now=PLUS_10)
    with pytest.raises(IllegalReviewTransition):
        machine.regenerate(resubmitted, now=PLUS_10)


def test_auto_approved_profile_skips_review(machine: ReviewStateMachine) -> None:
    approved = machine.create("r1", "auto", now=NOW)
    assert approved.state is ReviewState.APPROVED
    assert approved.answered_at == NOW
    assert stored(machine, "r1").state is ReviewState.APPROVED

    with pytest.raises(IllegalReviewTransition):
        machine.reject("r1", now=PLUS_5)


def test_illegal_machine_transition_persists_nothing(
    machine: ReviewStateMachine,
) -> None:
    review = machine.create("r1", "daily", now=NOW)
    machine.approve(review, now=PLUS_5)
    before = stored(machine, "r1")

    with pytest.raises(IllegalReviewTransition, match="approved to rejected"):
        machine.reject(review, now=PLUS_10)

    after = stored(machine, "r1")
    assert after == before
    assert after.state is ReviewState.APPROVED
    assert after.answered_at == NOW + timedelta(minutes=5)


def test_machine_reads_latest_state_not_a_stale_record(
    machine: ReviewStateMachine,
) -> None:
    stale = machine.create("r1", "daily", now=NOW)
    assert stale.state is ReviewState.PENDING
    machine.regenerate(stale, now=PLUS_5)

    assert stale.state is ReviewState.PENDING
    resubmitted = machine.resubmit(stale, now=PLUS_10)

    assert resubmitted.state is ReviewState.PENDING
    assert stored(machine, "r1").state is ReviewState.PENDING
    assert stored(machine, "r1").regenerated_count == 1


def test_machine_reports_a_missing_review(machine: ReviewStateMachine) -> None:
    assert machine.get("review_missing") is None
    with pytest.raises(ReviewNotFoundError, match="review not found"):
        machine.approve("review_missing")
    with pytest.raises(ReviewNotFoundError, match="review not found"):
        machine.transition(
            Review(
                id="review_missing",
                profile="daily",
                state=ReviewState.PENDING,
                created_at=NOW,
            ),
            ReviewState.APPROVED,
        )


def test_machine_helpers_report_legal_transitions(machine: ReviewStateMachine) -> None:
    review = machine.create("r", "daily", now=NOW)
    assert machine.legal_transitions(review) == legal_transitions(ReviewState.PENDING)
    assert machine.can_transition(review, ReviewState.APPROVED) is True
    assert machine.can_transition(review, ReviewState.TIMED_OUT) is True
    approved = machine.approve(review, now=PLUS_5)
    assert machine.can_transition(approved, ReviewState.REJECTED) is False
    assert machine.transitions is LEGAL_REVIEW_TRANSITIONS


def test_per_profile_approval_settings(machine: ReviewStateMachine) -> None:
    assert machine.approval("daily") == "ask"
    assert machine.approval("auto") == "never"
    assert machine.profile("daily").timeout_seconds == 60
    assert machine.timeout("daily").seconds == 60
    assert machine.timeout("daily").action == "reject"
    assert machine.timeout("notify").action == "notify"
    assert machine.timeout("approve").action == "approve"
    assert machine.allow_regenerate("noregen") is False
    assert machine.allow_regenerate("daily") is True
    assert machine.max_regenerations("oneregen") == 1
    assert machine.max_regenerations("daily") == 3


def test_unknown_profile_raises(machine: ReviewStateMachine) -> None:
    with pytest.raises(ValueError, match="unknown profile"):
        machine.create("r1", "missing", now=NOW)
    with pytest.raises(ValueError, match="unknown profile"):
        machine.approval("missing")


def test_timeout_before_deadline_raises(machine: ReviewStateMachine) -> None:
    review = machine.create("r1", "daily", now=NOW)
    with pytest.raises(IllegalReviewTransition):
        machine.timed_out(review, now=PLUS_30_SECONDS)
    assert stored(machine, "r1").state is ReviewState.PENDING


def test_timeout_at_deadline_is_rejected_by_default(
    machine: ReviewStateMachine,
) -> None:
    review = machine.create("r1", "daily", now=NOW)

    timed_out = machine.timed_out(review, now=PLUS_60_SECONDS)
    assert timed_out.state is ReviewState.REJECTED
    assert timed_out.timeout_action == "reject"
    assert timed_out.timeout_at == PLUS_60_SECONDS
    assert stored(machine, "r1").state is ReviewState.REJECTED


def test_timeout_applies_per_profile_seconds(machine: ReviewStateMachine) -> None:
    short = ReviewStateMachine(
        settings={"fast": review_settings(timeout_seconds=5)},
        clock=machine.clock,
    )
    review = short.create("r1", "fast", now=NOW)
    with pytest.raises(IllegalReviewTransition):
        short.timed_out(review, now=NOW + timedelta(seconds=4))
    timed_out = short.timed_out(review, now=NOW + timedelta(seconds=5))
    assert timed_out.state is ReviewState.REJECTED


def test_timeout_on_timeout_approve(machine: ReviewStateMachine) -> None:
    other = ReviewStateMachine(
        settings={"daily": review_settings(on_timeout="approve")},
        clock=machine.clock,
    )
    review = other.create("r1", "daily", now=NOW)
    timed_out = other.timed_out(review, now=PLUS_60_SECONDS)
    assert timed_out.state is ReviewState.APPROVED
    assert timed_out.timeout_action == "approve"


def test_timeout_on_timeout_notify(machine: ReviewStateMachine) -> None:
    other = ReviewStateMachine(
        settings={"daily": review_settings(on_timeout="notify")},
        clock=machine.clock,
    )
    review = other.create("r1", "daily", now=NOW)
    timed_out = other.timed_out(review, now=PLUS_60_SECONDS)
    assert timed_out.state is ReviewState.TIMED_OUT
    assert timed_out.timeout_action == "notify"


def test_timeout_only_applies_to_pending_reviews(machine: ReviewStateMachine) -> None:
    approved = machine.create("r1", "auto", now=NOW)
    with pytest.raises(IllegalReviewTransition):
        machine.timed_out(approved, now=PLUS_60_SECONDS)

    rejected = machine.create("r2", "daily", now=NOW)
    machine.reject(rejected, now=PLUS_5)
    with pytest.raises(IllegalReviewTransition):
        machine.timed_out(rejected, now=PLUS_60_SECONDS)


def test_transition_review_is_pure_and_does_not_mutate() -> None:
    original = Review(
        id="r", profile="daily", state=ReviewState.PENDING, created_at=NOW
    )
    updated = transition_review(
        original,
        ReviewState.APPROVED,
        now=PLUS_5,
        actor="alice",
        reason="ok",
    )
    assert original.state is ReviewState.PENDING
    assert original.answered_at is None
    assert updated.state is ReviewState.APPROVED
    assert updated.answered_at == PLUS_5
    assert updated.answered_by == "alice"
    assert updated.answer_reason == "ok"

"""The legal lifecycle of a review, driven by per-profile approval settings.

A review starts ``pending``. A reviewer may approve, reject or request a
regeneration. A regenerated review may be re-submitted for review, up to the
profile's ``max_regenerations``. If no answer arrives before the profile's
``timeout_seconds``, the profile's ``on_timeout`` decides whether the review is
rejected, approved or only escalated (``notify``). Terminal states never leave
themselves.

Every profile carries its own review settings (approval mode, timeout and
on-timeout action) so different audiences or brands can review differently.
The machine is constructed with a mapping of profile name to
:class:`~aivia.core.config.ops.ReviewConfig` and a clock; tests supply a fake
clock so they never depend on wall time.

The mapping in :data:`LEGAL_REVIEW_TRANSITIONS` is the single source of truth.
:func:`transition_review` applies one transition to a record without touching
any store, while :class:`ReviewStateMachine` keeps the reviews in memory,
validates each transition and records every accepted change. An illegal
transition raises :class:`IllegalReviewTransition` and persists nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from aivia.core.config.ops import ReviewConfig
from aivia.core.errors import AiviaError
from aivia.core.models import utcnow

__all__ = [
    "LEGAL_REVIEW_TRANSITIONS",
    "REVIEW_STATE_TRANSITIONS",
    "IllegalReviewTransition",
    "InvalidReviewTransition",
    "InvalidReviewTransitionError",
    "Review",
    "ReviewNotFoundError",
    "ReviewState",
    "ReviewStateError",
    "ReviewStateMachine",
    "ReviewStateStore",
    "ReviewTimeout",
    "can_transition",
    "is_legal_transition",
    "legal_transitions",
    "transition_review",
]

LEGAL_REVIEW_TRANSITIONS: Mapping[ReviewState, frozenset[ReviewState]] = (
    MappingProxyType({})  # placeholder, replaced below after ReviewState exists
)
"""Every review state mapped to the states it may move to."""


class ReviewStateError(AiviaError):
    """Base class for review lifecycle errors."""

    code = "review_state"


class ReviewNotFoundError(ReviewStateError):
    """The requested review does not exist in the machine."""

    code = "review_not_found"


class IllegalReviewTransition(ReviewStateError, ValueError):
    """A transition the state machine does not allow.

    The error is also a :class:`ValueError` so callers that validate a
    transition before applying it can use the usual exception handling.
    """

    code = "illegal_review_transition"

    def __init__(self, current: ReviewState, target: ReviewState) -> None:
        super().__init__(
            f"cannot transition a review from {current.value} to {target.value}",
            details={
                "current": current.value,
                "target": target.value,
                "allowed": sorted(
                    state.value for state in LEGAL_REVIEW_TRANSITIONS[current]
                ),
            },
        )
        self.current = current
        self.target = target


InvalidReviewTransition = IllegalReviewTransition
"""Alias for :class:`IllegalReviewTransition`."""

InvalidReviewTransitionError = IllegalReviewTransition
"""Alias for :class:`IllegalReviewTransition`."""


class ReviewState(StrEnum):
    """The lifecycle state of a review.

    ``pending`` is the initial state. ``approved``, ``rejected`` and
    ``timed_out`` are terminal. ``regenerated`` may be re-submitted as
    ``pending`` up to the profile's regeneration limit.
    """

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REGENERATED = "regenerated"
    TIMED_OUT = "timed_out"


REVIEW_STATE_ALIASES = {
    "approved": ReviewState.APPROVED,
    "ok": ReviewState.APPROVED,
    "yes": ReviewState.APPROVED,
    "rejected": ReviewState.REJECTED,
    "no": ReviewState.REJECTED,
    "regenerated": ReviewState.REGENERATED,
    "regenerate": ReviewState.REGENERATED,
    "redo": ReviewState.REGENERATED,
    "timed_out": ReviewState.TIMED_OUT,
    "timeout": ReviewState.TIMED_OUT,
    "expired": ReviewState.TIMED_OUT,
}


def _review_state(value: ReviewState | str) -> ReviewState:
    if isinstance(value, ReviewState):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in REVIEW_STATE_ALIASES:
            return REVIEW_STATE_ALIASES[text]
        return ReviewState(text)
    return value


LEGAL_REVIEW_TRANSITIONS = MappingProxyType(
    {
        ReviewState.PENDING: frozenset(
            {
                ReviewState.APPROVED,
                ReviewState.REJECTED,
                ReviewState.REGENERATED,
                ReviewState.TIMED_OUT,
            }
        ),
        ReviewState.REGENERATED: frozenset({ReviewState.PENDING}),
        ReviewState.APPROVED: frozenset(),
        ReviewState.REJECTED: frozenset(),
        ReviewState.TIMED_OUT: frozenset(),
    }
)
"""Every review state mapped to the states it may move to."""

_REVIEW_STATE_TRANSITIONS = LEGAL_REVIEW_TRANSITIONS
"""Alias for :data:`LEGAL_REVIEW_TRANSITIONS`."""

REVIEW_STATE_TRANSITIONS = LEGAL_REVIEW_TRANSITIONS
"""Alias for :data:`LEGAL_REVIEW_TRANSITIONS`."""


class ReviewTimeout:
    """The timeout configuration resolved for a profile.

    ``seconds`` is the review timeout in seconds. ``action`` is the
    :attr:`~aivia.core.config.ops.ReviewConfig.on_timeout` decision applied
    when the deadline passes: ``reject``, ``approve`` or ``notify``.
    """

    __slots__ = ("action", "seconds")

    def __init__(self, seconds: int, action: str) -> None:
        self.seconds = seconds
        self.action = action

    def __repr__(self) -> str:
        return f"ReviewTimeout(seconds={self.seconds}, action={self.action!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ReviewTimeout):
            return NotImplemented
        return self.seconds == other.seconds and self.action == other.action

    def __hash__(self) -> int:
        return hash((self.seconds, self.action))


@dataclass(frozen=True, slots=True)
class Review:
    """One review record: who reviewed which profile and with what outcome."""

    id: str
    profile: str
    state: ReviewState
    created_at: datetime
    updated_at: datetime | None = None
    answered_at: datetime | None = None
    answered_by: str = ""
    answer_reason: str = ""
    timeout_at: datetime | None = None
    timeout_action: str = ""
    regenerated_count: int = 0


_PERSISTED_FIELDS = (
    "state",
    "updated_at",
    "answered_at",
    "answered_by",
    "answer_reason",
    "timeout_at",
    "timeout_action",
    "regenerated_count",
)

_TERMINAL_STATES = frozenset(
    {ReviewState.APPROVED, ReviewState.REJECTED, ReviewState.TIMED_OUT}
)


def _moment(value: datetime | None) -> datetime:
    return value if value is not None else utcnow()


def legal_transitions(state: ReviewState | str) -> frozenset[ReviewState]:
    """Return the states a review in *state* may move to."""
    return LEGAL_REVIEW_TRANSITIONS[_review_state(state)]


def can_transition(source: ReviewState | str, target: ReviewState | str) -> bool:
    """Whether a review may move from *source* to *target*."""
    return _review_state(target) in LEGAL_REVIEW_TRANSITIONS[_review_state(source)]


is_legal_transition = can_transition
"""Alias for :func:`can_transition`."""


def transition_review(
    review: Review,
    target: ReviewState | str,
    *,
    now: datetime | None = None,
    actor: str = "",
    reason: str = "",
    timeout_action: str = "",
) -> Review:
    """Return *review* moved to *target*, or raise when the move is illegal.

    The original record is never mutated. ``updated_at`` is stamped on every
    transition. Answering stamps ``answered_at`` plus the actor and reason;
    timing out stamps ``timeout_at`` and the on-timeout action; regenerating
    increments ``regenerated_count``.
    """
    current = review.state
    destination = _review_state(target)
    if destination not in LEGAL_REVIEW_TRANSITIONS[current]:
        raise IllegalReviewTransition(current, destination)

    moment = _moment(now)
    changes: dict[str, Any] = {"state": destination, "updated_at": moment}

    if destination in {ReviewState.APPROVED, ReviewState.REJECTED}:
        changes["answered_at"] = moment
        changes["answered_by"] = actor.strip()
        changes["answer_reason"] = reason.strip()
    if timeout_action or destination is ReviewState.TIMED_OUT:
        changes["timeout_at"] = moment
        changes["timeout_action"] = timeout_action.strip()
    elif destination is ReviewState.REGENERATED:
        changes["regenerated_count"] = review.regenerated_count + 1

    return replace(review, **changes)


class ReviewStateMachine:
    """Move reviews through legal states, driven by per-profile settings.

    *settings* maps a profile name to its
    :class:`~aivia.core.config.ops.ReviewConfig`, so every profile can carry
    its own approval mode, timeout and on-timeout action. *clock* supplies the
    timestamp for each transition and defaults to
    :func:`~aivia.core.models.base.utcnow`. Reviews are kept in memory and are
    never mutated; every accepted transition returns a new record.
    """

    def __init__(
        self,
        *,
        settings: Mapping[str, ReviewConfig] | None = None,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._settings_map: dict[str, ReviewConfig] = dict(settings or {})
        self._clock = clock
        self._reviews: dict[str, Review] = {}

    @property
    def transitions(self) -> Mapping[ReviewState, frozenset[ReviewState]]:
        """The legal transition mapping used by this machine."""
        return LEGAL_REVIEW_TRANSITIONS

    def _settings(self, profile: str) -> ReviewConfig:
        """Return the review settings configured for *profile*."""
        try:
            return self._settings_map[profile]
        except KeyError:
            available = ", ".join(sorted(self._settings_map)) or "none"
            raise ValueError(
                f"unknown profile: {profile} (available: {available})"
            ) from None

    def profile(self, profile: str) -> ReviewConfig:
        """Return the review settings configured for *profile*."""
        return self._settings(profile)

    def approval(self, profile: str) -> str:
        """Return the approval mode configured for *profile*."""
        return self._settings(profile).approval_mode

    def timeout(self, profile: str) -> ReviewTimeout:
        """Return the timeout configuration resolved for *profile*."""
        settings = self._settings(profile)
        return ReviewTimeout(
            seconds=settings.timeout_seconds, action=settings.on_timeout
        )

    def allow_regenerate(self, profile: str) -> bool:
        """Whether regeneration is allowed for *profile*."""
        return self._settings(profile).allow_regenerate

    def max_regenerations(self, profile: str) -> int:
        """Return the regeneration limit configured for *profile*."""
        return self._settings(profile).max_regenerations

    def legal_transitions(
        self, review: Review | ReviewState | str
    ) -> frozenset[ReviewState]:
        """Return the states the given review or state may move to."""
        state = review.state if isinstance(review, Review) else _review_state(review)
        return LEGAL_REVIEW_TRANSITIONS[state]

    def can_transition(
        self, review: Review | ReviewState | str, target: ReviewState | str
    ) -> bool:
        """Whether *review* may move to *target*."""
        state = review.state if isinstance(review, Review) else _review_state(review)
        return _review_state(target) in LEGAL_REVIEW_TRANSITIONS[state]

    def create(
        self,
        review_id: str,
        profile: str,
        *,
        now: datetime | None = None,
    ) -> Review:
        """Create a pending review for *profile* and return the stored record.

        A profile whose approval mode is ``never`` is auto-approved: it skips
        human review and reaches the ``approved`` terminal state immediately.
        """
        settings = self._settings(profile)
        moment = _moment(now)
        if settings.approval_mode == "never":
            return self._store(
                Review(
                    id=review_id,
                    profile=profile,
                    state=ReviewState.APPROVED,
                    created_at=moment,
                    updated_at=moment,
                    answered_at=moment,
                )
            )
        return self._store(
            Review(
                id=review_id,
                profile=profile,
                state=ReviewState.PENDING,
                created_at=moment,
                updated_at=moment,
            )
        )

    def get(self, review_id: str) -> Review | None:
        """Return the stored review with *review_id*, if any."""
        return self._reviews.get(review_id)

    def transition(
        self,
        review: Review | str,
        target: ReviewState | str,
        *,
        now: datetime | None = None,
        actor: str = "",
        reason: str = "",
        timeout_action: str = "",
    ) -> Review:
        """Persist one legal transition and return the updated record."""
        current = self._current(review)
        updated = transition_review(
            current,
            target,
            now=now if now is not None else self.clock(),
            actor=actor,
            reason=reason,
            timeout_action=timeout_action,
        )
        self._reviews[updated.id] = updated
        return updated

    def _current(self, review: Review | str) -> Review:
        review_id = review.id if isinstance(review, Review) else review
        current = self._reviews.get(review_id)
        if current is None:
            raise ReviewNotFoundError(
                f"review not found: {review_id}",
                details={"review_id": review_id},
            )
        return current

    @property
    def clock(self) -> Callable[[], datetime]:
        """The clock used to stamp transitions."""
        return self._clock

    def approve(
        self,
        review: Review | str,
        *,
        now: datetime | None = None,
        actor: str = "",
        reason: str = "",
    ) -> Review:
        """Move a pending review to ``approved``, recording who approved it."""
        return self.transition(
            review,
            ReviewState.APPROVED,
            now=now,
            actor=actor,
            reason=reason,
        )

    def reject(
        self,
        review: Review | str,
        *,
        now: datetime | None = None,
        actor: str = "",
        reason: str = "",
    ) -> Review:
        """Move a pending review to ``rejected``, recording who rejected it."""
        return self.transition(
            review,
            ReviewState.REJECTED,
            now=now,
            actor=actor,
            reason=reason,
        )

    def regenerate(
        self,
        review: Review | str,
        *,
        now: datetime | None = None,
        actor: str = "",
        reason: str = "",
    ) -> Review:
        """Request a regeneration from ``pending`` to ``regenerated``.

        Regeneration is only allowed when the profile permits it and the
        review has not already been regenerated ``max_regenerations`` times.
        A regenerated review may be re-submitted to ``pending``.
        """
        current = self._current(review)
        settings = self._settings(current.profile)
        if not settings.allow_regenerate:
            raise IllegalReviewTransition(current.state, ReviewState.REGENERATED)
        if (
            current.state is ReviewState.PENDING
            and current.regenerated_count >= settings.max_regenerations
        ):
            raise IllegalReviewTransition(current.state, ReviewState.REGENERATED)
        return self.transition(
            current,
            ReviewState.REGENERATED,
            now=now,
            actor=actor,
            reason=reason,
        )

    def resubmit(
        self,
        review: Review | str,
        *,
        now: datetime | None = None,
    ) -> Review:
        """Re-submit a regenerated review to ``pending`` for another review."""
        current = self._current(review)
        if current.state is not ReviewState.REGENERATED:
            raise IllegalReviewTransition(current.state, ReviewState.PENDING)
        return self.transition(current, ReviewState.PENDING, now=now)

    def timed_out(
        self,
        review: Review | str,
        *,
        now: datetime | None = None,
    ) -> Review:
        """Apply the profile's on-timeout behaviour to a pending review.

        A review that has not passed its deadline raises
        :class:`IllegalReviewTransition`. Once past the deadline the profile's
        ``on_timeout`` decides the outcome: ``reject`` moves to ``rejected``,
        ``approve`` moves to ``approved`` and ``notify`` moves to
        ``timed_out`` (escalated without auto-resolving).
        """
        current = self._current(review)
        if current.state is not ReviewState.PENDING:
            raise IllegalReviewTransition(current.state, ReviewState.TIMED_OUT)
        settings = self._settings(current.profile)
        moment = _moment(now)
        deadline = current.created_at + timedelta(seconds=settings.timeout_seconds)
        if moment < deadline:
            raise IllegalReviewTransition(current.state, ReviewState.TIMED_OUT)
        if settings.on_timeout == "reject":
            target = ReviewState.REJECTED
            reason = "review timed out: rejected"
            timeout_action = "reject"
        elif settings.on_timeout == "approve":
            target = ReviewState.APPROVED
            reason = "review timed out: approved"
            timeout_action = "approve"
        else:
            target = ReviewState.TIMED_OUT
            reason = "review timed out: notified"
            timeout_action = "notify"
        return self.transition(
            current,
            target,
            now=moment,
            actor="system",
            reason=reason,
            timeout_action=timeout_action,
        )

    def _store(self, review: Review) -> Review:
        self._reviews[review.id] = review
        return review


ReviewStateStore = ReviewStateMachine
"""Alias for :class:`ReviewStateMachine`."""

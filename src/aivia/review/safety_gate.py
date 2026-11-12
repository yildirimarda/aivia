"""Content safety gates: block-list screening with mandatory-review escalation.

A gate binds a :class:`~aivia.review.review_state.ReviewStateMachine` to the
block-list screening from :mod:`aivia.media.screening`. Clean content passes
through untouched. Blocked content is either escalated to a mandatory review
(a new ``pending`` review a human must answer) or refused outright into the
distinct ``refused`` terminal state, depending on whether escalation is enabled.

The module is deterministic and offline: it never calls a provider, a model or
the network.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aivia.media.screening import (
    DEFAULT_CONTENT_KIND,
    DEFAULT_PROMPT_KIND,
    ScreeningResult,
    check_content,
    check_prompt,
)
from aivia.review.review_state import (
    Review,
    ReviewState,
    ReviewStateMachine,
)

__all__ = ["SafetyGate", "SafetyGateResult"]

_PROMPT_KINDS = frozenset(
    {
        "prompt",
        "text",
        "script",
        "caption",
        "title",
        "description",
        "lyrics",
        "query",
    }
)


def _is_prompt_kind(kind: str) -> bool:
    text = str(kind or DEFAULT_PROMPT_KIND).strip().lower()
    return text in _PROMPT_KINDS or text.startswith("prompt")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _escalate_enabled(config: Any) -> bool:
    """Whether blocked content is escalated to a mandatory review.

    Defaults to the safety config's ``allow_regenerate`` when a configuration
    is supplied, and to ``True`` otherwise. When disabled, blocked content is
    refused outright into the distinct ``refused`` state.
    """
    if config is None:
        return True
    allow_regenerate: Any = True
    if isinstance(config, Mapping):
        safety = config.get("safety")
        source: Mapping[str, Any] = safety if isinstance(safety, Mapping) else config
        allow_regenerate = source.get("allow_regenerate", True)
    else:
        allow_regenerate = getattr(config, "allow_regenerate", True)
    return bool(allow_regenerate)


@dataclass(frozen=True, slots=True)
class SafetyGateResult:
    """Outcome of screening content through a safety gate."""

    allowed: bool
    """Whether the content passed the block-list screen."""
    blocked: bool
    """Whether the content was blocked by a configured term."""
    blocked_terms: tuple[str, ...]
    """The configured terms that matched, if any."""
    review: Review | None
    """The review created for the content, if any."""
    state: ReviewState | None
    """The review state after the gate ran, if a review exists."""
    reason: str
    """A human readable reason for the outcome."""


class SafetyGate:
    """Screen content against configured block lists and escalate blocked items.

    *reviews* is the review state machine used to create a mandatory review for
    blocked content. *config* is the safety configuration (or a mapping carrying
    a ``safety`` section) whose ``blocked_terms`` and ``block_categories`` drive
    the block list. *profile* names the review profile a mandatory review is
    created under.
    """

    def __init__(
        self,
        *,
        reviews: ReviewStateMachine,
        config: Any = None,
        profile: str = "default",
    ) -> None:
        self._reviews = reviews
        self._config = config
        self._profile = profile

    @property
    def reviews(self) -> ReviewStateMachine:
        """The review state machine the gate creates mandatory reviews through."""
        return self._reviews

    @property
    def config(self) -> Any:
        """The safety configuration driving the block list."""
        return self._config

    @property
    def profile(self) -> str:
        """The review profile mandatory reviews are created under."""
        return self._profile

    def _screen(
        self,
        content: str,
        kind: str,
        terms: Sequence[str] | str | None,
        categories: Sequence[str] | str | None,
    ) -> ScreeningResult:
        prompt_kind = _is_prompt_kind(kind)
        if prompt_kind:
            return check_prompt(
                content, self._config, terms=terms, categories=categories
            )
        return check_content(
            content, self._config, kind=kind, terms=terms, categories=categories
        )

    def screen(
        self,
        text: str,
        *,
        kind: str = DEFAULT_CONTENT_KIND,
        terms: Sequence[str] | str | None = None,
        categories: Sequence[str] | str | None = None,
        review_id: str | None = None,
        escalate: bool | None = None,
    ) -> SafetyGateResult:
        """Screen *text* against the configured block lists.

        Clean content returns with ``allowed=True`` and no review. Blocked
        content creates a mandatory review (``pending``) when escalation is
        enabled, or is refused into the distinct ``refused`` terminal state
        when it is not.
        """
        content = str(text)
        result = self._screen(content, kind, terms, categories)

        if result.allowed:
            return SafetyGateResult(
                allowed=True,
                blocked=False,
                blocked_terms=(),
                review=None,
                state=None,
                reason="",
            )

        blocked_terms = tuple(result.blocked_terms)
        review_id = review_id or f"review-{_digest(content)[:12]}"
        review = self._reviews.create(review_id, self._profile)

        want_escalate = (
            _escalate_enabled(self._config) if escalate is None else escalate
        )
        if want_escalate:
            return SafetyGateResult(
                allowed=False,
                blocked=True,
                blocked_terms=blocked_terms,
                review=review,
                state=review.state,
                reason="blocked by screening: escalated to mandatory review",
            )

        if review.state is ReviewState.PENDING:
            review = self._reviews.refuse(
                review,
                actor="system",
                reason="blocked by screening: " + ", ".join(blocked_terms),
            )
        return SafetyGateResult(
            allowed=False,
            blocked=True,
            blocked_terms=blocked_terms,
            review=review,
            state=review.state,
            reason="blocked by screening: refused",
        )

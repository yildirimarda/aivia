"""The weakest-link stage: one dead rubric dimension blocks a topic.

The scoring stage ranks candidates by their weighted mean, which lets a topic
with one very low dimension hide behind three strong ones. This stage adds the
weakest-link rule on top of that mean:

* every score already records its weakest rubric dimension and that dimension's
  score, so a stored score can be explained after the fact;
* a candidate is selected only when the weighted mean clears ``min_score``
  **and** the weakest dimension clears ``min_dimension_score``;
* a candidate whose mean passes but whose weakest dimension fails is reported
  as blocked, with the blocking dimension named, instead of disappearing.

The stage reuses :class:`~aivia.research.scoring.ScoringStage`, so it is
deterministic and offline and accepts the same rubric, config, explicit scores
and injected scorer. It never calls a provider itself.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aivia.core.errors import ConfigError
from aivia.core.models.topic import TopicCandidate
from aivia.research.scoring import (
    DEFAULT_MIN_DIMENSION_SCORE,
    CandidateScore,
    ScoringResult,
    ScoringStage,
    score_candidates,
    score_candidates_async,
)

__all__ = [
    "DEFAULT_MIN_DIMENSION_SCORE",
    "WeakestLinkResult",
    "WeakestLinkScore",
    "WeakestLinkScorer",
    "WeakestLinkStage",
    "as_weakest_link_result",
    "build_weakest_link_stage",
    "filter_weakest_link",
    "passes_weakest_link",
    "score_candidates_weakest_link",
    "score_topics_weakest_link",
    "score_weakest_link",
    "score_weakest_link_async",
    "weakest_dimension",
    "weakest_link",
    "weakest_link_stage",
    "weakest_score",
]

WeakestLinkScore = CandidateScore
"""A candidate score that also exposes its weakest dimension."""


def _validate_floor(value: Any) -> float:
    if value is None:
        return DEFAULT_MIN_DIMENSION_SCORE
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError("minimum dimension score must be a number")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ConfigError("minimum dimension score must be between zero and one")
    return number


@dataclass(frozen=True, slots=True)
class WeakestLinkResult(ScoringResult):
    """Every score plus only the candidates that pass the weakest-link rule."""

    @property
    def passing(self) -> list[CandidateScore]:
        """The scores that pass both the mean and the weakest dimension."""
        return [score for score in self.scores if score.selected]

    @property
    def candidates(self) -> list[TopicCandidate]:
        """The candidates selected by the mean and weakest-dimension rules."""
        return [score.candidate for score in self.passing]

    @property
    def failed(self) -> list[CandidateScore]:
        """The scores dropped by either the mean or the weakest dimension."""
        return [score for score in self.scores if not score.selected]

    @property
    def blocked(self) -> list[CandidateScore]:
        """The scores whose mean passes but whose weakest dimension fails."""
        return [score for score in self.scores if score.blocked_by_weakest_link]

    @property
    def blocked_by_weakest_link(self) -> list[CandidateScore]:
        """Alias for :attr:`blocked`."""
        return self.blocked

    @property
    def blocked_candidates(self) -> list[TopicCandidate]:
        """The candidates blocked by a single dead dimension."""
        return [score.candidate for score in self.blocked]

    @property
    def blocked_titles(self) -> list[str]:
        """The titles of the candidates blocked by a dead dimension."""
        return [score.title for score in self.blocked]

    @property
    def weakest_links(self) -> list[tuple[str, float]]:
        """The weakest dimension name and score for every scored candidate."""
        links: list[tuple[str, float]] = []
        for score in self.scores:
            link = score.weakest_link
            if link is not None:
                links.append(link)
        return links

    def filter(
        self,
        min_score: float | None = None,
        *,
        threshold: float | None = None,
        min_dimension_score: float | None = None,
    ) -> list[TopicCandidate]:
        """Return candidates above both the mean and weakest-dimension limits."""
        limit = self.min_score if min_score is None else min_score
        if threshold is not None:
            limit = threshold
        floor = (
            self.min_dimension_score
            if min_dimension_score is None
            else _validate_floor(min_dimension_score)
        )
        selected: list[TopicCandidate] = []
        for score in self.scores:
            if limit is not None and score.overall < limit:
                continue
            weakest = score.weakest_score
            if weakest is not None and weakest < floor:
                continue
            selected.append(score.candidate)
        return selected


def as_weakest_link_result(result: ScoringResult) -> WeakestLinkResult:
    """Wrap a scoring result so its candidates also honour the weakest link."""
    if isinstance(result, WeakestLinkResult):
        return result
    selected = [score for score in result.scores if score.selected]
    blocked = [score for score in result.scores if score.blocked_by_weakest_link]
    metadata = dict(result.metadata)
    metadata["mean_kept"] = metadata.get(
        "kept",
        sum(1 for score in result.scores if score.passed),
    )
    metadata["kept"] = len(selected)
    metadata["dropped"] = len(result.scores) - len(selected)
    metadata["min_dimension_score"] = result.rubric.min_dimension_score
    metadata["weakest_link_blocked"] = len(blocked)
    metadata["weakest_link_blocked_titles"] = [score.title for score in blocked]
    return WeakestLinkResult(
        scores=result.scores,
        rubric=result.rubric,
        metadata=metadata,
    )


def score_weakest_link(
    candidates: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
    **options: Any,
) -> WeakestLinkResult:
    """Score *candidates* and keep only those passing the weakest-link rule."""
    return as_weakest_link_result(
        score_candidates(candidates, metadata=metadata, **options)
    )


async def score_weakest_link_async(
    candidates: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
    **options: Any,
) -> WeakestLinkResult:
    """Async variant of :func:`score_weakest_link`."""
    return as_weakest_link_result(
        await score_candidates_async(candidates, metadata=metadata, **options)
    )


weakest_link = score_weakest_link
"""Alias for :func:`score_weakest_link`."""

score_weakest_link_candidates = score_weakest_link
"""Alias for :func:`score_weakest_link`."""

score_candidates_weakest_link = score_weakest_link
"""Alias for :func:`score_weakest_link`."""

score_topics_weakest_link = score_weakest_link
"""Alias for :func:`score_weakest_link`."""


def _dimension_mapping(value: Mapping[Any, Any]) -> dict[str, float]:
    dimensions: dict[str, float] = {}
    for name, score in value.items():
        if not isinstance(name, str) or not name.strip():
            continue
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            continue
        number = float(score)
        if math.isfinite(number):
            dimensions[name.strip()] = number
    return dimensions


def _weakest_from_dimensions(
    dimensions: Mapping[str, float],
    order: Sequence[str] | None,
) -> str | None:
    if not dimensions:
        return None
    ordered = [name for name in (order or ()) if name in dimensions]
    ordered.extend(name for name in dimensions if name not in ordered)
    return min(ordered, key=lambda name: (dimensions[name], ordered.index(name)))


def weakest_dimension(
    value: Any,
    *,
    order: Sequence[str] | None = None,
) -> str | None:
    """Return the weakest dimension name for a score or a dimension mapping."""
    if isinstance(value, CandidateScore):
        return value.weakest_dimension
    if isinstance(value, Mapping):
        return _weakest_from_dimensions(_dimension_mapping(value), order)
    raise ConfigError(f"cannot read dimensions from {value!r}")


def weakest_score(
    value: Any,
    *,
    order: Sequence[str] | None = None,
) -> float | None:
    """Return the weakest dimension score for a score or a dimension mapping."""
    if isinstance(value, CandidateScore):
        return value.weakest_score
    if isinstance(value, Mapping):
        dimensions = _dimension_mapping(value)
        name = _weakest_from_dimensions(dimensions, order)
        return None if name is None else dimensions[name]
    raise ConfigError(f"cannot read dimensions from {value!r}")


def passes_weakest_link(
    value: Any,
    min_dimension_score: float | None = None,
) -> bool:
    """Whether *value*'s weakest dimension clears the floor."""
    if min_dimension_score is None:
        floor = (
            value.min_dimension_score
            if isinstance(value, CandidateScore)
            else DEFAULT_MIN_DIMENSION_SCORE
        )
    else:
        floor = _validate_floor(min_dimension_score)
    score = weakest_score(value)
    return True if score is None else score >= floor


def filter_weakest_link(
    result: Any,
    *,
    min_score: float | None = None,
    threshold: float | None = None,
    min_dimension_score: float | None = None,
) -> list[TopicCandidate]:
    """Return candidates above both the mean and weakest-dimension limits."""
    if isinstance(result, WeakestLinkResult):
        return result.filter(
            min_score,
            threshold=threshold,
            min_dimension_score=min_dimension_score,
        )
    if isinstance(result, ScoringResult):
        return as_weakest_link_result(result).filter(
            min_score,
            threshold=threshold,
            min_dimension_score=min_dimension_score,
        )
    if isinstance(result, CandidateScore):
        scores = [result]
    elif isinstance(result, Iterable) and not isinstance(
        result,
        (str, bytes, bytearray, Mapping),
    ):
        scores = [
            item
            if isinstance(item, CandidateScore)
            else CandidateScore.model_validate(item)
            for item in result
        ]
    else:
        raise ConfigError("expected a score, scores or a scoring result")
    limit = min_score
    if threshold is not None:
        limit = threshold
    floor = (
        None if min_dimension_score is None else _validate_floor(min_dimension_score)
    )
    selected: list[TopicCandidate] = []
    for score in scores:
        if limit is not None and score.overall < limit:
            continue
        weakest = score.weakest_score
        if weakest is not None:
            if floor is None:
                if not score.meets_min_dimension:
                    continue
            elif weakest < floor:
                continue
        selected.append(score.candidate)
    return selected


class WeakestLinkStage(ScoringStage):
    """A scoring stage that also enforces the weakest-dimension floor.

    The rubric, config, explicit scores and scorer are resolved exactly like
    :class:`~aivia.research.scoring.ScoringStage`; only :meth:`run` and
    :meth:`run_async` return a :class:`WeakestLinkResult` whose candidates must
    clear both thresholds.
    """

    def run(
        self,
        candidates: Any,
        *,
        scores: Any = None,
        scorer: Any = None,
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> WeakestLinkResult:
        """Score *candidates* and apply the weakest-link rule."""
        result = super().run(
            candidates,
            scores=scores,
            scorer=scorer,
            reason=reason,
            metadata=metadata,
        )
        return as_weakest_link_result(result)

    score_candidates = run
    scoring = run
    __call__ = run

    async def run_async(
        self,
        candidates: Any,
        *,
        scores: Any = None,
        scorer: Any = None,
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> WeakestLinkResult:
        """Async variant that can await an async scorer."""
        result = await super().run_async(
            candidates,
            scores=scores,
            scorer=scorer,
            reason=reason,
            metadata=metadata,
        )
        return as_weakest_link_result(result)

    score_candidates_async = run_async


WeakestLinkScorer = WeakestLinkStage
"""Alias for :class:`WeakestLinkStage`."""


def weakest_link_stage(
    rubric: Any = None,
    **options: Any,
) -> WeakestLinkStage:
    """Build a :class:`WeakestLinkStage` from a rubric or config."""
    return WeakestLinkStage(rubric, **options)


build_weakest_link_stage = weakest_link_stage
"""Alias for :func:`weakest_link_stage`."""

"""Select the topics a run will use from scored candidates.

Selection is the last research decision before a run starts writing. It takes
the scored candidates from :mod:`aivia.research.scoring` (or plain candidates
that already carry a score), keeps the ones above the minimum score, orders
them by score and then applies two limits:

* a per-run limit, so a run never starts with more topics than it can handle;
* a per-category diversity limit, so one category cannot crowd out the rest.

The stage is deterministic and offline. Equal scores keep their input order,
and the same candidates, score and limits always produce the same selection.
It never calls a provider or a model itself.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from aivia.core.errors import ConfigError
from aivia.core.models.topic import TopicCandidate
from aivia.providers.protocols.source import SourceResponse
from aivia.research.scoring import (
    CandidateScore,
    ScoringResult,
    score_candidates,
    score_candidates_async,
)

__all__ = [
    "DEFAULT_CATEGORY",
    "DEFAULT_CATEGORY_LIMIT",
    "DEFAULT_DIVERSITY",
    "DEFAULT_DIVERSITY_LIMIT",
    "DEFAULT_LIMIT",
    "DEFAULT_MAX_PER_CATEGORY",
    "DEFAULT_MAX_TOPICS",
    "DEFAULT_MIN_SCORE",
    "DEFAULT_PER_CATEGORY",
    "DEFAULT_TOPICS_PER_RUN",
    "SelectResult",
    "SelectStage",
    "SelectedTopic",
    "SelectedTopics",
    "Selection",
    "SelectionDecision",
    "SelectionResult",
    "SelectionStage",
    "SelectionStatus",
    "Selector",
    "TopicSelection",
    "TopicSelector",
    "build_selection_stage",
    "category_counts",
    "category_of",
    "choose_topics",
    "diverse_select",
    "diversify",
    "filter_selected",
    "filter_selection",
    "normalise_category",
    "normalize_category",
    "passes_selection",
    "pick_topics",
    "select",
    "select_candidate",
    "select_candidates",
    "select_candidates_async",
    "select_diverse",
    "select_scored",
    "select_scores",
    "select_topic",
    "select_topics",
    "select_topics_async",
    "select_topics_sync",
    "selected_topics",
    "selection",
    "selection_stage",
]

_UNSET: Any = object()

DEFAULT_LIMIT: int | None = None
"""No per-run limit is applied unless the caller sets one."""

DEFAULT_MAX_PER_CATEGORY: int | None = 1
"""At most one topic from each category is selected by default."""

DEFAULT_DIVERSITY = DEFAULT_MAX_PER_CATEGORY
"""Alias for :data:`DEFAULT_MAX_PER_CATEGORY`."""

DEFAULT_CATEGORY_LIMIT = DEFAULT_MAX_PER_CATEGORY
"""Alias for :data:`DEFAULT_MAX_PER_CATEGORY`."""

DEFAULT_PER_CATEGORY = DEFAULT_MAX_PER_CATEGORY
"""Alias for :data:`DEFAULT_MAX_PER_CATEGORY`."""

DEFAULT_DIVERSITY_LIMIT = DEFAULT_MAX_PER_CATEGORY
"""Alias for :data:`DEFAULT_MAX_PER_CATEGORY`."""

DEFAULT_MAX_TOPICS = DEFAULT_LIMIT
"""Alias for :data:`DEFAULT_LIMIT`."""

DEFAULT_TOPICS_PER_RUN = DEFAULT_LIMIT
"""Alias for :data:`DEFAULT_LIMIT`."""

DEFAULT_MIN_SCORE: float | None = None
"""Plain candidates are not filtered unless the caller sets a minimum score."""

DEFAULT_CATEGORY = "uncategorized"
"""The category used for candidates that do not declare one."""

_LIMIT_ALIASES = (
    "max_topics",
    "topics_per_run",
    "per_run",
    "count",
    "size",
    "max_results",
    "max_count",
)
_CATEGORY_ALIASES = (
    "per_category",
    "category_limit",
    "max_category",
    "max_per_category_count",
    "max_per_category_size",
    "category_cap",
)


class SelectionStatus(StrEnum):
    """Why a candidate was or was not selected for the run."""

    SELECTED = "selected"
    BELOW_MIN_SCORE = "below_min_score"
    CATEGORY_LIMIT = "category_limit"
    RUN_LIMIT = "run_limit"


def _validate_limit(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError("selection limit must be an integer")
    if value < 0:
        raise ConfigError("selection limit must not be negative")
    return value


def _validate_category_limit(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError("max per category must be an integer")
    if value < 0:
        raise ConfigError("max per category must not be negative")
    return None if value == 0 else value


def _validate_min_score(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError("minimum score must be a number")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ConfigError("minimum score must be between zero and one")
    return number


def _resolve_option(
    primary: Any,
    name: str,
    aliases: Sequence[str],
    options: dict[str, Any],
) -> Any:
    values: list[tuple[str, Any]] = []
    if primary is not _UNSET:
        values.append((name, primary))
    for alias in aliases:
        if alias in options:
            values.append((alias, options.pop(alias)))
    if not values:
        return _UNSET
    first_name, first = values[0]
    for alias, value in values[1:]:
        if value != first:
            raise ConfigError(
                f"conflicting {name} values: {first_name}={first!r}, {alias}={value!r}"
            )
    return first


def normalise_category(value: Any) -> str:
    """Return the lower-case category key used for diversity counting."""
    if value is None:
        return DEFAULT_CATEGORY
    text = str(value).strip().lower()
    return text or DEFAULT_CATEGORY


normalize_category = normalise_category
"""Alias for :func:`normalise_category`."""


def category_of(candidate: Any) -> str:
    """Return *candidate*'s normalised category, or ``uncategorized``."""
    return normalise_category(getattr(candidate, "category", ""))


def _coerce_candidate(value: Any) -> TopicCandidate:
    if isinstance(value, TopicCandidate):
        return value
    if isinstance(value, str):
        return TopicCandidate(title=value)
    if isinstance(value, Mapping):
        candidate = value.get("candidate", value.get("topic"))
        if isinstance(candidate, TopicCandidate):
            return candidate
        return TopicCandidate.model_validate(dict(value))
    candidate = getattr(value, "candidate", None)
    if isinstance(candidate, TopicCandidate):
        return candidate
    raise ConfigError("candidate must be a topic, a candidate or a mapping")


def _coerce_score(value: Any) -> CandidateScore:
    if isinstance(value, CandidateScore):
        return value
    try:
        return CandidateScore.model_validate(value)
    except Exception as error:  # pragma: no cover - re-raised as ConfigError
        raise ConfigError(f"invalid candidate score: {value!r}") from error


@dataclass(frozen=True, slots=True)
class _Entry:
    candidate: TopicCandidate
    score: float
    score_min: float | None
    candidate_score: CandidateScore | None
    index: int


def _entry_from_score(score: CandidateScore, index: int) -> _Entry:
    return _Entry(
        candidate=score.candidate,
        score=score.overall,
        score_min=score.min_score,
        candidate_score=score,
        index=index,
    )


def _entry_from_candidate(candidate: Any, index: int) -> _Entry:
    topic = _coerce_candidate(candidate)
    return _Entry(
        candidate=topic,
        score=float(topic.score),
        score_min=None,
        candidate_score=None,
        index=index,
    )


def _entries_from_scores(value: Any) -> list[_Entry]:
    return [
        _entry_from_score(_coerce_score(item), index)
        for index, item in enumerate(value)
    ]


def _entries(value: Any) -> list[_Entry]:
    if value is None:
        return []
    if isinstance(value, SelectionResult):
        return [
            _Entry(
                candidate=decision.candidate,
                score=decision.score,
                score_min=decision.min_score,
                candidate_score=decision.candidate_score,
                index=index,
            )
            for index, decision in enumerate(value.decisions)
        ]
    if isinstance(value, ScoringResult):
        return _entries_from_scores(value.scores)
    if isinstance(value, CandidateScore):
        return [_entry_from_score(value, 0)]
    if isinstance(value, (TopicCandidate, str)):
        return [_entry_from_candidate(value, 0)]
    if isinstance(value, SourceResponse):
        return [
            _entry_from_candidate(candidate, index)
            for index, candidate in enumerate(value.candidates)
        ]
    if isinstance(value, Mapping):
        if any(
            key in value
            for key in (
                "dimensions",
                "dimension_scores",
                "candidate",
                "overall",
                "weighted_score",
            )
        ):
            return [_entry_from_score(_coerce_score(value), 0)]
        for key in ("scores", "scored"):
            if key in value:
                return _entries(value[key])
        for key in ("candidates", "topics"):
            if key in value:
                return _entries(value[key])
        return [_entry_from_candidate(value, 0)]
    scores = getattr(value, "scores", None)
    if scores is not None and not isinstance(scores, (str, bytes, bytearray)):
        return _entries_from_scores(scores)
    candidates = getattr(value, "candidates", None)
    if candidates is not None and not isinstance(candidates, (str, bytes, bytearray)):
        return _entries(candidates)
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        entries: list[_Entry] = []
        for item in value:
            entries.extend(_entries(item))
        return entries
    raise ConfigError("candidates must be a score, a candidate or an iterable")


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    """One candidate, its score and the reason it was selected or skipped."""

    candidate: TopicCandidate
    score: float = 0.0
    category: str = DEFAULT_CATEGORY
    selected: bool = False
    reason: str = ""
    status: SelectionStatus = SelectionStatus.SELECTED
    rank: int | None = None
    min_score: float | None = None
    candidate_score: CandidateScore | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def title(self) -> str:
        """The candidate title, for convenient reporting."""
        return self.candidate.title

    @property
    def overall(self) -> float:
        """Alias for :attr:`score`."""
        return self.score

    @property
    def category_key(self) -> str:
        """The normalised category used for diversity counting."""
        return normalise_category(self.category)

    @property
    def rejected(self) -> bool:
        """Whether the candidate was not selected."""
        return not self.selected

    @property
    def dropped(self) -> bool:
        """Alias for :attr:`rejected`."""
        return self.rejected

    @property
    def below_min_score(self) -> bool:
        """Whether the candidate failed the minimum score."""
        return self.status is SelectionStatus.BELOW_MIN_SCORE

    @property
    def blocked_by_category(self) -> bool:
        """Whether the per-category diversity limit skipped the candidate."""
        return self.status is SelectionStatus.CATEGORY_LIMIT

    @property
    def blocked_by_limit(self) -> bool:
        """Whether the per-run limit skipped the candidate."""
        return self.status is SelectionStatus.RUN_LIMIT

    @property
    def passed(self) -> bool:
        """Whether the candidate cleared the minimum score."""
        return not self.below_min_score

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the decision."""
        return {
            "title": self.title,
            "score": self.score,
            "category": self.category,
            "selected": self.selected,
            "rejected": self.rejected,
            "status": self.status.value,
            "reason": self.reason,
            "rank": self.rank,
            "min_score": self.min_score,
            "candidate": self.candidate.model_dump(mode="json"),
        }


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Every selection decision plus the topics chosen for the run."""

    decisions: tuple[SelectionDecision, ...] = ()
    limit: int | None = DEFAULT_LIMIT
    max_per_category: int | None = DEFAULT_MAX_PER_CATEGORY
    min_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def selected_decisions(self) -> list[SelectionDecision]:
        """The decisions that selected a topic."""
        return [decision for decision in self.decisions if decision.selected]

    @property
    def selection_decisions(self) -> list[SelectionDecision]:
        """Alias for :attr:`selected_decisions`."""
        return self.selected_decisions

    @property
    def all_decisions(self) -> list[SelectionDecision]:
        """Every decision, in selection order."""
        return list(self.decisions)

    @property
    def rejected_decisions(self) -> list[SelectionDecision]:
        """The decisions that skipped a topic."""
        return [decision for decision in self.decisions if not decision.selected]

    @property
    def dropped_decisions(self) -> list[SelectionDecision]:
        """Alias for :attr:`rejected_decisions`."""
        return self.rejected_decisions

    @property
    def candidates(self) -> list[TopicCandidate]:
        """The selected candidates, in selection order."""
        return [decision.candidate for decision in self.selected_decisions]

    @property
    def selected(self) -> list[TopicCandidate]:
        """Alias for :attr:`candidates`."""
        return self.candidates

    @property
    def topics(self) -> list[TopicCandidate]:
        """Alias for :attr:`candidates`."""
        return self.candidates

    @property
    def kept(self) -> list[TopicCandidate]:
        """Alias for :attr:`candidates`."""
        return self.candidates

    @property
    def chosen(self) -> list[TopicCandidate]:
        """Alias for :attr:`candidates`."""
        return self.candidates

    @property
    def picked(self) -> list[TopicCandidate]:
        """Alias for :attr:`candidates`."""
        return self.candidates

    @property
    def selection(self) -> list[TopicCandidate]:
        """Alias for :attr:`candidates`."""
        return self.candidates

    @property
    def accepted(self) -> list[TopicCandidate]:
        """Alias for :attr:`candidates`."""
        return self.candidates

    @property
    def accepted_candidates(self) -> list[TopicCandidate]:
        """Alias for :attr:`candidates`."""
        return self.candidates

    @property
    def all_candidates(self) -> list[TopicCandidate]:
        """Every inspected candidate, in input order."""
        return [decision.candidate for decision in self.decisions]

    @property
    def rejected(self) -> list[TopicCandidate]:
        """The candidates that were not selected, in input order."""
        return [decision.candidate for decision in self.rejected_decisions]

    @property
    def dropped(self) -> list[TopicCandidate]:
        """Alias for :attr:`rejected`."""
        return self.rejected

    @property
    def excluded(self) -> list[TopicCandidate]:
        """Alias for :attr:`rejected`."""
        return self.rejected

    @property
    def skipped(self) -> list[TopicCandidate]:
        """Alias for :attr:`rejected`."""
        return self.rejected

    @property
    def skipped_candidates(self) -> list[TopicCandidate]:
        """Alias for :attr:`rejected`."""
        return self.rejected

    @property
    def titles(self) -> list[str]:
        """The titles of the selected candidates."""
        return [decision.title for decision in self.selected_decisions]

    @property
    def selected_titles(self) -> list[str]:
        """Alias for :attr:`titles`."""
        return self.titles

    @property
    def rejected_titles(self) -> list[str]:
        """The titles of the candidates that were not selected."""
        return [decision.title for decision in self.rejected_decisions]

    @property
    def all_titles(self) -> list[str]:
        """The titles of every inspected candidate."""
        return [decision.title for decision in self.decisions]

    @property
    def count(self) -> int:
        """The number of selected topics."""
        return len(self.selected_decisions)

    @property
    def selected_count(self) -> int:
        """Alias for :attr:`count`."""
        return self.count

    @property
    def input_count(self) -> int:
        """The number of candidates the stage received."""
        return len(self.decisions)

    @property
    def rejected_count(self) -> int:
        """The number of candidates that were not selected."""
        return len(self.rejected_decisions)

    @property
    def dropped_count(self) -> int:
        """Alias for :attr:`rejected_count`."""
        return self.rejected_count

    @property
    def below_min_score(self) -> list[SelectionDecision]:
        """The decisions dropped by the minimum score."""
        return [
            decision
            for decision in self.decisions
            if decision.status is SelectionStatus.BELOW_MIN_SCORE
        ]

    @property
    def blocked_by_category(self) -> list[SelectionDecision]:
        """The decisions skipped by the per-category limit."""
        return [
            decision
            for decision in self.decisions
            if decision.status is SelectionStatus.CATEGORY_LIMIT
        ]

    @property
    def category_limited(self) -> list[SelectionDecision]:
        """Alias for :attr:`blocked_by_category`."""
        return self.blocked_by_category

    @property
    def blocked_by_limit(self) -> list[SelectionDecision]:
        """The decisions skipped by the per-run limit."""
        return [
            decision
            for decision in self.decisions
            if decision.status is SelectionStatus.RUN_LIMIT
        ]

    @property
    def category_counts(self) -> dict[str, int]:
        """The number of selected topics in each category."""
        return category_counts(self.selected_decisions)

    @property
    def counts(self) -> dict[str, int]:
        """Alias for :attr:`category_counts`."""
        return self.category_counts

    @property
    def all_category_counts(self) -> dict[str, int]:
        """The number of inspected candidates in each category."""
        return category_counts(self.decisions)

    @property
    def categories(self) -> list[str]:
        """The selected categories in first-selected order."""
        categories: list[str] = []
        for decision in self.selected_decisions:
            key = decision.category_key
            if key not in categories:
                categories.append(key)
        return categories

    @property
    def by_category(self) -> dict[str, list[TopicCandidate]]:
        """The selected candidates grouped by normalised category."""
        grouped: dict[str, list[TopicCandidate]] = {}
        for decision in self.selected_decisions:
            grouped.setdefault(decision.category_key, []).append(decision.candidate)
        return grouped

    @property
    def scores(self) -> list[CandidateScore]:
        """The scores attached to the inspected candidates, when available."""
        return [
            decision.candidate_score
            for decision in self.decisions
            if decision.candidate_score is not None
        ]

    def filter(self) -> list[TopicCandidate]:
        """Return the selected candidates."""
        return self.candidates

    def to_response(
        self,
        *,
        provider: str = "select",
        query: str = "",
    ) -> SourceResponse:
        """Return the selected candidates as a :class:`SourceResponse`."""
        return SourceResponse(
            provider=provider,
            query=query,
            candidates=list(self.candidates),
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the stage."""
        return {
            "decisions": [decision.to_dict() for decision in self.decisions],
            "selected": [
                candidate.model_dump(mode="json") for candidate in self.candidates
            ],
            "rejected": self.rejected_titles,
            "category_counts": self.category_counts,
            "limit": self.limit,
            "max_per_category": self.max_per_category,
            "min_score": self.min_score,
            "metadata": dict(self.metadata),
        }

    def __iter__(self) -> Iterator[TopicCandidate]:
        return iter(self.candidates)

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(
        self,
        index: int | slice,
    ) -> TopicCandidate | list[TopicCandidate]:
        return self.candidates[index]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (list, tuple)):
            return list(self.candidates) == list(other)
        return super().__eq__(other)


Selection = SelectionResult
SelectResult = SelectionResult
SelectedTopic = SelectionDecision
SelectedTopics = SelectionResult
TopicSelection = SelectionResult


def category_counts(values: Any) -> dict[str, int]:
    """Count *values* by normalised category.

    *values* may be decisions, candidates or a :class:`SelectionResult`.
    """
    if isinstance(values, SelectionResult):
        values = values.selected_decisions
    elif isinstance(values, SelectionDecision):
        values = [values]
    counts: dict[str, int] = {}
    for value in values:
        if isinstance(value, SelectionDecision):
            key = value.category_key
        else:
            key = category_of(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _prepare(
    candidates: Any,
    *,
    scores: Any = _UNSET,
    scorer: Any = None,
    dimension_scorer: Any = None,
    rubric: Any = None,
    config: Any = None,
    derivation: Any = None,
    min_score: Any = _UNSET,
) -> tuple[list[_Entry], float | None]:
    if isinstance(candidates, ScoringResult):
        return _entries(candidates), candidates.min_score
    if isinstance(candidates, CandidateScore):
        return _entries(candidates), candidates.min_score
    if isinstance(candidates, SelectionResult):
        return _entries(candidates), candidates.min_score
    scoring = (
        scores is not _UNSET
        or scorer is not None
        or dimension_scorer is not None
        or rubric is not None
        or config is not None
        or derivation is not None
    )
    if scoring:
        options: dict[str, Any] = {
            "scores": None if scores is _UNSET else scores,
            "scorer": scorer,
            "dimension_scorer": dimension_scorer,
            "rubric": rubric,
            "config": config,
            "derivation": derivation,
        }
        if min_score is not _UNSET:
            options["min_score"] = min_score
        result = score_candidates(candidates, **options)
        return _entries(result), result.min_score
    return _entries(candidates), None


def _select(
    candidates: Any,
    *,
    limit: Any = _UNSET,
    max_per_category: Any = _UNSET,
    min_score: Any = _UNSET,
    scores: Any = _UNSET,
    scorer: Any = None,
    dimension_scorer: Any = None,
    rubric: Any = None,
    config: Any = None,
    derivation: Any = None,
    metadata: Mapping[str, Any] | None = None,
    options: dict[str, Any] | None = None,
) -> SelectionResult:
    options = dict(options or {})
    limit_value = _resolve_option(limit, "limit", _LIMIT_ALIASES, options)
    if limit_value is _UNSET:
        limit_value = DEFAULT_LIMIT
    limit_value = _validate_limit(limit_value)

    category_value = _resolve_option(
        max_per_category,
        "max_per_category",
        _CATEGORY_ALIASES,
        options,
    )
    if "diversity" in options:
        diversity = options.pop("diversity")
        if diversity is False or diversity is None:
            category_value = None
        elif diversity is True:
            category_value = (
                DEFAULT_MAX_PER_CATEGORY if category_value is _UNSET else category_value
            )
        elif isinstance(diversity, int) and not isinstance(diversity, bool):
            category_value = diversity
        else:
            raise ConfigError("diversity must be a boolean or an integer")
    if category_value is _UNSET:
        category_value = DEFAULT_MAX_PER_CATEGORY
    category_value = _validate_category_limit(category_value)

    threshold = _resolve_option(
        min_score, "min_score", ("threshold", "minimum_score"), options
    )
    explicit_threshold = threshold is not _UNSET
    entries, default_min = _prepare(
        candidates,
        scores=scores,
        scorer=scorer,
        dimension_scorer=dimension_scorer,
        rubric=rubric,
        config=config,
        derivation=derivation,
        min_score=threshold,
    )
    if threshold is _UNSET:
        threshold = default_min
    minimum = _validate_min_score(threshold)

    ordered = sorted(entries, key=lambda entry: (-entry.score, entry.index))
    decisions: list[SelectionDecision] = []
    counts: dict[str, int] = {}
    selected = 0
    for entry in ordered:
        candidate = entry.candidate
        category = normalise_category(candidate.category)
        if explicit_threshold:
            effective_min = minimum
        else:
            effective_min = minimum if minimum is not None else entry.score_min
        if effective_min is not None and entry.score < effective_min:
            decisions.append(
                SelectionDecision(
                    candidate=candidate,
                    score=entry.score,
                    category=category,
                    selected=False,
                    reason="below minimum score",
                    status=SelectionStatus.BELOW_MIN_SCORE,
                    rank=None,
                    min_score=effective_min,
                    candidate_score=entry.candidate_score,
                )
            )
            continue
        if category_value is not None and counts.get(category, 0) >= category_value:
            decisions.append(
                SelectionDecision(
                    candidate=candidate,
                    score=entry.score,
                    category=category,
                    selected=False,
                    reason=f"category limit reached: {category}",
                    status=SelectionStatus.CATEGORY_LIMIT,
                    rank=None,
                    min_score=effective_min,
                    candidate_score=entry.candidate_score,
                )
            )
            continue
        if limit_value is not None and selected >= limit_value:
            decisions.append(
                SelectionDecision(
                    candidate=candidate,
                    score=entry.score,
                    category=category,
                    selected=False,
                    reason="run limit reached",
                    status=SelectionStatus.RUN_LIMIT,
                    rank=None,
                    min_score=effective_min,
                    candidate_score=entry.candidate_score,
                )
            )
            continue
        selected += 1
        counts[category] = counts.get(category, 0) + 1
        decisions.append(
            SelectionDecision(
                candidate=candidate,
                score=entry.score,
                category=category,
                selected=True,
                reason="selected",
                status=SelectionStatus.SELECTED,
                rank=selected,
                min_score=effective_min,
                candidate_score=entry.candidate_score,
            )
        )

    result_metadata: dict[str, Any] = {
        "limit": limit_value,
        "max_per_category": category_value,
        "min_score": minimum,
        "input_count": len(decisions),
        "selected": selected,
        "rejected": len(decisions) - selected,
        "below_min_score": sum(
            1
            for decision in decisions
            if decision.status is SelectionStatus.BELOW_MIN_SCORE
        ),
        "category_limited": sum(
            1
            for decision in decisions
            if decision.status is SelectionStatus.CATEGORY_LIMIT
        ),
        "run_limited": sum(
            1 for decision in decisions if decision.status is SelectionStatus.RUN_LIMIT
        ),
        "category_counts": counts,
    }
    result_metadata.update(metadata or {})
    return SelectionResult(
        decisions=tuple(decisions),
        limit=limit_value,
        max_per_category=category_value,
        min_score=minimum,
        metadata=result_metadata,
    )


def select_topics(
    candidates: Any,
    limit: Any = _UNSET,
    *,
    max_per_category: Any = _UNSET,
    min_score: Any = _UNSET,
    scores: Any = _UNSET,
    scorer: Any = None,
    dimension_scorer: Any = None,
    rubric: Any = None,
    config: Any = None,
    derivation: Any = None,
    metadata: Mapping[str, Any] | None = None,
    **options: Any,
) -> SelectionResult:
    """Select the topics a run will use from scored candidates.

    Candidates are filtered by the minimum score (from the scoring rubric when
    a :class:`ScoringResult` is given), ordered by score and then accepted
    until either the per-run ``limit`` or the ``max_per_category`` diversity
    limit is reached. Pass ``max_per_category=None`` or ``0`` to disable the
    category limit.
    """
    return _select(
        candidates,
        limit=limit,
        max_per_category=max_per_category,
        min_score=min_score,
        scores=scores,
        scorer=scorer,
        dimension_scorer=dimension_scorer,
        rubric=rubric,
        config=config,
        derivation=derivation,
        metadata=metadata,
        options=options,
    )


select_candidates = select_topics
"""Alias for :func:`select_topics`."""

select = select_topics
"""Alias for :func:`select_topics`."""

select_topics_sync = select_topics
"""Alias for :func:`select_topics`."""

selection = select_topics
"""Alias for :func:`select_topics`."""

choose_topics = select_topics
"""Alias for :func:`select_topics`."""

pick_topics = select_topics
"""Alias for :func:`select_topics`."""

select_scores = select_topics
"""Alias for :func:`select_topics`."""

select_scored = select_topics
"""Alias for :func:`select_topics`."""

select_diverse = select_topics
"""Alias for :func:`select_topics`."""

diverse_select = select_topics
"""Alias for :func:`select_topics`."""

diversify = select_topics
"""Alias for :func:`select_topics`."""


async def select_candidates_async(
    candidates: Any,
    *,
    scorer: Any = None,
    dimension_scorer: Any = None,
    scores: Any = _UNSET,
    rubric: Any = None,
    config: Any = None,
    derivation: Any = None,
    min_score: Any = _UNSET,
    **options: Any,
) -> SelectionResult:
    """Async variant that can await a scorer before selecting."""
    active_scorer = scorer if scorer is not None else dimension_scorer
    if active_scorer is None:
        return select_topics(
            candidates,
            scores=scores,
            rubric=rubric,
            config=config,
            derivation=derivation,
            min_score=min_score,
            **options,
        )
    values = [entry.candidate for entry in _entries(candidates)]
    scoring_options: dict[str, Any] = {
        "rubric": rubric,
        "config": config,
        "derivation": derivation,
    }
    if scores is not _UNSET:
        scoring_options["scores"] = scores
    if min_score is not _UNSET:
        scoring_options["min_score"] = min_score
    scored = await score_candidates_async(
        values,
        scorer=active_scorer,
        **scoring_options,
    )
    return select_topics(scored, **options)


select_topics_async = select_candidates_async


def select_candidate(
    candidate: Any,
    *,
    min_score: Any = _UNSET,
    metadata: Mapping[str, Any] | None = None,
) -> SelectionDecision:
    """Select one candidate and return its decision."""
    result = select_topics(
        [candidate],
        max_per_category=None,
        min_score=min_score,
        metadata=metadata,
    )
    return result.decisions[0]


select_topic = select_candidate
"""Alias for :func:`select_candidate`."""


def passes_selection(
    candidate: Any,
    *,
    min_score: Any = _UNSET,
) -> bool:
    """Whether *candidate* alone clears the minimum score."""
    return select_candidate(candidate, min_score=min_score).selected


def selected_topics(value: Any, **options: Any) -> list[TopicCandidate]:
    """Return the selected topics from a selection or scoring result."""
    if isinstance(value, SelectionResult) and not options:
        return value.candidates
    return select_topics(value, **options).candidates


filter_selected = selected_topics
"""Alias for :func:`selected_topics`."""

filter_selection = selected_topics
"""Alias for :func:`selected_topics`."""


class SelectionStage:
    """A reusable selection stage bound to one limit and diversity rule."""

    def __init__(
        self,
        limit: Any = _UNSET,
        *,
        max_per_category: Any = _UNSET,
        min_score: Any = _UNSET,
        rubric: Any = None,
        config: Any = None,
        derivation: Any = None,
        scorer: Any = None,
        dimension_scorer: Any = None,
        scores: Any = _UNSET,
        name: str = "select",
        **options: Any,
    ) -> None:
        limit_value = _resolve_option(limit, "limit", _LIMIT_ALIASES, options)
        if limit_value is _UNSET:
            limit_value = DEFAULT_LIMIT
        self._limit = _validate_limit(limit_value)

        category_value = _resolve_option(
            max_per_category,
            "max_per_category",
            _CATEGORY_ALIASES,
            options,
        )
        if category_value is _UNSET:
            category_value = DEFAULT_MAX_PER_CATEGORY
        self._max_per_category = _validate_category_limit(category_value)

        threshold = _resolve_option(
            min_score,
            "min_score",
            ("threshold", "minimum_score"),
            options,
        )
        self._min_score = (
            _UNSET if threshold is _UNSET else _validate_min_score(threshold)
        )
        self._rubric = rubric
        self._config = config
        self._derivation = derivation
        self._scorer = scorer if scorer is not None else dimension_scorer
        self._scores = scores
        self._name = name.strip() or "select"

    @property
    def name(self) -> str:
        """The provider name reported by :meth:`to_response`."""
        return self._name

    @property
    def limit(self) -> int | None:
        """The maximum number of topics selected per run."""
        return self._limit

    @property
    def max_per_category(self) -> int | None:
        """The maximum number of topics selected from one category."""
        return self._max_per_category

    @property
    def category_limit(self) -> int | None:
        """Alias for :attr:`max_per_category`."""
        return self._max_per_category

    @property
    def min_score(self) -> float | None:
        """The minimum score a candidate must reach, if set."""
        return None if self._min_score is _UNSET else self._min_score

    @property
    def threshold(self) -> float | None:
        """Alias for :attr:`min_score`."""
        return self._min_score

    def select(
        self,
        candidates: Any,
        *,
        limit: Any = _UNSET,
        max_per_category: Any = _UNSET,
        min_score: Any = _UNSET,
        scores: Any = _UNSET,
        scorer: Any = None,
        metadata: Mapping[str, Any] | None = None,
        **options: Any,
    ) -> SelectionResult:
        """Select topics using this stage's configuration."""
        return _select(
            candidates,
            limit=self._limit if limit is _UNSET else limit,
            max_per_category=(
                self._max_per_category
                if max_per_category is _UNSET
                else max_per_category
            ),
            min_score=self._min_score if min_score is _UNSET else min_score,
            scores=self._scores if scores is _UNSET else scores,
            scorer=self._scorer if scorer is None else scorer,
            rubric=self._rubric,
            config=self._config,
            derivation=self._derivation,
            metadata=metadata,
            options=options,
        )

    run = select
    select_topics = select
    select_candidates = select
    __call__ = select

    async def run_async(
        self,
        candidates: Any,
        *,
        limit: Any = _UNSET,
        max_per_category: Any = _UNSET,
        min_score: Any = _UNSET,
        scores: Any = _UNSET,
        scorer: Any = None,
        metadata: Mapping[str, Any] | None = None,
        **options: Any,
    ) -> SelectionResult:
        """Async variant that can await an async scorer."""
        active_scorer = self._scorer if scorer is None else scorer
        if active_scorer is None:
            return self.select(
                candidates,
                limit=limit,
                max_per_category=max_per_category,
                min_score=min_score,
                scores=scores,
                metadata=metadata,
                **options,
            )
        result = await score_candidates_async(
            candidates,
            scorer=active_scorer,
            rubric=self._rubric,
            config=self._config,
            derivation=self._derivation,
            scores=None if scores is _UNSET else scores,
        )
        return self.select(
            result,
            limit=limit,
            max_per_category=max_per_category,
            min_score=min_score,
            metadata=metadata,
            **options,
        )

    select_async = run_async

    def filter(
        self,
        candidates: Any,
        *,
        min_score: Any = _UNSET,
    ) -> list[TopicCandidate]:
        """Return only the topics this stage selects."""
        return self.select(candidates, min_score=min_score).candidates

    def to_response(
        self,
        candidates: Any,
        *,
        query: str = "",
    ) -> SourceResponse:
        """Select topics and return them as a response."""
        return self.select(candidates).to_response(
            provider=self._name,
            query=query,
        )


Selector = SelectionStage
"""Alias for :class:`SelectionStage`."""

TopicSelector = SelectionStage
"""Alias for :class:`SelectionStage`."""

SelectStage = SelectionStage
"""Alias for :class:`SelectionStage`."""

selection_stage = SelectionStage
"""Alias for :class:`SelectionStage`."""


def build_selection_stage(
    limit: Any = _UNSET,
    *,
    max_per_category: Any = _UNSET,
    **options: Any,
) -> SelectionStage:
    """Build a :class:`SelectionStage`."""
    return SelectionStage(
        limit,
        max_per_category=max_per_category,
        **options,
    )

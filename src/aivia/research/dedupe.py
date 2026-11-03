"""The dedupe stage: remove topics already seen or too similar to one.

Collection returns every candidate a source offered, including the same story
from two feeds and a topic another profile already used. This stage applies the
cheap, deterministic filters before any scoring or paid judgement:

* a normalized title is an exact duplicate, regardless of case, accents,
  punctuation or whitespace;
* an embedding whose cosine similarity clears a configured threshold is a near
  duplicate, even when the title is phrased differently.

Both checks apply inside one batch and against a :class:`DedupeHistory` bounded
by a time window. The history is deliberately shared across profiles: the "no
duplicates" invariant says the same topic is not produced twice inside the
window, even by different profiles. Every dropped candidate carries a
:class:`DedupeDecision` whose ``reason`` says which rule matched and which
earlier title it matched.

The stage is offline by default. Embeddings may be supplied explicitly, read
from a candidate, or produced by an injected synchronous callable; an async
provider can be used through :func:`dedupe_candidates_async`.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from aivia.core.errors import ConfigError
from aivia.core.models.topic import TopicCandidate
from aivia.core.similarity import (
    DEFAULT_COSINE_THRESHOLD,
    cosine_similarity,
    title_similarity,
)
from aivia.providers.protocols.embeddings import EmbeddingRequest, EmbeddingResponse
from aivia.providers.protocols.source import SourceResponse
from aivia.research.modes.otd import (
    DEFAULT_DEDUPE_WINDOW,
    SeenTopic,
    TopicHistory,
    _coerce_embedding,
    _resolve_now,
    _validate_window,
    normalize_title,
)

__all__ = [
    "DEFAULT_DEDUPE_WINDOW",
    "DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD",
    "DEFAULT_EMBEDDING_THRESHOLD",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "Dedupe",
    "DedupeDecision",
    "DedupeHistory",
    "DedupeMatch",
    "DedupeReason",
    "DedupeResult",
    "DedupeStage",
    "Deduper",
    "Deduplicator",
    "Duplicate",
    "DuplicateReason",
    "dedupe",
    "dedupe_candidates",
    "dedupe_candidates_async",
    "dedupe_sync",
    "deduplicate",
]

DEFAULT_EMBEDDING_THRESHOLD = DEFAULT_COSINE_THRESHOLD
"""Cosine similarity at or above this marks two candidates as the same topic."""

DEFAULT_SIMILARITY_THRESHOLD = DEFAULT_EMBEDDING_THRESHOLD
"""Alias for :data:`DEFAULT_EMBEDDING_THRESHOLD`."""

DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD = DEFAULT_EMBEDDING_THRESHOLD
"""Alias for :data:`DEFAULT_EMBEDDING_THRESHOLD`."""

_IDENTICAL = 1.0
"""The similarity of two normalized titles that are exactly equal."""


class DedupeReason(StrEnum):
    """Why a candidate was dropped by the dedupe stage."""

    TITLE = "title"
    """The normalized title matches an earlier topic."""

    NORMALIZED_TITLE = "title"
    """Alias for :attr:`TITLE`."""

    EMBEDDING = "embedding"
    """The embedding similarity matches an earlier topic."""

    EMBEDDING_SIMILARITY = "embedding"
    """Alias for :attr:`EMBEDDING`."""


DuplicateReason = DedupeReason
"""Alias for :class:`DedupeReason`."""


def _validate_threshold(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ConfigError(f"{label} must be between zero and one")
    return number


def _resolve_threshold(
    embedding_threshold: Any,
    threshold: Any,
    similarity_threshold: Any,
) -> float | None:
    aliases = [
        value for value in (threshold, similarity_threshold) if value is not None
    ]
    if aliases:
        values = [
            _validate_threshold(value, "similarity threshold") for value in aliases
        ]
        first = values[0]
        if any(value != first for value in values[1:]):
            raise ConfigError("conflicting similarity thresholds")
        if (
            embedding_threshold is not None
            and embedding_threshold != DEFAULT_EMBEDDING_THRESHOLD
        ):
            base = _validate_threshold(embedding_threshold, "embedding threshold")
            if base != first:
                raise ConfigError("conflicting similarity thresholds")
        return first
    if embedding_threshold is None:
        return None
    return _validate_threshold(embedding_threshold, "embedding threshold")


def _validate_profile(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ConfigError("profile must be a string")
    return value.strip()


def _validate_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{label} must be a boolean")
    return value


def _coerce_candidate(value: Any) -> TopicCandidate:
    if isinstance(value, TopicCandidate):
        return value
    if isinstance(value, str):
        return TopicCandidate(title=value)
    if isinstance(value, Mapping):
        return TopicCandidate.model_validate(dict(value))
    raise ConfigError(f"invalid candidate: {value!r}")


def _coerce_candidates(value: Any) -> list[TopicCandidate]:
    if value is None:
        return []
    if isinstance(value, (TopicCandidate, str)):
        return [_coerce_candidate(value)]
    if isinstance(value, SourceResponse):
        return list(value.candidates)
    if isinstance(value, Mapping):
        nested = value.get("candidates", value.get("topics"))
        if nested is not None and not any(
            key in value for key in ("title", "name", "topic", "subject")
        ):
            return _coerce_candidates(nested)
        return [_coerce_candidate(value)]
    if hasattr(value, "candidates") and not isinstance(value, (bytes, bytearray)):
        return _coerce_candidates(value.candidates)
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        candidates: list[TopicCandidate] = []
        for item in value:
            candidates.extend(_coerce_candidates(item))
        return candidates
    raise ConfigError("candidates must be a candidate or an iterable of candidates")


def _maybe_vector(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    vector = _coerce_embedding(value)
    return vector or None


def _mapping_vector(
    embeddings: Mapping[str, Any] | None,
    candidate: TopicCandidate,
) -> tuple[float, ...] | None:
    if not embeddings:
        return None
    for key in (candidate.title, normalize_title(candidate.title), candidate.name):
        if key in embeddings:
            return _maybe_vector(embeddings[key])
    return None


def _candidate_vector(
    candidate: TopicCandidate,
    embeddings: Mapping[str, Any] | None,
    embedder: Any,
) -> tuple[float, ...] | None:
    for attribute in ("embedding", "vector"):
        vector = _maybe_vector(getattr(candidate, attribute, None))
        if vector is not None:
            return vector
    vector = _mapping_vector(embeddings, candidate)
    if vector is not None:
        return vector
    if embedder is None:
        return None
    if isinstance(embedder, Mapping):
        return _mapping_vector(embedder, candidate)
    method = embedder if callable(embedder) else getattr(embedder, "embed", None)
    if not callable(method):
        raise ConfigError("embedder must be callable or expose embed()")
    outcome = method(candidate)
    if inspect.isawaitable(outcome):
        close = getattr(outcome, "close", None)
        if callable(close):
            close()
        raise ConfigError(
            "embedder is asynchronous; use dedupe_candidates_async()",
        )
    return _maybe_vector(outcome)


def _as_seen(entry: Any, now: datetime) -> SeenTopic:
    if isinstance(entry, SeenTopic):
        return entry
    if isinstance(entry, TopicCandidate):
        vector = getattr(entry, "embedding", None)
        if vector is None:
            vector = getattr(entry, "vector", None)
        return SeenTopic(
            title=entry.title,
            seen_at=now,
            mode=entry.mode,
            embedding=vector or (),
        )
    if isinstance(entry, str):
        return SeenTopic(title=entry, seen_at=now)
    if isinstance(entry, Mapping):
        title = entry.get("title") or entry.get("name")
        if not isinstance(title, str) or not title.strip():
            raise ConfigError("history entry needs a title")
        raw_at = entry.get("seen_at") or entry.get("at")
        seen_at = raw_at if isinstance(raw_at, datetime) else now
        return SeenTopic(
            title=title,
            seen_at=seen_at,
            mode=str(entry.get("mode", "")),
            profile=str(entry.get("profile", entry.get("profile_name", "")) or ""),
            embedding=entry.get("embedding", entry.get("vector", ())),
        )
    raise ConfigError("history entries must be titles, records or mappings")


class DedupeHistory(TopicHistory):
    """A topic history that can match by title and embedding within a window."""

    def within(
        self,
        window: timedelta | int | float | None,
        *,
        now: datetime | Callable[[], datetime] | None = None,
        profile: str | None = None,
        across_profiles: bool = True,
    ) -> list[SeenTopic]:
        """Return entries seen inside *window* ending at *now*."""
        span = _validate_window(window)
        if span <= timedelta(0):
            return []
        current = _resolve_now(now)
        entries: list[SeenTopic] = []
        for entry in self.entries:
            if not across_profiles and profile is not None and entry.profile != profile:
                continue
            if current - entry.seen_at < span:
                entries.append(entry)
        return entries

    def find_title(
        self,
        title: str,
        window: timedelta | int | float | None,
        *,
        now: datetime | Callable[[], datetime] | None = None,
        profile: str | None = None,
        across_profiles: bool = True,
        threshold: float | None = None,
    ) -> SeenTopic | None:
        """Return the first matching title inside the window, if any."""
        key = normalize_title(title)
        limit = _validate_threshold(threshold, "title threshold")
        for entry in self.within(
            window,
            now=now,
            profile=profile,
            across_profiles=across_profiles,
        ):
            if entry.key == key:
                return entry
            if limit is not None and title_similarity(title, entry.title) >= limit:
                return entry
        return None

    def find_similar(
        self,
        embedding: Sequence[float],
        window: timedelta | int | float | None,
        *,
        threshold: float | None = DEFAULT_EMBEDDING_THRESHOLD,
        now: datetime | Callable[[], datetime] | None = None,
        profile: str | None = None,
        across_profiles: bool = True,
    ) -> tuple[SeenTopic, float] | None:
        """Return the closest embedding match inside the window, if any."""
        vector = _maybe_vector(embedding)
        limit = _validate_threshold(threshold, "embedding threshold")
        if vector is None or limit is None:
            return None
        best: tuple[SeenTopic, float] | None = None
        for entry in self.within(
            window,
            now=now,
            profile=profile,
            across_profiles=across_profiles,
        ):
            if not entry.embedding or len(entry.embedding) != len(vector):
                continue
            score = cosine_similarity(vector, entry.embedding)
            if score >= limit and (best is None or score > best[1]):
                best = (entry, score)
        return best


def _coerce_history(
    history: Any,
    now: datetime | Callable[[], datetime] | None,
) -> Any:
    if history is None:
        return DedupeHistory(now=now)
    if isinstance(history, DedupeHistory):
        return history
    if isinstance(history, TopicHistory):
        return history
    if isinstance(history, Iterable) and not isinstance(
        history, (str, bytes, bytearray, Mapping)
    ):
        return DedupeHistory(history, now=now)
    if hasattr(history, "entries"):
        return history
    if hasattr(history, "seen_within") and hasattr(history, "record"):
        return history
    raise ConfigError(
        "history must be a DedupeHistory, an iterable of entries or a store "
        "with entries"
    )


def _prune_history(
    history: Any,
    *,
    window: timedelta,
    now: datetime,
) -> int:
    """Forget entries the next match would ignore, keeping the history bounded."""
    prune = getattr(history, "prune", None)
    if not callable(prune):
        return 0
    removed = prune(window, now=now)
    return removed if isinstance(removed, int) else 0


def _history_entries(
    history: Any,
    *,
    window: timedelta | int | float | None,
    now: datetime,
    profile: str,
    across_profiles: bool,
) -> list[SeenTopic]:
    if history is None:
        return []
    span = _validate_window(window)
    if span <= timedelta(0):
        return []
    within = getattr(history, "within", None)
    if callable(within):
        result = within(
            span,
            now=now,
            profile=profile,
            across_profiles=across_profiles,
        )
        return [_as_seen(entry, now) for entry in result]
    entries = getattr(history, "entries", None)
    if entries is None:
        return []
    selected: list[SeenTopic] = []
    for raw in entries:
        entry = _as_seen(raw, now)
        if not across_profiles and entry.profile != profile:
            continue
        if now - entry.seen_at < span:
            selected.append(entry)
    return selected


def _record_seen(history: Any, records: Sequence[SeenTopic]) -> None:
    if not records:
        return
    add = getattr(history, "add", None)
    if callable(add):
        for entry in records:
            add(entry)
        return
    record_fn = getattr(history, "record", None)
    if callable(record_fn):
        record_fn(records, at=records[0].seen_at)
        return
    raise ConfigError("history must expose add() or record()")


def _best_entry_match(
    vector: Sequence[float],
    entries: Sequence[SeenTopic],
    threshold: float,
) -> tuple[SeenTopic, float] | None:
    best: tuple[SeenTopic, float] | None = None
    for entry in entries:
        if not entry.embedding or len(entry.embedding) != len(vector):
            continue
        score = cosine_similarity(vector, entry.embedding)
        if score >= threshold and (best is None or score > best[1]):
            best = (entry, score)
    return best


@dataclass(frozen=True, slots=True)
class DedupeDecision:
    """One candidate and the rule that kept or dropped it."""

    candidate: TopicCandidate
    reason: DedupeReason | None = None
    matched_title: str = ""
    matched_profile: str = ""
    matched_seen_at: datetime | None = None
    similarity: float | None = None
    age: timedelta | None = None
    profile: str = ""
    observed_at: datetime | None = None

    @property
    def duplicate(self) -> bool:
        """Whether this candidate was dropped."""
        return self.reason is not None

    @property
    def is_duplicate(self) -> bool:
        """Alias for :attr:`duplicate`."""
        return self.duplicate

    @property
    def kept(self) -> bool:
        """Whether this candidate survived deduplication."""
        return self.reason is None

    @property
    def title(self) -> str:
        """The candidate title."""
        return self.candidate.title

    @property
    def matched(self) -> str:
        """The title this candidate matched, or an empty string when kept."""
        return self.matched_title

    @property
    def rule(self) -> str:
        """The matched rule as a plain string, or an empty string when kept."""
        return self.reason.value if self.reason is not None else ""

    @property
    def kind(self) -> str:
        """Alias for :attr:`rule`."""
        return self.rule

    @property
    def score(self) -> float | None:
        """The embedding similarity when the reason is embedding matching."""
        return self.similarity

    @property
    def explanation(self) -> str:
        """A human-readable description of the decision."""
        if self.reason is None:
            return f"kept {self.candidate.title!r}"
        if self.reason is DedupeReason.TITLE:
            where = (
                f"profile {self.matched_profile!r}"
                if self.matched_profile
                else "this batch"
            )
            return f"title matches {self.matched_title!r} from {where}"
        value = 0.0 if self.similarity is None else self.similarity
        where = (
            f"profile {self.matched_profile!r}"
            if self.matched_profile
            else "this batch"
        )
        return (
            f"embedding similarity {value:.3f} to {self.matched_title!r} from {where}"
        )

    @property
    def reason_text(self) -> str:
        """Alias for :attr:`explanation`."""
        return self.explanation

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable decision record."""
        payload: dict[str, Any] = {
            "title": self.candidate.title,
            "duplicate": self.duplicate,
            "reason": self.reason.value if self.reason is not None else None,
            "matched_title": self.matched_title,
            "matched_profile": self.matched_profile,
            "profile": self.profile,
            "explanation": self.explanation,
        }
        if self.similarity is not None:
            payload["similarity"] = round(self.similarity, 6)
        if self.matched_seen_at is not None:
            payload["matched_seen_at"] = self.matched_seen_at.isoformat()
        if self.age is not None:
            payload["age_seconds"] = round(self.age.total_seconds(), 6)
        return payload


Duplicate = DedupeDecision
"""Alias for :class:`DedupeDecision`."""

DedupeMatch = DedupeDecision
"""Alias for :class:`DedupeDecision`."""


@dataclass(frozen=True, slots=True)
class DedupeResult:
    """The candidates kept and the decisions that dropped the rest."""

    decisions: tuple[DedupeDecision, ...] = ()
    window: timedelta = DEFAULT_DEDUPE_WINDOW
    embedding_threshold: float | None = DEFAULT_EMBEDDING_THRESHOLD
    profile: str = ""
    across_profiles: bool = True
    history: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def candidates(self) -> list[TopicCandidate]:
        """The candidates that survived deduplication, in input order."""
        return [decision.candidate for decision in self.decisions if decision.kept]

    @property
    def unique(self) -> list[TopicCandidate]:
        """Alias for :attr:`candidates`."""
        return self.candidates

    @property
    def kept(self) -> list[TopicCandidate]:
        """Alias for :attr:`candidates`."""
        return self.candidates

    @property
    def topics(self) -> list[TopicCandidate]:
        """Alias for :attr:`candidates`."""
        return self.candidates

    @property
    def unique_candidates(self) -> list[TopicCandidate]:
        """Alias for :attr:`candidates`."""
        return self.candidates

    @property
    def kept_candidates(self) -> list[TopicCandidate]:
        """Alias for :attr:`candidates`."""
        return self.candidates

    @property
    def duplicates(self) -> list[DedupeDecision]:
        """Every dropped candidate with its reason."""
        return [decision for decision in self.decisions if decision.duplicate]

    @property
    def duplicate_candidates(self) -> list[TopicCandidate]:
        """Alias for :attr:`dropped`."""
        return self.dropped

    @property
    def dropped(self) -> list[TopicCandidate]:
        """The dropped candidates in input order."""
        return [decision.candidate for decision in self.duplicates]

    @property
    def reasons(self) -> list[DedupeReason]:
        """The reason for each dropped candidate, in input order."""
        return [
            decision.reason
            for decision in self.duplicates
            if decision.reason is not None
        ]

    @property
    def duplicate_reasons(self) -> list[DedupeReason]:
        """Alias for :attr:`reasons`."""
        return self.reasons

    @property
    def duplicate_titles(self) -> list[str]:
        """The titles of the dropped candidates."""
        return [candidate.title for candidate in self.dropped]

    @property
    def count(self) -> int:
        """The number of candidates that survived."""
        return len(self.candidates)

    @property
    def input_count(self) -> int:
        """The number of candidates the stage received."""
        return len(self.decisions)

    @property
    def duplicate_count(self) -> int:
        """The number of dropped candidates."""
        return len(self.duplicates)

    @property
    def skipped_duplicates(self) -> int:
        """Alias for :attr:`duplicate_count`."""
        return self.duplicate_count

    @property
    def title_duplicates(self) -> int:
        """How many candidates were dropped by normalized title."""
        return sum(
            1 for decision in self.duplicates if decision.reason is DedupeReason.TITLE
        )

    @property
    def embedding_duplicates(self) -> int:
        """How many candidates were dropped by embedding similarity."""
        return sum(
            1
            for decision in self.duplicates
            if decision.reason is DedupeReason.EMBEDDING
        )

    @property
    def titles(self) -> list[str]:
        """The titles of the candidates that survived."""
        return [candidate.title for candidate in self.candidates]

    def to_response(
        self,
        *,
        provider: str = "dedupe",
        query: str = "",
    ) -> SourceResponse:
        """Return the kept candidates as a :class:`SourceResponse`."""
        return SourceResponse(
            provider=provider,
            query=query,
            candidates=self.candidates,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the stage."""
        return {
            "candidates": [
                candidate.model_dump(mode="json") for candidate in self.candidates
            ],
            "decisions": [decision.to_dict() for decision in self.decisions],
            "metadata": dict(self.metadata),
        }

    def __iter__(self) -> Iterator[TopicCandidate]:
        return iter(self.candidates)

    def __len__(self) -> int:
        return len(self.candidates)


def dedupe_candidates(
    candidates: Any,
    *,
    history: Any = None,
    window: timedelta | int | float | None = DEFAULT_DEDUPE_WINDOW,
    embedding_threshold: float | None = DEFAULT_EMBEDDING_THRESHOLD,
    threshold: float | None = None,
    similarity_threshold: float | None = None,
    title_threshold: float | None = None,
    embeddings: Mapping[str, Any] | None = None,
    embedder: Any = None,
    profile: str = "",
    across_profiles: bool = True,
    now: datetime | Callable[[], datetime] | None = None,
    record: bool = True,
    record_history: bool | None = None,
    mode: str = "",
) -> DedupeResult:
    """Drop candidates whose title or embedding matches an earlier topic.

    ``window`` bounds how far back the history is consulted; a zero window
    disables history matching but still removes duplicates inside the batch.
    ``embedding_threshold=None`` disables embedding matching. ``threshold`` and
    ``similarity_threshold`` are aliases for ``embedding_threshold``.

    Embeddings are read from the candidate's ``embedding`` or ``vector`` field,
    from ``embeddings`` keyed by title or normalized title, or produced by
    ``embedder``. With ``across_profiles`` true (the default) history entries
    from every profile are compared, which enforces the cross-profile no-repeat
    invariant.
    """
    if record_history is not None:
        record = _validate_bool(record_history, "record history")
    active_record = _validate_bool(record, "record")
    active_profile = _validate_profile(profile)
    active_across = _validate_bool(across_profiles, "across profiles")
    active_threshold = _resolve_threshold(
        embedding_threshold,
        threshold,
        similarity_threshold,
    )
    active_title_threshold = _validate_threshold(title_threshold, "title threshold")
    span = _validate_window(window)
    current = _resolve_now(now)
    active_history = _coerce_history(history, now)
    _prune_history(active_history, window=span, now=current)
    entries = _history_entries(
        active_history,
        window=span,
        now=current,
        profile=active_profile,
        across_profiles=active_across,
    )

    decisions: list[DedupeDecision] = []
    accepted: list[tuple[TopicCandidate, tuple[float, ...] | None]] = []
    batch_titles: dict[str, TopicCandidate] = {}
    batch_vectors: list[tuple[TopicCandidate, tuple[float, ...]]] = []

    for candidate in _coerce_candidates(candidates):
        key = normalize_title(candidate.title)
        vector = _candidate_vector(candidate, embeddings, embedder)
        decision: DedupeDecision | None = None

        earlier = batch_titles.get(key)
        if earlier is not None:
            decision = DedupeDecision(
                candidate=candidate,
                reason=DedupeReason.TITLE,
                matched_title=earlier.title,
                matched_profile=active_profile,
                similarity=_IDENTICAL,
                profile=active_profile,
                observed_at=current,
            )
        else:
            for entry in entries:
                matches = entry.key == key
                if not matches and active_title_threshold is not None:
                    matches = (
                        title_similarity(candidate.title, entry.title)
                        >= active_title_threshold
                    )
                if not matches:
                    continue
                decision = DedupeDecision(
                    candidate=candidate,
                    reason=DedupeReason.TITLE,
                    matched_title=entry.title,
                    matched_profile=entry.profile,
                    matched_seen_at=entry.seen_at,
                    similarity=_IDENTICAL,
                    age=current - entry.seen_at,
                    profile=active_profile,
                    observed_at=current,
                )
                break

        if decision is None and vector is not None and active_threshold is not None:
            best_batch: tuple[TopicCandidate, float] | None = None
            for earlier, earlier_vector in batch_vectors:
                if len(earlier_vector) != len(vector):
                    continue
                score = cosine_similarity(vector, earlier_vector)
                if score >= active_threshold and (
                    best_batch is None or score > best_batch[1]
                ):
                    best_batch = (earlier, score)
            if best_batch is not None:
                earlier, score = best_batch
                decision = DedupeDecision(
                    candidate=candidate,
                    reason=DedupeReason.EMBEDDING,
                    matched_title=earlier.title,
                    matched_profile=active_profile,
                    similarity=score,
                    profile=active_profile,
                    observed_at=current,
                )

        if decision is None and vector is not None and active_threshold is not None:
            best_entry = _best_entry_match(vector, entries, active_threshold)
            if best_entry is not None:
                entry, score = best_entry
                decision = DedupeDecision(
                    candidate=candidate,
                    reason=DedupeReason.EMBEDDING,
                    matched_title=entry.title,
                    matched_profile=entry.profile,
                    matched_seen_at=entry.seen_at,
                    similarity=score,
                    age=current - entry.seen_at,
                    profile=active_profile,
                    observed_at=current,
                )

        if decision is not None:
            decisions.append(decision)
            continue

        decisions.append(
            DedupeDecision(
                candidate=candidate,
                profile=active_profile,
                observed_at=current,
            )
        )
        batch_titles[key] = candidate
        if vector is not None:
            batch_vectors.append((candidate, vector))
        accepted.append((candidate, vector))

    if active_record:
        records = [
            SeenTopic(
                title=candidate.title,
                seen_at=current,
                mode=mode or "dedupe",
                profile=active_profile,
                embedding=vector or (),
            )
            for candidate, vector in accepted
        ]
        _record_seen(active_history, records)

    metadata: dict[str, Any] = {
        "window_seconds": span.total_seconds(),
        "embedding_threshold": active_threshold,
        "title_threshold": active_title_threshold,
        "profile": active_profile,
        "across_profiles": active_across,
        "input_count": len(decisions),
        "kept": len(accepted),
        "skipped_duplicates": len(decisions) - len(accepted),
        "title_duplicates": sum(
            1 for decision in decisions if decision.reason is DedupeReason.TITLE
        ),
        "embedding_duplicates": sum(
            1 for decision in decisions if decision.reason is DedupeReason.EMBEDDING
        ),
        "recorded": active_record,
    }
    return DedupeResult(
        decisions=tuple(decisions),
        window=span,
        embedding_threshold=active_threshold,
        profile=active_profile,
        across_profiles=active_across,
        history=active_history,
        metadata=metadata,
    )


dedupe = dedupe_candidates
"""Alias for :func:`dedupe_candidates`."""

dedupe_sync = dedupe_candidates
"""Alias for :func:`dedupe_candidates` for callers next to an async API."""

deduplicate = dedupe_candidates
"""Alias for :func:`dedupe_candidates`."""


def _vectors_from_outcome(
    outcome: Any,
    expected: int,
) -> list[tuple[float, ...]]:
    if isinstance(outcome, EmbeddingResponse):
        ordered = sorted(outcome.embeddings, key=lambda item: item.index)
        vectors = [_maybe_vector(item.vector) for item in ordered]
    elif hasattr(outcome, "embeddings"):
        return _vectors_from_outcome(outcome.embeddings, expected)
    else:
        vectors = [_maybe_vector(item) for item in outcome]
    if len(vectors) != expected:
        raise ConfigError(
            "embedder returned the wrong number of vectors",
            details={"expected": expected, "received": len(vectors)},
        )
    if any(vector is None for vector in vectors):
        raise ConfigError("embedder returned an empty vector")
    return [vector for vector in vectors if vector is not None]


async def _embed_candidates_async(
    candidates: Sequence[TopicCandidate],
    embedder: Any,
) -> dict[str, tuple[float, ...]]:
    if embedder is None:
        return {}
    if isinstance(embedder, Mapping):
        return {
            key: vector
            for key, value in embedder.items()
            if (vector := _maybe_vector(value)) is not None
        }
    texts = [candidate.title for candidate in candidates]
    method = getattr(embedder, "embed", None)
    if callable(method):
        try:
            outcome = method(EmbeddingRequest(inputs=texts))
        except TypeError:
            outcome = method(texts)
    elif callable(embedder):
        outcome = embedder(texts)
    else:
        raise ConfigError("embedder must be callable or expose embed()")
    if inspect.isawaitable(outcome):
        outcome = await outcome
    vectors = _vectors_from_outcome(outcome, len(texts))
    return {
        candidate.title: vector
        for candidate, vector in zip(candidates, vectors, strict=True)
    }


async def dedupe_candidates_async(
    candidates: Any,
    *,
    embedder: Any = None,
    embeddings: Mapping[str, Any] | None = None,
    **options: Any,
) -> DedupeResult:
    """Async wrapper that resolves embeddings through an async provider first."""
    values = _coerce_candidates(candidates)
    resolved = dict(embeddings or {})
    if embedder is not None:
        produced = await _embed_candidates_async(values, embedder)
        for title, vector in produced.items():
            resolved.setdefault(title, vector)
    return dedupe_candidates(
        values,
        embeddings=resolved,
        embedder=None,
        **options,
    )


class DedupeStage:
    """A reusable dedupe stage bound to one window, threshold and history."""

    def __init__(
        self,
        *,
        window: timedelta | int | float | None = DEFAULT_DEDUPE_WINDOW,
        embedding_threshold: float | None = DEFAULT_EMBEDDING_THRESHOLD,
        threshold: float | None = None,
        similarity_threshold: float | None = None,
        title_threshold: float | None = None,
        embeddings: Mapping[str, Any] | None = None,
        embedder: Any = None,
        history: Any = None,
        profile: str = "",
        across_profiles: bool = True,
        now: datetime | Callable[[], datetime] | None = None,
        record: bool = True,
        mode: str = "",
        name: str = "dedupe",
    ) -> None:
        self._window = _validate_window(window)
        self._embedding_threshold = _resolve_threshold(
            embedding_threshold,
            threshold,
            similarity_threshold,
        )
        self._title_threshold = _validate_threshold(
            title_threshold,
            "title threshold",
        )
        self._embeddings = dict(embeddings) if embeddings is not None else None
        self._embedder = embedder
        self._history = _coerce_history(history, now)
        self._profile = _validate_profile(profile)
        self._across_profiles = _validate_bool(across_profiles, "across profiles")
        self._now = now
        self._record = _validate_bool(record, "record")
        self._mode = mode
        self._name = name.strip() or "dedupe"

    @property
    def name(self) -> str:
        """The provider name reported by :meth:`to_response`."""
        return self._name

    @property
    def window(self) -> timedelta:
        """How far back the history is consulted."""
        return self._window

    @property
    def embedding_threshold(self) -> float | None:
        """The cosine threshold for embedding duplicates."""
        return self._embedding_threshold

    @property
    def title_threshold(self) -> float | None:
        """The optional fuzzy title threshold."""
        return self._title_threshold

    @property
    def history(self) -> Any:
        """The history this stage reads and writes."""
        return self._history

    @property
    def profile(self) -> str:
        """The profile the stage records new topics under."""
        return self._profile

    @property
    def across_profiles(self) -> bool:
        """Whether history from every profile is compared."""
        return self._across_profiles

    def dedupe(
        self,
        candidates: Any,
        *,
        history: Any = None,
        profile: str | None = None,
        now: datetime | Callable[[], datetime] | None = None,
        record: bool | None = None,
    ) -> DedupeResult:
        """Dedupe *candidates* using this stage's configuration."""
        return dedupe_candidates(
            candidates,
            history=self._history if history is None else history,
            window=self._window,
            embedding_threshold=self._embedding_threshold,
            title_threshold=self._title_threshold,
            embeddings=self._embeddings,
            embedder=self._embedder,
            profile=self._profile if profile is None else profile,
            across_profiles=self._across_profiles,
            now=self._now if now is None else now,
            record=self._record if record is None else record,
            mode=self._mode,
        )

    run = dedupe

    __call__ = dedupe

    async def dedupe_async(
        self,
        candidates: Any,
        *,
        history: Any = None,
        profile: str | None = None,
        now: datetime | Callable[[], datetime] | None = None,
        record: bool | None = None,
    ) -> DedupeResult:
        """Async variant that can await an embedding provider."""
        return await dedupe_candidates_async(
            candidates,
            embedder=self._embedder,
            embeddings=self._embeddings,
            history=self._history if history is None else history,
            window=self._window,
            embedding_threshold=self._embedding_threshold,
            title_threshold=self._title_threshold,
            profile=self._profile if profile is None else profile,
            across_profiles=self._across_profiles,
            now=self._now if now is None else now,
            record=self._record if record is None else record,
            mode=self._mode,
        )

    run_async = dedupe_async


Dedupe = DedupeStage
"""Alias for :class:`DedupeStage`."""

Deduplicator = DedupeStage
"""Alias for :class:`DedupeStage`."""

Deduper = DedupeStage
"""Alias for :class:`DedupeStage`."""

"""``aivia research``: curate topics without running source collection.

``research add`` is the manual entry point into the research engine. The editor
supplies one or more topics (and their source URLs) directly, so the collection
stage is skipped. The command still runs the two safety stages that matter for
a hand-picked topic: dedupe against every topic added before, and verification
against independent source domains.

Accepted topics are persisted in a small JSON store under the user data
directory. A topic that only one domain reports is still stored, but it is
marked unverified and the reason is recorded, matching the verification stage
used by the rest of research.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml
from pydantic import ValidationError

from aivia.cli.registry import register_command
from aivia.core import paths
from aivia.core.config.profile import ContentProfileConfig
from aivia.core.config.resolve import resolve_config
from aivia.core.errors import ConfigError
from aivia.core.models.base import canonical_json, utcnow
from aivia.core.models.topic import TopicCandidate
from aivia.research.brief import BriefDerivation, brief_hash
from aivia.research.dedupe import (
    DEFAULT_DEDUPE_WINDOW,
    DedupeDecision,
    DedupeHistory,
    dedupe_candidates,
)
from aivia.research.derived import BriefDerivationStore
from aivia.research.modes.niche import RelevanceGate
from aivia.research.modes.otd import SeenTopic
from aivia.research.scoring import (
    CandidateScore,
    ScoringRubric,
    score_candidates,
)
from aivia.research.verify import (
    DEFAULT_MIN_DOMAINS,
    VerificationDecision,
    VerificationResult,
    verify_candidates,
)

__all__ = [
    "DEFAULT_STORE_FILENAME",
    "AddResult",
    "PreviewResult",
    "PreviewTopic",
    "StoredTopic",
    "TopicStore",
    "add_topics",
    "configure",
    "preview_topics",
    "run_add",
    "run_preview",
    "run_research",
    "topic_store_path",
]

DEFAULT_STORE_FILENAME = "topics.json"
"""The file under ``data/research`` that holds manually added topics."""

_STORE_VERSION = 1


def topic_store_path(
    data_dir: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return the default persistent topic store path."""
    root = paths.data_dir(environ) if data_dir is None else Path(data_dir)
    return Path(root).expanduser() / "research" / DEFAULT_STORE_FILENAME


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_time(value: Any, default: datetime) -> datetime:
    if isinstance(value, datetime):
        return _ensure_utc(value)
    if isinstance(value, str) and value.strip():
        try:
            return _ensure_utc(datetime.fromisoformat(value.strip()))
        except ValueError:
            return default
    return default


@dataclass(frozen=True, slots=True)
class StoredTopic:
    """One accepted topic, when it was added and by which profile."""

    candidate: TopicCandidate
    added_at: datetime
    profile: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "added_at", _ensure_utc(self.added_at))
        if not isinstance(self.profile, str):
            raise ConfigError("stored topic profile must be a string")
        object.__setattr__(self, "profile", self.profile.strip())
        if not isinstance(self.reason, str):
            raise ConfigError("stored topic reason must be a string")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable store record."""
        payload = self.candidate.model_dump(mode="json")
        payload["added_at"] = self.added_at.isoformat()
        payload["profile"] = self.profile
        payload["verification_reason"] = self.reason
        return payload


class TopicStore:
    """A deterministic JSON store of manually added research topics."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        data_dir: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._path = (
            topic_store_path(data_dir, environ=environ)
            if path is None
            else Path(path).expanduser()
        )

    @property
    def path(self) -> Path:
        """The JSON file backing this store."""
        return self._path

    def load(self) -> list[StoredTopic]:
        """Read every stored topic in insertion order."""
        if not self._path.exists():
            return []
        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"cannot read topic store: {self._path}") from exc
        if not text.strip():
            return []
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid topic store JSON: {self._path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ConfigError(f"topic store root must be a mapping: {self._path}")
        raw_topics = payload.get("topics", [])
        if not isinstance(raw_topics, Sequence) or isinstance(
            raw_topics, (str, bytes, bytearray)
        ):
            raise ConfigError(f"topic store topics must be a list: {self._path}")
        now = utcnow()
        stored: list[StoredTopic] = []
        for item in raw_topics:
            if isinstance(item, str):
                item = {"title": item}
            if not isinstance(item, Mapping):
                raise ConfigError(f"invalid topic entry in {self._path}")
            values = dict(item)
            added_at = _parse_time(values.pop("added_at", None), now)
            profile = values.pop("profile", "")
            reason = values.pop("verification_reason", "")
            try:
                candidate = TopicCandidate.model_validate(values)
            except ValidationError as exc:
                raise ConfigError(
                    f"invalid topic entry in {self._path}: {exc}"
                ) from exc
            stored.append(
                StoredTopic(
                    candidate=candidate,
                    added_at=added_at,
                    profile=str(profile or ""),
                    reason=str(reason or ""),
                )
            )
        return stored

    def save(self, entries: Iterable[StoredTopic]) -> None:
        """Write *entries* atomically and deterministically."""
        payload = {
            "version": _STORE_VERSION,
            "topics": [entry.to_dict() for entry in entries],
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self._path.name}.",
            suffix=".tmp",
            dir=self._path.parent,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(canonical_json(payload, indent=2))
                handle.write("\n")
            os.replace(temporary, self._path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def append(self, entries: Iterable[StoredTopic]) -> list[StoredTopic]:
        """Append *entries* to the store and return the full list."""
        stored = self.load()
        stored.extend(entries)
        self.save(stored)
        return stored

    def history(
        self,
        *,
        now: datetime | None = None,
    ) -> DedupeHistory:
        """Return the store as dedupe history for the next add."""
        current = utcnow() if now is None else now
        return DedupeHistory(
            [
                SeenTopic(
                    title=entry.candidate.title,
                    seen_at=entry.added_at,
                    mode=entry.candidate.mode,
                    profile=entry.profile,
                    embedding=getattr(entry.candidate, "embedding", ()) or (),
                )
                for entry in self.load()
            ],
            now=current,
        )


@dataclass(frozen=True, slots=True)
class AddResult:
    """The accepted topics, rejected duplicates and verification decisions."""

    stored: tuple[StoredTopic, ...] = ()
    verification: VerificationResult | None = None
    duplicates: tuple[DedupeDecision, ...] = ()
    store_path: Path | None = None
    recorded: bool = True

    @property
    def added(self) -> list[TopicCandidate]:
        """The accepted candidates in input order."""
        return [entry.candidate for entry in self.stored]

    @property
    def added_count(self) -> int:
        """The number of accepted topics."""
        return len(self.stored)

    @property
    def rejected_count(self) -> int:
        """The number of rejected duplicate topics."""
        return len(self.duplicates)

    @property
    def verified_count(self) -> int:
        """The number of accepted topics that cleared verification."""
        if self.verification is None:
            return 0
        return self.verification.verified_count

    @property
    def unverified_count(self) -> int:
        """The number of accepted topics that did not clear verification."""
        if self.verification is None:
            return 0
        return self.verification.unverified_count

    def _decisions(self) -> list[VerificationDecision]:
        if self.verification is None:
            return []
        return list(self.verification.decisions)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the add operation."""
        added: list[dict[str, Any]] = []
        for entry, decision in zip(
            self.stored,
            self._decisions(),
            strict=False,
        ):
            payload = entry.candidate.model_dump(mode="json")
            payload.update(
                {
                    "added_at": entry.added_at.isoformat(),
                    "profile": entry.profile,
                    "verified": decision.verified,
                    "status": decision.status.value,
                    "domains": list(decision.domains),
                    "domain_count": decision.domain_count,
                    "verification_reason": decision.reason,
                }
            )
            added.append(payload)
        return {
            "added": added,
            "rejected": [decision.to_dict() for decision in self.duplicates],
            "store": None if self.store_path is None else str(self.store_path),
            "recorded": self.recorded,
            "counts": {
                "added": self.added_count,
                "rejected": self.rejected_count,
                "verified": self.verified_count,
                "unverified": self.unverified_count,
            },
        }


def _as_topic_candidate(value: Any) -> TopicCandidate:
    if isinstance(value, TopicCandidate):
        return value
    return TopicCandidate.model_validate(
        value.model_dump(exclude={"claims", "candidate"})
    )


def add_topics(
    candidates: Iterable[TopicCandidate],
    *,
    store: TopicStore,
    profile: str = "",
    min_domains: int = DEFAULT_MIN_DOMAINS,
    window: timedelta | int | float | None = DEFAULT_DEDUPE_WINDOW,
    now: datetime | None = None,
    record: bool = True,
) -> AddResult:
    """Dedupe, verify and store manually supplied *candidates*.

    Collection is deliberately not part of this function: the caller has
    already chosen the topics. Dedupe runs first so a repeat is rejected before
    any verification work, then the surviving topics are verified against
    independent domains and persisted in input order.
    """
    current = utcnow() if now is None else now
    values = [
        candidate.model_copy(update={"mode": candidate.mode or "manual"})
        for candidate in candidates
    ]
    deduped = dedupe_candidates(
        values,
        history=store.history(now=current),
        window=window,
        profile=profile,
        now=current,
        record=False,
    )
    verification = verify_candidates(deduped.candidates, min_domains=min_domains)
    stored = tuple(
        StoredTopic(
            candidate=_as_topic_candidate(decision.candidate),
            added_at=current,
            profile=profile,
            reason=decision.reason,
        )
        for decision in verification.decisions
    )
    if record and stored:
        store.append(stored)
    return AddResult(
        stored=stored,
        verification=verification,
        duplicates=tuple(deduped.duplicates),
        store_path=store.path,
        recorded=record,
    )


_WORD = re.compile(r"[a-z0-9][a-z0-9'_-]*", re.IGNORECASE)
_BRIEF_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "also",
        "and",
        "are",
        "but",
        "can",
        "for",
        "from",
        "has",
        "have",
        "into",
        "its",
        "not",
        "our",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "with",
        "you",
        "your",
    }
)


def _unique_terms(values: Iterable[str]) -> tuple[str, ...]:
    terms: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        term = " ".join(value.split())
        key = term.casefold()
        if term and key not in seen:
            seen.add(key)
            terms.append(term)
    return tuple(terms)


def _normalise_keyword_values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return _unique_terms([value])
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return _unique_terms(str(item) for item in value)
    return ()


def _option_keywords(args: argparse.Namespace) -> tuple[str, ...]:
    values: list[str] = []
    for value in list(getattr(args, "keyword_options", None) or []):
        if isinstance(value, str):
            values.append(value)
    for group in list(getattr(args, "keyword_lists", None) or []):
        if isinstance(group, str):
            values.extend(
                part.strip()
                for part in group.replace(";", ",").split(",")
                if part.strip()
            )
    return _unique_terms(values)


def _brief_words(text: str, *, limit: int = 24) -> tuple[str, ...]:
    if not isinstance(text, str) or not text.strip():
        return ()
    words = [
        match.group(0).casefold()
        for match in _WORD.finditer(text)
        if len(match.group(0)) >= 3
        and match.group(0).casefold() not in _BRIEF_STOPWORDS
    ]
    return _unique_terms(words)[:limit]


def _fallback_brief(
    profile: ContentProfileConfig,
    brief: str,
    keywords: Sequence[str],
) -> BriefDerivation:
    niche = profile.niche.strip()
    explicit = _unique_terms(keywords) or _unique_terms(profile.keywords)
    queries = _unique_terms(
        [
            *explicit,
            *([niche] if niche else []),
            *_brief_words(brief),
        ]
    )
    return BriefDerivation(
        queries=list(queries),
        subtopics=[],
        include_terms=list(explicit),
        exclude_terms=[],
        brief=brief,
        brief_hash=brief_hash(brief),
        keywords=list(explicit),
    )


def _candidate_text(candidate: TopicCandidate) -> str:
    parts = [
        candidate.title,
        candidate.summary,
        candidate.category,
        *candidate.keywords,
    ]
    text = " ".join(part for part in parts if part)
    return " ".join(text.casefold().split())


def _term_matches(text: str, term: str) -> bool:
    needle = " ".join(term.casefold().split())
    return bool(needle) and needle in text


def _preview_part_terms(
    derivation: BriefDerivation,
    profile: ContentProfileConfig,
) -> dict[str, tuple[str, ...]]:
    return {
        "keywords": _unique_terms(derivation.keywords),
        "include_terms": _unique_terms(derivation.include_terms),
        "subtopics": _unique_terms(derivation.subtopics),
        "queries": _unique_terms(derivation.queries),
        "niche": _unique_terms([profile.niche]),
    }


def _matched_parts(
    candidate: TopicCandidate,
    part_terms: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    text = _candidate_text(candidate)
    matched: dict[str, tuple[str, ...]] = {}
    for name, terms in part_terms.items():
        matched[name] = tuple(term for term in terms if _term_matches(text, term))
    return matched


def _matched_excluded(
    candidate: TopicCandidate,
    terms: Sequence[str],
) -> tuple[str, ...]:
    text = _candidate_text(candidate)
    return tuple(term for term in terms if _term_matches(text, term))


def _title_tokens(title: str) -> set[str]:
    return {
        match.group(0).casefold()
        for match in _WORD.finditer(title)
        if len(match.group(0)) >= 3
        and match.group(0).casefold() not in _BRIEF_STOPWORDS
    }


def _originality(
    candidate: TopicCandidate,
    pool: Sequence[TopicCandidate],
    index: int,
) -> float:
    tokens = _title_tokens(candidate.title)
    if not tokens:
        return 0.0
    best = 0.0
    for other_index, other in enumerate(pool):
        if other_index == index:
            continue
        other_tokens = _title_tokens(other.title)
        union = tokens | other_tokens
        if union:
            best = max(best, len(tokens & other_tokens) / len(union))
    return round(max(0.0, 1.0 - best), 6)


def _signal_score(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return round(min(1.0, max(0.0, float(value))), 6)
    return 0.5


def _preview_dimensions(
    candidate: TopicCandidate,
    rubric: ScoringRubric,
    *,
    relevance: float,
    originality: float,
) -> dict[str, float]:
    domains = len(candidate.source_domains)
    accuracy = round(min(1.0, domains / 2.0), 6)
    engagement = _signal_score(candidate.signal)
    dimensions: dict[str, float] = {}
    for name in rubric.dimensions:
        key = name.casefold()
        if key in {"relevance", "fit", "theme", "brief", "topic"}:
            value = relevance
        elif key in {"originality", "novelty"}:
            value = originality
        elif key in {"accuracy", "verification", "sources", "trust"}:
            value = accuracy
        elif key in {"engagement", "signal", "interest"}:
            value = engagement
        else:
            value = relevance
        dimensions[name] = round(min(1.0, max(0.0, value)), 6)
    return dimensions


def _preview_reason(
    candidate: TopicCandidate,
    *,
    matched_required: Sequence[str],
    required: Sequence[str],
    excluded: Sequence[str],
) -> str:
    if candidate.reason.strip():
        return candidate.reason.strip()
    if excluded:
        return "matches excluded brief terms: " + ", ".join(excluded)
    if matched_required:
        return (
            f"matches {len(matched_required)} of {len(required)} brief terms: "
            + ", ".join(matched_required)
        )
    if required:
        return "no brief terms matched"
    return "no brief terms configured"


def _preview_cost(
    config: Any,
    count: int,
) -> tuple[float, dict[str, Any]]:
    if count <= 0 or config is None:
        return 0.0, {
            "usd": 0.0,
            "paid_usd": 0.0,
            "calls": 0,
            "unit": "call",
            "estimated": True,
            "complete": True,
            "note": "nothing to price",
        }
    try:
        from aivia.cli.run import capability_configs, provider_metadata
        from aivia.core.config.formats import FormatRecipeConfig, StageConfig
        from aivia.core.estimate import Estimator

        recipe = FormatRecipeConfig(
            stages=[
                StageConfig(
                    stage="write",
                    options={"units": float(count), "unit": "call"},
                )
            ],
            capabilities=["llm"],
        )
        estimate = Estimator(
            provider_metadata(config),
            configs=capability_configs(config) or None,
        ).estimate("research_preview", recipe)
    except (ConfigError, TypeError, ValueError) as exc:
        return 0.0, {
            "usd": 0.0,
            "paid_usd": 0.0,
            "calls": count,
            "unit": "call",
            "estimated": True,
            "complete": False,
            "note": str(exc),
        }
    note = ""
    if not estimate.complete:
        note = "; ".join(stage.note for stage in estimate.missing_stages)
    return estimate.total_usd, {
        "usd": estimate.total_usd,
        "paid_usd": 0.0,
        "calls": count,
        "unit": "call",
        "estimated": True,
        "complete": estimate.complete,
        "note": note,
        "estimate": estimate.to_dict(),
    }


@dataclass(frozen=True, slots=True)
class PreviewTopic:
    """One scored topic with the brief parts it matched and its cost share."""

    score: CandidateScore
    matched_parts: dict[str, tuple[str, ...]] = field(default_factory=dict)
    matched_terms: tuple[str, ...] = ()
    excluded_terms: tuple[str, ...] = ()
    cost_usd: float = 0.0

    @property
    def candidate(self) -> TopicCandidate:
        """The previewed candidate."""
        return self.score.candidate

    @property
    def title(self) -> str:
        """The candidate title."""
        return self.score.title

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable preview record."""
        parts = {name: list(terms) for name, terms in self.matched_parts.items()}
        payload = self.score.to_dict()
        payload.update(
            {
                "score": round(self.score.overall, 6),
                "scores": dict(self.score.dimensions),
                "dimensions": dict(self.score.dimensions),
                "reason": self.score.reason,
                "matched": list(self.matched_terms),
                "matched_terms": list(self.matched_terms),
                "matched_brief_parts": parts,
                "brief_parts": parts,
                "excluded_terms": list(self.excluded_terms),
                "cost_usd": self.cost_usd,
                "cost": {
                    "usd": self.cost_usd,
                    "paid_usd": 0.0,
                    "estimated": True,
                },
            }
        )
        return payload


@dataclass(frozen=True, slots=True)
class PreviewResult:
    """The scored topics, the brief settings and the estimated preview cost."""

    topics: tuple[PreviewTopic, ...] = ()
    profile: str = ""
    brief: str = ""
    derivation: BriefDerivation | None = None
    rubric: ScoringRubric | None = None
    cost_usd: float = 0.0
    cost: dict[str, Any] = field(default_factory=dict)
    source_store: Path | None = None

    def __len__(self) -> int:
        return len(self.topics)

    @property
    def count(self) -> int:
        """The number of previewed topics."""
        return len(self.topics)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable preview summary."""
        topics = [topic.to_dict() for topic in self.topics]
        return {
            "profile": self.profile,
            "brief": self.brief,
            "brief_hash": None
            if self.derivation is None
            else self.derivation.brief_hash,
            "keywords": []
            if self.derivation is None
            else list(self.derivation.keywords),
            "queries": [] if self.derivation is None else list(self.derivation.queries),
            "subtopics": []
            if self.derivation is None
            else list(self.derivation.subtopics),
            "include_terms": (
                [] if self.derivation is None else list(self.derivation.include_terms)
            ),
            "exclude_terms": (
                [] if self.derivation is None else list(self.derivation.exclude_terms)
            ),
            "rubric": None if self.rubric is None else self.rubric.to_dict(),
            "topics": topics,
            "results": topics,
            "cost": dict(self.cost),
            "cost_usd": self.cost_usd,
            "store": None if self.source_store is None else str(self.source_store),
            "counts": {
                "topics": len(self.topics),
                "selected": sum(1 for topic in self.topics if topic.score.selected),
                "passed": sum(1 for topic in self.topics if topic.score.passed),
                "failed": sum(1 for topic in self.topics if not topic.score.passed),
            },
        }


def preview_topics(
    candidates: Iterable[TopicCandidate | Mapping[str, Any]],
    *,
    profile: ContentProfileConfig | None = None,
    profile_name: str = "",
    brief: str = "",
    keywords: Sequence[str] | str | None = None,
    derivation: BriefDerivation | None = None,
    rubric: ScoringRubric | Any = None,
    config: Any = None,
    cost: Mapping[str, Any] | None = None,
    cost_usd: float | None = None,
) -> PreviewResult:
    """Score *candidates* against the brief without calling a provider.

    The preview is deterministic: it derives brief terms from the supplied
    derivation or from the profile, matches them against each candidate and
    turns the matches into rubric dimension scores. The returned cost is an
    estimate of what scoring these candidates with the configured LLM would
    cost; the preview itself makes no paid call.
    """
    values = [
        candidate
        if isinstance(candidate, TopicCandidate)
        else TopicCandidate.model_validate(candidate)
        for candidate in candidates
    ]
    active_profile = profile if profile is not None else ContentProfileConfig()
    active_brief = (
        brief.strip()
        if isinstance(brief, str) and brief.strip()
        else active_profile.brief
    )
    explicit = _normalise_keyword_values(keywords)
    if derivation is None:
        active_derivation = _fallback_brief(
            active_profile,
            active_brief,
            explicit or active_profile.keywords,
        )
    elif explicit:
        active_derivation = derivation.with_explicit_keywords(explicit)
    else:
        active_derivation = derivation
    active_rubric = (
        rubric
        if isinstance(rubric, ScoringRubric)
        else ScoringRubric.from_config(
            rubric if rubric is not None else getattr(config, "scoring", None)
        )
    )
    gate = RelevanceGate.from_derivation(
        active_derivation,
        niche=active_profile.niche,
        min_relevance=None,
    )
    part_terms = _preview_part_terms(active_derivation, active_profile)
    required = tuple(gate.required_terms)
    payloads: list[dict[str, Any]] = []
    matches: list[
        tuple[dict[str, tuple[str, ...]], tuple[str, ...], tuple[str, ...]]
    ] = []
    for index, candidate in enumerate(values):
        text = _candidate_text(candidate)
        matched_parts = _matched_parts(candidate, part_terms)
        matched_terms = _unique_terms(
            term for terms in matched_parts.values() for term in terms
        )
        excluded = _matched_excluded(candidate, active_derivation.exclude_terms)
        matched_required = tuple(term for term in required if _term_matches(text, term))
        payloads.append(
            {
                "dimensions": _preview_dimensions(
                    candidate,
                    active_rubric,
                    relevance=gate.term_score(candidate),
                    originality=_originality(candidate, values, index),
                ),
                "reason": _preview_reason(
                    candidate,
                    matched_required=matched_required,
                    required=required,
                    excluded=excluded,
                ),
            }
        )
        matches.append((matched_parts, matched_terms, excluded))
    scored = score_candidates(values, scores=payloads, rubric=active_rubric)
    if cost_usd is None and cost is not None:
        cost_usd = float(cost.get("usd", 0.0))
    if cost_usd is None:
        total_cost, cost_payload = _preview_cost(config, len(values))
    else:
        total_cost = float(cost_usd)
        cost_payload = dict(cost or {})
        cost_payload.setdefault("usd", total_cost)
        cost_payload.setdefault("paid_usd", 0.0)
        cost_payload.setdefault("calls", len(values))
        cost_payload.setdefault("unit", "call")
        cost_payload.setdefault("estimated", True)
    share = total_cost / len(values) if values else 0.0
    topics = tuple(
        PreviewTopic(
            score=score,
            matched_parts=matched_parts,
            matched_terms=matched_terms,
            excluded_terms=excluded,
            cost_usd=share,
        )
        for score, (matched_parts, matched_terms, excluded) in zip(
            scored.scores,
            matches,
            strict=True,
        )
    )
    return PreviewResult(
        topics=topics,
        profile=profile_name,
        brief=active_brief,
        derivation=active_derivation,
        rubric=active_rubric,
        cost_usd=total_cost,
        cost=cost_payload,
    )


def _configure_add(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "topics",
        nargs="*",
        metavar="TITLE",
        help="topic title; a bare http(s) argument is treated as a source URL",
    )
    parser.add_argument(
        "--title",
        dest="title_options",
        action="append",
        default=None,
        metavar="TITLE",
        help="topic title; repeatable",
    )
    parser.add_argument(
        "--source",
        "--url",
        "--source-url",
        dest="sources",
        action="append",
        default=None,
        metavar="URL",
        help="source URL for the topic; repeatable",
    )
    parser.add_argument(
        "--sources",
        "--urls",
        "--source-urls",
        dest="source_lists",
        action="append",
        default=None,
        metavar="URLS",
        help="comma-separated source URLs",
    )
    parser.add_argument(
        "--file",
        "--input",
        "--topics-file",
        dest="file",
        metavar="PATH",
        help="read topics from a JSON or YAML file (use - for standard input)",
    )
    parser.add_argument(
        "--store",
        metavar="PATH",
        help="topic store file instead of data/research/topics.json",
    )
    parser.add_argument(
        "--min-domains",
        type=int,
        default=DEFAULT_MIN_DOMAINS,
        metavar="N",
        help="independent source domains required to verify a topic",
    )
    parser.add_argument(
        "--dedupe-window",
        "--window",
        dest="dedupe_window",
        type=float,
        default=DEFAULT_DEDUPE_WINDOW.total_seconds() / 86400,
        metavar="DAYS",
        help="days of topic history to check for duplicates",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="print a machine-readable summary",
    )
    parser.set_defaults(_research_handler=run_add)


def _configure_preview(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "topics",
        nargs="*",
        metavar="TITLE",
        help="topic title; a bare http(s) argument is treated as a source URL",
    )
    parser.add_argument(
        "--title",
        dest="title_options",
        action="append",
        default=None,
        metavar="TITLE",
        help="topic title; repeatable",
    )
    parser.add_argument(
        "--source",
        "--url",
        "--source-url",
        dest="sources",
        action="append",
        default=None,
        metavar="URL",
        help="source URL for the topic; repeatable",
    )
    parser.add_argument(
        "--sources",
        "--urls",
        "--source-urls",
        dest="source_lists",
        action="append",
        default=None,
        metavar="URLS",
        help="comma-separated source URLs",
    )
    parser.add_argument(
        "--file",
        "--input",
        "--topics-file",
        dest="file",
        metavar="PATH",
        help="read topics from a JSON or YAML file (use - for standard input)",
    )
    parser.add_argument(
        "--store",
        metavar="PATH",
        help="read topics from this store instead of data/research/topics.json",
    )
    parser.add_argument(
        "--brief",
        metavar="TEXT",
        help="brief text to score against instead of the selected profile brief",
    )
    parser.add_argument(
        "--niche",
        metavar="TEXT",
        help="niche to score against instead of the selected profile niche",
    )
    parser.add_argument(
        "--keyword",
        dest="keyword_options",
        action="append",
        default=None,
        metavar="TERM",
        help="explicit keyword that must take precedence; repeatable",
    )
    parser.add_argument(
        "--keywords",
        dest="keyword_lists",
        action="append",
        default=None,
        metavar="TERMS",
        help="comma-separated explicit keywords",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        metavar="SCORE",
        help="override the rubric minimum score for the preview",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="print a machine-readable preview",
    )
    parser.set_defaults(_research_handler=run_preview)


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the research group and its subcommands."""
    subparsers = parser.add_subparsers(dest="research_command", metavar="COMMAND")
    add = subparsers.add_parser(
        "add",
        help="Add a hand-picked topic without collecting sources",
        description="Add a hand-picked topic without collecting sources",
    )
    _configure_add(add)
    preview = subparsers.add_parser(
        "preview",
        help="Score topics against the brief without collecting or storing them",
        description="Score topics against the brief without collecting or storing them",
    )
    _configure_preview(preview)


def run_research(args: argparse.Namespace) -> int:
    """Dispatch to the selected research subcommand."""
    handler = getattr(args, "_research_handler", None)
    if handler is None:
        print("error: missing research command", file=sys.stderr)
        print("usage: aivia research {add,preview}", file=sys.stderr)
        return 2
    return int(handler(args))


def _looks_like_url(value: str) -> bool:
    parsed = urlsplit(value.strip())
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def _shared_sources(args: argparse.Namespace) -> list[str]:
    values: list[str] = []
    for source in list(getattr(args, "sources", None) or []):
        if isinstance(source, str) and source.strip():
            values.append(source.strip())
    for group in list(getattr(args, "source_lists", None) or []):
        if isinstance(group, str):
            values.extend(
                part.strip()
                for part in group.replace(";", ",").split(",")
                if part.strip()
            )
    unique: list[str] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def _read_file(path: str) -> Any:
    if path == "-":
        text = sys.stdin.read()
    else:
        text = Path(path).expanduser().read_text(encoding="utf-8")
    if not text.strip():
        return []
    return yaml.safe_load(text)


def _file_entries(path: str) -> list[Any]:
    payload = _read_file(path)
    if isinstance(payload, Mapping):
        nested = payload.get("topics")
        if nested is not None and not any(
            key in payload for key in ("title", "name", "topic", "subject")
        ):
            payload = nested
        else:
            payload = [payload]
    if isinstance(payload, str):
        payload = [payload]
    if not isinstance(payload, Sequence) or isinstance(
        payload, (str, bytes, bytearray)
    ):
        raise ConfigError("topic file must contain a list or a single topic")
    return list(payload)


def _candidate_from_file(item: Any, shared: Sequence[str]) -> TopicCandidate:
    if isinstance(item, str):
        return TopicCandidate(
            title=item,
            source_urls=list(shared),
            mode="manual",
        )
    if not isinstance(item, Mapping):
        raise ConfigError("topics in a file must be titles or mappings")
    candidate = TopicCandidate.model_validate(dict(item))
    if not candidate.source_urls and shared:
        candidate = candidate.model_copy(update={"source_urls": list(shared)})
    if not candidate.mode:
        candidate = candidate.model_copy(update={"mode": "manual"})
    return candidate


def _collect_candidates(args: argparse.Namespace) -> list[TopicCandidate]:
    shared = _shared_sources(args)
    titles: list[str] = []
    for value in list(getattr(args, "topics", None) or []):
        if not isinstance(value, str):
            raise ConfigError("topic titles must be strings")
        if _looks_like_url(value):
            if value.strip() not in shared:
                shared.append(value.strip())
        else:
            titles.append(value)
    for value in list(getattr(args, "title_options", None) or []):
        if not isinstance(value, str):
            raise ConfigError("topic titles must be strings")
        titles.append(value)

    candidates = [
        TopicCandidate(title=title, source_urls=list(shared), mode="manual")
        for title in titles
    ]
    path = getattr(args, "file", None)
    if path:
        candidates.extend(
            _candidate_from_file(item, shared) for item in _file_entries(path)
        )
    return candidates


def _profile_name(args: argparse.Namespace) -> str:
    value = getattr(args, "profile", None)
    return value.strip() if isinstance(value, str) else ""


def _store_for(args: argparse.Namespace) -> TopicStore:
    path = getattr(args, "store", None)
    if isinstance(path, str) and path.strip():
        return TopicStore(path.strip())
    config = getattr(args, "config", None)
    return TopicStore(data_dir=paths.resolve_paths(config).data_dir)


def _print_result(result: AddResult) -> None:
    decisions = list(result.verification.decisions) if result.verification else []
    for entry, decision in zip(result.stored, decisions, strict=False):
        status = "verified" if decision.verified else "unverified"
        print(f"added: {entry.candidate.title} ({status})")
    for duplicate in result.duplicates:
        print(f"rejected: {duplicate.title} ({duplicate.explanation})")
    print(
        f"added {result.added_count}, rejected {result.rejected_count}, "
        f"verified {result.verified_count}, unverified {result.unverified_count}"
    )


def run_add(args: argparse.Namespace) -> int:
    """Add, dedupe and verify hand-picked topics without collecting sources."""
    try:
        candidates = _collect_candidates(args)
        if not candidates:
            raise ConfigError("no topics to add")
        window = timedelta(days=float(getattr(args, "dedupe_window", 365)))
        result = add_topics(
            candidates,
            store=_store_for(args),
            profile=_profile_name(args),
            min_domains=int(getattr(args, "min_domains", DEFAULT_MIN_DOMAINS)),
            window=window,
            record=not bool(getattr(args, "dry_run", False)),
        )
    except (ConfigError, OSError, ValidationError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if bool(getattr(args, "json_output", False)):
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        _print_result(result)
    return 0 if result.added_count else 1


def _profiles(config: Any) -> dict[str, ContentProfileConfig]:
    section = getattr(config, "profiles", None)
    profiles = getattr(section, "profiles", None)
    return dict(profiles) if isinstance(profiles, dict) else {}


def _default_profile(
    config: Any,
    profiles: dict[str, ContentProfileConfig],
) -> str:
    publish = getattr(config, "publish", None)
    configured = getattr(publish, "default_profile", "")
    if isinstance(configured, str) and configured.strip() in profiles:
        return configured.strip()
    if "default" in profiles:
        return "default"
    return next(iter(profiles), "")


def _preview_context(
    args: argparse.Namespace,
) -> tuple[Any, str, ContentProfileConfig]:
    resolved = resolve_config(
        config_file=getattr(args, "config", None),
        environment=getattr(args, "environment", None),
        profile=getattr(args, "profile", None),
        preset=getattr(args, "preset", None),
        set_values=getattr(args, "set_values", None),
    )
    config = resolved.to_app_config()
    profiles = _profiles(config)
    selected = _profile_name(args)
    if selected:
        if profiles and selected not in profiles:
            available = ", ".join(sorted(profiles)) or "none"
            raise ConfigError(
                f"unknown profile: {selected} (available: {available})",
                details={"profile": selected, "available": list(profiles)},
            )
        return config, selected, profiles.get(selected, ContentProfileConfig())
    if profiles:
        name = _default_profile(config, profiles)
        return config, name, profiles[name]
    return config, "", ContentProfileConfig()


def _preview_derivation(
    args: argparse.Namespace,
    name: str,
    profile: ContentProfileConfig,
    brief: str,
    keywords: Sequence[str],
) -> BriefDerivation:
    cached: BriefDerivation | None = None
    if name:
        store = BriefDerivationStore(
            data_dir=paths.resolve_paths(getattr(args, "config", None)).data_dir
        )
        cached = store.get_for_brief(name, brief)
    explicit = _unique_terms(keywords) or _unique_terms(profile.keywords)
    if cached is not None:
        return cached.with_explicit_keywords(explicit)
    return _fallback_brief(profile, brief, explicit)


def _preview_candidates(
    args: argparse.Namespace,
) -> tuple[list[TopicCandidate], Path | None]:
    candidates = _collect_candidates(args)
    if candidates:
        return candidates, None
    store = _store_for(args)
    stored = store.load()
    return [entry.candidate for entry in stored], store.path


def _print_preview(result: PreviewResult) -> None:
    if result.profile:
        print(f"Profile: {result.profile}")
    if result.brief:
        print(f"Brief: {result.brief}")
    for topic in result.topics:
        print(f"Topic: {topic.title}")
        print(f"  Score: {topic.score.overall:.3f}")
        dimensions = ", ".join(
            f"{name}={value:.3f}" for name, value in topic.score.dimensions.items()
        )
        print(f"  Scores: {dimensions}")
        print(f"  Reason: {topic.score.reason}")
        parts = "; ".join(
            f"{name}: {', '.join(terms)}"
            for name, terms in topic.matched_parts.items()
            if terms
        )
        print(f"  Matched brief parts: {parts or 'none'}")
        if topic.excluded_terms:
            print(f"  Excluded terms: {', '.join(topic.excluded_terms)}")
        print(f"  Cost: ${topic.cost_usd:.6f}")
    print(f"Total cost: ${result.cost_usd:.6f}")


def run_preview(args: argparse.Namespace) -> int:
    """Score candidates against the brief without collecting or storing them."""
    try:
        config, name, profile = _preview_context(args)
        niche = getattr(args, "niche", None)
        if isinstance(niche, str):
            profile = profile.model_copy(update={"niche": niche})
        brief = getattr(args, "brief", None)
        if brief is None:
            brief = profile.brief
        elif isinstance(brief, str):
            brief = brief.strip()
        keywords = _option_keywords(args)
        derivation = _preview_derivation(
            args,
            name,
            profile,
            brief,
            keywords,
        )
        candidates, source_store = _preview_candidates(args)
        if not candidates:
            raise ConfigError("no topics to preview")
        min_score = getattr(args, "min_score", None)
        rubric_options = {} if min_score is None else {"min_score": min_score}
        rubric = ScoringRubric.from_config(
            getattr(config, "scoring", None),
            **rubric_options,
        )
        result = preview_topics(
            candidates,
            profile=profile,
            profile_name=name,
            brief=brief,
            keywords=keywords,
            derivation=derivation,
            rubric=rubric,
            config=config,
        )
        result = replace(result, source_store=source_store)
    except (ConfigError, OSError, ValidationError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if bool(getattr(args, "json_output", False)):
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        _print_preview(result)
    return 0


register_command(
    "research",
    run_research,
    help="Add hand-picked research topics with verification and dedupe",
    configure=configure,
)

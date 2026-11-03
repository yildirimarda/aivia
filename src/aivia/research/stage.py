"""The research stage: turn a topic request into selected, verified topics.

This is the deterministic control flow the rest of the pipeline uses. One call
runs the research stages in their fixed order:

    collection -> dedupe -> relevance gating -> scoring -> sensitivity
    -> verification -> selection

Each step is the stage already defined for that job; this module only chains
them, passes the candidates from one to the next and keeps every intermediate
result for auditing. Sensitivity runs after scoring and before verification, so
a blocked or held topic is never verified or selected. The output is a
:class:`TopicInput`: the selected topics a writing stage can consume plus the
collection, dedupe, gating, scoring, sensitivity, verification and selection
records that explain how they were chosen.

The stage is offline by default. Collection may call configured source
providers, but dedupe, gating, scoring, sensitivity, verification and selection
are local and deterministic; an injected scorer or embedder is the only way to
introduce a model call, and it is awaited when the caller runs the async entry
point.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, cast

from aivia.core.config.research import ResearchConfig
from aivia.core.errors import ConfigError
from aivia.core.models.topic import TopicCandidate
from aivia.providers.protocols.source import SourceRequest
from aivia.providers.sources import SourceRegistry
from aivia.research.collect import (
    DEFAULT_CONCURRENCY,
    DEFAULT_SOURCE_TIMEOUT,
    CollectionResult,
    collect_sources,
)
from aivia.research.dedupe import (
    DEFAULT_DEDUPE_WINDOW,
    DEFAULT_EMBEDDING_THRESHOLD,
    DedupeHistory,
    DedupeResult,
    _candidate_vector,
    _embed_candidates_async,
    _record_seen,
    dedupe_candidates,
    dedupe_candidates_async,
)
from aivia.research.modes.niche import DEFAULT_MIN_RELEVANCE, RelevanceGate
from aivia.research.modes.otd import SeenTopic, _resolve_now
from aivia.research.scoring import (
    CandidateScore,
    ScoringResult,
    score_candidates,
    score_candidates_async,
)
from aivia.research.select import SelectionResult, select_topics
from aivia.research.sensitivity import (
    DEFAULT_SENSITIVITY_RULES,
    SensitivityResult,
    SensitivityStage,
)
from aivia.research.verify import (
    DEFAULT_MIN_DOMAINS,
    VerificationResult,
    verify_candidates,
)

__all__ = [
    "DEFAULT_REQUIRE_VERIFIED",
    "GateDecision",
    "GateResult",
    "GatingDecision",
    "GatingResult",
    "Research",
    "ResearchInput",
    "ResearchOutput",
    "ResearchPipeline",
    "ResearchResult",
    "ResearchStage",
    "ResearchStageResult",
    "TopicInput",
    "TopicResearchStage",
    "build_research_stage",
    "research",
    "research_stage",
    "research_topics",
    "run_research_stage",
]

DEFAULT_REQUIRE_VERIFIED = True
"""Only topics with enough independent source domains are selected by default."""

_UNSET: Any = object()


@dataclass(frozen=True, slots=True)
class GatingDecision:
    """One candidate's relevance score and whether it cleared the gate."""

    candidate: TopicCandidate
    score: float
    threshold: float | None = None
    passed: bool = True
    reason: str = ""

    @property
    def title(self) -> str:
        """The candidate title."""
        return self.candidate.title

    @property
    def relevance(self) -> float:
        """Alias for :attr:`score`."""
        return self.score

    @property
    def rejected(self) -> bool:
        """Whether the candidate was dropped by the gate."""
        return not self.passed

    @property
    def dropped(self) -> bool:
        """Alias for :attr:`rejected`."""
        return self.rejected

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the decision."""
        return {
            "title": self.title,
            "score": self.score,
            "threshold": self.threshold,
            "passed": self.passed,
            "reason": self.reason,
            "candidate": self.candidate.model_dump(mode="json"),
        }


@dataclass(frozen=True, slots=True)
class GatingResult:
    """Every relevance decision plus the candidates that passed the gate."""

    decisions: tuple[GatingDecision, ...] = ()
    min_relevance: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed_decisions(self) -> list[GatingDecision]:
        """The decisions that accepted a candidate."""
        return [decision for decision in self.decisions if decision.passed]

    @property
    def rejected_decisions(self) -> list[GatingDecision]:
        """The decisions that dropped a candidate."""
        return [decision for decision in self.decisions if not decision.passed]

    @property
    def candidates(self) -> list[TopicCandidate]:
        """The candidates that passed the gate, in input order."""
        return [decision.candidate for decision in self.passed_decisions]

    @property
    def kept(self) -> list[TopicCandidate]:
        """Alias for :attr:`candidates`."""
        return self.candidates

    @property
    def topics(self) -> list[TopicCandidate]:
        """Alias for :attr:`candidates`."""
        return self.candidates

    @property
    def rejected(self) -> list[TopicCandidate]:
        """The candidates the gate dropped, in input order."""
        return [decision.candidate for decision in self.rejected_decisions]

    @property
    def dropped(self) -> list[TopicCandidate]:
        """Alias for :attr:`rejected`."""
        return self.rejected

    @property
    def titles(self) -> list[str]:
        """The titles of the candidates that passed."""
        return [decision.title for decision in self.passed_decisions]

    @property
    def rejected_titles(self) -> list[str]:
        """The titles of the candidates that were dropped."""
        return [decision.title for decision in self.rejected_decisions]

    @property
    def count(self) -> int:
        """The number of candidates that passed the gate."""
        return len(self.passed_decisions)

    @property
    def input_count(self) -> int:
        """The number of candidates the gate inspected."""
        return len(self.decisions)

    @property
    def rejected_count(self) -> int:
        """The number of candidates the gate dropped."""
        return len(self.rejected_decisions)

    @property
    def dropped_count(self) -> int:
        """Alias for :attr:`rejected_count`."""
        return self.rejected_count

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the gating stage."""
        return {
            "decisions": [decision.to_dict() for decision in self.decisions],
            "passed": [decision.title for decision in self.passed_decisions],
            "rejected": self.rejected_titles,
            "min_relevance": self.min_relevance,
            "metadata": dict(self.metadata),
        }

    def __iter__(self) -> Iterator[TopicCandidate]:
        return iter(self.candidates)

    def __len__(self) -> int:
        return len(self.candidates)


GateDecision = GatingDecision
"""Alias for :class:`GatingDecision`."""

GateResult = GatingResult
"""Alias for :class:`GatingResult`."""


@dataclass(frozen=True, slots=True)
class TopicInput:
    """The selected topics and every stage result that produced them.

    ``selection`` is the authoritative list a writing stage consumes; the other
    fields record what each research stage saw so a decision can be explained
    after the fact. ``metadata`` carries the run counts and the collection
    failure report summary.
    """

    request: SourceRequest = field(default_factory=SourceRequest)
    profile: str = ""
    brief: str = ""
    collection: CollectionResult | None = None
    dedupe: DedupeResult | None = None
    gating: GatingResult | None = None
    scoring: ScoringResult | None = None
    sensitivity: SensitivityResult | None = None
    verification: VerificationResult | None = None
    selection: SelectionResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def selected(self) -> list[TopicCandidate]:
        """The topics selected for the run, in selection order."""
        return [] if self.selection is None else self.selection.candidates

    @property
    def topics(self) -> list[TopicCandidate]:
        """Alias for :attr:`selected`."""
        return self.selected

    @property
    def selected_titles(self) -> list[str]:
        """The titles of the selected topics."""
        return [topic.title for topic in self.selected]

    @property
    def titles(self) -> list[str]:
        """Alias for :attr:`selected_titles`."""
        return self.selected_titles

    @property
    def candidates(self) -> list[TopicCandidate]:
        """Every candidate collection returned, before any filtering."""
        return [] if self.collection is None else self.collection.candidates

    @property
    def deduped(self) -> list[TopicCandidate]:
        """The candidates that survived dedupe."""
        return [] if self.dedupe is None else self.dedupe.candidates

    @property
    def gated(self) -> list[TopicCandidate]:
        """The candidates that survived relevance gating."""
        return [] if self.gating is None else self.gating.candidates

    @property
    def scored(self) -> list[TopicCandidate]:
        """The candidates that cleared the scoring rubric."""
        return [] if self.scoring is None else self.scoring.candidates

    @property
    def screened(self) -> list[TopicCandidate]:
        """The candidates allowed past the sensitivity filter."""
        return [] if self.sensitivity is None else self.sensitivity.candidates

    @property
    def allowed(self) -> list[TopicCandidate]:
        """Alias for :attr:`screened`."""
        return self.screened

    @property
    def sensitivity_candidates(self) -> list[TopicCandidate]:
        """Alias for :attr:`screened`."""
        return self.screened

    @property
    def blocked(self) -> list[TopicCandidate]:
        """The candidates the sensitivity filter blocked."""
        return [] if self.sensitivity is None else self.sensitivity.blocked_candidates

    @property
    def review(self) -> list[TopicCandidate]:
        """The candidates the sensitivity filter held for manual review."""
        return [] if self.sensitivity is None else self.sensitivity.review_candidates

    @property
    def manual_review(self) -> list[TopicCandidate]:
        """Alias for :attr:`review`."""
        return self.review

    @property
    def blocked_titles(self) -> list[str]:
        """The titles of the candidates the sensitivity filter blocked."""
        return [candidate.title for candidate in self.blocked]

    @property
    def review_titles(self) -> list[str]:
        """The titles of the candidates held for manual review."""
        return [candidate.title for candidate in self.review]

    @property
    def manual_review_titles(self) -> list[str]:
        """Alias for :attr:`review_titles`."""
        return self.review_titles

    @property
    def verified(self) -> list[TopicCandidate]:
        """The candidates with enough independent source domains."""
        if self.verification is None:
            return []
        return cast("list[TopicCandidate]", self.verification.verified)

    @property
    def unverified(self) -> list[TopicCandidate]:
        """The candidates that failed verification."""
        if self.verification is None:
            return []
        return cast("list[TopicCandidate]", self.verification.unverified)

    @property
    def scores(self) -> list[CandidateScore]:
        """The scores attached to the scored candidates."""
        return [] if self.scoring is None else list(self.scoring.scores)

    @property
    def failures(self) -> list[Any]:
        """The source failures collection recorded."""
        return [] if self.collection is None else list(self.collection.failures)

    @property
    def failed_sources(self) -> list[str]:
        """The names of the sources that failed during collection."""
        return [] if self.collection is None else self.collection.failed_sources

    @property
    def complete(self) -> bool:
        """Whether every attempted source succeeded."""
        return True if self.collection is None else self.collection.complete

    @property
    def ok(self) -> bool:
        """Alias for :attr:`complete`."""
        return self.complete

    @property
    def count(self) -> int:
        """The number of selected topics."""
        return len(self.selected)

    @property
    def selected_count(self) -> int:
        """Alias for :attr:`count`."""
        return self.count

    @property
    def input_count(self) -> int:
        """The number of candidates collection returned."""
        return len(self.candidates)

    @property
    def rejected_count(self) -> int:
        """The number of candidates not selected."""
        return 0 if self.selection is None else self.selection.rejected_count

    @property
    def category_counts(self) -> dict[str, int]:
        """The number of selected topics in each category."""
        return {} if self.selection is None else self.selection.category_counts

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the whole research stage."""
        return {
            "request": {
                "query": self.request.query,
                "language": self.request.language,
                "limit": self.request.limit,
                "mode": self.request.mode,
            },
            "profile": self.profile,
            "brief": self.brief,
            "collection": (
                None if self.collection is None else self.collection.to_dict()
            ),
            "dedupe": None if self.dedupe is None else self.dedupe.to_dict(),
            "gating": None if self.gating is None else self.gating.to_dict(),
            "scoring": None if self.scoring is None else self.scoring.to_dict(),
            "sensitivity": (
                None if self.sensitivity is None else self.sensitivity.to_dict()
            ),
            "verification": (
                None if self.verification is None else self.verification.to_dict()
            ),
            "selection": None if self.selection is None else self.selection.to_dict(),
            "selected": [
                candidate.model_dump(mode="json") for candidate in self.selected
            ],
            "titles": self.selected_titles,
            "blocked": self.blocked_titles,
            "manual_review": self.review_titles,
            "counts": {
                "candidates": self.input_count,
                "deduped": len(self.deduped),
                "gated": len(self.gated),
                "scored": len(self.scored),
                "screened": len(self.screened),
                "verified": len(self.verified),
                "selected": self.selected_count,
            },
            "failed_sources": self.failed_sources,
            "complete": self.complete,
            "metadata": dict(self.metadata),
        }

    def __iter__(self) -> Iterator[TopicCandidate]:
        return iter(self.selected)

    def __len__(self) -> int:
        return len(self.selected)

    def __getitem__(
        self,
        index: int | slice,
    ) -> TopicCandidate | list[TopicCandidate]:
        return self.selected[index]


ResearchResult = TopicInput
ResearchInput = TopicInput
ResearchOutput = TopicInput
ResearchStageResult = TopicInput


def _validate_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{label} must be a boolean")
    return value


def _validate_name(value: Any) -> str:
    if not isinstance(value, str):
        raise ConfigError("stage name must be a string")
    return value.strip() or "research"


def _coerce_request(value: Any) -> SourceRequest:
    if value is None:
        return SourceRequest()
    if isinstance(value, SourceRequest):
        return value
    if isinstance(value, str):
        return SourceRequest(query=value)
    if isinstance(value, Mapping):
        return SourceRequest.model_validate(dict(value))
    raise TypeError("request must be a SourceRequest, a string or a mapping")


def _gate_threshold(gate: Any, fallback: float | None) -> float | None:
    value = getattr(gate, "min_relevance", fallback)
    if value is None:
        return None
    return float(value)


def _gate_passes(gate: Any, score: float, threshold: float | None) -> bool:
    method = getattr(gate, "passes_score", None)
    if callable(method):
        return bool(method(score))
    return threshold is None or score >= threshold


def _score_gate_sync(gate: Any, candidate: TopicCandidate) -> float:
    method = getattr(gate, "score", None)
    if not callable(method):
        method = gate
    if not callable(method):
        raise ConfigError("gate must be callable or expose score()")
    return float(method(candidate))


async def _score_gate_async(gate: Any, candidate: TopicCandidate) -> float:
    method = getattr(gate, "score_async", None)
    if callable(method):
        return float(await method(candidate))
    method = getattr(gate, "score", None)
    if not callable(method):
        method = gate
    if not callable(method):
        raise ConfigError("gate must be callable or expose score()")
    outcome = method(candidate)
    if hasattr(outcome, "__await__"):
        outcome = await outcome
    return float(outcome)


def _gating_sync(
    gate: Any,
    candidates: Sequence[TopicCandidate],
    *,
    threshold: float | None,
) -> GatingResult:
    decisions: list[GatingDecision] = []
    for candidate in candidates:
        score = _score_gate_sync(gate, candidate)
        passed = _gate_passes(gate, score, threshold)
        decisions.append(
            GatingDecision(
                candidate=candidate,
                score=score,
                threshold=threshold,
                passed=passed,
                reason="relevant" if passed else "below minimum relevance",
            )
        )
    return _gating_result(decisions, threshold)


async def _gating_async(
    gate: Any,
    candidates: Sequence[TopicCandidate],
    *,
    threshold: float | None,
) -> GatingResult:
    decisions: list[GatingDecision] = []
    for candidate in candidates:
        score = await _score_gate_async(gate, candidate)
        passed = _gate_passes(gate, score, threshold)
        decisions.append(
            GatingDecision(
                candidate=candidate,
                score=score,
                threshold=threshold,
                passed=passed,
                reason="relevant" if passed else "below minimum relevance",
            )
        )
    return _gating_result(decisions, threshold)


def _gating_result(
    decisions: Sequence[GatingDecision],
    threshold: float | None,
) -> GatingResult:
    return GatingResult(
        decisions=tuple(decisions),
        min_relevance=threshold,
        metadata={
            "min_relevance": threshold,
            "input_count": len(decisions),
            "kept": sum(1 for decision in decisions if decision.passed),
            "dropped": sum(1 for decision in decisions if not decision.passed),
        },
    )


def _rule_name(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("name", "id", "rule", "rule_id", "label"):
            if key in value:
                return str(value[key]).strip().casefold()
        return ""
    for key in ("name", "id", "rule", "rule_id", "label"):
        name = getattr(value, key, None)
        if name is not None:
            return str(name).strip().casefold()
    return ""


def _without_disabled_rules(rules: Any, disabled: Any) -> Any:
    names = {str(name).strip().casefold() for name in disabled if str(name).strip()}
    if not names:
        return rules
    if isinstance(rules, (str, bytes, bytearray)) or not isinstance(rules, Sequence):
        return rules
    return [rule for rule in rules if _rule_name(rule) not in names]


class ResearchStage:
    """Chain collection, dedupe, gating, scoring, verification and selection.

    ``sources`` and ``request`` may be supplied to the constructor or to a
    call. Sources accept everything :func:`~aivia.research.collect.collect_sources`
    accepts. ``derivation`` supplies the relevance terms; an explicit ``gate``
    replaces it. The scoring options are forwarded to
    :func:`~aivia.research.scoring.score_candidates`; when no scorer is given a
    candidate's own dimension scores or ``score`` field are used.

    By default only verified topics are eligible for selection. Pass
    ``require_verified=False`` to let single-source topics compete. The stage
    records only the selected topics in its history so a second run does not
    repeat them while topics dropped by gating or scoring stay eligible; pass
    ``record=False`` or a per-call history to change that.
    """

    def __init__(
        self,
        sources: Any = None,
        *,
        request: SourceRequest | Mapping[str, Any] | str | None = None,
        profile: str = "",
        brief: str = "",
        derivation: Any = None,
        gate: Any = None,
        include_terms: Any = None,
        exclude_terms: Any = None,
        queries: Any = None,
        subtopics: Any = None,
        keywords: Any = None,
        niche: str | None = None,
        min_relevance: float | None = DEFAULT_MIN_RELEVANCE,
        relevance: Any = None,
        rubric: Any = None,
        config: Any = None,
        scorer: Any = None,
        dimension_scorer: Any = None,
        scores: Any = None,
        min_score: Any = _UNSET,
        min_dimension_score: Any = _UNSET,
        min_domains: int = DEFAULT_MIN_DOMAINS,
        require_verified: bool = DEFAULT_REQUIRE_VERIFIED,
        sensitivity: Any = None,
        sensitivity_rules: Any = None,
        sensitivity_blocked_terms: Any = None,
        sensitivity_review_terms: Any = None,
        sensitivity_disabled_rules: Any = None,
        sensitivity_enabled: bool | None = None,
        limit: int | None = None,
        max_per_category: int | None = 1,
        history: Any = None,
        window: timedelta | int | float | None = DEFAULT_DEDUPE_WINDOW,
        embedding_threshold: float | None = DEFAULT_EMBEDDING_THRESHOLD,
        title_threshold: float | None = None,
        embeddings: Mapping[str, Any] | None = None,
        embedder: Any = None,
        across_profiles: bool = True,
        record: bool = True,
        mode: str = "",
        now: datetime | Callable[[], datetime] | None = None,
        timeout: float | None = DEFAULT_SOURCE_TIMEOUT,
        timeouts: Mapping[str, float] | None = None,
        concurrency: int = DEFAULT_CONCURRENCY,
        continue_on_error: bool = True,
        registry: SourceRegistry | None = None,
        name: str = "research",
    ) -> None:
        self._sources = sources
        self._request = None if request is None else _coerce_request(request)
        self._profile = profile if isinstance(profile, str) else str(profile)
        self._brief = brief if isinstance(brief, str) else str(brief)
        self._derivation = derivation
        self._gate = gate
        self._include_terms = include_terms
        self._exclude_terms = exclude_terms
        self._queries = queries
        self._subtopics = subtopics
        self._keywords = keywords
        self._niche = niche
        self._min_relevance = min_relevance
        self._relevance = relevance
        self._rubric = rubric
        self._config = config
        self._scorer = scorer if scorer is not None else dimension_scorer
        self._scores = scores
        self._min_score = min_score
        self._min_dimension_score = min_dimension_score
        self._min_domains = min_domains
        self._require_verified = _validate_bool(require_verified, "require verified")
        self._sensitivity = sensitivity
        self._sensitivity_rules = sensitivity_rules
        self._sensitivity_blocked_terms = sensitivity_blocked_terms
        self._sensitivity_review_terms = sensitivity_review_terms
        self._sensitivity_disabled_rules = sensitivity_disabled_rules
        self._sensitivity_enabled = sensitivity_enabled
        self._limit = limit
        self._max_per_category = max_per_category
        self._history = history if history is not None else DedupeHistory(now=now)
        self._window = window
        self._embedding_threshold = embedding_threshold
        self._title_threshold = title_threshold
        self._embeddings = embeddings
        self._embedder = embedder
        self._across_profiles = _validate_bool(across_profiles, "across profiles")
        self._record = _validate_bool(record, "record")
        self._mode = mode if isinstance(mode, str) else str(mode)
        self._now = now
        self._timeout = timeout
        self._timeouts = dict(timeouts) if timeouts is not None else None
        self._concurrency = concurrency
        self._continue_on_error = _validate_bool(
            continue_on_error,
            "continue on error",
        )
        self._registry = registry
        self._name = _validate_name(name)

    @property
    def name(self) -> str:
        """The stage name used in reports."""
        return self._name

    @property
    def sources(self) -> Any:
        """The sources configured on this stage, if any."""
        return self._sources

    @property
    def request(self) -> SourceRequest | None:
        """The request configured on this stage, if any."""
        return self._request

    @property
    def profile(self) -> str:
        """The profile name recorded on the result."""
        return self._profile

    @property
    def brief(self) -> str:
        """The brief text recorded on the result."""
        return self._brief

    @property
    def derivation(self) -> Any:
        """The brief derivation that supplies relevance terms."""
        return self._derivation

    @property
    def gate(self) -> Any:
        """The explicit relevance gate, if any."""
        return self._gate

    @property
    def history(self) -> Any:
        """The dedupe history this stage reads and writes."""
        return self._history

    @property
    def min_domains(self) -> int:
        """The number of independent domains a topic needs."""
        return self._min_domains

    @property
    def require_verified(self) -> bool:
        """Whether only verified topics are eligible for selection."""
        return self._require_verified

    @property
    def sensitivity(self) -> Any:
        """The sensitivity filter configured on this stage, if any."""
        return self._sensitivity

    @property
    def limit(self) -> int | None:
        """The per-run topic limit."""
        return self._limit

    @property
    def max_per_category(self) -> int | None:
        """The per-category diversity limit."""
        return self._max_per_category

    def _request_for(self, value: Any = None) -> SourceRequest:
        if value is not None:
            return _coerce_request(value)
        if self._request is not None:
            return self._request
        return SourceRequest(query=self._brief)

    def _research_config(self) -> ResearchConfig | None:
        if self._config is None:
            return None
        try:
            return ResearchConfig.from_config(self._config)
        except (AttributeError, ConfigError, TypeError, ValueError):
            return None

    def _sensitivity_for(self) -> SensitivityStage:
        source = self._sensitivity
        options: dict[str, Any] = {}
        base_rules: Any = None
        if isinstance(source, SensitivityStage):
            if (
                self._sensitivity_rules is None
                and self._sensitivity_blocked_terms is None
                and self._sensitivity_review_terms is None
                and self._sensitivity_disabled_rules is None
                and self._sensitivity_enabled is None
            ):
                return source
            base_rules = source.rules
            options["enabled"] = source.enabled
        elif isinstance(source, Mapping):
            options = dict(source)
            base_rules = options.pop("rules", options.pop("rule", None))
        elif source is not None:
            base_rules = source

        if self._sensitivity_rules is not None:
            base_rules = self._sensitivity_rules
        blocked = self._sensitivity_blocked_terms
        review = self._sensitivity_review_terms
        disabled = self._sensitivity_disabled_rules
        enabled = self._sensitivity_enabled
        if blocked is None:
            blocked = options.get("blocked_terms")
        if review is None:
            review = options.get("review_terms")
        if disabled is None:
            disabled = options.get("disabled_rules")
        if enabled is None:
            enabled = options.get("enabled")

        if base_rules is None:
            config = self._research_config()
            if config is not None and source is None:
                config_options = config.sensitivity_stage_options(
                    DEFAULT_SENSITIVITY_RULES
                )
                base_rules = config_options["rules"]
                if enabled is None:
                    enabled = config_options["enabled"]
                if blocked is None:
                    blocked = config_options["blocked_terms"]
                if review is None:
                    review = config_options["review_terms"]
            else:
                base_rules = DEFAULT_SENSITIVITY_RULES

        if disabled:
            base_rules = _without_disabled_rules(base_rules, disabled)

        return SensitivityStage(
            base_rules,
            blocked_terms=blocked,
            review_terms=review,
            enabled=True if enabled is None else bool(enabled),
        )

    def _gate_for(self, derivation: Any = None) -> Any:
        if self._gate is not None:
            return self._gate
        active = self._derivation if derivation is None else derivation
        options: dict[str, Any] = {}
        for key, value in (
            ("include_terms", self._include_terms),
            ("exclude_terms", self._exclude_terms),
            ("queries", self._queries),
            ("subtopics", self._subtopics),
            ("keywords", self._keywords),
            ("niche", self._niche),
            ("relevance", self._relevance),
        ):
            if value is not None:
                options[key] = value
        if active is not None:
            return RelevanceGate.from_derivation(
                active,
                min_relevance=self._min_relevance,
                **options,
            )
        return RelevanceGate(min_relevance=self._min_relevance, **options)

    def _scoring_options(self, derivation: Any = None) -> dict[str, Any]:
        options: dict[str, Any] = {
            "scores": self._scores,
            "scorer": self._scorer,
            "rubric": self._rubric,
            "config": self._config,
            "derivation": self._derivation if derivation is None else derivation,
        }
        if self._min_score is not _UNSET:
            options["min_score"] = self._min_score
        if self._min_dimension_score is not _UNSET:
            options["min_dimension_score"] = self._min_dimension_score
        return options

    def _dedupe(
        self,
        candidates: Sequence[TopicCandidate],
        *,
        profile: str,
        now: datetime | Callable[[], datetime] | None,
        record: bool | None,
        history: Any = None,
    ) -> DedupeResult:
        return dedupe_candidates(
            candidates,
            history=self._history if history is None else history,
            window=self._window,
            embedding_threshold=self._embedding_threshold,
            title_threshold=self._title_threshold,
            embeddings=self._embeddings,
            embedder=self._embedder,
            profile=profile,
            across_profiles=self._across_profiles,
            now=self._now if now is None else now,
            record=False,
            mode=self._mode,
        )

    async def _dedupe_async(
        self,
        candidates: Sequence[TopicCandidate],
        *,
        profile: str,
        now: datetime | Callable[[], datetime] | None,
        record: bool | None,
        history: Any = None,
    ) -> DedupeResult:
        return await dedupe_candidates_async(
            candidates,
            embedder=self._embedder,
            embeddings=self._embeddings,
            history=self._history if history is None else history,
            window=self._window,
            embedding_threshold=self._embedding_threshold,
            title_threshold=self._title_threshold,
            profile=profile,
            across_profiles=self._across_profiles,
            now=self._now if now is None else now,
            record=False,
            mode=self._mode,
        )

    def _should_record(self, record: bool | None) -> bool:
        return self._record if record is None else _validate_bool(record, "record")

    def _selected_records(
        self,
        selected: Sequence[TopicCandidate],
        *,
        profile: str,
        now: datetime | Callable[[], datetime] | None,
        embeddings: Mapping[str, Any] | None = None,
        embedder: Any = _UNSET,
    ) -> list[SeenTopic]:
        current = _resolve_now(self._now if now is None else now)
        vectors = self._embeddings if embeddings is None else embeddings
        active_embedder = self._embedder if embedder is _UNSET else embedder
        return [
            SeenTopic(
                title=candidate.title,
                seen_at=current,
                mode=self._mode or "research",
                profile=profile,
                embedding=_candidate_vector(
                    candidate,
                    vectors,
                    active_embedder,
                )
                or (),
            )
            for candidate in selected
        ]

    def _record_selected(
        self,
        selected: Sequence[TopicCandidate],
        *,
        profile: str,
        now: datetime | Callable[[], datetime] | None,
        record: bool | None,
        history: Any = None,
    ) -> int:
        if not self._should_record(record) or not selected:
            return 0
        records = self._selected_records(selected, profile=profile, now=now)
        _record_seen(self._history if history is None else history, records)
        return len(records)

    async def _record_selected_async(
        self,
        selected: Sequence[TopicCandidate],
        *,
        profile: str,
        now: datetime | Callable[[], datetime] | None,
        record: bool | None,
        history: Any = None,
    ) -> int:
        if not self._should_record(record) or not selected:
            return 0
        vectors = dict(self._embeddings or {})
        produced = await _embed_candidates_async(selected, self._embedder)
        for title, vector in produced.items():
            vectors.setdefault(title, vector)
        records = self._selected_records(
            selected,
            profile=profile,
            now=now,
            embeddings=vectors,
            embedder=None,
        )
        _record_seen(self._history if history is None else history, records)
        return len(records)

    def _assemble(
        self,
        *,
        request: SourceRequest,
        profile: str,
        brief: str,
        collection: CollectionResult,
        dedupe: DedupeResult,
        gating: GatingResult,
        scoring: ScoringResult,
        sensitivity: SensitivityResult,
    ) -> TopicInput:
        verification = verify_candidates(
            sensitivity.candidates,
            min_domains=self._min_domains,
        )
        allowed_ids = {id(candidate) for candidate in sensitivity.candidates}
        scores = tuple(
            score for score in scoring.scores if id(score.candidate) in allowed_ids
        )
        if self._require_verified:
            scores = tuple(score for score in scores if score.candidate.verified)
        selection_source = ScoringResult(
            scores=scores,
            rubric=scoring.rubric,
            metadata=dict(scoring.metadata),
        )
        selection = select_topics(
            selection_source,
            limit=self._limit,
            max_per_category=self._max_per_category,
        )
        metadata: dict[str, Any] = {
            "query": request.query,
            "profile": profile,
            "brief": brief,
            "sources": {
                "attempted": collection.attempted,
                "succeeded": len(collection.results),
                "failed": len(collection.failures),
            },
            "failed_sources": collection.failed_sources,
            "complete": collection.complete,
            "counts": {
                "candidates": len(collection.candidates),
                "deduped": len(dedupe.candidates),
                "gated": len(gating.candidates),
                "scored": len(scoring.candidates),
                "screened": len(sensitivity.candidates),
                "verified": len(verification.verified),
                "selected": len(selection.candidates),
            },
            "sensitivity": {
                "input_count": sensitivity.input_count,
                "allowed": len(sensitivity.candidates),
                "blocked": len(sensitivity.blocked_candidates),
                "manual_review": len(sensitivity.review_candidates),
                "blocked_titles": sensitivity.blocked_titles,
                "review_titles": sensitivity.review_titles,
            },
            "require_verified": self._require_verified,
            "min_domains": self._min_domains,
        }
        return TopicInput(
            request=request,
            profile=profile,
            brief=brief,
            collection=collection,
            dedupe=dedupe,
            gating=gating,
            scoring=scoring,
            sensitivity=sensitivity,
            verification=verification,
            selection=selection,
            metadata=metadata,
        )

    def run_collected(
        self,
        collection: CollectionResult,
        *,
        request: SourceRequest | Mapping[str, Any] | str | None = None,
        profile: str | None = None,
        brief: str | None = None,
        derivation: Any = None,
        now: datetime | Callable[[], datetime] | None = None,
        record: bool | None = None,
        history: Any = None,
    ) -> TopicInput:
        """Run the stages after collection on an existing collection result."""
        if not isinstance(collection, CollectionResult):
            raise TypeError("collection must be a CollectionResult")
        active_request = self._request_for(request)
        active_profile = self._profile if profile is None else profile
        active_brief = self._brief if brief is None else brief
        dedupe = self._dedupe(
            collection.candidates,
            profile=active_profile,
            now=now,
            record=record,
            history=history,
        )
        gate = self._gate_for(derivation)
        threshold = _gate_threshold(gate, self._min_relevance)
        gating = _gating_sync(gate, dedupe.candidates, threshold=threshold)
        scoring = score_candidates(
            gating.candidates,
            **self._scoring_options(derivation),
        )
        sensitivity = self._sensitivity_for().run(scoring.candidates)
        result = self._assemble(
            request=active_request,
            profile=active_profile,
            brief=active_brief,
            collection=collection,
            dedupe=dedupe,
            gating=gating,
            scoring=scoring,
            sensitivity=sensitivity,
        )
        recorded = self._record_selected(
            result.selected,
            profile=active_profile,
            now=now,
            record=record,
            history=history,
        )
        if result.dedupe is not None:
            result.dedupe.metadata["recorded"] = self._should_record(record)
            result.dedupe.metadata["recorded_count"] = recorded
        return result

    async def run_collected_async(
        self,
        collection: CollectionResult,
        *,
        request: SourceRequest | Mapping[str, Any] | str | None = None,
        profile: str | None = None,
        brief: str | None = None,
        derivation: Any = None,
        now: datetime | Callable[[], datetime] | None = None,
        record: bool | None = None,
        history: Any = None,
    ) -> TopicInput:
        """Async variant of :meth:`run_collected` for async embedders/scorers."""
        if not isinstance(collection, CollectionResult):
            raise TypeError("collection must be a CollectionResult")
        active_request = self._request_for(request)
        active_profile = self._profile if profile is None else profile
        active_brief = self._brief if brief is None else brief
        dedupe = await self._dedupe_async(
            collection.candidates,
            profile=active_profile,
            now=now,
            record=record,
            history=history,
        )
        gate = self._gate_for(derivation)
        threshold = _gate_threshold(gate, self._min_relevance)
        gating = await _gating_async(gate, dedupe.candidates, threshold=threshold)
        scoring = await score_candidates_async(
            gating.candidates,
            **self._scoring_options(derivation),
        )
        sensitivity = self._sensitivity_for().run(scoring.candidates)
        result = self._assemble(
            request=active_request,
            profile=active_profile,
            brief=active_brief,
            collection=collection,
            dedupe=dedupe,
            gating=gating,
            scoring=scoring,
            sensitivity=sensitivity,
        )
        recorded = await self._record_selected_async(
            result.selected,
            profile=active_profile,
            now=now,
            record=record,
            history=history,
        )
        if result.dedupe is not None:
            result.dedupe.metadata["recorded"] = self._should_record(record)
            result.dedupe.metadata["recorded_count"] = recorded
        return result

    async def run(
        self,
        sources: Any = None,
        request: SourceRequest | Mapping[str, Any] | str | None = None,
    ) -> TopicInput:
        """Collect from *sources* and chain every research stage."""
        active_sources = self._sources if sources is None else sources
        active_request = self._request_for(request)
        collection = await collect_sources(
            active_sources,
            active_request,
            timeout=self._timeout,
            timeouts=self._timeouts,
            concurrency=self._concurrency,
            continue_on_error=self._continue_on_error,
            registry=self._registry,
        )
        return await self.run_collected_async(collection)

    run_async = run

    def run_sync(
        self,
        sources: Any = None,
        request: SourceRequest | Mapping[str, Any] | str | None = None,
    ) -> TopicInput:
        """Synchronous wrapper around :meth:`run` for CLI and test callers."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise ConfigError(
                "run_sync cannot be called from a running event loop; await run()"
            )
        return asyncio.run(self.run(sources, request))

    __call__ = run_sync

    def to_dict(self) -> dict[str, Any]:
        """Return the stage's configuration as a JSON-serializable mapping."""
        return {
            "name": self._name,
            "profile": self._profile,
            "brief": self._brief,
            "min_domains": self._min_domains,
            "require_verified": self._require_verified,
            "limit": self._limit,
            "max_per_category": self._max_per_category,
            "record": self._record,
            "across_profiles": self._across_profiles,
        }


Research = ResearchStage
ResearchPipeline = ResearchStage
TopicResearchStage = ResearchStage
research_stage = ResearchStage
research = ResearchStage


def build_research_stage(
    sources: Any = None,
    *,
    request: SourceRequest | Mapping[str, Any] | str | None = None,
    **options: Any,
) -> ResearchStage:
    """Build a :class:`ResearchStage` with the given sources and options."""
    return ResearchStage(sources, request=request, **options)


async def run_research_stage(
    sources: Any = None,
    request: SourceRequest | Mapping[str, Any] | str | None = None,
    *,
    stage: ResearchStage | None = None,
    **options: Any,
) -> TopicInput:
    """Build a stage (unless one is supplied) and run it asynchronously."""
    active = stage if stage is not None else ResearchStage(sources, **options)
    if stage is not None and (sources is not None or options):
        raise ConfigError("pass either a stage or sources/options, not both")
    return await active.run(sources if stage is None else None, request)


def research_topics(
    sources: Any = None,
    request: SourceRequest | Mapping[str, Any] | str | None = None,
    *,
    stage: ResearchStage | None = None,
    **options: Any,
) -> TopicInput:
    """Build a stage (unless one is supplied) and run it synchronously."""
    active = stage if stage is not None else ResearchStage(sources, **options)
    if stage is not None and (sources is not None or options):
        raise ConfigError("pass either a stage or sources/options, not both")
    return active.run_sync(sources if stage is None else None, request)

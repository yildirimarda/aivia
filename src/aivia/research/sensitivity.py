"""Classify sensitive topics as blocked or needing manual review.

The sensitivity stage runs after discovery and before verification or
selection. It matches a candidate's title, summary, category and keywords
against a small, ordered set of literal rules. Each rule names an action:
``blocked`` removes the topic from the run and ``manual_review`` keeps it aside
for a human instead of sending it on automatically.

Every decision records the matched rule, its category, the matched terms and a
human-readable reason, so a blocked or held topic can always be explained after
the fact. Matching is case-insensitive, Unicode-normalised and whole-word, so a
term such as ``war`` does not match ``software``. When several rules match the
most severe action wins, with the first rule in configuration order breaking a
tie. The stage is deterministic and offline; it never calls a provider.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)
from pydantic import (
    ValidationError as PydanticValidationError,
)

from aivia.core.errors import ConfigError
from aivia.core.models.topic import TopicCandidate
from aivia.providers.protocols.source import SourceResponse

__all__ = [
    "ALLOWED",
    "BLOCKED",
    "DEFAULT_RULES",
    "DEFAULT_SENSITIVITY_RULES",
    "MANUAL_REVIEW",
    "SensitiveTopicFilter",
    "SensitivityAction",
    "SensitivityDecision",
    "SensitivityFilter",
    "SensitivityResult",
    "SensitivityRule",
    "SensitivityStage",
    "build_sensitivity_stage",
    "classify_candidate",
    "filter_sensitive",
    "filter_sensitive_topics",
    "is_sensitive",
    "screen_candidate",
    "screen_candidates",
    "sensitive_topics",
    "sensitivity_stage",
]


class SensitivityAction(StrEnum):
    """What happens to a candidate that matches a rule."""

    ALLOW = "allow"
    REVIEW = "manual_review"
    BLOCK = "blocked"

    @property
    def severity(self) -> int:
        """The relative strength of the action, used to resolve matches."""
        return _SEVERITY[self]

    @property
    def blocked(self) -> bool:
        """Whether this action removes the candidate from the run."""
        return self is SensitivityAction.BLOCK

    @property
    def needs_review(self) -> bool:
        """Whether this action holds the candidate for a human."""
        return self is SensitivityAction.REVIEW

    @classmethod
    def parse(cls, value: Any) -> SensitivityAction:
        """Coerce an action name or alias into a :class:`SensitivityAction`."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ConfigError("sensitivity action must be a string")
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        try:
            return _ACTION_ALIASES[key]
        except KeyError:
            allowed = ", ".join(action.value for action in cls)
            raise ConfigError(
                f"unknown sensitivity action: {value!r} (allowed: {allowed})"
            ) from None


MANUAL_REVIEW = SensitivityAction.REVIEW
"""Alias for :attr:`SensitivityAction.REVIEW`."""

BLOCKED = SensitivityAction.BLOCK
"""Alias for :attr:`SensitivityAction.BLOCK`."""

ALLOWED = SensitivityAction.ALLOW
"""Alias for :attr:`SensitivityAction.ALLOW`."""

_SEVERITY: dict[SensitivityAction, int] = {
    SensitivityAction.ALLOW: 0,
    SensitivityAction.REVIEW: 1,
    SensitivityAction.BLOCK: 2,
}

_ACTION_ALIASES: dict[str, SensitivityAction] = {
    "allow": SensitivityAction.ALLOW,
    "allowed": SensitivityAction.ALLOW,
    "pass": SensitivityAction.ALLOW,
    "ok": SensitivityAction.ALLOW,
    "review": SensitivityAction.REVIEW,
    "manual": SensitivityAction.REVIEW,
    "manual_review": SensitivityAction.REVIEW,
    "manualreview": SensitivityAction.REVIEW,
    "flag": SensitivityAction.REVIEW,
    "hold": SensitivityAction.REVIEW,
    "pending": SensitivityAction.REVIEW,
    "block": SensitivityAction.BLOCK,
    "blocked": SensitivityAction.BLOCK,
    "deny": SensitivityAction.BLOCK,
    "reject": SensitivityAction.BLOCK,
    "exclude": SensitivityAction.BLOCK,
}


def _normalise(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _normalise_words(value: str) -> str:
    normalised = _normalise(value)
    cleaned = "".join(
        character if character.isalnum() or character == "_" else " "
        for character in normalised
    )
    return " ".join(cleaned.split())


def _normalise_terms(value: Any, *, label: str = "sensitivity term") -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw: Iterable[Any] = value.replace(";", ",").replace("\n", ",").split(",")
    elif isinstance(value, Iterable) and not isinstance(
        value, (bytes, bytearray, Mapping)
    ):
        raw = value
    else:
        raise ConfigError(f"{label}s must be a string or an iterable of strings")
    terms: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            raise ConfigError(f"{label}s must contain only strings")
        text = _normalise_words(item)
        if not text:
            continue
        if text not in seen:
            seen.add(text)
            terms.append(text)
    return tuple(terms)


def _search_text(candidate: TopicCandidate) -> str:
    parts = [
        candidate.title,
        candidate.summary,
        candidate.category,
        *candidate.keywords,
    ]
    return _normalise_words(" ".join(part for part in parts if part))


def _matches(text: str, terms: Sequence[str]) -> tuple[str, ...]:
    padded = f" {text} "
    matched: list[str] = []
    for term in terms:
        if f" {term} " in padded:
            matched.append(term)
    return tuple(matched)


class SensitivityRule(BaseModel):
    """One named policy rule with literal terms and an action."""

    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)

    name: str = Field(
        min_length=1,
        validation_alias=AliasChoices("name", "id", "rule", "rule_id", "label"),
    )
    terms: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "terms",
            "keywords",
            "matches",
            "phrases",
            "blocked_terms",
        ),
    )
    action: SensitivityAction = Field(
        default=SensitivityAction.REVIEW,
        validation_alias=AliasChoices("action", "decision", "outcome"),
    )
    category: str = ""
    reason: str = ""
    enabled: bool = True

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        name = value.strip()
        if not name:
            raise ValueError("sensitivity rule name must not be empty")
        return name

    @field_validator("terms", mode="before")
    @classmethod
    def _clean_terms(cls, value: Any) -> list[str]:
        terms = _normalise_terms(value)
        if not terms:
            raise ValueError("sensitivity rule must have at least one term")
        return list(terms)

    @field_validator("action", mode="before")
    @classmethod
    def _clean_action(cls, value: Any) -> SensitivityAction:
        return SensitivityAction.parse(value)

    @field_validator("category", "reason", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    def matches(self, text: str) -> tuple[str, ...]:
        """Return the rule's terms found in already-normalised *text*."""
        return _matches(text, self.terms)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the rule."""
        return {
            "name": self.name,
            "terms": list(self.terms),
            "action": self.action.value,
            "category": self.category,
            "reason": self.reason,
            "enabled": self.enabled,
        }


def _rule(
    name: str,
    terms: Sequence[str],
    action: SensitivityAction | str,
    *,
    category: str = "",
    reason: str = "",
) -> SensitivityRule:
    return SensitivityRule(
        name=name,
        terms=list(terms),
        action=SensitivityAction.parse(action),
        category=category,
        reason=reason,
    )


DEFAULT_SENSITIVITY_RULES: tuple[SensitivityRule, ...] = (
    _rule(
        "child-exploitation",
        (
            "child abuse",
            "child sexual abuse",
            "child pornography",
            "csam",
            "sexualisation of children",
        ),
        SensitivityAction.BLOCK,
        category="child-safety",
        reason="Content involving the abuse or sexualisation of children is blocked.",
    ),
    _rule(
        "graphic-violence",
        (
            "beheading",
            "massacre",
            "mass shooting",
            "school shooting",
            "graphic violence",
            "gore",
        ),
        SensitivityAction.BLOCK,
        category="violence",
        reason="Graphic depictions of violence are blocked.",
    ),
    _rule(
        "hate-speech",
        (
            "ethnic cleansing",
            "genocide",
            "hate speech",
            "white supremacy",
        ),
        SensitivityAction.BLOCK,
        category="hate",
        reason="Content promoting hate or atrocity is blocked.",
    ),
    _rule(
        "self-harm",
        (
            "eating disorder",
            "self harm",
            "self-harm",
            "suicide",
            "suicidal",
        ),
        SensitivityAction.REVIEW,
        category="self-harm",
        reason="Self-harm topics need a human to check the framing and support.",
    ),
    _rule(
        "sexual-content",
        (
            "explicit sex",
            "nudity",
            "porn",
            "pornography",
            "sexual content",
        ),
        SensitivityAction.REVIEW,
        category="sexuality",
        reason="Sexual content needs a human to check it suits the platform.",
    ),
    _rule(
        "substance-use",
        (
            "drug overdose",
            "drug use",
            "fentanyl",
            "heroin",
            "opioid crisis",
        ),
        SensitivityAction.REVIEW,
        category="substances",
        reason="Substance-use topics need a human to check the framing.",
    ),
    _rule(
        "terrorism",
        (
            "bomb making",
            "extremism",
            "terrorism",
            "terrorist attack",
        ),
        SensitivityAction.REVIEW,
        category="extremism",
        reason="Extremism topics need a human to check the framing and sources.",
    ),
    _rule(
        "tragedy",
        (
            "death toll",
            "fatal accident",
            "natural disaster",
            "plane crash",
            "war crime",
        ),
        SensitivityAction.REVIEW,
        category="tragedy",
        reason="Tragedy topics need a human to check tone and sensitivity.",
    ),
)

DEFAULT_RULES = DEFAULT_SENSITIVITY_RULES
"""Alias for :data:`DEFAULT_SENSITIVITY_RULES`."""


def _coerce_rule(value: Any) -> SensitivityRule:
    if isinstance(value, SensitivityRule):
        return value
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, Mapping):
        try:
            return SensitivityRule(**dict(value))
        except PydanticValidationError as error:
            details = error.errors()
            message = details[0]["msg"] if details else "invalid sensitivity rule"
            raise ConfigError(message, details={"errors": details}) from error
    raise ConfigError("sensitivity rule must be a mapping or SensitivityRule")


def _coerce_rules(value: Any) -> tuple[SensitivityRule, ...]:
    if value is None:
        return DEFAULT_SENSITIVITY_RULES
    if isinstance(value, (SensitivityRule, Mapping)):
        candidates: Sequence[Any] = (value,)
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        candidates = list(value)
    else:
        raise ConfigError(
            "sensitivity rules must be a rule, mapping or iterable of rules"
        )
    rules: list[SensitivityRule] = []
    seen: set[str] = set()
    for candidate in candidates:
        rule = _coerce_rule(candidate)
        key = rule.name.casefold()
        if key in seen:
            raise ConfigError(
                f"duplicate sensitivity rule name: {rule.name!r}",
                details={"rule": rule.name},
            )
        seen.add(key)
        rules.append(rule)
    return tuple(rules)


def _extra_rule(
    name: str,
    terms: Any,
    action: SensitivityAction,
    category: str,
) -> SensitivityRule | None:
    cleaned = _normalise_terms(terms)
    if not cleaned:
        return None
    return _rule(name, cleaned, action, category=category)


@dataclass(frozen=True, slots=True)
class SensitivityDecision:
    """The outcome for one candidate and the rule that produced it."""

    candidate: TopicCandidate
    action: SensitivityAction = SensitivityAction.ALLOW
    rule: str = ""
    category: str = ""
    reason: str = ""
    matched_terms: tuple[str, ...] = ()
    matched_rules: tuple[str, ...] = ()

    @property
    def matched_rule(self) -> str:
        """Alias for :attr:`rule`."""
        return self.rule

    @property
    def blocked(self) -> bool:
        """Whether the candidate is blocked."""
        return self.action is SensitivityAction.BLOCK

    @property
    def review(self) -> bool:
        """Whether the candidate is held for manual review."""
        return self.action is SensitivityAction.REVIEW

    @property
    def manual_review(self) -> bool:
        """Alias for :attr:`review`."""
        return self.review

    @property
    def allowed(self) -> bool:
        """Whether the candidate may continue automatically."""
        return self.action is SensitivityAction.ALLOW

    @property
    def title(self) -> str:
        """The candidate title, for convenient reporting."""
        return self.candidate.title

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the decision."""
        return {
            "title": self.candidate.title,
            "action": self.action.value,
            "blocked": self.blocked,
            "manual_review": self.review,
            "allowed": self.allowed,
            "rule": self.rule,
            "matched_rule": self.rule,
            "category": self.category,
            "reason": self.reason,
            "matched_terms": list(self.matched_terms),
            "matched_rules": list(self.matched_rules),
            "candidate": self.candidate.model_dump(mode="json"),
        }


@dataclass(frozen=True, slots=True)
class SensitivityResult:
    """Every decision plus the candidates that may continue automatically."""

    decisions: tuple[SensitivityDecision, ...] = ()
    rules: tuple[SensitivityRule, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def allowed_decisions(self) -> list[SensitivityDecision]:
        """The decisions that let a candidate continue."""
        return [decision for decision in self.decisions if decision.allowed]

    @property
    def blocked_decisions(self) -> list[SensitivityDecision]:
        """The decisions that blocked a candidate."""
        return [decision for decision in self.decisions if decision.blocked]

    @property
    def review_decisions(self) -> list[SensitivityDecision]:
        """The decisions that held a candidate for manual review."""
        return [decision for decision in self.decisions if decision.review]

    @property
    def manual_review_decisions(self) -> list[SensitivityDecision]:
        """Alias for :attr:`review_decisions`."""
        return self.review_decisions

    @property
    def candidates(self) -> list[TopicCandidate]:
        """The candidates allowed to continue, in input order."""
        return [decision.candidate for decision in self.allowed_decisions]

    @property
    def kept(self) -> list[TopicCandidate]:
        """Alias for :attr:`candidates`."""
        return self.candidates

    @property
    def topics(self) -> list[TopicCandidate]:
        """Alias for :attr:`candidates`."""
        return self.candidates

    @property
    def allowed_candidates(self) -> list[TopicCandidate]:
        """Alias for :attr:`candidates`."""
        return self.candidates

    @property
    def blocked_candidates(self) -> list[TopicCandidate]:
        """The candidates removed from the run."""
        return [decision.candidate for decision in self.blocked_decisions]

    @property
    def blocked(self) -> list[TopicCandidate]:
        """Alias for :attr:`blocked_candidates`."""
        return self.blocked_candidates

    @property
    def review_candidates(self) -> list[TopicCandidate]:
        """The candidates held for a human."""
        return [decision.candidate for decision in self.review_decisions]

    @property
    def manual_review(self) -> list[TopicCandidate]:
        """Alias for :attr:`review_candidates`."""
        return self.review_candidates

    @property
    def manual_review_candidates(self) -> list[TopicCandidate]:
        """Alias for :attr:`review_candidates`."""
        return self.review_candidates

    @property
    def titles(self) -> list[str]:
        """The titles of the allowed candidates."""
        return [candidate.title for candidate in self.candidates]

    @property
    def allowed_titles(self) -> list[str]:
        """Alias for :attr:`titles`."""
        return self.titles

    @property
    def blocked_titles(self) -> list[str]:
        """The titles of the blocked candidates."""
        return [candidate.title for candidate in self.blocked_candidates]

    @property
    def review_titles(self) -> list[str]:
        """The titles of the candidates held for review."""
        return [candidate.title for candidate in self.review_candidates]

    @property
    def manual_review_titles(self) -> list[str]:
        """Alias for :attr:`review_titles`."""
        return self.review_titles

    @property
    def matched_rules(self) -> list[str]:
        """The primary matched rule for every non-allowed decision."""
        return [decision.rule for decision in self.decisions if not decision.allowed]

    @property
    def count(self) -> int:
        """The number of candidates allowed to continue."""
        return len(self.candidates)

    @property
    def input_count(self) -> int:
        """The number of candidates the stage received."""
        return len(self.decisions)

    @property
    def blocked_count(self) -> int:
        """The number of blocked candidates."""
        return len(self.blocked_decisions)

    @property
    def review_count(self) -> int:
        """The number of candidates held for manual review."""
        return len(self.review_decisions)

    @property
    def manual_review_count(self) -> int:
        """Alias for :attr:`review_count`."""
        return self.review_count

    def filter(self) -> list[TopicCandidate]:
        """Return the candidates allowed to continue automatically."""
        return self.candidates

    def to_response(
        self,
        *,
        provider: str = "sensitivity",
        query: str = "",
    ) -> SourceResponse:
        """Return the allowed candidates as a :class:`SourceResponse`."""
        return SourceResponse(
            provider=provider,
            query=query,
            candidates=self.candidates,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the stage."""
        return {
            "decisions": [decision.to_dict() for decision in self.decisions],
            "candidates": [
                candidate.model_dump(mode="json") for candidate in self.candidates
            ],
            "blocked": [candidate.title for candidate in self.blocked_candidates],
            "manual_review": [candidate.title for candidate in self.review_candidates],
            "metadata": dict(self.metadata),
        }

    def __iter__(self) -> Iterator[TopicCandidate]:
        return iter(self.candidates)

    def __len__(self) -> int:
        return len(self.candidates)


def _coerce_candidates(candidates: Any) -> list[TopicCandidate]:
    if candidates is None:
        return []
    if isinstance(candidates, TopicCandidate):
        return [candidates]
    if isinstance(candidates, SourceResponse):
        return list(candidates.candidates)
    if isinstance(candidates, Mapping):
        value = candidates.get("candidates", candidates.get("topics"))
        if value is not None:
            return _coerce_candidates(value)
        return [_coerce_candidate(candidates)]
    if isinstance(candidates, Iterable) and not isinstance(
        candidates, (str, bytes, bytearray)
    ):
        return [_coerce_candidate(candidate) for candidate in candidates]
    return [_coerce_candidate(candidates)]


def _coerce_candidate(candidate: Any) -> TopicCandidate:
    if isinstance(candidate, TopicCandidate):
        return candidate
    if isinstance(candidate, Mapping):
        return TopicCandidate(**dict(candidate))
    raise ConfigError("candidate must be a TopicCandidate or mapping")


def _classify(
    candidate: TopicCandidate,
    rules: Sequence[SensitivityRule],
    *,
    enabled: bool = True,
) -> SensitivityDecision:
    if not enabled:
        return SensitivityDecision(candidate=candidate)
    text = _search_text(candidate)
    matched: list[tuple[int, SensitivityRule, tuple[str, ...]]] = []
    for index, rule in enumerate(rules):
        if not rule.enabled:
            continue
        terms = rule.matches(text)
        if terms:
            matched.append((index, rule, terms))
    if not matched:
        return SensitivityDecision(candidate=candidate)

    _, primary_rule, primary_terms = max(
        matched,
        key=lambda item: (item[1].action.severity, -item[0]),
    )
    return SensitivityDecision(
        candidate=candidate,
        action=primary_rule.action,
        rule=primary_rule.name,
        category=primary_rule.category,
        reason=primary_rule.reason,
        matched_terms=primary_terms,
        matched_rules=tuple(rule.name for _, rule, _ in matched),
    )


class SensitivityStage:
    """A reusable sensitivity filter bound to one ordered rule set."""

    def __init__(
        self,
        rules: Any = None,
        *,
        blocked_terms: Any = None,
        review_terms: Any = None,
        enabled: bool = True,
        name: str = "sensitivity",
    ) -> None:
        base = list(_coerce_rules(rules))
        extra: list[SensitivityRule] = []
        block_rule = _extra_rule(
            "blocked-terms",
            blocked_terms,
            SensitivityAction.BLOCK,
            "configured",
        )
        if block_rule is not None:
            extra.append(block_rule)
        review_rule = _extra_rule(
            "review-terms",
            review_terms,
            SensitivityAction.REVIEW,
            "configured",
        )
        if review_rule is not None:
            extra.append(review_rule)
        self._rules = _coerce_rules([*base, *extra])
        self._enabled = bool(enabled)
        self._name = name.strip() or "sensitivity"

    @property
    def name(self) -> str:
        """The provider name reported by :meth:`to_response`."""
        return self._name

    @property
    def rules(self) -> tuple[SensitivityRule, ...]:
        """The ordered rules this stage applies."""
        return self._rules

    @property
    def enabled(self) -> bool:
        """Whether matching is enabled; a disabled stage allows everything."""
        return self._enabled

    def classify(self, candidate: Any) -> SensitivityDecision:
        """Classify one candidate against this stage's rules."""
        return _classify(
            _coerce_candidate(candidate),
            self._rules,
            enabled=self._enabled,
        )

    screen = classify

    def is_sensitive(self, candidate: Any) -> bool:
        """Whether *candidate* is blocked or needs manual review."""
        return not self.classify(candidate).allowed

    def run(
        self,
        candidates: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> SensitivityResult:
        """Classify every candidate and collect the decisions."""
        values = _coerce_candidates(candidates)
        decisions = tuple(self.classify(candidate) for candidate in values)
        result_metadata = {
            "enabled": self._enabled,
            "rules": len(self._rules),
            "input_count": len(decisions),
            "allowed": sum(1 for decision in decisions if decision.allowed),
            "blocked": sum(1 for decision in decisions if decision.blocked),
            "manual_review": sum(1 for decision in decisions if decision.review),
        }
        if metadata:
            result_metadata.update(metadata)
        return SensitivityResult(
            decisions=decisions,
            rules=self._rules,
            metadata=result_metadata,
        )

    screen_candidates = run
    __call__ = run

    def filter(self, candidates: Any) -> list[TopicCandidate]:
        """Return only the candidates allowed to continue automatically."""
        return self.run(candidates).candidates

    def to_response(
        self,
        candidates: Any,
        *,
        query: str = "",
    ) -> SourceResponse:
        """Classify *candidates* and return the allowed ones as a response."""
        return self.run(candidates).to_response(
            provider=self._name,
            query=query,
        )


SensitivityFilter = SensitivityStage
"""Alias for :class:`SensitivityStage`."""

SensitiveTopicFilter = SensitivityStage
"""Alias for :class:`SensitivityStage`."""

sensitivity_stage = SensitivityStage
"""Alias for :class:`SensitivityStage`."""


def build_sensitivity_stage(
    rules: Any = None,
    *,
    blocked_terms: Any = None,
    review_terms: Any = None,
    enabled: bool = True,
    name: str = "sensitivity",
) -> SensitivityStage:
    """Build a sensitivity stage from rules and optional term shortcuts."""
    return SensitivityStage(
        rules,
        blocked_terms=blocked_terms,
        review_terms=review_terms,
        enabled=enabled,
        name=name,
    )


def screen_candidate(
    candidate: Any,
    *,
    rules: Any = None,
    blocked_terms: Any = None,
    review_terms: Any = None,
    enabled: bool = True,
) -> SensitivityDecision:
    """Classify one candidate and record the matched rule."""
    return SensitivityStage(
        rules,
        blocked_terms=blocked_terms,
        review_terms=review_terms,
        enabled=enabled,
    ).classify(candidate)


def classify_candidate(
    candidate: Any,
    *,
    rules: Any = None,
    blocked_terms: Any = None,
    review_terms: Any = None,
    enabled: bool = True,
) -> SensitivityDecision:
    """Alias for :func:`screen_candidate`."""
    return screen_candidate(
        candidate,
        rules=rules,
        blocked_terms=blocked_terms,
        review_terms=review_terms,
        enabled=enabled,
    )


def screen_candidates(
    candidates: Any,
    *,
    rules: Any = None,
    blocked_terms: Any = None,
    review_terms: Any = None,
    enabled: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> SensitivityResult:
    """Classify *candidates* into allowed, blocked and manual review."""
    return SensitivityStage(
        rules,
        blocked_terms=blocked_terms,
        review_terms=review_terms,
        enabled=enabled,
    ).run(candidates, metadata=metadata)


def filter_sensitive(
    candidates: Any,
    *,
    rules: Any = None,
    blocked_terms: Any = None,
    review_terms: Any = None,
    enabled: bool = True,
) -> list[TopicCandidate]:
    """Return only the candidates that are neither blocked nor held."""
    return screen_candidates(
        candidates,
        rules=rules,
        blocked_terms=blocked_terms,
        review_terms=review_terms,
        enabled=enabled,
    ).candidates


filter_sensitive_topics = filter_sensitive
"""Alias for :func:`filter_sensitive`."""

sensitive_topics = filter_sensitive
"""Alias for :func:`filter_sensitive`."""


def is_sensitive(
    candidate: Any,
    *,
    rules: Any = None,
    blocked_terms: Any = None,
    review_terms: Any = None,
    enabled: bool = True,
) -> bool:
    """Whether *candidate* is blocked or needs manual review."""
    return not screen_candidate(
        candidate,
        rules=rules,
        blocked_terms=blocked_terms,
        review_terms=review_terms,
        enabled=enabled,
    ).allowed

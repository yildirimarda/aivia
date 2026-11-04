"""Pre-score hook candidates with a deterministic, dependency free heuristic.

Hooks are cheap to generate but expensive to judge badly. Before a model is
asked to pick one, this module scores every candidate on a small set of
properties that need no provider, model, network or third party package:

* ``length`` keeps the line short enough to land in the first few seconds;
* ``clarity`` rewards ordinary words and a single readable clause;
* ``specificity`` rewards a number, a name, a time, a place or a quotation;
* ``curiosity`` rewards a question or an open loop the script can close;
* ``naturalness`` penalises links, emoji, shouting and repeated punctuation.

Each property is scored between zero and one. The final score is deliberately
weighted by the weakest property: half of it comes from the weighted mean and
half from the lowest property score, so a candidate cannot hide one dead
property behind four strong ones. The weights, the weakest-property share and
the acceptance floor are configurable, and every score records the properties,
the weakest one and the reason it failed.

The prefilter is pure code and fully offline. It never calls a provider and
never mutates the candidates it receives.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic import (
    ValidationError as PydanticValidationError,
)

from aivia.core.config.writing import (
    DEFAULT_HOOK_IDEAL_MAX_WORDS,
    DEFAULT_HOOK_IDEAL_MIN_WORDS,
    DEFAULT_HOOK_MIN_SCORE,
    DEFAULT_HOOK_PROPERTY_WEIGHTS,
    DEFAULT_HOOK_WEAKEST_WEIGHT,
    HOOK_CLARITY_AVG_WORD_LIMIT,
    HOOK_CLARITY_AVG_WORD_PENALTY,
    HOOK_CLARITY_CLAUSE_LIMIT,
    HOOK_CLARITY_CLAUSE_PENALTY,
    HOOK_CLARITY_LONG_RATIO_PENALTY,
    HOOK_CLARITY_LONG_WORD_LENGTH,
    HOOK_CLARITY_LONG_WORD_RATIO_LIMIT,
    HOOK_CLARITY_SENTENCE_LIMIT,
    HOOK_CLARITY_SENTENCE_PENALTY,
    HOOK_CURIOSITY_BASE,
    HOOK_CURIOSITY_OPEN_LOOP_BONUS,
    HOOK_CURIOSITY_QUESTION_BONUS,
    HOOK_CURIOSITY_SURPRISE_BONUS,
    HOOK_LENGTH_LONG_PENALTY_PER_WORD,
    HOOK_LENGTH_SHORT_BASE,
    HOOK_LENGTH_SHORT_SPAN,
    HOOK_MAX_PROPERTY_SCORE,
    HOOK_NATURALNESS_ADJACENT_REPEAT_PENALTY,
    HOOK_NATURALNESS_BASE,
    HOOK_NATURALNESS_CAPS_MAX_PENALTY,
    HOOK_NATURALNESS_CAPS_PENALTY_PER_WORD,
    HOOK_NATURALNESS_EMOJI_PENALTY,
    HOOK_NATURALNESS_EXCLAMATION_LIMIT,
    HOOK_NATURALNESS_EXCLAMATION_PENALTY,
    HOOK_NATURALNESS_HASHTAG_MAX_PENALTY,
    HOOK_NATURALNESS_HASHTAG_PENALTY,
    HOOK_NATURALNESS_REPEATED_PUNCT_PENALTY,
    HOOK_NATURALNESS_URL_PENALTY,
    HOOK_PROPERTY_REASON_FLOOR,
    HOOK_SPECIFICITY_BASE,
    HOOK_SPECIFICITY_NUMBER_BONUS,
    HOOK_SPECIFICITY_PROPER_BONUS,
    HOOK_SPECIFICITY_QUOTE_BONUS,
    HOOK_SPECIFICITY_TIME_BONUS,
)
from aivia.core.errors import ConfigError
from aivia.writing.hooks import HookCandidate, HookCandidates, HookSelection

__all__ = [
    "DEFAULT_HOOK_MIN_SCORE",
    "DEFAULT_HOOK_PROPERTY_WEIGHTS",
    "DEFAULT_HOOK_WEAKEST_WEIGHT",
    "HOOK_PROPERTIES",
    "HookCandidateScore",
    "HookPreFilter",
    "HookPreFilterConfig",
    "HookPreFilterResult",
    "HookPreFilterScore",
    "HookPreScore",
    "HookPrefilter",
    "HookPrefilterConfig",
    "HookPrefilterResult",
    "HookScore",
    "best_hook_candidate",
    "choose_hook_by_prefilter",
    "filter_hook_candidates",
    "filter_hooks",
    "hook_property_scores",
    "pre_score_hook_candidate",
    "pre_score_hook_candidates",
    "prefilter_hook_candidate",
    "prefilter_hook_candidates",
    "prefilter_hooks",
    "prescore_hook_candidate",
    "prescore_hook_candidates",
    "rank_hook_candidates",
    "rank_hooks",
    "score_hook",
    "score_hook_candidate",
    "score_hook_candidates",
    "score_hooks",
    "weakest_hook_property",
]

HOOK_PROPERTIES = (
    "length",
    "clarity",
    "specificity",
    "curiosity",
    "naturalness",
)
"""Every property the heuristic scores, in display order."""

_PROPERTY_ALIASES = {
    "length": "length",
    "brevity": "length",
    "shortness": "length",
    "conciseness": "length",
    "clarity": "clarity",
    "readability": "clarity",
    "specificity": "specificity",
    "concreteness": "specificity",
    "detail": "specificity",
    "curiosity": "curiosity",
    "tension": "curiosity",
    "intrigue": "curiosity",
    "open-loop": "curiosity",
    "open-loop-tension": "curiosity",
    "naturalness": "naturalness",
    "delivery": "naturalness",
    "speakability": "naturalness",
    "cleanliness": "naturalness",
}

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_NUMBER = re.compile(r"\d")
_NUMBER_WORD = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|hundred|thousand|million|billion|half|most|few|many|several)\b",
    re.IGNORECASE,
)
_PROPER = re.compile(r"\b[A-Z][a-z]+")
_OPEN_LOOP = re.compile(
    r"\b(?:why|how|what|when|where|who|which|secret|hidden|nobody|no one|"
    r"reason|actually|truth|surprising|untold|what happened|until|before|"
    r"discover|reveals?|you won.t believe)\b",
    re.IGNORECASE,
)
_SURPRISE = re.compile(
    r"\b(?:wrong|myth|mistake|never|stop|warning|beware|but|yet|however|"
    r"surprising|unexpected|actually|secret)\b",
    re.IGNORECASE,
)
_TIME_OR_PLACE = re.compile(
    r"\b(?:today|yesterday|tomorrow|now|then|here|there|every day|"
    r"last (?:week|month|year)|years? ago|in \d{4}|\d{4})\b",
    re.IGNORECASE,
)
_QUOTED = re.compile(r"[\"'\u2018\u2019\u201c\u201d]")
_URL = re.compile(
    r"(?:https?://|www\.|\b[\w.-]+\.(?:com|org|net|io|co|dev)\b)",
    re.IGNORECASE,
)
_EMOJI = re.compile("[\U0001f000-\U0001faff\u2600-\u27bf]")
_ALLCAPS = re.compile(r"\b[A-Z]{3,}\b")
_REPEATED_PUNCT = re.compile(r"[!?.,]{2,}")
_ADJACENT_REPEAT = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)


def _normalise_property(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value).casefold()).strip("-")
    return _PROPERTY_ALIASES.get(text, "")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _finite_weight(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"hook property weight for {label} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(
            f"hook property weight for {label} must be finite and non-negative"
        )
    return number


def _normalise_weights(value: Any) -> dict[str, float]:
    if value is None:
        return dict(DEFAULT_HOOK_PROPERTY_WEIGHTS)
    if not isinstance(value, Mapping):
        raise ValueError("hook property weights must be a mapping")
    weights: dict[str, float] = {}
    for raw_name, raw_weight in value.items():
        name = _normalise_property(raw_name)
        if not name:
            raise ValueError(f"unknown hook property: {raw_name!r}")
        if name in weights:
            raise ValueError(f"duplicate hook property weight: {name!r}")
        weights[name] = _finite_weight(raw_weight, name)
    if not weights:
        raise ValueError("at least one hook property weight is required")
    if sum(weights.values()) <= 0.0:
        raise ValueError("hook property weights must not all be zero")
    return weights


class HookPrefilterConfig(BaseModel):
    """Weights, weakest-property share and acceptance floor for hook scores.

    A partial ``weights`` mapping restricts scoring to those properties, which
    is useful in a test or a profile that cares about one dimension. The
    ``weakest_weight`` is how much of the final score comes from the lowest
    property score; zero reproduces a plain weighted mean and one returns the
    weakest property alone.
    """

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    weights: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_HOOK_PROPERTY_WEIGHTS),
        validation_alias=AliasChoices(
            "weights",
            "property_weights",
            "dimension_weights",
            "hook_weights",
        ),
    )
    weakest_weight: float = Field(
        default=DEFAULT_HOOK_WEAKEST_WEIGHT,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices(
            "weakest_weight",
            "weakest_property_weight",
            "weakest_share",
        ),
    )
    min_score: float = Field(
        default=DEFAULT_HOOK_MIN_SCORE,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices(
            "min_score",
            "minimum_score",
            "threshold",
            "floor",
        ),
    )
    ideal_min_words: int = Field(
        default=DEFAULT_HOOK_IDEAL_MIN_WORDS,
        ge=1,
        validation_alias=AliasChoices(
            "ideal_min_words",
            "min_words",
            "shortest_words",
        ),
    )
    ideal_max_words: int = Field(
        default=DEFAULT_HOOK_IDEAL_MAX_WORDS,
        ge=1,
        validation_alias=AliasChoices(
            "ideal_max_words",
            "max_words",
            "longest_words",
        ),
    )

    @field_validator("weights", mode="before")
    @classmethod
    def _validate_weights(cls, value: Any) -> dict[str, float]:
        return _normalise_weights(value)

    @model_validator(mode="after")
    def _validate_bands(self) -> HookPrefilterConfig:
        if self.ideal_min_words > self.ideal_max_words:
            raise ValueError("ideal_min_words must not be greater than ideal_max_words")
        return self


def _coerce_config(
    config: HookPrefilterConfig | Mapping[str, Any] | None = None,
    *,
    weights: Mapping[str, Any] | None = None,
) -> HookPrefilterConfig:
    if config is None:
        base: dict[str, Any] = {}
    elif isinstance(config, HookPrefilterConfig):
        base = config.model_dump()
    elif isinstance(config, Mapping):
        base = dict(config)
    else:
        raise ConfigError(
            "hook prefilter config must be a mapping or HookPrefilterConfig",
            details={"config": repr(config)},
        )
    if weights is not None:
        base["weights"] = weights
    try:
        return HookPrefilterConfig.model_validate(base)
    except PydanticValidationError as error:
        raise ConfigError(
            f"invalid hook prefilter config: {error}",
            details={"config": repr(config)},
        ) from error


def _coerce_candidate(value: Any) -> HookCandidate:
    if isinstance(value, HookCandidate):
        return value
    if isinstance(value, str):
        return HookCandidate(text=value)
    if isinstance(value, Mapping):
        return HookCandidate.model_validate(dict(value))
    raise ConfigError(
        "a hook candidate must be text, a mapping or a HookCandidate",
        details={"candidate": repr(value)},
    )


def _coerce_candidates(value: Any) -> list[HookCandidate]:
    if value is None:
        return []
    if isinstance(value, (HookCandidates, HookSelection)):
        return list(value.candidates)
    if isinstance(value, (HookCandidate, str, Mapping)):
        return [_coerce_candidate(value)]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_coerce_candidate(item) for item in value]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return [_coerce_candidate(item) for item in value]
    raise ConfigError(
        "hook candidates must be a candidate or a sequence of candidates",
        details={"candidates": repr(value)},
    )


def _count_words(text: str) -> int:
    return len(_WORD.findall(text))


def _length_score(text: str, config: HookPrefilterConfig) -> float:
    words = _count_words(text)
    if words == 0:
        return 0.0
    low = config.ideal_min_words
    high = config.ideal_max_words
    if low <= words <= high:
        return HOOK_MAX_PROPERTY_SCORE
    if words < low:
        return _clamp(HOOK_LENGTH_SHORT_BASE + HOOK_LENGTH_SHORT_SPAN * words / low)
    return _clamp(
        HOOK_MAX_PROPERTY_SCORE - HOOK_LENGTH_LONG_PENALTY_PER_WORD * (words - high)
    )


def _clarity_score(text: str, config: HookPrefilterConfig) -> float:
    words = _WORD.findall(text)
    if not words:
        return 0.0
    average_length = sum(len(word) for word in words) / len(words)
    long_ratio = sum(
        1 for word in words if len(word) > HOOK_CLARITY_LONG_WORD_LENGTH
    ) / len(words)
    sentences = max(1, len(re.findall(r"[.!?]+", text)))
    clauses = text.count(",") + text.count(";")
    value = HOOK_MAX_PROPERTY_SCORE
    value -= (
        max(0.0, average_length - HOOK_CLARITY_AVG_WORD_LIMIT)
        * HOOK_CLARITY_AVG_WORD_PENALTY
    )
    value -= (
        max(0, sentences - HOOK_CLARITY_SENTENCE_LIMIT) * HOOK_CLARITY_SENTENCE_PENALTY
    )
    value -= (
        max(0.0, long_ratio - HOOK_CLARITY_LONG_WORD_RATIO_LIMIT)
        * HOOK_CLARITY_LONG_RATIO_PENALTY
    )
    value -= max(0, clauses - HOOK_CLARITY_CLAUSE_LIMIT) * HOOK_CLARITY_CLAUSE_PENALTY
    return _clamp(value)


def _specificity_score(text: str, config: HookPrefilterConfig) -> float:
    value = HOOK_SPECIFICITY_BASE
    if _NUMBER.search(text) or _NUMBER_WORD.search(text):
        value += HOOK_SPECIFICITY_NUMBER_BONUS
    if any(match.start() > 0 for match in _PROPER.finditer(text)):
        value += HOOK_SPECIFICITY_PROPER_BONUS
    if _TIME_OR_PLACE.search(text):
        value += HOOK_SPECIFICITY_TIME_BONUS
    if _QUOTED.search(text):
        value += HOOK_SPECIFICITY_QUOTE_BONUS
    return _clamp(value)


def _curiosity_score(text: str, config: HookPrefilterConfig) -> float:
    value = HOOK_CURIOSITY_BASE
    if "?" in text:
        value += HOOK_CURIOSITY_QUESTION_BONUS
    if _OPEN_LOOP.search(text):
        value += HOOK_CURIOSITY_OPEN_LOOP_BONUS
    if _SURPRISE.search(text):
        value += HOOK_CURIOSITY_SURPRISE_BONUS
    return _clamp(value)


def _naturalness_score(text: str, config: HookPrefilterConfig) -> float:
    value = HOOK_NATURALNESS_BASE
    if _URL.search(text):
        value -= HOOK_NATURALNESS_URL_PENALTY
    if _EMOJI.search(text):
        value -= HOOK_NATURALNESS_EMOJI_PENALTY
    capitals = _ALLCAPS.findall(text)
    if capitals:
        value -= min(
            HOOK_NATURALNESS_CAPS_MAX_PENALTY,
            HOOK_NATURALNESS_CAPS_PENALTY_PER_WORD * len(capitals),
        )
    if _REPEATED_PUNCT.search(text):
        value -= HOOK_NATURALNESS_REPEATED_PUNCT_PENALTY
    if _ADJACENT_REPEAT.search(text):
        value -= HOOK_NATURALNESS_ADJACENT_REPEAT_PENALTY
    value -= (
        max(0, text.count("!") - HOOK_NATURALNESS_EXCLAMATION_LIMIT)
        * HOOK_NATURALNESS_EXCLAMATION_PENALTY
    )
    value -= min(
        HOOK_NATURALNESS_HASHTAG_MAX_PENALTY,
        text.count("#") * HOOK_NATURALNESS_HASHTAG_PENALTY,
    )
    return _clamp(value)


_PROPERTY_SCORERS: dict[str, Callable[[str, HookPrefilterConfig], float]] = {
    "length": _length_score,
    "clarity": _clarity_score,
    "specificity": _specificity_score,
    "curiosity": _curiosity_score,
    "naturalness": _naturalness_score,
}


def hook_property_scores(
    candidate: Any,
    *,
    config: HookPrefilterConfig | Mapping[str, Any] | None = None,
    weights: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Return the property scores for one candidate, in display order."""
    active = _coerce_config(config, weights=weights)
    hook = _coerce_candidate(candidate)
    return {
        name: _PROPERTY_SCORERS[name](hook.text, active) for name in HOOK_PROPERTIES
    }


def _score_reasons(
    properties: Mapping[str, float],
    *,
    text: str,
    config: HookPrefilterConfig,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if properties["length"] < HOOK_PROPERTY_REASON_FLOOR:
        words = _count_words(text)
        reasons.append(
            f"length {words} words outside the ideal "
            f"{config.ideal_min_words}-{config.ideal_max_words}"
        )
    if properties["clarity"] < HOOK_PROPERTY_REASON_FLOOR:
        reasons.append("clarity is low: long words or too many clauses")
    if properties["specificity"] < HOOK_PROPERTY_REASON_FLOOR:
        reasons.append("specificity lacks a number, name, time, place or quotation")
    if properties["curiosity"] < HOOK_PROPERTY_REASON_FLOOR:
        reasons.append("curiosity has no question or open loop")
    if properties["naturalness"] < HOOK_PROPERTY_REASON_FLOOR:
        reasons.append("naturalness is low: formatting noise or shouting")
    return tuple(reasons)


@dataclass(frozen=True, slots=True)
class HookCandidateScore:
    """One candidate, its property scores and the composite pre-score."""

    index: int
    candidate: HookCandidate
    properties: dict[str, float]
    weights: dict[str, float]
    weighted_mean: float
    weakest_property: str
    weakest_score: float
    score: float
    min_score: float
    weakest_weight: float
    reasons: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        """The candidate text."""
        return self.candidate.text

    @property
    def hook(self) -> str:
        """Alias for :attr:`text`."""
        return self.text

    @property
    def formula(self) -> str:
        """The candidate formula id, when it named one."""
        return self.candidate.formula

    @property
    def passed(self) -> bool:
        """Whether the composite score reaches the configured floor."""
        return self.score >= self.min_score

    accepted = passed
    """Alias for :attr:`passed`."""

    @property
    def rejected(self) -> bool:
        """Whether the composite score is below the configured floor."""
        return not self.passed

    @property
    def weakest_link(self) -> tuple[str, float]:
        """The weakest property name and score as a pair."""
        return (self.weakest_property, self.weakest_score)

    @property
    def property_scores(self) -> dict[str, float]:
        """A copy of the per-property scores."""
        return dict(self.properties)

    @property
    def rank_key(self) -> tuple[float, float, int]:
        """Sort key: highest score, then least weak, then input order."""
        return (-self.score, self.weakest_score, self.index)

    @property
    def reason(self) -> str:
        """The failure reasons joined into one string."""
        return "; ".join(self.reasons)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation for logs and prompts."""
        return {
            "index": self.index,
            "text": self.text,
            "formula": self.formula,
            "properties": dict(self.properties),
            "weights": dict(self.weights),
            "weighted_mean": self.weighted_mean,
            "weakest_property": self.weakest_property,
            "weakest_score": self.weakest_score,
            "weakest_weight": self.weakest_weight,
            "score": self.score,
            "min_score": self.min_score,
            "passed": self.passed,
            "reasons": list(self.reasons),
        }


HookScore = HookCandidateScore
HookPreScore = HookCandidateScore
HookPrefilterScore = HookCandidateScore


@dataclass(frozen=True, slots=True)
class HookPrefilterResult:
    """Every scored candidate, the preferred order and the accepted subset."""

    scores: tuple[HookCandidateScore, ...] = ()
    config: HookPrefilterConfig = field(default_factory=HookPrefilterConfig)

    def __iter__(self) -> Iterator[HookCandidateScore]:
        return iter(self.scores)

    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(self, index: int) -> HookCandidateScore:
        return self.scores[index]

    @property
    def scored(self) -> tuple[HookCandidateScore, ...]:
        """Alias for :attr:`scores`."""
        return self.scores

    @property
    def all_scores(self) -> tuple[HookCandidateScore, ...]:
        """Alias for :attr:`scores`."""
        return self.scores

    @property
    def ranked(self) -> tuple[HookCandidateScore, ...]:
        """Every score ordered from strongest to weakest, ties by input order."""
        return tuple(sorted(self.scores, key=lambda item: item.rank_key))

    @property
    def ranked_scores(self) -> tuple[HookCandidateScore, ...]:
        """Alias for :attr:`ranked`."""
        return self.ranked

    @property
    def accepted(self) -> tuple[HookCandidateScore, ...]:
        """Every score that clears the configured floor, in input order."""
        return tuple(score for score in self.scores if score.passed)

    @property
    def passing(self) -> tuple[HookCandidateScore, ...]:
        """Alias for :attr:`accepted`."""
        return self.accepted

    @property
    def rejected(self) -> tuple[HookCandidateScore, ...]:
        """Every score below the configured floor, in input order."""
        return tuple(score for score in self.scores if not score.passed)

    @property
    def dropped(self) -> tuple[HookCandidateScore, ...]:
        """Alias for :attr:`rejected`."""
        return self.rejected

    @property
    def candidates(self) -> tuple[HookCandidate, ...]:
        """Every candidate, in input order."""
        return tuple(score.candidate for score in self.scores)

    @property
    def accepted_candidates(self) -> tuple[HookCandidate, ...]:
        """Every candidate that clears the floor, in input order."""
        return tuple(score.candidate for score in self.accepted)

    @property
    def rejected_candidates(self) -> tuple[HookCandidate, ...]:
        """Every candidate below the floor, in input order."""
        return tuple(score.candidate for score in self.rejected)

    @property
    def best(self) -> HookCandidateScore | None:
        """The highest scoring candidate, or ``None`` when there are none."""
        ranked = self.ranked
        return ranked[0] if ranked else None

    @property
    def winner(self) -> HookCandidateScore | None:
        """Alias for :attr:`best`."""
        return self.best

    @property
    def best_index(self) -> int | None:
        """The input index of the highest scoring candidate."""
        best = self.best
        return None if best is None else best.index

    @property
    def chosen_index(self) -> int | None:
        """Alias for :attr:`best_index`."""
        return self.best_index

    @property
    def selected_index(self) -> int | None:
        """Alias for :attr:`best_index`."""
        return self.best_index

    @property
    def chosen(self) -> HookCandidate | None:
        """The highest scoring candidate, or ``None`` when there are none."""
        best = self.best
        return None if best is None else best.candidate

    @property
    def hook(self) -> str:
        """The text of the highest scoring candidate, or an empty string."""
        chosen = self.chosen
        return "" if chosen is None else chosen.text

    @property
    def ok(self) -> bool:
        """Whether at least one candidate clears the floor."""
        return bool(self.accepted)

    @property
    def reason(self) -> str:
        """The first rejection reason, or an empty string when one passed."""
        if self.ok:
            return ""
        for score in self.scores:
            if score.reasons:
                return score.reason
        return "no hook candidates"

    def require(self) -> HookPrefilterResult:
        """Return this result, raising when no candidate clears the floor."""
        if not self.ok:
            raise ConfigError(
                "no hook candidate passed the prefilter",
                details={"scores": [score.to_dict() for score in self.scores]},
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation including every score."""
        return {
            "scores": [score.to_dict() for score in self.scores],
            "best_index": self.best_index,
            "accepted": [score.index for score in self.accepted],
            "rejected": [score.index for score in self.rejected],
            "config": {
                "weights": dict(self.config.weights),
                "weakest_weight": self.config.weakest_weight,
                "min_score": self.config.min_score,
                "ideal_min_words": self.config.ideal_min_words,
                "ideal_max_words": self.config.ideal_max_words,
            },
        }

    def to_prompt_context(self) -> list[dict[str, Any]]:
        """Return scored candidates ready to be shown to a model that picks.

        The model sees the candidate text, its formula and the deterministic
        pre-score. It can still disagree, but it no longer picks blind.
        """
        return [
            {
                "index": score.index,
                "text": score.text,
                "formula": score.formula,
                "pre_score": score.score,
                "weakest_property": score.weakest_property,
                "weakest_score": score.weakest_score,
                "properties": dict(score.properties),
            }
            for score in self.ranked
        ]


def score_hook_candidate(
    candidate: Any,
    *,
    index: int = 0,
    config: HookPrefilterConfig | Mapping[str, Any] | None = None,
    weights: Mapping[str, Any] | None = None,
) -> HookCandidateScore:
    """Score one hook candidate with the deterministic heuristic."""
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ConfigError(
            "hook candidate index must be a non-negative integer",
            details={"index": repr(index)},
        )
    active = _coerce_config(config, weights=weights)
    hook = _coerce_candidate(candidate)
    properties = {
        name: _PROPERTY_SCORERS[name](hook.text, active) for name in HOOK_PROPERTIES
    }
    effective = {
        name: active.weights[name] for name in HOOK_PROPERTIES if name in active.weights
    }
    total_weight = sum(effective.values())
    weighted_mean = (
        sum(properties[name] * weight for name, weight in effective.items())
        / total_weight
    )
    weakest_property = min(
        effective,
        key=lambda name: (properties[name], HOOK_PROPERTIES.index(name)),
    )
    weakest_score = properties[weakest_property]
    score = (
        1.0 - active.weakest_weight
    ) * weighted_mean + active.weakest_weight * weakest_score
    return HookCandidateScore(
        index=index,
        candidate=hook,
        properties=properties,
        weights=dict(effective),
        weighted_mean=weighted_mean,
        weakest_property=weakest_property,
        weakest_score=weakest_score,
        score=score,
        min_score=active.min_score,
        weakest_weight=active.weakest_weight,
        reasons=_score_reasons(properties, text=hook.text, config=active),
    )


def score_hook_candidates(
    candidates: Any,
    *,
    config: HookPrefilterConfig | Mapping[str, Any] | None = None,
    weights: Mapping[str, Any] | None = None,
) -> HookPrefilterResult:
    """Score every candidate and return the full prefilter result."""
    active = _coerce_config(config, weights=weights)
    hooks = _coerce_candidates(candidates)
    return HookPrefilterResult(
        scores=tuple(
            score_hook_candidate(hook, index=index, config=active)
            for index, hook in enumerate(hooks)
        ),
        config=active,
    )


def rank_hook_candidates(
    candidates: Any,
    *,
    config: HookPrefilterConfig | Mapping[str, Any] | None = None,
    weights: Mapping[str, Any] | None = None,
) -> tuple[HookCandidateScore, ...]:
    """Return every score ordered from strongest to weakest."""
    return score_hook_candidates(candidates, config=config, weights=weights).ranked


def prefilter_hook_candidates(
    candidates: Any,
    *,
    config: HookPrefilterConfig | Mapping[str, Any] | None = None,
    weights: Mapping[str, Any] | None = None,
) -> HookPrefilterResult:
    """Score candidates and keep the accepted and rejected sets separate."""
    return score_hook_candidates(candidates, config=config, weights=weights)


def filter_hook_candidates(
    candidates: Any,
    *,
    config: HookPrefilterConfig | Mapping[str, Any] | None = None,
    weights: Mapping[str, Any] | None = None,
) -> tuple[HookCandidate, ...]:
    """Return only the candidates whose composite score clears the floor."""
    return prefilter_hook_candidates(
        candidates,
        config=config,
        weights=weights,
    ).accepted_candidates


def best_hook_candidate(
    candidates: Any,
    *,
    config: HookPrefilterConfig | Mapping[str, Any] | None = None,
    weights: Mapping[str, Any] | None = None,
) -> HookCandidateScore | None:
    """Return the highest scoring candidate, even when every one is rejected."""
    return prefilter_hook_candidates(
        candidates,
        config=config,
        weights=weights,
    ).best


def choose_hook_by_prefilter(
    candidates: Any,
    *,
    config: HookPrefilterConfig | Mapping[str, Any] | None = None,
    weights: Mapping[str, Any] | None = None,
) -> HookCandidate:
    """Return the highest scoring candidate, raising when none clears the floor."""
    chosen = (
        prefilter_hook_candidates(
            candidates,
            config=config,
            weights=weights,
        )
        .require()
        .chosen
    )
    assert chosen is not None
    return chosen


class HookPrefilter:
    """A reusable config bound to the scoring, ranking and filtering helpers."""

    def __init__(
        self,
        config: HookPrefilterConfig | Mapping[str, Any] | None = None,
        *,
        weights: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = _coerce_config(config, weights=weights)

    @property
    def weights(self) -> dict[str, float]:
        """A copy of the configured property weights."""
        return dict(self.config.weights)

    @property
    def min_score(self) -> float:
        """The configured acceptance floor."""
        return self.config.min_score

    @property
    def weakest_weight(self) -> float:
        """The configured weakest-property share."""
        return self.config.weakest_weight

    def score(self, candidate: Any, *, index: int = 0) -> HookCandidateScore:
        """Score one candidate with this filter's config."""
        return score_hook_candidate(candidate, index=index, config=self.config)

    def score_all(self, candidates: Any) -> HookPrefilterResult:
        """Score every candidate with this filter's config."""
        return score_hook_candidates(candidates, config=self.config)

    def prefilter(self, candidates: Any) -> HookPrefilterResult:
        """Alias for :meth:`score_all`."""
        return self.score_all(candidates)

    def rank(self, candidates: Any) -> tuple[HookCandidateScore, ...]:
        """Return every score ordered from strongest to weakest."""
        return self.score_all(candidates).ranked

    def filter(self, candidates: Any) -> tuple[HookCandidate, ...]:
        """Return only the candidates that clear the configured floor."""
        return self.score_all(candidates).accepted_candidates

    def best(self, candidates: Any) -> HookCandidateScore | None:
        """Return the highest scoring candidate, if there are any."""
        return self.score_all(candidates).best

    def choose(self, candidates: Any) -> HookCandidate:
        """Return the highest scoring candidate, raising when none passes."""
        return self.score_all(candidates).require().chosen  # type: ignore[return-value]

    def prompt_context(self, candidates: Any) -> list[dict[str, Any]]:
        """Return scored candidates ready for a model pick prompt."""
        return self.score_all(candidates).to_prompt_context()


HookPreFilter = HookPrefilter
"""Alias for :class:`HookPrefilter`."""

HookPreFilterConfig = HookPrefilterConfig
HookPreFilterResult = HookPrefilterResult
HookPreFilterScore = HookCandidateScore


def weakest_hook_property(
    candidate: Any,
    *,
    config: HookPrefilterConfig | Mapping[str, Any] | None = None,
    weights: Mapping[str, Any] | None = None,
) -> tuple[str, float]:
    """Return the weakest property name and score for one candidate."""
    score = score_hook_candidate(candidate, config=config, weights=weights)
    return score.weakest_link


prefilter_hook_candidate = score_hook_candidate
prefilter_hooks = prefilter_hook_candidates
filter_hooks = filter_hook_candidates
rank_hooks = rank_hook_candidates
score_hook = score_hook_candidate
score_hooks = score_hook_candidates
pre_score_hook_candidate = score_hook_candidate
prescore_hook_candidate = score_hook_candidate
pre_score_hook_candidates = score_hook_candidates
prescore_hook_candidates = score_hook_candidates

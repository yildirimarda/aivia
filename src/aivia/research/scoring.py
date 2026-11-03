"""Score topic candidates against a versioned, weighted rubric.

Scoring happens after collection, dedupe and relevance gating. Each candidate
receives one score per rubric dimension and a weighted mean overall score. The
rubric version is recorded on every score and result so a stored score can be
interpreted later, even after the configured weights change.

The stage is deterministic and offline: dimension scores may be supplied
explicitly, read from a candidate or produced by an injected synchronous
callable. An async scorer is available through :func:`score_candidates_async`
and :meth:`ScoringStage.run_async`. This module never calls a provider itself;
the caller decides whether the dimension scores come from a model, a heuristic
or a fixture.

The configured minimum score filters the result. Each score also records its
weakest rubric dimension and weakest dimension score, and ``min_dimension_score``
is carried on the rubric and every score. This stage deliberately keeps the
weighted mean as its only filter; :mod:`aivia.research.weakest_link` enforces
the weakest-dimension floor so one dead dimension blocks selection.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
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

from aivia.core.config.quality import ScoringConfig
from aivia.core.errors import ConfigError
from aivia.core.models.topic import TopicCandidate
from aivia.providers.protocols.source import SourceResponse

__all__ = [
    "DEFAULT_MIN_DIMENSION_SCORE",
    "DEFAULT_MIN_SCORE",
    "DEFAULT_RUBRIC_VERSION",
    "DEFAULT_SCORING_MIN_SCORE",
    "DEFAULT_SCORING_WEIGHTS",
    "CandidateScore",
    "Dimension",
    "Rubric",
    "RubricDimension",
    "ScoreResult",
    "ScoredCandidate",
    "Scorer",
    "ScoringResult",
    "ScoringRubric",
    "ScoringStage",
    "TopicScore",
    "TopicScorer",
    "build_scoring_rubric",
    "filter_scored",
    "filter_scores",
    "score",
    "score_candidate",
    "score_candidates",
    "score_candidates_async",
    "score_one",
    "score_topic",
    "score_topics",
    "scored_candidates",
    "scoring",
]

_TOLERANCE = 1e-6
_UNSET: Any = object()

_DEFAULT_CONFIG = ScoringConfig()
DEFAULT_RUBRIC_VERSION = _DEFAULT_CONFIG.rubric_version
"""The rubric version used when no configuration or explicit version is given."""

DEFAULT_SCORING_WEIGHTS: dict[str, float] = dict(_DEFAULT_CONFIG.weights)
"""The packaged dimension weights, copied so callers cannot mutate them."""

_EXPECTED_TOTAL = sum(DEFAULT_SCORING_WEIGHTS.values())

DEFAULT_MIN_SCORE = _DEFAULT_CONFIG.min_score
"""The default minimum weighted mean a candidate must reach."""

DEFAULT_SCORING_MIN_SCORE = DEFAULT_MIN_SCORE
"""Alias for :data:`DEFAULT_MIN_SCORE` that avoids a relevance name clash."""

DEFAULT_MIN_DIMENSION_SCORE = _DEFAULT_CONFIG.min_dimension_score
"""The default weakest-dimension floor carried by the rubric."""

_DIMENSION_ALIASES = (
    "dimensions",
    "dimension_scores",
    "per_dimension",
    "per_dimension_scores",
    "scores",
    "criteria",
    "factors",
    "checks",
)
_METADATA_KEYS = frozenset(
    {
        "candidate",
        "topic",
        "reason",
        "explanation",
        "rationale",
        "why",
        "score",
        "overall",
        "overall_score",
        "version",
        "rubric_version",
        "weights",
        "dimension_weights",
        "rubric_weights",
        "min_score",
        "minimum_score",
        "threshold",
        "passed",
    }
)


def _validate_unit(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be between zero and one")
    return number


def _validate_optional_unit(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return _validate_unit(value, label)


def _validate_weights(value: Any, label: str = "scoring") -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} weights must be a mapping")
    normalised: dict[str, float] = {}
    for name, weight in value.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{label} weight names must be non-empty")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError(f"{label} weight for {name} must be a number")
        number = float(weight)
        if not math.isfinite(number):
            raise ValueError(f"{label} weight for {name} must be finite")
        if number < 0:
            raise ValueError(f"{label} weight for {name} must not be negative")
        normalised[name.strip()] = number
    if not normalised:
        raise ValueError(f"{label} weights must not be empty")
    total = sum(normalised.values())
    if not math.isclose(
        total,
        _EXPECTED_TOTAL,
        rel_tol=0.0,
        abs_tol=_TOLERANCE,
    ):
        raise ValueError(
            f"{label} weights must sum to {_EXPECTED_TOTAL:g}, got {total:g}"
        )
    return normalised


def _validate_threshold(value: Any, label: str) -> float:
    try:
        return _validate_unit(value, label)
    except ValueError as error:
        raise ConfigError(str(error)) from error


def _validate_optional_threshold(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return _validate_threshold(value, label)


def _as_config_mapping(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, ScoringConfig):
        return config.model_dump()
    if isinstance(config, BaseModel):
        return config.model_dump()
    if isinstance(config, Mapping):
        values = dict(config)
    else:
        nested = getattr(config, "scoring", None)
        if nested is None:
            raise ConfigError("config must be a ScoringConfig, mapping or app config")
        if isinstance(nested, BaseModel):
            return nested.model_dump()
        if isinstance(nested, Mapping):
            return dict(nested)
        raise ConfigError("config.scoring must be a ScoringConfig or mapping")
    nested = values.get("scoring")
    if isinstance(nested, Mapping) and not any(
        key in values
        for key in (
            "version",
            "rubric_version",
            "weights",
            "dimension_weights",
            "rubric_weights",
            "min_score",
            "threshold",
            "criteria",
            "dimensions",
        )
    ):
        return dict(nested)
    return values


def _apply_aliases(values: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(values)
    aliases = {
        "rubric_version": "version",
        "dimension_weights": "weights",
        "rubric_weights": "weights",
        "minimum_score": "min_score",
        "threshold": "min_score",
        "min_weakest_dimension": "min_dimension_score",
        "weakest_dimension": "min_dimension_score",
        "factors": "criteria",
        "checks": "criteria",
    }
    for alias, canonical in aliases.items():
        if alias in result:
            value = result.pop(alias)
            result.setdefault(canonical, value)
    return result


class RubricDimension(BaseModel):
    """One named dimension of a scoring rubric."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    name: str = Field(
        validation_alias=AliasChoices(
            "name",
            "dimension",
            "criterion",
            "label",
            "title",
            "factor",
        )
    )
    description: str = Field(
        default="",
        validation_alias=AliasChoices(
            "description",
            "guidance",
            "detail",
            "question",
            "why",
        ),
    )
    weight: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _from_string(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"name": data}
        return data

    @field_validator("name", mode="before")
    @classmethod
    def _validate_name(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = " ".join(value.split())
            if not text:
                raise ValueError("rubric dimension name must not be empty")
            return text
        return value

    @field_validator("description", mode="before")
    @classmethod
    def _validate_description(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else (value or "")

    @field_validator("weight", mode="before")
    @classmethod
    def _validate_weight(cls, value: Any) -> Any:
        return _validate_optional_unit(value, "dimension weight")


class ScoringRubric(BaseModel):
    """A versioned rubric with named dimensions and configured weights.

    ``weights`` is authoritative: it maps each dimension to a non-negative
    weight and must sum to one, exactly like :class:`ScoringConfig`. Criteria
    carry the human-readable descriptions. When criteria omit weights, the
    packaged default weights are used if the names match, otherwise the
    dimensions share the weight equally.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    version: str = Field(
        default=DEFAULT_RUBRIC_VERSION,
        min_length=1,
        validation_alias=AliasChoices("version", "rubric_version"),
    )
    criteria: list[RubricDimension] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "criteria",
            "dimensions",
            "factors",
            "checks",
        ),
    )
    weights: dict[str, float] = Field(default_factory=dict)
    min_score: float | None = Field(
        default=DEFAULT_MIN_SCORE,
        validation_alias=AliasChoices(
            "min_score",
            "minimum_score",
            "threshold",
        ),
    )
    min_dimension_score: float = Field(
        default=DEFAULT_MIN_DIMENSION_SCORE,
        validation_alias=AliasChoices(
            "min_dimension_score",
            "min_weakest_dimension",
            "weakest_dimension",
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_rubric(cls, data: Any) -> Any:
        if data is None:
            return {}
        if isinstance(data, ScoringRubric):
            return data.model_dump()
        if isinstance(data, ScoringConfig):
            return data.model_dump()
        if isinstance(data, BaseModel):
            return data.model_dump()
        if isinstance(data, str):
            return {"version": data}
        if not isinstance(data, Mapping):
            return data
        values = _apply_aliases(data)
        for alias in _DIMENSION_ALIASES:
            if alias not in values:
                continue
            value = values.get(alias)
            if alias == "dimensions" and isinstance(value, Mapping):
                values.pop(alias, None)
                values.setdefault("weights", dict(value))
                values.setdefault(
                    "criteria",
                    [{"name": name} for name in value],
                )
            elif alias != "criteria" and "criteria" not in values:
                values["criteria"] = value
            if alias not in ("dimensions", "criteria"):
                values.pop(alias, None)
        return values

    @field_validator("version", mode="before")
    @classmethod
    def _validate_version(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise ValueError("rubric version must not be empty")
            return text
        return value

    @field_validator("criteria", mode="before")
    @classmethod
    def _coerce_criteria(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("weights", mode="before")
    @classmethod
    def _coerce_weights(cls, value: Any) -> Any:
        if value is None:
            return {}
        if isinstance(value, ScoringConfig):
            return value.weights
        if isinstance(value, BaseModel):
            return value.model_dump().get("weights", {})
        return value

    @field_validator("weights")
    @classmethod
    def _validate_weights_field(cls, value: dict[str, float]) -> dict[str, float]:
        return _validate_weights(value) if value else {}

    @field_validator("min_score", mode="before")
    @classmethod
    def _validate_min_score(cls, value: Any) -> Any:
        return _validate_optional_unit(value, "minimum score")

    @field_validator("min_dimension_score", mode="before")
    @classmethod
    def _validate_min_dimension_score(cls, value: Any) -> Any:
        return _validate_unit(value, "minimum dimension score")

    @model_validator(mode="after")
    def _complete_rubric(self) -> ScoringRubric:
        if not self.weights:
            if self.criteria:
                self.weights = self._weights_from_criteria()
            else:
                self.weights = dict(DEFAULT_SCORING_WEIGHTS)
        else:
            self.weights = _validate_weights(self.weights)
        if not self.criteria:
            self.criteria = [
                RubricDimension(name=name, weight=weight)
                for name, weight in self.weights.items()
            ]
        else:
            for criterion in self.criteria:
                if criterion.name in self.weights:
                    criterion.weight = self.weights[criterion.name]
        return self

    def _weights_from_criteria(self) -> dict[str, float]:
        names = [criterion.name for criterion in self.criteria]
        explicit = {
            criterion.name: criterion.weight
            for criterion in self.criteria
            if criterion.weight is not None
        }
        if len(explicit) == len(self.criteria):
            return _validate_weights(explicit)
        if set(names) <= set(DEFAULT_SCORING_WEIGHTS):
            return {name: DEFAULT_SCORING_WEIGHTS[name] for name in names}
        share = 1.0 / len(names)
        return dict.fromkeys(names, share)

    @classmethod
    def from_config(
        cls,
        config: Any = None,
        *,
        version: str | None = None,
        weights: Mapping[str, float] | None = None,
        min_score: Any = _UNSET,
        min_dimension_score: Any = _UNSET,
        criteria: Any = None,
        dimensions: Any = None,
    ) -> ScoringRubric:
        """Build a rubric from a config section and explicit overrides."""
        values = _as_config_mapping(config)
        if version is not None:
            values["version"] = version
        if weights is not None:
            values["weights"] = dict(weights)
        if min_score is not _UNSET:
            values["min_score"] = min_score
        if min_dimension_score is not _UNSET:
            values["min_dimension_score"] = min_dimension_score
        if criteria is not None:
            values["criteria"] = criteria
        if dimensions is not None:
            values["criteria"] = dimensions
        try:
            return cls.model_validate(values)
        except PydanticValidationError as error:
            raise ConfigError(str(error)) from error

    from_mapping = from_config
    """Alias for :meth:`from_config`."""

    @classmethod
    def from_derivation(
        cls,
        derivation: Any,
        config: Any = None,
        **overrides: Any,
    ) -> ScoringRubric:
        """Build a rubric from a brief derivation's criteria.

        The criteria names become the rubric dimensions. Configured weights
        apply when their names match; otherwise the dimensions share the
        weight equally.
        """
        brief_rubric = getattr(derivation, "rubric", None)
        if brief_rubric is None:
            brief_rubric = getattr(derivation, "relevance_rubric", None)
        criteria = []
        for criterion in getattr(brief_rubric, "criteria", ()) or ():
            name = getattr(criterion, "name", "")
            if name:
                criteria.append(
                    RubricDimension(
                        name=name,
                        description=getattr(criterion, "description", ""),
                    )
                )
        overrides = dict(overrides)
        explicit = overrides.pop("criteria", None)
        if explicit is None:
            explicit = overrides.pop("dimensions", None)
        else:
            overrides.pop("dimensions", None)
        if explicit is not None:
            criteria = explicit
        return cls.from_config(config, criteria=criteria or None, **overrides)

    @property
    def rubric_version(self) -> str:
        """Alias for :attr:`version`."""
        return self.version

    @property
    def dimensions(self) -> tuple[str, ...]:
        """The dimension names in configured weight order."""
        return tuple(self.weights)

    @property
    def names(self) -> tuple[str, ...]:
        """Alias for :attr:`dimensions`."""
        return self.dimensions

    @property
    def dimension_weights(self) -> dict[str, float]:
        """A copy of the configured weights."""
        return dict(self.weights)

    @property
    def threshold(self) -> float | None:
        """Alias for :attr:`min_score`."""
        return self.min_score

    @property
    def min_weakest_dimension(self) -> float:
        """Alias for :attr:`min_dimension_score`."""
        return self.min_dimension_score

    def weight_for(self, name: str) -> float:
        """Return the configured weight for *name*, or zero when unknown."""
        return self.weights.get(name, 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable rubric record."""
        return {
            "version": self.version,
            "dimensions": list(self.dimensions),
            "weights": dict(self.weights),
            "min_score": self.min_score,
            "min_dimension_score": self.min_dimension_score,
            "criteria": [
                criterion.model_dump(mode="json") for criterion in self.criteria
            ],
        }


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


def _dimensions_from_sequence(
    value: Sequence[Any],
    rubric: ScoringRubric,
) -> dict[str, float]:
    if all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
    ):
        return {
            name: float(score)
            for name, score in zip(rubric.dimensions, value, strict=False)
        }
    dimensions: dict[str, float] = {}
    for item in value:
        if isinstance(item, Mapping):
            name = item.get("name", item.get("dimension"))
            score = item.get("score", item.get("value"))
        elif isinstance(item, Sequence) and not isinstance(
            item,
            (str, bytes, bytearray),
        ):
            if len(item) != 2:
                raise ConfigError("dimension pairs must have a name and a score")
            name, score = item
        else:
            raise ConfigError(f"invalid dimension score: {item!r}")
        if not isinstance(name, str) or not name.strip():
            raise ConfigError("dimension name must be a non-empty string")
        dimensions[name.strip()] = score
    return dimensions


def _coerce_dimensions(value: Any, rubric: ScoringRubric) -> Any:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _dimensions_from_sequence(value, rubric)
    return value


def _coerce_dimension_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError("dimension scores must be a mapping")
    dimensions: dict[str, float] = {}
    for name, score in value.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("dimension name must be a non-empty string")
        dimensions[name.strip()] = _validate_unit(
            score,
            f"{name} dimension score",
        )
    if not dimensions:
        raise ValueError("at least one dimension score is required")
    return dimensions


class CandidateScore(BaseModel):
    """One candidate's per-dimension scores and weighted overall score."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    candidate: TopicCandidate = Field(
        validation_alias=AliasChoices("candidate", "topic"),
    )
    dimensions: dict[str, float] = Field(
        validation_alias=AliasChoices(
            "dimensions",
            "dimension_scores",
            "per_dimension",
            "per_dimension_scores",
            "scores",
        )
    )
    score: float | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "score",
            "overall",
            "overall_score",
            "weighted_score",
        ),
    )
    reason: str = Field(
        default="",
        validation_alias=AliasChoices(
            "reason",
            "explanation",
            "rationale",
            "why",
        ),
    )
    rubric_version: str = Field(
        default=DEFAULT_RUBRIC_VERSION,
        validation_alias=AliasChoices("rubric_version", "version"),
    )
    weights: dict[str, float] = Field(default_factory=dict)
    min_score: float | None = Field(
        default=DEFAULT_MIN_SCORE,
        validation_alias=AliasChoices(
            "min_score",
            "minimum_score",
            "threshold",
        ),
    )
    min_dimension_score: float = Field(
        default=DEFAULT_MIN_DIMENSION_SCORE,
        validation_alias=AliasChoices(
            "min_dimension_score",
            "min_weakest_dimension",
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_score(cls, data: Any) -> Any:
        if isinstance(data, CandidateScore):
            return data.model_dump()
        if isinstance(data, BaseModel):
            return data.model_dump()
        if not isinstance(data, Mapping):
            return data
        values = _apply_aliases(data)
        if "candidate" not in values and "topic" in values:
            values["candidate"] = values.pop("topic")
        for alias in _DIMENSION_ALIASES:
            if alias in values and "dimensions" not in values:
                values["dimensions"] = values.pop(alias)
        if "dimensions" not in values:
            weights = values.get("weights")
            if isinstance(weights, Mapping):
                dimensions = {
                    name: values.pop(name) for name in list(weights) if name in values
                }
                if dimensions:
                    values["dimensions"] = dimensions
        return values

    @field_validator("candidate", mode="before")
    @classmethod
    def _validate_candidate(cls, value: Any) -> Any:
        return _coerce_candidate(value)

    @field_validator("dimensions", mode="before")
    @classmethod
    def _coerce_dimensions_field(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return value
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            return _dimensions_from_sequence(value, ScoringRubric())
        return value

    @field_validator("dimensions")
    @classmethod
    def _validate_dimensions(cls, value: dict[str, float]) -> dict[str, float]:
        return _coerce_dimension_mapping(value)

    @field_validator("score", mode="before")
    @classmethod
    def _validate_score(cls, value: Any) -> Any:
        return _validate_optional_unit(value, "score")

    @field_validator("reason", mode="before")
    @classmethod
    def _validate_reason(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else (value or "")

    @field_validator("rubric_version", mode="before")
    @classmethod
    def _validate_rubric_version(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise ValueError("rubric version must not be empty")
            return text
        return value

    @field_validator("weights", mode="before")
    @classmethod
    def _coerce_weights_field(cls, value: Any) -> Any:
        if value is None:
            return {}
        if isinstance(value, ScoringConfig):
            return value.weights
        if isinstance(value, BaseModel):
            return value.model_dump().get("weights", {})
        return value

    @field_validator("weights")
    @classmethod
    def _validate_weights_field(cls, value: dict[str, float]) -> dict[str, float]:
        return _validate_weights(value) if value else {}

    @field_validator("min_score", mode="before")
    @classmethod
    def _validate_min_score(cls, value: Any) -> Any:
        return _validate_optional_unit(value, "minimum score")

    @field_validator("min_dimension_score", mode="before")
    @classmethod
    def _validate_min_dimension_score(cls, value: Any) -> Any:
        return _validate_unit(value, "minimum dimension score")

    @model_validator(mode="after")
    def _fill_overall(self) -> CandidateScore:
        dimensions = dict(self.dimensions)
        weights = dict(self.weights)
        if not weights:
            if set(dimensions) <= set(DEFAULT_SCORING_WEIGHTS):
                weights = {name: DEFAULT_SCORING_WEIGHTS[name] for name in dimensions}
            else:
                weights = {name: 1.0 / len(dimensions) for name in dimensions}
        present = {
            name: weight for name, weight in weights.items() if name in dimensions
        }
        if not present:
            present = {name: 1.0 / len(dimensions) for name in dimensions}
            weights = dict(present)
        if self.score is None:
            denominator = sum(present.values())
            if denominator <= 0:
                self.score = sum(dimensions.values()) / len(dimensions)
            else:
                self.score = (
                    sum(dimensions[name] * weight for name, weight in present.items())
                    / denominator
                )
        self.weights = weights
        return self

    @property
    def overall(self) -> float:
        """The weighted overall score."""
        return float(self.score or 0.0)

    @property
    def dimension_scores(self) -> dict[str, float]:
        """A copy of the per-dimension scores."""
        return dict(self.dimensions)

    @property
    def dimension_order(self) -> tuple[str, ...]:
        """The scored dimensions in rubric weight order, then any extras."""
        ordered = [name for name in self.weights if name in self.dimensions]
        ordered.extend(name for name in self.dimensions if name not in ordered)
        return tuple(ordered)

    @property
    def weakest_dimension(self) -> str | None:
        """The name of the lowest-scoring dimension, or ``None`` when unscored."""
        order = self.dimension_order
        if not order:
            return None
        return min(order, key=lambda name: (self.dimensions[name], order.index(name)))

    @property
    def weakest_score(self) -> float | None:
        """The lowest dimension score, or ``None`` when there are no dimensions."""
        name = self.weakest_dimension
        return None if name is None else self.dimensions[name]

    @property
    def weakest_link(self) -> tuple[str, float] | None:
        """The weakest dimension name and score as a pair, or ``None``."""
        name = self.weakest_dimension
        if name is None:
            return None
        return (name, self.dimensions[name])

    @property
    def meets_min_dimension(self) -> bool:
        """Whether the weakest dimension clears the configured floor."""
        value = self.weakest_score
        return True if value is None else value >= self.min_dimension_score

    passes_weakest_link = meets_min_dimension
    """Alias for :attr:`meets_min_dimension`."""

    @property
    def blocked_by_weakest_link(self) -> bool:
        """Whether the mean passes but a single dead dimension blocks selection."""
        return self.passed and not self.meets_min_dimension

    @property
    def selected(self) -> bool:
        """Whether both the weighted mean and the weakest dimension pass."""
        return self.passed and self.meets_min_dimension

    @property
    def title(self) -> str:
        """The candidate title."""
        return self.candidate.title

    @property
    def candidate_title(self) -> str:
        """Alias for :attr:`title`."""
        return self.title

    @property
    def version(self) -> str:
        """Alias for :attr:`rubric_version`."""
        return self.rubric_version

    @property
    def threshold(self) -> float | None:
        """Alias for :attr:`min_score`."""
        return self.min_score

    @property
    def passed(self) -> bool:
        """Whether the overall score reaches the configured minimum."""
        return self.min_score is None or self.overall >= self.min_score

    @property
    def relevant(self) -> bool:
        """Alias for :attr:`passed`."""
        return self.passed

    @property
    def accepted(self) -> bool:
        """Alias for :attr:`passed`."""
        return self.passed

    @property
    def kept(self) -> bool:
        """Alias for :attr:`passed`."""
        return self.passed

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable score record."""
        return {
            "title": self.title,
            "candidate": self.candidate.model_dump(mode="json"),
            "dimensions": dict(self.dimensions),
            "score": self.overall,
            "passed": self.passed,
            "reason": self.reason,
            "rubric_version": self.rubric_version,
            "weights": dict(self.weights),
            "min_score": self.min_score,
            "min_dimension_score": self.min_dimension_score,
            "weakest_dimension": self.weakest_dimension,
            "weakest_score": self.weakest_score,
            "meets_min_dimension": self.meets_min_dimension,
            "selected": self.selected,
        }


TopicScore = CandidateScore
"""Alias for :class:`CandidateScore`."""

ScoredCandidate = CandidateScore
"""Alias for :class:`CandidateScore`."""


@dataclass(frozen=True, slots=True)
class ScoringResult:
    """Every scored candidate plus the ones that clear the minimum score."""

    scores: tuple[CandidateScore, ...] = ()
    rubric: ScoringRubric = field(default_factory=ScoringRubric)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def scored(self) -> tuple[CandidateScore, ...]:
        """Alias for :attr:`scores`."""
        return self.scores

    @property
    def all_scores(self) -> tuple[CandidateScore, ...]:
        """Alias for :attr:`scores`."""
        return self.scores

    @property
    def candidates(self) -> list[TopicCandidate]:
        """The candidates that clear the minimum score, in input order."""
        return [score.candidate for score in self.scores if score.passed]

    @property
    def passing(self) -> list[CandidateScore]:
        """The passing scores in input order."""
        return [score for score in self.scores if score.passed]

    @property
    def kept(self) -> list[TopicCandidate]:
        """Alias for :attr:`candidates`."""
        return self.candidates

    @property
    def topics(self) -> list[TopicCandidate]:
        """Alias for :attr:`candidates`."""
        return self.candidates

    @property
    def passing_candidates(self) -> list[TopicCandidate]:
        """Alias for :attr:`candidates`."""
        return self.candidates

    @property
    def failed(self) -> list[CandidateScore]:
        """The scores that did not clear the minimum."""
        return [score for score in self.scores if not score.passed]

    @property
    def failures(self) -> list[CandidateScore]:
        """Alias for :attr:`failed`."""
        return self.failed

    @property
    def dropped(self) -> list[TopicCandidate]:
        """The candidates dropped by the minimum score, in input order."""
        return [score.candidate for score in self.failed]

    @property
    def dropped_candidates(self) -> list[TopicCandidate]:
        """Alias for :attr:`dropped`."""
        return self.dropped

    @property
    def titles(self) -> list[str]:
        """The titles of the passing candidates."""
        return [candidate.title for candidate in self.candidates]

    @property
    def count(self) -> int:
        """The number of candidates that cleared the minimum."""
        return len(self.candidates)

    @property
    def input_count(self) -> int:
        """The number of candidates the stage received."""
        return len(self.scores)

    @property
    def dropped_count(self) -> int:
        """The number of candidates dropped by the minimum score."""
        return len(self.failed)

    @property
    def version(self) -> str:
        """The rubric version the scores were produced with."""
        return self.rubric.version

    @property
    def min_score(self) -> float | None:
        """The minimum score the result was filtered with."""
        return self.rubric.min_score

    @property
    def min_dimension_score(self) -> float:
        """The weakest-dimension floor carried by the rubric."""
        return self.rubric.min_dimension_score

    @property
    def weights(self) -> dict[str, float]:
        """The rubric weights used for the overall scores."""
        return dict(self.rubric.weights)

    @property
    def dimensions(self) -> tuple[str, ...]:
        """The rubric dimensions in configured order."""
        return self.rubric.dimensions

    def filter(
        self,
        min_score: float | None = None,
        *,
        threshold: float | None = None,
    ) -> list[TopicCandidate]:
        """Return candidates above *min_score*, defaulting to the rubric."""
        limit = self.min_score if min_score is None else min_score
        if threshold is not None:
            limit = threshold
        if limit is None:
            return [score.candidate for score in self.scores]
        return [score.candidate for score in self.scores if score.overall >= limit]

    def to_response(
        self,
        *,
        provider: str = "scoring",
        query: str = "",
    ) -> SourceResponse:
        """Return the passing candidates as a :class:`SourceResponse`."""
        return SourceResponse(
            provider=provider,
            query=query,
            candidates=self.candidates,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the stage."""
        return {
            "scores": [score.to_dict() for score in self.scores],
            "candidates": [
                candidate.model_dump(mode="json") for candidate in self.candidates
            ],
            "rubric": self.rubric.to_dict(),
            "metadata": dict(self.metadata),
        }

    def __iter__(self) -> Iterator[TopicCandidate]:
        return iter(self.candidates)

    def __len__(self) -> int:
        return len(self.candidates)


ScoreResult = ScoringResult
"""Alias for :class:`ScoringResult`."""


def _extract_score_payload(
    value: Any,
    rubric: ScoringRubric,
) -> dict[str, Any]:
    if isinstance(value, CandidateScore):
        return value.model_dump()
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, Mapping):
        values = _apply_aliases(dict(value))
        dimensions = None
        for alias in _DIMENSION_ALIASES:
            if alias in values and alias != "criteria":
                dimensions = values[alias]
                break
        if dimensions is None:
            dimensions = {
                name: values[name] for name in rubric.dimensions if name in values
            }
        if not dimensions:
            overall = values.get("score", values.get("overall"))
            if overall is not None:
                dimensions = dict.fromkeys(rubric.dimensions, overall)
        if dimensions is None:
            raise ConfigError("scorer returned no dimension scores")
        payload = {
            "dimensions": dimensions,
            "reason": values.get("reason", values.get("explanation", "")),
            "score": values.get(
                "score",
                values.get("overall", values.get("overall_score")),
            ),
            "rubric_version": values.get(
                "rubric_version",
                values.get("version", rubric.version),
            ),
            "weights": values.get(
                "weights",
                values.get("dimension_weights", values.get("rubric_weights")),
            ),
            "min_score": values.get(
                "min_score",
                values.get("minimum_score", values.get("threshold")),
            ),
            "min_dimension_score": values.get(
                "min_dimension_score",
                values.get("min_weakest_dimension"),
            ),
        }
        return payload
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {
            "dimensions": {name: float(value) for name in rubric.dimensions},
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {"dimensions": _dimensions_from_sequence(value, rubric)}
    for attribute in ("dimension_scores", "dimensions", "scores"):
        dimensions = getattr(value, attribute, None)
        if dimensions is not None:
            payload = {"dimensions": dimensions}
            reason = getattr(value, "reason", "")
            if reason:
                payload["reason"] = reason
            score = getattr(value, "score", None)
            if score is not None:
                payload["score"] = score
            return payload
    raise ConfigError(f"invalid scorer output: {value!r}")


def _build_candidate_score(
    candidate: TopicCandidate,
    value: Any,
    rubric: ScoringRubric,
    *,
    reason: str = "",
) -> CandidateScore:
    if isinstance(value, CandidateScore):
        if value.candidate is candidate:
            return value
        return value.model_copy(update={"candidate": candidate})
    payload = _extract_score_payload(value, rubric)
    for key in (
        "rubric_version",
        "weights",
        "min_score",
        "min_dimension_score",
    ):
        if payload.get(key) is None:
            payload.pop(key, None)
    payload.setdefault("reason", reason)
    payload.setdefault("rubric_version", rubric.version)
    payload.setdefault("weights", dict(rubric.weights))
    payload.setdefault("min_score", rubric.min_score)
    payload.setdefault("min_dimension_score", rubric.min_dimension_score)
    payload["candidate"] = candidate
    try:
        return CandidateScore.model_validate(payload)
    except PydanticValidationError as error:
        raise ConfigError(str(error)) from error


def _call_scorer(scorer: Any, candidate: TopicCandidate) -> Any:
    method = scorer if callable(scorer) else getattr(scorer, "score", None)
    if not callable(method):
        raise ConfigError("scorer must be callable or expose score()")
    outcome = method(candidate)
    if inspect.isawaitable(outcome):
        close = getattr(outcome, "close", None)
        if callable(close):
            close()
        raise ConfigError(
            "scorer is asynchronous; use score_candidates_async()",
        )
    return outcome


async def _call_scorer_async(scorer: Any, candidate: TopicCandidate) -> Any:
    method = scorer if callable(scorer) else getattr(scorer, "score", None)
    if not callable(method):
        raise ConfigError("scorer must be callable or expose score()")
    outcome = method(candidate)
    if inspect.isawaitable(outcome):
        return await outcome
    return outcome


def _mapping_lookup(
    scores: Mapping[Any, Any],
    candidate: TopicCandidate,
) -> Any:
    for key in (
        candidate.title,
        candidate.name,
        candidate.title.strip(),
        candidate.title.casefold(),
        " ".join(candidate.title.split()).casefold(),
    ):
        if key in scores:
            return scores[key]
    return _UNSET


def _resolve_explicit_score(
    scores: Any,
    candidate: TopicCandidate,
    index: int,
    rubric: ScoringRubric,
) -> Any:
    if scores is None:
        return _UNSET
    if isinstance(scores, Mapping):
        found = _mapping_lookup(scores, candidate)
        if found is not _UNSET:
            return found
        if all(not isinstance(value, Mapping) for value in scores.values()) and any(
            name in scores for name in rubric.dimensions
        ):
            return dict(scores)
        return _UNSET
    if isinstance(scores, Sequence) and not isinstance(scores, (str, bytes, bytearray)):
        if index < len(scores):
            return scores[index]
        return _UNSET
    return scores


def _resolve_rubric(
    *,
    rubric: Any,
    config: Any,
    derivation: Any = None,
    version: Any,
    weights: Any,
    min_score: Any,
    min_dimension_score: Any,
    threshold: Any,
) -> ScoringRubric:
    if threshold is not _UNSET:
        if min_score is not _UNSET and min_score != threshold:
            raise ConfigError("conflicting minimum score values")
        min_score = threshold
    if isinstance(rubric, ScoringRubric):
        base = rubric
    elif rubric is not None:
        base = ScoringRubric.from_config(rubric)
    elif derivation is not None:
        base = ScoringRubric.from_derivation(derivation, config)
    else:
        base = ScoringRubric.from_config(config)
    if (
        version is _UNSET
        and weights is _UNSET
        and min_score is _UNSET
        and (min_dimension_score is _UNSET)
    ):
        return base
    values = base.model_dump()
    if version is not _UNSET:
        values["version"] = version
    if weights is not _UNSET:
        values["weights"] = weights
    if min_score is not _UNSET:
        values["min_score"] = min_score
    if min_dimension_score is not _UNSET:
        values["min_dimension_score"] = min_dimension_score
    try:
        return ScoringRubric.model_validate(values)
    except PydanticValidationError as error:
        raise ConfigError(str(error)) from error


def score_candidates(
    candidates: Any,
    *,
    scores: Any = None,
    scorer: Any = None,
    dimension_scorer: Any = None,
    rubric: Any = None,
    config: Any = None,
    derivation: Any = None,
    weights: Any = _UNSET,
    version: Any = _UNSET,
    min_score: Any = _UNSET,
    threshold: Any = _UNSET,
    min_dimension_score: Any = _UNSET,
    reason: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> ScoringResult:
    """Score *candidates* and keep the ones above the minimum score.

    Dimension scores come from ``scores`` first. ``scores`` may be a mapping
    keyed by candidate title, a sequence aligned with *candidates*, a single
    dimension mapping, or a callable. ``scorer`` and ``dimension_scorer`` are
    aliases for a callable scorer. When neither is supplied, a candidate's own
    ``dimension_scores``/``scores`` attribute and then its ``score`` field are
    used as a deterministic fallback.
    """
    active_scorer = scorer if scorer is not None else dimension_scorer
    if callable(scores) and active_scorer is None:
        active_scorer = scores
        scores = None
    active_rubric = _resolve_rubric(
        rubric=rubric,
        config=config,
        derivation=derivation,
        version=version,
        weights=weights,
        min_score=min_score,
        min_dimension_score=min_dimension_score,
        threshold=threshold,
    )
    values = _coerce_candidates(candidates)
    results: list[CandidateScore] = []
    for index, candidate in enumerate(values):
        value = _resolve_explicit_score(scores, candidate, index, active_rubric)
        if value is _UNSET:
            if active_scorer is not None:
                value = _call_scorer(active_scorer, candidate)
            else:
                for attribute in ("dimension_scores", "dimensions", "scores"):
                    found = getattr(candidate, attribute, None)
                    if found is not None:
                        value = found
                        break
                else:
                    value = candidate.score
        results.append(
            _build_candidate_score(
                candidate,
                value,
                active_rubric,
                reason=reason,
            )
        )
    passing = sum(1 for score in results if score.passed)
    result_metadata: dict[str, Any] = {
        "rubric_version": active_rubric.version,
        "weights": dict(active_rubric.weights),
        "min_score": active_rubric.min_score,
        "min_dimension_score": active_rubric.min_dimension_score,
        "input_count": len(results),
        "kept": passing,
        "dropped": len(results) - passing,
    }
    result_metadata.update(metadata or {})
    return ScoringResult(
        scores=tuple(results),
        rubric=active_rubric,
        metadata=result_metadata,
    )


async def score_candidates_async(
    candidates: Any,
    *,
    scorer: Any = None,
    dimension_scorer: Any = None,
    **options: Any,
) -> ScoringResult:
    """Async variant that awaits a scorer before scoring every candidate."""
    active_scorer = scorer if scorer is not None else dimension_scorer
    values = _coerce_candidates(candidates)
    if active_scorer is None:
        return score_candidates(values, **options)
    explicit: list[Any] = []
    for candidate in values:
        explicit.append(await _call_scorer_async(active_scorer, candidate))
    options.pop("scorer", None)
    options.pop("dimension_scorer", None)
    options.pop("scores", None)
    return score_candidates(values, scores=explicit, **options)


def score_topic(
    candidate: Any,
    *,
    scores: Any = None,
    scorer: Any = None,
    dimension_scorer: Any = None,
    rubric: Any = None,
    config: Any = None,
    derivation: Any = None,
    weights: Any = _UNSET,
    version: Any = _UNSET,
    min_score: Any = _UNSET,
    threshold: Any = _UNSET,
    min_dimension_score: Any = _UNSET,
    reason: str = "",
) -> CandidateScore:
    """Score one candidate and return its :class:`CandidateScore`."""
    result = score_candidates(
        [candidate],
        scores=scores,
        scorer=scorer,
        dimension_scorer=dimension_scorer,
        rubric=rubric,
        config=config,
        derivation=derivation,
        weights=weights,
        version=version,
        min_score=min_score,
        threshold=threshold,
        min_dimension_score=min_dimension_score,
        reason=reason,
    )
    return result.scores[0]


score_candidate = score_topic
"""Alias for :func:`score_topic` for direct module imports."""

score_one = score_topic
"""Alias for :func:`score_topic`."""

score_topics = score_candidates
"""Alias for :func:`score_candidates`."""

scored_candidates = score_candidates
"""Alias for :func:`score_candidates`."""

scoring = score_candidates
"""Alias for :func:`score_candidates`."""

score = score_candidates
"""Alias for :func:`score_candidates`."""


def filter_scored(
    result: ScoringResult | Sequence[CandidateScore] | Any,
    *,
    min_score: Any = _UNSET,
    threshold: Any = _UNSET,
) -> list[TopicCandidate]:
    """Return the candidates above a minimum score from a result or scores."""
    if isinstance(result, ScoringResult):
        if min_score is _UNSET and threshold is _UNSET:
            return result.candidates
        limit = threshold if threshold is not _UNSET else min_score
        if limit is None:
            return [score.candidate for score in result.scores]
        if (
            threshold is not _UNSET
            and min_score is not _UNSET
            and min_score != threshold
        ):
            raise ConfigError("conflicting minimum score values")
        value = _validate_threshold(limit, "minimum score")
        return [score.candidate for score in result.scores if score.overall >= value]
    if isinstance(result, CandidateScore):
        values = [result]
    else:
        values = [
            item
            if isinstance(item, CandidateScore)
            else CandidateScore.model_validate(item)
            for item in result
        ]
    limit = threshold if threshold is not _UNSET else min_score
    if limit is _UNSET or limit is None:
        return [score.candidate for score in values]
    value = _validate_threshold(limit, "minimum score")
    return [score.candidate for score in values if score.overall >= value]


filter_scores = filter_scored
"""Alias for :func:`filter_scored`."""


def build_scoring_rubric(
    config: Any = None,
    *,
    derivation: Any = None,
    version: str | None = None,
    weights: Mapping[str, float] | None = None,
    min_score: Any = _UNSET,
    threshold: Any = _UNSET,
    min_dimension_score: Any = _UNSET,
    criteria: Any = None,
    dimensions: Any = None,
) -> ScoringRubric:
    """Build a rubric from config, with explicit overrides and threshold aliases."""
    if threshold is not _UNSET:
        if min_score is not _UNSET and min_score != threshold:
            raise ConfigError("conflicting minimum score values")
        min_score = threshold
    if derivation is not None:
        return ScoringRubric.from_derivation(
            derivation,
            config,
            version=version,
            weights=weights,
            min_score=min_score,
            min_dimension_score=min_dimension_score,
            criteria=criteria,
            dimensions=dimensions,
        )
    return ScoringRubric.from_config(
        config,
        version=version,
        weights=weights,
        min_score=min_score,
        min_dimension_score=min_dimension_score,
        criteria=criteria,
        dimensions=dimensions,
    )


Rubric = ScoringRubric
"""Alias for :class:`ScoringRubric`."""

Dimension = RubricDimension
"""Alias for :class:`RubricDimension`."""


class ScoringStage:
    """A reusable scoring stage bound to one rubric and scorer."""

    def __init__(
        self,
        rubric: Any = None,
        *,
        config: Any = None,
        derivation: Any = None,
        weights: Any = _UNSET,
        version: Any = _UNSET,
        min_score: Any = _UNSET,
        threshold: Any = _UNSET,
        min_dimension_score: Any = _UNSET,
        scorer: Any = None,
        dimension_scorer: Any = None,
        scores: Any = None,
        reason: str = "",
        name: str = "scoring",
    ) -> None:
        if threshold is not _UNSET:
            if min_score is not _UNSET and min_score != threshold:
                raise ConfigError("conflicting minimum score values")
            min_score = threshold
        self._rubric = _resolve_rubric(
            rubric=rubric,
            config=config,
            derivation=derivation,
            version=version,
            weights=weights,
            min_score=min_score,
            min_dimension_score=min_dimension_score,
            threshold=_UNSET,
        )
        self._scorer = scorer if scorer is not None else dimension_scorer
        self._scores = scores
        self._reason = reason
        self._name = name.strip() or "scoring"

    @property
    def name(self) -> str:
        """The provider name reported by :meth:`to_response`."""
        return self._name

    @property
    def rubric(self) -> ScoringRubric:
        """The rubric this stage scores with."""
        return self._rubric

    @property
    def weights(self) -> dict[str, float]:
        """A copy of the configured weights."""
        return dict(self._rubric.weights)

    @property
    def version(self) -> str:
        """The rubric version recorded on every score."""
        return self._rubric.version

    @property
    def dimensions(self) -> tuple[str, ...]:
        """The rubric dimensions in configured order."""
        return self._rubric.dimensions

    @property
    def min_score(self) -> float | None:
        """The minimum weighted mean a candidate must reach."""
        return self._rubric.min_score

    @property
    def threshold(self) -> float | None:
        """Alias for :attr:`min_score`."""
        return self.min_score

    @property
    def min_dimension_score(self) -> float:
        """The weakest-dimension floor carried by the rubric."""
        return self._rubric.min_dimension_score

    @property
    def min_weakest_dimension(self) -> float:
        """Alias for :attr:`min_dimension_score`."""
        return self.min_dimension_score

    @property
    def scorer(self) -> Any:
        """The injected scorer, if any."""
        return self._scorer

    def score(
        self,
        candidate: Any,
        *,
        scores: Any = None,
        reason: str | None = None,
        rubric: ScoringRubric | None = None,
    ) -> CandidateScore:
        """Score one candidate using this stage's rubric and scorer."""
        active_rubric = self._rubric if rubric is None else rubric
        explicit = self._scores if scores is None else scores
        value = _resolve_explicit_score(
            explicit,
            _coerce_candidate(candidate),
            0,
            active_rubric,
        )
        if value is _UNSET:
            if self._scorer is not None:
                value = _call_scorer(self._scorer, _coerce_candidate(candidate))
            else:
                value = _coerce_candidate(candidate).score
        return _build_candidate_score(
            _coerce_candidate(candidate),
            value,
            active_rubric,
            reason=self._reason if reason is None else reason,
        )

    async def score_async(
        self,
        candidate: Any,
        *,
        scores: Any = None,
        reason: str | None = None,
        rubric: ScoringRubric | None = None,
    ) -> CandidateScore:
        """Score one candidate, awaiting an async scorer when needed."""
        active_rubric = self._rubric if rubric is None else rubric
        explicit = self._scores if scores is None else scores
        value = _resolve_explicit_score(
            explicit,
            _coerce_candidate(candidate),
            0,
            active_rubric,
        )
        if value is _UNSET:
            if self._scorer is not None:
                value = await _call_scorer_async(
                    self._scorer,
                    _coerce_candidate(candidate),
                )
            else:
                value = _coerce_candidate(candidate).score
        return _build_candidate_score(
            _coerce_candidate(candidate),
            value,
            active_rubric,
            reason=self._reason if reason is None else reason,
        )

    def run(
        self,
        candidates: Any,
        *,
        scores: Any = None,
        scorer: Any = None,
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ScoringResult:
        """Score *candidates* using this stage's configuration."""
        return score_candidates(
            candidates,
            scores=self._scores if scores is None else scores,
            scorer=self._scorer if scorer is None else scorer,
            rubric=self._rubric,
            reason=self._reason if reason is None else reason,
            metadata=metadata,
        )

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
    ) -> ScoringResult:
        """Async variant that can await an async scorer."""
        return await score_candidates_async(
            candidates,
            scorer=self._scorer if scorer is None else scorer,
            scores=self._scores if scores is None else scores,
            rubric=self._rubric,
            reason=self._reason if reason is None else reason,
            metadata=metadata,
        )

    score_candidates_async = run_async

    def filter(
        self,
        candidates: Any,
        *,
        scores: Any = None,
        scorer: Any = None,
    ) -> list[TopicCandidate]:
        """Return only the candidates that clear the minimum score."""
        return self.run(candidates, scores=scores, scorer=scorer).candidates

    def to_response(
        self,
        candidates: Any,
        *,
        query: str = "",
        scores: Any = None,
        scorer: Any = None,
    ) -> SourceResponse:
        """Score *candidates* and return the passing ones as a response."""
        return self.run(candidates, scores=scores, scorer=scorer).to_response(
            provider=self._name,
            query=query,
        )


Scorer = ScoringStage
"""Alias for :class:`ScoringStage`."""

TopicScorer = ScoringStage
"""Alias for :class:`ScoringStage`."""

"""Quality and scoring configuration: thresholds, retries and weighted rubrics.

Both sections score candidates on several dimensions. ``weights`` maps each
dimension to a non-negative weight and must sum to ``1.0`` so a weighted mean
stays comparable between rubrics. A missing mapping uses the packaged default
rubric; an explicit mapping is validated as a whole.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from pydantic import Field, field_validator, model_validator

from aivia.core.config.base import ConfigModel

__all__ = [
    "QualityConfig",
    "ScoringConfig",
]

_WEIGHT_SUM = 1.0
_WEIGHT_TOLERANCE = 1e-6

_DEFAULT_QUALITY_WEIGHTS = {
    "prompt_adherence": 0.4,
    "technical_quality": 0.3,
    "aesthetic_quality": 0.3,
}
_DEFAULT_SCORING_WEIGHTS = {
    "relevance": 0.35,
    "originality": 0.25,
    "accuracy": 0.25,
    "engagement": 0.15,
}

_QUALITY_ALIASES = {
    "retries": "max_retries",
    "threshold": "min_score",
    "score_threshold": "min_score",
    "frame_count": "sample_frames",
}
_SCORING_ALIASES = {
    "version": "rubric_version",
    "dimension_weights": "weights",
    "rubric_weights": "weights",
    "threshold": "min_score",
    "minimum_score": "min_score",
    "min_weakest_dimension": "min_dimension_score",
}


def _accept_aliases(data: Any, aliases: Mapping[str, str]) -> Any:
    if not isinstance(data, Mapping):
        return data
    values = dict(data)
    for alias, canonical in aliases.items():
        if alias in values:
            value = values.pop(alias)
            values.setdefault(canonical, value)
    return values


def _validate_weights(weights: Mapping[str, float], label: str) -> dict[str, float]:
    normalised: dict[str, float] = {}
    for name, weight in weights.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{label} weight names must be non-empty")
        if not math.isfinite(weight):
            raise ValueError(f"{label} weight for {name} must be finite")
        if weight < 0:
            raise ValueError(f"{label} weight for {name} must not be negative")
        normalised[name.strip()] = float(weight)
    if not normalised:
        raise ValueError(f"{label} weights must not be empty")
    total = sum(normalised.values())
    if not math.isclose(
        total,
        _WEIGHT_SUM,
        rel_tol=0.0,
        abs_tol=_WEIGHT_TOLERANCE,
    ):
        raise ValueError(f"{label} weights must sum to {_WEIGHT_SUM:g}, got {total:g}")
    return normalised


class QualityConfig(ConfigModel):
    """Thresholds and retry policy for generated media quality assurance."""

    enabled: bool = True
    min_score: float = Field(default=0.7, ge=0, le=1)
    max_retries: int = Field(default=2, ge=0)
    sample_frames: int = Field(default=3, ge=1)
    vision: bool = True
    weights: dict[str, float] = Field(
        default_factory=lambda: dict(_DEFAULT_QUALITY_WEIGHTS)
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        return _accept_aliases(data, _QUALITY_ALIASES)

    @field_validator("weights")
    @classmethod
    def _validate_quality_weights(cls, value: dict[str, float]) -> dict[str, float]:
        return _validate_weights(value, "quality")

    @property
    def retries(self) -> int:
        """Alias for :attr:`max_retries`."""
        return self.max_retries

    @property
    def threshold(self) -> float:
        """Alias for :attr:`min_score`."""
        return self.min_score

    @property
    def vision_check(self) -> bool:
        """Alias for :attr:`vision`."""
        return self.vision

    @property
    def weight_sum(self) -> float:
        """The sum of all configured dimension weights."""
        return sum(self.weights.values())

    @property
    def dimensions(self) -> tuple[str, ...]:
        """The configured dimension names in mapping order."""
        return tuple(self.weights)


class ScoringConfig(ConfigModel):
    """Versioned rubric, dimension weights and score thresholds for research."""

    rubric_version: str = Field(default="1", min_length=1)
    weights: dict[str, float] = Field(
        default_factory=lambda: dict(_DEFAULT_SCORING_WEIGHTS)
    )
    min_score: float = Field(default=0.6, ge=0, le=1)
    min_dimension_score: float = Field(default=0.0, ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        return _accept_aliases(data, _SCORING_ALIASES)

    @field_validator("weights")
    @classmethod
    def _validate_scoring_weights(cls, value: dict[str, float]) -> dict[str, float]:
        return _validate_weights(value, "scoring")

    @property
    def version(self) -> str:
        """Alias for :attr:`rubric_version`."""
        return self.rubric_version

    @property
    def threshold(self) -> float:
        """Alias for :attr:`min_score`."""
        return self.min_score

    @property
    def weight_sum(self) -> float:
        """The sum of all configured dimension weights."""
        return sum(self.weights.values())

    @property
    def dimensions(self) -> tuple[str, ...]:
        """The configured dimension names in mapping order."""
        return tuple(self.weights)

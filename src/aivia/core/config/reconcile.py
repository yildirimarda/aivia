"""Reconciliation settings for provider-reported amounts.

A provider may report the real amount it charged after a call. This section
holds the policy that decides when that amount has drifted far enough from the
pre-call estimate to be flagged. ``drift_threshold`` is the relative fraction
of the estimate, so ``0.1`` means ten percent. ``absolute_threshold_usd`` is a
floor that keeps a tiny absolute difference from being flagged when the
estimate itself is small or zero.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from pydantic import Field, field_validator, model_validator

from aivia.core.config.base import ConfigModel

__all__ = [
    "DEFAULT_ABSOLUTE_THRESHOLD_USD",
    "DEFAULT_DRIFT_THRESHOLD",
    "ReconcileConfig",
    "ReconcileSettings",
    "ReconciliationConfig",
]

DEFAULT_DRIFT_THRESHOLD = 0.1
"""The default relative drift threshold, as a fraction of the estimate."""

DEFAULT_ABSOLUTE_THRESHOLD_USD = 0.0
"""The default absolute drift floor in USD."""

_THRESHOLD_ALIASES = (
    "threshold",
    "drift_tolerance",
    "drift_limit",
    "max_drift",
    "max_drift_ratio",
)
_ABSOLUTE_ALIASES = (
    "absolute_tolerance_usd",
    "absolute_threshold",
    "tolerance_usd",
    "drift_absolute_usd",
    "drift_floor_usd",
)


class ReconcileConfig(ConfigModel):
    """Thresholds that turn an estimate/actual difference into a drift flag."""

    drift_threshold: float = Field(
        default=DEFAULT_DRIFT_THRESHOLD,
        ge=0,
        description="Relative drift above which a reported amount is flagged.",
    )
    absolute_threshold_usd: float = Field(
        default=DEFAULT_ABSOLUTE_THRESHOLD_USD,
        ge=0,
        description="Absolute drift floor that protects small estimates.",
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for alias in _THRESHOLD_ALIASES:
            if alias in values:
                threshold = values.pop(alias)
                values.setdefault("drift_threshold", threshold)
        for alias in _ABSOLUTE_ALIASES:
            if alias in values:
                threshold = values.pop(alias)
                values.setdefault("absolute_threshold_usd", threshold)
        return values

    @field_validator("drift_threshold", "absolute_threshold_usd")
    @classmethod
    def _finite_threshold(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("reconciliation thresholds must be finite")
        return value

    @property
    def threshold(self) -> float:
        """Alias for :attr:`drift_threshold`."""
        return self.drift_threshold

    @property
    def absolute_tolerance_usd(self) -> float:
        """Alias for :attr:`absolute_threshold_usd`."""
        return self.absolute_threshold_usd

    @property
    def tolerance_usd(self) -> float:
        """Alias for :attr:`absolute_threshold_usd`."""
        return self.absolute_threshold_usd


ReconcileSettings = ReconcileConfig
ReconciliationConfig = ReconcileConfig

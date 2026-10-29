"""Budget configuration: spending caps, provider limits and overrun policy.

A cap of ``0`` means the cap is not configured; a positive value is the maximum
amount that may be spent in that scope. ``per_provider`` holds the monthly limit
for each provider. ``confirm_above`` is the estimated spend above which a run
needs explicit confirmation; ``0`` disables the confirmation check. Alert
thresholds are sorted ascending so alerts fire from the lowest threshold upward,
``on_exceed`` selects what happens when a cap is reached, and
``reservation_ttl_seconds`` is how long an abandoned reservation may hold budget
before it can be expired.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator, model_validator

from aivia.core.config.base import ConfigModel

__all__ = [
    "BudgetConfig",
    "BudgetExceedBehavior",
    "BudgetExceedBehaviour",
    "OnExceed",
    "OnExceedBehavior",
    "OnExceedBehaviour",
]


class OnExceed(StrEnum):
    """What to do when a budget cap is reached."""

    PAUSE = "pause"
    DOWNGRADE = "downgrade"
    NOTIFY = "notify"


OnExceedBehavior = OnExceed
OnExceedBehaviour = OnExceed
BudgetExceedBehavior = OnExceed
BudgetExceedBehaviour = OnExceed

_PROVIDER_ALIASES = (
    "provider_monthly",
    "provider_monthly_limits",
    "per_provider_monthly",
    "provider_limits",
    "providers",
)
_ALERT_ALIASES = ("alert_thresholds", "thresholds")
_RESERVATION_TTL_ALIASES = (
    "reservation_ttl",
    "reservation_expiry_seconds",
    "reservation_max_age_seconds",
    "reservation_timeout_seconds",
)
_CONFIRM_ALIASES = (
    "confirm_over",
    "confirmation_threshold",
    "require_confirmation_above",
)
_PROVIDER_LIMIT_KEYS = ("monthly", "per_month", "limit", "cap", "value")


def _normalise_provider_limits(value: Any) -> Any:
    """Flatten ``provider: {monthly: limit}`` entries to ``provider: limit``."""
    if not isinstance(value, Mapping):
        return value
    normalised: dict[Any, Any] = {}
    for name, limit in value.items():
        if isinstance(limit, Mapping):
            for key in _PROVIDER_LIMIT_KEYS:
                if key in limit:
                    limit = limit[key]
                    break
        normalised[name] = limit
    return normalised


class BudgetConfig(ConfigModel):
    """Spending caps, provider monthly limits, alert thresholds and policy."""

    per_video: float = Field(default=0.0, ge=0)
    per_day: float = Field(default=0.0, ge=0)
    per_month: float = Field(default=0.0, ge=0)
    per_order: float = Field(default=0.0, ge=0)
    per_instruction: float = Field(default=0.0, ge=0)
    per_run: float = Field(default=0.0, ge=0)
    per_call: float = Field(default=0.0, ge=0)
    confirm_above: float = Field(default=0.0, ge=0)
    per_provider: dict[str, float] = Field(default_factory=dict)
    alerts: list[float] = Field(default_factory=list)
    reservation_ttl_seconds: float = Field(default=3600.0, ge=0)
    on_exceed: OnExceed = OnExceed.PAUSE

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for alias in _PROVIDER_ALIASES:
            if alias in values:
                provider_limits = values.pop(alias)
                values.setdefault("per_provider", provider_limits)
        for alias in _ALERT_ALIASES:
            if alias in values:
                thresholds = values.pop(alias)
                values.setdefault("alerts", thresholds)
        for alias in _CONFIRM_ALIASES:
            if alias in values:
                threshold = values.pop(alias)
                values.setdefault("confirm_above", threshold)
        for alias in _RESERVATION_TTL_ALIASES:
            if alias in values:
                ttl = values.pop(alias)
                if isinstance(ttl, timedelta):
                    ttl = ttl.total_seconds()
                values.setdefault("reservation_ttl_seconds", ttl)
        if "per_provider" in values:
            values["per_provider"] = _normalise_provider_limits(values["per_provider"])
        return values

    @field_validator("per_provider")
    @classmethod
    def _validate_provider_limits(cls, value: dict[str, float]) -> dict[str, float]:
        for name, limit in value.items():
            if not name.strip():
                raise ValueError("provider name must not be empty")
            if limit < 0:
                raise ValueError("provider monthly limit must not be negative")
        return value

    @field_validator("alerts")
    @classmethod
    def _sort_alerts(cls, value: list[float]) -> list[float]:
        if any(threshold < 0 for threshold in value):
            raise ValueError("alert thresholds must not be negative")
        return sorted(value)

    @property
    def provider_monthly(self) -> dict[str, float]:
        """Alias for :attr:`per_provider`, the monthly limit per provider."""
        return self.per_provider

    @property
    def provider_monthly_limits(self) -> dict[str, float]:
        """Alias for :attr:`per_provider`, the monthly limit per provider."""
        return self.per_provider

    @property
    def alert_thresholds(self) -> list[float]:
        """Alias for :attr:`alerts`, sorted ascending."""
        return self.alerts

    @property
    def confirmation_threshold(self) -> float:
        """Alias for :attr:`confirm_above`, the spend that needs confirmation."""
        return self.confirm_above

    @property
    def reservation_ttl(self) -> float:
        """Alias for :attr:`reservation_ttl_seconds`."""
        return self.reservation_ttl_seconds

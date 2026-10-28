"""Circuit breaker settings for provider selection.

The defaults live here with the rest of the configuration so the core router
contains policy but no tuning literals. ``failure_threshold`` is the number of
consecutive failures that opens a provider circuit, and ``cooldown_seconds`` is
how long it stays open before one half-open probe is allowed.
"""

from __future__ import annotations

import math

from pydantic import Field, field_validator

from aivia.core.config.base import ConfigModel

__all__ = [
    "CircuitBreakerConfig",
    "CircuitConfig",
]


class CircuitBreakerConfig(ConfigModel):
    """Failure threshold and cooldown for one provider circuit."""

    failure_threshold: int = Field(default=5, ge=1)
    cooldown_seconds: float = Field(default=60.0, ge=0)

    @field_validator("cooldown_seconds")
    @classmethod
    def _validate_cooldown(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("circuit cooldown must be finite")
        return value


CircuitConfig = CircuitBreakerConfig

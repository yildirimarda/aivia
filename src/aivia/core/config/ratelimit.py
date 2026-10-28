"""Token bucket rate limiter settings for provider calls.

The limiter itself lives in :mod:`aivia.core.ratelimit`; this module only holds
the policy values so the core contains no tuning literals. ``on_limit`` selects
whether a caller waits for the next token or receives a
:class:`~aivia.core.errors.RateLimitError` immediately. The default request rate
is used only when a provider does not declare its own limit.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from aivia.core.config.base import ConfigModel

__all__ = [
    "OnLimit",
    "OnLimitBehavior",
    "OnLimitBehaviour",
    "OnRateLimit",
    "RateLimitBehavior",
    "RateLimitBehaviour",
    "RateLimiterConfig",
    "RateLimiterSettings",
]


class OnRateLimit(StrEnum):
    """What a token bucket does when the next token is not available yet."""

    WAIT = "wait"
    RAISE = "raise"


OnLimit = OnRateLimit
OnLimitBehavior = OnRateLimit
OnLimitBehaviour = OnRateLimit
RateLimitBehavior = OnRateLimit
RateLimitBehaviour = OnRateLimit


class RateLimiterConfig(ConfigModel):
    """Default rate and on-limit policy for a provider token bucket.

    ``requests_per_minute`` is the sustained rate used when a provider does not
    declare its own limit. ``burst`` is the largest short-term allowance and
    ``0`` falls back to the sustained rate. ``on_limit`` chooses between
    waiting for the next token and raising a rate limited error.
    """

    requests_per_minute: int = Field(default=60, ge=0)
    burst: int = Field(default=0, ge=0)
    on_limit: OnRateLimit = OnRateLimit.WAIT

    @property
    def effective_burst(self) -> int:
        """The burst allowance, falling back to the sustained rate."""
        return self.burst or self.requests_per_minute


RateLimiterSettings = RateLimiterConfig

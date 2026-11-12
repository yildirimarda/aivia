"""Scope runs, orders and posts to an owner with optional per-owner budgets
and rate limits.

An owner is a named principal (a person, team or tenant) whose runs, orders and
posts are attributed to them. Each owner may declare optional per-owner budget
caps and rate limits. When no owner is configured -- the single-user default --
behaviour is unchanged: budgets fall back to the global config and no
per-owner rate limit applies.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from aivia.core.budget import BudgetGuard
from aivia.core.config.budget import BudgetConfig
from aivia.core.config.ratelimit import RateLimiterConfig
from aivia.core.ratelimit import TokenBucket

__all__ = [
    "OwnerConfig",
    "OwnerScope",
    "budget_guard_for_owner",
    "rate_limiter_for_owner",
]


@dataclass(frozen=True, slots=True)
class OwnerConfig:
    """Optional per-owner budget and rate limit policy.

    Both fields default to ``None``, meaning the owner inherits the global
    configuration. Set ``budget`` to cap an owner's spend and ``rate_limit`` to
    throttle an owner's calls.
    """

    budget: BudgetConfig | None = None
    rate_limit: RateLimiterConfig | None = None


class OwnerScope:
    """Attribute runs, orders and posts to an owner and apply optional per-owner caps.

    Pass a mapping of owner name to :class:`OwnerConfig`. An owner that is not
    present inherits the global configuration. An empty owner name means
    single-user mode: no per-owner caps apply.
    """

    def __init__(self, owners: Mapping[str, OwnerConfig] | None = None) -> None:
        self._owners: dict[str, OwnerConfig] = {
            str(name).strip(): config for name, config in (owners or {}).items()
        }

    def config_for(self, owner: str) -> OwnerConfig | None:
        """Return the per-owner config for *owner*, or ``None`` when unconfigured."""
        name = owner.strip() if isinstance(owner, str) else ""
        return self._owners.get(name) or None

    def budget_guard_for(
        self,
        owner: str,
        tracker: Any | None = None,
    ) -> BudgetGuard:
        """Return a :class:`~aivia.core.budget.BudgetGuard` for *owner*.

        When *owner* has a configured budget, the guard enforces those caps;
        otherwise it enforces the global defaults. An empty owner never applies
        per-owner caps, so single-user behaviour is unchanged.
        """
        config = self.config_for(owner)
        budget_config = config.budget if config is not None else None
        return BudgetGuard(budget_config, tracker)

    def rate_limiter_for(
        self,
        owner: str,
        clock: Any | None = None,
        sleep: Any | None = None,
    ) -> TokenBucket | None:
        """Return a :class:`~aivia.core.ratelimit.TokenBucket` for *owner*.

        Returns ``None`` when the owner has no configured rate limit (including
        single-user mode), so callers proceed without rate limiting.
        """
        config = self.config_for(owner)
        if config is None or config.rate_limit is None:
            return None
        rate_limit = config.rate_limit
        return TokenBucket(
            requests_per_minute=float(rate_limit.requests_per_minute),
            burst=rate_limit.burst,
            on_limit=rate_limit.on_limit,
            clock=clock,
            sleep=sleep,
            name=owner or None,
        )


def budget_guard_for_owner(
    scope: OwnerScope,
    owner: str,
    tracker: Any | None = None,
) -> BudgetGuard:
    """Return a budget guard scoped to *owner* (module-level convenience)."""
    return scope.budget_guard_for(owner, tracker)


def rate_limiter_for_owner(
    scope: OwnerScope,
    owner: str,
    clock: Any | None = None,
    sleep: Any | None = None,
) -> TokenBucket | None:
    """Return a rate limiter scoped to *owner* (module-level convenience)."""
    return scope.rate_limiter_for(owner, clock, sleep)

"""Enforce spending caps before a paid call is made.

The :class:`BudgetGuard` is the read side of the budget policy. A caller hands
it the amount a call is expected to cost and the run it belongs to; the guard
compares that estimate with the configured per-call and per-run caps and raises
:class:`~aivia.core.errors.BudgetError` when the call would exceed either one.
The check runs before the provider is contacted, so an over-budget call never
spends money.

A cap of ``0`` means the cap is not configured and never blocks a call. The
guard reads spend from the cost ledger through
:class:`~aivia.core.cost.CostTracker`; it never writes a cost event, resolves a
price or calls a provider. The media service composes those responsibilities.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Engine

from aivia.core.config.budget import BudgetConfig
from aivia.core.cost import CostTracker
from aivia.core.db.repos import CostEventRepository
from aivia.core.errors import BudgetError, ConfigError

__all__ = [
    "BudgetCheck",
    "BudgetChecker",
    "BudgetController",
    "BudgetEnforcer",
    "BudgetGuard",
    "BudgetManager",
    "BudgetResult",
    "BudgetStatus",
    "check_budget",
    "enforce_budget",
    "ensure_budget",
    "would_exceed_budget",
]

_PER_CALL = "per_call"
_PER_RUN = "per_run"


def _exceeds(amount: float, cap: float) -> bool:
    """Compare money without rejecting a call that is only float noise over."""
    return amount > cap and not math.isclose(
        amount,
        cap,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


def _coerce_amount(value: object, *, label: str = "amount_usd") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            f"{label} must be a finite, non-negative number",
            details={label: value},
        )
    amount = float(value)
    if not math.isfinite(amount) or amount < 0:
        raise ConfigError(
            f"{label} must be a finite, non-negative number",
            details={label: value},
        )
    return amount


def _coerce_run_id(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ConfigError(
            "run_id must be a string",
            details={"run_id": value},
        )
    return value.strip()


def _coerce_optional_name(value: object, *, label: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ConfigError(
            f"{label} must be a string",
            details={label: value},
        )
    return value.strip()


def _coerce_config(value: object) -> BudgetConfig:
    if value is None:
        return BudgetConfig()
    if isinstance(value, BudgetConfig):
        return value
    if isinstance(value, Mapping):
        return BudgetConfig.from_mapping(value)
    raise ConfigError(
        "budget config must be a BudgetConfig or a mapping",
        details={"config": type(value).__name__},
    )


def _coerce_tracker(value: object) -> CostTracker | None:
    if value is None:
        return None
    if isinstance(value, CostTracker):
        return value
    if isinstance(value, CostEventRepository):
        return CostTracker(repository=value)
    if isinstance(value, Engine):
        return CostTracker(value)
    raise ConfigError(
        "cost tracker must be a CostTracker, repository or engine",
        details={"tracker": type(value).__name__},
    )


@dataclass(frozen=True, slots=True)
class BudgetCheck:
    """The outcome of evaluating one estimated call against the active caps.

    ``scope`` names the cap that rejected the call, or is ``None`` when the
    call is allowed. ``spent_usd`` is the spend already recorded for the run
    and ``remaining_usd`` is how much the call may still cost after applying
    every active cap.
    """

    run_id: str
    amount_usd: float
    spent_usd: float
    per_call_cap: float = 0.0
    per_run_cap: float = 0.0
    scope: str | None = None
    provider: str = ""
    operation: str = ""

    @property
    def allowed(self) -> bool:
        """Whether the estimated call fits inside every configured cap."""
        return self.scope is None

    @property
    def exceeded(self) -> bool:
        """Whether a cap rejected the estimated call."""
        return self.scope is not None

    @property
    def projected_usd(self) -> float:
        """Spend for the run if this call is made."""
        return self.spent_usd + self.amount_usd

    @property
    def remaining_usd(self) -> float | None:
        """The most this call may cost, or ``None`` when no cap is configured."""
        limits: list[float] = []
        if self.per_call_cap:
            limits.append(self.per_call_cap)
        if self.per_run_cap:
            limits.append(max(self.per_run_cap - self.spent_usd, 0.0))
        if not limits:
            return None
        return min(limits)

    @property
    def remaining(self) -> float | None:
        """Alias for :attr:`remaining_usd`."""
        return self.remaining_usd

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for logs and errors."""
        return {
            "run_id": self.run_id,
            "provider": self.provider,
            "operation": self.operation,
            "amount_usd": self.amount_usd,
            "spent_usd": self.spent_usd,
            "projected_usd": self.projected_usd,
            "per_call_cap": self.per_call_cap,
            "per_run_cap": self.per_run_cap,
            "remaining_usd": self.remaining_usd,
            "scope": self.scope,
            "allowed": self.allowed,
        }


BudgetResult = BudgetCheck
BudgetStatus = BudgetCheck


class BudgetGuard:
    """Check a call estimate against the per-call and per-run caps.

    Pass a :class:`~aivia.core.config.budget.BudgetConfig` (or a mapping using
    its aliases) and the cost ledger to read spend from. The ledger may be a
    :class:`~aivia.core.cost.CostTracker`, a
    :class:`~aivia.core.db.repos.CostEventRepository` or an SQLAlchemy engine.
    A guard without a ledger can still enforce the per-call cap; the per-run
    cap needs the ledger to know what the run has already spent.

    ``evaluate`` returns a :class:`BudgetCheck` without raising, while
    ``check`` and its aliases raise :class:`~aivia.core.errors.BudgetError`
    before the paid call is made.
    """

    def __init__(
        self,
        config: BudgetConfig | Mapping[str, Any] | None = None,
        tracker: CostTracker | CostEventRepository | Engine | None = None,
        *,
        engine: Engine | None = None,
        repository: CostEventRepository | None = None,
        cost: CostTracker | None = None,
        ledger: CostTracker | None = None,
    ) -> None:
        if isinstance(config, (CostTracker, CostEventRepository, Engine)) and (
            isinstance(tracker, (BudgetConfig, Mapping)) or tracker is None
        ):
            config, tracker = tracker, config

        supplied = [
            item
            for item in (tracker, engine, repository, cost, ledger)
            if item is not None
        ]
        if len(supplied) > 1:
            raise TypeError("pass only one cost tracker, repository or engine")

        self._config = _coerce_config(config)
        self._tracker = _coerce_tracker(supplied[0] if supplied else None)

    @property
    def config(self) -> BudgetConfig:
        """The caps this guard enforces."""
        return self._config

    @property
    def tracker(self) -> CostTracker | None:
        """The cost ledger spend is read from, when one was supplied."""
        return self._tracker

    def spent(self, run_id: str | None = None) -> float:
        """Return the amount already recorded for *run_id*.

        An empty or omitted run id reads the events with no run attached. A
        guard without a ledger reports zero spend.
        """
        target = _coerce_run_id(run_id)
        if self._tracker is None:
            return 0.0
        return self._tracker.total(run_id=target)

    def evaluate(
        self,
        amount_usd: float | None = None,
        *,
        run_id: str | None = None,
        provider: str | None = None,
        operation: str | None = None,
        estimate: float | None = None,
        amount: float | None = None,
        cost_usd: float | None = None,
    ) -> BudgetCheck:
        """Evaluate an estimated call without raising on an exceeded cap."""
        if amount_usd is None:
            for candidate in (estimate, amount, cost_usd):
                if candidate is not None:
                    amount_usd = candidate
                    break
        if amount_usd is None:
            raise ConfigError(
                "a budget check needs the estimated amount",
                details={"amount_usd": None},
            )
        value = _coerce_amount(amount_usd)
        target = _coerce_run_id(run_id)
        per_call = self._config.per_call
        per_run = self._config.per_run
        if per_run > 0 and self._tracker is None:
            raise ConfigError(
                "a per-run budget cap needs a cost ledger",
                details={"per_run": per_run},
            )
        spent = self.spent(target)

        scope: str | None = None
        if per_call > 0 and _exceeds(value, per_call):
            scope = _PER_CALL
        elif per_run > 0 and _exceeds(spent + value, per_run):
            scope = _PER_RUN

        return BudgetCheck(
            run_id=target,
            amount_usd=value,
            spent_usd=spent,
            per_call_cap=per_call,
            per_run_cap=per_run,
            scope=scope,
            provider=_coerce_optional_name(provider, label="provider"),
            operation=_coerce_optional_name(operation, label="operation"),
        )

    def check(
        self,
        amount_usd: float | None = None,
        *,
        run_id: str | None = None,
        provider: str | None = None,
        operation: str | None = None,
        estimate: float | None = None,
        amount: float | None = None,
        cost_usd: float | None = None,
    ) -> BudgetCheck:
        """Raise :class:`~aivia.core.errors.BudgetError` when a cap is exceeded.

        The returned :class:`BudgetCheck` describes the allowed call, so a
        caller can log the remaining budget alongside the provider request.
        """
        result = self.evaluate(
            amount_usd,
            run_id=run_id,
            provider=provider,
            operation=operation,
            estimate=estimate,
            amount=amount,
            cost_usd=cost_usd,
        )
        if result.allowed:
            return result

        if result.scope == _PER_CALL:
            cap = result.per_call_cap
            message = (
                f"estimated call cost {result.amount_usd:.6g} USD exceeds the "
                f"per-call cap of {cap:.6g} USD"
            )
        else:
            cap = result.per_run_cap
            message = (
                f"estimated call cost {result.amount_usd:.6g} USD would bring "
                f"run {result.run_id or '<none>'} spend to "
                f"{result.projected_usd:.6g} USD, above the per-run cap of "
                f"{cap:.6g} USD"
            )
        raise BudgetError(
            message,
            details={
                "scope": result.scope,
                "cap": cap,
                "amount_usd": result.amount_usd,
                "requested": result.amount_usd,
                "spent_usd": result.spent_usd,
                "spent": result.spent_usd,
                "projected_usd": result.projected_usd,
                "remaining_usd": result.remaining_usd,
                "remaining": result.remaining_usd,
                "run_id": result.run_id,
                "provider": result.provider,
                "operation": result.operation,
            },
        )

    def ensure(self, amount_usd: float | None = None, **kwargs: Any) -> BudgetCheck:
        """Alias for :meth:`check`."""
        return self.check(amount_usd, **kwargs)

    def enforce(self, amount_usd: float | None = None, **kwargs: Any) -> BudgetCheck:
        """Alias for :meth:`check`."""
        return self.check(amount_usd, **kwargs)

    def authorize(self, amount_usd: float | None = None, **kwargs: Any) -> BudgetCheck:
        """Alias for :meth:`check`."""
        return self.check(amount_usd, **kwargs)

    def check_call(self, amount_usd: float | None = None, **kwargs: Any) -> BudgetCheck:
        """Alias for :meth:`check`."""
        return self.check(amount_usd, **kwargs)

    def reserve(self, amount_usd: float | None = None, **kwargs: Any) -> BudgetCheck:
        """Alias for :meth:`check`."""
        return self.check(amount_usd, **kwargs)

    def check_budget(
        self, amount_usd: float | None = None, **kwargs: Any
    ) -> BudgetCheck:
        """Alias for :meth:`check`."""
        return self.check(amount_usd, **kwargs)

    def enforce_budget(
        self, amount_usd: float | None = None, **kwargs: Any
    ) -> BudgetCheck:
        """Alias for :meth:`check`."""
        return self.check(amount_usd, **kwargs)

    def would_exceed_budget(
        self,
        amount_usd: float | None = None,
        **kwargs: Any,
    ) -> bool:
        """Alias for :meth:`would_exceed`."""
        return self.would_exceed(amount_usd, **kwargs)

    def would_exceed(
        self,
        amount_usd: float | None = None,
        **kwargs: Any,
    ) -> bool:
        """Return whether the estimated call exceeds an active cap."""
        return self.evaluate(amount_usd, **kwargs).exceeded

    def remaining(self, run_id: str | None = None) -> float | None:
        """Return the most a call may cost for *run_id*, if a cap applies."""
        return self.evaluate(0.0, run_id=run_id).remaining_usd


BudgetEnforcer = BudgetGuard
BudgetChecker = BudgetGuard
BudgetManager = BudgetGuard
BudgetController = BudgetGuard


def _guard(
    config: BudgetConfig | Mapping[str, Any] | None,
    tracker: CostTracker | CostEventRepository | Engine | None,
) -> BudgetGuard:
    return BudgetGuard(config, tracker)


def check_budget(
    amount_usd: float,
    *,
    run_id: str | None = None,
    config: BudgetConfig | Mapping[str, Any] | None = None,
    tracker: CostTracker | CostEventRepository | Engine | None = None,
    provider: str | None = None,
    operation: str | None = None,
) -> BudgetCheck:
    """Check one call estimate without constructing a long-lived guard."""
    return _guard(config, tracker).check(
        amount_usd,
        run_id=run_id,
        provider=provider,
        operation=operation,
    )


def enforce_budget(
    amount_usd: float,
    *,
    run_id: str | None = None,
    config: BudgetConfig | Mapping[str, Any] | None = None,
    tracker: CostTracker | CostEventRepository | Engine | None = None,
    provider: str | None = None,
    operation: str | None = None,
) -> BudgetCheck:
    """Alias for :func:`check_budget`."""
    return check_budget(
        amount_usd,
        run_id=run_id,
        config=config,
        tracker=tracker,
        provider=provider,
        operation=operation,
    )


def ensure_budget(
    amount_usd: float,
    *,
    run_id: str | None = None,
    config: BudgetConfig | Mapping[str, Any] | None = None,
    tracker: CostTracker | CostEventRepository | Engine | None = None,
    provider: str | None = None,
    operation: str | None = None,
) -> BudgetCheck:
    """Alias for :func:`check_budget`."""
    return check_budget(
        amount_usd,
        run_id=run_id,
        config=config,
        tracker=tracker,
        provider=provider,
        operation=operation,
    )


def would_exceed_budget(
    amount_usd: float,
    *,
    run_id: str | None = None,
    config: BudgetConfig | Mapping[str, Any] | None = None,
    tracker: CostTracker | CostEventRepository | Engine | None = None,
    provider: str | None = None,
    operation: str | None = None,
) -> bool:
    """Return whether a call estimate exceeds an active cap."""
    return _guard(config, tracker).would_exceed(
        amount_usd,
        run_id=run_id,
        provider=provider,
        operation=operation,
    )

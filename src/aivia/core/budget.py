"""Enforce spending caps before a paid call is made.

The :class:`BudgetGuard` is the read side of the budget policy. A caller hands
it the amount a call is expected to cost and the run it belongs to; the guard
compares that estimate with the configured per-call, per-run, per-day,
per-month and per-provider monthly caps and raises
:class:`~aivia.core.errors.BudgetError` when the call would exceed any one of
them. The check runs before the provider is contacted, so an over-budget call
never spends money.

Day and month spend is read from the cost events whose ``created_at`` falls in
the current UTC day or month. A provider monthly limit narrows the same month
window to the provider making the call.

:meth:`BudgetGuard.decide` applies the configured ``on_exceed`` behaviour to an
exceeded check. ``downgrade`` picks the cheapest affordable candidate provider,
``pause`` moves the run to ``paused_budget`` and ``notify`` lets the call
proceed while returning a notification. A downgrade with no affordable
candidate falls back to pausing so no paid call can bypass the cap.

A cap of ``0`` means the cap is not configured and never blocks a call. The
guard reads spend from the cost ledger through
:class:`~aivia.core.cost.CostTracker`. :meth:`BudgetGuard.check` never writes a
cost event, while :meth:`BudgetGuard.reserve` records the estimate as an
estimated event in the same transaction that checks it, so concurrent calls
cannot both pass the same cap. The guard never resolves a price or calls a
provider; the media service composes those responsibilities.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from enum import StrEnum
from typing import Any, NoReturn

from sqlalchemy import Connection, Engine, func, select

from aivia.core.budget_alerts import (
    AlertPeriod,
    BudgetAlert,
    BudgetAlertEmitter,
    BudgetAlertManager,
    BudgetAlertNotifier,
    BudgetAlertPeriod,
    BudgetAlertState,
    BudgetAlertTracker,
    ConsoleAlertNotifier,
    ConsoleNotifier,
    emit_budget_alerts,
)
from aivia.core.config.budget import BudgetConfig, OnExceed
from aivia.core.cost import CostTracker
from aivia.core.cost_agg import CostAggregator, CostGroup, _not_replaced
from aivia.core.db.repos import CostEventRepository
from aivia.core.db.tables import cost_events
from aivia.core.errors import BudgetError, ConfigError
from aivia.core.models.base import utcnow
from aivia.core.models.cost import CostEvent
from aivia.core.models.run import Run, RunState

__all__ = [
    "AlertPeriod",
    "BudgetAction",
    "BudgetAlert",
    "BudgetAlertEmitter",
    "BudgetAlertManager",
    "BudgetAlertNotifier",
    "BudgetAlertPeriod",
    "BudgetAlertState",
    "BudgetAlertTracker",
    "BudgetCheck",
    "BudgetChecker",
    "BudgetController",
    "BudgetDecision",
    "BudgetEnforcer",
    "BudgetGuard",
    "BudgetManager",
    "BudgetOutcome",
    "BudgetPolicy",
    "BudgetResolution",
    "BudgetResult",
    "BudgetStatus",
    "ConsoleAlertNotifier",
    "ConsoleNotifier",
    "apply_budget_policy",
    "apply_on_exceed",
    "check_budget",
    "decide_budget",
    "emit_budget_alerts",
    "enforce_budget",
    "ensure_budget",
    "resolve_budget",
    "would_exceed_budget",
]

_PER_CALL = "per_call"
_PER_RUN = "per_run"
_PER_DAY = "per_day"
_PER_MONTH = "per_month"
_PER_PROVIDER = "per_provider"


def _exceeds(amount: float, cap: float) -> bool:
    """Compare money without rejecting a call that is only float noise over."""
    return amount > cap and not math.isclose(
        amount,
        cap,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


def _coerce_now(value: object) -> datetime:
    """Return *value* as an aware UTC datetime, defaulting to the current time."""
    if value is None:
        return utcnow()
    if not isinstance(value, datetime):
        raise ConfigError(
            "budget timestamp must be a datetime",
            details={"now": value},
        )
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _day_window(now: datetime) -> tuple[datetime, datetime]:
    """Return the half-open UTC window covering *now*'s calendar day."""
    start = datetime.combine(now.date(), time.min, tzinfo=UTC)
    return start, start + timedelta(days=1)


def _month_window(now: datetime) -> tuple[datetime, datetime]:
    """Return the half-open UTC window covering *now*'s calendar month."""
    start = datetime(now.year, now.month, 1, tzinfo=UTC)
    if now.month == 12:
        end = datetime(now.year + 1, 1, 1, tzinfo=UTC)
    else:
        end = datetime(now.year, now.month + 1, 1, tzinfo=UTC)
    return start, end


def _sum_events(
    connection: Connection,
    *,
    run_id: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    provider: str | None = None,
) -> float:
    """Sum cost events on *connection* over a run or half-open time window."""
    statement = select(func.coalesce(func.sum(cost_events.c.amount_usd), 0.0))
    statement = statement.where(_not_replaced())
    if run_id is not None:
        statement = statement.where(cost_events.c.run_id == run_id)
    if start is not None:
        statement = statement.where(cost_events.c.created_at >= start)
    if end is not None:
        statement = statement.where(cost_events.c.created_at < end)
    if provider is not None:
        statement = statement.where(cost_events.c.provider == provider)
    return float(connection.execute(statement).scalar_one() or 0.0)


@contextmanager
def _write_transaction(engine: Engine) -> Iterator[Connection]:
    """Open a transaction that takes the write lock before any spend is read.

    SQLite starts a deferred transaction by default, so two reservations could
    both read the same spend and only then try to write. ``BEGIN IMMEDIATE``
    acquires the write lock first, making the read and the insert one atomic
    step and serialising concurrent reservations.
    """
    with engine.begin() as connection:
        if engine.dialect.name == "sqlite":
            connection.exec_driver_sql("BEGIN IMMEDIATE")
        yield connection


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
    per_day_cap: float = 0.0
    per_month_cap: float = 0.0
    per_provider_cap: float = 0.0
    day_spent_usd: float = 0.0
    month_spent_usd: float = 0.0
    provider_spent_usd: float = 0.0
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
    def day_projected_usd(self) -> float:
        """Spend for the current UTC day if this call is made."""
        return self.day_spent_usd + self.amount_usd

    @property
    def month_projected_usd(self) -> float:
        """Spend for the current UTC month if this call is made."""
        return self.month_spent_usd + self.amount_usd

    @property
    def provider_projected_usd(self) -> float:
        """Spend for the provider in the current UTC month if this call is made."""
        return self.provider_spent_usd + self.amount_usd

    @property
    def day_spent(self) -> float:
        """Alias for :attr:`day_spent_usd`."""
        return self.day_spent_usd

    @property
    def month_spent(self) -> float:
        """Alias for :attr:`month_spent_usd`."""
        return self.month_spent_usd

    @property
    def provider_spent(self) -> float:
        """Alias for :attr:`provider_spent_usd`."""
        return self.provider_spent_usd

    @property
    def per_provider_monthly_cap(self) -> float:
        """Alias for :attr:`per_provider_cap`."""
        return self.per_provider_cap

    @property
    def remaining_usd(self) -> float | None:
        """The most this call may cost, or ``None`` when no cap is configured."""
        limits: list[float] = []
        if self.per_call_cap:
            limits.append(self.per_call_cap)
        if self.per_run_cap:
            limits.append(max(self.per_run_cap - self.spent_usd, 0.0))
        if self.per_day_cap:
            limits.append(max(self.per_day_cap - self.day_spent_usd, 0.0))
        if self.per_month_cap:
            limits.append(max(self.per_month_cap - self.month_spent_usd, 0.0))
        if self.per_provider_cap:
            limits.append(max(self.per_provider_cap - self.provider_spent_usd, 0.0))
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
            "per_day_cap": self.per_day_cap,
            "per_month_cap": self.per_month_cap,
            "per_provider_cap": self.per_provider_cap,
            "day_spent_usd": self.day_spent_usd,
            "month_spent_usd": self.month_spent_usd,
            "provider_spent_usd": self.provider_spent_usd,
            "remaining_usd": self.remaining_usd,
            "scope": self.scope,
            "allowed": self.allowed,
        }


BudgetResult = BudgetCheck
BudgetStatus = BudgetCheck


class BudgetAction(StrEnum):
    """What a caller should do after a budget check.

    ``PROCEED`` and ``NOTIFY`` allow the call, ``DOWNGRADE`` allows it on a
    cheaper provider and ``PAUSE`` stops the run until a human resumes it.
    """

    PROCEED = "proceed"
    DOWNGRADE = "downgrade"
    PAUSE = "pause"
    NOTIFY = "notify"


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    """The configured ``on_exceed`` behaviour applied to one budget check.

    ``allowed`` tells the caller whether it may make the paid call. A
    downgrade names the cheaper ``provider``; a pause carries the updated
    ``run`` in ``paused_budget`` state; a notification is available through
    ``notifications`` and never blocks the call.
    """

    check: BudgetCheck
    action: BudgetAction = BudgetAction.PROCEED
    behaviour: OnExceed | None = None
    provider: str = ""
    previous_provider: str = ""
    estimate_usd: float | None = None
    run: Run | None = None
    notifications: tuple[str, ...] = ()
    message: str = ""

    @property
    def exceeded(self) -> bool:
        """Whether the underlying check exceeded a cap."""
        return self.check.exceeded

    @property
    def allowed(self) -> bool:
        """Whether the paid call may proceed."""
        return self.action in {
            BudgetAction.PROCEED,
            BudgetAction.DOWNGRADE,
            BudgetAction.NOTIFY,
        }

    @property
    def blocked(self) -> bool:
        """Whether the call must not proceed."""
        return not self.allowed

    @property
    def paused(self) -> bool:
        """Whether the run was paused on budget."""
        return self.action is BudgetAction.PAUSE

    @property
    def downgraded(self) -> bool:
        """Whether a cheaper provider was selected."""
        return self.action is BudgetAction.DOWNGRADE

    @property
    def notified(self) -> bool:
        """Whether the call proceeds with a notification."""
        return self.action is BudgetAction.NOTIFY

    @property
    def notify(self) -> bool:
        """Alias for :attr:`notified`."""
        return self.notified

    @property
    def alerts(self) -> tuple[str, ...]:
        """Alias for :attr:`notifications`."""
        return self.notifications

    @property
    def run_state(self) -> RunState | None:
        """The state the run has after applying the decision."""
        if self.run is not None:
            return self.run.state
        if self.paused:
            return RunState.PAUSED_BUDGET
        return None

    @property
    def paused_run(self) -> Run | None:
        """The paused run, when the decision paused one."""
        return self.run if self.paused else None

    @property
    def reason(self) -> str:
        """Alias for :attr:`message`."""
        return self.message

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for logs and alerts."""
        return {
            "action": self.action.value,
            "behaviour": self.behaviour.value if self.behaviour else None,
            "allowed": self.allowed,
            "blocked": self.blocked,
            "provider": self.provider,
            "previous_provider": self.previous_provider,
            "estimate_usd": self.estimate_usd,
            "run_id": self.check.run_id,
            "run_state": self.run_state.value if self.run_state else None,
            "notifications": list(self.notifications),
            "message": self.message,
            "check": self.check.to_dict(),
        }


BudgetOutcome = BudgetDecision
BudgetResolution = BudgetDecision


def _coerce_candidates(
    value: Mapping[str, float] | Iterable[tuple[str, float]] | None,
) -> list[tuple[str, float]]:
    """Validate candidate provider estimates in the order they were supplied."""
    if value is None:
        return []
    entries: Iterable[tuple[str, float]]
    entries = value.items() if isinstance(value, Mapping) else value
    candidates: list[tuple[str, float]] = []
    for entry in entries:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ConfigError(
                "budget candidates must be (provider, estimate) pairs",
                details={"candidate": entry},
            )
        name, estimate = entry
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(
                "candidate provider names must be non-empty strings",
                details={"provider": name},
            )
        candidates.append(
            (
                name.strip(),
                _coerce_amount(estimate, label="candidate estimate"),
            )
        )
    return candidates


def _exceeded_message(result: BudgetCheck) -> str:
    """Describe an exceeded check in the same words as :meth:`BudgetGuard.check`."""
    if result.scope == _PER_CALL:
        return (
            f"estimated call cost {result.amount_usd:.6g} USD exceeds the "
            f"per-call cap of {result.per_call_cap:.6g} USD"
        )
    if result.scope == _PER_DAY:
        return (
            f"estimated call cost {result.amount_usd:.6g} USD would bring "
            f"day spend to {result.day_projected_usd:.6g} USD, above the "
            f"per-day cap of {result.per_day_cap:.6g} USD"
        )
    if result.scope == _PER_MONTH:
        return (
            f"estimated call cost {result.amount_usd:.6g} USD would bring "
            f"month spend to {result.month_projected_usd:.6g} USD, above the "
            f"per-month cap of {result.per_month_cap:.6g} USD"
        )
    if result.scope == _PER_PROVIDER:
        return (
            f"estimated call cost {result.amount_usd:.6g} USD would bring "
            f"provider {result.provider or '<none>'} month spend to "
            f"{result.provider_projected_usd:.6g} USD, above the provider "
            f"monthly cap of {result.per_provider_cap:.6g} USD"
        )
    return (
        f"estimated call cost {result.amount_usd:.6g} USD would bring "
        f"run {result.run_id or '<none>'} spend to "
        f"{result.projected_usd:.6g} USD, above the per-run cap of "
        f"{result.per_run_cap:.6g} USD"
    )


def _pause_run(
    run: Run | None, result: BudgetCheck, now: datetime | None
) -> Run | None:
    """Return *run* moved to ``paused_budget``, or ``None`` when no run is given."""
    if run is None:
        return None
    moment = now if now is not None else utcnow()
    if not isinstance(moment, datetime):
        raise ConfigError(
            "pause timestamp must be a datetime",
            details={"now": moment},
        )
    return run.model_copy(
        update={
            "state": RunState.PAUSED_BUDGET,
            "paused_at": moment,
            "updated_at": moment,
            "pause_reason": f"budget: {_exceeded_message(result)}",
        }
    )


class BudgetGuard:
    """Check a call estimate against every configured spending cap.

    Pass a :class:`~aivia.core.config.budget.BudgetConfig` (or a mapping using
    its aliases) and the cost ledger to read spend from. The ledger may be a
    :class:`~aivia.core.cost.CostTracker`, a
    :class:`~aivia.core.db.repos.CostEventRepository` or an SQLAlchemy engine.
    A guard without a ledger can still enforce the per-call cap; the per-run,
    per-day, per-month and per-provider caps need the ledger to know what has
    already been spent.

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

    def _period_spent(
        self,
        start: datetime,
        end: datetime,
        *,
        provider: str | None = None,
    ) -> float:
        """Sum the ledger over the half-open UTC window ``[start, end)``."""
        if self._tracker is None:
            return 0.0
        buckets = CostAggregator(self._tracker.repository).aggregate(
            CostGroup.PROVIDER,
            start=start,
            end=end - timedelta.resolution,
            provider=provider,
        )
        return math.fsum(bucket.total_usd for bucket in buckets)

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
        now: datetime | None = None,
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
        current_provider = _coerce_optional_name(provider, label="provider")
        per_run = self._config.per_run
        per_day = self._config.per_day
        per_month = self._config.per_month
        per_provider = self._config.per_provider.get(current_provider, 0.0)
        if per_run > 0 and self._tracker is None:
            raise ConfigError(
                "a per-run budget cap needs a cost ledger",
                details={"per_run": per_run},
            )
        if per_day and self._tracker is None:
            raise ConfigError(
                "a per-day budget cap needs a cost ledger",
                details={"per_day": per_day},
            )
        if per_month and self._tracker is None:
            raise ConfigError(
                "a per-month budget cap needs a cost ledger",
                details={"per_month": per_month},
            )
        if per_provider and self._tracker is None:
            raise ConfigError(
                "a per-provider budget cap needs a cost ledger",
                details={"per_provider": per_provider},
            )
        spent = self.spent(target)

        day_spent = 0.0
        month_spent = 0.0
        provider_spent = 0.0
        if per_day or per_month or per_provider:
            moment = _coerce_now(now)
            day_start, day_end = _day_window(moment)
            month_start, month_end = _month_window(moment)
            if per_day:
                day_spent = self._period_spent(day_start, day_end)
            if per_month:
                month_spent = self._period_spent(month_start, month_end)
            if per_provider:
                provider_spent = self._period_spent(
                    month_start,
                    month_end,
                    provider=current_provider,
                )

        return self._result(
            value,
            run_id=target,
            provider=current_provider,
            operation=_coerce_optional_name(operation, label="operation"),
            spent=spent,
            day_spent=day_spent,
            month_spent=month_spent,
            provider_spent=provider_spent,
        )

    def _result(
        self,
        amount_usd: float,
        *,
        run_id: str,
        provider: str,
        operation: str,
        spent: float,
        day_spent: float,
        month_spent: float,
        provider_spent: float,
    ) -> BudgetCheck:
        """Build the check for the already-computed spend totals."""
        per_call = self._config.per_call
        per_run = self._config.per_run
        per_day = self._config.per_day
        per_month = self._config.per_month
        per_provider = self._config.per_provider.get(provider, 0.0)

        scope: str | None = None
        if per_call > 0 and _exceeds(amount_usd, per_call):
            scope = _PER_CALL
        elif per_run > 0 and _exceeds(spent + amount_usd, per_run):
            scope = _PER_RUN
        elif per_day and _exceeds(day_spent + amount_usd, per_day):
            scope = _PER_DAY
        elif per_month and _exceeds(month_spent + amount_usd, per_month):
            scope = _PER_MONTH
        elif per_provider and _exceeds(provider_spent + amount_usd, per_provider):
            scope = _PER_PROVIDER

        return BudgetCheck(
            run_id=run_id,
            amount_usd=amount_usd,
            spent_usd=spent,
            per_call_cap=per_call,
            per_run_cap=per_run,
            per_day_cap=per_day,
            per_month_cap=per_month,
            per_provider_cap=per_provider,
            day_spent_usd=day_spent,
            month_spent_usd=month_spent,
            provider_spent_usd=provider_spent,
            scope=scope,
            provider=provider,
            operation=operation,
        )

    def decide(
        self,
        value: BudgetCheck | float | None = None,
        *,
        check: BudgetCheck | None = None,
        amount_usd: float | None = None,
        run_id: str | None = None,
        provider: str | None = None,
        operation: str | None = None,
        estimate: float | None = None,
        amount: float | None = None,
        cost_usd: float | None = None,
        candidates: Mapping[str, float] | Iterable[tuple[str, float]] | None = None,
        prices: Mapping[str, float] | Iterable[tuple[str, float]] | None = None,
        run: Run | None = None,
        now: datetime | None = None,
    ) -> BudgetDecision:
        """Apply the configured ``on_exceed`` behaviour to one check.

        Pass either a :class:`BudgetCheck` or the same arguments accepted by
        :meth:`evaluate`. ``candidates`` maps provider names to their estimated
        cost for this call and is used by the ``downgrade`` behaviour. When no
        cheaper candidate fits the remaining budget the decision falls back to
        pausing, so a downgrade can never let an over-budget call proceed.
        """
        if check is not None:
            if value is not None:
                raise TypeError("pass a budget check or an amount, not both")
            result = check
        elif isinstance(value, BudgetCheck):
            result = value
        else:
            result = self.evaluate(
                value if value is not None else amount_usd,
                run_id=run_id,
                provider=provider,
                operation=operation,
                estimate=estimate,
                amount=amount,
                cost_usd=cost_usd,
                now=now,
            )
        if not isinstance(result, BudgetCheck):
            raise ConfigError(
                "budget decision needs a BudgetCheck",
                details={"check": type(result).__name__},
            )

        current = _coerce_optional_name(
            provider if provider is not None else result.provider,
            label="provider",
        )
        if not result.exceeded:
            return BudgetDecision(
                check=result,
                action=BudgetAction.PROCEED,
                provider=current,
                previous_provider=current,
                estimate_usd=result.amount_usd,
                run=run,
                message="call is within budget",
            )

        behaviour = self._config.on_exceed
        detail = _exceeded_message(result)

        if behaviour is OnExceed.NOTIFY:
            message = f"{detail}; notifying and continuing"
            return BudgetDecision(
                check=result,
                action=BudgetAction.NOTIFY,
                behaviour=behaviour,
                provider=current,
                previous_provider=current,
                estimate_usd=result.amount_usd,
                run=run,
                notifications=(message,),
                message=message,
            )

        if behaviour is OnExceed.DOWNGRADE:
            remaining = result.remaining_usd
            affordable: list[tuple[float, int, str]] = []
            for index, (name, price) in enumerate(
                _coerce_candidates(candidates if candidates is not None else prices)
            ):
                if name == current or not price < result.amount_usd:
                    continue
                if remaining is not None and _exceeds(price, remaining):
                    continue
                affordable.append((price, index, name))
            if affordable:
                price, _, name = min(affordable)
                message = (
                    f"{detail}; downgraded from {current or '<unknown>'} to "
                    f"{name} at {price:.6g} USD"
                )
                return BudgetDecision(
                    check=result,
                    action=BudgetAction.DOWNGRADE,
                    behaviour=behaviour,
                    provider=name,
                    previous_provider=current,
                    estimate_usd=price,
                    run=run,
                    message=message,
                )
            message = (
                f"{detail}; no cheaper provider fits the remaining budget, "
                "run paused on budget"
            )
            return BudgetDecision(
                check=result,
                action=BudgetAction.PAUSE,
                behaviour=behaviour,
                provider="",
                previous_provider=current,
                estimate_usd=result.amount_usd,
                run=_pause_run(run, result, now),
                message=message,
            )

        message = f"{detail}; run paused on budget"
        return BudgetDecision(
            check=result,
            action=BudgetAction.PAUSE,
            behaviour=behaviour,
            provider=current,
            previous_provider=current,
            estimate_usd=result.amount_usd,
            run=_pause_run(run, result, now),
            message=message,
        )

    def resolve(self, *args: Any, **kwargs: Any) -> BudgetDecision:
        """Alias for :meth:`decide`."""
        return self.decide(*args, **kwargs)

    def on_exceed(self, *args: Any, **kwargs: Any) -> BudgetDecision:
        """Alias for :meth:`decide`."""
        return self.decide(*args, **kwargs)

    def handle_exceed(self, *args: Any, **kwargs: Any) -> BudgetDecision:
        """Alias for :meth:`decide`."""
        return self.decide(*args, **kwargs)

    def handle_exceeded(self, *args: Any, **kwargs: Any) -> BudgetDecision:
        """Alias for :meth:`decide`."""
        return self.decide(*args, **kwargs)

    def apply_on_exceed(self, *args: Any, **kwargs: Any) -> BudgetDecision:
        """Alias for :meth:`decide`."""
        return self.decide(*args, **kwargs)

    def apply_exceed(self, *args: Any, **kwargs: Any) -> BudgetDecision:
        """Alias for :meth:`decide`."""
        return self.decide(*args, **kwargs)

    def apply_policy(self, *args: Any, **kwargs: Any) -> BudgetDecision:
        """Alias for :meth:`decide`."""
        return self.decide(*args, **kwargs)

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
        now: datetime | None = None,
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
            now=now,
        )
        if result.allowed:
            return result
        self._raise_budget_error(result)

    def _raise_budget_error(self, result: BudgetCheck) -> NoReturn:
        """Raise the budget error for an exceeded check."""
        caps = {
            _PER_CALL: result.per_call_cap,
            _PER_RUN: result.per_run_cap,
            _PER_DAY: result.per_day_cap,
            _PER_MONTH: result.per_month_cap,
            _PER_PROVIDER: result.per_provider_cap,
        }
        cap = caps.get(result.scope or "", 0.0)
        raise BudgetError(
            _exceeded_message(result),
            details={
                "scope": result.scope,
                "cap": cap,
                "amount_usd": result.amount_usd,
                "requested": result.amount_usd,
                "spent_usd": result.spent_usd,
                "spent": result.spent_usd,
                "projected_usd": result.projected_usd,
                "day_spent_usd": result.day_spent_usd,
                "day_projected_usd": result.day_projected_usd,
                "month_spent_usd": result.month_spent_usd,
                "month_projected_usd": result.month_projected_usd,
                "provider_spent_usd": result.provider_spent_usd,
                "provider_projected_usd": result.provider_projected_usd,
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

    def reserve(
        self,
        amount_usd: float | None = None,
        *,
        run_id: str | None = None,
        provider: str | None = None,
        operation: str | None = None,
        estimate: float | None = None,
        amount: float | None = None,
        cost_usd: float | None = None,
        now: datetime | None = None,
        model: str | None = None,
        units: float = 0.0,
        unit_kind: str = "call",
        pricing_version: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> BudgetCheck:
        """Check the estimate and atomically record it as a reservation.

        The spend is read and an estimated :class:`CostEvent` is inserted in
        one transaction that takes SQLite's write lock first, so two concurrent
        calls cannot both see the same remaining per-run budget and pass. A
        reservation that would exceed a cap is not written and raises
        :class:`~aivia.core.errors.BudgetError`, exactly like :meth:`check`.

        A guard without a ledger has nowhere to reserve and falls back to the
        read-only check; the per-call cap does not need a shared reservation.
        """
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
        current_provider = _coerce_optional_name(provider, label="provider")
        operation_name = _coerce_optional_name(operation, label="operation")
        if self._tracker is None:
            return self.check(
                value,
                run_id=target,
                provider=current_provider,
                operation=operation_name,
                now=now,
            )

        per_day = self._config.per_day
        per_month = self._config.per_month
        per_provider = self._config.per_provider.get(current_provider, 0.0)
        moment = (
            _coerce_now(now)
            if now is not None or per_day or per_month or per_provider
            else None
        )
        values: dict[str, Any] = {
            "provider": current_provider,
            "model": _coerce_optional_name(model, label="model"),
            "operation": operation_name,
            "units": units,
            "unit_kind": unit_kind,
            "amount_usd": value,
            "estimated": True,
            "run_id": target,
            "pricing_version": _coerce_optional_name(
                pricing_version, label="pricing_version"
            ),
            "metadata": dict(metadata or {}),
        }
        if moment is not None:
            values["created_at"] = moment
            values["updated_at"] = moment
        event = CostEvent.model_validate(values)
        repository = self._tracker.repository

        with _write_transaction(repository.engine) as connection:
            spent = _sum_events(connection, run_id=target)
            day_spent = 0.0
            month_spent = 0.0
            provider_spent = 0.0
            if moment is not None and (per_day or per_month or per_provider):
                day_start, day_end = _day_window(moment)
                month_start, month_end = _month_window(moment)
                if per_day:
                    day_spent = _sum_events(
                        connection,
                        start=day_start,
                        end=day_end,
                    )
                if per_month:
                    month_spent = _sum_events(
                        connection,
                        start=month_start,
                        end=month_end,
                    )
                if per_provider:
                    provider_spent = _sum_events(
                        connection,
                        start=month_start,
                        end=month_end,
                        provider=current_provider,
                    )
            result = self._result(
                value,
                run_id=target,
                provider=current_provider,
                operation=operation_name,
                spent=spent,
                day_spent=day_spent,
                month_spent=month_spent,
                provider_spent=provider_spent,
            )
            if result.allowed:
                repository.create_in(connection, event)

        if result.exceeded:
            self._raise_budget_error(result)
        return result

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

    def remaining(
        self,
        run_id: str | None = None,
        *,
        provider: str | None = None,
        operation: str | None = None,
        now: datetime | None = None,
    ) -> float | None:
        """Return the most a call may cost for *run_id*, if a cap applies."""
        return self.evaluate(
            0.0,
            run_id=run_id,
            provider=provider,
            operation=operation,
            now=now,
        ).remaining_usd


BudgetEnforcer = BudgetGuard
BudgetChecker = BudgetGuard
BudgetManager = BudgetGuard
BudgetController = BudgetGuard
BudgetPolicy = BudgetGuard


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
    now: datetime | None = None,
) -> BudgetCheck:
    """Check one call estimate without constructing a long-lived guard."""
    return _guard(config, tracker).check(
        amount_usd,
        run_id=run_id,
        provider=provider,
        operation=operation,
        now=now,
    )


def enforce_budget(
    amount_usd: float,
    *,
    run_id: str | None = None,
    config: BudgetConfig | Mapping[str, Any] | None = None,
    tracker: CostTracker | CostEventRepository | Engine | None = None,
    provider: str | None = None,
    operation: str | None = None,
    now: datetime | None = None,
) -> BudgetCheck:
    """Alias for :func:`check_budget`."""
    return check_budget(
        amount_usd,
        run_id=run_id,
        config=config,
        tracker=tracker,
        provider=provider,
        operation=operation,
        now=now,
    )


def ensure_budget(
    amount_usd: float,
    *,
    run_id: str | None = None,
    config: BudgetConfig | Mapping[str, Any] | None = None,
    tracker: CostTracker | CostEventRepository | Engine | None = None,
    provider: str | None = None,
    operation: str | None = None,
    now: datetime | None = None,
) -> BudgetCheck:
    """Alias for :func:`check_budget`."""
    return check_budget(
        amount_usd,
        run_id=run_id,
        config=config,
        tracker=tracker,
        provider=provider,
        operation=operation,
        now=now,
    )


def would_exceed_budget(
    amount_usd: float,
    *,
    run_id: str | None = None,
    config: BudgetConfig | Mapping[str, Any] | None = None,
    tracker: CostTracker | CostEventRepository | Engine | None = None,
    provider: str | None = None,
    operation: str | None = None,
    now: datetime | None = None,
) -> bool:
    """Return whether a call estimate exceeds an active cap."""
    return _guard(config, tracker).would_exceed(
        amount_usd,
        run_id=run_id,
        provider=provider,
        operation=operation,
        now=now,
    )


def decide_budget(
    amount_usd: float,
    *,
    run_id: str | None = None,
    config: BudgetConfig | Mapping[str, Any] | None = None,
    tracker: CostTracker | CostEventRepository | Engine | None = None,
    provider: str | None = None,
    operation: str | None = None,
    candidates: Mapping[str, float] | Iterable[tuple[str, float]] | None = None,
    prices: Mapping[str, float] | Iterable[tuple[str, float]] | None = None,
    run: Run | None = None,
    now: datetime | None = None,
) -> BudgetDecision:
    """Apply the configured overrun policy without a long-lived guard."""
    return _guard(config, tracker).decide(
        amount_usd,
        run_id=run_id,
        provider=provider,
        operation=operation,
        candidates=candidates,
        prices=prices,
        run=run,
        now=now,
    )


def resolve_budget(
    amount_usd: float,
    *,
    run_id: str | None = None,
    config: BudgetConfig | Mapping[str, Any] | None = None,
    tracker: CostTracker | CostEventRepository | Engine | None = None,
    provider: str | None = None,
    operation: str | None = None,
    candidates: Mapping[str, float] | Iterable[tuple[str, float]] | None = None,
    prices: Mapping[str, float] | Iterable[tuple[str, float]] | None = None,
    run: Run | None = None,
    now: datetime | None = None,
) -> BudgetDecision:
    """Alias for :func:`decide_budget`."""
    return decide_budget(
        amount_usd,
        run_id=run_id,
        config=config,
        tracker=tracker,
        provider=provider,
        operation=operation,
        candidates=candidates,
        prices=prices,
        run=run,
        now=now,
    )


def apply_budget_policy(
    amount_usd: float,
    *,
    run_id: str | None = None,
    config: BudgetConfig | Mapping[str, Any] | None = None,
    tracker: CostTracker | CostEventRepository | Engine | None = None,
    provider: str | None = None,
    operation: str | None = None,
    candidates: Mapping[str, float] | Iterable[tuple[str, float]] | None = None,
    prices: Mapping[str, float] | Iterable[tuple[str, float]] | None = None,
    run: Run | None = None,
    now: datetime | None = None,
) -> BudgetDecision:
    """Alias for :func:`decide_budget`."""
    return decide_budget(
        amount_usd,
        run_id=run_id,
        config=config,
        tracker=tracker,
        provider=provider,
        operation=operation,
        candidates=candidates,
        prices=prices,
        run=run,
        now=now,
    )


def apply_on_exceed(
    amount_usd: float,
    *,
    run_id: str | None = None,
    config: BudgetConfig | Mapping[str, Any] | None = None,
    tracker: CostTracker | CostEventRepository | Engine | None = None,
    provider: str | None = None,
    operation: str | None = None,
    candidates: Mapping[str, float] | Iterable[tuple[str, float]] | None = None,
    prices: Mapping[str, float] | Iterable[tuple[str, float]] | None = None,
    run: Run | None = None,
    now: datetime | None = None,
) -> BudgetDecision:
    """Alias for :func:`decide_budget`."""
    return decide_budget(
        amount_usd,
        run_id=run_id,
        config=config,
        tracker=tracker,
        provider=provider,
        operation=operation,
        candidates=candidates,
        prices=prices,
        run=run,
        now=now,
    )

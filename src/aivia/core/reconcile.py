"""Reconcile provider-reported actual amounts against pre-call estimates.

The pricing resolver produces an estimate before a paid call. Some providers
return the amount they actually charged with the response. This module compares
the two, reports the signed and absolute difference, and flags the call when the
difference is above a configured threshold.

The comparison is pure: it never calls a provider or changes the ledger. The
optional :meth:`Reconciler.record_actual` helper writes the reported amount as
an actual :class:`~aivia.core.models.cost.CostEvent` through an existing
:class:`~aivia.core.cost.CostTracker`, carrying the estimate, drift and flag in
the event metadata so the decision stays auditable.
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from aivia.core.config.reconcile import (
    DEFAULT_ABSOLUTE_THRESHOLD_USD,
    DEFAULT_DRIFT_THRESHOLD,
    ReconcileConfig,
)
from aivia.core.cost import CostTracker
from aivia.core.models.cost import CostEvent

__all__ = [
    "DEFAULT_ABSOLUTE_THRESHOLD_USD",
    "DEFAULT_DRIFT_THRESHOLD",
    "CostDriftWarning",
    "CostReconcileResult",
    "CostReconciler",
    "CostReconciliation",
    "Drift",
    "DriftReport",
    "ReconcileResult",
    "Reconciler",
    "Reconciliation",
    "actual_amount",
    "compare_amounts",
    "compare_costs",
    "reconcile",
    "reconcile_amounts",
    "reconcile_costs",
    "reported_amount",
]

_DIRECT_AMOUNT_KEYS = (
    "actual_cost_usd",
    "actual_amount_usd",
    "reported_cost_usd",
    "reported_amount_usd",
    "amount_usd",
    "cost_usd",
    "total_cost_usd",
    "price_usd",
    "amount",
    "cost",
    "usd",
)
_NESTED_AMOUNT_KEYS = ("metadata", "usage", "billing", "cost_details", "details")
_EVENT_FIELDS = (
    "provider",
    "model",
    "operation",
    "units",
    "unit_kind",
    "pricing_version",
    "run_id",
)


def _coerce_amount(value: object, *, label: str = "amount") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite, non-negative number")
    amount = float(value)
    if not math.isfinite(amount) or amount < 0:
        raise ValueError(f"{label} must be a finite, non-negative number")
    return amount


def _lookup_amount(source: object, *, depth: int = 0) -> float | None:
    if depth > 4:
        return None
    if isinstance(source, CostEvent):
        return source.amount_usd
    if isinstance(source, (int, float)) and not isinstance(source, bool):
        return _coerce_amount(source)

    if isinstance(source, Mapping):
        data: Mapping[str, Any] = source
    else:
        data = {}
        for key in _DIRECT_AMOUNT_KEYS:
            if hasattr(source, key):
                data = {key: getattr(source, key)}
                break
        else:
            for key in _NESTED_AMOUNT_KEYS:
                if hasattr(source, key):
                    data = {key: getattr(source, key)}
                    break
            else:
                dump = getattr(source, "model_dump", None)
                if callable(dump):
                    dumped = dump()
                    if isinstance(dumped, Mapping):
                        data = dumped

    for key in _DIRECT_AMOUNT_KEYS:
        if key not in data:
            continue
        value = data[key]
        if isinstance(value, Mapping):
            found = _lookup_amount(value, depth=depth + 1)
            if found is not None:
                return found
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            return _coerce_amount(value)

    for key in _NESTED_AMOUNT_KEYS:
        if key not in data:
            continue
        value = data[key]
        if value is None or isinstance(value, (str, bytes)):
            continue
        found = _lookup_amount(value, depth=depth + 1)
        if found is not None:
            return found
    return None


def reported_amount(source: object) -> float:
    """Return the actual amount a provider response reports.

    A number is accepted as-is. A mapping or object may expose the amount as
    ``amount_usd``, ``cost_usd`` or one of the usual aliases, either directly or
    inside ``metadata``, ``usage``, ``billing`` or ``cost_details``. A response
    without a reported amount raises :class:`ValueError` instead of silently
    treating the call as free.
    """
    amount = _lookup_amount(source)
    if amount is None:
        raise ValueError("provider response does not report an actual cost amount")
    return amount


actual_amount = reported_amount
"""Alias for :func:`reported_amount`."""


def _is_above(value: float, limit: float) -> bool:
    return value > limit and not math.isclose(
        value,
        limit,
        rel_tol=1e-9,
        abs_tol=1e-12,
    )


@dataclass(frozen=True, slots=True)
class Reconciliation:
    """The comparison of one estimate with the amount a provider reported.

    ``difference_usd`` is signed (actual minus estimate), while ``drift_usd``
    is its absolute value. ``allowed_drift_usd`` is the larger of the relative
    threshold applied to the estimate and the configured absolute floor. The
    call is flagged only when the drift is above that allowance.
    """

    estimate_usd: float
    actual_usd: float
    threshold: float
    absolute_threshold_usd: float = 0.0
    estimate_event_id: str = ""
    actual_event_id: str = ""
    estimate_event: CostEvent | None = None
    actual_event: CostEvent | None = None

    @property
    def difference_usd(self) -> float:
        """The signed difference between the actual and estimated amount."""
        return self.actual_usd - self.estimate_usd

    @property
    def difference(self) -> float:
        """Alias for :attr:`difference_usd`."""
        return self.difference_usd

    @property
    def drift_usd(self) -> float:
        """The absolute difference between the actual and estimated amount."""
        return abs(self.difference_usd)

    @property
    def drift(self) -> float:
        """Alias for :attr:`drift_usd`."""
        return self.drift_usd

    @property
    def drift_ratio(self) -> float:
        """The drift as a fraction of the estimate.

        A zero estimate has no meaningful ratio: it is ``0.0`` when the actual
        amount is also zero and infinite when the provider charged anything.
        """
        if self.estimate_usd == 0:
            return 0.0 if self.drift_usd == 0 else math.inf
        return self.drift_usd / self.estimate_usd

    @property
    def allowed_drift_usd(self) -> float:
        """The largest drift that is still considered within tolerance."""
        return max(
            self.estimate_usd * self.threshold,
            self.absolute_threshold_usd,
        )

    @property
    def exceeds_threshold(self) -> bool:
        """Whether the drift is above the configured allowance."""
        return _is_above(self.drift_usd, self.allowed_drift_usd)

    @property
    def flagged(self) -> bool:
        """Alias for :attr:`exceeds_threshold`."""
        return self.exceeds_threshold

    @property
    def drifted(self) -> bool:
        """Alias for :attr:`exceeds_threshold`."""
        return self.exceeds_threshold

    @property
    def within_threshold(self) -> bool:
        """Whether the drift is within the configured allowance."""
        return not self.exceeds_threshold

    @property
    def is_flagged(self) -> bool:
        """Alias for :attr:`exceeds_threshold`."""
        return self.exceeds_threshold

    @property
    def is_within_threshold(self) -> bool:
        """Alias for :attr:`within_threshold`."""
        return self.within_threshold

    @property
    def signed_drift_ratio(self) -> float:
        """The signed difference as a fraction of the estimate."""
        if self.estimate_usd == 0:
            return (
                0.0
                if self.difference_usd == 0
                else math.copysign(
                    math.inf,
                    self.difference_usd,
                )
            )
        return self.difference_usd / self.estimate_usd

    @property
    def direction(self) -> str:
        """``over``, ``under`` or ``none`` relative to the estimate."""
        if self.difference_usd > 0:
            return "over"
        if self.difference_usd < 0:
            return "under"
        return "none"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for logs and reports."""
        ratio = self.drift_ratio
        return {
            "estimate_usd": self.estimate_usd,
            "actual_usd": self.actual_usd,
            "difference_usd": self.difference_usd,
            "drift_usd": self.drift_usd,
            "drift_ratio": ratio if math.isfinite(ratio) else None,
            "threshold": self.threshold,
            "absolute_threshold_usd": self.absolute_threshold_usd,
            "allowed_drift_usd": self.allowed_drift_usd,
            "flagged": self.flagged,
            "direction": self.direction,
            "estimate_event_id": self.estimate_event_id,
            "actual_event_id": self.actual_event_id,
        }


ReconcileResult = Reconciliation
CostReconciliation = Reconciliation
CostReconcileResult = Reconciliation
DriftReport = Reconciliation
Drift = Reconciliation


class CostDriftWarning(UserWarning):
    """A provider-reported amount drifted above the configured threshold."""

    def __init__(self, reconciliation: Reconciliation) -> None:
        self.reconciliation = reconciliation
        self.message = (
            "reported amount drifted from the estimate: "
            f"estimate {reconciliation.estimate_usd:.6f}, "
            f"actual {reconciliation.actual_usd:.6f}, "
            f"drift {reconciliation.drift_usd:.6f} "
            f"({reconciliation.drift_ratio:.1%}), "
            f"threshold {reconciliation.threshold:.1%}"
        )
        super().__init__(self.message)


def _coerce_threshold(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite, non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be a finite, non-negative number")
    return number


def _event_fields(source: object) -> dict[str, Any]:
    if isinstance(source, CostEvent):
        values: dict[str, Any] = {name: getattr(source, name) for name in _EVENT_FIELDS}
        values["metadata"] = dict(source.metadata)
        return values
    if isinstance(source, Mapping):
        return {
            name: source[name]
            for name in (*_EVENT_FIELDS, "metadata")
            if name in source
        }
    return {}


def _drift_metadata(reconciliation: Reconciliation) -> dict[str, object]:
    ratio = reconciliation.drift_ratio
    return {
        "estimate_usd": reconciliation.estimate_usd,
        "drift_usd": reconciliation.drift_usd,
        "drift_ratio": ratio if math.isfinite(ratio) else None,
        "drift_threshold": reconciliation.threshold,
        "drift": reconciliation.flagged,
        "drift_flagged": reconciliation.flagged,
    }


class Reconciler:
    """Compare estimates with provider-reported amounts.

    ``config`` supplies the default relative and absolute thresholds. The
    explicit ``threshold`` and ``absolute_threshold_usd`` arguments override
    the corresponding config values for one instance. Set ``warn`` to emit a
    :class:`CostDriftWarning` whenever a comparison is flagged.
    """

    def __init__(
        self,
        config: ReconcileConfig | Mapping[str, Any] | None = None,
        *,
        threshold: float | None = None,
        absolute_threshold_usd: float | None = None,
        warn: bool = False,
    ) -> None:
        if config is None:
            parsed = ReconcileConfig()
        elif isinstance(config, ReconcileConfig):
            parsed = config
        else:
            parsed = ReconcileConfig.from_mapping(config)

        self._threshold = (
            parsed.drift_threshold
            if threshold is None
            else _coerce_threshold(threshold, label="drift threshold")
        )
        self._absolute_threshold_usd = (
            parsed.absolute_threshold_usd
            if absolute_threshold_usd is None
            else _coerce_threshold(
                absolute_threshold_usd,
                label="absolute drift threshold",
            )
        )
        self._warn = bool(warn)

    @property
    def threshold(self) -> float:
        """The relative drift threshold used by this reconciler."""
        return self._threshold

    @property
    def absolute_threshold_usd(self) -> float:
        """The absolute drift floor used by this reconciler."""
        return self._absolute_threshold_usd

    def _build(
        self,
        estimate_usd: float,
        actual_usd: float,
        *,
        estimate_event: CostEvent | None = None,
        actual_event: CostEvent | None = None,
    ) -> Reconciliation:
        return Reconciliation(
            estimate_usd=estimate_usd,
            actual_usd=actual_usd,
            threshold=self._threshold,
            absolute_threshold_usd=self._absolute_threshold_usd,
            estimate_event_id="" if estimate_event is None else estimate_event.id,
            actual_event_id="" if actual_event is None else actual_event.id,
            estimate_event=estimate_event,
            actual_event=actual_event,
        )

    def _maybe_warn(self, reconciliation: Reconciliation) -> None:
        if self._warn and reconciliation.flagged:
            warnings.warn(CostDriftWarning(reconciliation), stacklevel=3)

    def compare(
        self,
        estimate: object,
        actual: object,
    ) -> Reconciliation:
        """Compare an estimate with a provider-reported actual amount.

        Both arguments may be a :class:`CostEvent`, a number or a provider
        response carrying a cost amount. When an event is passed, its id is
        kept on the result so the two ledger rows stay linked.
        """
        estimate_event = estimate if isinstance(estimate, CostEvent) else None
        actual_event = actual if isinstance(actual, CostEvent) else None
        estimate_amount = _lookup_amount(estimate)
        if estimate_amount is None:
            raise ValueError("estimate does not contain a cost amount")
        actual_amount = reported_amount(actual)
        result = self._build(
            estimate_amount,
            actual_amount,
            estimate_event=estimate_event,
            actual_event=actual_event,
        )
        self._maybe_warn(result)
        return result

    def reconcile(self, estimate: object, actual: object) -> Reconciliation:
        """Alias for :meth:`compare`."""
        return self.compare(estimate, actual)

    def reconcile_events(self, estimate: object, actual: object) -> Reconciliation:
        """Alias for :meth:`compare` for callers passing ledger events."""
        return self.compare(estimate, actual)

    def check(self, estimate: object, actual: object) -> Reconciliation:
        """Alias for :meth:`compare`."""
        return self.compare(estimate, actual)

    def evaluate(self, estimate: object, actual: object) -> Reconciliation:
        """Alias for :meth:`compare`."""
        return self.compare(estimate, actual)

    def compare_events(self, estimate: object, actual: object) -> Reconciliation:
        """Alias for :meth:`compare` for callers passing ledger events."""
        return self.compare(estimate, actual)

    def reconcile_run(
        self,
        tracker: CostTracker,
        run_id: str,
    ) -> list[Reconciliation]:
        """Pair estimated and actual events for *run_id* in ledger order.

        Events are matched on provider, model, operation and unit kind. An
        estimate without a later actual (or an actual without an estimate) is
        left unpaired, because there is nothing meaningful to compare.
        """
        pending: dict[tuple[str, str, str, str], deque[CostEvent]] = {}
        results: list[Reconciliation] = []
        for event in tracker.for_run(run_id):
            key = (event.provider, event.model, event.operation, event.unit_kind)
            if event.estimated:
                pending.setdefault(key, deque()).append(event)
                continue
            estimates = pending.get(key)
            if not estimates:
                continue
            estimate_event = estimates.popleft()
            result = self._build(
                estimate_event.amount_usd,
                event.amount_usd,
                estimate_event=estimate_event,
                actual_event=event,
            )
            self._maybe_warn(result)
            results.append(result)
        return results

    def record_actual(
        self,
        tracker: CostTracker,
        estimate: object,
        actual: object | None = None,
        *,
        amount_usd: float | None = None,
        **values: Any,
    ) -> Reconciliation:
        """Record a provider-reported amount and return its reconciliation.

        *actual* may be the provider response or a number; pass *amount_usd*
        instead when the amount is already extracted. Provider, operation and
        the other call fields default to those of an estimate :class:`CostEvent`
        and can be overridden with keyword values. The returned result carries
        the stored actual event, its id and the drift flag.
        """
        estimate_event = estimate if isinstance(estimate, CostEvent) else None
        estimate_amount = _lookup_amount(estimate)
        if estimate_amount is None:
            raise ValueError("estimate does not contain a cost amount")
        if amount_usd is not None:
            actual_amount = _coerce_amount(amount_usd, label="actual amount")
            actual_event = actual if isinstance(actual, CostEvent) else None
        elif actual is None:
            raise ValueError("record_actual needs a reported amount")
        else:
            actual_amount = reported_amount(actual)
            actual_event = actual if isinstance(actual, CostEvent) else None

        result = self._build(
            estimate_amount,
            actual_amount,
            estimate_event=estimate_event,
            actual_event=actual_event,
        )
        self._maybe_warn(result)

        fields = _event_fields(estimate)
        metadata = dict(fields.pop("metadata", {}) or {})
        supplied_metadata = values.pop("metadata", None)
        if supplied_metadata is not None:
            metadata.update(dict(supplied_metadata))
        metadata.update(_drift_metadata(result))
        if result.estimate_event_id:
            metadata.setdefault("replaces", result.estimate_event_id)
            metadata.setdefault("estimate_event_id", result.estimate_event_id)
        fields.update(values)
        fields["amount_usd"] = actual_amount
        fields["metadata"] = metadata

        stored = tracker.record_actual(**fields)
        return replace(
            result,
            actual_event=stored,
            actual_event_id=stored.id,
        )

    def record_reported_actual(
        self,
        tracker: CostTracker,
        estimate: object,
        actual: object | None = None,
        *,
        amount_usd: float | None = None,
        **values: Any,
    ) -> Reconciliation:
        """Alias for :meth:`record_actual`."""
        return self.record_actual(
            tracker,
            estimate,
            actual,
            amount_usd=amount_usd,
            **values,
        )

    def record(
        self,
        tracker: CostTracker,
        estimate: object,
        actual: object | None = None,
        *,
        amount_usd: float | None = None,
        **values: Any,
    ) -> Reconciliation:
        """Alias for :meth:`record_actual`."""
        return self.record_actual(
            tracker,
            estimate,
            actual,
            amount_usd=amount_usd,
            **values,
        )


CostReconciler = Reconciler


def reconcile(
    estimate: object,
    actual: object,
    *,
    config: ReconcileConfig | Mapping[str, Any] | None = None,
    threshold: float | None = None,
    absolute_threshold_usd: float | None = None,
    warn: bool = False,
) -> Reconciliation:
    """Compare *estimate* with *actual* without keeping a reconciler."""
    return Reconciler(
        config,
        threshold=threshold,
        absolute_threshold_usd=absolute_threshold_usd,
        warn=warn,
    ).compare(estimate, actual)


reconcile_amounts = reconcile
reconcile_costs = reconcile
compare_amounts = reconcile
compare_costs = reconcile

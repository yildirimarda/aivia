"""Detect pricing drift between estimated and actual provider costs.

A pricing drift job pairs every pre-call estimate with the amount a provider
later reported, computes the drift, warns when it crosses a threshold and
prints a suggested per-provider override (the actual unit price). The
suggestions are printed only: nothing is written to config or the ledger, so
an operator can review and apply them manually.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TextIO

from aivia.core.cost import CostTracker
from aivia.core.reconcile import CostDriftWarning, Reconciler, Reconciliation

__all__ = [
    "CostDriftWarning",
    "DriftSuggestion",
    "PricingDriftJob",
    "check_pricing_drift",
]


@dataclass(frozen=True, slots=True)
class DriftSuggestion:
    """A suggested pricing override for one drifted provider.

    ``suggested_override_usd`` is the actual amount divided by the estimated
    units, expressed in the estimate's unit kind. It is a suggestion only and
    is never written to configuration.
    """

    provider: str
    estimate_usd: float
    actual_usd: float
    drift_usd: float
    suggested_override_usd: float
    units: float
    unit_kind: str


class PricingDriftJob:
    """Compare estimated and actual costs and suggest overrides without writing them."""

    def __init__(
        self,
        tracker: CostTracker,
        *,
        threshold: float = 0.1,
        absolute_threshold_usd: float = 0.0,
        warn: bool = True,
        print_suggestions: bool = True,
        file: TextIO | None = None,
    ) -> None:
        self._tracker = tracker
        self._reconciler = Reconciler(
            threshold=threshold,
            absolute_threshold_usd=absolute_threshold_usd,
            warn=warn,
        )
        self._print_suggestions = bool(print_suggestions)
        self._file = file if file is not None else sys.stdout

    @property
    def tracker(self) -> CostTracker:
        """The cost ledger the job reads estimates and actuals from."""
        return self._tracker

    @property
    def reconciler(self) -> Reconciler:
        """The reconciler used to compare estimates with reported amounts."""
        return self._reconciler

    def check_run(self, run_id: str) -> list[DriftSuggestion]:
        """Pair estimates and actuals for *run_id*, warn and print suggestions.

        Returns one :class:`DriftSuggestion` per provider whose reported amount
        drifted beyond the threshold. Nothing is written to config or the
        ledger.
        """
        results = self._reconciler.reconcile_run(self._tracker, run_id)
        by_provider: dict[str, list[Reconciliation]] = {}
        for result in results:
            if not result.flagged:
                continue
            estimate = result.estimate_event
            if estimate is None:
                continue
            by_provider.setdefault(estimate.provider, []).append(result)

        suggestions: list[DriftSuggestion] = []
        for provider, results_for_provider in by_provider.items():
            estimate_usd = sum(result.estimate_usd for result in results_for_provider)
            actual_usd = sum(result.actual_usd for result in results_for_provider)
            drift_usd = sum(result.drift_usd for result in results_for_provider)
            total_units = sum(
                result.estimate_event.units
                if result.estimate_event is not None
                else 0.0
                for result in results_for_provider
            )
            first_event = results_for_provider[0].estimate_event
            unit_kind = first_event.unit_kind if first_event is not None else "call"
            suggested = actual_usd / total_units if total_units else actual_usd
            suggestions.append(
                DriftSuggestion(
                    provider=provider,
                    estimate_usd=estimate_usd,
                    actual_usd=actual_usd,
                    drift_usd=drift_usd,
                    suggested_override_usd=suggested,
                    units=total_units,
                    unit_kind=unit_kind,
                )
            )

        if self._print_suggestions and suggestions:
            self._print(suggestions)
        return suggestions

    def _print(self, suggestions: list[DriftSuggestion]) -> None:
        print(
            "Pricing drift detected — suggested overrides (not written):",
            file=self._file,
        )
        for suggestion in suggestions:
            message = (
                f"  {suggestion.provider}: "
                f"{suggestion.suggested_override_usd:.6f} USD/{suggestion.unit_kind} "
                f"(estimate {suggestion.estimate_usd:.6f}, "
                f"actual {suggestion.actual_usd:.6f}, "
                f"drift {suggestion.drift_usd:.6f})"
            )
            print(message, file=self._file)


def check_pricing_drift(
    tracker: CostTracker,
    run_id: str,
    *,
    threshold: float = 0.1,
    absolute_threshold_usd: float = 0.0,
    warn: bool = True,
    print_suggestions: bool = True,
) -> list[DriftSuggestion]:
    """Check *run_id* for pricing drift and print suggested overrides.

    Convenience wrapper around :class:`PricingDriftJob`. Nothing is written
    to config or the ledger.
    """
    job = PricingDriftJob(
        tracker,
        threshold=threshold,
        absolute_threshold_usd=absolute_threshold_usd,
        warn=warn,
        print_suggestions=print_suggestions,
    )
    return job.check_run(run_id)

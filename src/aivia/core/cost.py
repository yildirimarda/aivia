"""Record a cost event for every paid provider call.

The :class:`CostTracker` is the write side of the cost ledger. A caller that
made a paid call hands it the provider, operation, units and amount together
with the pricing version that produced the amount and whether the amount is a
pre-call estimate or a provider-reported actual. The tracker validates the
record through :class:`~aivia.core.models.cost.CostEvent` and persists it with
:class:`~aivia.core.db.repos.CostEventRepository`.

The tracker deliberately does not resolve prices, enforce budgets or call a
provider; those responsibilities belong to the pricing resolver, the budget
guard and the media service. It only guarantees that a paid call leaves a
durable ledger entry instead of disappearing when a run ends.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from sqlalchemy import Engine

from aivia.core.cost_agg import CostAggregate, CostAggregator, CostGroup
from aivia.core.db.repos import CostEventRepository
from aivia.core.models.cost import CostEvent, CostKind

__all__ = [
    "CostAggregate",
    "CostAggregator",
    "CostEventRecorder",
    "CostGroup",
    "CostLedger",
    "CostRecorder",
    "CostTracker",
]


class CostTracker:
    """Persist one :class:`CostEvent` per paid call.

    Pass an SQLAlchemy :class:`~sqlalchemy.Engine` or a ready
    :class:`~aivia.core.db.repos.CostEventRepository`. The engine form builds
    the repository on demand, so a caller does not have to import the database
    layer just to record a call.
    """

    def __init__(
        self,
        engine: Engine | CostEventRepository | None = None,
        *,
        repository: CostEventRepository | None = None,
    ) -> None:
        if repository is not None:
            if engine is not None:
                raise TypeError("pass either engine or repository, not both")
            target: Engine | CostEventRepository = repository
        else:
            if engine is None:
                raise TypeError(
                    "CostTracker needs an engine or a cost event repository"
                )
            target = engine

        if isinstance(target, CostEventRepository):
            self._repository = target
        elif isinstance(target, Engine):
            self._repository = CostEventRepository(target)
        else:
            raise TypeError(
                "engine must be a SQLAlchemy Engine or a CostEventRepository"
            )

    @property
    def repository(self) -> CostEventRepository:
        """The repository every recorded event is written through."""
        return self._repository

    def record(
        self,
        event: CostEvent | Mapping[str, Any] | None = None,
        **values: Any,
    ) -> CostEvent:
        """Record one paid call and return the stored event.

        Pass a validated :class:`CostEvent`, a mapping using the model's
        aliases, or keyword values. A keyword value overrides the matching
        field of *event*. Provider, operation, units, amount, pricing version
        and the estimated/actual flag are all persisted on the returned event.
        """
        data = self._merge(event, values)
        if not data:
            raise ValueError("a cost event needs at least a provider and an operation")

        record = CostEvent.model_validate(data)
        return self._repository.create(record)

    @staticmethod
    def _merge(
        event: CostEvent | Mapping[str, Any] | None,
        values: Mapping[str, Any],
    ) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if isinstance(event, CostEvent):
            data.update(event.model_dump())
        elif event is not None:
            if not isinstance(event, Mapping):
                raise TypeError("event must be a CostEvent or a mapping")
            data.update(dict(event))
        data.update(values)
        return data

    def _link_replacement(
        self,
        event: CostEvent | Mapping[str, Any] | None,
        values: dict[str, Any],
    ) -> None:
        """Point an actual event at the estimate it replaces, when one exists."""
        data = self._merge(event, values)
        if not data:
            return
        probe = CostEvent.model_validate(data)
        if probe.replaces:
            return
        candidate = self._unpaired_estimate(probe)
        if candidate is None:
            return
        metadata = dict(probe.metadata)
        metadata.setdefault("replaces", candidate.id)
        metadata.setdefault("estimate_event_id", candidate.id)
        values["metadata"] = metadata

    def _unpaired_estimate(self, actual: CostEvent) -> CostEvent | None:
        """Return the earliest estimate for *actual*'s call that has no actual."""
        events = self.events()
        replaced = {
            event.replaces for event in events if not event.estimated and event.replaces
        }
        key = (
            actual.run_id,
            actual.provider,
            actual.model,
            actual.operation,
            actual.unit_kind,
        )
        for candidate in events:
            if not candidate.estimated or candidate.id in replaced:
                continue
            if (
                candidate.run_id,
                candidate.provider,
                candidate.model,
                candidate.operation,
                candidate.unit_kind,
            ) == key:
                return candidate
        return None

    def _effective_events(self, events: list[CostEvent]) -> list[CostEvent]:
        """Drop estimates that an actual event has replaced."""
        replaced = {
            event.replaces for event in events if not event.estimated and event.replaces
        }
        if not replaced:
            return events
        return [
            event for event in events if not (event.estimated and event.id in replaced)
        ]

    def record_event(
        self,
        event: CostEvent | Mapping[str, Any] | None = None,
        **values: Any,
    ) -> CostEvent:
        """Alias for :meth:`record`."""
        return self.record(event, **values)

    def record_estimate(
        self,
        event: CostEvent | Mapping[str, Any] | None = None,
        **values: Any,
    ) -> CostEvent:
        """Record a pre-call quote, marking the event as estimated."""
        values["estimated"] = True
        return self.record(event, **values)

    def record_actual(
        self,
        event: CostEvent | Mapping[str, Any] | None = None,
        **values: Any,
    ) -> CostEvent:
        """Record a provider-reported amount, marking the event as actual.

        When the event does not already name the estimate it replaces, the
        earliest unpaired estimate for the same call is linked automatically.
        """
        values["estimated"] = False
        self._link_replacement(event, values)
        return self.record(event, **values)

    def record_paid_call(
        self,
        event: CostEvent | Mapping[str, Any] | None = None,
        *,
        estimated: bool = False,
        **values: Any,
    ) -> CostEvent:
        """Record a paid call, choosing the estimated/actual flag explicitly."""
        values["estimated"] = estimated
        if not estimated:
            self._link_replacement(event, values)
        return self.record(event, **values)

    def events(
        self,
        *,
        run_id: str | None = None,
        provider: str | None = None,
        operation: str | None = None,
        estimated: bool | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[CostEvent]:
        """Return recorded events matching the given equality filters."""
        filters: dict[str, Any] = {}
        if run_id is not None:
            filters["run_id"] = run_id
        if provider is not None:
            filters["provider"] = provider
        if operation is not None:
            filters["operation"] = operation
        if estimated is not None:
            filters["estimated"] = estimated
        return self._repository.list(limit=limit, offset=offset, **filters)

    def list_events(
        self,
        *,
        run_id: str | None = None,
        provider: str | None = None,
        operation: str | None = None,
        estimated: bool | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[CostEvent]:
        """Alias for :meth:`events`."""
        return self.events(
            run_id=run_id,
            provider=provider,
            operation=operation,
            estimated=estimated,
            limit=limit,
            offset=offset,
        )

    def for_run(self, run_id: str) -> list[CostEvent]:
        """Return every event belonging to *run_id* in creation order."""
        return self.events(run_id=run_id)

    def total(
        self,
        *,
        run_id: str | None = None,
        provider: str | None = None,
        operation: str | None = None,
        estimated: bool | None = None,
    ) -> float:
        """Return the summed USD amount of the matching events.

        An estimate replaced by a later actual event is not counted, so one
        call contributes either its estimate or its actual amount, never both.
        """
        events = self._effective_events(
            self.events(
                run_id=run_id,
                provider=provider,
                operation=operation,
            )
        )
        if estimated is not None:
            events = [event for event in events if event.estimated is estimated]
        return math.fsum(event.amount_usd for event in events)

    def estimated_total(
        self,
        *,
        run_id: str | None = None,
        provider: str | None = None,
        operation: str | None = None,
    ) -> float:
        """Return the summed USD amount of estimated events only."""
        return self.total(
            run_id=run_id,
            provider=provider,
            operation=operation,
            estimated=True,
        )

    def actual_total(
        self,
        *,
        run_id: str | None = None,
        provider: str | None = None,
        operation: str | None = None,
    ) -> float:
        """Return the summed USD amount of actual events only."""
        return self.total(
            run_id=run_id,
            provider=provider,
            operation=operation,
            estimated=False,
        )

    def count(
        self,
        *,
        run_id: str | None = None,
        provider: str | None = None,
        operation: str | None = None,
        estimated: bool | None = None,
    ) -> int:
        """Return how many events match the given filters."""
        return len(
            self.events(
                run_id=run_id,
                provider=provider,
                operation=operation,
                estimated=estimated,
            )
        )

    def totals(
        self,
        *,
        run_id: str | None = None,
        provider: str | None = None,
        operation: str | None = None,
    ) -> dict[str, float | int]:
        """Return total, estimated, actual and count for the matching events.

        Estimates replaced by an actual event are excluded from every amount
        and from the count, because the actual amount already accounts for the
        call.
        """
        events = self._effective_events(
            self.events(
                run_id=run_id,
                provider=provider,
                operation=operation,
            )
        )
        estimated = math.fsum(event.amount_usd for event in events if event.estimated)
        actual = math.fsum(event.amount_usd for event in events if not event.estimated)
        return {
            "total": math.fsum(event.amount_usd for event in events),
            "estimated": estimated,
            "actual": actual,
            "count": len(events),
        }

    @property
    def kinds(self) -> tuple[str, str]:
        """The two accepted estimated/actual flag values."""
        return (CostKind.ESTIMATED, CostKind.ACTUAL)


CostLedger = CostTracker
"""Alias for :class:`CostTracker` for callers that prefer the ledger wording."""

CostRecorder = CostTracker
"""Alias for :class:`CostTracker` for callers that prefer the recorder wording."""

CostEventRecorder = CostTracker
"""Alias for :class:`CostTracker` for callers that prefer the event wording."""

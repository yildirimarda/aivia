"""SQL-level aggregation over the cost event ledger.

The ledger records one row per paid call. Reports, budgets and the dashboard
all ask the same question in different shapes: how much did we spend per run,
day, week, month, provider, model, profile, format or stage? This module
answers it with a single ``GROUP BY`` query instead of loading every event into
Python, so the answer stays cheap as the ledger grows.

``profile`` and ``format`` live on the run a cost belongs to, so those groups
join :data:`~aivia.core.db.tables.runs`. A cost event without a matching run is
still counted, under the empty key, because spend must never disappear from a
report. ``stage`` is the cost event's operation, which is the capability or
pipeline step that spent the money.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from enum import StrEnum
from typing import Any

from sqlalchemy import Engine, case, func, select

from aivia.core.db.repos import CostEventRepository
from aivia.core.db.tables import cost_events, runs

__all__ = [
    "CostAggregate",
    "CostAggregator",
    "CostGroup",
]


class CostGroup(StrEnum):
    """The dimensions a cost report can group by."""

    RUN = "run"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    PROVIDER = "provider"
    MODEL = "model"
    PROFILE = "profile"
    FORMAT = "format"
    STAGE = "stage"


@dataclass(frozen=True, slots=True)
class CostAggregate:
    """One bucket of the ledger: a group key and the totals behind it.

    ``key`` is the group value, for example a run id, ``2026-01-05`` for a day,
    ``2026-W01`` for a week, ``2026-01`` for a month or a provider name. An
    event whose run has no profile or format appears under the empty key.
    """

    group: CostGroup
    key: str
    total_usd: float
    estimated_usd: float
    actual_usd: float
    units: float
    count: int

    @property
    def total(self) -> float:
        """Alias for :attr:`total_usd`."""
        return self.total_usd

    @property
    def estimated(self) -> float:
        """Alias for :attr:`estimated_usd`."""
        return self.estimated_usd

    @property
    def actual(self) -> float:
        """Alias for :attr:`actual_usd`."""
        return self.actual_usd

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for exports and logs."""
        return {
            "group": str(self.group),
            "key": self.key,
            "total_usd": self.total_usd,
            "estimated_usd": self.estimated_usd,
            "actual_usd": self.actual_usd,
            "units": self.units,
            "count": self.count,
        }


def _coerce_group(group: CostGroup | str) -> CostGroup:
    if isinstance(group, CostGroup):
        return group
    try:
        return CostGroup(str(group).strip().lower())
    except ValueError as exc:
        allowed = ", ".join(item.value for item in CostGroup)
        raise ValueError(
            f"unknown cost group {group!r}; expected one of {allowed}"
        ) from exc


def _needs_run_join(group: CostGroup) -> bool:
    return group in {CostGroup.PROFILE, CostGroup.FORMAT}


def _not_replaced() -> Any:
    """Exclude estimates whose call has a later actual replacement."""
    link = func.json_extract(cost_events.c.metadata, "$.replaces")
    replaced = select(link).where(
        cost_events.c.estimated.is_(False),
        link.is_not(None),
        link != "",
    )
    return ~cost_events.c.id.in_(replaced)


def _group_expression(group: CostGroup) -> Any:
    if group is CostGroup.RUN:
        return cost_events.c.run_id
    if group is CostGroup.PROVIDER:
        return cost_events.c.provider
    if group is CostGroup.MODEL:
        return cost_events.c.model
    if group is CostGroup.STAGE:
        return cost_events.c.operation
    if group is CostGroup.PROFILE:
        return func.coalesce(runs.c.profile, "")
    if group is CostGroup.FORMAT:
        return func.coalesce(runs.c.format, "")
    if group is CostGroup.DAY:
        return func.strftime("%Y-%m-%d", cost_events.c.created_at)
    if group is CostGroup.WEEK:
        return func.strftime("%Y-W%W", cost_events.c.created_at)
    if group is CostGroup.MONTH:
        return func.strftime("%Y-%m", cost_events.c.created_at)
    raise ValueError(f"unsupported cost group {group!r}")


def _as_datetime(value: datetime | date | str) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min, tzinfo=UTC)
    return datetime.fromisoformat(str(value).strip())


def _upper_bound(value: datetime | date | str) -> datetime:
    if isinstance(value, date) and not isinstance(value, datetime):
        return datetime.combine(value, time.max, tzinfo=UTC)
    return _as_datetime(value)


class CostAggregator:
    """Run grouped cost queries against the ledger.

    Pass an SQLAlchemy :class:`~sqlalchemy.Engine` or a ready
    :class:`~aivia.core.db.repos.CostEventRepository`, mirroring
    :class:`~aivia.core.cost.CostTracker`.
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
                    "CostAggregator needs an engine or a cost event repository"
                )
            target = engine

        if isinstance(target, CostEventRepository):
            self._engine = target.engine
        elif isinstance(target, Engine):
            self._engine = target
        else:
            raise TypeError(
                "engine must be a SQLAlchemy Engine or a CostEventRepository"
            )

    @property
    def engine(self) -> Engine:
        """The engine every aggregation query runs on."""
        return self._engine

    def aggregate(
        self,
        group: CostGroup | str,
        *,
        start: datetime | date | str | None = None,
        end: datetime | date | str | None = None,
        run_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        stage: str | None = None,
        estimated: bool | None = None,
    ) -> list[CostAggregate]:
        """Return one bucket per *group* value, ordered by the group key.

        *start* is inclusive and *end* is inclusive; a date end includes the
        whole day. The optional equality filters narrow the ledger before the
        grouping, so a report can answer "spend per model for this run" without
        loading the events.
        """
        dimension = _coerce_group(group)
        key = _group_expression(dimension)
        statement = (
            select(
                key.label("key"),
                func.coalesce(func.sum(cost_events.c.amount_usd), 0.0).label(
                    "total_usd"
                ),
                func.coalesce(
                    func.sum(
                        case(
                            (
                                cost_events.c.estimated.is_(True),
                                cost_events.c.amount_usd,
                            ),
                            else_=0.0,
                        )
                    ),
                    0.0,
                ).label("estimated_usd"),
                func.coalesce(
                    func.sum(
                        case(
                            (
                                cost_events.c.estimated.is_(False),
                                cost_events.c.amount_usd,
                            ),
                            else_=0.0,
                        )
                    ),
                    0.0,
                ).label("actual_usd"),
                func.coalesce(func.sum(cost_events.c.units), 0.0).label("units"),
                func.count().label("count"),
            )
            .select_from(cost_events)
            .where(_not_replaced())
            .group_by(key)
            .order_by(key)
        )
        if _needs_run_join(dimension):
            statement = statement.outerjoin(runs, cost_events.c.run_id == runs.c.id)
        if start is not None:
            statement = statement.where(cost_events.c.created_at >= _as_datetime(start))
        if end is not None:
            statement = statement.where(cost_events.c.created_at <= _upper_bound(end))
        if run_id is not None:
            statement = statement.where(cost_events.c.run_id == run_id)
        if provider is not None:
            statement = statement.where(cost_events.c.provider == provider)
        if model is not None:
            statement = statement.where(cost_events.c.model == model)
        if stage is not None:
            statement = statement.where(cost_events.c.operation == stage)
        if estimated is not None:
            statement = statement.where(cost_events.c.estimated.is_(estimated))

        with self._engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()

        return [
            CostAggregate(
                group=dimension,
                key=str(row["key"] or ""),
                total_usd=float(row["total_usd"]),
                estimated_usd=float(row["estimated_usd"]),
                actual_usd=float(row["actual_usd"]),
                units=float(row["units"]),
                count=int(row["count"]),
            )
            for row in rows
        ]

    def by_run(self, **filters: Any) -> list[CostAggregate]:
        """Aggregate spend per run."""
        return self.aggregate(CostGroup.RUN, **filters)

    def by_day(self, **filters: Any) -> list[CostAggregate]:
        """Aggregate spend per UTC calendar day."""
        return self.aggregate(CostGroup.DAY, **filters)

    def by_week(self, **filters: Any) -> list[CostAggregate]:
        """Aggregate spend per UTC week, keyed ``YYYY-Www``."""
        return self.aggregate(CostGroup.WEEK, **filters)

    def by_month(self, **filters: Any) -> list[CostAggregate]:
        """Aggregate spend per UTC month, keyed ``YYYY-MM``."""
        return self.aggregate(CostGroup.MONTH, **filters)

    def by_provider(self, **filters: Any) -> list[CostAggregate]:
        """Aggregate spend per provider."""
        return self.aggregate(CostGroup.PROVIDER, **filters)

    def by_model(self, **filters: Any) -> list[CostAggregate]:
        """Aggregate spend per model."""
        return self.aggregate(CostGroup.MODEL, **filters)

    def by_profile(self, **filters: Any) -> list[CostAggregate]:
        """Aggregate spend per content profile, joining the run it belongs to."""
        return self.aggregate(CostGroup.PROFILE, **filters)

    def by_format(self, **filters: Any) -> list[CostAggregate]:
        """Aggregate spend per format, joining the run it belongs to."""
        return self.aggregate(CostGroup.FORMAT, **filters)

    def by_stage(self, **filters: Any) -> list[CostAggregate]:
        """Aggregate spend per stage, using the cost event's operation."""
        return self.aggregate(CostGroup.STAGE, **filters)

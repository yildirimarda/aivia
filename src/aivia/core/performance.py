"""Performance reporting across cost, views and content dimensions.

Reads the cost ledger together with the platform metric snapshots so spend can
be viewed beside the views it bought, grouped by profile, mode, category,
language, platform, format, hook style and visual strategy, with cost per
thousand views (CPM) for every group.

A run's spend is the sum of its effective cost events: an estimate that a later
actual event has replaced is excluded, so one call contributes either its
estimate or its actual amount, never both. A run's views are the sum of the
views snapshotted for the run's posts. Each run contributes its spend and views
to every dimension value it carries -- profile and format from the run,
platform and language from its posts, and mode, category, hook style and
visual strategy from the run's cost event metadata -- so a group's CPM is
spend divided by views times one thousand, or zero when the group has no
views.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine, func, select

from aivia.core.db.tables import cost_events, post_metrics, posts, runs

__all__ = [
    "PerformanceBucket",
    "PerformanceReport",
    "build_performance_report",
    "performance_by",
]

#: Metadata keys on a cost event that name a reporting dimension.
_DIMENSION_KEYS = ("mode", "category", "language", "hook_style", "visual_strategy")

#: Every dimension the report can group by.
_DIMENSIONS = (
    "profile",
    "mode",
    "category",
    "language",
    "platform",
    "format",
    "hook_style",
    "visual_strategy",
)


@dataclass(frozen=True, slots=True)
class PerformanceBucket:
    """One group of a performance report: a dimension value and its totals."""

    dimension: str
    """The dimension the bucket is grouped by, e.g. ``"platform"``."""

    key: str
    """The dimension value, e.g. ``"tiktok"``, or ``"(none)"`` when unset."""

    cost_usd: float
    """The spend attributed to the group."""

    views: int
    """The views attributed to the group."""

    cpm_usd: float
    """Cost per thousand views: ``cost_usd / views * 1000``, or ``0.0``."""


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    """The full performance report across every configured dimension."""

    generated_at: datetime
    by_profile: tuple[PerformanceBucket, ...]
    by_mode: tuple[PerformanceBucket, ...]
    by_category: tuple[PerformanceBucket, ...]
    by_language: tuple[PerformanceBucket, ...]
    by_platform: tuple[PerformanceBucket, ...]
    by_format: tuple[PerformanceBucket, ...]
    by_hook_style: tuple[PerformanceBucket, ...]
    by_visual_strategy: tuple[PerformanceBucket, ...]

    @property
    def total_cost_usd(self) -> float:
        """The total spend across runs that have any spend recorded."""
        return sum(bucket.cost_usd for bucket in self.by_profile)

    @property
    def total_views(self) -> int:
        """The total views across runs that have any views recorded."""
        return sum(bucket.views for bucket in self.by_platform)

    @property
    def overall_cpm_usd(self) -> float:
        """Overall cost per thousand views, or zero when there are no views."""
        views = self.total_views
        if not views:
            return 0.0
        return self.total_cost_usd / views * 1000.0


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _empty_key(value: Any) -> str:
    return str(value or "") or "(none)"


def _replaced_estimate_condition() -> Any:
    """A where clause excluding estimates an actual event has replaced.

    The ``replaces`` link lives inside an actual event's metadata JSON under
    ``$.replaces``. Any estimate whose id is referenced there is superseded and
    must not be counted alongside its actual replacement.
    """
    link = func.json_extract(cost_events.c.metadata, "$.replaces")
    replaced = select(link).where(
        cost_events.c.estimated.is_(False),
        link.is_not(None),
        link != "",
    )
    return ~cost_events.c.id.in_(replaced)


def _run_facts(engine: Engine) -> list[dict[str, Any]]:
    """Return one row per run: its spend, views and dimension values."""
    # Spend per run, excluding estimates an actual event has replaced.
    spend_statement = (
        select(
            cost_events.c.run_id.label("run_id"),
            func.coalesce(func.sum(cost_events.c.amount_usd), 0.0).label("cost_usd"),
        )
        .select_from(cost_events)
        .where(_replaced_estimate_condition())
        .group_by(cost_events.c.run_id)
    )
    spend_by_run: dict[str, float] = {}
    with engine.connect() as connection:
        for row in connection.execute(spend_statement).mappings():
            spend_by_run[str(row["run_id"] or "")] = float(row["cost_usd"] or 0.0)

    # Views per run, summed over the run's posts and every snapshot mark.
    views_statement = (
        select(
            posts.c.run_id.label("run_id"),
            func.coalesce(func.sum(post_metrics.c.views), 0).label("views"),
        )
        .select_from(posts)
        .outerjoin(post_metrics, post_metrics.c.post_id == posts.c.id)
        .group_by(posts.c.run_id)
    )
    views_by_run: dict[str, int] = {}
    with engine.connect() as connection:
        for row in connection.execute(views_statement).mappings():
            views_by_run[str(row["run_id"] or "")] = int(row["views"] or 0)

    # Run level dimensions: profile and format.
    run_statement = select(runs.c.id, runs.c.profile, runs.c.format)
    runs_by_id: dict[str, dict[str, str]] = {}
    with engine.connect() as connection:
        for row in connection.execute(run_statement).mappings():
            runs_by_id[str(row["id"] or "")] = {
                "profile": str(row["profile"] or ""),
                "format": str(row["format"] or ""),
            }

    # Post level dimensions: platform and language, grouped by run.
    post_statement = select(
        posts.c.id, posts.c.run_id, posts.c.platform, posts.c.language
    )
    run_posts: dict[str, list[dict[str, str]]] = {}
    with engine.connect() as connection:
        for row in connection.execute(post_statement).mappings():
            run_id = str(row["run_id"] or "")
            run_posts.setdefault(run_id, []).append(
                {
                    "platform": str(row["platform"] or ""),
                    "language": str(row["language"] or ""),
                }
            )

    # Cost event metadata dimensions: mode, category, hook style, visual strategy.
    metadata_statement = select(
        cost_events.c.run_id,
        cost_events.c.metadata,
    )
    run_metadata: dict[str, dict[str, str]] = {}
    with engine.connect() as connection:
        for row in connection.execute(metadata_statement).mappings():
            run_id = str(row["run_id"] or "")
            metadata = row["metadata"]
            if not isinstance(metadata, Mapping):
                continue
            entry = run_metadata.setdefault(run_id, {})
            for key in _DIMENSION_KEYS:
                value = metadata.get(key)
                if value is None or value == "":
                    continue
                entry.setdefault(key, str(value))

    # Assemble one fact row per run. A run that carries neither spend nor views
    # contributes nothing to a performance report, so it is left out entirely.
    run_ids = (
        set(spend_by_run)
        | set(views_by_run)
        | set(runs_by_id)
        | set(run_posts)
        | set(run_metadata)
    )
    facts: list[dict[str, Any]] = []
    for run_id in run_ids:
        cost_usd = spend_by_run.get(run_id, 0.0)
        views = views_by_run.get(run_id, 0)
        if not cost_usd and not views:
            continue
        facts.append(
            {
                "run_id": run_id,
                "cost_usd": cost_usd,
                "views": views,
                "profile": runs_by_id.get(run_id, {}).get("profile", ""),
                "format": runs_by_id.get(run_id, {}).get("format", ""),
                "platforms": [p["platform"] for p in run_posts.get(run_id, [])],
                "languages": [p["language"] for p in run_posts.get(run_id, [])],
                "mode": run_metadata.get(run_id, {}).get("mode", ""),
                "category": run_metadata.get(run_id, {}).get("category", ""),
                "hook_style": run_metadata.get(run_id, {}).get("hook_style", ""),
                "visual_strategy": run_metadata.get(run_id, {}).get(
                    "visual_strategy", ""
                ),
            }
        )
    return facts


def _expand(
    facts: list[dict[str, Any]], dimension: str
) -> list[tuple[str, float, int]]:
    """Return ``(key, cost_usd, views)`` rows for *facts* grouped by dimension.

    A run that carries several values for a dimension (for example posts on two
    platforms) contributes its full spend and views to each value, so a group's
    CPM stays internally consistent. A run that carries no value for a dimension
    reports under ``"(none)"``.
    """
    rows: list[tuple[str, float, int]] = []
    for fact in facts:
        cost_usd = float(fact.get("cost_usd") or 0.0)
        views = int(fact.get("views") or 0)
        if dimension in {"platform", "language"}:
            values = fact.get(dimension + "s") or []
            if not values:
                rows.append((_empty_key(None), cost_usd, views))
                continue
            for value in values:
                rows.append((_empty_key(value), cost_usd, views))
        else:
            rows.append((_empty_key(fact.get(dimension)), cost_usd, views))
    return rows


def _buckets(
    facts: list[dict[str, Any]],
    dimension: str,
) -> tuple[PerformanceBucket, ...]:
    """Aggregate *facts* by *dimension* into cost, views and CPM buckets."""
    totals: dict[str, dict[str, Any]] = {}
    for key, cost_usd, views in _expand(facts, dimension):
        bucket = totals.setdefault(key, {"cost_usd": 0.0, "views": 0})
        bucket["cost_usd"] += cost_usd
        bucket["views"] += views

    ordered: list[PerformanceBucket] = []
    for key in sorted(totals):
        measure = totals[key]
        cost_usd = measure["cost_usd"]
        views = measure["views"]
        cpm_usd = cost_usd / views * 1000.0 if views else 0.0
        ordered.append(
            PerformanceBucket(
                dimension=dimension,
                key=key,
                cost_usd=cost_usd,
                views=views,
                cpm_usd=cpm_usd,
            )
        )
    return tuple(ordered)


def performance_by(engine: Engine, dimension: str) -> tuple[PerformanceBucket, ...]:
    """Return performance buckets grouped by *dimension*.

    *dimension* is one of ``profile``, ``mode``, ``category``, ``language``,
    ``platform``, ``format``, ``hook_style`` or ``visual_strategy``.
    """
    if dimension not in _DIMENSIONS:
        raise ValueError(
            f"unknown performance dimension {dimension!r}; "
            f"expected one of {', '.join(_DIMENSIONS)}"
        )
    return _buckets(_run_facts(engine), dimension)


def build_performance_report(engine: Engine) -> PerformanceReport:
    """Build the full performance report across every dimension."""
    facts = _run_facts(engine)
    moment = _as_utc(datetime.now(UTC))
    return PerformanceReport(
        generated_at=moment,
        by_profile=_buckets(facts, "profile"),
        by_mode=_buckets(facts, "mode"),
        by_category=_buckets(facts, "category"),
        by_language=_buckets(facts, "language"),
        by_platform=_buckets(facts, "platform"),
        by_format=_buckets(facts, "format"),
        by_hook_style=_buckets(facts, "hook_style"),
        by_visual_strategy=_buckets(facts, "visual_strategy"),
    )

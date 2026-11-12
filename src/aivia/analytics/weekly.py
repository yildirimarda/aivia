"""Weekly report of spend, outputs, top performers and provider comparison.

The report scopes the cost ledger and the platform metric snapshots to a single
UTC week (Monday to Sunday) and summarizes how much was spent, what was
produced, which posts performed best and how providers compare on spend and
call volume. The result can be rendered as a
:class:`~aivia.providers.protocols.notify.NotificationMessage` and delivered
    through any :class:`~aivia.providers.protocols.notify.Notifier`.
"""

# config-coverage: allow-file — TOP_PERFORMER_LIMIT is an internal default
# tuning constant for the weekly report, not user-facing vendor, model or
# price configuration.

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import Engine, func, select

from aivia.core.db.tables import cost_events, post_metrics, posts
from aivia.core.models.base import utcnow
from aivia.providers.protocols.notify import (
    NotificationLevel,
    NotificationMessage,
    NotificationReceipt,
    Notifier,
)

__all__ = [
    "TOP_PERFORMER_LIMIT",
    "WeeklyProviderComparison",
    "WeeklyReport",
    "WeeklyTopPerformer",
    "build_weekly_report",
    "deliver_weekly_report",
    "weekly_report_message",
]

#: How many top performing posts the report lists.
TOP_PERFORMER_LIMIT = 10


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _week_bounds(week_start: date) -> tuple[datetime, datetime]:
    """Return the inclusive start and exclusive end of the UTC week."""
    start = datetime(week_start.year, week_start.month, week_start.day, tzinfo=UTC)
    return start, start + timedelta(days=7)


def _not_replaced() -> Any:
    """Exclude estimates an actual event has replaced.

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


@dataclass(frozen=True, slots=True)
class WeeklyProviderComparison:
    """One provider's share of the week's spend and call volume."""

    provider: str
    """The provider name."""

    spend_usd: float
    """The spend attributed to the provider, excluding replaced estimates."""

    calls: int
    """The number of cost events attributed to the provider."""


@dataclass(frozen=True, slots=True)
class WeeklyTopPerformer:
    """One post and its engagement, ranked by views."""

    post_id: str
    """The post id."""

    platform: str
    """The platform the post was published on."""

    views: int
    """The views snapshotted for the post."""

    likes: int
    """The likes snapshotted for the post."""

    comments: int
    """The comments snapshotted for the post."""

    shares: int
    """The shares snapshotted for the post."""


@dataclass(frozen=True, slots=True)
class WeeklyReport:
    """The full weekly report across spend, outputs and providers."""

    week_start: date
    """The Monday the week starts on."""

    week_end: date
    """The Sunday the week ends on (exclusive bound)."""

    spend_usd: float
    """The total spend for the week."""

    outputs: int
    """The number of posts produced in the week."""

    top_performers: tuple[WeeklyTopPerformer, ...]
    """The top performing posts, ranked by views."""

    provider_comparison: tuple[WeeklyProviderComparison, ...]
    """Each provider's spend and call volume, ranked by spend."""

    generated_at: datetime
    """The instant the report was generated."""


def build_weekly_report(
    engine: Engine, week_start: date, *, now: datetime | None = None
) -> WeeklyReport:
    """Build the weekly report for the week starting on *week_start*.

    The week covers ``[week_start, week_start + 7 days)`` in UTC. Spend excludes
    estimates that a later actual event has replaced. Outputs counts posts
    created in the week. Top performers are posts with metric snapshots ranked
    by views. Provider comparison groups spend and call volume by provider.
    """
    week_start_dt, week_end_dt = _week_bounds(week_start)
    generated_at = _as_utc(now if now is not None else utcnow())

    # Spend, excluding estimates an actual event has replaced.
    spend_statement = (
        select(func.coalesce(func.sum(cost_events.c.amount_usd), 0.0))
        .select_from(cost_events)
        .where(cost_events.c.created_at >= week_start_dt)
        .where(cost_events.c.created_at < week_end_dt)
        .where(_not_replaced())
    )
    with engine.connect() as connection:
        spend_usd = float(connection.execute(spend_statement).scalar() or 0.0)

    # Outputs: posts produced in the week.
    outputs_statement = (
        select(func.count())
        .select_from(posts)
        .where(posts.c.created_at >= week_start_dt)
        .where(posts.c.created_at < week_end_dt)
    )
    with engine.connect() as connection:
        outputs = int(connection.execute(outputs_statement).scalar() or 0)

    # Top performers: posts with metrics, ranked by views.
    top_statement = (
        select(
            posts.c.id,
            posts.c.platform,
            post_metrics.c.views,
            post_metrics.c.likes,
            post_metrics.c.comments,
            post_metrics.c.shares,
        )
        .select_from(posts)
        .join(post_metrics, post_metrics.c.post_id == posts.c.id)
        .where(posts.c.created_at >= week_start_dt)
        .where(posts.c.created_at < week_end_dt)
        .order_by(post_metrics.c.views.desc(), posts.c.id)
        .limit(TOP_PERFORMER_LIMIT)
    )
    top_performers: list[WeeklyTopPerformer] = []
    with engine.connect() as connection:
        for row in connection.execute(top_statement).mappings():
            top_performers.append(
                WeeklyTopPerformer(
                    post_id=str(row["id"]),
                    platform=str(row["platform"] or ""),
                    views=int(row["views"] or 0),
                    likes=int(row["likes"] or 0),
                    comments=int(row["comments"] or 0),
                    shares=int(row["shares"] or 0),
                )
            )

    # Provider comparison: spend and call volume per provider.
    provider_statement = (
        select(
            cost_events.c.provider,
            func.coalesce(func.sum(cost_events.c.amount_usd), 0.0).label("spend_usd"),
            func.count().label("calls"),
        )
        .select_from(cost_events)
        .where(cost_events.c.created_at >= week_start_dt)
        .where(cost_events.c.created_at < week_end_dt)
        .where(_not_replaced())
        .group_by(cost_events.c.provider)
        .order_by(func.sum(cost_events.c.amount_usd).desc(), cost_events.c.provider)
    )
    provider_comparison: list[WeeklyProviderComparison] = []
    with engine.connect() as connection:
        for row in connection.execute(provider_statement).mappings():
            provider_comparison.append(
                WeeklyProviderComparison(
                    provider=str(row["provider"] or ""),
                    spend_usd=float(row["spend_usd"] or 0.0),
                    calls=int(row["calls"] or 0),
                )
            )

    return WeeklyReport(
        week_start=week_start,
        week_end=week_start + timedelta(days=7),
        spend_usd=spend_usd,
        outputs=outputs,
        top_performers=tuple(top_performers),
        provider_comparison=tuple(provider_comparison),
        generated_at=generated_at,
    )


def _money(amount: float) -> str:
    return f"${amount:,.2f}"


def weekly_report_message(report: WeeklyReport) -> NotificationMessage:
    """Render *report* as a notification message."""
    lines: list[str] = [
        f"Weekly report {report.week_start} to {report.week_end}",
        f"Spend: {_money(report.spend_usd)}",
        f"Outputs: {report.outputs} post{'s' if report.outputs != 1 else ''}",
    ]
    if report.top_performers:
        lines.append("Top performers:")
        for performer in report.top_performers:
            lines.append(
                f"  {performer.post_id} ({performer.platform}): {performer.views} views"
            )
    if report.provider_comparison:
        lines.append("Provider comparison:")
        for provider in report.provider_comparison:
            lines.append(
                f"  {provider.provider}: {_money(provider.spend_usd)} "
                f"({provider.calls} call{'s' if provider.calls != 1 else ''})"
            )
    return NotificationMessage(
        title="Weekly report",
        text="\n".join(lines),
        level=NotificationLevel.INFO,
        metadata={
            "week_start": report.week_start.isoformat(),
            "week_end": report.week_end.isoformat(),
            "spend_usd": report.spend_usd,
            "outputs": report.outputs,
        },
    )


async def deliver_weekly_report(
    engine: Engine,
    week_start: date,
    notifier: Notifier,
    *,
    now: datetime | None = None,
) -> NotificationReceipt:
    """Build the weekly report and deliver it through *notifier*."""
    report = build_weekly_report(engine, week_start, now=now)
    message = weekly_report_message(report)
    return await notifier.send(message)

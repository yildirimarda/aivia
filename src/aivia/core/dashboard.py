"""Render the cost ledger as a static, self-contained HTML dashboard.

The dashboard is a read-only view over the same aggregates the ``costs``
reports use: spend per day, per run and per provider plus progress bars for the
configured budget caps. It contains no scripts, stylesheets or fonts fetched
from the network, so the generated file can be opened offline or archived next
to a run.

Budget bars use the current UTC day, month and the configured provider monthly
limits, matching :class:`~aivia.core.budget.BudgetGuard`. A per-run bar is shown
when ``budget.per_run`` is configured. A cap of ``0`` means the scope has no cap
and is omitted instead of rendered as an empty bar.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta
from typing import Any

from sqlalchemy import Engine

from aivia.core.config.budget import BudgetConfig
from aivia.core.cost_agg import CostAggregate, CostAggregator
from aivia.core.performance import PerformanceBucket, build_performance_report

__all__ = [
    "BudgetBar",
    "DashboardData",
    "build_dashboard",
    "render_dashboard",
]


@dataclass(frozen=True, slots=True)
class BudgetBar:
    """One configured cap together with the spend counted against it."""

    label: str
    scope: str
    spent_usd: float
    cap_usd: float

    @property
    def fraction(self) -> float:
        """The used share of the cap, where ``1.0`` means exactly at the cap."""
        if not self.cap_usd:
            return 0.0
        return self.spent_usd / self.cap_usd

    @property
    def percent(self) -> float:
        """The used share as a percentage."""
        return self.fraction * 100.0

    @property
    def width_percent(self) -> float:
        """The bar width, clamped to the 0-100 range."""
        return min(max(self.percent, 0.0), 100.0)

    @property
    def exceeded(self) -> bool:
        """Whether spend is above the cap."""
        return self.spent_usd > self.cap_usd

    @property
    def remaining_usd(self) -> float:
        """The part of the cap that is still available."""
        return max(self.cap_usd - self.spent_usd, 0.0)


@dataclass(frozen=True, slots=True)
class DashboardData:
    """Everything the static dashboard renders."""

    generated_at: datetime
    days: tuple[CostAggregate, ...]
    runs: tuple[CostAggregate, ...]
    providers: tuple[CostAggregate, ...]
    budgets: tuple[BudgetBar, ...]
    performance: dict[str, tuple[PerformanceBucket, ...]] = field(default_factory=dict)
    overall_cpm_usd: float = 0.0

    @property
    def total_usd(self) -> float:
        """The total of the day buckets, which matches the other sections."""
        return sum(bucket.total_usd for bucket in self.days)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _day_window(now: datetime) -> tuple[datetime, datetime]:
    start = datetime.combine(now.date(), time.min, tzinfo=UTC)
    return start, start + timedelta(days=1)


def _month_window(now: datetime) -> tuple[datetime, datetime]:
    start = datetime(now.year, now.month, 1, tzinfo=UTC)
    if now.month == 12:
        end = datetime(now.year + 1, 1, 1, tzinfo=UTC)
    else:
        end = datetime(now.year, now.month + 1, 1, tzinfo=UTC)
    return start, end


def _inclusive_end(exclusive: datetime) -> datetime:
    return exclusive - timedelta.resolution


def _spent(
    aggregator: CostAggregator,
    *,
    start: datetime,
    end: datetime,
) -> float:
    buckets = aggregator.by_day(start=start, end=_inclusive_end(end))
    return sum(bucket.total_usd for bucket in buckets)


def _provider_spend(
    aggregator: CostAggregator,
    *,
    start: datetime,
    end: datetime,
) -> dict[str, float]:
    buckets = aggregator.by_provider(start=start, end=_inclusive_end(end))
    return {bucket.key: bucket.total_usd for bucket in buckets}


def _budget_bars(
    aggregator: CostAggregator,
    budget: BudgetConfig,
    *,
    now: datetime,
    runs: tuple[CostAggregate, ...],
) -> tuple[BudgetBar, ...]:
    bars: list[BudgetBar] = []

    day_start, day_end = _day_window(now)
    if budget.per_day:
        bars.append(
            BudgetBar(
                label="Today",
                scope="per_day",
                spent_usd=_spent(aggregator, start=day_start, end=day_end),
                cap_usd=budget.per_day,
            )
        )

    month_start, month_end = _month_window(now)
    if budget.per_month:
        bars.append(
            BudgetBar(
                label="This month",
                scope="per_month",
                spent_usd=_spent(aggregator, start=month_start, end=month_end),
                cap_usd=budget.per_month,
            )
        )

    if budget.per_run:
        for bucket in runs:
            bars.append(
                BudgetBar(
                    label=f"Run {bucket.key or '(none)'}",
                    scope="per_run",
                    spent_usd=bucket.total_usd,
                    cap_usd=budget.per_run,
                )
            )

    if budget.per_provider:
        provider_spend = _provider_spend(
            aggregator,
            start=month_start,
            end=month_end,
        )
        for name in sorted(budget.per_provider):
            cap = budget.per_provider[name]
            if not cap:
                continue
            bars.append(
                BudgetBar(
                    label=f"Provider {name}",
                    scope="per_provider",
                    spent_usd=provider_spend.get(name, 0.0),
                    cap_usd=cap,
                )
            )

    return tuple(bars)


def build_dashboard(
    engine: Engine,
    budget: BudgetConfig,
    *,
    now: datetime | None = None,
    start: Any = None,
    end: Any = None,
    run_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    stage: str | None = None,
) -> DashboardData:
    """Read the ledger and return the data a dashboard renders.

    *now* selects the UTC day and month the budget bars measure; it defaults to
    the current time. The remaining arguments filter the three spend tables but
    never the budget bars, because a budget is global rather than a report
    filter.
    """
    moment = _as_utc(now or datetime.now(UTC))
    aggregator = CostAggregator(engine)
    filters: dict[str, Any] = {
        "start": start,
        "end": end,
        "run_id": run_id,
        "provider": provider,
        "model": model,
        "stage": stage,
    }
    filters = {key: value for key, value in filters.items() if value is not None}

    days = tuple(aggregator.by_day(**filters))
    runs = tuple(aggregator.by_run(**filters))
    providers = tuple(aggregator.by_provider(**filters))
    all_runs = tuple(aggregator.by_run())
    budgets = _budget_bars(aggregator, budget, now=moment, runs=all_runs)
    performance = build_performance_report(engine)

    return DashboardData(
        generated_at=moment,
        days=days,
        runs=runs,
        providers=providers,
        budgets=budgets,
        performance={
            "profile": performance.by_profile,
            "mode": performance.by_mode,
            "category": performance.by_category,
            "language": performance.by_language,
            "platform": performance.by_platform,
            "format": performance.by_format,
            "hook_style": performance.by_hook_style,
            "visual_strategy": performance.by_visual_strategy,
        },
        overall_cpm_usd=performance.overall_cpm_usd,
    )


def _money(value: float) -> str:
    return f"${value:.6f}"


def _table(title: str, key_header: str, buckets: tuple[CostAggregate, ...]) -> str:
    rows = [
        (
            "<tr>"
            f"<td>{html.escape(bucket.key or '(none)')}</td>"
            f'<td class="num">{_money(bucket.total_usd)}</td>'
            f'<td class="num">{_money(bucket.estimated_usd)}</td>'
            f'<td class="num">{_money(bucket.actual_usd)}</td>'
            f'<td class="num">{bucket.units:g}</td>'
            f'<td class="num">{bucket.count}</td>'
            "</tr>"
        )
        for bucket in buckets
    ]
    body = "".join(rows) or (
        '<tr><td colspan="6" class="empty">No spend recorded.</td></tr>'
    )
    return (
        "<section>"
        f"<h2>{html.escape(title)}</h2>"
        "<table>"
        "<thead><tr>"
        f"<th>{html.escape(key_header)}</th>"
        '<th class="num">Total</th>'
        '<th class="num">Estimated</th>'
        '<th class="num">Actual</th>'
        '<th class="num">Units</th>'
        '<th class="num">Count</th>'
        "</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
        "</section>"
    )


def _performance_table(
    title: str,
    key_header: str,
    buckets: tuple[PerformanceBucket, ...],
) -> str:
    rows = [
        (
            "<tr>"
            f"<td>{html.escape(bucket.key)}</td>"
            f'<td class="num">{_money(bucket.cost_usd)}</td>'
            f'<td class="num">{bucket.views:g}</td>'
            f'<td class="num">{_money(bucket.cpm_usd)}</td>'
            "</tr>"
        )
        for bucket in buckets
    ]
    body = "".join(rows) or (
        '<tr><td colspan="4" class="empty">No spend recorded.</td></tr>'
    )
    return (
        "<section>"
        f"<h2>{html.escape(title)}</h2>"
        "<table>"
        "<thead><tr>"
        f"<th>{html.escape(key_header)}</th>"
        '<th class="num">Cost</th>'
        '<th class="num">Views</th>'
        '<th class="num">CPM per 1k views</th>'
        "</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
        "</section>"
    )


def _budget_bar(bar: BudgetBar) -> str:
    if bar.exceeded:
        state = "over"
        note = f"Over by {_money(bar.spent_usd - bar.cap_usd)}"
    elif bar.fraction >= 0.8:
        state = "warn"
        note = f"{_money(bar.remaining_usd)} remaining"
    else:
        state = "ok"
        note = f"{_money(bar.remaining_usd)} remaining"
    return (
        '<div class="budget">'
        '<div class="budget-head">'
        f"<span>{html.escape(bar.label)}</span>"
        f"<span>{_money(bar.spent_usd)} / {_money(bar.cap_usd)} "
        f"({bar.percent:.1f}%)</span>"
        "</div>"
        f'<div class="bar" role="progressbar" aria-label="{html.escape(bar.label)}" '
        f'aria-valuenow="{bar.percent:.1f}" aria-valuemin="0" aria-valuemax="100">'
        f'<span class="fill {state}" style="width:{bar.width_percent:.1f}%"></span>'
        "</div>"
        f'<div class="budget-note">{note}</div>'
        "</div>"
    )


_STYLE = """
:root { color-scheme: light dark; }
body { font-family: system-ui, -apple-system, Segoe UI, sans-serif;
       margin: 2rem auto; max-width: 72rem; padding: 0 1rem; line-height: 1.45; }
header { margin-bottom: 2rem; }
h1 { margin: 0 0 .25rem; font-size: 1.6rem; }
h2 { margin: 0 0 .75rem; font-size: 1.15rem; }
.generated, .budget-note, .empty { color: #666; font-size: .875rem; }
section { margin-bottom: 2rem; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: .4rem .5rem;
         border-bottom: 1px solid rgba(127, 127, 127, .35); }
th { font-size: .8rem; text-transform: uppercase; letter-spacing: .04em; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.budget { margin-bottom: 1rem; }
.budget-head { display: flex; justify-content: space-between; gap: 1rem;
               font-size: .9rem; margin-bottom: .25rem; }
.bar { background: rgba(127, 127, 127, .25); border-radius: .5rem;
       height: .75rem; overflow: hidden; }
.fill { display: block; height: 100%; background: #2f855a; }
.fill.warn { background: #b7791f; }
.fill.over { background: #c53030; }
""".strip()


def render_dashboard(
    data: DashboardData,
    *,
    title: str = "AIVIA spend dashboard",
) -> str:
    """Render *data* as one self-contained HTML document."""
    generated = _as_utc(data.generated_at).isoformat()
    budgets = "".join(_budget_bar(bar) for bar in data.budgets) or (
        '<p class="empty">No budget caps configured.</p>'
    )
    performance_sections = "\n".join(
        _performance_table(title, key_header, buckets)
        for title, key_header, buckets in (
            ("Performance by profile", "Profile", data.performance.get("profile", ())),
            ("Performance by mode", "Mode", data.performance.get("mode", ())),
            (
                "Performance by category",
                "Category",
                data.performance.get("category", ()),
            ),
            (
                "Performance by language",
                "Language",
                data.performance.get("language", ()),
            ),
            (
                "Performance by platform",
                "Platform",
                data.performance.get("platform", ()),
            ),
            ("Performance by format", "Format", data.performance.get("format", ())),
            (
                "Performance by hook style",
                "Hook style",
                data.performance.get("hook_style", ()),
            ),
            (
                "Performance by visual strategy",
                "Visual strategy",
                data.performance.get("visual_strategy", ()),
            ),
        )
    )
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n"
        f"<style>{_STYLE}</style>\n"
        "</head>\n"
        "<body>\n"
        "<header>\n"
        f"<h1>{html.escape(title)}</h1>\n"
        f'<p class="generated">Generated {html.escape(generated)} · '
        f"total {_money(data.total_usd)}"
        f" · overall CPM {_money(data.overall_cpm_usd)}</p>\n"
        "</header>\n"
        "<main>\n"
        f"{_table('Spend per day', 'Day', data.days)}\n"
        f"{_table('Spend per run', 'Run', data.runs)}\n"
        f"{_table('Spend per provider', 'Provider', data.providers)}\n"
        f"{performance_sections}\n"
        "<section>\n"
        "<h2>Budgets</h2>\n"
        f"{budgets}\n"
        "</section>\n"
        "</main>\n"
        "</body>\n"
        "</html>\n"
    )

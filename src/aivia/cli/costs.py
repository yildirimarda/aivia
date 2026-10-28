"""``aivia costs``: report spend recorded in the cost ledger."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path

from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError

from aivia.cli.registry import register_command
from aivia.core import paths
from aivia.core.config.budget import BudgetConfig
from aivia.core.config.database import DatabaseConfig
from aivia.core.config.resolve import resolve_app_config
from aivia.core.cost import CostAggregator, CostTracker
from aivia.core.cost_agg import CostAggregate, CostGroup
from aivia.core.dashboard import build_dashboard, render_dashboard
from aivia.core.db import AssetRepository, RunRepository, create_database_engine
from aivia.core.errors import ConfigError
from aivia.core.models.asset import Asset
from aivia.core.models.cost import CostEvent

__all__ = [
    "configure",
    "run_costs",
    "run_costs_dashboard",
    "run_costs_export",
    "run_costs_run",
    "run_costs_summary",
]


def _configure_resolution(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="config file to merge instead of the default location",
    )
    parser.add_argument(
        "--env",
        "--environment",
        dest="environment",
        metavar="NAME",
        help="merge config.<NAME>.yaml beside the user config",
    )
    parser.add_argument(
        "--profile",
        metavar="NAME",
        help="merge the overrides of a configured profile",
    )
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        metavar="KEY=VALUE",
        help="override a dotted key; repeatable and applied after the profile",
    )


def _configure_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "run_id",
        nargs="?",
        metavar="RUN_ID",
        help="run whose spend to report",
    )
    parser.add_argument(
        "--run",
        "--run-id",
        dest="run_option",
        metavar="RUN_ID",
        help="run whose spend to report",
    )
    parser.add_argument(
        "--database",
        "--db",
        dest="database",
        metavar="PATH",
        help="database file to read instead of the one in the user data directory",
    )
    _configure_resolution(parser)
    parser.set_defaults(_costs_handler=run_costs_run)


def _configure_report(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-g",
        "--group",
        dest="groups",
        action="append",
        metavar="GROUP",
        help=(
            "group the spend by run, day, week, month, provider, model, profile, "
            "format or stage; repeatable or comma separated (default: day)"
        ),
    )
    parser.add_argument(
        "--start",
        "--since",
        "--from",
        dest="start",
        metavar="DATE",
        help="only count spend on or after this ISO date or datetime",
    )
    parser.add_argument(
        "--end",
        "--until",
        "--to",
        dest="end",
        metavar="DATE",
        help="only count spend on or before this ISO date or datetime",
    )
    parser.add_argument(
        "--run",
        "--run-id",
        dest="run_id",
        metavar="RUN_ID",
        help="only count the spend of one run",
    )
    parser.add_argument(
        "--provider",
        metavar="NAME",
        help="only count the spend of one provider",
    )
    parser.add_argument(
        "--model",
        metavar="NAME",
        help="only count the spend of one model",
    )
    parser.add_argument(
        "--stage",
        metavar="NAME",
        help="only count the spend of one stage",
    )
    parser.add_argument(
        "--database",
        "--db",
        dest="database",
        metavar="PATH",
        help="database file to read instead of the one in the user data directory",
    )
    _configure_resolution(parser)


def _configure_export_options(
    parser: argparse.ArgumentParser,
    *,
    default_format: str,
) -> None:
    parser.add_argument(
        "-f",
        "--format",
        dest="export_format",
        choices=("table", "csv", "json"),
        default=None,
        help=f"output format (default: {default_format})",
    )
    parser.add_argument(
        "--csv",
        dest="export_format",
        action="store_const",
        const="csv",
        help="shorthand for --format csv",
    )
    parser.add_argument(
        "--json",
        dest="export_format",
        action="store_const",
        const="json",
        help="shorthand for --format json",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="write the export to PATH instead of standard output",
    )
    parser.set_defaults(_default_export_format=default_format)


def _configure_summary(parser: argparse.ArgumentParser) -> None:
    _configure_report(parser)
    _configure_export_options(parser, default_format="table")
    parser.set_defaults(_costs_handler=run_costs_summary)


def _configure_export(parser: argparse.ArgumentParser) -> None:
    _configure_report(parser)
    _configure_export_options(parser, default_format="csv")
    parser.set_defaults(_costs_handler=run_costs_export)


def _configure_dashboard(parser: argparse.ArgumentParser) -> None:
    _configure_report(parser)
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="write the dashboard to PATH instead of standard output",
    )
    parser.add_argument(
        "--now",
        metavar="DATETIME",
        help=(
            "evaluate budget bars at this ISO date or datetime "
            "instead of the current time"
        ),
    )
    parser.set_defaults(_costs_handler=run_costs_dashboard)


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the costs group and its subcommands."""
    subparsers = parser.add_subparsers(dest="costs_command", metavar="COMMAND")
    run_parser = subparsers.add_parser(
        "run",
        help="Break down the spend of one run by stage, provider, model and asset",
        description=(
            "Break down the spend of one run by stage, provider, model and asset"
        ),
    )
    _configure_run(run_parser)
    summary_parser = subparsers.add_parser(
        "summary",
        help="Summarize spend by a dimension with optional date filters",
        description="Summarize spend by a dimension with optional date filters",
    )
    _configure_summary(summary_parser)
    export_parser = subparsers.add_parser(
        "export",
        help="Export spend as CSV or JSON with the same totals as the summary",
        description="Export spend as CSV or JSON with the same totals as the summary",
    )
    _configure_export(export_parser)
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Write a static HTML dashboard with spend and budget bars",
        description="Write a static HTML dashboard with spend and budget bars",
    )
    _configure_dashboard(dashboard_parser)


def run_costs(args: argparse.Namespace) -> int:
    """Dispatch to the selected costs subcommand."""
    handler = getattr(args, "_costs_handler", None)
    if handler is None:
        print("error: missing costs command", file=sys.stderr)
        print(
            "usage: aivia costs {run,summary,export,dashboard}",
            file=sys.stderr,
        )
        return 2
    return int(handler(args))


def _run_id(args: argparse.Namespace) -> str:
    positional = (args.run_id or "").strip()
    option = (args.run_option or "").strip()
    return positional or option


def _resolve_database(
    args: argparse.Namespace,
) -> tuple[Path, DatabaseConfig | None]:
    explicit = getattr(args, "database", None)
    if explicit:
        return Path(explicit).expanduser(), None

    config = resolve_app_config(
        config_file=args.config,
        environment=args.environment,
        profile=args.profile,
        set_values=args.set_values,
    )
    database = getattr(config, "database", None)
    if not isinstance(database, DatabaseConfig):
        database = None
    filename = database.filename if database is not None else DatabaseConfig().filename
    resolved = paths.resolve_paths(args.config)
    return resolved.data_dir / filename, database


def _money(value: float) -> str:
    return f"${value:.6f}"


def _units(value: float) -> str:
    return f"{value:g}"


def _summary_groups(values: list[str] | None) -> list[CostGroup]:
    groups: list[CostGroup] = []
    for raw in values or []:
        for part in raw.split(","):
            text = part.strip()
            if not text:
                continue
            try:
                group = CostGroup(text.lower())
            except ValueError as exc:
                allowed = ", ".join(item.value for item in CostGroup)
                raise ValueError(
                    f"unknown group {text!r}; expected one of {allowed}"
                ) from exc
            if group not in groups:
                groups.append(group)
    return groups or [CostGroup.DAY]


def _parse_moment(value: str) -> date | datetime:
    text = value.strip()
    if not text:
        raise ValueError("a date filter must not be empty")
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _parse_moment_datetime(value: str) -> datetime:
    parsed = _parse_moment(value)
    if isinstance(parsed, datetime):
        return parsed
    return datetime.combine(parsed, time.min, tzinfo=UTC)


def _resolve_budget(args: argparse.Namespace) -> BudgetConfig:
    config = resolve_app_config(
        config_file=args.config,
        environment=args.environment,
        profile=args.profile,
        set_values=args.set_values,
    )
    budget = getattr(config, "budget", None)
    return budget if isinstance(budget, BudgetConfig) else BudgetConfig()


def _render_aggregates(
    title: str,
    key_header: str,
    buckets: list[CostAggregate],
) -> list[str]:
    lines = [title, f"{key_header}\tTOTAL\tESTIMATED\tACTUAL\tUNITS\tCOUNT"]
    for bucket in buckets:
        lines.append(
            "\t".join(
                (
                    bucket.key or "(none)",
                    _money(bucket.total_usd),
                    _money(bucket.estimated_usd),
                    _money(bucket.actual_usd),
                    _units(bucket.units),
                    str(bucket.count),
                )
            )
        )
    return lines


@dataclass(slots=True)
class _AssetSpend:
    """The spend linked to one asset, or to no asset at all."""

    asset_id: str
    label: str
    kind: str
    total_usd: float = 0.0
    estimated_usd: float = 0.0
    actual_usd: float = 0.0
    units: float = 0.0
    count: int = 0

    def add(self, event: CostEvent) -> None:
        self.total_usd += event.amount_usd
        if event.estimated:
            self.estimated_usd += event.amount_usd
        else:
            self.actual_usd += event.amount_usd
        self.units += event.units
        self.count += 1


def _asset_buckets(assets: list[Asset], events: list[CostEvent]) -> list[_AssetSpend]:
    by_id = {asset.id: asset for asset in assets}
    asset_for_event: dict[str, str] = {}
    for asset in assets:
        for link in asset.cost_links:
            asset_for_event.setdefault(link.cost_event_id, asset.id)

    buckets: dict[str, _AssetSpend] = {}

    def bucket(asset_id: str) -> _AssetSpend:
        found = buckets.get(asset_id)
        if found is not None:
            return found
        asset = by_id.get(asset_id) if asset_id else None
        created = _AssetSpend(
            asset_id=asset_id,
            label=(
                asset.label
                if asset is not None
                else ("(unassigned)" if not asset_id else "")
            ),
            kind=str(asset.kind) if asset is not None else "",
        )
        buckets[asset_id] = created
        return created

    for event in events:
        asset_id = asset_for_event.get(event.id)
        if asset_id is None:
            metadata_id = event.metadata.get("asset_id")
            asset_id = str(metadata_id) if metadata_id else ""
        bucket(asset_id).add(event)

    return sorted(buckets.values(), key=lambda item: (not item.asset_id, item.asset_id))


def _render_assets(buckets: list[_AssetSpend]) -> list[str]:
    lines = [
        "BY ASSET",
        "ASSET\tLABEL\tKIND\tTOTAL\tESTIMATED\tACTUAL\tUNITS\tCOUNT",
    ]
    for bucket in buckets:
        lines.append(
            "\t".join(
                (
                    bucket.asset_id or "-",
                    bucket.label or "-",
                    bucket.kind or "-",
                    _money(bucket.total_usd),
                    _money(bucket.estimated_usd),
                    _money(bucket.actual_usd),
                    _units(bucket.units),
                    str(bucket.count),
                )
            )
        )
    return lines


def run_costs_run(args: argparse.Namespace) -> int:
    """Print one run's spend broken down by stage, provider, model and asset."""
    run_id = _run_id(args)
    if not run_id:
        print("error: a run id is required", file=sys.stderr)
        return 2

    try:
        database_path, database_config = _resolve_database(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not database_path.exists():
        print(f"error: database not found: {database_path}", file=sys.stderr)
        return 1

    engine = create_database_engine(path=database_path, config=database_config)
    try:
        run = RunRepository(engine).get(run_id)
        if run is None:
            print(f"error: run not found: {run_id}", file=sys.stderr)
            return 1

        events = CostTracker(engine).for_run(run_id)
        aggregator = CostAggregator(engine)
        assets = AssetRepository(engine).list(run_id=run_id)
        total = math.fsum(event.amount_usd for event in events)

        lines = [f"Run: {run.id}", f"Total: {_money(total)}", ""]
        lines.extend(
            _render_aggregates("BY STAGE", "STAGE", aggregator.by_stage(run_id=run_id))
        )
        lines.append("")
        lines.extend(
            _render_aggregates(
                "BY PROVIDER", "PROVIDER", aggregator.by_provider(run_id=run_id)
            )
        )
        lines.append("")
        lines.extend(
            _render_aggregates("BY MODEL", "MODEL", aggregator.by_model(run_id=run_id))
        )
        lines.append("")
        lines.extend(_render_assets(_asset_buckets(assets, events)))
        print("\n".join(lines))
    except SQLAlchemyError as exc:
        print(f"error: cannot read the cost ledger: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()
    return 0


def _collect_report(
    engine: Engine,
    groups: list[CostGroup],
    *,
    start: date | datetime | None,
    end: date | datetime | None,
    run_id: str | None,
    provider: str | None,
    model: str | None,
    stage: str | None,
) -> tuple[list[tuple[CostGroup, list[CostAggregate]]], float]:
    aggregator = CostAggregator(engine)
    sections: list[tuple[CostGroup, list[CostAggregate]]] = []
    total = 0.0
    for index, group in enumerate(groups):
        buckets = aggregator.aggregate(
            group,
            start=start,
            end=end,
            run_id=run_id,
            provider=provider,
            model=model,
            stage=stage,
        )
        if index == 0:
            total = math.fsum(bucket.total_usd for bucket in buckets)
        sections.append((group, buckets))
    return sections, total


def _export_format(args: argparse.Namespace) -> str:
    chosen = getattr(args, "export_format", None)
    output = getattr(args, "output", None)
    if chosen is None and output:
        suffix = Path(output).suffix.lower()
        if suffix == ".json":
            chosen = "json"
        elif suffix == ".csv":
            chosen = "csv"
    if chosen is None:
        chosen = getattr(args, "_default_export_format", "csv")
    return str(chosen)


def _export_json(
    sections: list[tuple[CostGroup, list[CostAggregate]]],
    total: float,
) -> str:
    group_payloads: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for group, buckets in sections:
        bucket_rows = [bucket.to_dict() for bucket in buckets]
        rows.extend(bucket_rows)
        group_payloads.append(
            {
                "group": group.value,
                "total_usd": math.fsum(bucket.total_usd for bucket in buckets),
                "estimated_usd": math.fsum(bucket.estimated_usd for bucket in buckets),
                "actual_usd": math.fsum(bucket.actual_usd for bucket in buckets),
                "rows": bucket_rows,
            }
        )

    first = sections[0][1] if sections else []
    estimated = math.fsum(bucket.estimated_usd for bucket in first)
    actual = math.fsum(bucket.actual_usd for bucket in first)
    payload: dict[str, object] = {
        "group": sections[0][0].value if sections else None,
        "total_usd": total,
        "total": total,
        "estimated_usd": estimated,
        "actual_usd": actual,
        "totals": {
            "total_usd": total,
            "estimated_usd": estimated,
            "actual_usd": actual,
        },
        "groups": group_payloads,
        "rows": rows,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _export_csv(
    sections: list[tuple[CostGroup, list[CostAggregate]]],
) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "group",
            "key",
            "total_usd",
            "estimated_usd",
            "actual_usd",
            "units",
            "count",
        ]
    )
    for group, buckets in sections:
        for bucket in buckets:
            writer.writerow(
                [
                    group.value,
                    bucket.key,
                    f"{bucket.total_usd:.6f}",
                    f"{bucket.estimated_usd:.6f}",
                    f"{bucket.actual_usd:.6f}",
                    _units(bucket.units),
                    bucket.count,
                ]
            )
    return buffer.getvalue()


def _emit_export(text: str, output: str | None) -> int:
    if not output:
        sys.stdout.write(text)
        return 0
    target = Path(output).expanduser()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    except OSError as exc:
        print(f"error: cannot write {target}: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {target}")
    return 0


def run_costs_summary(args: argparse.Namespace) -> int:
    """Print spend grouped by one or more dimensions, optionally date-filtered."""
    try:
        groups = _summary_groups(args.groups)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        start = _parse_moment(args.start) if args.start else None
        end = _parse_moment(args.end) if args.end else None
    except ValueError as exc:
        print(f"error: invalid date: {exc}", file=sys.stderr)
        return 2

    try:
        database_path, database_config = _resolve_database(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not database_path.exists():
        print(f"error: database not found: {database_path}", file=sys.stderr)
        return 1

    run_id = getattr(args, "run_id", None) or None
    provider = getattr(args, "provider", None) or None
    model = getattr(args, "model", None) or None
    stage = getattr(args, "stage", None) or None
    output = getattr(args, "output", None)

    engine = create_database_engine(path=database_path, config=database_config)
    try:
        sections, total = _collect_report(
            engine,
            groups,
            start=start,
            end=end,
            run_id=run_id,
            provider=provider,
            model=model,
            stage=stage,
        )
        export_format = _export_format(args)
        if export_format == "json":
            return _emit_export(_export_json(sections, total), output)
        if export_format == "csv":
            return _emit_export(_export_csv(sections), output)

        lines = [f"Total: {_money(total)}"]
        for group, buckets in sections:
            lines.append("")
            lines.extend(
                _render_aggregates(
                    f"BY {group.value.upper()}", group.value.upper(), buckets
                )
            )
        if output:
            return _emit_export("\n".join(lines) + "\n", output)
        print("\n".join(lines))
    except SQLAlchemyError as exc:
        print(f"error: cannot read the cost ledger: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()
    return 0


def run_costs_export(args: argparse.Namespace) -> int:
    """Export grouped spend as CSV or JSON, matching the summary totals."""
    try:
        groups = _summary_groups(args.groups)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        start = _parse_moment(args.start) if args.start else None
        end = _parse_moment(args.end) if args.end else None
    except ValueError as exc:
        print(f"error: invalid date: {exc}", file=sys.stderr)
        return 2

    try:
        database_path, database_config = _resolve_database(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not database_path.exists():
        print(f"error: database not found: {database_path}", file=sys.stderr)
        return 1

    run_id = getattr(args, "run_id", None) or None
    provider = getattr(args, "provider", None) or None
    model = getattr(args, "model", None) or None
    stage = getattr(args, "stage", None) or None
    output = getattr(args, "output", None)
    export_format = _export_format(args)

    engine = create_database_engine(path=database_path, config=database_config)
    try:
        sections, total = _collect_report(
            engine,
            groups,
            start=start,
            end=end,
            run_id=run_id,
            provider=provider,
            model=model,
            stage=stage,
        )
        if export_format == "json":
            text = _export_json(sections, total)
        elif export_format == "csv":
            text = _export_csv(sections)
        else:
            lines = [f"Total: {_money(total)}"]
            for group, buckets in sections:
                lines.append("")
                lines.extend(
                    _render_aggregates(
                        f"BY {group.value.upper()}", group.value.upper(), buckets
                    )
                )
            text = "\n".join(lines) + "\n"
        return _emit_export(text, output)
    except SQLAlchemyError as exc:
        print(f"error: cannot read the cost ledger: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()


def run_costs_dashboard(args: argparse.Namespace) -> int:
    """Write a static HTML dashboard of spend and configured budget caps."""
    try:
        start = _parse_moment(args.start) if args.start else None
        end = _parse_moment(args.end) if args.end else None
        now = _parse_moment_datetime(args.now) if args.now else None
    except ValueError as exc:
        print(f"error: invalid date: {exc}", file=sys.stderr)
        return 2

    try:
        database_path, database_config = _resolve_database(args)
        budget = _resolve_budget(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not database_path.exists():
        print(f"error: database not found: {database_path}", file=sys.stderr)
        return 1

    engine = create_database_engine(path=database_path, config=database_config)
    try:
        data = build_dashboard(
            engine,
            budget,
            now=now,
            start=start,
            end=end,
            run_id=getattr(args, "run_id", None) or None,
            provider=getattr(args, "provider", None) or None,
            model=getattr(args, "model", None) or None,
            stage=getattr(args, "stage", None) or None,
        )
    except SQLAlchemyError as exc:
        print(f"error: cannot read the cost ledger: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()

    return _emit_export(render_dashboard(data), getattr(args, "output", None))


register_command(
    "costs",
    run_costs,
    help="Report spend from the local cost ledger",
    configure=configure,
)

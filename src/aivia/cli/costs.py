"""``aivia costs``: report spend recorded in the cost ledger."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

from aivia.cli.registry import register_command
from aivia.core import paths
from aivia.core.config.database import DatabaseConfig
from aivia.core.config.resolve import resolve_app_config
from aivia.core.cost import CostAggregator, CostTracker
from aivia.core.cost_agg import CostAggregate
from aivia.core.db import AssetRepository, RunRepository, create_database_engine
from aivia.core.errors import ConfigError
from aivia.core.models.asset import Asset
from aivia.core.models.cost import CostEvent

__all__ = ["configure", "run_costs", "run_costs_run"]


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


def run_costs(args: argparse.Namespace) -> int:
    """Dispatch to the selected costs subcommand."""
    handler = getattr(args, "_costs_handler", None)
    if handler is None:
        print("error: missing costs command", file=sys.stderr)
        print("usage: aivia costs {run}", file=sys.stderr)
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
            _render_aggregates(
                "BY STAGE", "STAGE", aggregator.by_stage(run_id=run_id)
            )
        )
        lines.append("")
        lines.extend(
            _render_aggregates(
                "BY PROVIDER", "PROVIDER", aggregator.by_provider(run_id=run_id)
            )
        )
        lines.append("")
        lines.extend(
            _render_aggregates(
                "BY MODEL", "MODEL", aggregator.by_model(run_id=run_id)
            )
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


register_command(
    "costs",
    run_costs,
    help="Report spend from the local cost ledger",
    configure=configure,
)

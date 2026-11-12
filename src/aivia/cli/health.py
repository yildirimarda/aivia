"""``aivia health``: report queue depth, worker liveness, disk space and providers."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy import Engine

from aivia.cli.registry import register_command
from aivia.core import paths
from aivia.core.config.database import DatabaseConfig
from aivia.core.config.providers import ProvidersConfig
from aivia.core.config.resolve import resolve_app_config
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.errors import ConfigError
from aivia.core.health import (
    DEFAULT_MIN_FREE_BYTES,
    DiskFullError,
    HealthReport,
    check_health,
)
from aivia.providers.provider_factory import build_providers

__all__ = ["configure", "run_health"]

DEFAULT_STALE_AFTER_SECONDS: float = 120.0


def _configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--database",
        "--db",
        dest="database",
        metavar="PATH",
        help="database file to use instead of the one in the user data directory",
    )
    parser.add_argument(
        "--min-free-bytes",
        dest="min_free_bytes",
        type=int,
        metavar="BYTES",
        default=None,
        help=(
            "minimum free bytes required for a new run "
            f"(default: {DEFAULT_MIN_FREE_BYTES})"
        ),
    )
    parser.set_defaults(_health_handler=run_health)


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the health command's options."""
    _configure(parser)


def _database_path(args: argparse.Namespace, config: object) -> Path:
    explicit = getattr(args, "database", None)
    if explicit:
        return Path(explicit).expanduser()
    database = getattr(config, "database", None)
    if not isinstance(database, DatabaseConfig):
        database = None
    filename = database.filename if database is not None else DatabaseConfig().filename
    return paths.resolve_paths(args.config).data_dir / filename


def _open_database(path: Path) -> Engine:
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_database_engine(path=path)
    try:
        upgrade_database(engine=engine)
    except BaseException:
        engine.dispose()
        raise
    return engine


def _providers_config(config: object) -> ProvidersConfig:
    section = getattr(config, "providers", None)
    return section if isinstance(section, ProvidersConfig) else ProvidersConfig()


def _min_free_bytes(args: argparse.Namespace) -> int:
    value = getattr(args, "min_free_bytes", None)
    if value is None:
        return DEFAULT_MIN_FREE_BYTES
    return int(value)


def _print_report(report: HealthReport) -> None:
    lines = [
        "QUEUE",
        f"  depth: {report.queue_depth}",
        "WORKERS",
        f"  running: {report.worker_liveness.running_jobs}",
        f"  fresh heartbeats: {report.worker_liveness.fresh_count}",
        f"  stale heartbeats: {report.worker_liveness.stale_count}",
        f"  live: {report.worker_liveness.live}",
        "DISK",
        f"  path: {report.disk_space.path}",
        f"  total: {report.disk_space.total_bytes} bytes",
        f"  free: {report.disk_space.free_bytes} bytes",
        f"  used: {report.disk_space.used_bytes} bytes",
        f"  min required: {report.disk_space.min_free_bytes} bytes",
        f"  ok: {report.disk_space.ok}",
        "PROVIDERS",
    ]
    for status in report.provider_status:
        line = f"  {status.name}: {status.status}"
        if status.latency_ms is not None:
            line += f" ({status.latency_ms:.1f} ms)"
        if status.message:
            line += f" - {status.message}"
        lines.append(line)
    if not report.provider_status:
        lines.append("  (none configured)")
    lines.append("")
    lines.append(f"overall: {'healthy' if report.healthy else 'unhealthy'}")
    print("\n".join(lines))


def run_health(args: argparse.Namespace) -> int:
    """Print a health snapshot and exit nonzero when unhealthy."""
    try:
        config = resolve_app_config(
            config_file=getattr(args, "config", None),
            environment=getattr(args, "environment", None),
            profile=getattr(args, "profile", None),
            preset=getattr(args, "preset", None),
            set_values=getattr(args, "set_values", None),
        )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        database_path = _database_path(args, config)
        engine = _open_database(database_path)
    except (OSError, ConfigError) as exc:
        print(f"error: cannot open the database: {exc}", file=sys.stderr)
        return 1

    try:
        providers = build_providers(_providers_config(config))
        report = asyncio.run(
            check_health(
                engine,
                paths.resolve_paths(args.config).data_dir,
                providers.values(),
                stale_after_seconds=DEFAULT_STALE_AFTER_SECONDS,
                min_free_bytes=_min_free_bytes(args),
            )
        )
    except DiskFullError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: cannot build health report: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()

    _print_report(report)
    return 0 if report.healthy else 1


register_command(
    "health",
    run_health,
    help="Report queue depth, worker liveness, disk space and provider status",
    configure=configure,
)

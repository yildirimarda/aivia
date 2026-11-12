"""``aivia retention``: remove stale intermediates, cache, orders, deliveries,
logs and hits.

Final outputs (pinned assets and compositions) are never removed unless
``retention.delete_final_outputs`` is enabled. With ``--dry-run`` the command
lists what would be deleted without removing anything.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import overload

from sqlalchemy import Engine

from aivia.cli.registry import register_command
from aivia.core import paths
from aivia.core.config.database import DatabaseConfig
from aivia.core.config.ops import RetentionConfig
from aivia.core.config.resolve import resolve_config
from aivia.core.db import (
    AssetCacheRepository,
    AssetRepository,
    CompositionRepository,
    DeliveryRepository,
    OrderRepository,
    RunRepository,
    WebCardHitRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.db.order_state import OrderStateMachine
from aivia.core.errors import ConfigError
from aivia.core.models import utcnow

__all__ = ["configure", "plan_retention", "retention"]

_LOGS_DIR_NAME = "logs"

_CATEGORY_INTERMEDIATES = "intermediates"
_CATEGORY_CACHE = "cache"
_CATEGORY_ORDERS = "expired orders"
_CATEGORY_DELIVERIES = "deliveries"
_CATEGORY_LOGS = "logs"
_CATEGORY_HITS = "hits"


def _load_config(args: argparse.Namespace) -> object:
    return resolve_config(
        config_file=args.config,
        environment=getattr(args, "environment", None),
        profile=getattr(args, "profile", None),
        preset=getattr(args, "preset", None),
        set_values=getattr(args, "set_values", None) or (),
    ).to_app_config()


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


@overload
def _as_utc(value: datetime) -> datetime: ...


@overload
def _as_utc(value: None) -> None: ...


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _remove_path(path: str | Path | None) -> None:
    """Remove one file or directory, ignoring missing paths and errors."""
    text = str(path).strip() if path is not None else ""
    if not text:
        return
    target = Path(text)
    try:
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        else:
            target.unlink(missing_ok=True)
    except OSError:
        pass


def _retention_config(config: object) -> RetentionConfig:
    retention = getattr(config, "retention", None)
    if isinstance(retention, RetentionConfig):
        return retention
    return RetentionConfig.from_mapping({})


def plan_retention(
    engine: Engine,
    retention_config: RetentionConfig,
    *,
    now: datetime | None = None,
    logs_dir: Path | None = None,
) -> list[dict[str, object]]:
    """Return the deletion plan without removing anything.

    Each entry is a mapping with ``category``, ``target`` and an optional
    ``path`` describing what would be removed.
    """
    current = _as_utc(now) if now is not None else utcnow()
    assert current is not None
    planned: list[dict[str, object]] = []

    def add(category: str, target: str, path: Path | None = None) -> None:
        planned.append({"category": category, "target": target, "path": path})

    orders_days = int(retention_config.orders_days)
    deliveries_days = int(retention_config.deliveries_days)
    hits_days = int(retention_config.web_card_hits_days)
    cache_days = int(retention_config.cache_days)
    intermediates_days = int(retention_config.intermediates_days)
    logs_days = int(retention_config.logs_days)

    # Hits: anonymous web card views older than the window.
    if hits_days >= 0:
        hit_cutoff = current - timedelta(days=hits_days)
        for hit in WebCardHitRepository(engine).list():
            hit_created = _as_utc(hit.get("created_at"))
            if hit_created is not None and hit_created < hit_cutoff:
                add(_CATEGORY_HITS, str(hit["id"]))

    # Deliveries: delivery rows and their files older than the window.
    if deliveries_days >= 0:
        delivery_cutoff = current - timedelta(days=deliveries_days)
        for delivery in DeliveryRepository(engine).list():
            if _as_utc(delivery.created_at) < delivery_cutoff:
                path = Path(delivery.path) if delivery.path else None
                add(_CATEGORY_DELIVERIES, delivery.id, path)

    # Expired orders: whole orders older than the window. Order deletion
    # cascades to deliveries, posts, runs and hits, and honours pinned finals.
    if orders_days >= 0:
        order_cutoff = current - timedelta(days=orders_days)
        for order in OrderRepository(engine).list():
            order_created = _as_utc(order.get("created_at"))
            if order_created is not None and order_created < order_cutoff:
                add(_CATEGORY_ORDERS, str(order["id"]))

    # Cache: cache entries older than the window; the referenced asset file is
    # only removed when it is not pinned (or final output deletion is enabled).
    if cache_days >= 0:
        cache_cutoff = current - timedelta(days=cache_days)
        for entry in AssetCacheRepository(engine).list():
            if _as_utc(entry.created_at) < cache_cutoff:
                add(_CATEGORY_CACHE, entry.cache_key)

    # Intermediates: run directories older than the window.
    if intermediates_days >= 0:
        intermediate_cutoff = current - timedelta(days=intermediates_days)
        for run in RunRepository(engine).list():
            if _as_utc(run.created_at) < intermediate_cutoff:
                path = Path(run.run_dir) if run.run_dir else None
                add(_CATEGORY_INTERMEDIATES, run.id, path)

    # Logs: log files older than the window.
    if logs_dir is not None and logs_days >= 0 and logs_dir.is_dir():
        log_cutoff = current - timedelta(days=logs_days)
        for log_file in sorted(logs_dir.iterdir()):
            if not log_file.is_file():
                continue
            try:
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime, tz=UTC)
            except OSError:
                continue
            if mtime < log_cutoff:
                add(_CATEGORY_LOGS, log_file.name, log_file)

    return planned


def _perform(
    engine: Engine,
    planned: list[dict[str, object]],
    retention_config: RetentionConfig,
) -> int:
    delete_final_outputs = bool(retention_config.delete_final_outputs)
    machine = OrderStateMachine(engine)
    deleted_order_ids: set[str] = set()

    for item in planned:
        category = str(item["category"])
        target = str(item["target"])
        raw_path = item.get("path")
        path: Path | None = (
            Path(raw_path) if isinstance(raw_path, (str, Path)) else None
        )

        if category == _CATEGORY_HITS:
            WebCardHitRepository(engine).delete(target)
        elif category == _CATEGORY_DELIVERIES:
            delivery = DeliveryRepository(engine).get(target)
            if delivery is not None:
                for file_path in delivery.files:
                    _remove_path(file_path)
                _remove_path(delivery.path)
            DeliveryRepository(engine).delete(target)
        elif category == _CATEGORY_ORDERS:
            machine.delete(target, delete_final_outputs=delete_final_outputs)
            deleted_order_ids.add(target)
        elif category == _CATEGORY_CACHE:
            entry = AssetCacheRepository(engine).get(target)
            if entry is not None:
                asset = AssetRepository(engine).get(entry.asset_id)
                if asset is not None and (not asset.pinned or delete_final_outputs):
                    _remove_path(asset.path)
                AssetCacheRepository(engine).delete(target)
        elif category == _CATEGORY_INTERMEDIATES:
            run = RunRepository(engine).get(target)
            if run is not None:
                _remove_path(run.run_dir)
        elif category == _CATEGORY_LOGS:
            _remove_path(path)

    # Order deletion cascades its runs; remove any remaining run directories
    # whose order was deleted so no intermediate files linger.
    if deleted_order_ids:
        for run in RunRepository(engine).list():
            if run.order_id in deleted_order_ids:
                _remove_path(run.run_dir)

    # CompositionRepository is retained for symmetry with order deletion; pinned
    # compositions are honoured by OrderStateMachine.delete already.
    _ = CompositionRepository
    return len(planned)


def retention(args: argparse.Namespace) -> int:
    """Run the retention job, previewing deletions when ``--dry-run`` is set."""
    try:
        config = _load_config(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    retention_config = _retention_config(config)

    database_path = _database_path(args, config)
    if not database_path.exists():
        print(f"error: database not found: {database_path}", file=sys.stderr)
        return 1

    logs_dir = database_path.parent / _LOGS_DIR_NAME

    engine = _open_database(database_path)
    try:
        planned = plan_retention(engine, retention_config, logs_dir=logs_dir)
    finally:
        engine.dispose()

    dry_run = bool(getattr(args, "dry_run", False))
    if dry_run:
        if not planned:
            print("retention: dry run, nothing to delete")
            return 0
        print(f"retention: dry run, would delete {len(planned)} item(s)")
        for item in planned:
            target = str(item["target"])
            path = item.get("path")
            detail = f" ({path})" if path is not None else ""
            print(f"  {item['category']}: {target}{detail}")
        return 0

    engine = _open_database(database_path)
    try:
        count = _perform(engine, planned, retention_config)
    finally:
        engine.dispose()
    print(f"retention: deleted {count} item(s)")
    return 0


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the retention command and its arguments."""
    parser.add_argument(
        "--database",
        "--db",
        dest="database",
        metavar="PATH",
        help="database file to use instead of the one in the user data directory",
    )
    parser.set_defaults(_retention_handler=retention)


def _dispatch(args: argparse.Namespace) -> int:
    handler = getattr(args, "_retention_handler", None)
    if handler is None:
        print("error: a retention command is required", file=sys.stderr)
        return 2
    return int(handler(args))


register_command(
    "retention",
    _dispatch,
    help="Remove stale intermediates, cache, orders, deliveries, logs and hits",
    configure=configure,
)

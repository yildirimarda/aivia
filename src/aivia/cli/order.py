"""``aivia order``: create, show and delete personalized orders.

An order may carry an idempotency key. The first create request with a key
stores the order and returns it. A later create request with the same key
returns that existing order instead of inserting a duplicate, so repeating a
request never creates a second order.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError

from aivia.cli.registry import register_command
from aivia.core import paths
from aivia.core.config.database import DatabaseConfig
from aivia.core.config.resolve import resolve_config
from aivia.core.db import (
    OrderRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.errors import ConfigError

__all__ = ["configure", "order_create", "order_delete", "order_import", "order_show"]

KNOWN_ORDER_FIELDS = frozenset(
    {
        "occasion",
        "name",
        "message",
        "language",
        "profile",
        "level",
        "timezone",
        "currency",
        "amount_usd",
        "budget_usd",
        "idempotency_key",
    }
)
"""Fields a batch row may carry; anything else is reported as a bad row."""


def _parse_batch_file(path: Path, fmt: str | None) -> list[dict[str, Any]]:
    """Read *path* as CSV or JSON and return one mapping per row."""
    if fmt is None:
        suffix = path.suffix.lower()
        if suffix == ".json":
            fmt = "json"
        elif suffix == ".csv":
            fmt = "csv"
        else:
            raise ValueError(f"unsupported batch file {path!r}: expected .csv or .json")
    text = path.read_text(encoding="utf-8")
    if fmt == "csv":
        reader = csv.DictReader(io.StringIO(text))
        return [
            {key: value for key, value in row.items() if key is not None}
            for row in reader
        ]
    data = json.loads(text)
    if isinstance(data, list):
        return [dict(row) for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        items = data.get("items")
        if isinstance(items, list):
            return [dict(row) for row in items if isinstance(row, dict)]
        return [data]
    raise ValueError("JSON batch must be a list of rows or an object")


def _validate_row(
    row: dict[str, Any], number: int
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate one batch row, returning its order values or an error message."""
    values: dict[str, Any] = {}
    problems: list[str] = []
    for key, value in row.items():
        if key not in KNOWN_ORDER_FIELDS:
            problems.append(f"unknown field: {key}")
            continue
        values[key] = value
    for money_field in ("amount_usd", "budget_usd"):
        raw = values.get(money_field)
        if raw is None or raw == "":
            continue
        try:
            values[money_field] = float(raw)
        except (TypeError, ValueError):
            problems.append(f"bad {money_field}: {raw!r}")
    if problems:
        return None, "; ".join(problems)
    return values, None


def _configure_database(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--database",
        "--db",
        dest="database",
        metavar="PATH",
        help="database file to use instead of the one in the user data directory",
    )


def _configure_create(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--occasion",
        default="birthday",
        metavar="NAME",
        help="occasion the order is for (default: birthday)",
    )
    parser.add_argument(
        "--name",
        default="",
        metavar="NAME",
        help="recipient name",
    )
    parser.add_argument(
        "--message",
        default="",
        metavar="TEXT",
        help="personal message",
    )
    parser.add_argument(
        "--language",
        default="",
        metavar="CODE",
        help="language code",
    )
    parser.add_argument(
        "--profile",
        default="",
        metavar="NAME",
        help="profile the order belongs to",
    )
    parser.add_argument(
        "--level",
        default="",
        metavar="NAME",
        help="personalization level",
    )
    parser.add_argument(
        "--timezone",
        default="UTC",
        metavar="ZONE",
        help="recipient timezone (default: UTC)",
    )
    parser.add_argument(
        "--amount",
        type=float,
        default=0.0,
        metavar="USD",
        help="line-item amount before tax (default: 0)",
    )
    parser.add_argument(
        "--currency",
        default="USD",
        metavar="CODE",
        help="currency code (default: USD)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        metavar="USD",
        help="optional spend ceiling for the order",
    )
    parser.add_argument(
        "--idempotency-key",
        "--idempotency",
        dest="idempotency_key",
        metavar="KEY",
        help="return the existing order when KEY was already used for this request",
    )
    parser.set_defaults(_order_handler=order_create)


def _configure_show_delete(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "order_id",
        nargs="?",
        default=None,
        metavar="ORDER_ID",
        help="order to show or delete",
    )
    parser.add_argument(
        "--order",
        "--order-id",
        dest="order_option",
        default=None,
        metavar="ORDER_ID",
        help="order to show or delete",
    )


def _configure_show(parser: argparse.ArgumentParser) -> None:
    _configure_show_delete(parser)
    parser.set_defaults(_order_handler=order_show)


def _configure_delete(parser: argparse.ArgumentParser) -> None:
    _configure_show_delete(parser)
    parser.set_defaults(_order_handler=order_delete)


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the order command and its create, show and delete subcommands."""
    subparsers = parser.add_subparsers(
        dest="order_command", required=True, metavar="COMMAND"
    )
    create = subparsers.add_parser(
        "create",
        help="Create a personalized order",
        description="Create a personalized order",
    )
    _configure_create(create)
    _configure_database(create)
    import_ = subparsers.add_parser(
        "import",
        help="Import orders from a CSV or JSON batch file",
        description="Import orders from a CSV or JSON batch file",
    )
    _configure_import(import_)
    show = subparsers.add_parser(
        "show",
        help="Show a personalized order",
        description="Show a personalized order",
    )
    _configure_show(show)
    _configure_database(show)
    delete = subparsers.add_parser(
        "delete",
        help="Delete a personalized order",
        description="Delete a personalized order",
    )
    _configure_delete(delete)
    _configure_database(delete)


def _order_id(args: argparse.Namespace) -> str:
    positional = (getattr(args, "order_id", None) or "").strip()
    option = (getattr(args, "order_option", None) or "").strip()
    return positional or option


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


def _money(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-"
    return f"${float(value):.6f}"


def _print_order(order: dict[str, Any]) -> None:
    lines = [
        f"Order: {order['id']}",
        f"State: {order['state']}",
        f"Occasion: {order['occasion']}",
        f"Name: {order['name']}",
        f"Message: {order['message']}",
        f"Language: {order['language']}",
        f"Profile: {order['profile']}",
        f"Level: {order['level']}",
        f"Timezone: {order['timezone']}",
        f"Currency: {order['currency']}",
        f"Amount: {_money(order['amount_usd'])}",
        f"Tax: {_money(order['tax_usd'])}",
        f"Cost: {_money(order['cost_usd'])}",
        f"Budget: {_money(order['budget_usd'])}",
        f"Run: {order['run_id'] or '-'}",
        f"Idempotency key: {order['idempotency_key'] or '-'}",
    ]
    print("\n".join(lines))


def order_create(args: argparse.Namespace) -> int:
    """Create an order, returning the existing one for a repeated idempotency key."""
    try:
        config = _load_config(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    idempotency_key = (getattr(args, "idempotency_key", None) or "").strip()
    values: dict[str, Any] = {
        "occasion": (getattr(args, "occasion", None) or "").strip(),
        "name": (getattr(args, "name", None) or "").strip(),
        "message": (getattr(args, "message", None) or "").strip(),
        "language": (getattr(args, "language", None) or "").strip(),
        "profile": (getattr(args, "profile", None) or "").strip(),
        "level": (getattr(args, "level", None) or "").strip(),
        "timezone": (getattr(args, "timezone", None) or "UTC").strip(),
        "currency": (getattr(args, "currency", None) or "USD").strip(),
        "amount_usd": float(getattr(args, "amount", 0.0) or 0.0),
        "budget_usd": getattr(args, "budget", None),
        "idempotency_key": idempotency_key,
    }

    database_path = _database_path(args, config)
    try:
        engine = _open_database(database_path)
    except (OSError, SQLAlchemyError) as exc:
        print(f"error: cannot open the database: {exc}", file=sys.stderr)
        return 1

    try:
        repository = OrderRepository(engine)
        existing = None
        if idempotency_key:
            existing = repository.find_by_idempotency_key(idempotency_key)
        if existing is not None:
            _print_order(existing)
            return 0
        stored = repository.create(**values)
        _print_order(stored)
        return 0
    except (OSError, SQLAlchemyError) as exc:
        print(f"error: cannot create the order: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()


def _configure_import(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--file",
        "--batch-file",
        dest="batch_file",
        metavar="PATH",
        required=True,
        help="CSV or JSON file with one order per row",
    )
    parser.add_argument(
        "--format",
        dest="format",
        choices=("csv", "json"),
        default=None,
        metavar="FORMAT",
        help=(
            "treat the batch file as csv or json instead of detecting from "
            "its extension"
        ),
    )
    _configure_database(parser)
    parser.set_defaults(_order_handler=order_import)


def order_import(args: argparse.Namespace) -> int:
    """Import orders from a CSV or JSON file, one row per order.

    Valid rows are inserted into the database. Invalid rows are isolated:
    they are reported one per row with the reason they failed and nothing is
    stored for them. The exit code is non zero when any row fails.
    """
    try:
        config = _load_config(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    batch_file = getattr(args, "batch_file", None)
    if not batch_file:
        print("error: a batch file is required", file=sys.stderr)
        return 2
    path = Path(batch_file).expanduser()
    if not path.is_file():
        print(f"error: batch file not found: {path}", file=sys.stderr)
        return 1

    try:
        rows = _parse_batch_file(path, getattr(args, "format", None))
    except (OSError, ValueError) as exc:
        print(f"error: cannot read the batch file: {exc}", file=sys.stderr)
        return 1

    database_path = _database_path(args, config)
    try:
        engine = _open_database(database_path)
    except (OSError, SQLAlchemyError) as exc:
        print(f"error: cannot open the database: {exc}", file=sys.stderr)
        return 1

    succeeded = 0
    failed = 0
    reports: list[str] = []
    try:
        repository = OrderRepository(engine)
        for number, row in enumerate(rows, start=1):
            values, error = _validate_row(row, number)
            if error is not None:
                failed += 1
                reports.append(f"Row {number}: {error}")
                continue
            assert values is not None
            try:
                repository.create(**values)
            except (OSError, SQLAlchemyError) as exc:
                failed += 1
                reports.append(f"Row {number}: cannot create the order: {exc}")
                continue
            succeeded += 1
    finally:
        engine.dispose()

    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")
    for report in reports:
        print(report, file=sys.stderr)
    return 1 if failed else 0


def order_show(args: argparse.Namespace) -> int:
    """Print an order's fields."""
    order_id = _order_id(args)
    if not order_id:
        print("error: an order id is required", file=sys.stderr)
        return 2

    try:
        config = _load_config(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    database_path = _database_path(args, config)
    if not database_path.exists():
        print(f"error: database not found: {database_path}", file=sys.stderr)
        return 1

    engine = create_database_engine(path=database_path)
    try:
        order = OrderRepository(engine).get(order_id)
        if order is None:
            print(f"error: order not found: {order_id}", file=sys.stderr)
            return 1
        _print_order(order)
        return 0
    except SQLAlchemyError as exc:
        print(f"error: cannot read the order: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()


def order_delete(args: argparse.Namespace) -> int:
    """Delete an order and return it."""
    order_id = _order_id(args)
    if not order_id:
        print("error: an order id is required", file=sys.stderr)
        return 2

    try:
        config = _load_config(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    database_path = _database_path(args, config)
    if not database_path.exists():
        print(f"error: database not found: {database_path}", file=sys.stderr)
        return 1

    engine = create_database_engine(path=database_path)
    try:
        repository = OrderRepository(engine)
        order = repository.get(order_id)
        if order is None:
            print(f"error: order not found: {order_id}", file=sys.stderr)
            return 1
        repository.delete(order_id)
        print(f"Deleted: {order_id}")
        return 0
    except SQLAlchemyError as exc:
        print(f"error: cannot delete the order: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()


def _dispatch(args: argparse.Namespace) -> int:
    handler = getattr(args, "_order_handler", None)
    if handler is None:
        print("error: an order command is required", file=sys.stderr)
        return 2
    return int(handler(args))


register_command(
    "order",
    _dispatch,
    help="Create, show and delete personalized orders",
    configure=configure,
)

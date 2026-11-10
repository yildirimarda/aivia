"""``aivia order``: create, show and delete personalized orders.

An order may carry an idempotency key. The first create request with a key
stores the order and returns it. A later create request with the same key
returns that existing order instead of inserting a duplicate, so repeating a
request never creates a second order.
"""

from __future__ import annotations

import argparse
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

__all__ = ["configure", "order_create", "order_delete", "order_show"]


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

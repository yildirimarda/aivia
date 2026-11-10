"""An optional character on orders: both the characterless and the charactered
paths run.

An order may be created with no character, in which case it proceeds normally
and records no character, or with a selected character, in which case that
choice is stored on the order. Blank values are treated as absent.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, cast

import pytest
from sqlalchemy import Engine

from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.db.repos import OrderRepository
from aivia.orders import order


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    database = tmp_path / "data" / "aivia.db"
    upgrade_database(database)
    created = create_database_engine(database)
    try:
        yield created
    finally:
        created.dispose()


@pytest.fixture
def orders(engine: Engine) -> OrderRepository:
    return OrderRepository(engine)


def _base_order_values() -> dict[str, Any]:
    return {
        "occasion": "birthday",
        "name": "Ada",
        "message": "Happy birthday",
        "language": "en",
        "profile": "",
        "level": "standard",
        "timezone": "UTC",
        "currency": "USD",
        "amount_usd": 0.0,
        "budget_usd": None,
        "idempotency_key": "",
    }


def _make_order(repository: OrderRepository, **values: Any) -> dict[str, Any]:
    base = _base_order_values()
    base.update(values)
    return repository.create(**cast("Mapping[str, Any]", base))


def test_order_runs_without_a_character(orders: OrderRepository) -> None:
    stored = order(orders, **_base_order_values())

    assert stored["id"]
    assert stored["character"] == ""
    assert stored["occasion"] == "birthday"
    assert stored["state"] == "draft"


def test_order_runs_with_a_character(orders: OrderRepository) -> None:
    stored = order(orders, character="hero", **_base_order_values())

    assert stored["id"]
    assert stored["character"] == "hero"
    assert stored["occasion"] == "birthday"


def test_blank_character_is_treated_as_absent(orders: OrderRepository) -> None:
    stored = order(orders, character="   ", **_base_order_values())

    assert stored["id"]
    assert stored["character"] == ""


def test_character_is_persisted_on_the_order(orders: OrderRepository) -> None:
    stored = order(orders, character="mascot", **_base_order_values())

    fetched = orders.get(stored["id"])
    assert fetched is not None
    assert fetched["character"] == "mascot"

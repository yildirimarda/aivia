"""Linking orders to their runs and deriving an order's cost from its runs."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, inspect
from sqlalchemy.exc import IntegrityError

from aivia.core.db import (
    OrderRepository,
    RunRepository,
    create_database_engine,
    runs,
    upgrade_database,
)


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    database = tmp_path / "data" / "aivia.db"
    upgrade_database(database)
    created = create_database_engine(database)
    try:
        yield created
    finally:
        created.dispose()


def _columns(engine: Engine, table: str) -> set[str]:
    return {column["name"] for column in inspect(engine).get_columns(table)}


def test_the_migration_creates_the_order_link_on_runs(engine: Engine) -> None:
    assert "order_id" in _columns(engine, "runs")


def test_the_order_link_column_is_registered_on_the_metadata() -> None:
    assert "order_id" in runs.c


def test_runs_have_an_order_lookup_index(engine: Engine) -> None:
    indexes = {
        index["name"]: index
        for index in inspect(engine).get_indexes("runs")
        if index["name"] is not None
    }
    assert "ix_runs_order_id" in indexes
    assert indexes["ix_runs_order_id"]["column_names"] == ["order_id"]
    assert not indexes["ix_runs_order_id"]["unique"]


def test_a_run_can_be_linked_to_an_order(engine: Engine) -> None:
    order = OrderRepository(engine).create(id="order_1", occasion="birthday")
    run = RunRepository(engine).create(id="run_1", order_id=order["id"], cost_usd=4.00)

    assert run.order_id == order["id"]
    fetched = RunRepository(engine).get(run.id)
    assert fetched is not None
    assert fetched.order_id == order["id"]


def test_a_run_without_an_order_is_allowed(engine: Engine) -> None:
    run = RunRepository(engine).create(id="run_1", cost_usd=1.00)

    assert run.order_id == ""
    fetched = RunRepository(engine).get(run.id)
    assert fetched is not None
    assert fetched.order_id == ""


def test_a_run_rejects_a_missing_order(engine: Engine) -> None:
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(runs.insert().values(id="run_1", order_id="missing-order"))


def test_order_cost_equals_the_sum_of_its_runs(engine: Engine) -> None:
    order = OrderRepository(engine).create(id="order_1", cost_usd=0.0)
    RunRepository(engine).create(id="run_1", order_id=order["id"], cost_usd=2.50)
    RunRepository(engine).create(id="run_2", order_id=order["id"], cost_usd=3.75)
    RunRepository(engine).create(id="run_3", cost_usd=99.00)

    assert OrderRepository(engine).cost_for_order(order["id"]) == pytest.approx(6.25)


def test_an_order_with_no_runs_costs_nothing(engine: Engine) -> None:
    order = OrderRepository(engine).create(id="order_1", cost_usd=0.0)

    assert OrderRepository(engine).cost_for_order(order["id"]) == 0.0


def test_recompute_order_cost_stores_the_sum_on_the_order(engine: Engine) -> None:
    order = OrderRepository(engine).create(id="order_1", cost_usd=0.0)
    RunRepository(engine).create(id="run_1", order_id=order["id"], cost_usd=1.25)
    RunRepository(engine).create(id="run_2", order_id=order["id"], cost_usd=2.50)

    updated = OrderRepository(engine).recompute_order_cost(order["id"])
    assert updated is not None
    assert updated["cost_usd"] == pytest.approx(3.75)

    stored = OrderRepository(engine).get(order["id"])
    assert stored is not None
    assert stored["cost_usd"] == pytest.approx(3.75)


def test_recompute_order_cost_on_a_missing_order_returns_none(engine: Engine) -> None:
    assert OrderRepository(engine).recompute_order_cost("missing-order") is None


def test_runs_belonging_to_a_deleted_order_are_cascaded(engine: Engine) -> None:
    order = OrderRepository(engine).create(id="order_1")
    RunRepository(engine).create(id="run_1", order_id=order["id"], cost_usd=5.00)

    OrderRepository(engine).delete(order["id"])

    assert RunRepository(engine).get("run_1") is None

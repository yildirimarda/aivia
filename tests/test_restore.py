"""A scripted backup and restore drill proving runs, orders and costs survive.

A head schema database is seeded with a run, an order and a cost event, then
``sqlite3.Connection.backup`` scripts a copy of the whole database to a backup
file. The backup is restored into a fresh file and every seeded record is
re-read through the repositories and compared against the original values.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

from sqlalchemy import Engine

from aivia.core.db import (
    CostEventRepository,
    OrderRepository,
    RunRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.models import CostEvent, RunKind, RunState


def _live_engine(database: Path) -> Engine:
    """Upgrade *database* to head and return a pragma-enabled engine."""
    upgrade_database(database)
    return create_database_engine(database)


def _backup_file(source: Path, target: Path) -> None:
    """Script a SQLite backup of *source* into *target*."""
    source_conn = sqlite3.connect(source)
    target_conn = sqlite3.connect(target)
    try:
        source_conn.backup(target_conn)
    finally:
        target_conn.close()
        source_conn.close()


def _seed(engine: Engine) -> dict[str, str]:
    """Write one run, one order and one cost event and return their ids."""
    order = OrderRepository(engine).create(
        id="order_drill",
        state="paid",
        occasion="birthday",
        name="Ada",
        amount_usd=9.99,
        currency="USD",
    )
    run = RunRepository(engine).create(
        id="run_drill",
        kind=RunKind.ORDER,
        state=RunState.SUCCEEDED,
        cost_usd=12.50,
        profile="birthday",
        format="card",
        order_id=order["id"],
    )
    cost = CostEventRepository(engine).create(
        CostEvent(
            id="cost_drill",
            provider="mock",
            model="mock-image",
            operation="generate_image",
            units=2,
            amount_usd=0.50,
            run_id=run.id,
        )
    )
    return {"order": order["id"], "run": run.id, "cost": cost.id}


def _assert_seeded_intact(engine: Engine, ids: dict[str, str]) -> None:
    """Re-read every seeded record and assert its values survived."""
    run = RunRepository(engine).get(ids["run"])
    assert run is not None
    assert run.id == ids["run"]
    assert run.kind is RunKind.ORDER
    assert run.state is RunState.SUCCEEDED
    assert run.cost_usd == 12.50
    assert run.profile == "birthday"
    assert run.format == "card"
    assert run.order_id == ids["order"]

    order = OrderRepository(engine).get(ids["order"])
    assert order is not None
    assert order["id"] == ids["order"]
    assert order["state"] == "paid"
    assert order["occasion"] == "birthday"
    assert order["name"] == "Ada"
    assert order["amount_usd"] == 9.99
    assert order["run_id"] == ""

    cost = CostEventRepository(engine).get(ids["cost"])
    assert cost is not None
    assert cost.id == ids["cost"]
    assert cost.provider == "mock"
    assert cost.model == "mock-image"
    assert cost.operation == "generate_image"
    assert cost.units == 2
    assert cost.amount_usd == 0.50
    assert cost.run_id == ids["run"]

    # Counts prove nothing was added, dropped or duplicated by the drill.
    assert len(RunRepository(engine).list()) == 1
    assert len(OrderRepository(engine).list()) == 1
    assert len(CostEventRepository(engine).list()) == 1


def test_scripted_backup_and_restore_preserves_runs_orders_and_costs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "data" / "aivia.db"
    backup = tmp_path / "backup" / "aivia.bak"
    restored = tmp_path / "restored" / "aivia.db"

    engine = _live_engine(source)
    try:
        ids = _seed(engine)
    finally:
        engine.dispose()

    backup.parent.mkdir(parents=True, exist_ok=True)
    _backup_file(source, backup)

    # Restore: recover the backup onto a fresh file as if from backup media.
    restored.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup, restored)

    restore_engine = create_database_engine(restored)
    try:
        _assert_seeded_intact(restore_engine, ids)
    finally:
        restore_engine.dispose()

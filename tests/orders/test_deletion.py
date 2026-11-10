"""Order deletion removes inputs, intermediates, outputs and deliveries while
retention honours pinned finals and the audit entry keeps no personal data."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine

from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.db.order_state import OrderAuditLog, OrderState, OrderStateMachine
from aivia.core.db.repos import (
    AssetRepository,
    CompositionRepository,
    DeliveryRepository,
    OrderRepository,
    RunRepository,
)


def _database_dir(engine: Engine) -> Path:
    url = str(engine.url)
    prefix = "sqlite+pysqlite:///"
    assert url.startswith(prefix)
    return Path(url[len(prefix) :]).parent


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    database = tmp_path / "data" / "aivia.db"
    upgrade_database(database)
    created = create_database_engine(database)
    try:
        yield created
    finally:
        created.dispose()


def _write_file(path: Path, content: bytes = b"data") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_delete_order_removes_run_directory_inputs_and_intermediates(
    engine: Engine,
) -> None:
    base = _database_dir(engine)
    order_repo = OrderRepository(engine)
    run_repo = RunRepository(engine)
    order_repo.create(
        id="order_1",
        state="delivered",
        occasion="birthday",
        name="Ada",
        message="Happy birthday",
        run_id="run_1",
    )
    run_dir = base / "runs" / "run_1"
    run_repo.create(
        id="run_1",
        order_id="order_1",
        run_dir=str(run_dir),
        state="succeeded",
    )
    _write_file(run_dir / "input.txt")
    _write_file(run_dir / "intermediate.png")
    _write_file(run_dir / "artifacts.json")

    machine = OrderStateMachine(engine)
    machine.delete("order_1")

    assert not (run_dir / "input.txt").exists()
    assert not (run_dir / "intermediate.png").exists()
    assert not run_dir.exists()


def test_delete_order_removes_delivery_files_and_rows(engine: Engine) -> None:
    base = _database_dir(engine)
    order_repo = OrderRepository(engine)
    delivery_repo = DeliveryRepository(engine)
    order_repo.create(id="order_1", state="delivered", occasion="birthday")
    delivery_dir = base / "deliveries"
    _write_file(delivery_dir / "final.mp4")
    _write_file(delivery_dir / "poster.png")
    delivery_repo.create(
        id="delivery_1",
        order_id="order_1",
        state="delivered",
        path=str(delivery_dir / "final.mp4"),
        files=[str(delivery_dir / "final.mp4"), str(delivery_dir / "poster.png")],
    )

    machine = OrderStateMachine(engine)
    machine.delete("order_1")

    assert not (delivery_dir / "final.mp4").exists()
    assert not (delivery_dir / "poster.png").exists()
    assert delivery_repo.get("delivery_1") is None


def test_delete_order_removes_non_pinned_outputs(engine: Engine) -> None:
    base = _database_dir(engine)
    order_repo = OrderRepository(engine)
    run_repo = RunRepository(engine)
    asset_repo = AssetRepository(engine)
    order_repo.create(id="order_1", state="delivered", run_id="run_1")
    run_repo.create(id="run_1", order_id="order_1", run_dir="")
    output_dir = base / "outputs"
    _write_file(output_dir / "master.mp4")
    asset_repo.create(
        id="asset_1",
        run_id="run_1",
        path=str(output_dir / "master.mp4"),
        pinned=False,
        kind="video",
    )

    machine = OrderStateMachine(engine)
    machine.delete("order_1")

    assert not (output_dir / "master.mp4").exists()
    assert asset_repo.get("asset_1") is None


def test_retention_keeps_pinned_final_outputs(engine: Engine) -> None:
    base = _database_dir(engine)
    order_repo = OrderRepository(engine)
    run_repo = RunRepository(engine)
    asset_repo = AssetRepository(engine)
    order_repo.create(id="order_1", state="delivered", run_id="run_1")
    run_repo.create(id="run_1", order_id="order_1", run_dir="")
    output_dir = base / "outputs"
    _write_file(output_dir / "final.mp4")
    asset_repo.create(
        id="asset_1",
        run_id="run_1",
        path=str(output_dir / "final.mp4"),
        pinned=True,
        kind="video",
    )

    machine = OrderStateMachine(engine)
    machine.delete("order_1", delete_final_outputs=False)

    assert (output_dir / "final.mp4").exists()
    assert asset_repo.get("asset_1") is not None


def test_delete_final_outputs_removes_pinned_finals(engine: Engine) -> None:
    base = _database_dir(engine)
    order_repo = OrderRepository(engine)
    run_repo = RunRepository(engine)
    asset_repo = AssetRepository(engine)
    order_repo.create(id="order_1", state="delivered", run_id="run_1")
    run_repo.create(id="run_1", order_id="order_1", run_dir="")
    output_dir = base / "outputs"
    _write_file(output_dir / "final.mp4")
    asset_repo.create(
        id="asset_1",
        run_id="run_1",
        path=str(output_dir / "final.mp4"),
        pinned=True,
        kind="video",
    )

    machine = OrderStateMachine(engine)
    machine.delete("order_1", delete_final_outputs=True)

    assert not (output_dir / "final.mp4").exists()
    assert asset_repo.get("asset_1") is None


def test_delete_order_removes_compositions(engine: Engine) -> None:
    base = _database_dir(engine)
    order_repo = OrderRepository(engine)
    run_repo = RunRepository(engine)
    composition_repo = CompositionRepository(engine)
    order_repo.create(id="order_1", state="delivered", run_id="run_1")
    run_repo.create(id="run_1", order_id="order_1", run_dir="")
    card_dir = base / "cards"
    _write_file(card_dir / "card.html")
    composition_repo.create(
        id="composition_1",
        run_id="run_1",
        path=str(card_dir / "card.html"),
        pinned=False,
    )

    machine = OrderStateMachine(engine)
    machine.delete("order_1")

    assert not (card_dir / "card.html").exists()
    assert composition_repo.get("composition_1") is None


def test_delete_order_records_audit_entry_without_personal_data(engine: Engine) -> None:
    entries: list[Any] = []
    order_repo = OrderRepository(engine)
    order_repo.create(
        id="order_1",
        state="delivered",
        occasion="birthday",
        name="Ada",
        message="Happy birthday",
        language="en",
        timezone="UTC",
    )

    machine = OrderStateMachine(engine, audit_log=OrderAuditLog(sink=entries.append))
    machine.delete("order_1")

    assert len(entries) == 1
    entry = entries[0]
    assert entry.order_id == "order_1"
    assert entry.reason == "deleted"
    payload = entry.to_dict()
    for field in ("name", "message", "occasion", "language", "timezone", "variables"):
        assert field not in payload
    assert "Ada" not in str(payload)
    assert "Happy birthday" not in str(payload)
    assert entry.from_state == OrderState.DELIVERED


def test_delete_order_returns_none_when_unknown(engine: Engine) -> None:
    machine = OrderStateMachine(engine)
    assert machine.delete("order_missing") is None

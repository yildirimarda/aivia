"""A populated database from the previous schema version upgrades without loss.

The upgrade path is exercised against a real file database and the shipped
Alembic scripts, not against ``metadata.create_all``. The previous revision is
derived from the migration graph so the test keeps testing a real previous to
head upgrade as new revisions are added.
"""

from __future__ import annotations

from pathlib import Path

from alembic.script import ScriptDirectory
from sqlalchemy import Engine, inspect

from aivia.core.db import (
    AssetRepository,
    CostEventRepository,
    OrderRepository,
    RunRepository,
    alembic_config,
    create_database_engine,
    current_revision,
    upgrade_database,
)
from aivia.core.models import Asset, AssetKind, CostEvent, Run, RunKind, RunState


def _revisions() -> tuple[str, str]:
    """Return the revision before head and the head revision."""
    script = ScriptDirectory.from_config(
        alembic_config(url="sqlite+pysqlite:///:memory:")
    )
    head = script.get_current_head()
    assert head is not None
    previous = script.get_revision(head).down_revision
    assert isinstance(previous, str)
    return previous, head


def _seed(engine: Engine) -> None:
    """Write one run, asset and cost event through their repositories."""
    RunRepository(engine).create(
        Run(
            id="run_previous",
            kind=RunKind.ORDER,
            state=RunState.SUCCEEDED,
            cost_usd=1.25,
            profile="birthday",
            format="card",
        )
    )
    AssetRepository(engine).create(
        Asset(
            id="asset_previous",
            kind=AssetKind.IMAGE,
            label="hero",
            prompt="a cake",
            pinned=True,
            run_id="run_previous",
        )
    )
    CostEventRepository(engine).create(
        CostEvent(
            id="cost_previous",
            provider="mock",
            model="mock-image",
            operation="generate_image",
            units=2,
            amount_usd=0.5,
            run_id="run_previous",
        )
    )


def _assert_seed_intact(engine: Engine) -> None:
    """Assert the run, asset and cost event written by :func:`_seed` survived."""
    run = RunRepository(engine).get("run_previous")
    assert run is not None
    assert run.state is RunState.SUCCEEDED
    assert run.cost_usd == 1.25
    assert run.profile == "birthday"

    asset = AssetRepository(engine).get("asset_previous")
    assert asset is not None
    assert asset.label == "hero"
    assert asset.prompt == "a cake"
    assert asset.pinned is True
    assert asset.run_id == "run_previous"

    cost = CostEventRepository(engine).get("cost_previous")
    assert cost is not None
    assert cost.provider == "mock"
    assert cost.amount_usd == 0.5
    assert cost.run_id == "run_previous"


def test_a_previous_schema_database_upgrades_with_its_records_intact(
    tmp_path: Path,
) -> None:
    previous, head = _revisions()
    database = tmp_path / "data" / "aivia.db"

    upgrade_database(database, revision=previous)
    engine = create_database_engine(database)
    try:
        assert current_revision(engine) == previous
        _seed(engine)
    finally:
        engine.dispose()

    upgrade_database(database)

    engine = create_database_engine(database)
    try:
        assert current_revision(engine) == head
        _assert_seed_intact(engine)
        assert inspect(engine).has_table("orders")
    finally:
        engine.dispose()


def test_an_order_survives_a_subsequent_upgrade(tmp_path: Path) -> None:
    _, head = _revisions()
    database = tmp_path / "data" / "aivia.db"

    upgrade_database(database)
    engine = create_database_engine(database)
    try:
        _seed(engine)
        OrderRepository(engine).create(
            id="order_previous",
            state="paid",
            occasion="birthday",
            name="Ada",
            amount_usd=9.99,
            run_id="run_previous",
        )
    finally:
        engine.dispose()

    upgrade_database(database)

    engine = create_database_engine(database)
    try:
        assert current_revision(engine) == head
        _assert_seed_intact(engine)
        order = OrderRepository(engine).get("order_previous")
        assert order is not None
        assert order["state"] == "paid"
        assert order["occasion"] == "birthday"
        assert order["name"] == "Ada"
        assert order["amount_usd"] == 9.99
        assert order["run_id"] == "run_previous"
    finally:
        engine.dispose()

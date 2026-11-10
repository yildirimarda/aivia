"""``aivia costs order``: spend per order, per output type and per occasion."""

from __future__ import annotations

from pathlib import Path

import pytest

from aivia.cli import main
from aivia.core.cost import CostTracker
from aivia.core.db import (
    AssetRepository,
    OrderRepository,
    RunRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.models import Asset, AssetKind, Run
from aivia.core.models.asset import AssetCostLink


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "aivia.db"
    upgrade_database(path)
    return path


def seed(database: Path) -> tuple[str, str, str]:
    """Three orders across two occasions with image and video spend."""
    engine = create_database_engine(database)
    try:
        orders = OrderRepository(engine)
        orders.create(
            {
                "id": "order_b1",
                "occasion": "birthday",
                "name": "Ada",
                "state": "paid",
            }
        )
        orders.create(
            {
                "id": "order_b2",
                "occasion": "birthday",
                "name": "Bob",
                "state": "paid",
            }
        )
        orders.create(
            {
                "id": "order_w1",
                "occasion": "wedding",
                "name": "Carol",
                "state": "paid",
            }
        )

        runs = RunRepository(engine)
        runs.create(
            Run(id="run_b1", order_id="order_b1", profile="daily", format="short")
        )
        runs.create(
            Run(id="run_b2", order_id="order_b2", profile="daily", format="short")
        )
        runs.create(
            Run(id="run_w1", order_id="order_w1", profile="daily", format="short")
        )

        tracker = CostTracker(engine)
        image_event = tracker.record(
            provider="acme",
            model="image-1",
            operation="image_gen",
            units=1.0,
            unit_kind="image",
            amount_usd=0.10,
            run_id="run_b1",
            estimated=False,
        )
        video_event = tracker.record(
            provider="other",
            model="video-1",
            operation="video_gen",
            units=3.0,
            unit_kind="second",
            amount_usd=0.30,
            run_id="run_b2",
            estimated=False,
        )
        wedding_event = tracker.record(
            provider="acme",
            model="image-1",
            operation="image_gen",
            units=1.0,
            unit_kind="image",
            amount_usd=0.20,
            run_id="run_w1",
            estimated=False,
        )

        assets = AssetRepository(engine)
        assets.create(
            Asset(
                id="asset_b1",
                kind=AssetKind.IMAGE,
                label="card",
                run_id="run_b1",
                cost_links=[AssetCostLink(cost_event_id=image_event.id)],
            )
        )
        assets.create(
            Asset(
                id="asset_b2",
                kind=AssetKind.VIDEO,
                label="clip",
                run_id="run_b2",
                cost_links=[AssetCostLink(cost_event_id=video_event.id)],
            )
        )
        assets.create(
            Asset(
                id="asset_w1",
                kind=AssetKind.IMAGE,
                label="card",
                run_id="run_w1",
                cost_links=[AssetCostLink(cost_event_id=wedding_event.id)],
            )
        )
        return "order_b1", "order_b2", "order_w1"
    finally:
        engine.dispose()


def section(output: str, title: str) -> list[list[str]]:
    lines = output.splitlines()
    start = lines.index(title) + 1
    rows: list[list[str]] = []
    for line in lines[start:]:
        if not line:
            break
        rows.append(line.split("\t"))
    return rows


def run_command(
    capsys: pytest.CaptureFixture[str],
    *args: str,
) -> tuple[int, str, str]:
    code = main(["costs", "order", *args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_costs_order_reports_per_order_output_type_and_occasion(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed(database)

    code, out, err = run_command(capsys, "--database", str(database))

    assert code == 0
    assert err == ""
    assert "Total: $0.600000" in out

    assert section(out, "BY ORDER") == [
        ["ORDER", "OCCASION", "TOTAL", "ESTIMATED", "ACTUAL", "UNITS", "COUNT"],
        ["order_b1", "birthday", "$0.100000", "$0.000000", "$0.100000", "1", "1"],
        ["order_b2", "birthday", "$0.300000", "$0.000000", "$0.300000", "3", "1"],
        ["order_w1", "wedding", "$0.200000", "$0.000000", "$0.200000", "1", "1"],
    ]
    assert section(out, "BY OUTPUT TYPE") == [
        ["OUTPUT TYPE", "TOTAL", "ESTIMATED", "ACTUAL", "UNITS", "COUNT"],
        ["image", "$0.300000", "$0.000000", "$0.300000", "2", "2"],
        ["video", "$0.300000", "$0.000000", "$0.300000", "3", "1"],
    ]
    assert section(out, "BY OCCASION") == [
        ["OCCASION", "ORDERS", "TOTAL", "AVERAGE"],
        ["birthday", "2", "$0.400000", "$0.200000"],
        ["wedding", "1", "$0.200000", "$0.200000"],
    ]


def test_costs_order_reports_a_single_order(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed(database)

    code, out, err = run_command(capsys, "order_b2", "--database", str(database))

    assert code == 0
    assert err == ""
    assert "Total: $0.300000" in out
    assert section(out, "BY ORDER") == [
        ["ORDER", "OCCASION", "TOTAL", "ESTIMATED", "ACTUAL", "UNITS", "COUNT"],
        ["order_b2", "birthday", "$0.300000", "$0.000000", "$0.300000", "3", "1"],
    ]
    assert section(out, "BY OUTPUT TYPE") == [
        ["OUTPUT TYPE", "TOTAL", "ESTIMATED", "ACTUAL", "UNITS", "COUNT"],
        ["video", "$0.300000", "$0.000000", "$0.300000", "3", "1"],
    ]
    assert section(out, "BY OCCASION") == [
        ["OCCASION", "ORDERS", "TOTAL", "AVERAGE"],
        ["birthday", "1", "$0.300000", "$0.300000"],
    ]


def test_costs_order_excludes_a_replaced_estimate(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = create_database_engine(database)
    try:
        orders = OrderRepository(engine)
        orders.create({"id": "order_r", "occasion": "birthday", "state": "paid"})
        RunRepository(engine).create(
            Run(id="run_r", order_id="order_r", profile="daily", format="short")
        )
        tracker = CostTracker(engine)
        estimate = tracker.record_estimate(
            provider="acme",
            model="image-1",
            operation="image_gen",
            units=2.0,
            unit_kind="image",
            amount_usd=0.20,
            run_id="run_r",
        )
        actual = tracker.record_actual(
            provider="acme",
            model="image-1",
            operation="image_gen",
            units=1.0,
            unit_kind="image",
            amount_usd=0.10,
            run_id="run_r",
            metadata={"replaces": estimate.id},
        )
        AssetRepository(engine).create(
            Asset(
                id="asset_r",
                kind=AssetKind.IMAGE,
                label="card",
                run_id="run_r",
                cost_links=[
                    AssetCostLink(cost_event_id=estimate.id),
                    AssetCostLink(cost_event_id=actual.id),
                ],
            )
        )
    finally:
        engine.dispose()

    code, out, err = run_command(capsys, "--database", str(database))

    assert code == 0
    assert err == ""
    assert "Total: $0.100000" in out
    assert section(out, "BY ORDER") == [
        ["ORDER", "OCCASION", "TOTAL", "ESTIMATED", "ACTUAL", "UNITS", "COUNT"],
        ["order_r", "birthday", "$0.100000", "$0.000000", "$0.100000", "1", "1"],
    ]


def test_costs_order_reports_an_empty_ledger(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = create_database_engine(database)
    try:
        OrderRepository(engine).create(
            {"id": "order_empty", "occasion": "birthday", "state": "paid"}
        )
    finally:
        engine.dispose()

    code, out, err = run_command(capsys, "--database", str(database))

    assert code == 0
    assert err == ""
    assert "Total: $0.000000" in out
    assert section(out, "BY ORDER") == [
        ["ORDER", "OCCASION", "TOTAL", "ESTIMATED", "ACTUAL", "UNITS", "COUNT"],
        ["order_empty", "birthday", "$0.000000", "$0.000000", "$0.000000", "0", "0"],
    ]
    assert section(out, "BY OUTPUT TYPE") == [
        ["OUTPUT TYPE", "TOTAL", "ESTIMATED", "ACTUAL", "UNITS", "COUNT"]
    ]
    assert section(out, "BY OCCASION") == [
        ["OCCASION", "ORDERS", "TOTAL", "AVERAGE"],
        ["birthday", "1", "$0.000000", "$0.000000"],
    ]


def test_costs_order_reports_a_missing_order(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, err = run_command(capsys, "order_missing", "--database", str(database))

    assert code == 1
    assert out == ""
    assert "order not found: order_missing" in err


def test_costs_order_reports_a_missing_database(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.db"

    code, out, err = run_command(capsys, "--database", str(missing))

    assert code == 1
    assert out == ""
    assert f"database not found: {missing}" in err


def test_costs_order_honours_provider_and_model_filters(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed(database)

    code, out, err = run_command(
        capsys, "--provider", "other", "--database", str(database)
    )

    assert code == 0
    assert err == ""
    assert "Total: $0.300000" in out
    assert section(out, "BY ORDER") == [
        ["ORDER", "OCCASION", "TOTAL", "ESTIMATED", "ACTUAL", "UNITS", "COUNT"],
        ["order_b1", "birthday", "$0.000000", "$0.000000", "$0.000000", "0", "0"],
        ["order_b2", "birthday", "$0.300000", "$0.000000", "$0.300000", "3", "1"],
        ["order_w1", "wedding", "$0.000000", "$0.000000", "$0.000000", "0", "0"],
    ]

    code, out, err = run_command(
        capsys, "--model", "image-1", "--database", str(database)
    )

    assert code == 0
    assert err == ""
    assert "Total: $0.300000" in out
    assert section(out, "BY ORDER") == [
        ["ORDER", "OCCASION", "TOTAL", "ESTIMATED", "ACTUAL", "UNITS", "COUNT"],
        ["order_b1", "birthday", "$0.100000", "$0.000000", "$0.100000", "1", "1"],
        ["order_b2", "birthday", "$0.000000", "$0.000000", "$0.000000", "0", "0"],
        ["order_w1", "wedding", "$0.200000", "$0.000000", "$0.200000", "1", "1"],
    ]

"""``aivia costs run``: per-run spend broken down by dimension."""

from __future__ import annotations

from pathlib import Path

import pytest

from aivia.cli import main
from aivia.core.cost import CostTracker
from aivia.core.db import (
    AssetRepository,
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


def seed(database: Path) -> str:
    """Write one run with spend across stages, providers, models and assets."""
    engine = create_database_engine(database)
    try:
        run = RunRepository(engine).create(
            Run(id="run_costs", profile="daily", format="short")
        )
        tracker = CostTracker(engine)
        image_actual = tracker.record(
            provider="acme",
            model="image-1",
            operation="image_gen",
            units=1.0,
            unit_kind="image",
            amount_usd=0.10,
            run_id=run.id,
            estimated=False,
        )
        image_estimate = tracker.record(
            provider="acme",
            model="image-1",
            operation="image_gen",
            units=2.0,
            unit_kind="image",
            amount_usd=0.20,
            run_id=run.id,
            estimated=True,
        )
        video = tracker.record(
            provider="other",
            model="video-1",
            operation="video_gen",
            units=3.0,
            unit_kind="second",
            amount_usd=0.30,
            run_id=run.id,
            estimated=False,
        )
        tracker.record(
            provider="other",
            model="llm-1",
            operation="write",
            units=10.0,
            unit_kind="token",
            amount_usd=0.05,
            run_id=run.id,
            estimated=False,
        )

        assets = AssetRepository(engine)
        assets.create(
            Asset(
                id="asset_hero",
                kind=AssetKind.IMAGE,
                label="hero",
                run_id=run.id,
                cost_links=[
                    AssetCostLink(cost_event_id=image_actual.id),
                    AssetCostLink(cost_event_id=image_estimate.id),
                ],
            )
        )
        assets.create(
            Asset(
                id="asset_clip",
                kind=AssetKind.VIDEO,
                label="clip",
                run_id=run.id,
                cost_links=[AssetCostLink(cost_event_id=video.id)],
            )
        )
        return run.id
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
    code = main(["costs", "run", *args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_costs_run_breaks_down_by_stage_provider_model_and_asset(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_id = seed(database)

    code, out, err = run_command(capsys, run_id, "--database", str(database))

    assert code == 0
    assert err == ""
    assert "Run: run_costs" in out
    assert "Total: $0.650000" in out

    assert section(out, "BY STAGE") == [
        ["STAGE", "TOTAL", "ESTIMATED", "ACTUAL", "UNITS", "COUNT"],
        ["image_gen", "$0.300000", "$0.200000", "$0.100000", "3", "2"],
        ["video_gen", "$0.300000", "$0.000000", "$0.300000", "3", "1"],
        ["write", "$0.050000", "$0.000000", "$0.050000", "10", "1"],
    ]
    assert section(out, "BY PROVIDER") == [
        ["PROVIDER", "TOTAL", "ESTIMATED", "ACTUAL", "UNITS", "COUNT"],
        ["acme", "$0.300000", "$0.200000", "$0.100000", "3", "2"],
        ["other", "$0.350000", "$0.000000", "$0.350000", "13", "2"],
    ]
    assert section(out, "BY MODEL") == [
        ["MODEL", "TOTAL", "ESTIMATED", "ACTUAL", "UNITS", "COUNT"],
        ["image-1", "$0.300000", "$0.200000", "$0.100000", "3", "2"],
        ["llm-1", "$0.050000", "$0.000000", "$0.050000", "10", "1"],
        ["video-1", "$0.300000", "$0.000000", "$0.300000", "3", "1"],
    ]
    assert section(out, "BY ASSET") == [
        [
            "ASSET",
            "LABEL",
            "KIND",
            "TOTAL",
            "ESTIMATED",
            "ACTUAL",
            "UNITS",
            "COUNT",
        ],
        [
            "asset_clip",
            "clip",
            "video",
            "$0.300000",
            "$0.000000",
            "$0.300000",
            "3",
            "1",
        ],
        [
            "asset_hero",
            "hero",
            "image",
            "$0.300000",
            "$0.200000",
            "$0.100000",
            "3",
            "2",
        ],
        [
            "-",
            "(unassigned)",
            "-",
            "$0.050000",
            "$0.000000",
            "$0.050000",
            "10",
            "1",
        ],
    ]


def test_costs_run_accepts_the_run_option_and_db_alias(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_id = seed(database)

    code, out, err = run_command(
        capsys,
        "--run-id",
        run_id,
        "--db",
        str(database),
    )

    assert code == 0
    assert err == ""
    assert "Run: run_costs" in out


def test_costs_run_reads_the_database_beside_an_explicit_config(
    database: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_id = seed(database)
    config = tmp_path / "config.yaml"
    config.write_text("database:\n  filename: aivia.db\n", encoding="utf-8")

    code, out, err = run_command(
        capsys,
        run_id,
        "--config",
        str(config),
    )

    assert code == 0
    assert err == ""
    assert "Run: run_costs" in out
    assert f"database not found: {database}" not in err


def test_costs_run_reports_a_missing_run(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, err = run_command(capsys, "run_missing", "--database", str(database))

    assert code == 1
    assert out == ""
    assert "run not found: run_missing" in err


def test_costs_run_reports_a_missing_database(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.db"

    code, out, err = run_command(capsys, "run_1", "--database", str(missing))

    assert code == 1
    assert out == ""
    assert f"database not found: {missing}" in err


def test_costs_run_requires_a_run_id(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, err = run_command(capsys, "--database", str(database))

    assert code == 2
    assert out == ""
    assert "a run id is required" in err


def test_costs_run_excludes_a_replaced_estimate_from_asset_totals(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = create_database_engine(database)
    try:
        run = RunRepository(engine).create(
            Run(id="run_replaced", profile="daily", format="short")
        )
        tracker = CostTracker(engine)
        estimate = tracker.record_estimate(
            provider="acme",
            model="image-1",
            operation="image_gen",
            units=2.0,
            unit_kind="image",
            amount_usd=0.20,
            run_id=run.id,
        )
        actual = tracker.record_actual(
            provider="acme",
            model="image-1",
            operation="image_gen",
            units=1.0,
            unit_kind="image",
            amount_usd=0.10,
            run_id=run.id,
            metadata={"replaces": estimate.id},
        )
        AssetRepository(engine).create(
            Asset(
                id="asset_hero",
                kind=AssetKind.IMAGE,
                label="hero",
                run_id=run.id,
                cost_links=[
                    AssetCostLink(cost_event_id=estimate.id),
                    AssetCostLink(cost_event_id=actual.id),
                ],
            )
        )
    finally:
        engine.dispose()

    code, out, err = run_command(capsys, "run_replaced", "--database", str(database))

    assert code == 0
    assert err == ""
    assert section(out, "BY ASSET") == [
        [
            "ASSET",
            "LABEL",
            "KIND",
            "TOTAL",
            "ESTIMATED",
            "ACTUAL",
            "UNITS",
            "COUNT",
        ],
        [
            "asset_hero",
            "hero",
            "image",
            "$0.100000",
            "$0.000000",
            "$0.100000",
            "1",
            "1",
        ],
    ]


def test_costs_run_reports_an_empty_ledger_for_an_existing_run(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = create_database_engine(database)
    try:
        RunRepository(engine).create(Run(id="run_empty"))
    finally:
        engine.dispose()

    code, out, err = run_command(capsys, "run_empty", "--database", str(database))

    assert code == 0
    assert err == ""
    assert "Total: $0.000000" in out
    assert section(out, "BY STAGE") == [
        ["STAGE", "TOTAL", "ESTIMATED", "ACTUAL", "UNITS", "COUNT"]
    ]
    assert section(out, "BY ASSET") == [
        [
            "ASSET",
            "LABEL",
            "KIND",
            "TOTAL",
            "ESTIMATED",
            "ACTUAL",
            "UNITS",
            "COUNT",
        ]
    ]

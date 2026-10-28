"""``aivia costs summary``: grouped spend with optional date filters."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from aivia.cli import main
from aivia.core.cost import CostTracker
from aivia.core.db import (
    RunRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.models import Run


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "aivia.db"
    upgrade_database(path)
    return path


def seed(database: Path) -> str:
    """Write spend across two days, providers, models, stages and one run."""
    engine = create_database_engine(database)
    try:
        run = RunRepository(engine).create(
            Run(id="run_costs", profile="daily", format="short")
        )
        tracker = CostTracker(engine)
        tracker.record(
            provider="acme",
            model="image-1",
            operation="image_gen",
            units=1.0,
            unit_kind="image",
            amount_usd=0.10,
            run_id=run.id,
            estimated=False,
            created_at=datetime(2026, 1, 5, 10, 0, tzinfo=UTC),
        )
        tracker.record(
            provider="acme",
            model="image-1",
            operation="image_gen",
            units=2.0,
            unit_kind="image",
            amount_usd=0.20,
            run_id=run.id,
            estimated=True,
            created_at=datetime(2026, 1, 5, 15, 0, tzinfo=UTC),
        )
        tracker.record(
            provider="other",
            model="video-1",
            operation="video_gen",
            units=3.0,
            unit_kind="second",
            amount_usd=0.30,
            run_id=run.id,
            estimated=False,
            created_at=datetime(2026, 1, 12, 9, 0, tzinfo=UTC),
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
            created_at=datetime(2026, 1, 12, 12, 0, tzinfo=UTC),
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
    code = main(["costs", "summary", *args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_costs_summary_defaults_to_grouping_by_day(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed(database)

    code, out, err = run_command(capsys, "--database", str(database))

    assert code == 0
    assert err == ""
    assert "Total: $0.650000" in out
    assert section(out, "BY DAY") == [
        ["DAY", "TOTAL", "ESTIMATED", "ACTUAL", "UNITS", "COUNT"],
        ["2026-01-05", "$0.300000", "$0.200000", "$0.100000", "3", "2"],
        ["2026-01-12", "$0.350000", "$0.000000", "$0.350000", "13", "2"],
    ]


def test_costs_summary_groups_by_a_chosen_dimension(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed(database)

    code, out, err = run_command(
        capsys, "--group", "provider", "--database", str(database)
    )

    assert code == 0
    assert err == ""
    assert section(out, "BY PROVIDER") == [
        ["PROVIDER", "TOTAL", "ESTIMATED", "ACTUAL", "UNITS", "COUNT"],
        ["acme", "$0.300000", "$0.200000", "$0.100000", "3", "2"],
        ["other", "$0.350000", "$0.000000", "$0.350000", "13", "2"],
    ]
    assert "BY DAY" not in out


def test_costs_summary_accepts_several_groups(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed(database)

    code, out, err = run_command(
        capsys,
        "--group",
        "provider",
        "--group",
        "model",
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
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


def test_costs_summary_accepts_comma_separated_groups(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed(database)

    code, out, err = run_command(
        capsys, "--group", "stage,format", "--database", str(database)
    )

    assert code == 0
    assert err == ""
    assert section(out, "BY STAGE") == [
        ["STAGE", "TOTAL", "ESTIMATED", "ACTUAL", "UNITS", "COUNT"],
        ["image_gen", "$0.300000", "$0.200000", "$0.100000", "3", "2"],
        ["video_gen", "$0.300000", "$0.000000", "$0.300000", "3", "1"],
        ["write", "$0.050000", "$0.000000", "$0.050000", "10", "1"],
    ]
    assert section(out, "BY FORMAT") == [
        ["FORMAT", "TOTAL", "ESTIMATED", "ACTUAL", "UNITS", "COUNT"],
        ["short", "$0.650000", "$0.200000", "$0.450000", "16", "4"],
    ]


def test_costs_summary_filters_by_an_inclusive_date_range(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed(database)

    code, out, err = run_command(
        capsys,
        "--start",
        "2026-01-12",
        "--end",
        "2026-01-12",
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    assert "Total: $0.350000" in out
    assert section(out, "BY DAY") == [
        ["DAY", "TOTAL", "ESTIMATED", "ACTUAL", "UNITS", "COUNT"],
        ["2026-01-12", "$0.350000", "$0.000000", "$0.350000", "13", "2"],
    ]


def test_costs_summary_filters_by_a_start_datetime(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed(database)

    code, out, err = run_command(
        capsys,
        "--since",
        "2026-01-05T12:00:00Z",
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    assert "Total: $0.550000" in out
    assert section(out, "BY DAY") == [
        ["DAY", "TOTAL", "ESTIMATED", "ACTUAL", "UNITS", "COUNT"],
        ["2026-01-05", "$0.200000", "$0.200000", "$0.000000", "2", "1"],
        ["2026-01-12", "$0.350000", "$0.000000", "$0.350000", "13", "2"],
    ]


def test_costs_summary_filters_by_an_end_date(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed(database)

    code, out, err = run_command(
        capsys, "--until", "2026-01-05", "--database", str(database)
    )

    assert code == 0
    assert err == ""
    assert "Total: $0.300000" in out
    assert section(out, "BY DAY") == [
        ["DAY", "TOTAL", "ESTIMATED", "ACTUAL", "UNITS", "COUNT"],
        ["2026-01-05", "$0.300000", "$0.200000", "$0.100000", "3", "2"],
    ]


def test_costs_summary_narrows_by_run_provider_model_and_stage(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_id = seed(database)

    code, out, err = run_command(
        capsys,
        "--group",
        "stage",
        "--run-id",
        run_id,
        "--provider",
        "acme",
        "--model",
        "image-1",
        "--stage",
        "image_gen",
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    assert "Total: $0.300000" in out
    assert section(out, "BY STAGE") == [
        ["STAGE", "TOTAL", "ESTIMATED", "ACTUAL", "UNITS", "COUNT"],
        ["image_gen", "$0.300000", "$0.200000", "$0.100000", "3", "2"],
    ]


def test_costs_summary_reports_an_empty_ledger(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = create_database_engine(database)
    try:
        RunRepository(engine).create(Run(id="run_empty"))
    finally:
        engine.dispose()

    code, out, err = run_command(
        capsys, "--group", "provider", "--database", str(database)
    )

    assert code == 0
    assert err == ""
    assert "Total: $0.000000" in out
    assert section(out, "BY PROVIDER") == [
        ["PROVIDER", "TOTAL", "ESTIMATED", "ACTUAL", "UNITS", "COUNT"]
    ]


def test_costs_summary_reads_the_database_beside_an_explicit_config(
    database: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed(database)
    config = tmp_path / "config.yaml"
    config.write_text("database:\n  filename: aivia.db\n", encoding="utf-8")

    code, out, err = run_command(capsys, "--config", str(config))

    assert code == 0
    assert err == ""
    assert "Total: $0.650000" in out


def test_costs_summary_rejects_an_unknown_group(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, err = run_command(
        capsys, "--group", "asset", "--database", str(database)
    )

    assert code == 2
    assert out == ""
    assert "unknown group 'asset'" in err


def test_costs_summary_rejects_an_invalid_date(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, err = run_command(
        capsys, "--start", "not-a-date", "--database", str(database)
    )

    assert code == 2
    assert out == ""
    assert "invalid date" in err


def test_costs_summary_reports_a_missing_database(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.db"

    code, out, err = run_command(capsys, "--database", str(missing))

    assert code == 1
    assert out == ""
    assert f"database not found: {missing}" in err

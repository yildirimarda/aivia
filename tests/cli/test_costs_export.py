"""``aivia costs export``: CSV and JSON spend with summary-matching totals."""

from __future__ import annotations

import csv
import io
import json
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
    """Write spend across two days, providers, models and stages."""
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


def run_command(
    capsys: pytest.CaptureFixture[str],
    *args: str,
) -> tuple[int, str, str]:
    code = main(["costs", *args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def summary_total(capsys: pytest.CaptureFixture[str], database: Path) -> float:
    code, out, err = run_command(capsys, "summary", "--database", str(database))
    assert code == 0
    assert err == ""
    line = next(line for line in out.splitlines() if line.startswith("Total: $"))
    return float(line.removeprefix("Total: $"))


def test_costs_export_json_totals_match_the_summary(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed(database)
    expected_total = summary_total(capsys, database)

    code, out, err = run_command(
        capsys, "export", "--format", "json", "--database", str(database)
    )

    assert code == 0
    assert err == ""
    payload = json.loads(out)
    assert payload["total_usd"] == pytest.approx(expected_total)
    assert payload["total"] == pytest.approx(expected_total)
    assert payload["group"] == "day"
    assert payload["rows"] == [
        {
            "group": "day",
            "key": "2026-01-05",
            "total_usd": pytest.approx(0.30),
            "estimated_usd": pytest.approx(0.20),
            "actual_usd": pytest.approx(0.10),
            "units": pytest.approx(3.0),
            "count": 2,
        },
        {
            "group": "day",
            "key": "2026-01-12",
            "total_usd": pytest.approx(0.35),
            "estimated_usd": pytest.approx(0.0),
            "actual_usd": pytest.approx(0.35),
            "units": pytest.approx(13.0),
            "count": 2,
        },
    ]
    assert sum(row["total_usd"] for row in payload["rows"]) == pytest.approx(
        expected_total
    )
    assert payload["totals"]["total_usd"] == pytest.approx(expected_total)
    assert payload["groups"][0]["total_usd"] == pytest.approx(expected_total)


def test_costs_export_csv_totals_match_the_summary(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed(database)
    expected_total = summary_total(capsys, database)

    code, out, err = run_command(
        capsys,
        "export",
        "--format",
        "csv",
        "--group",
        "provider",
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    reader = csv.DictReader(io.StringIO(out))
    assert reader.fieldnames == [
        "group",
        "key",
        "total_usd",
        "estimated_usd",
        "actual_usd",
        "units",
        "count",
    ]
    rows = list(reader)
    assert [row["key"] for row in rows] == ["acme", "other"]
    assert rows[0]["group"] == "provider"
    assert float(rows[0]["total_usd"]) == pytest.approx(0.30)
    assert float(rows[1]["total_usd"]) == pytest.approx(0.35)
    assert sum(float(row["total_usd"]) for row in rows) == pytest.approx(expected_total)


def test_costs_export_defaults_to_csv_and_honours_filters(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed(database)

    code, out, err = run_command(
        capsys,
        "export",
        "--provider",
        "other",
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    rows = list(csv.DictReader(io.StringIO(out)))
    assert len(rows) == 1
    assert rows[0]["key"] == "2026-01-12"
    assert float(rows[0]["total_usd"]) == pytest.approx(0.35)


def test_costs_export_writes_a_file_and_infers_its_format(
    database: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed(database)
    target = tmp_path / "reports" / "costs.json"

    code, out, err = run_command(
        capsys,
        "export",
        "--output",
        str(target),
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    assert f"wrote {target}" in out
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["total_usd"] == pytest.approx(0.65)


def test_costs_summary_can_export_json_too(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed(database)
    expected_total = summary_total(capsys, database)

    code, out, err = run_command(
        capsys, "summary", "--format", "json", "--database", str(database)
    )

    assert code == 0
    assert err == ""
    assert json.loads(out)["total_usd"] == pytest.approx(expected_total)


def test_costs_export_reports_an_empty_ledger(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = create_database_engine(database)
    try:
        RunRepository(engine).create(Run(id="run_empty"))
    finally:
        engine.dispose()

    code, out, err = run_command(
        capsys, "export", "--format", "json", "--database", str(database)
    )

    assert code == 0
    assert err == ""
    payload = json.loads(out)
    assert payload["total_usd"] == 0.0
    assert payload["rows"] == []


def test_costs_export_reports_a_missing_database(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.db"

    code, out, err = run_command(
        capsys, "export", "--format", "json", "--database", str(missing)
    )

    assert code == 1
    assert out == ""
    assert f"database not found: {missing}" in err


def test_costs_export_rejects_an_unknown_format(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, err = run_command(
        capsys, "export", "--format", "xml", "--database", str(database)
    )

    assert code == 2
    assert out == ""
    assert "invalid choice" in err

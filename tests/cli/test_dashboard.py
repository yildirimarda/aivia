"""``aivia costs dashboard``: a static HTML spend dashboard with budget bars."""

from __future__ import annotations

from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

import pytest
from sqlalchemy import text

from aivia.cli import main
from aivia.core.cost import CostTracker
from aivia.core.db import (
    RunRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.models import Run

NOW = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "aivia.db"
    upgrade_database(path)
    return path


def seed(database: Path) -> None:
    """Write spend across three days, two runs and two providers."""
    engine = create_database_engine(database)
    try:
        runs = RunRepository(engine)
        alpha = runs.create(Run(id="run_alpha", profile="daily", format="short"))
        beta = runs.create(Run(id="run_beta", profile="daily", format="short"))
        tracker = CostTracker(engine)
        tracker.record(
            provider="acme",
            model="image-1",
            operation="image_gen",
            units=1.0,
            unit_kind="image",
            amount_usd=0.10,
            run_id=alpha.id,
            estimated=False,
            created_at=datetime(2026, 1, 15, 9, 0, tzinfo=UTC),
        )
        tracker.record(
            provider="acme",
            model="image-1",
            operation="image_gen",
            units=2.0,
            unit_kind="image",
            amount_usd=0.20,
            run_id=alpha.id,
            estimated=True,
            created_at=datetime(2026, 1, 15, 10, 0, tzinfo=UTC),
        )
        tracker.record(
            provider="other",
            model="video-1",
            operation="video_gen",
            units=3.0,
            unit_kind="second",
            amount_usd=0.30,
            run_id=beta.id,
            estimated=False,
            created_at=datetime(2026, 1, 14, 9, 0, tzinfo=UTC),
        )
        tracker.record(
            provider="other",
            model="llm-1",
            operation="write",
            units=10.0,
            unit_kind="token",
            amount_usd=0.05,
            run_id=beta.id,
            estimated=False,
            created_at=datetime(2026, 1, 2, 12, 0, tzinfo=UTC),
        )
    finally:
        engine.dispose()


class _SectionParser(HTMLParser):
    """Collect table rows under each ``h2`` heading."""

    def __init__(self) -> None:
        super().__init__()
        self.sections: dict[str, list[list[str]]] = {}
        self._heading = ""
        self._in_heading = False
        self._in_cell = False
        self._cell: list[str] = []
        self._row: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag == "h2":
            self._in_heading = True
            self._heading = ""
        elif tag == "tr":
            self._row = []
        elif tag in {"td", "th"}:
            self._in_cell = True
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "h2":
            self._in_heading = False
            self.sections.setdefault(self._heading, [])
        elif tag in {"td", "th"}:
            self._in_cell = False
            self._row.append("".join(self._cell).strip())
        elif tag == "tr":
            self.sections.setdefault(self._heading, []).append(self._row)

    def handle_data(self, data: str) -> None:
        if self._in_heading:
            self._heading += data
        if self._in_cell:
            self._cell.append(data)


def tables(text: str) -> dict[str, list[list[str]]]:
    parser = _SectionParser()
    parser.feed(text)
    return parser.sections


def run_dashboard(
    capsys: pytest.CaptureFixture[str],
    database: Path,
    *args: str,
) -> tuple[int, str, str]:
    code = main(["costs", "dashboard", "--database", str(database), *args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def budget_args() -> list[str]:
    return [
        "--now",
        "2026-01-15T12:00:00Z",
        "--set",
        "budget.per_day=0.50",
        "--set",
        "budget.per_month=1.00",
        "--set",
        "budget.per_run=0.25",
        "--set",
        "budget.per_provider.acme=0.20",
        "--set",
        "budget.per_provider.other=0.50",
    ]


def test_dashboard_writes_spend_per_day_run_and_provider(
    database: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed(database)
    output = tmp_path / "dashboard.html"

    code, out, err = run_dashboard(
        capsys,
        database,
        "--output",
        str(output),
        *budget_args(),
    )

    assert code == 0
    assert err == ""
    assert f"wrote {output}" in out
    text = output.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    assert "<h1>AIVIA spend dashboard</h1>" in text
    assert "Generated 2026-01-15T12:00:00+00:00" in text
    assert "total $0.650000" in text
    assert "<script" not in text

    sections = tables(text)
    assert sections["Spend per day"] == [
        ["Day", "Total", "Estimated", "Actual", "Units", "Count"],
        ["2026-01-02", "$0.050000", "$0.000000", "$0.050000", "10", "1"],
        ["2026-01-14", "$0.300000", "$0.000000", "$0.300000", "3", "1"],
        ["2026-01-15", "$0.300000", "$0.200000", "$0.100000", "3", "2"],
    ]
    assert sections["Spend per run"] == [
        ["Run", "Total", "Estimated", "Actual", "Units", "Count"],
        ["run_alpha", "$0.300000", "$0.200000", "$0.100000", "3", "2"],
        ["run_beta", "$0.350000", "$0.000000", "$0.350000", "13", "2"],
    ]
    assert sections["Spend per provider"] == [
        ["Provider", "Total", "Estimated", "Actual", "Units", "Count"],
        ["acme", "$0.300000", "$0.200000", "$0.100000", "3", "2"],
        ["other", "$0.350000", "$0.000000", "$0.350000", "13", "2"],
    ]


def test_dashboard_renders_budget_bars_for_every_configured_scope(
    database: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed(database)
    output = tmp_path / "dashboard.html"

    code, _, err = run_dashboard(
        capsys,
        database,
        "--output",
        str(output),
        *budget_args(),
    )

    assert code == 0
    assert err == ""
    text = output.read_text(encoding="utf-8")
    assert 'role="progressbar"' in text
    assert "<span>Today</span><span>$0.300000 / $0.500000 (60.0%)</span>" in text
    assert "<span>This month</span><span>$0.650000 / $1.000000 (65.0%)</span>" in text
    assert (
        "<span>Run run_alpha</span><span>$0.300000 / $0.250000 (120.0%)</span>" in text
    )
    assert (
        "<span>Run run_beta</span><span>$0.350000 / $0.250000 (140.0%)</span>" in text
    )
    assert (
        "<span>Provider acme</span><span>$0.300000 / $0.200000 (150.0%)</span>" in text
    )
    assert (
        "<span>Provider other</span><span>$0.350000 / $0.500000 (70.0%)</span>" in text
    )
    assert "Over by $0.100000" in text
    assert "Over by $0.050000" in text
    assert "$0.150000 remaining" in text


def test_dashboard_filters_tables_without_hiding_global_budgets(
    database: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed(database)
    output = tmp_path / "dashboard.html"

    code, _, err = run_dashboard(
        capsys,
        database,
        "--output",
        str(output),
        "--run",
        "run_beta",
        *budget_args(),
    )

    assert code == 0
    assert err == ""
    sections = tables(output.read_text(encoding="utf-8"))
    assert sections["Spend per run"] == [
        ["Run", "Total", "Estimated", "Actual", "Units", "Count"],
        ["run_beta", "$0.350000", "$0.000000", "$0.350000", "13", "2"],
    ]
    assert [row[0] for row in sections["Spend per day"][1:]] == [
        "2026-01-02",
        "2026-01-14",
    ]
    assert [row[0] for row in sections["Spend per provider"][1:]] == ["other"]


def test_dashboard_reports_when_no_budget_caps_are_configured(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed(database)

    code, out, err = run_dashboard(
        capsys,
        database,
        "--now",
        "2026-01-15T12:00:00Z",
    )

    assert code == 0
    assert err == ""
    assert out.startswith("<!doctype html>")
    assert "No budget caps configured." in out
    assert 'role="progressbar"' not in out


def test_dashboard_escapes_ledger_text(
    database: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = create_database_engine(database)
    try:
        run = RunRepository(engine).create(Run(id="run_escape"))
        CostTracker(engine).record(
            provider="<script>alert(1)</script>",
            model="image-1",
            operation="image_gen",
            units=1.0,
            amount_usd=0.01,
            run_id=run.id,
            created_at=NOW,
        )
    finally:
        engine.dispose()
    output = tmp_path / "dashboard.html"

    code, _, err = run_dashboard(
        capsys,
        database,
        "--output",
        str(output),
        "--now",
        "2026-01-15T12:00:00Z",
    )

    assert code == 0
    assert err == ""
    text = output.read_text(encoding="utf-8")
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text
    assert "<script>alert(1)</script>" not in text


def test_dashboard_rejects_an_invalid_now(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, err = run_dashboard(capsys, database, "--now", "not-a-date")

    assert code == 2
    assert out == ""
    assert "invalid date" in err


def test_dashboard_reports_a_missing_database(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.db"

    code = main(["costs", "dashboard", "--database", str(missing)])
    captured = capsys.readouterr()

    assert code == 1
    assert captured.out == ""
    assert f"database not found: {missing}" in captured.err


def test_dashboard_reports_performance_by_dimension_and_cpm(
    database: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = create_database_engine(database)
    try:
        run = RunRepository(engine).create(
            Run(id="run_alpha", profile="daily", format="short")
        )
        tracker = CostTracker(engine)
        tracker.record(
            provider="acme",
            model="gen-1",
            operation="image",
            units=1.0,
            unit_kind="image",
            amount_usd=0.10,
            run_id=run.id,
            estimated=False,
            created_at=datetime(2026, 1, 15, 9, 0, tzinfo=UTC),
            metadata={
                "mode": "script",
                "category": "news",
                "hook_style": "question",
                "visual_strategy": "hybrid",
            },
        )
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO posts (id, run_id, platform, language) "
                    "VALUES (:id, :run_id, :platform, :language)"
                ),
                {
                    "id": "post_alpha",
                    "run_id": run.id,
                    "platform": "tiktok",
                    "language": "en",
                },
            )
            connection.execute(
                text(
                    "INSERT INTO post_metrics (id, post_id, views) "
                    "VALUES (:id, :post_id, :views)"
                ),
                {"id": "metric_alpha", "post_id": "post_alpha", "views": 1000},
            )
    finally:
        engine.dispose()

    output = tmp_path / "dashboard.html"
    code, _, err = run_dashboard(
        capsys,
        database,
        "--output",
        str(output),
        "--now",
        "2026-01-15T12:00:00Z",
    )

    assert code == 0
    assert err == ""
    text_output = output.read_text(encoding="utf-8")
    assert "overall CPM $0.100000" in text_output
    sections = tables(text_output)
    assert sections["Performance by platform"] == [
        ["Platform", "Cost", "Views", "CPM per 1k views"],
        ["tiktok", "$0.100000", "1000", "$0.100000"],
    ]
    assert sections["Performance by profile"] == [
        ["Profile", "Cost", "Views", "CPM per 1k views"],
        ["daily", "$0.100000", "1000", "$0.100000"],
    ]
    assert sections["Performance by format"] == [
        ["Format", "Cost", "Views", "CPM per 1k views"],
        ["short", "$0.100000", "1000", "$0.100000"],
    ]
    assert sections["Performance by mode"] == [
        ["Mode", "Cost", "Views", "CPM per 1k views"],
        ["script", "$0.100000", "1000", "$0.100000"],
    ]
    assert sections["Performance by category"] == [
        ["Category", "Cost", "Views", "CPM per 1k views"],
        ["news", "$0.100000", "1000", "$0.100000"],
    ]
    assert sections["Performance by hook style"] == [
        ["Hook style", "Cost", "Views", "CPM per 1k views"],
        ["question", "$0.100000", "1000", "$0.100000"],
    ]
    assert sections["Performance by visual strategy"] == [
        ["Visual strategy", "Cost", "Views", "CPM per 1k views"],
        ["hybrid", "$0.100000", "1000", "$0.100000"],
    ]

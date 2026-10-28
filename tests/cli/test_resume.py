"""``aivia resume``: continue a budget-paused run with an audit trail."""

from __future__ import annotations

import io
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from aivia.cli import main
from aivia.core.db import (
    RunRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.models import Run, RunState

PAUSED_AT = datetime(2026, 1, 2, 10, 0, tzinfo=UTC)
PAUSE_REASON = "budget: estimated call cost would exceed the per-run cap"


class _Stream(io.StringIO):
    """A stdin replacement whose terminal status the test controls."""

    def __init__(self, text: str = "", *, tty: bool) -> None:
        super().__init__(text)
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "aivia.db"
    upgrade_database(path)
    return path


def seed(database: Path, *, state: RunState = RunState.PAUSED_BUDGET) -> str:
    engine = create_database_engine(database)
    try:
        run = RunRepository(engine).create(
            Run(
                id="run_paused",
                state=state,
                paused_at=PAUSED_AT if state is RunState.PAUSED_BUDGET else None,
                pause_reason=(PAUSE_REASON if state is RunState.PAUSED_BUDGET else ""),
            )
        )
        return run.id
    finally:
        engine.dispose()


def stored_run(database: Path, run_id: str) -> Run:
    engine = create_database_engine(database)
    try:
        run = RunRepository(engine).get(run_id)
        assert run is not None
        return run
    finally:
        engine.dispose()


def run_command(
    capsys: pytest.CaptureFixture[str],
    *args: str,
) -> tuple[int, str, str]:
    code = main(["resume", *args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_resume_requires_yes_in_a_non_interactive_run(
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = seed(database)
    monkeypatch.setattr(sys, "stdin", _Stream("yes\n", tty=False))

    code, out, err = run_command(
        capsys,
        run_id,
        "--by",
        "alice",
        "--reason",
        "cap raised",
        "--database",
        str(database),
    )

    assert code == 1
    assert "non-interactive runs require --yes" in err
    assert out == ""
    assert stored_run(database, run_id).state is RunState.PAUSED_BUDGET


def test_yes_resumes_and_records_who_and_why(
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = seed(database)
    monkeypatch.setattr(sys, "stdin", _Stream(tty=False))

    code, out, err = run_command(
        capsys,
        run_id,
        "--by",
        "alice",
        "--reason",
        "cap raised",
        "--yes",
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    assert "Resumed run run_paused as alice: cap raised" in out

    run = stored_run(database, run_id)
    assert run.state is RunState.RUNNING
    assert run.is_paused is False
    assert run.resumed_by == "alice"
    assert run.resume_reason == "cap raised"
    assert run.resumed_at is not None
    assert run.resumed_at.tzinfo is UTC
    assert run.paused_at == PAUSED_AT
    assert run.pause_reason == PAUSE_REASON


def test_interactive_confirmation_accepts_yes(
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = seed(database)
    monkeypatch.setattr(sys, "stdin", _Stream("y\n", tty=True))

    code, out, err = run_command(
        capsys,
        run_id,
        "--who",
        "bob",
        "--why",
        "reviewed the budget",
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    assert "Resume paused budget run run_paused? [y/N]" in out
    assert stored_run(database, run_id).resumed_by == "bob"


def test_interactive_confirmation_declined_leaves_the_run_paused(
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = seed(database)
    monkeypatch.setattr(sys, "stdin", _Stream("n\n", tty=True))

    code, out, err = run_command(
        capsys,
        run_id,
        "--by",
        "alice",
        "--reason",
        "cap raised",
        "--database",
        str(database),
    )

    assert code == 1
    assert "Resume paused budget run run_paused? [y/N]" in out
    assert "confirmation declined" in err
    run = stored_run(database, run_id)
    assert run.state is RunState.PAUSED_BUDGET
    assert run.resumed_by == ""


def test_resume_rejects_a_run_that_is_not_paused(
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = seed(database, state=RunState.RUNNING)
    monkeypatch.setattr(sys, "stdin", _Stream(tty=False))

    code, out, err = run_command(
        capsys,
        run_id,
        "--by",
        "alice",
        "--reason",
        "cap raised",
        "--yes",
        "--database",
        str(database),
    )

    assert code == 1
    assert out == ""
    assert "is not paused on budget (state: running)" in err
    assert stored_run(database, run_id).state is RunState.RUNNING


def test_resume_rejects_an_unknown_run(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, err = run_command(
        capsys,
        "run_missing",
        "--by",
        "alice",
        "--reason",
        "cap raised",
        "--yes",
        "--database",
        str(database),
    )

    assert code == 1
    assert out == ""
    assert "error: run not found: run_missing" in err


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("--by", "--by is required"),
        ("--reason", "--reason is required"),
    ],
)
def test_resume_requires_who_and_why(
    database: Path,
    capsys: pytest.CaptureFixture[str],
    missing: str,
    message: str,
) -> None:
    run_id = seed(database)
    args = {
        "--by": "alice",
        "--reason": "cap raised",
    }
    args.pop(missing)
    command = [run_id, "--yes", "--database", str(database)]
    for flag, value in args.items():
        command.extend([flag, value])

    code, _, err = run_command(capsys, *command)

    assert code == 2
    assert message in err
    assert stored_run(database, run_id).state is RunState.PAUSED_BUDGET


def test_resume_requires_a_run_id(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, _, err = run_command(
        capsys,
        "--by",
        "alice",
        "--reason",
        "cap raised",
        "--yes",
    )

    assert code == 2
    assert "a run id is required" in err


def test_resume_reports_a_missing_database(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "data" / "missing.db"

    code, _, err = run_command(
        capsys,
        "run_paused",
        "--by",
        "alice",
        "--reason",
        "cap raised",
        "--yes",
        "--database",
        str(missing),
    )

    assert code == 1
    assert f"database not found: {missing}" in err


def test_resume_accepts_the_run_option(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_id = seed(database)

    code, out, err = run_command(
        capsys,
        "--run",
        run_id,
        "--by",
        "alice",
        "--reason",
        "cap raised",
        "--yes",
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    assert "Resumed run run_paused" in out

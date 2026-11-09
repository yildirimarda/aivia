"""``aivia session repl``: the interactive session REPL.

Each message refines or replans the session by mapping to a composition patch.
The five sample edits are applied, undone and redone, including branching when
an edit follows an undo.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from aivia.cli import main

_FIVE_EDITS = "background\nduration\nfps\ntitle\nopacity\n"


def _run(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    stdin: str = "",
) -> tuple[int, str, str]:
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    code = main(["session", "repl"])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_repl_applies_each_of_the_five_sample_edits(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, out, err = _run(capsys, monkeypatch, _FIVE_EDITS)

    assert code == 0, err
    assert err == ""

    assert "applied Set canvas background to white. -> version 2" in out
    assert "bg=#ffffff" in out

    assert "applied Extend duration to 15 seconds. -> version 3" in out
    assert "duration=15.0" in out

    assert "applied Set frame rate to 60 fps. -> version 4" in out
    assert "fps=60" in out

    assert "applied Refine the title text. -> version 5" in out
    assert "text='Refined title'" in out

    assert "applied Fade the title to 50% opacity. -> version 6" in out
    assert "opacity=0.5" in out

    assert out.count("applied") == 5


def test_repl_undo_and_redo_walk_the_version_history(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, out, err = _run(capsys, monkeypatch, "background\nundo\nredo\nquit\n")

    assert code == 0, err
    assert err == ""

    assert "applied Set canvas background to white. -> version 2" in out
    assert "undone to version 1" in out
    assert "bg=#000000" in out
    assert "redone to version 2" in out
    assert "bg=#ffffff" in out
    assert "Goodbye." in out


def test_repl_editing_after_undo_starts_a_new_branch(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, out, err = _run(
        capsys, monkeypatch, "background\nduration\nundo\nfps\nredo\n"
    )

    assert code == 0, err
    assert err == ""

    # The duration edit is on the abandoned branch and is gone.
    assert "applied Extend duration to 15 seconds. -> version 3" in out
    assert "undone to version 2" in out
    assert "applied Set frame rate to 60 fps. -> version 4" in out
    assert "fps=60" in out
    # The new branch keeps the background from version 2 but not duration 15.
    assert "bg=#ffffff" in out
    assert "duration=10.0" in out
    # Redo cannot restore the abandoned branch.
    assert "(nothing to redo)" in out


def test_repl_unknown_message_is_ignored_without_a_new_version(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, out, err = _run(capsys, monkeypatch, "frobnicate\nBACKGROUND\nquit\n")

    assert code == 0, err
    assert err == ""
    assert "(no sample edit matches 'frobnicate')" in out
    # Case-insensitive matching still applies the edit.
    assert "applied Set canvas background to white. -> version 2" in out


def test_repl_quit_exits_cleanly(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, out, err = _run(capsys, monkeypatch, "quit\n")

    assert code == 0
    assert err == ""
    assert "Goodbye." in out


def test_repl_end_of_input_exits_cleanly(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, out, err = _run(capsys, monkeypatch, "")

    assert code == 0
    assert err == ""
    assert "AIVIA session REPL" in out


def test_repl_undo_at_the_start_reports_nothing_to_undo(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, out, err = _run(capsys, monkeypatch, "undo\nquit\n")

    assert code == 0, err
    assert err == ""
    assert "(nothing to undo)" in out


def test_repl_show_prints_the_current_composition(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, out, err = _run(capsys, monkeypatch, "show\nquit\n")

    assert code == 0, err
    assert err == ""
    assert "canvas=1280x720 bg=#000000 duration=10.0 fps=30" in out
    assert "title text='Hello' opacity=1.0" in out

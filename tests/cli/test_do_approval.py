"""``aivia do``: plan preview and approval with always, ask and never modes."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from aivia.cli import main
from aivia.core.config import PlannerApprovalMode
from aivia.core.config.planner import PlannerConfig


class _Stream(io.StringIO):
    """A stdin replacement whose terminal status the test controls."""

    def __init__(self, text: str = "", *, tty: bool) -> None:
        super().__init__(text)
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def write_config(
    tmp_path: Path,
    *,
    approval_mode: str = "ask",
    preview: bool = True,
) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
providers:
  text:
    type: mock-llm
formats:
  sample:
    description: A sample recipe for plan approval.
    stages:
      - stage: input_topic
      - stage: write
    capabilities: [llm]
    output_type: video
    handler: save
planner:
  approval_mode: {approval_mode}
  preview: {str(preview).lower()}
""".lstrip(),
        encoding="utf-8",
    )
    return path


def do_command(
    capsys: pytest.CaptureFixture[str],
    *args: str,
) -> tuple[int, str, str]:
    code = main(["do", *args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_do_preview_shows_format_and_estimate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stream("y\n", tty=True))
    config = write_config(tmp_path)

    code, out, err = do_command(capsys, "--format", "sample", "--config", str(config))

    assert code == 0, err
    assert err == ""
    assert "Format: sample" in out
    assert "STAGE" in out
    assert "TOTAL" in out


def test_do_never_mode_never_prompts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stream(tty=False))
    config = write_config(tmp_path, approval_mode="never")

    code, out, err = do_command(capsys, "--format", "sample", "--config", str(config))

    assert code == 0, err
    assert "Approve plan" not in out
    assert "plan approval declined" not in err


def test_do_always_mode_prompts_interactively(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stream("y\n", tty=True))
    config = write_config(tmp_path, approval_mode="always")

    code, out, err = do_command(capsys, "--format", "sample", "--config", str(config))

    assert code == 0, err
    assert "Approve plan sample?" in out
    assert "[y/N]" in out


def test_do_always_mode_refuses_non_interactive_without_yes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stream(tty=False))
    config = write_config(tmp_path, approval_mode="always")

    code, out, err = do_command(capsys, "--format", "sample", "--config", str(config))

    assert code == 1
    assert "non-interactive runs require --yes" in err
    assert "Approve plan" not in out


def test_do_yes_flag_skips_prompt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stream("no\n", tty=True))
    config = write_config(tmp_path, approval_mode="always")

    code, out, err = do_command(
        capsys, "--format", "sample", "--config", str(config), "--yes"
    )

    assert code == 0, err
    assert "Approve plan" not in out


def test_do_ask_mode_prompts_interactively(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stream("y\n", tty=True))
    config = write_config(tmp_path, approval_mode="ask")

    code, out, err = do_command(capsys, "--format", "sample", "--config", str(config))

    assert code == 0, err
    assert "Approve plan sample?" in out


def test_do_declined_approval_cancels(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stream("no\n", tty=True))
    config = write_config(tmp_path, approval_mode="always")

    code, _out, err = do_command(capsys, "--format", "sample", "--config", str(config))

    assert code == 1
    assert "plan approval declined" in err


def test_do_edit_flag_shows_edit_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stream("y\n", tty=True))
    config = write_config(tmp_path, approval_mode="ask")

    code, out, err = do_command(
        capsys, "--format", "sample", "--config", str(config), "--edit"
    )

    assert code == 0, err
    assert "Edit path" in out


def test_do_unknown_format_rejects(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stream(tty=False))
    config = write_config(tmp_path, approval_mode="never")

    code, _out, err = do_command(
        capsys, "--format", "nonexistent", "--config", str(config)
    )

    assert code == 1
    assert "unknown format" in err


def test_do_missing_format_rejects(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stream(tty=False))
    config = write_config(tmp_path, approval_mode="never")

    code, _out, err = do_command(capsys, "--config", str(config))

    assert code == 2
    assert "a format is required" in err


def test_planner_approval_mode_enum_values() -> None:
    assert PlannerApprovalMode.ALWAYS.value == "always"
    assert PlannerApprovalMode.ASK.value == "ask"
    assert PlannerApprovalMode.NEVER.value == "never"


def test_planner_config_defaults_are_ask_with_preview() -> None:
    planner = PlannerConfig.from_mapping({})
    assert planner.approval_mode is PlannerApprovalMode.ASK
    assert planner.preview is True

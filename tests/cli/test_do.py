"""``aivia do``: resolve a format label, select an output and print a breakdown."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from aivia.cli import main
from aivia.core.config.planner import PlannerApprovalMode


class _Stream(io.StringIO):
    """A stdin replacement whose terminal status the test controls."""

    def __init__(self, text: str = "", *, tty: bool) -> None:
        super().__init__(text)
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def write_config(tmp_path: Path, *, approval_mode: str = "ask") -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
providers:
  text:
    type: mock-llm
formats:
  sample:
    description: A sample recipe for the do command.
    stages:
      - stage: input_topic
      - stage: write
    capabilities: [llm]
    output_type: video
    handler: save
planner:
  approval_mode: {approval_mode}
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


def test_do_resolves_a_format_by_label_and_prints_a_cost_breakdown(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stream("y\n", tty=True))
    config = write_config(tmp_path, approval_mode="never")

    code, out, err = do_command(capsys, "--format", "sample", "--config", str(config))

    assert code == 0, err
    assert err == ""
    assert "Format: sample" in out
    assert "STAGE" in out
    assert "TOTAL" in out


def test_do_writes_the_breakdown_to_the_selected_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stream(tty=False))
    config = write_config(tmp_path, approval_mode="never")
    output = tmp_path / "breakdown.txt"

    code, out, err = do_command(
        capsys,
        "--format",
        "sample",
        "--output",
        str(output),
        "--config",
        str(config),
    )

    assert code == 0, err
    assert err == ""
    assert output.is_file()
    contents = output.read_text(encoding="utf-8")
    assert "Format: sample" in contents
    assert "STAGE" in contents
    assert "TOTAL" in contents
    assert f"Output: {output}" in out


def test_do_never_mode_with_output_writes_without_prompting(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stream(tty=False))
    config = write_config(tmp_path, approval_mode="never")
    output = tmp_path / "breakdown.txt"

    code, out, err = do_command(
        capsys,
        "--format",
        "sample",
        "--output",
        str(output),
        "--config",
        str(config),
    )

    assert code == 0, err
    assert "Approve plan" not in out
    assert output.is_file()
    assert "TOTAL" in output.read_text(encoding="utf-8")


def test_do_yes_flag_writes_output_without_prompting(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stream("no\n", tty=True))
    config = write_config(tmp_path, approval_mode="always")
    output = tmp_path / "breakdown.txt"

    code, out, err = do_command(
        capsys,
        "--format",
        "sample",
        "--output",
        str(output),
        "--yes",
        "--config",
        str(config),
    )

    assert code == 0, err
    assert "Approve plan" not in out
    assert output.is_file()
    assert "TOTAL" in output.read_text(encoding="utf-8")


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


def test_do_approval_mode_enum_values() -> None:
    assert PlannerApprovalMode.ALWAYS.value == "always"
    assert PlannerApprovalMode.ASK.value == "ask"
    assert PlannerApprovalMode.NEVER.value == "never"

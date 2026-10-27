"""Makefile contract: developer targets, tools and check ordering."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
MAKE = shutil.which("make")

REQUIRED_TARGETS = {"lint", "type", "test", "check", "docs", "install-smoke"}


def _prerequisites(text: str) -> dict[str, list[str]]:
    targets: dict[str, list[str]] = {}
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z0-9_-]+):(.*)$", line)
        if match:
            targets[match.group(1)] = match.group(2).split()
    return targets


def _phony_targets(text: str) -> set[str]:
    for line in text.splitlines():
        if line.startswith(".PHONY:"):
            return set(line.split(":", 1)[1].split())
    return set()


def _write_fake_uv(bin_dir: Path) -> None:
    script = bin_dir / "uv"
    script.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$*" >> "$UV_LOG"\n'
        'case "$UV_LOG" in\n'
        "  *fail*) exit 1 ;;\n"
        "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    script.chmod(0o755)


def _make_env(tmp_path: Path, *, fail: bool = False) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_uv(bin_dir)
    log = tmp_path / ("uv-fail.log" if fail else "uv.log")
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["UV_LOG"] = str(log)
    return env, log


def _run_make(
    tmp_path: Path, args: list[str], env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    assert MAKE is not None
    return subprocess.run(
        [MAKE, "-f", str(MAKEFILE), *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_required_targets_exist_and_are_phony() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")

    assert set(_prerequisites(text)) >= REQUIRED_TARGETS
    assert _phony_targets(text) >= REQUIRED_TARGETS


def test_check_depends_on_lint_type_and_test_in_that_order() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")

    assert _prerequisites(text)["check"] == ["lint", "type", "test"]


def test_targets_use_the_project_tools() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")

    assert "uv run ruff check ." in text
    assert "uv run mypy" in text
    assert "uv run pytest -q" in text
    assert "scripts/install_smoke.sh" in text


@pytest.mark.skipif(MAKE is None, reason="make is not installed")
def test_make_check_runs_lint_type_and_test_in_order(tmp_path: Path) -> None:
    env, log = _make_env(tmp_path)

    result = _run_make(tmp_path, ["check"], env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert log.read_text(encoding="utf-8").splitlines() == [
        "run ruff check .",
        "run mypy",
        "run pytest -q",
    ]


@pytest.mark.skipif(MAKE is None, reason="make is not installed")
def test_make_check_stops_when_lint_fails(tmp_path: Path) -> None:
    env, log = _make_env(tmp_path, fail=True)

    result = _run_make(tmp_path, ["check"], env)

    assert result.returncode != 0
    assert log.read_text(encoding="utf-8").splitlines() == ["run ruff check ."]


@pytest.mark.skipif(MAKE is None, reason="make is not installed")
def test_make_docs_checks_the_documentation_files(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "DESIGN.md").write_text("design", encoding="utf-8")
    (tmp_path / "docs" / "decisions.md").write_text("decisions", encoding="utf-8")
    (tmp_path / "docs" / "PROGRESS.md").write_text("progress", encoding="utf-8")

    result = _run_make(tmp_path, ["docs"], os.environ.copy())

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(MAKE is None, reason="make is not installed")
def test_make_docs_fails_when_a_document_is_missing(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "DESIGN.md").write_text("design", encoding="utf-8")

    result = _run_make(tmp_path, ["docs"], os.environ.copy())

    assert result.returncode != 0


@pytest.mark.skipif(MAKE is None, reason="make is not installed")
def test_make_install_smoke_runs_the_script_when_present(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    marker = tmp_path / "installed.txt"
    script = scripts / "install_smoke.sh"
    script.write_text(
        f"#!/bin/sh\nprintf installed > {marker}\n",
        encoding="utf-8",
    )
    script.chmod(0o755)

    result = _run_make(tmp_path, ["install-smoke"], os.environ.copy())

    assert result.returncode == 0, result.stdout + result.stderr
    assert marker.read_text(encoding="utf-8") == "installed"


@pytest.mark.skipif(MAKE is None, reason="make is not installed")
def test_make_install_smoke_falls_back_to_the_packaging_test(tmp_path: Path) -> None:
    env, log = _make_env(tmp_path)

    result = _run_make(tmp_path, ["install-smoke"], env)

    assert result.returncode == 0, result.stdout + result.stderr
    expected = (
        "run pytest -q tests/test_packaging.py "
        "-k installed_console_script or module_entry_point"
    )
    assert log.read_text(encoding="utf-8").splitlines() == [expected]

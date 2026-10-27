"""Toolchain contract: lint, type and test settings, with strict markers."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "pyproject.toml"


def _run(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)


def _tool_config() -> dict[str, Any]:
    config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    return cast(dict[str, Any], config["tool"])


def _run_pytest(test_file: Path) -> subprocess.CompletedProcess[str]:
    return _run(
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "-c",
        str(CONFIG),
        str(test_file),
    )


def test_pyproject_configures_ruff() -> None:
    ruff = _tool_config()["ruff"]
    assert ruff["target-version"] == "py312"
    assert ruff["line-length"] == 88
    assert {"src", "tests"} <= set(ruff["src"])
    assert {"E", "F", "I"} <= set(ruff["lint"]["select"])


def test_ruff_enforces_the_configured_rules(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text("import os\n", encoding="utf-8")

    result = _run(
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--no-cache",
        "--config",
        str(CONFIG),
        str(sample),
    )

    assert result.returncode != 0, result.stdout
    assert "F401" in result.stdout


def test_pyproject_configures_mypy() -> None:
    mypy = _tool_config()["mypy"]
    assert mypy["python_version"] == "3.12"
    assert mypy["strict"] is True
    assert set(mypy["files"]) == {"src", "tests"}


def test_mypy_enforces_strict_typing(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text("def identity(value):\n    return value\n", encoding="utf-8")

    result = _run(
        sys.executable,
        "-m",
        "mypy",
        "--config-file",
        str(CONFIG),
        "--cache-dir",
        str(tmp_path / "mypy_cache"),
        str(sample),
    )

    assert result.returncode != 0, result.stdout
    assert "missing a type annotation" in result.stdout


def test_pytest_has_strict_markers_and_the_project_markers() -> None:
    ini = _tool_config()["pytest"]["ini_options"]
    assert "--strict-markers" in ini["addopts"].split()

    declared = {marker.split(":", 1)[0] for marker in ini["markers"]}
    assert declared == {"live", "slow", "golden"}


def test_declared_markers_are_accepted(tmp_path: Path) -> None:
    sample = tmp_path / "test_markers.py"
    sample.write_text(
        "\n".join(
            [
                "import pytest",
                "",
                "@pytest.mark.live",
                "def test_live():",
                "    pass",
                "",
                "@pytest.mark.slow",
                "def test_slow():",
                "    pass",
                "",
                "@pytest.mark.golden",
                "def test_golden():",
                "    pass",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = _run_pytest(sample)

    assert result.returncode == 0, result.stdout + result.stderr


def test_unknown_markers_fail_the_run(tmp_path: Path) -> None:
    sample = tmp_path / "test_unknown_marker.py"
    sample.write_text(
        "\n".join(
            [
                "import pytest",
                "",
                "@pytest.mark.not_a_real_marker",
                "def test_unknown():",
                "    pass",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = _run_pytest(sample)

    assert result.returncode != 0
    assert "not_a_real_marker" in result.stdout + result.stderr

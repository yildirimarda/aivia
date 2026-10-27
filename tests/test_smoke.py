"""Toolchain contract: lint, type and test settings, with strict markers."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, cast

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "pyproject.toml"
LIVE_BUDGET_ENV = "AIVA_LIVE_BUDGET_USD"


def _run(
    *args: str, cwd: Path = ROOT, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, cwd=cwd, env=env, capture_output=True, text=True, check=False
    )


def _tool_config() -> dict[str, Any]:
    config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    return cast(dict[str, Any], config["tool"])


def _run_pytest(
    test_file: Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    run_env = os.environ.copy()
    run_env.pop(LIVE_BUDGET_ENV, None)
    existing_pythonpath = run_env.get("PYTHONPATH")
    run_env["PYTHONPATH"] = (
        str(ROOT)
        if not existing_pythonpath
        else f"{ROOT}{os.pathsep}{existing_pythonpath}"
    )
    if env is not None:
        run_env.update(env)
    return _run(
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "-p",
        "conftest",
        "-c",
        str(CONFIG),
        *args,
        str(test_file),
        env=run_env,
    )


def _write_live_test(tmp_path: Path) -> Path:
    sample = tmp_path / "test_live_sample.py"
    sample.write_text(
        "\n".join(
            [
                "from decimal import Decimal",
                "",
                "import pytest",
                "",
                "@pytest.mark.live",
                "def test_live(live_budget_usd):",
                "    assert live_budget_usd == Decimal('1.25')",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return sample


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


def test_pytest_exposes_the_live_option() -> None:
    result = _run(sys.executable, "-m", "pytest", "--help", "-c", str(CONFIG))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "--live" in result.stdout


def test_live_tests_are_skipped_unless_enabled(tmp_path: Path) -> None:
    sample = _write_live_test(tmp_path)

    result = _run_pytest(sample)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 skipped" in result.stdout
    assert "live tests require --live" in result.stdout


def test_live_tests_run_when_enabled_with_budget(tmp_path: Path) -> None:
    sample = _write_live_test(tmp_path)

    result = _run_pytest(sample, "--live", env={LIVE_BUDGET_ENV: "1.25"})

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout


@pytest.mark.parametrize("budget", ["", "0", "-1", "not-a-number", "NaN"])
def test_live_requires_a_positive_budget(tmp_path: Path, budget: str) -> None:
    sample = _write_live_test(tmp_path)

    result = _run_pytest(sample, "--live", env={LIVE_BUDGET_ENV: budget})

    assert result.returncode != 0, result.stdout + result.stderr
    assert LIVE_BUDGET_ENV in result.stdout + result.stderr

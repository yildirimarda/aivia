"""Packaging contract: one top-level package, package data and a console script."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from aivia import __version__

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out_dir = tmp_path_factory.mktemp("wheel")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(out_dir),
            str(ROOT),
        ],
        check=True,
    )
    wheels = sorted(out_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel, found {wheels}"
    return wheels[0]


def test_wheel_has_exactly_one_top_level_package(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()

    top_level = {name.split("/", 1)[0] for name in names if "/" in name}
    packages = {
        name for name in top_level if not name.endswith((".dist-info", ".data"))
    }
    assert packages == {"aivia"}


def test_wheel_contains_package_data(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())

    assert "aivia/py.typed" in names
    assert "aivia/data/defaults/README.txt" in names


def test_wheel_declares_console_script(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        entry_points = next(
            name
            for name in archive.namelist()
            if name.endswith(".dist-info/entry_points.txt")
        )
        contents = archive.read(entry_points).decode()

    assert "[console_scripts]" in contents
    assert "aivia = aivia.cli:main" in contents


def test_installed_console_script_runs_from_any_directory(tmp_path: Path) -> None:
    script = Path(sys.executable).with_name("aivia")
    assert script.is_file(), "the aivia console script is not installed"

    result = subprocess.run(
        [str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"aivia {__version__}"


def test_module_entry_point_runs_from_any_directory(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "aivia"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"aivia {__version__}"

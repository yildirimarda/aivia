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


def test_wheel_contains_bundled_ofl_font_family_and_license(wheel: Path) -> None:
    font_dir = "aivia/assets/fonts/lato/"
    font_files = [
        f"{font_dir}Lato-Regular.ttf",
        f"{font_dir}Lato-Bold.ttf",
        f"{font_dir}Lato-Italic.ttf",
        f"{font_dir}Lato-BoldItalic.ttf",
    ]

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())

        assert set(font_files) <= names
        assert f"{font_dir}OFL.txt" in names

        for font_file in font_files:
            assert archive.read(font_file)[:4] == b"\x00\x01\x00\x00"

        license_text = archive.read(f"{font_dir}OFL.txt").decode()
        assert "SIL OPEN FONT LICENSE Version 1.1" in license_text
        assert 'Reserved Font Name "Lato"' in license_text


def test_wheel_contains_default_config_prompts_and_templates(wheel: Path) -> None:
    expected = {
        "aivia/data/defaults/config.example.yaml",
        "aivia/data/defaults/prompts/image.md",
        "aivia/data/defaults/templates/web_card.html",
    }

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert expected <= names

        config = archive.read("aivia/data/defaults/config.example.yaml").decode()
        assert "schema_version: 1" in config
        assert "fonts_dir: package" in config


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

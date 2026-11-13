"""README contract: prerequisites, installs, init, CI walkthrough, no publishing."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def _read() -> str:
    return README.read_text(encoding="utf-8")


def test_readme_exists_and_is_non_empty() -> None:
    assert README.is_file(), "missing README.md"
    assert _read().strip(), "README.md is empty"


def test_readme_records_prerequisites() -> None:
    text = _read().lower()
    assert "python 3.12" in text
    assert "uv" in text
    assert "ffmpeg" in text
    assert "ffprobe" in text


def test_readme_records_uv_tool_install_path() -> None:
    text = _read()
    assert "uv tool install ." in text
    assert "uv tool install --force ." in text


def test_readme_records_editable_install_path() -> None:
    text = _read()
    assert "uv pip install -e ." in text
    assert "pip install -e ." in text


def test_readme_records_first_run_init() -> None:
    text = _read()
    assert "aivia init" in text
    assert "--force" in text
    assert "--dry-run" in text
    assert "--config" in text
    assert "xdg_config_home" in text.lower()


def test_readme_records_ci_verified_walkthrough() -> None:
    text = _read()
    assert "make check" in text
    assert "uv run ruff check ." in text
    assert "uv run mypy" in text
    assert "uv run pytest -q" in text


def test_readme_records_no_publishing_path() -> None:
    text = _read().lower()
    assert "no pypi publishing path" in text
    assert "do not publish to pypi" in text

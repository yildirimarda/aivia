"""``aivia init`` contract: create the skeleton, print paths and guard edits."""

from __future__ import annotations

from pathlib import Path

import pytest

from aivia.cli import main
from aivia.core import paths


def packaged_config() -> str:
    return (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )


def test_init_creates_the_config_skeleton_and_prints_resolved_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.chdir(tmp_path)

    assert main(["init"]) == 0

    config_dir = xdg / "aivia"
    config = config_dir / "config.yaml"
    assert config.read_text(encoding="utf-8") == packaged_config()
    assert (config_dir / "prompts" / "image.md").is_file()
    assert (config_dir / "templates" / "web_card.html").is_file()
    assert (config_dir / "data").is_dir()

    output = capsys.readouterr().out
    assert f"Config directory: {config_dir}" in output
    assert f"Config file: {config}" in output
    assert f"Data directory: {config_dir / 'data'}" in output
    assert f"Defaults directory: {paths.package_defaults_dir()}" in output
    assert f"Created: {config}" in output


def test_init_refuses_to_overwrite_without_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    config_dir = xdg / "aivia"
    config_dir.mkdir(parents=True)
    config = config_dir / "config.yaml"
    config.write_text("user: edited\n", encoding="utf-8")

    assert main(["init"]) == 1

    captured = capsys.readouterr()
    assert "already exists" in captured.err
    assert "--force" in captured.err
    assert config.read_text(encoding="utf-8") == "user: edited\n"
    assert not (config_dir / "prompts").exists()


def test_init_dry_run_prints_paths_and_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))

    assert main(["init", "--dry-run"]) == 0

    config_dir = xdg / "aivia"
    assert not config_dir.exists()
    output = capsys.readouterr().out
    assert "Dry run: no files written." in output
    assert f"Would create: {config_dir / 'config.yaml'}" in output


def test_init_dry_run_with_force_reports_overwrite_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    config_dir = xdg / "aivia"
    config_dir.mkdir(parents=True)
    config = config_dir / "config.yaml"
    config.write_text("user: edited\n", encoding="utf-8")

    assert main(["init", "--dry-run", "--force"]) == 0

    assert config.read_text(encoding="utf-8") == "user: edited\n"
    output = capsys.readouterr().out
    assert f"Would overwrite: {config}" in output
    assert f"Would preserve: {config}" not in output


def test_init_force_overwrites_the_config_but_preserves_other_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    config_dir = xdg / "aivia"
    (config_dir / "prompts").mkdir(parents=True)
    config = config_dir / "config.yaml"
    config.write_text("user: edited\n", encoding="utf-8")
    prompt = config_dir / "prompts" / "image.md"
    prompt.write_text("my prompt\n", encoding="utf-8")

    assert main(["init", "--force"]) == 0

    assert config.read_text(encoding="utf-8") == packaged_config()
    assert prompt.read_text(encoding="utf-8") == "my prompt\n"
    output = capsys.readouterr().out
    assert f"Overwrote: {config}" in output
    assert f"Preserved: {prompt}" in output


def test_init_force_on_a_fresh_install_reports_a_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))

    assert main(["init", "--force"]) == 0

    output = capsys.readouterr().out
    assert "Overwrote:" not in output
    assert f"Created: {xdg / 'aivia' / 'config.yaml'}" in output


def test_init_honours_an_explicit_config_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "custom" / "config.yaml"

    assert main(["init", "--config", str(config)]) == 0

    assert config.read_text(encoding="utf-8") == packaged_config()
    assert (config.parent / "data").is_dir()
    output = capsys.readouterr().out
    assert f"Config file: {config.resolve()}" in output
    assert f"Data directory: {config.parent.resolve() / 'data'}" in output

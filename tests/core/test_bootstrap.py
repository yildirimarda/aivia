"""Bootstrap contract: copy defaults once, preserve edits and add upgrades."""

from __future__ import annotations

from pathlib import Path

import pytest

from aivia.core.bootstrap import BootstrapResult, bootstrap

CONFIG_EXAMPLE = "schema_version: 1\npipeline:\n  timezone: Europe/Berlin\n"
PROMPT = "Write a concise image prompt.\n"
TEMPLATE = "<!doctype html><title>Card</title>\n"


def make_defaults(root: Path) -> Path:
    defaults = root / "defaults"
    (defaults / "prompts").mkdir(parents=True)
    (defaults / "templates").mkdir(parents=True)
    (defaults / "config.example.yaml").write_text(CONFIG_EXAMPLE, encoding="utf-8")
    (defaults / "prompts" / "image.md").write_text(PROMPT, encoding="utf-8")
    (defaults / "templates" / "web_card.html").write_text(TEMPLATE, encoding="utf-8")
    (defaults / "README.txt").write_text("package internals\n", encoding="utf-8")
    return defaults


def test_first_run_copies_defaults_and_renames_the_example_config(
    tmp_path: Path,
) -> None:
    defaults = make_defaults(tmp_path)
    config = tmp_path / "config"

    result = bootstrap(config_dir=config, defaults_dir=defaults)

    assert isinstance(result, BootstrapResult)
    assert result.first_run is True
    assert result.changed is True
    assert result.config_dir == config.resolve()
    assert result.data_dir == config.resolve() / "data"
    assert result.data_dir.is_dir()

    assert (config / "config.yaml").read_text(encoding="utf-8") == CONFIG_EXAMPLE
    assert (config / "prompts" / "image.md").read_text(encoding="utf-8") == PROMPT
    assert (config / "templates" / "web_card.html").read_text(
        encoding="utf-8"
    ) == TEMPLATE
    assert not (config / "config.example.yaml").exists()
    assert not (config / "README.txt").exists()
    assert result.created == (
        config.resolve() / "config.yaml",
        config.resolve() / "prompts" / "image.md",
        config.resolve() / "templates" / "web_card.html",
    )
    assert result.preserved == ()


def test_second_run_is_idempotent_and_preserves_every_file(tmp_path: Path) -> None:
    defaults = make_defaults(tmp_path)
    config = tmp_path / "config"
    first = bootstrap(config_dir=config, defaults_dir=defaults)

    second = bootstrap(config_dir=config, defaults_dir=defaults)

    assert second.first_run is False
    assert second.changed is False
    assert second.created == ()
    assert second.preserved == first.created
    assert (config / "config.yaml").read_text(encoding="utf-8") == CONFIG_EXAMPLE


def test_edited_files_are_never_overwritten(tmp_path: Path) -> None:
    defaults = make_defaults(tmp_path)
    config = tmp_path / "config"
    bootstrap(config_dir=config, defaults_dir=defaults)
    edited = "schema_version: 1\nuser: edited\n"
    (config / "config.yaml").write_text(edited, encoding="utf-8")
    (config / "prompts" / "image.md").write_text("my prompt\n", encoding="utf-8")
    (defaults / "config.example.yaml").write_text("new packaged\n", encoding="utf-8")
    (defaults / "prompts" / "image.md").write_text("new prompt\n", encoding="utf-8")

    result = bootstrap(config_dir=config, defaults_dir=defaults)

    assert (config / "config.yaml").read_text(encoding="utf-8") == edited
    assert (config / "prompts" / "image.md").read_text(
        encoding="utf-8"
    ) == "my prompt\n"
    assert result.created == ()
    assert result.preserved == (
        config.resolve() / "config.yaml",
        config.resolve() / "prompts" / "image.md",
        config.resolve() / "templates" / "web_card.html",
    )


def test_upgrade_adds_new_defaults_without_touching_existing_files(
    tmp_path: Path,
) -> None:
    defaults = make_defaults(tmp_path)
    config = tmp_path / "config"
    bootstrap(config_dir=config, defaults_dir=defaults)
    (config / "config.yaml").write_text("edited\n", encoding="utf-8")
    (defaults / "prompts" / "video.md").write_text("video prompt\n", encoding="utf-8")
    (defaults / "styles").mkdir()
    (defaults / "styles" / "plain.yaml").write_text("tone: plain\n", encoding="utf-8")

    result = bootstrap(config_dir=config, defaults_dir=defaults)

    assert (config / "config.yaml").read_text(encoding="utf-8") == "edited\n"
    assert (config / "prompts" / "video.md").read_text(
        encoding="utf-8"
    ) == "video prompt\n"
    assert (config / "styles" / "plain.yaml").read_text(
        encoding="utf-8"
    ) == "tone: plain\n"
    assert result.created == (
        config.resolve() / "prompts" / "video.md",
        config.resolve() / "styles" / "plain.yaml",
    )
    assert result.changed is True


def test_existing_symlink_is_preserved_even_when_its_target_is_missing(
    tmp_path: Path,
) -> None:
    defaults = make_defaults(tmp_path)
    config = tmp_path / "config"
    config.mkdir()
    link = config / "config.yaml"
    link.symlink_to(tmp_path / "missing.yaml")

    result = bootstrap(config_dir=config, defaults_dir=defaults)

    assert link.is_symlink()
    assert link.readlink() == tmp_path / "missing.yaml"
    assert link in result.preserved
    assert link not in result.created


def test_bootstrap_uses_the_xdg_config_directory_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    defaults = make_defaults(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.chdir(tmp_path)

    result = bootstrap(defaults_dir=defaults)

    config = tmp_path / "xdg" / "aivia"
    assert result.config_dir == config
    assert result.data_dir == config / "data"
    assert (config / "config.yaml").is_file()
    assert (config / "data").is_dir()


def test_an_explicit_data_directory_is_created_independently(tmp_path: Path) -> None:
    defaults = make_defaults(tmp_path)
    data = tmp_path / "elsewhere" / "data"

    result = bootstrap(
        config_dir=tmp_path / "config",
        data_dir=data,
        defaults_dir=defaults,
    )

    assert result.data_dir == data.resolve()
    assert data.is_dir()


def test_a_missing_defaults_directory_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="defaults directory is missing"):
        bootstrap(config_dir=tmp_path / "config", defaults_dir=tmp_path / "missing")

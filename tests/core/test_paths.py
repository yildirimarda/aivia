"""Path contract: user config/data locations and packaged resources."""

from __future__ import annotations

from pathlib import Path

import pytest

from aivia.core.paths import (
    DEFAULT_CONFIG_FILENAME,
    Paths,
    config_dir,
    data_dir,
    default_config_file,
    package_assets_dir,
    package_defaults_dir,
    package_fonts_dir,
    package_resource,
    package_root,
    resolve_config_file,
    resolve_paths,
)


def test_config_dir_uses_xdg_config_home(tmp_path: Path) -> None:
    paths = config_dir({"XDG_CONFIG_HOME": str(tmp_path)})

    assert paths == tmp_path / "aivia"


def test_config_dir_expands_a_tilde_in_xdg_config_home(tmp_path: Path) -> None:
    paths = config_dir({"XDG_CONFIG_HOME": "~/" + tmp_path.name})

    assert paths == Path.home() / tmp_path.name / "aivia"


@pytest.mark.parametrize("value", ["", "   ", "relative/config"])
def test_config_dir_ignores_an_empty_or_relative_xdg_config_home(
    value: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    assert config_dir({"XDG_CONFIG_HOME": value}) == tmp_path / ".config" / "aivia"


def test_config_dir_falls_back_to_the_home_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    assert config_dir({}) == tmp_path / ".config" / "aivia"


def test_data_dir_is_rooted_beside_the_config(tmp_path: Path) -> None:
    environ = {"XDG_CONFIG_HOME": str(tmp_path)}

    assert data_dir(environ) == tmp_path / "aivia" / "data"


def test_default_config_file_uses_the_config_directory(tmp_path: Path) -> None:
    environ = {"XDG_CONFIG_HOME": str(tmp_path)}

    assert default_config_file(environ) == tmp_path / "aivia" / DEFAULT_CONFIG_FILENAME


def test_resolve_config_file_returns_the_default(tmp_path: Path) -> None:
    environ = {"XDG_CONFIG_HOME": str(tmp_path)}

    assert resolve_config_file(environ=environ) == (
        tmp_path / "aivia" / DEFAULT_CONFIG_FILENAME
    )


def test_explicit_config_flag_takes_precedence_over_the_config_directory(
    tmp_path: Path,
) -> None:
    config = tmp_path / "elsewhere" / "custom.yaml"
    environ = {"XDG_CONFIG_HOME": str(tmp_path / "xdg")}

    assert resolve_config_file(config, environ) == config.resolve()
    assert resolve_config_file(str(config), environ) == config.resolve()


def test_explicit_config_flag_expands_the_user_directory() -> None:
    resolved = resolve_config_file("~/aivia/custom.yaml", {})

    assert resolved == (Path.home() / "aivia" / "custom.yaml").resolve()


def test_explicit_relative_config_flag_is_resolved_from_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    assert (
        resolve_config_file("custom.yaml", {}) == (tmp_path / "custom.yaml").resolve()
    )


def test_resolve_paths_uses_the_default_config_and_data_locations(
    tmp_path: Path,
) -> None:
    environ = {"XDG_CONFIG_HOME": str(tmp_path)}

    paths = resolve_paths(environ=environ)

    assert isinstance(paths, Paths)
    assert paths.config_file == tmp_path / "aivia" / DEFAULT_CONFIG_FILENAME
    assert paths.config_dir == tmp_path / "aivia"
    assert paths.data_dir == tmp_path / "aivia" / "data"


def test_resolve_paths_follows_an_explicit_config_flag(tmp_path: Path) -> None:
    config = tmp_path / "project" / "aivia.yaml"
    environ = {"XDG_CONFIG_HOME": str(tmp_path / "xdg")}

    paths = resolve_paths(config, environ)

    assert paths.config_file == config.resolve()
    assert paths.config_dir == config.parent.resolve()
    assert paths.data_dir == config.parent.resolve() / "data"


def test_package_resources_resolve_from_any_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    defaults = package_defaults_dir()

    assert defaults.is_absolute()
    assert (defaults / "config.example.yaml").is_file()
    assert (defaults / "README.txt").is_file()
    assert package_resource("data", "defaults", "README.txt") == (
        defaults / "README.txt"
    )


def test_package_assets_include_the_bundled_fonts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    assert package_assets_dir() == package_root() / "assets"
    assert (package_fonts_dir() / "lato" / "OFL.txt").is_file()


def test_resolve_paths_includes_package_resources(tmp_path: Path) -> None:
    paths = resolve_paths(environ={"XDG_CONFIG_HOME": str(tmp_path)})

    assert paths.defaults_dir == package_defaults_dir()
    assert paths.fonts_dir == package_fonts_dir()
    assert paths.assets_dir == package_assets_dir()

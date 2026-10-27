"""Resolve user config/data locations and packaged runtime resources.

User paths follow the XDG base directory rules and never depend on the current
working directory. Packaged resources are looked up through the import system so
they resolve from an installed wheel just as well as from a source checkout.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

__all__ = [
    "APP_NAME",
    "CONFIG_DIR_NAME",
    "DATA_DIR_NAME",
    "DEFAULT_CONFIG_FILENAME",
    "ENV_PREFIX",
    "Paths",
    "config_dir",
    "data_dir",
    "default_config_file",
    "default_config_path",
    "package_assets_dir",
    "package_data_dir",
    "package_defaults_dir",
    "package_fonts_dir",
    "package_resource",
    "package_root",
    "resolve_config_file",
    "resolve_paths",
    "user_config_dir",
    "user_data_dir",
]

APP_NAME = "aivia"
CONFIG_DIR_NAME = APP_NAME
DATA_DIR_NAME = "data"
DEFAULT_CONFIG_FILENAME = "config.yaml"
ENV_PREFIX = "AIVA_"


@dataclass(frozen=True, slots=True)
class Paths:
    """The resolved user and package paths for one invocation."""

    config_file: Path
    data_dir: Path
    defaults_dir: Path
    fonts_dir: Path

    @property
    def config_dir(self) -> Path:
        """The directory containing the selected config file."""
        return self.config_file.parent

    @property
    def assets_dir(self) -> Path:
        """The directory containing packaged runtime assets."""
        return self.fonts_dir.parent


def _environ(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def config_dir(environ: Mapping[str, str] | None = None) -> Path:
    """Return the user config directory without consulting the CWD.

    ``XDG_CONFIG_HOME`` is honoured when it is an absolute path; otherwise the
    XDG fallback ``~/.config`` is used. An empty or relative value is ignored,
    as required by the XDG base directory specification.
    """
    env = _environ(environ)
    configured = env.get("XDG_CONFIG_HOME", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_absolute():
            return candidate / CONFIG_DIR_NAME
    return Path.home() / ".config" / CONFIG_DIR_NAME


def data_dir(environ: Mapping[str, str] | None = None) -> Path:
    """Return the user data directory, rooted beside the default config."""
    return config_dir(environ) / DATA_DIR_NAME


def default_config_file(environ: Mapping[str, str] | None = None) -> Path:
    """Return the config file used when no explicit config is supplied."""
    return config_dir(environ) / DEFAULT_CONFIG_FILENAME


def resolve_config_file(
    config: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return the config file, giving an explicit config flag precedence.

    An explicit path is expanded and made absolute relative to the current
    directory. Without one, the default lives in :func:`config_dir`.
    """
    if config is not None:
        return Path(config).expanduser().resolve()
    return default_config_file(environ)


def resolve_paths(
    config: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Paths:
    """Resolve every user and package path for an invocation."""
    config_file = resolve_config_file(config, environ)
    return Paths(
        config_file=config_file,
        data_dir=config_file.parent / DATA_DIR_NAME,
        defaults_dir=package_defaults_dir(),
        fonts_dir=package_fonts_dir(),
    )


def package_root() -> Path:
    """Return the root directory of the installed ``aivia`` package."""
    return Path(str(resources.files(APP_NAME)))


def package_resource(*parts: str) -> Path:
    """Return a path inside the package, independent of the working directory."""
    return package_root().joinpath(*parts)


def package_assets_dir() -> Path:
    """Return the packaged runtime assets directory."""
    return package_resource("assets")


def package_defaults_dir() -> Path:
    """Return the packaged default config, prompt and template directory."""
    return package_resource("data", "defaults")


def package_fonts_dir() -> Path:
    """Return the packaged font directory."""
    return package_resource("assets", "fonts")


user_config_dir = config_dir
user_data_dir = data_dir
default_config_path = default_config_file
package_data_dir = package_defaults_dir

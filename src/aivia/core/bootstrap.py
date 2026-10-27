"""First-run bootstrap of the user config directory from packaged defaults.

The packaged defaults are the source of truth for a fresh installation.  On the
first run they are copied into the user config directory, with the example
config renamed to the active ``config.yaml``.  Later runs only add defaults that
are still missing, so a file the user edited is never overwritten and an upgrade
can introduce new defaults without touching existing ones.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from aivia.core import paths

__all__ = [
    "DEFAULT_CONFIG_SOURCE",
    "BootstrapResult",
    "bootstrap",
    "bootstrap_user_config",
]

DEFAULT_CONFIG_SOURCE = "config.example.yaml"

_INTERNAL_DEFAULTS = frozenset({"README.txt"})


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """What one bootstrap run created and what it left untouched."""

    config_dir: Path
    data_dir: Path
    created: tuple[Path, ...]
    preserved: tuple[Path, ...]

    @property
    def first_run(self) -> bool:
        """Whether this run populated the config directory for the first time."""
        return bool(self.created)

    @property
    def changed(self) -> bool:
        """Whether any default was added by this run."""
        return bool(self.created)


def _destination_for(relative: Path) -> Path | None:
    """Map a packaged default to its user path, or skip internal files."""
    if relative == Path(DEFAULT_CONFIG_SOURCE):
        return Path(paths.DEFAULT_CONFIG_FILENAME)
    if relative.parent == Path() and relative.name in _INTERNAL_DEFAULTS:
        return None
    return relative


def _copy_if_missing(source: Path, destination: Path) -> bool:
    """Copy *source* to *destination* unless something already lives there."""
    if os.path.lexists(destination):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
    except FileExistsError:
        return False
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return True


def bootstrap(
    *,
    config_dir: Path | None = None,
    data_dir: Path | None = None,
    defaults_dir: Path | None = None,
    environ: Mapping[str, str] | None = None,
    dry_run: bool = False,
) -> BootstrapResult:
    """Populate the user config and data directories from packaged defaults.

    Existing files are preserved regardless of their contents.  New files in
    *defaults_dir* are copied on every run, which is what makes an upgrade add
    defaults without rewriting the user's edits.  The example config is copied
    as the active config file rather than under its packaged name.

    With *dry_run* nothing is written or created; the result reports what a
    real run would create and preserve.
    """
    if config_dir is None:
        target_config = paths.config_dir(environ)
    else:
        target_config = Path(config_dir).expanduser().resolve()

    if data_dir is None:
        if config_dir is None:
            target_data = paths.data_dir(environ)
        else:
            target_data = target_config / paths.DATA_DIR_NAME
    else:
        target_data = Path(data_dir).expanduser().resolve()

    source_dir = (
        paths.package_defaults_dir()
        if defaults_dir is None
        else Path(defaults_dir).expanduser().resolve()
    )
    if not source_dir.is_dir():
        raise FileNotFoundError(f"packaged defaults directory is missing: {source_dir}")

    if not dry_run:
        target_config.mkdir(parents=True, exist_ok=True)
        target_data.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    preserved: list[Path] = []
    for source in sorted(source_dir.rglob("*"), key=lambda path: path.as_posix()):
        if not source.is_file():
            continue
        relative = source.relative_to(source_dir)
        destination_relative = _destination_for(relative)
        if destination_relative is None:
            continue
        destination = target_config / destination_relative
        if dry_run:
            if os.path.lexists(destination):
                preserved.append(destination)
            else:
                created.append(destination)
        elif _copy_if_missing(source, destination):
            created.append(destination)
        else:
            preserved.append(destination)

    return BootstrapResult(
        config_dir=target_config,
        data_dir=target_data,
        created=tuple(created),
        preserved=tuple(preserved),
    )


bootstrap_user_config = bootstrap

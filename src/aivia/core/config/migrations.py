"""Versioned configuration migrations.

``schema_version`` declares which schema a user config was written for. A
config without the key is treated as version 0, the unversioned format. Each
migration upgrades exactly one version and only touches the keys it knows
about, so unknown keys, module sections and user comments survive an upgrade.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import yaml

from aivia.core.errors import ConfigError

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "MIGRATIONS",
    "ConfigMigration",
    "MigrationResult",
    "dump_migrated_config",
    "migrate_config",
    "render_migration_diff",
    "rewrite_schema_version",
    "schema_version_of",
]

CURRENT_SCHEMA_VERSION = 1
"""The schema version written by this release."""

_UNVERSIONED = 0
_SCHEMA_VERSION_LINE = re.compile(r'(?m)^["\']?schema_version["\']?[ \t]*:.*$')

MigrationFunction = Callable[[dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class ConfigMigration:
    """One single-step upgrade from ``from_version`` to ``to_version``."""

    from_version: int
    to_version: int
    description: str
    apply: MigrationFunction

    def __post_init__(self) -> None:
        if self.to_version != self.from_version + 1:
            raise ValueError(
                "a config migration must advance exactly one schema version"
            )


@dataclass(frozen=True, slots=True)
class MigrationResult:
    """The migrated values and the steps that produced them."""

    values: dict[str, Any]
    from_version: int
    to_version: int
    steps: tuple[ConfigMigration, ...]

    @property
    def changed(self) -> bool:
        """Whether at least one migration ran."""
        return bool(self.steps)


def _add_schema_version(values: dict[str, Any]) -> None:
    """Record version 1, keeping the key first when it has to be added."""
    if "schema_version" in values:
        values["schema_version"] = 1
        return
    items = [("schema_version", 1), *values.items()]
    values.clear()
    values.update(items)


MIGRATIONS: tuple[ConfigMigration, ...] = (
    ConfigMigration(
        from_version=_UNVERSIONED,
        to_version=1,
        description="add the mandatory schema_version key",
        apply=_add_schema_version,
    ),
)


def schema_version_of(values: Mapping[str, Any]) -> int:
    """Return the declared schema version, treating an absent key as 0."""
    if "schema_version" not in values:
        return _UNVERSIONED
    version = values["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise ConfigError(
            f"invalid schema_version: expected a positive integer, got {version!r}"
        )
    if version < 0:
        raise ConfigError(
            f"invalid schema_version: expected a positive integer, got {version!r}"
        )
    return version


def _find_migration(
    version: int,
    migrations: Sequence[ConfigMigration],
) -> ConfigMigration | None:
    for migration in migrations:
        if migration.from_version == version:
            return migration
    return None


def migrate_config(
    values: Mapping[str, Any],
    *,
    migrations: Sequence[ConfigMigration] | None = None,
    target_version: int = CURRENT_SCHEMA_VERSION,
) -> MigrationResult:
    """Upgrade *values* to *target_version* without dropping unknown keys."""
    available = MIGRATIONS if migrations is None else tuple(migrations)
    migrated = dict(values)
    from_version = schema_version_of(migrated)
    version = from_version
    if version > target_version:
        raise ConfigError(
            f"config schema version {version} is newer than supported "
            f"version {target_version}",
            details={"schema_version": version, "supported": target_version},
        )
    steps: list[ConfigMigration] = []
    while version < target_version:
        migration = _find_migration(version, available)
        if migration is None:
            raise ConfigError(
                "no config migration from schema_version "
                f"{version} to {target_version}",
                details={"schema_version": version, "target": target_version},
            )
        migration.apply(migrated)
        steps.append(migration)
        version = migration.to_version
    return MigrationResult(
        values=migrated,
        from_version=from_version,
        to_version=version,
        steps=tuple(steps),
    )


def _only_schema_version_changed(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> bool:
    return {key: value for key, value in before.items() if key != "schema_version"} == {
        key: value for key, value in after.items() if key != "schema_version"
    }


def rewrite_schema_version(text: str, version: int) -> str:
    """Set ``schema_version`` in *text*, preserving comments and formatting."""
    replacement = f"schema_version: {version}"
    if _SCHEMA_VERSION_LINE.search(text):
        return _SCHEMA_VERSION_LINE.sub(replacement, text, count=1)
    prefix = f"{replacement}\n"
    if not text:
        return prefix
    return prefix + text


def dump_migrated_config(
    values: Mapping[str, Any],
    *,
    before: Mapping[str, Any] | None = None,
    original_text: str | None = None,
) -> str:
    """Render the migrated config as YAML text.

    When the only change is the schema version, the original text is kept and
    just that line is inserted or replaced, so comments survive the upgrade.
    Any structural migration falls back to a deterministic YAML dump.
    """
    if original_text is not None and (
        before is None or _only_schema_version_changed(before, values)
    ):
        version = schema_version_of(values)
        candidate = rewrite_schema_version(original_text, version)
        try:
            parsed = yaml.safe_load(candidate)
        except yaml.YAMLError:
            parsed = None
        if parsed == dict(values):
            return candidate
    return yaml.safe_dump(
        dict(values),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )


def render_migration_diff(
    path: str,
    before: str,
    after: str,
    *,
    from_version: int,
    to_version: int,
) -> str:
    """Return a unified diff between two config texts."""
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"{path} (schema_version {from_version})",
            tofile=f"{path} (schema_version {to_version})",
        )
    )

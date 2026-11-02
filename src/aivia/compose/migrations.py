"""Versioned composition migrations.

``schema_version`` declares which schema a composition was written for. A
document without the key is treated as version 0, the unversioned format. Each
migration upgrades exactly one version and only touches the keys it knows
about, so unknown fields survive an upgrade. Loading always migrates an older
document to the current version before validation and refuses a document from
a newer release with a clear error instead of validating it against the wrong
schema.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aivia.compose.schema import CURRENT_COMPOSITION_SCHEMA_VERSION
from aivia.core.errors import ValidationError

__all__ = [
    "CURRENT_COMPOSITION_SCHEMA_VERSION",
    "MIGRATIONS",
    "CompositionMigration",
    "CompositionMigrationError",
    "CompositionMigrationResult",
    "MigrationResult",
    "migrate_composition",
    "schema_version_of",
]

_UNVERSIONED = 0
_VERSION_KEYS = ("schema_version", "version")

MigrationFunction = Callable[[dict[str, Any]], None]


class CompositionMigrationError(ValidationError):
    """A composition declares an unsupported or invalid schema version."""

    code = "composition_migration"


@dataclass(frozen=True, slots=True)
class CompositionMigration:
    """One single-step upgrade from ``from_version`` to ``to_version``."""

    from_version: int
    to_version: int
    description: str
    apply: MigrationFunction

    def __post_init__(self) -> None:
        if self.to_version != self.from_version + 1:
            raise ValueError(
                "a composition migration must advance exactly one schema version"
            )


@dataclass(frozen=True, slots=True)
class CompositionMigrationResult:
    """The migrated values and the steps that produced them."""

    values: dict[str, Any]
    from_version: int
    to_version: int
    steps: tuple[CompositionMigration, ...]

    @property
    def changed(self) -> bool:
        """Whether at least one migration ran."""
        return bool(self.steps)


MigrationResult = CompositionMigrationResult
"""Alias for :class:`CompositionMigrationResult`."""


def _add_schema_version(values: dict[str, Any]) -> None:
    """Record version 1, keeping the existing key name when there is one."""
    for key in _VERSION_KEYS:
        if key in values:
            values[key] = 1
            return
    items = [("schema_version", 1), *values.items()]
    values.clear()
    values.update(items)


MIGRATIONS: tuple[CompositionMigration, ...] = (
    CompositionMigration(
        from_version=_UNVERSIONED,
        to_version=1,
        description="add the mandatory schema_version key",
        apply=_add_schema_version,
    ),
)


def schema_version_of(values: Mapping[str, Any]) -> int:
    """Return the declared schema version, treating an absent key as 0."""
    key = next((name for name in _VERSION_KEYS if name in values), None)
    if key is None:
        return _UNVERSIONED
    version = values[key]
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise CompositionMigrationError(
            f"invalid schema_version: expected a non-negative integer, got {version!r}"
        )
    return version


def _find_migration(
    version: int,
    migrations: Sequence[CompositionMigration],
) -> CompositionMigration | None:
    for migration in migrations:
        if migration.from_version == version:
            return migration
    return None


def migrate_composition(
    values: Mapping[str, Any],
    *,
    migrations: Sequence[CompositionMigration] | None = None,
    target_version: int = CURRENT_COMPOSITION_SCHEMA_VERSION,
) -> CompositionMigrationResult:
    """Upgrade *values* to *target_version* without dropping unknown fields."""
    available = MIGRATIONS if migrations is None else tuple(migrations)
    migrated = dict(values)
    from_version = schema_version_of(migrated)
    version = from_version
    if version > target_version:
        raise CompositionMigrationError(
            f"composition schema version {version} is newer than supported "
            f"version {target_version}; upgrade aivia",
            details={"schema_version": version, "supported": target_version},
        )
    steps: list[CompositionMigration] = []
    while version < target_version:
        migration = _find_migration(version, available)
        if migration is None:
            raise CompositionMigrationError(
                "no composition migration from schema_version "
                f"{version} to {target_version}",
                details={"schema_version": version, "target": target_version},
            )
        migration.apply(migrated)
        steps.append(migration)
        version = migration.to_version
    return CompositionMigrationResult(
        values=migrated,
        from_version=from_version,
        to_version=version,
        steps=tuple(steps),
    )

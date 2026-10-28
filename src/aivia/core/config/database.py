"""Local database configuration: the SQLite file name and lock wait."""

from __future__ import annotations

from pydantic import Field, field_validator

from aivia.core.config.base import ConfigModel

__all__ = ["DatabaseConfig"]


class DatabaseConfig(ConfigModel):
    """The SQLite file inside the user data directory and its busy timeout.

    ``filename`` is a bare file name; the directory is always the user data
    directory so a config file cannot point the database at another location.
    ``busy_timeout_ms`` is how long SQLite waits for a lock before raising
    ``database is locked``.
    """

    filename: str = Field(default="aivia.db", min_length=1)
    busy_timeout_ms: int = Field(default=5_000, ge=0)

    @field_validator("filename")
    @classmethod
    def _validate_filename(cls, value: str) -> str:
        name = value.strip()
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError("database.filename must be a file name without a path")
        return name

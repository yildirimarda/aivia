"""Database engine and migrations for the local SQLite store."""

from aivia.core.db.engine import (
    DEFAULT_BUSY_TIMEOUT_MS,
    DEFAULT_DATABASE_FILENAME,
    configure_connection,
    create_database_engine,
    database_path,
    database_url,
    install_sqlite_pragmas,
)
from aivia.core.db.migrate import (
    MIGRATIONS_ENV,
    alembic_config,
    current_revision,
    migrations_dir,
    upgrade_database,
)
from aivia.core.db.schema import NAMING_CONVENTION, metadata
from aivia.core.db.tables import (
    CLAIM_INDEX_NAME,
    IDEMPOTENCY_INDEX_NAME,
    jobs,
    runs,
    stage_runs,
)

__all__ = [
    "CLAIM_INDEX_NAME",
    "DEFAULT_BUSY_TIMEOUT_MS",
    "DEFAULT_DATABASE_FILENAME",
    "IDEMPOTENCY_INDEX_NAME",
    "MIGRATIONS_ENV",
    "NAMING_CONVENTION",
    "alembic_config",
    "configure_connection",
    "create_database_engine",
    "current_revision",
    "database_path",
    "database_url",
    "install_sqlite_pragmas",
    "jobs",
    "metadata",
    "migrations_dir",
    "runs",
    "stage_runs",
    "upgrade_database",
]

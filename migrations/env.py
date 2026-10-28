"""Alembic environment for the local SQLite database."""

from __future__ import annotations

from alembic import context
from sqlalchemy import Connection

from aivia.core.db.engine import create_database_engine
from aivia.core.db.schema import metadata
from aivia.core.db.tables import assets as _asset_tables  # noqa: F401
from aivia.core.db.tables import aux as _aux_tables  # noqa: F401
from aivia.core.db.tables import core as _core_tables  # noqa: F401

config = context.config
target_metadata = metadata


def _run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations on the supplied connection or a new configured engine."""
    supplied = config.attributes.get("connection")
    if supplied is not None:
        _run_migrations(supplied)
        return

    engine = create_database_engine(url=config.get_main_option("sqlalchemy.url"))
    try:
        with engine.connect() as connection:
            _run_migrations(connection)
    finally:
        engine.dispose()


def run_migrations_offline() -> None:
    """Emit SQL without a live database connection."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

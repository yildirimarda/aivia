"""Give queued jobs an idempotency key that is unique per kind.

Revision ID: 0014
Revises: 0013
Create Date: 2026-11-01
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0014"
down_revision: str | None = "0013"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None

_IDEMPOTENCY_PREDICATE = sa.text("idempotency_key != ''")


def upgrade() -> None:
    """Add the key column and the unique index that guards async submissions."""
    op.add_column(
        "jobs",
        sa.Column(
            "idempotency_key",
            sa.String(length=255),
            nullable=False,
            server_default=sa.text("''"),
        ),
    )
    op.create_index(
        "uq_jobs_kind_idempotency_key",
        "jobs",
        ["kind", "idempotency_key"],
        unique=True,
        sqlite_where=_IDEMPOTENCY_PREDICATE,
        postgresql_where=_IDEMPOTENCY_PREDICATE,
    )


def downgrade() -> None:
    """Drop the unique job idempotency index and its column."""
    op.drop_index("uq_jobs_kind_idempotency_key", table_name="jobs")
    op.drop_column("jobs", "idempotency_key")

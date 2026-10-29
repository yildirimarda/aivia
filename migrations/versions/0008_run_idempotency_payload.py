"""Record the request digest behind a run idempotency key.

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-29
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    """Add the request digest used to reject a reused key with new input."""
    op.add_column(
        "runs",
        sa.Column(
            "idempotency_payload_hash",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
    )


def downgrade() -> None:
    """Remove the idempotency request digest from the runs table."""
    op.drop_column("runs", "idempotency_payload_hash")

"""Record who resumed a paused budget run, when and why.

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-28
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    """Add the resume attribution columns to the runs table."""
    op.add_column(
        "runs",
        sa.Column(
            "resumed_by",
            sa.Text(),
            nullable=False,
            server_default=sa.text("''"),
        ),
    )
    op.add_column(
        "runs",
        sa.Column(
            "resume_reason",
            sa.Text(),
            nullable=False,
            server_default=sa.text("''"),
        ),
    )
    op.add_column(
        "runs",
        sa.Column("resumed_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    """Remove the resume attribution columns from the runs table."""
    op.drop_column("runs", "resumed_at")
    op.drop_column("runs", "resume_reason")
    op.drop_column("runs", "resumed_by")

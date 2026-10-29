"""Record the config environment selected when a run was created.

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-29
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    """Add the selected config environment to the runs table."""
    op.add_column(
        "runs",
        sa.Column(
            "environment",
            sa.Text(),
            nullable=False,
            server_default=sa.text("''"),
        ),
    )


def downgrade() -> None:
    """Remove the selected config environment from the runs table."""
    op.drop_column("runs", "environment")

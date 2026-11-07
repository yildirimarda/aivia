"""Record the character version a run pinned when it started.

Revision ID: 0015
Revises: 0014
Create Date: 2026-11-07
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0015"
down_revision: str | None = "0014"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    """Add the pinned character version to the runs table."""
    op.add_column(
        "runs",
        sa.Column(
            "character_version",
            sa.Text(),
            nullable=False,
            server_default=sa.text("''"),
        ),
    )


def downgrade() -> None:
    """Remove the pinned character version from the runs table."""
    op.drop_column("runs", "character_version")

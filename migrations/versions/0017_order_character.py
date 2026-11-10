"""Record the optional character selected for an order.

Revision ID: 0017
Revises: 0016
Create Date: 2026-11-07
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0017"
down_revision: str | None = "0016"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    """Add an optional character to the orders table."""
    op.add_column(
        "orders",
        sa.Column(
            "character",
            sa.String(length=128),
            nullable=False,
            server_default=sa.text("''"),
        ),
    )


def downgrade() -> None:
    """Remove the optional character from the orders table."""
    op.drop_column("orders", "character")

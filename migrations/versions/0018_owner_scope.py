"""Scope runs, orders and posts to an owner.

Revision ID: 0018
Revises: 0017
Create Date: 2026-11-12
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0018"
down_revision: str | None = "0017"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    """Add an optional owner to runs, orders and posts."""
    op.add_column(
        "runs",
        sa.Column(
            "owner",
            sa.String(length=255),
            nullable=False,
            server_default=sa.text("''"),
        ),
    )
    op.add_column(
        "orders",
        sa.Column(
            "owner",
            sa.String(length=255),
            nullable=False,
            server_default=sa.text("''"),
        ),
    )
    op.add_column(
        "posts",
        sa.Column(
            "owner",
            sa.String(length=255),
            nullable=False,
            server_default=sa.text("''"),
        ),
    )


def downgrade() -> None:
    """Remove the owner from runs, orders and posts."""
    op.drop_column("runs", "owner")
    op.drop_column("orders", "owner")
    op.drop_column("posts", "owner")

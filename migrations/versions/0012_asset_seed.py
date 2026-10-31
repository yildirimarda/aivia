"""Record the generation seed that produced each asset.

Revision ID: 0012
Revises: 0011
Create Date: 2026-10-31
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    """Add the per-output generation seed to the assets table."""
    op.add_column("assets", sa.Column("seed", sa.Integer(), nullable=True))


def downgrade() -> None:
    """Remove the per-output generation seed from the assets table."""
    op.drop_column("assets", "seed")

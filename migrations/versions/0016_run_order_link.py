"""Link runs to their orders so an order's cost is the sum of its runs.

Revision ID: 0016
Revises: 0015
Create Date: 2026-11-07
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0016"
down_revision: str | None = "0015"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    """Add an optional order link to the runs table."""
    with op.batch_alter_table("runs") as batch_op:
        batch_op.add_column(
            sa.Column(
                "order_id",
                sa.String(length=64),
                nullable=True,
            )
        )
        batch_op.create_foreign_key(
            "fk_runs_order_id",
            "orders",
            ["order_id"],
            ["id"],
            ondelete="CASCADE",
        )
    op.create_index("ix_runs_order_id", "runs", ["order_id"])


def downgrade() -> None:
    """Remove the order link from the runs table."""
    op.drop_index("ix_runs_order_id", table_name="runs")
    with op.batch_alter_table("runs") as batch_op:
        batch_op.drop_constraint("fk_runs_order_id", type_="foreignkey")
        batch_op.drop_column("order_id")

"""Create the asset cache table.

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-29
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    """Add the request digest cache pointing at stored assets."""
    op.create_table(
        "asset_cache",
        sa.Column("cache_key", sa.String(length=64), primary_key=True),
        sa.Column(
            "asset_id",
            sa.String(length=64),
            sa.ForeignKey("assets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "provider",
            sa.String(length=128),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "model",
            sa.String(length=128),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "prompt_hash",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "parameters", sa.JSON(), nullable=False, server_default=sa.text("'{}'")
        ),
        sa.Column(
            "reference_hashes",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
        sa.Column(
            "hit_count", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "schema_version",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("1"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
    )
    op.create_index("ix_asset_cache_asset_id", "asset_cache", ["asset_id"])


def downgrade() -> None:
    """Drop the asset cache table."""
    op.drop_index("ix_asset_cache_asset_id", table_name="asset_cache")
    op.drop_table("asset_cache")

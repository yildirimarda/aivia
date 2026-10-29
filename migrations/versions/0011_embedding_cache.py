"""Create the embedding vector cache table.

Revision ID: 0011
Revises: 0010
Create Date: 2026-10-29
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    """Add the request digest cache holding embedding vectors."""
    op.create_table(
        "embedding_cache",
        sa.Column("cache_key", sa.String(length=64), primary_key=True),
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
            "text",
            sa.String(),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("dimensions", sa.Integer(), nullable=True),
        sa.Column(
            "input_type",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "normalize",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "truncate",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("1"),
        ),
        sa.Column("vector", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
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
    op.create_index(
        "ix_embedding_cache_provider",
        "embedding_cache",
        ["provider"],
    )


def downgrade() -> None:
    """Drop the embedding vector cache table."""
    op.drop_index("ix_embedding_cache_provider", table_name="embedding_cache")
    op.drop_table("embedding_cache")

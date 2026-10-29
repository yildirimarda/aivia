"""Create the LLM response cache table.

Revision ID: 0010
Revises: 0009
Create Date: 2026-10-29
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    """Add the request digest cache pointing at serialized LLM responses."""
    op.create_table(
        "llm_cache",
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
            "prompt_version",
            sa.String(length=128),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "messages", sa.JSON(), nullable=False, server_default=sa.text("'[]'")
        ),
        sa.Column(
            "parameters", sa.JSON(), nullable=False, server_default=sa.text("'{}'")
        ),
        sa.Column(
            "response", sa.JSON(), nullable=False, server_default=sa.text("'{}'")
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
    op.create_index("ix_llm_cache_provider", "llm_cache", ["provider"])


def downgrade() -> None:
    """Drop the LLM response cache table."""
    op.drop_index("ix_llm_cache_provider", table_name="llm_cache")
    op.drop_table("llm_cache")

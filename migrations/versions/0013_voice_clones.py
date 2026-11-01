"""Create the local voice clone index.

Revision ID: 0013
Revises: 0012
Create Date: 2026-11-01
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    """Add the table that ties provider voice ids to an owner and consent."""
    op.create_table(
        "voice_clones",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column(
            "provider",
            sa.String(length=128),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "provider_voice_id",
            sa.String(length=255),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "owner",
            sa.String(length=255),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "language",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("consent", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column(
            "metadata", sa.JSON(), nullable=False, server_default=sa.text("'{}'")
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
    op.create_index("ix_voice_clones_owner", "voice_clones", ["owner"])
    op.create_index("ix_voice_clones_provider", "voice_clones", ["provider"])
    op.create_index(
        "uq_voice_clones_provider_voice",
        "voice_clones",
        ["provider", "provider_voice_id"],
        unique=True,
    )


def downgrade() -> None:
    """Drop the voice clone table and its indexes."""
    op.drop_index("uq_voice_clones_provider_voice", table_name="voice_clones")
    op.drop_index("ix_voice_clones_provider", table_name="voice_clones")
    op.drop_index("ix_voice_clones_owner", table_name="voice_clones")
    op.drop_table("voice_clones")

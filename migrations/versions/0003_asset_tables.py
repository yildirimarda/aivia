"""Create the assets, compositions and cost events tables.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-28
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    """Add the asset, composition and cost event tables and their indexes."""
    op.create_table(
        "assets",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "kind",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'image'"),
        ),
        sa.Column("label", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("path", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column(
            "content_hash",
            sa.String(length=128),
            nullable=False,
            server_default=sa.text("''"),
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
        sa.Column("prompt", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column(
            "prompt_hash",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("license", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column(
            "cost_links", sa.JSON(), nullable=False, server_default=sa.text("'[]'")
        ),
        sa.Column(
            "analysis", sa.JSON(), nullable=False, server_default=sa.text("'{}'")
        ),
        sa.Column("alt_text", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("pinned", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("quality_score", sa.Float(), nullable=True),
        sa.Column(
            "run_id",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
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
    op.create_index("ix_assets_prompt_hash", "assets", ["prompt_hash"])
    op.create_index("ix_assets_run_id", "assets", ["run_id"])
    op.create_index("ix_assets_provider", "assets", ["provider"])
    op.create_index("ix_assets_created_at", "assets", ["created_at"])
    op.create_index("ix_assets_pinned", "assets", ["pinned"])

    op.create_table(
        "compositions",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("path", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column(
            "content_hash",
            sa.String(length=128),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "format",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "version", sa.Integer(), nullable=False, server_default=sa.text("1")
        ),
        sa.Column(
            "renderer_version",
            sa.String(length=128),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "toolchain_fingerprint",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("fps", sa.Float(), nullable=True),
        sa.Column("assets", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
        sa.Column("pinned", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "run_id",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "language",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "platform",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
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
    op.create_index("ix_compositions_run_id", "compositions", ["run_id"])
    op.create_index("ix_compositions_created_at", "compositions", ["created_at"])
    op.create_index("ix_compositions_pinned", "compositions", ["pinned"])

    op.create_table(
        "cost_events",
        sa.Column("id", sa.String(length=64), primary_key=True),
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
            "operation",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("units", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "unit_kind",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'call'"),
        ),
        sa.Column(
            "amount_usd", sa.Float(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "estimated", sa.Boolean(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "run_id",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "pricing_version",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
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
    op.create_index("ix_cost_events_run_id", "cost_events", ["run_id"])
    op.create_index("ix_cost_events_provider", "cost_events", ["provider"])
    op.create_index("ix_cost_events_created_at", "cost_events", ["created_at"])


def downgrade() -> None:
    """Drop the asset, composition and cost event tables."""
    op.drop_index("ix_cost_events_created_at", table_name="cost_events")
    op.drop_index("ix_cost_events_provider", table_name="cost_events")
    op.drop_index("ix_cost_events_run_id", table_name="cost_events")
    op.drop_table("cost_events")
    op.drop_index("ix_compositions_pinned", table_name="compositions")
    op.drop_index("ix_compositions_created_at", table_name="compositions")
    op.drop_index("ix_compositions_run_id", table_name="compositions")
    op.drop_table("compositions")
    op.drop_index("ix_assets_pinned", table_name="assets")
    op.drop_index("ix_assets_created_at", table_name="assets")
    op.drop_index("ix_assets_provider", table_name="assets")
    op.drop_index("ix_assets_run_id", table_name="assets")
    op.drop_index("ix_assets_prompt_hash", table_name="assets")
    op.drop_table("assets")

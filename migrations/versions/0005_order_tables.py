"""Create the order, delivery, post, post metric and web card hit tables.

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-28
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    """Add the order, delivery, post and analytics tables."""
    op.create_table(
        "orders",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "state",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'draft'"),
        ),
        sa.Column(
            "occasion",
            sa.String(length=128),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "name",
            sa.String(length=255),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("message", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column(
            "language",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "profile",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "level",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "timezone",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("'UTC'"),
        ),
        sa.Column(
            "variables", sa.JSON(), nullable=False, server_default=sa.text("'{}'")
        ),
        sa.Column("outputs", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
        sa.Column("pricing", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column(
            "currency",
            sa.String(length=8),
            nullable=False,
            server_default=sa.text("'USD'"),
        ),
        sa.Column(
            "amount_usd", sa.Float(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column("tax_usd", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column("cost_usd", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column("budget_usd", sa.Float(), nullable=True),
        sa.Column(
            "run_id",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "idempotency_key",
            sa.String(length=255),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("scheduled_for", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "error_code",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "error_message", sa.Text(), nullable=False, server_default=sa.text("''")
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
    op.create_index("ix_orders_state", "orders", ["state"])
    op.create_index("ix_orders_occasion", "orders", ["occasion"])
    op.create_index("ix_orders_run_id", "orders", ["run_id"])
    op.create_index("ix_orders_idempotency_key", "orders", ["idempotency_key"])
    op.create_index("ix_orders_scheduled_for", "orders", ["scheduled_for"])
    op.create_index("ix_orders_created_at", "orders", ["created_at"])

    op.create_table(
        "deliveries",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "order_id",
            sa.String(length=64),
            sa.ForeignKey("orders.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "run_id",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "provider",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("'folder'"),
        ),
        sa.Column(
            "state",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column("target", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("path", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("url", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column(
            "email",
            sa.String(length=320),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("files", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
        sa.Column("scheduled_for", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "attempts", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "max_attempts", sa.Integer(), nullable=False, server_default=sa.text("3")
        ),
        sa.Column(
            "idempotency_key",
            sa.String(length=255),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "confirmation", sa.Text(), nullable=False, server_default=sa.text("''")
        ),
        sa.Column(
            "error_code",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "error_message", sa.Text(), nullable=False, server_default=sa.text("''")
        ),
        sa.Column("cost_usd", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "attestation", sa.JSON(), nullable=False, server_default=sa.text("'{}'")
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
    op.create_index("ix_deliveries_order_id", "deliveries", ["order_id"])
    op.create_index("ix_deliveries_state", "deliveries", ["state"])
    op.create_index("ix_deliveries_scheduled_for", "deliveries", ["scheduled_for"])
    op.create_index("ix_deliveries_created_at", "deliveries", ["created_at"])

    op.create_table(
        "posts",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "order_id",
            sa.String(length=64),
            sa.ForeignKey("orders.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "platform",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "account",
            sa.String(length=128),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "profile",
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
            "format",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("title", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("caption", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column(
            "hashtags", sa.JSON(), nullable=False, server_default=sa.text("'[]'")
        ),
        sa.Column(
            "mentions", sa.JSON(), nullable=False, server_default=sa.text("'[]'")
        ),
        sa.Column("media", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
        sa.Column(
            "media_type",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "visibility",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "state",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'draft'"),
        ),
        sa.Column(
            "remote_id",
            sa.String(length=255),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("url", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("scheduled_for", sa.DateTime(timezone=True), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dry_run", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "idempotency_key",
            sa.String(length=255),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "run_id",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "composition_id",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "disclosure_text", sa.Text(), nullable=False, server_default=sa.text("''")
        ),
        sa.Column(
            "disclosure_required",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "disclosure_applied",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "attestation", sa.JSON(), nullable=False, server_default=sa.text("'{}'")
        ),
        sa.Column(
            "error_code",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "error_message", sa.Text(), nullable=False, server_default=sa.text("''")
        ),
        sa.Column("cost_usd", sa.Float(), nullable=False, server_default=sa.text("0")),
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
    op.create_index("ix_posts_state", "posts", ["state"])
    op.create_index("ix_posts_platform", "posts", ["platform"])
    op.create_index("ix_posts_run_id", "posts", ["run_id"])
    op.create_index("ix_posts_created_at", "posts", ["created_at"])

    op.create_table(
        "post_metrics",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "post_id",
            sa.String(length=64),
            sa.ForeignKey("posts.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "platform",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "impressions", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column("reach", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("views", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("likes", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "comments", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column("shares", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("saves", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("clicks", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "watch_time_seconds",
            sa.Float(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "engagement_rate", sa.Float(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("raw", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
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
    op.create_index("ix_post_metrics_post_id", "post_metrics", ["post_id"])
    op.create_index("ix_post_metrics_captured_at", "post_metrics", ["captured_at"])

    op.create_table(
        "web_card_hits",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "order_id",
            sa.String(length=64),
            sa.ForeignKey("orders.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "card_id",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "day", sa.String(length=10), nullable=False, server_default=sa.text("''")
        ),
        sa.Column(
            "visitor_hash",
            sa.String(length=128),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("hit_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "user_agent", sa.Text(), nullable=False, server_default=sa.text("''")
        ),
        sa.Column("referrer", sa.Text(), nullable=False, server_default=sa.text("''")),
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
    op.create_index("ix_web_card_hits_order_id", "web_card_hits", ["order_id"])
    op.create_index("ix_web_card_hits_day", "web_card_hits", ["day"])
    op.create_index("ix_web_card_hits_visitor_hash", "web_card_hits", ["visitor_hash"])
    op.create_index("ix_web_card_hits_created_at", "web_card_hits", ["created_at"])


def downgrade() -> None:
    """Drop the order, delivery, post and analytics tables."""
    op.drop_index("ix_web_card_hits_created_at", table_name="web_card_hits")
    op.drop_index("ix_web_card_hits_visitor_hash", table_name="web_card_hits")
    op.drop_index("ix_web_card_hits_day", table_name="web_card_hits")
    op.drop_index("ix_web_card_hits_order_id", table_name="web_card_hits")
    op.drop_table("web_card_hits")
    op.drop_index("ix_post_metrics_captured_at", table_name="post_metrics")
    op.drop_index("ix_post_metrics_post_id", table_name="post_metrics")
    op.drop_table("post_metrics")
    op.drop_index("ix_posts_created_at", table_name="posts")
    op.drop_index("ix_posts_run_id", table_name="posts")
    op.drop_index("ix_posts_platform", table_name="posts")
    op.drop_index("ix_posts_state", table_name="posts")
    op.drop_table("posts")
    op.drop_index("ix_deliveries_created_at", table_name="deliveries")
    op.drop_index("ix_deliveries_scheduled_for", table_name="deliveries")
    op.drop_index("ix_deliveries_state", table_name="deliveries")
    op.drop_index("ix_deliveries_order_id", table_name="deliveries")
    op.drop_table("deliveries")
    op.drop_index("ix_orders_created_at", table_name="orders")
    op.drop_index("ix_orders_scheduled_for", table_name="orders")
    op.drop_index("ix_orders_idempotency_key", table_name="orders")
    op.drop_index("ix_orders_run_id", table_name="orders")
    op.drop_index("ix_orders_occasion", table_name="orders")
    op.drop_index("ix_orders_state", table_name="orders")
    op.drop_table("orders")

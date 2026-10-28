"""Create the plan, template, reference, style spec, feedback and rating tables.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-28
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    """Add the assistant and reference tables and their lookup indexes."""
    op.create_table(
        "plans",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "instruction", sa.Text(), nullable=False, server_default=sa.text("''")
        ),
        sa.Column(
            "state",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'draft'"),
        ),
        sa.Column("steps", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
        sa.Column(
            "assumptions", sa.JSON(), nullable=False, server_default=sa.text("'[]'")
        ),
        sa.Column(
            "approval_mode",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'ask'"),
        ),
        sa.Column(
            "approved", sa.Boolean(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "approved_by",
            sa.String(length=128),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "estimate_usd", sa.Float(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column("cost_usd", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column("budget_usd", sa.Float(), nullable=True),
        sa.Column(
            "run_id",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("profile", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("format", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column(
            "idempotency_key",
            sa.String(length=255),
            nullable=False,
            server_default=sa.text("''"),
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
    op.create_index("ix_plans_run_id", "plans", ["run_id"])
    op.create_index("ix_plans_state", "plans", ["state"])
    op.create_index("ix_plans_created_at", "plans", ["created_at"])

    op.create_table(
        "templates",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "name",
            sa.String(length=255),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "kind",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'plan'"),
        ),
        sa.Column(
            "description", sa.Text(), nullable=False, server_default=sa.text("''")
        ),
        sa.Column("path", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column(
            "content_hash",
            sa.String(length=128),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("version", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column(
            "variables", sa.JSON(), nullable=False, server_default=sa.text("'{}'")
        ),
        sa.Column(
            "asset_slots", sa.JSON(), nullable=False, server_default=sa.text("'[]'")
        ),
        sa.Column(
            "source_plan_id",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "source_composition_id",
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
    op.create_index("ix_templates_name", "templates", ["name"])
    op.create_index("ix_templates_kind", "templates", ["kind"])
    op.create_index("ix_templates_created_at", "templates", ["created_at"])

    op.create_table(
        "references",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "kind",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'image'"),
        ),
        sa.Column("label", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("path", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("url", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column(
            "content_hash",
            sa.String(length=128),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "mime_type",
            sa.String(length=128),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column(
            "language",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("source", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("owned", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "analysis", sa.JSON(), nullable=False, server_default=sa.text("'{}'")
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
    op.create_index("ix_references_kind", "references", ["kind"])
    op.create_index("ix_references_content_hash", "references", ["content_hash"])
    op.create_index("ix_references_created_at", "references", ["created_at"])

    op.create_table(
        "style_specs",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "name",
            sa.String(length=255),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("path", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column(
            "content_hash",
            sa.String(length=128),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("version", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("spec", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column(
            "reference_ids", sa.JSON(), nullable=False, server_default=sa.text("'[]'")
        ),
        sa.Column("recipe", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
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
    op.create_index("ix_style_specs_name", "style_specs", ["name"])
    op.create_index("ix_style_specs_created_at", "style_specs", ["created_at"])

    op.create_table(
        "feedback",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "subject_kind",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'topic'"),
        ),
        sa.Column(
            "subject_id",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "sentiment",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'neutral'"),
        ),
        sa.Column("comment", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column(
            "source",
            sa.String(length=64),
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
    op.create_index("ix_feedback_subject", "feedback", ["subject_kind", "subject_id"])
    op.create_index("ix_feedback_created_at", "feedback", ["created_at"])

    op.create_table(
        "ratings",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "subject_kind",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'provider'"),
        ),
        sa.Column(
            "subject_id",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("score", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "rater",
            sa.String(length=128),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("comment", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column(
            "run_id",
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
    op.create_index(
        "uq_ratings_subject",
        "ratings",
        ["subject_kind", "subject_id"],
        unique=True,
    )
    op.create_index("ix_ratings_created_at", "ratings", ["created_at"])


def downgrade() -> None:
    """Drop the assistant and reference tables."""
    op.drop_index("ix_ratings_created_at", table_name="ratings")
    op.drop_index("uq_ratings_subject", table_name="ratings")
    op.drop_table("ratings")
    op.drop_index("ix_feedback_created_at", table_name="feedback")
    op.drop_index("ix_feedback_subject", table_name="feedback")
    op.drop_table("feedback")
    op.drop_index("ix_style_specs_created_at", table_name="style_specs")
    op.drop_index("ix_style_specs_name", table_name="style_specs")
    op.drop_table("style_specs")
    op.drop_index("ix_references_created_at", table_name="references")
    op.drop_index("ix_references_content_hash", table_name="references")
    op.drop_index("ix_references_kind", table_name="references")
    op.drop_table("references")
    op.drop_index("ix_templates_created_at", table_name="templates")
    op.drop_index("ix_templates_kind", table_name="templates")
    op.drop_index("ix_templates_name", table_name="templates")
    op.drop_table("templates")
    op.drop_index("ix_plans_created_at", table_name="plans")
    op.drop_index("ix_plans_state", table_name="plans")
    op.drop_index("ix_plans_run_id", table_name="plans")
    op.drop_table("plans")

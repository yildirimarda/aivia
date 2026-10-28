"""Create the runs, stage runs and jobs tables.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-28
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None

_IDEMPOTENCY_PREDICATE = sa.text(
    "idempotency_key IS NOT NULL AND idempotency_key != ''"
)


def upgrade() -> None:
    """Add the core run and job tables and their queue indexes."""
    op.create_table(
        "runs",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "kind",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'pipeline'"),
        ),
        sa.Column(
            "state",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'queued'"),
        ),
        sa.Column("idempotency_key", sa.String(length=255), nullable=True),
        sa.Column(
            "toolchain_fingerprint",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column("profile", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("format", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("run_dir", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column(
            "config_snapshot",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column("cost_usd", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column("estimated_cost_usd", sa.Float(), nullable=True),
        sa.Column(
            "error_code",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "error_message", sa.Text(), nullable=False, server_default=sa.text("''")
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("paused_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "pause_reason", sa.Text(), nullable=False, server_default=sa.text("''")
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
    op.create_index("ix_runs_kind", "runs", ["kind"])
    op.create_index("ix_runs_state", "runs", ["state"])
    op.create_index(
        "uq_runs_idempotency_key",
        "runs",
        ["idempotency_key"],
        unique=True,
        sqlite_where=_IDEMPOTENCY_PREDICATE,
        postgresql_where=_IDEMPOTENCY_PREDICATE,
    )

    op.create_table(
        "stage_runs",
        sa.Column(
            "run_id",
            sa.String(length=64),
            sa.ForeignKey("runs.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("stage", sa.String(length=128), primary_key=True),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column("index", sa.Integer(), nullable=True),
        sa.Column(
            "input_hash",
            sa.String(length=128),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "output_hash",
            sa.String(length=128),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "artifacts", sa.JSON(), nullable=False, server_default=sa.text("'[]'")
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
        sa.Column("estimated_cost_usd", sa.Float(), nullable=True),
        sa.Column(
            "toolchain_fingerprint",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column(
            "metadata", sa.JSON(), nullable=False, server_default=sa.text("'{}'")
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
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
    op.create_index("ix_stage_runs_status", "stage_runs", ["status"])

    op.create_table(
        "jobs",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "kind",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("'run'"),
        ),
        sa.Column(
            "state",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column(
            "run_id",
            sa.String(length=64),
            sa.ForeignKey("runs.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "attempts", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "max_attempts", sa.Integer(), nullable=False, server_default=sa.text("3")
        ),
        sa.Column("not_before", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "claimed_by",
            sa.String(length=128),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "error_code",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "error_message", sa.Text(), nullable=False, server_default=sa.text("''")
        ),
        sa.Column("payload", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
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
        sa.CheckConstraint(
            "attempts BETWEEN 0 AND max_attempts",
            name="ck_jobs_attempts_within_limit",
        ),
        sa.CheckConstraint(
            "max_attempts BETWEEN 1 AND 2147483647",
            name="ck_jobs_max_attempts_positive",
        ),
    )
    op.create_index("ix_jobs_claim", "jobs", ["state", "not_before"])
    op.create_index("ix_jobs_run_id", "jobs", ["run_id"])


def downgrade() -> None:
    """Drop the core run and job tables."""
    op.drop_index("ix_jobs_run_id", table_name="jobs")
    op.drop_index("ix_jobs_claim", table_name="jobs")
    op.drop_table("jobs")
    op.drop_index("ix_stage_runs_status", table_name="stage_runs")
    op.drop_table("stage_runs")
    op.drop_index("uq_runs_idempotency_key", table_name="runs")
    op.drop_index("ix_runs_state", table_name="runs")
    op.drop_index("ix_runs_kind", table_name="runs")
    op.drop_table("runs")

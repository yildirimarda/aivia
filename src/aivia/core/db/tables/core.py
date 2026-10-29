"""Core tables for runs, their stage results and the persistent job queue.

The tables mirror the run, stage result and job records. They live on the
shared :data:`aivia.core.db.schema.metadata` so migrations, tests and future
repositories all describe one schema.

Two indexes carry the queue guarantees. ``ix_jobs_claim`` makes selecting the
next due pending job cheap, and ``uq_runs_idempotency_key`` makes a repeated
idempotency key fail instead of producing a second paid run. The unique index
is partial so runs without a key, including rows that store an empty string,
do not collide with each other.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
    func,
    text,
)

from aivia.core.db.schema import metadata

__all__ = [
    "CLAIM_INDEX_NAME",
    "IDEMPOTENCY_INDEX_NAME",
    "jobs",
    "runs",
    "stage_runs",
]

CLAIM_INDEX_NAME = "ix_jobs_claim"
"""The index a worker uses to find the next claimable job."""

IDEMPOTENCY_INDEX_NAME = "uq_runs_idempotency_key"
"""The unique index that rejects a repeated non-empty run idempotency key."""

_IDEMPOTENCY_PREDICATE = text("idempotency_key IS NOT NULL AND idempotency_key != ''")

runs = Table(
    "runs",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("kind", String(32), nullable=False, server_default=text("'pipeline'")),
    Column("state", String(32), nullable=False, server_default=text("'queued'")),
    Column("idempotency_key", String(255), nullable=True),
    Column(
        "idempotency_payload_hash",
        String(64),
        nullable=False,
        server_default=text("''"),
    ),
    Column("toolchain_fingerprint", JSON, nullable=False, server_default=text("'{}'")),
    Column("profile", Text, nullable=False, server_default=text("''")),
    Column("environment", Text, nullable=False, server_default=text("''")),
    Column("format", Text, nullable=False, server_default=text("''")),
    Column("run_dir", Text, nullable=False, server_default=text("''")),
    Column("config_snapshot", JSON, nullable=False, server_default=text("'{}'")),
    Column("cost_usd", Float, nullable=False, server_default=text("0")),
    Column("estimated_cost_usd", Float, nullable=True),
    Column("error_code", String(64), nullable=False, server_default=text("''")),
    Column("error_message", Text, nullable=False, server_default=text("''")),
    Column("started_at", DateTime(timezone=True), nullable=True),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    Column("paused_at", DateTime(timezone=True), nullable=True),
    Column("pause_reason", Text, nullable=False, server_default=text("''")),
    Column("resumed_by", Text, nullable=False, server_default=text("''")),
    Column("resume_reason", Text, nullable=False, server_default=text("''")),
    Column("resumed_at", DateTime(timezone=True), nullable=True),
    Column("schema_version", Integer, nullable=False, server_default=text("1")),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    ),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    ),
)

Index("ix_runs_kind", runs.c.kind)
Index("ix_runs_state", runs.c.state)
Index(
    IDEMPOTENCY_INDEX_NAME,
    runs.c.idempotency_key,
    unique=True,
    sqlite_where=_IDEMPOTENCY_PREDICATE,
    postgresql_where=_IDEMPOTENCY_PREDICATE,
)

stage_runs = Table(
    "stage_runs",
    metadata,
    Column(
        "run_id",
        String(64),
        ForeignKey("runs.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("stage", String(128), primary_key=True),
    Column("status", String(32), nullable=False, server_default=text("'pending'")),
    Column("index", Integer, nullable=True),
    Column("input_hash", String(128), nullable=False, server_default=text("''")),
    Column("output_hash", String(128), nullable=False, server_default=text("''")),
    Column("artifacts", JSON, nullable=False, server_default=text("'[]'")),
    Column("error_code", String(64), nullable=False, server_default=text("''")),
    Column("error_message", Text, nullable=False, server_default=text("''")),
    Column("cost_usd", Float, nullable=False, server_default=text("0")),
    Column("estimated_cost_usd", Float, nullable=True),
    Column("toolchain_fingerprint", JSON, nullable=False, server_default=text("'{}'")),
    Column("metadata", JSON, nullable=False, server_default=text("'{}'")),
    Column("started_at", DateTime(timezone=True), nullable=True),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    Column("schema_version", Integer, nullable=False, server_default=text("1")),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    ),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    ),
)

Index("ix_stage_runs_status", stage_runs.c.status)

jobs = Table(
    "jobs",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("kind", String(64), nullable=False, server_default=text("'run'")),
    Column("state", String(32), nullable=False, server_default=text("'pending'")),
    Column(
        "run_id",
        String(64),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=True,
    ),
    Column("attempts", Integer, nullable=False, server_default=text("0")),
    Column("max_attempts", Integer, nullable=False, server_default=text("3")),
    Column("not_before", DateTime(timezone=True), nullable=True),
    Column("heartbeat_at", DateTime(timezone=True), nullable=True),
    Column("claimed_by", String(128), nullable=False, server_default=text("''")),
    Column("claimed_at", DateTime(timezone=True), nullable=True),
    Column("started_at", DateTime(timezone=True), nullable=True),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    Column("error_code", String(64), nullable=False, server_default=text("''")),
    Column("error_message", Text, nullable=False, server_default=text("''")),
    Column("payload", JSON, nullable=False, server_default=text("'{}'")),
    Column("metadata", JSON, nullable=False, server_default=text("'{}'")),
    Column("schema_version", Integer, nullable=False, server_default=text("1")),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    ),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    ),
    CheckConstraint(
        "attempts BETWEEN 0 AND max_attempts",
        name="ck_jobs_attempts_within_limit",
    ),
    CheckConstraint(
        "max_attempts BETWEEN 1 AND 2147483647",
        name="ck_jobs_max_attempts_positive",
    ),
)

Index(CLAIM_INDEX_NAME, jobs.c.state, jobs.c.not_before)
Index("ix_jobs_run_id", jobs.c.run_id)

"""Tables for plans, templates, references, style specs, feedback and ratings.

Plans persist the validated tool-call sequence produced from an instruction.
Templates save a plan or composition with variables and asset slots so it can be
run again or in batch. References record the user-supplied examples a run was
built from, including the ownership flag the rights policy needs. Style specs
hold the structured, human-editable description extracted from references.

Feedback is deliberately not unique: a subject can collect many likes, dislikes
and comments over time. Ratings are the opposite. ``uq_ratings_subject`` makes
``(subject_kind, subject_id)`` unique, so a subject has exactly one rating that
later provider-selection strategies can read without resolving conflicts.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
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
    "FEEDBACK_CREATED_AT_INDEX_NAME",
    "FEEDBACK_SUBJECT_INDEX_NAME",
    "PLAN_CREATED_AT_INDEX_NAME",
    "PLAN_RUN_INDEX_NAME",
    "PLAN_STATE_INDEX_NAME",
    "RATING_CREATED_AT_INDEX_NAME",
    "RATING_SUBJECT_INDEX_NAME",
    "REFERENCE_CONTENT_HASH_INDEX_NAME",
    "REFERENCE_CREATED_AT_INDEX_NAME",
    "REFERENCE_KIND_INDEX_NAME",
    "STYLE_SPEC_CREATED_AT_INDEX_NAME",
    "STYLE_SPEC_NAME_INDEX_NAME",
    "TEMPLATE_CREATED_AT_INDEX_NAME",
    "TEMPLATE_KIND_INDEX_NAME",
    "TEMPLATE_NAME_INDEX_NAME",
    "feedback",
    "plans",
    "ratings",
    "references",
    "style_specs",
    "templates",
]

PLAN_RUN_INDEX_NAME = "ix_plans_run_id"
"""List the plans belonging to one run."""

PLAN_STATE_INDEX_NAME = "ix_plans_state"
"""Find plans awaiting approval or currently executing."""

PLAN_CREATED_AT_INDEX_NAME = "ix_plans_created_at"
"""Scan plans by creation time for reports and retention."""

TEMPLATE_NAME_INDEX_NAME = "ix_templates_name"
"""Look up a saved template by name."""

TEMPLATE_KIND_INDEX_NAME = "ix_templates_kind"
"""Group templates by what they save, a plan or a composition."""

TEMPLATE_CREATED_AT_INDEX_NAME = "ix_templates_created_at"
"""Scan templates by creation time."""

REFERENCE_KIND_INDEX_NAME = "ix_references_kind"
"""Group references by media kind before analysis."""

REFERENCE_CONTENT_HASH_INDEX_NAME = "ix_references_content_hash"
"""Find a reference by the hash of its content to avoid duplicates."""

REFERENCE_CREATED_AT_INDEX_NAME = "ix_references_created_at"
"""Scan references by creation time for retention."""

STYLE_SPEC_NAME_INDEX_NAME = "ix_style_specs_name"
"""Look up a stored style spec by name."""

STYLE_SPEC_CREATED_AT_INDEX_NAME = "ix_style_specs_created_at"
"""Scan style specs by creation time."""

FEEDBACK_SUBJECT_INDEX_NAME = "ix_feedback_subject"
"""Collect the feedback recorded for one subject."""

FEEDBACK_CREATED_AT_INDEX_NAME = "ix_feedback_created_at"
"""Scan feedback by creation time when summarizing it."""

RATING_SUBJECT_INDEX_NAME = "uq_ratings_subject"
"""The unique index that allows one rating per subject."""

RATING_CREATED_AT_INDEX_NAME = "ix_ratings_created_at"
"""Scan ratings by creation time."""


plans = Table(
    "plans",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("instruction", Text, nullable=False, server_default=text("''")),
    Column("state", String(32), nullable=False, server_default=text("'draft'")),
    Column("steps", JSON, nullable=False, server_default=text("'[]'")),
    Column("assumptions", JSON, nullable=False, server_default=text("'[]'")),
    Column("approval_mode", String(32), nullable=False, server_default=text("'ask'")),
    Column("approved", Boolean, nullable=False, server_default=text("0")),
    Column("approved_by", String(128), nullable=False, server_default=text("''")),
    Column("approved_at", DateTime(timezone=True), nullable=True),
    Column("estimate_usd", Float, nullable=False, server_default=text("0")),
    Column("cost_usd", Float, nullable=False, server_default=text("0")),
    Column("budget_usd", Float, nullable=True),
    Column("run_id", String(64), nullable=False, server_default=text("''")),
    Column("profile", Text, nullable=False, server_default=text("''")),
    Column("format", Text, nullable=False, server_default=text("''")),
    Column("idempotency_key", String(255), nullable=False, server_default=text("''")),
    Column("error_code", String(64), nullable=False, server_default=text("''")),
    Column("error_message", Text, nullable=False, server_default=text("''")),
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
)

Index(PLAN_RUN_INDEX_NAME, plans.c.run_id)
Index(PLAN_STATE_INDEX_NAME, plans.c.state)
Index(PLAN_CREATED_AT_INDEX_NAME, plans.c.created_at)


templates = Table(
    "templates",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("name", String(255), nullable=False, server_default=text("''")),
    Column("kind", String(32), nullable=False, server_default=text("'plan'")),
    Column("description", Text, nullable=False, server_default=text("''")),
    Column("path", Text, nullable=False, server_default=text("''")),
    Column("content_hash", String(128), nullable=False, server_default=text("''")),
    Column("version", Integer, nullable=False, server_default=text("1")),
    Column("variables", JSON, nullable=False, server_default=text("'{}'")),
    Column("asset_slots", JSON, nullable=False, server_default=text("'[]'")),
    Column("source_plan_id", String(64), nullable=False, server_default=text("''")),
    Column(
        "source_composition_id",
        String(64),
        nullable=False,
        server_default=text("''"),
    ),
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
)

Index(TEMPLATE_NAME_INDEX_NAME, templates.c.name)
Index(TEMPLATE_KIND_INDEX_NAME, templates.c.kind)
Index(TEMPLATE_CREATED_AT_INDEX_NAME, templates.c.created_at)


references = Table(
    "references",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("kind", String(32), nullable=False, server_default=text("'image'")),
    Column("label", Text, nullable=False, server_default=text("''")),
    Column("path", Text, nullable=False, server_default=text("''")),
    Column("url", Text, nullable=False, server_default=text("''")),
    Column("content_hash", String(128), nullable=False, server_default=text("''")),
    Column("mime_type", String(128), nullable=False, server_default=text("''")),
    Column("size_bytes", Integer, nullable=True),
    Column("language", String(32), nullable=False, server_default=text("''")),
    Column("source", Text, nullable=False, server_default=text("''")),
    Column("owned", Boolean, nullable=False, server_default=text("0")),
    Column("analysis", JSON, nullable=False, server_default=text("'{}'")),
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
)

Index(REFERENCE_KIND_INDEX_NAME, references.c.kind)
Index(REFERENCE_CONTENT_HASH_INDEX_NAME, references.c.content_hash)
Index(REFERENCE_CREATED_AT_INDEX_NAME, references.c.created_at)


style_specs = Table(
    "style_specs",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("name", String(255), nullable=False, server_default=text("''")),
    Column("path", Text, nullable=False, server_default=text("''")),
    Column("content_hash", String(128), nullable=False, server_default=text("''")),
    Column("version", Integer, nullable=False, server_default=text("1")),
    Column("spec", JSON, nullable=False, server_default=text("'{}'")),
    Column("reference_ids", JSON, nullable=False, server_default=text("'[]'")),
    Column("recipe", JSON, nullable=False, server_default=text("'{}'")),
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
)

Index(STYLE_SPEC_NAME_INDEX_NAME, style_specs.c.name)
Index(STYLE_SPEC_CREATED_AT_INDEX_NAME, style_specs.c.created_at)


feedback = Table(
    "feedback",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("subject_kind", String(32), nullable=False, server_default=text("'topic'")),
    Column("subject_id", String(64), nullable=False, server_default=text("''")),
    Column("sentiment", String(16), nullable=False, server_default=text("'neutral'")),
    Column("comment", Text, nullable=False, server_default=text("''")),
    Column("source", String(64), nullable=False, server_default=text("''")),
    Column("run_id", String(64), nullable=False, server_default=text("''")),
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
)

Index(FEEDBACK_SUBJECT_INDEX_NAME, feedback.c.subject_kind, feedback.c.subject_id)
Index(FEEDBACK_CREATED_AT_INDEX_NAME, feedback.c.created_at)


ratings = Table(
    "ratings",
    metadata,
    Column("id", String(64), primary_key=True),
    Column(
        "subject_kind",
        String(32),
        nullable=False,
        server_default=text("'provider'"),
    ),
    Column("subject_id", String(64), nullable=False, server_default=text("''")),
    Column("score", Float, nullable=False, server_default=text("0")),
    Column("rater", String(128), nullable=False, server_default=text("''")),
    Column("comment", Text, nullable=False, server_default=text("''")),
    Column("run_id", String(64), nullable=False, server_default=text("''")),
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
)

Index(
    RATING_SUBJECT_INDEX_NAME, ratings.c.subject_kind, ratings.c.subject_id, unique=True
)
Index(RATING_CREATED_AT_INDEX_NAME, ratings.c.created_at)

"""Tables for assets, composition references and cost events.

The three tables record what a run used or produced and what it cost. Assets
keep their provenance, including the prompt hash used for caching, the license
and the pinned flag that protects a final output from retention. Compositions
store the reference to a stored composition together with the renderer
toolchain fingerprint. Cost events are the immutable ledger of every paid call.

The indexes answer the questions the pipeline asks most often: which asset was
made from a prompt, what belongs to a run, what a provider produced, what was
created in a time window and which outputs are pinned.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
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
    "ASSET_CACHE_ASSET_INDEX_NAME",
    "ASSET_CREATED_AT_INDEX_NAME",
    "ASSET_PINNED_INDEX_NAME",
    "ASSET_PROMPT_HASH_INDEX_NAME",
    "ASSET_PROVIDER_INDEX_NAME",
    "ASSET_RUN_INDEX_NAME",
    "COMPOSITION_CREATED_AT_INDEX_NAME",
    "COMPOSITION_PINNED_INDEX_NAME",
    "COMPOSITION_RUN_INDEX_NAME",
    "COST_EVENT_CREATED_AT_INDEX_NAME",
    "COST_EVENT_PROVIDER_INDEX_NAME",
    "COST_EVENT_RUN_INDEX_NAME",
    "asset_cache",
    "assets",
    "compositions",
    "cost_events",
]

ASSET_CACHE_ASSET_INDEX_NAME = "ix_asset_cache_asset_id"
"""Find the cache entry pointing at one asset."""

ASSET_PROMPT_HASH_INDEX_NAME = "ix_assets_prompt_hash"
"""Look up a cached asset by the hash of the prompt that produced it."""

ASSET_RUN_INDEX_NAME = "ix_assets_run_id"
"""List the assets belonging to one run."""

ASSET_PROVIDER_INDEX_NAME = "ix_assets_provider"
"""Group generated assets by the provider that made them."""

ASSET_CREATED_AT_INDEX_NAME = "ix_assets_created_at"
"""Scan assets by creation time for retention and reporting."""

ASSET_PINNED_INDEX_NAME = "ix_assets_pinned"
"""Find pinned final outputs that retention must not delete."""

COMPOSITION_RUN_INDEX_NAME = "ix_compositions_run_id"
"""List the compositions belonging to one run."""

COMPOSITION_CREATED_AT_INDEX_NAME = "ix_compositions_created_at"
"""Scan compositions by creation time."""

COMPOSITION_PINNED_INDEX_NAME = "ix_compositions_pinned"
"""Find pinned final compositions that retention must not delete."""

COST_EVENT_RUN_INDEX_NAME = "ix_cost_events_run_id"
"""Total and list the cost events of one run."""

COST_EVENT_PROVIDER_INDEX_NAME = "ix_cost_events_provider"
"""Aggregate spend per provider."""

COST_EVENT_CREATED_AT_INDEX_NAME = "ix_cost_events_created_at"
"""Scan cost events by creation time for budgets and reports."""


assets = Table(
    "assets",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("kind", String(32), nullable=False, server_default=text("'image'")),
    Column("label", Text, nullable=False, server_default=text("''")),
    Column("path", Text, nullable=False, server_default=text("''")),
    Column("content_hash", String(128), nullable=False, server_default=text("''")),
    Column("provider", String(128), nullable=False, server_default=text("''")),
    Column("model", String(128), nullable=False, server_default=text("''")),
    Column("prompt", Text, nullable=False, server_default=text("''")),
    Column("prompt_hash", String(64), nullable=False, server_default=text("''")),
    Column("license", JSON, nullable=False, server_default=text("'{}'")),
    Column("cost_links", JSON, nullable=False, server_default=text("'[]'")),
    Column("analysis", JSON, nullable=False, server_default=text("'{}'")),
    Column("alt_text", Text, nullable=False, server_default=text("''")),
    Column("pinned", Boolean, nullable=False, server_default=text("0")),
    Column("quality_score", Float, nullable=True),
    Column("run_id", String(64), nullable=False, server_default=text("''")),
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

Index(ASSET_PROMPT_HASH_INDEX_NAME, assets.c.prompt_hash)
Index(ASSET_RUN_INDEX_NAME, assets.c.run_id)
Index(ASSET_PROVIDER_INDEX_NAME, assets.c.provider)
Index(ASSET_CREATED_AT_INDEX_NAME, assets.c.created_at)
Index(ASSET_PINNED_INDEX_NAME, assets.c.pinned)

asset_cache = Table(
    "asset_cache",
    metadata,
    Column("cache_key", String(64), primary_key=True),
    Column(
        "asset_id",
        String(64),
        ForeignKey("assets.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("provider", String(128), nullable=False, server_default=text("''")),
    Column("model", String(128), nullable=False, server_default=text("''")),
    Column("prompt_hash", String(64), nullable=False, server_default=text("''")),
    Column("parameters", JSON, nullable=False, server_default=text("'{}'")),
    Column("reference_hashes", JSON, nullable=False, server_default=text("'[]'")),
    Column("hit_count", Integer, nullable=False, server_default=text("0")),
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

Index(ASSET_CACHE_ASSET_INDEX_NAME, asset_cache.c.asset_id)

compositions = Table(
    "compositions",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("path", Text, nullable=False, server_default=text("''")),
    Column("content_hash", String(128), nullable=False, server_default=text("''")),
    Column("format", String(32), nullable=False, server_default=text("''")),
    Column("version", Integer, nullable=False, server_default=text("1")),
    Column("renderer_version", String(128), nullable=False, server_default=text("''")),
    Column(
        "toolchain_fingerprint",
        JSON,
        nullable=False,
        server_default=text("'{}'"),
    ),
    Column("duration_seconds", Float, nullable=True),
    Column("width", Integer, nullable=True),
    Column("height", Integer, nullable=True),
    Column("fps", Float, nullable=True),
    Column("assets", JSON, nullable=False, server_default=text("'[]'")),
    Column("pinned", Boolean, nullable=False, server_default=text("0")),
    Column("run_id", String(64), nullable=False, server_default=text("''")),
    Column("language", String(32), nullable=False, server_default=text("''")),
    Column("platform", String(64), nullable=False, server_default=text("''")),
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

Index(COMPOSITION_RUN_INDEX_NAME, compositions.c.run_id)
Index(COMPOSITION_CREATED_AT_INDEX_NAME, compositions.c.created_at)
Index(COMPOSITION_PINNED_INDEX_NAME, compositions.c.pinned)

cost_events = Table(
    "cost_events",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("provider", String(128), nullable=False, server_default=text("''")),
    Column("model", String(128), nullable=False, server_default=text("''")),
    Column("operation", String(64), nullable=False, server_default=text("''")),
    Column("units", Float, nullable=False, server_default=text("0")),
    Column("unit_kind", String(32), nullable=False, server_default=text("'call'")),
    Column("amount_usd", Float, nullable=False, server_default=text("0")),
    Column("estimated", Boolean, nullable=False, server_default=text("0")),
    Column("run_id", String(64), nullable=False, server_default=text("''")),
    Column("pricing_version", String(64), nullable=False, server_default=text("''")),
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

Index(COST_EVENT_RUN_INDEX_NAME, cost_events.c.run_id)
Index(COST_EVENT_PROVIDER_INDEX_NAME, cost_events.c.provider)
Index(COST_EVENT_CREATED_AT_INDEX_NAME, cost_events.c.created_at)

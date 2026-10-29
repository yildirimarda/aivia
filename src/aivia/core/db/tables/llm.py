"""Table for cached LLM responses.

The cache key is the canonical digest of the provider, model, prompt version,
messages and call parameters. The serialized response is stored beside it so a
hit can be returned without calling a provider.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Index,
    Integer,
    String,
    Table,
    func,
    text,
)

from aivia.core.db.schema import metadata

__all__ = [
    "LLM_CACHE_PROVIDER_INDEX_NAME",
    "llm_cache",
]

LLM_CACHE_PROVIDER_INDEX_NAME = "ix_llm_cache_provider"
"""Find cached responses by the provider that produced them."""

llm_cache = Table(
    "llm_cache",
    metadata,
    Column("cache_key", String(64), primary_key=True),
    Column("provider", String(128), nullable=False, server_default=text("''")),
    Column("model", String(128), nullable=False, server_default=text("''")),
    Column("prompt_version", String(128), nullable=False, server_default=text("''")),
    Column("messages", JSON, nullable=False, server_default=text("'[]'")),
    Column("parameters", JSON, nullable=False, server_default=text("'{}'")),
    Column("response", JSON, nullable=False, server_default=text("'{}'")),
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

Index(LLM_CACHE_PROVIDER_INDEX_NAME, llm_cache.c.provider)

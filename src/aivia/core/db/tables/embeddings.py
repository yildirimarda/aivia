"""Table for cached embedding vectors.

Embedding the same input twice is wasted spend, so the cache stores one vector
per provider, model, dimension, input type and normalization flag. The key is
the canonical digest of those values, which lets a later process reuse a vector
without calling a provider. ``hit_count`` counts the lookups that were served
from the entry.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
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
    "EMBEDDING_CACHE_PROVIDER_INDEX_NAME",
    "embedding_cache",
]

EMBEDDING_CACHE_PROVIDER_INDEX_NAME = "ix_embedding_cache_provider"
"""Find cached vectors by the provider that produced them."""

embedding_cache = Table(
    "embedding_cache",
    metadata,
    Column("cache_key", String(64), primary_key=True),
    Column("provider", String(128), nullable=False, server_default=text("''")),
    Column("model", String(128), nullable=False, server_default=text("''")),
    Column("text", String, nullable=False, server_default=text("''")),
    Column("dimensions", Integer, nullable=True),
    Column("input_type", String(32), nullable=False, server_default=text("''")),
    Column("normalize", Boolean, nullable=False, server_default=text("0")),
    Column("truncate", Boolean, nullable=False, server_default=text("1")),
    Column("vector", JSON, nullable=False, server_default=text("'[]'")),
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

Index(EMBEDDING_CACHE_PROVIDER_INDEX_NAME, embedding_cache.c.provider)

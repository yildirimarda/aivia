"""Table for locally registered voice clones.

A clone record keeps the provider voice id that later speech requests pin,
together with the owner whose voice it is and the consent metadata that made
the clone legal to use. The table is deliberately small: the provider adapter
owns cloning and deletion at the vendor, while this table is the durable local
index that survives restarts and ties a provider id to an owner.
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
    Text,
    func,
    text,
)

from aivia.core.db.schema import metadata

__all__ = [
    "VOICE_CLONE_OWNER_INDEX_NAME",
    "VOICE_CLONE_PROVIDER_INDEX_NAME",
    "VOICE_CLONE_PROVIDER_VOICE_INDEX_NAME",
    "voice_clones",
]

VOICE_CLONE_OWNER_INDEX_NAME = "ix_voice_clones_owner"
"""List every clone owned by one person or account."""

VOICE_CLONE_PROVIDER_INDEX_NAME = "ix_voice_clones_provider"
"""Group clones by the provider that hosts them."""

VOICE_CLONE_PROVIDER_VOICE_INDEX_NAME = "uq_voice_clones_provider_voice"
"""Keep one local record per provider voice id."""

voice_clones = Table(
    "voice_clones",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("name", Text, nullable=False, server_default=text("''")),
    Column("provider", String(128), nullable=False, server_default=text("''")),
    Column(
        "provider_voice_id",
        String(255),
        nullable=False,
        server_default=text("''"),
    ),
    Column("owner", String(255), nullable=False, server_default=text("''")),
    Column("language", String(32), nullable=False, server_default=text("''")),
    Column("consent", JSON, nullable=False, server_default=text("'{}'")),
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

Index(VOICE_CLONE_OWNER_INDEX_NAME, voice_clones.c.owner)
Index(VOICE_CLONE_PROVIDER_INDEX_NAME, voice_clones.c.provider)
Index(
    VOICE_CLONE_PROVIDER_VOICE_INDEX_NAME,
    voice_clones.c.provider,
    voice_clones.c.provider_voice_id,
    unique=True,
)

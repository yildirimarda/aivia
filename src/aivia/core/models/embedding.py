"""Persisted embedding cache records.

An embedding vector is reusable only when the provider, model, input text,
dimension, input type, normalization flag and truncation flag are all
identical. The cache stores the canonical request digest next to the vector so
a later process can serve the input without contacting a provider.

The model only describes data. Hashing requests and deciding whether a hit is
allowed belongs to :mod:`aivia.providers.embeddings.cache`.
"""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, Field, field_validator

from aivia.core.models.base import ModelBase

__all__ = [
    "EmbeddingCacheEntry",
    "EmbeddingCacheRecord",
]


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


class EmbeddingCacheEntry(ModelBase):
    """One request digest pointing at the vector that satisfied it.

    ``text`` and the request flags are kept for audit and for rebuilding the
    key without the original request. ``vector`` is the stored embedding and
    ``hit_count`` counts lookups served from the entry.
    """

    cache_key: str = Field(
        min_length=1,
        validation_alias=AliasChoices(
            "cache_key",
            "key",
            "digest",
            "embedding_cache_key",
        ),
    )
    provider: str = Field(
        default="",
        validation_alias=AliasChoices("provider", "vendor"),
    )
    model: str = Field(
        default="",
        validation_alias=AliasChoices("model", "model_id"),
    )
    text: str = Field(
        default="",
        validation_alias=AliasChoices("text", "input", "content"),
    )
    dimensions: int | None = Field(
        default=None,
        validation_alias=AliasChoices("dimensions", "dimension"),
    )
    input_type: str = Field(
        default="",
        validation_alias=AliasChoices("input_type", "type"),
    )
    normalize: bool = Field(
        default=False,
        validation_alias=AliasChoices("normalize", "normalized"),
    )
    truncate: bool = Field(
        default=True,
        validation_alias=AliasChoices("truncate", "truncated"),
    )
    vector: list[float] = Field(
        default_factory=list,
        validation_alias=AliasChoices("vector", "embedding", "values"),
    )
    hit_count: int = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices("hit_count", "hits"),
    )

    @field_validator(
        "cache_key",
        "provider",
        "model",
        "text",
        "input_type",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("vector", mode="before")
    @classmethod
    def _coerce_vector(cls, value: Any) -> Any:
        return [] if value is None else value

    @property
    def embedding_cache_key(self) -> str:
        """Alias for :attr:`cache_key`."""
        return self.cache_key

    @property
    def hits(self) -> int:
        """Alias for :attr:`hit_count`."""
        return self.hit_count


EmbeddingCacheRecord = EmbeddingCacheEntry
"""Alias for callers that prefer the record wording."""

"""Embeddings configuration: chain, dimensions, batching, cache and prices.

Embedding calls are routed like every other capability, so the section owns an
ordered provider ``chain``. ``dimensions`` is the vector size requested from a
provider; leaving it unset uses the provider's own default. ``batch_size``
limits how many inputs one provider request carries and ``cache_size`` bounds
the in-memory vector cache. ``pricing`` maps a model id to USD per million
input tokens, matching the unit the embedding cost calculator expects.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from pydantic import Field, field_validator, model_validator

from aivia.core.config.base import ConfigModel
from aivia.core.config.capability import CapabilityChainConfig
from aivia.core.errors import ConfigError

__all__ = [
    "DEFAULT_EMBEDDING_BATCH_SIZE",
    "DEFAULT_EMBEDDING_CACHE_SIZE",
    "EmbeddingConfig",
    "EmbeddingsConfig",
]

DEFAULT_EMBEDDING_BATCH_SIZE = 64
"""The default number of inputs sent in one provider request."""

DEFAULT_EMBEDDING_CACHE_SIZE = 10_000
"""The default number of vectors kept in the in-memory cache."""

_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "chain": ("provider_chain", "providers", "order"),
    "dimensions": ("dimension", "vector_dimensions", "vector_size"),
    "batch_size": ("batch", "embed_batch_size", "max_batch_size"),
    "cache_size": (
        "cache",
        "cache_entries",
        "max_cache_entries",
        "cache_max_entries",
    ),
    "pricing": (
        "prices",
        "model_prices",
        "price_per_million",
    ),
}


class EmbeddingsConfig(ConfigModel):
    """Provider chain, vector dimensions, batching, cache size and model prices.

    ``chain`` is ordered from most to least preferred and rejects duplicates so
    a typo never silently changes routing. ``pricing`` values are USD per
    million input tokens; a model without an entry falls back to the provider
    metadata and the router's pricing overrides.
    """

    chain: list[str] = Field(default_factory=list)
    dimensions: int | None = Field(default=None, ge=1)
    batch_size: int = Field(default=DEFAULT_EMBEDDING_BATCH_SIZE, ge=1)
    cache_size: int = Field(default=DEFAULT_EMBEDDING_CACHE_SIZE, ge=1)
    pricing: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for canonical, aliases in _FIELD_ALIASES.items():
            for alias in aliases:
                if alias in values:
                    value = values.pop(alias)
                    values.setdefault(canonical, value)
        return values

    @field_validator("chain")
    @classmethod
    def _validate_chain(cls, value: list[str]) -> list[str]:
        chain: list[str] = []
        seen: set[str] = set()
        for entry in value:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError("chain entries must be non-empty provider names")
            provider = entry.strip()
            if provider in seen:
                raise ValueError(f"duplicate provider in chain: {provider}")
            seen.add(provider)
            chain.append(provider)
        return chain

    @field_validator("pricing")
    @classmethod
    def _validate_pricing(cls, value: dict[str, float]) -> dict[str, float]:
        pricing: dict[str, float] = {}
        for model, price in value.items():
            if not isinstance(model, str) or not model.strip():
                raise ValueError("pricing model names must be non-empty")
            if not math.isfinite(price):
                raise ValueError(f"price for {model} must be finite")
            if price < 0:
                raise ValueError(f"price for {model} must not be negative")
            pricing[model.strip()] = float(price)
        return pricing

    @property
    def provider_chain(self) -> list[str]:
        """Alias for :attr:`chain`."""
        return self.chain

    @property
    def batch(self) -> int:
        """Alias for :attr:`batch_size`."""
        return self.batch_size

    @property
    def cache_entries(self) -> int:
        """Alias for :attr:`cache_size`."""
        return self.cache_size

    @property
    def prices(self) -> dict[str, float]:
        """Alias for :attr:`pricing`."""
        return self.pricing

    @property
    def model_prices(self) -> dict[str, float]:
        """Alias for :attr:`pricing`."""
        return self.pricing

    def pricing_for(self, model: str) -> float:
        """Return the configured per-million price for *model*."""
        key = model.strip()
        try:
            return self.pricing[key]
        except KeyError:
            available = ", ".join(sorted(self.pricing)) or "none"
            raise ConfigError(
                f"no pricing configured for embeddings model: {model} "
                f"(available: {available})",
                details={"model": model, "available": list(self.pricing)},
            ) from None

    def to_capability_config(self) -> CapabilityChainConfig:
        """Return the router/estimator view of this section.

        The embedding section owns an ordered provider chain and per-model
        prices. The shared capability config accepts those prices as provider
        overrides, which is how the estimator consumes them.
        """
        return CapabilityChainConfig(
            chain=list(self.chain),
            pricing_overrides=dict(self.pricing),
        )

    def as_capability_config(self) -> CapabilityChainConfig:
        """Alias for :meth:`to_capability_config`."""
        return self.to_capability_config()


EmbeddingConfig = EmbeddingsConfig

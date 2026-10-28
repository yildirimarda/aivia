"""Reusable capability chain configuration shared by every provider capability.

A capability such as ``image_gen``, ``video_gen`` or ``tts`` selects its
implementations from an ordered ``chain``. ``strategy`` decides which candidate
wins, ``weights`` supplies optional per-provider relative weights for the
``weighted_ab`` strategy, ``pricing_overrides`` supplies local prices for
providers whose metadata is stale or missing, and the per-clip cap plus split
flag describe how long video requests may be divided into several provider
calls.

The model is deliberately independent of any concrete capability so LLM,
media and module sections can reuse it without inheriting their own defaults.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator, model_validator

from aivia.core.config.base import ConfigModel

__all__ = [
    "CapabilityChainConfig",
    "CapabilityConfig",
    "CapabilityStrategy",
    "ChainStrategy",
]


class CapabilityStrategy(StrEnum):
    """How the router picks a provider from a capability chain."""

    FIRST_AVAILABLE = "first_available"
    CHEAPEST = "cheapest"
    BEST_QUALITY = "best_quality"
    ROUND_ROBIN = "round_robin"
    WEIGHTED_AB = "weighted_ab"


ChainStrategy = CapabilityStrategy

_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "strategy": ("chain_strategy", "selection_strategy"),
    "chain": ("provider_chain", "providers", "order"),
    "weights": ("provider_weights", "chain_weights", "ab_weights"),
    "pricing_overrides": ("pricing", "prices", "price_overrides"),
    "max_clip_seconds": (
        "per_clip_cap",
        "clip_cap",
        "clip_cap_seconds",
        "max_clip",
        "max_clip_duration_seconds",
    ),
    "split_long_requests": (
        "split",
        "split_long",
        "split_long_clips",
        "split_clips",
        "allow_split",
    ),
}


class CapabilityChainConfig(ConfigModel):
    """Ordered providers, selection strategy and limits for one capability.

    ``max_clip_seconds`` is a cap of ``0`` when it is not configured; a positive
    value is the longest clip a single provider call may produce. When
    ``split_long_requests`` is enabled, a longer request may be divided into
    multiple clips instead of being clamped.
    """

    strategy: CapabilityStrategy = CapabilityStrategy.FIRST_AVAILABLE
    chain: list[str] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)
    pricing_overrides: dict[str, float] = Field(default_factory=dict)
    max_clip_seconds: float = Field(default=0.0, ge=0)
    split_long_requests: bool = False

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

    @field_validator("pricing_overrides")
    @classmethod
    def _validate_pricing_overrides(cls, value: dict[str, float]) -> dict[str, float]:
        overrides: dict[str, float] = {}
        for name, price in value.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("pricing override keys must be non-empty")
            if not math.isfinite(price):
                raise ValueError(f"pricing override for {name} must be finite")
            if price < 0:
                raise ValueError(f"pricing override for {name} must not be negative")
            overrides[name.strip()] = float(price)
        return overrides

    @field_validator("weights")
    @classmethod
    def _validate_weights(cls, value: dict[str, float]) -> dict[str, float]:
        weights: dict[str, float] = {}
        for name, weight in value.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("weight keys must be non-empty provider names")
            if not math.isfinite(weight):
                raise ValueError(f"weight for {name} must be finite")
            if weight < 0:
                raise ValueError(f"weight for {name} must not be negative")
            weights[name.strip()] = float(weight)
        return weights

    @model_validator(mode="after")
    def _require_weights_for_weighted_ab(self) -> CapabilityChainConfig:
        if self.strategy is CapabilityStrategy.WEIGHTED_AB and not self.weights:
            raise ValueError("weighted_ab strategy requires per-provider weights")
        if self.chain and self.weights:
            unknown = sorted(set(self.weights) - set(self.chain))
            if unknown:
                names = ", ".join(unknown)
                raise ValueError(
                    f"weights reference providers not in the chain: {names}"
                )
        return self

    @field_validator("max_clip_seconds")
    @classmethod
    def _validate_max_clip_seconds(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("max clip seconds must be finite")
        return value

    @property
    def per_clip_cap(self) -> float:
        """Alias for :attr:`max_clip_seconds`; ``0`` means no cap."""
        return self.max_clip_seconds

    @property
    def provider_weights(self) -> dict[str, float]:
        """Alias for :attr:`weights`."""
        return self.weights

    @property
    def weight_sum(self) -> float:
        """The sum of all configured provider weights."""
        return sum(self.weights.values())

    @property
    def split(self) -> bool:
        """Alias for :attr:`split_long_requests`."""
        return self.split_long_requests

    @property
    def pricing(self) -> dict[str, float]:
        """Alias for :attr:`pricing_overrides`."""
        return self.pricing_overrides


CapabilityConfig = CapabilityChainConfig

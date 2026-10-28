"""Provider metadata shared by the registry, router and documentation.

A :class:`ProviderMeta` describes one provider implementation: the capabilities
it can serve, the options it supports, the price of each capability, the rate
limit it enforces, its static quality rank and the licensing terms attached to
its outputs. The router uses this metadata to filter providers, estimate cost,
pick the best quality candidate and respect rate limits without knowing any
concrete provider class.

The models only describe data. They never call a provider, read configuration
or decide which provider to use.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import AliasChoices, Field, field_validator

from aivia.core.models.asset import AssetLicense
from aivia.core.models.base import ModelBase

__all__ = [
    "ProviderMeta",
    "ProviderMetadata",
    "ProviderPrice",
    "ProviderPricing",
    "RateLimit",
    "RateLimits",
]


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _coerce_license(value: Any) -> Any:
    if value is None:
        return {}
    if isinstance(value, AssetLicense):
        return value
    if isinstance(value, str):
        return {"name": value}
    return value


def _coerce_rate_limit(value: Any) -> Any:
    if value is None:
        return {}
    if isinstance(value, RateLimit):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {"requests_per_minute": value}
    return value


def _coerce_pricing(value: Any) -> Any:
    if value is None:
        return {}
    if isinstance(value, ProviderPrice):
        return {"default": value}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {"default": {"amount_usd": value}}
    if isinstance(value, Mapping):
        pricing: dict[str, Any] = {}
        for key, price in value.items():
            if isinstance(price, (int, float)) and not isinstance(price, bool):
                pricing[key] = {"amount_usd": price}
            else:
                pricing[key] = price
        return pricing
    return value


def _coerce_string_list(value: Any) -> Any:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return value
    return value


def _validate_string_list(values: Sequence[str], *, label: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        if not isinstance(raw, str):
            raise ValueError(f"{label} entries must be strings")
        entry = raw.strip()
        if not entry:
            raise ValueError(f"{label} entries must be non-empty")
        if entry not in seen:
            seen.add(entry)
            result.append(entry)
    return result


class ProviderPrice(ModelBase):
    """The price of one provider operation.

    ``amount_usd`` is the price for ``units_per_price`` units of ``unit``. The
    default covers one unit, so a price of ``0.04`` for a single image is
    expressed with ``unit="image"`` and ``units_per_price=1``. ``cost_for``
    scales the price for a requested number of units without guessing a unit
    conversion.
    """

    amount_usd: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices(
            "amount_usd",
            "price_usd",
            "amount",
            "price",
            "usd",
        ),
    )
    unit: str = Field(
        default="call",
        validation_alias=AliasChoices("unit", "unit_kind", "unit_type"),
    )
    units_per_price: float = Field(
        default=1.0,
        gt=0.0,
        validation_alias=AliasChoices("units_per_price", "per_units", "unit_count"),
    )
    currency: str = Field(
        default="USD",
        validation_alias=AliasChoices("currency", "currency_code"),
    )

    @field_validator("amount_usd", mode="before")
    @classmethod
    def _validate_amount(cls, value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("amount_usd must be finite")
        return value

    @field_validator("unit", "currency", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("currency")
    @classmethod
    def _normalize_currency(cls, value: str) -> str:
        return value.upper()

    def cost_for(self, units: float) -> float:
        """Return the price for *units*, rejecting invalid quantities."""
        if not math.isfinite(units) or units < 0:
            raise ValueError("units must be a finite, non-negative number")
        return self.amount_usd * units / self.units_per_price

    @property
    def price_usd(self) -> float:
        """Alias for :attr:`amount_usd`."""
        return self.amount_usd

    @property
    def price(self) -> float:
        """Alias for :attr:`amount_usd`."""
        return self.amount_usd


class RateLimit(ModelBase):
    """The request budget a provider allows.

    ``requests_per_minute`` is the sustained rate. ``burst`` is the largest
    short-term allowance; ``0`` means the sustained rate is also the burst.
    ``max_concurrency`` limits simultaneous calls, where ``0`` means unlimited.
    """

    requests_per_minute: int = Field(
        default=60,
        ge=0,
        validation_alias=AliasChoices(
            "requests_per_minute",
            "per_minute",
            "rpm",
            "rate_limit_per_minute",
        ),
    )
    burst: int = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices("burst", "burst_size", "max_burst"),
    )
    max_concurrency: int = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices(
            "max_concurrency",
            "concurrency",
            "max_parallel",
        ),
    )

    @property
    def requests_per_second(self) -> float:
        """The sustained rate expressed per second."""
        return self.requests_per_minute / 60.0

    @property
    def effective_burst(self) -> int:
        """The burst allowance, falling back to the sustained rate."""
        return self.burst or self.requests_per_minute

    @property
    def is_unlimited(self) -> bool:
        """Whether the provider declares no sustained request limit."""
        return self.requests_per_minute == 0


class ProviderMeta(ModelBase):
    """Metadata for one provider implementation.

    ``capabilities`` names every operation the provider can serve.
    ``pricing`` maps a capability (or ``default``) to its price.
    ``supported_options`` lists the option values accepted per capability.
    ``quality_rank`` is a static ranking used before human ratings exist;
    ``None`` means the provider has no rank yet. ``license`` and
    ``commercial_use_notes`` describe the rights attached to generated output.
    """

    name: str = Field(
        default="",
        validation_alias=AliasChoices("name", "provider", "provider_name"),
    )
    capabilities: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("capabilities", "capability", "supports"),
    )
    supported_options: dict[str, list[str]] = Field(
        default_factory=dict,
        validation_alias=AliasChoices(
            "supported_options",
            "options",
            "capability_options",
        ),
    )
    pricing: dict[str, ProviderPrice] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("pricing", "prices"),
    )
    rate_limit: RateLimit = Field(
        default_factory=RateLimit,
        validation_alias=AliasChoices("rate_limit", "rate_limits", "limits"),
    )
    quality_rank: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("quality_rank", "quality", "rank"),
    )
    license: AssetLicense = Field(
        default_factory=AssetLicense,
        validation_alias=AliasChoices("license", "licence", "licensing"),
    )
    commercial_use_notes: str = Field(
        default="",
        validation_alias=AliasChoices(
            "commercial_use_notes",
            "commercial_notes",
            "licensing_notes",
        ),
    )

    @field_validator("name", "commercial_use_notes", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("capabilities", mode="before")
    @classmethod
    def _coerce_capabilities(cls, value: Any) -> Any:
        return _coerce_string_list(value)

    @field_validator("capabilities")
    @classmethod
    def _validate_capabilities(cls, value: list[str]) -> list[str]:
        return _validate_string_list(value, label="capability")

    @field_validator("supported_options", mode="before")
    @classmethod
    def _coerce_supported_options(cls, value: Any) -> Any:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            return value
        return {key: _coerce_string_list(options) for key, options in value.items()}

    @field_validator("supported_options")
    @classmethod
    def _validate_supported_options(
        cls, value: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        options: dict[str, list[str]] = {}
        for capability, entries in value.items():
            key = capability.strip()
            if not key:
                raise ValueError("supported option keys must be non-empty")
            options[key] = _validate_string_list(entries, label=f"option for {key}")
        return options

    @field_validator("pricing", mode="before")
    @classmethod
    def _coerce_pricing_values(cls, value: Any) -> Any:
        return _coerce_pricing(value)

    @field_validator("pricing")
    @classmethod
    def _validate_pricing(
        cls, value: dict[str, ProviderPrice]
    ) -> dict[str, ProviderPrice]:
        pricing: dict[str, ProviderPrice] = {}
        for capability, price in value.items():
            key = capability.strip()
            if not key:
                raise ValueError("pricing keys must be non-empty")
            pricing[key] = price
        return pricing

    @field_validator("rate_limit", mode="before")
    @classmethod
    def _coerce_rate_limit_value(cls, value: Any) -> Any:
        return _coerce_rate_limit(value)

    @field_validator("license", mode="before")
    @classmethod
    def _coerce_license_value(cls, value: Any) -> Any:
        return _coerce_license(value)

    @property
    def capability(self) -> str:
        """The first declared capability, or an empty string."""
        return self.capabilities[0] if self.capabilities else ""

    @property
    def primary_capability(self) -> str:
        """Alias for :attr:`capability`."""
        return self.capability

    @property
    def has_quality_rank(self) -> bool:
        """Whether a static quality rank is declared."""
        return self.quality_rank is not None

    @property
    def is_commercial(self) -> bool:
        """Whether the provider license explicitly allows commercial use."""
        return self.license.is_commercial

    @property
    def rate_limit_per_minute(self) -> int:
        """Alias for :attr:`RateLimit.requests_per_minute`."""
        return self.rate_limit.requests_per_minute

    def supports(self, capability: str) -> bool:
        """Return whether this provider declares *capability*."""
        return capability.strip() in self.capabilities

    def pricing_for(self, capability: str) -> ProviderPrice | None:
        """Return the price for *capability*, falling back to ``default``."""
        key = capability.strip()
        price = self.pricing.get(key)
        if price is not None:
            return price
        return self.pricing.get("default")

    def price_for(self, capability: str) -> ProviderPrice | None:
        """Alias for :meth:`pricing_for`."""
        return self.pricing_for(capability)


ProviderPricing = ProviderPrice
"""Alias for :class:`ProviderPrice` for callers that prefer the longer name."""

ProviderMetadata = ProviderMeta
"""Alias for :class:`ProviderMeta` for callers that prefer the longer name."""

RateLimits = RateLimit
"""Alias for :class:`RateLimit` for callers that use the plural form."""

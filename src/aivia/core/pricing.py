"""Resolve the USD price of one provider call.

A price comes from one of two places. An explicit override keyed by the
configured provider name wins, because it is how an operator corrects a stale
or missing upstream price. An override replaces the metadata amount but keeps
its unit and bulk factor when metadata exists; without metadata it is a
per-call price. Otherwise the provider's
:class:`~aivia.core.models.provider.ProviderMeta` price for the operation is
used, including its ``unit`` and ``units_per_price``.

Requested quantities are converted into the price's unit before the amount is
calculated. A price expressed per second can therefore price a request given
in minutes, and a price expressed per thousand tokens can price a token count
without the caller repeating the arithmetic.

A missing price is not a crash. The resolver returns a zero-cost resolution
flagged ``missing`` and emits :class:`MissingPriceWarning`, so the media
service can record the call and keep the run visible instead of aborting. The
resolver never calls a provider, enforces a budget or writes the cost ledger.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from aivia.core.errors import ConfigError
from aivia.core.models.provider import ProviderMeta, ProviderPrice
from aivia.providers.base import Provider

__all__ = [
    "MissingPriceWarning",
    "PriceResolution",
    "PriceResolver",
    "PriceSource",
    "PricingResolver",
    "PricingResult",
    "PricingSource",
    "ResolvedPrice",
    "ResolvedPricing",
    "convert_units",
    "resolve_price",
    "resolve_pricing",
]


class MissingPriceWarning(UserWarning):
    """A provider has no price for an operation, so a zero-cost call is used.

    The warning instance carries the provider and operation that lacked a
    price, so a caller that records warnings as events does not have to parse
    the message.
    """

    def __init__(self, provider: str, operation: str) -> None:
        self.provider = provider
        self.operation = operation
        self.message = (
            f"provider {provider!r} has no price for {operation!r}; "
            "recording a zero-cost call"
        )
        super().__init__(self.message)


class PriceSource(StrEnum):
    """Where a resolved price came from."""

    METADATA = "metadata"
    OVERRIDE = "override"
    MISSING = "missing"


PricingSource = PriceSource

_Unit = tuple[str, float]

_UNITS: dict[str, _Unit] = {
    "call": ("call", 1.0),
    "calls": ("call", 1.0),
    "request": ("call", 1.0),
    "requests": ("call", 1.0),
    "operation": ("call", 1.0),
    "operations": ("call", 1.0),
    "unit": ("call", 1.0),
    "units": ("call", 1.0),
    "image": ("image", 1.0),
    "images": ("image", 1.0),
    "picture": ("image", 1.0),
    "pictures": ("image", 1.0),
    "clip": ("clip", 1.0),
    "clips": ("clip", 1.0),
    "video": ("clip", 1.0),
    "videos": ("clip", 1.0),
    "track": ("clip", 1.0),
    "tracks": ("clip", 1.0),
    "millisecond": ("time", 0.001),
    "milliseconds": ("time", 0.001),
    "ms": ("time", 0.001),
    "second": ("time", 1.0),
    "seconds": ("time", 1.0),
    "sec": ("time", 1.0),
    "secs": ("time", 1.0),
    "s": ("time", 1.0),
    "minute": ("time", 60.0),
    "minutes": ("time", 60.0),
    "min": ("time", 60.0),
    "mins": ("time", 60.0),
    "hour": ("time", 3600.0),
    "hours": ("time", 3600.0),
    "hr": ("time", 3600.0),
    "hrs": ("time", 3600.0),
    "h": ("time", 3600.0),
    "day": ("time", 86400.0),
    "days": ("time", 86400.0),
    "token": ("token", 1.0),
    "tokens": ("token", 1.0),
    "thousand_token": ("token", 1000.0),
    "thousand_tokens": ("token", 1000.0),
    "1k_token": ("token", 1000.0),
    "1k_tokens": ("token", 1000.0),
    "ktoken": ("token", 1000.0),
    "ktokens": ("token", 1000.0),
    "million_token": ("token", 1_000_000.0),
    "million_tokens": ("token", 1_000_000.0),
    "1m_token": ("token", 1_000_000.0),
    "1m_tokens": ("token", 1_000_000.0),
    "mtoken": ("token", 1_000_000.0),
    "mtokens": ("token", 1_000_000.0),
    "character": ("character", 1.0),
    "characters": ("character", 1.0),
    "char": ("character", 1.0),
    "chars": ("character", 1.0),
    "thousand_character": ("character", 1000.0),
    "thousand_characters": ("character", 1000.0),
    "thousand_char": ("character", 1000.0),
    "thousand_chars": ("character", 1000.0),
    "1k_character": ("character", 1000.0),
    "1k_characters": ("character", 1000.0),
    "1k_char": ("character", 1000.0),
    "1k_chars": ("character", 1000.0),
    "kchar": ("character", 1000.0),
    "kchars": ("character", 1000.0),
    "million_character": ("character", 1_000_000.0),
    "million_characters": ("character", 1_000_000.0),
    "million_char": ("character", 1_000_000.0),
    "million_chars": ("character", 1_000_000.0),
    "1m_character": ("character", 1_000_000.0),
    "1m_characters": ("character", 1_000_000.0),
    "1m_char": ("character", 1_000_000.0),
    "1m_chars": ("character", 1_000_000.0),
    "word": ("word", 1.0),
    "words": ("word", 1.0),
    "frame": ("frame", 1.0),
    "frames": ("frame", 1.0),
    "megapixel": ("megapixel", 1.0),
    "megapixels": ("megapixel", 1.0),
    "mp": ("megapixel", 1.0),
    "byte": ("byte", 1.0),
    "bytes": ("byte", 1.0),
    "kilobyte": ("byte", 1000.0),
    "kilobytes": ("byte", 1000.0),
    "kb": ("byte", 1000.0),
    "megabyte": ("byte", 1_000_000.0),
    "megabytes": ("byte", 1_000_000.0),
    "mb": ("byte", 1_000_000.0),
    "gigabyte": ("byte", 1_000_000_000.0),
    "gigabytes": ("byte", 1_000_000_000.0),
    "gb": ("byte", 1_000_000_000.0),
}


def _coerce_units(value: object, *, label: str = "units") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            f"{label} must be a finite, non-negative number",
            details={label: value},
        )
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ConfigError(
            f"{label} must be a finite, non-negative number",
            details={label: value},
        )
    return number


def _unit_key(unit: str) -> str:
    if not isinstance(unit, str):
        raise ConfigError(
            "pricing unit must be a non-empty string",
            details={"unit": unit},
        )
    key = "_".join(unit.strip().lower().replace("-", " ").split())
    if not key:
        raise ConfigError(
            "pricing unit must be a non-empty string",
            details={"unit": unit},
        )
    return key


def _resolve_unit(unit: str) -> _Unit:
    key = _unit_key(unit)
    try:
        return _UNITS[key]
    except KeyError:
        raise ConfigError(
            f"unknown pricing unit: {unit!r}",
            details={"unit": unit, "known": sorted(_UNITS)},
        ) from None


def convert_units(units: float, from_unit: str, to_unit: str) -> float:
    """Convert *units* from one pricing unit to another.

    Units of the same dimension convert through a canonical base unit, so
    ``convert_units(2, "minutes", "seconds")`` is ``120``. A generic ``call``
    unit is treated as a raw count and is compatible with every unit, which
    keeps a per-call price usable for an image or clip count. Unknown units and
    incompatible dimensions raise :class:`~aivia.core.errors.ConfigError`.
    """
    quantity = _coerce_units(units)
    from_key = _unit_key(from_unit)
    to_key = _unit_key(to_unit)
    if from_key == to_key:
        return quantity
    from_dimension, from_factor = _resolve_unit(from_key)
    to_dimension, to_factor = _resolve_unit(to_key)
    if from_dimension == to_dimension:
        return quantity * from_factor / to_factor
    if "call" in (from_dimension, to_dimension):
        return quantity
    raise ConfigError(
        f"cannot convert pricing units from {from_unit!r} to {to_unit!r}",
        details={
            "from_unit": from_unit,
            "to_unit": to_unit,
            "from_dimension": from_dimension,
            "to_dimension": to_dimension,
        },
    )


@dataclass(frozen=True, slots=True)
class ResolvedPrice:
    """The price of one provider call after overrides and unit conversion.

    ``units`` is the quantity the caller asked to price and ``billed_units`` is
    the same quantity expressed in the price's own unit. ``missing`` is true
    when no price was known; the amount is then zero and
    :class:`MissingPriceWarning` was emitted.
    """

    provider: str
    operation: str
    units: float
    unit_kind: str
    amount_usd: float
    source: PriceSource
    billed_units: float = 0.0
    price_unit: str = "call"
    units_per_price: float = 1.0
    currency: str = "USD"
    pricing_version: str = ""
    missing: bool = False

    @property
    def amount(self) -> float:
        """Alias for :attr:`amount_usd`."""
        return self.amount_usd

    @property
    def cost_usd(self) -> float:
        """Alias for :attr:`amount_usd`."""
        return self.amount_usd

    @property
    def price(self) -> float:
        """Alias for :attr:`amount_usd`."""
        return self.amount_usd

    @property
    def price_usd(self) -> float:
        """Alias for :attr:`amount_usd`."""
        return self.amount_usd

    @property
    def unit(self) -> str:
        """Alias for :attr:`unit_kind`."""
        return self.unit_kind

    @property
    def is_missing(self) -> bool:
        """Whether no price was available for the call."""
        return self.missing

    @property
    def is_override(self) -> bool:
        """Whether an explicit override supplied the price."""
        return self.source is PriceSource.OVERRIDE

    @property
    def is_metadata(self) -> bool:
        """Whether provider metadata supplied the price."""
        return self.source is PriceSource.METADATA

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for logs and sidecars."""
        return {
            "provider": self.provider,
            "operation": self.operation,
            "units": self.units,
            "unit_kind": self.unit_kind,
            "billed_units": self.billed_units,
            "amount_usd": self.amount_usd,
            "source": self.source.value,
            "price_unit": self.price_unit,
            "units_per_price": self.units_per_price,
            "currency": self.currency,
            "pricing_version": self.pricing_version,
            "missing": self.missing,
        }


PriceResolution = ResolvedPrice
PricingResult = ResolvedPrice
ResolvedPricing = ResolvedPrice


def _coerce_overrides(
    value: Mapping[str, float] | None,
) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(
            "pricing overrides must be a mapping of provider name to price",
            details={"overrides": type(value).__name__},
        )
    overrides: dict[str, float] = {}
    for name, price in value.items():
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(
                "pricing override names must be non-empty strings",
                details={"name": name},
            )
        if isinstance(price, bool) or not isinstance(price, (int, float)):
            raise ConfigError(
                f"pricing override for {name!r} must be a finite number",
                details={"provider": name, "price": price},
            )
        amount = float(price)
        if not math.isfinite(amount) or amount < 0:
            raise ConfigError(
                f"pricing override for {name!r} must be finite and non-negative",
                details={"provider": name, "price": price},
            )
        overrides[name.strip()] = amount
    return overrides


def _provider_meta(provider: Provider | ProviderMeta) -> tuple[ProviderMeta, str]:
    if isinstance(provider, ProviderMeta):
        return provider, provider.name
    if isinstance(provider, Provider):
        return provider.metadata, provider.name
    raise TypeError(
        f"provider must be a Provider or ProviderMeta, not {type(provider).__name__}"
    )


def _override_price(
    amount_usd: float,
    metadata_price: ProviderPrice | None,
) -> ProviderPrice:
    """Replace a metadata price's amount while keeping its unit semantics."""
    if metadata_price is None:
        return ProviderPrice(amount_usd=amount_usd)
    return ProviderPrice(
        amount_usd=amount_usd,
        unit=metadata_price.unit,
        units_per_price=metadata_price.units_per_price,
        currency=metadata_price.currency,
    )


class PricingResolver:
    """Resolve provider metadata and local overrides into a call price.

    ``overrides`` maps a configured provider name to a USD price per unit and
    wins over metadata. ``pricing_version`` is copied onto every resolution so
    a cost event can record which price table produced the amount.
    """

    def __init__(
        self,
        overrides: Mapping[str, float] | None = None,
        *,
        pricing_version: str = "",
    ) -> None:
        if not isinstance(pricing_version, str):
            raise ConfigError(
                "pricing version must be a string",
                details={"pricing_version": pricing_version},
            )
        self._overrides = _coerce_overrides(overrides)
        self._pricing_version = pricing_version.strip()

    @property
    def overrides(self) -> dict[str, float]:
        """A copy of the per-provider USD overrides."""
        return dict(self._overrides)

    @property
    def pricing_version(self) -> str:
        """The version copied onto every resolved price."""
        return self._pricing_version

    def resolve(
        self,
        provider: Provider | ProviderMeta,
        operation: str,
        units: float = 1.0,
        unit_kind: str | None = None,
        *,
        provider_name: str | None = None,
    ) -> ResolvedPrice:
        """Return the price of *operation* for *units* of *unit_kind*.

        The configured *provider_name* is used to look up overrides and may
        differ from the metadata name when one adapter serves several
        instances. When *unit_kind* is omitted, the price's own unit is used,
        so a caller only has to name the unit when it differs from the price.
        A missing price emits :class:`MissingPriceWarning` and returns a
        zero-cost resolution instead of raising.
        """
        metadata, default_name = _provider_meta(provider)
        name = (provider_name if provider_name is not None else default_name).strip()
        if not name:
            raise ConfigError(
                "provider name must be a non-empty string",
                details={"provider": default_name},
            )
        if not isinstance(operation, str) or not operation.strip():
            raise ConfigError(
                "operation must be a non-empty string",
                details={"operation": operation},
            )
        target = operation.strip()
        quantity = _coerce_units(units)

        metadata_price = metadata.pricing_for(target)
        override = self._overrides.get(name)
        price: ProviderPrice | None
        if override is not None:
            price = _override_price(override, metadata_price)
            source = PriceSource.OVERRIDE
        else:
            price = metadata_price
            source = PriceSource.METADATA
            if price is None:
                requested = _unit_key(unit_kind) if unit_kind is not None else "call"
                warnings.warn(
                    MissingPriceWarning(name, target),
                    stacklevel=2,
                )
                return ResolvedPrice(
                    provider=name,
                    operation=target,
                    units=quantity,
                    unit_kind=requested,
                    billed_units=quantity,
                    amount_usd=0.0,
                    source=PriceSource.MISSING,
                    pricing_version=self._pricing_version,
                    missing=True,
                )

        requested = _unit_key(unit_kind) if unit_kind is not None else price.unit
        billed = convert_units(quantity, requested, price.unit)
        return ResolvedPrice(
            provider=name,
            operation=target,
            units=quantity,
            unit_kind=requested,
            billed_units=billed,
            amount_usd=price.cost_for(billed),
            source=source,
            price_unit=price.unit,
            units_per_price=price.units_per_price,
            currency=price.currency,
            pricing_version=self._pricing_version,
        )

    def resolve_price(
        self,
        provider: Provider | ProviderMeta,
        operation: str,
        units: float = 1.0,
        unit_kind: str | None = None,
        *,
        provider_name: str | None = None,
    ) -> ResolvedPrice:
        """Alias for :meth:`resolve`."""
        return self.resolve(
            provider,
            operation,
            units,
            unit_kind=unit_kind,
            provider_name=provider_name,
        )

    def price_for(
        self,
        provider: Provider | ProviderMeta,
        operation: str,
        units: float = 1.0,
        unit_kind: str | None = None,
        *,
        provider_name: str | None = None,
    ) -> float:
        """Return only the resolved USD amount."""
        return self.resolve(
            provider,
            operation,
            units,
            unit_kind=unit_kind,
            provider_name=provider_name,
        ).amount_usd


PriceResolver = PricingResolver


def resolve_price(
    provider: Provider | ProviderMeta,
    operation: str,
    units: float = 1.0,
    unit_kind: str | None = None,
    *,
    overrides: Mapping[str, float] | None = None,
    provider_name: str | None = None,
    pricing_version: str = "",
) -> ResolvedPrice:
    """Resolve one price without constructing a long-lived resolver."""
    return PricingResolver(
        overrides,
        pricing_version=pricing_version,
    ).resolve(
        provider,
        operation,
        units,
        unit_kind=unit_kind,
        provider_name=provider_name,
    )


resolve_pricing = resolve_price

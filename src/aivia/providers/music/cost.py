"""Price a music generation call and record the cost event.

Music providers quote either a per-second rate for the generated audio or a
flat price for every track. This module resolves the price from the provider's
metadata or a configured override, converts the requested quantity into the
price's own unit and turns it into a
:class:`~aivia.core.models.cost.CostEvent`. It never calls a provider and never
enforces a budget.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from aivia.core.cost import CostTracker
from aivia.core.errors import ConfigError
from aivia.core.models.cost import CostEvent
from aivia.core.models.provider import ProviderMeta
from aivia.core.pricing import PricingResolver, ResolvedPrice
from aivia.providers.base import Provider
from aivia.providers.protocols.music import MusicResponse

__all__ = [
    "MUSIC_OPERATION",
    "MUSIC_UNIT",
    "TRACK_UNIT",
    "MusicCost",
    "MusicCostCalculator",
    "MusicPrice",
    "music_cost_event",
]

MUSIC_OPERATION = "music_gen"
"""The cost event operation for one music generation call."""

MUSIC_UNIT = "second"
"""The default cost unit for generated music, matching per-second pricing."""

TRACK_UNIT = "track"
"""The cost unit used when a provider prices every generated track."""

MusicPrice = ResolvedPrice
"""Alias for the resolved price of one music call."""

MusicCost = ResolvedPrice
"""Alias for the resolved cost of one music call."""


def _units(value: object, *, label: str = "units") -> float:
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


class MusicCostCalculator:
    """Resolve music prices and build or record their cost events."""

    def __init__(
        self,
        overrides: Mapping[str, float] | None = None,
        *,
        pricing_version: str = "",
    ) -> None:
        self._resolver = PricingResolver(
            overrides,
            pricing_version=pricing_version,
        )

    @property
    def resolver(self) -> PricingResolver:
        """The resolver used for every price lookup."""
        return self._resolver

    @property
    def pricing_version(self) -> str:
        """The version copied onto every cost event."""
        return self._resolver.pricing_version

    def resolve(
        self,
        provider: Provider | ProviderMeta,
        units: float,
        *,
        provider_name: str | None = None,
        unit_kind: str | None = MUSIC_UNIT,
        operation: str = MUSIC_OPERATION,
    ) -> ResolvedPrice:
        """Return the price of *units* of music from *provider*."""
        return self._resolver.resolve(
            provider,
            operation,
            _units(units),
            unit_kind=unit_kind,
            provider_name=provider_name,
        )

    def cost_for(
        self,
        provider: Provider | ProviderMeta,
        units: float,
        *,
        provider_name: str | None = None,
        unit_kind: str | None = MUSIC_UNIT,
        operation: str = MUSIC_OPERATION,
    ) -> ResolvedPrice:
        """Alias for :meth:`resolve`."""
        return self.resolve(
            provider,
            units,
            provider_name=provider_name,
            unit_kind=unit_kind,
            operation=operation,
        )

    def cost_event(
        self,
        provider: Provider | ProviderMeta,
        *,
        units: float,
        unit_kind: str | None = MUSIC_UNIT,
        response: MusicResponse | None = None,
        track_count: int | None = None,
        provider_name: str | None = None,
        run_id: str = "",
        operation: str = MUSIC_OPERATION,
        estimated: bool = False,
        metadata: Mapping[str, Any] | None = None,
        price: ResolvedPrice | None = None,
    ) -> CostEvent:
        """Build the cost event for one music generation call."""
        if response is not None and not isinstance(response, MusicResponse):
            raise TypeError("response must be a MusicResponse or None")
        resolved = (
            price
            if price is not None
            else self.resolve(
                provider,
                units,
                provider_name=provider_name,
                unit_kind=unit_kind,
                operation=operation,
            )
        )
        details: dict[str, Any] = {
            "billed_units": resolved.billed_units,
            "price_unit": resolved.price_unit,
            "price_source": resolved.source.value,
        }
        if resolved.missing:
            details["missing_price"] = True
        count = track_count
        if count is None and response is not None:
            count = response.count
        if count is not None:
            details["track_count"] = count
        if response is not None:
            if response.request_id:
                details["request_id"] = response.request_id
            if response.model:
                details["model"] = response.model
        if metadata is not None:
            details.update(metadata)
        return CostEvent(
            provider=resolved.provider,
            model=response.model if response is not None else "",
            operation=operation,
            units=resolved.units,
            unit_kind=resolved.unit_kind,
            amount_usd=resolved.amount_usd,
            estimated=estimated,
            run_id=run_id,
            pricing_version=resolved.pricing_version,
            metadata=details,
        )

    def record(
        self,
        provider: Provider | ProviderMeta,
        tracker: CostTracker,
        *,
        units: float,
        unit_kind: str | None = MUSIC_UNIT,
        response: MusicResponse | None = None,
        track_count: int | None = None,
        provider_name: str | None = None,
        run_id: str = "",
        operation: str = MUSIC_OPERATION,
        estimated: bool = False,
        metadata: Mapping[str, Any] | None = None,
        price: ResolvedPrice | None = None,
    ) -> CostEvent:
        """Record the cost event for one music generation call."""
        if not isinstance(tracker, CostTracker):
            raise TypeError("tracker must be a CostTracker")
        event = self.cost_event(
            provider,
            units=units,
            unit_kind=unit_kind,
            response=response,
            track_count=track_count,
            provider_name=provider_name,
            run_id=run_id,
            operation=operation,
            estimated=estimated,
            metadata=metadata,
            price=price,
        )
        if estimated:
            return tracker.record_estimate(event)
        return tracker.record_actual(event)


def music_cost_event(
    provider: Provider | ProviderMeta,
    *,
    units: float,
    overrides: Mapping[str, float] | None = None,
    pricing_version: str = "",
    **values: Any,
) -> CostEvent:
    """Build one music cost event without constructing a calculator."""
    calculator = MusicCostCalculator(overrides, pricing_version=pricing_version)
    return calculator.cost_event(provider, units=units, **values)

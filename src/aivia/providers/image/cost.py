"""Price a routed image generation call and record it in the cost ledger.

Image providers charge per generated image. The price comes from the provider's
:class:`~aivia.core.models.provider.ProviderMeta` metadata for the ``image_gen``
capability, with a configured per-provider override winning when an operator
corrected a stale or missing upstream price. A missing price still records a
zero-cost event and emits :class:`~aivia.core.pricing.MissingPriceWarning`, so
an image never disappears from the ledger just because its price was not
configured.

The calculator only resolves prices and builds events. The image service calls
it after a successful routed request, so an event always names the provider that
actually served the call.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aivia.core.cost import CostTracker
from aivia.core.models.cost import CostEvent
from aivia.core.models.provider import ProviderMeta
from aivia.core.pricing import PricingResolver, ResolvedPrice
from aivia.providers.base import Provider
from aivia.providers.protocols.image import CAPABILITY_IMAGE_GEN, ImageResponse

__all__ = [
    "IMAGE_OPERATION",
    "IMAGE_UNIT",
    "ImageCost",
    "ImageCostCalculator",
    "ImagePrice",
    "image_cost_event",
]

IMAGE_OPERATION = CAPABILITY_IMAGE_GEN
"""The default cost event operation for one image generation call."""

IMAGE_UNIT = "image"
"""The default cost unit for one generated image."""

ImagePrice = ResolvedPrice
"""Alias for the resolved price of one image generation call."""

ImageCost = ResolvedPrice
"""Alias for the resolved cost of one image generation call."""


class ImageCostCalculator:
    """Resolve image prices and build or record their cost events."""

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
        *,
        provider_name: str | None = None,
        units: float = 1.0,
        unit_kind: str | None = IMAGE_UNIT,
        operation: str = IMAGE_OPERATION,
    ) -> ResolvedPrice:
        """Return the price of *units* generated images served by *provider*."""
        return self._resolver.resolve(
            provider,
            operation,
            units,
            unit_kind=unit_kind,
            provider_name=provider_name,
        )

    def cost_for(
        self,
        provider: Provider | ProviderMeta,
        *,
        provider_name: str | None = None,
        units: float = 1.0,
        unit_kind: str | None = IMAGE_UNIT,
        operation: str = IMAGE_OPERATION,
    ) -> ResolvedPrice:
        """Alias for :meth:`resolve`."""
        return self.resolve(
            provider,
            provider_name=provider_name,
            units=units,
            unit_kind=unit_kind,
            operation=operation,
        )

    def cost_event(
        self,
        provider: Provider | ProviderMeta,
        *,
        provider_name: str | None = None,
        response: ImageResponse | None = None,
        run_id: str = "",
        operation: str = IMAGE_OPERATION,
        estimated: bool = False,
        units: float = 1.0,
        unit_kind: str | None = IMAGE_UNIT,
        metadata: Mapping[str, Any] | None = None,
        price: ResolvedPrice | None = None,
    ) -> CostEvent:
        """Build the cost event for one image generation call.

        *price* lets a caller reuse a resolution it already made, so a missing
        price warning is emitted once instead of twice. When *response* is
        given, its model, image count and request id are copied into the event
        metadata for provenance. Prompt text is never copied.
        """
        if response is not None and not isinstance(response, ImageResponse):
            raise TypeError("response must be an ImageResponse or None")
        resolved = (
            price
            if price is not None
            else self.resolve(
                provider,
                provider_name=provider_name,
                units=units,
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
        if response is not None:
            details["image_count"] = response.count
            if response.model:
                details["model"] = response.model
            if response.request_id:
                details["request_id"] = response.request_id
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
        provider_name: str | None = None,
        response: ImageResponse | None = None,
        run_id: str = "",
        operation: str = IMAGE_OPERATION,
        estimated: bool = False,
        units: float = 1.0,
        unit_kind: str | None = IMAGE_UNIT,
        metadata: Mapping[str, Any] | None = None,
        price: ResolvedPrice | None = None,
    ) -> CostEvent:
        """Record the cost event for one image generation call."""
        if not isinstance(tracker, CostTracker):
            raise TypeError("tracker must be a CostTracker")
        event = self.cost_event(
            provider,
            provider_name=provider_name,
            response=response,
            run_id=run_id,
            operation=operation,
            estimated=estimated,
            units=units,
            unit_kind=unit_kind,
            metadata=metadata,
            price=price,
        )
        if estimated:
            return tracker.record_estimate(event)
        return tracker.record_actual(event)

    def record_event(
        self,
        provider: Provider | ProviderMeta,
        tracker: CostTracker,
        **values: Any,
    ) -> CostEvent:
        """Alias for :meth:`record`."""
        return self.record(provider, tracker, **values)


def image_cost_event(
    provider: Provider | ProviderMeta,
    *,
    overrides: Mapping[str, float] | None = None,
    pricing_version: str = "",
    **values: Any,
) -> CostEvent:
    """Build one image cost event without constructing a calculator."""
    return ImageCostCalculator(
        overrides,
        pricing_version=pricing_version,
    ).cost_event(provider, **values)

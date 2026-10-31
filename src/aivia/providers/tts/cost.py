"""Price a text to speech call per character and record the cost event.

Text to speech is billed by the number of input characters, not by the length
of the encoded audio. This module resolves a provider's per-character price
from its metadata or a configured override and turns the character count into a
:class:`~aivia.core.models.cost.CostEvent`. The calculator never calls a
provider and never enforces a budget.
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
from aivia.providers.protocols.tts import TTSResponse

__all__ = [
    "CHARACTER_UNIT",
    "TTS_OPERATION",
    "TTSCost",
    "TTSCostCalculator",
    "TTSPrice",
    "tts_cost_event",
]

TTS_OPERATION = "tts"
"""The cost event operation for one text to speech call."""

CHARACTER_UNIT = "character"
"""The cost unit for text to speech input."""

TTSPrice = ResolvedPrice
"""Alias for the resolved price of one text to speech call."""

TTSCost = ResolvedPrice
"""Alias for the resolved cost of one text to speech call."""


def _characters(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            "characters must be a finite, non-negative number",
            details={"characters": value},
        )
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ConfigError(
            "characters must be a finite, non-negative number",
            details={"characters": value},
        )
    return number


def _characters_from_response(response: TTSResponse) -> float:
    raw = response.metadata.get("characters")
    if raw is None:
        raise ConfigError(
            "text to speech response has no character count",
            details={"provider": response.provider},
        )
    return _characters(raw)


class TTSCostCalculator:
    """Resolve per-character prices and build or record their cost events."""

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
        characters: float,
        *,
        provider_name: str | None = None,
        unit_kind: str | None = CHARACTER_UNIT,
        operation: str = TTS_OPERATION,
    ) -> ResolvedPrice:
        """Return the price of *characters* spoken by *provider*."""
        return self._resolver.resolve(
            provider,
            operation,
            _characters(characters),
            unit_kind=unit_kind,
            provider_name=provider_name,
        )

    def cost_for(
        self,
        provider: Provider | ProviderMeta,
        characters: float,
        *,
        provider_name: str | None = None,
        unit_kind: str | None = CHARACTER_UNIT,
        operation: str = TTS_OPERATION,
    ) -> ResolvedPrice:
        """Alias for :meth:`resolve`."""
        return self.resolve(
            provider,
            characters,
            provider_name=provider_name,
            unit_kind=unit_kind,
            operation=operation,
        )

    def cost_event(
        self,
        provider: Provider | ProviderMeta,
        *,
        characters: float | None = None,
        response: TTSResponse | None = None,
        provider_name: str | None = None,
        run_id: str = "",
        operation: str = TTS_OPERATION,
        estimated: bool = False,
        unit_kind: str | None = CHARACTER_UNIT,
        metadata: Mapping[str, Any] | None = None,
        price: ResolvedPrice | None = None,
    ) -> CostEvent:
        """Build the cost event for one text to speech call.

        When *characters* is omitted it is read from ``response.metadata``. A
        response also contributes its model, voice and request id for
        provenance. The input text itself is never copied into the event.
        """
        if response is not None and not isinstance(response, TTSResponse):
            raise TypeError("response must be a TTSResponse or None")
        if characters is None:
            if response is None:
                raise ConfigError("a character count or response is required")
            characters = _characters_from_response(response)
        resolved = (
            price
            if price is not None
            else self.resolve(
                provider,
                characters,
                provider_name=provider_name,
                unit_kind=unit_kind,
                operation=operation,
            )
        )
        details: dict[str, Any] = {
            "characters": resolved.units,
            "billed_units": resolved.billed_units,
            "price_unit": resolved.price_unit,
            "price_source": resolved.source.value,
        }
        if resolved.missing:
            details["missing_price"] = True
        if response is not None:
            if response.voice:
                details["voice"] = response.voice
            if response.request_id:
                details["request_id"] = response.request_id
            if response.format:
                details["format"] = response.format.value
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
        characters: float | None = None,
        response: TTSResponse | None = None,
        provider_name: str | None = None,
        run_id: str = "",
        operation: str = TTS_OPERATION,
        estimated: bool = False,
        unit_kind: str | None = CHARACTER_UNIT,
        metadata: Mapping[str, Any] | None = None,
        price: ResolvedPrice | None = None,
    ) -> CostEvent:
        """Record the cost event for one text to speech call."""
        if not isinstance(tracker, CostTracker):
            raise TypeError("tracker must be a CostTracker")
        event = self.cost_event(
            provider,
            characters=characters,
            response=response,
            provider_name=provider_name,
            run_id=run_id,
            operation=operation,
            estimated=estimated,
            unit_kind=unit_kind,
            metadata=metadata,
            price=price,
        )
        if estimated:
            return tracker.record_estimate(event)
        return tracker.record_actual(event)


def tts_cost_event(
    provider: Provider | ProviderMeta,
    *,
    overrides: Mapping[str, float] | None = None,
    pricing_version: str = "",
    **values: Any,
) -> CostEvent:
    """Build one text to speech cost event without constructing a calculator."""
    return TTSCostCalculator(
        overrides,
        pricing_version=pricing_version,
    ).cost_event(provider, **values)

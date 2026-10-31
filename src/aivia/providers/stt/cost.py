"""Per-minute speech to text cost calculation.

Hosted transcription services bill by the duration of the audio they process,
while a local model is free. This module resolves a provider's per-minute price
from its metadata or a configured override and turns the audio duration into a
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
from aivia.providers.protocols.stt import STTResponse
from aivia.providers.stt.common import response_duration_seconds

__all__ = [
    "MINUTE_UNIT",
    "STT_OPERATION",
    "STTCost",
    "STTCostCalculator",
    "STTPrice",
    "stt_cost_event",
]

STT_OPERATION = "stt"
"""The cost event operation for one speech to text call."""

MINUTE_UNIT = "minute"
"""The cost unit for speech to text audio duration."""

STTPrice = ResolvedPrice
"""Alias for the resolved price of one speech to text call."""

STTCost = ResolvedPrice
"""Alias for the resolved cost of one speech to text call."""


def _minutes(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            "minutes must be a finite, non-negative number",
            details={"minutes": value},
        )
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ConfigError(
            "minutes must be a finite, non-negative number",
            details={"minutes": value},
        )
    return number


def _minutes_from_response(response: STTResponse) -> float:
    duration = response_duration_seconds(response)
    if duration is None:
        raise ConfigError(
            "speech to text response has no duration",
            details={"provider": response.provider},
        )
    return _minutes(duration / 60.0)


class STTCostCalculator:
    """Resolve per-minute prices and build or record their cost events."""

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
        minutes: float,
        *,
        provider_name: str | None = None,
        unit_kind: str | None = MINUTE_UNIT,
        operation: str = STT_OPERATION,
    ) -> ResolvedPrice:
        """Return the price of *minutes* of audio transcribed by *provider*."""
        return self._resolver.resolve(
            provider,
            operation,
            _minutes(minutes),
            unit_kind=unit_kind,
            provider_name=provider_name,
        )

    def cost_for(
        self,
        provider: Provider | ProviderMeta,
        minutes: float,
        *,
        provider_name: str | None = None,
        unit_kind: str | None = MINUTE_UNIT,
        operation: str = STT_OPERATION,
    ) -> ResolvedPrice:
        """Alias for :meth:`resolve`."""
        return self.resolve(
            provider,
            minutes,
            provider_name=provider_name,
            unit_kind=unit_kind,
            operation=operation,
        )

    def cost_event(
        self,
        provider: Provider | ProviderMeta,
        *,
        minutes: float | None = None,
        response: STTResponse | None = None,
        provider_name: str | None = None,
        run_id: str = "",
        operation: str = STT_OPERATION,
        estimated: bool = False,
        unit_kind: str | None = MINUTE_UNIT,
        metadata: Mapping[str, Any] | None = None,
        price: ResolvedPrice | None = None,
    ) -> CostEvent:
        """Build the cost event for one speech to text call.

        When *minutes* is omitted it is derived from ``response``. A response
        also contributes its model, language, word count and request id for
        provenance. The transcript text itself is never copied into the event.
        """
        if response is not None and not isinstance(response, STTResponse):
            raise TypeError("response must be a STTResponse or None")
        if minutes is None:
            if response is None:
                raise ConfigError("a duration or response is required")
            minutes = _minutes_from_response(response)
        resolved = (
            price
            if price is not None
            else self.resolve(
                provider,
                minutes,
                provider_name=provider_name,
                unit_kind=unit_kind,
                operation=operation,
            )
        )
        details: dict[str, Any] = {
            "minutes": resolved.units,
            "billed_units": resolved.billed_units,
            "price_unit": resolved.price_unit,
            "price_source": resolved.source.value,
        }
        if resolved.missing:
            details["missing_price"] = True
        if response is not None:
            if response.language:
                details["language"] = response.language
            if response.word_count:
                details["words"] = response.word_count
            if response.request_id:
                details["request_id"] = response.request_id
            duration = response_duration_seconds(response)
            if duration is not None:
                details["duration_seconds"] = duration
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
        minutes: float | None = None,
        response: STTResponse | None = None,
        provider_name: str | None = None,
        run_id: str = "",
        operation: str = STT_OPERATION,
        estimated: bool = False,
        unit_kind: str | None = MINUTE_UNIT,
        metadata: Mapping[str, Any] | None = None,
        price: ResolvedPrice | None = None,
    ) -> CostEvent:
        """Record the cost event for one speech to text call."""
        if not isinstance(tracker, CostTracker):
            raise TypeError("tracker must be a CostTracker")
        event = self.cost_event(
            provider,
            minutes=minutes,
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


def stt_cost_event(
    provider: Provider | ProviderMeta,
    *,
    overrides: Mapping[str, float] | None = None,
    pricing_version: str = "",
    **values: Any,
) -> CostEvent:
    """Build one speech to text cost event without constructing a calculator."""
    return STTCostCalculator(
        overrides,
        pricing_version=pricing_version,
    ).cost_event(provider, **values)

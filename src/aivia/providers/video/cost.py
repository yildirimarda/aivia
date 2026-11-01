"""Turn a video pricing table into cost events.

Video is priced along four dimensions at once: every generated second, a flat
fee for every clip, a rate that depends on the resolution tier and a surcharge
for a generated soundtrack.
:class:`~aivia.media.video_pricing.VideoPricing` already validates those rates
and :class:`~aivia.media.video_pricing.VideoPricingEstimator` combines them
into one estimate. This module turns that estimate into a
:class:`~aivia.core.models.cost.CostEvent` so a video call reaches the ledger
with the same provenance as every other paid call.

An event records the resolved amount and the unit it was billed in. When a
per-second rate applies the units are billable seconds; otherwise they are
clips. The tier, resolution, base and audio components are copied into the
event metadata, so a cost can be explained after the fact.

The calculator never calls a provider and never enforces a budget. Recording an
estimate followed by an actual event uses
:class:`~aivia.core.cost.CostTracker`, which links the actual event back to the
estimate it replaces. The pricing models are imported lazily so a provider
module does not pull in the media package at import time.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from aivia.core.cost import CostTracker
from aivia.core.errors import ConfigError
from aivia.core.models.cost import CostEvent
from aivia.providers.protocols.video import (
    CAPABILITY_VIDEO_GEN,
    VideoRequest,
    VideoResponse,
)

if TYPE_CHECKING:
    from aivia.media.video_pricing import (
        VideoCostEstimate,
        VideoPricing,
        VideoPricingEstimator,
    )

__all__ = [
    "CLIP_UNIT",
    "SECOND_UNIT",
    "VIDEO_OPERATION",
    "VIDEO_UNIT",
    "VideoCostCalculator",
    "VideoCostEvent",
    "video_cost_event",
    "video_cost_events",
]

VIDEO_OPERATION = CAPABILITY_VIDEO_GEN
"""The cost event operation for one video generation call."""

VIDEO_UNIT = "second"
"""The cost unit used when a per-second rate applies."""

SECOND_UNIT = VIDEO_UNIT
"""Alias for :data:`VIDEO_UNIT`."""

CLIP_UNIT = "clip"
"""The cost unit used when only per-clip pricing applies."""

VideoCostEvent = CostEvent
"""The event type produced by the calculator."""


def _pricing_version(value: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(
            "pricing version must be a string",
            details={"pricing_version": value},
        )
    return value.strip()


def _duration(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            "duration_seconds must be a finite, non-negative number",
            details={"duration_seconds": value},
        )
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ConfigError(
            "duration_seconds must be a finite, non-negative number",
            details={"duration_seconds": value},
        )
    return number


def _provider_name(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    name = getattr(value, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    metadata = getattr(value, "metadata", None)
    name = getattr(metadata, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    raise TypeError("provider must be a name, Provider or ProviderMeta")


def _request(value: Any, request: VideoRequest | None) -> VideoRequest:
    if isinstance(value, VideoRequest):
        if request is not None and request is not value:
            raise TypeError("pass either a request or request=, not both")
        return value
    if request is not None:
        if not isinstance(request, VideoRequest):
            raise TypeError("request must be a VideoRequest")
        return request
    if value is None:
        raise TypeError("a VideoRequest is required")
    raise TypeError("request must be a VideoRequest")


def _request_for_result(
    request: VideoRequest,
    response: VideoResponse,
    result: Any,
) -> VideoRequest:
    return request.model_copy(
        update={
            "count": 1,
            "duration_seconds": (
                result.duration_seconds
                if result.duration_seconds is not None
                else request.duration_seconds
            ),
            "width": result.width if result.width is not None else request.width,
            "height": result.height if result.height is not None else request.height,
            "generate_audio": bool(result.has_audio or request.generate_audio),
        }
    )


def _units(estimate: VideoCostEstimate) -> tuple[float, str]:
    time_priced = estimate.per_second_usd > 0.0 or estimate.audio_per_second_usd > 0.0
    if time_priced:
        return estimate.billable_seconds, VIDEO_UNIT
    return float(estimate.videos), CLIP_UNIT


class VideoCostCalculator:
    """Resolve video prices and build or record their cost events.

    *pricing* may be a validated
    :class:`~aivia.media.video_pricing.VideoPricing` table, a mapping or a
    ready estimator. Pass *pricing_version* to stamp a configured table version
    onto every event; otherwise the version on the pricing table is used.
    """

    def __init__(
        self,
        pricing: VideoPricing | Mapping[str, Any] | VideoPricingEstimator | None = None,
        *,
        estimator: VideoPricingEstimator | None = None,
        pricing_version: str = "",
    ) -> None:
        if pricing is not None and estimator is not None:
            raise TypeError("pass either pricing or estimator, not both")
        from aivia.media.video_pricing import (
            VideoPricing,
            VideoPricingEstimator,
        )

        if estimator is not None:
            if not isinstance(estimator, VideoPricingEstimator):
                raise TypeError("estimator must be a VideoPricingEstimator")
            resolved = estimator
        elif isinstance(pricing, VideoPricingEstimator):
            resolved = pricing
        elif isinstance(pricing, VideoPricing):
            resolved = VideoPricingEstimator(pricing)
        elif isinstance(pricing, Mapping):
            resolved = VideoPricingEstimator(VideoPricing.from_mapping(pricing))
        elif pricing is None:
            resolved = VideoPricingEstimator()
        else:
            raise TypeError(
                "pricing must be a VideoPricing, mapping or VideoPricingEstimator"
            )
        self._estimator = resolved
        self._pricing_version = _pricing_version(pricing_version)

    @property
    def estimator(self) -> VideoPricingEstimator:
        """The estimator every price is resolved through."""
        return self._estimator

    @property
    def pricing(self) -> VideoPricing:
        """The validated pricing table behind :attr:`estimator`."""
        return self._estimator.pricing

    @property
    def pricing_version(self) -> str:
        """The version copied onto every event."""
        return self._pricing_version or self._estimator.pricing.pricing_version

    def estimate(
        self,
        request: VideoRequest,
        *,
        tier: str | None = None,
        duration_seconds: float | None = None,
    ) -> VideoCostEstimate:
        """Return the estimate for one request."""
        if not isinstance(request, VideoRequest):
            raise TypeError("request must be a VideoRequest")
        return self._estimator.estimate(
            request,
            tier=tier,
            duration_seconds=_duration(duration_seconds),
        )

    def cost_for(
        self,
        request: VideoRequest,
        *,
        tier: str | None = None,
        duration_seconds: float | None = None,
    ) -> VideoCostEstimate:
        """Alias for :meth:`estimate`."""
        return self.estimate(
            request,
            tier=tier,
            duration_seconds=duration_seconds,
        )

    def resolve(
        self,
        request: VideoRequest,
        *,
        tier: str | None = None,
        duration_seconds: float | None = None,
    ) -> VideoCostEstimate:
        """Alias for :meth:`estimate`."""
        return self.estimate(
            request,
            tier=tier,
            duration_seconds=duration_seconds,
        )

    def estimate_requests(
        self,
        requests: Iterable[VideoRequest],
        *,
        tier: str | None = None,
    ) -> VideoCostEstimate:
        """Return one combined estimate for a sequence of clips."""
        return self._estimator.estimate_requests(requests, tier=tier)

    def estimate_response(
        self,
        request: VideoRequest,
        response: VideoResponse,
        *,
        tier: str | None = None,
    ) -> VideoCostEstimate:
        """Price the clips a provider actually returned.

        Each result contributes its own duration, resolution and audio state,
        falling back to the submitted request when the provider omitted one.
        """
        if not isinstance(request, VideoRequest):
            raise TypeError("request must be a VideoRequest")
        if not isinstance(response, VideoResponse):
            raise TypeError("response must be a VideoResponse")
        return self.estimate_requests(
            [
                _request_for_result(request, response, result)
                for result in response.videos
            ],
            tier=tier,
        )

    def cost_event(
        self,
        subject: Any = None,
        response: VideoResponse | None = None,
        *,
        request: VideoRequest | None = None,
        requests: Iterable[VideoRequest] | None = None,
        provider: Any = None,
        provider_name: str | None = None,
        model: str | None = None,
        run_id: str = "",
        operation: str = VIDEO_OPERATION,
        estimated: bool = False,
        metadata: Mapping[str, Any] | None = None,
        tier: str | None = None,
        duration_seconds: float | None = None,
        estimate: VideoCostEstimate | None = None,
    ) -> CostEvent:
        """Build the cost event for one video call.

        Pass the request as the first argument or with ``request=``. A
        provider, provider metadata or name may be passed first instead, as the
        other capability calculators do. ``requests`` prices a split
        adaptation as one event with one per-clip fee per provider call.
        """
        if response is not None and not isinstance(response, VideoResponse):
            raise TypeError("response must be a VideoResponse or None")
        if (
            subject is not None
            and not isinstance(subject, VideoRequest)
            and provider is None
        ):
            provider = subject
        if requests is not None:
            if request is not None or isinstance(subject, VideoRequest):
                raise TypeError("pass either requests or a request, not both")
            items = tuple(requests)
            if not items:
                raise ValueError("at least one video request is required")
            active_request = items[0]
            resolved_estimate = self.estimate_requests(items, tier=tier)
        else:
            active_request = _request(subject, request)
            if estimate is not None:
                resolved_estimate = estimate
            elif response is not None:
                resolved_estimate = self.estimate_response(
                    active_request,
                    response,
                    tier=tier,
                )
            else:
                resolved_estimate = self.estimate(
                    active_request,
                    tier=tier,
                    duration_seconds=duration_seconds,
                )

        name = (
            provider_name.strip()
            if isinstance(provider_name, str)
            else _provider_name(provider)
        )
        if not name and response is not None:
            name = response.provider
        model_name = (
            model.strip()
            if isinstance(model, str)
            else (response.model if response is not None else "")
        )
        if not model_name:
            model_name = (active_request.model or "").strip()

        units, unit_kind = _units(resolved_estimate)
        details: dict[str, Any] = {
            "video_count": resolved_estimate.videos,
            "clips": resolved_estimate.clips,
            "duration_seconds": resolved_estimate.duration_seconds,
            "billable_seconds": resolved_estimate.billable_seconds,
            "billed_units": units,
            "price_unit": unit_kind,
            "price_source": "pricing",
            "per_second_usd": resolved_estimate.per_second_usd,
            "per_clip_usd": resolved_estimate.per_clip_usd,
            "audio_per_second_usd": resolved_estimate.audio_per_second_usd,
            "audio_per_clip_usd": resolved_estimate.audio_per_clip_usd,
            "base_cost_usd": resolved_estimate.base_cost_usd,
            "audio_cost_usd": resolved_estimate.audio_cost_usd,
            "audio": resolved_estimate.audio,
            "currency": resolved_estimate.currency,
            "estimate": resolved_estimate.to_dict(),
        }
        if resolved_estimate.tier is not None:
            details["tier"] = resolved_estimate.tier
        if resolved_estimate.resolution is not None:
            details["resolution"] = list(resolved_estimate.resolution)
        if response is not None:
            if response.request_id:
                details["request_id"] = response.request_id
            if response.model:
                details["model"] = response.model
        if metadata is not None:
            details.update(metadata)

        version = self.pricing_version or resolved_estimate.pricing_version
        return CostEvent(
            provider=name,
            model=model_name,
            operation=operation,
            units=units,
            unit_kind=unit_kind,
            amount_usd=resolved_estimate.amount_usd,
            estimated=estimated,
            run_id=run_id,
            pricing_version=version,
            metadata=details,
        )

    def record(
        self,
        subject: Any,
        tracker: CostTracker,
        response: VideoResponse | None = None,
        *,
        request: VideoRequest | None = None,
        requests: Iterable[VideoRequest] | None = None,
        provider: Any = None,
        provider_name: str | None = None,
        model: str | None = None,
        run_id: str = "",
        operation: str = VIDEO_OPERATION,
        estimated: bool = False,
        metadata: Mapping[str, Any] | None = None,
        tier: str | None = None,
        duration_seconds: float | None = None,
        estimate: VideoCostEstimate | None = None,
    ) -> CostEvent:
        """Record the cost event for one video call."""
        if not isinstance(tracker, CostTracker):
            raise TypeError("tracker must be a CostTracker")
        event = self.cost_event(
            subject,
            response,
            request=request,
            requests=requests,
            provider=provider,
            provider_name=provider_name,
            model=model,
            run_id=run_id,
            operation=operation,
            estimated=estimated,
            metadata=metadata,
            tier=tier,
            duration_seconds=duration_seconds,
            estimate=estimate,
        )
        if estimated:
            return tracker.record_estimate(event)
        return tracker.record_actual(event)

    def record_event(
        self,
        subject: Any,
        tracker: CostTracker,
        **values: Any,
    ) -> CostEvent:
        """Alias for :meth:`record`."""
        return self.record(subject, tracker, **values)

    def record_estimate(
        self,
        subject: Any,
        tracker: CostTracker,
        **values: Any,
    ) -> CostEvent:
        """Record a pre-call quote for one video call."""
        values["estimated"] = True
        return self.record(subject, tracker, **values)

    def record_actual(
        self,
        subject: Any,
        tracker: CostTracker,
        **values: Any,
    ) -> CostEvent:
        """Record the provider-reported amount, linking it to its estimate."""
        values["estimated"] = False
        return self.record(subject, tracker, **values)


def video_cost_event(
    subject: Any = None,
    response: VideoResponse | None = None,
    *,
    pricing: VideoPricing | Mapping[str, Any] | VideoPricingEstimator | None = None,
    estimator: VideoPricingEstimator | None = None,
    pricing_version: str = "",
    **values: Any,
) -> CostEvent:
    """Build one video cost event without constructing a calculator."""
    calculator = VideoCostCalculator(
        pricing,
        estimator=estimator,
        pricing_version=pricing_version,
    )
    return calculator.cost_event(subject, response, **values)


def video_cost_events(
    requests: Iterable[VideoRequest],
    *,
    pricing: VideoPricing | Mapping[str, Any] | VideoPricingEstimator | None = None,
    estimator: VideoPricingEstimator | None = None,
    pricing_version: str = "",
    **values: Any,
) -> tuple[CostEvent, ...]:
    """Build one event per clip for a sequence of video requests."""
    calculator = VideoCostCalculator(
        pricing,
        estimator=estimator,
        pricing_version=pricing_version,
    )
    return tuple(calculator.cost_event(request, **values) for request in requests)

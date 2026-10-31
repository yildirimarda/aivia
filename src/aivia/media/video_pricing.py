"""Price a video request from per-second, per-clip, resolution and audio rates.

Video providers rarely quote a single number. A model may charge for every
generated second, a flat fee for every clip, more for a larger resolution and a
surcharge when native audio is enabled. This module keeps those dimensions
separate and combines them in one deterministic estimate, so a caller can
budget a request before submitting it and explain the amount afterwards.

Pricing is pure data. :class:`VideoPricing` validates the rates, resolves the
resolution tier for a request and returns a :class:`VideoRate`. The
:class:`VideoPricingEstimator` multiplies that rate by the request duration and
variation count. Neither model calls a provider, reads configuration or mutates
the request.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Self

from pydantic import AliasChoices, Field, field_validator, model_validator

from aivia.core.config.base import ConfigModel
from aivia.core.errors import ConfigError
from aivia.providers.protocols.video import VideoRequest

__all__ = [
    "ResolutionTier",
    "VideoAudioPricing",
    "VideoCost",
    "VideoCostEstimate",
    "VideoEstimate",
    "VideoEstimator",
    "VideoPrice",
    "VideoPriceTier",
    "VideoPricing",
    "VideoPricingEstimator",
    "VideoPricingModel",
    "VideoRate",
    "VideoResolutionTier",
    "estimate_video_adaptation",
    "estimate_video_cost",
    "estimate_video_price",
    "estimate_video_requests",
    "price_video",
    "video_cost",
]

_TIER_HEIGHT = re.compile(r"^(?P<height>\d{3,4})p$", re.IGNORECASE)
_TIER_K = re.compile(r"^(?P<k>\d+)k$", re.IGNORECASE)


def _finite_non_negative(value: Any, *, label: str) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite, non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be a finite, non-negative number")
    return number


def _tier_pixels_from_name(name: str) -> int | None:
    text = name.strip()
    match = _TIER_HEIGHT.fullmatch(text)
    height: int | None = None
    if match is not None:
        height = int(match.group("height"))
    else:
        match = _TIER_K.fullmatch(text)
        if match is not None:
            height = int(match.group("k")) * 540
    if height is None or height <= 0:
        return None
    return round(height * height * 16 / 9)


class VideoAudioPricing(ConfigModel):
    """The surcharge for a generated soundtrack.

    ``per_second`` and ``per_clip`` are added to the base video cost for every
    generated second and every generated clip. ``multiplier`` scales that
    surcharge, so ``1.5`` means audio costs one and a half times the listed
    add-on rates. Audio is never charged when the request did not ask for it.
    """

    per_second: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices(
            "per_second",
            "per_second_usd",
            "audio_per_second",
            "cost_per_second",
        ),
    )
    per_clip: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices(
            "per_clip",
            "per_clip_usd",
            "audio_per_clip",
            "cost_per_clip",
        ),
    )
    multiplier: float = Field(
        default=1.0,
        ge=0.0,
        validation_alias=AliasChoices(
            "multiplier",
            "audio_multiplier",
            "factor",
        ),
    )

    @field_validator("per_second", "per_clip", "multiplier")
    @classmethod
    def _validate_rates(cls, value: float) -> float:
        return float(_finite_non_negative(value, label="audio price"))

    def cost(self, *, duration_seconds: float, videos: int) -> float:
        """Return the audio surcharge for *videos* of *duration_seconds* each."""
        return (
            (self.per_second * duration_seconds + self.per_clip)
            * videos
            * self.multiplier
        )


class VideoResolutionTier(ConfigModel):
    """A resolution band with optional rate overrides.

    A request matches the smallest tier whose ``max_pixels`` or
    ``max_short_side`` covers it. When both bounds are unset, the name is
    parsed for a familiar value such as ``720p``, ``1080p`` or ``4k`` and the
    equivalent pixel budget is derived from a 16:9 frame. A tier without any
    bound is a catch-all and is used only when nothing else matches.

    ``per_second`` and ``per_clip`` replace the base rates for the tier;
    ``multiplier`` scales whichever rate applies. ``audio`` replaces the
    request-wide audio pricing, while the individual ``audio_*`` fields
    override only the named component.
    """

    name: str = ""
    max_pixels: int | None = Field(default=None, gt=0)
    max_short_side: int | None = Field(
        default=None,
        gt=0,
        validation_alias=AliasChoices(
            "max_short_side",
            "max_height",
            "max_side",
        ),
    )
    per_second: float | None = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices(
            "per_second",
            "per_second_usd",
            "cost_per_second",
        ),
    )
    per_clip: float | None = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices(
            "per_clip",
            "per_clip_usd",
            "cost_per_clip",
        ),
    )
    multiplier: float = Field(default=1.0, ge=0.0)
    audio: VideoAudioPricing | None = None
    audio_per_second: float | None = Field(default=None, ge=0.0)
    audio_per_clip: float | None = Field(default=None, ge=0.0)
    audio_multiplier: float | None = Field(default=None, ge=0.0)

    @field_validator("name", mode="before")
    @classmethod
    def _strip_name(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("per_second", "per_clip", "audio_per_second", "audio_per_clip")
    @classmethod
    def _validate_optional_rates(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return float(_finite_non_negative(value, label="tier price"))

    @field_validator("multiplier", "audio_multiplier")
    @classmethod
    def _validate_optional_multiplier(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return float(_finite_non_negative(value, label="tier multiplier"))

    @model_validator(mode="after")
    def _derive_pixel_bound(self) -> Self:
        if self.max_pixels is None and self.max_short_side is None:
            pixels = _tier_pixels_from_name(self.name)
            if pixels is not None:
                object.__setattr__(self, "max_pixels", pixels)
        return self

    @property
    def bounded(self) -> bool:
        """Whether the tier declares a pixel or short-side bound."""
        return self.max_pixels is not None or self.max_short_side is not None

    def matches(self, width: int, height: int) -> bool:
        """Whether a *width* by *height* frame belongs to this tier."""
        if self.max_pixels is not None and width * height > self.max_pixels:
            return False
        return self.max_short_side is None or min(width, height) <= self.max_short_side

    def rank(self) -> tuple[int, int]:
        """A sortable size used to pick the smallest matching tier."""
        if self.max_pixels is not None:
            return (0, self.max_pixels)
        if self.max_short_side is not None:
            return (1, self.max_short_side * self.max_short_side * 16 // 9)
        return (2, 0)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for logs and reports."""
        return self.model_dump(mode="json")


ResolutionTier = VideoResolutionTier
VideoPriceTier = VideoResolutionTier


class VideoRate(ConfigModel):
    """The effective rates for one request after the tier and audio options.

    The audio rates are present even when audio is disabled so a report can
    show what enabling it would cost; the estimator charges them only when the
    request asked for a soundtrack.
    """

    tier: str | None = None
    per_second: float = Field(default=0.0, ge=0.0)
    per_clip: float = Field(default=0.0, ge=0.0)
    audio_per_second: float = Field(default=0.0, ge=0.0)
    audio_per_clip: float = Field(default=0.0, ge=0.0)
    currency: str = "USD"
    pricing_version: str = ""

    def cost(
        self,
        *,
        duration_seconds: float,
        videos: int,
        generate_audio: bool,
    ) -> float:
        """Return the total price for *videos* clips of *duration_seconds*."""
        base = (self.per_second * duration_seconds + self.per_clip) * videos
        if not generate_audio:
            return base
        audio = (
            self.audio_per_second * duration_seconds + self.audio_per_clip
        ) * videos
        return base + audio

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for logs and reports."""
        return self.model_dump(mode="json")


class VideoPricing(ConfigModel):
    """Rates for video generation with resolution and audio dimensions.

    ``per_second`` prices every generated second and ``per_clip`` adds a flat
    fee for every generated clip, so both can be combined. ``resolution_tiers``
    overrides either rate for a resolution band. ``audio`` is the default
    soundtrack surcharge and ``audio_options`` may name separate rates for the
    ``native`` and ``prompt`` audio modes.

    ``currency`` and ``pricing_version`` are copied onto every estimate so a
    cost event or sidecar records which price table produced the amount.
    """

    per_second: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices(
            "per_second",
            "per_second_usd",
            "price_per_second",
            "cost_per_second",
        ),
    )
    per_clip: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices(
            "per_clip",
            "per_clip_usd",
            "price_per_clip",
            "cost_per_clip",
        ),
    )
    resolution_tiers: dict[str, VideoResolutionTier] = Field(
        default_factory=dict,
        validation_alias=AliasChoices(
            "resolution_tiers",
            "tiers",
            "resolutions",
        ),
    )
    default_resolution_tier: str | None = None
    audio: VideoAudioPricing = Field(default_factory=VideoAudioPricing)
    audio_options: dict[str, VideoAudioPricing] = Field(default_factory=dict)
    currency: str = "USD"
    pricing_version: str = ""

    @field_validator("per_second", "per_clip")
    @classmethod
    def _validate_rates(cls, value: float) -> float:
        return float(_finite_non_negative(value, label="video price"))

    @field_validator("currency", mode="before")
    @classmethod
    def _normalize_currency(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("currency")
    @classmethod
    def _validate_currency(cls, value: str) -> str:
        if len(value) != 3 or not value.isalpha():
            raise ValueError("currency must be a three-letter code")
        return value

    @field_validator("default_resolution_tier", mode="before")
    @classmethod
    def _strip_default_tier(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            return text or None
        return value

    @field_validator("pricing_version", mode="before")
    @classmethod
    def _strip_version(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _name_tiers(self) -> Self:
        tiers: dict[str, VideoResolutionTier] = {}
        for key, tier in self.resolution_tiers.items():
            name = key.strip()
            if not name:
                raise ValueError("resolution tier names must be non-empty")
            if not tier.name:
                tier = tier.model_copy(update={"name": name})
            if not tier.bounded:
                pixels = _tier_pixels_from_name(tier.name)
                if pixels is not None:
                    tier = tier.model_copy(update={"max_pixels": pixels})
            tiers[name] = tier
        object.__setattr__(self, "resolution_tiers", tiers)
        if (
            self.default_resolution_tier is not None
            and self.default_resolution_tier not in tiers
        ):
            available = ", ".join(sorted(tiers)) or "none"
            raise ValueError(
                "default_resolution_tier names an unknown tier "
                f"{self.default_resolution_tier!r} (available: {available})"
            )
        return self

    @property
    def tiers(self) -> dict[str, VideoResolutionTier]:
        """Alias for :attr:`resolution_tiers`."""
        return self.resolution_tiers

    def tier(self, name: str) -> VideoResolutionTier:
        """Return the tier called *name*, raising when it is unknown."""
        key = name.strip()
        try:
            return self.resolution_tiers[key]
        except KeyError:
            available = ", ".join(sorted(self.resolution_tiers)) or "none"
            raise ConfigError(
                f"unknown video resolution tier: {name!r} (available: {available})",
                details={
                    "tier": name,
                    "available": list(self.resolution_tiers),
                },
            ) from None

    def tier_for(
        self,
        width: int | None,
        height: int | None,
        *,
        name: str | None = None,
    ) -> VideoResolutionTier | None:
        """Return the tier a frame belongs to, or ``None`` without a tier.

        An explicit *name* wins. Without dimensions the configured default is
        used. A frame larger than every bound falls back to the largest tier,
        so an unusual resolution is never silently underpriced.
        """
        if name is not None:
            return self.tier(name)
        if not self.resolution_tiers:
            return None
        if width is None or height is None:
            if self.default_resolution_tier is None:
                return None
            return self.tier(self.default_resolution_tier)
        matching = [
            tier
            for tier in self.resolution_tiers.values()
            if tier.matches(width, height)
        ]
        bounded = [tier for tier in matching if tier.bounded]
        if bounded:
            return min(bounded, key=lambda tier: (tier.rank(), tier.name))
        if matching:
            return min(matching, key=lambda tier: tier.name)
        bounded = [tier for tier in self.resolution_tiers.values() if tier.bounded]
        if bounded:
            return max(bounded, key=lambda tier: (tier.rank(), tier.name))
        return None

    def _audio_pricing(
        self,
        tier: VideoResolutionTier | None,
        *,
        audio_prompt: str | None,
    ) -> VideoAudioPricing:
        option = "prompt" if audio_prompt else "native"
        pricing = self.audio_options.get(option, self.audio)
        if tier is None:
            return pricing
        if tier.audio is not None:
            return tier.audio
        updates: dict[str, float] = {}
        if tier.audio_per_second is not None:
            updates["per_second"] = tier.audio_per_second
        if tier.audio_per_clip is not None:
            updates["per_clip"] = tier.audio_per_clip
        if tier.audio_multiplier is not None:
            updates["multiplier"] = tier.audio_multiplier
        if not updates:
            return pricing
        return pricing.model_copy(update=updates)

    def rate_for(
        self,
        width: int | None = None,
        height: int | None = None,
        *,
        generate_audio: bool = False,
        audio_prompt: str | None = None,
        tier: str | None = None,
    ) -> VideoRate:
        """Resolve the effective rates for one frame and audio mode."""
        selected = self.tier_for(width, height, name=tier)
        per_second = self.per_second
        per_clip = self.per_clip
        if selected is not None:
            if selected.per_second is not None:
                per_second = selected.per_second
            if selected.per_clip is not None:
                per_clip = selected.per_clip
            per_second *= selected.multiplier
            per_clip *= selected.multiplier
        audio = self._audio_pricing(selected, audio_prompt=audio_prompt)
        return VideoRate(
            tier=selected.name if selected is not None else None,
            per_second=per_second,
            per_clip=per_clip,
            audio_per_second=audio.per_second * audio.multiplier,
            audio_per_clip=audio.per_clip * audio.multiplier,
            currency=self.currency,
            pricing_version=self.pricing_version,
        )


VideoPricingModel = VideoPricing
VideoPrice = VideoPricing


@dataclass(frozen=True, slots=True)
class VideoCostEstimate:
    """The estimated cost of one or more video requests.

    ``videos`` counts every generated variation, ``clips`` counts provider
    calls and ``duration_seconds`` is the summed length of one variation.
    ``billable_seconds`` is the duration multiplied by the video count, which
    is the quantity a per-second price is applied to.
    """

    videos: int
    clips: int
    duration_seconds: float
    billable_seconds: float
    resolution: tuple[int, int] | None
    tier: str | None
    audio: bool
    per_second_usd: float
    per_clip_usd: float
    audio_per_second_usd: float
    audio_per_clip_usd: float
    base_cost_usd: float
    audio_cost_usd: float
    amount_usd: float
    currency: str = "USD"
    pricing_version: str = ""

    @property
    def total_usd(self) -> float:
        """Alias for :attr:`amount_usd`."""
        return self.amount_usd

    @property
    def cost_usd(self) -> float:
        """Alias for :attr:`amount_usd`."""
        return self.amount_usd

    @property
    def price_usd(self) -> float:
        """Alias for :attr:`amount_usd`."""
        return self.amount_usd

    @property
    def total_cost_usd(self) -> float:
        """Alias for :attr:`amount_usd`."""
        return self.amount_usd

    @property
    def estimated_cost_usd(self) -> float:
        """Alias for :attr:`amount_usd`."""
        return self.amount_usd

    @property
    def amount(self) -> float:
        """Alias for :attr:`amount_usd`."""
        return self.amount_usd

    @property
    def total(self) -> float:
        """Alias for :attr:`amount_usd`."""
        return self.amount_usd

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for logs and sidecars."""
        return {
            "videos": self.videos,
            "clips": self.clips,
            "duration_seconds": self.duration_seconds,
            "billable_seconds": self.billable_seconds,
            "resolution": list(self.resolution) if self.resolution else None,
            "tier": self.tier,
            "audio": self.audio,
            "per_second_usd": self.per_second_usd,
            "per_clip_usd": self.per_clip_usd,
            "audio_per_second_usd": self.audio_per_second_usd,
            "audio_per_clip_usd": self.audio_per_clip_usd,
            "base_cost_usd": self.base_cost_usd,
            "audio_cost_usd": self.audio_cost_usd,
            "amount_usd": self.amount_usd,
            "currency": self.currency,
            "pricing_version": self.pricing_version,
        }


VideoCost = VideoCostEstimate
VideoEstimate = VideoCostEstimate


class VideoPricingEstimator:
    """Estimate video cost from a :class:`VideoPricing` table.

    ``pricing`` may be a validated model or a mapping. Every estimate is pure:
    the request is read but never changed. Pass several adapted clips to
    :meth:`estimate_requests` to price a split request, where the per-clip fee
    applies once per provider call.
    """

    def __init__(
        self,
        pricing: VideoPricing | Mapping[str, Any] | None = None,
    ) -> None:
        if pricing is None:
            model = VideoPricing()
        elif isinstance(pricing, VideoPricing):
            model = pricing
        elif isinstance(pricing, Mapping):
            model = VideoPricing.from_mapping(pricing)
        else:
            raise TypeError(
                "pricing must be a VideoPricing or a mapping, "
                f"not {type(pricing).__name__}"
            )
        self._pricing = model

    @property
    def pricing(self) -> VideoPricing:
        """The pricing table this estimator applies."""
        return self._pricing

    def rate_for(
        self,
        request: VideoRequest,
        *,
        tier: str | None = None,
    ) -> VideoRate:
        """Return the effective rates for *request*."""
        if not isinstance(request, VideoRequest):
            raise TypeError("request must be a VideoRequest")
        return self._pricing.rate_for(
            request.width,
            request.height,
            generate_audio=request.generate_audio,
            audio_prompt=request.audio_prompt,
            tier=tier,
        )

    def estimate(
        self,
        request: VideoRequest,
        *,
        tier: str | None = None,
        duration_seconds: float | None = None,
    ) -> VideoCostEstimate:
        """Estimate one request, optionally overriding its clip duration."""
        if not isinstance(request, VideoRequest):
            raise TypeError("request must be a VideoRequest")
        clip_length = (
            request.duration_seconds if duration_seconds is None else duration_seconds
        )
        if clip_length is None:
            clip_length = 0.0
        if not math.isfinite(clip_length) or clip_length < 0:
            raise ValueError("duration_seconds must be finite and non-negative")
        rate = self.rate_for(request, tier=tier)
        videos = request.count
        billable = clip_length * videos
        base = rate.cost(
            duration_seconds=clip_length,
            videos=videos,
            generate_audio=False,
        )
        audio_cost = (
            rate.cost(
                duration_seconds=clip_length,
                videos=videos,
                generate_audio=True,
            )
            - base
            if request.generate_audio
            else 0.0
        )
        resolution = (
            (request.width, request.height)
            if request.width is not None and request.height is not None
            else None
        )
        return VideoCostEstimate(
            videos=videos,
            clips=1,
            duration_seconds=clip_length,
            billable_seconds=billable,
            resolution=resolution,
            tier=rate.tier,
            audio=request.generate_audio,
            per_second_usd=rate.per_second,
            per_clip_usd=rate.per_clip,
            audio_per_second_usd=rate.audio_per_second,
            audio_per_clip_usd=rate.audio_per_clip,
            base_cost_usd=base,
            audio_cost_usd=audio_cost,
            amount_usd=base + audio_cost,
            currency=rate.currency,
            pricing_version=rate.pricing_version,
        )

    def estimate_requests(
        self,
        requests: Iterable[VideoRequest],
        *,
        tier: str | None = None,
    ) -> VideoCostEstimate:
        """Estimate a sequence of clips, such as a split adaptation."""
        items = tuple(requests)
        if not items:
            raise ValueError("at least one video request is required")
        estimates = tuple(self.estimate(request, tier=tier) for request in items)
        tiers = {item.tier for item in estimates if item.tier is not None}
        tier_name: str | None
        if not tiers:
            tier_name = None
        elif len(tiers) == 1:
            tier_name = next(iter(tiers))
        else:
            tier_name = "+".join(sorted(tiers))
        first = estimates[0]
        return VideoCostEstimate(
            videos=sum(item.videos for item in estimates),
            clips=len(estimates),
            duration_seconds=math.fsum(item.duration_seconds for item in estimates),
            billable_seconds=math.fsum(item.billable_seconds for item in estimates),
            resolution=first.resolution
            if len({item.resolution for item in estimates}) == 1
            else None,
            tier=tier_name,
            audio=any(item.audio for item in estimates),
            per_second_usd=first.per_second_usd,
            per_clip_usd=first.per_clip_usd,
            audio_per_second_usd=first.audio_per_second_usd,
            audio_per_clip_usd=first.audio_per_clip_usd,
            base_cost_usd=math.fsum(item.base_cost_usd for item in estimates),
            audio_cost_usd=math.fsum(item.audio_cost_usd for item in estimates),
            amount_usd=math.fsum(item.amount_usd for item in estimates),
            currency=first.currency,
            pricing_version=first.pricing_version,
        )

    def estimate_adaptation(
        self,
        adaptation: Any,
        *,
        tier: str | None = None,
    ) -> VideoCostEstimate:
        """Estimate an adapted request by pricing each of its clips."""
        clips = getattr(adaptation, "clips", None)
        if clips is None:
            raise TypeError("adaptation must expose a clips sequence")
        return self.estimate_requests(clips, tier=tier)


VideoEstimator = VideoPricingEstimator


def estimate_video_cost(
    request: VideoRequest,
    pricing: VideoPricing | Mapping[str, Any] | None = None,
    *,
    tier: str | None = None,
    duration_seconds: float | None = None,
) -> VideoCostEstimate:
    """Estimate one request without constructing a long-lived estimator."""
    return VideoPricingEstimator(pricing).estimate(
        request,
        tier=tier,
        duration_seconds=duration_seconds,
    )


def estimate_video_requests(
    requests: Iterable[VideoRequest],
    pricing: VideoPricing | Mapping[str, Any] | None = None,
    *,
    tier: str | None = None,
) -> VideoCostEstimate:
    """Estimate a sequence of clips without constructing an estimator."""
    return VideoPricingEstimator(pricing).estimate_requests(requests, tier=tier)


def estimate_video_adaptation(
    adaptation: Any,
    pricing: VideoPricing | Mapping[str, Any] | None = None,
    *,
    tier: str | None = None,
) -> VideoCostEstimate:
    """Estimate an adapted video request without constructing an estimator."""
    return VideoPricingEstimator(pricing).estimate_adaptation(
        adaptation,
        tier=tier,
    )


estimate_video_price = estimate_video_cost
price_video = estimate_video_cost
video_cost = estimate_video_cost

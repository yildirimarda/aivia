"""Adapt a video request to per-clip limits and optionally split it.

The shared media adaptation layer already snaps sizes, clamps durations and
rejects unsupported options. Video adds one decision that only makes sense for
a timeline: a request longer than the provider's per-clip cap can either be
clamped (the default) or divided into several shorter clips that are generated
separately and joined afterwards. Splitting multiplies the number of paid calls,
so it happens only when it is explicitly enabled. A request that pins a first or
last frame is never split, because each piece would otherwise be generated
against the same frame instead of continuing the intended motion; the request is
clamped instead and the reason is recorded.

Every change is recorded as an :class:`AdaptationWarning` and the original
request is never mutated. The result carries the adapted clips together with
the combined cost of every provider call, so a caller can budget a split before
submitting it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from aivia.core.config.capability import CapabilityChainConfig
from aivia.providers.media.adapt import (
    AdaptationAction,
    AdaptationWarning,
    MediaConstraints,
    UnadaptableRequestError,
)
from aivia.providers.media.adapt import (
    adapt_video_request as _adapt_video_request,
)
from aivia.providers.protocols.video import VideoRequest

__all__ = [
    "VideoAdaptation",
    "adapt_video_request",
    "split_duration",
]

_EPSILON = 1e-9


def _positive(value: float, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def _non_negative(value: float, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return number


def split_duration(
    seconds: float,
    *,
    max_clip_seconds: float,
    min_clip_seconds: float | None = None,
) -> tuple[float, ...]:
    """Divide *seconds* into the fewest clips that respect *max_clip_seconds*.

    The clips are equal in length and their sum is exactly *seconds*, so no
    content is lost to rounding. An even split also keeps every clip above the
    provider minimum when a capped-chunk split would leave a short tail. When
    the minimum makes an even split impossible,
    :class:`UnadaptableRequestError` is raised instead of returning clips the
    provider would reject.
    """
    total = _positive(seconds, label="seconds")
    maximum = _positive(max_clip_seconds, label="max_clip_seconds")
    minimum = (
        None
        if min_clip_seconds is None
        else _positive(min_clip_seconds, label="min_clip_seconds")
    )
    if minimum is not None and minimum > maximum:
        raise UnadaptableRequestError(
            "the provider minimum clip length exceeds the per-clip cap",
            details={
                "min_clip_seconds": minimum,
                "max_clip_seconds": maximum,
            },
        )
    count = max(1, math.ceil(total / maximum - _EPSILON))
    part = total / count
    if minimum is not None and part < minimum - _EPSILON:
        raise UnadaptableRequestError(
            "the request cannot be split into clips that satisfy both the "
            "minimum and the per-clip cap",
            details={
                "duration_seconds": total,
                "clips": count,
                "clip_seconds": part,
                "min_clip_seconds": minimum,
                "max_clip_seconds": maximum,
            },
        )
    return tuple(part for _ in range(count))


def _clip_cost(
    request: VideoRequest,
    *,
    cost_per_second: float,
    cost_per_clip: float,
) -> float:
    duration = request.duration_seconds or 0.0
    return (cost_per_clip + cost_per_second * duration) * request.count


def _pinned_frame_text(request: VideoRequest) -> str | None:
    has_first = request.first_frame is not None
    has_last = request.last_frame is not None
    if has_first and has_last:
        return "first and last frames"
    if has_first:
        return "first frame"
    if has_last:
        return "last frame"
    return None


@dataclass(frozen=True, slots=True)
class VideoAdaptation:
    """The adapted video clips, their warnings and their combined cost.

    ``clips`` holds one request per provider call. A request that fits the
    provider cap produces a single clip; a split produces several requests with
    the same options and a share of the duration. ``combined_cost_usd`` is the
    sum over every generated video, so ``count`` variations and every split
    call are included.
    """

    clips: tuple[VideoRequest, ...]
    warnings: tuple[AdaptationWarning, ...] = ()
    combined_cost_usd: float = 0.0

    @property
    def request(self) -> VideoRequest:
        """The first adapted clip, matching the single-request media result."""
        return self.clips[0]

    @property
    def requests(self) -> tuple[VideoRequest, ...]:
        """Alias for :attr:`clips`."""
        return self.clips

    @property
    def split(self) -> bool:
        """Whether the request was divided into more than one clip."""
        return len(self.clips) > 1

    @property
    def clip_count(self) -> int:
        """The number of provider calls the adaptation needs."""
        return len(self.clips)

    @property
    def video_count(self) -> int:
        """The number of videos produced, including every variation."""
        return sum(clip.count for clip in self.clips)

    @property
    def total_duration_seconds(self) -> float:
        """The summed duration of one variation across every clip."""
        return math.fsum(clip.duration_seconds or 0.0 for clip in self.clips)

    @property
    def cost_usd(self) -> float:
        """Alias for :attr:`combined_cost_usd`."""
        return self.combined_cost_usd

    @property
    def changed(self) -> bool:
        """Whether adaptation changed at least one value."""
        return bool(self.warnings)

    @property
    def messages(self) -> tuple[str, ...]:
        """The human-readable warning messages in adaptation order."""
        return tuple(warning.message for warning in self.warnings)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary for logs and sidecars."""
        return {
            "clips": len(self.clips),
            "videos": self.video_count,
            "durations": [clip.duration_seconds for clip in self.clips],
            "combined_cost_usd": self.combined_cost_usd,
            "warnings": [warning.to_dict() for warning in self.warnings],
        }


def adapt_video_request(
    request: VideoRequest,
    constraints: MediaConstraints,
    *,
    split: bool | None = None,
    max_clip_seconds: float | None = None,
    config: CapabilityChainConfig | None = None,
    cost_per_second: float = 0.0,
    cost_per_clip: float = 0.0,
) -> VideoAdaptation:
    """Adapt *request* to *constraints*, optionally splitting a long clip.

    The per-clip cap is the smaller of *max_clip_seconds* and the provider's
    ``constraints.max_duration_seconds``. *config* supplies the split flag and
    the cap from a capability chain; explicit arguments win. A split is made
    only when it is enabled, a positive cap exists and the requested duration
    exceeds that cap. A request that pins a first or last frame is never split:
    it is clamped instead and a warning records that the pinned frame prevented
    the split. Otherwise the request is clamped exactly like the shared media
    adaptation layer does.

    *cost_per_second* and *cost_per_clip* price each generated video, and
    ``count`` multiplies the result. They are intentionally simple inputs: the
    full video pricing model is a separate concern.
    """
    per_second = _non_negative(cost_per_second, label="cost_per_second")
    per_clip = _non_negative(cost_per_clip, label="cost_per_clip")

    if config is not None:
        if split is None:
            split = config.split_long_requests
        if max_clip_seconds is None and config.max_clip_seconds > 0:
            max_clip_seconds = config.max_clip_seconds
    split_requested = bool(split)
    pinned_frames = _pinned_frame_text(request)
    allow_split = split_requested and pinned_frames is None

    caps = [
        _positive(cap, label="max_clip_seconds")
        for cap in (max_clip_seconds, constraints.max_duration_seconds)
        if cap is not None and cap > 0
    ]
    cap = min(caps) if caps else None

    base_constraints = constraints
    if cap is not None:
        if allow_split:
            base_constraints = constraints.model_copy(
                update={"max_duration_seconds": None}
            )
        elif split_requested and pinned_frames is not None:
            base_constraints = constraints.model_copy(
                update={"max_duration_seconds": cap}
            )
    adapted = _adapt_video_request(request, base_constraints)
    warnings = list(adapted.warnings)
    requested_duration = request.duration_seconds
    duration = adapted.request.duration_seconds

    clips: tuple[VideoRequest, ...]
    needs_split = (
        cap is not None
        and requested_duration is not None
        and requested_duration > cap + _EPSILON
    )
    if not split_requested or not needs_split:
        clips = (adapted.request,)
    elif pinned_frames is not None:
        assert cap is not None
        clips = (adapted.request,)
        warnings.append(
            AdaptationWarning(
                action=AdaptationAction.SKIPPED_SPLIT,
                option="duration_seconds",
                message=(
                    "duration_seconds was not split because the request pins "
                    f"the {pinned_frames}; it was clamped to the {cap:g}s "
                    "per-clip limit"
                ),
                requested=requested_duration,
                applied=adapted.request.duration_seconds,
            )
        )
    else:
        assert cap is not None and duration is not None
        parts = split_duration(
            duration,
            max_clip_seconds=cap,
            min_clip_seconds=constraints.min_duration_seconds,
        )
        clips = tuple(
            adapted.request.model_copy(update={"duration_seconds": part})
            for part in parts
        )
        warnings.append(
            AdaptationWarning(
                action=AdaptationAction.SPLIT_DURATION,
                option="duration_seconds",
                message=(
                    f"duration_seconds was split into {len(parts)} clips of "
                    f"{parts[0]:g}s each to fit the {cap:g}s per-clip limit"
                ),
                requested=duration,
                applied=parts,
            )
        )

    combined = math.fsum(
        _clip_cost(
            clip,
            cost_per_second=per_second,
            cost_per_clip=per_clip,
        )
        for clip in clips
    )
    return VideoAdaptation(
        clips=clips,
        warnings=tuple(warnings),
        combined_cost_usd=combined,
    )

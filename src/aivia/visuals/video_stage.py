"""Resolve video beats into clips that fit their beat, cap and platform.

A video beat carries a generation prompt rather than a stock query. This stage
turns every video request into a typed
:class:`~aivia.providers.protocols.video.VideoRequest`, hands it to the media
service and keeps the stored asset together with the prompt hash, provider and
model that produced it. Callers never choose a vendor or write a cost event:
the default path is ``context.media.video``, which runs the same
screen -> cache -> budget -> router -> provider -> validate -> store -> cost ->
sidecar order as every other paid call.

A clip never outlives its beat. The requested duration is clamped to the beat's
own duration and to a per-clip cap (the stage option or the router's
``video_gen`` capability configuration), and the platform's declared video
aspect ratio is applied when a platform is given. Resolution is deterministic
for the same service responses and requests: beats keep their input order,
non-video beats are ignored and an empty prompt is reported as unresolved
instead of being sent to a provider.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from aivia.core.errors import ConfigError
from aivia.core.models.asset import Asset, compute_prompt_hash
from aivia.core.models.run import StageResult, StageState
from aivia.core.models.script import MediaKind, VisualType
from aivia.core.platforms import Platform, PlatformCatalog, get_platform
from aivia.core.router import Router
from aivia.core.stages.base import Stage, StageContext
from aivia.providers.protocols.video import VideoRequest
from aivia.visuals.image_stage import (
    _asset_from,
    _call_method,
    _normalise_asset,
    _outcome_values,
    _positive_int,
    _text,
    _write_json,
)
from aivia.visuals.strategy import VisualRequest, resolve_visual_requests

__all__ = [
    "DEFAULT_VIDEOS_FILENAME",
    "DEFAULT_VIDEO_FILENAME",
    "DEFAULT_VIDEO_MANIFEST_FILENAME",
    "VIDEO_MANIFEST_FILENAME",
    "VIDEO_STAGE_NAME",
    "ResolvedVideo",
    "VideoBeat",
    "VideoBeatResult",
    "VideoDecision",
    "VideoResolver",
    "VideoResult",
    "VideoStage",
    "VideoStageResult",
    "build_video_request",
    "build_video_stage",
    "clamp_duration",
    "clamp_video_duration",
    "resolve_video",
    "resolve_video_beat",
    "resolve_video_beats",
    "resolve_video_request",
    "resolve_video_request_sync",
    "resolve_video_requests",
    "resolve_video_requests_sync",
    "resolve_videos",
    "run_video_stage",
    "video_duration",
    "video_stage",
]

VIDEO_STAGE_NAME = "video"
"""The recipe stage name this implementation serves."""

DEFAULT_VIDEO_FILENAME = "video.json"
"""The manifest that records every resolved video beat under the run directory."""

DEFAULT_VIDEOS_FILENAME = "videos.json"
"""Plural alias of :data:`DEFAULT_VIDEO_FILENAME` kept for callers and manifests."""

DEFAULT_VIDEO_MANIFEST_FILENAME = DEFAULT_VIDEO_FILENAME
"""Alias for :data:`DEFAULT_VIDEO_FILENAME`."""

VIDEO_MANIFEST_FILENAME = DEFAULT_VIDEO_FILENAME
"""Alias for :data:`DEFAULT_VIDEO_FILENAME`."""

DEFAULT_VIDEO_COUNT = 1
"""How many variations one beat requests unless the beat says otherwise."""

DEFAULT_PROMPT_METADATA_KEY = "prompt"
"""The request metadata key checked when a beat has no prompt of its own."""

_CAP_ALIASES = (
    "max_clip_seconds",
    "max_clip",
    "max_clip_duration_seconds",
    "max_clip_duration",
    "max_duration_seconds",
    "per_clip_cap",
    "clip_cap",
    "clip_cap_seconds",
)

_DURATION_ALIASES = ("duration_seconds", "duration", "seconds", "length_seconds")

_BEAT_DURATION_KEYS = ("beat_duration_seconds", "beat_seconds", "beat_duration")

_VIDEO_STAGE_OPTION_KEYS = frozenset(
    {
        "model",
        "aspect_ratio",
        "negative_prompt",
        "seed",
        "count",
        "width",
        "height",
        "format",
        "fps",
        "generate_audio",
        "audio_prompt",
        "duration_seconds",
        "max_clip_seconds",
        "platform",
        "platforms",
    }
)


def _request_metadata(request: VisualRequest) -> Mapping[str, Any]:
    metadata = getattr(request, "metadata", None)
    return metadata if isinstance(metadata, Mapping) else {}


def _coerce_request(value: Any) -> VisualRequest:
    if isinstance(value, VisualRequest):
        return value
    if isinstance(value, Mapping):
        return VisualRequest.model_validate(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ConfigError("video resolution needs a non-empty prompt")
        return VisualRequest(
            visual_type=VisualType.VIDEO,
            kind=MediaKind.VIDEO,
            prompt=text,
            text=text,
        )
    requests = resolve_visual_requests(value, strategy="video")
    return requests[0]


def _prompt_of(request: VisualRequest, metadata: Mapping[str, Any]) -> str:
    return (
        _text(request.prompt)
        or _text(metadata.get(DEFAULT_PROMPT_METADATA_KEY))
        or _text(request.text)
    )


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number", details={"value": repr(value)})
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ConfigError(
            f"{name} must be finite and positive",
            details={"value": repr(value)},
        )
    return number


def _clip_cap(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            "max_clip_seconds must be a number",
            details={"value": repr(value)},
        )
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ConfigError(
            "max_clip_seconds must be finite and non-negative",
            details={"value": repr(value)},
        )
    return number or None


def _beat_seconds(
    request: VisualRequest,
    metadata: Mapping[str, Any],
    explicit: float | None = None,
) -> float | None:
    if explicit is not None:
        return _positive_float(explicit, "beat_duration_seconds")
    for key in _BEAT_DURATION_KEYS:
        value = metadata.get(key)
        if value is not None:
            return _positive_float(value, key)
    return request.duration_seconds


def clamp_video_duration(
    duration_seconds: float | None,
    *,
    beat_seconds: float | None = None,
    max_clip_seconds: float | None = None,
) -> float | None:
    """Return the clip duration clamped to the beat and the per-clip cap.

    ``duration_seconds`` is the requested length. When it is missing the beat's
    own duration is used, and when that is missing too the result is ``None`` so
    the provider default applies. The per-clip cap is only applied to a known
    duration; a cap of ``0`` means there is no cap.
    """
    target = (
        _positive_float(duration_seconds, "duration_seconds")
        if duration_seconds is not None
        else None
    )
    beat = (
        _positive_float(beat_seconds, "beat_seconds")
        if beat_seconds is not None
        else None
    )
    cap = _clip_cap(max_clip_seconds)
    if target is None:
        target = beat
    if target is None:
        return None
    if beat is not None:
        target = min(target, beat)
    if cap is not None:
        target = min(target, cap)
    return target


video_duration = clamp_video_duration
clamp_duration = clamp_video_duration
"""Aliases for :func:`clamp_video_duration`."""


def _platform_catalog(value: Any) -> PlatformCatalog | None:
    if value is None:
        return None
    if isinstance(value, PlatformCatalog):
        return value
    if isinstance(value, Mapping):
        payload = dict(value)
        if "platforms" not in payload:
            payload = {"platforms": payload}
        return PlatformCatalog.from_mapping(payload)
    raise ConfigError(
        "platforms must be a PlatformCatalog or mapping",
        details={"value": type(value).__name__},
    )


def _resolve_platform(
    value: Platform | str | None,
    *,
    catalog: PlatformCatalog | None = None,
) -> Platform | None:
    if value is None:
        return None
    if isinstance(value, Platform):
        return value
    try:
        return get_platform(str(value), catalog=catalog)
    except (KeyError, ValueError) as exc:
        raise ConfigError(
            f"unknown platform: {value!r}",
            details={"platform": str(value)},
        ) from exc


def _platform_aspect_ratio(platform: Platform | None) -> str:
    if platform is None:
        return ""
    limits = platform.limits.for_media_type("video") or platform.limits.primary
    return _text(limits.aspect_ratio)


def _chosen_aspect(
    request: VisualRequest,
    *,
    aspect_ratio: str | None,
    platform: Platform | None,
) -> str:
    explicit = _text(aspect_ratio)
    if explicit:
        return explicit
    from_platform = _platform_aspect_ratio(platform)
    if from_platform:
        return from_platform
    return _text(request.aspect_ratio)


def _metadata_value(
    request: VisualRequest,
    metadata: Mapping[str, Any],
    name: str,
) -> Any:
    if name in metadata:
        return metadata[name]
    return getattr(request, name, None)


def _video_plan(
    request: VisualRequest,
    *,
    metadata: Mapping[str, Any],
    model: str | None = None,
    aspect_ratio: str | None = None,
    duration_seconds: float | None = None,
    max_clip_seconds: float | None = None,
    beat_duration_seconds: float | None = None,
    platform: Platform | None = None,
    fps: int | None = None,
    fmt: Any = None,
    generate_audio: bool | None = None,
    audio_prompt: str | None = None,
    count: int | None = None,
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
) -> tuple[VideoRequest | None, dict[str, Any]]:
    request_meta = _request_metadata(request)
    active_meta = dict(request_meta)
    active_meta.update(metadata)
    prompt = _prompt_of(request, active_meta)
    beat = _beat_seconds(request, active_meta, beat_duration_seconds)
    requested = (
        request.duration_seconds
        if request.duration_seconds is not None
        else duration_seconds
    )
    cap = _clip_cap(max_clip_seconds)
    duration = clamp_video_duration(
        requested,
        beat_seconds=beat,
        max_clip_seconds=cap,
    )
    chosen_aspect = _chosen_aspect(
        request,
        aspect_ratio=aspect_ratio,
        platform=platform,
    )
    clamped = bool(
        requested is not None and duration is not None and duration < requested - 1e-9
    )
    info: dict[str, Any] = {
        "prompt": prompt,
        "duration_seconds": duration,
        "requested_duration_seconds": requested,
        "beat_duration_seconds": beat,
        "max_clip_seconds": cap,
        "aspect_ratio": chosen_aspect,
        "clamped": clamped,
        "platform": platform.id if platform is not None else "",
    }
    if not prompt:
        return None, info

    values: dict[str, Any] = {"prompt": prompt}
    chosen_negative = _text(_metadata_value(request, active_meta, "negative_prompt"))
    if chosen_negative:
        values["negative_prompt"] = chosen_negative
    chosen_model = _text(model) or _text(_metadata_value(request, active_meta, "model"))
    if chosen_model:
        values["model"] = chosen_model
    if chosen_aspect:
        values["aspect_ratio"] = chosen_aspect
    if duration is not None:
        values["duration_seconds"] = duration
    chosen_seed = (
        seed if seed is not None else _metadata_value(request, active_meta, "seed")
    )
    if isinstance(chosen_seed, int) and not isinstance(chosen_seed, bool):
        values["seed"] = chosen_seed
    chosen_count = count if count is not None else active_meta.get("count")
    if chosen_count is None:
        chosen_count = DEFAULT_VIDEO_COUNT
    values["count"] = _positive_int(chosen_count, "count") or DEFAULT_VIDEO_COUNT
    chosen_width = width if width is not None else active_meta.get("width")
    chosen_height = height if height is not None else active_meta.get("height")
    if chosen_width is not None and chosen_height is not None:
        values["width"] = _positive_int(chosen_width, "width")
        values["height"] = _positive_int(chosen_height, "height")
    chosen_format = (
        fmt if fmt is not None else _metadata_value(request, active_meta, "format")
    )
    if chosen_format is not None:
        values["format"] = chosen_format
    chosen_fps = (
        fps if fps is not None else _metadata_value(request, active_meta, "fps")
    )
    if chosen_fps is not None:
        values["fps"] = _positive_int(chosen_fps, "fps")
    chosen_audio = (
        generate_audio
        if generate_audio is not None
        else _metadata_value(request, active_meta, "generate_audio")
    )
    if chosen_audio is not None:
        values["generate_audio"] = bool(chosen_audio)
    chosen_audio_prompt = _text(
        audio_prompt
        if audio_prompt is not None
        else _metadata_value(request, active_meta, "audio_prompt")
    )
    if chosen_audio_prompt:
        values["audio_prompt"] = chosen_audio_prompt
    call_metadata = dict(active_meta)
    call_metadata.update(
        {
            "stage": VIDEO_STAGE_NAME,
            "beat_index": request.index,
            "duration_seconds": duration,
            "aspect_ratio": chosen_aspect,
        }
    )
    values["metadata"] = call_metadata
    return VideoRequest(**values), info


def _video_method_of(target: Any) -> Any:
    if callable(target) and not any(
        callable(getattr(target, name, None))
        for name in ("video", "generate", "generate_video")
    ):
        return target
    for name in ("video", "generate_video", "generate", "resolve"):
        candidate = getattr(target, name, None)
        if callable(candidate):
            return candidate
    if callable(target):
        return target
    return None


async def _call_video(
    target: Any,
    request: VideoRequest,
    *,
    context: StageContext | None,
    run_id: str,
    metadata: Mapping[str, Any] | None,
    token: Any = None,
    operation: str | None = None,
    **options: Any,
) -> Any:
    if target is None:
        if context is None:
            raise ConfigError(
                "video resolution needs a video service or a stage context router",
                details={"stage": VIDEO_STAGE_NAME},
            )
        try:
            target = context.media
        except ValueError as exc:
            raise ConfigError(
                "video resolution needs a video service or a stage context router",
                details={"stage": VIDEO_STAGE_NAME},
            ) from exc
    if isinstance(target, Router):
        from aivia.media.video_service import VideoService

        target = VideoService(target)
    method = _video_method_of(target)
    if method is None:
        raise ConfigError(
            "video service must be callable or expose a video or generate method",
            details={"value": type(target).__name__},
        )
    active_token = token
    if active_token is None:
        screen = getattr(target, "screen", None)
        if callable(screen):
            active_token = screen(request.prompt)
    values: dict[str, Any] = {
        "run_id": run_id,
        "metadata": metadata,
        "token": active_token,
        "operation": operation,
        **options,
    }
    values = {name: value for name, value in values.items() if value is not None}
    return await _call_method(method, cast(Any, request), values)


class VideoBeatResult(BaseModel):
    """The resolved video asset for one beat, or why it was not resolved.

    Attribute access falls through to the stored
    :class:`~aivia.core.models.asset.Asset`, so a caller may read
    ``result.path`` or ``result.seed`` as well as ``result.asset``.
    ``duration_seconds`` is the length sent to the provider after clamping to
    the beat and the per-clip cap, while ``requested_duration_seconds`` keeps
    the unclamped value for the manifest.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    index: int = Field(default=0, ge=0)
    request: VisualRequest | None = None
    asset: Asset | None = None
    prompt_hash: str = Field(
        default="",
        validation_alias=AliasChoices("prompt_hash", "prompt_sha256"),
    )
    provider: str = ""
    model: str = ""
    cost_usd: float = Field(default=0.0, ge=0.0)
    cached: bool = False
    error: str = ""
    duration_seconds: float | None = None
    requested_duration_seconds: float | None = None
    beat_duration_seconds: float | None = None
    max_clip_seconds: float | None = None
    aspect_ratio: str = ""
    clamped: bool = False
    platform: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    outcome: Any = Field(default=None, exclude=True)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        asset = self.__dict__.get("asset")
        if asset is None:
            raise AttributeError(name)
        return getattr(asset, name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Asset):
            return self.asset == other
        return super().__eq__(other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    @property
    def resolved(self) -> bool:
        """Whether a video asset was stored."""
        return self.asset is not None

    @property
    def passed(self) -> bool:
        """Alias for :attr:`resolved`."""
        return self.resolved

    @property
    def video(self) -> Asset | None:
        """Alias for :attr:`asset`."""
        return self.asset

    @property
    def clip(self) -> Asset | None:
        """Alias for :attr:`asset`."""
        return self.asset

    @property
    def selected(self) -> Asset | None:
        """Alias for :attr:`asset`."""
        return self.asset

    @property
    def first(self) -> Asset | None:
        """Alias for :attr:`asset`."""
        return self.asset

    @property
    def first_asset(self) -> Asset | None:
        """Alias for :attr:`asset`."""
        return self.asset

    @property
    def prompt_sha256(self) -> str:
        """Alias for :attr:`prompt_hash`."""
        return self.prompt_hash

    @property
    def prompt_digest(self) -> str:
        """Alias for :attr:`prompt_hash`."""
        return self.prompt_hash

    @property
    def cost(self) -> float:
        """Alias for :attr:`cost_usd`."""
        return self.cost_usd

    @property
    def result(self) -> Any:
        """The raw service outcome, when one was returned."""
        return self.outcome

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for the manifest."""
        return {
            "index": self.index,
            "request": (self.request.to_dict() if self.request is not None else None),
            "asset": (
                self.asset.model_dump(mode="json") if self.asset is not None else None
            ),
            "prompt_hash": self.prompt_hash,
            "provider": self.provider,
            "model": self.model,
            "cost_usd": self.cost_usd,
            "cached": self.cached,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "requested_duration_seconds": self.requested_duration_seconds,
            "beat_duration_seconds": self.beat_duration_seconds,
            "max_clip_seconds": self.max_clip_seconds,
            "aspect_ratio": self.aspect_ratio,
            "clamped": self.clamped,
            "platform": self.platform,
            "metadata": dict(self.metadata),
        }


VideoResult = VideoBeatResult
VideoBeat = VideoBeatResult
ResolvedVideo = VideoBeatResult
VideoDecision = VideoBeatResult


class VideoStageResult(StageResult):
    """A stage result carrying one :class:`VideoBeatResult` per video beat."""

    beats: list[VideoBeatResult] = Field(
        default_factory=list,
        validation_alias=AliasChoices("beats", "results", "items", "selections"),
    )

    @property
    def results(self) -> list[VideoBeatResult]:
        """Alias for :attr:`beats`."""
        return self.beats

    @property
    def videos(self) -> list[VideoBeatResult]:
        """Alias for :attr:`beats`."""
        return self.beats

    @property
    def clips(self) -> list[VideoBeatResult]:
        """Alias for :attr:`beats`."""
        return self.beats

    @property
    def assets(self) -> list[Asset]:
        """Every stored asset in beat order."""
        return [beat.asset for beat in self.beats if beat.asset is not None]

    @property
    def selected(self) -> list[Asset]:
        """Alias for :attr:`assets`."""
        return self.assets

    @property
    def unresolved(self) -> list[VideoBeatResult]:
        """Beats for which no video asset was stored."""
        return [beat for beat in self.beats if beat.asset is None]

    @property
    def prompt_hashes(self) -> dict[int, str]:
        """The prompt hash keyed by beat index."""
        return {beat.index: beat.prompt_hash for beat in self.beats}

    @property
    def durations(self) -> dict[int, float | None]:
        """The clamped clip duration keyed by beat index."""
        return {beat.index: beat.duration_seconds for beat in self.beats}

    @property
    def aspect_ratios(self) -> dict[int, str]:
        """The applied aspect ratio keyed by beat index."""
        return {beat.index: beat.aspect_ratio for beat in self.beats}

    @property
    def total_duration_seconds(self) -> float:
        """The summed clamped duration of every resolved beat."""
        return math.fsum(
            beat.duration_seconds or 0.0
            for beat in self.beats
            if beat.asset is not None
        )

    @property
    def providers(self) -> list[str]:
        """Every provider in beat order, de-duplicated."""
        return list(
            dict.fromkeys(beat.provider for beat in self.beats if beat.provider)
        )

    @property
    def models(self) -> list[str]:
        """Every model in beat order, de-duplicated."""
        return list(dict.fromkeys(beat.model for beat in self.beats if beat.model))

    @property
    def asset(self) -> Asset | None:
        """The first stored asset, or ``None``."""
        assets = self.assets
        return assets[0] if assets else None

    @property
    def first(self) -> Asset | None:
        """Alias for :attr:`asset`."""
        return self.asset

    @property
    def provider(self) -> str:
        """The first resolved provider, or an empty string."""
        providers = self.providers
        return providers[0] if providers else ""

    @property
    def model(self) -> str:
        """The first resolved model, or an empty string."""
        models = self.models
        return models[0] if models else ""

    @property
    def prompt_hash(self) -> str:
        """The first resolved prompt hash, or an empty string."""
        return next((beat.prompt_hash for beat in self.beats if beat.prompt_hash), "")


async def resolve_video_request(
    value: Any,
    *,
    service: Any = None,
    video_service: Any = None,
    video: Any = None,
    generator: Any = None,
    provider: Any = None,
    router: Any = None,
    context: StageContext | None = None,
    config: Any = None,
    model: str | None = None,
    aspect_ratio: str | None = None,
    negative_prompt: str | None = None,
    seed: int | None = None,
    count: int | None = None,
    width: int | None = None,
    height: int | None = None,
    duration_seconds: float | None = None,
    beat_duration_seconds: float | None = None,
    max_clip_seconds: float | None = None,
    platform: Platform | str | None = None,
    platforms: PlatformCatalog | Mapping[str, Any] | None = None,
    fps: int | None = None,
    fmt: Any = None,
    format: Any = None,
    generate_audio: bool | None = None,
    audio_prompt: str | None = None,
    token: Any = None,
    screening_token: Any = None,
    run_id: str = "",
    metadata: Mapping[str, Any] | None = None,
    operation: str | None = None,
    **values: Any,
) -> VideoBeatResult:
    """Resolve one video beat through the media service.

    ``service`` (or its aliases ``video``, ``generator``, ``provider``) may be a
    :class:`~aivia.media.video_service.VideoService`, a media facade, a router
    or an async callable. When it is omitted the stage context's media facade
    is used. The returned result is unresolved only when the beat has no prompt
    or the service returned no asset.
    """
    request = _coerce_request(value)
    active_meta = dict(metadata or {})
    if negative_prompt is not None:
        active_meta.setdefault("negative_prompt", negative_prompt)
    active_catalog = _platform_catalog(platforms)
    active_platform = _resolve_platform(platform, catalog=active_catalog)
    video_request, info = _video_plan(
        request,
        metadata=active_meta,
        model=model,
        aspect_ratio=aspect_ratio,
        duration_seconds=duration_seconds,
        max_clip_seconds=max_clip_seconds,
        beat_duration_seconds=beat_duration_seconds,
        platform=active_platform,
        fps=fps,
        fmt=fmt if fmt is not None else format,
        generate_audio=generate_audio,
        audio_prompt=audio_prompt,
        count=count,
        width=width,
        height=height,
        seed=seed,
    )
    common = {
        "index": request.index,
        "request": request,
        "requested_duration_seconds": info["requested_duration_seconds"],
        "beat_duration_seconds": info["beat_duration_seconds"],
        "max_clip_seconds": info["max_clip_seconds"],
        "aspect_ratio": info["aspect_ratio"],
        "clamped": info["clamped"],
        "platform": info["platform"],
    }
    if video_request is None:
        return VideoBeatResult(
            **common,
            metadata={**active_meta, "reason": "no prompt"},
        )

    target = service if service is not None else video_service
    if target is None:
        target = video
    if target is None:
        target = generator if generator is not None else provider
    if target is None and router is not None:
        target = router
    active_token = token if token is not None else screening_token
    outcome = await _call_video(
        target,
        video_request,
        context=context,
        run_id=run_id,
        metadata=metadata,
        token=active_token,
        operation=operation,
        **values,
    )
    asset = _asset_from(outcome)
    outcome_provider, outcome_model, cost, cached = _outcome_values(outcome)
    if asset is None:
        return VideoBeatResult(
            **common,
            provider=outcome_provider,
            model=outcome_model,
            cost_usd=cost,
            cached=cached,
            duration_seconds=video_request.duration_seconds,
            error="video service returned no asset",
            metadata=dict(active_meta),
            outcome=outcome,
        )
    asset = _normalise_asset(
        asset,
        cast(Any, video_request),
        provider=outcome_provider,
        model=outcome_model,
    )
    prompt_hash = asset.prompt_hash or compute_prompt_hash(video_request.prompt)
    return VideoBeatResult(
        **common,
        asset=asset,
        prompt_hash=prompt_hash,
        provider=asset.provider or outcome_provider,
        model=asset.model or outcome_model,
        cost_usd=cost,
        cached=cached,
        duration_seconds=video_request.duration_seconds,
        metadata=dict(active_meta),
        outcome=outcome,
    )


resolve_video_beat = resolve_video_request
resolve_video = resolve_video_request
build_video_request = resolve_video_request
"""Aliases for :func:`resolve_video_request`."""


async def resolve_video_requests(
    values: Any,
    **kwargs: Any,
) -> list[VideoBeatResult]:
    """Resolve every video request in *values*, preserving input order."""
    if isinstance(values, VisualRequest):
        requests = [values]
    elif isinstance(values, Mapping):
        requests = [_coerce_request(values)]
    elif isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray)
    ):
        requests = [_coerce_request(item) for item in values]
    else:
        requests = resolve_visual_requests(values, strategy="video")
    results: list[VideoBeatResult] = []
    for position, request in enumerate(requests):
        result = await resolve_video_request(request, **kwargs)
        if result.index != position and "index" not in request.model_fields_set:
            result = result.model_copy(update={"index": position})
        results.append(result)
    return results


resolve_video_beats = resolve_video_requests
resolve_videos = resolve_video_requests
"""Aliases for :func:`resolve_video_requests`."""


def resolve_video_request_sync(value: Any, **kwargs: Any) -> VideoBeatResult:
    """Run :func:`resolve_video_request` from synchronous code."""
    import asyncio

    return asyncio.run(resolve_video_request(value, **kwargs))


def resolve_video_requests_sync(values: Any, **kwargs: Any) -> list[VideoBeatResult]:
    """Run :func:`resolve_video_requests` from synchronous code."""
    import asyncio

    return asyncio.run(resolve_video_requests(values, **kwargs))


class VideoResolver:
    """Reusable video resolver carrying the service and request defaults."""

    def __init__(
        self,
        service: Any = None,
        *,
        video_service: Any = None,
        video: Any = None,
        generator: Any = None,
        provider: Any = None,
        router: Any = None,
        config: Any = None,
        model: str | None = None,
        aspect_ratio: str | None = None,
        negative_prompt: str | None = None,
        seed: int | None = None,
        count: int | None = None,
        width: int | None = None,
        height: int | None = None,
        duration_seconds: float | None = None,
        beat_duration_seconds: float | None = None,
        max_clip_seconds: float | None = None,
        platform: Platform | str | None = None,
        platforms: PlatformCatalog | Mapping[str, Any] | None = None,
        fps: int | None = None,
        fmt: Any = None,
        format: Any = None,
        generate_audio: bool | None = None,
        audio_prompt: str | None = None,
        token: Any = None,
        run_id: str = "",
        metadata: Mapping[str, Any] | None = None,
        operation: str | None = None,
        **options: Any,
    ) -> None:
        self.service = service if service is not None else video_service
        if self.service is None:
            self.service = video
        if self.service is None:
            self.service = generator if generator is not None else provider
        self.router = router
        self.config = config
        self.model = model
        self.aspect_ratio = aspect_ratio
        self.negative_prompt = negative_prompt
        self.seed = seed
        self.count = count
        self.width = width
        self.height = height
        self.duration_seconds = duration_seconds
        self.beat_duration_seconds = beat_duration_seconds
        self.max_clip_seconds = max_clip_seconds
        self.platform = platform
        self.platforms = platforms
        self.fps = fps
        self.fmt = fmt if fmt is not None else format
        self.generate_audio = generate_audio
        self.audio_prompt = audio_prompt
        self.token = token
        self.run_id = run_id
        self.metadata = metadata
        self.operation = operation
        self.options = dict(options)

    def _values(self, **overrides: Any) -> dict[str, Any]:
        values: dict[str, Any] = {
            "service": self.service,
            "router": self.router,
            "config": self.config,
            "model": self.model,
            "aspect_ratio": self.aspect_ratio,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
            "count": self.count,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.duration_seconds,
            "beat_duration_seconds": self.beat_duration_seconds,
            "max_clip_seconds": self.max_clip_seconds,
            "platform": self.platform,
            "platforms": self.platforms,
            "fps": self.fps,
            "fmt": self.fmt,
            "generate_audio": self.generate_audio,
            "audio_prompt": self.audio_prompt,
            "token": self.token,
            "run_id": self.run_id,
            "metadata": self.metadata,
            "operation": self.operation,
            **self.options,
        }
        values.update(overrides)
        return values

    async def resolve(
        self,
        value: Any,
        *,
        context: StageContext | None = None,
        **overrides: Any,
    ) -> VideoBeatResult:
        return await resolve_video_request(
            value,
            context=context,
            **self._values(**overrides),
        )

    async def resolve_beat(
        self,
        value: Any,
        **overrides: Any,
    ) -> VideoBeatResult:
        return await self.resolve(value, **overrides)

    async def resolve_requests(
        self,
        values: Any,
        *,
        context: StageContext | None = None,
        **overrides: Any,
    ) -> list[VideoBeatResult]:
        return await resolve_video_requests(
            values,
            context=context,
            **self._values(**overrides),
        )

    __call__ = resolve


VideoBeatResolver = VideoResolver
VideoRequestResolver = VideoResolver


def _config_cap(config: Any) -> float | None:
    if not isinstance(config, Mapping):
        return None
    media = config.get("media")
    if not isinstance(media, Mapping):
        media = config
    video_gen = media.get("video_gen")
    if not isinstance(video_gen, Mapping):
        return None
    for key in _CAP_ALIASES:
        if key in video_gen:
            return _clip_cap(video_gen[key])
    return None


class VideoStage(Stage):
    """Resolve every video beat into a stored, costed clip.

    ``requests`` may be a script, a beat sequence, one or more visual requests
    or mappings. A ``script``/``beats`` option is resolved with the video
    strategy first. ``service``/``video``/``generator`` supplies the video
    service; without one the stage uses ``context.media.video``. Each request
    duration is clamped to the beat and to ``max_clip_seconds`` (falling back
    to the router's ``video_gen`` cap), and ``platform`` supplies the video
    aspect ratio. Non-video beats are ignored so a mixed visual plan can be
    resolved in one pass.
    """

    name = VIDEO_STAGE_NAME
    artifacts: tuple[str, ...] = (DEFAULT_VIDEO_FILENAME,)

    def __init__(
        self,
        name: str | None = None,
        *,
        options: Mapping[str, Any] | None = None,
        requests: Any = None,
        visuals: Any = None,
        visual_requests: Any = None,
        script: Any = None,
        beats: Any = None,
        service: Any = None,
        video_service: Any = None,
        video: Any = None,
        generator: Any = None,
        provider: Any = None,
        video_provider: Any = None,
        router: Any = None,
        config: Any = None,
        model: str | None = None,
        aspect_ratio: str | None = None,
        negative_prompt: str | None = None,
        seed: int | None = None,
        count: int | None = None,
        width: int | None = None,
        height: int | None = None,
        duration_seconds: float | None = None,
        beat_duration_seconds: float | None = None,
        max_clip_seconds: float | None = None,
        platform: Platform | str | None = None,
        platforms: PlatformCatalog | Mapping[str, Any] | None = None,
        fps: int | None = None,
        format: Any = None,
        generate_audio: bool | None = None,
        audio_prompt: str | None = None,
        filename: str | None = None,
        manifest_filename: str | None = None,
        **kwargs: Any,
    ) -> None:
        merged = dict(options or {})
        supplied = {
            "requests": requests if requests is not None else visual_requests,
            "visuals": visuals,
            "script": script,
            "beats": beats,
            "service": service,
            "video_service": video_service,
            "video": video,
            "generator": generator,
            "provider": provider,
            "video_provider": video_provider,
            "router": router,
            "config": config,
            "model": model,
            "aspect_ratio": aspect_ratio,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "count": count,
            "width": width,
            "height": height,
            "duration_seconds": duration_seconds,
            "beat_duration_seconds": beat_duration_seconds,
            "max_clip_seconds": max_clip_seconds,
            "platform": platform,
            "platforms": platforms,
            "fps": fps,
            "format": format,
            "generate_audio": generate_audio,
            "audio_prompt": audio_prompt,
            "filename": filename,
            "manifest_filename": manifest_filename,
        }
        for key, value in supplied.items():
            if value is not None:
                merged[key] = value
        self.filename = _text(
            merged.pop("filename", None)
            or merged.pop("manifest_filename", None)
            or DEFAULT_VIDEO_FILENAME
        )
        if not self.filename:
            self.filename = DEFAULT_VIDEO_FILENAME
        self.artifacts = (self.filename,)

        self._requests_input = merged.pop("requests", None)
        if self._requests_input is None:
            self._requests_input = merged.pop("visuals", None)
        self._script = merged.pop("script", None)
        self._beats = merged.pop("beats", None)
        self._service = merged.pop("service", None)
        if self._service is None:
            self._service = merged.pop("video_service", None)
        if self._service is None:
            self._service = merged.pop("video", None)
        if self._service is None:
            self._service = merged.pop("generator", None)
        if self._service is None:
            self._service = merged.pop("provider", None)
        if self._service is None:
            self._service = merged.pop("video_provider", None)
        self._router = merged.pop("router", None)
        self._config = merged.pop("config", None)
        self.model = merged.pop("model", None)
        self.aspect_ratio = _text(merged.pop("aspect_ratio", None))
        self.negative_prompt = merged.pop("negative_prompt", None)
        self.seed = merged.pop("seed", None)
        self.count = merged.pop("count", None)
        self.width = merged.pop("width", None)
        self.height = merged.pop("height", None)
        self.duration_seconds = merged.pop("duration_seconds", None)
        if self.duration_seconds is None:
            for alias in _DURATION_ALIASES:
                if alias in merged:
                    self.duration_seconds = merged.pop(alias)
                    break
        self.beat_duration_seconds = merged.pop("beat_duration_seconds", None)
        self.max_clip_seconds = merged.pop("max_clip_seconds", None)
        if self.max_clip_seconds is None:
            for alias in _CAP_ALIASES:
                if alias in merged:
                    self.max_clip_seconds = merged.pop(alias)
                    break
        self.platform = merged.pop("platform", None)
        self.platforms = merged.pop("platforms", None)
        self.fps = merged.pop("fps", None)
        self.format = merged.pop("format", None)
        self.generate_audio = merged.pop("generate_audio", None)
        self.audio_prompt = merged.pop("audio_prompt", None)
        self.results: list[VideoBeatResult] = []
        super().__init__(name=name, options=merged, **kwargs)

    def _requests(self) -> list[VisualRequest]:
        value = self._requests_input
        if value is None:
            value = self._script if self._script is not None else self._beats
        if value is None:
            return []
        if isinstance(value, VisualRequest):
            return [value]
        if isinstance(value, Mapping):
            return [_coerce_request(value)]
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return [_coerce_request(item) for item in value]
        return resolve_visual_requests(
            value,
            strategy="video",
            aspect_ratio=self.aspect_ratio,
            duration_seconds=self.duration_seconds,
        )

    def _cap(self, context: StageContext) -> float | None:
        if self.max_clip_seconds is not None:
            return _clip_cap(self.max_clip_seconds)
        router = self._router if self._router is not None else context.router
        if router is not None and hasattr(router, "config_for"):
            capability = router.config_for("video_gen")
            cap = _clip_cap(getattr(capability, "max_clip_seconds", 0.0))
            if cap is not None:
                return cap
        config = self._config if self._config is not None else context.config
        return _config_cap(config)

    def _platform(self) -> Platform | None:
        catalog = _platform_catalog(self.platforms)
        return _resolve_platform(self.platform, catalog=catalog)

    async def run(self, context: StageContext) -> StageResult:
        """Resolve every video beat and write the ``video.json`` manifest."""
        requests = self._requests()
        video_requests = [
            request for request in requests if request.visual_type is VisualType.VIDEO
        ]
        run_id = context.run.id if context.run is not None else ""
        platform = self._platform()
        max_clip_seconds = self._cap(context)
        results: list[VideoBeatResult] = []
        for position, request in enumerate(video_requests):
            result = await resolve_video_request(
                request,
                service=self._service,
                router=self._router if self._router is not None else context.router,
                context=context,
                config=self._config if self._config is not None else context.config,
                model=self.model,
                aspect_ratio=self.aspect_ratio,
                negative_prompt=self.negative_prompt,
                seed=self.seed,
                count=self.count,
                width=self.width,
                height=self.height,
                duration_seconds=self.duration_seconds,
                beat_duration_seconds=self.beat_duration_seconds,
                max_clip_seconds=max_clip_seconds,
                platform=platform,
                fps=self.fps,
                fmt=self.format,
                generate_audio=self.generate_audio,
                audio_prompt=self.audio_prompt,
                run_id=run_id,
                metadata={"position": position},
            )
            results.append(result)
        self.results = results

        resolved_count = sum(1 for result in results if result.resolved)
        unresolved_count = len(results) - resolved_count
        payload: dict[str, Any] = {
            "version": 1,
            "stage": self.name,
            "beats": [result.to_dict() for result in results],
            "metadata": {
                "resolved": resolved_count,
                "unresolved": unresolved_count,
                "max_clip_seconds": max_clip_seconds,
                "platform": platform.id if platform is not None else "",
                "aspect_ratio": _platform_aspect_ratio(platform),
                "durations": [result.duration_seconds for result in results],
                "prompt_hashes": [
                    result.prompt_hash for result in results if result.prompt_hash
                ],
            },
        }
        relative = self.filename
        _write_json(context.run_dir / relative, payload)
        if relative == DEFAULT_VIDEO_FILENAME:
            _write_json(context.run_dir / DEFAULT_VIDEOS_FILENAME, payload)
        total_cost = sum(result.cost_usd for result in results)
        providers = [result.provider for result in results if result.provider]
        models = [result.model for result in results if result.model]
        return VideoStageResult(
            run_id=run_id,
            stage=self.name,
            status=StageState.SUCCEEDED,
            artifacts=[relative],
            cost_usd=total_cost,
            beats=results,
            metadata={
                "beats": len(results),
                "resolved": resolved_count,
                "unresolved": unresolved_count,
                "max_clip_seconds": max_clip_seconds,
                "platform": platform.id if platform is not None else "",
                "aspect_ratio": _platform_aspect_ratio(platform),
                "durations": [result.duration_seconds for result in results],
                "providers": list(dict.fromkeys(providers)),
                "models": list(dict.fromkeys(models)),
                "prompt_hashes": [
                    result.prompt_hash for result in results if result.prompt_hash
                ],
                "manifest": relative,
            },
        )


video_stage = VideoStage
"""Alias for :class:`VideoStage`."""


def build_video_stage(
    *,
    options: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> VideoStage:
    """Build a :class:`VideoStage` from recipe options and overrides."""
    return VideoStage(options=options, **kwargs)


async def run_video_stage(
    context: StageContext,
    *,
    stage: VideoStage | None = None,
    **options: Any,
) -> StageResult:
    """Build a stage unless one is supplied, then run it asynchronously."""
    if stage is not None and options:
        raise ConfigError("pass either a stage or options, not both")
    active = stage if stage is not None else VideoStage(**options)
    return await active.run(context)

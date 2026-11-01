"""Run one video generation request through the single paid path.

The service is the only place a video call is executed. It requires a
screening token for the prompt, answers an identical request from the video
asset cache when possible, applies the remaining budget, asks the router for a
provider, validates and stores every returned clip, records one actual cost
event and writes a cost sidecar next to each output.

The order matches the project invariant: screen -> cache -> budget -> router ->
provider -> validate -> store -> cost -> sidecar. A cache hit short-circuits the
paid path but still records a zero-amount event per reused clip, so the reuse is
visible in the ledger without counting as spend.

Provider failures are kept as :class:`VideoFailure` records even when the
router falls back to another provider, so a caller can explain a run that
succeeded only after a retry. A failed call never records a cost event.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlsplit

import httpx
from sqlalchemy import Engine

from aivia.core.asset_cache import AssetCache, AssetCacheKey
from aivia.core.budget import BudgetGuard
from aivia.core.config.budget import BudgetConfig
from aivia.core.cost import CostTracker
from aivia.core.errors import (
    BudgetError,
    ConfigError,
    ProviderExhaustedError,
    ValidationError,
)
from aivia.core.media_budget import (
    CheaperHint,
    ClipBudgetDecision,
    ClipOption,
    PerClipBudgetPolicy,
)
from aivia.core.media_probe import MediaDownloader, MediaLimits, MediaProbe
from aivia.core.models.asset import (
    Asset,
    AssetAnalysis,
    AssetCostLink,
    AssetKind,
)
from aivia.core.models.base import canonical_json
from aivia.core.models.cost import CostEvent
from aivia.core.models.run import ToolchainFingerprint
from aivia.core.pricing import PricingResolver, ResolvedPrice
from aivia.core.router import Router, RouteSelection
from aivia.core.sidecar import SIDECAR_SUFFIX, build_cost_sidecar, write_cost_sidecar
from aivia.media.accessibility import write_accessibility_sidecars
from aivia.media.adaptation import adapt_for_provider
from aivia.media.screening import (
    DEFAULT_PROMPT_KIND,
    ScreeningToken,
    require_screened,
    screen_prompt,
)
from aivia.media.video_cache import (
    VIDEO_OPERATION,
    MaterialisedVideo,
    VideoCache,
    VideoCacheOutcome,
    validated_video_format,
    validated_video_fps,
    validated_video_has_audio,
    video_cache_key,
)
from aivia.media.video_pricing import (
    VideoCostEstimate,
    VideoPricing,
    VideoPricingEstimator,
)
from aivia.providers.base import Provider
from aivia.providers.media.adapt import AdaptationResult, AdaptationWarning
from aivia.providers.media.dto import MediaKind
from aivia.providers.protocols.storage import (
    StorageObject,
    StorageProvider,
    StoragePutRequest,
)
from aivia.providers.protocols.video import (
    CAPABILITY_VIDEO_FIRST_FRAME,
    CAPABILITY_VIDEO_GEN,
    CAPABILITY_VIDEO_INPUT,
    CAPABILITY_VIDEO_LAST_FRAME,
    VideoProvider,
    VideoRequest,
    VideoResponse,
    VideoResult,
)
from aivia.providers.video.v2v import ensure_source_video_media_type

__all__ = [
    "VIDEO_OPERATION",
    "CheaperHint",
    "ClipBudgetDecision",
    "PerClipBudgetPolicy",
    "VideoFailure",
    "VideoOutcome",
    "VideoService",
    "VideoServiceResult",
    "generate_video",
    "video_service_from_config",
]

ToolchainLike = ToolchainFingerprint | Mapping[str, Any] | str | None
BudgetLike = BudgetGuard | BudgetConfig | Mapping[str, Any] | None
ClipBudgetLike = (
    PerClipBudgetPolicy | BudgetConfig | Mapping[str, Any] | float | int | None
)
ClipCandidates = Mapping[str, float] | Iterable[Any] | None

_TIME_UNITS = frozenset(
    {
        "ms",
        "millisecond",
        "milliseconds",
        "s",
        "sec",
        "secs",
        "second",
        "seconds",
        "min",
        "mins",
        "minute",
        "minutes",
        "h",
        "hr",
        "hrs",
        "hour",
        "hours",
    }
)


def _file_path(url: str) -> Path | None:
    parsed = urlsplit(url)
    if parsed.scheme.lower() != "file":
        return None
    return Path(unquote(parsed.path))


def _redacted_message(error: BaseException) -> str:
    text = str(error).strip()
    return text or type(error).__name__


def _coerce_clip_budget_policy(value: Any) -> PerClipBudgetPolicy:
    if isinstance(value, PerClipBudgetPolicy):
        return value
    return PerClipBudgetPolicy(value)


@dataclass(frozen=True, slots=True)
class VideoFailure:
    """One provider attempt or validation step that did not produce a clip.

    ``stage`` is ``provider`` for a call the router made, ``validation`` for
    bytes that arrived but were rejected, and ``budget`` for a request stopped
    before submit. ``error`` is the stable error code when the exception was an
    :class:`~aivia.core.errors.AiviaError`.
    """

    provider: str
    error: str
    error_type: str
    message: str
    stage: str = "provider"
    retryable: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for logs and manifests."""
        payload: dict[str, Any] = {
            "provider": self.provider,
            "error": self.error,
            "error_type": self.error_type,
            "message": self.message,
            "stage": self.stage,
            "retryable": self.retryable,
        }
        if self.details:
            payload["details"] = dict(self.details)
        return payload


@dataclass(frozen=True, slots=True)
class VideoOutcome:
    """The response, stored clips, price, events, failures and sidecars.

    ``stored`` holds one :class:`~aivia.media.video_cache.VideoCacheOutcome`
    per clip, in response order. ``events`` holds the one actual event for the
    provider call on a miss, or one zero-amount hit event per clip on a hit.
    """

    request: VideoRequest
    response: VideoResponse
    assets: tuple[Asset, ...] = ()
    stored: tuple[VideoCacheOutcome, ...] = ()
    probes: tuple[MediaProbe, ...] = ()
    storage_objects: tuple[StorageObject, ...] = ()
    events: tuple[CostEvent, ...] = ()
    price: VideoCostEstimate | ResolvedPrice | None = None
    selection: RouteSelection | None = None
    sidecars: tuple[Path, ...] = ()
    accessibility_sidecars: tuple[Any, ...] = ()
    cache_hit: bool = False
    failures: tuple[VideoFailure, ...] = ()
    adaptation_warnings: tuple[AdaptationWarning, ...] = ()
    audio_stripped: bool = False
    clip_budget: ClipBudgetDecision | None = None
    cheaper_hint: CheaperHint | None = None

    @property
    def stripped_audio(self) -> bool:
        """Alias for :attr:`audio_stripped`."""
        return self.audio_stripped

    @property
    def budget_decision(self) -> ClipBudgetDecision | None:
        """The per-clip budget decision applied before submit, when any."""
        return self.clip_budget

    @property
    def clip_budget_decision(self) -> ClipBudgetDecision | None:
        """Alias for :attr:`clip_budget`."""
        return self.clip_budget

    @property
    def hint(self) -> CheaperHint | None:
        """The cheaper hint chosen instead of the original option, when any."""
        return self.cheaper_hint

    @property
    def cheaper(self) -> bool:
        """Whether a cheaper option was chosen before submit."""
        return self.cheaper_hint is not None

    @property
    def downgraded(self) -> bool:
        """Whether the per-clip budget policy chose a cheaper option."""
        return bool(self.clip_budget is not None and self.clip_budget.downgraded)

    @property
    def warnings(self) -> tuple[AdaptationWarning, ...]:
        """Every change the adaptation layer made before submission."""
        return self.adaptation_warnings

    @property
    def adaptation_messages(self) -> tuple[str, ...]:
        """The human-readable adaptation warnings in order."""
        return tuple(warning.message for warning in self.adaptation_warnings)

    @property
    def accessibility(self) -> Any:
        """The first accessibility sidecar result, or ``None``."""
        return self.accessibility_sidecars[0] if self.accessibility_sidecars else None

    @property
    def accessibility_sidecar(self) -> Any:
        """Alias for :attr:`accessibility`."""
        return self.accessibility

    @property
    def accessibility_paths(self) -> tuple[Path, ...]:
        """Every written accessibility sidecar path."""
        return tuple(
            path
            for result in self.accessibility_sidecars
            for path in getattr(result, "paths", ())
        )

    @property
    def all_sidecars(self) -> tuple[Path, ...]:
        """Cost and accessibility sidecar paths together."""
        return (*self.sidecars, *self.accessibility_paths)

    @property
    def event(self) -> CostEvent | None:
        """The first recorded cost event, or ``None``."""
        return self.events[0] if self.events else None

    @property
    def videos(self) -> list[VideoResult]:
        """Every validated clip in response order."""
        return list(self.response.videos)

    @property
    def results(self) -> list[VideoResult]:
        """Alias for :attr:`videos`."""
        return self.videos

    @property
    def video(self) -> VideoCacheOutcome | None:
        """The first stored clip outcome, or ``None``."""
        return self.stored[0] if self.stored else None

    @property
    def first(self) -> VideoCacheOutcome | None:
        """Alias for :attr:`video`."""
        return self.video

    @property
    def asset(self) -> Asset | None:
        """The first stored asset, or ``None``."""
        return self.assets[0] if self.assets else None

    @property
    def outputs(self) -> tuple[Asset, ...]:
        """Alias for :attr:`assets`."""
        return self.assets

    @property
    def paths(self) -> tuple[str, ...]:
        """Every stored asset path in response order."""
        return tuple(asset.path for asset in self.assets)

    @property
    def urls(self) -> tuple[str, ...]:
        """Alias for :attr:`paths`."""
        return self.paths

    @property
    def sidecar(self) -> Path | None:
        """The first written sidecar, or ``None``."""
        return self.sidecars[0] if self.sidecars else None

    @property
    def sidecar_path(self) -> Path | None:
        """Alias for :attr:`sidecar`."""
        return self.sidecar

    @property
    def provider(self) -> str:
        """The provider that served or cached the call."""
        if self.response.provider:
            return self.response.provider
        return self.assets[0].provider if self.assets else ""

    @property
    def model(self) -> str:
        """The model that served or cached the call."""
        return self.response.model or (self.assets[0].model if self.assets else "")

    @property
    def cached(self) -> bool:
        """Whether this outcome came from the asset cache."""
        return self.cache_hit

    @property
    def amount_usd(self) -> float:
        """The sum of every recorded event amount."""
        return math.fsum(event.amount_usd for event in self.events)

    @property
    def cost_usd(self) -> float:
        """Alias for :attr:`amount_usd`."""
        return self.amount_usd

    @property
    def total_cost_usd(self) -> float:
        """Alias for :attr:`amount_usd`."""
        return self.amount_usd

    @property
    def estimate(self) -> VideoCostEstimate | None:
        """The estimator result when pricing was supplied as a table."""
        return self.price if isinstance(self.price, VideoCostEstimate) else None

    @property
    def failure(self) -> VideoFailure | None:
        """The first recorded failure, or ``None``."""
        return self.failures[0] if self.failures else None

    @property
    def failed(self) -> bool:
        """Whether any provider attempt or validation step failed."""
        return bool(self.failures)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary for logs and manifests."""
        return {
            "provider": self.provider,
            "model": self.model,
            "cache_hit": self.cache_hit,
            "audio_stripped": self.audio_stripped,
            "amount_usd": self.amount_usd,
            "paths": list(self.paths),
            "failures": [failure.to_dict() for failure in self.failures],
            "adaptation_warnings": [
                warning.to_dict() for warning in self.adaptation_warnings
            ],
            "clip_budget": (
                self.clip_budget.to_dict() if self.clip_budget is not None else None
            ),
            "cheaper_hint": (
                self.cheaper_hint.to_dict() if self.cheaper_hint is not None else None
            ),
        }


VideoServiceResult = VideoOutcome
"""Alias for :class:`VideoOutcome`."""


class VideoService:
    """Execute video generation through a :class:`~aivia.core.router.Router`.

    ``cache`` answers an identical request without a provider call and persists
    the assets of a miss. When it is omitted but an engine or tracker is given,
    the service builds a :class:`~aivia.media.video_cache.VideoCache` on that
    ledger so caching is never silently disabled. ``storage`` stores validated
    clip bytes; without one the asset record is still built but no file is
    written.

    ``pricing`` may be a :class:`~aivia.media.video_pricing.VideoPricing` table
    or estimator. When it is supplied the service prices the validated clips
    with it, using the duration and resolution ffprobe reported. Without one,
    the selected provider's metadata price and ``pricing_overrides`` are used.
    """

    def __init__(
        self,
        router: Router,
        *,
        tracker: CostTracker | None = None,
        cost_tracker: CostTracker | None = None,
        cache: VideoCache | AssetCache | None = None,
        video_cache: VideoCache | None = None,
        asset_cache: AssetCache | None = None,
        engine: Engine | None = None,
        storage: StorageProvider | None = None,
        budget: BudgetLike = None,
        budget_guard: BudgetGuard | None = None,
        clip_budget: ClipBudgetLike = None,
        clip_budget_policy: ClipBudgetLike = None,
        per_clip_budget: ClipBudgetLike = None,
        per_clip_cap: float | None = None,
        media_budget: ClipBudgetLike = None,
        clip_candidates: ClipCandidates = None,
        clip_options: ClipCandidates = None,
        safety: Any = None,
        safety_config: Any = None,
        screening: Any = None,
        screening_config: Any = None,
        screener: Any = None,
        limits: MediaLimits | None = None,
        media_limits: MediaLimits | None = None,
        downloader: MediaDownloader | None = None,
        http: Any = None,
        transport: httpx.AsyncBaseTransport | None = None,
        pricing: VideoPricing | Mapping[str, Any] | VideoPricingEstimator | None = None,
        pricing_estimator: VideoPricingEstimator | None = None,
        pricing_overrides: Mapping[str, float] | None = None,
        pricing_version: str = "",
        operation: str = VIDEO_OPERATION,
        toolchain: ToolchainLike = None,
        toolchain_fingerprint: ToolchainLike = None,
        key_prefix: str | None = None,
        record_cost: bool = True,
        record_sidecar: bool = True,
        accessibility: Any = None,
        accessibility_config: Any = None,
        no_cache: bool = False,
        ffprobe_executable: str = "ffprobe",
        ffprobe_timeout_seconds: float | None = None,
        ffprobe_runner: Any = None,
        ffmpeg_executable: str = "ffmpeg",
        ffmpeg_runner: Any = None,
    ) -> None:
        if not isinstance(router, Router):
            raise TypeError("router must be a Router")
        if tracker is not None and cost_tracker is not None:
            raise TypeError("pass either tracker or cost_tracker, not both")
        if budget is not None and budget_guard is not None:
            raise TypeError("pass either budget or budget_guard, not both")
        if limits is not None and media_limits is not None:
            raise TypeError("pass either limits or media_limits, not both")
        if toolchain is not None and toolchain_fingerprint is not None:
            raise TypeError("pass either toolchain or toolchain_fingerprint, not both")
        if downloader is not None and (http is not None or transport is not None):
            raise TypeError(
                "pass either a downloader or http/transport options, not both"
            )
        if pricing is not None and pricing_estimator is not None:
            raise TypeError("pass either pricing or pricing_estimator, not both")
        if cache is not None and (video_cache is not None or asset_cache is not None):
            raise TypeError("pass only one of cache, video_cache or asset_cache")
        if video_cache is not None and asset_cache is not None:
            raise TypeError("pass either video_cache or asset_cache, not both")

        active_tracker = tracker if tracker is not None else cost_tracker
        if active_tracker is not None and not isinstance(active_tracker, CostTracker):
            raise TypeError("tracker must be a CostTracker")

        selected_video: VideoCache | None = None
        selected_asset: AssetCache | None = asset_cache
        if isinstance(cache, VideoCache):
            selected_video = cache
        elif isinstance(cache, AssetCache):
            selected_asset = cache
        elif cache is not None:
            raise TypeError("cache must be a VideoCache or AssetCache")
        if video_cache is not None:
            if not isinstance(video_cache, VideoCache):
                raise TypeError("video_cache must be a VideoCache")
            selected_video = video_cache

        selected_limits = limits if limits is not None else media_limits
        if selected_limits is not None and not isinstance(selected_limits, MediaLimits):
            raise TypeError("limits must be a MediaLimits")

        selected_downloader = downloader
        if selected_downloader is not None and not isinstance(
            selected_downloader, MediaDownloader
        ):
            raise TypeError("downloader must be a MediaDownloader")
        if selected_downloader is None and (http is not None or transport is not None):
            selected_downloader = MediaDownloader(
                http=http,
                transport=transport,
                limits=selected_limits,
            )

        if selected_video is None:
            if selected_asset is None and engine is not None:
                selected_asset = AssetCache(
                    engine,
                    cost=active_tracker,
                    no_cache=no_cache,
                )
            elif (
                selected_asset is None
                and active_tracker is not None
                and active_tracker.repository.engine is not None
            ):
                selected_asset = AssetCache(
                    active_tracker.repository.engine,
                    cost=active_tracker,
                    no_cache=no_cache,
                )
            if selected_asset is not None:
                if active_tracker is None:
                    active_tracker = selected_asset.cost
                selected_video = VideoCache(
                    cache=selected_asset,
                    storage=storage,
                    downloader=selected_downloader,
                    limits=selected_limits,
                    no_cache=no_cache,
                    ffprobe_executable=ffprobe_executable,
                    ffprobe_timeout_seconds=(
                        ffprobe_timeout_seconds
                        if ffprobe_timeout_seconds is not None
                        else _default_ffprobe_timeout()
                    ),
                    ffprobe_runner=ffprobe_runner,
                    ffmpeg_executable=ffmpeg_executable,
                    ffmpeg_runner=ffmpeg_runner,
                )
            else:
                selected_video = VideoCache(
                    storage=storage,
                    downloader=selected_downloader,
                    limits=selected_limits,
                    no_cache=no_cache,
                    ffprobe_executable=ffprobe_executable,
                    ffprobe_timeout_seconds=(
                        ffprobe_timeout_seconds
                        if ffprobe_timeout_seconds is not None
                        else _default_ffprobe_timeout()
                    ),
                    ffprobe_runner=ffprobe_runner,
                    ffmpeg_executable=ffmpeg_executable,
                    ffmpeg_runner=ffmpeg_runner,
                )
        else:
            if no_cache and selected_video.cache is not None:
                selected_video.cache.no_cache = True
            if storage is None:
                storage = selected_video.storage
            if selected_video.cache is not None and active_tracker is None:
                active_tracker = selected_video.cache.cost
            if (
                selected_downloader is not None
                or selected_limits is not None
                or ffprobe_runner is not None
                or ffprobe_timeout_seconds is not None
                or ffprobe_executable != "ffprobe"
                or ffmpeg_runner is not None
                or ffmpeg_executable != "ffmpeg"
            ) and selected_video.cache is not None:
                selected_video = VideoCache(
                    cache=selected_video.cache,
                    storage=storage,
                    downloader=selected_downloader,
                    limits=selected_limits,
                    no_cache=no_cache,
                    ffprobe_executable=ffprobe_executable,
                    ffprobe_timeout_seconds=(
                        ffprobe_timeout_seconds
                        if ffprobe_timeout_seconds is not None
                        else _default_ffprobe_timeout()
                    ),
                    ffprobe_runner=ffprobe_runner,
                    ffmpeg_executable=ffmpeg_executable,
                    ffmpeg_runner=ffmpeg_runner,
                )

        supplied_safety = [
            value
            for value in (safety, safety_config, screening, screening_config)
            if value is not None
        ]
        if len(supplied_safety) > 1:
            raise TypeError("pass only one safety or screening configuration")
        selected_safety = supplied_safety[0] if supplied_safety else None
        if (
            screener is None
            and selected_safety is not None
            and hasattr(selected_safety, "require")
        ):
            screener = selected_safety
            selected_safety = None
        if not isinstance(operation, str) or not operation.strip():
            raise ValueError("operation must be a non-empty string")
        if key_prefix is not None:
            if not isinstance(key_prefix, str) or not key_prefix.strip():
                raise ValueError("key_prefix must be a non-empty string")
            key_prefix = key_prefix.strip().strip("/")

        budget_config: BudgetConfig | None = None
        if isinstance(budget, BudgetConfig):
            budget_config = budget
        elif isinstance(budget, Mapping):
            budget_config = BudgetConfig.from_mapping(budget)
        elif isinstance(budget_guard, BudgetGuard):
            budget_config = budget_guard.config

        supplied_clip_budgets = [
            value
            for value in (
                clip_budget,
                clip_budget_policy,
                per_clip_budget,
                per_clip_cap,
                media_budget,
            )
            if value is not None
        ]
        if len(supplied_clip_budgets) > 1:
            raise TypeError(
                "pass only one of clip_budget, clip_budget_policy, "
                "per_clip_budget, per_clip_cap or media_budget"
            )
        selected_clip_budget: Any = (
            supplied_clip_budgets[0] if supplied_clip_budgets else None
        )
        if (
            selected_clip_budget is None
            and budget_config is not None
            and bool(budget_config.per_video)
        ):
            selected_clip_budget = budget_config
        active_clip_budget = (
            _coerce_clip_budget_policy(selected_clip_budget)
            if selected_clip_budget is not None
            else None
        )

        supplied_candidates = [
            value for value in (clip_candidates, clip_options) if value is not None
        ]
        if len(supplied_candidates) > 1:
            raise TypeError("pass either clip_candidates or clip_options, not both")
        active_clip_candidates = supplied_candidates[0] if supplied_candidates else None

        active_budget = budget_guard if budget_guard is not None else budget
        if isinstance(active_budget, (BudgetConfig, Mapping)):
            active_budget = BudgetGuard(active_budget, active_tracker)
        if active_budget is not None and not isinstance(active_budget, BudgetGuard):
            raise TypeError("budget must be a BudgetGuard, BudgetConfig or mapping")

        active_pricing = pricing if pricing is not None else pricing_estimator
        estimator: VideoPricingEstimator | None
        if active_pricing is None:
            estimator = None
        elif isinstance(active_pricing, VideoPricingEstimator):
            estimator = active_pricing
        elif isinstance(active_pricing, VideoPricing):
            estimator = VideoPricingEstimator(active_pricing)
        elif isinstance(active_pricing, Mapping):
            estimator = VideoPricingEstimator(VideoPricing.from_mapping(active_pricing))
        else:
            raise TypeError(
                "pricing must be a VideoPricing, mapping or VideoPricingEstimator"
            )

        self._router = router
        self._tracker = active_tracker
        self._videos = selected_video
        self._storage = storage
        self._budget = active_budget
        self._clip_budget = active_clip_budget
        self._clip_candidates = active_clip_candidates
        self._safety = selected_safety
        self._screener = screener
        self._limits = selected_limits or MediaLimits()
        self._downloader = selected_downloader
        self._operation = operation.strip()
        self._toolchain = toolchain if toolchain is not None else toolchain_fingerprint
        self._key_prefix = key_prefix
        self._record_cost = bool(record_cost)
        self._record_sidecar = bool(record_sidecar)
        self._accessibility = (
            accessibility if accessibility is not None else accessibility_config
        )
        self._no_cache = bool(no_cache)
        self._estimator = estimator
        self._pricing = PricingResolver(
            pricing_overrides,
            pricing_version=pricing_version,
        )
        self._last_outcome: VideoOutcome | None = None
        self._last_selection: RouteSelection | None = None
        self._last_event: CostEvent | None = None
        self._last_price: VideoCostEstimate | ResolvedPrice | None = None
        self._last_failures: tuple[VideoFailure, ...] = ()

    @property
    def router(self) -> Router:
        """The router every call goes through."""
        return self._router

    @property
    def tracker(self) -> CostTracker | None:
        """The default cost tracker, when one was configured."""
        return self._tracker

    @property
    def videos(self) -> VideoCache:
        """The video cache that validates, stores and reuses clips."""
        return self._videos

    @property
    def cache(self) -> VideoCache:
        """Alias for :attr:`videos`."""
        return self._videos

    @property
    def storage(self) -> StorageProvider | None:
        """The storage provider, when one was configured."""
        return self._storage

    @property
    def budget(self) -> BudgetGuard | None:
        """The budget guard, when one was configured."""
        return self._budget

    @property
    def clip_budget(self) -> PerClipBudgetPolicy | None:
        """The per-clip budget policy applied before every submit."""
        return self._clip_budget

    @property
    def clip_budget_policy(self) -> PerClipBudgetPolicy | None:
        """Alias for :attr:`clip_budget`."""
        return self._clip_budget

    @property
    def per_clip_budget(self) -> PerClipBudgetPolicy | None:
        """Alias for :attr:`clip_budget`."""
        return self._clip_budget

    @property
    def limits(self) -> MediaLimits:
        """The caps applied to every downloaded or returned clip."""
        return self._limits

    @property
    def estimator(self) -> VideoPricingEstimator | None:
        """The pricing estimator supplied to the service, if any."""
        return self._estimator

    @property
    def last_outcome(self) -> VideoOutcome | None:
        """The most recent outcome."""
        return self._last_outcome

    @property
    def last_selection(self) -> RouteSelection | None:
        """The most recent successful router selection."""
        return self._last_selection

    @property
    def last_event(self) -> CostEvent | None:
        """The most recent recorded cost event."""
        return self._last_event

    @property
    def last_price(self) -> VideoCostEstimate | ResolvedPrice | None:
        """The most recent resolved price or estimate."""
        return self._last_price

    @property
    def failures(self) -> tuple[VideoFailure, ...]:
        """Failures recorded by the most recent call."""
        return self._last_failures

    @property
    def last_failures(self) -> tuple[VideoFailure, ...]:
        """Alias for :attr:`failures`."""
        return self._last_failures

    def screen(self, prompt: str) -> ScreeningToken:
        """Screen *prompt* with this service's configuration and return a token."""
        if self._screener is not None:
            return cast(
                ScreeningToken,
                cast(Any, self._screener).screen_prompt(prompt),
            )
        return screen_prompt(prompt, self._safety)

    def _require_screened(
        self,
        token: Any,
        prompt: str,
    ) -> ScreeningToken:
        if self._screener is not None:
            return cast(
                ScreeningToken,
                cast(Any, self._screener).require(
                    token,
                    prompt,
                    kind=DEFAULT_PROMPT_KIND,
                ),
            )
        return require_screened(
            token,
            prompt,
            kind=DEFAULT_PROMPT_KIND,
            config=self._safety,
        )

    def cache_key(self, request: VideoRequest, *, index: int = 0) -> AssetCacheKey:
        """Return the cache key for one output of *request*."""
        if not isinstance(request, VideoRequest):
            raise TypeError("request must be a VideoRequest")
        if self._videos.cache is None:
            return video_cache_key(request, index=index)
        return self._videos.key(request, index=index)

    @staticmethod
    def _required_capabilities(request: VideoRequest) -> tuple[str, ...]:
        required: list[str] = []
        if request.first_frame is not None:
            required.append(CAPABILITY_VIDEO_FIRST_FRAME)
        if request.last_frame is not None:
            required.append(CAPABILITY_VIDEO_LAST_FRAME)
        if request.source_video is not None:
            required.append(CAPABILITY_VIDEO_INPUT)
        return tuple(required)

    @staticmethod
    def _router_units(request: VideoRequest) -> tuple[float, str]:
        if request.duration_seconds is not None:
            return request.duration_seconds * request.count, "second"
        return float(request.count), "clip"

    @staticmethod
    def _failure(
        provider: str,
        error: BaseException,
        *,
        stage: str = "provider",
    ) -> VideoFailure:
        return VideoFailure(
            provider=provider,
            error=getattr(error, "code", type(error).__name__),
            error_type=type(error).__name__,
            message=_redacted_message(error),
            stage=stage,
            retryable=bool(getattr(error, "retryable", False)),
            details=dict(getattr(error, "details", {}) or {}),
        )

    def _remember(self, outcome: VideoOutcome) -> None:
        self._last_outcome = outcome
        self._last_selection = outcome.selection
        self._last_event = outcome.event
        self._last_price = outcome.price
        self._last_failures = outcome.failures

    def _provider_clip_estimate(
        self,
        provider: Provider,
        request: VideoRequest,
    ) -> float | None:
        """Price one provider's metadata for *request*, or ``None`` if unpriced."""
        config = self._router.config_for(CAPABILITY_VIDEO_GEN)
        resolver = PricingResolver(config.pricing_overrides)
        units, unit_kind = self._router_units(request)
        try:
            resolved = resolver.resolve(
                provider,
                CAPABILITY_VIDEO_GEN,
                units,
                unit_kind=unit_kind,
                provider_name=provider.name,
            )
        except ConfigError:
            resolved = resolver.resolve(
                provider,
                CAPABILITY_VIDEO_GEN,
                float(request.count),
                provider_name=provider.name,
            )
        if resolved.missing:
            return None
        return resolved.amount_usd

    def _auto_clip_candidates(
        self,
        request: VideoRequest,
        providers: Iterable[Provider],
    ) -> list[ClipOption]:
        """Build cheaper hints from the priced providers in the chain."""
        options: list[ClipOption] = []
        for provider in providers:
            amount = self._provider_clip_estimate(provider, request)
            if amount is None:
                continue
            options.append(ClipOption(provider=provider.name, estimate_usd=amount))
        return options

    def _evaluate_clip_budget(
        self,
        request: VideoRequest,
        *,
        estimator: VideoPricingEstimator | None,
        policy: PerClipBudgetPolicy | None,
        candidates: Any,
        exclude: Iterable[str],
        required: tuple[str, ...],
    ) -> tuple[ClipBudgetDecision | None, set[str]]:
        """Apply the per-clip cap before any provider is called."""
        excluded = frozenset(exclude)
        if policy is None or not policy.cap:
            return None, set(excluded)
        providers = tuple(
            provider
            for provider in self._router.candidates(
                CAPABILITY_VIDEO_GEN,
                required=required or None,
            )
            if provider.name not in excluded
        )
        if not providers:
            return None, set(excluded)
        current = providers[0]
        amount: float | None
        if estimator is not None:
            amount = estimator.estimate(request).amount_usd
        else:
            amount = self._provider_clip_estimate(current, request)
            if amount is None:
                return None, set(excluded)
        options = candidates
        if options is None:
            options = self._auto_clip_candidates(request, providers)
        width, height = request.width, request.height
        decision = policy.evaluate(
            amount,
            provider=current.name,
            model=request.model or "",
            name=current.name,
            resolution=f"{width}x{height}" if width and height else "",
            duration_seconds=request.duration_seconds,
            candidates=options,
        )
        routing_exclude = set(excluded)
        if decision.blocked:
            details = {
                "amount_usd": decision.estimate_usd,
                "clip_budget": decision.to_dict(),
            }
            failure = VideoFailure(
                provider=current.name,
                error="budget_exceeded",
                error_type="BudgetError",
                message=decision.message,
                stage="budget",
                details=details,
            )
            self._last_failures = (failure,)
            raise BudgetError(
                decision.message,
                details={
                    "scope": "per_clip",
                    "cap": decision.cap_usd,
                    "per_clip_cap": decision.cap_usd,
                    "amount_usd": decision.estimate_usd,
                    "provider": current.name,
                    "clip_budget": decision.to_dict(),
                },
            )
        if decision.downgraded and decision.hint is not None:
            hint_name = decision.hint.provider
            available = {provider.name for provider in providers}
            if hint_name in available:
                routing_exclude.update(
                    provider.name
                    for provider in providers
                    if provider.name != hint_name
                )
        return decision, routing_exclude

    async def generate(
        self,
        request: VideoRequest,
        *,
        token: Any = None,
        screening_token: Any = None,
        screen_token: Any = None,
        run_id: str = "",
        tracker: CostTracker | None = None,
        operation: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        record_cost: bool | None = None,
        record_sidecar: bool | None = None,
        remaining_budget: float | None = None,
        no_cache: bool | None = None,
        storage: StorageProvider | None = None,
        key: str | None = None,
        key_prefix: str | None = None,
        pinned: bool = True,
        estimated: bool = False,
        exclude: Iterable[str] | None = None,
        clip_budget: ClipBudgetLike = None,
        clip_budget_policy: ClipBudgetLike = None,
        per_clip_budget: ClipBudgetLike = None,
        per_clip_cap: float | None = None,
        media_budget: ClipBudgetLike = None,
        clip_candidates: ClipCandidates = None,
        clip_options: ClipCandidates = None,
        pricing: VideoPricing | Mapping[str, Any] | VideoPricingEstimator | None = None,
        cost_usd: float | None = None,
        cost_event: CostEvent | Mapping[str, Any] | None = None,
    ) -> VideoOutcome:
        """Generate *request* through the screened paid path.

        ``token`` (or its alias ``screening_token``) must be the token returned
        by :func:`aivia.media.screening.screen_prompt` for this prompt. A
        missing, malformed or mismatched token raises
        :class:`~aivia.core.errors.ContentPolicyError` before the cache, the
        budget, the router or any provider is touched.
        """
        if not isinstance(request, VideoRequest):
            raise TypeError("request must be a VideoRequest")
        ensure_source_video_media_type(request)
        supplied_tokens = [
            value
            for value in (token, screening_token, screen_token)
            if value is not None
        ]
        if len(supplied_tokens) > 1 and any(
            value != supplied_tokens[0] for value in supplied_tokens[1:]
        ):
            raise ValueError("token, screening_token and screen_token disagree")
        active_token = supplied_tokens[0] if supplied_tokens else None
        self._require_screened(active_token, request.prompt)

        if tracker is not None and not isinstance(tracker, CostTracker):
            raise TypeError("tracker must be a CostTracker")
        active_tracker = tracker if tracker is not None else self._tracker
        operation_name = self._operation if operation is None else operation.strip()
        if not operation_name:
            raise ValueError("operation must be a non-empty string")
        active_storage = storage if storage is not None else self._storage
        active_prefix = (
            key_prefix.strip().strip("/")
            if key_prefix is not None
            else self._key_prefix
        )
        if key_prefix is not None and not active_prefix:
            raise ValueError("key_prefix must be a non-empty string")
        use_cache = (
            self._videos.cache is not None
            and not self._videos.cache.no_cache
            and not self._no_cache
            and not bool(no_cache)
        )
        write_sidecar = (
            self._record_sidecar if record_sidecar is None else bool(record_sidecar)
        )
        write_cost = self._record_cost if record_cost is None else bool(record_cost)
        estimator = self._estimator
        if pricing is not None:
            if isinstance(pricing, VideoPricingEstimator):
                estimator = pricing
            elif isinstance(pricing, VideoPricing):
                estimator = VideoPricingEstimator(pricing)
            elif isinstance(pricing, Mapping):
                estimator = VideoPricingEstimator(VideoPricing.from_mapping(pricing))
            else:
                raise TypeError(
                    "pricing must be a VideoPricing, mapping or VideoPricingEstimator"
                )

        supplied_clip_budgets = [
            value
            for value in (
                clip_budget,
                clip_budget_policy,
                per_clip_budget,
                per_clip_cap,
                media_budget,
            )
            if value is not None
        ]
        if len(supplied_clip_budgets) > 1:
            raise TypeError(
                "pass only one of clip_budget, clip_budget_policy, "
                "per_clip_budget, per_clip_cap or media_budget"
            )
        active_clip_budget = self._clip_budget
        if supplied_clip_budgets:
            active_clip_budget = _coerce_clip_budget_policy(supplied_clip_budgets[0])
        supplied_candidates = [
            value for value in (clip_candidates, clip_options) if value is not None
        ]
        if len(supplied_candidates) > 1:
            raise TypeError("pass either clip_candidates or clip_options, not both")
        active_clip_candidates = (
            supplied_candidates[0] if supplied_candidates else self._clip_candidates
        )
        active_exclude = tuple(exclude or ())

        failures: list[VideoFailure] = []

        if use_cache:
            cached = await self._cached_outcome(
                request,
                run_id=run_id,
                operation=operation_name,
                storage=active_storage,
                write_sidecar=write_sidecar,
            )
            if cached is not None:
                self._remember(cached)
                return cached

        budget = remaining_budget
        if budget is None and self._budget is not None:
            budget = self._budget.remaining(run_id, operation=operation_name)

        if estimator is not None and budget is not None:
            estimate = estimator.estimate(request)
            if estimate.amount_usd > budget and not math.isclose(
                estimate.amount_usd,
                budget,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                failure = VideoFailure(
                    provider="",
                    error="budget_exceeded",
                    error_type="BudgetError",
                    message="video request exceeds the remaining budget",
                    stage="budget",
                    details={
                        "amount_usd": estimate.amount_usd,
                        "remaining_budget": budget,
                    },
                )
                self._last_failures = (failure,)
                raise BudgetError(
                    "video request exceeds the remaining budget",
                    details={
                        "amount_usd": estimate.amount_usd,
                        "remaining_budget": budget,
                        "estimate": estimate.to_dict(),
                    },
                )

        required = self._required_capabilities(request)
        clip_budget_decision, routing_exclude = self._evaluate_clip_budget(
            request,
            estimator=estimator,
            policy=active_clip_budget,
            candidates=active_clip_candidates,
            exclude=active_exclude,
            required=required,
        )
        units, unit_kind = self._router_units(request)
        adaptation: AdaptationResult[Any] | None = None

        async def run(provider: Provider) -> VideoResponse:
            nonlocal adaptation
            try:
                adaptation = adapt_for_provider(request, provider, MediaKind.VIDEO)
                adapted = cast(VideoRequest, adaptation.request)
                return await cast(VideoProvider, provider).generate(adapted)
            except BaseException as error:
                failures.append(self._failure(provider.name, error))
                raise

        try:
            response = await self._router.execute(
                CAPABILITY_VIDEO_GEN,
                run,
                required=required or None,
                remaining_budget=budget,
                units=units,
                unit_kind=unit_kind,
                exclude=routing_exclude or None,
            )
        except ProviderExhaustedError as error:
            recorded_providers = {item.provider for item in failures}
            for attempt in error.details.get("attempted", ()):
                if not isinstance(attempt, Mapping):
                    continue
                provider = str(attempt.get("provider", ""))
                if provider in recorded_providers:
                    continue
                failures.append(
                    VideoFailure(
                        provider=provider,
                        error=str(attempt.get("error", "")),
                        error_type=str(attempt.get("error_type", "")),
                        message=str(attempt.get("message", "")),
                        stage="provider",
                        retryable=True,
                        details=dict(attempt),
                    )
                )
            self._last_failures = tuple(failures)
            raise
        if not isinstance(response, VideoResponse):
            raise TypeError("video provider must return a VideoResponse")
        selection = self._router.last_selection
        if selection is None:
            raise RuntimeError("router completed a video call without a selection")
        provider_name = response.provider or selection.name
        model_name = response.model or request.model or ""
        submitted = (
            cast(VideoRequest, adaptation.request)
            if adaptation is not None
            else request
        )
        raw_results = list(response.videos)
        if response.count != submitted.count:
            failure = VideoFailure(
                provider=selection.name,
                error="video_count_mismatch",
                error_type="ValidationError",
                message=(
                    "video provider returned "
                    f"{response.count} clip(s) for a request asking for "
                    f"{submitted.count}"
                ),
                stage="validation",
            )
            failures.append(failure)
            self._last_failures = tuple(failures)
            raise ValidationError(failure.message)

        materialised: list[MaterialisedVideo] = []
        for index, result in enumerate(response.videos):
            try:
                item = await self._videos.materialise_with_policy(
                    result,
                    submitted,
                    index=index,
                    run_id=run_id,
                    storage=active_storage,
                    key=key,
                    key_prefix=active_prefix,
                )
            except BaseException as error:
                failures.append(
                    self._failure(selection.name, error, stage="validation")
                )
                self._last_failures = tuple(failures)
                raise
            materialised.append(item)
        if not materialised:
            raise ValueError("video provider returned no clips")
        audio_stripped = any(item.audio_stripped for item in materialised)

        validated: list[VideoResult] = []
        for result, item in zip(response.videos, materialised, strict=True):
            probe = item.probe
            seed = result.seed if result.seed is not None else submitted.seed
            validated.append(
                VideoResult(
                    data=probe.data,
                    format=validated_video_format(probe, result.format),
                    width=probe.width,
                    height=probe.height,
                    duration_seconds=probe.duration_seconds,
                    fps=validated_video_fps(
                        probe,
                        result.fps if result.fps is not None else submitted.fps,
                    ),
                    has_audio=validated_video_has_audio(
                        probe,
                        result.has_audio,
                    ),
                    seed=seed,
                    metadata=dict(result.metadata),
                )
            )
        response = response.model_copy(
            update={
                "videos": validated,
                "provider": provider_name,
                "model": model_name,
            }
        )

        price: VideoCostEstimate | ResolvedPrice | None
        event: CostEvent
        if estimator is not None:
            estimate = self._estimate_results(
                estimator,
                submitted,
                validated,
            )
            total_duration = math.fsum(
                item.duration_seconds or 0.0 for item in validated
            )
            time_priced = (
                estimate.per_second_usd > 0.0
                or estimate.audio_per_second_usd > 0.0
                or estimate.billable_seconds != float(len(validated))
            )
            units = estimate.billable_seconds if time_priced else float(len(validated))
            unit_kind = "second" if time_priced else "clip"
            details: dict[str, Any] = {
                "video_count": len(validated),
                "duration_seconds": total_duration,
                "billed_units": units,
                "price_unit": unit_kind,
                "price_source": "pricing",
                "estimate": estimate.to_dict(),
            }
            if estimate.tier is not None:
                details["tier"] = estimate.tier
            if estimate.resolution is not None:
                details["resolution"] = list(estimate.resolution)
            details["audio"] = estimate.audio
            details["audio_stripped"] = audio_stripped
            if response.request_id:
                details["request_id"] = response.request_id
            if metadata is not None:
                details.update(metadata)
            if clip_budget_decision is not None:
                details["clip_budget"] = clip_budget_decision.to_dict()
                if clip_budget_decision.hint is not None:
                    details["cheaper_hint"] = clip_budget_decision.hint.to_dict()
            event = CostEvent(
                provider=provider_name,
                model=model_name,
                operation=operation_name,
                units=units,
                unit_kind=unit_kind,
                amount_usd=estimate.amount_usd,
                estimated=estimated,
                run_id=run_id,
                pricing_version=estimate.pricing_version,
                metadata=details,
            )
            price = estimate
        else:
            units, unit_kind = self._metadata_units(
                submitted,
                validated,
                selection.provider,
                operation_name,
            )
            resolved = self._pricing.resolve(
                selection.provider,
                operation_name,
                units,
                unit_kind=unit_kind,
                provider_name=selection.name,
            )
            total_duration = math.fsum(
                item.duration_seconds or 0.0 for item in validated
            )
            details = {
                "video_count": len(validated),
                "duration_seconds": total_duration,
                "billed_units": resolved.billed_units,
                "price_unit": resolved.price_unit,
                "price_source": resolved.source.value,
                "audio_stripped": audio_stripped,
            }
            if resolved.missing:
                details["missing_price"] = True
            if response.request_id:
                details["request_id"] = response.request_id
            if metadata is not None:
                details.update(metadata)
            if clip_budget_decision is not None:
                details["clip_budget"] = clip_budget_decision.to_dict()
                if clip_budget_decision.hint is not None:
                    details["cheaper_hint"] = clip_budget_decision.hint.to_dict()
            event = CostEvent(
                provider=resolved.provider,
                model=model_name,
                operation=operation_name,
                units=resolved.units,
                unit_kind=resolved.unit_kind,
                amount_usd=resolved.amount_usd,
                estimated=estimated,
                run_id=run_id,
                pricing_version=resolved.pricing_version,
                metadata=details,
            )
            price = resolved

        if cost_event is not None:
            if isinstance(cost_event, CostEvent):
                event = cost_event
            else:
                data = dict(cost_event)
                data.setdefault("provider", provider_name)
                data.setdefault("model", model_name)
                data.setdefault("operation", operation_name)
                data.setdefault("units", units)
                data.setdefault("unit_kind", unit_kind)
                data.setdefault("run_id", run_id)
                data.setdefault("estimated", estimated)
                extra = dict(data.get("metadata") or {})
                extra.update(event.metadata)
                data["metadata"] = extra
                event = CostEvent.model_validate(data)
        if cost_usd is not None:
            if isinstance(cost_usd, bool) or not isinstance(cost_usd, (int, float)):
                raise TypeError("cost_usd must be a number")
            amount = float(cost_usd)
            if not math.isfinite(amount) or amount < 0.0:
                raise ValueError("cost_usd must be finite and non-negative")
            event = event.model_copy(update={"amount_usd": amount})

        recorded: CostEvent | None = None
        if write_cost and active_tracker is not None:
            recorded = (
                active_tracker.record_estimate(event)
                if estimated
                else active_tracker.record_actual(event)
            )
            event = recorded

        cost_links = (
            [
                AssetCostLink(
                    cost_event_id=event.id,
                    amount_usd=event.amount_usd,
                    provider=event.provider,
                    model=event.model,
                    operation=event.operation,
                )
            ]
            if recorded is not None
            else []
        )
        assets: list[Asset] = []
        stored_outcomes: list[VideoCacheOutcome] = []
        for index, (result, item) in enumerate(
            zip(response.videos, materialised, strict=True)
        ):
            probe = item.probe
            stored = item.storage_object
            raw = raw_results[index]
            asset = Asset(
                kind=AssetKind.VIDEO,
                provider=provider_name,
                model=model_name,
                prompt=request.prompt,
                path=(
                    (stored.url or stored.key)
                    if stored is not None
                    else (raw.url or "")
                ),
                content_hash=probe.sha256,
                run_id=run_id,
                pinned=pinned,
                seed=result.seed,
                analysis=AssetAnalysis(
                    width=probe.width,
                    height=probe.height,
                    duration_seconds=probe.duration_seconds,
                    fps=result.fps,
                    format=result.format.value,
                    has_audio=result.has_audio,
                    audio_stripped=item.audio_stripped,
                ),
                cost_links=list(cost_links),
            )
            if use_cache and self._videos.cache is not None:
                key_for_asset = self.cache_key(request, index=index)
                asset = self._videos.cache.store(asset, key=key_for_asset)
            assets.append(asset)
            stored_outcomes.append(
                VideoCacheOutcome(
                    request=request,
                    result=result,
                    asset=asset,
                    probe=probe,
                    storage_object=stored,
                    event=event,
                    key=self.cache_key(request, index=index),
                    cache_hit=False,
                    audio_stripped=item.audio_stripped,
                )
            )

        sidecars: tuple[Path, ...] = ()
        if write_sidecar:
            sidecars = await self._write_sidecars(
                assets,
                [event] * len(assets),
                run_id=run_id,
                storage=active_storage,
                stored_objects=[item.storage_object for item in materialised],
            )

        accessibility = self._write_accessibility_sidecars(
            assets,
            request,
        )
        outcome = VideoOutcome(
            request=request,
            response=response,
            assets=tuple(assets),
            stored=tuple(stored_outcomes),
            probes=tuple(item.probe for item in materialised),
            storage_objects=tuple(
                item.storage_object
                for item in materialised
                if item.storage_object is not None
            ),
            events=(event,),
            price=price,
            selection=selection,
            sidecars=sidecars,
            accessibility_sidecars=accessibility,
            cache_hit=False,
            failures=tuple(failures),
            adaptation_warnings=(adaptation.warnings if adaptation is not None else ()),
            audio_stripped=audio_stripped,
            clip_budget=clip_budget_decision,
            cheaper_hint=(
                clip_budget_decision.hint if clip_budget_decision is not None else None
            ),
        )
        self._remember(outcome)
        return outcome

    async def generate_with_cost(
        self,
        request: VideoRequest,
        **values: Any,
    ) -> VideoOutcome:
        """Alias for :meth:`generate` returning the outcome with cost data."""
        return await self.generate(request, **values)

    async def generate_video(
        self,
        request: VideoRequest,
        **values: Any,
    ) -> VideoOutcome:
        """Alias for :meth:`generate`."""
        return await self.generate(request, **values)

    @staticmethod
    def _metadata_units(
        request: VideoRequest,
        results: list[VideoResult],
        provider: Provider,
        operation: str,
    ) -> tuple[float, str]:
        price = provider.metadata.pricing_for(operation)
        unit = price.unit.strip() if price is not None and price.unit else ""
        if unit.lower() in _TIME_UNITS:
            total = math.fsum(
                result.duration_seconds or request.duration_seconds or 0.0
                for result in results
            )
            return total, unit
        if unit:
            return float(len(results)), unit
        return float(len(results)), "clip"

    @staticmethod
    def _estimate_results(
        estimator: VideoPricingEstimator,
        request: VideoRequest,
        results: list[VideoResult],
    ) -> VideoCostEstimate:
        requests = [
            request.model_copy(
                update={
                    "count": 1,
                    "duration_seconds": (
                        result.duration_seconds
                        if result.duration_seconds is not None
                        else request.duration_seconds
                    ),
                    "width": (
                        result.width if result.width is not None else request.width
                    ),
                    "height": (
                        result.height if result.height is not None else request.height
                    ),
                    "generate_audio": bool(result.has_audio or request.generate_audio),
                }
            )
            for result in results
        ]
        return estimator.estimate_requests(requests)

    async def _cached_outcome(
        self,
        request: VideoRequest,
        *,
        run_id: str,
        operation: str,
        storage: StorageProvider | None,
        write_sidecar: bool,
    ) -> VideoOutcome | None:
        cache = self._videos
        if cache.cache is None:
            return None
        hits: list[VideoCacheOutcome] = []
        for index in range(request.count):
            hit = await cache.lookup(
                request,
                index=index,
                run_id=run_id,
                operation=operation,
                record_hit=False,
                storage=storage,
            )
            if hit is None:
                return None
            hits.append(hit)

        events: list[CostEvent] = []
        recorded_hits: list[VideoCacheOutcome] = []
        for index, hit in enumerate(hits):
            key = self.cache_key(request, index=index)
            event = cache.cache.record_hit(
                key,
                hit.asset,
                run_id=run_id,
                operation=operation,
            )
            events.append(event)
            recorded_hits.append(replace(hit, event=event, key=key))
        results = [hit.result for hit in recorded_hits]
        response = VideoResponse(
            videos=results,
            model=recorded_hits[0].asset.model,
            provider=recorded_hits[0].asset.provider,
        )
        assets = tuple(hit.asset for hit in recorded_hits)
        sidecars: tuple[Path, ...] = ()
        if write_sidecar:
            sidecars = await self._write_sidecars(
                list(assets),
                events,
                run_id=run_id,
                storage=storage,
                stored_objects=[],
            )
        accessibility = self._write_accessibility_sidecars(assets, request)
        return VideoOutcome(
            request=request,
            response=response,
            assets=assets,
            stored=tuple(recorded_hits),
            probes=tuple(hit.probe for hit in recorded_hits if hit.probe is not None),
            events=tuple(events),
            price=None,
            selection=None,
            sidecars=sidecars,
            accessibility_sidecars=accessibility,
            cache_hit=True,
            failures=(),
            audio_stripped=any(hit.audio_stripped for hit in recorded_hits),
        )

    def _write_accessibility_sidecars(
        self,
        assets: list[Asset] | tuple[Asset, ...],
        request: VideoRequest,
    ) -> tuple[Any, ...]:
        if self._accessibility is None:
            return ()
        metadata = request.metadata or {}
        written: list[Any] = []
        for asset in assets:
            if not asset.path:
                continue
            path = _file_path(asset.path)
            if path is None:
                continue
            result = write_accessibility_sidecars(
                path,
                self._accessibility,
                alt_text=(
                    metadata.get("alt_text") or metadata.get("alt") or request.prompt
                ),
                subtitle=metadata.get("subtitle"),
                words=metadata.get("words"),
                transcript=metadata.get("transcript"),
                text=metadata.get("text") or request.prompt,
                duration_seconds=asset.analysis.duration_seconds,
            )
            if result:
                written.append(result)
        return tuple(written)

    async def _write_sidecars(
        self,
        assets: list[Asset] | tuple[Asset, ...],
        events: list[CostEvent] | tuple[CostEvent, ...],
        *,
        run_id: str,
        storage: StorageProvider | None,
        stored_objects: list[StorageObject | None],
    ) -> tuple[Path, ...]:
        written: list[Path] = []
        for index, asset in enumerate(assets):
            asset_events = [events[index]] if index < len(events) else list(events)
            if not asset_events:
                continue
            path = _file_path(asset.path) if asset.path else None
            if path is not None:
                written.append(
                    write_cost_sidecar(
                        path,
                        asset_events,
                        toolchain_fingerprint=self._toolchain,
                        run_id=run_id or None,
                    )
                )
                continue
            stored = stored_objects[index] if index < len(stored_objects) else None
            if storage is None or stored is None:
                continue
            sidecar = build_cost_sidecar(
                stored.url or stored.key,
                asset_events,
                toolchain_fingerprint=self._toolchain,
                run_id=run_id or None,
            )
            payload = (canonical_json(sidecar.to_dict(), indent=2) + "\n").encode(
                "utf-8"
            )
            await storage.put(
                StoragePutRequest(
                    data=payload,
                    key=f"{stored.key}{SIDECAR_SUFFIX}",
                    content_type="application/json",
                    overwrite=True,
                )
            )
        return tuple(written)


def _default_ffprobe_timeout() -> float:
    from aivia.core.config.media import DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS

    return DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS


async def generate_video(
    request: VideoRequest,
    router: Router,
    *,
    token: Any = None,
    screening_token: Any = None,
    tracker: CostTracker | None = None,
    cache: VideoCache | AssetCache | None = None,
    storage: StorageProvider | None = None,
    budget: BudgetLike = None,
    safety: Any = None,
    limits: MediaLimits | None = None,
    pricing: VideoPricing | Mapping[str, Any] | VideoPricingEstimator | None = None,
    pricing_overrides: Mapping[str, float] | None = None,
    pricing_version: str = "",
    run_id: str = "",
    metadata: Mapping[str, Any] | None = None,
    remaining_budget: float | None = None,
    **values: Any,
) -> VideoOutcome:
    """Execute one video request without constructing a long-lived service."""
    service = VideoService(
        router,
        tracker=tracker,
        cache=cache,
        storage=storage,
        budget=budget,
        safety=safety,
        limits=limits,
        pricing=pricing,
        pricing_overrides=pricing_overrides,
        pricing_version=pricing_version,
    )
    return await service.generate(
        request,
        token=token,
        screening_token=screening_token,
        run_id=run_id,
        metadata=metadata,
        remaining_budget=remaining_budget,
        **values,
    )


def video_service_from_config(
    router: Router,
    config: Any = None,
    **values: Any,
) -> VideoService:
    """Build a :class:`VideoService` for *router* and an optional config.

    A mapping with ``chain`` or ``strategy`` keys is applied as the
    ``video_gen`` capability chain, mirroring the image service helper.
    """
    if isinstance(config, Mapping):
        section = config.get("video_gen", config)
        if isinstance(section, Mapping) and (
            "chain" in section or "strategy" in section
        ):
            base = router.config_for(CAPABILITY_VIDEO_GEN)
            merged = base.model_copy(
                update={
                    key: section[key]
                    for key in ("chain", "strategy", "weights", "pricing_overrides")
                    if key in section
                }
            )
            router = Router(
                router.providers,
                configs={CAPABILITY_VIDEO_GEN: merged},
                seed=router.seed,
                circuit_breakers=router.circuits,
                rate_limiters=router.limiters,
            )
    return VideoService(router, **values)

"""Generate or select a still, animate it and keep both cost events linked.

Image to video is two paid stages in one flow. The still may already exist (a
user supplied file, a stock result or a stored :class:`~aivia.core.models.asset.Asset`),
or it may be generated through :class:`~aivia.media.image_service.ImageService`.
The still is then handed to a video provider as the first frame, the validated
clip is stored through :class:`~aivia.media.video_cache.VideoCache` and a cost
event is recorded for the animation.

The image service already records and links its own cost event. This helper
keeps that event, records the video event and links it to the stored video
asset, so a run that generated a still and animated it never loses either half
of the spend. A video cache hit short-circuits the paid animation but still
returns the original linked event together with the zero-amount hit record.

The video call is never made by this module directly. It either runs through a
:class:`~aivia.core.router.Router` (the default, which gives fallback, rate
limits and capability filtering) or through a caller supplied async generator,
so a future video service can replace the transport without changing the flow.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlsplit

from aivia.core.cost import CostTracker
from aivia.core.errors import ValidationError
from aivia.core.models.asset import (
    Asset,
    AssetAnalysis,
    AssetCostLink,
    AssetKind,
    AssetLicense,
)
from aivia.core.models.cost import CostEvent
from aivia.core.pricing import PricingResolver
from aivia.core.router import Router, RouteSelection
from aivia.media.image_service import IMAGE_OPERATION, ImageOutcome, ImageService
from aivia.media.video_cache import VIDEO_OPERATION, VideoCache, VideoCacheOutcome
from aivia.media.video_pricing import VideoPricing, VideoPricingEstimator
from aivia.providers.base import Provider
from aivia.providers.protocols.image import ImageRequest, ImageResult
from aivia.providers.protocols.media import MediaInput
from aivia.providers.protocols.stock import StockAsset
from aivia.providers.protocols.video import (
    CAPABILITY_VIDEO_FIRST_FRAME,
    CAPABILITY_VIDEO_GEN,
    VideoProvider,
    VideoRequest,
    VideoResponse,
    VideoResult,
)

__all__ = [
    "I2V_OPERATION",
    "I2V_UNIT",
    "I2VFlow",
    "I2VOutcome",
    "I2VResult",
    "StillToVideo",
    "StillToVideoFlow",
    "StillToVideoOutcome",
    "StillToVideoResult",
    "animate_still",
    "generate_and_animate",
    "i2v",
    "run_i2v_flow",
    "still_to_video",
]

I2V_OPERATION = VIDEO_OPERATION
"""The cost operation recorded for the animation call."""

I2V_UNIT = "clip"
"""The default cost unit for one animated still."""

_IMAGE_SUFFIXES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".avif": "image/avif",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}

_TIME_UNITS = {
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
}

_COST_KEYS = ("cost_usd", "amount_usd", "cost", "amount", "total_cost_usd")

VideoGenerator = Callable[[VideoRequest], Awaitable[VideoResponse]]
StillSelector = Callable[[], Any]
VideoCost = Callable[..., Any]


def _strip(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _mime_type(suffix: str) -> str | None:
    return _IMAGE_SUFFIXES.get(suffix.lower())


def _file_from_url(value: str) -> Path | None:
    parsed = urlsplit(value)
    if parsed.scheme.lower() != "file":
        return None
    return Path(unquote(parsed.path))


def _path_media_input(path: Path) -> MediaInput:
    if not path.is_file():
        raise ValidationError(
            "still file does not exist",
            details={"path": str(path)},
        )
    return MediaInput(
        data=path.read_bytes(),
        media_type=_mime_type(path.suffix),
        filename=path.name,
    )


def _asset_media_input(asset: Asset) -> MediaInput:
    path = _strip(asset.path)
    if not path:
        raise ValidationError(
            "still asset has no file or URL",
            details={"asset_id": asset.id},
        )
    if path.startswith(("http://", "https://")):
        return MediaInput(url=path)
    local = _file_from_url(path)
    return _path_media_input(local if local is not None else Path(path))


def _as_media_input(still: Any) -> MediaInput:
    """Return *still* as a provider frame input.

    Accepts the shapes a still can arrive in: an :class:`ImageRequest` result,
    an :class:`ImageOutcome`, a stored :class:`Asset`, a stock result, raw
    bytes, a local path or a remote URL.
    """
    if isinstance(still, MediaInput):
        return still
    if isinstance(still, StockAsset):
        return MediaInput(url=still.url)
    if isinstance(still, ImageResult):
        if still.data is not None:
            return MediaInput(
                data=still.data,
                media_type=f"image/{still.format.value}",
            )
        if still.url is not None:
            return MediaInput(url=still.url)
        raise ValidationError("still image result has neither data nor URL")
    if isinstance(still, ImageOutcome):
        if still.images:
            return _as_media_input(still.images[0])
        if still.asset is not None:
            return _as_media_input(still.asset)
        raise ValidationError("still outcome has no image")
    if isinstance(still, Asset):
        return _asset_media_input(still)
    if isinstance(still, (bytes, bytearray, memoryview)):
        return MediaInput(data=bytes(still), media_type="image/png")
    if isinstance(still, (str, Path)):
        text = str(still)
        if text.startswith(("http://", "https://")):
            return MediaInput(url=text)
        local = _file_from_url(text)
        return _path_media_input(local if local is not None else Path(text))

    data = getattr(still, "data", None)
    if isinstance(data, (bytes, bytearray, memoryview)):
        return MediaInput(data=bytes(data))
    url = getattr(still, "url", None)
    if isinstance(url, str) and url.strip():
        return MediaInput(url=url.strip())
    raise TypeError(
        "still must be an Asset, MediaInput, ImageOutcome, ImageResult, "
        "StockAsset, bytes, path or URL"
    )


def _still_asset(still: Any) -> Asset | None:
    if isinstance(still, Asset):
        return still
    if isinstance(still, ImageOutcome):
        return still.asset
    if isinstance(still, StockAsset):
        return Asset(
            kind=AssetKind.IMAGE,
            provider=still.provider,
            path=still.url,
            analysis=AssetAnalysis(width=still.width, height=still.height),
            license=AssetLicense(
                name=still.license or "",
                url=still.license_url or "",
                attribution=still.attribution or "",
                attribution_required=bool(still.attribution),
            ),
        )
    return None


def _still_event(still: Any) -> CostEvent | None:
    if isinstance(still, ImageOutcome):
        return still.event
    return None


def _event_link(event: CostEvent) -> AssetCostLink:
    return AssetCostLink(
        cost_event_id=event.id,
        amount_usd=event.amount_usd,
        provider=event.provider,
        model=event.model,
        operation=event.operation,
    )


def _is_image_request(value: Any) -> bool:
    return isinstance(value, ImageRequest)


async def _resolve(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _call_cost(
    cost: VideoCost,
    request: VideoRequest,
    response: VideoResponse,
    selection: RouteSelection | None,
) -> Any:
    """Call a cost builder with the arguments its signature accepts."""
    try:
        parameters = list(inspect.signature(cost).parameters.values())
    except (TypeError, ValueError):
        return cost(request, response, selection)
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters
    ):
        return cost(request, response, selection)
    if len(positional) >= 3:
        return cost(request, response, selection)
    if len(positional) == 2:
        return cost(request, response)
    if len(positional) == 1:
        return cost(request)
    return cost()


def _amount_from(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Mapping):
        for key in _COST_KEYS:
            candidate = value.get(key)
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                return float(candidate)
        return None
    for key in _COST_KEYS:
        candidate = getattr(value, key, None)
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            return float(candidate)
    return None


def _event_from_value(value: Any) -> CostEvent | None:
    if isinstance(value, CostEvent):
        return value
    if isinstance(value, Mapping):
        return CostEvent.model_validate(dict(value))
    event = getattr(value, "event", None)
    if isinstance(event, CostEvent):
        return event
    return None


def _units_for(
    request: VideoRequest,
    provider: Provider | None,
    operation: str,
) -> tuple[float, str]:
    price = provider.metadata.pricing_for(operation) if provider is not None else None
    unit = _strip(price.unit).lower() if price is not None else I2V_UNIT
    if unit in _TIME_UNITS:
        return (request.duration_seconds or 1.0) * request.count, unit
    return float(request.count), unit or I2V_UNIT


@dataclass(frozen=True, slots=True)
class I2VOutcome:
    """One still animated into one clip, with both stages' cost events.

    ``video`` is the stored (or cache hit) clip. ``still`` is the asset the
    frame came from, when it is known. ``still_event`` is the image service
    event for a generated still; ``video_event`` is the event for the
    animation, or the zero-amount hit record on a cache hit.
    """

    request: VideoRequest
    video: VideoCacheOutcome
    still: Asset | None = None
    still_event: CostEvent | None = None
    video_event: CostEvent | None = None
    still_outcome: ImageOutcome | None = None
    selection: RouteSelection | None = None

    @property
    def video_asset(self) -> Asset:
        """The stored video asset."""
        return self.video.asset

    @property
    def asset(self) -> Asset:
        """Alias for :attr:`video_asset`."""
        return self.video_asset

    @property
    def still_asset(self) -> Asset | None:
        """The still asset, when the frame came from one."""
        return self.still

    @property
    def video_assets(self) -> tuple[Asset, ...]:
        """Every stored asset, video first."""
        return self.video.assets

    @property
    def assets(self) -> tuple[Asset, ...]:
        """The still (when known) followed by the video."""
        if self.still is None:
            return self.video_assets
        return (self.still, *self.video_assets)

    @property
    def events(self) -> tuple[CostEvent, ...]:
        """Every event this flow recorded or reused, still first."""
        values = [
            event for event in (self.still_event, self.video_event) if event is not None
        ]
        return tuple(values)

    @property
    def event(self) -> CostEvent | None:
        """The animation event, or the still event when there is no video one."""
        return self.video_event or self.still_event

    @property
    def video_event_id(self) -> str | None:
        """The identifier of the animation event, when one was recorded."""
        return None if self.video_event is None else self.video_event.id

    @property
    def still_event_id(self) -> str | None:
        """The identifier of the still generation event, when one was recorded."""
        return None if self.still_event is None else self.still_event.id

    @property
    def amount_usd(self) -> float:
        """The sum of the still and animation events."""
        return sum(event.amount_usd for event in self.events)

    @property
    def total_cost_usd(self) -> float:
        """Alias for :attr:`amount_usd`."""
        return self.amount_usd

    @property
    def cache_hit(self) -> bool:
        """Whether the animation was answered from the video cache."""
        return self.video.cache_hit

    @property
    def cached(self) -> bool:
        """Alias for :attr:`cache_hit`."""
        return self.cache_hit

    @property
    def videos(self) -> list[VideoResult]:
        """The stored clips, in provider order."""
        return self.video.videos

    @property
    def video_result(self) -> VideoResult:
        """The first stored clip."""
        return self.video.result

    @property
    def first(self) -> VideoResult:
        """Alias for :attr:`video_result`."""
        return self.video_result

    @property
    def path(self) -> str:
        """The stored video path or URL."""
        return self.video.path

    @property
    def video_path(self) -> str:
        """Alias for :attr:`path`."""
        return self.path

    @property
    def still_path(self) -> str | None:
        """The still path or URL, when known."""
        return None if self.still is None else self.still.path

    @property
    def provider(self) -> str:
        """The provider that served the animation."""
        return self.video.asset.provider

    @property
    def model(self) -> str:
        """The model that served the animation."""
        return self.video.asset.model

    @property
    def format(self) -> Any:
        """The validated video format."""
        return self.video.format

    @property
    def width(self) -> int | None:
        """The validated video width."""
        return self.video.width

    @property
    def height(self) -> int | None:
        """The validated video height."""
        return self.video.height

    @property
    def duration_seconds(self) -> float | None:
        """The validated video duration."""
        return self.video.duration_seconds

    @property
    def fps(self) -> int | None:
        """The requested frame rate."""
        return self.video.fps

    @property
    def has_audio(self) -> bool:
        """Whether the clip carries an audio stream."""
        return self.video.has_audio

    @property
    def data(self) -> bytes | None:
        """The validated video bytes."""
        return self.video.data

    def __getattr__(self, name: str) -> Any:
        """Delegate remaining asset fields such as ``id`` and ``analysis``."""
        return getattr(self.video.asset, name)


StillToVideoOutcome = I2VOutcome
I2VResult = I2VOutcome
StillToVideoResult = I2VOutcome


class I2VFlow:
    """Generate or select a still, animate it and keep linked cost events.

    ``videos`` is the :class:`VideoCache` that validates, stores and caches the
    clip. ``images`` is optional and only needed when the still is generated
    from an :class:`ImageRequest`; a supplied or selected still needs no image
    service. The animation runs through ``router`` when given, otherwise
    through ``generate_video``. ``tracker`` persists the animation cost event;
    without one the event is still returned but cannot be linked to a stored
    asset because it has no durable identifier.
    """

    def __init__(
        self,
        *,
        videos: VideoCache | None = None,
        video_cache: VideoCache | None = None,
        cache: VideoCache | None = None,
        images: ImageService | None = None,
        image_service: ImageService | None = None,
        router: Router | None = None,
        generate_video: VideoGenerator | None = None,
        video_generator: VideoGenerator | None = None,
        generator: VideoGenerator | None = None,
        tracker: CostTracker | None = None,
        cost_tracker: CostTracker | None = None,
        pricing: VideoPricing | VideoPricingEstimator | None = None,
        pricing_estimator: VideoPricingEstimator | None = None,
        pricing_overrides: Mapping[str, float] | None = None,
        pricing_version: str = "",
        operation: str = I2V_OPERATION,
        image_operation: str = IMAGE_OPERATION,
        record_cost: bool = True,
    ) -> None:
        supplied_videos = [
            value for value in (videos, video_cache, cache) if value is not None
        ]
        if len(supplied_videos) > 1:
            raise TypeError("pass only one of videos, video_cache or cache")
        active_videos = supplied_videos[0] if supplied_videos else None
        if active_videos is None:
            raise TypeError("I2VFlow needs a VideoCache")
        if not isinstance(active_videos, VideoCache):
            raise TypeError("videos must be a VideoCache")
        if images is not None and image_service is not None:
            raise TypeError("pass either images or image_service, not both")
        active_images = images if images is not None else image_service
        if active_images is not None and not isinstance(active_images, ImageService):
            raise TypeError("images must be an ImageService")
        if tracker is not None and cost_tracker is not None:
            raise TypeError("pass either tracker or cost_tracker, not both")
        if router is not None and not isinstance(router, Router):
            raise TypeError("router must be a Router")
        supplied_generators = [
            value
            for value in (generate_video, video_generator, generator)
            if value is not None
        ]
        if len(supplied_generators) > 1:
            raise TypeError(
                "pass only one of generate_video, video_generator or generator"
            )
        active_generator = supplied_generators[0] if supplied_generators else None
        if pricing is not None and pricing_estimator is not None:
            raise TypeError("pass either pricing or pricing_estimator, not both")
        active_pricing = pricing if pricing is not None else pricing_estimator
        estimator: VideoPricingEstimator | None
        if isinstance(active_pricing, VideoPricingEstimator):
            estimator = active_pricing
        elif isinstance(active_pricing, VideoPricing):
            estimator = VideoPricingEstimator(active_pricing)
        elif active_pricing is None:
            estimator = None
        else:
            raise TypeError("pricing must be a VideoPricing or estimator")

        operation_name = _strip(operation)
        if not operation_name:
            raise ValueError("operation must be a non-empty string")
        image_operation_name = _strip(image_operation)
        if not image_operation_name:
            raise ValueError("image_operation must be a non-empty string")

        self._videos = active_videos
        self._images = active_images
        self._router = router
        self._generator = active_generator
        self._tracker = tracker if tracker is not None else cost_tracker
        self._estimator = estimator
        self._pricing = PricingResolver(
            pricing_overrides,
            pricing_version=pricing_version,
        )
        self._operation = operation_name
        self._image_operation = image_operation_name
        self._record_cost = bool(record_cost)

    @property
    def videos(self) -> VideoCache:
        """The video cache used by this flow."""
        return self._videos

    @property
    def images(self) -> ImageService | None:
        """The image service used to generate a still, when configured."""
        return self._images

    @property
    def tracker(self) -> CostTracker | None:
        """The tracker that persists the animation event."""
        return self._tracker

    async def animate(
        self,
        still: Any = None,
        *,
        image_request: ImageRequest | None = None,
        still_request: ImageRequest | None = None,
        selector: StillSelector | None = None,
        select_still: StillSelector | None = None,
        video_request: VideoRequest | None = None,
        prompt: str | None = None,
        run_id: str = "",
        token: Any = None,
        screening_token: Any = None,
        cost_usd: float | None = None,
        cost_event: CostEvent | Mapping[str, Any] | None = None,
        video_cost: VideoCost | None = None,
        tracker: CostTracker | None = None,
        remaining_budget: float | None = None,
        storage: Any = None,
        pinned: bool = True,
        estimated: bool = False,
        record_cost: bool | None = None,
        **video_options: Any,
    ) -> I2VOutcome:
        """Obtain a still, animate it and return both assets and cost events.

        Supply exactly one still source: ``still`` itself, ``selector`` (a
        callable returning any accepted still shape) or ``image_request`` /
        ``still_request`` for a generated still. The animation prompt comes
        from ``video_request`` or ``prompt`` plus any extra video options.
        """
        active_selector = selector if selector is not None else select_still
        still_outcome: ImageOutcome | None = None
        selected = still

        if selected is None and active_selector is not None:
            selected = await _resolve(active_selector())
        if selected is None:
            still_input = image_request if image_request is not None else still_request
            if still_input is not None:
                if not _is_image_request(still_input):
                    raise TypeError("image_request must be an ImageRequest")
                images = self._images
                if images is None:
                    raise ValueError("an ImageService is required to generate a still")
                active_token = token if token is not None else screening_token
                still_outcome = await images.generate(
                    still_input,
                    token=active_token,
                    run_id=run_id,
                    operation=self._image_operation,
                )
                selected = still_outcome

        if selected is None:
            raise ValueError("a still, selector or image request is required")
        if _is_image_request(selected):
            images = self._images
            if images is None:
                raise ValueError("an ImageService is required to generate a still")
            active_token = token if token is not None else screening_token
            still_outcome = await images.generate(
                cast("ImageRequest", selected),
                token=active_token,
                run_id=run_id,
                operation=self._image_operation,
            )
            selected = still_outcome

        if isinstance(selected, ImageOutcome):
            still_outcome = selected
        still_asset = _still_asset(selected)
        still_event = _still_event(selected) or (
            still_outcome.event if still_outcome is not None else None
        )
        frame = _as_media_input(selected)

        request = self._video_request(
            video_request,
            prompt=prompt,
            frame=frame,
            options=video_options,
        )
        if request.count != 1:
            raise ValueError(
                "the still to video flow animates one still into one clip; "
                "count must be 1"
            )

        cached = await self._videos.lookup(
            request,
            run_id=run_id,
            operation=self._operation,
            storage=storage,
        )
        if cached is not None:
            return I2VOutcome(
                request=request,
                video=cached,
                still=still_asset,
                still_event=still_event,
                video_event=cached.event,
                still_outcome=still_outcome,
            )

        response, selection = await self._generate(
            request,
            remaining_budget=remaining_budget,
        )
        if not response.videos:
            raise ValueError("video provider returned no clips")
        result = response.videos[0]
        provider_name = response.provider or (
            selection.name if selection is not None else ""
        )
        model_name = response.model or request.model or ""
        event = self._build_event(
            request,
            response,
            selection=selection,
            provider_name=provider_name,
            model_name=model_name,
            run_id=run_id,
            cost_usd=cost_usd,
            cost_event=cost_event,
            video_cost=video_cost,
            still_asset=still_asset,
            still_event=still_event,
            estimated=estimated,
        )

        active_tracker = tracker if tracker is not None else self._tracker
        should_record = self._record_cost if record_cost is None else bool(record_cost)
        recorded: CostEvent | None = None
        if event is not None and should_record and active_tracker is not None:
            recorded = (
                active_tracker.record_estimate(event)
                if estimated
                else active_tracker.record_actual(event)
            )
            event = recorded

        links = [_event_link(recorded)] if recorded is not None else []
        stored = await self._videos.store(
            request,
            result,
            provider=provider_name,
            model=model_name,
            index=0,
            run_id=run_id,
            seed=result.seed if result.seed is not None else request.seed,
            pinned=pinned,
            storage=storage,
            cost_links=links,
        )
        if (
            recorded is not None
            and stored.asset.cost_links != links
            and self._videos.cache is not None
        ):
            updated = self._videos.cache.assets.update(
                stored.asset.id,
                cost_links=links,
            )
            if updated is not None:
                stored = replace(stored, asset=updated)

        return I2VOutcome(
            request=request,
            video=stored,
            still=still_asset,
            still_event=still_event,
            video_event=event,
            still_outcome=still_outcome,
            selection=selection,
        )

    async def run(self, still: Any = None, **values: Any) -> I2VOutcome:
        """Alias for :meth:`animate`."""
        return await self.animate(still, **values)

    async def generate_and_animate(
        self,
        still: Any = None,
        **values: Any,
    ) -> I2VOutcome:
        """Alias for :meth:`animate`."""
        return await self.animate(still, **values)

    async def still_to_video(
        self,
        still: Any = None,
        **values: Any,
    ) -> I2VOutcome:
        """Alias for :meth:`animate`."""
        return await self.animate(still, **values)

    def _video_request(
        self,
        request: VideoRequest | None,
        *,
        prompt: str | None,
        frame: MediaInput,
        options: Mapping[str, Any],
    ) -> VideoRequest:
        if request is not None:
            if not isinstance(request, VideoRequest):
                raise TypeError("video_request must be a VideoRequest")
            return request.model_copy(update={"first_frame": frame})
        values = dict(options)
        values.pop("first_frame", None)
        prompt_value = prompt if prompt is not None else values.pop("prompt", None)
        text = _strip(prompt_value)
        if not text:
            raise ValueError("an animation prompt is required")
        return VideoRequest(prompt=text, first_frame=frame, **values)

    async def _generate(
        self,
        request: VideoRequest,
        *,
        remaining_budget: float | None,
    ) -> tuple[VideoResponse, RouteSelection | None]:
        if self._router is not None:

            async def run(provider: Provider) -> VideoResponse:
                return await cast("VideoProvider", provider).generate(request)

            response = await self._router.execute(
                CAPABILITY_VIDEO_GEN,
                run,
                required=(CAPABILITY_VIDEO_FIRST_FRAME,),
                remaining_budget=remaining_budget,
                units=float(request.count),
                unit_kind=I2V_UNIT,
            )
            if not isinstance(response, VideoResponse):
                raise TypeError("video provider must return a VideoResponse")
            return response, self._router.last_selection

        generator = self._generator
        if generator is None:
            raise ValueError(
                "I2VFlow needs a router or a video generator to animate a still"
            )
        generated = await generator(request)
        if isinstance(generated, VideoCacheOutcome):
            raise TypeError(
                "video generator must return a VideoResponse, not a cached outcome"
            )
        if not isinstance(generated, VideoResponse):
            raise TypeError("video generator must return a VideoResponse")
        return generated, None

    def _build_event(
        self,
        request: VideoRequest,
        response: VideoResponse,
        *,
        selection: RouteSelection | None,
        provider_name: str,
        model_name: str,
        run_id: str,
        cost_usd: float | None,
        cost_event: CostEvent | Mapping[str, Any] | None,
        video_cost: VideoCost | None,
        still_asset: Asset | None,
        still_event: CostEvent | None,
        estimated: bool,
    ) -> CostEvent | None:
        provider = selection.provider if selection is not None else None
        units, unit_kind = _units_for(request, provider, self._operation)
        metadata: dict[str, Any] = {
            "video_count": request.count,
            "first_frame": "still",
        }
        if still_asset is not None:
            metadata["still_asset_id"] = still_asset.id
        if still_event is not None:
            metadata["still_cost_event_id"] = still_event.id
            metadata["still_operation"] = still_event.operation

        if cost_event is not None:
            if isinstance(cost_event, CostEvent):
                return cost_event
            data = dict(cost_event)
            data.setdefault("provider", provider_name)
            data.setdefault("model", model_name)
            data.setdefault("operation", self._operation)
            data.setdefault("units", units)
            data.setdefault("unit_kind", unit_kind)
            data.setdefault("run_id", run_id)
            data.setdefault("estimated", estimated)
            details = dict(data.get("metadata") or {})
            details.update(metadata)
            data["metadata"] = details
            return CostEvent.model_validate(data)

        amount: float | None = None
        if video_cost is not None:
            raw = _call_cost(video_cost, request, response, selection)
            event = _event_from_value(raw)
            if event is not None:
                return event
            amount = _amount_from(raw)
        if amount is None:
            amount = cost_usd
        if amount is None:
            amount = _amount_from(response.metadata)
        if amount is None and self._estimator is not None:
            amount = self._estimator.estimate(request).amount_usd
        if amount is None and provider is not None:
            resolved = self._pricing.resolve(
                provider,
                self._operation,
                units,
                unit_kind=unit_kind,
                provider_name=provider_name or None,
            )
            amount = resolved.amount_usd
            units = resolved.units
            unit_kind = resolved.unit_kind
            metadata["price_source"] = resolved.source.value
            if resolved.missing:
                metadata["missing_price"] = True
        if amount is None:
            amount = 0.0
            metadata["missing_price"] = True
        metadata["billed_units"] = units
        metadata["price_unit"] = unit_kind
        if response.request_id:
            metadata["request_id"] = response.request_id
        return CostEvent(
            provider=provider_name,
            model=model_name,
            operation=self._operation,
            units=units,
            unit_kind=unit_kind,
            amount_usd=float(amount),
            estimated=estimated,
            run_id=run_id,
            metadata=metadata,
        )


StillToVideoFlow = I2VFlow
StillToVideo = I2VFlow


async def animate_still(
    still: Any = None,
    *,
    videos: VideoCache | None = None,
    video_cache: VideoCache | None = None,
    **values: Any,
) -> I2VOutcome:
    """Run one still to video flow without constructing a long-lived object."""
    flow_values = dict(values)
    flow_keys = {
        "videos",
        "video_cache",
        "cache",
        "images",
        "image_service",
        "router",
        "generate_video",
        "video_generator",
        "generator",
        "tracker",
        "cost_tracker",
        "pricing",
        "pricing_estimator",
        "pricing_overrides",
        "pricing_version",
        "operation",
        "image_operation",
        "record_cost",
    }
    flow_options = {
        key: flow_values.pop(key) for key in list(flow_values) if key in flow_keys
    }
    flow = I2VFlow(videos=videos, video_cache=video_cache, **flow_options)
    return await flow.animate(still, **flow_values)


still_to_video = animate_still
run_i2v_flow = animate_still
generate_and_animate = animate_still
i2v = animate_still

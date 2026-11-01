"""Run one image generation request through the single paid path.

The service is the only place an image call is executed. It requires a
screening token for the prompt, answers an identical request from the asset
cache when possible, applies the remaining budget, asks the router for a
provider, validates and stores every returned image, records one actual cost
event and writes a sidecar next to each output. Callers never choose a vendor,
handle a vendor payload or write a cost record themselves.

The order is deliberate and matches the project invariant: screen -> cache ->
budget -> router -> provider -> validate -> store -> cost -> sidecar. A cache hit
short-circuits the paid path but still records a zero-amount cost event, so a
reused image is visible in the ledger without counting as spend.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlsplit

import httpx
from sqlalchemy import Engine

from aivia.core.asset_cache import AssetCache, AssetCacheKey
from aivia.core.budget import BudgetGuard
from aivia.core.config.budget import BudgetConfig
from aivia.core.cost import CostTracker
from aivia.core.media_probe import (
    MediaDownloader,
    MediaLimits,
    MediaProbe,
    validate_media,
)
from aivia.core.models.asset import (
    Asset,
    AssetAnalysis,
    AssetCostLink,
    AssetKind,
)
from aivia.core.models.base import canonical_json
from aivia.core.models.cost import CostEvent
from aivia.core.models.run import ToolchainFingerprint
from aivia.core.pricing import ResolvedPrice
from aivia.core.router import Router, RouteSelection
from aivia.core.sidecar import SIDECAR_SUFFIX, build_cost_sidecar, write_cost_sidecar
from aivia.media.adaptation import (
    adapt_for_cache,
    adapt_for_provider,
    cache_request_variants,
)
from aivia.media.screening import (
    DEFAULT_PROMPT_KIND,
    ScreeningToken,
    require_screened,
    screen_prompt,
)
from aivia.providers.base import Provider
from aivia.providers.image.cost import (
    IMAGE_OPERATION,
    IMAGE_UNIT,
    ImageCostCalculator,
)
from aivia.providers.media.adapt import (
    AdaptationError,
    AdaptationResult,
    AdaptationWarning,
)
from aivia.providers.media.dto import MediaKind
from aivia.providers.protocols.image import (
    CAPABILITY_IMAGE_GEN,
    CAPABILITY_IMAGE_INPUT,
    ImageProvider,
    ImageRequest,
    ImageResponse,
    ImageResult,
)
from aivia.providers.protocols.media import ImageFormat
from aivia.providers.protocols.storage import (
    StorageObject,
    StorageProvider,
    StoragePutRequest,
)

__all__ = [
    "IMAGE_OPERATION",
    "IMAGE_UNIT",
    "ImageGeneration",
    "ImageOutcome",
    "ImageService",
    "ImageServiceResult",
    "generate_image",
    "image_service_from_config",
    "service_from_config",
    "variation_seeds",
]

ToolchainLike = ToolchainFingerprint | Mapping[str, Any] | str | None
BudgetLike = BudgetGuard | BudgetConfig | Mapping[str, Any] | None


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_path(url: str) -> Path | None:
    parsed = urlsplit(url)
    if parsed.scheme.lower() != "file":
        return None
    return Path(unquote(parsed.path))


def _image_format(value: str) -> ImageFormat:
    try:
        return ImageFormat(value)
    except ValueError:
        return ImageFormat.PNG


def variation_seeds(
    seed: int | None,
    count: int,
    *,
    fallback: int | None = None,
) -> tuple[int, ...]:
    """Return one distinct seed per requested variation.

    An explicit *seed* is expanded to ``seed + index`` so the same request
    always yields the same values. Without one, a fresh 32-bit base is drawn
    unless *fallback* supplies a base; every output still receives its own
    seed so the variation is reproducible from the recorded value.
    """
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer")
    if count < 1:
        raise ValueError("count must be positive")
    base = fallback if seed is None and fallback is not None else seed
    if base is None:
        base = secrets.randbits(32)
    return tuple(base + index for index in range(count))


@dataclass(frozen=True, slots=True)
class ImageOutcome:
    """The response together with its stored assets, price, events and sidecars.

    ``events`` holds one zero-amount event per cached asset on a hit, or the one
    actual event for the provider call on a miss. ``event`` is the first event
    for callers that only need a single record.
    """

    response: ImageResponse
    assets: tuple[Asset, ...]
    probes: tuple[MediaProbe, ...] = ()
    storage_objects: tuple[StorageObject, ...] = ()
    events: tuple[CostEvent, ...] = ()
    price: ResolvedPrice | None = None
    selection: RouteSelection | None = None
    sidecars: tuple[Path, ...] = ()
    cache_hit: bool = False
    adaptation_warnings: tuple[AdaptationWarning, ...] = ()

    @property
    def event(self) -> CostEvent | None:
        """The first recorded cost event, or ``None``."""
        return self.events[0] if self.events else None

    @property
    def warnings(self) -> tuple[AdaptationWarning, ...]:
        """Every change the adaptation layer made before submission."""
        return self.adaptation_warnings

    @property
    def adaptation_messages(self) -> tuple[str, ...]:
        """The human-readable adaptation warnings in order."""
        return tuple(warning.message for warning in self.adaptation_warnings)

    @property
    def images(self) -> list[ImageResult]:
        """Every image in the provider response."""
        return list(self.response.images)

    @property
    def seeds(self) -> tuple[int | None, ...]:
        """The recorded seed for every output, in response order."""
        return tuple(image.seed for image in self.response.images)

    @property
    def seed(self) -> int | None:
        """The first recorded seed, or ``None``."""
        return self.seeds[0] if self.seeds else None

    @property
    def variation_seeds(self) -> tuple[int | None, ...]:
        """Alias for :attr:`seeds`."""
        return self.seeds

    @property
    def outputs(self) -> tuple[Asset, ...]:
        """Alias for :attr:`assets`."""
        return self.assets

    @property
    def asset(self) -> Asset | None:
        """The first stored asset, or ``None``."""
        return self.assets[0] if self.assets else None

    @property
    def first(self) -> Asset | None:
        """Alias for :attr:`asset`."""
        return self.asset

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
        """The first sidecar written, or ``None``."""
        return self.sidecars[0] if self.sidecars else None

    @property
    def sidecar_paths(self) -> tuple[Path, ...]:
        """Alias for :attr:`sidecars`."""
        return self.sidecars

    @property
    def sidecar_path(self) -> Path | None:
        """Alias for :attr:`sidecar`."""
        return self.sidecar

    @property
    def provider(self) -> str:
        """The provider that served the call."""
        return self.response.provider

    @property
    def model(self) -> str:
        """The model that served the call."""
        return self.response.model

    @property
    def cached(self) -> bool:
        """Whether this outcome came from the asset cache."""
        return self.cache_hit

    @property
    def amount_usd(self) -> float:
        """The recorded cost, or zero when no cost was recorded."""
        if self.events:
            return sum(event.amount_usd for event in self.events)
        if self.price is not None:
            return self.price.amount_usd
        return 0.0


ImageGeneration = ImageOutcome
"""Alias for :class:`ImageOutcome`."""

ImageServiceResult = ImageOutcome
"""Alias for :class:`ImageOutcome`."""


class ImageService:
    """Execute image generation through a :class:`~aivia.core.router.Router`.

    ``cache`` answers an identical request without a provider call and persists
    the assets of a miss. When it is omitted but a tracker or engine is given,
    the service builds an :class:`~aivia.core.asset_cache.AssetCache` on that
    ledger so the cache is never silently disabled. ``storage`` stores validated
    image bytes; without one the asset record is still built but no file is
    written. ``budget`` may be a :class:`~aivia.core.budget.BudgetGuard` or a
    budget section, and its remaining amount is passed to the router.
    """

    def __init__(
        self,
        router: Router,
        *,
        tracker: CostTracker | None = None,
        cost_tracker: CostTracker | None = None,
        cache: AssetCache | None = None,
        engine: Engine | None = None,
        storage: StorageProvider | None = None,
        budget: BudgetLike = None,
        budget_guard: BudgetGuard | None = None,
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
        pricing_overrides: Mapping[str, float] | None = None,
        pricing_version: str = "",
        operation: str = IMAGE_OPERATION,
        toolchain: ToolchainLike = None,
        toolchain_fingerprint: ToolchainLike = None,
        key_prefix: str | None = None,
        record_cost: bool = True,
        record_sidecar: bool = True,
        no_cache: bool = False,
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
        active_tracker = tracker if tracker is not None else cost_tracker
        if active_tracker is not None and not isinstance(active_tracker, CostTracker):
            raise TypeError("tracker must be a CostTracker")
        if cache is not None and not isinstance(cache, AssetCache):
            raise TypeError("cache must be an AssetCache")
        if engine is not None and not isinstance(engine, Engine):
            raise TypeError("engine must be a SQLAlchemy Engine")
        if downloader is not None and not isinstance(downloader, MediaDownloader):
            raise TypeError("downloader must be a MediaDownloader")
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

        selected_cache = cache
        if selected_cache is None and engine is not None:
            selected_cache = AssetCache(engine, no_cache=no_cache)
        elif (
            selected_cache is None
            and active_tracker is not None
            and active_tracker.repository.engine is not None
        ):
            selected_cache = AssetCache(
                active_tracker.repository.engine,
                cost=active_tracker,
                no_cache=no_cache,
            )
        if selected_cache is not None and no_cache:
            selected_cache.no_cache = True
        if active_tracker is None and selected_cache is not None:
            active_tracker = selected_cache.cost

        selected_limits = limits if limits is not None else media_limits
        if selected_limits is not None and not isinstance(selected_limits, MediaLimits):
            raise TypeError("limits must be a MediaLimits")

        active_budget = budget_guard if budget_guard is not None else budget
        if isinstance(active_budget, (BudgetConfig, Mapping)):
            active_budget = BudgetGuard(active_budget, active_tracker)
        if active_budget is not None and not isinstance(active_budget, BudgetGuard):
            raise TypeError("budget must be a BudgetGuard, BudgetConfig or mapping")

        selected_toolchain = (
            toolchain if toolchain is not None else toolchain_fingerprint
        )
        if pricing_overrides is None:
            overrides: Mapping[str, float] = router.config_for(
                CAPABILITY_IMAGE_GEN
            ).pricing_overrides
        else:
            if not isinstance(pricing_overrides, Mapping):
                raise TypeError("pricing_overrides must be a mapping")
            overrides = pricing_overrides

        self._router = router
        self._tracker = active_tracker
        self._cache = selected_cache
        self._storage = storage
        self._budget = active_budget
        self._safety = selected_safety
        self._screener = screener
        self._limits = selected_limits or MediaLimits()
        self._downloader = downloader
        self._http = http
        self._transport = transport
        self._operation = operation.strip()
        self._toolchain = selected_toolchain
        self._key_prefix = key_prefix
        self._record_cost = bool(record_cost)
        self._record_sidecar = bool(record_sidecar)
        self._no_cache = bool(no_cache)
        self._calculator = ImageCostCalculator(
            overrides,
            pricing_version=pricing_version,
        )
        self._last_outcome: ImageOutcome | None = None
        self._last_selection: RouteSelection | None = None
        self._last_event: CostEvent | None = None
        self._last_price: ResolvedPrice | None = None

    @property
    def router(self) -> Router:
        """The router every call goes through."""
        return self._router

    @property
    def tracker(self) -> CostTracker | None:
        """The default cost tracker, when one was configured."""
        return self._tracker

    @property
    def cache(self) -> AssetCache | None:
        """The asset cache, when one was configured."""
        return self._cache

    @property
    def storage(self) -> StorageProvider | None:
        """The storage provider, when one was configured."""
        return self._storage

    @property
    def budget(self) -> BudgetGuard | None:
        """The budget guard, when one was configured."""
        return self._budget

    @property
    def limits(self) -> MediaLimits:
        """The caps applied to every downloaded or returned image."""
        return self._limits

    @property
    def calculator(self) -> ImageCostCalculator:
        """The calculator that prices and records image calls."""
        return self._calculator

    @property
    def last_outcome(self) -> ImageOutcome | None:
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
    def last_price(self) -> ResolvedPrice | None:
        """The most recent resolved price."""
        return self._last_price

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

    @staticmethod
    def _seed_for(request: ImageRequest, index: int) -> int | None:
        """Return the seed one output of *request* will be recorded with."""
        if request.seed is None:
            return None
        return request.seed + index

    @staticmethod
    def _response_seeds(
        request: ImageRequest,
        response: ImageResponse,
        base_seed: int,
    ) -> tuple[int, ...]:
        """Choose the seed recorded for each image in *response*.

        An explicit request seed always expands to ``seed + index``. Without
        one, distinct seeds reported by the provider are kept; missing or
        duplicate values fall back to a fresh base so every output differs.
        """
        if request.seed is None:
            reported = [image.seed for image in response.images]
            if all(seed is not None for seed in reported) and len(set(reported)) == len(
                reported
            ):
                return tuple(cast(int, seed) for seed in reported)
        return variation_seeds(base_seed, response.count)

    def _parameters(self, request: ImageRequest, index: int = 0) -> dict[str, Any]:
        return {
            "negative_prompt": request.negative_prompt,
            "model": request.model,
            "width": request.width,
            "height": request.height,
            "aspect_ratio": request.aspect_ratio,
            "format": request.format.value,
            "quality": request.quality,
            "style": request.style,
            "seed": self._seed_for(request, index),
            "count": request.count,
            "index": index,
        }

    def _reference_hashes(self, request: ImageRequest) -> tuple[str, ...]:
        hashes: list[str] = []
        for reference in request.reference_images:
            if reference.data is not None:
                hashes.append(hashlib.sha256(reference.data).hexdigest())
            elif reference.url is not None:
                hashes.append(_sha256_text(reference.url))
        return tuple(hashes)

    def _cache_key_for(
        self,
        request: ImageRequest,
        *,
        index: int = 0,
    ) -> AssetCacheKey:
        if self._cache is None:
            return AssetCacheKey(
                provider="",
                model=request.model or "",
                prompt=request.prompt,
                parameters=self._parameters(request, index),
                reference_hashes=self._reference_hashes(request),
            )
        return self._cache.key(
            "",
            request.model or "",
            request.prompt,
            parameters=self._parameters(request, index),
            reference_hashes=self._reference_hashes(request),
        )

    def _adapted_cache_request(
        self,
        request: ImageRequest,
        provider: Provider | None,
    ) -> ImageRequest:
        if provider is None:
            return request
        try:
            return cast(
                ImageRequest,
                adapt_for_cache(request, provider, MediaKind.IMAGE),
            )
        except AdaptationError:
            return request

    def cache_key(
        self,
        request: ImageRequest,
        *,
        index: int = 0,
        provider: Provider | None = None,
    ) -> AssetCacheKey:
        """Return the cache key for one output of *request*.

        The key is built from the request the provider would actually submit,
        so a snapped size or clamped count is reused on a repeat. The provider
        is left empty in the key so a repeated request can be answered before
        the router has selected a provider. The stored asset still records the
        provider and model that actually served the call.
        """
        if not isinstance(request, ImageRequest):
            raise TypeError("request must be an ImageRequest")
        if provider is not None:
            request = self._adapted_cache_request(request, provider)
        else:
            variants = cache_request_variants(
                request,
                self._router,
                MediaKind.IMAGE,
                required=(
                    (CAPABILITY_IMAGE_INPUT,) if request.reference_images else ()
                ),
            )
            if variants:
                request = cast(ImageRequest, variants[0])
        return self._cache_key_for(request, index=index)

    async def generate(
        self,
        request: ImageRequest,
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
    ) -> ImageOutcome:
        """Generate *request* through the screened paid path.

        ``token`` (or its alias ``screening_token``) must be the token returned
        by :func:`aivia.media.screening.screen_prompt` for this prompt. A
        missing, malformed or mismatched token raises
        :class:`~aivia.core.errors.ContentPolicyError` before the cache, the
        budget, the router or any provider is touched.
        """
        if not isinstance(request, ImageRequest):
            raise TypeError("request must be an ImageRequest")
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
            self._cache is not None and not self._no_cache and not bool(no_cache)
        )
        write_sidecar = (
            self._record_sidecar if record_sidecar is None else bool(record_sidecar)
        )
        write_cost = self._record_cost if record_cost is None else bool(record_cost)

        required = (CAPABILITY_IMAGE_INPUT,) if request.reference_images else None
        if use_cache:
            cached = await self._cached_outcome(
                cache_request_variants(
                    request,
                    self._router,
                    MediaKind.IMAGE,
                    required=required or (),
                )
                or (request,),
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

        base_seed = request.seed if request.seed is not None else secrets.randbits(32)
        adaptation: AdaptationResult[Any] | None = None

        async def run(provider: Provider) -> ImageResponse:
            nonlocal adaptation
            adaptation = adapt_for_provider(
                request,
                provider,
                MediaKind.IMAGE,
                seed=base_seed,
            )
            adapted = cast(ImageRequest, adaptation.request)
            return await cast(ImageProvider, provider).generate(adapted)

        response = await self._router.execute(
            CAPABILITY_IMAGE_GEN,
            run,
            required=required,
            remaining_budget=budget,
            units=float(request.count),
            unit_kind=IMAGE_UNIT,
            exclude=exclude,
        )
        if not isinstance(response, ImageResponse):
            raise TypeError("image provider must return an ImageResponse")
        selection = self._router.last_selection
        if selection is None:
            raise RuntimeError("router completed an image call without a selection")
        provider_name = response.provider or selection.name

        seeds = self._response_seeds(request, response, base_seed)
        response = response.model_copy(
            update={
                "images": [
                    image.model_copy(update={"seed": seed})
                    for image, seed in zip(response.images, seeds, strict=True)
                ]
            }
        )

        materialised: list[tuple[MediaProbe, StorageObject | None]] = []
        for index, result in enumerate(response.images):
            probe, stored = await self._materialise(
                result,
                request,
                index=index,
                storage=active_storage,
                key=key,
                key_prefix=active_prefix,
                run_id=run_id,
            )
            materialised.append((probe, stored))
        if not materialised:
            raise ValueError("image provider returned no images")

        price = self._calculator.resolve(
            selection.provider,
            provider_name=selection.name,
            units=float(response.count),
            unit_kind=IMAGE_UNIT,
            operation=operation_name,
        )
        event = self._calculator.cost_event(
            selection.provider,
            provider_name=selection.name,
            response=response,
            run_id=run_id,
            operation=operation_name,
            estimated=estimated,
            units=float(response.count),
            unit_kind=IMAGE_UNIT,
            metadata=metadata,
            price=price,
        )
        recorded: CostEvent | None = None
        if write_cost and active_tracker is not None:
            recorded = (
                active_tracker.record_estimate(event)
                if estimated
                else active_tracker.record_actual(event)
            )
            event = recorded

        key_request = self._adapted_cache_request(request, selection.provider)
        assets: list[Asset] = []
        for index, (probe, stored) in enumerate(materialised):
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
            asset = Asset(
                kind=AssetKind.IMAGE,
                provider=provider_name,
                model=response.model or request.model or "",
                prompt=request.prompt,
                path=(stored.url or stored.key) if stored is not None else "",
                content_hash=probe.sha256,
                run_id=run_id,
                pinned=pinned,
                seed=seeds[index],
                analysis=AssetAnalysis(width=probe.width, height=probe.height),
                cost_links=cost_links,
            )
            if use_cache and self._cache is not None:
                key_for_asset = self._cache_key_for(key_request, index=index)
                asset = self._cache.store(asset, key=key_for_asset)
            assets.append(asset)

        sidecars: tuple[Path, ...] = ()
        if write_sidecar:
            sidecars = await self._write_sidecars(
                assets,
                [event] * len(assets),
                run_id=run_id,
                storage=active_storage,
                stored_objects=[stored for _, stored in materialised],
            )

        outcome = ImageOutcome(
            response=response,
            assets=tuple(assets),
            probes=tuple(probe for probe, _ in materialised),
            storage_objects=tuple(
                stored for _, stored in materialised if stored is not None
            ),
            events=(event,),
            price=price,
            selection=selection,
            sidecars=sidecars,
            cache_hit=False,
            adaptation_warnings=(adaptation.warnings if adaptation is not None else ()),
        )
        self._last_selection = selection
        self._last_event = event
        self._last_price = price
        self._remember(outcome)
        return outcome

    async def generate_with_cost(
        self,
        request: ImageRequest,
        **values: Any,
    ) -> ImageOutcome:
        """Alias for :meth:`generate` returning the outcome with cost data."""
        return await self.generate(request, **values)

    async def generate_images(
        self,
        request: ImageRequest,
        **values: Any,
    ) -> ImageOutcome:
        """Alias for :meth:`generate`."""
        return await self.generate(request, **values)

    def _remember(self, outcome: ImageOutcome) -> None:
        self._last_outcome = outcome
        self._last_selection = outcome.selection
        self._last_event = outcome.event
        self._last_price = outcome.price

    async def _cached_outcome(
        self,
        requests: Iterable[ImageRequest],
        *,
        run_id: str,
        operation: str,
        storage: StorageProvider | None,
        write_sidecar: bool,
    ) -> ImageOutcome | None:
        cache = self._cache
        if cache is None:
            return None
        keys: list[AssetCacheKey] = []
        assets: list[Asset] = []
        payloads: list[bytes] = []
        for request in requests:
            candidate_keys = [
                self._cache_key_for(request, index=index)
                for index in range(request.count)
            ]
            candidate_assets: list[Asset] = []
            candidate_payloads: list[bytes] = []
            for key in candidate_keys:
                entry = cache.entry(key)
                if entry is None:
                    break
                asset = cache.assets.get(entry.asset_id)
                if asset is None:
                    break
                payload = await self._cached_payload(asset, storage)
                if payload is None:
                    break
                candidate_assets.append(asset)
                candidate_payloads.append(payload)
            else:
                keys = candidate_keys
                assets = candidate_assets
                payloads = candidate_payloads
                break
        else:
            return None

        events: list[CostEvent] = [
            cache.record_hit(key, asset, run_id=run_id, operation=operation)
            for key, asset in zip(keys, assets, strict=True)
        ]
        images: list[ImageResult] = []
        for payload, asset in zip(payloads, assets, strict=True):
            probe = validate_media(payload, limits=self._limits)
            images.append(
                ImageResult(
                    data=payload,
                    format=_image_format(probe.format.value),
                    width=probe.width,
                    height=probe.height,
                    seed=asset.seed,
                )
            )
        response = ImageResponse(
            images=images,
            model=assets[0].model,
            provider=assets[0].provider,
        )
        sidecars: tuple[Path, ...] = ()
        if write_sidecar:
            sidecars = await self._write_sidecars(
                assets,
                events,
                run_id=run_id,
                storage=storage,
                stored_objects=[],
            )
        return ImageOutcome(
            response=response,
            assets=tuple(assets),
            events=tuple(events),
            sidecars=sidecars,
            cache_hit=True,
        )

    async def _cached_payload(
        self,
        asset: Asset,
        storage: StorageProvider | None,
    ) -> bytes | None:
        if asset.path:
            path = _file_path(asset.path)
            if path is not None and path.is_file():
                return path.read_bytes()
        if storage is not None and asset.content_hash:
            digest = asset.content_hash
            key = f"objects/{digest[:2]}/{digest}"
            try:
                return await storage.get(key)
            except KeyError:
                return None
        return None

    async def _materialise(
        self,
        result: ImageResult,
        request: ImageRequest,
        *,
        index: int,
        storage: StorageProvider | None,
        key: str | None,
        key_prefix: str | None,
        run_id: str,
    ) -> tuple[MediaProbe, StorageObject | None]:
        if result.data is not None:
            payload = result.data
            probe = validate_media(
                payload,
                limits=self._limits,
                expected_format=result.format.value,
            )
        else:
            if result.url is None:
                raise ValueError("image result has neither data nor url")
            probe = await self._downloader_for().download(
                result.url,
                limits=self._limits,
                expected_format=result.format.value,
            )
            payload = probe.data

        stored: StorageObject | None = None
        if storage is not None:
            storage_key = self._storage_key(
                probe,
                index=index,
                key=key,
                key_prefix=key_prefix,
                count=request.count,
            )
            stored = await storage.put(
                StoragePutRequest(
                    data=payload,
                    key=storage_key,
                    content_type=probe.mime_type,
                    metadata={
                        "kind": "image",
                        "format": probe.format.value,
                        "run_id": run_id,
                    },
                )
            )
        return probe, stored

    def _downloader_for(self) -> MediaDownloader:
        if self._downloader is None:
            self._downloader = MediaDownloader(
                http=self._http,
                transport=self._transport,
                limits=self._limits,
            )
        return self._downloader

    @staticmethod
    def _storage_key(
        probe: MediaProbe,
        *,
        index: int,
        key: str | None,
        key_prefix: str | None,
        count: int,
    ) -> str | None:
        if key:
            base = key.strip()
            if not base:
                raise ValueError("storage key must be non-empty")
            if count > 1:
                return f"{base}-{index}.{probe.extension}"
            return base
        if key_prefix:
            digest = probe.sha256
            return f"{key_prefix}/{digest[:2]}/{digest}.{probe.extension}"
        return None

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


async def generate_image(
    request: ImageRequest,
    router: Router,
    *,
    token: Any = None,
    screening_token: Any = None,
    tracker: CostTracker | None = None,
    cache: AssetCache | None = None,
    storage: StorageProvider | None = None,
    budget: BudgetLike = None,
    safety: Any = None,
    limits: MediaLimits | None = None,
    pricing_overrides: Mapping[str, float] | None = None,
    pricing_version: str = "",
    run_id: str = "",
    metadata: Mapping[str, Any] | None = None,
    remaining_budget: float | None = None,
    **values: Any,
) -> ImageOutcome:
    """Execute one image request without constructing a long-lived service."""
    service = ImageService(
        router,
        tracker=tracker,
        cache=cache,
        storage=storage,
        budget=budget,
        safety=safety,
        limits=limits,
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


def image_service_from_config(
    router: Router,
    config: Any = None,
    **values: Any,
) -> ImageService:
    """Build an :class:`ImageService` for *router* and an optional config.

    The core has no dedicated image section yet, so *config* is accepted for a
    future module and ignored when ``None``. A mapping with ``chain`` or
    ``strategy`` keys is applied as the ``image_gen`` capability chain.
    """
    if isinstance(config, Mapping):
        section = config.get("image_gen", config)
        if isinstance(section, Mapping) and (
            "chain" in section or "strategy" in section
        ):
            base = router.config_for(CAPABILITY_IMAGE_GEN)
            merged = base.model_copy(
                update={
                    key: section[key]
                    for key in ("chain", "strategy", "weights", "pricing_overrides")
                    if key in section
                }
            )
            router = Router(
                router.providers,
                configs={CAPABILITY_IMAGE_GEN: merged},
                seed=router.seed,
                circuit_breakers=router.circuits,
                rate_limiters=router.limiters,
            )
    return ImageService(router, **values)


service_from_config = image_service_from_config
"""Alias for :func:`image_service_from_config`."""

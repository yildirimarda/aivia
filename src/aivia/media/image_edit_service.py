"""Run one image edit request through the single paid path.

The service mirrors :mod:`aivia.media.image_service` for the ``image_edit``
capability: it screens a prompt when one is present, answers an identical
request from the asset cache, applies the remaining budget, asks the router for
a provider, validates and stores every returned image, records one actual cost
event and writes a sidecar next to each output. An instruction edit, an
outpaint, a background removal and an upscale all use this one path; only the
operation and the provider change.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from aivia.core.asset_cache import AssetCacheKey
from aivia.core.models.asset import (
    Asset,
    AssetAnalysis,
    AssetCostLink,
    AssetKind,
)
from aivia.core.models.cost import CostEvent
from aivia.core.pricing import ResolvedPrice
from aivia.core.router import RouteSelection
from aivia.media.image_service import ImageService, _image_format
from aivia.providers.image.cost import IMAGE_UNIT, ImageCostCalculator
from aivia.providers.protocols.image import ImageRequest, ImageResult
from aivia.providers.protocols.image_edit import (
    CAPABILITY_IMAGE_EDIT,
    ImageEditProvider,
    ImageEditRequest,
    ImageEditResponse,
)
from aivia.providers.protocols.media import ImageFormat
from aivia.providers.protocols.storage import (
    StorageObject,
    StorageProvider,
)

__all__ = [
    "ImageEditOutcome",
    "ImageEditService",
    "edit_image",
]

EDIT_OPERATION = CAPABILITY_IMAGE_EDIT
"""The default cost event operation for one image edit call."""


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _media_hash(media: Any) -> str | None:
    if media is None:
        return None
    if media.data is not None:
        return _hash_bytes(media.data)
    if media.url is not None:
        return _hash_text(media.url)
    return None


@dataclass(frozen=True, slots=True)
class ImageEditOutcome:
    """The edit response together with its assets, price, events and sidecars."""

    response: ImageEditResponse
    assets: tuple[Asset, ...]
    probes: tuple[Any, ...] = ()
    storage_objects: tuple[StorageObject, ...] = ()
    events: tuple[CostEvent, ...] = ()
    price: ResolvedPrice | None = None
    selection: RouteSelection | None = None
    sidecars: tuple[Path, ...] = ()
    cache_hit: bool = False

    @property
    def event(self) -> CostEvent | None:
        """The first recorded cost event, or ``None``."""
        return self.events[0] if self.events else None

    @property
    def images(self) -> list[ImageResult]:
        """Every edited image in the provider response."""
        return list(self.response.images)

    @property
    def outputs(self) -> tuple[Asset, ...]:
        """Alias for :attr:`assets`."""
        return self.assets

    @property
    def asset(self) -> Asset | None:
        """The first stored asset, or ``None``."""
        return self.assets[0] if self.assets else None

    @property
    def paths(self) -> tuple[str, ...]:
        """Every stored asset path in response order."""
        return tuple(asset.path for asset in self.assets)

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


class ImageEditService(ImageService):
    """Execute image edits through a :class:`~aivia.core.router.Router`.

    The constructor accepts the same collaborators as :class:`ImageService`.
    The generation calculator inherited from it is unused here; every edit is
    priced against the ``image_edit`` capability instead.
    """

    def __init__(
        self,
        router: Any,
        *,
        pricing_overrides: Mapping[str, float] | None = None,
        pricing_version: str = "",
        **values: Any,
    ) -> None:
        values.setdefault("operation", EDIT_OPERATION)
        super().__init__(router, **values)
        if pricing_overrides is None:
            pricing_overrides = router.config_for(
                CAPABILITY_IMAGE_EDIT
            ).pricing_overrides
        self._edit_calculator = ImageCostCalculator(
            pricing_overrides,
            pricing_version=pricing_version,
        )

    @property
    def calculator(self) -> ImageCostCalculator:
        """The calculator that prices and records image edit calls."""
        return self._edit_calculator

    def _edit_parameters(
        self,
        request: ImageEditRequest,
        index: int = 0,
    ) -> dict[str, Any]:
        seed = request.seed + index if request.seed is not None else None
        return {
            "operation": request.operation.value,
            "model": request.model,
            "width": request.width,
            "height": request.height,
            "scale": request.scale,
            "format": (request.format or ImageFormat.PNG).value,
            "seed": seed,
            "count": request.count,
            "index": index,
        }

    def edit_cache_key(
        self,
        request: ImageEditRequest,
        *,
        index: int = 0,
    ) -> AssetCacheKey:
        """Return the cache key for one output of an edit request."""
        if not isinstance(request, ImageEditRequest):
            raise TypeError("request must be an ImageEditRequest")
        prompt = request.prompt or request.operation.value
        references = tuple(
            value
            for value in (_media_hash(request.image), _media_hash(request.mask))
            if value is not None
        )
        if self._cache is None:
            return AssetCacheKey(
                provider="",
                model=request.model or "",
                prompt=prompt,
                parameters=self._edit_parameters(request, index),
                reference_hashes=references,
            )
        return self._cache.key(
            "",
            request.model or "",
            prompt,
            parameters=self._edit_parameters(request, index),
            reference_hashes=references,
        )

    async def edit(
        self,
        request: ImageEditRequest,
        *,
        token: Any = None,
        screening_token: Any = None,
        screen_token: Any = None,
        run_id: str = "",
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
    ) -> ImageEditOutcome:
        """Edit *request* through the screened paid path."""
        if not isinstance(request, ImageEditRequest):
            raise TypeError("request must be an ImageEditRequest")
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
        if request.prompt:
            self._require_screened(active_token, request.prompt)

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

        if use_cache:
            cached = await self._cached_edit_outcome(
                request,
                run_id=run_id,
                operation=operation_name,
                storage=active_storage,
                write_sidecar=write_sidecar,
            )
            if cached is not None:
                return cached

        budget = remaining_budget
        if budget is None and self._budget is not None:
            budget = self._budget.remaining(run_id, operation=operation_name)

        base_seed = request.seed if request.seed is not None else secrets.randbits(32)
        provider_request = (
            request
            if request.seed is not None
            else request.model_copy(update={"seed": base_seed})
        )

        async def run(provider: Any) -> ImageEditResponse:
            return await cast(ImageEditProvider, provider).edit(provider_request)

        response = await self._router.execute(
            CAPABILITY_IMAGE_EDIT,
            run,
            remaining_budget=budget,
            units=float(request.count),
            unit_kind=IMAGE_UNIT,
            exclude=exclude,
        )
        if not isinstance(response, ImageEditResponse):
            raise TypeError("image edit provider must return an ImageEditResponse")
        selection = self._router.last_selection
        if selection is None:
            raise RuntimeError("router completed an image edit without a selection")
        provider_name = response.provider or selection.name

        seeds = tuple(
            image.seed
            if image.seed is not None
            else (
                request.seed + index if request.seed is not None else base_seed + index
            )
            for index, image in enumerate(response.images)
        )
        response = response.model_copy(
            update={
                "images": [
                    image.model_copy(update={"seed": seed})
                    for image, seed in zip(response.images, seeds, strict=True)
                ]
            }
        )

        pseudo_request = ImageRequest(
            prompt=request.prompt or request.operation.value,
            model=request.model,
            width=request.width,
            height=request.height,
            format=request.format or ImageFormat.PNG,
            seed=request.seed,
            count=request.count,
        )
        materialised: list[tuple[Any, StorageObject | None]] = []
        for index, result in enumerate(response.images):
            probe, stored = await self._materialise(
                result,
                pseudo_request,
                index=index,
                storage=active_storage,
                key=key,
                key_prefix=active_prefix,
                run_id=run_id,
            )
            materialised.append((probe, stored))
        if not materialised:
            raise ValueError("image edit provider returned no images")

        price = self._edit_calculator.resolve(
            selection.provider,
            provider_name=selection.name,
            units=float(response.count),
            unit_kind=IMAGE_UNIT,
            operation=operation_name,
        )
        event = self._edit_calculator.cost_event(
            selection.provider,
            provider_name=selection.name,
            run_id=run_id,
            operation=operation_name,
            estimated=estimated,
            units=float(response.count),
            unit_kind=IMAGE_UNIT,
            metadata=metadata,
            price=price,
        )
        details = dict(event.metadata)
        details["image_count"] = response.count
        details["operation"] = request.operation.value
        if response.model:
            details["model"] = response.model
        if response.request_id:
            details["request_id"] = response.request_id
        event = event.model_copy(update={"model": response.model, "metadata": details})
        recorded: CostEvent | None = None
        if write_cost and self._tracker is not None:
            recorded = (
                self._tracker.record_estimate(event)
                if estimated
                else self._tracker.record_actual(event)
            )
            event = recorded

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
                prompt=request.prompt or "",
                path=(stored.url or stored.key) if stored is not None else "",
                content_hash=probe.sha256,
                run_id=run_id,
                pinned=pinned,
                seed=seeds[index],
                analysis=AssetAnalysis(width=probe.width, height=probe.height),
                cost_links=cost_links,
            )
            if use_cache and self._cache is not None:
                asset = self._cache.store(
                    asset,
                    key=self.edit_cache_key(request, index=index),
                )
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

        return ImageEditOutcome(
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
        )

    async def _cached_edit_outcome(
        self,
        request: ImageEditRequest,
        *,
        run_id: str,
        operation: str,
        storage: StorageProvider | None,
        write_sidecar: bool,
    ) -> ImageEditOutcome | None:
        cache = self._cache
        if cache is None:
            return None
        keys = [
            self.edit_cache_key(request, index=index) for index in range(request.count)
        ]
        assets: list[Asset] = []
        payloads: list[bytes] = []
        for key in keys:
            entry = cache.entry(key)
            if entry is None:
                return None
            asset = cache.assets.get(entry.asset_id)
            if asset is None:
                return None
            payload = await self._cached_payload(asset, storage)
            if payload is None:
                return None
            assets.append(asset)
            payloads.append(payload)

        events = [
            cache.record_hit(key, asset, run_id=run_id, operation=operation)
            for key, asset in zip(keys, assets, strict=True)
        ]
        images: list[ImageResult] = []
        for payload, asset in zip(payloads, assets, strict=True):
            from aivia.core.media_probe import validate_media

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
        response = ImageEditResponse(
            images=images,
            operation=request.operation,
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
        return ImageEditOutcome(
            response=response,
            assets=tuple(assets),
            events=tuple(events),
            sidecars=sidecars,
            cache_hit=True,
        )


async def edit_image(
    request: ImageEditRequest,
    router: Any,
    *,
    token: Any = None,
    screening_token: Any = None,
    **values: Any,
) -> ImageEditOutcome:
    """Execute one image edit without constructing a long-lived service."""
    service = ImageEditService(router, **values)
    return await service.edit(
        request,
        token=token,
        screening_token=screening_token,
    )

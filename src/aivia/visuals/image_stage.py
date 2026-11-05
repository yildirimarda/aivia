"""Resolve image beats through the single screened, cached and costed path.

An image beat carries a generation prompt rather than a stock query. This stage
turns every image request into a typed
:class:`~aivia.providers.protocols.image.ImageRequest`, hands it to the media
service and keeps the stored asset together with the prompt hash, provider and
model that produced it. Callers never choose a vendor or write a cost event:
the default path is ``context.media.image``, which runs the same
screen -> cache -> budget -> router -> provider -> validate -> store -> cost ->
sidecar order as every other paid call.

An injected image service or callable keeps the stage offline in tests. The
stage is deterministic for the same service responses and requests: beats keep
their input order, non-image beats are ignored and an empty prompt is reported
as unresolved instead of being sent to a provider.
"""

from __future__ import annotations

import contextlib
import inspect
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from aivia.core.errors import ConfigError
from aivia.core.models.asset import Asset, compute_prompt_hash
from aivia.core.models.base import canonical_json
from aivia.core.models.run import StageResult, StageState
from aivia.core.models.script import MediaKind, VisualType
from aivia.core.router import Router
from aivia.core.stages.base import Stage, StageContext
from aivia.providers.protocols.image import ImageRequest
from aivia.visuals.strategy import VisualRequest, resolve_visual_requests

__all__ = [
    "DEFAULT_IMAGES_FILENAME",
    "DEFAULT_IMAGE_FILENAME",
    "DEFAULT_IMAGE_MANIFEST_FILENAME",
    "IMAGE_MANIFEST_FILENAME",
    "IMAGE_STAGE_NAME",
    "ImageAssetResult",
    "ImageBeat",
    "ImageBeatResult",
    "ImageDecision",
    "ImageResolver",
    "ImageResult",
    "ImageStage",
    "ImageStageResult",
    "ResolvedImage",
    "build_image_request",
    "build_image_stage",
    "image_stage",
    "resolve_image",
    "resolve_image_beat",
    "resolve_image_request",
    "resolve_image_request_sync",
    "resolve_image_requests",
    "resolve_image_requests_sync",
    "resolve_images",
    "run_image_stage",
]

IMAGE_STAGE_NAME = "image"
"""The recipe stage name this implementation serves."""

DEFAULT_IMAGE_FILENAME = "image.json"
"""The manifest that records every resolved image beat under the run directory."""

DEFAULT_IMAGES_FILENAME = "images.json"
"""Plural alias of :data:`DEFAULT_IMAGE_FILENAME` kept for callers and manifests."""

DEFAULT_IMAGE_MANIFEST_FILENAME = DEFAULT_IMAGE_FILENAME
"""Alias for :data:`DEFAULT_IMAGE_FILENAME`."""

IMAGE_MANIFEST_FILENAME = DEFAULT_IMAGE_FILENAME
"""Alias for :data:`DEFAULT_IMAGE_FILENAME`."""

DEFAULT_IMAGE_COUNT = 1
"""How many variations one beat requests unless the beat says otherwise."""

DEFAULT_PROMPT_METADATA_KEY = "prompt"
"""The request metadata key checked when a beat has no prompt of its own."""

_OPTIONAL_REQUEST_FIELDS = (
    "negative_prompt",
    "style",
    "aspect_ratio",
    "model",
    "format",
    "quality",
    "width",
    "height",
    "seed",
    "count",
)


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer", details={"value": repr(value)})
    if value < 1:
        raise ConfigError(f"{name} must be positive", details={"value": repr(value)})
    return value


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
            raise ConfigError("image resolution needs a non-empty prompt")
        return VisualRequest(
            visual_type=VisualType.IMAGE,
            kind=MediaKind.IMAGE,
            prompt=text,
            text=text,
        )
    requests = resolve_visual_requests(value, strategy="image")
    return requests[0]


def _prompt_of(request: VisualRequest, metadata: Mapping[str, Any]) -> str:
    return (
        _text(request.prompt)
        or _text(metadata.get(DEFAULT_PROMPT_METADATA_KEY))
        or _text(request.text)
    )


def _request_values(
    request: VisualRequest,
    *,
    metadata: Mapping[str, Any],
    model: str | None = None,
    aspect_ratio: str | None = None,
    negative_prompt: str | None = None,
    style: str | None = None,
    seed: int | None = None,
    count: int | None = None,
    width: int | None = None,
    height: int | None = None,
    fmt: Any = None,
    quality: str | None = None,
) -> dict[str, Any]:
    values: dict[str, Any] = {"prompt": _prompt_of(request, metadata)}
    chosen_negative = (
        negative_prompt if negative_prompt is not None else request.negative_prompt
    )
    if _text(chosen_negative):
        values["negative_prompt"] = _text(chosen_negative)
    chosen_style = style if style is not None else request.style
    if _text(chosen_style):
        values["style"] = _text(chosen_style)
    chosen_aspect = aspect_ratio if aspect_ratio is not None else request.aspect_ratio
    if _text(chosen_aspect):
        values["aspect_ratio"] = _text(chosen_aspect)
    chosen_model = model if model is not None else _text(metadata.get("model"))
    if chosen_model:
        values["model"] = chosen_model
    chosen_seed = seed if seed is not None else request.seed
    if chosen_seed is None:
        raw_seed = metadata.get("seed")
        if isinstance(raw_seed, int) and not isinstance(raw_seed, bool):
            chosen_seed = raw_seed
    if chosen_seed is not None:
        values["seed"] = chosen_seed
    chosen_count = count if count is not None else metadata.get("count")
    if chosen_count is None:
        chosen_count = DEFAULT_IMAGE_COUNT
    values["count"] = _positive_int(chosen_count, "count") or DEFAULT_IMAGE_COUNT
    chosen_width = width if width is not None else metadata.get("width")
    chosen_height = height if height is not None else metadata.get("height")
    if chosen_width is not None and chosen_height is not None:
        values["width"] = _positive_int(chosen_width, "width")
        values["height"] = _positive_int(chosen_height, "height")
    chosen_format = fmt if fmt is not None else metadata.get("format")
    if chosen_format is not None:
        values["format"] = chosen_format
    chosen_quality = quality if quality is not None else _text(metadata.get("quality"))
    if chosen_quality:
        values["quality"] = chosen_quality
    return values


def _image_request(
    request: VisualRequest,
    *,
    metadata: Mapping[str, Any] | None = None,
    **options: Any,
) -> ImageRequest | None:
    request_meta = _request_metadata(request)
    active_meta = dict(request_meta)
    active_meta.update(metadata or {})
    values = _request_values(request, metadata=active_meta, **options)
    prompt = _text(values.get("prompt"))
    if not prompt:
        return None
    values["prompt"] = prompt
    call_metadata = dict(active_meta)
    call_metadata.update(
        {
            "stage": IMAGE_STAGE_NAME,
            "beat_index": request.index,
        }
    )
    values["metadata"] = call_metadata
    return ImageRequest(**values)


def _asset_values(asset: Any) -> dict[str, Any]:
    if isinstance(asset, Mapping):
        return dict(asset)
    values: dict[str, Any] = {}
    for name in Asset.model_fields:
        if hasattr(asset, name):
            values[name] = getattr(asset, name)
    return values


def _asset_from(value: Any) -> Asset | None:
    if value is None:
        return None
    if isinstance(value, Asset):
        return value
    if isinstance(value, Mapping):
        if "asset" in value:
            return _asset_from(value.get("asset"))
        return Asset.model_validate(value)
    outcome = getattr(value, "outcome", None)
    if outcome is not None and outcome is not value:
        found = _asset_from(outcome)
        if found is not None:
            return found
    for name in ("asset", "first", "image"):
        candidate = getattr(value, name, None)
        if candidate is not None and not callable(candidate):
            found = _asset_from(candidate)
            if found is not None:
                return found
    for name in ("assets", "outputs", "images"):
        candidates = getattr(value, name, None)
        if isinstance(candidates, Sequence) and not isinstance(
            candidates, (str, bytes, bytearray)
        ):
            for candidate in candidates:
                found = _asset_from(candidate)
                if found is not None:
                    return found
    if hasattr(value, "prompt") or hasattr(value, "path"):
        return Asset.model_validate(_asset_values(value))
    return None


def _outcome_values(value: Any) -> tuple[str, str, float, bool]:
    if value is None:
        return "", "", 0.0, False
    outcome = getattr(value, "outcome", None)
    if outcome is not None and outcome is not value:
        return _outcome_values(outcome)
    if isinstance(value, Mapping):
        provider = _text(value.get("provider"))
        model = _text(value.get("model"))
        amount = value.get("cost_usd", value.get("amount_usd", 0.0))
        cached = bool(value.get("cached") or value.get("cache_hit"))
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            amount = 0.0
        return provider, model, max(0.0, float(amount)), cached
    provider = _text(getattr(value, "provider", ""))
    model = _text(getattr(value, "model", ""))
    amount = getattr(value, "cost_usd", None)
    if amount is None:
        amount = getattr(value, "amount_usd", 0.0)
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        amount = 0.0
    cached = bool(getattr(value, "cached", False) or getattr(value, "cache_hit", False))
    return provider, model, max(0.0, float(amount)), cached


def _normalise_asset(
    asset: Asset,
    request: ImageRequest,
    *,
    provider: str,
    model: str,
) -> Asset:
    prompt = asset.prompt or request.prompt
    prompt_hash = asset.prompt_hash or compute_prompt_hash(prompt)
    updates: dict[str, Any] = {}
    if asset.prompt != prompt:
        updates["prompt"] = prompt
    if asset.prompt_hash != prompt_hash:
        updates["prompt_hash"] = prompt_hash
    if not asset.provider and provider:
        updates["provider"] = provider
    if not asset.model and model:
        updates["model"] = model
    return asset.model_copy(update=updates) if updates else asset


def _method_of(target: Any) -> Any:
    if callable(target) and not any(
        callable(getattr(target, name, None))
        for name in ("image", "generate", "generate_image")
    ):
        return target
    for name in ("image", "generate", "generate_image", "resolve"):
        candidate = getattr(target, name, None)
        if callable(candidate):
            return candidate
    if callable(target):
        return target
    return None


def _supported_kwargs(method: Any, values: Mapping[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return {}
    parameters = signature.parameters
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return dict(values)
    return {name: value for name, value in values.items() if name in parameters}


async def _call_method(
    method: Any, request: ImageRequest, values: Mapping[str, Any]
) -> Any:
    kwargs = _supported_kwargs(method, values)
    try:
        result = method(request, **kwargs)
    except TypeError:
        if not kwargs:
            raise
        result = method(request)
    if inspect.isawaitable(result):
        result = await result
    return result


async def _call_image(
    target: Any,
    request: ImageRequest,
    *,
    context: StageContext | None,
    run_id: str,
    metadata: Mapping[str, Any] | None,
    token: Any = None,
    operation: str | None = None,
    **options: Any,
) -> Any:
    if target is None:
        if context is None or context.router is None:
            raise ConfigError(
                "image resolution needs an image service or a stage context router",
                details={"stage": IMAGE_STAGE_NAME},
            )
        target = context.media
    if isinstance(target, Router):
        from aivia.media.image_service import ImageService

        target = ImageService(target)
    method = _method_of(target)
    if method is None:
        raise ConfigError(
            "image service must be callable or expose an image or generate method",
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
    return await _call_method(method, request, values)


class ImageBeatResult(BaseModel):
    """The resolved image asset for one beat, or why it was not resolved.

    Attribute access falls through to the stored
    :class:`~aivia.core.models.asset.Asset`,
    so a caller may read ``result.path`` or ``result.seed`` as well as
    ``result.asset``. ``prompt_hash``, ``provider`` and ``model`` are repeated
    at the top level so a manifest reader does not have to unpack the asset.
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
        """Whether an image asset was stored."""
        return self.asset is not None

    @property
    def passed(self) -> bool:
        """Alias for :attr:`resolved`."""
        return self.resolved

    @property
    def image(self) -> Asset | None:
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
            "metadata": dict(self.metadata),
        }


ImageResult = ImageBeatResult
ImageBeat = ImageBeatResult
ResolvedImage = ImageBeatResult
ImageDecision = ImageBeatResult
ImageAssetResult = ImageBeatResult


class ImageStageResult(StageResult):
    """A stage result carrying one :class:`ImageBeatResult` per image beat."""

    beats: list[ImageBeatResult] = Field(
        default_factory=list,
        validation_alias=AliasChoices("beats", "results", "items", "selections"),
    )

    @property
    def results(self) -> list[ImageBeatResult]:
        """Alias for :attr:`beats`."""
        return self.beats

    @property
    def images(self) -> list[ImageBeatResult]:
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
    def unresolved(self) -> list[ImageBeatResult]:
        """Beats for which no image asset was stored."""
        return [beat for beat in self.beats if beat.asset is None]

    @property
    def prompt_hashes(self) -> dict[int, str]:
        """The prompt hash keyed by beat index."""
        return {beat.index: beat.prompt_hash for beat in self.beats}

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


async def resolve_image_request(
    value: Any,
    *,
    service: Any = None,
    image_service: Any = None,
    image: Any = None,
    generator: Any = None,
    provider: Any = None,
    router: Any = None,
    context: StageContext | None = None,
    config: Any = None,
    model: str | None = None,
    aspect_ratio: str | None = None,
    negative_prompt: str | None = None,
    style: str | None = None,
    seed: int | None = None,
    count: int | None = None,
    width: int | None = None,
    height: int | None = None,
    fmt: Any = None,
    quality: str | None = None,
    token: Any = None,
    screening_token: Any = None,
    run_id: str = "",
    metadata: Mapping[str, Any] | None = None,
    operation: str | None = None,
    **values: Any,
) -> ImageBeatResult:
    """Resolve one image beat through the media service.

    ``service`` (or its aliases ``image``, ``generator``, ``provider``) may be
    an :class:`~aivia.media.image_service.ImageService`, a media facade, a
    router or an async callable. When it is omitted the stage context's media
    facade is used. The returned result is unresolved only when the beat has no
    prompt or the service returned no asset.
    """
    request = _coerce_request(value)
    active_meta = dict(metadata or {})
    image_request = _image_request(
        request,
        metadata=active_meta,
        model=model,
        aspect_ratio=aspect_ratio,
        negative_prompt=negative_prompt,
        style=style,
        seed=seed,
        count=count,
        width=width,
        height=height,
        fmt=fmt,
        quality=quality,
    )
    if image_request is None:
        return ImageBeatResult(
            index=request.index,
            request=request,
            metadata={**active_meta, "reason": "no prompt"},
        )

    target = service if service is not None else image_service
    if target is None:
        target = image
    if target is None:
        target = generator if generator is not None else provider
    if target is None and router is not None:
        target = router
    active_token = token if token is not None else screening_token
    outcome = await _call_image(
        target,
        image_request,
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
        return ImageBeatResult(
            index=request.index,
            request=request,
            provider=outcome_provider,
            model=outcome_model,
            cost_usd=cost,
            cached=cached,
            error="image service returned no asset",
            metadata=dict(active_meta),
            outcome=outcome,
        )
    asset = _normalise_asset(
        asset,
        image_request,
        provider=outcome_provider,
        model=outcome_model,
    )
    prompt_hash = asset.prompt_hash or compute_prompt_hash(image_request.prompt)
    return ImageBeatResult(
        index=request.index,
        request=request,
        asset=asset,
        prompt_hash=prompt_hash,
        provider=asset.provider or outcome_provider,
        model=asset.model or outcome_model,
        cost_usd=cost,
        cached=cached,
        metadata=dict(active_meta),
        outcome=outcome,
    )


resolve_image_beat = resolve_image_request
resolve_image = resolve_image_request
build_image_request = resolve_image_request
"""Aliases for :func:`resolve_image_request`."""


async def resolve_image_requests(
    values: Any,
    **kwargs: Any,
) -> list[ImageBeatResult]:
    """Resolve every image request in *values*, preserving input order."""
    if isinstance(values, VisualRequest):
        requests = [values]
    elif isinstance(values, Mapping):
        requests = [_coerce_request(values)]
    elif isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray)
    ):
        requests = [_coerce_request(item) for item in values]
    else:
        requests = resolve_visual_requests(values, strategy="image")
    results: list[ImageBeatResult] = []
    for position, request in enumerate(requests):
        result = await resolve_image_request(request, **kwargs)
        if result.index != position and "index" not in request.model_fields_set:
            result = result.model_copy(update={"index": position})
        results.append(result)
    return results


resolve_image_beats = resolve_image_requests
resolve_images = resolve_image_requests
"""Aliases for :func:`resolve_image_requests`."""


def resolve_image_request_sync(value: Any, **kwargs: Any) -> ImageBeatResult:
    """Run :func:`resolve_image_request` from synchronous code."""
    import asyncio

    return asyncio.run(resolve_image_request(value, **kwargs))


def resolve_image_requests_sync(values: Any, **kwargs: Any) -> list[ImageBeatResult]:
    """Run :func:`resolve_image_requests` from synchronous code."""
    import asyncio

    return asyncio.run(resolve_image_requests(values, **kwargs))


class ImageResolver:
    """Reusable image resolver carrying the service and request defaults."""

    def __init__(
        self,
        service: Any = None,
        *,
        image_service: Any = None,
        image: Any = None,
        generator: Any = None,
        provider: Any = None,
        router: Any = None,
        config: Any = None,
        model: str | None = None,
        aspect_ratio: str | None = None,
        negative_prompt: str | None = None,
        style: str | None = None,
        seed: int | None = None,
        count: int | None = None,
        width: int | None = None,
        height: int | None = None,
        fmt: Any = None,
        quality: str | None = None,
        token: Any = None,
        run_id: str = "",
        metadata: Mapping[str, Any] | None = None,
        operation: str | None = None,
        **options: Any,
    ) -> None:
        self.service = service if service is not None else image_service
        if self.service is None:
            self.service = image
        if self.service is None:
            self.service = generator if generator is not None else provider
        self.router = router
        self.config = config
        self.model = model
        self.aspect_ratio = aspect_ratio
        self.negative_prompt = negative_prompt
        self.style = style
        self.seed = seed
        self.count = count
        self.width = width
        self.height = height
        self.fmt = fmt
        self.quality = quality
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
            "style": self.style,
            "seed": self.seed,
            "count": self.count,
            "width": self.width,
            "height": self.height,
            "fmt": self.fmt,
            "quality": self.quality,
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
    ) -> ImageBeatResult:
        return await resolve_image_request(
            value,
            context=context,
            **self._values(**overrides),
        )

    async def resolve_beat(
        self,
        value: Any,
        **overrides: Any,
    ) -> ImageBeatResult:
        return await self.resolve(value, **overrides)

    async def resolve_requests(
        self,
        values: Any,
        *,
        context: StageContext | None = None,
        **overrides: Any,
    ) -> list[ImageBeatResult]:
        return await resolve_image_requests(
            values,
            context=context,
            **self._values(**overrides),
        )

    __call__ = resolve


ImageBeatResolver = ImageResolver
ImageRequestResolver = ImageResolver


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical_json(value, indent=2)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


class ImageStage(Stage):
    """Resolve every image beat into a stored, costed asset.

    ``requests`` may be a script, a beat sequence, one or more visual requests
    or mappings. A ``script``/``beats`` option is resolved with the image
    strategy first. ``service``/``image``/``generator`` supplies the image
    service; without one the stage uses ``context.media.image``. Non-image
    beats are ignored so a mixed visual plan can be resolved in one pass.
    """

    name = IMAGE_STAGE_NAME
    artifacts: tuple[str, ...] = (DEFAULT_IMAGE_FILENAME,)

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
        image_service: Any = None,
        image: Any = None,
        generator: Any = None,
        provider: Any = None,
        image_provider: Any = None,
        router: Any = None,
        config: Any = None,
        model: str | None = None,
        aspect_ratio: str | None = None,
        negative_prompt: str | None = None,
        style: str | None = None,
        seed: int | None = None,
        count: int | None = None,
        width: int | None = None,
        height: int | None = None,
        format: Any = None,
        quality: str | None = None,
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
            "image_service": image_service,
            "image": image,
            "generator": generator,
            "provider": provider,
            "image_provider": image_provider,
            "router": router,
            "config": config,
            "model": model,
            "aspect_ratio": aspect_ratio,
            "negative_prompt": negative_prompt,
            "style": style,
            "seed": seed,
            "count": count,
            "width": width,
            "height": height,
            "format": format,
            "quality": quality,
            "filename": filename,
            "manifest_filename": manifest_filename,
        }
        for key, value in supplied.items():
            if value is not None:
                merged[key] = value
        self.filename = _text(
            merged.pop("filename", None)
            or merged.pop("manifest_filename", None)
            or DEFAULT_IMAGE_FILENAME
        )
        if not self.filename:
            self.filename = DEFAULT_IMAGE_FILENAME
        self.artifacts = (self.filename,)

        self._requests_input = merged.pop("requests", None)
        if self._requests_input is None:
            self._requests_input = merged.pop("visuals", None)
        self._script = merged.pop("script", None)
        self._beats = merged.pop("beats", None)
        self._service = merged.pop("service", None)
        if self._service is None:
            self._service = merged.pop("image_service", None)
        if self._service is None:
            self._service = merged.pop("image", None)
        if self._service is None:
            self._service = merged.pop("generator", None)
        if self._service is None:
            self._service = merged.pop("provider", None)
        if self._service is None:
            self._service = merged.pop("image_provider", None)
        self._router = merged.pop("router", None)
        self._config = merged.pop("config", None)
        self.model = merged.pop("model", None)
        self.aspect_ratio = merged.pop("aspect_ratio", None)
        self.negative_prompt = merged.pop("negative_prompt", None)
        self.style = merged.pop("style", None)
        self.seed = merged.pop("seed", None)
        self.count = merged.pop("count", None)
        self.width = merged.pop("width", None)
        self.height = merged.pop("height", None)
        self.format = merged.pop("format", None)
        self.quality = merged.pop("quality", None)
        self.results: list[ImageBeatResult] = []
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
        return resolve_visual_requests(value, strategy="image")

    async def run(self, context: StageContext) -> StageResult:
        """Resolve every image beat and write the ``images.json`` manifest."""
        requests = self._requests()
        image_requests = [
            request for request in requests if request.visual_type is VisualType.IMAGE
        ]
        run_id = context.run.id if context.run is not None else ""
        results: list[ImageBeatResult] = []
        for position, request in enumerate(image_requests):
            result = await resolve_image_request(
                request,
                service=self._service,
                router=self._router if self._router is not None else context.router,
                context=context,
                config=self._config if self._config is not None else context.config,
                model=self.model,
                aspect_ratio=self.aspect_ratio,
                negative_prompt=self.negative_prompt,
                style=self.style,
                seed=self.seed,
                count=self.count,
                width=self.width,
                height=self.height,
                fmt=self.format,
                quality=self.quality,
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
                "prompt_hashes": [
                    result.prompt_hash for result in results if result.prompt_hash
                ],
            },
        }
        relative = self.filename
        _write_json(context.run_dir / relative, payload)
        if relative == DEFAULT_IMAGE_FILENAME:
            _write_json(context.run_dir / DEFAULT_IMAGES_FILENAME, payload)
        total_cost = sum(result.cost_usd for result in results)
        providers = [result.provider for result in results if result.provider]
        models = [result.model for result in results if result.model]
        return ImageStageResult(
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
                "providers": list(dict.fromkeys(providers)),
                "models": list(dict.fromkeys(models)),
                "prompt_hashes": [
                    result.prompt_hash for result in results if result.prompt_hash
                ],
                "manifest": relative,
            },
        )


image_stage = ImageStage
"""Alias for :class:`ImageStage`."""


def build_image_stage(
    *,
    options: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> ImageStage:
    """Build an :class:`ImageStage` from recipe options and overrides."""
    return ImageStage(options=options, **kwargs)


async def run_image_stage(
    context: StageContext,
    *,
    stage: ImageStage | None = None,
    **options: Any,
) -> StageResult:
    """Build a stage unless one is supplied, then run it asynchronously."""
    if stage is not None and options:
        raise ConfigError("pass either a stage or options, not both")
    active = stage if stage is not None else ImageStage(**options)
    return await active.run(context)

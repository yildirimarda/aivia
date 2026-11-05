"""Run the shared image and video quality gates for every visual asset.

The image and video stages turn a beat into one stored asset. That asset can
still miss the intended description, so this module applies the same quality
assurance used by the media layer: the vision model only returns a score while
retries, provider fallback and the final stock or image to video fallback stay
in deterministic code.

Every attempt is kept, including the rejected ones, with the provider, model,
score and the cost that attempt actually incurred. A beat therefore reports the
full spend, not only the cost of the asset that happened to pass. The accepted
asset carries its quality score so a later persistence step can store it.

The stage is the pipeline entry point: it resolves mixed image and video
requests, runs the matching quality gate per beat and writes a manifest that
lists every attempt and its cost. Stock beats are left to the stock stage,
which applies relevance scoring instead of generation quality checks.
"""

from __future__ import annotations

import inspect
from collections.abc import Coroutine, Mapping, Sequence
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from aivia.core.errors import ConfigError
from aivia.core.models.asset import Asset
from aivia.core.models.run import StageResult, StageState
from aivia.core.models.script import MediaKind, VisualType
from aivia.core.stages.base import Stage, StageContext
from aivia.media.image_qa import ImageQAResult, run_image_qa
from aivia.media.video_qa import VideoQAResult, run_video_qa
from aivia.providers.protocols.stock import StockAsset
from aivia.visuals.image_stage import (
    _image_request,
    _supported_kwargs,
    _text,
    _write_json,
)
from aivia.visuals.strategy import (
    DEFAULT_VISUAL_STRATEGY,
    VisualRequest,
    resolve_visual_requests,
)
from aivia.visuals.video_stage import (
    _config_cap,
    _platform_catalog,
    _resolve_platform,
    _video_plan,
)

__all__ = [
    "DEFAULT_VISUAL_QA_FILENAME",
    "DEFAULT_VISUAL_QA_MANIFEST_FILENAME",
    "VISUAL_QA_STAGE_NAME",
    "ResolvedVisualQA",
    "VisualQA",
    "VisualQAAttempt",
    "VisualQABeat",
    "VisualQAResult",
    "VisualQAService",
    "VisualQAStage",
    "VisualQAStageResult",
    "build_visual_qa_stage",
    "resolve_image_qa",
    "resolve_visual_qa",
    "resolve_visual_qa_requests",
    "resolve_visual_qa_sync",
    "run_visual_qa",
    "run_visual_qa_stage",
    "run_visual_qa_sync",
    "visual_qa_stage",
]

VISUAL_QA_STAGE_NAME = "visual_qa"
"""The recipe stage name this implementation serves."""

DEFAULT_VISUAL_QA_FILENAME = "visual-qa.json"
"""The manifest that records every quality attempt under the run directory."""

DEFAULT_VISUAL_QA_MANIFEST_FILENAME = DEFAULT_VISUAL_QA_FILENAME
"""Alias for :data:`DEFAULT_VISUAL_QA_FILENAME`."""

_IMAGE_KINDS = frozenset({MediaKind.IMAGE, MediaKind.STILL})
_VIDEO_KINDS = frozenset({MediaKind.VIDEO, MediaKind.MOTION})
_SUPPORTED_TYPES = frozenset(
    {
        VisualType.IMAGE,
        VisualType.STILL,
        VisualType.TEXT,
        VisualType.VIDEO,
    }
)

_IMAGE_REQUEST_KEYS = (
    "model",
    "aspect_ratio",
    "negative_prompt",
    "style",
    "seed",
    "count",
    "width",
    "height",
    "format",
)
_VIDEO_REQUEST_KEYS = (
    "model",
    "aspect_ratio",
    "negative_prompt",
    "seed",
    "count",
    "width",
    "height",
    "fps",
    "generate_audio",
    "audio_prompt",
    "duration_seconds",
    "max_clip_seconds",
    "beat_duration_seconds",
    "platform",
    "platforms",
    "format",
)
_IMAGE_QA_KEYS = (
    "config",
    "quality",
    "scorer",
    "vision",
    "helper",
    "stock",
    "stock_provider",
    "stock_assets",
    "stock_ranker",
    "providers",
    "chain",
    "min_score",
    "max_retries",
    "weights",
    "enabled",
    "vision_check",
    "no_cache",
    "score_stock",
    "prompt",
    "role",
)
_VIDEO_QA_KEYS = (
    "config",
    "quality",
    "scorer",
    "vision",
    "helper",
    "sampler",
    "frame_sampler",
    "fallback",
    "i2v",
    "i2v_flow",
    "images",
    "image_service",
    "still",
    "still_request",
    "image_request",
    "providers",
    "chain",
    "min_score",
    "max_retries",
    "weights",
    "enabled",
    "vision_check",
    "no_cache",
    "prompt",
    "role",
    "sample_frames",
    "scene_threshold",
    "ffmpeg_executable",
    "ffmpeg_runner",
    "ffprobe_executable",
    "ffprobe_runner",
    "timeout_seconds",
)


def _coerce_request(value: Any) -> VisualRequest:
    if isinstance(value, VisualRequest):
        return value
    if isinstance(value, Mapping):
        return VisualRequest.model_validate(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ConfigError("visual quality checks need a non-empty prompt")
        return VisualRequest(prompt=text, text=text)
    requests = resolve_visual_requests(value)
    return requests[0]


def _kind_of(request: VisualRequest) -> MediaKind:
    if request.visual_type is VisualType.VIDEO or request.kind in _VIDEO_KINDS:
        return MediaKind.VIDEO
    if request.visual_type is VisualType.STOCK or request.kind is MediaKind.STOCK:
        return MediaKind.STOCK
    return MediaKind.IMAGE


def _score_number(score: Any) -> float | None:
    if score is None:
        return None
    value = getattr(score, "overall", score)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _model_of(value: Any) -> str:
    if value is None:
        return ""
    outcome = getattr(value, "outcome", None)
    if outcome is not None and outcome is not value:
        return _model_of(outcome)
    model = getattr(value, "model", "")
    return model if isinstance(model, str) else ""


def _cached_of(value: Any) -> bool:
    if value is None:
        return False
    return bool(getattr(value, "cached", False) or getattr(value, "cache_hit", False))


def _request_values(
    options: Mapping[str, Any],
    keys: Sequence[str],
    *,
    rename: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    values = {key: options[key] for key in keys if key in options}
    for source, target in (rename or {}).items():
        if source in values:
            values[target] = values.pop(source)
    return values


def _qa_values(options: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    values = {key: options[key] for key in keys if key in options}
    if "vision_model" in options:
        values["model"] = options["vision_model"]
    return values


def _with_quality(asset: Asset | None, score: float | None) -> Asset | None:
    if asset is None or score is None or asset.quality_score is not None:
        return asset
    return asset.model_copy(update={"quality_score": score})


class VisualQAAttempt(BaseModel):
    """One generation, fallback or stock attempt and the cost it incurred.

    ``retry`` marks a second or later try against the same provider and
    ``fallback`` names the step that produced the attempt (``provider``,
    ``stock`` or ``image_to_video``). ``cost_usd`` is the amount recorded for
    this attempt alone, so rejected candidates are never lost from the total.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    index: int = Field(default=0, ge=0)
    kind: MediaKind = MediaKind.IMAGE
    provider: str = ""
    model: str = ""
    asset: Asset | None = None
    stock: StockAsset | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    passed: bool = False
    retry: bool = False
    fallback: str = ""
    error: str = ""
    cached: bool = False
    cost_usd: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def amount_usd(self) -> float:
        """Alias for :attr:`cost_usd`."""
        return self.cost_usd

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for the manifest."""
        return {
            "index": self.index,
            "kind": self.kind.value,
            "provider": self.provider,
            "model": self.model,
            "asset": (
                self.asset.model_dump(mode="json") if self.asset is not None else None
            ),
            "stock": (
                self.stock.model_dump(mode="json") if self.stock is not None else None
            ),
            "score": self.score,
            "passed": self.passed,
            "retry": self.retry,
            "fallback": self.fallback,
            "error": self.error,
            "cached": self.cached,
            "cost_usd": self.cost_usd,
            "metadata": dict(self.metadata),
        }


class VisualQAResult(BaseModel):
    """The accepted asset for one beat together with every quality attempt.

    ``cost_usd`` is the total spent across every attempt, not only the accepted
    one. ``attempts`` preserves the order the media layer tried them in.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    index: int = Field(default=0, ge=0)
    request: VisualRequest | None = None
    kind: MediaKind = MediaKind.IMAGE
    asset: Asset | None = None
    stock: StockAsset | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    passed: bool = False
    provider: str = ""
    model: str = ""
    fallback: str = ""
    attempts: list[VisualQAAttempt] = Field(default_factory=list)
    cost_usd: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    qa: Any = Field(default=None, exclude=True)

    @property
    def resolved(self) -> bool:
        """Whether an asset was accepted."""
        return (self.asset is not None or self.stock is not None) and self.passed

    @property
    def failed(self) -> bool:
        """Whether no attempt was accepted."""
        return not self.passed

    @property
    def total_cost_usd(self) -> float:
        """The total cost of every attempt."""
        return self.cost_usd

    @property
    def retries(self) -> int:
        """How many attempts followed the first one."""
        return max(0, len(self.attempts) - 1)

    @property
    def used_fallback(self) -> bool:
        """Whether the accepted attempt came from a fallback."""
        return bool(self.fallback and self.fallback != "exhausted")

    @property
    def attempt_costs(self) -> list[float]:
        """The cost of every attempt in order."""
        return [attempt.cost_usd for attempt in self.attempts]

    @property
    def prompt_hash(self) -> str:
        """The prompt hash of the accepted asset, or an empty string."""
        return self.asset.prompt_hash if self.asset is not None else ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for the manifest."""
        return {
            "index": self.index,
            "request": (self.request.to_dict() if self.request is not None else None),
            "kind": self.kind.value,
            "asset": (
                self.asset.model_dump(mode="json") if self.asset is not None else None
            ),
            "stock": (
                self.stock.model_dump(mode="json") if self.stock is not None else None
            ),
            "score": self.score,
            "passed": self.passed,
            "provider": self.provider,
            "model": self.model,
            "fallback": self.fallback,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "cost_usd": self.cost_usd,
            "metadata": dict(self.metadata),
        }


VisualQABeat = VisualQAResult
ResolvedVisualQA = VisualQAResult


def _from_image_qa(
    result: ImageQAResult,
    request: VisualRequest,
    metadata: Mapping[str, Any],
) -> VisualQAResult:
    attempts = [
        VisualQAAttempt(
            index=index,
            kind=MediaKind.IMAGE,
            provider=attempt.provider,
            model=_model_of(attempt.outcome),
            asset=attempt.asset,
            stock=attempt.stock,
            score=_score_number(attempt.score),
            passed=attempt.passed,
            retry=attempt.retry,
            fallback=attempt.fallback,
            error=attempt.error,
            cached=_cached_of(attempt.outcome),
            cost_usd=attempt.amount_usd,
        )
        for index, attempt in enumerate(result.attempts)
    ]
    score = _score_number(result.score)
    return VisualQAResult(
        index=request.index,
        request=request,
        kind=MediaKind.IMAGE,
        asset=_with_quality(result.asset, score),
        stock=result.stock,
        score=score,
        passed=result.passed,
        provider=result.provider,
        model=_model_of(result.outcome),
        fallback=result.fallback,
        attempts=attempts,
        cost_usd=result.total_amount_usd,
        metadata=dict(metadata),
        qa=result,
    )


def _from_video_qa(
    result: VideoQAResult,
    request: VisualRequest,
    metadata: Mapping[str, Any],
) -> VisualQAResult:
    attempts: list[VisualQAAttempt] = []
    for index, attempt in enumerate(result.attempts):
        outcome = attempt.outcome if attempt.outcome is not None else attempt.i2v
        attempts.append(
            VisualQAAttempt(
                index=index,
                kind=MediaKind.VIDEO,
                provider=attempt.provider,
                model=_model_of(outcome),
                asset=attempt.asset,
                score=_score_number(attempt.score),
                passed=attempt.passed,
                retry=attempt.retry,
                fallback=attempt.fallback,
                error=attempt.error,
                cached=_cached_of(outcome),
                cost_usd=attempt.amount_usd,
            )
        )
    score = _score_number(result.score)
    accepted = result.outcome if result.outcome is not None else result.i2v
    return VisualQAResult(
        index=request.index,
        request=request,
        kind=MediaKind.VIDEO,
        asset=_with_quality(result.asset, score),
        score=score,
        passed=result.passed,
        provider=result.provider,
        model=_model_of(accepted),
        fallback=result.fallback,
        attempts=attempts,
        cost_usd=result.total_amount_usd,
        metadata=dict(metadata),
        qa=result,
    )


async def _call_qa_runner(
    runner: Any,
    request: Any,
    *,
    run_id: str,
    metadata: Mapping[str, Any] | None,
    token: Any,
) -> Any:
    method = getattr(runner, "run", None)
    function = method if callable(method) else runner
    if not callable(function):
        raise ConfigError(
            "quality runner must be callable or expose run()",
            details={"value": type(runner).__name__},
        )
    values: dict[str, Any] = {"run_id": run_id, "metadata": metadata, "token": token}
    values = {name: value for name, value in values.items() if value is not None}
    kwargs = _supported_kwargs(function, values)
    result = function(request, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


def _image_target(target: Any) -> tuple[Any, Any]:
    if target is None:
        return None, None
    service = getattr(target, "image_service", None)
    if service is not None and callable(getattr(service, "generate", None)):
        target = service
    if callable(getattr(target, "generate", None)):
        return target, None
    image_method = getattr(target, "image", None)
    if callable(image_method):

        async def generate(request: Any, **values: Any) -> Any:
            result = image_method(request, **values)
            if inspect.isawaitable(result):
                result = await result
            return getattr(result, "outcome", result)

        return None, generate
    if callable(target):
        return None, target
    raise ConfigError(
        "image quality checks need a service exposing generate() or image()",
        details={"value": type(target).__name__},
    )


def _video_target(target: Any) -> tuple[Any, Any]:
    if target is None:
        return None, None
    service = getattr(target, "video_service", None)
    if service is not None and callable(getattr(service, "generate", None)):
        target = service
    if callable(getattr(target, "generate", None)):
        return target, None
    video_method = getattr(target, "video", None)
    if callable(video_method):

        async def generate(request: Any, **values: Any) -> Any:
            result = video_method(request, **values)
            if inspect.isawaitable(result):
                result = await result
            return getattr(result, "outcome", result)

        return None, generate
    if callable(target):
        return None, target
    raise ConfigError(
        "video quality checks need a service exposing generate() or video()",
        details={"value": type(target).__name__},
    )


def _context_service(context: StageContext | None) -> Any:
    if context is None:
        return None
    try:
        return context.media
    except ValueError:
        return None


class VisualQA:
    """Run the image or video quality gate for one visual request.

    ``image``/``video`` may be the media services, a media facade or a router.
    ``image_qa``/``video_qa`` accept a ready quality assurance object or an
    async runner, which keeps tests independent of a provider. The remaining
    keyword options are split by the request builder and the quality gate, so
    the same object can carry generation and scoring defaults.
    """

    def __init__(
        self,
        *,
        image: Any = None,
        video: Any = None,
        image_qa: Any = None,
        video_qa: Any = None,
        config: Any = None,
        **options: Any,
    ) -> None:
        self.image = image
        self.video = video
        self.image_qa = image_qa
        self.video_qa = video_qa
        self.config = config
        self.options = dict(options)

    def _options(self, overrides: Mapping[str, Any]) -> dict[str, Any]:
        options = dict(self.options)
        options.update(overrides)
        if self.config is not None:
            options.setdefault("config", self.config)
        return options

    def _image_service(
        self,
        options: Mapping[str, Any],
        context: StageContext | None,
    ) -> Any:
        service = options.get("image", options.get("image_service", self.image))
        return service if service is not None else _context_service(context)

    def _video_service(
        self,
        options: Mapping[str, Any],
        context: StageContext | None,
    ) -> Any:
        service = options.get("video", options.get("video_service", self.video))
        return service if service is not None else _context_service(context)

    async def _run_image(
        self,
        request: VisualRequest,
        *,
        context: StageContext | None,
        run_id: str,
        metadata: Mapping[str, Any] | None,
        token: Any,
        options: Mapping[str, Any],
    ) -> VisualQAResult:
        active_meta = dict(metadata or {})
        values = _request_values(
            options,
            _IMAGE_REQUEST_KEYS,
            rename={"image_quality": "quality", "format": "fmt"},
        )
        image_request = _image_request(request, metadata=active_meta, **values)
        if image_request is None:
            return VisualQAResult(
                index=request.index,
                request=request,
                kind=MediaKind.IMAGE,
                passed=False,
                metadata={**active_meta, "reason": "no prompt"},
            )

        runner = options.get("qa", self.image_qa)
        if runner is not None:
            result = await _call_qa_runner(
                runner,
                image_request,
                run_id=run_id,
                metadata=active_meta,
                token=token,
            )
            if isinstance(result, VisualQAResult):
                return result
            if not isinstance(result, ImageQAResult):
                raise ConfigError(
                    "image quality runner must return an ImageQAResult",
                    details={"value": type(result).__name__},
                )
            return _from_image_qa(result, request, active_meta)

        service = self._image_service(options, context)
        if service is None:
            raise ConfigError(
                "image quality checks need an image service or a QA runner",
                details={"stage": VISUAL_QA_STAGE_NAME},
            )
        qa_service, qa_generate = _image_target(service)
        qa_values = _qa_values(options, _IMAGE_QA_KEYS)
        if qa_generate is not None:
            qa_values["generate"] = qa_generate
        result = await run_image_qa(
            image_request,
            qa_service,
            token=token,
            run_id=run_id,
            metadata=active_meta,
            **qa_values,
        )
        return _from_image_qa(result, request, active_meta)

    async def _run_video(
        self,
        request: VisualRequest,
        *,
        context: StageContext | None,
        run_id: str,
        metadata: Mapping[str, Any] | None,
        token: Any,
        options: Mapping[str, Any],
    ) -> VisualQAResult:
        active_meta = dict(metadata or {})
        values = _request_values(
            options,
            _VIDEO_REQUEST_KEYS,
            rename={"format": "fmt"},
        )
        catalog = _platform_catalog(values.pop("platforms", None))
        platform = _resolve_platform(values.pop("platform", None), catalog=catalog)
        if "max_clip_seconds" not in values:
            cap = _config_cap(options.get("config"))
            if cap is not None:
                values["max_clip_seconds"] = cap
        video_request, info = _video_plan(
            request,
            metadata=active_meta,
            platform=platform,
            **values,
        )
        if video_request is None:
            return VisualQAResult(
                index=request.index,
                request=request,
                kind=MediaKind.VIDEO,
                passed=False,
                metadata={**active_meta, **info, "reason": "no prompt"},
            )

        runner = options.get("qa", self.video_qa)
        if runner is not None:
            result = await _call_qa_runner(
                runner,
                video_request,
                run_id=run_id,
                metadata=active_meta,
                token=token,
            )
            if isinstance(result, VisualQAResult):
                return result
            if not isinstance(result, VideoQAResult):
                raise ConfigError(
                    "video quality runner must return a VideoQAResult",
                    details={"value": type(result).__name__},
                )
            return _from_video_qa(result, request, {**active_meta, **info})

        service = self._video_service(options, context)
        if service is None:
            raise ConfigError(
                "video quality checks need a video service or a QA runner",
                details={"stage": VISUAL_QA_STAGE_NAME},
            )
        qa_service, qa_generate = _video_target(service)
        qa_values = _qa_values(options, _VIDEO_QA_KEYS)
        if qa_generate is not None:
            qa_values["generate"] = qa_generate
        result = await run_video_qa(
            video_request,
            qa_service,
            token=token,
            run_id=run_id,
            metadata=active_meta,
            **qa_values,
        )
        return _from_video_qa(result, request, {**active_meta, **info})

    async def run(
        self,
        value: Any,
        *,
        context: StageContext | None = None,
        run_id: str = "",
        metadata: Mapping[str, Any] | None = None,
        token: Any = None,
        **overrides: Any,
    ) -> VisualQAResult:
        """Run the matching quality gate for one visual request."""
        request = _coerce_request(value)
        options = self._options(overrides)
        if context is not None:
            options.setdefault("config", context.config)
        kind = _kind_of(request)
        if request.visual_type not in _SUPPORTED_TYPES:
            if request.visual_type is VisualType.NONE:
                reason = "no visual requested"
            else:
                reason = "stock quality is handled by the stock stage"
            return VisualQAResult(
                index=request.index,
                request=request,
                kind=kind,
                passed=False,
                metadata={**dict(metadata or {}), "reason": reason},
            )
        if kind is MediaKind.VIDEO:
            return await self._run_video(
                request,
                context=context,
                run_id=run_id,
                metadata=metadata,
                token=token,
                options=options,
            )
        return await self._run_image(
            request,
            context=context,
            run_id=run_id,
            metadata=metadata,
            token=token,
            options=options,
        )

    __call__ = run


VisualQAService = VisualQA


async def resolve_image_qa(
    value: Any,
    *,
    service: Any = None,
    image: Any = None,
    image_service: Any = None,
    qa: Any = None,
    image_qa: Any = None,
    context: StageContext | None = None,
    run_id: str = "",
    metadata: Mapping[str, Any] | None = None,
    token: Any = None,
    **options: Any,
) -> VisualQAResult:
    """Run the image quality gate for one image beat."""
    target = service if service is not None else image
    if target is None:
        target = image_service
    active = VisualQA(
        image=target,
        image_qa=qa if qa is not None else image_qa,
        **options,
    )
    request = _coerce_request(value)
    return await active.run(
        request,
        context=context,
        run_id=run_id,
        metadata=metadata,
        token=token,
    )


async def resolve_visual_qa(
    value: Any,
    *,
    image: Any = None,
    video: Any = None,
    image_qa: Any = None,
    video_qa: Any = None,
    config: Any = None,
    context: StageContext | None = None,
    run_id: str = "",
    metadata: Mapping[str, Any] | None = None,
    token: Any = None,
    **options: Any,
) -> VisualQAResult:
    """Run the matching quality gate for one visual request."""
    active = VisualQA(
        image=image,
        video=video,
        image_qa=image_qa,
        video_qa=video_qa,
        config=config,
        **options,
    )
    return await active.run(
        value,
        context=context,
        run_id=run_id,
        metadata=metadata,
        token=token,
    )


def _run_sync[T](coroutine: Coroutine[Any, Any, T]) -> T:
    import asyncio

    return asyncio.run(coroutine)


def resolve_visual_qa_sync(value: Any, **kwargs: Any) -> VisualQAResult:
    """Run :func:`resolve_visual_qa` from synchronous code."""
    return _run_sync(resolve_visual_qa(value, **kwargs))


run_visual_qa = resolve_visual_qa
run_visual_qa_sync = resolve_visual_qa_sync


async def resolve_visual_qa_requests(
    values: Any,
    **kwargs: Any,
) -> list[VisualQAResult]:
    """Run the quality gate for every image and video request in *values*."""
    if isinstance(values, VisualRequest):
        requests = [values]
    elif isinstance(values, Mapping):
        requests = [_coerce_request(values)]
    elif isinstance(values, Sequence) and not isinstance(
        values,
        (str, bytes, bytearray),
    ):
        requests = [_coerce_request(item) for item in values]
    else:
        requests = resolve_visual_requests(values)
    results: list[VisualQAResult] = []
    for position, request in enumerate(requests):
        if request.visual_type not in _SUPPORTED_TYPES:
            continue
        result = await resolve_visual_qa(request, **kwargs)
        if result.index != position and "index" not in request.model_fields_set:
            result = result.model_copy(update={"index": position})
        results.append(result)
    return results


class VisualQAStageResult(StageResult):
    """A stage result carrying one :class:`VisualQAResult` per checked beat."""

    beats: list[VisualQAResult] = Field(
        default_factory=list,
        validation_alias=AliasChoices("beats", "results", "items", "selections"),
    )

    @property
    def results(self) -> list[VisualQAResult]:
        """Alias for :attr:`beats`."""
        return self.beats

    @property
    def assets(self) -> list[Asset]:
        """Every accepted generated asset in beat order."""
        return [beat.asset for beat in self.beats if beat.asset is not None]

    @property
    def stock_assets(self) -> list[StockAsset]:
        """Every accepted stock asset in beat order."""
        return [beat.stock for beat in self.beats if beat.stock is not None]

    @property
    def passed(self) -> list[VisualQAResult]:
        """Beats for which an asset passed the quality gate."""
        return [beat for beat in self.beats if beat.passed]

    @property
    def unresolved(self) -> list[VisualQAResult]:
        """Beats for which no attempt passed the quality gate."""
        return [beat for beat in self.beats if not beat.passed]

    @property
    def attempts(self) -> list[VisualQAAttempt]:
        """Every attempt across every beat, in beat order."""
        return [attempt for beat in self.beats for attempt in beat.attempts]

    @property
    def total_cost_usd(self) -> float:
        """The total cost of every attempt."""
        return sum(beat.cost_usd for beat in self.beats)


class VisualQAStage(Stage):
    """Run the shared quality gate for every image and video beat.

    ``requests`` may be a script, a beat sequence, one or more visual requests
    or mappings. ``image``/``video`` supply the media services; without one the
    stage uses ``context.media``. ``image_qa``/``video_qa`` inject a ready
    quality assurance object or runner. Stock beats are ignored because the
    stock stage applies relevance scoring instead.
    """

    name = VISUAL_QA_STAGE_NAME
    artifacts: tuple[str, ...] = (DEFAULT_VISUAL_QA_FILENAME,)

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
        image: Any = None,
        image_service: Any = None,
        video: Any = None,
        video_service: Any = None,
        image_qa: Any = None,
        video_qa: Any = None,
        config: Any = None,
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
            "image": image,
            "image_service": image_service,
            "video": video,
            "video_service": video_service,
            "image_qa": image_qa,
            "video_qa": video_qa,
            "config": config,
            "filename": filename,
            "manifest_filename": manifest_filename,
        }
        for key, value in supplied.items():
            if value is not None:
                merged[key] = value
        self.filename = _text(
            merged.pop("filename", None)
            or merged.pop("manifest_filename", None)
            or DEFAULT_VISUAL_QA_FILENAME
        )
        if not self.filename:
            self.filename = DEFAULT_VISUAL_QA_FILENAME
        self.artifacts = (self.filename,)

        self._requests_input = merged.pop("requests", None)
        if self._requests_input is None:
            self._requests_input = merged.pop("visuals", None)
        self._script = merged.pop("script", None)
        self._beats = merged.pop("beats", None)
        self._image = merged.pop("image", None)
        if self._image is None:
            self._image = merged.pop("image_service", None)
        self._video = merged.pop("video", None)
        if self._video is None:
            self._video = merged.pop("video_service", None)
        self._image_qa = merged.pop("image_qa", None)
        self._video_qa = merged.pop("video_qa", None)
        self._config = merged.pop("config", None)
        merged.update(kwargs)
        self.results: list[VisualQAResult] = []
        super().__init__(name=name, options=merged)

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
            value,
            (str, bytes, bytearray),
        ):
            return [_coerce_request(item) for item in value]
        return resolve_visual_requests(
            value,
            strategy=self.options.get("strategy", DEFAULT_VISUAL_STRATEGY),
            aspect_ratio=_text(self.options.get("aspect_ratio")),
            duration_seconds=self.options.get("duration_seconds"),
        )

    def _quality(self, context: StageContext) -> VisualQA:
        return VisualQA(
            image=self._image,
            video=self._video,
            image_qa=self._image_qa,
            video_qa=self._video_qa,
            config=self._config if self._config is not None else context.config,
            **self.options,
        )

    async def run(self, context: StageContext) -> StageResult:
        """Check every image and video beat and write the quality manifest."""
        quality = self._quality(context)
        run_id = context.run.id if context.run is not None else ""
        requests = [
            request
            for request in self._requests()
            if request.visual_type in _SUPPORTED_TYPES
        ]
        results: list[VisualQAResult] = []
        for position, request in enumerate(requests):
            result = await quality.run(
                request,
                context=context,
                run_id=run_id,
                metadata={"position": position},
            )
            results.append(result)
        self.results = results

        passed = sum(1 for result in results if result.passed)
        attempts = sum(len(result.attempts) for result in results)
        retries = sum(result.retries for result in results)
        total_cost = sum(result.cost_usd for result in results)
        providers = [result.provider for result in results if result.provider]
        models = [result.model for result in results if result.model]
        metadata = {
            "beats": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "attempts": attempts,
            "retries": retries,
            "cost_usd": total_cost,
            "providers": list(dict.fromkeys(providers)),
            "models": list(dict.fromkeys(models)),
            "attempt_costs": [
                attempt.cost_usd for result in results for attempt in result.attempts
            ],
        }
        payload: dict[str, Any] = {
            "version": 1,
            "stage": self.name,
            "beats": [result.to_dict() for result in results],
            "metadata": metadata,
        }
        relative = self.filename
        _write_json(context.run_dir / relative, payload)
        return VisualQAStageResult(
            run_id=run_id,
            stage=self.name,
            status=StageState.SUCCEEDED,
            artifacts=[relative],
            cost_usd=total_cost,
            beats=results,
            metadata={**metadata, "manifest": relative},
        )


visual_qa_stage = VisualQAStage
"""Alias for :class:`VisualQAStage`."""


def build_visual_qa_stage(
    *,
    options: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> VisualQAStage:
    """Build a :class:`VisualQAStage` from recipe options and overrides."""
    return VisualQAStage(options=options, **kwargs)


async def run_visual_qa_stage(
    context: StageContext,
    *,
    stage: VisualQAStage | None = None,
    **options: Any,
) -> StageResult:
    """Build a stage unless one is supplied, then run it asynchronously."""
    if stage is not None and options:
        raise ConfigError("pass either a stage or options, not both")
    active = stage if stage is not None else VisualQAStage(**options)
    return await active.run(context)

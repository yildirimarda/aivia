"""Resolve each beat's visual strategy into one concrete deterministic request.

The writing stage assigns a visual type to every beat and the prompt builder
attaches one still and one motion prompt. The visual stages need a single
request per beat: which kind of asset to obtain, the exact prompt to send, the
aspect ratio and duration to honour, and for stock the search query.

Resolution is pure data transformation. It never reads a file, calls a
provider or draws a random number, so the same script, strategy and limit
always produce the same requests. The AI video cap is enforced by the shared
:class:`~aivia.writing.visual_assign.VisualAssigner`, which means a request
never asks for more generated video than the configured limit.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from aivia.core.models.base import canonical_json
from aivia.core.models.script import Beat, MediaKind, MediaPrompt, Script, VisualType
from aivia.writing.visual_assign import (
    DEFAULT_MAX_AI_VIDEO_BEATS,
    DEFAULT_VISUAL_STRATEGY,
    VisualAssigner,
    VisualStrategy,
    normalise_visual_strategy,
)

__all__ = [
    "DEFAULT_MAX_AI_VIDEO_BEATS",
    "DEFAULT_VISUAL_STRATEGY",
    "BeatVisualRequest",
    "ResolvedVisual",
    "VisualBeatRequest",
    "VisualDecision",
    "VisualPlan",
    "VisualPlanBuilder",
    "VisualRequest",
    "VisualRequestBuilder",
    "VisualRequestFactory",
    "VisualRequestPlanner",
    "VisualRequestResolver",
    "VisualResolver",
    "VisualStrategy",
    "VisualStrategyPlanner",
    "VisualStrategyRequest",
    "VisualStrategyResolver",
    "beat_visual_request",
    "build_visual_plan",
    "build_visual_request",
    "build_visual_requests",
    "concrete_visual_requests",
    "plan_visuals",
    "requests_for",
    "requests_for_script",
    "resolve_beat",
    "resolve_beat_strategy",
    "resolve_beat_visual",
    "resolve_script",
    "resolve_strategy",
    "resolve_visual_request",
    "resolve_visual_requests",
    "resolve_visual_strategy",
    "resolve_visuals",
    "visual_plan",
    "visual_requests",
    "visual_requests_for",
    "visual_requests_from_script",
]

_VISUAL_PROMPT_KINDS = (
    MediaKind.IMAGE,
    MediaKind.STILL,
    MediaKind.STOCK,
    MediaKind.MOTION,
    MediaKind.VIDEO,
)

_PROMPT_ORDER: dict[VisualType, tuple[MediaKind, ...]] = {
    VisualType.VIDEO: (
        MediaKind.MOTION,
        MediaKind.VIDEO,
        MediaKind.IMAGE,
        MediaKind.STILL,
        MediaKind.STOCK,
    ),
    VisualType.STOCK: (
        MediaKind.STOCK,
        MediaKind.IMAGE,
        MediaKind.STILL,
        MediaKind.MOTION,
        MediaKind.VIDEO,
    ),
    VisualType.IMAGE: (
        MediaKind.IMAGE,
        MediaKind.STILL,
        MediaKind.STOCK,
        MediaKind.MOTION,
        MediaKind.VIDEO,
    ),
    VisualType.STILL: (
        MediaKind.STILL,
        MediaKind.IMAGE,
        MediaKind.STOCK,
        MediaKind.MOTION,
        MediaKind.VIDEO,
    ),
    VisualType.TEXT: (
        MediaKind.IMAGE,
        MediaKind.STILL,
        MediaKind.STOCK,
        MediaKind.MOTION,
        MediaKind.VIDEO,
    ),
    VisualType.NONE: (),
}

_MEDIA_KIND: dict[VisualType, MediaKind] = {
    VisualType.IMAGE: MediaKind.IMAGE,
    VisualType.VIDEO: MediaKind.VIDEO,
    VisualType.STOCK: MediaKind.STOCK,
    VisualType.STILL: MediaKind.STILL,
    VisualType.TEXT: MediaKind.IMAGE,
    VisualType.NONE: MediaKind.IMAGE,
}


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _visual_type(value: Any) -> VisualType:
    if isinstance(value, VisualType):
        return value
    if isinstance(value, str):
        try:
            return VisualType(value.strip().lower())
        except ValueError:
            pass
    return VisualType.IMAGE


def _prompt_of(beat: Beat, visual_type: VisualType) -> MediaPrompt | None:
    for kind in _PROMPT_ORDER[visual_type]:
        for prompt in beat.media_prompts:
            if prompt.kind is kind:
                return prompt
    return None


def _visual_prompts(beat: Beat) -> list[MediaPrompt]:
    return [
        prompt
        for prompt in beat.media_prompts
        if prompt.kind in _VISUAL_PROMPT_KINDS
    ]


def _first_text(values: Sequence[str]) -> str:
    for value in values:
        text = _text(value)
        if text:
            return text
    return ""


def _first_number(values: Sequence[float | None]) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


class VisualRequest(BaseModel):
    """One concrete, provider agnostic request for a single beat.

    ``visual_type`` is the resolved source (image, video, stock, still, text or
    none) and ``kind`` is the media kind a provider will be asked for. ``query``
    is only meaningful for stock requests and otherwise stays empty. All values
    come from the beat or the caller, never from a provider or a random source.
    """

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        validate_assignment=True,
    )

    index: int = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices(
            "index",
            "beat_index",
            "order",
            "position",
            "number",
        ),
    )
    visual_type: VisualType = Field(
        default=VisualType.IMAGE,
        validation_alias=AliasChoices(
            "visual_type",
            "visual",
            "source_type",
            "visual_kind",
        ),
    )
    kind: MediaKind = Field(
        default=MediaKind.IMAGE,
        validation_alias=AliasChoices(
            "kind",
            "media_kind",
            "media_type",
            "target",
        ),
    )
    prompt: str = Field(
        default="",
        validation_alias=AliasChoices(
            "prompt",
            "text",
            "description",
            "instruction",
        ),
    )
    negative_prompt: str = Field(
        default="",
        validation_alias=AliasChoices(
            "negative_prompt",
            "negative",
            "avoid",
        ),
    )
    style: str = ""
    aspect_ratio: str = Field(
        default="",
        validation_alias=AliasChoices(
            "aspect_ratio",
            "aspect",
            "ratio",
        ),
    )
    duration_seconds: float | None = Field(
        default=None,
        gt=0,
        validation_alias=AliasChoices(
            "duration_seconds",
            "duration",
            "seconds",
            "length_seconds",
        ),
    )
    seed: int | None = None
    query: str = ""
    text: str = ""
    strategy: str = ""

    @field_validator("visual_type", mode="before")
    @classmethod
    def _validate_visual_type(cls, value: Any) -> Any:
        return _visual_type(value)

    @field_validator("kind", mode="before")
    @classmethod
    def _validate_kind(cls, value: Any) -> Any:
        if isinstance(value, MediaKind):
            return value
        if isinstance(value, str):
            text = value.strip().lower()
            try:
                return MediaKind(text)
            except ValueError:
                return MediaKind.IMAGE
        return value

    @field_validator(
        "prompt",
        "negative_prompt",
        "style",
        "aspect_ratio",
        "query",
        "text",
        "strategy",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _text(value)

    @property
    def beat(self) -> int:
        """Alias for :attr:`index`."""
        return self.index

    @property
    def beat_index(self) -> int:
        """Alias for :attr:`index`."""
        return self.index

    @property
    def visual(self) -> VisualType:
        """Alias for :attr:`visual_type`."""
        return self.visual_type

    @property
    def source(self) -> VisualType:
        """Alias for :attr:`visual_type`."""
        return self.visual_type

    @property
    def type(self) -> VisualType:
        """Alias for :attr:`visual_type`."""
        return self.visual_type

    @property
    def media_type(self) -> MediaKind:
        """Alias for :attr:`kind`."""
        return self.kind

    @property
    def target(self) -> MediaKind:
        """Alias for :attr:`kind`."""
        return self.kind

    @property
    def aspect(self) -> str:
        """Alias for :attr:`aspect_ratio`."""
        return self.aspect_ratio

    @property
    def duration(self) -> float | None:
        """Alias for :attr:`duration_seconds`."""
        return self.duration_seconds

    @property
    def negative(self) -> str:
        """Alias for :attr:`negative_prompt`."""
        return self.negative_prompt

    @property
    def is_video(self) -> bool:
        """Whether this request asks for video."""
        return self.visual_type is VisualType.VIDEO

    @property
    def is_image(self) -> bool:
        """Whether this request asks for a generated image."""
        return self.visual_type is VisualType.IMAGE

    @property
    def is_stock(self) -> bool:
        """Whether this request searches a stock library."""
        return self.visual_type is VisualType.STOCK

    @property
    def is_ai(self) -> bool:
        """Whether this request needs a paid generation call."""
        return self.visual_type in {VisualType.IMAGE, VisualType.VIDEO}

    @property
    def is_free(self) -> bool:
        """Whether this request needs no paid generation call."""
        return not self.is_ai

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for logs and manifests."""
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        """Return the request as canonical JSON."""
        return canonical_json(self.to_dict())


BeatVisualRequest = VisualRequest
VisualBeatRequest = VisualRequest
VisualDecision = VisualRequest
VisualStrategyRequest = VisualRequest
ResolvedVisual = VisualRequest


class VisualPlan(list[VisualRequest]):
    """An ordered list of requests that also records how it was resolved."""

    def __init__(
        self,
        requests: Sequence[VisualRequest] = (),
        *,
        strategy: VisualStrategy | str = DEFAULT_VISUAL_STRATEGY,
        max_ai_video_beats: int = DEFAULT_MAX_AI_VIDEO_BEATS,
    ) -> None:
        super().__init__(requests)
        self.strategy = normalise_visual_strategy(strategy)
        self.max_ai_video_beats = max_ai_video_beats

    @property
    def requests(self) -> list[VisualRequest]:
        """Return the requests as a plain list."""
        return list(self)


def _beat_index(beat: Beat, position: int) -> int:
    if "index" in beat.model_fields_set and beat.index is not None:
        return beat.index
    return position


def _request_for(
    beat: Beat,
    *,
    position: int,
    strategy: VisualStrategy,
    aspect_ratio: str,
    duration_seconds: float | None,
) -> VisualRequest:
    visual_type = _visual_type(beat.visual_type)
    prompt = _prompt_of(beat, visual_type)
    prompts = _visual_prompts(beat)
    fallback = _text(beat.text)

    if visual_type is VisualType.NONE:
        prompt_text = ""
    elif prompt is not None:
        prompt_text = prompt.prompt
    else:
        prompt_text = fallback

    query = (
        (prompt_text or fallback)
        if visual_type is VisualType.STOCK
        else ""
    )

    chosen_aspect = _first_text(
        [
            prompt.aspect_ratio if prompt is not None else "",
            *(item.aspect_ratio for item in prompts),
            aspect_ratio,
        ]
    )
    chosen_duration = _first_number(
        [
            beat.duration_seconds,
            prompt.duration_seconds if prompt is not None else None,
            *(item.duration_seconds for item in prompts),
            duration_seconds,
        ]
    )
    chosen_seed = prompt.seed if prompt is not None else None
    if chosen_seed is None:
        chosen_seed = next(
            (item.seed for item in prompts if item.seed is not None),
            None,
        )

    return VisualRequest(
        index=_beat_index(beat, position),
        visual_type=visual_type,
        kind=_MEDIA_KIND[visual_type],
        prompt=prompt_text,
        negative_prompt=prompt.negative_prompt if prompt is not None else "",
        style=prompt.style if prompt is not None else "",
        aspect_ratio=chosen_aspect,
        duration_seconds=chosen_duration,
        seed=chosen_seed,
        query=query,
        text=fallback,
        strategy=strategy.value,
    )


class VisualStrategyResolver:
    """Turn a script, beat or beat sequence into concrete visual requests.

    ``strategy`` and ``max_ai_video_beats`` follow the same rules as
    :class:`~aivia.writing.visual_assign.VisualAssigner`. ``aspect_ratio`` and
    ``duration_seconds`` are only fallbacks for beats and prompts that do not
    carry their own value, so a caller can apply a platform preset without
    overwriting beat level intent.
    """

    def __init__(
        self,
        strategy: VisualStrategy | str = DEFAULT_VISUAL_STRATEGY,
        max_ai_video_beats: int = DEFAULT_MAX_AI_VIDEO_BEATS,
        *,
        aspect_ratio: str = "",
        duration_seconds: float | None = None,
    ) -> None:
        self.assigner = VisualAssigner(
            strategy=strategy,
            max_ai_video_beats=max_ai_video_beats,
        )
        self.strategy = self.assigner.strategy
        self.max_ai_video_beats = self.assigner.max_ai_video_beats
        self.aspect_ratio = _text(aspect_ratio)
        self.duration_seconds = duration_seconds

    def requests_for(self, script: Script) -> list[VisualRequest]:
        """Return one request per beat in *script* without changing it."""
        return self.resolve_beats(script.beats)

    def resolve_beats(self, beats: Sequence[Beat]) -> list[VisualRequest]:
        """Return one request per beat, applying the cap across the sequence."""
        assigned = self.assigner.assign_beats(beats)
        return [
            _request_for(
                beat,
                position=position,
                strategy=self.strategy,
                aspect_ratio=self.aspect_ratio,
                duration_seconds=self.duration_seconds,
            )
            for position, beat in enumerate(assigned)
        ]

    def resolve_beat(self, beat: Beat) -> VisualRequest:
        """Return the request for one *beat*."""
        return self.resolve_beats([beat])[0]

    def resolve_script(self, script: Script) -> list[VisualRequest]:
        """Alias for :meth:`requests_for`."""
        return self.requests_for(script)

    def types_for(self, value: Script | Sequence[Beat]) -> list[str]:
        """Return the resolved visual type value of each beat."""
        if isinstance(value, Script):
            return self.assigner.types_for(value)
        return [visual.value for visual in self.assigner._types_for(value)]

    def resolve(
        self,
        value: Script | Beat | Sequence[Beat],
    ) -> VisualRequest | list[VisualRequest]:
        """Resolve a script or sequence to requests, or one beat to a request."""
        if isinstance(value, Script):
            return self.requests_for(value)
        if isinstance(value, Beat):
            return self.resolve_beat(value)
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            return self.resolve_beats(value)
        raise TypeError(
            "visual resolution needs a script, a beat or a sequence of beats"
        )

    def plan(self, value: Script | Sequence[Beat]) -> VisualPlan:
        """Return a :class:`VisualPlan` with the same requests."""
        if isinstance(value, Script):
            requests = self.requests_for(value)
        elif isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            requests = self.resolve_beats(value)
        else:
            raise TypeError("a visual plan needs a script or a sequence of beats")
        return VisualPlan(
            requests,
            strategy=self.strategy,
            max_ai_video_beats=self.max_ai_video_beats,
        )

    build = resolve
    generate = resolve
    apply = resolve
    __call__ = resolve
    resolve_script = resolve_script
    resolve_all = resolve


VisualResolver = VisualStrategyResolver
VisualRequestResolver = VisualStrategyResolver
VisualRequestBuilder = VisualStrategyResolver
VisualRequestFactory = VisualStrategyResolver
VisualRequestPlanner = VisualStrategyResolver
VisualStrategyPlanner = VisualStrategyResolver
VisualPlanBuilder = VisualStrategyResolver


def resolve_visual_request(
    beat: Beat,
    *,
    strategy: VisualStrategy | str = DEFAULT_VISUAL_STRATEGY,
    max_ai_video_beats: int = DEFAULT_MAX_AI_VIDEO_BEATS,
    aspect_ratio: str = "",
    duration_seconds: float | None = None,
) -> VisualRequest:
    """Resolve one *beat* into a concrete visual request."""
    return VisualStrategyResolver(
        strategy=strategy,
        max_ai_video_beats=max_ai_video_beats,
        aspect_ratio=aspect_ratio,
        duration_seconds=duration_seconds,
    ).resolve_beat(beat)


def resolve_visual_requests(
    value: Script | Beat | Sequence[Beat],
    *,
    strategy: VisualStrategy | str = DEFAULT_VISUAL_STRATEGY,
    max_ai_video_beats: int = DEFAULT_MAX_AI_VIDEO_BEATS,
    aspect_ratio: str = "",
    duration_seconds: float | None = None,
) -> list[VisualRequest]:
    """Resolve a script, beat or sequence into a list of visual requests."""
    resolver = VisualStrategyResolver(
        strategy=strategy,
        max_ai_video_beats=max_ai_video_beats,
        aspect_ratio=aspect_ratio,
        duration_seconds=duration_seconds,
    )
    if isinstance(value, Script):
        return resolver.requests_for(value)
    if isinstance(value, Beat):
        return [resolver.resolve_beat(value)]
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return resolver.resolve_beats(value)
    raise TypeError("visual resolution needs a script, a beat or a sequence of beats")


def resolve_visual_strategy(
    value: Script | Beat | Sequence[Beat],
    *,
    strategy: VisualStrategy | str = DEFAULT_VISUAL_STRATEGY,
    max_ai_video_beats: int = DEFAULT_MAX_AI_VIDEO_BEATS,
    aspect_ratio: str = "",
    duration_seconds: float | None = None,
) -> VisualRequest | list[VisualRequest]:
    """Resolve a beat to one request, or a script/sequence to a request list."""
    resolver = VisualStrategyResolver(
        strategy=strategy,
        max_ai_video_beats=max_ai_video_beats,
        aspect_ratio=aspect_ratio,
        duration_seconds=duration_seconds,
    )
    return resolver.resolve(value)


def build_visual_plan(
    value: Script | Sequence[Beat],
    *,
    strategy: VisualStrategy | str = DEFAULT_VISUAL_STRATEGY,
    max_ai_video_beats: int = DEFAULT_MAX_AI_VIDEO_BEATS,
    aspect_ratio: str = "",
    duration_seconds: float | None = None,
) -> VisualPlan:
    """Resolve *value* into a :class:`VisualPlan`."""
    return VisualStrategyResolver(
        strategy=strategy,
        max_ai_video_beats=max_ai_video_beats,
        aspect_ratio=aspect_ratio,
        duration_seconds=duration_seconds,
    ).plan(value)


build_visual_request = resolve_visual_request
build_visual_requests = resolve_visual_requests
resolve_beat = resolve_visual_request
resolve_beat_visual = resolve_visual_request
resolve_beat_strategy = resolve_visual_request
resolve_script = resolve_visual_requests
resolve_strategy = resolve_visual_strategy
beat_visual_request = resolve_visual_request
concrete_visual_requests = resolve_visual_requests
resolve_visuals = resolve_visual_requests
visual_requests = resolve_visual_requests
visual_requests_for = resolve_visual_requests
visual_requests_from_script = resolve_visual_requests
requests_for = resolve_visual_requests
requests_for_script = resolve_visual_requests
plan_visuals = build_visual_plan
visual_plan = build_visual_plan

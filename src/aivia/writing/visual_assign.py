"""Assign a visual source type to every script beat, capped for AI video.

The writing stage produces ordered beats, but the visual stage needs to know
what each beat should become before it can request an asset. This module makes
that decision deterministically from a strategy and a hard limit on the number
of AI generated video beats.

``hybrid`` is the default: the first beats, up to the configured limit, become
AI video and the rest become AI images. ``video`` follows the same rule so the
limit can never be exceeded, while ``image``, ``stock``, ``still``, ``text`` and
``none`` assign that source to every beat. ``existing`` keeps a visual type
already present on a beat, fills blanks with an image and still downgrades any
video beyond the limit. No provider is called and no model is involved, so the
same script, strategy and limit always produce the same assignment.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Any, overload

from aivia.core.config.writing import (
    DEFAULT_MAX_AI_VIDEO_BEATS,
    DEFAULT_VISUAL_STRATEGY,
)
from aivia.core.models.script import Beat, Script, VisualType

__all__ = [
    "DEFAULT_MAX_AI_VIDEO_BEATS",
    "DEFAULT_VISUAL_STRATEGY",
    "VisualAssigner",
    "VisualStrategy",
    "assign_beat_visual",
    "assign_visual_types",
    "assign_visuals",
    "build_visual_assignment",
    "normalise_visual_strategy",
    "normalize_visual_strategy",
    "visual_types",
    "visual_types_for",
]

class VisualStrategy(StrEnum):
    """The ways a script's beats can be assigned a visual source."""

    HYBRID = "hybrid"
    IMAGE = "image"
    VIDEO = "video"
    STOCK = "stock"
    STILL = "still"
    TEXT = "text"
    NONE = "none"
    EXISTING = "existing"


_STRATEGY_ALIASES: dict[str, VisualStrategy] = {
    "ai_image": VisualStrategy.IMAGE,
    "ai_video": VisualStrategy.VIDEO,
    "assets": VisualStrategy.EXISTING,
    "auto": VisualStrategy.HYBRID,
    "existing_assets": VisualStrategy.EXISTING,
    "image_gen": VisualStrategy.IMAGE,
    "images": VisualStrategy.IMAGE,
    "keep": VisualStrategy.EXISTING,
    "mix": VisualStrategy.HYBRID,
    "mixed": VisualStrategy.HYBRID,
    "provided": VisualStrategy.EXISTING,
    "provided_assets": VisualStrategy.EXISTING,
    "video_gen": VisualStrategy.VIDEO,
    "videos": VisualStrategy.VIDEO,
}

_VISUAL_TYPE_ALIASES: dict[str, VisualType] = {
    "ai_image": VisualType.IMAGE,
    "ai_video": VisualType.VIDEO,
    "images": VisualType.IMAGE,
    "videos": VisualType.VIDEO,
}

_STRATEGY_TYPES: dict[VisualStrategy, VisualType] = {
    VisualStrategy.IMAGE: VisualType.IMAGE,
    VisualStrategy.STOCK: VisualType.STOCK,
    VisualStrategy.STILL: VisualType.STILL,
    VisualStrategy.TEXT: VisualType.TEXT,
    VisualStrategy.NONE: VisualType.NONE,
}


def _normalise_name(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("visual strategy must be a string")
    text = value.strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def normalise_visual_strategy(value: Any = None) -> VisualStrategy:
    """Return *value* as a :class:`VisualStrategy`.

    An empty value means the default. Common aliases such as ``ai_video`` or
    ``provided`` are accepted so callers can use the vocabulary of the config
    and estimate layers. Anything else fails with the allowed names.
    """
    if value is None or value == "":
        return VisualStrategy(DEFAULT_VISUAL_STRATEGY)
    if isinstance(value, VisualStrategy):
        return value
    name = _normalise_name(value)
    if name in _STRATEGY_ALIASES:
        return _STRATEGY_ALIASES[name]
    try:
        return VisualStrategy(name)
    except ValueError:
        allowed = ", ".join(strategy.value for strategy in VisualStrategy)
        raise ValueError(
            f"unknown visual strategy: {value!r} (allowed: {allowed})"
        ) from None


normalize_visual_strategy = normalise_visual_strategy


def _validate_limit(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("max_ai_video_beats must be an integer")
    if value < 0:
        raise ValueError("max_ai_video_beats must not be negative")
    return value


def _normalise_visual_type(value: Any) -> VisualType | None:
    if isinstance(value, VisualType):
        return value
    if not isinstance(value, str):
        return None
    name = _normalise_name(value)
    if not name:
        return None
    if name in _VISUAL_TYPE_ALIASES:
        return _VISUAL_TYPE_ALIASES[name]
    try:
        return VisualType(name)
    except ValueError:
        return None


class VisualAssigner:
    """Assign a visual type to each beat under a hard AI video limit.

    ``strategy`` accepts a :class:`VisualStrategy` or any supported alias.
    ``max_ai_video_beats`` only limits :attr:`VisualType.VIDEO`; stock, still,
    text and none beats are never counted against it.
    """

    def __init__(
        self,
        *,
        strategy: VisualStrategy | str = DEFAULT_VISUAL_STRATEGY,
        max_ai_video_beats: int = DEFAULT_MAX_AI_VIDEO_BEATS,
    ) -> None:
        self.strategy = normalise_visual_strategy(strategy)
        self.max_ai_video_beats = _validate_limit(max_ai_video_beats)

    def _types_for(self, beats: Sequence[Beat]) -> list[VisualType]:
        strategy = self.strategy
        count = len(beats)
        if strategy in _STRATEGY_TYPES:
            return [_STRATEGY_TYPES[strategy]] * count

        if strategy is VisualStrategy.EXISTING:
            types: list[VisualType] = []
            videos = 0
            for beat in beats:
                visual = _normalise_visual_type(beat.visual_type) or VisualType.IMAGE
                if visual is VisualType.VIDEO:
                    if videos >= self.max_ai_video_beats:
                        visual = VisualType.IMAGE
                    else:
                        videos += 1
                types.append(visual)
            return types

        limit = min(self.max_ai_video_beats, count)
        return [
            VisualType.VIDEO if index < limit else VisualType.IMAGE
            for index in range(count)
        ]

    def assign_beats(self, beats: Sequence[Beat]) -> list[Beat]:
        """Return copies of *beats* with their visual type assigned."""
        types = self._types_for(beats)
        return [
            beat.model_copy(update={"visual_type": visual.value}, deep=True)
            for beat, visual in zip(beats, types, strict=True)
        ]

    def assign_beat(self, beat: Beat) -> Beat:
        """Return a copy of one *beat* with its visual type assigned."""
        return self.assign_beats([beat])[0]

    def assign(self, script: Script) -> Script:
        """Return a copy of *script* with every beat's visual type assigned."""
        beats = self.assign_beats(script.beats)
        return script.model_copy(update={"beats": beats}, deep=True)

    def types_for(self, script: Script) -> list[str]:
        """Return the visual type value of each beat without changing *script*."""
        return [visual.value for visual in self._types_for(script.beats)]

    apply = assign
    generate = assign
    __call__ = assign
    assign_script = assign
    assign_types = assign
    visual_types = types_for

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(strategy={self.strategy.value!r}, "
            f"max_ai_video_beats={self.max_ai_video_beats!r})"
        )


@overload
def assign_visual_types(
    value: Script,
    *,
    strategy: VisualStrategy | str = DEFAULT_VISUAL_STRATEGY,
    max_ai_video_beats: int = DEFAULT_MAX_AI_VIDEO_BEATS,
) -> Script: ...


@overload
def assign_visual_types(
    value: Beat,
    *,
    strategy: VisualStrategy | str = DEFAULT_VISUAL_STRATEGY,
    max_ai_video_beats: int = DEFAULT_MAX_AI_VIDEO_BEATS,
) -> Beat: ...


@overload
def assign_visual_types(
    value: Sequence[Beat],
    *,
    strategy: VisualStrategy | str = DEFAULT_VISUAL_STRATEGY,
    max_ai_video_beats: int = DEFAULT_MAX_AI_VIDEO_BEATS,
) -> list[Beat]: ...


def assign_visual_types(
    value: Script | Beat | Sequence[Beat],
    *,
    strategy: VisualStrategy | str = DEFAULT_VISUAL_STRATEGY,
    max_ai_video_beats: int = DEFAULT_MAX_AI_VIDEO_BEATS,
) -> Script | Beat | list[Beat]:
    """Assign a visual type to a script, one beat or a sequence of beats.

    A :class:`Script` returns a new script, one :class:`Beat` returns a new
    beat and a sequence returns one beat per input. The limit is applied across
    the whole sequence, so splitting a script never produces extra video beats.
    """
    assigner = VisualAssigner(
        strategy=strategy,
        max_ai_video_beats=max_ai_video_beats,
    )
    if isinstance(value, Script):
        return assigner.assign(value)
    if isinstance(value, Beat):
        return assigner.assign_beat(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return assigner.assign_beats(value)
    raise TypeError("visual assignment needs a script, a beat or a sequence of beats")


assign_visuals = assign_visual_types
build_visual_assignment = assign_visual_types
assign_beat_visual = assign_visual_types


def visual_types(
    value: Script | Beat | Sequence[Beat],
    *,
    strategy: VisualStrategy | str = DEFAULT_VISUAL_STRATEGY,
    max_ai_video_beats: int = DEFAULT_MAX_AI_VIDEO_BEATS,
) -> list[str]:
    """Return the visual type values for *value* without changing it."""
    assigner = VisualAssigner(
        strategy=strategy,
        max_ai_video_beats=max_ai_video_beats,
    )
    if isinstance(value, Script):
        return assigner.types_for(value)
    if isinstance(value, Beat):
        return assigner.types_for(Script(beats=[value]))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [visual.value for visual in assigner._types_for(value)]
    raise TypeError("visual assignment needs a script, a beat or a sequence of beats")


visual_types_for = visual_types

"""Entrance and exit animation for text layers.

Text animations are declarative presets: an entrance plays over the first part
of a layer's visible window and an exit over the last part. Each preset becomes
ordinary keyframes for opacity, position or scale, so the same animation can be
sampled deterministically by the still renderer and translated into FFmpeg
expressions by the video builder.

A preset never mutates the composition. Explicit keyframes always win over the
tracks a preset would generate for the same property.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .errors import RenderError
from .keyframes import (
    AnimatedProperty,
    AnimatedValue,
    Easing,
    KeyframeTrack,
    keyframe_track,
    keyframes_for_layer,
)

__all__ = [
    "EntranceAnimation",
    "ExitAnimation",
    "TextAnimation",
    "TextAnimationError",
    "TextAnimationKind",
    "TextAnimationPreset",
    "TextAnimationSet",
    "TextAnimationSpec",
    "TextAnimationTracks",
    "TextAnimationType",
    "TextAnimations",
    "animation_for_layer",
    "animation_tracks",
    "entrance_animation",
    "entrance_animation_for_layer",
    "exit_animation",
    "exit_animation_for_layer",
    "fade",
    "pop",
    "sample_animation",
    "sample_text_animation",
    "slide",
    "text_animation_for_layer",
    "text_animation_tracks",
    "text_animation_tracks_for_layer",
    "text_animations_for_layer",
    "text_tracks_for_layer",
    "zoom",
]


class TextAnimationError(RenderError):
    """A text animation declaration could not be built."""

    code = "text_animation"


class TextAnimationType(StrEnum):
    """The entrance and exit presets a text layer can declare."""

    NONE = "none"
    FADE = "fade"
    FADE_IN = "fade"
    FADE_OUT = "fade"
    SLIDE = "slide"
    SLIDE_IN = "slide"
    SLIDE_OUT = "slide"
    SLIDE_UP = "slide_up"
    SLIDE_DOWN = "slide_down"
    SLIDE_LEFT = "slide_left"
    SLIDE_RIGHT = "slide_right"
    ZOOM = "zoom"
    ZOOM_IN = "zoom"
    ZOOM_OUT = "zoom"
    SCALE = "zoom"
    POP = "pop"

    @classmethod
    def _missing_(cls, value: Any) -> TextAnimationType | None:
        if not isinstance(value, str):
            return None
        key = "".join(character for character in value.lower() if character.isalnum())
        aliases = {
            "none": cls.NONE,
            "off": cls.NONE,
            "false": cls.NONE,
            "fade": cls.FADE,
            "fadein": cls.FADE,
            "fadeout": cls.FADE,
            "dissolve": cls.FADE,
            "slide": cls.SLIDE,
            "slidein": cls.SLIDE,
            "slideout": cls.SLIDE,
            "slideup": cls.SLIDE_UP,
            "up": cls.SLIDE_UP,
            "slidedown": cls.SLIDE_DOWN,
            "down": cls.SLIDE_DOWN,
            "slideleft": cls.SLIDE_LEFT,
            "left": cls.SLIDE_LEFT,
            "slideright": cls.SLIDE_RIGHT,
            "right": cls.SLIDE_RIGHT,
            "zoom": cls.ZOOM,
            "zoomin": cls.ZOOM,
            "zoomout": cls.ZOOM,
            "scale": cls.ZOOM,
            "grow": cls.ZOOM,
            "shrink": cls.ZOOM,
            "pop": cls.POP,
            "bounce": cls.POP,
        }
        return aliases.get(key)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _pair(value: Any, name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a pair of numbers")
    if not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{name} must be a pair of numbers")
    return _finite(value[0], name), _finite(value[1], name)


class TextAnimation(BaseModel):
    """One entrance or exit preset attached to a text layer.

    ``duration`` is measured in seconds from the start of the visible window for
    an entrance and from its end for an exit. ``distance`` is the travel of a
    slide preset in pixels; ``start_scale`` is the scale a zoom preset starts at.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    type: TextAnimationType = Field(
        default=TextAnimationType.FADE,
        validation_alias=AliasChoices(
            "type",
            "kind",
            "preset",
            "effect",
            "style",
            "animation",
            "motion",
        ),
    )
    duration: float = Field(
        default=0.5,
        gt=0.0,
        validation_alias=AliasChoices(
            "duration",
            "duration_seconds",
            "length",
            "length_seconds",
            "time",
            "seconds",
        ),
    )
    easing: Easing = Field(
        default=Easing.EASE_OUT,
        validation_alias=AliasChoices("easing", "ease", "curve", "timing"),
    )
    distance: float = Field(
        default=48.0,
        ge=0.0,
        validation_alias=AliasChoices(
            "distance",
            "distance_px",
            "offset_px",
            "travel",
            "shift",
        ),
    )
    offset: tuple[float, float] | None = Field(
        default=None,
        validation_alias=AliasChoices("offset", "translate", "movement"),
    )
    start_scale: float = Field(
        default=0.8,
        gt=0.0,
        validation_alias=AliasChoices(
            "start_scale",
            "from_scale",
            "scale_from",
            "zoom_from",
        ),
    )
    end_scale: float = Field(
        default=1.0,
        gt=0.0,
        validation_alias=AliasChoices("end_scale", "to_scale", "scale_to", "zoom_to"),
    )
    opacity_from: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("opacity_from", "from_opacity", "start_opacity"),
    )
    opacity_to: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("opacity_to", "to_opacity", "end_opacity"),
    )
    delay: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("delay", "start_delay", "delay_seconds"),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_shorthand(cls, data: Any) -> Any:
        if isinstance(data, bool):
            return {"type": "fade" if data else "none"}
        if isinstance(data, str):
            return {"type": data}
        return data

    @field_validator("duration", "distance", "start_scale", "end_scale", mode="before")
    @classmethod
    def _validate_number(cls, value: Any) -> Any:
        return _finite(value, "text animation")

    @field_validator("opacity_from", "opacity_to", "delay", mode="before")
    @classmethod
    def _validate_fraction(cls, value: Any) -> Any:
        return _finite(value, "text animation")

    @field_validator("offset", mode="before")
    @classmethod
    def _validate_offset(cls, value: Any) -> Any:
        if value is None:
            return None
        return _pair(value, "text animation offset")

    @property
    def kind(self) -> TextAnimationType:
        """Alias for :attr:`type`."""
        return self.type

    @property
    def preset(self) -> TextAnimationType:
        """Alias for :attr:`type`."""
        return self.type

    @property
    def duration_seconds(self) -> float:
        """Alias for :attr:`duration`."""
        return self.duration

    @property
    def is_none(self) -> bool:
        """Whether this preset disables the animation."""
        return self.type is TextAnimationType.NONE

    @property
    def is_fade(self) -> bool:
        """Whether this preset only changes opacity."""
        return self.type is TextAnimationType.FADE

    @property
    def is_slide(self) -> bool:
        """Whether this preset moves the text."""
        return self.type in {
            TextAnimationType.SLIDE,
            TextAnimationType.SLIDE_UP,
            TextAnimationType.SLIDE_DOWN,
            TextAnimationType.SLIDE_LEFT,
            TextAnimationType.SLIDE_RIGHT,
        }

    @property
    def is_zoom(self) -> bool:
        """Whether this preset scales the text."""
        return self.type in {
            TextAnimationType.ZOOM,
            TextAnimationType.POP,
        }

    def offset_for(self, *, entrance: bool) -> tuple[float, float]:
        """Return the position offset this preset starts or ends at."""
        if self.offset is not None:
            x, y = self.offset
            return (x, y) if entrance else (-x, -y)
        distance = self.distance
        if self.type is TextAnimationType.SLIDE_UP:
            return (0.0, distance) if entrance else (0.0, -distance)
        if self.type is TextAnimationType.SLIDE_DOWN:
            return (0.0, -distance) if entrance else (0.0, distance)
        if self.type is TextAnimationType.SLIDE_LEFT:
            return (distance, 0.0) if entrance else (-distance, 0.0)
        if self.type is TextAnimationType.SLIDE_RIGHT:
            return (-distance, 0.0) if entrance else (distance, 0.0)
        return (distance, 0.0) if entrance else (-distance, 0.0)

    def scale_for(self, *, entrance: bool) -> float:
        """Return the scale this preset starts or ends at."""
        return self.start_scale


def entrance_animation(
    type: TextAnimationType | str = TextAnimationType.FADE,
    *,
    duration: float = 0.5,
    easing: Easing | str = Easing.EASE_OUT,
    distance: float = 48.0,
    start_scale: float = 0.8,
) -> TextAnimation:
    """Build an entrance animation preset."""
    return TextAnimation(
        type=TextAnimationType(type),
        duration=duration,
        easing=Easing(easing),
        distance=distance,
        start_scale=start_scale,
    )


def exit_animation(
    type: TextAnimationType | str = TextAnimationType.FADE,
    *,
    duration: float = 0.5,
    easing: Easing | str = Easing.EASE_OUT,
    distance: float = 48.0,
    start_scale: float = 0.8,
) -> TextAnimation:
    """Build an exit animation preset."""
    return TextAnimation(
        type=TextAnimationType(type),
        duration=duration,
        easing=Easing(easing),
        distance=distance,
        start_scale=start_scale,
    )


def fade(
    *, duration: float = 0.5, easing: Easing | str = Easing.EASE_OUT
) -> TextAnimation:
    """Build a fade preset."""
    return TextAnimation(
        type=TextAnimationType.FADE,
        duration=duration,
        easing=Easing(easing),
    )


def slide(
    direction: str = "up",
    *,
    duration: float = 0.5,
    distance: float = 48.0,
    easing: Easing | str = Easing.EASE_OUT,
) -> TextAnimation:
    """Build a slide preset in *direction*."""
    return TextAnimation(
        type=TextAnimationType(f"slide_{direction}"),
        duration=duration,
        distance=distance,
        easing=Easing(easing),
    )


def zoom(
    *,
    duration: float = 0.5,
    start_scale: float = 0.8,
    easing: Easing | str = Easing.EASE_OUT,
) -> TextAnimation:
    """Build a zoom preset."""
    return TextAnimation(
        type=TextAnimationType.ZOOM,
        duration=duration,
        start_scale=start_scale,
        easing=Easing(easing),
    )


def pop(*, duration: float = 0.5, start_scale: float = 0.5) -> TextAnimation:
    """Build a pop preset."""
    return TextAnimation(
        type=TextAnimationType.POP,
        duration=duration,
        start_scale=start_scale,
        easing=Easing.EASE_OUT,
    )


EntranceAnimation = TextAnimation
ExitAnimation = TextAnimation


@dataclass(frozen=True, slots=True)
class TextAnimationSet:
    """The entrance and exit presets declared on one layer."""

    entrance: TextAnimation | None = None
    exit: TextAnimation | None = None

    @property
    def is_empty(self) -> bool:
        """Whether neither preset is active."""
        return (self.entrance is None or self.entrance.is_none) and (
            self.exit is None or self.exit.is_none
        )

    @property
    def enter(self) -> TextAnimation | None:
        """Alias for :attr:`entrance`."""
        return self.entrance

    @property
    def leave(self) -> TextAnimation | None:
        """Alias for :attr:`exit`."""
        return self.exit


TextAnimations = TextAnimationSet

TextAnimationPreset = TextAnimationType
"""Alias for :class:`TextAnimationType`."""

TextAnimationKind = TextAnimationType
"""Alias for :class:`TextAnimationType`."""

TextAnimationSpec = TextAnimation
"""Alias for :class:`TextAnimation`."""


@dataclass(frozen=True, slots=True)
class TextAnimationTracks:
    """The keyframe tracks generated by a text animation set."""

    opacity: KeyframeTrack | None = None
    position: KeyframeTrack | None = None
    scale: KeyframeTrack | None = None

    @property
    def is_empty(self) -> bool:
        """Whether no property is animated."""
        return self.opacity is None and self.position is None and self.scale is None


def _coerce_animation(value: Any) -> TextAnimation | None:
    if value is None or value is False:
        return None
    if isinstance(value, TextAnimation):
        return None if value.is_none else value
    if isinstance(value, (str, bool, Mapping)):
        animation = TextAnimation.model_validate(value)
        return None if animation.is_none else animation
    raise TextAnimationError(f"unsupported text animation declaration {value!r}")


def _layer_value(layer: Any, *names: str) -> Any:
    extras = getattr(layer, "model_extra", None) or {}
    for name in names:
        if name in extras:
            return extras[name]
        value = getattr(layer, name, None)
        if value is not None:
            return value
    return None


def entrance_animation_for_layer(layer: Any) -> TextAnimation | None:
    """Return the entrance preset declared on *layer*, or ``None``."""
    raw = _layer_value(
        layer,
        "entrance",
        "entrance_animation",
        "enter",
        "in",
        "appear",
        "in_animation",
    )
    return _coerce_animation(raw)


def exit_animation_for_layer(layer: Any) -> TextAnimation | None:
    """Return the exit preset declared on *layer*, or ``None``."""
    raw = _layer_value(
        layer,
        "exit",
        "exit_animation",
        "leave",
        "out",
        "disappear",
        "out_animation",
    )
    return _coerce_animation(raw)


def text_animation_for_layer(layer: Any) -> TextAnimationSet | None:
    """Return both presets declared on *layer*, or ``None`` when absent."""
    animations = TextAnimationSet(
        entrance=entrance_animation_for_layer(layer),
        exit=exit_animation_for_layer(layer),
    )
    return None if animations.is_empty else animations


def _dedupe_frames(
    frames: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted(
        (dict(frame) for frame in frames),
        key=lambda frame: float(frame["time"]),
    )
    unique: list[dict[str, Any]] = []
    for frame in ordered:
        if unique and float(frame["time"]) == float(unique[-1]["time"]):
            unique[-1] = frame
        else:
            unique.append(frame)
    return unique


def _track(
    property_name: AnimatedProperty,
    frames: Sequence[Mapping[str, Any]],
) -> KeyframeTrack | None:
    unique = _dedupe_frames(frames)
    if len(unique) < 2:
        return None
    return keyframe_track(property_name, unique)


def _duration(animation: TextAnimation | None, span: float) -> float:
    if animation is None or animation.is_none:
        return 0.0
    return min(max(animation.duration, 0.0), span)


def animation_tracks(
    entrance: TextAnimation | None,
    exit: TextAnimation | None,
    start_seconds: float,
    end_seconds: float,
    *,
    opacity: float = 1.0,
) -> TextAnimationTracks:
    """Turn entrance and exit presets into keyframes over one visible window."""
    if end_seconds <= start_seconds:
        return TextAnimationTracks()
    entrance = _coerce_animation(entrance)
    exit = _coerce_animation(exit)
    if entrance is None and exit is None:
        return TextAnimationTracks()

    span = end_seconds - start_seconds
    entrance_duration = _duration(entrance, span)
    exit_duration = _duration(exit, span)
    if entrance_duration + exit_duration > span:
        factor = span / (entrance_duration + exit_duration)
        entrance_duration *= factor
        exit_duration *= factor

    entrance_start = start_seconds
    if entrance is not None:
        delay = min(entrance.delay, max(0.0, span - entrance_duration))
        entrance_start += delay
    entrance_end = entrance_start + entrance_duration
    exit_start = end_seconds - exit_duration

    opacity_frames: list[dict[str, Any]] = []
    position_frames: list[dict[str, Any]] = []
    scale_frames: list[dict[str, Any]] = []

    if entrance is not None:
        from_opacity = entrance.opacity_from * opacity
        to_opacity = opacity
        opacity_frames.extend(
            (
                {
                    "time": entrance_start,
                    "value": from_opacity,
                    "easing": entrance.easing,
                },
                {"time": entrance_end, "value": to_opacity},
            )
        )
        if entrance.is_slide:
            position_frames.extend(
                (
                    {
                        "time": entrance_start,
                        "value": entrance.offset_for(entrance=True),
                        "easing": entrance.easing,
                    },
                    {"time": entrance_end, "value": (0.0, 0.0)},
                )
            )
        if entrance.is_zoom:
            scale_frames.extend(
                (
                    {
                        "time": entrance_start,
                        "value": entrance.scale_for(entrance=True),
                        "easing": entrance.easing,
                    },
                    {"time": entrance_end, "value": 1.0},
                )
            )

    if exit is not None:
        if not opacity_frames:
            opacity_frames.append(
                {"time": start_seconds, "value": opacity, "easing": exit.easing}
            )
        if float(opacity_frames[-1]["time"]) < exit_start:
            opacity_frames.append(
                {"time": exit_start, "value": opacity, "easing": exit.easing}
            )
        else:
            opacity_frames[-1]["easing"] = exit.easing
        opacity_frames.append(
            {"time": end_seconds, "value": exit.opacity_from * opacity}
        )

        if exit.is_slide:
            if not position_frames:
                position_frames.append(
                    {"time": start_seconds, "value": (0.0, 0.0), "easing": exit.easing}
                )
            if float(position_frames[-1]["time"]) < exit_start:
                position_frames.append(
                    {"time": exit_start, "value": (0.0, 0.0), "easing": exit.easing}
                )
            else:
                position_frames[-1]["easing"] = exit.easing
            position_frames.append(
                {"time": end_seconds, "value": exit.offset_for(entrance=False)}
            )

        if exit.is_zoom:
            if not scale_frames:
                scale_frames.append(
                    {"time": start_seconds, "value": 1.0, "easing": exit.easing}
                )
            if float(scale_frames[-1]["time"]) < exit_start:
                scale_frames.append(
                    {"time": exit_start, "value": 1.0, "easing": exit.easing}
                )
            else:
                scale_frames[-1]["easing"] = exit.easing
            scale_frames.append(
                {"time": end_seconds, "value": exit.scale_for(entrance=False)}
            )

    return TextAnimationTracks(
        opacity=_track(AnimatedProperty.OPACITY, opacity_frames),
        position=_track(AnimatedProperty.POSITION, position_frames),
        scale=_track(AnimatedProperty.SCALE, scale_frames),
    )


def text_animation_tracks(
    entrance: TextAnimation | None,
    exit: TextAnimation | None,
    start_seconds: float,
    end_seconds: float,
    *,
    opacity: float = 1.0,
) -> TextAnimationTracks:
    """Alias for :func:`animation_tracks`."""
    return animation_tracks(entrance, exit, start_seconds, end_seconds, opacity=opacity)


def text_animation_tracks_for_layer(
    layer: Any,
    start_seconds: float,
    end_seconds: float,
) -> TextAnimationTracks:
    """Generate the tracks declared by *layer*'s entrance and exit presets."""
    opacity = _layer_value(layer, "opacity")
    value = 1.0 if opacity is None else float(opacity)
    return animation_tracks(
        entrance_animation_for_layer(layer),
        exit_animation_for_layer(layer),
        start_seconds,
        end_seconds,
        opacity=value,
    )


def text_tracks_for_layer(
    layer: Any,
    start_seconds: float,
    end_seconds: float,
) -> TextAnimationTracks:
    """Merge declared keyframes with the tracks generated by text presets."""
    declared = keyframes_for_layer(layer)
    generated = text_animation_tracks_for_layer(layer, start_seconds, end_seconds)
    return TextAnimationTracks(
        opacity=declared.opacity or generated.opacity,
        position=declared.position or generated.position,
        scale=declared.scale or generated.scale,
    )


def sample_text_animation(
    layer: Any,
    time_seconds: float,
    start_seconds: float,
    end_seconds: float,
) -> dict[AnimatedProperty, AnimatedValue]:
    """Sample a layer's generated text animation tracks at *time_seconds*."""
    tracks = text_animation_tracks_for_layer(layer, start_seconds, end_seconds)
    sampled: dict[AnimatedProperty, AnimatedValue] = {}
    for property_name, track in (
        (AnimatedProperty.OPACITY, tracks.opacity),
        (AnimatedProperty.POSITION, tracks.position),
        (AnimatedProperty.SCALE, tracks.scale),
    ):
        if track is not None:
            sampled[property_name] = track.sample(time_seconds)
    return sampled


text_animations_for_layer = text_animation_for_layer
"""Alias for :func:`text_animation_for_layer`."""

animation_for_layer = text_animation_for_layer
"""Alias for :func:`text_animation_for_layer`."""

sample_animation = sample_text_animation
"""Alias for :func:`sample_text_animation`."""

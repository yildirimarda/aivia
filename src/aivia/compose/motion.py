"""Keyframed motion for video, including Ken Burns and parallax.

Motion is expressed as ordinary keyframes so it samples deterministically. A
``ken_burns`` motion adds a slow zoom and an optional pan inside the layer's
source; a ``parallax`` motion drifts the layer across the canvas by a distance
scaled by its depth. Both become ordinary keyframe tracks and, for the FFmpeg
renderer, pure expression strings.

The expression builders never run ffmpeg. They translate the same easing curves
used by the still renderer into the small expression language understood by the
``overlay``, ``zoompan``, ``rotate`` and ``geq`` filters.
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
from .text_animation import text_animation_tracks_for_layer

__all__ = [
    "LayerMotion",
    "Motion",
    "MotionError",
    "MotionTracks",
    "MotionType",
    "ken_burns",
    "layer_motion",
    "motion_for_layer",
    "motion_tracks",
    "pair_expression",
    "parallax",
    "sample_motion",
    "track_expression",
    "tracks_for_layer",
]


class MotionError(RenderError):
    """A motion preset or expression could not be built."""

    code = "motion"


class MotionType(StrEnum):
    """The motion presets a layer can declare."""

    KEN_BURNS = "ken_burns"
    PARALLAX = "parallax"

    @classmethod
    def _missing_(cls, value: Any) -> MotionType | None:
        if not isinstance(value, str):
            return None
        key = "".join(character for character in value.lower() if character.isalnum())
        aliases = {
            "kenburns": cls.KEN_BURNS,
            "kenburn": cls.KEN_BURNS,
            "zoompan": cls.KEN_BURNS,
            "zoom": cls.KEN_BURNS,
            "parallax": cls.PARALLAX,
            "depth": cls.PARALLAX,
            "drift": cls.PARALLAX,
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


class Motion(BaseModel):
    """A declarative motion preset attached to one layer.

    ``ken_burns`` zooms from ``start_scale`` to ``end_scale`` and pans from
    ``start_pan`` to ``end_pan`` inside the layer source. ``parallax`` moves the
    layer from ``start_pan`` by ``drift * depth`` over its visible window.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    type: MotionType = Field(
        default=MotionType.KEN_BURNS,
        validation_alias=AliasChoices("type", "kind", "preset", "motion", "effect"),
    )
    start_scale: float = Field(
        default=1.0,
        gt=0.0,
        validation_alias=AliasChoices(
            "start_scale",
            "from_scale",
            "scale_start",
            "zoom_start",
            "start_zoom",
        ),
    )
    end_scale: float = Field(
        default=1.0,
        gt=0.0,
        validation_alias=AliasChoices(
            "end_scale",
            "to_scale",
            "scale_end",
            "zoom_end",
            "end_zoom",
            "zoom",
        ),
    )
    start_pan: tuple[float, float] = Field(
        default=(0.0, 0.0),
        validation_alias=AliasChoices(
            "start_pan",
            "pan_start",
            "from_pan",
            "from_position",
            "start_position",
            "from",
        ),
    )
    end_pan: tuple[float, float] = Field(
        default=(0.0, 0.0),
        validation_alias=AliasChoices(
            "end_pan",
            "pan_end",
            "to_pan",
            "to_position",
            "end_position",
            "to",
        ),
    )
    easing: Easing = Field(
        default=Easing.EASE_IN_OUT,
        validation_alias=AliasChoices("easing", "ease", "timing", "curve"),
    )
    depth: float = Field(
        default=1.0,
        ge=0.0,
        validation_alias=AliasChoices("depth", "parallax_depth", "factor"),
    )
    drift: tuple[float, float] = Field(
        default=(0.0, 0.0),
        validation_alias=AliasChoices("drift", "shift", "travel", "movement"),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_string_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"type": data}
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        if not any(
            key in values for key in ("type", "kind", "preset", "motion", "effect")
        ) and any(
            key in values
            for key in (
                "depth",
                "parallax_depth",
                "factor",
                "drift",
                "shift",
                "travel",
            )
        ):
            values["type"] = MotionType.PARALLAX
        zoom = values.get("zoom")
        if (
            isinstance(zoom, Sequence)
            and not isinstance(zoom, (str, bytes))
            and len(zoom) == 2
            and all(isinstance(item, (int, float)) for item in zoom)
        ):
            values.pop("zoom", None)
            values.setdefault("start_scale", zoom[0])
            values.setdefault("end_scale", zoom[1])
        pan = values.get("pan")
        if (
            isinstance(pan, Sequence)
            and not isinstance(pan, (str, bytes))
            and len(pan) == 2
            and all(
                isinstance(item, Sequence)
                and not isinstance(item, (str, bytes))
                and len(item) == 2
                for item in pan
            )
        ):
            values.pop("pan", None)
            values.setdefault("start_pan", pan[0])
            values.setdefault("end_pan", pan[1])
        return values

    @field_validator("start_scale", "end_scale", "depth", mode="before")
    @classmethod
    def _validate_number(cls, value: Any) -> Any:
        return _finite(value, "motion")

    @field_validator("start_pan", "end_pan", "drift", mode="before")
    @classmethod
    def _validate_pair(cls, value: Any) -> Any:
        return _pair(value, "motion offset")

    @property
    def is_ken_burns(self) -> bool:
        """Whether this is the Ken Burns preset."""
        return self.type is MotionType.KEN_BURNS

    @property
    def is_parallax(self) -> bool:
        """Whether this is the parallax preset."""
        return self.type is MotionType.PARALLAX

    @property
    def zoom(self) -> tuple[float, float]:
        """The start and end scale as a pair."""
        return self.start_scale, self.end_scale

    @property
    def pan(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """The start and end pan as a pair of pairs."""
        return self.start_pan, self.end_pan


def ken_burns(
    *,
    start_scale: float = 1.0,
    end_scale: float = 1.15,
    start_pan: tuple[float, float] = (0.0, 0.0),
    end_pan: tuple[float, float] = (0.0, 0.0),
    easing: Easing | str = Easing.EASE_IN_OUT,
) -> Motion:
    """Build a Ken Burns motion preset."""
    return Motion(
        type=MotionType.KEN_BURNS,
        start_scale=start_scale,
        end_scale=end_scale,
        start_pan=start_pan,
        end_pan=end_pan,
        easing=Easing(easing),
    )


def parallax(
    *,
    depth: float = 0.5,
    drift: tuple[float, float] = (0.0, 0.0),
    start_pan: tuple[float, float] = (0.0, 0.0),
    easing: Easing | str = Easing.LINEAR,
) -> Motion:
    """Build a parallax motion preset that drifts a layer across the canvas."""
    return Motion(
        type=MotionType.PARALLAX,
        depth=depth,
        drift=drift,
        start_pan=start_pan,
        easing=Easing(easing),
    )


def _layer_value(layer: Any, *names: str) -> Any:
    extras = getattr(layer, "model_extra", None) or {}
    for name in names:
        if name in extras:
            return extras[name]
        value = getattr(layer, name, None)
        if value is not None:
            return value
    return None


def motion_for_layer(layer: Any) -> Motion | None:
    """Return the motion preset declared on *layer*, or ``None``."""
    raw = _layer_value(layer, "motion", "camera", "movement")
    if raw is None or raw is False:
        return None
    if isinstance(raw, Motion):
        return raw
    if isinstance(raw, (str, Mapping)):
        return Motion.model_validate(raw)
    raise MotionError(f"unsupported motion declaration {raw!r}")


layer_motion = motion_for_layer
"""Alias for :func:`motion_for_layer`."""


@dataclass(frozen=True, slots=True)
class MotionTracks:
    """The keyframe tracks that describe one layer's motion."""

    scale: KeyframeTrack | None = None
    position: KeyframeTrack | None = None
    rotation: KeyframeTrack | None = None
    opacity: KeyframeTrack | None = None
    pan: KeyframeTrack | None = None

    @property
    def is_empty(self) -> bool:
        """Whether no property is animated."""
        return not any(
            track is not None
            for track in (
                self.scale,
                self.position,
                self.rotation,
                self.opacity,
                self.pan,
            )
        )


LayerMotion = MotionTracks
"""Alias for :class:`MotionTracks`."""


def _track(
    property_name: AnimatedProperty | None,
    start: float,
    end: float,
    start_value: AnimatedValue,
    end_value: AnimatedValue,
    easing: Easing,
    *,
    always: bool = False,
) -> KeyframeTrack | None:
    if start_value == end_value and not always:
        return None
    frames = [
        {"time": start, "value": start_value, "easing": easing},
        {"time": end, "value": end_value},
    ]
    if property_name is None:
        return KeyframeTrack.model_validate({"keyframes": frames})
    return keyframe_track(property_name, frames)


def motion_tracks(
    motion: Motion | None,
    start_seconds: float,
    end_seconds: float,
) -> MotionTracks:
    """Turn *motion* into keyframes over one visible layer window."""
    if motion is None or end_seconds <= start_seconds:
        return MotionTracks()

    easing = motion.easing
    scale_changes = motion.start_scale != motion.end_scale or motion.start_scale != 1.0
    scale = _track(
        AnimatedProperty.SCALE,
        start_seconds,
        end_seconds,
        motion.start_scale,
        motion.end_scale,
        easing,
        always=scale_changes,
    )

    if motion.is_parallax:
        travel = (
            motion.drift[0] * motion.depth,
            motion.drift[1] * motion.depth,
        )
        target = (
            motion.start_pan[0] + travel[0],
            motion.start_pan[1] + travel[1],
        )
        position = _track(
            AnimatedProperty.POSITION,
            start_seconds,
            end_seconds,
            motion.start_pan,
            target,
            easing,
        )
        return MotionTracks(scale=scale, position=position)

    pan = _track(
        None,
        start_seconds,
        end_seconds,
        motion.start_pan,
        motion.end_pan,
        easing,
    )
    return MotionTracks(scale=scale, pan=pan)


def tracks_for_layer(
    layer: Any,
    start_seconds: float,
    end_seconds: float,
) -> MotionTracks:
    """Merge declared layer keyframes with the layer's motion preset.

    Explicit keyframes win over the values a preset would generate for the same
    property, so a caller can start from Ken Burns and then override its scale
    or position track.
    """
    declared = keyframes_for_layer(layer)
    generated = motion_tracks(
        motion_for_layer(layer),
        start_seconds,
        end_seconds,
    )
    text = text_animation_tracks_for_layer(layer, start_seconds, end_seconds)
    return MotionTracks(
        scale=declared.scale or generated.scale or text.scale,
        position=declared.position or generated.position or text.position,
        rotation=declared.rotation or generated.rotation,
        opacity=declared.opacity or generated.opacity or text.opacity,
        pan=generated.pan,
    )


def sample_motion(
    motion: Motion | None,
    time_seconds: float,
    start_seconds: float,
    end_seconds: float,
) -> dict[AnimatedProperty, AnimatedValue]:
    """Sample a preset's generated tracks at *time_seconds*."""
    tracks = motion_tracks(motion, start_seconds, end_seconds)
    sampled: dict[AnimatedProperty, AnimatedValue] = {}
    for property_name in (
        AnimatedProperty.SCALE,
        AnimatedProperty.POSITION,
        AnimatedProperty.ROTATION,
        AnimatedProperty.OPACITY,
    ):
        track = getattr(tracks, property_name.value)
        if track is not None:
            sampled[property_name] = track.sample(time_seconds)
    return sampled


def _number(value: float) -> str:
    if value == 0.0:
        return "0"
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _component(value: AnimatedValue, index: int | None) -> float:
    if index is None:
        if isinstance(value, tuple):
            raise MotionError("expected a scalar keyframe value")
        return value
    if not isinstance(value, tuple):
        raise MotionError("expected a pair keyframe value")
    return value[index]


def _easing_expression(easing: Easing, progress: str) -> str:
    p = progress
    if easing is Easing.LINEAR:
        return p
    if easing is Easing.STEP:
        return f"if(gte({p},1),1,0)"
    if easing is Easing.SMOOTHSTEP:
        return f"({p}*{p}*(3-2*{p}))"
    if easing is Easing.EASE_IN:
        return f"({p}*{p})"
    if easing is Easing.EASE_OUT:
        return f"(1-(1-{p})*(1-{p}))"
    if easing is Easing.EASE_IN_OUT:
        return f"if(lt({p},0.5),2*{p}*{p},1-2*(1-{p})*(1-{p}))"
    if easing is Easing.EASE_IN_SINE:
        return f"(1-cos({p}*PI/2))"
    if easing is Easing.EASE_OUT_SINE:
        return f"sin({p}*PI/2)"
    if easing is Easing.EASE_IN_OUT_SINE:
        return f"((1-cos({p}*PI))/2)"
    if easing is Easing.EASE_IN_QUAD:
        return f"({p}*{p})"
    if easing is Easing.EASE_OUT_QUAD:
        return f"(1-(1-{p})*(1-{p}))"
    if easing is Easing.EASE_IN_OUT_QUAD:
        return f"if(lt({p},0.5),2*{p}*{p},1-2*(1-{p})*(1-{p}))"
    if easing is Easing.EASE_IN_CUBIC:
        return f"({p}*{p}*{p})"
    if easing is Easing.EASE_OUT_CUBIC:
        return f"(1-(1-{p})*(1-{p})*(1-{p}))"
    if easing is Easing.EASE_IN_OUT_CUBIC:
        return f"if(lt({p},0.5),4*{p}*{p}*{p},1-4*(1-{p})*(1-{p})*(1-{p}))"
    if easing is Easing.EASE_IN_EXPO:
        return f"if(eq({p},0),0,pow(2,10*({p}-1)))"
    if easing is Easing.EASE_OUT_EXPO:
        return f"if(eq({p},1),1,1-pow(2,-10*{p}))"
    if easing is Easing.EASE_IN_OUT_EXPO:
        return (
            f"if(eq({p},0),0,if(eq({p},1),1,"
            f"if(lt({p},0.5),pow(2,20*{p}-10)/2,"
            f"(2-pow(2,-20*{p}+10))/2)))"
        )
    return p


def _component_expression(
    track: KeyframeTrack,
    variable: str,
    index: int | None,
) -> str:
    frames = track.keyframes
    expression = _number(_component(frames[-1].value, index))
    for position in range(len(frames) - 2, -1, -1):
        start = frames[position]
        end = frames[position + 1]
        start_value = _component(start.value, index)
        end_value = _component(end.value, index)
        span = end.time_seconds - start.time_seconds
        progress = f"(({variable}-{_number(start.time_seconds)})/{_number(span)})"
        eased = _easing_expression(start.easing, progress)
        delta = end_value - start_value
        segment = f"({_number(start_value)}+{_number(delta)}*{eased})"
        expression = (
            f"if(lt({variable},{_number(start.time_seconds)}),"
            f"{_number(start_value)},"
            f"if(lt({variable},{_number(end.time_seconds)}),"
            f"{segment},{expression}))"
        )
    return expression


def track_expression(track: KeyframeTrack, *, variable: str = "t") -> str:
    """Return a scalar ffmpeg expression sampling *track*.

    ``variable`` is the time expression the filters expose; use ``t`` for
    ``overlay``, ``rotate`` and ``geq``, or ``on / fps`` for ``zoompan``.
    """
    return _component_expression(track, variable, None)


def pair_expression(
    track: KeyframeTrack,
    *,
    variable: str = "t",
) -> tuple[str, str]:
    """Return the ``(x, y)`` ffmpeg expressions sampling a pair *track*."""
    return (
        _component_expression(track, variable, 0),
        _component_expression(track, variable, 1),
    )

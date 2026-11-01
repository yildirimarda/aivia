"""Keyframes and easing for animated layer properties.

A composition never stores animation as a function: it lists keyframes with a
time and a value, and this module turns that list into a pure function of time.
Sampling is deterministic and depends only on the keyframes, so two renders of
the same composition evaluate every frame to the same numbers.

Position and scale may be scalar or a ``(x, y)`` pair. Rotation is expressed in
degrees and opacity is a fraction between zero and one. Values before the first
keyframe and after the last one are held, never extrapolated.
"""

from __future__ import annotations

import builtins
import math
import re
from bisect import bisect_right
from collections.abc import Callable, Iterator, Mapping, Sequence
from enum import StrEnum
from types import MappingProxyType
from typing import Any, cast

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

__all__ = [
    "EASING_FUNCTIONS",
    "AnimatedProperty",
    "AnimatedValue",
    "Easing",
    "EasingFunction",
    "Keyframe",
    "KeyframeAnimation",
    "KeyframeTrack",
    "LayerAnimation",
    "LayerKeyframes",
    "apply_easing",
    "ease",
    "ease_in",
    "ease_in_cubic",
    "ease_in_expo",
    "ease_in_out",
    "ease_in_out_cubic",
    "ease_in_out_expo",
    "ease_in_out_quad",
    "ease_in_out_sine",
    "ease_in_quad",
    "ease_in_sine",
    "ease_out",
    "ease_out_cubic",
    "ease_out_expo",
    "ease_out_quad",
    "ease_out_sine",
    "keyframe_track",
    "keyframes_for_layer",
    "layer_keyframes",
    "linear",
    "sample_layer",
    "smoothstep",
    "step",
]

type AnimatedValue = float | tuple[float, float]
"""A scalar value or an ``(x, y)`` pair, depending on the property."""

type EasingFunction = Callable[[float], float]
"""A function mapping progress in ``[0, 1]`` to eased progress."""


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def _linear(progress: float) -> float:
    return _clamp01(progress)


def _step(progress: float) -> float:
    return 1.0 if progress >= 1.0 else 0.0


def _smoothstep(progress: float) -> float:
    value = _clamp01(progress)
    return value * value * (3.0 - 2.0 * value)


def _ease_in_sine(progress: float) -> float:
    value = _clamp01(progress)
    return 1.0 - math.cos(value * math.pi / 2.0)


def _ease_out_sine(progress: float) -> float:
    value = _clamp01(progress)
    return math.sin(value * math.pi / 2.0)


def _ease_in_out_sine(progress: float) -> float:
    value = _clamp01(progress)
    return -(math.cos(math.pi * value) - 1.0) / 2.0


def _ease_in_quad(progress: float) -> float:
    value = _clamp01(progress)
    return value * value


def _ease_out_quad(progress: float) -> float:
    value = _clamp01(progress)
    return value * (2.0 - value)


def _ease_in_out_quad(progress: float) -> float:
    value = _clamp01(progress)
    if value < 0.5:
        return 2.0 * value * value
    return 1.0 - (-2.0 * value + 2.0) ** 2 / 2.0


def _ease_in_cubic(progress: float) -> float:
    value = _clamp01(progress)
    return value * value * value


def _ease_out_cubic(progress: float) -> float:
    value = _clamp01(progress)
    return 1.0 - (1.0 - value) ** 3


def _ease_in_out_cubic(progress: float) -> float:
    value = _clamp01(progress)
    if value < 0.5:
        return 4.0 * value * value * value
    return 1.0 - (-2.0 * value + 2.0) ** 3 / 2.0


def _ease_in_expo(progress: float) -> float:
    value = _clamp01(progress)
    if value == 0.0:
        return 0.0
    return float(2.0 ** (10.0 * value - 10.0))


def _ease_out_expo(progress: float) -> float:
    value = _clamp01(progress)
    if value == 1.0:
        return 1.0
    return float(1.0 - 2.0 ** (-10.0 * value))


def _ease_in_out_expo(progress: float) -> float:
    value = _clamp01(progress)
    if value == 0.0:
        return 0.0
    if value == 1.0:
        return 1.0
    if value < 0.5:
        return float(2.0 ** (20.0 * value - 10.0) / 2.0)
    return float((2.0 - 2.0 ** (-20.0 * value + 10.0)) / 2.0)


class Easing(StrEnum):
    """The named interpolation curves a keyframe can use."""

    LINEAR = "linear"
    STEP = "step"
    SMOOTHSTEP = "smoothstep"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    EASE_IN_SINE = "ease_in_sine"
    EASE_OUT_SINE = "ease_out_sine"
    EASE_IN_OUT_SINE = "ease_in_out_sine"
    EASE_IN_QUAD = "ease_in_quad"
    EASE_OUT_QUAD = "ease_out_quad"
    EASE_IN_OUT_QUAD = "ease_in_out_quad"
    EASE_IN_CUBIC = "ease_in_cubic"
    EASE_OUT_CUBIC = "ease_out_cubic"
    EASE_IN_OUT_CUBIC = "ease_in_out_cubic"
    EASE_IN_EXPO = "ease_in_expo"
    EASE_OUT_EXPO = "ease_out_expo"
    EASE_IN_OUT_EXPO = "ease_in_out_expo"

    @classmethod
    def _missing_(cls, value: Any) -> Easing | None:
        if not isinstance(value, str):
            return None
        key = _compact(value)
        aliases = {
            "ease": cls.EASE_IN_OUT,
            "easeinout": cls.EASE_IN_OUT,
            "easein": cls.EASE_IN,
            "easeout": cls.EASE_OUT,
            "in": cls.EASE_IN,
            "out": cls.EASE_OUT,
            "inout": cls.EASE_IN_OUT,
            "smooth": cls.SMOOTHSTEP,
            "smoothstep": cls.SMOOTHSTEP,
            "quadin": cls.EASE_IN_QUAD,
            "quadout": cls.EASE_OUT_QUAD,
            "quadinout": cls.EASE_IN_OUT_QUAD,
            "cubicin": cls.EASE_IN_CUBIC,
            "cubicout": cls.EASE_OUT_CUBIC,
            "cubicinout": cls.EASE_IN_OUT_CUBIC,
            "expoin": cls.EASE_IN_EXPO,
            "expoout": cls.EASE_OUT_EXPO,
            "expoinout": cls.EASE_IN_OUT_EXPO,
        }
        if key in aliases:
            return aliases[key]
        for member in cls:
            if _compact(member.value) == key:
                return member
        return None

    @builtins.property
    def function(self) -> EasingFunction:
        """The function implementing this easing."""
        return _EASING_FUNCTIONS[self]

    def apply(self, progress: float) -> float:
        """Ease *progress* with this curve."""
        return apply_easing(self, progress)


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.strip().lower())


_EASING_FUNCTIONS: dict[Easing, EasingFunction] = {
    Easing.LINEAR: _linear,
    Easing.STEP: _step,
    Easing.SMOOTHSTEP: _smoothstep,
    Easing.EASE_IN: _ease_in_quad,
    Easing.EASE_OUT: _ease_out_quad,
    Easing.EASE_IN_OUT: _ease_in_out_quad,
    Easing.EASE_IN_SINE: _ease_in_sine,
    Easing.EASE_OUT_SINE: _ease_out_sine,
    Easing.EASE_IN_OUT_SINE: _ease_in_out_sine,
    Easing.EASE_IN_QUAD: _ease_in_quad,
    Easing.EASE_OUT_QUAD: _ease_out_quad,
    Easing.EASE_IN_OUT_QUAD: _ease_in_out_quad,
    Easing.EASE_IN_CUBIC: _ease_in_cubic,
    Easing.EASE_OUT_CUBIC: _ease_out_cubic,
    Easing.EASE_IN_OUT_CUBIC: _ease_in_out_cubic,
    Easing.EASE_IN_EXPO: _ease_in_expo,
    Easing.EASE_OUT_EXPO: _ease_out_expo,
    Easing.EASE_IN_OUT_EXPO: _ease_in_out_expo,
}

EASING_FUNCTIONS: Mapping[Easing, EasingFunction] = MappingProxyType(_EASING_FUNCTIONS)
"""Read-only mapping from every easing name to its function."""


def apply_easing(
    easing: Easing | str | EasingFunction,
    progress: float,
) -> float:
    """Return the eased *progress* using *easing*.

    Progress is clamped to ``[0, 1]`` before the curve is applied. Passing a
    callable is allowed so a caller can supply a custom curve without adding a
    name to :class:`Easing`.
    """
    if isinstance(progress, bool) or not isinstance(progress, (int, float)):
        raise TypeError("progress must be a number")
    value = float(progress)
    if not math.isfinite(value):
        raise ValueError("progress must be finite")
    if callable(easing) and not isinstance(easing, str):
        function = easing
    else:
        function = _EASING_FUNCTIONS[Easing(easing)]
    return float(function(_clamp01(value)))


ease = apply_easing
"""Alias for :func:`apply_easing`."""

linear = _linear
step = _step
smoothstep = _smoothstep
ease_in = _ease_in_quad
ease_out = _ease_out_quad
ease_in_out = _ease_in_out_quad
ease_in_sine = _ease_in_sine
ease_out_sine = _ease_out_sine
ease_in_out_sine = _ease_in_out_sine
ease_in_quad = _ease_in_quad
ease_out_quad = _ease_out_quad
ease_in_out_quad = _ease_in_out_quad
ease_in_cubic = _ease_in_cubic
ease_out_cubic = _ease_out_cubic
ease_in_out_cubic = _ease_in_out_cubic
ease_in_expo = _ease_in_expo
ease_out_expo = _ease_out_expo
ease_in_out_expo = _ease_in_out_expo


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _coerce_value(value: Any) -> AnimatedValue:
    if isinstance(value, (str, bytes)):
        raise ValueError("keyframe value must be a number or a pair of numbers")
    if isinstance(value, Sequence):
        components = tuple(_finite(item, "keyframe value") for item in value)
        if len(components) != 2:
            raise ValueError("a keyframe pair must have exactly two components")
        return components[0], components[1]
    return _finite(value, "keyframe value")


class Keyframe(BaseModel):
    """One value of one property at an exact time."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    time_seconds: float = Field(
        ge=0.0,
        validation_alias=AliasChoices(
            "time_seconds", "time", "t", "at", "offset_seconds", "offset", "start"
        ),
    )
    value: AnimatedValue = Field(
        validation_alias=AliasChoices("value", "v", "to", "target")
    )
    easing: Easing = Field(
        default=Easing.LINEAR,
        validation_alias=AliasChoices("easing", "ease", "timing", "curve"),
    )

    @field_validator("time_seconds", mode="before")
    @classmethod
    def _validate_time(cls, value: Any) -> Any:
        return _finite(value, "keyframe time")

    @field_validator("value", mode="before")
    @classmethod
    def _validate_value(cls, value: Any) -> Any:
        return _coerce_value(value)

    @builtins.property
    def time(self) -> float:
        """Alias for :attr:`time_seconds`."""
        return self.time_seconds

    @builtins.property
    def at(self) -> float:
        """Alias for :attr:`time_seconds`."""
        return self.time_seconds


class AnimatedProperty(StrEnum):
    """The layer properties that can be animated."""

    POSITION = "position"
    SCALE = "scale"
    ROTATION = "rotation"
    OPACITY = "opacity"

    @classmethod
    def _missing_(cls, value: Any) -> AnimatedProperty | None:
        if not isinstance(value, str):
            return None
        aliases = {
            "pos": cls.POSITION,
            "translate": cls.POSITION,
            "translation": cls.POSITION,
            "move": cls.POSITION,
            "offset": cls.POSITION,
            "zoom": cls.SCALE,
            "size": cls.SCALE,
            "rotate": cls.ROTATION,
            "angle": cls.ROTATION,
            "turn": cls.ROTATION,
            "alpha": cls.OPACITY,
            "transparency": cls.OPACITY,
            "fade": cls.OPACITY,
        }
        return aliases.get(_compact(value))

    @builtins.property
    def is_pair(self) -> bool:
        """Whether the property is normally a pair of numbers."""
        return self in (AnimatedProperty.POSITION, AnimatedProperty.SCALE)


def _coerce_frame(raw: Any) -> Any:
    if isinstance(raw, Keyframe):
        return raw
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = list(raw)
        if len(values) == 2:
            return {"time": values[0], "value": values[1]}
        if len(values) == 3:
            return {"time": values[0], "value": values[1], "easing": values[2]}
    return raw


def _coerce_frames(raw: Any) -> Any:
    if isinstance(raw, Mapping):
        frames: list[dict[str, Any]] = []
        for time, value in raw.items():
            if isinstance(value, Mapping) and any(
                key in value for key in ("value", "v", "to", "target")
            ):
                frame = dict(value)
                frame.setdefault("time", time)
            else:
                frame = {"time": time, "value": value}
            frames.append(frame)
        return frames
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [_coerce_frame(item) for item in raw]
    return raw


def _coerce_track(prop: AnimatedProperty, raw: Any) -> dict[str, Any]:
    if isinstance(raw, KeyframeTrack):
        return raw.model_dump()
    if isinstance(raw, Mapping):
        values = dict(raw)
        if not any(key in values for key in ("keyframes", "frames", "keys", "values")):
            values = {"keyframes": values}
    else:
        values = {"keyframes": raw}
    values.setdefault("property", prop)
    return values


class KeyframeTrack(BaseModel):
    """An ordered list of keyframes for one animated property.

    A keyframe's easing controls the segment that starts at that keyframe. The
    first and last values are held outside the keyframe range.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    property: AnimatedProperty | None = Field(
        default=None,
        validation_alias=AliasChoices("property", "name", "prop", "kind"),
    )
    keyframes: tuple[Keyframe, ...] = Field(
        min_length=1,
        validation_alias=AliasChoices("keyframes", "frames", "keys", "values"),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_shorthand(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            values = dict(data)
            property_value = None
            for key in ("property", "name", "prop", "kind"):
                if key in values:
                    property_value = values[key]
                    break
            frames = None
            for key in ("keyframes", "frames", "keys", "values"):
                if key in values:
                    frames = values[key]
                    break
            if property_value is not None:
                values["property"] = AnimatedProperty(property_value)
            if frames is not None:
                values["keyframes"] = _coerce_frames(frames)
            return values
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            return {"keyframes": _coerce_frames(data)}
        return data

    @model_validator(mode="after")
    def _validate_frames(self) -> KeyframeTrack:
        frames = self.keyframes
        previous: float | None = None
        shape: int | None = None
        for frame in frames:
            if previous is not None and frame.time_seconds <= previous:
                raise ValueError("keyframe times must be strictly ascending")
            previous = frame.time_seconds
            current_shape = 2 if isinstance(frame.value, tuple) else 1
            if shape is not None and current_shape != shape:
                raise ValueError(
                    "all keyframes of a property must use the same value shape"
                )
            shape = current_shape
            self._validate_frame_value(frame.value)
        return self

    def _validate_frame_value(self, value: AnimatedValue) -> None:
        if self.property is None:
            return
        if self.property is AnimatedProperty.POSITION:
            if not isinstance(value, tuple):
                raise ValueError("position keyframes need an (x, y) pair")
        elif self.property is AnimatedProperty.SCALE:
            components = value if isinstance(value, tuple) else (value,)
            if any(component < 0.0 for component in components):
                raise ValueError("scale keyframe values must not be negative")
        elif self.property is AnimatedProperty.ROTATION:
            if isinstance(value, tuple):
                raise ValueError("rotation keyframes need a single number")
        elif self.property is AnimatedProperty.OPACITY:
            if isinstance(value, tuple):
                raise ValueError("opacity keyframes need a single number")
            assert isinstance(value, float)
            if not 0.0 <= value <= 1.0:
                raise ValueError("opacity keyframe values must be between 0 and 1")

    @builtins.property
    def name(self) -> AnimatedProperty | None:
        """Alias for :attr:`property`."""
        return self.property

    @builtins.property
    def prop(self) -> AnimatedProperty | None:
        """Alias for :attr:`property`."""
        return self.property

    @builtins.property
    def times(self) -> tuple[float, ...]:
        """The keyframe times in ascending order."""
        return tuple(frame.time_seconds for frame in self.keyframes)

    @builtins.property
    def values(self) -> tuple[AnimatedValue, ...]:
        """The keyframe values in time order."""
        return tuple(frame.value for frame in self.keyframes)

    @builtins.property
    def start_seconds(self) -> float:
        """The time of the first keyframe."""
        return self.keyframes[0].time_seconds

    @builtins.property
    def end_seconds(self) -> float:
        """The time of the last keyframe."""
        return self.keyframes[-1].time_seconds

    @builtins.property
    def duration_seconds(self) -> float:
        """The span between the first and last keyframe."""
        return self.end_seconds - self.start_seconds

    def sample(self, time_seconds: float) -> AnimatedValue:
        """Return the value at *time_seconds*.

        Times before the first keyframe return the first value and times after
        the last one return the last value.
        """
        time = _finite(time_seconds, "time")
        frames = self.keyframes
        if time <= frames[0].time_seconds:
            return frames[0].value
        if time >= frames[-1].time_seconds:
            return frames[-1].value

        index = bisect_right(self.times, time)
        start = frames[index - 1]
        end = frames[index]
        span = end.time_seconds - start.time_seconds
        progress = (time - start.time_seconds) / span
        eased = apply_easing(start.easing, progress)
        return _interpolate(start.value, end.value, eased)

    def at(self, time_seconds: float) -> AnimatedValue:
        """Alias for :meth:`sample`."""
        return self.sample(time_seconds)

    def value_at(self, time_seconds: float) -> AnimatedValue:
        """Alias for :meth:`sample`."""
        return self.sample(time_seconds)

    def evaluate(self, time_seconds: float) -> AnimatedValue:
        """Alias for :meth:`sample`."""
        return self.sample(time_seconds)

    def __call__(self, time_seconds: float) -> AnimatedValue:
        return self.sample(time_seconds)


def _interpolate(
    start: AnimatedValue,
    end: AnimatedValue,
    progress: float,
) -> AnimatedValue:
    if isinstance(start, tuple) and isinstance(end, tuple):
        return (
            start[0] + (end[0] - start[0]) * progress,
            start[1] + (end[1] - start[1]) * progress,
        )
    if isinstance(start, tuple) or isinstance(end, tuple):
        raise ValueError("cannot interpolate a scalar and a pair")
    return start + (end - start) * progress


def keyframe_track(
    property: AnimatedProperty | str,
    keyframes: Any,
) -> KeyframeTrack:
    """Build a :class:`KeyframeTrack` from a property and raw keyframes."""
    return KeyframeTrack.model_validate(
        {"property": AnimatedProperty(property), "keyframes": keyframes}
    )


class LayerKeyframes(BaseModel):
    """Every animated property of one layer, addressed by property name."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    position: KeyframeTrack | None = None
    scale: KeyframeTrack | None = None
    rotation: KeyframeTrack | None = None
    opacity: KeyframeTrack | None = None

    @model_validator(mode="before")
    @classmethod
    def _prepare_tracks(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for container in ("keyframes", "animation", "animations"):
            grouped = values.pop(container, None)
            if isinstance(grouped, Mapping):
                for key, frames in grouped.items():
                    values.setdefault(str(key), frames)

        prepared: dict[str, Any] = {}
        for key, raw in values.items():
            try:
                prop = AnimatedProperty(str(key))
            except ValueError:
                prepared[str(key)] = raw
                continue
            if raw is None:
                prepared[prop.value] = None
            elif isinstance(raw, KeyframeTrack):
                if raw.property is not prop:
                    raw = KeyframeTrack.model_validate(
                        {"property": prop, "keyframes": raw.keyframes}
                    )
                prepared[prop.value] = raw
            else:
                prepared[prop.value] = _coerce_track(prop, raw)
        return prepared

    @model_validator(mode="after")
    def _check_track_properties(self) -> LayerKeyframes:
        for prop in AnimatedProperty:
            track = getattr(self, prop.value)
            if track is not None and track.property is not prop:
                raise ValueError(f"{prop.value} track is for {track.property!s}")
        return self

    @builtins.property
    def tracks(self) -> dict[AnimatedProperty, KeyframeTrack]:
        """The tracks that are present, keyed by property."""
        return {
            prop: getattr(self, prop.value)
            for prop in AnimatedProperty
            if getattr(self, prop.value) is not None
        }

    def track(self, property: AnimatedProperty | str) -> KeyframeTrack | None:
        """Return the track for *property*, or ``None`` when absent."""
        return cast(
            KeyframeTrack | None,
            getattr(self, AnimatedProperty(property).value),
        )

    def sample(self, time_seconds: float) -> dict[AnimatedProperty, AnimatedValue]:
        """Sample every animated property at *time_seconds*."""
        return {prop: track.sample(time_seconds) for prop, track in self.tracks.items()}

    def at(self, time_seconds: float) -> dict[AnimatedProperty, AnimatedValue]:
        """Alias for :meth:`sample`."""
        return self.sample(time_seconds)

    def sample_property(
        self,
        property: AnimatedProperty | str,
        time_seconds: float,
    ) -> AnimatedValue | None:
        """Sample one property, returning ``None`` when it is not animated."""
        track = self.track(property)
        return None if track is None else track.sample(time_seconds)

    def value(self, property: AnimatedProperty | str, time_seconds: float) -> Any:
        """Alias for :meth:`sample_property`."""
        return self.sample_property(property, time_seconds)

    def __getitem__(self, property: AnimatedProperty | str) -> KeyframeTrack:
        track = self.track(property)
        if track is None:
            raise KeyError(str(property))
        return track

    def __contains__(self, property: object) -> bool:
        try:
            return self.track(cast(Any, property)) is not None
        except ValueError:
            return False

    @classmethod
    def from_layer(cls, layer: Any) -> LayerKeyframes:
        """Read the keyframes declared on a layer.

        Both the grouped ``keyframes``/``animation``/``animations`` containers
        and the per property ``<property>_keyframes``/``<property>_animation``
        spellings are accepted, matching composition validation.
        """
        return cls.model_validate(_layer_keyframe_data(layer))


KeyframeAnimation = LayerKeyframes
"""Alias for :class:`LayerKeyframes`."""

LayerAnimation = LayerKeyframes
"""Alias for :class:`LayerKeyframes`."""


def _layer_value(layer: Any, key: str) -> Any:
    extras = getattr(layer, "model_extra", None) or {}
    if key in extras:
        return extras[key]
    return getattr(layer, key, None)


def _layer_keyframes(layer: Any) -> Iterator[tuple[str, Any]]:
    for container in ("keyframes", "animation", "animations"):
        raw = _layer_value(layer, container)
        if isinstance(raw, Mapping):
            for prop, frames in raw.items():
                yield str(prop), frames
        elif raw is not None and container == "keyframes":
            yield "value", raw

    extras = getattr(layer, "model_extra", None) or {}
    for key, raw in extras.items():
        if key.endswith("_keyframes"):
            yield key[: -len("_keyframes")], raw
        elif key.endswith("_animation"):
            yield key[: -len("_animation")], raw


def _layer_keyframe_data(layer: Any) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for name, frames in _layer_keyframes(layer):
        try:
            prop = AnimatedProperty(name)
        except ValueError:
            continue
        data[prop.value] = {"property": prop, "keyframes": frames}
    return data


def keyframes_for_layer(layer: Any) -> LayerKeyframes:
    """Return the parsed keyframes declared on *layer*."""
    return LayerKeyframes.from_layer(layer)


layer_keyframes = keyframes_for_layer
"""Alias for :func:`keyframes_for_layer`."""


def sample_layer(
    layer: Any,
    time_seconds: float,
) -> dict[AnimatedProperty, AnimatedValue]:
    """Sample every keyframed property of *layer* at *time_seconds*."""
    return keyframes_for_layer(layer).sample(time_seconds)

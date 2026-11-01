"""Deterministic cut, crossfade and slide transitions.

A transition describes how one visual replaces another. The same model is used
for two related jobs:

* between two complete frames, through :func:`render_transition`; and
* as the entrance of a layer, through :func:`compose_transition`.

Both paths are pure functions of their inputs. No transition reads the clock,
uses randomness or calls a provider, so sampling the same time twice always
produces the same pixels.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from PIL import Image
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .errors import RenderError
from .keyframes import Easing, apply_easing

if TYPE_CHECKING:
    from .schema import Composition, Effect, Layer

__all__ = [
    "TRANSITION_VERSION",
    "SlideDirection",
    "Transition",
    "TransitionConfig",
    "TransitionError",
    "TransitionSpec",
    "TransitionType",
    "apply_transition",
    "compose_transition",
    "crossfade",
    "cut",
    "dissolve",
    "fade",
    "layer_transition_for",
    "render_transition",
    "slide",
    "slide_transition",
    "transition_from_effect",
    "transition_progress",
    "transition_window",
]

TRANSITION_VERSION = "1"
"""The version of the transition model."""


class TransitionError(RenderError):
    """A transition could not be built or rendered."""

    code = "transition"


def _compact(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


class TransitionType(StrEnum):
    """The kinds of transition the renderer knows how to draw."""

    CUT = "cut"
    CROSSFADE = "crossfade"
    SLIDE = "slide"

    @classmethod
    def _missing_(cls, value: Any) -> TransitionType | None:
        if not isinstance(value, str):
            return None
        aliases = {
            "cut": cls.CUT,
            "hardcut": cls.CUT,
            "none": cls.CUT,
            "instant": cls.CUT,
            "crossfade": cls.CROSSFADE,
            "fade": cls.CROSSFADE,
            "dissolve": cls.CROSSFADE,
            "mix": cls.CROSSFADE,
            "xfade": cls.CROSSFADE,
            "slide": cls.SLIDE,
            "push": cls.SLIDE,
            "wipe": cls.SLIDE,
        }
        return aliases.get(_compact(value))


class SlideDirection(StrEnum):
    """The direction a sliding visual travels."""

    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"

    @classmethod
    def _missing_(cls, value: Any) -> SlideDirection | None:
        if not isinstance(value, str):
            return None
        aliases = {
            "left": cls.LEFT,
            "ltr": cls.LEFT,
            "right": cls.RIGHT,
            "rtl": cls.RIGHT,
            "up": cls.UP,
            "top": cls.UP,
            "down": cls.DOWN,
            "bottom": cls.DOWN,
        }
        return aliases.get(_compact(value))


def _parse_declaration(value: Any) -> Any:
    """Expand string shorthands such as ``slide_left`` into model data."""
    if not isinstance(value, str):
        return value
    key = _compact(value)
    if key in {"cut", "hardcut", "none", "instant"}:
        return {"type": TransitionType.CUT}
    if key in {"crossfade", "fade", "dissolve", "mix", "xfade"}:
        return {"type": TransitionType.CROSSFADE}
    if key == "slide":
        return {"type": TransitionType.SLIDE}
    if key.startswith("slide"):
        direction = key.removeprefix("slide")
        if direction in {"left", "right", "up", "down"}:
            return {"type": TransitionType.SLIDE, "direction": direction}
    if key in {"left", "right", "up", "down"}:
        return {"type": TransitionType.SLIDE, "direction": key}
    return {"type": value}


class Transition(BaseModel):
    """One transition between two visuals.

    ``duration_seconds`` is the time spent changing from the outgoing visual to
    the incoming one. A cut is instantaneous and may use a zero duration.
    ``direction`` applies to slides and names the direction the incoming visual
    travels, matching the text animation presets.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    type: TransitionType = Field(
        validation_alias=AliasChoices(
            "type",
            "kind",
            "preset",
            "transition",
            "style",
        ),
    )
    duration_seconds: float = Field(
        default=0.5,
        ge=0.0,
        validation_alias=AliasChoices(
            "duration_seconds",
            "duration",
            "seconds",
            "length_seconds",
        ),
    )
    direction: SlideDirection = Field(
        default=SlideDirection.LEFT,
        validation_alias=AliasChoices("direction", "from", "side"),
    )
    easing: Easing = Field(
        default=Easing.LINEAR,
        validation_alias=AliasChoices("easing", "curve"),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_string_shorthand(cls, data: Any) -> Any:
        if isinstance(data, bool):
            return {"type": TransitionType.CROSSFADE if data else TransitionType.CUT}
        if isinstance(data, Mapping):
            values = dict(data)
            for key in ("type", "kind", "preset", "transition", "style"):
                raw = values.get(key)
                if not isinstance(raw, str):
                    continue
                parsed = _parse_declaration(raw)
                if not isinstance(parsed, Mapping):
                    continue
                values["type"] = parsed.get("type", raw)
                if "direction" in parsed:
                    values.setdefault("direction", parsed["direction"])
                break
            return values
        return _parse_declaration(data)

    @field_validator("duration_seconds", mode="before")
    @classmethod
    def _validate_duration(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("duration_seconds must be a number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("duration_seconds must be finite")
        return number

    @field_validator("easing", mode="before")
    @classmethod
    def _validate_easing(cls, value: Any) -> Any:
        if value is None:
            return Easing.LINEAR
        return value

    @model_validator(mode="after")
    def _validate_required_duration(self) -> Transition:
        if self.type is not TransitionType.CUT and not self.duration_seconds:
            raise ValueError(
                "duration_seconds must be positive for a gradual transition"
            )
        return self

    @property
    def kind(self) -> TransitionType:
        """Alias for :attr:`type`."""
        return self.type

    @property
    def duration(self) -> float:
        """Alias for :attr:`duration_seconds`."""
        return self.duration_seconds

    @property
    def is_cut(self) -> bool:
        """Whether the transition replaces the visual instantly."""
        return self.type is TransitionType.CUT

    @property
    def is_crossfade(self) -> bool:
        """Whether the transition fades one visual into another."""
        return self.type is TransitionType.CROSSFADE

    @property
    def is_slide(self) -> bool:
        """Whether the incoming visual moves across the canvas."""
        return self.type is TransitionType.SLIDE

    def progress(self, time_seconds: float, start_seconds: float = 0.0) -> float:
        """Return the eased progress at *time_seconds* inside the transition."""
        if self.is_cut:
            return 1.0 if time_seconds >= start_seconds else 0.0
        span = self.duration_seconds
        if span <= 0.0:
            return 1.0
        raw = (time_seconds - start_seconds) / span
        return apply_easing(self.easing, raw)

    def offset_for(
        self,
        progress: float,
        size: tuple[int, int],
    ) -> tuple[float, float]:
        """Return the incoming visual's pixel offset at *progress*."""
        width, height = size
        remaining = 1.0 - apply_easing(self.easing, progress)
        if self.direction is SlideDirection.LEFT:
            return remaining * width, 0.0
        if self.direction is SlideDirection.RIGHT:
            return -remaining * width, 0.0
        if self.direction is SlideDirection.UP:
            return 0.0, remaining * height
        return 0.0, -remaining * height

    def expression_offset(self, size: tuple[int, int]) -> tuple[str, str]:
        """Return the slide offset as FFmpeg-friendly x and y expressions."""
        width, height = size
        if self.direction is SlideDirection.LEFT:
            return f"{width}", "0"
        if self.direction is SlideDirection.RIGHT:
            return f"-{width}", "0"
        if self.direction is SlideDirection.UP:
            return "0", f"{height}"
        return "0", f"-{height}"


TransitionSpec = Transition
TransitionConfig = Transition


def cut(*, duration: float = 0.0) -> Transition:
    """Build an instant cut."""
    return Transition(type=TransitionType.CUT, duration_seconds=duration)


def crossfade(
    *, duration: float = 0.5, easing: Easing | str = Easing.LINEAR
) -> Transition:
    """Build a crossfade over *duration* seconds."""
    return Transition(
        type=TransitionType.CROSSFADE,
        duration_seconds=duration,
        easing=Easing(easing),
    )


def slide_transition(
    direction: SlideDirection | str = SlideDirection.LEFT,
    *,
    duration: float = 0.5,
    easing: Easing | str = Easing.LINEAR,
) -> Transition:
    """Build a slide in *direction* over *duration* seconds."""
    return Transition(
        type=TransitionType.SLIDE,
        direction=SlideDirection(direction),
        duration_seconds=duration,
        easing=Easing(easing),
    )


fade = crossfade
"""Alias for :func:`crossfade`."""

dissolve = crossfade
"""Alias for :func:`crossfade`."""

slide = slide_transition
"""Alias for :func:`slide_transition`."""


def _rgba(image: Image.Image, name: str) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TransitionError(f"{name} must be a Pillow image")
    return image if image.mode == "RGBA" else image.convert("RGBA")


def _check_frames(before: Image.Image, after: Image.Image) -> None:
    if before.size != after.size:
        raise TransitionError("transition frames must have the same size")


def _opacity(image: Image.Image, value: float) -> Image.Image:
    value = max(0.0, min(1.0, value))
    if value >= 1.0:
        return image
    alpha = image.getchannel("A").point(lambda channel: round(channel * value))
    result = image.copy()
    result.putalpha(alpha)
    return result


def compose_transition(
    base: Image.Image,
    incoming: Image.Image,
    transition: Transition | Mapping[str, Any],
    progress: float,
) -> Image.Image:
    """Composite *incoming* over *base* as an entrance transition.

    ``progress`` runs from zero (only the base is visible) to one (the incoming
    visual is fully visible). The result is a new RGBA image; neither input is
    modified.
    """
    value = (
        transition
        if isinstance(transition, Transition)
        else Transition.model_validate(transition)
    )
    before = _rgba(base, "base")
    after = _rgba(incoming, "incoming")
    _check_frames(before, after)
    amount = max(0.0, min(1.0, float(progress)))
    result = before.copy()
    if value.is_cut:
        result.alpha_composite(after)
        return result
    if value.is_crossfade:
        result.alpha_composite(_opacity(after, amount))
        return result
    dx, dy = value.offset_for(amount, after.size)
    result.alpha_composite(after, dest=(round(dx), round(dy)))
    return result


def render_transition(
    before: Image.Image,
    after: Image.Image,
    transition: Transition | Mapping[str, Any],
    progress: float = 1.0,
) -> Image.Image:
    """Render one complete frame changing from *before* to *after*.

    A cut selects the incoming frame as soon as the transition starts. A
    crossfade blends the two complete frames. A slide moves the incoming frame
    across the outgoing one.
    """
    value = (
        transition
        if isinstance(transition, Transition)
        else Transition.model_validate(transition)
    )
    outgoing = _rgba(before, "before")
    incoming = _rgba(after, "after")
    _check_frames(outgoing, incoming)
    if value.is_cut:
        return incoming.copy()
    amount = max(0.0, min(1.0, float(progress)))
    if value.is_crossfade:
        return Image.blend(outgoing, incoming, amount)
    result = outgoing.copy()
    dx, dy = value.offset_for(amount, incoming.size)
    result.alpha_composite(incoming, dest=(round(dx), round(dy)))
    return result


apply_transition = render_transition
"""Alias for :func:`render_transition`."""


def _coerce_transition(value: Any) -> Transition | None:
    if value is None or value is False:
        return None
    if isinstance(value, Transition):
        return value
    if isinstance(value, (str, bool, Mapping)):
        return Transition.model_validate(value)
    raise TransitionError(f"unsupported transition declaration {value!r}")


def layer_transition_for(layer: Layer | Any) -> Transition | None:
    """Return the entrance transition declared on *layer*, or ``None``."""
    extras = getattr(layer, "model_extra", None) or {}
    for name in ("transition", "transition_in", "enter_transition", "in_transition"):
        if name in extras:
            return _coerce_transition(extras[name])
        value = getattr(layer, name, None)
        if value is not None:
            return _coerce_transition(value)
    return None


def transition_window(
    layer: Layer | Any,
    transition: Transition | None = None,
) -> tuple[float, float]:
    """Return the ``(start, end)`` seconds of a layer's entrance transition."""
    value = transition or layer_transition_for(layer)
    start = float(getattr(layer, "start_seconds", 0.0))
    if value is None:
        return start, start
    if value.is_cut:
        return start, start
    return start, start + value.duration_seconds


def transition_progress(
    layer: Layer | Any,
    time_seconds: float,
    transition: Transition | None = None,
) -> float:
    """Return the eased entrance progress of *layer* at *time_seconds*."""
    value = transition or layer_transition_for(layer)
    start = float(getattr(layer, "start_seconds", 0.0))
    if value is None:
        return 1.0
    return value.progress(time_seconds, start)


_TRANSITION_EFFECTS = frozenset(
    {
        "transition",
        "transitions",
        "cut",
        "crossfade",
        "fade",
        "dissolve",
        "mix",
        "xfade",
        "slide",
        "slideleft",
        "slideright",
        "slideup",
        "slidedown",
    }
)


def transition_from_effect(effect: Effect | Any) -> Transition | None:
    """Return the transition declared by *effect*, or ``None`` for another effect.

    The effect type may be a transition name directly (``crossfade``) or the
    generic ``transition`` with a ``type`` in its params. Unknown effect types
    are ignored so other effects keep working.
    """
    raw_type = getattr(effect, "type", "")
    key = _compact(raw_type) if isinstance(raw_type, str) else ""
    if key not in _TRANSITION_EFFECTS:
        return None
    params = dict(getattr(effect, "params", {}) or {})
    if key in {"transition", "transitions"}:
        raw = params.pop("type", params.pop("kind", params.pop("preset", None)))
        if raw is None:
            identifier = getattr(effect, "id", "") or raw_type
            raise TransitionError(f"effect {identifier!r} has no transition type")
        params["type"] = raw
    else:
        params.setdefault("type", raw_type)
    try:
        return Transition.model_validate(params)
    except ValueError as exc:
        raise TransitionError(
            f"effect {getattr(effect, 'id', '') or raw_type!r} has invalid transition "
            "parameters"
        ) from exc


def _layer_end(layer: Layer | Any, composition: Composition | Any) -> float:
    end = getattr(layer, "end", None)
    if end is None:
        end = float(getattr(composition, "duration_seconds", 0.0))
    return float(end)


def outgoing_transition_end(
    layer: Layer | Any,
    composition: Composition | Any,
) -> float | None:
    """Return how long *layer* must stay visible under a later transition.

    A slideshow often places the next slide exactly at the previous slide's end.
    Extending the outgoing layer through the incoming transition keeps a real
    crossfade or slide instead of revealing the background.
    """
    end = _layer_end(layer, composition)
    start = float(getattr(layer, "start_seconds", 0.0))
    extended: float | None = None
    for other in getattr(composition, "layers", ()):
        if other is layer:
            continue
        transition = layer_transition_for(other)
        if transition is None or transition.is_cut:
            continue
        other_start = float(getattr(other, "start_seconds", 0.0))
        window_end = other_start + transition.duration_seconds
        if other_start - 1e-9 <= end <= window_end + 1e-9 and start < other_start:
            extended = window_end if extended is None else max(extended, window_end)
    return extended

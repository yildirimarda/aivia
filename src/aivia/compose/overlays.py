"""Deterministic procedural overlays without external assets.

Confetti, sparkles, snow and bokeh are drawn from a small particle model whose
initial values come from a seeded pseudo random generator. The same seed, type
and canvas always produce the same frame at the same time, so an overlay is as
reproducible as any other composition layer and never needs a downloaded asset.

An overlay can be used directly through :func:`render_overlay`, or declared as
a composition effect whose type is one of the overlay names. Effects are applied
after the layers by the Pillow still renderer.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import Any, cast

from PIL import Image, ImageColor, ImageDraw, ImageFilter
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .errors import RenderError
from .schema import Composition, Effect

__all__ = [
    "OVERLAY_VERSION",
    "Overlay",
    "OverlayError",
    "OverlayType",
    "Particle",
    "apply_overlay_effects",
    "apply_overlays",
    "overlay_from_effect",
    "particles",
    "render_overlay",
]

OVERLAY_VERSION = "1"
"""The version of the procedural overlay model."""


class OverlayError(RenderError):
    """A procedural overlay could not be built or drawn."""

    code = "overlay"


class OverlayType(StrEnum):
    """The procedural overlays the renderer knows how to draw."""

    CONFETTI = "confetti"
    SPARKLES = "sparkles"
    SNOW = "snow"
    BOKEH = "bokeh"

    @classmethod
    def _missing_(cls, value: Any) -> OverlayType | None:
        if not isinstance(value, str):
            return None
        key = "".join(character for character in value.lower() if character.isalnum())
        aliases = {
            "confetti": cls.CONFETTI,
            "confettiburst": cls.CONFETTI,
            "paper": cls.CONFETTI,
            "party": cls.CONFETTI,
            "sparkle": cls.SPARKLES,
            "sparkles": cls.SPARKLES,
            "glitter": cls.SPARKLES,
            "stars": cls.SPARKLES,
            "star": cls.SPARKLES,
            "snow": cls.SNOW,
            "snowfall": cls.SNOW,
            "winter": cls.SNOW,
            "bokeh": cls.BOKEH,
            "bokehlights": cls.BOKEH,
            "lights": cls.BOKEH,
            "blurcircles": cls.BOKEH,
        }
        return aliases.get(key)


@dataclass(frozen=True, slots=True)
class _Preset:
    count: int
    colors: tuple[str, ...]
    min_size: float
    max_size: float
    fall: tuple[float, float]
    sway: tuple[float, float]
    rate: tuple[float, float]
    spin: tuple[float, float]
    alpha: tuple[float, float]


_PRESETS: dict[OverlayType, _Preset] = {
    OverlayType.CONFETTI: _Preset(
        count=90,
        colors=("#ff4d6d", "#ffd166", "#06d6a0", "#4cc9f0", "#c77dff"),
        min_size=0.006,
        max_size=0.018,
        fall=(0.12, 0.30),
        sway=(0.010, 0.040),
        rate=(0.30, 1.00),
        spin=(-240.0, 240.0),
        alpha=(0.75, 1.00),
    ),
    OverlayType.SPARKLES: _Preset(
        count=70,
        colors=("#ffffff", "#fff1a8", "#bde0fe", "#ffd6ff"),
        min_size=0.004,
        max_size=0.014,
        fall=(-0.015, 0.020),
        sway=(0.002, 0.010),
        rate=(0.60, 1.60),
        spin=(-20.0, 20.0),
        alpha=(0.40, 1.00),
    ),
    OverlayType.SNOW: _Preset(
        count=120,
        colors=("#ffffff", "#f4f8ff", "#e6f0ff"),
        min_size=0.0025,
        max_size=0.0090,
        fall=(0.045, 0.120),
        sway=(0.005, 0.030),
        rate=(0.20, 0.80),
        spin=(-10.0, 10.0),
        alpha=(0.60, 1.00),
    ),
    OverlayType.BOKEH: _Preset(
        count=28,
        colors=("#ffd6a5", "#a0c4ff", "#bdb2ff", "#caffbf", "#fdffb6"),
        min_size=0.040,
        max_size=0.130,
        fall=(-0.010, 0.015),
        sway=(0.005, 0.020),
        rate=(0.10, 0.40),
        spin=(0.0, 0.0),
        alpha=(0.08, 0.28),
    ),
}

_ALIASES = {
    "confetti": OverlayType.CONFETTI,
    "confettiburst": OverlayType.CONFETTI,
    "paper": OverlayType.CONFETTI,
    "party": OverlayType.CONFETTI,
    "sparkle": OverlayType.SPARKLES,
    "sparkles": OverlayType.SPARKLES,
    "glitter": OverlayType.SPARKLES,
    "stars": OverlayType.SPARKLES,
    "star": OverlayType.SPARKLES,
    "snow": OverlayType.SNOW,
    "snowfall": OverlayType.SNOW,
    "winter": OverlayType.SNOW,
    "bokeh": OverlayType.BOKEH,
    "bokehlights": OverlayType.BOKEH,
    "lights": OverlayType.BOKEH,
    "blurcircles": OverlayType.BOKEH,
}
_GENERIC_EFFECTS = frozenset({"overlay", "overlays", "particles", "proceduraloverlay"})


def _normalise(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


class Overlay(BaseModel):
    """A seeded particle overlay.

    ``count``, ``colors`` and the size range default per overlay type. Sizes are
    fractions of the canvas's shorter side, so the same overlay keeps its
    proportions at any resolution. ``speed`` scales the motion, ``wind`` adds a
    horizontal drift per second and ``opacity`` scales the whole overlay.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    type: OverlayType = Field(
        validation_alias=AliasChoices("type", "kind", "preset", "overlay", "style"),
    )
    seed: int = Field(
        default=0,
        validation_alias=AliasChoices("seed", "random_seed", "rng"),
    )
    count: int | None = Field(
        default=None,
        ge=0,
        le=5000,
        validation_alias=AliasChoices("count", "particles", "amount"),
    )
    colors: tuple[str, ...] | None = Field(
        default=None,
        validation_alias=AliasChoices("colors", "colours", "colour", "palette"),
    )
    min_size: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("min_size", "size_min", "smallest"),
    )
    max_size: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("max_size", "size_max", "largest"),
    )
    speed: float = Field(
        default=1.0,
        ge=0.0,
        validation_alias=AliasChoices("speed", "velocity", "motion"),
    )
    wind: float = Field(
        default=0.0,
        validation_alias=AliasChoices("wind", "drift", "horizontal_drift"),
    )
    opacity: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("opacity", "alpha"),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"type": data}
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        size = values.pop("size", None)
        if size is not None:
            if isinstance(size, (int, float)) and not isinstance(size, bool):
                values.setdefault("min_size", size)
                values.setdefault("max_size", size)
            elif (
                isinstance(size, Sequence)
                and not isinstance(size, (str, bytes))
                and len(size) == 2
            ):
                values.setdefault("min_size", size[0])
                values.setdefault("max_size", size[1])
            else:
                raise ValueError("size must be a number or a pair of numbers")
        if values.get("min_size") is not None and values.get("max_size") is None:
            values["max_size"] = values["min_size"]
        if values.get("max_size") is not None and values.get("min_size") is None:
            values["min_size"] = values["max_size"]
        return values

    @field_validator("seed", mode="before")
    @classmethod
    def _normalise_seed(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            return 0
        try:
            return int(stripped)
        except ValueError:
            digest = hashlib.sha256(stripped.encode("utf-8")).digest()
            return int.from_bytes(digest[:8], "big")

    @field_validator("colors", mode="before")
    @classmethod
    def _validate_colors(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("colors must be a non-empty string or list of colours")
        colors: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("colors must contain non-empty strings")
            colour = item.strip()
            try:
                ImageColor.getcolor(colour, "RGBA")
            except ValueError as exc:
                raise ValueError(f"unknown overlay colour {colour!r}") from exc
            colors.append(colour)
        return tuple(colors)

    @field_validator("min_size", "max_size", "speed", "wind", mode="before")
    @classmethod
    def _validate_number(cls, value: Any) -> Any:
        if value is None:
            return None
        return _finite(value, "overlay")

    @model_validator(mode="after")
    def _validate_size_range(self) -> Overlay:
        if (
            self.min_size is not None
            and self.max_size is not None
            and self.max_size < self.min_size
        ):
            raise ValueError("max_size must be greater than or equal to min_size")
        return self

    @property
    def preset(self) -> _Preset:
        """The defaults for this overlay type."""
        return _PRESETS[self.type]

    @property
    def particle_count(self) -> int:
        """The number of particles to draw."""
        return self.preset.count if self.count is None else self.count

    @property
    def palette(self) -> tuple[str, ...]:
        """The colours to choose from."""
        return self.preset.colors if self.colors is None else self.colors

    @property
    def size_range(self) -> tuple[float, float]:
        """The minimum and maximum particle size as canvas fractions."""
        return (
            self.preset.min_size if self.min_size is None else self.min_size,
            self.preset.max_size if self.max_size is None else self.max_size,
        )


@dataclass(frozen=True, slots=True)
class Particle:
    """One sampled particle at a point in time."""

    x: float
    y: float
    size: float
    color: tuple[int, int, int, int]
    rotation: float
    shape: int


@dataclass(frozen=True, slots=True)
class _Blueprint:
    x: float
    y: float
    size: float
    color: tuple[int, int, int, int]
    alpha: float
    rotation: float
    spin: float
    fall: float
    sway: float
    rate: float
    phase: float
    shape: int


def _rgba(value: str) -> tuple[int, int, int, int]:
    red, green, blue, alpha = cast(
        tuple[int, int, int, int], ImageColor.getcolor(value, "RGBA")
    )
    return int(red), int(green), int(blue), int(alpha)


@lru_cache(maxsize=128)
def _blueprints(
    overlay_type: OverlayType,
    seed: int,
    count: int,
    colors: tuple[str, ...],
    min_size: float,
    max_size: float,
) -> tuple[_Blueprint, ...]:
    preset = _PRESETS[overlay_type]
    rng = random.Random(seed)
    palette = tuple(_rgba(colour) for colour in colors)
    blueprints: list[_Blueprint] = []
    for _ in range(count):
        color = palette[rng.randrange(len(palette))]
        alpha = rng.uniform(*preset.alpha)
        blueprints.append(
            _Blueprint(
                x=rng.random(),
                y=rng.random(),
                size=rng.uniform(min_size, max_size),
                color=color,
                alpha=alpha,
                rotation=rng.uniform(0.0, 360.0),
                spin=rng.uniform(*preset.spin),
                fall=rng.uniform(*preset.fall),
                sway=rng.uniform(*preset.sway),
                rate=rng.uniform(*preset.rate),
                phase=rng.uniform(0.0, math.tau),
                shape=rng.randrange(3),
            )
        )
    return tuple(blueprints)


def _sample(
    overlay: Overlay,
    blueprint: _Blueprint,
    time_seconds: float,
) -> Particle:
    speed = overlay.speed
    phase = time_seconds * blueprint.rate * math.tau + blueprint.phase
    sway = math.sin(phase) * blueprint.sway
    drift = overlay.wind * time_seconds

    if overlay.type in (OverlayType.CONFETTI, OverlayType.SNOW):
        x = blueprint.x + drift + sway
        y = blueprint.y + blueprint.fall * speed * time_seconds
    elif overlay.type is OverlayType.SPARKLES:
        x = blueprint.x + drift + sway * 0.5
        y = blueprint.y + blueprint.fall * speed * time_seconds
        y += math.cos(phase) * blueprint.sway * 0.5
    else:
        x = blueprint.x + drift * 0.5 + sway
        y = blueprint.y + blueprint.fall * speed * time_seconds
        y += math.cos(phase) * blueprint.sway

    pulse = 1.0
    alpha = blueprint.alpha
    if overlay.type is OverlayType.SPARKLES:
        twinkle = 0.5 + 0.5 * math.sin(phase)
        alpha *= 0.25 + 0.75 * twinkle
        pulse = 0.55 + 0.45 * twinkle
    elif overlay.type is OverlayType.BOKEH:
        pulse = 0.85 + 0.15 * math.sin(phase)

    return Particle(
        x=x % 1.0,
        y=y % 1.0,
        size=blueprint.size * pulse,
        color=(
            blueprint.color[0],
            blueprint.color[1],
            blueprint.color[2],
            round(blueprint.color[3] * alpha * overlay.opacity),
        ),
        rotation=blueprint.rotation + blueprint.spin * time_seconds,
        shape=blueprint.shape,
    )


def particles(
    overlay: Overlay | Mapping[str, Any],
    time_seconds: float = 0.0,
) -> tuple[Particle, ...]:
    """Sample *overlay* at *time_seconds*.

    Positions are normalised to the canvas, so ``0.0`` is the top or left edge
    and ``1.0`` the bottom or right edge. Sampling is a pure function of the
    overlay and the time.
    """
    value = overlay if isinstance(overlay, Overlay) else Overlay.model_validate(overlay)
    time = _finite(time_seconds, "time_seconds")
    if time < 0.0:
        raise OverlayError("time_seconds must not be negative")
    min_size, max_size = value.size_range
    blueprints = _blueprints(
        value.type,
        value.seed,
        value.particle_count,
        value.palette,
        min_size,
        max_size,
    )
    return tuple(_sample(value, blueprint, time) for blueprint in blueprints)


def _rotated_rectangle(
    x: float,
    y: float,
    width: float,
    height: float,
    angle_degrees: float,
) -> list[tuple[float, float]]:
    angle = math.radians(angle_degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    corners = (
        (-width / 2.0, -height / 2.0),
        (width / 2.0, -height / 2.0),
        (width / 2.0, height / 2.0),
        (-width / 2.0, height / 2.0),
    )
    return [
        (x + dx * cosine - dy * sine, y + dx * sine + dy * cosine)
        for dx, dy in corners
    ]


def _star(
    x: float,
    y: float,
    radius: float,
    angle_degrees: float,
) -> list[tuple[float, float]]:
    inner = max(0.5, radius * 0.18)
    points: list[tuple[float, float]] = []
    for index in range(8):
        length = radius if index % 2 == 0 else inner
        angle = math.radians(angle_degrees + index * 45.0)
        points.append((x + math.cos(angle) * length, y + math.sin(angle) * length))
    return points


def render_overlay(
    overlay: Overlay | Mapping[str, Any],
    size: tuple[int, int],
    time_seconds: float = 0.0,
) -> Image.Image:
    """Draw *overlay* onto a transparent RGBA image of *size*.

    The result contains only the overlay, ready to be composited over a frame.
    ``size`` is ``(width, height)`` in pixels.
    """
    value = overlay if isinstance(overlay, Overlay) else Overlay.model_validate(overlay)
    width, height = size
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width < 1
        or height < 1
    ):
        raise OverlayError("size must be a pair of positive integers")
    frame = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(frame, "RGBA")
    scale = float(min(width, height))
    time = _finite(time_seconds, "time_seconds")
    if time < 0.0:
        raise OverlayError("time_seconds must not be negative")

    min_size, max_size = value.size_range
    blueprints = _blueprints(
        value.type,
        value.seed,
        value.particle_count,
        value.palette,
        min_size,
        max_size,
    )
    for blueprint in blueprints:
        particle = _sample(value, blueprint, time)
        radius = particle.size * scale
        if radius <= 0.0:
            continue
        x = particle.x * width
        y = particle.y * height
        if value.type is OverlayType.CONFETTI:
            if particle.shape == 0:
                points = _rotated_rectangle(
                    x,
                    y,
                    max(1.0, radius * 2.0),
                    max(1.0, radius * 0.7),
                    particle.rotation,
                )
                draw.polygon(points, fill=particle.color)
            elif particle.shape == 1:
                draw.ellipse(
                    (x - radius, y - radius * 0.6, x + radius, y + radius * 0.6),
                    fill=particle.color,
                )
            else:
                points = _rotated_rectangle(
                    x,
                    y,
                    max(1.0, radius * 1.2),
                    max(1.0, radius * 1.2),
                    particle.rotation + 45.0,
                )
                draw.polygon(points, fill=particle.color)
        elif value.type is OverlayType.SPARKLES:
            draw.polygon(
                _star(x, y, max(1.0, radius), particle.rotation),
                fill=particle.color,
            )
        elif value.type is OverlayType.SNOW:
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=particle.color,
            )
        else:
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=particle.color,
            )

    if value.type is OverlayType.BOKEH:
        frame = frame.filter(ImageFilter.GaussianBlur(max(1.0, scale * 0.012)))
    return frame


def overlay_from_effect(effect: Effect) -> Overlay | None:
    """Return the overlay declared by *effect*, or ``None`` for another effect.

    The effect type may be an overlay name directly (``snow``) or the generic
    ``overlay`` with a ``type``/``preset`` in its params. Unknown effect types
    are ignored so other effects keep working.
    """
    key = _normalise(effect.type)
    if key in _ALIASES:
        overlay_type = _ALIASES[key]
        params = dict(effect.params)
    elif key in _GENERIC_EFFECTS:
        params = dict(effect.params)
        raw = params.pop("type", params.pop("kind", params.pop("preset", None)))
        if raw is None:
            return None
        try:
            overlay_type = OverlayType(raw)
        except ValueError as exc:
            raise OverlayError(f"unknown overlay type {raw!r}") from exc
    else:
        return None

    data = dict(params)
    data["type"] = overlay_type
    try:
        return Overlay.model_validate(data)
    except ValueError as exc:
        raise OverlayError(
            f"effect {effect.id or effect.type!r} has invalid overlay parameters"
        ) from exc


def apply_overlay_effects(
    image: Image.Image,
    composition: Composition,
    time_seconds: float = 0.0,
) -> Image.Image:
    """Composite every active overlay effect over *image*.

    Effects are drawn after the layers, in declaration order. An effect without
    an end time lasts until the composition ends. Non-overlay effects are left
    to their own renderer and are ignored here.
    """
    time = _finite(time_seconds, "time_seconds")
    for effect in composition.effects:
        if not effect.enabled:
            continue
        end = (
            composition.duration_seconds
            if effect.end_seconds is None
            else effect.end_seconds
        )
        if time < effect.start_seconds or time >= end:
            continue
        overlay = overlay_from_effect(effect)
        if overlay is None:
            continue
        frame = render_overlay(overlay, image.size, time - effect.start_seconds)
        image.alpha_composite(frame)
    return image


apply_overlays = apply_overlay_effects
"""Alias for :func:`apply_overlay_effects`."""

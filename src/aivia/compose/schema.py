"""Declarative composition schema shared by every renderer.

A composition describes *what* to render, never *how*: a canvas, an exact
duration, a frame rate, ordered layers, audio tracks and effects. Rendering is
deterministic and performs no AI call or network request, so this module is
plain data and validation only.

The schema is versioned. :data:`CURRENT_COMPOSITION_SCHEMA_VERSION` is the
version written by this release; readers keep the stored version so a later
migration can upgrade older documents instead of guessing.

Audio track and effect objects carry a small common envelope. Layers share the
same envelope and add a concrete class per kind with the fields that kind needs.
Unknown fields are rejected on a concrete layer while the generic :class:`Layer`
envelope keeps them for forward compatibility.
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal, Self, cast

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from aivia.core.platforms import SafeArea

from .motion import Motion
from .subtitles import SubtitleTrack
from .text_animation import TextAnimation
from .transitions import Transition

__all__ = [
    "COMPOSITION_SCHEMA_VERSION",
    "CURRENT_COMPOSITION_SCHEMA_VERSION",
    "Anchor",
    "AnyLayer",
    "AudioTrack",
    "AudioTrackConfig",
    "AudioTrackRole",
    "AudioTrackSpec",
    "BaseLayer",
    "BlendMode",
    "Canvas",
    "CanvasConfig",
    "CanvasSpec",
    "Composition",
    "CompositionConfig",
    "CompositionSchema",
    "CompositionSpec",
    "Effect",
    "EffectConfig",
    "EffectSpec",
    "FaceBox",
    "FilterType",
    "FitMode",
    "Gradient",
    "GradientStop",
    "GradientType",
    "ImageFilter",
    "ImageLayer",
    "ImageMask",
    "Layer",
    "LayerAdapter",
    "LayerConfig",
    "LayerSpec",
    "LayerType",
    "LayerUnion",
    "LogoLayer",
    "MaskMode",
    "ShapeLayer",
    "ShapeType",
    "StickerLayer",
    "TextAlign",
    "TextAnimation",
    "TextLayer",
    "TextOverflow",
    "TextPlacement",
    "TextPlacementMode",
    "TextStyle",
    "TextWeight",
    "TrackRole",
    "Transition",
    "VerticalAlign",
    "VideoLayer",
    "parse_layer",
]

CURRENT_COMPOSITION_SCHEMA_VERSION = 1
"""The composition schema version written by this release."""

COMPOSITION_SCHEMA_VERSION = CURRENT_COMPOSITION_SCHEMA_VERSION
"""Alias for :data:`CURRENT_COMPOSITION_SCHEMA_VERSION`."""

_RESOLUTION = re.compile(r"^(?P<width>\d+)\s*[xX*\u00d7]\s*(?P<height>\d+)$")


class LayerType(StrEnum):
    """The kinds of layer a composition can stack."""

    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"
    SHAPE = "shape"
    STICKER = "sticker"
    LOGO = "logo"


class FitMode(StrEnum):
    """How a media layer fits its source into the layer rectangle."""

    COVER = "cover"
    CONTAIN = "contain"
    STRETCH = "stretch"
    NONE = "none"

    @classmethod
    def _missing_(cls, value: Any) -> FitMode | None:
        if not isinstance(value, str):
            return None
        aliases = {
            "crop": cls.COVER,
            "fill": cls.STRETCH,
            "scale": cls.STRETCH,
            "original": cls.NONE,
            "actual": cls.NONE,
        }
        return aliases.get(value.strip().lower())


class ShapeType(StrEnum):
    """The primitive shapes a shape layer can draw."""

    RECTANGLE = "rectangle"
    ROUNDED_RECTANGLE = "rounded_rectangle"
    CIRCLE = "circle"
    LINE = "line"

    @classmethod
    def _missing_(cls, value: Any) -> ShapeType | None:
        if not isinstance(value, str):
            return None
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "rect": cls.RECTANGLE,
            "square": cls.RECTANGLE,
            "rounded_rect": cls.ROUNDED_RECTANGLE,
            "round_rect": cls.ROUNDED_RECTANGLE,
            "round_rectangle": cls.ROUNDED_RECTANGLE,
            "ellipse": cls.CIRCLE,
            "oval": cls.CIRCLE,
            "line_segment": cls.LINE,
        }
        return aliases.get(key)


class TextAlign(StrEnum):
    """Horizontal alignment of a text layer."""

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"

    @classmethod
    def _missing_(cls, value: Any) -> TextAlign | None:
        if not isinstance(value, str):
            return None
        aliases = {
            "start": cls.LEFT,
            "centre": cls.CENTER,
            "middle": cls.CENTER,
            "end": cls.RIGHT,
        }
        return aliases.get(value.strip().lower())


class VerticalAlign(StrEnum):
    """Vertical placement of the text block inside its layer rectangle."""

    TOP = "top"
    MIDDLE = "middle"
    BOTTOM = "bottom"

    @classmethod
    def _missing_(cls, value: Any) -> VerticalAlign | None:
        if not isinstance(value, str):
            return None
        aliases = {
            "start": cls.TOP,
            "center": cls.MIDDLE,
            "centre": cls.MIDDLE,
            "end": cls.BOTTOM,
        }
        return aliases.get(value.strip().lower())


class Anchor(StrEnum):
    """One of the nine positions a sticker or logo can be pinned to."""

    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    MIDDLE_LEFT = "middle_left"
    CENTER = "center"
    MIDDLE_RIGHT = "middle_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"

    @classmethod
    def _missing_(cls, value: Any) -> Anchor | None:
        if not isinstance(value, str):
            return None
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "top_left": cls.TOP_LEFT,
            "top_center": cls.TOP_CENTER,
            "top_right": cls.TOP_RIGHT,
            "middle_left": cls.MIDDLE_LEFT,
            "center": cls.CENTER,
            "middle_right": cls.MIDDLE_RIGHT,
            "bottom_left": cls.BOTTOM_LEFT,
            "bottom_center": cls.BOTTOM_CENTER,
            "bottom_right": cls.BOTTOM_RIGHT,
            "top": cls.TOP_CENTER,
            "upper": cls.TOP_CENTER,
            "upper_center": cls.TOP_CENTER,
            "top_middle": cls.TOP_CENTER,
            "north": cls.TOP_CENTER,
            "bottom": cls.BOTTOM_CENTER,
            "lower": cls.BOTTOM_CENTER,
            "lower_center": cls.BOTTOM_CENTER,
            "bottom_middle": cls.BOTTOM_CENTER,
            "south": cls.BOTTOM_CENTER,
            "left": cls.MIDDLE_LEFT,
            "center_left": cls.MIDDLE_LEFT,
            "centre_left": cls.MIDDLE_LEFT,
            "west": cls.MIDDLE_LEFT,
            "right": cls.MIDDLE_RIGHT,
            "center_right": cls.MIDDLE_RIGHT,
            "centre_right": cls.MIDDLE_RIGHT,
            "east": cls.MIDDLE_RIGHT,
            "middle": cls.CENTER,
            "centre": cls.CENTER,
            "middle_center": cls.CENTER,
            "center_middle": cls.CENTER,
            "centre_middle": cls.CENTER,
            "topleft": cls.TOP_LEFT,
            "upper_left": cls.TOP_LEFT,
            "top_start": cls.TOP_LEFT,
            "northwest": cls.TOP_LEFT,
            "nw": cls.TOP_LEFT,
            "north_west": cls.TOP_LEFT,
            "topright": cls.TOP_RIGHT,
            "upper_right": cls.TOP_RIGHT,
            "top_end": cls.TOP_RIGHT,
            "northeast": cls.TOP_RIGHT,
            "ne": cls.TOP_RIGHT,
            "north_east": cls.TOP_RIGHT,
            "bottomleft": cls.BOTTOM_LEFT,
            "lower_left": cls.BOTTOM_LEFT,
            "bottom_start": cls.BOTTOM_LEFT,
            "southwest": cls.BOTTOM_LEFT,
            "sw": cls.BOTTOM_LEFT,
            "south_west": cls.BOTTOM_LEFT,
            "bottomright": cls.BOTTOM_RIGHT,
            "lower_right": cls.BOTTOM_RIGHT,
            "bottom_end": cls.BOTTOM_RIGHT,
            "southeast": cls.BOTTOM_RIGHT,
            "se": cls.BOTTOM_RIGHT,
            "south_east": cls.BOTTOM_RIGHT,
        }
        return aliases.get(key)


class BlendMode(StrEnum):
    """How an image layer is combined with the pixels below it."""

    NORMAL = "normal"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    ADD = "add"
    SUBTRACT = "subtract"
    DARKEN = "darken"
    LIGHTEN = "lighten"
    DIFFERENCE = "difference"
    HARD_LIGHT = "hard_light"
    SOFT_LIGHT = "soft_light"

    @classmethod
    def _missing_(cls, value: Any) -> BlendMode | None:
        if not isinstance(value, str):
            return None
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "source_over": cls.NORMAL,
            "src_over": cls.NORMAL,
            "over": cls.NORMAL,
            "plus": cls.ADD,
            "addition": cls.ADD,
            "linear_dodge": cls.ADD,
            "sub": cls.SUBTRACT,
            "minus": cls.SUBTRACT,
            "darker": cls.DARKEN,
            "lighter": cls.LIGHTEN,
            "hardlight": cls.HARD_LIGHT,
            "hard_light": cls.HARD_LIGHT,
            "softlight": cls.SOFT_LIGHT,
            "soft_light": cls.SOFT_LIGHT,
        }
        return aliases.get(key)


class MaskMode(StrEnum):
    """Which channel of a mask image controls an image layer's visibility."""

    ALPHA = "alpha"
    LUMINANCE = "luminance"

    @classmethod
    def _missing_(cls, value: Any) -> MaskMode | None:
        if not isinstance(value, str):
            return None
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "luma": cls.LUMINANCE,
            "brightness": cls.LUMINANCE,
            "gray": cls.LUMINANCE,
            "grey": cls.LUMINANCE,
            "greyscale": cls.LUMINANCE,
            "grayscale": cls.LUMINANCE,
            "opacity": cls.ALPHA,
            "transparency": cls.ALPHA,
        }
        return aliases.get(key)


class FilterType(StrEnum):
    """A basic, deterministic image filter."""

    BLUR = "blur"
    BOX_BLUR = "box_blur"
    SHARPEN = "sharpen"
    UNSHARP_MASK = "unsharp_mask"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    SATURATION = "saturation"
    GRAYSCALE = "grayscale"
    INVERT = "invert"
    AUTOCONTRAST = "autocontrast"

    @classmethod
    def _missing_(cls, value: Any) -> FilterType | None:
        if not isinstance(value, str):
            return None
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "gaussian_blur": cls.BLUR,
            "gaussian": cls.BLUR,
            "blur_gaussian": cls.BLUR,
            "boxblur": cls.BOX_BLUR,
            "sharpness": cls.SHARPEN,
            "unsharp": cls.UNSHARP_MASK,
            "unsharp_mask_filter": cls.UNSHARP_MASK,
            "bright": cls.BRIGHTNESS,
            "lightness": cls.BRIGHTNESS,
            "color": cls.SATURATION,
            "colour": cls.SATURATION,
            "colorfulness": cls.SATURATION,
            "greyscale": cls.GRAYSCALE,
            "grey": cls.GRAYSCALE,
            "gray": cls.GRAYSCALE,
            "negate": cls.INVERT,
            "negative": cls.INVERT,
            "auto_contrast": cls.AUTOCONTRAST,
        }
        return aliases.get(key)


class TextStyle(StrEnum):
    """The slant of a text layer."""

    NORMAL = "normal"
    ITALIC = "italic"

    @classmethod
    def _missing_(cls, value: Any) -> TextStyle | None:
        if not isinstance(value, str):
            return None
        aliases = {
            "regular": cls.NORMAL,
            "roman": cls.NORMAL,
            "oblique": cls.ITALIC,
        }
        return aliases.get(value.strip().lower())


class TextWeight(StrEnum):
    """The weight of a text layer, mapped onto the bundled font files."""

    THIN = "thin"
    LIGHT = "light"
    REGULAR = "regular"
    MEDIUM = "medium"
    SEMIBOLD = "semibold"
    BOLD = "bold"
    BLACK = "black"

    @classmethod
    def _missing_(cls, value: Any) -> TextWeight | None:
        aliases = {
            "normal": cls.REGULAR,
            "book": cls.REGULAR,
            "demibold": cls.SEMIBOLD,
            "semi_bold": cls.SEMIBOLD,
            "semi-bold": cls.SEMIBOLD,
            "heavy": cls.BLACK,
            "extra_bold": cls.BLACK,
            "extrabold": cls.BLACK,
            "extra-bold": cls.BLACK,
        }
        if isinstance(value, int) and not isinstance(value, bool):
            if value < 1 or value > 1000:
                return None
            if value < 300:
                return cls.THIN if value < 200 else cls.LIGHT
            if value < 500:
                return cls.REGULAR
            if value < 600:
                return cls.MEDIUM
            if value < 700:
                return cls.SEMIBOLD
            if value < 800:
                return cls.BOLD
            return cls.BLACK
        if not isinstance(value, str):
            return None
        key = value.strip().lower()
        if key.isdigit():
            return cls._missing_(int(key))
        return aliases.get(key)

    @property
    def is_bold(self) -> bool:
        """Whether this weight selects the bold bundled font member."""
        return self in (TextWeight.SEMIBOLD, TextWeight.BOLD, TextWeight.BLACK)


class TextOverflow(StrEnum):
    """What happens to text that still does not fit after wrapping."""

    CLIP = "clip"
    ELLIPSIS = "ellipsis"

    @classmethod
    def _missing_(cls, value: Any) -> TextOverflow | None:
        if not isinstance(value, str):
            return None
        aliases = {
            "none": cls.CLIP,
            "visible": cls.CLIP,
            "hide": cls.CLIP,
            "truncate": cls.ELLIPSIS,
            "dots": cls.ELLIPSIS,
        }
        return aliases.get(value.strip().lower())


class AudioTrackRole(StrEnum):
    """How an audio track participates in the mix and ducking."""

    VOICE = "voice"
    MUSIC = "music"
    BACKGROUND = "background"
    SFX = "sfx"
    EFFECTS = "effects"

    @property
    def is_voice(self) -> bool:
        """Whether this role is a voice that drives ducking."""
        return self is AudioTrackRole.VOICE


TrackRole = AudioTrackRole


class _SchemaModel(BaseModel):
    """Strict base for composition values."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )


class GradientType(StrEnum):
    """The kinds of gradient a shape can paint with."""

    LINEAR = "linear"

    @classmethod
    def _missing_(cls, value: Any) -> GradientType | None:
        if not isinstance(value, str):
            return None
        aliases = {"linear_gradient": cls.LINEAR, "stripes": cls.LINEAR}
        return aliases.get(value.strip().lower().replace("-", "_").replace(" ", "_"))


class GradientStop(_SchemaModel):
    """One colour at a position along a gradient."""

    color: str = Field(
        min_length=1,
        validation_alias=AliasChoices("color", "colour"),
    )
    offset: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("offset", "position", "stop", "at"),
    )

    @field_validator("color", mode="before")
    @classmethod
    def _validate_color(cls, value: Any) -> Any:
        if not isinstance(value, str):
            raise ValueError("gradient colour must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("gradient colour must not be empty")
        return stripped

    @field_validator("offset", mode="before")
    @classmethod
    def _validate_offset(cls, value: Any) -> Any:
        return _finite(value, "gradient offset")


class Gradient(_SchemaModel):
    """A linear gradient made of ordered colour stops.

    The two colour shorthand accepts ``start``/``end`` (and their aliases),
    while ``stops`` and ``colors`` describe multi-colour gradients. Stops are
    ordered by offset. An angle of zero runs left to right and ninety runs top
    to bottom.
    """

    type: GradientType = Field(
        default=GradientType.LINEAR,
        validation_alias=AliasChoices("type", "kind"),
    )
    stops: list[GradientStop] = Field(
        min_length=2,
        validation_alias=AliasChoices("stops", "colors", "colours"),
    )
    angle: float = Field(
        default=0.0,
        validation_alias=AliasChoices("angle", "direction", "rotation"),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_color_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {
                "stops": [
                    {"color": data, "offset": 0.0},
                    {"color": data, "offset": 1.0},
                ]
            }
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        if "stops" not in values:
            colors = values.get("colors", values.get("colours"))
            if colors is not None and not isinstance(colors, (str, bytes)):
                values["stops"] = [
                    {
                        "color": color,
                        "offset": 0.0
                        if len(colors) == 1
                        else index / (len(colors) - 1),
                    }
                    for index, color in enumerate(colors)
                ]
                values.pop("colors", None)
                values.pop("colours", None)
            else:
                start = _first_present(
                    values,
                    "start_color",
                    "start",
                    "from_color",
                    "from",
                    "color",
                )
                end = _first_present(
                    values,
                    "end_color",
                    "end",
                    "to_color",
                    "to",
                    "color2",
                )
                if start is not None or end is not None:
                    values["stops"] = [
                        {"color": start or "#ffffff", "offset": 0.0},
                        {"color": end or "#00000000", "offset": 1.0},
                    ]
        stops = values.get("stops")
        if isinstance(stops, list):
            normalized: list[Any] = []
            for index, stop in enumerate(stops):
                default_offset = index / (len(stops) - 1) if len(stops) > 1 else 0.0
                if isinstance(stop, str):
                    normalized.append({"color": stop, "offset": default_offset})
                elif isinstance(stop, Mapping):
                    item = dict(stop)
                    if not any(
                        key in item for key in ("offset", "position", "stop", "at")
                    ):
                        item["offset"] = default_offset
                    normalized.append(item)
                else:
                    normalized.append(stop)
            values["stops"] = normalized
        return values

    @field_validator("angle", mode="before")
    @classmethod
    def _validate_angle(cls, value: Any) -> Any:
        return _finite(value, "gradient angle")

    @model_validator(mode="after")
    def _validate_stops(self) -> Self:
        offsets = [stop.offset for stop in self.stops]
        if any(left > right for left, right in itertools.pairwise(offsets)):
            raise ValueError("gradient stops must be ordered by offset")
        return self


class ImageMask(_SchemaModel):
    """A second image that limits which pixels of a layer are visible.

    ``alpha`` uses the mask image's alpha channel while ``luminance`` uses its
    brightness, so a plain grayscale PNG works as a mask. The mask is scaled to
    the layer box, optionally inverted and softened with a Gaussian feather.
    """

    source: str = Field(
        min_length=1,
        validation_alias=AliasChoices(
            "source",
            "path",
            "file",
            "asset",
            "asset_id",
            "mask",
        ),
    )
    mode: MaskMode = Field(
        default=MaskMode.ALPHA,
        validation_alias=AliasChoices("mode", "mask_mode", "channel"),
    )
    invert: bool = Field(
        default=False,
        validation_alias=AliasChoices("invert", "inverted", "reverse"),
    )
    feather: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("feather", "blur", "softness"),
    )
    fit: FitMode = Field(
        default=FitMode.STRETCH,
        validation_alias=AliasChoices("fit", "fit_mode", "mode_name"),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_string_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"source": data}
        return data

    @field_validator("source", mode="before")
    @classmethod
    def _validate_source(cls, value: Any) -> Any:
        if not isinstance(value, str):
            raise ValueError("mask source must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("mask source must not be empty")
        return stripped

    @field_validator("feather", mode="before")
    @classmethod
    def _validate_feather(cls, value: Any) -> Any:
        return _finite(value, "mask feather")

    @field_validator("invert", mode="before")
    @classmethod
    def _validate_invert(cls, value: Any) -> Any:
        if not isinstance(value, bool):
            raise ValueError("mask invert must be a boolean")
        return value


class ImageFilter(_SchemaModel):
    """One basic filter applied to an image layer before compositing.

    Colour and tone filters use ``amount`` as their strength. Blur and sharpen
    filters use ``radius``; a scalar shorthand such as ``{"blur": 2}`` fills the
    matching field. Filters run in the order they are declared.
    """

    type: FilterType = Field(
        validation_alias=AliasChoices("type", "kind", "filter", "name"),
    )
    amount: float = Field(
        default=1.0,
        ge=0.0,
        validation_alias=AliasChoices(
            "amount",
            "value",
            "factor",
            "strength",
            "intensity",
        ),
    )
    radius: float | None = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices("radius", "size", "blur_radius", "softness"),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_string_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"type": data}
        return data

    @field_validator("amount", "radius", mode="before")
    @classmethod
    def _validate_number(cls, value: Any) -> Any:
        if value is None:
            return None
        return _finite(value, "filter")


def _filter_spec(name: str, value: Any) -> Any:
    """Turn one ``{name: value}`` filter shorthand into a spec mapping."""
    if isinstance(value, bool):
        return {"type": name} if value else None
    if isinstance(value, Mapping):
        item = dict(value)
        item.setdefault("type", name)
        return item
    try:
        filter_type = FilterType(name)
    except (TypeError, ValueError):
        return {"type": name, "amount": value}
    if filter_type in (
        FilterType.BLUR,
        FilterType.BOX_BLUR,
        FilterType.SHARPEN,
        FilterType.UNSHARP_MASK,
    ):
        return {"type": name, "radius": value}
    return {"type": name, "amount": value}


def normalise_filters(value: Any) -> Any:
    """Normalise filter shorthand into a list of :class:`ImageFilter` specs."""
    if value is None:
        return []
    if isinstance(value, (ImageFilter, str)):
        return [value]
    if isinstance(value, Mapping):
        if any(key in value for key in ("type", "kind", "filter", "name")):
            return [value]
        specs: list[Any] = []
        for name, spec in value.items():
            normalized = _filter_spec(str(name), spec)
            if normalized is not None:
                specs.append(normalized)
        return specs
    if isinstance(value, (list, tuple)):
        specs = []
        for item in value:
            if (
                isinstance(item, Mapping)
                and len(item) == 1
                and not any(key in item for key in ("type", "kind", "filter", "name"))
            ):
                ((name, spec),) = item.items()
                normalized = _filter_spec(str(name), spec)
                if normalized is not None:
                    specs.append(normalized)
            else:
                specs.append(item)
        return specs
    return value


def _first_present(values: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in values:
            return values.pop(name)
    return None


def _parse_size(value: Any) -> tuple[int, int]:
    if isinstance(value, str):
        parsed = _RESOLUTION.fullmatch(value.strip())
        if parsed is None:
            raise ValueError("size must be WIDTHxHEIGHT, for example 1080x1920")
        return int(parsed.group("width")), int(parsed.group("height"))
    if isinstance(value, Mapping):
        return int(value["width"]), int(value["height"])
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError("size must be WIDTHxHEIGHT or a pair of integers")


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _integer(value: Any, name: str) -> Any:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer")
    return value


def _margin_value(value: Any) -> float | tuple[float, float]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("margin must be a number or a pair of numbers")
        horizontal = _finite(value[0], "margin")
        vertical = _finite(value[1], "margin")
        if horizontal < 0.0 or vertical < 0.0:
            raise ValueError("margin must not be negative")
        return horizontal, vertical
    number = _finite(value, "margin")
    if number < 0.0:
        raise ValueError("margin must not be negative")
    return number


class TextPlacementMode(StrEnum):
    """How a text layer chooses its rectangle on the canvas."""

    MANUAL = "manual"
    SMART = "smart"
    AUTO = "auto"

    @classmethod
    def _missing_(cls, value: Any) -> TextPlacementMode | None:
        if not isinstance(value, str):
            return None
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "automatic": cls.AUTO,
            "auto_place": cls.AUTO,
            "low_detail": cls.SMART,
            "low_detail_region": cls.SMART,
            "none": cls.MANUAL,
            "off": cls.MANUAL,
            "disabled": cls.MANUAL,
        }
        return aliases.get(key)

    @property
    def is_smart(self) -> bool:
        """Whether this mode asks the renderer to find a calm region."""
        return self is not TextPlacementMode.MANUAL


class FaceBox(_SchemaModel):
    """A rectangle around one face, in canvas pixels.

    A composition builder fills this from a vision check so a render stays
    deterministic and offline.
    """

    x: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("x", "left"),
    )
    y: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("y", "top"),
    )
    width: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("width", "w"),
    )
    height: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("height", "h"),
    )
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("x", "y", "width", "height", "confidence", mode="before")
    @classmethod
    def _validate_number(cls, value: Any) -> Any:
        if value is None:
            return None
        return _finite(value, "face box")

    @property
    def box(self) -> tuple[float, float, float, float]:
        """The region as ``(x, y, width, height)``."""
        return self.x, self.y, self.width, self.height

    @property
    def area(self) -> float:
        """The area of the rectangle in square pixels."""
        return self.width * self.height


class TextPlacement(_SchemaModel):
    """Options for smart text placement.

    ``mode`` defaults to smart when a placement block is present. ``faces``
    carries rectangles produced by an earlier vision check; rendering itself
    never runs that check.
    """

    mode: TextPlacementMode = Field(
        default=TextPlacementMode.SMART,
        validation_alias=AliasChoices("mode", "strategy", "placement_mode"),
    )
    avoid_faces: bool = Field(
        default=True,
        validation_alias=AliasChoices("avoid_faces", "faces_avoid", "skip_faces"),
    )
    padding: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("padding", "pad"),
    )
    margin: float | tuple[float, float] = Field(
        default=0.0,
        validation_alias=AliasChoices("margin", "margins"),
    )
    face_padding: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("face_padding", "face_margin"),
    )
    faces: list[FaceBox] = Field(
        default_factory=list,
        validation_alias=AliasChoices("faces", "face_regions", "detected_faces"),
    )
    step: int | None = Field(
        default=None,
        gt=0,
        validation_alias=AliasChoices("step", "grid_step"),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_shorthand(cls, data: Any) -> Any:
        if data is None:
            return {}
        if isinstance(data, bool):
            return {"mode": "smart" if data else "manual"}
        if isinstance(data, str):
            return {"mode": data}
        return data

    @field_validator("avoid_faces", mode="before")
    @classmethod
    def _validate_flag(cls, value: Any) -> Any:
        if not isinstance(value, bool):
            raise ValueError("avoid_faces must be a boolean")
        return value

    @field_validator("padding", "face_padding", mode="before")
    @classmethod
    def _validate_padding(cls, value: Any) -> Any:
        return _finite(value, "padding")

    @field_validator("margin", mode="before")
    @classmethod
    def _validate_margin(cls, value: Any) -> Any:
        return _margin_value(value)

    @field_validator("faces", mode="before")
    @classmethod
    def _validate_faces(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, (FaceBox, Mapping)):
            return [value]
        return value

    @property
    def smart(self) -> bool:
        """Whether placement should scan the image."""
        return self.mode.is_smart


class Canvas(_SchemaModel):
    """The pixel canvas a composition renders into.

    ``background`` is any colour string the renderer understands. The canvas
    may also be written as ``"1080x1920"`` or as a mapping with a
    ``resolution``/``size`` key.
    """

    width: int = Field(gt=0, validation_alias=AliasChoices("width", "w"))
    height: int = Field(gt=0, validation_alias=AliasChoices("height", "h"))
    background: str = Field(
        default="#000000",
        validation_alias=AliasChoices("background", "background_color", "bg", "color"),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_size_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            width, height = _parse_size(data)
            return {"width": width, "height": height}
        if isinstance(data, (tuple, list)):
            width, height = _parse_size(data)
            return {"width": width, "height": height}
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for key in ("resolution", "size", "dimensions"):
            if key in values:
                width, height = _parse_size(values.pop(key))
                values.setdefault("width", width)
                values.setdefault("height", height)
        return values

    @field_validator("width", "height", mode="before")
    @classmethod
    def _validate_dimension(cls, value: Any) -> Any:
        return _integer(value, "canvas dimension")

    @field_validator("background")
    @classmethod
    def _strip_background(cls, value: str) -> str:
        return value.strip()

    @property
    def size(self) -> tuple[int, int]:
        """The canvas as ``(width, height)``."""
        return self.width, self.height

    @property
    def resolution(self) -> str:
        """The canvas as a ``WIDTHxHEIGHT`` string."""
        return f"{self.width}x{self.height}"

    @property
    def aspect_ratio(self) -> float:
        """The width divided by the height."""
        return self.width / self.height

    @property
    def pixels(self) -> int:
        """The total number of pixels on the canvas."""
        return self.width * self.height


CanvasSpec = Canvas
CanvasConfig = Canvas


class Layer(_SchemaModel):
    """The common envelope of one visual layer.

    Concrete layer kinds add their own fields later. The envelope keeps a
    stable id, a timing window, stacking order, opacity and visibility so the
    renderer and the patch language can address any layer.
    """

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        validate_assignment=True,
    )

    id: str = Field(
        default="",
        validation_alias=AliasChoices("id", "layer_id", "name"),
    )
    type: LayerType = Field(
        validation_alias=AliasChoices("type", "kind", "layer_type"),
    )
    start_seconds: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("start_seconds", "start", "offset_seconds", "at"),
    )
    end_seconds: float | None = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices("end_seconds", "end", "stop"),
    )
    duration_seconds: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("duration_seconds", "duration", "length_seconds"),
    )
    z_index: int = Field(
        default=0,
        validation_alias=AliasChoices("z_index", "z", "order", "index"),
    )
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    visible: bool = Field(
        default=True, validation_alias=AliasChoices("visible", "enabled")
    )
    blend_mode: BlendMode = Field(
        default=BlendMode.NORMAL,
        validation_alias=AliasChoices(
            "blend_mode",
            "blend",
            "blending",
            "blending_mode",
        ),
    )
    mask: ImageMask | None = Field(
        default=None,
        validation_alias=AliasChoices("mask", "image_mask", "mask_source"),
    )
    filters: list[ImageFilter] = Field(
        default_factory=list,
        validation_alias=AliasChoices("filters", "filter", "image_filters"),
    )
    motion: Motion | None = Field(
        default=None,
        validation_alias=AliasChoices("motion", "camera", "movement"),
    )
    entrance: TextAnimation | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "entrance",
            "entrance_animation",
            "enter",
            "in",
            "appear",
            "in_animation",
        ),
    )
    exit: TextAnimation | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "exit",
            "exit_animation",
            "leave",
            "out",
            "disappear",
            "out_animation",
        ),
    )
    transition: Transition | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "transition",
            "transition_in",
            "enter_transition",
            "in_transition",
        ),
    )

    @field_validator("transition", mode="before")
    @classmethod
    def _validate_transition(cls, value: Any) -> Any:
        if value is None or value is False:
            return None
        if isinstance(value, Transition):
            return value
        if isinstance(value, (str, bool, Mapping)):
            return Transition.model_validate(value)
        return value

    @model_validator(mode="before")
    @classmethod
    def _fold_text_animation(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        bundle = values.pop("text_animation", None)
        if bundle is not None:
            if isinstance(bundle, Mapping):
                for key in ("entrance", "enter", "in", "exit", "leave", "out"):
                    if key in bundle:
                        target = (
                            "exit" if key in {"exit", "leave", "out"} else "entrance"
                        )
                        values.setdefault(target, bundle[key])
            elif isinstance(bundle, (str, bool)):
                values.setdefault("entrance", bundle)
        for container in ("animation", "animations"):
            raw = values.get(container)
            if isinstance(raw, (str, bool)):
                values.pop(container, None)
                values.setdefault("entrance", raw)
                continue
            if not isinstance(raw, Mapping):
                continue
            keys = {
                "".join(
                    character for character in str(key).lower() if character.isalnum()
                )
                for key in raw
            }
            if keys and keys <= {
                "entrance",
                "enter",
                "in",
                "exit",
                "leave",
                "out",
            }:
                values.pop(container, None)
                for key, item in raw.items():
                    normalized = "".join(
                        character
                        for character in str(key).lower()
                        if character.isalnum()
                    )
                    target = (
                        "exit" if normalized in {"exit", "leave", "out"} else "entrance"
                    )
                    values.setdefault(target, item)
        return values

    @field_validator("entrance", "exit", mode="before")
    @classmethod
    def _validate_text_animation(cls, value: Any) -> Any:
        if value is None or value is False:
            return None
        if isinstance(value, TextAnimation):
            return value
        if isinstance(value, (str, bool, Mapping)):
            return TextAnimation.model_validate(value)
        return value

    @model_validator(mode="before")
    @classmethod
    def _merge_flat_mask_options(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        flat = {
            "mask_mode": "mode",
            "mask_channel": "mode",
            "mask_invert": "invert",
            "mask_feather": "feather",
            "mask_fit": "fit",
        }
        if not any(name in values for name in flat):
            return values
        raw = values.get("mask", values.get("image_mask", values.get("mask_source")))
        if isinstance(raw, ImageMask):
            return values
        if raw is None:
            raw = {}
        elif isinstance(raw, str):
            raw = {"source": raw}
        elif isinstance(raw, Mapping):
            raw = dict(raw)
        else:
            return values
        for name, target in flat.items():
            if name in values and target not in raw:
                raw[target] = values.pop(name)
        for name in ("mask", "image_mask", "mask_source"):
            values.pop(name, None)
        values["mask"] = raw
        return values

    @model_validator(mode="before")
    @classmethod
    def _accept_string_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"type": data}
        return data

    @field_validator("mask", mode="before")
    @classmethod
    def _validate_mask(cls, value: Any) -> Any:
        if value is None or value is False:
            return None
        if isinstance(value, ImageMask):
            return value
        if isinstance(value, str):
            return {"source": value}
        return value

    @field_validator("filters", mode="before")
    @classmethod
    def _validate_filters(cls, value: Any) -> Any:
        return normalise_filters(value)

    @field_validator("id", mode="before")
    @classmethod
    def _strip_id(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("start_seconds", "end_seconds", "duration_seconds", mode="before")
    @classmethod
    def _validate_seconds(cls, value: Any) -> Any:
        if value is None:
            return None
        return _finite(value, "layer time")

    @field_validator("z_index", mode="before")
    @classmethod
    def _validate_z_index(cls, value: Any) -> Any:
        return _integer(value, "z_index")

    @field_validator("opacity", mode="before")
    @classmethod
    def _validate_opacity(cls, value: Any) -> Any:
        return _finite(value, "opacity")

    @model_validator(mode="after")
    def _validate_timing(self) -> Self:
        if self.end_seconds is not None and self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        if self.end_seconds is not None and self.duration_seconds is not None:
            expected = self.start_seconds + self.duration_seconds
            if not math.isclose(
                self.end_seconds,
                expected,
                rel_tol=1e-9,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    "end_seconds must equal start_seconds + duration_seconds"
                )
        return self

    @property
    def layer_type(self) -> LayerType:
        """Alias for :attr:`type`."""
        return self.type

    @property
    def kind(self) -> LayerType:
        """Alias for :attr:`type`."""
        return self.type

    @property
    def start(self) -> float:
        """Alias for :attr:`start_seconds`."""
        return self.start_seconds

    @property
    def end(self) -> float | None:
        """The exclusive end of the layer, when it can be determined."""
        if self.end_seconds is not None:
            return self.end_seconds
        if self.duration_seconds is not None:
            return self.start_seconds + self.duration_seconds
        return None

    @property
    def duration(self) -> float | None:
        """The layer duration, when it can be determined."""
        if self.duration_seconds is not None:
            return self.duration_seconds
        if self.end_seconds is not None:
            return self.end_seconds - self.start_seconds
        return None

    @property
    def entrance_animation(self) -> TextAnimation | None:
        """Alias for :attr:`entrance`."""
        return self.entrance

    @property
    def exit_animation(self) -> TextAnimation | None:
        """Alias for :attr:`exit`."""
        return self.exit

    @property
    def z(self) -> int:
        """Alias for :attr:`z_index`."""
        return self.z_index


LayerSpec = Layer
LayerConfig = Layer
BaseLayer = Layer


class _AssetLayer(Layer):
    """Shared source validation for layers backed by a stored asset."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    source: str = Field(
        min_length=1,
        validation_alias=AliasChoices(
            "source",
            "path",
            "file",
            "asset",
            "asset_id",
        ),
    )

    @field_validator("source", mode="before")
    @classmethod
    def _validate_source(cls, value: Any) -> Any:
        if not isinstance(value, str):
            raise ValueError("source must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("source must not be empty")
        return stripped


class _OverlayLayer(_AssetLayer):
    """Shared placement and shadow fields for stickers and logos.

    An overlay is pinned to one of nine canvas anchors. ``margin`` is either a
    single number applied to both axes or a ``(horizontal, vertical)`` pair.
    A drop shadow is optional and disabled unless ``shadow`` is true.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    anchor: Anchor = Field(
        default=Anchor.CENTER,
        validation_alias=AliasChoices("anchor", "corner", "alignment"),
    )
    margin: float | tuple[float, float] = Field(
        default=0.0,
        validation_alias=AliasChoices("margin", "margins", "inset"),
    )
    width: int | None = Field(
        default=None,
        gt=0,
        validation_alias=AliasChoices("width", "w"),
    )
    height: int | None = Field(
        default=None,
        gt=0,
        validation_alias=AliasChoices("height", "h"),
    )
    scale: float = Field(
        default=1.0,
        gt=0.0,
        validation_alias=AliasChoices("scale", "zoom"),
    )
    fit: FitMode = Field(
        default=FitMode.CONTAIN,
        validation_alias=AliasChoices("fit", "fit_mode", "mode"),
    )
    shadow: bool = Field(
        default=False,
        validation_alias=AliasChoices("shadow", "drop_shadow"),
    )
    shadow_color: str = Field(
        default="#000000",
        validation_alias=AliasChoices("shadow_color", "shadow_colour"),
    )
    shadow_offset: tuple[float, float] = Field(
        default=(4.0, 4.0),
        validation_alias=AliasChoices("shadow_offset", "shadow_position"),
    )
    shadow_blur: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices(
            "shadow_blur",
            "shadow_softness",
            "shadow_radius",
        ),
    )
    shadow_opacity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("shadow_opacity", "shadow_alpha"),
    )

    @field_validator("margin", mode="before")
    @classmethod
    def _validate_margin(cls, value: Any) -> Any:
        return _margin_value(value)

    @field_validator("width", "height", mode="before")
    @classmethod
    def _validate_dimension(cls, value: Any) -> Any:
        if value is None:
            return None
        return _integer(value, "overlay dimension")

    @field_validator("scale", "shadow_blur", "shadow_opacity", mode="before")
    @classmethod
    def _validate_overlay_number(cls, value: Any) -> Any:
        return _finite(value, "overlay")

    @field_validator("shadow", mode="before")
    @classmethod
    def _validate_shadow_flag(cls, value: Any) -> Any:
        if not isinstance(value, bool):
            raise ValueError("shadow must be a boolean")
        return value

    @field_validator("shadow_color", mode="before")
    @classmethod
    def _validate_shadow_color(cls, value: Any) -> Any:
        if not isinstance(value, str):
            raise ValueError("shadow_color must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("shadow_color must not be empty")
        return stripped

    @field_validator("shadow_offset", mode="before")
    @classmethod
    def _validate_shadow_offset(cls, value: Any) -> Any:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError("shadow_offset must be a pair of numbers")
        return _finite(value[0], "shadow_offset"), _finite(value[1], "shadow_offset")

    @property
    def margin_x(self) -> float:
        """The horizontal margin in pixels."""
        if isinstance(self.margin, tuple):
            return self.margin[0]
        return self.margin

    @property
    def margin_y(self) -> float:
        """The vertical margin in pixels."""
        if isinstance(self.margin, tuple):
            return self.margin[1]
        return self.margin


class ImageLayer(_AssetLayer):
    """A still image placed on the canvas."""

    type: Literal[LayerType.IMAGE] = Field(
        default=LayerType.IMAGE,
        validation_alias=AliasChoices("type", "kind", "layer_type"),
    )
    fit: FitMode = Field(
        default=FitMode.COVER,
        validation_alias=AliasChoices("fit", "fit_mode", "mode"),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_string_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"type": LayerType.IMAGE, "source": data}
        return data


class VideoLayer(_AssetLayer):
    """A video clip placed on the canvas."""

    type: Literal[LayerType.VIDEO] = Field(
        default=LayerType.VIDEO,
        validation_alias=AliasChoices("type", "kind", "layer_type"),
    )
    fit: FitMode = Field(
        default=FitMode.COVER,
        validation_alias=AliasChoices("fit", "fit_mode", "mode"),
    )
    trim_start_seconds: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices(
            "trim_start_seconds", "trim_start", "source_start", "clip_start"
        ),
    )
    trim_end_seconds: float | None = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices(
            "trim_end_seconds", "trim_end", "source_end", "clip_end"
        ),
    )
    loop: bool = Field(default=False, validation_alias=AliasChoices("loop", "looped"))
    muted: bool = Field(default=False, validation_alias=AliasChoices("muted", "mute"))
    playback_rate: float = Field(
        default=1.0,
        gt=0.0,
        validation_alias=AliasChoices("playback_rate", "speed", "rate"),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_string_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"type": LayerType.VIDEO, "source": data}
        return data

    @field_validator("trim_start_seconds", "trim_end_seconds", mode="before")
    @classmethod
    def _validate_trim(cls, value: Any) -> Any:
        if value is None:
            return None
        return _finite(value, "trim time")

    @field_validator("playback_rate", mode="before")
    @classmethod
    def _validate_playback_rate(cls, value: Any) -> Any:
        return _finite(value, "playback_rate")

    @model_validator(mode="after")
    def _validate_trim_window(self) -> Self:
        if (
            self.trim_end_seconds is not None
            and self.trim_end_seconds <= self.trim_start_seconds
        ):
            raise ValueError("trim_end_seconds must be greater than trim_start_seconds")
        return self


class TextLayer(Layer):
    """A text run drawn on the canvas."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    type: Literal[LayerType.TEXT] = Field(
        default=LayerType.TEXT,
        validation_alias=AliasChoices("type", "kind", "layer_type"),
    )
    text: str = Field(
        min_length=1,
        validation_alias=AliasChoices("text", "content", "value"),
    )
    font_family: str = Field(
        default="",
        validation_alias=AliasChoices("font_family", "font", "family"),
    )
    font_size: float = Field(
        default=48.0,
        gt=0.0,
        validation_alias=AliasChoices("font_size", "size", "font_size_px"),
    )
    color: str = Field(
        default="#ffffff",
        validation_alias=AliasChoices("color", "fill", "text_color"),
    )
    align: TextAlign = Field(
        default=TextAlign.LEFT,
        validation_alias=AliasChoices("align", "text_align", "alignment"),
    )
    vertical_align: VerticalAlign = Field(
        default=VerticalAlign.TOP,
        validation_alias=AliasChoices("vertical_align", "valign", "vertical_alignment"),
    )
    font_weight: TextWeight = Field(
        default=TextWeight.REGULAR,
        validation_alias=AliasChoices("font_weight", "weight", "font-weight"),
    )
    font_style: TextStyle = Field(
        default=TextStyle.NORMAL,
        validation_alias=AliasChoices("font_style", "style", "font-style"),
    )
    line_height: float = Field(
        default=1.2,
        gt=0.0,
        validation_alias=AliasChoices("line_height", "line_spacing", "line-height"),
    )
    letter_spacing: float = Field(
        default=0.0,
        validation_alias=AliasChoices("letter_spacing", "tracking", "letter-spacing"),
    )
    wrap: bool = Field(
        default=True,
        validation_alias=AliasChoices("wrap", "wrap_text", "word_wrap"),
    )
    max_lines: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("max_lines", "lines", "max-lines"),
    )
    shrink_to_fit: bool = Field(
        default=False,
        validation_alias=AliasChoices("shrink_to_fit", "shrink", "fit", "auto_shrink"),
    )
    min_font_size: float = Field(
        default=8.0,
        gt=0.0,
        validation_alias=AliasChoices("min_font_size", "min_size", "minimum_font_size"),
    )
    overflow: TextOverflow = Field(
        default=TextOverflow.ELLIPSIS,
        validation_alias=AliasChoices("overflow", "text_overflow"),
    )
    ellipsis: str = Field(
        default="\u2026",
        min_length=1,
        validation_alias=AliasChoices("ellipsis", "truncation"),
    )
    stroke: str | None = Field(
        default=None,
        validation_alias=AliasChoices("stroke", "stroke_color", "outline"),
    )
    stroke_width: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("stroke_width", "outline_width"),
    )
    placement: TextPlacement | None = Field(
        default=None,
        validation_alias=AliasChoices("placement", "text_placement"),
    )
    keyframes: Any | None = Field(
        default=None,
        validation_alias=AliasChoices("keyframes", "animation", "animations"),
    )

    @model_validator(mode="before")
    @classmethod
    def _fold_placement_options(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        raw = values.get("placement", values.get("text_placement"))
        options: dict[str, Any] = {}
        if isinstance(raw, TextPlacement):
            options.update(raw.model_dump())
        elif isinstance(raw, Mapping):
            options.update(raw)
        elif isinstance(raw, bool):
            options["mode"] = "smart" if raw else "manual"
        elif raw is not None:
            options["mode"] = raw

        flat = {
            "smart_placement": "mode",
            "auto_place": "mode",
            "placement_mode": "mode",
            "avoid_faces": "avoid_faces",
            "placement_padding": "padding",
            "placement_margin": "margin",
            "face_padding": "face_padding",
            "face_regions": "faces",
            "detected_faces": "faces",
            "faces": "faces",
        }
        found = False
        for name, target in flat.items():
            if name not in values:
                continue
            value = values.pop(name)
            if name in ("smart_placement", "auto_place"):
                value = "smart" if value else "manual"
            options.setdefault(target, value)
            found = True
        if not found:
            return values
        values.pop("placement", None)
        values.pop("text_placement", None)
        values["placement"] = options
        return values

    @model_validator(mode="before")
    @classmethod
    def _accept_string_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"type": LayerType.TEXT, "text": data}
        return data

    @field_validator("text", mode="before")
    @classmethod
    def _validate_text(cls, value: Any) -> Any:
        if not isinstance(value, str):
            raise ValueError("text must be a string")
        if not value.strip():
            raise ValueError("text must not be empty")
        return value

    @field_validator("font_family", mode="before")
    @classmethod
    def _strip_font_family(cls, value: Any) -> Any:
        if not isinstance(value, str):
            raise ValueError("value must be a string")
        return value.strip()

    @field_validator("color", "ellipsis", mode="before")
    @classmethod
    def _strip_text_field(cls, value: Any) -> Any:
        if not isinstance(value, str):
            raise ValueError("value must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be empty")
        return stripped

    @field_validator("stroke", mode="before")
    @classmethod
    def _validate_stroke(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("stroke must be a string")
        return value.strip() or None

    @field_validator(
        "font_size",
        "line_height",
        "letter_spacing",
        "min_font_size",
        "stroke_width",
        mode="before",
    )
    @classmethod
    def _validate_text_number(cls, value: Any) -> Any:
        return _finite(value, "text style")

    @field_validator("max_lines", mode="before")
    @classmethod
    def _validate_max_lines(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("max_lines must be an integer")
        return value

    @field_validator("wrap", "shrink_to_fit", mode="before")
    @classmethod
    def _validate_text_flag(cls, value: Any) -> Any:
        if not isinstance(value, bool):
            raise ValueError("value must be a boolean")
        return value

    @model_validator(mode="after")
    def _validate_shrink_bounds(self) -> Self:
        if self.shrink_to_fit and self.min_font_size > self.font_size:
            raise ValueError("min_font_size must not exceed font_size")
        return self

    @property
    def smart_placement(self) -> bool:
        """Whether this layer asks for smart text placement."""
        return self.placement is not None and self.placement.smart

    @property
    def auto_place(self) -> bool:
        """Alias for :attr:`smart_placement`."""
        return self.smart_placement

    @property
    def placement_mode(self) -> TextPlacementMode | None:
        """The configured placement mode, if any."""
        return None if self.placement is None else self.placement.mode

    @property
    def avoid_faces(self) -> bool:
        """Whether smart placement avoids configured face rectangles."""
        return self.placement is not None and self.placement.avoid_faces


class ShapeLayer(Layer):
    """A primitive shape drawn on the canvas."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    type: Literal[LayerType.SHAPE] = Field(
        default=LayerType.SHAPE,
        validation_alias=AliasChoices("type", "kind", "layer_type"),
    )
    shape: ShapeType = Field(
        validation_alias=AliasChoices("shape", "shape_type", "form"),
    )
    fill: str | Gradient | None = Field(
        default="#ffffff",
        validation_alias=AliasChoices("fill", "fill_color", "color", "background"),
    )
    stroke: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "stroke",
            "stroke_color",
            "border",
            "border_color",
            "outline",
        ),
    )
    stroke_width: float = Field(
        default=1.0,
        ge=0.0,
        validation_alias=AliasChoices(
            "stroke_width",
            "border_width",
            "outline_width",
            "line_width",
        ),
    )
    radius: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("radius", "corner_radius", "border_radius"),
    )
    gradient: Gradient | None = Field(
        default=None,
        validation_alias=AliasChoices("gradient", "fill_gradient"),
    )

    @field_validator("fill", mode="before")
    @classmethod
    def _strip_fill(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("stroke", mode="before")
    @classmethod
    def _validate_stroke(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("stroke must be a string")
        stripped = value.strip()
        return stripped or None

    @field_validator("stroke_width", "radius", mode="before")
    @classmethod
    def _validate_shape_number(cls, value: Any) -> Any:
        return _finite(value, "shape geometry")

    @model_validator(mode="after")
    def _validate_gradient_fill(self) -> Self:
        if isinstance(self.fill, Gradient) and self.gradient is not None:
            raise ValueError("set either a gradient fill or a gradient, not both")
        return self

    @model_validator(mode="before")
    @classmethod
    def _accept_string_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"type": LayerType.SHAPE, "shape": data}
        return data


class StickerLayer(_OverlayLayer):
    """A decorative sticker placed on the canvas."""

    type: Literal[LayerType.STICKER] = Field(
        default=LayerType.STICKER,
        validation_alias=AliasChoices("type", "kind", "layer_type"),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_string_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"type": LayerType.STICKER, "source": data}
        return data


class LogoLayer(_OverlayLayer):
    """A brand logo placed on the canvas."""

    type: Literal[LayerType.LOGO] = Field(
        default=LayerType.LOGO,
        validation_alias=AliasChoices("type", "kind", "layer_type"),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_string_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"type": LayerType.LOGO, "source": data}
        return data


AnyLayer = ImageLayer | VideoLayer | TextLayer | ShapeLayer | StickerLayer | LogoLayer
"""Any concrete layer, discriminated by its ``type`` field."""

LayerUnion = AnyLayer
"""Alias for :data:`AnyLayer`."""

LayerAdapter: TypeAdapter[AnyLayer] = TypeAdapter(AnyLayer)
"""A reusable pydantic adapter that parses a mapping into a concrete layer."""


class _LayerTypeGuard(BaseModel):
    """Require an explicit layer type before the concrete union is tried."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    type: LayerType = Field(
        validation_alias=AliasChoices("type", "kind", "layer_type"),
    )


def parse_layer(data: Any) -> AnyLayer:
    """Parse *data* into the concrete layer class named by its type."""
    if isinstance(data, Layer):
        if data.__class__ is Layer:
            data = data.model_dump()
        else:
            return cast(AnyLayer, data)
    _LayerTypeGuard.model_validate(data)
    return LayerAdapter.validate_python(data)


class AudioTrack(_SchemaModel):
    """One audio input with its placement, gain and fades."""

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        validate_assignment=True,
    )

    id: str = Field(
        default="",
        validation_alias=AliasChoices("id", "track_id", "name"),
    )
    source: str = Field(
        min_length=1,
        validation_alias=AliasChoices(
            "source",
            "path",
            "file",
            "audio",
            "asset",
            "asset_id",
        ),
    )
    role: AudioTrackRole = Field(
        default=AudioTrackRole.MUSIC,
        validation_alias=AliasChoices("role", "kind", "type"),
    )
    start_seconds: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices(
            "start_seconds", "start", "offset_seconds", "offset"
        ),
    )
    duration_seconds: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("duration_seconds", "duration", "length_seconds"),
    )
    volume_db: float = Field(
        default=0.0,
        validation_alias=AliasChoices("volume_db", "volume", "gain_db", "gain"),
    )
    fade_in_seconds: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("fade_in_seconds", "fade_in"),
    )
    fade_out_seconds: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("fade_out_seconds", "fade_out"),
    )
    loop: bool = Field(default=False, validation_alias=AliasChoices("loop", "looped"))
    duck: bool | None = Field(
        default=None,
        validation_alias=AliasChoices("duck", "ducked", "allow_ducking"),
    )
    enabled: bool = Field(
        default=True, validation_alias=AliasChoices("enabled", "visible")
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_string_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"source": data}
        return data

    @field_validator("id", "source", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator(
        "start_seconds",
        "duration_seconds",
        "fade_in_seconds",
        "fade_out_seconds",
        mode="before",
    )
    @classmethod
    def _validate_seconds(cls, value: Any) -> Any:
        if value is None:
            return None
        return _finite(value, "audio time")

    @field_validator("volume_db", mode="before")
    @classmethod
    def _validate_volume(cls, value: Any) -> Any:
        return _finite(value, "volume_db")

    @model_validator(mode="after")
    def _validate_fades(self) -> Self:
        if (
            self.duration_seconds is not None
            and self.fade_out_seconds > self.duration_seconds
        ):
            raise ValueError("fade_out_seconds must not exceed duration_seconds")
        return self

    @property
    def start(self) -> float:
        """Alias for :attr:`start_seconds`."""
        return self.start_seconds

    @property
    def duration(self) -> float | None:
        """Alias for :attr:`duration_seconds`."""
        return self.duration_seconds

    @property
    def volume(self) -> float:
        """Alias for :attr:`volume_db`."""
        return self.volume_db

    @property
    def offset(self) -> float:
        """Alias for :attr:`start_seconds`."""
        return self.start_seconds

    @property
    def is_voice(self) -> bool:
        """Whether this track is a voice that drives ducking."""
        return self.role.is_voice

    @property
    def should_duck(self) -> bool:
        """Whether this non-voice track may be ducked under a voice."""
        if self.is_voice:
            return False
        if self.duck is not None:
            return self.duck
        return self.role in (AudioTrackRole.MUSIC, AudioTrackRole.BACKGROUND)


AudioTrackSpec = AudioTrack
AudioTrackConfig = AudioTrack


class Effect(_SchemaModel):
    """One effect applied to a time window of the composition."""

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        validate_assignment=True,
    )

    id: str = Field(
        default="",
        validation_alias=AliasChoices("id", "effect_id", "name"),
    )
    type: str = Field(
        min_length=1,
        validation_alias=AliasChoices("type", "kind", "effect", "name"),
    )
    start_seconds: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("start_seconds", "start", "offset_seconds", "at"),
    )
    end_seconds: float | None = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices("end_seconds", "end", "stop"),
    )
    enabled: bool = Field(
        default=True, validation_alias=AliasChoices("enabled", "visible")
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("params", "options", "parameters", "settings"),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_string_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"type": data}
        return data

    @field_validator("id", "type", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("start_seconds", "end_seconds", mode="before")
    @classmethod
    def _validate_seconds(cls, value: Any) -> Any:
        if value is None:
            return None
        return _finite(value, "effect time")

    @model_validator(mode="after")
    def _validate_timing(self) -> Self:
        if self.end_seconds is not None and self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self

    @property
    def kind(self) -> str:
        """Alias for :attr:`type`."""
        return self.type

    @property
    def name(self) -> str:
        """Alias for :attr:`type`."""
        return self.type

    @property
    def start(self) -> float:
        """Alias for :attr:`start_seconds`."""
        return self.start_seconds

    @property
    def end(self) -> float | None:
        """Alias for :attr:`end_seconds`."""
        return self.end_seconds


EffectSpec = Effect
EffectConfig = Effect


class Composition(_SchemaModel):
    """A complete, versioned description of one deterministic render.

    ``duration_seconds`` is the exact output duration and must be positive.
    ``layers`` are stacked by ``z_index``; ``audio_tracks`` and ``effects``
    are optional. Cross-object checks such as unique ids and layer timings
    inside the duration belong to composition validation, not to this schema.
    """

    id: str = Field(
        default="",
        validation_alias=AliasChoices("id", "composition_id"),
    )
    schema_version: int = Field(
        default=CURRENT_COMPOSITION_SCHEMA_VERSION,
        ge=1,
        validation_alias=AliasChoices("schema_version", "version"),
    )
    canvas: Canvas
    duration_seconds: float = Field(
        gt=0.0,
        validation_alias=AliasChoices("duration_seconds", "duration", "length_seconds"),
    )
    fps: int = Field(
        default=30,
        gt=0,
        validation_alias=AliasChoices("fps", "frame_rate", "frames_per_second"),
    )
    layers: list[Layer] = Field(default_factory=list)
    audio_tracks: list[AudioTrack] = Field(
        default_factory=list,
        validation_alias=AliasChoices("audio_tracks", "tracks", "audio"),
    )
    effects: list[Effect] = Field(default_factory=list)
    subtitles: SubtitleTrack | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "subtitles",
            "subtitle",
            "subtitle_track",
            "captions",
            "caption",
        ),
    )
    subtitle_file: str = Field(
        default="",
        validation_alias=AliasChoices(
            "subtitle_file",
            "subtitle_path",
            "captions_file",
            "captions_path",
        ),
    )
    platform: str = Field(
        default="",
        validation_alias=AliasChoices("platform", "target_platform"),
    )
    safe_area: SafeArea | None = Field(
        default=None,
        validation_alias=AliasChoices("safe_area", "safe_areas", "margins"),
    )
    loudness_lufs: float | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "loudness_lufs",
            "target_lufs",
            "target_loudness",
        ),
    )

    @field_validator("subtitles", mode="before")
    @classmethod
    def _coerce_subtitles(cls, value: Any) -> Any:
        if value is None or isinstance(value, SubtitleTrack):
            return value
        if isinstance(value, BaseModel):
            return value.model_dump()
        if isinstance(value, Mapping):
            if {"cues", "words", "style", "language"} & set(value):
                return value
            return {"cues": [value]}
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            if all(
                isinstance(item, Mapping)
                and ({"cues", "words"} & set(item))
                for item in value
            ):
                cues = [
                    cue
                    for item in value
                    for cue in item.get("cues", [])
                ]
                words = [
                    word
                    for item in value
                    for word in item.get("words", [])
                ]
                return {"cues": cues, "words": words}
            return {"cues": list(value)}
        raise ValueError("subtitles must be a subtitle track, cues or mappings")

    @field_validator("subtitle_file", "platform", mode="before")
    @classmethod
    def _strip_subtitle_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("loudness_lufs", mode="before")
    @classmethod
    def _validate_loudness(cls, value: Any) -> Any:
        if value is None:
            return None
        return _finite(value, "loudness_lufs")

    @field_validator("id", mode="before")
    @classmethod
    def _strip_id(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("duration_seconds", mode="before")
    @classmethod
    def _validate_duration(cls, value: Any) -> Any:
        return _finite(value, "duration_seconds")

    @field_validator("fps", mode="before")
    @classmethod
    def _validate_fps(cls, value: Any) -> Any:
        return _integer(value, "fps")

    @property
    def width(self) -> int:
        """The canvas width in pixels."""
        return self.canvas.width

    @property
    def height(self) -> int:
        """The canvas height in pixels."""
        return self.canvas.height

    @property
    def aspect_ratio(self) -> float:
        """The canvas width divided by its height."""
        return self.canvas.aspect_ratio

    @property
    def duration(self) -> float:
        """Alias for :attr:`duration_seconds`."""
        return self.duration_seconds

    @property
    def version(self) -> int:
        """Alias for :attr:`schema_version`."""
        return self.schema_version

    @property
    def frame_rate(self) -> int:
        """Alias for :attr:`fps`."""
        return self.fps

    @property
    def tracks(self) -> list[AudioTrack]:
        """Alias for :attr:`audio_tracks`."""
        return self.audio_tracks

    @property
    def audio(self) -> list[AudioTrack]:
        """Alias for :attr:`audio_tracks`."""
        return self.audio_tracks

    @property
    def subtitle(self) -> SubtitleTrack | None:
        """Alias for :attr:`subtitles`."""
        return self.subtitles

    @property
    def subtitle_track(self) -> SubtitleTrack | None:
        """Alias for :attr:`subtitles`."""
        return self.subtitles

    @property
    def captions(self) -> SubtitleTrack | None:
        """Alias for :attr:`subtitles`."""
        return self.subtitles

    @property
    def frame_count(self) -> int:
        """The number of frames in the exact duration."""
        return round(self.duration_seconds * self.fps)

    @property
    def layer_count(self) -> int:
        """The number of layers in the composition."""
        return len(self.layers)

    def layer(self, layer_id: str) -> Layer | None:
        """Return the layer with *layer_id*, or ``None`` when absent."""
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        return None

    def track(self, track_id: str) -> AudioTrack | None:
        """Return the audio track with *track_id*, or ``None`` when absent."""
        for track in self.audio_tracks:
            if track.id == track_id:
                return track
        return None

    def effect(self, effect_id: str) -> Effect | None:
        """Return the effect with *effect_id*, or ``None`` when absent."""
        for effect in self.effects:
            if effect.id == effect_id:
                return effect
        return None


CompositionSpec = Composition
CompositionConfig = Composition
CompositionSchema = Composition

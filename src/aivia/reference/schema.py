"""Reference inputs and the structured style spec extracted from them.

A reference is one user-supplied example: an image, a video, an audio clip or a
piece of text. Analysis turns it into a :class:`StyleSpec`: a human-editable,
versioned description of the structure, pacing, palette, typography, music
mood and aspect that should be reproduced with new content. The models here are
plain data. They never read a file, call a provider or import the media
modules.

Every nested section is optional because analysis may learn only part of the
style from a given reference. A section that was not measured stays empty
instead of guessing, and a style spec can be merged with another later. The
schema is versioned and serializes canonically, so a spec written by one run can
be read back, edited and reloaded without losing its meaning.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, Self, cast

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from aivia.core.models.base import canonical_json

__all__ = [
    "CURRENT_REFERENCE_SCHEMA_VERSION",
    "CURRENT_STYLE_SPEC_SCHEMA_VERSION",
    "REFERENCE_SCHEMA_VERSION",
    "STYLE_SPEC_SCHEMA_VERSION",
    "Analysis",
    "Aspect",
    "AspectConfig",
    "AspectModel",
    "AspectOrientation",
    "AspectRatio",
    "AspectSchema",
    "AspectSpec",
    "CaptionStyle",
    "ColorTemperature",
    "Music",
    "MusicMood",
    "MusicMoodConfig",
    "MusicMoodModel",
    "MusicMoodSchema",
    "MusicMoodSpec",
    "MusicSpec",
    "Pacing",
    "PacingConfig",
    "PacingModel",
    "PacingSchema",
    "PacingSpec",
    "Palette",
    "PaletteConfig",
    "PaletteModel",
    "PaletteSchema",
    "PaletteSpec",
    "Reference",
    "ReferenceAnalysis",
    "ReferenceConfig",
    "ReferenceKind",
    "ReferenceModel",
    "ReferenceSchema",
    "ReferenceSpec",
    "ReferenceSpecModel",
    "Scene",
    "SceneConfig",
    "SceneModel",
    "SceneSchema",
    "SceneSpec",
    "Shot",
    "ShotSpec",
    "ShotType",
    "Structure",
    "StructureConfig",
    "StructureModel",
    "StructureSchema",
    "StructureSpec",
    "Style",
    "StyleSpec",
    "StyleSpecConfig",
    "StyleSpecModel",
    "StyleSpecSchema",
    "StyleSpecification",
    "TextAlignment",
    "TextCase",
    "Typography",
    "TypographyConfig",
    "TypographyModel",
    "TypographySchema",
    "TypographySpec",
    "build_reference_schema",
    "build_style_spec_schema",
    "reference_json_schema",
    "style_spec_json_schema",
]

CURRENT_REFERENCE_SCHEMA_VERSION = 1
"""The reference schema version written by this release."""

REFERENCE_SCHEMA_VERSION = CURRENT_REFERENCE_SCHEMA_VERSION
"""Alias for :data:`CURRENT_REFERENCE_SCHEMA_VERSION`."""

CURRENT_STYLE_SPEC_SCHEMA_VERSION = 1
"""The style spec schema version written by this release."""

STYLE_SPEC_SCHEMA_VERSION = CURRENT_STYLE_SPEC_SCHEMA_VERSION
"""Alias for :data:`CURRENT_STYLE_SPEC_SCHEMA_VERSION`."""

_HEX_COLOR = re.compile(r"^#(?:[0-9a-f]{3,4}|[0-9a-f]{6}|[0-9a-f]{8})$")
_RATIO = re.compile(r"^(?P<width>\d+(?:\.\d+)?)\s*[:/]\s*(?P<height>\d+(?:\.\d+)?)$")
_RESOLUTION = re.compile(r"^(?P<width>\d+)\s*[xX\u00d7]\s*(?P<height>\d+)$")
_WEIGHT_NAMES = {
    "thin": 100,
    "hairline": 100,
    "extralight": 200,
    "extra_light": 200,
    "ultralight": 200,
    "light": 300,
    "regular": 400,
    "normal": 400,
    "book": 400,
    "medium": 500,
    "semibold": 600,
    "semi_bold": 600,
    "demibold": 600,
    "bold": 700,
    "extrabold": 800,
    "extra_bold": 800,
    "black": 900,
    "heavy": 900,
}


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if isinstance(value, Mapping):
        value = list(value.values())
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return []
    names: list[str] = []
    seen: set[str] = set()
    for entry in value:
        if not isinstance(entry, str):
            continue
        name = entry.strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _optional_finite(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return _finite(value, label)


def _ratio_unit(value: Any, label: str) -> float:
    number = _finite(value, label)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{label} must be between 0 and 1")
    return number


def _first(data: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _fold_aliases(
    data: Any,
    aliases: Mapping[str, tuple[str, ...]],
) -> Any:
    if not isinstance(data, Mapping):
        return data
    values = dict(data)
    for canonical, names in aliases.items():
        for name in names:
            if name in values:
                value = values.pop(name)
                values.setdefault(canonical, value)
    return values


def _normalise_color(value: Any, label: str = "colour") -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    colour = value.strip().lower()
    if not colour:
        raise ValueError(f"{label} must not be empty")
    if not colour.startswith("#"):
        return colour
    if not _HEX_COLOR.fullmatch(colour):
        raise ValueError(f"{label} must be a hex colour like #112233: {value!r}")
    digits = colour[1:]
    if len(digits) in (3, 4):
        colour = "#" + "".join(character * 2 for character in digits)
    return colour


def _coerce_colors(value: Any) -> Any:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return list(value.values())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        colors: list[Any] = []
        for entry in value:
            if isinstance(entry, Mapping):
                colors.append(_first(entry, "hex", "color", "colour", "value", "name"))
            else:
                colors.append(entry)
        return colors
    return value


def _normalise_colors(value: Any) -> list[str]:
    colors = _coerce_colors(value)
    if not isinstance(colors, list):
        raise ValueError("colors must be a string or a list of colours")
    normalised: list[str] = []
    for raw in colors:
        if raw is None:
            continue
        colour = _normalise_color(raw)
        if colour not in normalised:
            normalised.append(colour)
    return normalised


class AspectOrientation(StrEnum):
    """How a frame is oriented relative to its width and height."""

    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"
    SQUARE = "square"

    @classmethod
    def _missing_(cls, value: Any) -> AspectOrientation | None:
        if not isinstance(value, str):
            return None
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "vertical": cls.PORTRAIT,
            "tall": cls.PORTRAIT,
            "portrait_mode": cls.PORTRAIT,
            "horizontal": cls.LANDSCAPE,
            "wide": cls.LANDSCAPE,
            "landscape_mode": cls.LANDSCAPE,
            "square_1_1": cls.SQUARE,
            "1_1": cls.SQUARE,
        }
        member = cls._value2member_map_.get(key)
        if member is not None:
            return cast(AspectOrientation, member)
        return aliases.get(key)


class ShotType(StrEnum):
    """The camera framing of one scene."""

    WIDE = "wide"
    MEDIUM = "medium"
    CLOSE_UP = "close_up"
    EXTREME_CLOSE_UP = "extreme_close_up"
    OVER_THE_SHOULDER = "over_the_shoulder"
    POINT_OF_VIEW = "point_of_view"
    AERIAL = "aerial"
    MACRO = "macro"
    INSERT = "insert"
    B_ROLL = "b_roll"
    TALKING_HEAD = "talking_head"
    ANIMATION = "animation"
    STILL = "still"

    @classmethod
    def _missing_(cls, value: Any) -> ShotType | None:
        if not isinstance(value, str):
            return None
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "wide_shot": cls.WIDE,
            "establishing": cls.WIDE,
            "establishing_shot": cls.WIDE,
            "medium_shot": cls.MEDIUM,
            "mid": cls.MEDIUM,
            "mid_shot": cls.MEDIUM,
            "close": cls.CLOSE_UP,
            "closeup": cls.CLOSE_UP,
            "close_shot": cls.CLOSE_UP,
            "extreme_close": cls.EXTREME_CLOSE_UP,
            "ecu": cls.EXTREME_CLOSE_UP,
            "ots": cls.OVER_THE_SHOULDER,
            "pov": cls.POINT_OF_VIEW,
            "drone": cls.AERIAL,
            "birds_eye": cls.AERIAL,
            "bird_eye": cls.AERIAL,
            "broll": cls.B_ROLL,
            "b-roll": cls.B_ROLL,
            "talkinghead": cls.TALKING_HEAD,
            "talking_head_shot": cls.TALKING_HEAD,
            "animated": cls.ANIMATION,
            "image": cls.STILL,
            "photo": cls.STILL,
        }
        member = cls._value2member_map_.get(key)
        if member is not None:
            return cast(ShotType, member)
        return aliases.get(key)


class TextAlignment(StrEnum):
    """Horizontal alignment of typography."""

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    JUSTIFY = "justify"

    @classmethod
    def _missing_(cls, value: Any) -> TextAlignment | None:
        if not isinstance(value, str):
            return None
        key = value.strip().lower()
        aliases = {
            "centre": cls.CENTER,
            "middle": cls.CENTER,
            "start": cls.LEFT,
            "end": cls.RIGHT,
            "both": cls.JUSTIFY,
        }
        member = cls._value2member_map_.get(key)
        if member is not None:
            return cast(TextAlignment, member)
        return aliases.get(key)


class TextCase(StrEnum):
    """How letter case is presented."""

    NONE = "none"
    UPPER = "upper"
    LOWER = "lower"
    TITLE = "title"
    SENTENCE = "sentence"

    @classmethod
    def _missing_(cls, value: Any) -> TextCase | None:
        if not isinstance(value, str):
            return None
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "uppercase": cls.UPPER,
            "all_caps": cls.UPPER,
            "caps": cls.UPPER,
            "lowercase": cls.LOWER,
            "title_case": cls.TITLE,
            "sentence_case": cls.SENTENCE,
            "normal": cls.NONE,
            "original": cls.NONE,
        }
        member = cls._value2member_map_.get(key)
        if member is not None:
            return cast(TextCase, member)
        return aliases.get(key)


class ColorTemperature(StrEnum):
    """The overall warmth of a palette."""

    WARM = "warm"
    COOL = "cool"
    NEUTRAL = "neutral"

    @classmethod
    def _missing_(cls, value: Any) -> ColorTemperature | None:
        if not isinstance(value, str):
            return None
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "cold": cls.COOL,
            "cool_tones": cls.COOL,
            "warm_tones": cls.WARM,
            "balanced": cls.NEUTRAL,
        }
        member = cls._value2member_map_.get(key)
        if member is not None:
            return cast(ColorTemperature, member)
        return aliases.get(key)


class CaptionStyle(StrEnum):
    """The visual treatment of burned-in captions."""

    NONE = "none"
    BURNED_IN = "burned_in"
    SUBTITLES = "subtitles"
    KARAOKE = "karaoke"
    LOWER_THIRD = "lower_third"
    CAPTIONS = "captions"

    @classmethod
    def _missing_(cls, value: Any) -> CaptionStyle | None:
        if not isinstance(value, str):
            return None
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "burn_in": cls.BURNED_IN,
            "burned": cls.BURNED_IN,
            "hardcoded": cls.BURNED_IN,
            "subtitle": cls.SUBTITLES,
            "srt": cls.SUBTITLES,
            "word_by_word": cls.KARAOKE,
            "lowerthird": cls.LOWER_THIRD,
            "caption": cls.CAPTIONS,
            "on_screen": cls.CAPTIONS,
        }
        member = cls._value2member_map_.get(key)
        if member is not None:
            return cast(CaptionStyle, member)
        return aliases.get(key)


class _SpecModel(BaseModel):
    """Base for reference values with canonical serialization helpers."""

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        validate_assignment=True,
    )

    def to_dict(
        self,
        *,
        exclude_none: bool = False,
        by_alias: bool = False,
    ) -> dict[str, Any]:
        """Return this value as a JSON-compatible mapping."""
        return self.model_dump(
            mode="json",
            exclude_none=exclude_none,
            by_alias=by_alias,
        )

    def to_json(
        self,
        *,
        indent: int | None = None,
        exclude_none: bool = False,
    ) -> str:
        """Return this value as canonical JSON."""
        return canonical_json(
            self.to_dict(exclude_none=exclude_none),
            indent=indent,
        )

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Self:
        """Validate a mapping into this model."""
        return cls.model_validate(dict(values))

    @classmethod
    def from_json(cls, text: str | bytes) -> Self:
        """Read canonical JSON back into this model."""
        return cls.model_validate_json(text)

    def model_dump_json(self, **kwargs: Any) -> str:
        """Return canonical JSON, overriding Pydantic's field-order output."""
        indent = kwargs.pop("indent", None)
        ensure_ascii = kwargs.pop("ensure_ascii", False)
        data = self.model_dump(mode="json", **kwargs)
        return canonical_json(data, indent=indent, ensure_ascii=ensure_ascii)


def _aspect_token(value: Any) -> tuple[str, Any]:
    """Return ``("ratio", ...)`` or ``("resolution", ...)`` for one token."""
    if not isinstance(value, str):
        raise ValueError("aspect must be a string")
    token = value.strip()
    if not token:
        return ("", token)
    if _RATIO.fullmatch(token):
        return ("ratio", token)
    if _RESOLUTION.fullmatch(token):
        return ("resolution", token)
    raise ValueError(f"aspect must look like 9:16 or 1080x1920: {value!r}")


def _ratio_parts(value: str) -> tuple[float, float]:
    match = _RATIO.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"aspect ratio must look like 9:16: {value!r}")
    width = float(match.group("width"))
    height = float(match.group("height"))
    if width <= 0.0 or height <= 0.0:
        raise ValueError("aspect ratio sides must be positive")
    return width, height


def _resolution_parts(value: str) -> tuple[int, int]:
    match = _RESOLUTION.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"resolution must look like 1080x1920: {value!r}")
    width = int(match.group("width"))
    height = int(match.group("height"))
    if width <= 0 or height <= 0:
        raise ValueError("resolution sides must be positive")
    return width, height


def _orientation_for(width: float, height: float) -> AspectOrientation:
    if width > height:
        return AspectOrientation.LANDSCAPE
    if width < height:
        return AspectOrientation.PORTRAIT
    return AspectOrientation.SQUARE


def _reduce_ratio(width: float, height: float) -> str:
    if width.is_integer() and height.is_integer():
        left = int(width)
        right = int(height)
        divisor = math.gcd(left, right)
        return f"{left // divisor}:{right // divisor}"
    return f"{width:g}:{height:g}"


class AspectSpec(_SpecModel):
    """The frame shape: an aspect ratio, pixel resolution or orientation.

    A single string is accepted as shorthand. ``9:16`` is read as a ratio and
    ``1080x1920`` as a resolution. Pixel dimensions derive the reduced ratio
    and the orientation, while a ratio alone derives the orientation. The
    fields are independent so an extracted spec may know only the orientation.
    """

    ratio: str = Field(
        default="",
        validation_alias=AliasChoices("ratio", "aspect_ratio", "aspect", "shape"),
    )
    width: int | None = Field(
        default=None,
        gt=0,
        validation_alias=AliasChoices("width", "w", "width_px"),
    )
    height: int | None = Field(
        default=None,
        gt=0,
        validation_alias=AliasChoices("height", "h", "height_px"),
    )
    orientation: AspectOrientation | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "orientation",
            "direction",
            "format",
            "layout",
        ),
    )
    resolution: str = Field(
        default="",
        validation_alias=AliasChoices("resolution", "size", "dimensions"),
    )

    @model_validator(mode="before")
    @classmethod
    def _fold_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            kind, token = _aspect_token(data)
            return {kind: token} if kind else {}
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        ratio_value = _first(values, "ratio", "aspect_ratio", "aspect", "shape")
        resolution_value = _first(values, "resolution", "size", "dimensions")
        if ratio_value is not None:
            kind, token = _aspect_token(ratio_value)
            if kind == "resolution":
                values.setdefault("resolution", token)
                values.pop("ratio", None)
                values.pop("aspect_ratio", None)
                values.pop("aspect", None)
                values.pop("shape", None)
            elif kind == "ratio":
                values["ratio"] = token
        if resolution_value is not None:
            kind, token = _aspect_token(resolution_value)
            if kind == "ratio":
                values.setdefault("ratio", token)
                values.pop("resolution", None)
                values.pop("size", None)
                values.pop("dimensions", None)
            elif kind == "resolution":
                values["resolution"] = token
        width = _first(values, "width", "w", "width_px")
        height = _first(values, "height", "h", "height_px")
        if width is not None:
            values["width"] = width
        if height is not None:
            values["height"] = height
        orientation = _first(
            values,
            "orientation",
            "direction",
            "format",
            "layout",
        )
        if orientation is not None:
            values["orientation"] = orientation
        return values

    @model_validator(mode="after")
    def _derive_shape(self) -> Self:
        ratio = self.ratio.strip()
        resolution = self.resolution.strip()
        width = self.width
        height = self.height

        if ratio:
            ratio_width, ratio_height = _ratio_parts(ratio)
            ratio = _reduce_ratio(ratio_width, ratio_height)
            if self.orientation is None:
                object.__setattr__(
                    self,
                    "orientation",
                    _orientation_for(ratio_width, ratio_height),
                )
        if resolution:
            width, height = _resolution_parts(resolution)
            resolution = f"{width}x{height}"
            if not ratio:
                ratio = _reduce_ratio(float(width), float(height))
            if self.orientation is None:
                object.__setattr__(
                    self,
                    "orientation",
                    _orientation_for(float(width), float(height)),
                )
        elif width is not None and height is not None:
            resolution = f"{width}x{height}"
            if not ratio:
                ratio = _reduce_ratio(float(width), float(height))
            if self.orientation is None:
                object.__setattr__(
                    self,
                    "orientation",
                    _orientation_for(float(width), float(height)),
                )

        object.__setattr__(self, "ratio", ratio)
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        return self

    @property
    def aspect_ratio(self) -> float | None:
        """Width divided by height, or ``None`` until the shape is known."""
        if self.width and self.height:
            return self.width / self.height
        if self.ratio:
            width, height = _ratio_parts(self.ratio)
            return width / height
        return None

    @property
    def dimensions(self) -> tuple[int, int] | None:
        """The ``(width, height)`` pair, or ``None`` when no pixels are known."""
        if self.width is None or self.height is None:
            return None
        return (self.width, self.height)

    @property
    def is_portrait(self) -> bool:
        """Whether the frame is taller than it is wide."""
        return self.orientation is AspectOrientation.PORTRAIT

    @property
    def is_landscape(self) -> bool:
        """Whether the frame is wider than it is tall."""
        return self.orientation is AspectOrientation.LANDSCAPE

    @property
    def is_square(self) -> bool:
        """Whether the frame has equal sides."""
        return self.orientation is AspectOrientation.SQUARE


class SceneSpec(_SpecModel):
    """One shot or scene in the reference structure.

    Start and end are absolute seconds. A duration may be supplied instead of
    one endpoint and the missing endpoint is derived, so analysis can report
    whichever pair it measured. Scenes are kept in the order supplied.
    """

    index: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("index", "order", "position", "number"),
    )
    start_seconds: float | None = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices("start_seconds", "start", "from"),
    )
    end_seconds: float | None = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices("end_seconds", "end", "to"),
    )
    duration_seconds: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("duration_seconds", "duration", "length"),
    )
    shot_type: ShotType | None = Field(
        default=None,
        validation_alias=AliasChoices("shot_type", "shot", "framing", "camera"),
    )
    description: str = Field(
        default="",
        validation_alias=AliasChoices("description", "summary", "prompt"),
    )
    text: str = Field(
        default="",
        validation_alias=AliasChoices(
            "text",
            "on_screen_text",
            "onscreen_text",
            "caption",
            "title",
        ),
    )
    transition: str = ""
    camera_motion: str = Field(
        default="",
        validation_alias=AliasChoices(
            "camera_motion",
            "motion",
            "movement",
            "camera_move",
        ),
    )
    has_caption: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "has_caption",
            "captioned",
            "has_text",
            "has_subtitle",
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _fold_aliases(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        start = _first(values, "start_seconds", "start", "from")
        end = _first(values, "end_seconds", "end", "to")
        duration = _first(values, "duration_seconds", "duration", "length")
        if start is not None:
            values["start_seconds"] = start
        if end is not None:
            values["end_seconds"] = end
        if duration is not None:
            values["duration_seconds"] = duration
        return values

    @model_validator(mode="after")
    def _derive_timing(self) -> Self:
        start = self.start_seconds
        end = self.end_seconds
        span = self.duration_seconds
        if start is not None and end is not None:
            if end < start:
                raise ValueError("scene end_seconds must not precede start_seconds")
            span = end - start
        elif start is not None and span is not None:
            end = start + span
        elif end is not None and span is not None:
            start = max(end - span, 0.0)
        if span is not None and span <= 0.0:
            raise ValueError("scene duration_seconds must be positive")
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)
        object.__setattr__(self, "duration_seconds", span)
        return self

    @field_validator(
        "description", "text", "transition", "camera_motion", mode="before"
    )
    @classmethod
    def _strip_text(cls, value: Any) -> str:
        return _text(value)

    @property
    def start(self) -> float | None:
        """Alias for :attr:`start_seconds`."""
        return self.start_seconds

    @property
    def end(self) -> float | None:
        """Alias for :attr:`end_seconds`."""
        return self.end_seconds

    @property
    def duration(self) -> float | None:
        """Alias for :attr:`duration_seconds`."""
        return self.duration_seconds


class StructureSpec(_SpecModel):
    """The ordered scenes, hook and duration of a reference.

    ``scenes`` carries the measured shots. ``beats`` is the narrative outline
    when the analyser reports one without exact timings. ``total_duration`` and
    the derived ``duration`` always agree with the sum of timed scenes when
    scenes are present, so later structure matching can compare a candidate to
    the same number.
    """

    scenes: list[SceneSpec] = Field(
        default_factory=list,
        validation_alias=AliasChoices("scenes", "shots", "segments", "timeline"),
    )
    scene_count: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("scene_count", "shots_count", "count"),
    )
    hook: str = ""
    hook_seconds: float | None = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices("hook_seconds", "hook_duration"),
    )
    total_duration_seconds: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices(
            "total_duration_seconds",
            "duration_seconds",
            "duration",
            "total_duration",
        ),
    )
    beats: list[str] = Field(default_factory=list)
    shot_types: list[ShotType] = Field(
        default_factory=list,
        validation_alias=AliasChoices("shot_types", "shots", "framings"),
    )
    transitions: list[str] = Field(default_factory=list)
    narrative_arc: str = Field(
        default="",
        validation_alias=AliasChoices(
            "narrative_arc",
            "arc",
            "story_arc",
            "structure",
        ),
    )
    acts: int | None = Field(default=None, ge=1)

    @model_validator(mode="before")
    @classmethod
    def _fold_aliases(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        scenes = _first(values, "scenes", "shots", "segments", "timeline")
        if scenes is not None:
            values["scenes"] = scenes
            values.pop("shots", None)
        duration = _first(
            values,
            "total_duration_seconds",
            "duration_seconds",
            "duration",
            "total_duration",
        )
        if duration is not None:
            values["total_duration_seconds"] = duration
        beats = values.get("beats")
        if beats is not None and isinstance(beats, Mapping):
            values["beats"] = list(beats.values())
        return values

    @field_validator("beats", "transitions", mode="before")
    @classmethod
    def _coerce_names(cls, value: Any) -> list[str]:
        return _text_list(value)

    @field_validator("shot_types", mode="before")
    @classmethod
    def _coerce_shot_types(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("hook", "narrative_arc", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> str:
        return _text(value)

    @field_validator("hook_seconds", "total_duration_seconds", mode="before")
    @classmethod
    def _finite_timing(cls, value: Any) -> Any:
        return _optional_finite(value, "structure timing")

    @model_validator(mode="after")
    def _derive_structure(self) -> Self:
        if self.shot_types == [] and self.scenes:
            seen: list[ShotType] = []
            for scene in self.scenes:
                if scene.shot_type is not None and scene.shot_type not in seen:
                    seen.append(scene.shot_type)
            object.__setattr__(self, "shot_types", seen)
        durations = [
            scene.duration_seconds
            for scene in self.scenes
            if scene.duration_seconds is not None
        ]
        if self.scene_count is None and self.scenes:
            object.__setattr__(self, "scene_count", len(self.scenes))
        if durations and self.total_duration_seconds is None:
            object.__setattr__(self, "total_duration_seconds", sum(durations))
        return self

    @property
    def duration(self) -> float | None:
        """Alias for :attr:`total_duration_seconds`."""
        return self.total_duration_seconds

    @property
    def duration_seconds(self) -> float | None:
        """Alias for :attr:`total_duration_seconds`."""
        return self.total_duration_seconds

    @property
    def cuts(self) -> int:
        """The number of cuts implied by the scenes."""
        return max(len(self.scenes) - 1, 0)

    @property
    def scene_durations(self) -> list[float]:
        """The duration of every timed scene, in order."""
        return [
            scene.duration_seconds
            for scene in self.scenes
            if scene.duration_seconds is not None
        ]


class PacingSpec(_SpecModel):
    """How quickly a reference moves: cuts, shot length and energy.

    ``cuts_per_minute`` is the primary measure. The shot length bounds let a
    later structure matcher accept a candidate that varies inside the observed
    range instead of requiring exact cut times. ``style`` keeps the analyser's
    own word, such as ``fast`` or ``slow``, without replacing the numbers.
    """

    cuts_per_minute: float | None = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices(
            "cuts_per_minute",
            "cuts",
            "cut_rate",
            "cpm",
        ),
    )
    average_shot_seconds: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices(
            "average_shot_seconds",
            "average_shot_length",
            "average_scene_seconds",
            "avg_shot_seconds",
            "shot_length",
        ),
    )
    min_shot_seconds: float | None = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices(
            "min_shot_seconds",
            "shortest_shot_seconds",
            "min_scene_seconds",
        ),
    )
    max_shot_seconds: float | None = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices(
            "max_shot_seconds",
            "longest_shot_seconds",
            "max_scene_seconds",
        ),
    )
    words_per_second: float | None = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices(
            "words_per_second",
            "wps",
            "speech_rate",
        ),
    )
    energy: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("energy", "intensity", "pace_energy"),
    )
    style: str = Field(
        default="",
        validation_alias=AliasChoices("style", "tempo", "feel", "label"),
    )

    @field_validator(
        "cuts_per_minute",
        "average_shot_seconds",
        "min_shot_seconds",
        "max_shot_seconds",
        "words_per_second",
        "energy",
        mode="before",
    )
    @classmethod
    def _finite_number(cls, value: Any) -> Any:
        return _optional_finite(value, "pacing value")

    @field_validator("style", mode="before")
    @classmethod
    def _strip_style(cls, value: Any) -> str:
        return _text(value)

    @model_validator(mode="after")
    def _validate_bounds(self) -> Self:
        if (
            self.min_shot_seconds is not None
            and self.max_shot_seconds is not None
            and self.min_shot_seconds > self.max_shot_seconds
        ):
            raise ValueError("min_shot_seconds must not exceed max_shot_seconds")
        return self

    @property
    def cuts(self) -> float | None:
        """Alias for :attr:`cuts_per_minute`."""
        return self.cuts_per_minute

    @property
    def avg_shot_seconds(self) -> float | None:
        """Alias for :attr:`average_shot_seconds`."""
        return self.average_shot_seconds

    @property
    def is_fast(self) -> bool:
        """Whether the measured cut rate is at least one cut per second."""
        return (self.cuts_per_minute or 0.0) >= 60.0

    @property
    def is_slow(self) -> bool:
        """Whether the measured cut rate is at most one cut every four seconds."""
        return (self.cuts_per_minute or 0.0) <= 15.0


class PaletteSpec(_SpecModel):
    """The dominant colours and colour treatment of a reference.

    ``colors`` is ordered most-frequent first and normalized to lowercase hex
    when a hex value is supplied. Named role fields are independent, so an
    analyser may report only a background or accent. Saturation and contrast are
    normalized to ``0..1``; an unknown value stays ``None``.
    """

    colors: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "colors",
            "colours",
            "dominant_colors",
            "palette",
            "swatches",
        ),
    )
    primary: str = ""
    secondary: str = ""
    background: str = ""
    foreground: str = ""
    accent: str = ""
    temperature: ColorTemperature | None = None
    saturation: float | None = Field(default=None, ge=0.0, le=1.0)
    contrast: float | None = Field(default=None, ge=0.0, le=1.0)
    scheme: str = Field(
        default="",
        validation_alias=AliasChoices("scheme", "harmony", "palette_type"),
    )

    @field_validator("colors", mode="before")
    @classmethod
    def _validate_colors(cls, value: Any) -> list[str]:
        return _normalise_colors(value)

    @field_validator(
        "primary",
        "secondary",
        "background",
        "foreground",
        "accent",
        mode="before",
    )
    @classmethod
    def _validate_role_color(cls, value: Any) -> str:
        if value is None or value == "":
            return ""
        return _normalise_color(value)

    @field_validator("temperature", mode="before")
    @classmethod
    def _normalise_temperature(cls, value: Any) -> Any:
        if value is None or isinstance(value, ColorTemperature):
            return value
        if isinstance(value, str):
            key = value.strip().lower()
            if key in ("", "unknown", "mixed"):
                return None
        return value

    @field_validator("saturation", "contrast", mode="before")
    @classmethod
    def _validate_ratio(cls, value: Any) -> Any:
        if value is None:
            return None
        return _ratio_unit(value, "palette value")

    @field_validator("scheme", mode="before")
    @classmethod
    def _strip_scheme(cls, value: Any) -> str:
        return _text(value)

    @property
    def dominant_colors(self) -> list[str]:
        """Alias for :attr:`colors`."""
        return self.colors

    @property
    def swatches(self) -> list[str]:
        """Alias for :attr:`colors`."""
        return self.colors

    @property
    def roles(self) -> dict[str, str]:
        """The named role colours that are set."""
        roles = {
            "primary": self.primary,
            "secondary": self.secondary,
            "background": self.background,
            "foreground": self.foreground,
            "accent": self.accent,
        }
        return {name: value for name, value in roles.items() if value}


class TypographySpec(_SpecModel):
    """Fonts, sizes, alignment and caption treatment.

    ``font_weight`` accepts either a CSS number or a familiar name such as
    ``bold``. Sizes are in pixels, ``line_height`` is a multiplier and
    ``letter_spacing`` is in pixels, matching the composition text layer. A
    reference may know only a font family or only a caption style.
    """

    font_family: str = Field(
        default="",
        validation_alias=AliasChoices(
            "font_family",
            "font",
            "family",
            "primary_font",
        ),
    )
    fonts: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("fonts", "typefaces", "families"),
    )
    heading_font: str = ""
    body_font: str = ""
    font_size: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("font_size", "size", "font_size_px"),
    )
    heading_size: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("heading_size", "title_size", "heading_size_px"),
    )
    body_size: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("body_size", "body_size_px"),
    )
    font_weight: int | None = Field(
        default=None,
        ge=1,
        le=1000,
        validation_alias=AliasChoices("font_weight", "weight", "font-weight"),
    )
    font_style: str = Field(
        default="",
        validation_alias=AliasChoices("font_style", "style", "font-style"),
    )
    align: TextAlignment = Field(
        default=TextAlignment.CENTER,
        validation_alias=AliasChoices("align", "text_align", "alignment"),
    )
    case: TextCase = Field(
        default=TextCase.NONE,
        validation_alias=AliasChoices("case", "text_case", "capitalization"),
    )
    line_height: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("line_height", "line_spacing", "line-height"),
    )
    letter_spacing: float | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "letter_spacing",
            "tracking",
            "letter-spacing",
        ),
    )
    color: str = Field(
        default="",
        validation_alias=AliasChoices("color", "colour", "text_color", "fill"),
    )
    placement: str = Field(
        default="",
        validation_alias=AliasChoices("placement", "position", "location"),
    )
    caption_style: CaptionStyle = Field(
        default=CaptionStyle.NONE,
        validation_alias=AliasChoices(
            "caption_style",
            "captions",
            "caption",
            "subtitles",
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _fold_aliases(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        family = _first(
            values,
            "font_family",
            "font",
            "family",
            "primary_font",
        )
        if family is not None:
            values["font_family"] = family
        fonts = _first(values, "fonts", "typefaces", "families")
        if fonts is not None:
            values["fonts"] = fonts
        weight = _first(values, "font_weight", "weight", "font-weight")
        if weight is not None:
            values["font_weight"] = weight
        return values

    @field_validator("fonts", mode="before")
    @classmethod
    def _coerce_fonts(cls, value: Any) -> list[str]:
        return _text_list(value)

    @field_validator(
        "font_family",
        "heading_font",
        "body_font",
        "font_style",
        "placement",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: Any) -> str:
        return _text(value)

    @field_validator("color", mode="before")
    @classmethod
    def _validate_color(cls, value: Any) -> str:
        if value is None or value == "":
            return ""
        return _normalise_color(value, "typography colour")

    @field_validator("font_weight", mode="before")
    @classmethod
    def _normalise_weight(cls, value: Any) -> Any:
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, str):
            key = value.strip().lower().replace("-", "_").replace(" ", "_")
            if not key:
                return None
            if key in _WEIGHT_NAMES:
                return _WEIGHT_NAMES[key]
            if key.isdigit():
                return int(key)
            raise ValueError(f"unknown font weight: {value!r}")
        return value

    @field_validator(
        "font_size",
        "heading_size",
        "body_size",
        "line_height",
        "letter_spacing",
        mode="before",
    )
    @classmethod
    def _finite_number(cls, value: Any) -> Any:
        return _optional_finite(value, "typography value")

    @property
    def typefaces(self) -> list[str]:
        """The known font families, primary first."""
        names: list[str] = []
        for name in [self.font_family, *self.fonts, self.heading_font, self.body_font]:
            if name and name not in names:
                names.append(name)
        return names

    @property
    def primary_font(self) -> str:
        """The first known font family, or an empty string."""
        names = self.typefaces
        return names[0] if names else ""

    @property
    def font(self) -> str:
        """Alias for :attr:`font_family`."""
        return self.font_family

    @property
    def weight(self) -> int | None:
        """Alias for :attr:`font_weight`."""
        return self.font_weight


class MusicMoodSpec(_SpecModel):
    """Genre, mood and musical traits that a soundtrack should match.

    ``bpm`` is tempo in beats per minute, ``energy`` and ``valence`` are
    normalized to ``0..1`` and ``vocals`` is a short description such as
    ``instrumental`` or ``female``. The model stores no melody or verbatim
    lyric, only descriptive traits.
    """

    genre: str = ""
    genres: list[str] = Field(default_factory=list)
    mood: str = ""
    moods: list[str] = Field(default_factory=list)
    bpm: float | None = Field(
        default=None,
        ge=0.0,
        le=400.0,
        validation_alias=AliasChoices("bpm", "tempo", "beats_per_minute"),
    )
    key: str = ""
    scale: str = Field(
        default="",
        validation_alias=AliasChoices("scale", "mode", "tonality"),
    )
    instruments: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("instruments", "instrumentation"),
    )
    vocals: str = Field(
        default="",
        validation_alias=AliasChoices("vocals", "vocal", "voice"),
    )
    energy: float | None = Field(default=None, ge=0.0, le=1.0)
    valence: float | None = Field(default=None, ge=0.0, le=1.0)
    loudness_lufs: float | None = Field(
        default=None,
        validation_alias=AliasChoices("loudness_lufs", "loudness", "lufs"),
    )

    @model_validator(mode="before")
    @classmethod
    def _fold_aliases(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        tempo = _first(values, "bpm", "tempo", "beats_per_minute")
        if tempo is not None:
            values["bpm"] = tempo
        genres = _first(values, "genres", "genre")
        if genres is not None and "genres" not in values:
            values["genres"] = genres
        moods = _first(values, "moods", "mood")
        if moods is not None and "moods" not in values:
            values["moods"] = moods
        return values

    @field_validator("genres", "moods", "instruments", mode="before")
    @classmethod
    def _coerce_names(cls, value: Any) -> list[str]:
        return _text_list(value)

    @field_validator("genre", "mood", "key", "scale", "vocals", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> str:
        return _text(value)

    @field_validator("vocals", mode="before")
    @classmethod
    def _coerce_vocals(cls, value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "vocal" if value else "instrumental"
        return value

    @field_validator("bpm", "loudness_lufs", mode="before")
    @classmethod
    def _finite_number(cls, value: Any) -> Any:
        return _optional_finite(value, "music value")

    @field_validator("energy", "valence", mode="before")
    @classmethod
    def _validate_ratio(cls, value: Any) -> Any:
        if value is None:
            return None
        return _ratio_unit(value, "music mood value")

    @property
    def tempo(self) -> float | None:
        """Alias for :attr:`bpm`."""
        return self.bpm

    @property
    def beat_rate(self) -> float | None:
        """Alias for :attr:`bpm`."""
        return self.bpm

    @property
    def is_instrumental(self) -> bool:
        """Whether the vocals field describes an instrumental track."""
        return self.vocals.strip().lower() in {
            "instrumental",
            "none",
            "no vocals",
            "no_vocals",
        }


class StyleSpec(_SpecModel):
    """A structured, human-editable description extracted from references.

    The six nested sections are independent so a spec may be partially known.
    ``name`` and ``description`` make it selectable and explain it to a person,
    while ``prompt_suffix`` carries the short phrase a media prompt can reuse.
    ``references`` lists the ids or paths the spec was derived from and
    ``metadata`` keeps analyser provenance. Serialization is canonical and
    versioned so an edited file round-trips without losing data.
    """

    schema_version: int = Field(default=CURRENT_STYLE_SPEC_SCHEMA_VERSION, ge=1)
    name: str = ""
    description: str = ""
    prompt_suffix: str = Field(
        default="",
        validation_alias=AliasChoices(
            "prompt_suffix",
            "suffix",
            "prompt",
        ),
    )
    structure: StructureSpec = Field(default_factory=StructureSpec)
    pacing: PacingSpec = Field(default_factory=PacingSpec)
    palette: PaletteSpec = Field(default_factory=PaletteSpec)
    typography: TypographySpec = Field(default_factory=TypographySpec)
    music: MusicMoodSpec = Field(
        default_factory=MusicMoodSpec,
        validation_alias=AliasChoices(
            "music",
            "music_mood",
            "music_style",
            "audio",
            "sound",
        ),
    )
    aspect: AspectSpec = Field(
        default_factory=AspectSpec,
        validation_alias=AliasChoices(
            "aspect",
            "aspect_ratio",
            "shape",
            "format",
        ),
    )
    duration_seconds: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("duration_seconds", "duration", "length"),
    )
    shot_types: list[ShotType] = Field(
        default_factory=list,
        validation_alias=AliasChoices("shot_types", "shots", "framings"),
    )
    caption_style: CaptionStyle = Field(
        default=CaptionStyle.NONE,
        validation_alias=AliasChoices(
            "caption_style",
            "captions",
            "caption",
            "subtitles",
        ),
    )
    references: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "references",
            "reference_ids",
            "sources",
        ),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _fold_aliases(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        music = _first(
            values,
            "music",
            "music_mood",
            "music_style",
            "audio",
            "sound",
        )
        if music is not None:
            values["music"] = music
        aspect = _first(
            values,
            "aspect",
            "aspect_ratio",
            "shape",
            "format",
        )
        if aspect is not None:
            values["aspect"] = aspect
        duration = _first(
            values,
            "duration_seconds",
            "duration",
            "length",
        )
        if duration is not None:
            values["duration_seconds"] = duration
        references = _first(
            values,
            "references",
            "reference_ids",
            "sources",
        )
        if references is not None:
            values["references"] = references
        if "structure" not in values and any(
            key in values for key in ("scenes", "shots", "timeline")
        ):
            structure: dict[str, Any] = {}
            for key in (
                "scenes",
                "shots",
                "timeline",
                "beats",
                "hook",
                "hook_seconds",
                "narrative_arc",
                "transitions",
            ):
                if key in values:
                    structure[key] = values.pop(key)
            values["structure"] = structure
        return values

    @field_validator("name", "description", "prompt_suffix", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> str:
        return _text(value)

    @field_validator("shot_types", mode="before")
    @classmethod
    def _coerce_shot_types(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("references", mode="before")
    @classmethod
    def _coerce_references(cls, value: Any) -> list[str]:
        return _text_list(value)

    @field_validator("duration_seconds", mode="before")
    @classmethod
    def _finite_duration(cls, value: Any) -> Any:
        return _optional_finite(value, "style spec duration")

    @field_validator("metadata", mode="before")
    @classmethod
    def _coerce_metadata(cls, value: Any) -> dict[str, Any]:
        return {} if value is None else dict(value)

    @model_validator(mode="after")
    def _derive_sections(self) -> Self:
        if self.duration_seconds is None:
            object.__setattr__(
                self,
                "duration_seconds",
                self.structure.total_duration_seconds,
            )
        if not self.shot_types:
            object.__setattr__(
                self,
                "shot_types",
                list(self.structure.shot_types),
            )
        if (
            self.caption_style is not CaptionStyle.NONE
            and self.typography.caption_style is CaptionStyle.NONE
        ):
            object.__setattr__(
                self.typography,
                "caption_style",
                self.caption_style,
            )
        return self

    @property
    def music_mood(self) -> MusicMoodSpec:
        """Alias for :attr:`music`."""
        return self.music

    @property
    def duration(self) -> float | None:
        """Alias for :attr:`duration_seconds`."""
        return self.duration_seconds

    @property
    def aspect_ratio(self) -> str:
        """The declared aspect ratio, if any."""
        return self.aspect.ratio

    @property
    def composition(self) -> StructureSpec:
        """Alias for :attr:`structure`."""
        return self.structure

    @property
    def style(self) -> str:
        """The analyser's style phrase, falling back to the prompt suffix."""
        value = self.metadata.get("style")
        if isinstance(value, str) and value.strip():
            return value.strip()
        return self.prompt_suffix or self.description

    @property
    def is_empty(self) -> bool:
        """Whether the spec carries no name and no extracted section."""
        if self.name or self.description or self.prompt_suffix:
            return False
        sections = (
            self.structure,
            self.pacing,
            self.palette,
            self.typography,
            self.music,
            self.aspect,
        )
        return all(
            section.model_dump() == type(section)().model_dump() for section in sections
        )

    def to_style_dict(self) -> dict[str, Any]:
        """Return the spec with only the fields that carry information."""
        return self.to_dict(exclude_none=True)


class ReferenceKind(StrEnum):
    """The kind of example a reference provides."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"

    @classmethod
    def _missing_(cls, value: Any) -> ReferenceKind | None:
        if not isinstance(value, str):
            return None
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "picture": cls.IMAGE,
            "photo": cls.IMAGE,
            "photograph": cls.IMAGE,
            "still": cls.IMAGE,
            "image_file": cls.IMAGE,
            "clip": cls.VIDEO,
            "movie": cls.VIDEO,
            "film": cls.VIDEO,
            "video_file": cls.VIDEO,
            "song": cls.AUDIO,
            "music": cls.AUDIO,
            "track": cls.AUDIO,
            "sound": cls.AUDIO,
            "audio_file": cls.AUDIO,
            "document": cls.TEXT,
            "doc": cls.TEXT,
            "script": cls.TEXT,
            "article": cls.TEXT,
            "text_file": cls.TEXT,
        }
        member = cls._value2member_map_.get(key)
        if member is not None:
            return cast(ReferenceKind, member)
        return aliases.get(key)

    @classmethod
    def from_path(cls, path: str) -> ReferenceKind | None:
        """Infer the kind from a path or URL extension, if it is known."""
        return _kind_from_path(path)

    @classmethod
    def from_mime(cls, mime: str) -> ReferenceKind | None:
        """Infer the kind from a MIME type, if it is known."""
        return _kind_from_mime(mime)


_EXTENSION_KINDS = {
    ".png": ReferenceKind.IMAGE,
    ".jpg": ReferenceKind.IMAGE,
    ".jpeg": ReferenceKind.IMAGE,
    ".webp": ReferenceKind.IMAGE,
    ".gif": ReferenceKind.IMAGE,
    ".bmp": ReferenceKind.IMAGE,
    ".tif": ReferenceKind.IMAGE,
    ".tiff": ReferenceKind.IMAGE,
    ".avif": ReferenceKind.IMAGE,
    ".heic": ReferenceKind.IMAGE,
    ".mp4": ReferenceKind.VIDEO,
    ".mov": ReferenceKind.VIDEO,
    ".mkv": ReferenceKind.VIDEO,
    ".webm": ReferenceKind.VIDEO,
    ".avi": ReferenceKind.VIDEO,
    ".m4v": ReferenceKind.VIDEO,
    ".mpg": ReferenceKind.VIDEO,
    ".mpeg": ReferenceKind.VIDEO,
    ".mp3": ReferenceKind.AUDIO,
    ".wav": ReferenceKind.AUDIO,
    ".m4a": ReferenceKind.AUDIO,
    ".aac": ReferenceKind.AUDIO,
    ".flac": ReferenceKind.AUDIO,
    ".ogg": ReferenceKind.AUDIO,
    ".opus": ReferenceKind.AUDIO,
    ".wma": ReferenceKind.AUDIO,
    ".txt": ReferenceKind.TEXT,
    ".md": ReferenceKind.TEXT,
    ".markdown": ReferenceKind.TEXT,
    ".srt": ReferenceKind.TEXT,
    ".vtt": ReferenceKind.TEXT,
    ".csv": ReferenceKind.TEXT,
    ".json": ReferenceKind.TEXT,
    ".yaml": ReferenceKind.TEXT,
    ".yml": ReferenceKind.TEXT,
}

_MIME_KINDS = (
    ("image/", ReferenceKind.IMAGE),
    ("video/", ReferenceKind.VIDEO),
    ("audio/", ReferenceKind.AUDIO),
    ("text/", ReferenceKind.TEXT),
)


def _extension(value: str) -> str:
    name = value.strip().split("?", 1)[0].split("#", 1)[0]
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    if "." not in name:
        return ""
    return "." + name.rsplit(".", 1)[-1].lower()


def _kind_from_path(path: str) -> ReferenceKind | None:
    return _EXTENSION_KINDS.get(_extension(path))


def _kind_from_mime(mime: str) -> ReferenceKind | None:
    key = mime.strip().lower()
    for prefix, kind in _MIME_KINDS:
        if key.startswith(prefix):
            return kind
    return None


class ReferenceAnalysis(StyleSpec):
    """The style spec plus what the analyser saw in one reference.

    ``confidence`` is the analyser's own confidence in the extraction,
    ``detected_text`` is OCR or transcript text and ``reproducing_prompt`` is a
    generic prompt that describes the style without naming the original
    subject. The remaining provenance fields name the analyser, not a vendor
    that will generate the new content.
    """

    reference_id: str = ""
    kind: ReferenceKind | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    detected_text: str = Field(
        default="",
        validation_alias=AliasChoices(
            "detected_text",
            "ocr_text",
            "on_screen_text",
            "text",
        ),
    )
    language: str = Field(
        default="",
        validation_alias=AliasChoices(
            "language",
            "language_code",
            "lang",
        ),
    )
    provider: str = ""
    model: str = ""
    subject: str = Field(
        default="",
        validation_alias=AliasChoices(
            "subject",
            "original_subject",
            "main_subject",
        ),
    )
    reproducing_prompt: str = Field(
        default="",
        validation_alias=AliasChoices(
            "reproducing_prompt",
            "generic_prompt",
            "style_prompt",
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _fold_analysis_aliases(cls, data: Any) -> Any:
        """Accept the words an analyser naturally uses for these sections."""
        if not isinstance(data, Mapping):
            return data
        values = dict(data)

        composition = _first(values, "composition", "layout", "frame")
        if composition is not None and "structure" not in values:
            if isinstance(composition, Mapping):
                values["structure"] = composition
            elif isinstance(composition, str) and composition.strip():
                values.setdefault("description", composition.strip())
        for key in ("composition", "layout", "frame"):
            values.pop(key, None)

        style = _first(values, "style", "visual_style", "look", "aesthetic")
        if style is not None:
            if isinstance(style, str) and style.strip():
                values.setdefault("description", style.strip())
                values.setdefault("prompt_suffix", style.strip())
            elif isinstance(style, Mapping):
                for key in (
                    "name",
                    "description",
                    "prompt_suffix",
                    "structure",
                    "pacing",
                    "palette",
                    "typography",
                    "music",
                    "aspect",
                    "caption_style",
                ):
                    if key in style and key not in values:
                        values[key] = style[key]
            metadata = dict(values.get("metadata") or {})
            metadata.setdefault("style", style)
            values["metadata"] = metadata
        for key in ("style", "visual_style", "look", "aesthetic"):
            values.pop(key, None)

        subject = _first(values, "subject", "original_subject", "main_subject")
        if subject is not None:
            values["subject"] = subject
        for key in ("original_subject", "main_subject"):
            values.pop(key, None)

        return values

    @field_validator(
        "reference_id",
        "detected_text",
        "language",
        "provider",
        "model",
        "subject",
        "reproducing_prompt",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: Any) -> str:
        return _text(value)

    @field_validator("confidence", mode="before")
    @classmethod
    def _validate_confidence(cls, value: Any) -> Any:
        if value is None:
            return None
        return _ratio_unit(value, "confidence")

    @property
    def analysis(self) -> ReferenceAnalysis:
        """Return this analysis itself for callers holding a spec-like value."""
        return self


class ReferenceSpec(_SpecModel):
    """One user-supplied reference and the analysis derived from it.

    ``kind`` is inferred from ``mime_type`` and then the file extension when it
    is not supplied, so a CLI can accept a path without repeating its type.
    ``owned`` records the rights decision; an unowned reference may still guide
    style and structure but a later policy decides whether it can be matched
    closely. ``analysis`` is optional because a reference may be registered
    before the paid analysis runs.
    """

    schema_version: int = Field(default=CURRENT_REFERENCE_SCHEMA_VERSION, ge=1)
    kind: ReferenceKind = ReferenceKind.IMAGE
    label: str = ""
    path: str = ""
    url: str = ""
    content_hash: str = ""
    mime_type: str = Field(
        default="",
        validation_alias=AliasChoices("mime_type", "mime", "media_type"),
    )
    size_bytes: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("size_bytes", "size", "bytes"),
    )
    language: str = ""
    source: str = ""
    owned: bool = False
    analysis: ReferenceAnalysis = Field(default_factory=ReferenceAnalysis)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _infer_kind(cls, data: Any) -> Any:
        if isinstance(data, str):
            data = {"path": data}
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        kind = _first(values, "kind", "type", "reference_kind")
        if kind is not None:
            values["kind"] = kind
        path = _first(values, "path", "file", "filename")
        if path is not None:
            values["path"] = path
        url = _first(values, "url", "uri", "link")
        if url is not None:
            values["url"] = url
        if "kind" not in values:
            mime = _first(values, "mime_type", "mime", "media_type") or ""
            path_value = values.get("path") or ""
            url_value = values.get("url") or ""
            inferred = (
                _kind_from_mime(str(mime))
                or _kind_from_path(str(path_value))
                or _kind_from_path(str(url_value))
            )
            values["kind"] = inferred or ReferenceKind.IMAGE
        analysis = _first(values, "analysis", "style", "spec")
        if analysis is not None:
            if isinstance(analysis, StyleSpec) and not isinstance(
                analysis, ReferenceAnalysis
            ):
                analysis = analysis.model_dump()
            values["analysis"] = analysis
        return values

    @field_validator(
        "label",
        "path",
        "url",
        "content_hash",
        "mime_type",
        "language",
        "source",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: Any) -> str:
        return _text(value)

    @field_validator("content_hash", "mime_type", "language", mode="before")
    @classmethod
    def _lower_text(cls, value: Any) -> str:
        return _text(value).lower()

    @field_validator("metadata", mode="before")
    @classmethod
    def _coerce_metadata(cls, value: Any) -> dict[str, Any]:
        return {} if value is None else dict(value)

    @property
    def is_remote(self) -> bool:
        """Whether the reference is fetched from a URL."""
        return bool(self.url) and not self.path

    @property
    def is_local(self) -> bool:
        """Whether the reference is a local file."""
        return bool(self.path)

    @property
    def extension(self) -> str:
        """The lowercased file extension of the path or URL, including the dot."""
        return _extension(self.path or self.url)

    @property
    def style_spec(self) -> ReferenceAnalysis:
        """Alias for :attr:`analysis`."""
        return self.analysis

    @property
    def has_analysis(self) -> bool:
        """Whether analysis produced any information."""
        return not self.analysis.is_empty


Reference = ReferenceSpec
ReferenceConfig = ReferenceSpec
Style = StyleSpec
StyleSpecification = StyleSpec
StyleSpecConfig = StyleSpec
Analysis = ReferenceAnalysis
Structure = StructureSpec
StructureConfig = StructureSpec
Pacing = PacingSpec
PacingConfig = PacingSpec
Palette = PaletteSpec
PaletteConfig = PaletteSpec
Typography = TypographySpec
TypographyConfig = TypographySpec
MusicMood = MusicMoodSpec
MusicMoodConfig = MusicMoodSpec
MusicSpec = MusicMoodSpec
Music = MusicMoodSpec
Aspect = AspectSpec
AspectConfig = AspectSpec
AspectRatio = AspectSpec
Scene = SceneSpec
SceneConfig = SceneSpec
Shot = SceneSpec
ShotSpec = SceneSpec


ReferenceSchema = ReferenceSpec
ReferenceModel = ReferenceSpec
ReferenceSpecModel = ReferenceSpec
StyleSpecSchema = StyleSpec
StyleSpecModel = StyleSpec
StructureModel = StructureSpec
StructureSchema = StructureSpec
PacingModel = PacingSpec
PacingSchema = PacingSpec
PaletteModel = PaletteSpec
PaletteSchema = PaletteSpec
TypographyModel = TypographySpec
TypographySchema = TypographySpec
MusicMoodModel = MusicMoodSpec
MusicMoodSchema = MusicMoodSpec
AspectModel = AspectSpec
AspectSchema = AspectSpec
SceneModel = SceneSpec
SceneSchema = SceneSpec


def build_style_spec_schema() -> dict[str, Any]:
    """Return the JSON schema for :class:`StyleSpec`."""
    return StyleSpec.model_json_schema()


def build_reference_schema() -> dict[str, Any]:
    """Return the JSON schema for :class:`ReferenceSpec`."""
    return ReferenceSpec.model_json_schema()


style_spec_json_schema = build_style_spec_schema
reference_json_schema = build_reference_schema

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

__all__ = [
    "COMPOSITION_SCHEMA_VERSION",
    "CURRENT_COMPOSITION_SCHEMA_VERSION",
    "AnyLayer",
    "AudioTrack",
    "AudioTrackConfig",
    "AudioTrackRole",
    "AudioTrackSpec",
    "BaseLayer",
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
    "FitMode",
    "ImageLayer",
    "Layer",
    "LayerAdapter",
    "LayerConfig",
    "LayerSpec",
    "LayerType",
    "LayerUnion",
    "LogoLayer",
    "ShapeLayer",
    "ShapeType",
    "StickerLayer",
    "TextAlign",
    "TextLayer",
    "TrackRole",
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

    @model_validator(mode="before")
    @classmethod
    def _accept_string_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"type": data}
        return data

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

    @field_validator("font_family", "color", mode="before")
    @classmethod
    def _strip_text_field(cls, value: Any) -> Any:
        if not isinstance(value, str):
            raise ValueError("value must be a string")
        return value.strip()

    @field_validator("font_size", mode="before")
    @classmethod
    def _validate_font_size(cls, value: Any) -> Any:
        return _finite(value, "font_size")


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

    @model_validator(mode="before")
    @classmethod
    def _accept_string_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"type": LayerType.SHAPE, "shape": data}
        return data


class StickerLayer(_AssetLayer):
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


class LogoLayer(_AssetLayer):
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

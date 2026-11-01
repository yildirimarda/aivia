"""Deterministic still rendering with Pillow.

The renderer interface separates *what* a composition contains from *how* a
frame is produced. This module implements the first backend: a pure Pillow
still renderer for the canvas, its background colour and image layers.

Rendering is deliberately self contained. It imports no AI provider and no
HTTP client, so a render cannot make a network request or spend money. Image
sources are local paths, encoded bytes or already decoded Pillow images, and
the same composition always produces the same pixels.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from PIL import (
    Image,
    ImageColor,
    ImageDraw,
    ImageFont,
    ImageOps,
    UnidentifiedImageError,
)

from aivia.core.config.media import (
    DEFAULT_MEDIA_MAX_ANIMATED_FRAMES,
    DEFAULT_MEDIA_MAX_ANIMATED_PIXELS,
    DEFAULT_MEDIA_MAX_BYTES,
    DEFAULT_MEDIA_MAX_PIXELS,
)

from .errors import FontError, RenderError
from .fonts import FontManager
from .keyframes import AnimatedProperty, keyframes_for_layer
from .schema import (
    Composition,
    FitMode,
    Gradient,
    Layer,
    LayerType,
    ShapeType,
    TextAlign,
    TextOverflow,
    TextStyle,
    TextWeight,
    VerticalAlign,
)

__all__ = [
    "PILLOW_STILL_RENDERER_VERSION",
    "RENDERER_VERSION",
    "AssetResolver",
    "FontError",
    "ImageSource",
    "LayerPlacement",
    "PillowRenderer",
    "PillowStillRenderer",
    "RenderError",
    "Renderer",
    "StillRenderer",
    "TextLayout",
    "layout_text",
    "placement_for_layer",
    "render_still",
]

PILLOW_STILL_RENDERER_VERSION = "1"
"""The version of the Pillow still renderer, recorded with every output."""

RENDERER_VERSION = PILLOW_STILL_RENDERER_VERSION
"""Alias for the current renderer version."""

type ImageSource = str | Path | bytes | Image.Image
"""A local image path, encoded bytes or an already decoded image."""

type AssetResolver = Mapping[str, ImageSource] | Callable[[str], ImageSource]
"""How a layer source string is resolved to image data."""


@dataclass(frozen=True, slots=True)
class LayerPlacement:
    """The destination rectangle of one layer on the canvas, in pixels."""

    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> tuple[float, float]:
        """The centre of the rectangle."""
        return self.x + self.width / 2.0, self.y + self.height / 2.0


@runtime_checkable
class Renderer(Protocol):
    """The contract every composition renderer implements."""

    version: str

    def render_still(
        self,
        composition: Composition,
        time_seconds: float = 0.0,
    ) -> Image.Image:
        """Render one deterministic still frame at *time_seconds*."""


StillRenderer = Renderer
"""Alias for :class:`Renderer`."""


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RenderError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise RenderError(f"{name} must be finite")
    return number


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise RenderError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise RenderError(f"{name} must be an integer")


def _layer_value(layer: Layer, *names: str) -> Any:
    extras = layer.model_extra or {}
    for name in names:
        if name in extras:
            return extras[name]
        value = getattr(layer, name, None)
        if value is not None:
            return value
    return None


def _pair(value: Any, name: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part for part in value.replace("x", ",").split(",") if part.strip()]
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        parts = list(value)
    else:
        raise RenderError(f"{name} must be a pair of integers")
    if len(parts) != 2:
        raise RenderError(f"{name} must be a pair of integers")
    return _integer(parts[0], name), _integer(parts[1], name)


def _offset_pair(value: Any, name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return _finite(value[0], name), _finite(value[1], name)
    raise RenderError(f"{name} must be a pair of numbers")


def placement_for_layer(layer: Layer, canvas: Image.Image) -> LayerPlacement:
    """Return the destination rectangle for *layer* on *canvas*.

    Geometry is optional. Without it a layer covers the whole canvas, which is
    what a background image needs. ``x``/``y`` and ``width``/``height`` (with
    the usual aliases) may be stored on the layer envelope; a ``position`` or
    ``size`` pair is accepted too.
    """
    position = _pair(_layer_value(layer, "position", "pos", "offset"), "position")
    size = _pair(_layer_value(layer, "size", "dimensions"), "size")
    x = _layer_value(layer, "x", "left")
    y = _layer_value(layer, "y", "top")
    width = _layer_value(layer, "width", "w")
    height = _layer_value(layer, "height", "h")

    if x is None and position is not None:
        x = position[0]
    if y is None and position is not None:
        y = position[1]
    if width is None and size is not None:
        width = size[0]
    if height is None and size is not None:
        height = size[1]

    placement = LayerPlacement(
        x=0 if x is None else _integer(x, "layer x"),
        y=0 if y is None else _integer(y, "layer y"),
        width=canvas.width if width is None else _integer(width, "layer width"),
        height=canvas.height if height is None else _integer(height, "layer height"),
    )
    if placement.width < 1 or placement.height < 1:
        raise RenderError("layer width and height must be positive")
    return placement


def _check_limits(
    image: Image.Image,
    *,
    max_pixels: int,
    max_frames: int,
    max_total_pixels: int,
) -> None:
    if image.width * image.height > max_pixels:
        raise RenderError("image is above the pixel cap")
    frames = getattr(image, "n_frames", 1)
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        frames = 1
    if frames > max_frames:
        raise RenderError("image is above the frame cap")
    if image.width * image.height * frames > max_total_pixels:
        raise RenderError("image is above the total decoded pixel cap")


def _open_source(
    source: ImageSource,
    *,
    max_bytes: int,
    max_pixels: int,
    max_frames: int,
    max_total_pixels: int,
) -> Image.Image:
    if isinstance(source, Image.Image):
        image = source.copy()
        _check_limits(
            image,
            max_pixels=max_pixels,
            max_frames=max_frames,
            max_total_pixels=max_total_pixels,
        )
        return image.convert("RGBA")
    try:
        if isinstance(source, bytes):
            if len(source) > max_bytes:
                raise RenderError("image is above the byte cap")
            with Image.open(BytesIO(source)) as opened:
                _check_limits(
                    opened,
                    max_pixels=max_pixels,
                    max_frames=max_frames,
                    max_total_pixels=max_total_pixels,
                )
                opened.load()
                return opened.convert("RGBA")
        path = Path(source)
        if path.stat().st_size > max_bytes:
            raise RenderError("image is above the byte cap")
        with Image.open(path) as opened:
            _check_limits(
                opened,
                max_pixels=max_pixels,
                max_frames=max_frames,
                max_total_pixels=max_total_pixels,
            )
            opened.load()
            return opened.convert("RGBA")
    except RenderError:
        raise
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise RenderError("could not open image data") from exc


def _resize_to(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.resize(size, Image.Resampling.LANCZOS)


def _fit(image: Image.Image, size: tuple[int, int], fit: FitMode) -> Image.Image:
    if fit is FitMode.STRETCH:
        return _resize_to(image, size)
    if fit is FitMode.COVER:
        return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)
    if fit is FitMode.CONTAIN:
        return ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)
    return image


def _composite(
    canvas: Image.Image,
    image: Image.Image,
    x: int,
    y: int,
) -> None:
    left = max(0, x)
    top = max(0, y)
    right = min(canvas.width, x + image.width)
    bottom = min(canvas.height, y + image.height)
    if right <= left or bottom <= top:
        return
    crop = (left - x, top - y, right - x, bottom - y)
    canvas.alpha_composite(image.crop(crop), dest=(left, top))


def _opacity(image: Image.Image, value: float) -> Image.Image:
    if value >= 1.0:
        return image
    if value <= 0.0:
        return image.copy().convert("RGBA").point(lambda _: 0)
    alpha = image.getchannel("A").point(lambda channel: round(channel * value))
    result = image.copy()
    result.putalpha(alpha)
    return result


def _sample(layer: Layer, property_name: AnimatedProperty, time: float) -> Any:
    track = keyframes_for_layer(layer).track(property_name)
    return None if track is None else track.sample(time)


def _color(value: Any, name: str) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RenderError(f"{name} must be a colour string")
    text = value.strip()
    if not text or text.lower() in {"none", "transparent"}:
        return None
    try:
        return cast(tuple[int, int, int, int], ImageColor.getcolor(text, "RGBA"))
    except ValueError as exc:
        raise RenderError(f"unsupported {name} {value!r}") from exc


def _as_gradient(value: Any) -> Gradient:
    if isinstance(value, Gradient):
        return value
    try:
        return Gradient.model_validate(value)
    except ValueError as exc:
        raise RenderError("invalid gradient") from exc


def _stop_color(
    stops: list[tuple[float, tuple[int, int, int, int]]], position: float
) -> tuple[int, int, int, int]:
    if position <= stops[0][0]:
        return stops[0][1]
    if position >= stops[-1][0]:
        return stops[-1][1]
    for (left_offset, left_color), (right_offset, right_color) in itertools.pairwise(
        stops
    ):
        if left_offset <= position <= right_offset:
            if right_offset == left_offset:
                return right_color
            fraction = (position - left_offset) / (right_offset - left_offset)
            return (
                round(left_color[0] + (right_color[0] - left_color[0]) * fraction),
                round(left_color[1] + (right_color[1] - left_color[1]) * fraction),
                round(left_color[2] + (right_color[2] - left_color[2]) * fraction),
                round(left_color[3] + (right_color[3] - left_color[3]) * fraction),
            )
    return stops[-1][1]


def _gradient_image(
    size: tuple[int, int],
    gradient: Gradient,
) -> Image.Image:
    stops: list[tuple[float, tuple[int, int, int, int]]] = []
    for stop in gradient.stops:
        color = _color(stop.color, "gradient colour")
        stops.append((stop.offset, (0, 0, 0, 0) if color is None else color))
    if not stops:
        return Image.new("RGBA", size, (0, 0, 0, 0))

    table = [_stop_color(stops, index / 255.0) for index in range(256)]
    width, height = size
    angle = math.radians(gradient.angle)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    corners = (
        (0, 0),
        (max(0, width - 1), 0),
        (0, max(0, height - 1)),
        (max(0, width - 1), max(0, height - 1)),
    )
    projections = [x * cosine + y * sine for x, y in corners]
    lowest = min(projections)
    span = max(projections) - lowest
    data = bytearray(width * height * 4)
    index = 0
    for y in range(height):
        for x in range(width):
            position = 0.0 if span <= 1e-12 else (x * cosine + y * sine - lowest) / span
            color = table[max(0, min(255, round(position * 255.0)))]
            data[index : index + 4] = bytes(color)
            index += 4
    return Image.frombytes("RGBA", size, bytes(data))


def _shape_radius(size: tuple[int, int], radius: float) -> float:
    return max(0.0, min(radius, (size[0] - 1) / 2.0, (size[1] - 1) / 2.0))


def _shape_mask(
    shape: ShapeType,
    size: tuple[int, int],
    *,
    radius: float = 0.0,
    line_width: int = 1,
) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    box = (0, 0, size[0] - 1, size[1] - 1)
    if shape is ShapeType.LINE:
        draw.line(
            ((0, 0), (size[0] - 1, size[1] - 1)),
            fill=255,
            width=max(1, line_width),
        )
    elif shape is ShapeType.CIRCLE:
        draw.ellipse(box, fill=255)
    elif shape is ShapeType.ROUNDED_RECTANGLE:
        draw.rounded_rectangle(
            box,
            radius=_shape_radius(size, radius),
            fill=255,
        )
    else:
        draw.rectangle(box, fill=255)
    return mask


def _shape_image(
    shape: ShapeType,
    size: tuple[int, int],
    *,
    fill: Any,
    stroke: Any,
    stroke_width: float,
    radius: float,
    gradient: Gradient | None,
) -> Image.Image:
    transparent = Image.new("RGBA", size, (0, 0, 0, 0))
    width = max(1, round(stroke_width))
    paint: Image.Image | None

    if shape is ShapeType.LINE:
        mask = _shape_mask(shape, size, line_width=width)
        paint_color = stroke if stroke is not None else fill
        if gradient is not None:
            paint = _gradient_image(size, gradient)
        else:
            color = _color(paint_color, "line colour")
            if color is None:
                return transparent
            paint = Image.new("RGBA", size, color)
        return Image.composite(paint, transparent, mask)

    mask = _shape_mask(shape, size, radius=radius)
    if gradient is not None:
        paint = _gradient_image(size, gradient)
    else:
        color = _color(fill, "fill")
        paint = None if color is None else Image.new("RGBA", size, color)
    image = transparent if paint is None else Image.composite(paint, transparent, mask)

    stroke_color = _color(stroke, "stroke")
    if stroke_color is None or width <= 0:
        return image

    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    box = (0, 0, size[0] - 1, size[1] - 1)
    if shape is ShapeType.CIRCLE:
        draw.ellipse(box, outline=stroke_color, width=width)
    elif shape is ShapeType.ROUNDED_RECTANGLE:
        draw.rounded_rectangle(
            box,
            radius=_shape_radius(size, radius),
            outline=stroke_color,
            width=width,
        )
    else:
        draw.rectangle(box, outline=stroke_color, width=width)
    return Image.alpha_composite(image, overlay)


@lru_cache(maxsize=1)
def _default_font_manager() -> FontManager:
    return FontManager()


@lru_cache(maxsize=32)
def _font_path(family: str, weight: TextWeight, style: TextStyle) -> Path:
    return _default_font_manager().resolve(family or None, weight=weight, style=style)


@lru_cache(maxsize=128)
def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size=size)
    except OSError as exc:
        raise RenderError("could not load font file") from exc


def _measure_text(text: str, font: ImageFont.FreeTypeFont, spacing: float) -> float:
    if not text:
        return 0.0
    if spacing == 0.0:
        return float(font.getlength(text))
    return float(
        sum(font.getlength(character) for character in text) + spacing * (len(text) - 1)
    )


def _split_word(
    word: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    spacing: float,
) -> list[str]:
    fragments: list[str] = []
    current = ""
    for character in word:
        candidate = f"{current}{character}"
        if current and _measure_text(candidate, font, spacing) > max_width:
            fragments.append(current)
            current = character
        else:
            current = candidate
    if current:
        fragments.append(current)
    return fragments or [""]


def _wrap_paragraph(
    paragraph: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    spacing: float,
) -> list[str]:
    words = paragraph.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}" if current else word
        if _measure_text(candidate, font, spacing) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = ""
        fragments = _split_word(word, font, max_width, spacing)
        lines.extend(fragments[:-1])
        current = fragments[-1]
    if current or not lines:
        lines.append(current)
    return lines


def _wrap_text(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    spacing: float,
    wrap: bool,
) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for paragraph in normalized.split("\n"):
        if wrap:
            lines.extend(_wrap_paragraph(paragraph, font, max_width, spacing))
        else:
            lines.append(paragraph)
    return lines


def _ellipsize(
    line: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    spacing: float,
    ellipsis: str,
    *,
    force: bool = False,
) -> str:
    if line.endswith(ellipsis) and _measure_text(line, font, spacing) <= max_width:
        return line
    if not force and _measure_text(line, font, spacing) <= max_width:
        return line
    target = max_width - _measure_text(ellipsis, font, spacing)
    prefix = ""
    for character in line:
        candidate = f"{prefix}{character}"
        if _measure_text(candidate, font, spacing) > target:
            break
        prefix = candidate
    return f"{prefix.rstrip()}{ellipsis}"


@dataclass(frozen=True, slots=True)
class TextLayout:
    """The lines of a text block and the box they occupy."""

    lines: tuple[str, ...]
    font_size: int
    line_height: int
    width: float
    height: float
    truncated: bool


def _fits(
    lines: list[str],
    line_height: int,
    max_lines: int | None,
    max_height: int,
) -> bool:
    if max_lines is not None and len(lines) > max_lines:
        return False
    return len(lines) * line_height <= max_height


def layout_text(
    text: str,
    *,
    max_width: int,
    max_height: int,
    font_family: str = "",
    font_size: float = 48.0,
    font_weight: TextWeight = TextWeight.REGULAR,
    font_style: TextStyle = TextStyle.NORMAL,
    line_height: float = 1.2,
    letter_spacing: float = 0.0,
    wrap: bool = True,
    max_lines: int | None = None,
    shrink_to_fit: bool = False,
    min_font_size: float = 8.0,
    overflow: TextOverflow = TextOverflow.ELLIPSIS,
    ellipsis: str = "\u2026",
    fonts: FontManager | None = None,
) -> TextLayout:
    """Lay *text* out inside a pixel box, wrapping and optionally shrinking it."""
    if not isinstance(text, str) or not text.strip():
        raise RenderError("text must not be empty")
    if max_width < 1 or max_height < 1:
        raise RenderError("text box must have a positive size")
    base_size = round(_finite(font_size, "font_size"))
    minimum = round(_finite(min_font_size, "min_font_size"))
    if base_size < 1 or minimum < 1:
        raise RenderError("font size must be positive")
    factor = _finite(line_height, "line_height")
    if factor <= 0.0:
        raise RenderError("line_height must be positive")
    spacing = _finite(letter_spacing, "letter_spacing")
    if max_lines is not None:
        max_lines = _integer(max_lines, "max_lines")
        if max_lines < 1:
            raise RenderError("max_lines must be positive")
    if shrink_to_fit and minimum > base_size:
        raise RenderError("min_font_size must not exceed font_size")
    if not isinstance(ellipsis, str) or not ellipsis.strip():
        raise RenderError("ellipsis must not be empty")

    manager = _default_font_manager() if fonts is None else fonts
    path = manager.resolve(
        font_family or None,
        weight=font_weight,
        style=font_style,
    )
    size = base_size
    font = _load_font(str(path), size)
    line_height_px = max(1, round(size * factor))
    lines = _wrap_text(text, font, max_width, spacing, wrap)

    if shrink_to_fit:
        while size > minimum and not _fits(
            lines, line_height_px, max_lines, max_height
        ):
            size -= 1
            font = _load_font(str(path), size)
            line_height_px = max(1, round(size * factor))
            lines = _wrap_text(text, font, max_width, spacing, wrap)

    limit = len(lines)
    if max_lines is not None:
        limit = min(limit, max_lines)
    limit = min(limit, max(1, max_height // line_height_px))
    limit = max(1, limit)
    truncated = limit < len(lines)
    visible = lines[:limit]

    if overflow is TextOverflow.ELLIPSIS:
        adjusted: list[str] = []
        for index, line in enumerate(visible):
            too_wide = _measure_text(line, font, spacing) > max_width
            if too_wide:
                truncated = True
            if too_wide or (truncated and index == len(visible) - 1):
                adjusted.append(
                    _ellipsize(
                        line,
                        font,
                        max_width,
                        spacing,
                        ellipsis.strip(),
                        force=truncated and index == len(visible) - 1,
                    )
                )
            else:
                adjusted.append(line)
        visible = adjusted
    else:
        for line in visible:
            if _measure_text(line, font, spacing) > max_width:
                truncated = True

    width = max(
        (_measure_text(line, font, spacing) for line in visible),
        default=0.0,
    )
    return TextLayout(
        lines=tuple(visible),
        font_size=size,
        line_height=line_height_px,
        width=width,
        height=len(visible) * line_height_px,
        truncated=truncated,
    )


def _enum_value(
    enum_type: type[Any],
    value: Any,
    name: str,
    default: Any,
) -> Any:
    if value is None:
        return default
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise RenderError(f"unsupported {name} {value!r}") from exc


def _text_flag(layer: Layer, default: bool, *names: str) -> bool:
    value = _layer_value(layer, *names)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise RenderError(f"{names[0]} must be a boolean")
    return value


def _draw_text_line(
    draw: ImageDraw.ImageDraw,
    position: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int, int],
    spacing: float,
    stroke: tuple[int, int, int, int] | None,
    stroke_width: float,
) -> None:
    x, y = position
    options: dict[str, Any] = {"font": font, "fill": fill, "anchor": "la"}
    if stroke is not None and stroke_width > 0.0:
        options["stroke_width"] = max(1, round(stroke_width))
        options["stroke_fill"] = stroke
    if spacing == 0.0:
        draw.text((x, y), text, **options)
        return
    for character in text:
        draw.text((x, y), character, **options)
        x += font.getlength(character) + spacing


class PillowStillRenderer:
    """Render one still frame of a composition with Pillow.

    ``assets`` maps layer source strings to local paths, encoded bytes or
    decoded images. Without a resolver, a source is treated as a local path,
    optionally relative to ``source_root``. Image, shape and text layers are
    drawn; other kinds are skipped until their own renderer support lands, so
    adding a layer kind cannot break an existing still render.
    """

    version = PILLOW_STILL_RENDERER_VERSION
    supported_layer_types = frozenset(
        {LayerType.IMAGE, LayerType.SHAPE, LayerType.TEXT}
    )

    def __init__(
        self,
        assets: AssetResolver | None = None,
        *,
        source_root: str | Path | None = None,
        fonts: FontManager | None = None,
        max_bytes: int = DEFAULT_MEDIA_MAX_BYTES,
        max_pixels: int = DEFAULT_MEDIA_MAX_PIXELS,
        max_frames: int = DEFAULT_MEDIA_MAX_ANIMATED_FRAMES,
        max_total_pixels: int = DEFAULT_MEDIA_MAX_ANIMATED_PIXELS,
    ) -> None:
        self._assets = assets
        self._source_root = None if source_root is None else Path(source_root)
        self._fonts = fonts
        self._max_bytes = max_bytes
        self._max_pixels = max_pixels
        self._max_frames = max_frames
        self._max_total_pixels = max_total_pixels

    def _resolve(self, source: str) -> ImageSource:
        if callable(self._assets):
            return self._assets(source)
        if self._assets is not None and source in self._assets:
            return self._assets[source]
        path = Path(source)
        if self._source_root is not None and not path.is_absolute():
            path = self._source_root / path
        if self._assets is not None and not path.exists():
            raise RenderError(f"no asset registered for {source!r}")
        return path

    def _open(self, source: str) -> Image.Image:
        return _open_source(
            self._resolve(source),
            max_bytes=self._max_bytes,
            max_pixels=self._max_pixels,
            max_frames=self._max_frames,
            max_total_pixels=self._max_total_pixels,
        )

    def _draw_image_layer(
        self,
        canvas: Image.Image,
        layer: Layer,
        time_seconds: float,
    ) -> None:
        source = _layer_value(layer, "source", "path", "file", "asset")
        if not isinstance(source, str) or not source.strip():
            raise RenderError(f"image layer {layer.id!r} has no source")
        placement = placement_for_layer(layer, canvas)
        image = self._open(source.strip())
        fit = _layer_value(layer, "fit", "fit_mode", "mode") or FitMode.COVER
        try:
            mode = FitMode(fit)
        except ValueError as exc:
            raise RenderError(f"unsupported fit mode {fit!r}") from exc
        fitted = _fit(image, (placement.width, placement.height), mode)

        x = placement.x
        y = placement.y
        if mode in (FitMode.CONTAIN, FitMode.NONE):
            x += (placement.width - fitted.width) // 2
            y += (placement.height - fitted.height) // 2
        position = _sample(layer, AnimatedProperty.POSITION, time_seconds)
        if position is not None:
            offset = _offset_pair(position, "position keyframe")
            if offset is not None:
                x += round(offset[0])
                y += round(offset[1])

        scale = _sample(layer, AnimatedProperty.SCALE, time_seconds)
        if scale is not None:
            if isinstance(scale, tuple):
                scale_x, scale_y = scale
            else:
                scale_x = scale_y = float(scale)
            if scale_x <= 0.0 or scale_y <= 0.0:
                return
            fitted = _resize_to(
                fitted,
                (
                    max(1, round(fitted.width * scale_x)),
                    max(1, round(fitted.height * scale_y)),
                ),
            )
            x += (placement.width - fitted.width) // 2
            y += (placement.height - fitted.height) // 2

        rotation = _sample(layer, AnimatedProperty.ROTATION, time_seconds)
        if rotation is not None:
            angle = float(rotation)
            rotated = fitted.rotate(
                angle,
                resample=Image.Resampling.BICUBIC,
                expand=True,
            )
            x += (fitted.width - rotated.width) // 2
            y += (fitted.height - rotated.height) // 2
            fitted = rotated

        opacity = _sample(layer, AnimatedProperty.OPACITY, time_seconds)
        value = layer.opacity if opacity is None else float(opacity)
        if value <= 0.0:
            return
        _composite(canvas, _opacity(fitted, min(1.0, value)), x, y)

    def _draw_shape_layer(
        self,
        canvas: Image.Image,
        layer: Layer,
        time_seconds: float,
    ) -> None:
        raw_shape = _layer_value(layer, "shape", "shape_type", "form")
        try:
            shape = ShapeType(raw_shape)
        except ValueError as exc:
            raise RenderError(f"unsupported shape {raw_shape!r}") from exc

        placement = placement_for_layer(layer, canvas)
        fill = _layer_value(layer, "fill", "fill_color", "color", "background")
        stroke = _layer_value(
            layer,
            "stroke",
            "stroke_color",
            "border",
            "border_color",
            "outline",
        )
        stroke_width = _layer_value(
            layer,
            "stroke_width",
            "border_width",
            "outline_width",
            "line_width",
        )
        stroke_width = (
            1.0 if stroke_width is None else _finite(stroke_width, "stroke_width")
        )
        if stroke_width < 0.0:
            raise RenderError("stroke_width must not be negative")
        radius = _layer_value(layer, "radius", "corner_radius", "border_radius")
        radius = 0.0 if radius is None else _finite(radius, "radius")
        if radius < 0.0:
            raise RenderError("radius must not be negative")

        gradient_raw = _layer_value(layer, "gradient")
        gradient = None if gradient_raw is None else _as_gradient(gradient_raw)
        if gradient is None and isinstance(fill, (Gradient, Mapping)):
            gradient = _as_gradient(fill)
            fill = None

        image = _shape_image(
            shape,
            (placement.width, placement.height),
            fill=fill,
            stroke=stroke,
            stroke_width=stroke_width,
            radius=radius,
            gradient=gradient,
        )

        x = placement.x
        y = placement.y
        position = _sample(layer, AnimatedProperty.POSITION, time_seconds)
        if position is not None:
            offset = _offset_pair(position, "position keyframe")
            if offset is not None:
                x += round(offset[0])
                y += round(offset[1])

        scale = _sample(layer, AnimatedProperty.SCALE, time_seconds)
        if scale is not None:
            if isinstance(scale, tuple):
                scale_x, scale_y = scale
            else:
                scale_x = scale_y = float(scale)
            if scale_x <= 0.0 or scale_y <= 0.0:
                return
            image = _resize_to(
                image,
                (
                    max(1, round(image.width * scale_x)),
                    max(1, round(image.height * scale_y)),
                ),
            )
            x += (placement.width - image.width) // 2
            y += (placement.height - image.height) // 2

        rotation = _sample(layer, AnimatedProperty.ROTATION, time_seconds)
        if rotation is not None:
            angle = float(rotation)
            rotated = image.rotate(
                angle,
                resample=Image.Resampling.BICUBIC,
                expand=True,
            )
            x += (image.width - rotated.width) // 2
            y += (image.height - rotated.height) // 2
            image = rotated

        opacity = _sample(layer, AnimatedProperty.OPACITY, time_seconds)
        value = layer.opacity if opacity is None else float(opacity)
        if value <= 0.0:
            return
        _composite(canvas, _opacity(image, min(1.0, value)), x, y)

    def _draw_text_layer(
        self,
        canvas: Image.Image,
        layer: Layer,
        time_seconds: float,
    ) -> None:
        raw_text = _layer_value(layer, "text", "content", "value")
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise RenderError(f"text layer {layer.id!r} has no text")

        raw_family = _layer_value(layer, "font_family", "font", "family")
        if raw_family is None:
            family = ""
        elif isinstance(raw_family, str):
            family = raw_family.strip()
        else:
            raise RenderError("font_family must be a string")

        raw_size = _layer_value(layer, "font_size", "size", "font_size_px")
        font_size = 48.0 if raw_size is None else _finite(raw_size, "font_size")
        if font_size <= 0.0:
            raise RenderError("font_size must be positive")

        raw_min_size = _layer_value(
            layer, "min_font_size", "min_size", "minimum_font_size"
        )
        min_font_size = (
            8.0 if raw_min_size is None else _finite(raw_min_size, "min_font_size")
        )
        if min_font_size <= 0.0:
            raise RenderError("min_font_size must be positive")

        raw_line_height = _layer_value(
            layer, "line_height", "line_spacing", "line-height"
        )
        line_height = (
            1.2 if raw_line_height is None else _finite(raw_line_height, "line_height")
        )
        if line_height <= 0.0:
            raise RenderError("line_height must be positive")

        raw_spacing = _layer_value(
            layer, "letter_spacing", "tracking", "letter-spacing"
        )
        letter_spacing = (
            0.0 if raw_spacing is None else _finite(raw_spacing, "letter_spacing")
        )

        raw_max_lines = _layer_value(layer, "max_lines", "lines", "max-lines")
        if raw_max_lines is None:
            max_lines = None
        else:
            max_lines = _integer(raw_max_lines, "max_lines")
            if max_lines < 1:
                raise RenderError("max_lines must be positive")

        raw_ellipsis = _layer_value(layer, "ellipsis", "truncation")
        if raw_ellipsis is None:
            ellipsis = "\u2026"
        elif isinstance(raw_ellipsis, str) and raw_ellipsis.strip():
            ellipsis = raw_ellipsis.strip()
        else:
            raise RenderError("ellipsis must not be empty")

        raw_stroke = _layer_value(layer, "stroke", "stroke_color", "outline")
        stroke = _color(raw_stroke, "text stroke")
        raw_stroke_width = _layer_value(layer, "stroke_width", "outline_width")
        stroke_width = (
            0.0
            if raw_stroke_width is None
            else _finite(raw_stroke_width, "stroke_width")
        )
        if stroke_width < 0.0:
            raise RenderError("stroke_width must not be negative")
        if stroke is None:
            stroke_width = 0.0

        raw_color = _layer_value(layer, "color", "fill", "text_color")
        color = _color("#ffffff" if raw_color is None else raw_color, "text colour")
        if color is None:
            return

        align = _enum_value(
            TextAlign,
            _layer_value(layer, "align", "text_align", "alignment"),
            "text alignment",
            TextAlign.LEFT,
        )
        vertical_align = _enum_value(
            VerticalAlign,
            _layer_value(layer, "vertical_align", "valign", "vertical_alignment"),
            "vertical alignment",
            VerticalAlign.TOP,
        )
        weight = _enum_value(
            TextWeight,
            _layer_value(layer, "font_weight", "weight", "font-weight"),
            "font weight",
            TextWeight.REGULAR,
        )
        style = _enum_value(
            TextStyle,
            _layer_value(layer, "font_style", "style", "font-style"),
            "font style",
            TextStyle.NORMAL,
        )
        overflow = _enum_value(
            TextOverflow,
            _layer_value(layer, "overflow", "text_overflow"),
            "text overflow",
            TextOverflow.ELLIPSIS,
        )
        wrap = _text_flag(layer, True, "wrap", "wrap_text", "word_wrap")
        shrink = _text_flag(
            layer,
            False,
            "shrink_to_fit",
            "shrink",
            "fit",
            "auto_shrink",
        )

        placement = placement_for_layer(layer, canvas)
        manager = _default_font_manager() if self._fonts is None else self._fonts
        path = manager.resolve(family or None, weight=weight, style=style)
        layout = layout_text(
            raw_text,
            max_width=placement.width,
            max_height=placement.height,
            font_family=family,
            font_size=font_size,
            font_weight=weight,
            font_style=style,
            line_height=line_height,
            letter_spacing=letter_spacing,
            wrap=wrap,
            max_lines=max_lines,
            shrink_to_fit=shrink,
            min_font_size=min_font_size,
            overflow=overflow,
            ellipsis=ellipsis,
            fonts=self._fonts,
        )
        font = _load_font(str(path), layout.font_size)
        text_image = Image.new(
            "RGBA", (placement.width, placement.height), (0, 0, 0, 0)
        )
        draw = ImageDraw.Draw(text_image)
        total_height = len(layout.lines) * layout.line_height
        if vertical_align is VerticalAlign.MIDDLE:
            top = (placement.height - total_height) / 2.0
        elif vertical_align is VerticalAlign.BOTTOM:
            top = placement.height - total_height
        else:
            top = 0.0
        for index, line in enumerate(layout.lines):
            line_width = _measure_text(line, font, letter_spacing)
            if align is TextAlign.CENTER:
                left = (placement.width - line_width) / 2.0
            elif align is TextAlign.RIGHT:
                left = placement.width - line_width
            else:
                left = 0.0
            _draw_text_line(
                draw,
                (left, top + index * layout.line_height),
                line,
                font,
                color,
                letter_spacing,
                stroke,
                stroke_width,
            )

        x = placement.x
        y = placement.y
        position = _sample(layer, AnimatedProperty.POSITION, time_seconds)
        if position is not None:
            offset = _offset_pair(position, "position keyframe")
            if offset is not None:
                x += round(offset[0])
                y += round(offset[1])

        scale = _sample(layer, AnimatedProperty.SCALE, time_seconds)
        if scale is not None:
            if isinstance(scale, tuple):
                scale_x, scale_y = scale
            else:
                scale_x = scale_y = float(scale)
            if scale_x <= 0.0 or scale_y <= 0.0:
                return
            text_image = _resize_to(
                text_image,
                (
                    max(1, round(text_image.width * scale_x)),
                    max(1, round(text_image.height * scale_y)),
                ),
            )
            x += (placement.width - text_image.width) // 2
            y += (placement.height - text_image.height) // 2

        rotation = _sample(layer, AnimatedProperty.ROTATION, time_seconds)
        if rotation is not None:
            angle = float(rotation)
            rotated = text_image.rotate(
                angle,
                resample=Image.Resampling.BICUBIC,
                expand=True,
            )
            x += (text_image.width - rotated.width) // 2
            y += (text_image.height - rotated.height) // 2
            text_image = rotated

        opacity = _sample(layer, AnimatedProperty.OPACITY, time_seconds)
        value = layer.opacity if opacity is None else float(opacity)
        if value <= 0.0:
            return
        _composite(canvas, _opacity(text_image, min(1.0, value)), x, y)

    def render_still(
        self,
        composition: Composition,
        time_seconds: float = 0.0,
    ) -> Image.Image:
        """Render one still frame of *composition* at *time_seconds*."""
        time = _finite(time_seconds, "time_seconds")
        if time < 0.0 or time > composition.duration_seconds:
            raise RenderError(
                "time_seconds must be between 0 and the composition duration"
            )
        try:
            background = ImageColor.getcolor(composition.canvas.background, "RGBA")
        except ValueError as exc:
            raise RenderError(
                f"unsupported canvas background {composition.canvas.background!r}"
            ) from exc
        canvas = Image.new("RGBA", composition.canvas.size, background)

        active = [
            layer
            for layer in composition.layers
            if layer.visible
            and layer.start_seconds <= time
            and (layer.end is None or time < layer.end)
            and layer.type in self.supported_layer_types
        ]
        for layer in sorted(active, key=lambda item: item.z_index):
            if layer.type is LayerType.IMAGE:
                self._draw_image_layer(canvas, layer, time)
            elif layer.type is LayerType.SHAPE:
                self._draw_shape_layer(canvas, layer, time)
            elif layer.type is LayerType.TEXT:
                self._draw_text_layer(canvas, layer, time)
        return canvas

    def render(
        self,
        composition: Composition,
        time_seconds: float = 0.0,
    ) -> Image.Image:
        """Alias for :meth:`render_still`."""
        return self.render_still(composition, time_seconds)


PillowRenderer = PillowStillRenderer
"""Alias for :class:`PillowStillRenderer`."""


def render_still(
    composition: Composition,
    time_seconds: float = 0.0,
    *,
    assets: AssetResolver | None = None,
    source_root: str | Path | None = None,
) -> Image.Image:
    """Render one still frame with the default Pillow renderer."""
    return PillowStillRenderer(assets, source_root=source_root).render_still(
        composition, time_seconds
    )

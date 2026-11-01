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
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from PIL import Image, ImageColor, ImageDraw, ImageOps, UnidentifiedImageError

from aivia.core.config.media import (
    DEFAULT_MEDIA_MAX_ANIMATED_FRAMES,
    DEFAULT_MEDIA_MAX_ANIMATED_PIXELS,
    DEFAULT_MEDIA_MAX_BYTES,
    DEFAULT_MEDIA_MAX_PIXELS,
)
from aivia.core.errors import ValidationError

from .keyframes import AnimatedProperty, keyframes_for_layer
from .schema import Composition, FitMode, Gradient, Layer, LayerType, ShapeType

__all__ = [
    "PILLOW_STILL_RENDERER_VERSION",
    "RENDERER_VERSION",
    "AssetResolver",
    "ImageSource",
    "LayerPlacement",
    "PillowRenderer",
    "PillowStillRenderer",
    "RenderError",
    "Renderer",
    "StillRenderer",
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


class RenderError(ValidationError):
    """A composition could not be rendered deterministically."""

    code = "render"


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


class PillowStillRenderer:
    """Render one still frame of a composition with Pillow.

    ``assets`` maps layer source strings to local paths, encoded bytes or
    decoded images. Without a resolver, a source is treated as a local path,
    optionally relative to ``source_root``. Image and shape layers are drawn;
    other kinds are skipped until their own renderer support lands, so adding
    text later cannot break an existing still render.
    """

    version = PILLOW_STILL_RENDERER_VERSION
    supported_layer_types = frozenset({LayerType.IMAGE, LayerType.SHAPE})

    def __init__(
        self,
        assets: AssetResolver | None = None,
        *,
        source_root: str | Path | None = None,
        max_bytes: int = DEFAULT_MEDIA_MAX_BYTES,
        max_pixels: int = DEFAULT_MEDIA_MAX_PIXELS,
        max_frames: int = DEFAULT_MEDIA_MAX_ANIMATED_FRAMES,
        max_total_pixels: int = DEFAULT_MEDIA_MAX_ANIMATED_PIXELS,
    ) -> None:
        self._assets = assets
        self._source_root = None if source_root is None else Path(source_root)
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

"""Animate a still image with slow zoom, overlays and a name entrance.

The animation is pure Pillow: a background still is fitted to the canvas and
zoomed slowly (a Ken Burns motion), procedural overlays are composited over it
and the recipient's name fades or slides in through an entrance animation. No
model, network or external tool is involved, so the same inputs always produce
the same pixels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any, overload

from PIL import Image

from aivia.compose.errors import RenderError
from aivia.compose.keyframes import Easing
from aivia.compose.motion import ken_burns
from aivia.compose.render import render_still
from aivia.compose.schema import (
    Composition,
    Effect,
    ImageLayer,
    Layer,
    LayerType,
    TextAlign,
    TextLayer,
    VerticalAlign,
)
from aivia.compose.text_animation import TextAnimation, entrance_animation

__all__ = ["animate_still"]

DEFAULT_CANVAS: tuple[int, int] = (1080, 1920)
DEFAULT_DURATION_SECONDS: float = 5.0
DEFAULT_ZOOM: tuple[float, float] = (1.0, 1.15)
DEFAULT_NAME_FONT_SIZE: float = 64.0


def _open_background(source: str | Path | bytes | Image.Image) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.convert("RGBA")
    try:
        if isinstance(source, (bytes, bytearray, memoryview)):
            with Image.open(BytesIO(bytes(source))) as opened:
                return opened.convert("RGBA")
        with Image.open(Path(source)) as opened:
            return opened.convert("RGBA")
    except (OSError, ValueError) as exc:
        raise RenderError("cannot open background image") from exc


def _fit_to_size(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    width, height = size
    if image.size == (width, height):
        return image
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _canvas_size(value: tuple[int, int] | str) -> tuple[int, int]:
    if isinstance(value, str):
        parts = [part for part in value.replace("x", ",").split(",") if part.strip()]
        if len(parts) != 2:
            raise RenderError("canvas must be a (width, height) pair or WxH string")
        return int(parts[0]), int(parts[1])
    return int(value[0]), int(value[1])


def _overlay_effects(
    overlays: Sequence[Mapping[str, Any] | str] | None,
) -> list[Effect]:
    effects: list[Effect] = []
    if not overlays:
        return effects
    for index, spec in enumerate(overlays):
        if isinstance(spec, str):
            type_value = spec
            params: dict[str, Any] = {}
        elif isinstance(spec, Mapping):
            data = dict(spec)
            type_value = data.pop("type", data.pop("kind", f"overlay-{index}"))
            params = data
        else:
            raise RenderError(f"overlay {index!r} must be a name or a mapping")
        effects.append(
            Effect(id=f"overlay-{index}", type=str(type_value), params=params)
        )
    return effects


def _name_entrance(value: Mapping[str, Any] | str | None) -> TextAnimation | None:
    if value is None:
        return None
    if isinstance(value, str):
        return entrance_animation(type=value)
    if isinstance(value, Mapping):
        return TextAnimation.model_validate(value)
    return entrance_animation(type=str(value))


@overload
def animate_still(
    background: str | Path | bytes | Image.Image,
    name: str,
    *,
    destination: None = None,
    canvas: tuple[int, int] | str = ...,
    duration_seconds: float = ...,
    time_seconds: float = ...,
    zoom: tuple[float, float] = ...,
    easing: Easing | str = ...,
    overlays: Sequence[Mapping[str, Any] | str] | None = ...,
    name_entrance: Mapping[str, Any] | str | None = ...,
    name_font_family: str = ...,
    name_font_size: float = ...,
    name_color: str | None = ...,
    name_align: TextAlign = ...,
    name_vertical_align: VerticalAlign = ...,
    image_format: str | None = ...,
    quality: int = ...,
) -> Image.Image: ...


@overload
def animate_still(
    background: str | Path | bytes | Image.Image,
    name: str,
    *,
    destination: str | Path,
    canvas: tuple[int, int] | str = ...,
    duration_seconds: float = ...,
    time_seconds: float = ...,
    zoom: tuple[float, float] = ...,
    easing: Easing | str = ...,
    overlays: Sequence[Mapping[str, Any] | str] | None = ...,
    name_entrance: Mapping[str, Any] | str | None = ...,
    name_font_family: str = ...,
    name_font_size: float = ...,
    name_color: str | None = ...,
    name_align: TextAlign = ...,
    name_vertical_align: VerticalAlign = ...,
    image_format: str | None = ...,
    quality: int = ...,
) -> Path: ...


def animate_still(
    background: str | Path | bytes | Image.Image,
    name: str,
    *,
    destination: str | Path | None = None,
    canvas: tuple[int, int] | str = DEFAULT_CANVAS,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    time_seconds: float = 0.0,
    zoom: tuple[float, float] = DEFAULT_ZOOM,
    easing: Easing | str = Easing.EASE_IN_OUT,
    overlays: Sequence[Mapping[str, Any] | str] | None = None,
    name_entrance: Mapping[str, Any] | str | None = "fade",
    name_font_family: str = "",
    name_font_size: float = DEFAULT_NAME_FONT_SIZE,
    name_color: str | None = None,
    name_align: TextAlign = TextAlign.CENTER,
    name_vertical_align: VerticalAlign = VerticalAlign.MIDDLE,
    image_format: str | None = None,
    quality: int = 90,
) -> Path | Image.Image:
    """Render one animated frame of *background* with *name*.

    The background is fitted to *canvas* and zooms from ``zoom[0]`` to
    ``zoom[1]`` over *duration_seconds* (a slow Ken Burns zoom). *overlays* are
    procedural overlay effects composited after the layers. *name* is drawn
    with *name_entrance* so it fades or slides in over the first part of the
    animation.

    When *destination* is given the result is written there and the path is
    returned; otherwise the composited :class:`PIL.Image.Image` is returned.
    """
    canvas_width, canvas_height = _canvas_size(canvas)
    duration = float(duration_seconds)
    if duration <= 0.0:
        raise RenderError("duration_seconds must be positive")
    time_value = float(time_seconds)
    if time_value < 0.0 or time_value > duration:
        raise RenderError("time_seconds must be between 0 and duration_seconds")

    fitted = _fit_to_size(_open_background(background), (canvas_width, canvas_height))

    layers: list[Layer] = [
        ImageLayer(
            id="background",
            type=LayerType.IMAGE,
            source="background",
            motion=ken_burns(
                start_scale=float(zoom[0]),
                end_scale=float(zoom[1]),
                easing=easing,
            ),
        )
    ]
    if isinstance(name, str) and name.strip():
        layers.append(
            TextLayer(
                id="name",
                type=LayerType.TEXT,
                text=name,
                entrance=_name_entrance(name_entrance),
                font_family=name_font_family,
                font_size=name_font_size,
                color=name_color if name_color is not None else "#ffffff",
                align=name_align,
                vertical_align=name_vertical_align,
            )
        )

    composition = Composition.model_validate(
        {
            "canvas": {
                "width": canvas_width,
                "height": canvas_height,
                "background": "#000000",
            },
            "duration_seconds": duration,
            "layers": layers,
            "effects": _overlay_effects(overlays),
        }
    )

    image = render_still(composition, time_value, assets={"background": fitted})

    if destination is not None:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        fmt = image_format
        if fmt is None:
            fmt = target.suffix.lstrip(".") if target.suffix else "png"
        options: dict[str, object] = {}
        if fmt.lower() in {"jpeg", "jpg", "webp"}:
            options["quality"] = quality
        canvas_image = image.convert("RGB") if fmt.lower() in {"jpeg", "jpg"} else image
        canvas_image.info.clear()
        canvas_image.getexif().clear()
        canvas_image.save(target, format=fmt.upper(), **options)
        return target
    return image

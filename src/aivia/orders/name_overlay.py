"""Place a recipient name on a provided background image.

The name is drawn over a calm, low-detail region of the background, steered
away from any known face rectangles. When the background under the text is low
contrast, an outline, shadow or backing box is added so the name stays
readable. All work is pure Pillow: no model, network or external tool is
involved, so the same inputs always produce the same pixels.
"""

from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any, cast

from PIL import Image, ImageDraw, ImageFont

from aivia.compose.contrast import (
    ContrastResult,
    ContrastStrategy,
    add_backing_box,
    add_shadow,
    ensure_contrast,
    parse_color,
)
from aivia.compose.errors import RenderError
from aivia.compose.placement import find_text_placement

__all__ = [
    "DEFAULT_FONT_FAMILY",
    "DEFAULT_FONT_SIZE",
    "overlay_name",
]

DEFAULT_FONT_FAMILY = "Lato"
DEFAULT_FONT_SIZE = 48.0

_MIN_FONT_SIZE = 8.0
_LINE_HEIGHT = 1.2


def _open_background(
    source: str | Path | bytes | Image.Image,
) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.convert("RGBA")
    try:
        if isinstance(source, bytes):
            with Image.open(BytesIO(source)) as opened:
                return opened.convert("RGBA")
        with Image.open(Path(source)) as opened:
            return opened.convert("RGBA")
    except (OSError, ValueError) as exc:
        raise RenderError("cannot open background image") from exc


def _fit_to_size(
    image: Image.Image,
    size: tuple[int, int],
) -> Image.Image:
    width, height = size
    if image.size == (width, height):
        return image
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _font_path(font_family: str) -> str:
    from aivia.compose.fonts import FontManager

    return str(FontManager().resolve(font_family or DEFAULT_FONT_FAMILY))


def _load_font(font_family: str, font_size: float) -> ImageFont.FreeTypeFont:
    size = max(1, round(font_size))
    try:
        return ImageFont.truetype(_font_path(font_family), size=size)
    except OSError as exc:
        raise RenderError("could not load font file") from exc


def _measure_text(
    text: str,
    font: ImageFont.FreeTypeFont,
) -> tuple[int, int]:
    width = max(1, round(font.getlength(text)))
    height = max(1, round(font.size * _LINE_HEIGHT))
    return width, height


def _shrink_to_fit(
    text: str,
    font_family: str,
    font_size: float,
    max_width: int,
) -> ImageFont.FreeTypeFont:
    size = max(1, round(font_size))
    font = _load_font(font_family, size)
    while size > _MIN_FONT_SIZE and font.getlength(text) > max_width:
        size -= 1
        font = _load_font(font_family, float(size))
    return font


def _render_text(
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int, int],
    stroke: tuple[int, int, int, int] | None = None,
    stroke_width: float = 0.0,
) -> Image.Image:
    width = max(1, round(font.getlength(text)))
    height = max(1, round(font.size * _LINE_HEIGHT))
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.text(
        (0.0, 0.0),
        text,
        font=font,
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=stroke,
    )
    return image


def _sample_background(
    image: Image.Image,
    box: tuple[int, int, int, int] | None,
) -> str:
    if box is None:
        box = (0, 0, image.width, image.height)
    left, top, right, bottom = box
    left = max(0, min(image.width, left))
    top = max(0, min(image.height, top))
    right = max(left, min(image.width, right))
    bottom = max(top, min(image.height, bottom))
    if right == left or bottom == top:
        return "#000000"
    region = image.crop((left, top, right, bottom)).convert("RGBA")
    flattened = Image.alpha_composite(
        Image.new("RGBA", region.size, (255, 255, 255, 255)),
        region,
    ).convert("RGB")
    pixel = flattened.resize((1, 1), Image.Resampling.BOX).getpixel((0, 0))
    red, green, blue = cast(tuple[int, int, int], pixel)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _shadow_offset_pair(
    offset: tuple[float, float] | float,
) -> tuple[float, float]:
    if isinstance(offset, (tuple, list)) and len(offset) == 2:
        return float(offset[0]), float(offset[1])
    return float(offset), float(offset)


def _apply_treatment(
    text_image: Image.Image,
    result: ContrastResult,
) -> tuple[Image.Image, int, int]:
    """Apply the contrast result treatment and return its pixel offset."""
    if not result.applied:
        return text_image, 0, 0
    if result.strategy is ContrastStrategy.SHADOW and result.shadow_color:
        treated = add_shadow(
            text_image,
            result.shadow_color,
            result.shadow_offset,
            result.shadow_blur,
        )
        offset_x, offset_y = _shadow_offset_pair(result.shadow_offset)
        margin = max(1, int(result.shadow_blur * 3)) + max(
            abs(round(offset_x)),
            abs(round(offset_y)),
        )
        return treated, -margin, -margin
    if result.strategy is ContrastStrategy.BOX and result.box_color:
        treated = add_backing_box(
            text_image,
            result.box_color,
            result.box_padding,
        )
        padding = round(result.box_padding)
        return treated, -padding, -padding
    return text_image, 0, 0


def overlay_name(
    background: str | Path | bytes | Image.Image,
    name: str,
    *,
    destination: str | Path | None = None,
    size: tuple[int, int] | None = None,
    font_family: str = "",
    font_size: float = DEFAULT_FONT_SIZE,
    color: str | None = None,
    avoid_faces: bool = True,
    faces: Sequence[Any] | None = None,
    face_padding: float = 0.0,
    margin: float | tuple[float, float] = 0.0,
    min_contrast: float | None = None,
    contrast_strategy: ContrastStrategy | str | None = None,
    outline_color: str | None = None,
    outline_width: float = 2.0,
    shadow_color: str | None = None,
    shadow_offset: tuple[float, float] | float = (2.0, 2.0),
    shadow_blur: float = 2.0,
    box_color: str | None = None,
    box_padding: float = 8.0,
    image_format: str | None = None,
    quality: int = 90,
) -> Path | Image.Image:
    """Draw *name* on *background* and return or save the result.

    The name is placed over the calmest region of the background, steered away
    from *faces* when *avoid_faces* is set. The background under the text is
    sampled and the text colour is chosen to contrast with it; when the ratio
    falls below *min_contrast*, an outline, shadow or backing box is added.

    When *destination* is given the result is written there and the path is
    returned; otherwise the composited :class:`PIL.Image.Image` is returned.
    """
    if not isinstance(name, str) or not name.strip():
        raise RenderError("name must not be empty")

    canvas = _open_background(background)
    if size is not None:
        canvas = _fit_to_size(canvas, size)

    font = _load_font(font_family, font_size)
    text_width, text_height = _measure_text(name, font)

    region = find_text_placement(
        canvas,
        (text_width, text_height),
        faces=faces,
        avoid_faces=avoid_faces,
        face_padding=face_padding,
        margin=margin,
    )
    if region.width < text_width or region.height < text_height:
        font = _shrink_to_fit(name, font_family, font_size, region.width)
        text_width, text_height = _measure_text(name, font)

    box = (
        region.x,
        region.y,
        region.x + region.width,
        region.y + region.height,
    )
    background_hex = _sample_background(canvas, box)

    result = ensure_contrast(
        background_hex,
        color,
        min_contrast=min_contrast,
        strategy=contrast_strategy,
        outline_color=outline_color,
        outline_width=outline_width,
        shadow_color=shadow_color,
        shadow_offset=shadow_offset,
        shadow_blur=shadow_blur,
        box_color=box_color,
        box_padding=box_padding,
    )

    fill = parse_color(result.text_color)
    stroke = None
    stroke_width = 0.0
    if result.applied and result.strategy is ContrastStrategy.OUTLINE:
        outline = result.outline_color
        if outline is not None:
            stroke = parse_color(outline)
            stroke_width = result.outline_width

    text_image = _render_text(
        name,
        font,
        fill,
        stroke=stroke,
        stroke_width=stroke_width,
    )
    text_image, offset_x, offset_y = _apply_treatment(text_image, result)

    x = region.x + offset_x
    y = region.y + offset_y
    canvas.alpha_composite(text_image, dest=(x, y))

    if destination is not None:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        fmt = image_format
        if fmt is None:
            fmt = target.suffix.lstrip(".") if target.suffix else "png"
        options: dict[str, object] = {}
        if fmt.lower() in {"jpeg", "jpg", "webp"}:
            options["quality"] = quality
        canvas.info.clear()
        canvas.getexif().clear()
        canvas.save(target, format=fmt.upper(), **options)
        return target
    return canvas

"""Deterministic text contrast selection and treatments.

Text drawn over a busy or mid tone background can vanish even when its colour
is valid. This module implements the WCAG relative luminance and contrast
ratio formulas, picks the light or dark text colour with the best ratio and,
when that ratio is below the configured minimum, describes the outline, shadow
or backing box that makes the text readable.

The module also applies those treatments to an already rendered text image.
All work is pure Pillow and colour arithmetic: no model, network or external
tool is involved, so the same inputs always produce the same pixels.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

from PIL import Image, ImageColor, ImageDraw, ImageFilter

from aivia.core.config.sections import RenderConfig

from .errors import RenderError

__all__ = [
    "ContrastResult",
    "ContrastStrategy",
    "ContrastStyle",
    "ContrastUtility",
    "add_backing_box",
    "add_contrast_treatment",
    "add_outline",
    "add_shadow",
    "apply_contrast",
    "apply_contrast_treatment",
    "best_text_color",
    "choose_text_color",
    "choose_text_colour",
    "contrast",
    "contrast_ratio",
    "ensure_contrast",
    "ensure_text_contrast",
    "luminance",
    "make_readable",
    "parse_color",
    "pick_text_color",
    "relative_luminance",
    "resolve_contrast",
]

type ColorInput = str | Sequence[int]
"""A colour string, an RGB triple or an RGBA quadruple."""

type ImageInput = Image.Image | str
"""A decoded image or a local path."""

_LIGHT = "#ffffff"
_DARK = "#000000"


class ContrastStrategy(StrEnum):
    """How to lift text contrast when the plain colour pair is too close."""

    OUTLINE = "outline"
    SHADOW = "shadow"
    BOX = "box"

    @classmethod
    def parse(cls, value: Any) -> ContrastStrategy:
        """Return *value* as a strategy, raising :class:`RenderError` if invalid."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise RenderError("contrast strategy must be a string")
        try:
            return cls(value.strip().lower())
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise RenderError(
                f"unsupported contrast strategy {value!r} (expected: {choices})"
            ) from exc


ContrastStyle = ContrastStrategy


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RenderError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise RenderError(f"{name} must be finite")
    return number


def _rgba(value: Any, name: str = "colour") -> tuple[int, int, int, int]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise RenderError(f"{name} must not be empty")
        try:
            return cast(tuple[int, int, int, int], ImageColor.getcolor(text, "RGBA"))
        except ValueError as exc:
            raise RenderError(f"unsupported {name} {value!r}") from exc
    if isinstance(value, (tuple, list)) and len(value) in (3, 4):
        channels: list[int] = []
        for channel in value:
            if isinstance(channel, bool) or not isinstance(channel, (int, float)):
                raise RenderError(f"{name} channels must be numbers")
            number = round(float(channel))
            if number < 0 or number > 255:
                raise RenderError(f"{name} channels must be between 0 and 255")
            channels.append(number)
        if len(channels) == 3:
            channels.append(255)
        return channels[0], channels[1], channels[2], channels[3]
    raise RenderError(f"{name} must be a colour string, RGB or RGBA sequence")


def parse_color(value: Any, name: str = "colour") -> tuple[int, int, int, int]:
    """Return *value* as an RGBA tuple, raising :class:`RenderError` if invalid."""
    return _rgba(value, name)


def _hex(value: tuple[int, int, int, int]) -> str:
    red, green, blue, alpha = value
    if alpha >= 255:
        return f"#{red:02x}{green:02x}{blue:02x}"
    return f"#{red:02x}{green:02x}{blue:02x}{alpha:02x}"


def _over(
    foreground: tuple[int, int, int, int],
    background: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    alpha = foreground[3] / 255.0
    if alpha >= 1.0:
        return foreground
    return (
        round(foreground[0] * alpha + background[0] * (1.0 - alpha)),
        round(foreground[1] * alpha + background[1] * (1.0 - alpha)),
        round(foreground[2] * alpha + background[2] * (1.0 - alpha)),
        255,
    )


def _linear(channel: int) -> float:
    value = channel / 255.0
    if value <= 0.03928:
        return value / 12.92
    return float(((value + 0.055) / 1.055) ** 2.4)


def relative_luminance(color: ColorInput) -> float:
    """Return the WCAG relative luminance of *color* in ``0..1``."""
    red, green, blue, _ = _rgba(color)
    return 0.2126 * _linear(red) + 0.7152 * _linear(green) + 0.0722 * _linear(blue)


luminance = relative_luminance


def _ratio_from_luminance(first: float, second: float) -> float:
    lighter = max(first, second)
    darker = min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def contrast_ratio(foreground: ColorInput, background: ColorInput) -> float:
    """Return the WCAG contrast ratio between two colours, from 1 to 21."""
    foreground_rgba = _rgba(foreground, "foreground colour")
    background_rgba = _rgba(background, "background colour")
    if background_rgba[3] < 255:
        background_rgba = _over(background_rgba, (255, 255, 255, 255))
    if foreground_rgba[3] < 255:
        foreground_rgba = _over(foreground_rgba, background_rgba)
    return _ratio_from_luminance(
        relative_luminance(foreground_rgba),
        relative_luminance(background_rgba),
    )


contrast = contrast_ratio


def choose_text_color(
    background: ColorInput,
    *,
    light: ColorInput = _LIGHT,
    dark: ColorInput = _DARK,
) -> str:
    """Return the light or dark colour with the higher contrast on *background*."""
    light_rgba = _rgba(light, "light colour")
    dark_rgba = _rgba(dark, "dark colour")
    light_ratio = contrast_ratio(light_rgba, background)
    dark_ratio = contrast_ratio(dark_rgba, background)
    chosen = light_rgba if light_ratio >= dark_ratio else dark_rgba
    return _hex(chosen)


choose_text_colour = choose_text_color
best_text_color = choose_text_color
pick_text_color = choose_text_color


@dataclass(frozen=True, slots=True)
class ContrastResult:
    """The chosen text colour and the treatment needed to keep it readable."""

    text_color: str
    background_color: str
    ratio: float
    base_ratio: float
    min_contrast: float
    strategy: ContrastStrategy
    applied: bool
    meets_contrast: bool
    outline_color: str | None = None
    outline_width: float = 0.0
    shadow_color: str | None = None
    shadow_offset: tuple[float, float] = (0.0, 0.0)
    shadow_blur: float = 0.0
    box_color: str | None = None
    box_padding: float = 0.0

    @property
    def color(self) -> str:
        """Alias for :attr:`text_color`."""
        return self.text_color

    @property
    def foreground(self) -> str:
        """Alias for :attr:`text_color`."""
        return self.text_color

    @property
    def background(self) -> str:
        """Alias for :attr:`background_color`."""
        return self.background_color

    @property
    def contrast(self) -> float:
        """Alias for :attr:`ratio`."""
        return self.ratio

    @property
    def satisfied(self) -> bool:
        """Whether the resulting colour pair meets the configured contrast."""
        return self.meets_contrast

    @property
    def treatment(self) -> ContrastStrategy | None:
        """The applied strategy, or ``None`` when no treatment was needed."""
        return self.strategy if self.applied else None


def _render_config(
    config: Any,
    render: Any,
) -> RenderConfig | None:
    for value in (config, render):
        if value is None:
            continue
        if isinstance(value, RenderConfig):
            return value
        if isinstance(value, Mapping):
            return RenderConfig.from_mapping(value)
        candidate = getattr(value, "render", None)
        if candidate is not None:
            return _render_config(candidate, None)
        return RenderConfig.from_mapping(cast(Any, value))
    return None


def _contrast_bounds() -> tuple[float | None, float | None]:
    field = RenderConfig.model_fields["min_contrast"]
    lower = next(
        (
            float(value)
            for item in field.metadata
            if (value := getattr(item, "ge", None)) is not None
        ),
        None,
    )
    upper = next(
        (
            float(value)
            for item in field.metadata
            if (value := getattr(item, "le", None)) is not None
        ),
        None,
    )
    return lower, upper


def _min_contrast(value: Any, config: RenderConfig | None) -> float:
    if value is None:
        value = RenderConfig().min_contrast if config is None else config.min_contrast
    number = _finite(value, "min_contrast")
    lower, upper = _contrast_bounds()
    if (lower is not None and number < lower) or (upper is not None and number > upper):
        raise RenderError(f"min_contrast must be between {lower} and {upper}")
    return number


def _strategy(value: Any, config: RenderConfig | None) -> ContrastStrategy:
    if value is None:
        value = ContrastStrategy.OUTLINE if config is None else config.contrast_strategy
    return ContrastStrategy.parse(value)


def _candidate_color(
    text: tuple[int, int, int, int],
    background: tuple[int, int, int, int],
    light: tuple[int, int, int, int],
    dark: tuple[int, int, int, int],
    explicit: Any,
) -> tuple[int, int, int, int]:
    if explicit is not None:
        return _rgba(explicit, "treatment colour")
    options = [light, dark]
    different = [option for option in options if option[:3] != text[:3]]
    pool = different or options
    return max(pool, key=lambda option: contrast_ratio(option, background))


def _shadow_offset(value: Any) -> tuple[float, float]:
    if value is None:
        return 2.0, 2.0
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return _finite(value[0], "shadow_offset"), _finite(value[1], "shadow_offset")
    offset = _finite(value, "shadow_offset")
    return offset, offset


def ensure_contrast(
    background: ColorInput,
    text_color: ColorInput | None = None,
    *,
    min_contrast: float | None = None,
    strategy: ContrastStrategy | str | None = None,
    config: Any | None = None,
    render: Any | None = None,
    light: ColorInput = _LIGHT,
    dark: ColorInput = _DARK,
    outline_color: ColorInput | None = None,
    outline_width: float = 2.0,
    shadow_color: ColorInput | None = None,
    shadow_offset: tuple[float, float] | float = (2.0, 2.0),
    shadow_blur: float = 2.0,
    box_color: ColorInput | None = None,
    box_padding: float = 8.0,
    foreground: ColorInput | None = None,
    colour: ColorInput | None = None,
) -> ContrastResult:
    """Choose readable text and describe the treatment needed on *background*.

    With no *text_color* the light or dark colour with the best ratio is used.
    When that ratio is below *min_contrast*, the configured *strategy* adds an
    outline, a drop shadow or a backing box. Passing a :class:`RenderConfig`
    through *config* supplies ``min_contrast`` and ``contrast_strategy`` unless
    they are given explicitly.
    """
    if text_color is None:
        text_color = foreground if foreground is not None else colour
    elif foreground is not None or colour is not None:
        raise RenderError("pass only one of text_color, foreground or colour")

    background_rgba = _rgba(background, "background colour")
    if background_rgba[3] < 255:
        background_rgba = _over(background_rgba, (255, 255, 255, 255))
    background_hex = _hex(background_rgba)

    configured = _render_config(config, render)
    minimum = _min_contrast(min_contrast, configured)
    chosen_strategy = _strategy(strategy, configured)

    if text_color is None:
        text_rgba = _rgba(choose_text_color(background_rgba, light=light, dark=dark))
    else:
        text_rgba = _rgba(text_color, "text colour")
        if text_rgba[3] < 255:
            text_rgba = _over(text_rgba, background_rgba)

    light_rgba = _rgba(light, "light colour")
    dark_rgba = _rgba(dark, "dark colour")
    base_ratio = contrast_ratio(text_rgba, background_rgba)
    ratio = base_ratio
    applied = False

    outline_hex: str | None = None
    outline_width_value = 0.0
    shadow_hex: str | None = None
    offset = (0.0, 0.0)
    blur_value = 0.0
    box_hex: str | None = None
    box_padding_value = 0.0

    if base_ratio < minimum:
        applied = True
        if chosen_strategy is ContrastStrategy.OUTLINE:
            width = _finite(outline_width, "outline_width")
            if width < 0.0:
                raise RenderError("outline_width must not be negative")
            outline_rgba = _candidate_color(
                text_rgba,
                background_rgba,
                light_rgba,
                dark_rgba,
                outline_color,
            )
            outline_hex = _hex(outline_rgba)
            outline_width_value = width
            ratio = max(base_ratio, contrast_ratio(outline_rgba, background_rgba))
        elif chosen_strategy is ContrastStrategy.SHADOW:
            offset = _shadow_offset(shadow_offset)
            blur_value = _finite(shadow_blur, "shadow_blur")
            if blur_value < 0.0:
                raise RenderError("shadow_blur must not be negative")
            shadow_rgba = _candidate_color(
                text_rgba,
                background_rgba,
                light_rgba,
                dark_rgba,
                shadow_color,
            )
            shadow_hex = _hex(shadow_rgba)
            ratio = max(base_ratio, contrast_ratio(shadow_rgba, background_rgba))
        else:
            padding = _finite(box_padding, "box_padding")
            if padding < 0.0:
                raise RenderError("box_padding must not be negative")
            box_rgba = _candidate_color(
                text_rgba,
                background_rgba,
                light_rgba,
                dark_rgba,
                box_color,
            )
            if text_color is None:
                text_rgba = _rgba(choose_text_color(box_rgba, light=light, dark=dark))
            box_hex = _hex(box_rgba)
            box_padding_value = padding
            ratio = contrast_ratio(text_rgba, box_rgba)

    return ContrastResult(
        text_color=_hex(text_rgba),
        background_color=background_hex,
        ratio=ratio,
        base_ratio=base_ratio,
        min_contrast=minimum,
        strategy=chosen_strategy,
        applied=applied,
        meets_contrast=ratio >= minimum,
        outline_color=outline_hex,
        outline_width=outline_width_value,
        shadow_color=shadow_hex,
        shadow_offset=offset,
        shadow_blur=blur_value,
        box_color=box_hex,
        box_padding=box_padding_value,
    )


ensure_text_contrast = ensure_contrast
resolve_contrast = ensure_contrast
make_readable = ensure_contrast


def _image(source: ImageInput) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.convert("RGBA")
    try:
        with Image.open(source) as opened:
            return opened.convert("RGBA")
    except (OSError, ValueError) as exc:
        raise RenderError("cannot open image for contrast treatment") from exc


def _opacity_alpha(alpha: Image.Image, opacity: float) -> Image.Image:
    value = _finite(opacity, "opacity")
    if value < 0.0 or value > 1.0:
        raise RenderError("opacity must be between 0 and 1")
    if value >= 1.0:
        return alpha
    return alpha.point(lambda channel: round(channel * value))


def add_outline(
    image: ImageInput,
    color: ColorInput = _DARK,
    width: float = 2.0,
    *,
    opacity: float = 1.0,
) -> Image.Image:
    """Return *image* with an outline of *color* around its opaque pixels."""
    source = _image(image)
    outline_width = _finite(width, "outline_width")
    if outline_width <= 0.0:
        return source.copy()
    outline_rgba = _rgba(color, "outline colour")
    alpha = source.getchannel("A")
    for _ in range(max(1, round(outline_width))):
        alpha = alpha.filter(ImageFilter.MaxFilter(3))
    outline = Image.new("RGBA", source.size, (*outline_rgba[:3], 0))
    outline.putalpha(_opacity_alpha(alpha, opacity))
    return Image.alpha_composite(outline, source)


def add_shadow(
    image: ImageInput,
    color: ColorInput = _DARK,
    offset: tuple[float, float] | float = (2.0, 2.0),
    blur: float = 2.0,
    *,
    opacity: float = 0.5,
) -> Image.Image:
    """Return *image* with a drop shadow of *color* behind its opaque pixels."""
    source = _image(image)
    shadow_rgba = _rgba(color, "shadow colour")
    offset_x, offset_y = _shadow_offset(offset)
    blur_value = _finite(blur, "shadow_blur")
    if blur_value < 0.0:
        raise RenderError("shadow_blur must not be negative")
    margin = max(1, math.ceil(blur_value * 3)) + max(
        abs(round(offset_x)), abs(round(offset_y))
    )
    size = source.width + margin * 2, source.height + margin * 2
    alpha = Image.new("L", size, 0)
    alpha.paste(
        source.getchannel("A"),
        (margin + round(offset_x), margin + round(offset_y)),
    )
    if blur_value > 0.0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(blur_value))
    shadow = Image.new("RGBA", size, (*shadow_rgba[:3], 0))
    shadow.putalpha(_opacity_alpha(alpha, opacity))
    text = Image.new("RGBA", size, (0, 0, 0, 0))
    text.paste(source, (margin, margin))
    return Image.alpha_composite(shadow, text)


def add_backing_box(
    image: ImageInput,
    color: ColorInput = _DARK,
    padding: float = 8.0,
    *,
    radius: float = 0.0,
    opacity: float = 1.0,
) -> Image.Image:
    """Return *image* with a backing box of *color* behind its opaque pixels."""
    source = _image(image)
    box_rgba = _rgba(color, "box colour")
    pad = _finite(padding, "box_padding")
    if pad < 0.0:
        raise RenderError("box_padding must not be negative")
    corner = _finite(radius, "box_radius")
    if corner < 0.0:
        raise RenderError("box_radius must not be negative")
    bounds = source.getchannel("A").getbbox()
    if bounds is None:
        return source.copy()
    left, top, right, bottom = bounds
    pad_px = round(pad)
    expand_left = max(0, pad_px - left)
    expand_top = max(0, pad_px - top)
    expand_right = max(0, pad_px - (source.width - right))
    expand_bottom = max(0, pad_px - (source.height - bottom))
    size = (
        source.width + expand_left + expand_right,
        source.height + expand_top + expand_bottom,
    )
    box = Image.new("RGBA", size, (*box_rgba[:3], 0))
    draw = ImageDraw.Draw(box)
    rectangle = (
        left - pad_px + expand_left,
        top - pad_px + expand_top,
        right + pad_px + expand_left,
        bottom + pad_px + expand_top,
    )
    draw.rounded_rectangle(
        rectangle,
        radius=round(corner),
        fill=(*box_rgba[:3], round(255 * _finite(opacity, "opacity"))),
    )
    box.paste(source, (expand_left, expand_top), source)
    return box


def apply_contrast_treatment(
    image: ImageInput,
    strategy: ContrastStrategy | str = ContrastStrategy.OUTLINE,
    *,
    color: ColorInput | None = None,
    width: float = 2.0,
    offset: tuple[float, float] | float = (2.0, 2.0),
    blur: float = 2.0,
    padding: float = 8.0,
    radius: float = 0.0,
    opacity: float | None = None,
) -> Image.Image:
    """Apply an outline, shadow or backing box to a rendered text image."""
    chosen = ContrastStrategy.parse(strategy)
    if chosen is ContrastStrategy.OUTLINE:
        return add_outline(
            image,
            _DARK if color is None else color,
            width,
            opacity=1.0 if opacity is None else opacity,
        )
    if chosen is ContrastStrategy.SHADOW:
        return add_shadow(
            image,
            _DARK if color is None else color,
            offset,
            blur,
            opacity=0.5 if opacity is None else opacity,
        )
    return add_backing_box(
        image,
        _DARK if color is None else color,
        padding,
        radius=radius,
        opacity=1.0 if opacity is None else opacity,
    )


add_contrast_treatment = apply_contrast_treatment


def apply_contrast(
    target: ColorInput | Image.Image,
    text_color: ColorInput | None = None,
    *,
    strategy: ContrastStrategy | str | None = None,
    **kwargs: Any,
) -> ContrastResult | Image.Image:
    """Resolve colours, or apply a treatment when *target* is a text image."""
    if isinstance(target, Image.Image):
        chosen = strategy
        if chosen is None and isinstance(text_color, str):
            try:
                chosen = ContrastStrategy.parse(text_color)
            except RenderError:
                chosen = None
        return apply_contrast_treatment(
            target,
            ContrastStrategy.OUTLINE if chosen is None else chosen,
            **kwargs,
        )
    return ensure_contrast(target, text_color, strategy=strategy, **kwargs)


class ContrastUtility:
    """A small object facade over the module level contrast helpers."""

    def __init__(
        self,
        config: Any | None = None,
        *,
        min_contrast: float | None = None,
        strategy: ContrastStrategy | str | None = None,
    ) -> None:
        self.config = config
        self.min_contrast = min_contrast
        self.strategy = strategy

    def choose_text_color(
        self,
        background: ColorInput,
        *,
        light: ColorInput = _LIGHT,
        dark: ColorInput = _DARK,
    ) -> str:
        """Return the best light or dark text colour on *background*."""
        return choose_text_color(background, light=light, dark=dark)

    def ensure_contrast(
        self,
        background: ColorInput,
        text_color: ColorInput | None = None,
        **kwargs: Any,
    ) -> ContrastResult:
        """Resolve contrast using this instance's config and defaults."""
        kwargs.setdefault("config", self.config)
        kwargs.setdefault("min_contrast", self.min_contrast)
        kwargs.setdefault("strategy", self.strategy)
        return ensure_contrast(background, text_color, **kwargs)

    def apply(
        self,
        image: ImageInput,
        strategy: ContrastStrategy | str | None = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Apply a treatment using this instance's configured strategy."""
        chosen = self.strategy if strategy is None else strategy
        if chosen is None:
            configured = _render_config(self.config, None)
            chosen = (
                ContrastStrategy.OUTLINE
                if configured is None
                else configured.contrast_strategy
            )
        return apply_contrast_treatment(image, chosen, **kwargs)

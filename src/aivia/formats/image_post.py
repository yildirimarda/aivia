"""Image post and carousel formats with aspect presets and text per slide.

An image post is one still with a caption; a carousel is an ordered set of
stills, each with its own text. Both share the same composition builder and
aspect presets, so a profile can switch between a square feed post and a
portrait carousel without changing the writing stage. The format is
deterministic and free: the visuals stage generates the images and this module
only places them and their text.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from aivia.compose.io import composition_to_yaml
from aivia.compose.render import render_still
from aivia.compose.schema import (
    CURRENT_COMPOSITION_SCHEMA_VERSION,
    Canvas,
    Composition,
    FitMode,
    ImageLayer,
    TextAlign,
    TextLayer,
    TextWeight,
    VerticalAlign,
)
from aivia.core.errors import ConfigError
from aivia.core.models.run import StageResult
from aivia.core.models.script import Script
from aivia.core.stages import Stage, StageContext

__all__ = [
    "ASPECT_PRESETS",
    "CAROUSEL_SLIDE_COMPOSITION_PATTERN",
    "CAROUSEL_SLIDE_FILENAME_PATTERN",
    "DEFAULT_ASPECT_PRESET",
    "DEFAULT_CAROUSEL_DIRNAME",
    "DEFAULT_CAROUSEL_MANIFEST_FILENAME",
    "DEFAULT_IMAGE_POST_COMPOSITION_FILENAME",
    "DEFAULT_IMAGE_POST_FILENAME",
    "DEFAULT_VISUAL_MANIFEST_FILENAME",
    "AspectPreset",
    "Carousel",
    "CarouselFormat",
    "CarouselRenderStage",
    "CarouselRenderer",
    "CarouselStage",
    "ImagePost",
    "ImagePostFormat",
    "ImagePostRenderStage",
    "ImagePostRenderer",
    "ImagePostSlide",
    "ImagePostStage",
    "build_carousel_stage",
    "build_image_post_stage",
    "carousel",
    "carousel_compositions",
    "carousel_stage",
    "image_post",
    "image_post_composition",
    "image_post_slides",
    "image_post_stage",
    "render_carousel",
    "render_image_post",
    "resolve_aspect_preset",
    "save_carousel",
    "save_image_post",
    "write_carousel",
    "write_image_post",
]

DEFAULT_ASPECT_PRESET = "square"
"""The preset used when a caller does not name an aspect."""

DEFAULT_IMAGE_POST_COMPOSITION_FILENAME = "image-post.yaml"
DEFAULT_IMAGE_POST_FILENAME = "image-post.png"
DEFAULT_CAROUSEL_MANIFEST_FILENAME = "carousel.json"
DEFAULT_CAROUSEL_DIRNAME = "carousel"
DEFAULT_VISUAL_MANIFEST_FILENAME = "visual-reuse.json"
CAROUSEL_SLIDE_FILENAME_PATTERN = "slide-{index}.png"
CAROUSEL_SLIDE_COMPOSITION_PATTERN = "slide-{index}.yaml"

_DEFAULT_BACKGROUND = "#000000"
_DEFAULT_TEXT_COLOR = "#ffffff"
_DEFAULT_TITLE_COLOR = "#ffffff"


@dataclass(frozen=True, slots=True)
class AspectPreset:
    """One named output shape, with the pixel canvas it renders into."""

    name: str
    width: int
    height: int
    label: str = ""

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ConfigError("aspect preset dimensions must be positive")

    @property
    def size(self) -> tuple[int, int]:
        """The canvas as ``(width, height)``."""
        return self.width, self.height

    @property
    def aspect_ratio(self) -> float:
        """The width divided by the height."""
        return self.width / self.height

    @property
    def ratio(self) -> str:
        """The reduced ``width:height`` ratio."""
        divisor = math.gcd(self.width, self.height) or 1
        return f"{self.width // divisor}:{self.height // divisor}"

    @property
    def canvas(self) -> Canvas:
        """A validated composition canvas for this preset."""
        return Canvas(width=self.width, height=self.height)

    def to_dict(self) -> dict[str, Any]:
        """Return the preset as a JSON-compatible mapping."""
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "ratio": self.ratio,
            "label": self.label,
        }


ASPECT_PRESETS: dict[str, AspectPreset] = {
    "square": AspectPreset("square", 1080, 1080, "1:1 feed post"),
    "portrait": AspectPreset("portrait", 1080, 1350, "4:5 feed post"),
    "story": AspectPreset("story", 1080, 1920, "9:16 story or reel"),
    "landscape": AspectPreset("landscape", 1920, 1080, "16:9 landscape post"),
}
"""The built-in aspect presets, keyed by their stable names."""

_ASPECT_ALIASES = {
    "feed": "portrait",
    "post": "square",
    "vertical": "story",
    "reel": "story",
    "reels": "story",
    "horizontal": "landscape",
    "wide": "landscape",
}
_RATIO_PRESETS = {preset.ratio: preset for preset in ASPECT_PRESETS.values()}
_SIZE_PRESETS = {preset.size: preset for preset in ASPECT_PRESETS.values()}


def _custom_preset(name: str, width: int, height: int) -> AspectPreset:
    if width <= 0 or height <= 0:
        raise ConfigError("aspect preset dimensions must be positive")
    return AspectPreset(
        name=name, width=width, height=height, label=f"{width}x{height}"
    )


def resolve_aspect_preset(value: Any = None) -> AspectPreset:
    """Resolve a preset name, a ratio, a size or a mapping to an aspect preset.

    A known name keeps its stable identity. A ratio such as ``4:5`` or a size
    such as ``1080x1350`` returns the matching built-in preset when there is
    one, otherwise a custom preset scaled to a sensible social canvas.
    """
    if value is None:
        return ASPECT_PRESETS[DEFAULT_ASPECT_PRESET]
    if isinstance(value, AspectPreset):
        return value
    if isinstance(value, Mapping):
        values = dict(value)
        named = (
            values.get("preset")
            or values.get("name")
            or values.get("aspect")
            or values.get("aspect_ratio")
        )
        if named:
            return resolve_aspect_preset(named)
        width = values.get("width", values.get("w"))
        height = values.get("height", values.get("h"))
        if width is not None and height is not None:
            try:
                return _custom_preset("custom", int(width), int(height))
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"invalid aspect preset: {value!r}") from exc
        raise ConfigError(f"invalid aspect preset: {value!r}")
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and len(value) == 2
    ):
        try:
            return _custom_preset("custom", int(value[0]), int(value[1]))
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"invalid aspect preset: {value!r}") from exc

    text = str(value).strip().lower().replace(" ", "")
    if not text:
        return ASPECT_PRESETS[DEFAULT_ASPECT_PRESET]
    if text in ASPECT_PRESETS:
        return ASPECT_PRESETS[text]
    if text in _ASPECT_ALIASES:
        return ASPECT_PRESETS[_ASPECT_ALIASES[text]]

    if ":" in text:
        left, right = text.split(":", 1)
        try:
            width, height = int(left), int(right)
        except ValueError as exc:
            raise ConfigError(f"invalid aspect preset: {value!r}") from exc
        if width <= 0 or height <= 0:
            raise ConfigError("aspect preset dimensions must be positive")
        divisor = math.gcd(width, height) or 1
        ratio = f"{width // divisor}:{height // divisor}"
        if ratio in _RATIO_PRESETS:
            return _RATIO_PRESETS[ratio]
        if width == height:
            return _custom_preset(text, 1080, 1080)
        if width > height:
            return _custom_preset(text, 1920, round(1920 * height / width))
        return _custom_preset(text, 1080, round(1080 * height / width))

    if "x" in text or "*" in text:
        parts = re.split(r"[x*]", text)
        if len(parts) == 2:
            try:
                width, height = int(parts[0]), int(parts[1])
            except ValueError as exc:
                raise ConfigError(f"invalid aspect preset: {value!r}") from exc
            if (width, height) in _SIZE_PRESETS:
                return _SIZE_PRESETS[(width, height)]
            return _custom_preset(text, width, height)

    available = ", ".join(sorted(ASPECT_PRESETS))
    raise ConfigError(
        f"unknown aspect preset: {value!r} (available: {available})",
        details={"aspect": str(value), "available": sorted(ASPECT_PRESETS)},
    )


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


class ImagePostSlide(BaseModel):
    """One slide's text, optional title and optional background image."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    text: str = Field(
        default="",
        validation_alias=AliasChoices(
            "text",
            "caption",
            "copy",
            "body",
            "narration",
        ),
    )
    title: str = Field(
        default="",
        validation_alias=AliasChoices("title", "heading", "headline"),
    )
    source: str = Field(
        default="",
        validation_alias=AliasChoices(
            "source",
            "image",
            "image_path",
            "path",
            "file",
            "asset",
            "asset_id",
        ),
    )
    prompt: str = Field(
        default="",
        validation_alias=AliasChoices(
            "prompt",
            "image_prompt",
            "visual_prompt",
        ),
    )
    alt_text: str = Field(
        default="",
        validation_alias=AliasChoices(
            "alt_text",
            "alt",
            "accessibility_text",
        ),
    )

    @field_validator("text", "title", "source", "prompt", "alt_text", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> str:
        return _text(value)

    @property
    def caption(self) -> str:
        """Alias for :attr:`text`."""
        return self.text

    @property
    def image(self) -> str:
        """Alias for :attr:`source`."""
        return self.source

    @property
    def has_text(self) -> bool:
        """Whether this slide has any visible text."""
        return bool(self.text or self.title)

    @property
    def has_image(self) -> bool:
        """Whether this slide has a background image."""
        return bool(self.source)


class ImagePost(BaseModel):
    """A title, an aspect preset and the ordered slides of one post."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    title: str = ""
    aspect: str = DEFAULT_ASPECT_PRESET
    background: str = _DEFAULT_BACKGROUND
    slides: list[ImagePostSlide] = Field(
        default_factory=list,
        validation_alias=AliasChoices("slides", "items", "pages"),
    )

    @field_validator("title", "aspect", "background", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> str:
        return _text(value)


Carousel = ImagePost
"""A carousel is an image post with more than one slide."""


def _image_source(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    path = getattr(value, "path", None)
    if isinstance(path, str):
        return path.strip()
    if isinstance(value, Mapping):
        for key in ("path", "source", "file", "asset", "url"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return ""


def _slide_from(value: Any) -> ImagePostSlide:
    if isinstance(value, ImagePostSlide):
        return value
    if isinstance(value, Script):
        slides = _slides_from_script(value)
        if not slides:
            raise ConfigError("script has no slide text")
        return slides[0]
    if isinstance(value, Mapping):
        return ImagePostSlide.model_validate(value)
    if isinstance(value, str):
        return ImagePostSlide(text=value)
    raise ConfigError(f"cannot read an image post slide from {type(value).__name__}")


def _slides_from_script(script: Script) -> list[ImagePostSlide]:
    slides: list[ImagePostSlide] = []
    for beat in script.beats:
        text = _text(beat.text)
        if not text and beat.dialogue:
            text = " ".join(
                line.text.strip() for line in beat.dialogue if line.text.strip()
            )
        prompt = _text(beat.image_prompt.prompt) if beat.image_prompt else ""
        slides.append(ImagePostSlide(text=text, prompt=prompt))
    if not slides and script.title:
        slides.append(ImagePostSlide(text=script.title))
    return slides


def _apply_images(
    slides: Sequence[ImagePostSlide],
    images: Any,
) -> list[ImagePostSlide]:
    if images is None:
        return list(slides)
    if isinstance(images, Mapping):
        resolved: list[ImagePostSlide] = []
        for index, slide in enumerate(slides):
            if slide.source:
                resolved.append(slide)
                continue
            candidate = (
                images.get(index)
                or images.get(str(index))
                or images.get(str(index + 1))
            )
            source = _image_source(candidate)
            resolved.append(
                slide.model_copy(update={"source": source}) if source else slide
            )
        return resolved
    if isinstance(images, (str, bytes, bytearray)):
        values: Sequence[Any] = [images]
    elif isinstance(images, Iterable):
        values = list(images)
    else:
        values = [images]
    resolved = []
    for index, slide in enumerate(slides):
        if slide.source or index >= len(values):
            resolved.append(slide)
            continue
        source = _image_source(values[index])
        resolved.append(
            slide.model_copy(update={"source": source}) if source else slide
        )
    return resolved


def _normalise_slides(
    source: Any,
    *,
    images: Any = None,
    text: str | None = None,
) -> list[ImagePostSlide]:
    slides: list[ImagePostSlide] = []
    if isinstance(source, ImagePost):
        slides = list(source.slides)
    elif isinstance(source, Script):
        slides = _slides_from_script(source)
    elif isinstance(source, Mapping):
        values = dict(source)
        if "script" in values:
            slides = _normalise_slides(values["script"])
        elif "slides" in values:
            raw = values["slides"]
            if isinstance(raw, (str, bytes, bytearray)):
                slides = [_slide_from(raw)]
            else:
                slides = [_slide_from(item) for item in raw]
        elif "beats" in values:
            slides = _slides_from_script(Script.model_validate(values))
        else:
            slides = [_slide_from(values)]
    elif isinstance(source, Sequence) and not isinstance(
        source, (str, bytes, bytearray)
    ):
        for item in source:
            if isinstance(item, Script):
                slides.extend(_slides_from_script(item))
            elif isinstance(item, Mapping) and "beats" in item and "slides" not in item:
                slides.extend(_slides_from_script(Script.model_validate(item)))
            else:
                slides.append(_slide_from(item))
    elif source is None:
        slides = []
    else:
        slides = [_slide_from(source)]

    slides = _apply_images(slides, images)
    if text is not None and text.strip():
        if slides:
            slides[0] = slides[0].model_copy(update={"text": text.strip()})
        else:
            slides = [ImagePostSlide(text=text)]
    return slides


def image_post_slides(
    source: Any,
    *,
    images: Any = None,
    text: str | None = None,
) -> list[ImagePostSlide]:
    """Return the ordered slides read from a script, mapping or slide list."""
    return _normalise_slides(source, images=images, text=text)


def _text_layer(
    *,
    layer_id: str,
    text: str,
    x: int,
    y: int,
    width: int,
    height: int,
    font_size: float,
    color: str,
    vertical_align: VerticalAlign,
    bold: bool,
    over_image: bool,
) -> TextLayer:
    stroke_width = max(1, round(font_size * 0.06)) if over_image else 0.0
    return TextLayer(
        id=layer_id,
        text=text,
        x=x,
        y=y,
        width=width,
        height=height,
        font_family="Lato",
        font_size=font_size,
        font_weight=TextWeight.BOLD if bold else TextWeight.SEMIBOLD,
        color=color,
        align=TextAlign.CENTER,
        vertical_align=vertical_align,
        shrink_to_fit=True,
        wrap=True,
        max_lines=6,
        stroke="#000000" if over_image else None,
        stroke_width=stroke_width,
    )


def _slide_composition(
    slide: ImagePostSlide,
    *,
    preset: AspectPreset,
    title: str = "",
    background: str = _DEFAULT_BACKGROUND,
    composition_id: str = "",
    language: str = "",
    platform: str = "",
    duration_seconds: float = 1.0,
    text_color: str = _DEFAULT_TEXT_COLOR,
    title_color: str = _DEFAULT_TITLE_COLOR,
) -> Composition:
    width, height = preset.size
    margin = max(1, round(min(width, height) * 0.06))
    content_width = max(1, width - margin * 2)
    layers: list[dict[str, Any]] = []
    over_image = bool(slide.source)
    if slide.source:
        layers.append(
            ImageLayer(
                id="image",
                source=slide.source,
                fit=FitMode.COVER,
                x=0,
                y=0,
                width=width,
                height=height,
                z_index=0,
            ).model_dump(mode="python", by_alias=False)
        )

    heading = slide.title or title
    if heading:
        title_size = max(24.0, round(min(width, height) * 0.07))
        layers.append(
            _text_layer(
                layer_id="title",
                text=heading,
                x=margin,
                y=margin,
                width=content_width,
                height=max(1, round(height * 0.2)),
                font_size=title_size,
                color=title_color,
                vertical_align=VerticalAlign.TOP,
                bold=True,
                over_image=over_image,
            ).model_dump(mode="python", by_alias=False)
        )

    if slide.text:
        text_size = max(24.0, round(min(width, height) * 0.055))
        if over_image:
            text_height = max(1, round(height * 0.36))
            text_y = max(margin, height - margin - text_height)
            valign = VerticalAlign.BOTTOM
        else:
            text_height = max(1, height - margin * 2)
            text_y = margin
            valign = VerticalAlign.MIDDLE
        layers.append(
            _text_layer(
                layer_id="text",
                text=slide.text,
                x=margin,
                y=text_y,
                width=content_width,
                height=text_height,
                font_size=text_size,
                color=text_color,
                vertical_align=valign,
                bold=False,
                over_image=over_image,
            ).model_dump(mode="python", by_alias=False)
        )

    if not layers:
        raise ConfigError("an image post slide needs text or an image")
    span = duration_seconds
    if span <= 0.0:
        raise ConfigError("image post duration must be positive")
    payload = {
        "id": composition_id or f"image-post-slide-{preset.name}",
        "schema_version": CURRENT_COMPOSITION_SCHEMA_VERSION,
        "canvas": {"width": width, "height": height, "background": background},
        "duration_seconds": span,
        "fps": 30,
        "layers": layers,
        "language": language,
        "platform": platform,
    }
    try:
        return Composition.model_validate(payload)
    except ValueError as exc:
        raise ConfigError(f"invalid image post composition: {exc}") from exc


def _content(
    source: Any,
    *,
    images: Any = None,
    text: str | None = None,
    title: str | None = None,
    aspect: Any = None,
    background: str | None = None,
) -> tuple[list[ImagePostSlide], str, AspectPreset, str]:
    source_title = ""
    source_aspect: Any = DEFAULT_ASPECT_PRESET
    source_background = _DEFAULT_BACKGROUND
    if isinstance(source, ImagePost):
        source_title = source.title
        source_aspect = source.aspect or source_aspect
        source_background = source.background or source_background
    elif isinstance(source, Script):
        source_title = source.title
    elif isinstance(source, Mapping):
        values = dict(source)
        if "script" in values:
            _, nested_title, nested_aspect, nested_background = _content(
                values["script"]
            )
            source_title = nested_title
            source_aspect = nested_aspect
            source_background = nested_background
        source_title = _text(values.get("title")) or source_title
        source_aspect = (
            values.get("aspect") or values.get("aspect_preset") or source_aspect
        )
        source_background = _text(values.get("background")) or source_background

    slides = _normalise_slides(source, images=images, text=text)
    final_title = (title if title is not None else source_title).strip()
    if not slides and final_title:
        slides = [ImagePostSlide(title=final_title)]
    final_aspect = resolve_aspect_preset(source_aspect if aspect is None else aspect)
    final_background = (
        background if background is not None else source_background
    ).strip() or _DEFAULT_BACKGROUND
    return slides, final_title, final_aspect, final_background


def image_post_composition(
    source: Any,
    *,
    aspect: Any = None,
    title: str | None = None,
    text: str | None = None,
    image: Any = None,
    images: Any = None,
    background: str | None = None,
    composition_id: str = "",
    language: str = "",
    platform: str = "",
    duration_seconds: float = 1.0,
) -> Composition:
    """Build the single still composition for an image post."""
    supplied_images = image if image is not None else images
    slides, heading, preset, canvas_background = _content(
        source,
        images=supplied_images,
        text=text,
        title=title,
        aspect=aspect,
        background=background,
    )
    if not slides:
        raise ConfigError("an image post needs text or an image")
    return _slide_composition(
        slides[0],
        preset=preset,
        title=heading,
        background=canvas_background,
        composition_id=composition_id,
        language=language,
        platform=platform,
        duration_seconds=duration_seconds,
    )


def carousel_compositions(
    source: Any,
    *,
    aspect: Any = None,
    title: str | None = None,
    text: str | None = None,
    images: Any = None,
    background: str | None = None,
    language: str = "",
    platform: str = "",
    duration_seconds: float = 1.0,
) -> list[Composition]:
    """Build one still composition per carousel slide."""
    explicit_title = bool(title and title.strip())
    slides, heading, preset, canvas_background = _content(
        source,
        images=images,
        text=text,
        title=title,
        aspect=aspect,
        background=background,
    )
    if not slides:
        raise ConfigError("a carousel needs at least one slide")
    return [
        _slide_composition(
            slide,
            preset=preset,
            title=heading if explicit_title and index == 1 else "",
            background=canvas_background,
            composition_id=f"carousel-slide-{index}",
            language=language,
            platform=platform,
            duration_seconds=duration_seconds,
        )
        for index, slide in enumerate(slides, start=1)
    ]


def _save_image(image: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def render_image_post(
    source: Any,
    *,
    output: str | Path | None = None,
    time_seconds: float = 0.0,
    source_root: str | Path | None = None,
    **kwargs: Any,
) -> Any:
    """Render one image post and optionally save it to *output*."""
    composition = image_post_composition(source, **kwargs)
    image = render_still(composition, time_seconds, source_root=source_root)
    if output is not None:
        _save_image(image, Path(output).expanduser())
    return image


def save_image_post(
    source: Any,
    output: str | Path,
    *,
    time_seconds: float = 0.0,
    source_root: str | Path | None = None,
    **kwargs: Any,
) -> Path:
    """Render *source* to *output* and return the written path."""
    path = Path(output).expanduser()
    render_image_post(
        source,
        output=path,
        time_seconds=time_seconds,
        source_root=source_root,
        **kwargs,
    )
    return path


write_image_post = save_image_post


def _write_carousel_images(
    compositions: Sequence[Composition],
    directory: Path,
    *,
    pattern: str,
    source_root: str | Path | None = None,
    time_seconds: float = 0.0,
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, composition in enumerate(compositions, start=1):
        path = directory / pattern.format(index=index)
        image = render_still(composition, time_seconds, source_root=source_root)
        paths.append(_save_image(image, path))
    return paths


def render_carousel(
    source: Any,
    *,
    output_dir: str | Path | None = None,
    pattern: str = CAROUSEL_SLIDE_FILENAME_PATTERN,
    time_seconds: float = 0.0,
    source_root: str | Path | None = None,
    **kwargs: Any,
) -> list[Any]:
    """Render every carousel slide, optionally writing them to *output_dir*."""
    compositions = carousel_compositions(source, **kwargs)
    if output_dir is not None:
        _write_carousel_images(
            compositions,
            Path(output_dir).expanduser(),
            pattern=pattern,
            source_root=source_root,
            time_seconds=time_seconds,
        )
    return [
        render_still(composition, time_seconds, source_root=source_root)
        for composition in compositions
    ]


def save_carousel(
    source: Any,
    output_dir: str | Path,
    *,
    pattern: str = CAROUSEL_SLIDE_FILENAME_PATTERN,
    time_seconds: float = 0.0,
    source_root: str | Path | None = None,
    **kwargs: Any,
) -> list[Path]:
    """Render every carousel slide into *output_dir* and return the paths."""
    directory = Path(output_dir).expanduser()
    compositions = carousel_compositions(source, **kwargs)
    return _write_carousel_images(
        compositions,
        directory,
        pattern=pattern,
        source_root=source_root,
        time_seconds=time_seconds,
    )


def write_carousel(
    source: Any,
    output_dir: str | Path,
    *,
    pattern: str = CAROUSEL_SLIDE_FILENAME_PATTERN,
    source_root: str | Path | None = None,
    **kwargs: Any,
) -> list[Path]:
    """Alias for :func:`save_carousel`."""
    return save_carousel(
        source,
        output_dir,
        pattern=pattern,
        source_root=source_root,
        **kwargs,
    )


class ImagePostRenderer:
    """Pin an aspect preset and render many image posts with it."""

    def __init__(
        self,
        aspect: Any = DEFAULT_ASPECT_PRESET,
        *,
        title: str | None = None,
        text: str | None = None,
        background: str | None = None,
    ) -> None:
        self.preset = resolve_aspect_preset(aspect)
        self.title = title
        self.text = text
        self.background = background

    def composition(self, source: Any, **kwargs: Any) -> Composition:
        """Build one composition with the pinned aspect preset."""
        options: dict[str, Any] = {
            "aspect": self.preset,
            "title": self.title,
            "text": self.text,
            "background": self.background,
        }
        options.update(kwargs)
        return image_post_composition(source, **options)

    build = composition
    compose = composition

    def render(
        self,
        source: Any,
        *,
        output: str | Path | None = None,
        **kwargs: Any,
    ) -> Any:
        """Render one image post with the pinned aspect preset."""
        options: dict[str, Any] = {
            "aspect": self.preset,
            "title": self.title,
            "text": self.text,
            "background": self.background,
        }
        options.update(kwargs)
        return render_image_post(source, output=output, **options)

    def save(self, source: Any, output: str | Path, **kwargs: Any) -> Path:
        """Save one image post with the pinned aspect preset."""
        options: dict[str, Any] = {
            "aspect": self.preset,
            "title": self.title,
            "text": self.text,
            "background": self.background,
        }
        options.update(kwargs)
        return save_image_post(source, output, **options)


class CarouselRenderer:
    """Pin an aspect preset and render many carousels with it."""

    def __init__(
        self,
        aspect: Any = DEFAULT_ASPECT_PRESET,
        *,
        title: str | None = None,
        text: str | None = None,
        background: str | None = None,
    ) -> None:
        self.preset = resolve_aspect_preset(aspect)
        self.title = title
        self.text = text
        self.background = background

    def compositions(self, source: Any, **kwargs: Any) -> list[Composition]:
        """Build one composition per slide with the pinned aspect preset."""
        options: dict[str, Any] = {
            "aspect": self.preset,
            "title": self.title,
            "text": self.text,
            "background": self.background,
        }
        options.update(kwargs)
        return carousel_compositions(source, **options)

    build = compositions
    compose = compositions

    def render(
        self,
        source: Any,
        *,
        output_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Render every slide with the pinned aspect preset."""
        options: dict[str, Any] = {
            "aspect": self.preset,
            "title": self.title,
            "text": self.text,
            "background": self.background,
        }
        options.update(kwargs)
        return render_carousel(source, output_dir=output_dir, **options)

    def save(
        self,
        source: Any,
        output_dir: str | Path,
        **kwargs: Any,
    ) -> list[Path]:
        """Save every slide with the pinned aspect preset."""
        options: dict[str, Any] = {
            "aspect": self.preset,
            "title": self.title,
            "text": self.text,
            "background": self.background,
        }
        options.update(kwargs)
        return save_carousel(source, output_dir, **options)


ImagePostFormat = ImagePostRenderer
"""Alias for :class:`ImagePostRenderer`."""

CarouselFormat = CarouselRenderer
"""Alias for :class:`CarouselRenderer`."""

image_post = image_post_composition
"""Alias for :func:`image_post_composition`."""

carousel = carousel_compositions
"""Alias for :func:`carousel_compositions`."""


def _read_script(directory: Path, language: str) -> Script:
    names = (
        f"script.{language}.json",
        f"script_{language}.json",
        "script.json",
    )
    for name in names:
        path = directory / name
        if path.is_file():
            return Script.model_validate_json(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"no written script for language {language!r} in {directory}"
    )


def _visual_images(directory: Path, filename: str) -> list[str]:
    path = directory / filename
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    beats = payload.get("beats") if isinstance(payload, Mapping) else None
    if not isinstance(beats, list):
        return []
    images: list[str] = []
    for beat in beats:
        asset = beat.get("asset") if isinstance(beat, Mapping) else None
        images.append(_image_source(asset))
    return images


class ImagePostStage(Stage):
    """Render a written script as one image post.

    The stage is deterministic and free. It reads the script and, when it is
    present, the visual reuse manifest so generated images become slide
    backgrounds. Without images it still renders the text on the preset canvas.
    """

    name = "render"
    artifacts: tuple[str, ...] = (
        DEFAULT_IMAGE_POST_COMPOSITION_FILENAME,
        DEFAULT_IMAGE_POST_FILENAME,
    )

    def __init__(
        self,
        *,
        scripts: Mapping[str, Script] | None = None,
        source_language: str = "en",
        aspect: Any = None,
        carousel: bool | None = None,
        images: Any = None,
        text: str | None = None,
        title: str | None = None,
        background: str | None = None,
        render: bool = True,
        composition_filename: str = DEFAULT_IMAGE_POST_COMPOSITION_FILENAME,
        image_filename: str = DEFAULT_IMAGE_POST_FILENAME,
        manifest_filename: str = DEFAULT_CAROUSEL_MANIFEST_FILENAME,
        slide_dirname: str = DEFAULT_CAROUSEL_DIRNAME,
        slide_pattern: str = CAROUSEL_SLIDE_FILENAME_PATTERN,
        slide_composition_pattern: str = CAROUSEL_SLIDE_COMPOSITION_PATTERN,
        visual_manifest: str = DEFAULT_VISUAL_MANIFEST_FILENAME,
        options: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(options=options, **kwargs)
        self.scripts = dict(scripts or {})
        self.source_language = source_language or self.options.get(
            "source_language", "en"
        )
        self.aspect = (
            aspect
            if aspect is not None
            else self.options.get("aspect", self.options.get("aspect_preset"))
        )
        self.carousel = (
            carousel
            if carousel is not None
            else bool(self.options.get("carousel", False))
            or self.options.get("mode") in {"carousel", "slides"}
        )
        self.images = images
        self.text = text if text is not None else self.options.get("text")
        self.title = title if title is not None else self.options.get("title")
        self.background = (
            background if background is not None else self.options.get("background")
        )
        self.render = render and bool(self.options.get("render", True))
        self.composition_filename = composition_filename or self.options.get(
            "composition_filename", DEFAULT_IMAGE_POST_COMPOSITION_FILENAME
        )
        self.image_filename = image_filename or self.options.get(
            "image_filename", DEFAULT_IMAGE_POST_FILENAME
        )
        self.manifest_filename = manifest_filename or self.options.get(
            "manifest_filename", DEFAULT_CAROUSEL_MANIFEST_FILENAME
        )
        self.slide_dirname = slide_dirname or self.options.get(
            "slide_dirname", DEFAULT_CAROUSEL_DIRNAME
        )
        self.slide_pattern = slide_pattern or self.options.get(
            "slide_pattern", CAROUSEL_SLIDE_FILENAME_PATTERN
        )
        self.slide_composition_pattern = slide_composition_pattern or self.options.get(
            "slide_composition_pattern",
            CAROUSEL_SLIDE_COMPOSITION_PATTERN,
        )
        self.visual_manifest = visual_manifest or self.options.get(
            "visual_manifest", DEFAULT_VISUAL_MANIFEST_FILENAME
        )
        if not self.carousel:
            self.artifacts = (
                (self.composition_filename, self.image_filename)
                if self.render
                else (self.composition_filename,)
            )

    def _script(self, context: StageContext) -> Script:
        script = self.scripts.get(self.source_language)
        if script is not None:
            return script
        return _read_script(context.run_dir, self.source_language)

    def _images(self, context: StageContext) -> Any:
        if self.images is not None:
            return self.images
        return _visual_images(context.run_dir, self.visual_manifest)

    async def run(self, context: StageContext) -> StageResult:
        script = self._script(context)
        images = self._images(context)
        common = {
            "aspect": self.aspect,
            "title": self.title,
            "text": self.text,
            "images": images,
            "background": self.background,
        }
        if not self.carousel:
            composition = image_post_composition(script, **common)
            composition_path = context.run_dir / self.composition_filename
            composition_path.parent.mkdir(parents=True, exist_ok=True)
            composition_path.write_text(
                composition_to_yaml(composition),
                encoding="utf-8",
            )
            post_artifacts = [self.composition_filename]
            if self.render:
                image_path = context.run_dir / self.image_filename
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image = render_still(
                    composition,
                    0.0,
                    source_root=context.run_dir,
                )
                image.save(image_path)
                post_artifacts.append(self.image_filename)
            return StageResult(
                run_id=context.run.id if context.run is not None else "",
                stage=self.name,
                artifacts=post_artifacts,
                cost_usd=0.0,
            )

        slides = image_post_slides(script, images=images, text=self.text)
        compositions = carousel_compositions(
            slides,
            aspect=self.aspect,
            background=self.background,
        )
        slide_dir = context.run_dir / self.slide_dirname
        slide_dir.mkdir(parents=True, exist_ok=True)
        manifest_slides: list[dict[str, Any]] = []
        artifacts: list[str] = []
        for index, (slide, composition) in enumerate(
            zip(slides, compositions, strict=True),
            start=1,
        ):
            composition_path = slide_dir / self.slide_composition_pattern.format(
                index=index
            )
            composition_path.parent.mkdir(parents=True, exist_ok=True)
            composition_path.write_text(
                composition_to_yaml(composition),
                encoding="utf-8",
            )
            artifacts.append(composition_path.relative_to(context.run_dir).as_posix())
            image_relative = ""
            if self.render:
                image_path = slide_dir / self.slide_pattern.format(index=index)
                image = render_still(
                    composition,
                    0.0,
                    source_root=context.run_dir,
                )
                image.save(image_path)
                image_relative = image_path.relative_to(context.run_dir).as_posix()
                artifacts.append(image_relative)
            manifest_slides.append(
                {
                    "index": index,
                    "text": slide.text,
                    "title": slide.title,
                    "source": slide.source,
                    "alt_text": slide.alt_text,
                    "composition": composition_path.relative_to(
                        context.run_dir
                    ).as_posix(),
                    "image": image_relative,
                }
            )
        preset = resolve_aspect_preset(self.aspect)
        manifest = {
            "version": 1,
            "format": "carousel",
            "aspect": preset.name,
            "width": preset.width,
            "height": preset.height,
            "slides": manifest_slides,
        }
        manifest_path = context.run_dir / self.manifest_filename
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifacts.insert(0, self.manifest_filename)
        return StageResult(
            run_id=context.run.id if context.run is not None else "",
            stage=self.name,
            artifacts=artifacts,
            cost_usd=0.0,
        )


ImagePostRenderStage = ImagePostStage
"""Alias for :class:`ImagePostStage`."""


class CarouselStage(ImagePostStage):
    """Render a written script as a multi-slide carousel."""

    name = "render"

    def __init__(
        self,
        *,
        carousel: bool = True,
        options: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(carousel=carousel, options=options, **kwargs)


CarouselRenderStage = CarouselStage
"""Alias for :class:`CarouselStage`."""


def build_image_post_stage(**kwargs: Any) -> ImagePostStage:
    """Build an :class:`ImagePostStage` from keyword options."""
    return ImagePostStage(**kwargs)


def image_post_stage(**kwargs: Any) -> ImagePostStage:
    """Alias for :func:`build_image_post_stage`."""
    return ImagePostStage(**kwargs)


def build_carousel_stage(**kwargs: Any) -> CarouselStage:
    """Build a :class:`CarouselStage` from keyword options."""
    return CarouselStage(**kwargs)


def carousel_stage(**kwargs: Any) -> CarouselStage:
    """Alias for :func:`build_carousel_stage`."""
    return CarouselStage(**kwargs)

"""Generate cover frames and apply brand-kit branding to rendered media.

Cover frames are deterministic stills rendered from a composition or extracted
from an existing image or video. Candidate frames are scored by contrast and
edge detail so the best one can be selected without a paid vision call.

The branding block is a :class:`~aivia.core.config.character.BrandKitConfig`:
its ``watermark`` becomes a logo layer on a composition or an overlay on an
image or video, and its ``intro`` and ``outro`` clips are joined around the
main video. Every operation is local and offline; ffmpeg is invoked with an
argv list and both runners are injectable.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from PIL import (
    Image,
    ImageFilter,
    ImageOps,
    ImageStat,
    UnidentifiedImageError,
)

from aivia.core.config.character import BrandKitConfig
from aivia.core.config.media import (
    DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
    DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
)
from aivia.core.errors import ConfigError, ValidationError
from aivia.core.media_probe import run_ffprobe
from aivia.media.video_utils import concatenate_videos, extract_frame

from .errors import RenderError
from .ffmpeg_cmd import FFmpegCommand, FFmpegInput
from .render import PillowStillRenderer
from .schema import Anchor, Composition
from .urls import local_source_path

__all__ = [
    "COVER_BRANDING_VERSION",
    "DEFAULT_COVER_CANDIDATES",
    "DEFAULT_THUMBNAIL_WIDTH",
    "DEFAULT_WATERMARK_MARGIN",
    "DEFAULT_WATERMARK_OPACITY",
    "BrandedVideoResult",
    "BrandingError",
    "CoverBrandingError",
    "CoverFrame",
    "CoverRenderResult",
    "apply_branding_to_video",
    "apply_intro_outro",
    "apply_watermark",
    "best_cover_frame",
    "build_watermark_command",
    "cover_frame_times",
    "cover_frames",
    "export_cover",
    "extract_cover_frames",
    "frame_score",
    "generate_cover",
    "generate_cover_frames",
    "generate_thumbnail",
    "make_cover",
    "make_thumbnail",
    "render_cover",
    "render_thumbnail",
    "select_cover_frame",
    "watermark_composition",
    "watermark_image",
    "watermark_video",
]

COVER_BRANDING_VERSION = "1"
"""The version of the cover and branding pipeline."""

DEFAULT_COVER_CANDIDATES = 5
"""How many candidate frames are rendered when no time is given."""

DEFAULT_THUMBNAIL_WIDTH = 640
"""The default width of a generated thumbnail."""

DEFAULT_WATERMARK_MARGIN = 24.0
"""The default distance between a watermark and its canvas edge, in pixels."""

DEFAULT_WATERMARK_OPACITY = 0.8
"""The default opacity of a watermark."""

_FRAME_TONE = 0.6
_FRAME_EDGE = 0.4
_FRAME_EXPOSURE = 0.5
_MIN_LENGTH = 0.0


class BrandingError(RenderError):
    """Branding could not be applied to a composition or rendered media."""

    code = "branding"


class CoverBrandingError(BrandingError):
    """A cover frame could not be generated or branded."""

    code = "cover_branding"


@dataclass(frozen=True, slots=True, eq=False)
class CoverFrame:
    """One written cover or thumbnail frame and its selection score."""

    output: Path
    time_seconds: float
    score: float
    width: int
    height: int
    watermarked: bool = False

    @property
    def path(self) -> Path:
        """Alias for :attr:`output`."""
        return self.output

    @property
    def size(self) -> tuple[int, int]:
        """The pixel size as ``(width, height)``."""
        return self.width, self.height

    @property
    def bytes(self) -> int:
        """The size of the written file in bytes."""
        return self.output.stat().st_size

    def __fspath__(self) -> str:
        return str(self.output)

    def __str__(self) -> str:
        return str(self.output)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.output, name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (str, Path)):
            return self.output == Path(other)
        if not isinstance(other, CoverFrame):
            return NotImplemented
        return (
            self.output == other.output
            and self.time_seconds == other.time_seconds
            and self.score == other.score
            and self.width == other.width
            and self.height == other.height
            and self.watermarked == other.watermarked
        )

    def __hash__(self) -> int:
        return hash((self.output, self.time_seconds, self.score))


@dataclass(frozen=True, slots=True, eq=False)
class CoverRenderResult:
    """The selected cover frame, the candidates and the branding used."""

    output: Path
    selected: CoverFrame
    frames: tuple[CoverFrame, ...]
    branding: BrandKitConfig | None = None
    watermark_applied: bool = False

    @property
    def path(self) -> Path:
        """Alias for :attr:`output`."""
        return self.output

    @property
    def time_seconds(self) -> float:
        """The timestamp of the selected frame."""
        return self.selected.time_seconds

    @property
    def score(self) -> float:
        """The selection score of the selected frame."""
        return self.selected.score

    def __fspath__(self) -> str:
        return str(self.output)

    def __str__(self) -> str:
        return str(self.output)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.output, name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (str, Path)):
            return self.output == Path(other)
        if not isinstance(other, CoverRenderResult):
            return NotImplemented
        return self.output == other.output and self.selected == other.selected

    def __hash__(self) -> int:
        return hash((self.output, self.selected))


@dataclass(frozen=True, slots=True, eq=False)
class BrandedVideoResult:
    """A video with its branding applied and the clips that were joined."""

    output: Path
    intro: Path | None
    outro: Path | None
    watermarked: bool
    sources: tuple[Path, ...]

    @property
    def path(self) -> Path:
        """Alias for :attr:`output`."""
        return self.output

    def __fspath__(self) -> str:
        return str(self.output)

    def __str__(self) -> str:
        return str(self.output)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.output, name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (str, Path)):
            return self.output == Path(other)
        if not isinstance(other, BrandedVideoResult):
            return NotImplemented
        return self.output == other.output and self.sources == other.sources

    def __hash__(self) -> int:
        return hash((self.output, self.sources))


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CoverBrandingError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise CoverBrandingError(f"{name} must be finite")
    return number


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CoverBrandingError(f"{name} must be a positive integer")
    return value


def _opacity(value: Any) -> float:
    number = _finite(value, "opacity")
    if number < 0.0 or number > 1.0:
        raise CoverBrandingError("opacity must be between 0 and 1")
    return number


def _margin(value: Any) -> tuple[float, float]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise CoverBrandingError("margin must be a number or a pair of numbers")
        horizontal = _finite(value[0], "margin")
        vertical = _finite(value[1], "margin")
    else:
        horizontal = vertical = _finite(value, "margin")
    if horizontal < 0.0 or vertical < 0.0:
        raise CoverBrandingError("margin must not be negative")
    return horizontal, vertical


def _anchor(value: Any) -> Anchor:
    try:
        return Anchor(value)
    except (TypeError, ValueError) as exc:
        raise CoverBrandingError(f"unsupported anchor {value!r}") from exc


def _branding(
    value: BrandKitConfig | Mapping[str, Any] | Any | None,
) -> BrandKitConfig | None:
    if value is None:
        return None
    if isinstance(value, BrandKitConfig):
        return value
    if isinstance(value, Mapping):
        try:
            return BrandKitConfig.from_mapping(dict(value))
        except (ConfigError, ValidationError) as exc:
            raise CoverBrandingError(str(exc)) from exc
    nested = getattr(value, "brand", None)
    if nested is not None and nested is not value:
        return _branding(nested)
    if isinstance(getattr(value, "watermark", None), str):
        return cast(BrandKitConfig, value)
    raise CoverBrandingError("branding must be a brand kit or a mapping")


def _composition(
    value: Composition | Mapping[str, Any],
) -> Composition:
    if isinstance(value, Composition):
        return value
    try:
        return Composition.model_validate(dict(value))
    except ValidationError as exc:
        raise CoverBrandingError(str(exc)) from exc


def _resolve_branding(
    composition: Composition | None,
    branding: BrandKitConfig | Mapping[str, Any] | Any | None,
) -> BrandKitConfig | None:
    resolved = _branding(branding)
    if resolved is not None:
        return resolved
    if composition is not None:
        return _branding(getattr(composition, "branding", None))
    return None


def _watermark_source(
    watermark: str | Path | Image.Image | None,
    branding: BrandKitConfig | None,
) -> str | Path | Image.Image | None:
    if watermark is not None:
        if not isinstance(watermark, (str, Path, Image.Image)):
            raise CoverBrandingError("watermark must be a path or a PIL image")
        if isinstance(watermark, Image.Image):
            return watermark
        return local_source_path(watermark)
    if branding is not None and branding.watermark:
        return local_source_path(branding.watermark)
    return None


def _clip(value: str | Path | None, configured: str) -> Path | None:
    if value is not None:
        text = str(value).strip()
        return Path(text) if text else None
    text = configured.strip()
    return Path(text) if text else None


def _open_image(source: str | Path | Image.Image, name: str) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.copy().convert("RGBA")
    path = local_source_path(source)
    try:
        with Image.open(path) as opened:
            opened.load()
            return opened.convert("RGBA")
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise CoverBrandingError(f"{name} could not be read: {path}") from exc


def _anchor_position(
    canvas_size: tuple[int, int],
    image_size: tuple[int, int],
    anchor: Anchor,
    margin: tuple[float, float],
) -> tuple[int, int]:
    canvas_width, canvas_height = canvas_size
    width, height = image_size
    margin_x, margin_y = margin
    left = margin_x
    top = margin_y
    right = canvas_width - width - margin_x
    bottom = canvas_height - height - margin_y
    center_x = (canvas_width - width) / 2.0
    center_y = (canvas_height - height) / 2.0
    positions = {
        Anchor.TOP_LEFT: (left, top),
        Anchor.TOP_CENTER: (center_x, top),
        Anchor.TOP_RIGHT: (right, top),
        Anchor.MIDDLE_LEFT: (left, center_y),
        Anchor.CENTER: (center_x, center_y),
        Anchor.MIDDLE_RIGHT: (right, center_y),
        Anchor.BOTTOM_LEFT: (left, bottom),
        Anchor.BOTTOM_CENTER: (center_x, bottom),
        Anchor.BOTTOM_RIGHT: (right, bottom),
    }
    x, y = positions[anchor]
    return round(x), round(y)


def _with_opacity(image: Image.Image, opacity: float) -> Image.Image:
    if opacity >= 1.0:
        return image
    result = image.copy()
    alpha = result.getchannel("A").point(lambda value: round(value * opacity))
    result.putalpha(alpha)
    return result


def _target_size(
    natural: tuple[int, int],
    *,
    scale: float,
    width: int | None,
    height: int | None,
) -> tuple[int, int]:
    natural_width, natural_height = natural
    if width is not None and height is not None:
        return _positive_int(width, "width"), _positive_int(height, "height")
    if width is not None:
        target_width = _positive_int(width, "width")
        return target_width, max(
            1, round(natural_height * target_width / natural_width)
        )
    if height is not None:
        target_height = _positive_int(height, "height")
        return max(
            1, round(natural_width * target_height / natural_height)
        ), target_height
    if scale == 1.0:
        return natural_width, natural_height
    return (
        max(1, round(natural_width * scale)),
        max(1, round(natural_height * scale)),
    )


def _frame_score(image: Image.Image) -> float:
    """Score one frame by contrast, edge detail and exposure, from 0 to 1."""
    gray = image.convert("L")
    statistics = ImageStat.Stat(gray)
    contrast = statistics.stddev[0] / 128.0
    brightness = statistics.mean[0] / 255.0
    edges = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).mean[0] / 255.0
    score = (
        _FRAME_TONE * min(contrast, 1.0)
        + _FRAME_EDGE * min(edges, 1.0)
        - _FRAME_EXPOSURE * abs(brightness - 0.5)
    )
    return max(0.0, min(1.0, score))


def frame_score(image: Image.Image) -> float:
    """Return a deterministic selection score for *image*."""
    if not isinstance(image, Image.Image):
        raise CoverBrandingError("frame_score expects a PIL image")
    return _frame_score(image)


def _size(value: Any, name: str = "size") -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = value.lower().replace("x", ",").replace("*", ",").split(",")
        if len(parts) != 2:
            raise CoverBrandingError(f"{name} must be WIDTHxHEIGHT or a pair")
        return _positive_int(int(parts[0]), name), _positive_int(int(parts[1]), name)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return _positive_int(value[0], name), _positive_int(value[1], name)
    raise CoverBrandingError(f"{name} must be WIDTHxHEIGHT or a pair")


def _fit_size(image: Image.Image, size: tuple[int, int] | None) -> Image.Image:
    if size is None:
        return image
    return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)


def _save_image(image: Image.Image, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix or ".png"
    if suffix.lower() in {".jpg", ".jpeg"} and image.mode in {"RGBA", "LA"}:
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.getchannel("A"))
        image = background
    descriptor, name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.stem}-",
        suffix=suffix,
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        image.save(temporary)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise CoverBrandingError("the cover frame could not be written")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def apply_watermark(
    target: Image.Image | str | Path | Composition | Mapping[str, Any],
    watermark: str | Path | Image.Image | None = None,
    *,
    branding: BrandKitConfig | Mapping[str, Any] | Any | None = None,
    anchor: Anchor | str = Anchor.BOTTOM_RIGHT,
    margin: float | tuple[float, float] = DEFAULT_WATERMARK_MARGIN,
    opacity: float = DEFAULT_WATERMARK_OPACITY,
    scale: float = 1.0,
    width: int | None = None,
    height: int | None = None,
) -> Image.Image | Composition:
    """Apply a watermark to an image or add it as a logo layer to a composition."""
    if isinstance(target, Composition) or (
        isinstance(target, Mapping) and "canvas" in target
    ):
        if isinstance(watermark, Image.Image):
            raise CoverBrandingError("a composition watermark must be a local path")
        return watermark_composition(
            target,
            watermark,
            branding=branding,
            anchor=anchor,
            margin=margin,
            opacity=opacity,
            scale=scale,
            width=width,
            height=height,
        )
    if not isinstance(target, (str, Path, Image.Image)):
        raise CoverBrandingError("target must be an image, path or composition")
    image = _open_image(target, "image")
    brand = _branding(branding)
    source = _watermark_source(watermark, brand)
    if source is None:
        return image
    mark = _open_image(source, "watermark")
    factor = _finite(scale, "scale")
    if factor <= 0.0:
        raise CoverBrandingError("scale must be positive")
    target_size = _target_size(
        mark.size,
        scale=factor,
        width=width,
        height=height,
    )
    if target_size != mark.size:
        mark = mark.resize(target_size, Image.Resampling.LANCZOS)
    mark = _with_opacity(mark, _opacity(opacity))
    x, y = _anchor_position(image.size, mark.size, _anchor(anchor), _margin(margin))
    result = image.copy()
    result.alpha_composite(mark, dest=(max(0, x), max(0, y)))
    return result


def watermark_image(
    image: Image.Image | str | Path,
    watermark: str | Path | Image.Image | None = None,
    **options: Any,
) -> Image.Image:
    """Apply a watermark to an image and always return an image."""
    result = apply_watermark(image, watermark, **options)
    if not isinstance(result, Image.Image):
        raise CoverBrandingError("watermark_image expects an image")
    return result


def _watermark_layer_id(composition: Composition) -> str:
    used = {layer.id for layer in composition.layers}
    if "brand-watermark" not in used:
        return "brand-watermark"
    index = 2
    while f"brand-watermark-{index}" in used:
        index += 1
    return f"brand-watermark-{index}"


def watermark_composition(
    composition: Composition | Mapping[str, Any],
    watermark: str | Path | None = None,
    *,
    branding: BrandKitConfig | Mapping[str, Any] | Any | None = None,
    anchor: Anchor | str = Anchor.BOTTOM_RIGHT,
    margin: float | tuple[float, float] = DEFAULT_WATERMARK_MARGIN,
    opacity: float = DEFAULT_WATERMARK_OPACITY,
    scale: float = 1.0,
    width: int | None = None,
    height: int | None = None,
    z_index: int | None = None,
) -> Composition:
    """Return *composition* with the branding watermark as a logo layer."""
    value = _composition(composition)
    brand = _resolve_branding(value, branding)
    source = _watermark_source(watermark, brand)
    if source is None:
        return value
    if isinstance(source, Image.Image):
        raise CoverBrandingError("a composition watermark must be a local path")
    factor = _finite(scale, "scale")
    if factor <= 0.0:
        raise CoverBrandingError("scale must be positive")
    margin_value = _margin(margin)
    layer: dict[str, Any] = {
        "id": _watermark_layer_id(value),
        "type": "logo",
        "source": str(source),
        "anchor": _anchor(anchor).value,
        "margin": margin_value,
        "opacity": _opacity(opacity),
        "scale": factor,
        "start_seconds": 0.0,
        "end_seconds": value.duration_seconds,
        "z_index": (
            max((item.z_index for item in value.layers), default=0) + 1
            if z_index is None
            else z_index
        ),
    }
    if width is not None:
        layer["width"] = _positive_int(width, "width")
    if height is not None:
        layer["height"] = _positive_int(height, "height")
    data = value.model_dump(by_alias=False, exclude_none=True)
    data["layers"] = [*data.get("layers", []), layer]
    try:
        return Composition.model_validate(data)
    except ValidationError as exc:
        raise CoverBrandingError(str(exc)) from exc


def cover_frame_times(
    composition: Composition | Mapping[str, Any],
    count: int = DEFAULT_COVER_CANDIDATES,
    *,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> list[float]:
    """Return evenly spaced candidate timestamps inside a composition."""
    value = _composition(composition)
    number = _positive_int(count, "count")
    start = 0.0 if start_seconds is None else _finite(start_seconds, "start_seconds")
    end = (
        value.duration_seconds
        if end_seconds is None
        else _finite(end_seconds, "end_seconds")
    )
    if start < 0.0 or start >= value.duration_seconds:
        raise CoverBrandingError(
            "start_seconds must be between 0 and the composition duration"
        )
    if end <= start or end > value.duration_seconds:
        raise CoverBrandingError(
            "end_seconds must be greater than start and at most the duration"
        )
    if number == 1:
        return [round((start + end) / 2.0, 6)]
    step = (end - start) / number
    return [round(start + step * (index + 0.5), 6) for index in range(number)]


def _source_kind(source: Any) -> str:
    if isinstance(source, Composition) or (
        isinstance(source, Mapping) and "canvas" in source
    ):
        return "composition"
    if isinstance(source, (str, Path)):
        suffix = Path(source).suffix.lower()
        if suffix in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".gif"}:
            return "video"
        return "image"
    raise CoverBrandingError("source must be a composition, image or video path")


def _source_duration(
    source: Any,
    *,
    ffprobe_executable: str,
    ffprobe_runner: Any,
    timeout_seconds: float,
) -> float:
    kind = _source_kind(source)
    if kind == "composition":
        return _composition(source).duration_seconds
    if kind == "image":
        return 1.0
    try:
        info = run_ffprobe(
            Path(source),
            executable=ffprobe_executable,
            timeout_seconds=timeout_seconds,
            runner=ffprobe_runner,
        )
    except ValidationError as exc:
        raise CoverBrandingError(str(exc)) from exc
    length = info.duration_seconds
    if length is None or length <= _MIN_LENGTH:
        raise CoverBrandingError("could not determine the video duration")
    return length


def _render_source_frame(
    source: Any,
    time_seconds: float,
    directory: Path,
    index: int,
    *,
    renderer: PillowStillRenderer | None,
    ffmpeg_executable: str,
    ffmpeg_runner: Any,
    timeout_seconds: float,
) -> Image.Image:
    kind = _source_kind(source)
    if kind == "composition":
        engine = renderer or PillowStillRenderer()
        return engine.render_still(_composition(source), time_seconds).convert("RGBA")
    if kind == "image":
        return _open_image(source, "image")
    destination = directory / f"frame-{index}.png"
    try:
        extract_frame(
            Path(source),
            destination,
            timestamp_seconds=time_seconds,
            ffmpeg_executable=ffmpeg_executable,
            ffmpeg_runner=ffmpeg_runner,
            timeout_seconds=timeout_seconds,
        )
    except ValidationError as exc:
        raise CoverBrandingError(str(exc)) from exc
    return _open_image(destination, "video frame")


def _candidate_times(
    source: Any,
    *,
    times: Sequence[float] | None,
    count: int,
    duration_seconds: float | None,
    ffprobe_executable: str,
    ffprobe_runner: Any,
    timeout_seconds: float,
) -> list[float]:
    if times is not None:
        values = [_finite(item, "time") for item in times]
        if not values:
            raise CoverBrandingError("times must not be empty")
        return values
    if _source_kind(source) == "composition":
        return cover_frame_times(source, count)
    if _source_kind(source) == "image":
        return [0.0]
    duration = (
        _source_duration(
            source,
            ffprobe_executable=ffprobe_executable,
            ffprobe_runner=ffprobe_runner,
            timeout_seconds=timeout_seconds,
        )
        if duration_seconds is None
        else _finite(duration_seconds, "duration_seconds")
    )
    number = _positive_int(count, "count")
    if number == 1:
        return [round(duration / 2.0, 6)]
    step = duration / number
    return [round(step * (index + 0.5), 6) for index in range(number)]


def generate_cover_frames(
    source: Composition | Mapping[str, Any] | str | Path,
    output_dir: str | Path,
    *,
    times: Sequence[float] | None = None,
    count: int = DEFAULT_COVER_CANDIDATES,
    duration_seconds: float | None = None,
    size: tuple[int, int] | str | None = None,
    format: str = "png",
    prefix: str = "cover",
    branding: BrandKitConfig | Mapping[str, Any] | Any | None = None,
    watermark: str | Path | Image.Image | None = None,
    watermark_enabled: bool = True,
    anchor: Anchor | str = Anchor.BOTTOM_RIGHT,
    margin: float | tuple[float, float] = DEFAULT_WATERMARK_MARGIN,
    opacity: float = DEFAULT_WATERMARK_OPACITY,
    scale: float = 1.0,
    renderer: PillowStillRenderer | None = None,
    ffmpeg_executable: str = "ffmpeg",
    ffmpeg_runner: Any = None,
    ffprobe_executable: str = "ffprobe",
    ffprobe_runner: Any = None,
    timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
    ffprobe_timeout_seconds: float = DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
) -> list[CoverFrame]:
    """Write candidate cover frames and return them in timestamp order."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    suffix = format.strip().lower().lstrip(".") or "png"
    if not suffix.isalnum():
        raise CoverBrandingError(f"unsupported frame format {format!r}")
    target_size = _size(size)
    brand = _resolve_branding(
        _composition(cast(Composition | Mapping[str, Any], source))
        if _source_kind(source) == "composition"
        else None,
        branding,
    )
    mark = _watermark_source(watermark, brand) if watermark_enabled else None
    timestamps = _candidate_times(
        source,
        times=times,
        count=count,
        duration_seconds=duration_seconds,
        ffprobe_executable=ffprobe_executable,
        ffprobe_runner=ffprobe_runner,
        timeout_seconds=ffprobe_timeout_seconds,
    )
    frames: list[CoverFrame] = []
    with tempfile.TemporaryDirectory(
        prefix="aivia-cover-",
        dir=directory,
    ) as temporary:
        working = Path(temporary)
        for index, timestamp in enumerate(timestamps):
            image = _render_source_frame(
                source,
                timestamp,
                working,
                index,
                renderer=renderer,
                ffmpeg_executable=ffmpeg_executable,
                ffmpeg_runner=ffmpeg_runner,
                timeout_seconds=timeout_seconds,
            )
            score = _frame_score(image)
            if mark is not None:
                image = watermark_image(
                    image,
                    mark,
                    anchor=anchor,
                    margin=margin,
                    opacity=opacity,
                    scale=scale,
                )
            image = _fit_size(image, target_size)
            output = directory / f"{prefix}-{index:03d}.{suffix}"
            _save_image(image, output)
            frames.append(
                CoverFrame(
                    output=output,
                    time_seconds=timestamp,
                    score=score,
                    width=image.width,
                    height=image.height,
                    watermarked=mark is not None,
                )
            )
    return frames


def select_cover_frame(frames: Sequence[CoverFrame]) -> CoverFrame:
    """Return the highest scoring frame, preferring earlier ties."""
    if not frames:
        raise CoverBrandingError("at least one cover frame is required")
    return max(frames, key=lambda frame: (frame.score, -frame.time_seconds))


def best_cover_frame(frames: Sequence[CoverFrame]) -> CoverFrame:
    """Alias for :func:`select_cover_frame`."""
    return select_cover_frame(frames)


def render_cover(
    source: Composition | Mapping[str, Any] | str | Path,
    output: str | Path,
    *,
    time_seconds: float | None = None,
    count: int = DEFAULT_COVER_CANDIDATES,
    duration_seconds: float | None = None,
    size: tuple[int, int] | str | None = None,
    branding: BrandKitConfig | Mapping[str, Any] | Any | None = None,
    watermark: str | Path | Image.Image | None = None,
    watermark_enabled: bool = True,
    anchor: Anchor | str = Anchor.BOTTOM_RIGHT,
    margin: float | tuple[float, float] = DEFAULT_WATERMARK_MARGIN,
    opacity: float = DEFAULT_WATERMARK_OPACITY,
    scale: float = 1.0,
    renderer: PillowStillRenderer | None = None,
    ffmpeg_executable: str = "ffmpeg",
    ffmpeg_runner: Any = None,
    ffprobe_executable: str = "ffprobe",
    ffprobe_runner: Any = None,
    timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
    ffprobe_timeout_seconds: float = DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
) -> CoverRenderResult:
    """Render or extract candidate frames and write the best one to *output*."""
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    brand = _resolve_branding(
        _composition(cast(Composition | Mapping[str, Any], source))
        if _source_kind(source) == "composition"
        else None,
        branding,
    )
    mark = _watermark_source(watermark, brand) if watermark_enabled else None
    timestamps = (
        [_finite(time_seconds, "time_seconds")]
        if time_seconds is not None
        else _candidate_times(
            source,
            times=None,
            count=count,
            duration_seconds=duration_seconds,
            ffprobe_executable=ffprobe_executable,
            ffprobe_runner=ffprobe_runner,
            timeout_seconds=ffprobe_timeout_seconds,
        )
    )
    target_size = _size(size)
    candidates: list[CoverFrame] = []
    best_image: Image.Image | None = None
    best_time = timestamps[0]
    best_score = -math.inf
    with tempfile.TemporaryDirectory(
        prefix="aivia-cover-",
        dir=destination.parent,
    ) as temporary:
        working = Path(temporary)
        for index, timestamp in enumerate(timestamps):
            image = _render_source_frame(
                source,
                timestamp,
                working,
                index,
                renderer=renderer,
                ffmpeg_executable=ffmpeg_executable,
                ffmpeg_runner=ffmpeg_runner,
                timeout_seconds=timeout_seconds,
            )
            score = _frame_score(image)
            if score > best_score:
                best_score = score
                best_time = timestamp
                best_image = image
            candidates.append(
                CoverFrame(
                    output=destination,
                    time_seconds=timestamp,
                    score=score,
                    width=image.width,
                    height=image.height,
                )
            )
        if best_image is None:
            raise CoverBrandingError("no cover frame could be rendered")
        if mark is not None:
            best_image = watermark_image(
                best_image,
                mark,
                anchor=anchor,
                margin=margin,
                opacity=opacity,
                scale=scale,
            )
        best_image = _fit_size(best_image, target_size)
        _save_image(best_image, destination)
    selected = CoverFrame(
        output=destination,
        time_seconds=best_time,
        score=best_score,
        width=best_image.width,
        height=best_image.height,
        watermarked=mark is not None,
    )
    return CoverRenderResult(
        output=destination,
        selected=selected,
        frames=tuple(candidates),
        branding=brand,
        watermark_applied=mark is not None,
    )


def _thumbnail_size(
    source: Any,
    size: tuple[int, int] | str | None,
    renderer: PillowStillRenderer | None,
) -> tuple[int, int] | None:
    if size is not None:
        return _size(size)
    if _source_kind(source) != "composition":
        return None
    value = _composition(source)
    width = min(DEFAULT_THUMBNAIL_WIDTH, value.width)
    height = max(1, round(value.height * width / value.width))
    return width, height


def render_thumbnail(
    source: Composition | Mapping[str, Any] | str | Path,
    output: str | Path,
    *,
    size: tuple[int, int] | str | None = None,
    **options: Any,
) -> CoverRenderResult:
    """Render a smaller cover frame using the thumbnail width by default."""
    renderer = options.get("renderer")
    return render_cover(
        source,
        output,
        size=_thumbnail_size(source, size, renderer),
        **options,
    )


def _overlay_expressions(
    anchor: Anchor,
    margin: tuple[float, float],
) -> tuple[str, str]:
    margin_x, margin_y = margin
    horizontal = {
        Anchor.TOP_LEFT: f"{margin_x:g}",
        Anchor.TOP_CENTER: "(main_w-overlay_w)/2",
        Anchor.TOP_RIGHT: f"main_w-overlay_w-{margin_x:g}",
        Anchor.MIDDLE_LEFT: f"{margin_x:g}",
        Anchor.CENTER: "(main_w-overlay_w)/2",
        Anchor.MIDDLE_RIGHT: f"main_w-overlay_w-{margin_x:g}",
        Anchor.BOTTOM_LEFT: f"{margin_x:g}",
        Anchor.BOTTOM_CENTER: "(main_w-overlay_w)/2",
        Anchor.BOTTOM_RIGHT: f"main_w-overlay_w-{margin_x:g}",
    }
    vertical = {
        Anchor.TOP_LEFT: f"{margin_y:g}",
        Anchor.TOP_CENTER: f"{margin_y:g}",
        Anchor.TOP_RIGHT: f"{margin_y:g}",
        Anchor.MIDDLE_LEFT: "(main_h-overlay_h)/2",
        Anchor.CENTER: "(main_h-overlay_h)/2",
        Anchor.MIDDLE_RIGHT: "(main_h-overlay_h)/2",
        Anchor.BOTTOM_LEFT: f"main_h-overlay_h-{margin_y:g}",
        Anchor.BOTTOM_CENTER: f"main_h-overlay_h-{margin_y:g}",
        Anchor.BOTTOM_RIGHT: f"main_h-overlay_h-{margin_y:g}",
    }
    return horizontal[anchor], vertical[anchor]


def build_watermark_command(
    video: str | Path,
    watermark: str | Path,
    output: str | Path,
    *,
    anchor: Anchor | str = Anchor.BOTTOM_RIGHT,
    margin: float | tuple[float, float] = DEFAULT_WATERMARK_MARGIN,
    opacity: float = DEFAULT_WATERMARK_OPACITY,
    scale: float = 1.0,
    width: int | None = None,
    height: int | None = None,
    executable: str = "ffmpeg",
    video_codec: str = "libx264",
    crf: int = 18,
    preset: str = "medium",
    pixel_format: str = "yuv420p",
) -> FFmpegCommand:
    """Build the pure ffmpeg command that overlays a watermark on a video."""
    factor = _finite(scale, "scale")
    if factor <= 0.0:
        raise CoverBrandingError("scale must be positive")
    filters = ["format=rgba"]
    if width is not None and height is not None:
        filters.append(
            f"scale={_positive_int(width, 'width')}:{_positive_int(height, 'height')}"
        )
    elif width is not None:
        filters.append(f"scale={_positive_int(width, 'width')}:-1")
    elif height is not None:
        filters.append(f"scale=-1:{_positive_int(height, 'height')}")
    elif factor != 1.0:
        filters.append(f"scale=iw*{factor:g}:ih*{factor:g}")
    alpha = _opacity(opacity)
    if alpha < 1.0:
        filters.append(f"colorchannelmixer=aa={alpha:g}")
    overlay_x, overlay_y = _overlay_expressions(_anchor(anchor), _margin(margin))
    graph = (
        f"[1:v]{','.join(filters)}[wm];"
        f"[0:v][wm]overlay={overlay_x}:{overlay_y}:format=auto[v]"
    )
    argv = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-i",
        str(watermark),
        "-filter_complex",
        graph,
        "-map",
        "[v]",
        "-map",
        "0:a?",
        "-c:v",
        video_codec,
        "-preset",
        preset,
        "-crf",
        str(_positive_int(crf, "crf")),
        "-pix_fmt",
        pixel_format,
        "-c:a",
        "aac",
        str(output),
    ]
    return FFmpegCommand(
        argv,
        filtergraph=graph,
        output=str(output),
        inputs=(
            FFmpegInput(source=str(video), layer_id="video"),
            FFmpegInput(source=str(watermark), layer_id="watermark"),
        ),
    )


def _run_ffmpeg(
    argv: Sequence[str],
    *,
    runner: Any,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    if runner is not None:
        result = runner(argv)
        if isinstance(result, subprocess.CompletedProcess):
            return result
        return subprocess.CompletedProcess(list(argv), 0, stdout=result, stderr="")
    try:
        return subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise CoverBrandingError(
            f"ffmpeg is not available: {argv[0]}",
            details={"executable": argv[0]},
        ) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise CoverBrandingError(
            "ffmpeg could not apply the branding",
            details={"executable": argv[0], "error": type(exc).__name__},
        ) from exc


def _run_to_file(
    command: Sequence[str],
    output: Path,
    *,
    runner: Any,
    timeout_seconds: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.stem}-",
        suffix=output.suffix or ".tmp",
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        argv = list(command)
        argv[-1] = str(temporary)
        completed = _run_ffmpeg(
            argv,
            runner=runner,
            timeout_seconds=timeout_seconds,
        )
        if completed.returncode != 0:
            raise CoverBrandingError(
                "ffmpeg could not apply the branding",
                details={
                    "returncode": completed.returncode,
                    "stderr": str(completed.stderr or "")[:500],
                },
            )
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise CoverBrandingError("ffmpeg produced no output")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_video(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == output.resolve():
        return
    descriptor, name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.stem}-",
        suffix=output.suffix or source.suffix or ".tmp",
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_bytes(source.read_bytes())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def apply_intro_outro(
    video: str | Path,
    output: str | Path,
    *,
    intro: str | Path | None = None,
    outro: str | Path | None = None,
    branding: BrandKitConfig | Mapping[str, Any] | Any | None = None,
    crossfade_seconds: float = 0.0,
    ffmpeg_executable: str = "ffmpeg",
    ffmpeg_runner: Any = None,
    ffprobe_executable: str = "ffprobe",
    ffprobe_runner: Any = None,
    timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
) -> BrandedVideoResult:
    """Join the branding intro and outro around *video*."""
    source = Path(video)
    destination = Path(output)
    brand = _branding(branding)
    intro_path = _clip(intro, brand.intro if brand is not None else "")
    outro_path = _clip(outro, brand.outro if brand is not None else "")
    clips = [clip for clip in (intro_path, source, outro_path) if clip is not None]
    if len(clips) == 1:
        _copy_video(source, destination)
    else:
        try:
            concatenate_videos(
                clips,
                destination,
                crossfade_seconds=crossfade_seconds,
                normalize=True,
                reference=1 if intro_path is not None else 0,
                ffmpeg_executable=ffmpeg_executable,
                ffmpeg_runner=ffmpeg_runner,
                ffprobe_executable=ffprobe_executable,
                ffprobe_runner=ffprobe_runner,
                timeout_seconds=timeout_seconds,
            )
        except ValidationError as exc:
            raise CoverBrandingError(str(exc)) from exc
    return BrandedVideoResult(
        output=destination,
        intro=intro_path,
        outro=outro_path,
        watermarked=False,
        sources=tuple(clips),
    )


def watermark_video(
    video: str | Path,
    output: str | Path,
    watermark: str | Path | Image.Image | None = None,
    *,
    branding: BrandKitConfig | Mapping[str, Any] | Any | None = None,
    anchor: Anchor | str = Anchor.BOTTOM_RIGHT,
    margin: float | tuple[float, float] = DEFAULT_WATERMARK_MARGIN,
    opacity: float = DEFAULT_WATERMARK_OPACITY,
    scale: float = 1.0,
    width: int | None = None,
    height: int | None = None,
    ffmpeg_executable: str = "ffmpeg",
    ffmpeg_runner: Any = None,
    timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
) -> BrandedVideoResult:
    """Overlay the branding watermark on a video."""
    source = Path(video)
    destination = Path(output)
    brand = _branding(branding)
    mark = _watermark_source(watermark, brand)
    if mark is None:
        _copy_video(source, destination)
        return BrandedVideoResult(
            output=destination,
            intro=None,
            outro=None,
            watermarked=False,
            sources=(source,),
        )
    if isinstance(mark, Image.Image):
        with tempfile.TemporaryDirectory(prefix="aivia-watermark-") as directory:
            temporary = Path(directory) / "watermark.png"
            mark.save(temporary)
            command = build_watermark_command(
                source,
                temporary,
                destination,
                anchor=anchor,
                margin=margin,
                opacity=opacity,
                scale=scale,
                width=width,
                height=height,
                executable=ffmpeg_executable,
            )
            _run_to_file(
                command,
                destination,
                runner=ffmpeg_runner,
                timeout_seconds=timeout_seconds,
            )
    else:
        command = build_watermark_command(
            source,
            Path(mark),
            destination,
            anchor=anchor,
            margin=margin,
            opacity=opacity,
            scale=scale,
            width=width,
            height=height,
            executable=ffmpeg_executable,
        )
        _run_to_file(
            command,
            destination,
            runner=ffmpeg_runner,
            timeout_seconds=timeout_seconds,
        )
    return BrandedVideoResult(
        output=destination,
        intro=None,
        outro=None,
        watermarked=True,
        sources=(source,),
    )


def apply_branding_to_video(
    video: str | Path,
    output: str | Path,
    *,
    branding: BrandKitConfig | Mapping[str, Any] | Any | None = None,
    intro: str | Path | None = None,
    outro: str | Path | None = None,
    watermark: str | Path | None = None,
    watermark_enabled: bool = True,
    anchor: Anchor | str = Anchor.BOTTOM_RIGHT,
    margin: float | tuple[float, float] = DEFAULT_WATERMARK_MARGIN,
    opacity: float = DEFAULT_WATERMARK_OPACITY,
    scale: float = 1.0,
    crossfade_seconds: float = 0.0,
    ffmpeg_executable: str = "ffmpeg",
    ffmpeg_runner: Any = None,
    ffprobe_executable: str = "ffprobe",
    ffprobe_runner: Any = None,
    timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
) -> BrandedVideoResult:
    """Apply the branding intro, outro and watermark to a video in order."""
    source = Path(video)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    brand = _branding(branding)
    intro_path = _clip(intro, brand.intro if brand is not None else "")
    outro_path = _clip(outro, brand.outro if brand is not None else "")
    mark = _watermark_source(watermark, brand) if watermark_enabled else None
    clips = [clip for clip in (intro_path, source, outro_path) if clip is not None]
    if len(clips) == 1 and mark is None:
        _copy_video(source, destination)
        return BrandedVideoResult(
            output=destination,
            intro=intro_path,
            outro=outro_path,
            watermarked=False,
            sources=tuple(clips),
        )
    if mark is None:
        return apply_intro_outro(
            source,
            destination,
            intro=intro_path,
            outro=outro_path,
            crossfade_seconds=crossfade_seconds,
            ffmpeg_executable=ffmpeg_executable,
            ffmpeg_runner=ffmpeg_runner,
            ffprobe_executable=ffprobe_executable,
            ffprobe_runner=ffprobe_runner,
            timeout_seconds=timeout_seconds,
        )
    if len(clips) == 1:
        return watermark_video(
            source,
            destination,
            mark,
            anchor=anchor,
            margin=margin,
            opacity=opacity,
            scale=scale,
            ffmpeg_executable=ffmpeg_executable,
            ffmpeg_runner=ffmpeg_runner,
            timeout_seconds=timeout_seconds,
        )
    with tempfile.TemporaryDirectory(
        prefix="aivia-branding-",
        dir=destination.parent,
    ) as directory:
        joined = Path(directory) / f"joined{destination.suffix or '.mp4'}"
        apply_intro_outro(
            source,
            joined,
            intro=intro_path,
            outro=outro_path,
            crossfade_seconds=crossfade_seconds,
            ffmpeg_executable=ffmpeg_executable,
            ffmpeg_runner=ffmpeg_runner,
            ffprobe_executable=ffprobe_executable,
            ffprobe_runner=ffprobe_runner,
            timeout_seconds=timeout_seconds,
        )
        branded = watermark_video(
            joined,
            destination,
            mark,
            anchor=anchor,
            margin=margin,
            opacity=opacity,
            scale=scale,
            ffmpeg_executable=ffmpeg_executable,
            ffmpeg_runner=ffmpeg_runner,
            timeout_seconds=timeout_seconds,
        )
        return BrandedVideoResult(
            output=branded.output,
            intro=intro_path,
            outro=outro_path,
            watermarked=branded.watermarked,
            sources=tuple(clips),
        )


cover_frames = generate_cover_frames
extract_cover_frames = generate_cover_frames
generate_cover = render_cover
make_cover = render_cover
export_cover = render_cover
generate_thumbnail = render_thumbnail
make_thumbnail = render_thumbnail

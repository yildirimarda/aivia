"""Deterministic image transforms that never touch the source file.

Generated and user-supplied images arrive in arbitrary sizes, aspect ratios,
formats and metadata. Composition and delivery need a small, predictable set of
operations before an image is stored or drawn:

* resize to an exact size;
* crop to an aspect ratio and then resize to the requested size;
* pad to a target size with a blurred, cover-cropped copy of the same image;
* convert to a supported output format, flattening transparency when the
  format cannot carry it;
* remove EXIF, ICC, text and other metadata before the image is written.

Every operation returns a new :class:`PIL.Image.Image`; the input image is
never modified. :func:`transform_image` writes through a temporary file and
refuses to use the source path as its destination, so an in-place edit cannot
destroy the original. Metadata is always stripped on save, matching the
project's privacy invariant.
"""

from __future__ import annotations

import os
import tempfile
from enum import StrEnum
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

from aivia.core.config.media import DEFAULT_MEDIA_MAX_BYTES as DEFAULT_MAX_IMAGE_BYTES
from aivia.core.config.media import (
    DEFAULT_MEDIA_MAX_PIXELS as DEFAULT_MAX_IMAGE_PIXELS,
)
from aivia.core.errors import ValidationError
from aivia.core.media_probe import image_dimensions

__all__ = [
    "DEFAULT_BLUR_RADIUS",
    "DEFAULT_MAX_IMAGE_BYTES",
    "DEFAULT_MAX_IMAGE_PIXELS",
    "ImageFit",
    "ImageFormat",
    "ImageSource",
    "aspect_crop",
    "blurred_fill_pad",
    "convert_image",
    "open_image",
    "resize_image",
    "save_image",
    "strip_metadata",
    "transform_image",
]

type ImageSource = str | Path | bytes
"""An image path or the raw bytes of an encoded image."""

DEFAULT_BLUR_RADIUS = 24.0
"""Default Gaussian blur radius, in pixels, for padded backgrounds."""

_HEADER_BYTES = 64 * 1024
"""Bytes read from a file to learn an image's declared dimensions."""


def _validate_limit(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{label} must be an integer")
    if value < 1:
        raise ValidationError(f"{label} must be a positive integer")
    return value


def _check_byte_cap(size: int, max_bytes: int) -> None:
    if size > max_bytes:
        raise ValidationError(
            f"image is {size} bytes, above the {max_bytes} byte cap",
            details={"size_bytes": size, "limit": max_bytes},
        )


def _check_pixel_cap(width: int, height: int, max_pixels: int) -> None:
    if width <= 0 or height <= 0:
        raise ValidationError(
            "image dimensions must be positive",
            details={"width": width, "height": height},
        )
    pixels = width * height
    if pixels > max_pixels:
        raise ValidationError(
            f"image declares {pixels} pixels, above the "
            f"{max_pixels} pixel decompression bomb guard",
            details={
                "width": width,
                "height": height,
                "pixels": pixels,
                "limit": max_pixels,
            },
        )


class ImageFormat(StrEnum):
    """Output formats the utilities can write."""

    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"
    GIF = "gif"
    BMP = "bmp"
    TIFF = "tiff"

    @classmethod
    def parse(cls, value: str | ImageFormat) -> ImageFormat:
        """Return a format from a name, alias or file suffix."""
        if isinstance(value, cls):
            return value
        key = value.strip().lower().lstrip(".")
        aliases = {
            "jpg": cls.JPEG,
            "jpe": cls.JPEG,
            "tif": cls.TIFF,
        }
        if key in aliases:
            return aliases[key]
        try:
            return cls(key)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValidationError(
                f"unsupported image format {value!r}; expected one of {allowed}"
            ) from exc

    @classmethod
    def from_path(cls, path: Path) -> ImageFormat:
        """Return the format implied by a destination file suffix."""
        if not path.suffix:
            raise ValidationError(
                f"cannot infer an image format from {path.name!r}; "
                "pass an explicit format"
            )
        return cls.parse(path.suffix)

    @property
    def pillow_name(self) -> str:
        """The format name Pillow expects when saving."""
        return self.name

    @property
    def supports_alpha(self) -> bool:
        """Whether the format can store an alpha channel."""
        return self in {ImageFormat.PNG, ImageFormat.WEBP, ImageFormat.GIF}


class ImageFit(StrEnum):
    """How an image should reach a requested size."""

    STRETCH = "stretch"
    CONTAIN = "contain"
    COVER = "cover"
    BLURRED_PADDING = "blurred_padding"


def _validate_size(size: tuple[int, int]) -> tuple[int, int]:
    width, height = size
    for value, label in ((width, "width"), (height, "height")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValidationError(f"image {label} must be a positive integer")
    return width, height


def _validate_centering(centering: tuple[float, float]) -> tuple[float, float]:
    if len(centering) != 2:
        raise ValidationError("centering must contain exactly two values")
    for value in centering:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValidationError("centering values must be numbers")
        if not 0.0 <= float(value) <= 1.0:
            raise ValidationError("centering values must be between 0 and 1")
    return float(centering[0]), float(centering[1])


def _validate_quality(quality: int) -> int:
    if isinstance(quality, bool) or not isinstance(quality, int):
        raise ValidationError("image quality must be an integer")
    if not 1 <= quality <= 100:
        raise ValidationError("image quality must be between 1 and 100")
    return quality


def _validate_radius(radius: float) -> float:
    if isinstance(radius, bool) or not isinstance(radius, (int, float)):
        raise ValidationError("blur radius must be a number")
    if float(radius) < 0.0:
        raise ValidationError("blur radius must not be negative")
    return float(radius)


def _rgba(background: tuple[int, ...]) -> tuple[int, int, int, int]:
    if len(background) not in (3, 4):
        raise ValidationError("background must have three or four channels")
    channels = [int(value) for value in background]
    if any(not 0 <= value <= 255 for value in channels):
        raise ValidationError("background channels must be between 0 and 255")
    if len(channels) == 3:
        channels.append(255)
    return channels[0], channels[1], channels[2], channels[3]


def _has_alpha(image: Image.Image) -> bool:
    return image.mode in {"RGBA", "LA", "PA"} or (
        image.mode == "P" and "transparency" in image.info
    )


def _flatten(image: Image.Image, background: tuple[int, ...]) -> Image.Image:
    rgba = image.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, _rgba(background))
    canvas.alpha_composite(rgba)
    return canvas.convert("RGB")


def _prepare_for_format(
    image: Image.Image,
    image_format: ImageFormat,
    *,
    background: tuple[int, ...],
) -> Image.Image:
    if image_format in {ImageFormat.JPEG, ImageFormat.BMP}:
        if _has_alpha(image):
            return _flatten(image, background)
        if image.mode not in {"RGB", "L"}:
            return image.convert("RGB")
    return image.copy()


def open_image(
    source: ImageSource,
    *,
    max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> Image.Image:
    """Open an image path or encoded bytes and return a detached copy.

    The file handle is closed before returning, so callers may delete or move
    the source immediately. Oversized payloads and images whose header
    declares more than ``max_pixels`` are rejected from their headers before
    any pixel data is decoded. Invalid data raises :class:`ValidationError`.
    """
    byte_limit = _validate_limit(max_bytes, "max_bytes")
    pixel_limit = _validate_limit(max_pixels, "max_pixels")
    try:
        if isinstance(source, bytes):
            _check_byte_cap(len(source), byte_limit)
            declared = image_dimensions(source)
            if declared is not None:
                _check_pixel_cap(*declared, pixel_limit)
            with Image.open(BytesIO(source)) as opened:
                _check_pixel_cap(*opened.size, pixel_limit)
                opened.load()
                return opened.copy()
        path = Path(source)
        _check_byte_cap(path.stat().st_size, byte_limit)
        with path.open("rb") as handle:
            declared = image_dimensions(handle.read(_HEADER_BYTES))
        if declared is not None:
            _check_pixel_cap(*declared, pixel_limit)
        with Image.open(path) as opened:
            _check_pixel_cap(*opened.size, pixel_limit)
            opened.load()
            return opened.copy()
    except Image.DecompressionBombError as exc:
        raise ValidationError(
            f"image exceeds the {pixel_limit} pixel decompression bomb guard",
            details={"limit": pixel_limit},
        ) from exc
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ValidationError("could not open image data") from exc


def resize_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Resize an image to exactly ``size`` without cropping or padding."""
    width, height = _validate_size(size)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def aspect_crop(
    image: Image.Image,
    size: tuple[int, int],
    *,
    centering: tuple[float, float] = (0.5, 0.5),
) -> Image.Image:
    """Crop to the target aspect ratio, then resize to exactly ``size``.

    ``centering`` selects the crop window: ``(0, 0)`` keeps the top-left,
    ``(1, 1)`` the bottom-right and the default ``(0.5, 0.5)`` centers it.
    """
    width, height = _validate_size(size)
    center_x, center_y = _validate_centering(centering)
    source_width, source_height = image.size
    target_ratio = width / height
    source_ratio = source_width / source_height

    if source_ratio > target_ratio:
        crop_width = max(1, round(source_height * target_ratio))
        crop_height = source_height
    else:
        crop_width = source_width
        crop_height = max(1, round(source_width / target_ratio))

    left = round((source_width - crop_width) * center_x)
    top = round((source_height - crop_height) * center_y)
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    if cropped.size != (width, height):
        cropped = cropped.resize((width, height), Image.Resampling.LANCZOS)
    return cropped


def blurred_fill_pad(
    image: Image.Image,
    size: tuple[int, int],
    *,
    blur_radius: float = DEFAULT_BLUR_RADIUS,
    background: tuple[int, ...] = (0, 0, 0, 255),
) -> Image.Image:
    """Fit the whole image inside ``size`` and pad with a blurred copy.

    The foreground keeps its aspect ratio and is centered. The background is
    the same image cover-cropped to the target size and blurred, so no hard
    bars or stretched pixels appear. Transparent pixels are composited over
    ``background`` before blurring.
    """
    width, height = _validate_size(size)
    radius = _validate_radius(blur_radius)
    background_rgba = _rgba(background)

    rgba = image.convert("RGBA")
    flattened = Image.new("RGBA", rgba.size, background_rgba)
    flattened.alpha_composite(rgba)
    background_layer = aspect_crop(flattened, (width, height)).filter(
        ImageFilter.GaussianBlur(radius)
    )

    scale = min(width / rgba.width, height / rgba.height)
    foreground_size = (
        max(1, round(rgba.width * scale)),
        max(1, round(rgba.height * scale)),
    )
    foreground = rgba.resize(foreground_size, Image.Resampling.LANCZOS)
    offset = ((width - foreground.width) // 2, (height - foreground.height) // 2)
    canvas = background_layer.copy()
    canvas.alpha_composite(foreground, dest=offset)

    if image.mode in {"RGB", "RGBA", "L", "LA"}:
        return canvas.convert(image.mode)
    return canvas


def convert_image(
    image: Image.Image,
    image_format: str | ImageFormat,
    *,
    background: tuple[int, ...] = (255, 255, 255, 255),
) -> Image.Image:
    """Return a copy prepared for saving as ``image_format``.

    Transparency is flattened over ``background`` for formats that cannot
    store an alpha channel. The original image is not changed.
    """
    target = ImageFormat.parse(image_format)
    return _prepare_for_format(image, target, background=background)


def strip_metadata(image: Image.Image) -> Image.Image:
    """Return a copy without EXIF, ICC, text or other auxiliary metadata."""
    clean = image.copy()
    clean.info.clear()
    clean.getexif().clear()
    return clean


def save_image(
    image: Image.Image,
    destination: str | Path,
    *,
    image_format: str | ImageFormat | None = None,
    quality: int = 90,
    background: tuple[int, ...] = (255, 255, 255, 255),
) -> Path:
    """Write ``image`` to ``destination`` atomically with metadata removed.

    The format comes from ``image_format`` or, when omitted, from the
    destination suffix. The write happens in a temporary file beside the
    destination and is moved into place only after Pillow succeeds, so a
    failed encode never leaves a partial output.
    """
    path = Path(destination)
    target = (
        ImageFormat.from_path(path)
        if image_format is None
        else ImageFormat.parse(image_format)
    )
    checked_quality = _validate_quality(quality)
    prepared = _prepare_for_format(image, target, background=background)
    clean = strip_metadata(prepared)

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        options: dict[str, object] = {}
        if target in {ImageFormat.JPEG, ImageFormat.WEBP}:
            options["quality"] = checked_quality
        clean.save(temporary, format=target.pillow_name, **options)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _same_file(source: Path, destination: Path) -> bool:
    if source.resolve() == destination.resolve():
        return True
    try:
        return destination.exists() and os.path.samefile(source, destination)
    except OSError:
        return False


def transform_image(
    source: ImageSource,
    destination: str | Path,
    *,
    size: tuple[int, int] | None = None,
    fit: str | ImageFit = ImageFit.CONTAIN,
    image_format: str | ImageFormat | None = None,
    quality: int = 90,
    blur_radius: float = DEFAULT_BLUR_RADIUS,
    background: tuple[int, ...] = (255, 255, 255, 255),
) -> Path:
    """Transform ``source`` into a new file at ``destination``.

    The source is never modified: its path cannot be the destination, and the
    output is written atomically. ``fit`` chooses how ``size`` is reached:
    ``stretch`` resizes exactly, ``contain`` scales down without padding,
    ``cover`` aspect-crops and ``blurred_padding`` centers the image on a
    blurred fill. EXIF orientation is applied before metadata is stripped.
    """
    target = Path(destination)
    if isinstance(source, (str, Path)) and _same_file(Path(source), target):
        raise ValidationError("destination must not overwrite the source image")

    image = ImageOps.exif_transpose(open_image(source))
    if size is not None:
        fit_mode = fit if isinstance(fit, ImageFit) else ImageFit(fit)
        if fit_mode is ImageFit.STRETCH:
            image = resize_image(image, size)
        elif fit_mode is ImageFit.CONTAIN:
            width, height = _validate_size(size)
            scale = min(width / image.width, height / image.height)
            contained = (
                max(1, round(image.width * scale)),
                max(1, round(image.height * scale)),
            )
            image = image.resize(contained, Image.Resampling.LANCZOS)
        elif fit_mode is ImageFit.COVER:
            image = aspect_crop(image, size)
        else:
            image = blurred_fill_pad(
                image,
                size,
                blur_radius=blur_radius,
                background=background,
            )
    return save_image(
        image,
        target,
        image_format=image_format,
        quality=quality,
        background=background,
    )

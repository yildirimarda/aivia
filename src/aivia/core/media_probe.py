"""Download media safely and validate what actually arrived.

A provider result is untrusted input. The URL may serve something other than
the requested format, the body may be larger than the caller allowed, and an
image header may declare a canvas so large that decoding it would exhaust
memory. This module centralises the checks every downloaded asset needs:

* format sniffing from magic bytes, never from the filename or Content-Type;
* image dimensions read from the container header before any decode;
* byte, pixel, frame and total decoded pixel caps, including a
  decompression-bomb guard;
* ffprobe inspection of video and audio files for streams, dimensions and
  duration;
* an optional destination write that happens only after validation succeeds.

The downloader uses the shared HTTP client, so retries, timeouts, redaction and
mock transports behave exactly as they do for every provider. ffprobe is run
without a shell and accepts an injectable runner so the offline suite never
needs the real binary.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import httpx

from aivia.core.config.http import HttpClientConfig
from aivia.core.config.media import (
    DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS as DEFAULT_FFPROBE_TIMEOUT_SECONDS,
)
from aivia.core.config.media import (
    DEFAULT_MEDIA_MAX_ANIMATED_FRAMES as DEFAULT_MAX_FRAMES,
)
from aivia.core.config.media import (
    DEFAULT_MEDIA_MAX_ANIMATED_PIXELS as DEFAULT_MAX_TOTAL_PIXELS,
)
from aivia.core.config.media import DEFAULT_MEDIA_MAX_BYTES as DEFAULT_MAX_BYTES
from aivia.core.config.media import DEFAULT_MEDIA_MAX_PIXELS as DEFAULT_MAX_PIXELS
from aivia.core.errors import ValidationError
from aivia.core.http import HttpClient

__all__ = [
    "DEFAULT_FFPROBE_TIMEOUT_SECONDS",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_FRAMES",
    "DEFAULT_MAX_PIXELS",
    "DEFAULT_MAX_TOTAL_PIXELS",
    "FFprobeInfo",
    "FFprobeStream",
    "MediaCategory",
    "MediaDownloader",
    "MediaFormat",
    "MediaLimits",
    "MediaProbe",
    "MediaSignature",
    "image_dimensions",
    "image_frame_count",
    "run_ffprobe",
    "sniff_media",
    "validate_media",
]

FFprobeRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
"""A callable that runs an ffprobe argv and returns the completed process."""


class MediaCategory(StrEnum):
    """The broad family a sniffed media format belongs to."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class MediaFormat(StrEnum):
    """Container formats this module can recognise from magic bytes."""

    PNG = "png"
    JPEG = "jpeg"
    GIF = "gif"
    WEBP = "webp"
    BMP = "bmp"
    TIFF = "tiff"
    AVIF = "avif"
    MP4 = "mp4"
    MOV = "mov"
    WEBM = "webm"
    MKV = "mkv"
    AVI = "avi"
    MPEG = "mpeg"
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    M4A = "m4a"


@dataclass(frozen=True, slots=True)
class _FormatSpec:
    category: MediaCategory
    mime_type: str
    extension: str


_FORMATS: dict[MediaFormat, _FormatSpec] = {
    MediaFormat.PNG: _FormatSpec(MediaCategory.IMAGE, "image/png", "png"),
    MediaFormat.JPEG: _FormatSpec(MediaCategory.IMAGE, "image/jpeg", "jpg"),
    MediaFormat.GIF: _FormatSpec(MediaCategory.IMAGE, "image/gif", "gif"),
    MediaFormat.WEBP: _FormatSpec(MediaCategory.IMAGE, "image/webp", "webp"),
    MediaFormat.BMP: _FormatSpec(MediaCategory.IMAGE, "image/bmp", "bmp"),
    MediaFormat.TIFF: _FormatSpec(MediaCategory.IMAGE, "image/tiff", "tiff"),
    MediaFormat.AVIF: _FormatSpec(MediaCategory.IMAGE, "image/avif", "avif"),
    MediaFormat.MP4: _FormatSpec(MediaCategory.VIDEO, "video/mp4", "mp4"),
    MediaFormat.MOV: _FormatSpec(MediaCategory.VIDEO, "video/quicktime", "mov"),
    MediaFormat.WEBM: _FormatSpec(MediaCategory.VIDEO, "video/webm", "webm"),
    MediaFormat.MKV: _FormatSpec(MediaCategory.VIDEO, "video/x-matroska", "mkv"),
    MediaFormat.AVI: _FormatSpec(MediaCategory.VIDEO, "video/x-msvideo", "avi"),
    MediaFormat.MPEG: _FormatSpec(MediaCategory.VIDEO, "video/mpeg", "mpeg"),
    MediaFormat.WAV: _FormatSpec(MediaCategory.AUDIO, "audio/wav", "wav"),
    MediaFormat.MP3: _FormatSpec(MediaCategory.AUDIO, "audio/mpeg", "mp3"),
    MediaFormat.FLAC: _FormatSpec(MediaCategory.AUDIO, "audio/flac", "flac"),
    MediaFormat.OGG: _FormatSpec(MediaCategory.AUDIO, "audio/ogg", "ogg"),
    MediaFormat.M4A: _FormatSpec(MediaCategory.AUDIO, "audio/mp4", "m4a"),
}

_JPEG_SOF = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)
_MP4_BRANDS = frozenset(
    {
        b"isom",
        b"iso2",
        b"iso3",
        b"iso4",
        b"iso5",
        b"iso6",
        b"mp41",
        b"mp42",
        b"avc1",
        b"dash",
        b"M4V ",
        b"M4VH",
        b"M4VP",
        b"f4v ",
    }
)
_M4A_BRANDS = frozenset({b"M4A ", b"M4B ", b"M4P ", b"M4P0", b"M4P1", b"F4A "})
_AVIF_BRANDS = frozenset({b"avif", b"avis"})


@dataclass(frozen=True, slots=True)
class MediaSignature:
    """The format detected from a payload's magic bytes."""

    format: MediaFormat
    category: MediaCategory
    mime_type: str
    extension: str


@dataclass(frozen=True, slots=True)
class FFprobeStream:
    """The subset of one ffprobe stream this module validates."""

    index: int
    codec_type: str
    codec_name: str
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    frame_rate: str | None = None
    pixel_format: str | None = None


@dataclass(frozen=True, slots=True)
class FFprobeInfo:
    """The validated result of one ffprobe inspection."""

    format_name: str
    format_long_name: str
    duration_seconds: float | None
    size_bytes: int | None
    streams: tuple[FFprobeStream, ...]

    @property
    def video_streams(self) -> tuple[FFprobeStream, ...]:
        """Every stream ffprobe reports as video."""
        return tuple(stream for stream in self.streams if stream.codec_type == "video")

    @property
    def audio_streams(self) -> tuple[FFprobeStream, ...]:
        """Every stream ffprobe reports as audio."""
        return tuple(stream for stream in self.streams if stream.codec_type == "audio")

    @property
    def primary_video(self) -> FFprobeStream | None:
        """The first video stream, if the file has one."""
        return self.video_streams[0] if self.video_streams else None


@dataclass(frozen=True, slots=True)
class MediaLimits:
    """Caps and dimension bounds applied to one downloaded asset.

    ``max_pixels`` is the decompression-bomb guard: it is checked from the
    image header before any decoder sees the pixels. Animated images are also
    rejected when they declare more than ``max_frames`` frames or would decode
    more than ``max_total_pixels`` pixels across every frame. The optional
    dimension and duration bounds let a caller enforce what was requested, and
    ``allowed_formats`` rejects a result whose real container is not acceptable.
    """

    max_bytes: int = DEFAULT_MAX_BYTES
    max_pixels: int = DEFAULT_MAX_PIXELS
    min_width: int | None = None
    max_width: int | None = None
    min_height: int | None = None
    max_height: int | None = None
    min_duration_seconds: float | None = None
    max_duration_seconds: float | None = None
    allowed_formats: frozenset[MediaFormat] | None = None
    max_frames: int = DEFAULT_MAX_FRAMES
    max_total_pixels: int = DEFAULT_MAX_TOTAL_PIXELS

    def __post_init__(self) -> None:
        for label in ("max_bytes", "max_pixels", "max_frames", "max_total_pixels"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{label} must be an integer")
            if value <= 0:
                raise ValueError(f"{label} must be positive")
        for label in ("min_width", "max_width", "min_height", "max_height"):
            value = getattr(self, label)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{label} must be an integer or None")
            if value <= 0:
                raise ValueError(f"{label} must be positive")
        if (
            self.min_width is not None
            and self.max_width is not None
            and self.min_width > self.max_width
        ):
            raise ValueError("min_width must not exceed max_width")
        if (
            self.min_height is not None
            and self.max_height is not None
            and self.min_height > self.max_height
        ):
            raise ValueError("min_height must not exceed max_height")
        for label in ("min_duration_seconds", "max_duration_seconds"):
            value = getattr(self, label)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{label} must be a number or None")
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{label} must be finite and non-negative")
        if (
            self.min_duration_seconds is not None
            and self.max_duration_seconds is not None
            and self.min_duration_seconds > self.max_duration_seconds
        ):
            raise ValueError(
                "min_duration_seconds must not exceed max_duration_seconds"
            )
        if self.allowed_formats is not None:
            formats = frozenset(self.allowed_formats)
            if any(not isinstance(item, MediaFormat) for item in formats):
                raise TypeError("allowed_formats must contain MediaFormat values")
            object.__setattr__(self, "allowed_formats", formats)


@dataclass(frozen=True, slots=True)
class MediaProbe:
    """Everything validation learned about one media payload."""

    signature: MediaSignature
    size_bytes: int
    sha256: str
    width: int | None = None
    height: int | None = None
    frames: int | None = None
    duration_seconds: float | None = None
    ffprobe: FFprobeInfo | None = None
    path: Path | None = None
    data: bytes = field(default=b"", repr=False)

    @property
    def format(self) -> MediaFormat:
        """The sniffed container format."""
        return self.signature.format

    @property
    def category(self) -> MediaCategory:
        """The image, video or audio family of the format."""
        return self.signature.category

    @property
    def mime_type(self) -> str:
        """The MIME type implied by the sniffed format."""
        return self.signature.mime_type

    @property
    def extension(self) -> str:
        """A conventional filename extension without a dot."""
        return self.signature.extension

    @property
    def pixels(self) -> int | None:
        """The declared pixel count for an image, when known."""
        if self.width is None or self.height is None:
            return None
        return self.width * self.height

    @property
    def total_pixels(self) -> int | None:
        """The declared pixel count across every frame, when known."""
        if self.width is None or self.height is None or self.frames is None:
            return None
        return self.width * self.height * self.frames


def _signature(format: MediaFormat) -> MediaSignature:
    spec = _FORMATS[format]
    return MediaSignature(
        format=format,
        category=spec.category,
        mime_type=spec.mime_type,
        extension=spec.extension,
    )


def sniff_media(data: bytes | bytearray | memoryview) -> MediaSignature | None:
    """Return the format named by *data*'s magic bytes, or ``None``.

    The filename and any HTTP Content-Type are deliberately ignored: a provider
    may label a PNG as ``image/jpeg`` or serve an HTML error page with a 200.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("media data must be bytes-like")
    payload = bytes(data)
    if not payload:
        return None

    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return _signature(MediaFormat.PNG)
    if payload.startswith(b"\xff\xd8\xff"):
        return _signature(MediaFormat.JPEG)
    if payload.startswith((b"GIF87a", b"GIF89a")):
        return _signature(MediaFormat.GIF)
    if payload.startswith(b"BM"):
        return _signature(MediaFormat.BMP)
    if payload.startswith((b"II*\x00", b"MM\x00*")):
        return _signature(MediaFormat.TIFF)
    if payload.startswith(b"fLaC"):
        return _signature(MediaFormat.FLAC)
    if payload.startswith(b"OggS"):
        return _signature(MediaFormat.OGG)
    if payload.startswith(b"ID3"):
        return _signature(MediaFormat.MP3)
    if len(payload) >= 2 and payload[0] == 0xFF and payload[1] & 0xE0 == 0xE0:
        return _signature(MediaFormat.MP3)
    if payload.startswith(b"\x1aE\xdf\xa3"):
        head = payload[:64].lower()
        if b"webm" in head:
            return _signature(MediaFormat.WEBM)
        return _signature(MediaFormat.MKV)
    if payload.startswith((b"\x00\x00\x01\xba", b"\x00\x00\x01\xb3")):
        return _signature(MediaFormat.MPEG)
    if len(payload) >= 12 and payload[:4] == b"RIFF":
        kind = payload[8:12]
        if kind == b"WEBP":
            return _signature(MediaFormat.WEBP)
        if kind == b"WAVE":
            return _signature(MediaFormat.WAV)
        if kind == b"AVI ":
            return _signature(MediaFormat.AVI)
    if len(payload) >= 12 and payload[4:8] == b"ftyp":
        brand = payload[8:12]
        if brand in _AVIF_BRANDS:
            return _signature(MediaFormat.AVIF)
        if brand in _M4A_BRANDS:
            return _signature(MediaFormat.M4A)
        if brand == b"qt  ":
            return _signature(MediaFormat.MOV)
        if brand in _MP4_BRANDS:
            return _signature(MediaFormat.MP4)
        return _signature(MediaFormat.MP4)
    return None


def _uint(
    data: bytes, start: int, length: int, *, byteorder: Literal["little", "big"]
) -> int | None:
    end = start + length
    if start < 0 or end > len(data):
        return None
    return int.from_bytes(data[start:end], byteorder)


def _png_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 24 or data[12:16] != b"IHDR":
        return None
    width = _uint(data, 16, 4, byteorder="big")
    height = _uint(data, 20, 4, byteorder="big")
    if width is None or height is None:
        return None
    return width, height


def _gif_dimensions(data: bytes) -> tuple[int, int] | None:
    width = _uint(data, 6, 2, byteorder="little")
    height = _uint(data, 8, 2, byteorder="little")
    if width is None or height is None:
        return None
    return width, height


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    index = 2
    length = len(data)
    while index + 4 <= length:
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker == 0xFF:
            index += 1
            continue
        if marker in {0x00, 0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        segment = _uint(data, index + 2, 2, byteorder="big")
        if segment is None or segment < 2:
            return None
        if marker in _JPEG_SOF:
            height = _uint(data, index + 5, 2, byteorder="big")
            width = _uint(data, index + 7, 2, byteorder="big")
            if width is None or height is None:
                return None
            return width, height
        index += 2 + segment
    return None


def _webp_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 16:
        return None
    chunk = data[12:16]
    if chunk == b"VP8X":
        width = _uint(data, 24, 3, byteorder="little")
        height = _uint(data, 27, 3, byteorder="little")
        if width is None or height is None:
            return None
        return width + 1, height + 1
    if chunk == b"VP8 ":
        if len(data) < 30 or data[23:26] != b"\x9d\x01\x2a":
            return None
        width = _uint(data, 26, 2, byteorder="little")
        height = _uint(data, 28, 2, byteorder="little")
        if width is None or height is None:
            return None
        return width & 0x3FFF, height & 0x3FFF
    if chunk == b"VP8L":
        if len(data) < 25 or data[20] != 0x2F:
            return None
        b0, b1, b2, b3 = data[21], data[22], data[23], data[24]
        width = (b0 | ((b1 & 0x3F) << 8)) + 1
        height = ((b1 >> 6) | (b2 << 2) | ((b3 & 0x0F) << 10)) + 1
        return width, height
    return None


def _bmp_dimensions(data: bytes) -> tuple[int, int] | None:
    width = _uint(data, 18, 4, byteorder="little")
    height = _uint(data, 22, 4, byteorder="little")
    if width is None or height is None:
        return None
    if height >= 0x80000000:
        height = 0x100000000 - height
    return width, height


def _tiff_dimensions(data: bytes) -> tuple[int, int] | None:
    byteorder: Literal["little", "big"]
    if data[:2] == b"II":
        byteorder = "little"
    elif data[:2] == b"MM":
        byteorder = "big"
    else:
        return None
    offset = _uint(data, 4, 4, byteorder=byteorder)
    if offset is None:
        return None
    count = _uint(data, offset, 2, byteorder=byteorder)
    if count is None:
        return None
    width: int | None = None
    height: int | None = None
    for index in range(count):
        entry = offset + 2 + index * 12
        tag = _uint(data, entry, 2, byteorder=byteorder)
        kind = _uint(data, entry + 2, 2, byteorder=byteorder)
        value = _uint(data, entry + 8, 4, byteorder=byteorder)
        if tag is None or kind is None or value is None:
            return None
        if tag not in {0x0100, 0x0101}:
            continue
        if kind == 3:
            parsed = _uint(data, entry + 8, 2, byteorder=byteorder)
        elif kind == 4:
            parsed = value
        else:
            return None
        if parsed is None:
            return None
        if tag == 0x0100:
            width = parsed
        else:
            height = parsed
    if width is None or height is None:
        return None
    return width, height


def _avif_dimensions(data: bytes) -> tuple[int, int] | None:
    marker = data.find(b"ispe")
    if marker < 0 or marker + 16 > len(data):
        return None
    width = _uint(data, marker + 8, 4, byteorder="big")
    height = _uint(data, marker + 12, 4, byteorder="big")
    if width is None or height is None:
        return None
    return width, height


def image_dimensions(data: bytes | bytearray | memoryview) -> tuple[int, int] | None:
    """Read declared ``(width, height)`` from an image header without decoding.

    PNG, JPEG, GIF, WebP, BMP, TIFF and AVIF are supported. ``None`` means the
    header is incomplete or the format does not carry dimensions this reader
    understands; callers must treat that as a validation failure rather than
    guessing.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("media data must be bytes-like")
    payload = bytes(data)
    signature = sniff_media(payload)
    if signature is None or signature.category is not MediaCategory.IMAGE:
        return None
    parsers: dict[MediaFormat, Callable[[bytes], tuple[int, int] | None]] = {
        MediaFormat.PNG: _png_dimensions,
        MediaFormat.JPEG: _jpeg_dimensions,
        MediaFormat.GIF: _gif_dimensions,
        MediaFormat.WEBP: _webp_dimensions,
        MediaFormat.BMP: _bmp_dimensions,
        MediaFormat.TIFF: _tiff_dimensions,
        MediaFormat.AVIF: _avif_dimensions,
    }
    parser = parsers.get(signature.format)
    if parser is None:
        return None
    return parser(payload)


def _skip_gif_sub_blocks(data: bytes, offset: int) -> int:
    length = len(data)
    while offset < length:
        size = data[offset]
        offset += 1
        if size == 0:
            return offset
        offset += size
    return offset


def _gif_frame_count(data: bytes) -> int:
    if len(data) < 13:
        return 1
    packed = data[10]
    offset = 13
    if packed & 0x80:
        offset += 3 * (1 << ((packed & 0x07) + 1))
    frames = 0
    length = len(data)
    while offset < length:
        block = data[offset]
        if block == 0x3B:
            break
        if block == 0x2C:
            frames += 1
            if offset + 10 > length:
                break
            frame_packed = data[offset + 9]
            offset += 10
            if frame_packed & 0x80:
                offset += 3 * (1 << ((frame_packed & 0x07) + 1))
            if offset >= length:
                break
            offset += 1
            offset = _skip_gif_sub_blocks(data, offset)
        elif block == 0x21:
            offset = _skip_gif_sub_blocks(data, offset + 2)
        else:
            break
    return max(frames, 1)


def _png_frame_count(data: bytes) -> int:
    offset = 8
    length = len(data)
    while offset + 8 <= length:
        size = int.from_bytes(data[offset : offset + 4], "big")
        chunk = data[offset + 4 : offset + 8]
        start = offset + 8
        if chunk == b"acTL":
            if start + 4 <= length:
                frames = int.from_bytes(data[start : start + 4], "big")
                return max(frames, 1)
            return 1
        if chunk in {b"IDAT", b"IEND"}:
            return 1
        offset = start + size + 4
    return 1


def _webp_frame_count(data: bytes) -> int:
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return 1
    offset = 12
    length = len(data)
    animated = False
    frames = 0
    while offset + 8 <= length:
        chunk = data[offset : offset + 4]
        size = int.from_bytes(data[offset + 4 : offset + 8], "little")
        start = offset + 8
        if chunk == b"VP8X" and start < length:
            animated = bool(data[start] & 0x02)
        elif chunk == b"ANMF":
            frames += 1
        if size < 0 or start + size > length:
            break
        offset = start + size + (size & 1)
    if not animated:
        return 1
    return max(frames, 1)


def _avif_frame_count(data: bytes) -> int:
    if b"avis" not in data[:64]:
        return 1
    marker = data.find(b"stsz")
    if marker < 0 or marker + 16 > len(data):
        return 1
    frames = int.from_bytes(data[marker + 12 : marker + 16], "big")
    return max(frames, 1)


def image_frame_count(
    data: bytes | bytearray | memoryview,
) -> int:
    """Read the declared frame count from an image header without decoding.

    GIF, APNG, animated WebP and animated AVIF are understood. A static image
    or a format that does not declare frames counts as one, so callers can
    apply a total decoded pixel cap without a separate static path.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("media data must be bytes-like")
    payload = bytes(data)
    signature = sniff_media(payload)
    if signature is None or signature.category is not MediaCategory.IMAGE:
        return 1
    counters: dict[MediaFormat, Callable[[bytes], int]] = {
        MediaFormat.GIF: _gif_frame_count,
        MediaFormat.PNG: _png_frame_count,
        MediaFormat.WEBP: _webp_frame_count,
        MediaFormat.AVIF: _avif_frame_count,
    }
    counter = counters.get(signature.format)
    if counter is None:
        return 1
    return counter(payload)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _as_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _parse_stream(raw: Mapping[str, Any], index: int) -> FFprobeStream:
    reported = _as_int(raw.get("index"))
    return FFprobeStream(
        index=index if reported is None else reported,
        codec_type=_as_text(raw.get("codec_type")).lower(),
        codec_name=_as_text(raw.get("codec_name")).lower(),
        width=_as_int(raw.get("width")),
        height=_as_int(raw.get("height")),
        duration_seconds=_as_float(raw.get("duration")),
        sample_rate=_as_int(raw.get("sample_rate")),
        channels=_as_int(raw.get("channels")),
        frame_rate=_as_text(raw.get("avg_frame_rate")) or None,
        pixel_format=_as_text(raw.get("pix_fmt")) or None,
    )


def run_ffprobe(
    path: str | Path,
    *,
    executable: str = "ffprobe",
    timeout_seconds: float = DEFAULT_FFPROBE_TIMEOUT_SECONDS,
    runner: FFprobeRunner | None = None,
) -> FFprobeInfo:
    """Inspect *path* with ffprobe and return the parsed streams and format.

    ffprobe is invoked with an argv list and no shell. A missing binary, a
    non-zero exit, malformed JSON or a file with no streams all become a
    :class:`ValidationError`, because none of those can be trusted as a media
    result.
    """
    target = Path(path)
    if not target.is_file():
        raise ValidationError(
            f"ffprobe target does not exist: {target}",
            details={"path": str(target)},
        )
    if not isinstance(executable, str) or not executable.strip():
        raise TypeError("ffprobe executable must be a non-empty string")
    deadline = timeout_seconds
    if (
        isinstance(deadline, bool)
        or not isinstance(deadline, (int, float))
        or not math.isfinite(float(deadline))
        or deadline <= 0
    ):
        raise ValueError("ffprobe timeout must be a finite positive number")

    argv = [
        executable,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "-i",
        str(target),
    ]
    try:
        completed = (
            runner(argv)
            if runner is not None
            else subprocess.run(
                argv,
                capture_output=True,
                text=True,
                check=False,
                timeout=float(timeout_seconds),
            )
        )
    except FileNotFoundError as exc:
        raise ValidationError(
            f"ffprobe is not available: {executable}",
            details={"executable": executable},
        ) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValidationError(
            "ffprobe could not inspect the media",
            details={"executable": executable, "error": type(exc).__name__},
        ) from exc

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise ValidationError(
            "ffprobe rejected the media",
            details={
                "returncode": completed.returncode,
                "stderr": stderr[:500],
            },
        )
    try:
        payload = json.loads(completed.stdout or "")
    except (TypeError, ValueError) as exc:
        raise ValidationError("ffprobe returned invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValidationError("ffprobe returned an unexpected JSON document")

    raw_format = payload.get("format")
    format_data = raw_format if isinstance(raw_format, Mapping) else {}
    raw_streams = payload.get("streams")
    stream_data = raw_streams if isinstance(raw_streams, list) else []
    streams = tuple(
        _parse_stream(item, index)
        for index, item in enumerate(stream_data)
        if isinstance(item, Mapping)
    )
    if not streams:
        raise ValidationError("ffprobe found no media streams")

    duration = _as_float(format_data.get("duration"))
    if duration is None:
        candidates = [
            stream.duration_seconds
            for stream in streams
            if stream.duration_seconds is not None
        ]
        duration = max(candidates) if candidates else None
    return FFprobeInfo(
        format_name=_as_text(format_data.get("format_name")),
        format_long_name=_as_text(format_data.get("format_long_name")),
        duration_seconds=duration,
        size_bytes=_as_int(format_data.get("size")),
        streams=streams,
    )


def _check_dimensions(
    width: int,
    height: int,
    limits: MediaLimits,
    *,
    expected_width: int | None,
    expected_height: int | None,
    source: str,
) -> None:
    if width <= 0 or height <= 0:
        raise ValidationError(
            "media dimensions must be positive",
            details={"width": width, "height": height, "source": source},
        )
    pixels = width * height
    if pixels > limits.max_pixels:
        raise ValidationError(
            f"media declares {pixels} pixels, above the "
            f"{limits.max_pixels} pixel decompression bomb guard",
            details={
                "width": width,
                "height": height,
                "pixels": pixels,
                "limit": limits.max_pixels,
                "source": source,
            },
        )
    for label, value, minimum, maximum in (
        ("width", width, limits.min_width, limits.max_width),
        ("height", height, limits.min_height, limits.max_height),
    ):
        if minimum is not None and value < minimum:
            raise ValidationError(
                f"media {label} {value} is below the minimum {minimum}",
                details={label: value, "minimum": minimum, "source": source},
            )
        if maximum is not None and value > maximum:
            raise ValidationError(
                f"media {label} {value} is above the maximum {maximum}",
                details={label: value, "maximum": maximum, "source": source},
            )
    if expected_width is not None and width != expected_width:
        raise ValidationError(
            f"media width {width} does not match the requested {expected_width}",
            details={
                "width": width,
                "expected_width": expected_width,
                "source": source,
            },
        )
    if expected_height is not None and height != expected_height:
        raise ValidationError(
            f"media height {height} does not match the requested {expected_height}",
            details={
                "height": height,
                "expected_height": expected_height,
                "source": source,
            },
        )


def _check_frame_cap(frames: int, max_frames: int) -> None:
    if frames > max_frames:
        raise ValidationError(
            f"image has {frames} frames, above the {max_frames} frame cap",
            details={"frames": frames, "limit": max_frames},
        )


def _check_total_pixel_cap(
    width: int,
    height: int,
    frames: int,
    max_total_pixels: int,
) -> None:
    pixels = width * height * frames
    if pixels > max_total_pixels:
        raise ValidationError(
            f"image decodes {pixels} pixels across {frames} frames, above the "
            f"{max_total_pixels} total decoded pixel cap",
            details={
                "width": width,
                "height": height,
                "frames": frames,
                "pixels": pixels,
                "limit": max_total_pixels,
            },
        )


def _check_duration(duration: float | None, limits: MediaLimits) -> None:
    if duration is None:
        if (
            limits.min_duration_seconds is not None
            or limits.max_duration_seconds is not None
        ):
            raise ValidationError("media duration is unknown")
        return
    if (
        limits.min_duration_seconds is not None
        and duration < limits.min_duration_seconds
    ):
        raise ValidationError(
            f"media duration {duration:.3f}s is below the minimum "
            f"{limits.min_duration_seconds:.3f}s",
            details={
                "duration_seconds": duration,
                "minimum": limits.min_duration_seconds,
            },
        )
    if (
        limits.max_duration_seconds is not None
        and duration > limits.max_duration_seconds
    ):
        raise ValidationError(
            f"media duration {duration:.3f}s is above the maximum "
            f"{limits.max_duration_seconds:.3f}s",
            details={
                "duration_seconds": duration,
                "maximum": limits.max_duration_seconds,
            },
        )


def validate_media(
    data: bytes | bytearray | memoryview,
    *,
    path: str | Path | None = None,
    limits: MediaLimits | None = None,
    expected_format: MediaFormat | str | None = None,
    expected_width: int | None = None,
    expected_height: int | None = None,
    ffprobe_executable: str = "ffprobe",
    ffprobe_timeout_seconds: float = DEFAULT_FFPROBE_TIMEOUT_SECONDS,
    ffprobe_runner: FFprobeRunner | None = None,
) -> MediaProbe:
    """Validate one payload and return what was actually detected.

    Images are checked from their headers, including the decompression-bomb
    guard, without decoding. Video and audio are written to a temporary file
    when *path* is not supplied and inspected with ffprobe. A rejected payload
    raises :class:`~aivia.core.errors.ValidationError` with structured details.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("media data must be bytes-like")
    payload = bytes(data)
    active = limits or MediaLimits()
    if not payload:
        raise ValidationError("media payload is empty")
    if len(payload) > active.max_bytes:
        raise ValidationError(
            f"media is {len(payload)} bytes, above the {active.max_bytes} byte cap",
            details={"size_bytes": len(payload), "limit": active.max_bytes},
        )

    signature = sniff_media(payload)
    if signature is None:
        raise ValidationError(
            "media format could not be recognised from its magic bytes",
            details={"size_bytes": len(payload)},
        )
    if active.allowed_formats is not None and signature.format not in (
        active.allowed_formats
    ):
        allowed = ", ".join(sorted(item.value for item in active.allowed_formats))
        raise ValidationError(
            f"media format {signature.format.value!r} is not allowed "
            f"(allowed: {allowed})",
            details={"format": signature.format.value, "allowed": allowed},
        )
    if expected_format is not None:
        wanted = (
            expected_format
            if isinstance(expected_format, MediaFormat)
            else MediaFormat(expected_format)
        )
        if signature.format is not wanted:
            raise ValidationError(
                f"media format {signature.format.value!r} does not match the "
                f"requested {wanted.value!r}",
                details={
                    "format": signature.format.value,
                    "expected_format": wanted.value,
                },
            )

    digest = hashlib.sha256(payload).hexdigest()
    width: int | None = None
    height: int | None = None
    frames: int | None = None
    duration: float | None = None
    info: FFprobeInfo | None = None
    resolved_path = Path(path) if path is not None else None

    if signature.category is MediaCategory.IMAGE:
        dimensions = image_dimensions(payload)
        if dimensions is None:
            raise ValidationError(
                "image dimensions could not be read from the header",
                details={"format": signature.format.value},
            )
        width, height = dimensions
        _check_dimensions(
            width,
            height,
            active,
            expected_width=expected_width,
            expected_height=expected_height,
            source="header",
        )
        frames = image_frame_count(payload)
        _check_frame_cap(frames, active.max_frames)
        _check_total_pixel_cap(width, height, frames, active.max_total_pixels)
    else:
        temporary: Path | None = None
        target = resolved_path
        if target is None:
            with tempfile.NamedTemporaryFile(
                prefix="aivia-probe-",
                suffix=f".{signature.extension}",
                delete=False,
            ) as handle:
                handle.write(payload)
                temporary = Path(handle.name)
            target = temporary
        try:
            info = run_ffprobe(
                target,
                executable=ffprobe_executable,
                timeout_seconds=ffprobe_timeout_seconds,
                runner=ffprobe_runner,
            )
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

        if signature.category is MediaCategory.VIDEO:
            if not info.video_streams:
                raise ValidationError(
                    "video payload has no video stream",
                    details={"format": signature.format.value},
                )
            primary = info.primary_video
            if primary is not None:
                width = primary.width
                height = primary.height
            if width is not None and height is not None:
                _check_dimensions(
                    width,
                    height,
                    active,
                    expected_width=expected_width,
                    expected_height=expected_height,
                    source="ffprobe",
                )
        elif not info.audio_streams:
            raise ValidationError(
                "audio payload has no audio stream",
                details={"format": signature.format.value},
            )
        duration = info.duration_seconds
        _check_duration(duration, active)

    return MediaProbe(
        signature=signature,
        size_bytes=len(payload),
        sha256=digest,
        width=width,
        height=height,
        frames=frames,
        duration_seconds=duration,
        ffprobe=info,
        path=resolved_path,
        data=payload,
    )


class MediaDownloader:
    """Fetch a remote media URL and validate it before returning or writing it.

    The downloader shares :class:`~aivia.core.http.HttpClient` retry and timeout
    behaviour, enforces the byte cap both from ``Content-Length`` and from the
    actual body, and writes *destination* only after validation has passed.
    Pass a ``transport`` in tests to avoid the network.
    """

    def __init__(
        self,
        *,
        http: HttpClient | None = None,
        http_config: HttpClientConfig | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        limits: MediaLimits | None = None,
        ffprobe_executable: str = "ffprobe",
        ffprobe_timeout_seconds: float = DEFAULT_FFPROBE_TIMEOUT_SECONDS,
        ffprobe_runner: FFprobeRunner | None = None,
    ) -> None:
        if http is not None and transport is not None:
            raise TypeError("pass either an http client or a transport, not both")
        if http is not None and http_config is not None:
            raise TypeError("pass either an http client or an http config, not both")
        self.limits = limits or MediaLimits()
        self.ffprobe_executable = ffprobe_executable
        self.ffprobe_timeout_seconds = ffprobe_timeout_seconds
        self.ffprobe_runner = ffprobe_runner
        self._http = http
        self._http_config = http_config or HttpClientConfig()
        self._transport = transport
        self._owns_http = http is None

    @property
    def http(self) -> HttpClient:
        """The HTTP client used for downloads."""
        if self._http is None:
            self._http = HttpClient(self._http_config, transport=self._transport)
        return self._http

    async def download(
        self,
        url: str,
        *,
        destination: str | Path | None = None,
        limits: MediaLimits | None = None,
        expected_format: MediaFormat | str | None = None,
        expected_width: int | None = None,
        expected_height: int | None = None,
        timeout_seconds: float | None = None,
    ) -> MediaProbe:
        """Download *url*, validate it, then optionally write *destination*.

        The destination is only created after every check passes, so a rejected
        download never leaves a partially trusted file behind.
        """
        from aivia.providers.protocols.base import validate_http_url

        target = validate_http_url(url, field="download url")
        active = limits or self.limits
        response = await self.http.get(
            target,
            timeout=timeout_seconds,
            raise_for_status=True,
        )
        try:
            declared = response.headers.get("content-length", "").strip()
            if declared.isdigit():
                size = int(declared)
                if size > active.max_bytes:
                    raise ValidationError(
                        f"download declares {size} bytes, above the "
                        f"{active.max_bytes} byte cap",
                        details={
                            "url": target,
                            "size_bytes": size,
                            "limit": active.max_bytes,
                        },
                    )
            payload = response.content
        finally:
            await response.aclose()

        probe = validate_media(
            payload,
            limits=active,
            expected_format=expected_format,
            expected_width=expected_width,
            expected_height=expected_height,
            ffprobe_executable=self.ffprobe_executable,
            ffprobe_timeout_seconds=self.ffprobe_timeout_seconds,
            ffprobe_runner=self.ffprobe_runner,
        )
        if destination is not None:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            probe = MediaProbe(
                signature=probe.signature,
                size_bytes=probe.size_bytes,
                sha256=probe.sha256,
                width=probe.width,
                height=probe.height,
                frames=probe.frames,
                duration_seconds=probe.duration_seconds,
                ffprobe=probe.ffprobe,
                path=path,
                data=payload,
            )
        return probe

    async def aclose(self) -> None:
        """Close the HTTP client when this downloader created it."""
        if self._owns_http and self._http is not None:
            await self._http.aclose()
            self._http = None

    async def close(self) -> None:
        """Alias for :meth:`aclose`."""
        await self.aclose()

    async def __aenter__(self) -> MediaDownloader:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        await self.aclose()

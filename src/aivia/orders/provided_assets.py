"""Caller-supplied background, loop, audio and logo assets for an order.

A caller may already own the media an order needs. This module accepts those
files, validates that each is a recognised format of the expected category and
that its resolution (and, for audio or video, duration) is within the requested
bounds, then fits each one to the target canvas. Fitted assets are written as
new files inside ``destination``; the source files are never modified.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from aivia.core.config.media import DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS
from aivia.core.errors import ValidationError
from aivia.core.media_probe import (
    MediaCategory,
    MediaFormat,
    MediaLimits,
    MediaProbe,
    validate_media,
)
from aivia.media.audio_utils import trim_audio
from aivia.media.image_utils import ImageFit, ImageFormat, transform_image

__all__ = [
    "AcceptProvidedAssetsError",
    "AssetKind",
    "ProvidedAsset",
    "ProvidedAssets",
    "accept_provided_assets",
]


type _Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[Any]]
"""An ffmpeg or ffprobe runner that may report output as str or bytes."""


class AcceptProvidedAssetsError(ValidationError):
    """Raised when a caller-supplied asset cannot be accepted or fitted."""


_IMAGE_FORMATS = frozenset(
    {
        MediaFormat.PNG,
        MediaFormat.JPEG,
        MediaFormat.GIF,
        MediaFormat.WEBP,
        MediaFormat.BMP,
        MediaFormat.TIFF,
        MediaFormat.AVIF,
    }
)
_AUDIO_FORMATS = frozenset(
    {
        MediaFormat.WAV,
        MediaFormat.MP3,
        MediaFormat.FLAC,
        MediaFormat.OGG,
        MediaFormat.M4A,
    }
)
_VIDEO_FORMATS = frozenset(
    {
        MediaFormat.MP4,
        MediaFormat.MOV,
        MediaFormat.WEBM,
        MediaFormat.MKV,
        MediaFormat.AVI,
        MediaFormat.MPEG,
    }
)
_LOOP_FORMATS = _AUDIO_FORMATS | _VIDEO_FORMATS


def _default_formats(kind: AssetKind) -> frozenset[MediaFormat]:
    if kind in (AssetKind.BACKGROUND, AssetKind.LOGO):
        return _IMAGE_FORMATS
    if kind is AssetKind.AUDIO:
        return _AUDIO_FORMATS
    return _LOOP_FORMATS


def _expected_category(kind: AssetKind) -> MediaCategory | None:
    if kind in (AssetKind.BACKGROUND, AssetKind.LOGO):
        return MediaCategory.IMAGE
    if kind is AssetKind.AUDIO:
        return MediaCategory.AUDIO
    return None


class AssetKind(StrEnum):
    """The role a caller-supplied asset plays in an order."""

    BACKGROUND = "background"
    """A backdrop image fitted to the canvas."""
    LOOP = "loop"
    """A looping audio or video bed."""
    AUDIO = "audio"
    """A music or voice audio track."""
    LOGO = "logo"
    """A brand image placed on the canvas."""


@dataclass(frozen=True, slots=True)
class ProvidedAsset:
    """One validated and fitted caller-supplied asset."""

    kind: AssetKind
    """The role the asset plays."""
    source: Path
    """The original file, left untouched."""
    fitted: Path
    """The fitted copy written into the destination directory."""
    probe: MediaProbe
    """What validation detected about the original file."""


@dataclass(frozen=True, slots=True)
class ProvidedAssets:
    """The full set of caller-supplied assets for an order."""

    background: ProvidedAsset | None = None
    loop: ProvidedAsset | None = None
    audio: ProvidedAsset | None = None
    logo: ProvidedAsset | None = None

    @property
    def supplied(self) -> bool:
        """Whether any asset file was supplied."""
        return (
            self.background is not None
            or self.loop is not None
            or self.audio is not None
            or self.logo is not None
        )

    def by_kind(self, kind: AssetKind) -> ProvidedAsset | None:
        """Return the fitted asset for *kind*, if one was supplied."""
        return cast("ProvidedAsset | None", getattr(self, kind.value))


def _validate_size(size: tuple[int, int]) -> tuple[int, int]:
    width, height = size
    for value, label in ((width, "width"), (height, "height")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValidationError(f"image {label} must be a positive integer")
    return width, height


def _destination_path(
    destination: Path,
    kind: AssetKind,
    source: Path,
    extension: str,
) -> Path:
    stem = source.stem or "asset"
    safe_stem = (
        "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in stem)
        or "asset"
    )
    return destination / f"{kind.value}-{safe_stem}.{extension.lstrip('.')}"


def _copy_file(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


def _fit_image(
    source: Path,
    destination: Path,
    *,
    size: tuple[int, int] | None,
    fit: str,
    image_format: str | ImageFormat | None,
) -> Path:
    if size is None:
        return _copy_file(source, destination)
    return transform_image(
        source,
        destination,
        size=size,
        fit=fit,
        image_format=image_format,
    )


def _fit_audio(
    source: Path,
    destination: Path,
    *,
    duration_seconds: float | None,
    ffmpeg_executable: str,
    ffmpeg_runner: _Runner | None,
    timeout_seconds: float,
) -> Path:
    if duration_seconds is None:
        return _copy_file(source, destination)
    return trim_audio(
        source,
        destination,
        duration_seconds=duration_seconds,
        ffmpeg_executable=ffmpeg_executable,
        ffmpeg_runner=ffmpeg_runner,
        timeout_seconds=timeout_seconds,
    )


def _base_ffmpeg(executable: str) -> list[str]:
    return [executable, "-hide_banner", "-loglevel", "error", "-y"]


def _run_ffmpeg_to_file(
    argv: Sequence[str],
    target: Path,
    *,
    runner: _Runner | None,
    timeout_seconds: float,
    failure: str,
) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    completed = (
        runner(argv)
        if runner is not None
        else subprocess.run(
            argv,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    )
    if not isinstance(completed, subprocess.CompletedProcess):
        completed = subprocess.CompletedProcess(
            list(argv), 0, stdout=completed, stderr=b""
        )
    if completed.returncode != 0:
        stderr = (completed.stderr or b"").decode("utf-8", "replace")
        raise ValidationError(
            failure,
            details={
                "returncode": completed.returncode,
                "stderr": stderr[:500],
                "target": str(target),
            },
        )
    if (
        (not target.is_file() or target.stat().st_size == 0)
        and isinstance(completed.stdout, (bytes, bytearray))
        and completed.stdout
    ):
        target.write_bytes(bytes(completed.stdout))
    if not target.is_file() or target.stat().st_size == 0:
        raise ValidationError(
            f"{failure}: ffmpeg produced no output",
            details={"target": str(target)},
        )
    return target


def _fit_video(
    source: Path,
    destination: Path,
    *,
    size: tuple[int, int] | None,
    ffmpeg_executable: str,
    ffmpeg_runner: _Runner | None,
    timeout_seconds: float,
) -> Path:
    if size is None:
        return _copy_file(source, destination)
    width, height = _validate_size(size)
    target = Path(destination)
    with tempfile.TemporaryDirectory(prefix="aivia-provided-") as directory:
        temporary = Path(directory) / "fitted"
        argv = _base_ffmpeg(ffmpeg_executable)
        argv.extend(
            ["-i", str(source), "-vf", f"scale={width}:{height}", "-c:a", "copy"]
        )
        argv.append(str(temporary))
        _run_ffmpeg_to_file(
            argv,
            temporary,
            runner=ffmpeg_runner,
            timeout_seconds=timeout_seconds,
            failure="ffmpeg could not fit the loop video",
        )
        os.replace(temporary, target)
    return target


def _fit_asset(
    kind: AssetKind,
    source: Path,
    destination: Path,
    *,
    size: tuple[int, int] | None,
    duration_seconds: float | None,
    fit: str,
    image_format: str | ImageFormat | None,
    ffmpeg_executable: str,
    ffmpeg_runner: _Runner | None,
    timeout_seconds: float,
) -> Path:
    if kind in (AssetKind.BACKGROUND, AssetKind.LOGO):
        return _fit_image(
            source,
            destination,
            size=size,
            fit=fit,
            image_format=image_format,
        )
    if kind is AssetKind.AUDIO:
        return _fit_audio(
            source,
            destination,
            duration_seconds=duration_seconds,
            ffmpeg_executable=ffmpeg_executable,
            ffmpeg_runner=ffmpeg_runner,
            timeout_seconds=timeout_seconds,
        )
    return _fit_video(
        source,
        destination,
        size=size,
        ffmpeg_executable=ffmpeg_executable,
        ffmpeg_runner=ffmpeg_runner,
        timeout_seconds=timeout_seconds,
    )


def _accept_one(
    kind: AssetKind,
    source: str | Path | None,
    destination: Path,
    *,
    size: tuple[int, int] | None,
    duration_seconds: float | None,
    limits: MediaLimits | None,
    formats: frozenset[MediaFormat] | None,
    fit: str,
    image_format: str | ImageFormat | None,
    ffmpeg_executable: str,
    ffmpeg_runner: _Runner | None,
    ffprobe_executable: str,
    ffprobe_runner: _Runner | None,
    timeout_seconds: float,
) -> ProvidedAsset | None:
    if source is None:
        return None
    source_path = Path(source).expanduser()
    if not source_path.is_file():
        raise AcceptProvidedAssetsError(
            f"provided {kind.value} asset does not exist: {source_path}",
            details={"kind": kind.value, "path": str(source_path)},
        )
    payload = source_path.read_bytes()
    active_limits = limits or MediaLimits()
    expected_category = _expected_category(kind)
    probe = validate_media(
        payload,
        path=source_path,
        limits=active_limits,
        expected_format=None,
        expected_width=None,
        expected_height=None,
        ffprobe_executable=ffprobe_executable,
        ffprobe_timeout_seconds=timeout_seconds,
        ffprobe_runner=ffprobe_runner,
    )
    allowed = formats if formats is not None else _default_formats(kind)
    if probe.format not in allowed:
        allowed_names = ", ".join(sorted(item.value for item in allowed))
        raise AcceptProvidedAssetsError(
            f"provided {kind.value} asset format {probe.format.value!r} is not "
            f"allowed (allowed: {allowed_names})",
            details={
                "kind": kind.value,
                "format": probe.format.value,
                "allowed": allowed_names,
                "path": str(source_path),
            },
        )
    if expected_category is not None and probe.category is not expected_category:
        raise AcceptProvidedAssetsError(
            f"provided {kind.value} asset must be {expected_category.value}, "
            f"got {probe.category.value}",
            details={
                "kind": kind.value,
                "category": probe.category.value,
                "expected_category": expected_category.value,
                "path": str(source_path),
            },
        )
    extension = probe.extension
    fitted = _fit_asset(
        kind,
        source_path,
        _destination_path(destination, kind, source_path, extension),
        size=size,
        duration_seconds=duration_seconds,
        fit=fit,
        image_format=image_format,
        ffmpeg_executable=ffmpeg_executable,
        ffmpeg_runner=ffmpeg_runner,
        timeout_seconds=timeout_seconds,
    )
    return ProvidedAsset(
        kind=kind,
        source=source_path,
        fitted=fitted,
        probe=probe,
    )


def accept_provided_assets(
    *,
    destination: str | Path,
    background: str | Path | None = None,
    loop: str | Path | None = None,
    audio: str | Path | None = None,
    logo: str | Path | None = None,
    size: tuple[int, int] | None = None,
    duration_seconds: float | None = None,
    limits: MediaLimits | None = None,
    background_formats: frozenset[MediaFormat] | None = None,
    loop_formats: frozenset[MediaFormat] | None = None,
    audio_formats: frozenset[MediaFormat] | None = None,
    logo_formats: frozenset[MediaFormat] | None = None,
    fit: str = ImageFit.COVER,
    image_format: str | ImageFormat | None = None,
    ffmpeg_executable: str = "ffmpeg",
    ffmpeg_runner: _Runner | None = None,
    ffprobe_executable: str = "ffprobe",
    ffprobe_runner: _Runner | None = None,
    timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
) -> ProvidedAssets:
    """Accept and fit the caller-supplied assets for an order.

    Each supplied file is validated for existence, recognised format and
    resolution/duration bounds, then fitted to the requested canvas and written
    as a new file inside ``destination``. Source files are never modified.

    ``size`` fits image and video assets to a canvas ``(width, height)``.
    ``duration_seconds`` trims audio assets. ``limits`` sets resolution and
    duration bounds enforced during validation. Per-kind ``*_formats`` override
    the default recognised container formats.
    """
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    validated_size = None if size is None else _validate_size(size)

    return ProvidedAssets(
        background=_accept_one(
            AssetKind.BACKGROUND,
            background,
            target,
            size=validated_size,
            duration_seconds=None,
            limits=limits,
            formats=background_formats,
            fit=fit,
            image_format=image_format,
            ffmpeg_executable=ffmpeg_executable,
            ffmpeg_runner=ffmpeg_runner,
            ffprobe_executable=ffprobe_executable,
            ffprobe_runner=ffprobe_runner,
            timeout_seconds=timeout_seconds,
        ),
        loop=_accept_one(
            AssetKind.LOOP,
            loop,
            target,
            size=validated_size,
            duration_seconds=duration_seconds,
            limits=limits,
            formats=loop_formats,
            fit=fit,
            image_format=image_format,
            ffmpeg_executable=ffmpeg_executable,
            ffmpeg_runner=ffmpeg_runner,
            ffprobe_executable=ffprobe_executable,
            ffprobe_runner=ffprobe_runner,
            timeout_seconds=timeout_seconds,
        ),
        audio=_accept_one(
            AssetKind.AUDIO,
            audio,
            target,
            size=None,
            duration_seconds=duration_seconds,
            limits=limits,
            formats=audio_formats,
            fit=fit,
            image_format=image_format,
            ffmpeg_executable=ffmpeg_executable,
            ffmpeg_runner=ffmpeg_runner,
            ffprobe_executable=ffprobe_executable,
            ffprobe_runner=ffprobe_runner,
            timeout_seconds=timeout_seconds,
        ),
        logo=_accept_one(
            AssetKind.LOGO,
            logo,
            target,
            size=validated_size,
            duration_seconds=None,
            limits=limits,
            formats=logo_formats,
            fit=fit,
            image_format=image_format,
            ffmpeg_executable=ffmpeg_executable,
            ffmpeg_runner=ffmpeg_runner,
            ffprobe_executable=ffprobe_executable,
            ffprobe_runner=ffprobe_runner,
            timeout_seconds=timeout_seconds,
        ),
    )

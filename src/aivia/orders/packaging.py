"""Package order outputs into final deliverable formats.

The packager takes rendered master artifacts (a video, an audio track, a still
image and/or a composition) and produces the order's finals: MP4 video, an
animated GIF, an animated WebP, an MP3, a poster frame and a self-contained web
card. Every output can be rendered at one of the configured size presets and
every final is marked pinned so retention never removes it.

All encoding is local and deterministic: ffmpeg is invoked with an argv list and
both runners are injectable, so the offline suite never needs the real binaries.
"""

from __future__ import annotations

import io
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from aivia.compose.web_card import render_web_card
from aivia.core.config.media import (
    DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
    DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
)
from aivia.core.errors import ConfigError, ValidationError
from aivia.formats.image_post import ASPECT_PRESETS, resolve_aspect_preset
from aivia.media.video_utils import (
    FFmpegRunner,
    VideoOutputFormat,
    convert_video,
    extract_audio,
    extract_frame,
)

__all__ = [
    "DEFAULT_SIZE_PRESETS",
    "PackagedOutput",
    "Packager",
    "PackagingError",
    "SizePreset",
    "package_outputs",
]


class PackagingError(ValidationError):
    """An output could not be packaged into a final deliverable."""

    code = "packaging"


@dataclass(frozen=True, slots=True)
class SizePreset:
    """One named output shape, with the pixel canvas it renders into."""

    name: str
    width: int
    height: int
    label: str = ""

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise PackagingError("size preset dimensions must be positive")

    @property
    def size(self) -> tuple[int, int]:
        """The canvas as ``(width, height)``."""
        return self.width, self.height

    @property
    def aspect_ratio(self) -> float:
        """The width divided by the height."""
        return self.width / self.height


def _builtin_presets() -> dict[str, SizePreset]:
    return {
        name: SizePreset(
            name=preset.name,
            width=preset.width,
            height=preset.height,
            label=preset.label,
        )
        for name, preset in ASPECT_PRESETS.items()
    }


DEFAULT_SIZE_PRESETS: dict[str, SizePreset] = _builtin_presets()
"""The built-in size presets: square, portrait, story and landscape."""


@dataclass(frozen=True, slots=True)
class PackagedOutput:
    """One packaged final output, always marked pinned."""

    kind: str
    """The deliverable kind: ``mp4``, ``gif``, ``webp``, ``mp3``,
    ``poster`` or ``web_card``."""

    path: Path
    """The written output file."""

    preset: str | None
    """The size preset applied, or ``None`` when the source size was kept."""

    width: int
    """The output width in pixels."""

    height: int
    """The output height in pixels."""

    size_bytes: int
    """The size of the written file in bytes."""

    format: str
    """The container format, without a leading dot."""

    source: str
    """The source the output was packaged from."""

    pinned: bool = True
    """Finals are always pinned; retention must never remove them."""

    @property
    def is_pinned(self) -> bool:
        """Alias for :attr:`pinned`."""
        return self.pinned

    @property
    def extension(self) -> str:
        """The file suffix without a leading dot."""
        return self.path.suffix.lstrip(".")

    def __fspath__(self) -> str:
        return str(self.path)

    def __str__(self) -> str:
        return str(self.path)


_DEFAULT_OUTPUTS: dict[str, str] = {
    "mp4": "video.mp4",
    "gif": "animation.gif",
    "webp": "animation.webp",
    "mp3": "audio.mp3",
    "poster": "poster.png",
    "web_card": "card.html",
}

_EXTENSION_BY_KIND: dict[str, str] = {
    "mp4": "mp4",
    "gif": "gif",
    "webp": "webp",
    "mp3": "mp3",
    "poster": "png",
    "web_card": "html",
}

_KINDS = frozenset(_DEFAULT_OUTPUTS)


class Packager:
    """Package rendered master artifacts into pinned final outputs.

    ``destination`` is the directory finals are written into. ``presets``
    overrides the built-in size presets. The ffmpeg and ffprobe runners are
    injectable so the packager can be exercised without the real binaries.
    """

    def __init__(
        self,
        destination: str | Path,
        *,
        presets: Mapping[str, SizePreset] | None = None,
        ffmpeg_executable: str = "ffmpeg",
        ffprobe_executable: str = "ffprobe",
        ffmpeg_runner: FFmpegRunner | None = None,
        ffprobe_runner: FFmpegRunner | None = None,
        timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
        ffprobe_timeout_seconds: float = DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
    ) -> None:
        self.destination = Path(destination)
        self.presets = (
            dict(presets) if presets is not None else dict(DEFAULT_SIZE_PRESETS)
        )
        self.ffmpeg_executable = ffmpeg_executable
        self.ffprobe_executable = ffprobe_executable
        self.ffmpeg_runner = ffmpeg_runner
        self.ffprobe_runner = ffprobe_runner
        self.timeout_seconds = timeout_seconds
        self.ffprobe_timeout_seconds = ffprobe_timeout_seconds

    def _output_path(self, kind: str, output: str | Path | None) -> Path:
        if output is not None:
            return Path(output)
        return self.destination / _DEFAULT_OUTPUTS[kind]

    def _preset_dimensions(
        self, preset: str | SizePreset | None
    ) -> tuple[int, int] | None:
        if preset is None:
            return None
        if isinstance(preset, SizePreset):
            return preset.size
        key = str(preset).strip()
        if not key:
            return None
        resolved = self.presets.get(key)
        if resolved is None:
            try:
                aspect = resolve_aspect_preset(key)
            except (ValidationError, ConfigError) as exc:
                raise PackagingError(
                    "unknown size preset "
                    f"{key!r}; expected one of {sorted(self.presets)}",
                ) from exc
            resolved = SizePreset(
                name=aspect.name,
                width=aspect.width,
                height=aspect.height,
                label=aspect.label,
            )
        return resolved.size

    def _check_not_source(self, source: Any, destination: Path) -> None:
        if (
            isinstance(source, (str, Path))
            and Path(source).resolve() == destination.resolve()
        ):
            raise PackagingError("destination must not overwrite the source")

    @contextmanager
    def _materialize(
        self, source: Any, *, prefix: str = "aivia-packaging-"
    ) -> Iterator[Path]:
        if isinstance(source, (bytes, bytearray, memoryview)):
            payload = bytes(source)
            if not payload:
                raise PackagingError("packaging source is empty")
            with tempfile.TemporaryDirectory(prefix=prefix) as directory:
                path = Path(directory) / "source"
                path.write_bytes(payload)
                yield path
            return
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.is_file():
                raise PackagingError(
                    f"packaging source does not exist: {path}",
                    details={"path": str(path)},
                )
            yield path
            return
        raise TypeError("packaging source must be bytes or a path")

    def _size_bytes(self, path: Path) -> int:
        return path.stat().st_size if path.is_file() else 0

    def _record(
        self,
        kind: str,
        path: Path,
        preset: str | SizePreset | None,
        width: int,
        height: int,
        source: Any,
    ) -> PackagedOutput:
        dimensions = self._preset_dimensions(preset)
        if dimensions is not None:
            width, height = dimensions
        preset_name = (
            preset.name if isinstance(preset, SizePreset) else (str(preset) or None)
        )
        if isinstance(source, (str, Path)):
            source_text = str(source)
        elif isinstance(source, (bytes, bytearray, memoryview)):
            source_text = f"{len(bytes(source))} bytes"
        else:
            source_text = ""
        return PackagedOutput(
            kind=kind,
            path=path,
            preset=preset_name,
            width=width,
            height=height,
            size_bytes=self._size_bytes(path),
            format=_EXTENSION_BY_KIND[kind],
            source=source_text,
        )

    def package_mp4(
        self,
        source: Any,
        *,
        output: str | Path | None = None,
        preset: str | SizePreset | None = None,
        fps: float | None = None,
        loop: int = 0,
        quality: int | None = None,
        crf: int | None = None,
        keep_audio: bool = True,
        audio_bitrate_kbps: int = 192,
    ) -> PackagedOutput:
        """Package *source* as an MP4 video, optionally at a size preset."""
        destination = self._output_path("mp4", output)
        self._check_not_source(source, destination)
        dimensions = self._preset_dimensions(preset)
        width = dimensions[0] if dimensions else None
        height = dimensions[1] if dimensions else None
        self.destination.mkdir(parents=True, exist_ok=True)
        with self._materialize(source) as active:
            convert_video(
                active,
                destination,
                output_format=VideoOutputFormat.MP4,
                fps=fps,
                width=width,
                height=height,
                loop=loop,
                quality=quality,
                crf=crf,
                keep_audio=keep_audio,
                audio_bitrate_kbps=audio_bitrate_kbps,
                ffmpeg_executable=self.ffmpeg_executable,
                ffmpeg_runner=self.ffmpeg_runner,
                timeout_seconds=self.timeout_seconds,
            )
        return self._record("mp4", destination, preset, width or 0, height or 0, source)

    def package_gif(
        self,
        source: Any,
        *,
        output: str | Path | None = None,
        preset: str | SizePreset | None = None,
        fps: float | None = None,
        loop: int = 0,
        quality: int | None = None,
        crf: int | None = None,
    ) -> PackagedOutput:
        """Package *source* as an animated GIF, optionally at a size preset."""
        destination = self._output_path("gif", output)
        self._check_not_source(source, destination)
        dimensions = self._preset_dimensions(preset)
        width = dimensions[0] if dimensions else None
        height = dimensions[1] if dimensions else None
        self.destination.mkdir(parents=True, exist_ok=True)
        with self._materialize(source) as active:
            convert_video(
                active,
                destination,
                output_format=VideoOutputFormat.GIF,
                fps=fps,
                width=width,
                height=height,
                loop=loop,
                quality=quality,
                crf=crf,
                keep_audio=False,
                ffmpeg_executable=self.ffmpeg_executable,
                ffmpeg_runner=self.ffmpeg_runner,
                timeout_seconds=self.timeout_seconds,
            )
        return self._record("gif", destination, preset, width or 0, height or 0, source)

    def package_webp(
        self,
        source: Any,
        *,
        output: str | Path | None = None,
        preset: str | SizePreset | None = None,
        fps: float | None = None,
        loop: int = 0,
        quality: int | None = None,
        crf: int | None = None,
    ) -> PackagedOutput:
        """Package *source* as an animated WebP, optionally at a size preset."""
        destination = self._output_path("webp", output)
        self._check_not_source(source, destination)
        dimensions = self._preset_dimensions(preset)
        width = dimensions[0] if dimensions else None
        height = dimensions[1] if dimensions else None
        self.destination.mkdir(parents=True, exist_ok=True)
        with self._materialize(source) as active:
            convert_video(
                active,
                destination,
                output_format=VideoOutputFormat.WEBP,
                fps=fps,
                width=width,
                height=height,
                loop=loop,
                quality=quality,
                crf=crf,
                keep_audio=False,
                ffmpeg_executable=self.ffmpeg_executable,
                ffmpeg_runner=self.ffmpeg_runner,
                timeout_seconds=self.timeout_seconds,
            )
        return self._record(
            "webp", destination, preset, width or 0, height or 0, source
        )

    def package_mp3(
        self,
        source: Any,
        *,
        output: str | Path | None = None,
        bitrate_kbps: int = 192,
    ) -> PackagedOutput:
        """Package *source* as an MP3 audio track."""
        destination = self._output_path("mp3", output)
        self._check_not_source(source, destination)
        self.destination.mkdir(parents=True, exist_ok=True)
        with self._materialize(source) as active:
            extract_audio(
                active,
                destination,
                bitrate_kbps=bitrate_kbps,
                ffmpeg_executable=self.ffmpeg_executable,
                ffmpeg_runner=self.ffmpeg_runner,
                timeout_seconds=self.timeout_seconds,
            )
        return self._record("mp3", destination, None, 0, 0, source)

    def _open_image(self, source: Any) -> Image.Image | None:
        try:
            if isinstance(source, (bytes, bytearray, memoryview)):
                with Image.open(io.BytesIO(bytes(source))) as opened:
                    return opened.convert("RGBA")
            if isinstance(source, (str, Path)):
                with Image.open(Path(source)) as opened:
                    return opened.convert("RGBA")
        except (OSError, UnidentifiedImageError, ValueError):
            return None
        return None

    def package_poster(
        self,
        source: Any,
        *,
        output: str | Path | None = None,
        preset: str | SizePreset | None = None,
        timestamp_seconds: float = 0.0,
    ) -> PackagedOutput:
        """Package *source* as a poster frame, optionally at a size preset.

        When *source* is a still image it is resized directly; when it is a
        video a frame is extracted with ffmpeg and then resized.
        """
        destination = self._output_path("poster", output)
        self._check_not_source(source, destination)
        dimensions = self._preset_dimensions(preset)
        width = dimensions[0] if dimensions else None
        height = dimensions[1] if dimensions else None
        self.destination.mkdir(parents=True, exist_ok=True)

        image = self._open_image(source)
        if image is None:
            temporary = Path(
                tempfile.mkstemp(
                    dir=destination.parent,
                    prefix=f".{destination.name}.",
                    suffix=".tmp",
                )[1]
            )
            try:
                with self._materialize(source) as active:
                    extract_frame(
                        active,
                        temporary,
                        timestamp_seconds=timestamp_seconds,
                        ffmpeg_executable=self.ffmpeg_executable,
                        ffmpeg_runner=self.ffmpeg_runner,
                        timeout_seconds=self.timeout_seconds,
                    )
                image = Image.open(temporary).convert("RGBA")
            finally:
                temporary.unlink(missing_ok=True)
        else:
            image = image.convert("RGBA")

        actual_width, actual_height = image.size
        if width is not None and height is not None:
            image = image.convert("RGB").resize(
                (width, height), Image.Resampling.LANCZOS
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, format="PNG")
        return self._record(
            "poster",
            destination,
            preset,
            width if width is not None else actual_width,
            height if height is not None else actual_height,
            source,
        )

    def package_web_card(
        self,
        composition: Mapping[str, Any] | Any,
        *,
        output: str | Path | None = None,
        preset: str | SizePreset | None = None,
        background_image: str | Path | bytes | None = None,
        title: str = "",
        summary: str = "",
        language: str = "en",
        play_label: str = "Play",
        animation: Any = None,
        background_color: str | None = None,
        expires_at: Any = None,
        hit_url: str | None = None,
        card_id: str = "",
        order_id: str = "",
    ) -> PackagedOutput:
        """Package *composition* as a self-contained web card.

        When *preset* and *background_image* are both given the embedded
        background is resized to the preset canvas before the card is rendered.
        """
        destination = self._output_path("web_card", output)
        self.destination.mkdir(parents=True, exist_ok=True)

        preset_dimensions = self._preset_dimensions(preset)
        resolved_background: str | Path | None = None
        if background_image is not None:
            opened = self._open_image(background_image)
            if opened is None:
                raise PackagingError(
                    "web card background image could not be read",
                    details={"background_image": str(background_image)},
                )
            if preset_dimensions is not None:
                card_width, card_height = preset_dimensions
                resized = opened.convert("RGB").resize(
                    (card_width, card_height), Image.Resampling.LANCZOS
                )
                directory = Path(tempfile.mkdtemp(prefix="aivia-packaging-"))
                resolved_background = directory / "background.png"
                resized.save(resolved_background, format="PNG")
            elif isinstance(background_image, (str, Path)):
                resolved_background = background_image
            else:
                directory = Path(tempfile.mkdtemp(prefix="aivia-packaging-"))
                resolved_background = directory / "background.png"
                opened.convert("RGB").save(resolved_background, format="PNG")

        result = render_web_card(
            composition,
            destination,
            title=title,
            summary=summary,
            language=language,
            play_label=play_label,
            animation=animation,
            background_color=background_color,
            background_image=resolved_background,
            expires_at=expires_at,
            hit_url=hit_url,
            card_id=card_id,
            order_id=order_id,
        )
        card_width, card_height = (
            preset_dimensions if preset_dimensions is not None else (0, 0)
        )
        return self._record(
            "web_card",
            result.output,
            preset,
            card_width,
            card_height,
            composition,
        )

    def package_all(
        self,
        *,
        video: Any = None,
        audio: Any = None,
        image: Any = None,
        composition: Mapping[str, Any] | Any = None,
        formats: Sequence[str] | None = None,
        preset: str | SizePreset | None = None,
    ) -> dict[str, PackagedOutput]:
        """Package every requested final from the supplied masters.

        ``video`` feeds mp4, gif, webp, mp3 and poster. ``image`` feeds poster
        and the web card background. ``composition`` feeds the web card.
        """
        requested = list(formats) if formats is not None else list(_KINDS)
        unknown = [item for item in requested if item not in _KINDS]
        if unknown:
            raise PackagingError(
                "unknown output kinds: "
                f"{sorted(unknown)}; expected one of {sorted(_KINDS)}",
                details={"kinds": requested},
            )

        outputs: dict[str, PackagedOutput] = {}
        poster_source = image if image is not None else video

        if "mp4" in requested:
            if video is None:
                raise PackagingError("a video source is required to package mp4")
            outputs["mp4"] = self.package_mp4(video, preset=preset)
        if "gif" in requested:
            if video is None:
                raise PackagingError("a video source is required to package gif")
            outputs["gif"] = self.package_gif(video, preset=preset)
        if "webp" in requested:
            if video is None:
                raise PackagingError("a video source is required to package webp")
            outputs["webp"] = self.package_webp(video, preset=preset)
        if "mp3" in requested:
            source = audio if audio is not None else video
            if source is None:
                raise PackagingError(
                    "an audio or video source is required to package mp3"
                )
            outputs["mp3"] = self.package_mp3(source)
        if "poster" in requested:
            if poster_source is None:
                raise PackagingError(
                    "a video or image source is required to package poster"
                )
            outputs["poster"] = self.package_poster(poster_source, preset=preset)
        if "web_card" in requested:
            if composition is None:
                raise PackagingError("a composition is required to package web_card")
            outputs["web_card"] = self.package_web_card(
                composition, preset=preset, background_image=image
            )
        return outputs


def package_outputs(
    destination: str | Path,
    *,
    video: Any = None,
    audio: Any = None,
    image: Any = None,
    composition: Mapping[str, Any] | Any = None,
    formats: Sequence[str] | None = None,
    preset: str | SizePreset | None = None,
    **packager_kwargs: Any,
) -> dict[str, PackagedOutput]:
    """Convenience wrapper around :class:`Packager.package_all`."""
    packager = Packager(destination, **packager_kwargs)
    return packager.package_all(
        video=video,
        audio=audio,
        image=image,
        composition=composition,
        formats=list(formats) if formats is not None else None,
        preset=preset,
    )

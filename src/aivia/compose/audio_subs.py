"""Audio mixing and subtitle burn-in for a composition.

The composition schema already describes timed audio tracks and optional
subtitles. This module turns those declarations into the two products the
FFmpeg renderer needs: a deterministic audio filtergraph and an ASS document
whose margins respect the platform safe area. It also exposes the small
helpers a caller needs to mix audio on its own or write the subtitle file
before rendering.

Everything here is pure apart from the explicit ``write`` helpers. The command
builder only receives paths, never a live file handle, so building an argv can
still be tested without touching the filesystem or executing a binary.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from aivia.core.config.video import VideoConfig
from aivia.core.errors import ValidationError
from aivia.core.platforms import Platform, SafeArea

from .contrast import contrast_subtitle_style
from .errors import RenderError
from .schema import Composition
from .subtitles import SubtitleTrack

if TYPE_CHECKING:
    from aivia.media.audio_mix import (
        AudioMixResult,
        AudioMixSettings,
        DuckingSettings,
        FFmpegRunner,
    )
    from aivia.media.audio_mix import (
        AudioTrack as MediaAudioTrack,
    )
    from aivia.media.subtitles import Subtitle, SubtitleStyleInput

__all__ = [
    "DEFAULT_LOUDNESS_LUFS",
    "AudioSubtitleError",
    "audio_filtergraph",
    "audio_inputs",
    "audio_mix_settings",
    "audio_tracks_for_mix",
    "build_audio_filtergraph",
    "build_subtitle_filter",
    "burn_subtitles",
    "mix_composition_audio",
    "subtitle_ass",
    "subtitle_document",
    "subtitle_filter",
    "subtitle_filtergraph",
    "subtitle_for",
    "write_subtitle_document",
]

DEFAULT_LOUDNESS_LUFS = VideoConfig().loudness_lufs
"""The loudness target used when neither the composition nor config names one."""


class AudioSubtitleError(RenderError):
    """Audio or subtitle settings could not be turned into a render product."""

    code = "audio_subtitles"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AudioSubtitleError(f"{name} must be a number")
    import math

    number = float(value)
    if not math.isfinite(number):
        raise AudioSubtitleError(f"{name} must be finite")
    return number


def _positive(value: Any, name: str) -> float:
    number = _finite(value, name)
    if number <= 0.0:
        raise AudioSubtitleError(f"{name} must be positive")
    return number


def audio_tracks_for_mix(composition: Composition) -> list[MediaAudioTrack]:
    """Convert enabled composition tracks into mixable media tracks.

    Composition tracks use ``start_seconds`` while the mixer calls the same
    value ``offset_seconds``. Roles, gains, fades, durations, looping and the
    ducking opt-in are carried across unchanged.
    """
    from aivia.media.audio_mix import (
        AudioTrack as MediaAudioTrack,
    )
    from aivia.media.audio_mix import (
        AudioTrackRole as MediaAudioTrackRole,
    )

    tracks: list[MediaAudioTrack] = []
    for track in composition.audio_tracks:
        if not track.enabled:
            continue
        tracks.append(
            MediaAudioTrack(
                source=track.source,
                role=MediaAudioTrackRole(track.role.value),
                duck=track.duck,
                volume_db=track.volume_db,
                offset_seconds=track.start_seconds,
                fade_in_seconds=track.fade_in_seconds,
                fade_out_seconds=track.fade_out_seconds,
                duration_seconds=track.duration_seconds,
                loop=track.loop,
            )
        )
    return tracks


def audio_mix_settings(
    composition: Composition | None = None,
    *,
    loudness_lufs: float | None = None,
    target_lufs: float | None = None,
    normalize_loudness: bool | None = None,
    true_peak_db: float | None = None,
    sample_rate: int | None = None,
    channels: int | None = None,
    duration_seconds: float | None = None,
    ducking: DuckingSettings | Mapping[str, Any] | None = None,
    config: Any | None = None,
    video: Any | None = None,
) -> AudioMixSettings:
    """Return mix settings for *composition*, normalizing to the loudness target.

    An explicit argument wins over the composition's field, which wins over the
    video config default. Mixing normalizes by default because the renderer
    promises the configured loudness, not the source track's level.
    """
    from aivia.media.audio_mix import AudioMixSettings, DuckingSettings

    value = composition
    configured = None if value is None else value.loudness_lufs
    if configured is None:
        active_config = config if config is not None else video
        if isinstance(active_config, Mapping):
            configured = active_config.get("loudness_lufs")
        elif active_config is not None:
            configured = getattr(active_config, "loudness_lufs", None)
    target = loudness_lufs
    if target is None:
        target = target_lufs
    if target is None:
        target = configured
    if target is None:
        target = DEFAULT_LOUDNESS_LUFS
    target = _finite(target, "loudness_lufs")
    if target > 0.0:
        raise AudioSubtitleError("loudness_lufs must not be positive")

    duration = duration_seconds
    if duration is None and value is not None:
        duration = value.duration_seconds
    if duration is not None:
        duration = _positive(duration, "duration_seconds")

    normalize = True if normalize_loudness is None else bool(normalize_loudness)
    overrides: dict[str, Any] = {
        "target_lufs": target,
        "normalize_loudness": normalize,
    }
    if duration is not None:
        overrides["duration_seconds"] = duration
    if true_peak_db is not None:
        overrides["true_peak_db"] = _finite(true_peak_db, "true_peak_db")
    if sample_rate is not None:
        overrides["sample_rate"] = sample_rate
    if channels is not None:
        overrides["channels"] = channels
    if ducking is not None:
        overrides["ducking"] = DuckingSettings.from_value(ducking)
    return AudioMixSettings.model_validate(overrides)


def build_audio_filtergraph(
    composition: Composition,
    *,
    settings: AudioMixSettings | Mapping[str, Any] | None = None,
    input_offset: int = 0,
    loudness_lufs: float | None = None,
    normalize_loudness: bool | None = None,
    ducking: DuckingSettings | Mapping[str, Any] | None = None,
) -> str:
    """Build the audio half of a composition's ``filter_complex``.

    Returns an empty string when every track is disabled or absent. The
    ``input_offset`` accounts for the background and visual layer inputs that
    precede the audio inputs in a combined command.
    """
    from aivia.media.audio_mix import AudioMixSettings, build_mix_filtergraph

    tracks = audio_tracks_for_mix(composition)
    if not tracks:
        return ""
    if input_offset < 0:
        raise AudioSubtitleError("input_offset must not be negative")
    active = (
        AudioMixSettings.from_value(settings)
        if settings is not None
        else audio_mix_settings(
            composition,
            loudness_lufs=loudness_lufs,
            normalize_loudness=normalize_loudness,
            ducking=ducking,
        )
    )
    if active.duration_seconds is None:
        active = active.model_copy(
            update={"duration_seconds": composition.duration_seconds}
        )
    return build_mix_filtergraph(tracks, active, input_offset=input_offset)


def _as_subtitle(
    source: (
        Subtitle
        | SubtitleTrack
        | Composition
        | Mapping[str, Any]
        | Iterable[Any]
        | str
        | Path
        | None
    ),
) -> Subtitle | None:
    from aivia.media.subtitles import Subtitle

    if source is None:
        return None
    if isinstance(source, (str, Path)):
        return Subtitle.from_file(source)
    if isinstance(source, Subtitle):
        return source
    if isinstance(source, SubtitleTrack):
        return cast(Subtitle, source.to_media())
    if isinstance(source, Composition):
        referenced = source.subtitle_file.strip()
        external = bool(referenced) and Path(referenced).suffix.lower() in {
            ".srt",
            ".vtt",
        }
        if source.subtitles is not None:
            track = cast(Subtitle, source.subtitles.to_media())
            if track.cues or not external:
                return track
            loaded = Subtitle.from_file(referenced)
            loaded.style = track.style
            return loaded
        if external:
            return Subtitle.from_file(referenced)
        return None
    if isinstance(source, Mapping):
        return Subtitle.model_validate(dict(source))
    items = list(source)
    if not items:
        return None
    return Subtitle.model_validate({"cues": items})


def subtitle_for(
    composition: Composition,
    subtitles: (
        Subtitle | SubtitleTrack | Mapping[str, Any] | Iterable[Any] | str | Path | None
    ) = None,
) -> Subtitle | None:
    """Return the explicit subtitle track or the composition's own track."""
    explicit = _as_subtitle(subtitles)
    if explicit is not None:
        return explicit
    return _as_subtitle(composition)


def subtitle_document(
    source: (
        Subtitle
        | SubtitleTrack
        | Composition
        | Mapping[str, Any]
        | Iterable[Any]
        | str
        | Path
        | None
    ),
    *,
    width: int | None = None,
    height: int | None = None,
    style: SubtitleStyleInput = None,
    preset: str | None = None,
    safe_area: SafeArea | Mapping[str, float] | None = None,
    platform: Platform | str | None = None,
    highlight: bool | str = True,
    background: str | None = None,
    render: Any | None = None,
    config: Any | None = None,
) -> str:
    """Return the ASS document that should be burned into the video.

    A composition supplies its canvas, platform and safe area unless the
    caller overrides them. An absent subtitle track produces an empty string.
    When a *background* is known (from the composition canvas by default), the
    subtitle style takes the contrast treatment named by the render config.
    """
    track = _as_subtitle(source)
    if track is None:
        return ""
    if isinstance(source, Composition):
        width = source.canvas.width if width is None else width
        height = source.canvas.height if height is None else height
        if safe_area is None:
            safe_area = source.safe_area
        if platform is None and source.platform:
            platform = source.platform
        if background is None:
            background = source.canvas.background
    if width is None:
        width = 1920
    if height is None:
        height = 1080
    chosen_style = style
    if background is not None:
        from aivia.media.subtitles import resolve_style

        chosen = resolve_style(track.style if style is None else style)
        if preset is not None:
            chosen = resolve_style(preset)
            preset = None
        chosen_style = contrast_subtitle_style(
            chosen,
            background,
            render=render,
            config=config,
        )
    return track.to_ass(
        width=width,
        height=height,
        style=chosen_style,
        preset=preset,
        safe_area=safe_area,
        platform=platform,
        highlight=highlight,
    )


subtitle_ass = subtitle_document
"""Alias for :func:`subtitle_document`."""


def write_subtitle_document(
    source: (
        Subtitle
        | SubtitleTrack
        | Composition
        | Mapping[str, Any]
        | Iterable[Any]
        | str
        | Path
        | None
    ),
    path: str | Path,
    **options: Any,
) -> Path:
    """Write the burned ASS document to *path* and return the path."""
    document = subtitle_document(source, **options)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    return destination


burn_subtitles = write_subtitle_document
"""Alias for :func:`write_subtitle_document`."""


def _escape_filter_path(value: str | Path) -> str:
    return str(value).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def subtitle_filter(path: str | Path) -> str:
    """Return the FFmpeg ``ass`` filter that burns the ASS file at *path*."""
    if not str(path).strip():
        raise AudioSubtitleError("subtitle path must not be empty")
    return f"ass=filename='{_escape_filter_path(path)}'"


build_subtitle_filter = subtitle_filter
"""Alias for :func:`subtitle_filter`."""

subtitle_filtergraph = subtitle_filter
"""Alias for :func:`subtitle_filter`."""


def mix_composition_audio(
    composition: Composition,
    destination: str | Path,
    *,
    settings: AudioMixSettings | Mapping[str, Any] | None = None,
    loudness_lufs: float | None = None,
    normalize_loudness: bool | None = None,
    ducking: DuckingSettings | Mapping[str, Any] | None = None,
    ffmpeg_executable: str = "ffmpeg",
    ffmpeg_runner: FFmpegRunner | None = None,
    timeout_seconds: float | None = None,
) -> AudioMixResult:
    """Mix a composition's audio tracks into a standalone file."""
    from aivia.media.audio_mix import AudioMixSettings, mix_audio

    tracks = audio_tracks_for_mix(composition)
    if not tracks:
        raise AudioSubtitleError("a composition needs an enabled audio track to mix")
    active = (
        AudioMixSettings.from_value(settings)
        if settings is not None
        else audio_mix_settings(
            composition,
            loudness_lufs=loudness_lufs,
            normalize_loudness=normalize_loudness,
            ducking=ducking,
        )
    )
    options: dict[str, Any] = {}
    if timeout_seconds is not None:
        options["timeout_seconds"] = timeout_seconds
    try:
        return mix_audio(
            tracks,
            destination,
            active,
            ffmpeg_executable=ffmpeg_executable,
            ffmpeg_runner=ffmpeg_runner,
            **options,
        )
    except ValidationError as exc:
        raise AudioSubtitleError(
            "the composition audio could not be mixed",
            details={"error": exc.message},
        ) from exc


audio_filtergraph = build_audio_filtergraph
"""Alias for :func:`build_audio_filtergraph`."""

audio_inputs = audio_tracks_for_mix
"""Alias for :func:`audio_tracks_for_mix`."""

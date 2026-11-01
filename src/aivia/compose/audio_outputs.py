"""Render a composition to an audio-only file with metadata.

Audio-only output reuses the same deterministic mix graph as a video render but
drops every visual input. The output extension selects the container and its
default encoder, the composition duration is enforced with silence padding and
trimming, and requested tags are written as ID3, Vorbis comment or iTunes
metadata depending on the container. The file is written through a temporary
path and verified with ffprobe before publication.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aivia.core.config.media import (
    DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
    DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
)
from aivia.core.errors import ValidationError

from .audio_subs import audio_mix_settings, build_audio_filtergraph
from .errors import RenderError
from .ffmpeg_cmd import FFmpegCommand, FFmpegInput
from .output_profiles import (
    OutputProfile,
    apply_output_profile,
    resolve_output_profile,
)
from .schema import Composition

if TYPE_CHECKING:
    from aivia.core.media_probe import FFprobeInfo, FFprobeRunner

__all__ = [
    "AUDIO_RENDERER_VERSION",
    "AudioMetadata",
    "AudioOutputError",
    "AudioRenderResult",
    "AudioRenderer",
    "audio_output_format",
    "build_audio_command",
    "export_audio",
    "render_audio",
]

AUDIO_RENDERER_VERSION = "1"
"""The version of the audio-only renderer, recorded with every output."""

AudioFFmpegRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
"""A callable that runs an ffmpeg argv and returns the completed process."""

_AUDIO_FORMATS: dict[str, str] = {
    "mp3": "libmp3lame",
    "wav": "pcm_s16le",
    "flac": "flac",
    "ogg": "libvorbis",
    "opus": "libopus",
    "m4a": "aac",
    "aac": "aac",
}

_FORMAT_ALIASES: dict[str, str] = {
    "oga": "ogg",
    "wave": "wav",
    "mp4": "m4a",
}

_COMPATIBLE_CODECS: dict[str, frozenset[str]] = {
    "mp3": frozenset({"libmp3lame", "mp3"}),
    "wav": frozenset({"pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_f32le", "pcm"}),
    "flac": frozenset({"flac"}),
    "ogg": frozenset({"libvorbis", "vorbis"}),
    "opus": frozenset({"libopus", "opus"}),
    "m4a": frozenset({"aac", "libfdk_aac", "alac"}),
    "aac": frozenset({"aac", "libfdk_aac"}),
}

_CODEC_NAMES: dict[str, str] = {
    "libmp3lame": "mp3",
    "mp3": "mp3",
    "pcm": "pcm_s16le",
    "libvorbis": "vorbis",
    "vorbis": "vorbis",
    "libopus": "opus",
    "opus": "opus",
    "libfdk_aac": "aac",
    "aac": "aac",
}

_LOSSLESS_CODECS = frozenset(
    {"pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_f32le", "flac"}
)

_METADATA_FIELDS = (
    "title",
    "artist",
    "album",
    "album_artist",
    "genre",
    "comment",
    "date",
    "track",
    "composer",
    "copyright",
)

_METADATA_ALIASES: dict[str, str] = {
    "albumartist": "album_artist",
    "album-artist": "album_artist",
    "year": "date",
    "tracknumber": "track",
    "track_number": "track",
}


class AudioOutputError(RenderError):
    """A composition could not be rendered as a verified audio-only file."""

    code = "audio_output"


def _metadata_text(value: Any, name: str) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    if not isinstance(value, str):
        raise AudioOutputError(f"metadata {name!r} must be text")
    text = value.strip()
    if "\x00" in text:
        raise AudioOutputError(f"metadata {name!r} must not contain NUL")
    return text


@dataclass(frozen=True, slots=True)
class AudioMetadata:
    """The tags written into an audio-only output."""

    title: str = ""
    artist: str = ""
    album: str = ""
    album_artist: str = ""
    genre: str = ""
    comment: str = ""
    date: str = ""
    track: str = ""
    composer: str = ""
    copyright: str = ""
    extra: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in _METADATA_FIELDS:
            object.__setattr__(
                self,
                name,
                _metadata_text(getattr(self, name), name),
            )
        cleaned: dict[str, str] = {}
        for key, value in self.extra.items():
            name = _metadata_text(key, "extra key").lower()
            if not name:
                continue
            if "=" in name:
                raise AudioOutputError("metadata keys must not contain '='")
            cleaned[name] = _metadata_text(value, name)
        object.__setattr__(self, "extra", cleaned)

    @classmethod
    def from_value(
        cls, value: AudioMetadata | Mapping[str, Any] | None
    ) -> AudioMetadata:
        """Coerce ``None``, a mapping or an instance into metadata."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise AudioOutputError("metadata must be a mapping or AudioMetadata")
        known = {item.name for item in fields(cls) if item.name != "extra"}
        values: dict[str, str] = {}
        extra: dict[str, str] = {}
        for raw_key, raw_value in value.items():
            key = _metadata_text(raw_key, "metadata key").lower().replace(" ", "_")
            key = _METADATA_ALIASES.get(key, key)
            if key in known:
                values[key] = _metadata_text(raw_value, key)
            else:
                extra[key] = _metadata_text(raw_value, key)
        return cls(**values, extra=extra)

    def as_tags(self) -> dict[str, str]:
        """Return the non-empty tags in the order they are written."""
        tags: dict[str, str] = {}
        for name in _METADATA_FIELDS:
            value = getattr(self, name)
            if value:
                tags[name] = value
        for key, value in self.extra.items():
            if key and value:
                tags[key] = value
        return tags

    def __bool__(self) -> bool:
        return bool(self.as_tags())


@dataclass(frozen=True, slots=True)
class AudioRenderResult:
    """A rendered audio file, the command that produced it and its probe."""

    output: Path
    command: FFmpegCommand
    probe: FFprobeInfo
    format: str
    codec: str
    metadata: AudioMetadata

    @property
    def path(self) -> Path:
        """Alias for :attr:`output`."""
        return self.output

    @property
    def duration_seconds(self) -> float | None:
        """The probed duration of the rendered file."""
        return self.probe.duration_seconds

    @property
    def tags(self) -> dict[str, str]:
        """The probed tags, with stream tags filling gaps in format tags."""
        tags = dict(self.probe.tags)
        for stream in self.probe.audio_streams:
            for key, value in stream.tags.items():
                tags.setdefault(key, value)
        return tags

    def __fspath__(self) -> str:
        return str(self.output)

    def __str__(self) -> str:
        return str(self.output)


def _format_text(value: Any) -> str:
    if value is None:
        return ""
    text = getattr(value, "value", value)
    return str(text).strip().lower().lstrip(".")


def audio_output_format(
    output: str | Path,
    output_format: str | None = None,
) -> str:
    """Return the canonical audio container for *output*.

    An explicit *output_format* wins over the file suffix. Common aliases such
    as ``oga`` and ``wave`` resolve to their canonical name.
    """
    fmt = _format_text(output_format) or _format_text(Path(output).suffix)
    fmt = _FORMAT_ALIASES.get(fmt, fmt)
    if not fmt:
        raise AudioOutputError(
            "cannot infer an audio format; use a .mp3, .m4a, .wav, .flac, "
            ".ogg or .opus output or pass output_format"
        )
    if fmt not in _AUDIO_FORMATS:
        allowed = ", ".join(sorted(_AUDIO_FORMATS))
        raise AudioOutputError(
            f"unsupported audio output format {fmt!r}; expected one of {allowed}"
        )
    return fmt


def _codec_for(fmt: str, requested: str | None) -> str:
    if requested is None:
        return _AUDIO_FORMATS[fmt]
    codec = requested.strip()
    if not codec:
        raise AudioOutputError("audio_codec must be a non-empty string")
    if codec not in _COMPATIBLE_CODECS[fmt]:
        allowed = ", ".join(sorted(_COMPATIBLE_CODECS[fmt]))
        raise AudioOutputError(
            f"codec {codec!r} cannot be written into a {fmt!r} container; "
            f"expected one of {allowed}"
        )
    return codec


def _expected_codec_name(codec: str) -> str:
    if codec in _CODEC_NAMES:
        return _CODEC_NAMES[codec]
    if codec.startswith("pcm_"):
        return codec
    return codec


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AudioOutputError(f"{name} must be a positive integer")
    return value


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AudioOutputError(f"{name} must be a positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise AudioOutputError(f"{name} must be a positive number")
    return number


def _loudness(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AudioOutputError("loudness_lufs must be a number")
    number = float(value)
    if not math.isfinite(number) or number > 0.0:
        raise AudioOutputError("loudness_lufs must be finite and not positive")
    return number


def _composition(value: Composition | Mapping[str, Any]) -> Composition:
    if isinstance(value, Composition):
        return value
    return Composition.model_validate(value)


def _audio_inputs(composition: Composition) -> tuple[FFmpegInput, ...]:
    inputs: list[FFmpegInput] = []
    for track in composition.audio_tracks:
        if not track.enabled:
            continue
        options = ("-stream_loop", "-1") if track.loop else ()
        inputs.append(
            FFmpegInput(
                source=track.source,
                options=options,
                layer_id=track.id,
            )
        )
    return tuple(inputs)


def _settings(
    composition: Composition,
    *,
    audio_settings: Any | None,
    loudness_lufs: float | None,
    normalize_loudness: bool | None,
    ducking: Any | None,
    sample_rate: int | None,
    channels: int | None,
) -> Any:
    from aivia.media.audio_mix import AudioMixSettings, DuckingSettings

    overrides: dict[str, Any] = {}
    if sample_rate is not None:
        overrides["sample_rate"] = sample_rate
    if channels is not None:
        overrides["channels"] = channels
    if loudness_lufs is not None:
        overrides["target_lufs"] = loudness_lufs
        overrides.setdefault("normalize_loudness", True)
    if normalize_loudness is not None:
        overrides["normalize_loudness"] = bool(normalize_loudness)
    if ducking is not None:
        overrides["ducking"] = DuckingSettings.from_value(ducking)
    if audio_settings is not None:
        base = AudioMixSettings.from_value(audio_settings)
        return base.model_copy(update=overrides) if overrides else base
    return audio_mix_settings(
        composition,
        loudness_lufs=loudness_lufs,
        normalize_loudness=normalize_loudness,
        sample_rate=sample_rate,
        channels=channels,
        ducking=ducking,
        duration_seconds=composition.duration_seconds,
    )


def build_audio_command(
    composition: Composition | Mapping[str, Any],
    output: str | Path = "out.mp3",
    *,
    executable: str = "ffmpeg",
    audio_codec: str | None = None,
    audio_bitrate: int | None = None,
    sample_rate: int | None = None,
    channels: int | None = None,
    metadata: AudioMetadata | Mapping[str, Any] | None = None,
    loudness_lufs: float | None = None,
    normalize_loudness: bool | None = None,
    ducking: Any | None = None,
    audio_settings: Any | None = None,
    output_format: str | None = None,
    output_profile: OutputProfile | Mapping[str, Any] | None = None,
    config: Any | None = None,
    video: Any | None = None,
    platform: str | None = None,
    profile: str | None = None,
    overwrite: bool = True,
) -> FFmpegCommand:
    """Build an audio-only ffmpeg argv for *composition* without executing it.

    The output suffix selects the container and its default encoder. An
    explicit *audio_codec* must be compatible with that container. The
    composition's duration is enforced by padding silence and trimming, and
    every requested metadata tag is written to the output.
    """
    if not isinstance(executable, str) or not executable.strip():
        raise AudioOutputError("executable must be a non-empty string")
    value = _composition(composition)
    fmt = audio_output_format(output, output_format)
    resolved: OutputProfile | None = None
    if output_profile is not None or config is not None or video is not None:
        resolved = (
            output_profile
            if isinstance(output_profile, OutputProfile)
            else (
                OutputProfile.from_mapping(output_profile)
                if output_profile is not None
                else resolve_output_profile(
                    config if config is not None else video,
                    platform=platform,
                    profile=profile,
                )
            )
        )
        value = apply_output_profile(value, resolved)
        if loudness_lufs is None:
            loudness_lufs = resolved.loudness_lufs
        if audio_bitrate is None:
            audio_bitrate = resolved.audio_bitrate
        if audio_codec is None and resolved.audio_codec in _COMPATIBLE_CODECS[fmt]:
            audio_codec = resolved.audio_codec

    codec = _codec_for(fmt, audio_codec)
    if audio_bitrate is not None:
        audio_bitrate = _positive_int(audio_bitrate, "audio_bitrate")
    if sample_rate is not None:
        sample_rate = _positive_int(sample_rate, "sample_rate")
    if channels is not None:
        channels = _positive_int(channels, "channels")
    if loudness_lufs is not None:
        loudness_lufs = _loudness(loudness_lufs)

    inputs = _audio_inputs(value)
    if not inputs:
        raise AudioOutputError(
            "an audio-only output needs at least one enabled audio track"
        )
    active = _settings(
        value,
        audio_settings=audio_settings,
        loudness_lufs=loudness_lufs,
        normalize_loudness=normalize_loudness,
        ducking=ducking,
        sample_rate=sample_rate,
        channels=channels,
    )
    graph = build_audio_filtergraph(
        value,
        settings=active,
        input_offset=0,
        loudness_lufs=loudness_lufs,
        normalize_loudness=normalize_loudness,
        ducking=ducking,
    )
    if not graph:
        raise AudioOutputError(
            "an audio-only output needs at least one enabled audio track"
        )
    duration = value.duration_seconds
    graph = (
        f"{graph};[out]apad=whole_dur={duration:.6f},"
        f"atrim=duration={duration:.6f}[aout]"
    )

    tags = AudioMetadata.from_value(metadata).as_tags()
    argv: list[str] = [executable.strip(), "-hide_banner", "-loglevel", "error"]
    if overwrite:
        argv.append("-y")
    for item in inputs:
        argv.extend(item.argv())
    argv.extend(
        [
            "-filter_complex",
            graph,
            "-map",
            "[aout]",
            "-t",
            f"{duration:.6f}",
            "-vn",
            "-map_metadata",
            "-1",
            "-c:a",
            codec,
        ]
    )
    if audio_bitrate is not None and codec not in _LOSSLESS_CODECS:
        argv.extend(["-b:a", f"{audio_bitrate}k"])
    if sample_rate is not None:
        argv.extend(["-ar", str(sample_rate)])
    if channels is not None:
        argv.extend(["-ac", str(channels)])
    if fmt == "mp3":
        argv.extend(["-id3v2_version", "3", "-write_id3v1", "1"])
    for key, value_text in tags.items():
        argv.extend(["-metadata", f"{key}={value_text}"])
    argv.append(str(output))
    return FFmpegCommand(
        argv,
        filtergraph=graph,
        output=str(output),
        inputs=inputs,
    )


class AudioRenderer:
    """Execute the pure audio command and verify the resulting file."""

    version = AUDIO_RENDERER_VERSION

    def __init__(
        self,
        *,
        ffmpeg_executable: str = "ffmpeg",
        ffprobe_executable: str = "ffprobe",
        timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
        ffprobe_timeout_seconds: float = DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
        ffmpeg_runner: AudioFFmpegRunner | None = None,
        ffprobe_runner: FFprobeRunner | None = None,
    ) -> None:
        if not isinstance(ffmpeg_executable, str) or not ffmpeg_executable.strip():
            raise AudioOutputError("ffmpeg_executable must be a non-empty string")
        if not isinstance(ffprobe_executable, str) or not ffprobe_executable.strip():
            raise AudioOutputError("ffprobe_executable must be a non-empty string")
        self.ffmpeg_executable = ffmpeg_executable.strip()
        self.ffprobe_executable = ffprobe_executable.strip()
        self.timeout_seconds = _positive_number(timeout_seconds, "timeout_seconds")
        self.ffprobe_timeout_seconds = _positive_number(
            ffprobe_timeout_seconds, "ffprobe_timeout_seconds"
        )
        self.ffmpeg_runner = ffmpeg_runner
        self.ffprobe_runner = ffprobe_runner

    def _run(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if self.ffmpeg_runner is not None:
            return self.ffmpeg_runner(argv)
        try:
            return subprocess.run(
                list(argv),
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise AudioOutputError(
                f"ffmpeg is not available: {argv[0]}",
                details={"executable": argv[0]},
            ) from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise AudioOutputError(
                "ffmpeg could not render the audio",
                details={"executable": argv[0], "error": type(exc).__name__},
            ) from exc

    def _verify(
        self,
        *,
        duration: float,
        codec: str,
        metadata: AudioMetadata,
        probe: FFprobeInfo,
    ) -> None:
        streams = probe.audio_streams
        if not streams:
            raise AudioOutputError("the rendered file has no audio stream")
        if probe.video_streams:
            raise AudioOutputError(
                "an audio-only output must not contain a video stream"
            )
        stream = streams[0]
        expected_codec = _expected_codec_name(codec)
        if stream.codec_name != expected_codec:
            raise AudioOutputError(
                "the rendered audio codec does not match the requested codec",
                details={"expected": expected_codec, "actual": stream.codec_name},
            )
        actual = probe.duration_seconds
        if actual is None:
            actual = stream.duration_seconds
        tolerance = max(0.15, duration * 0.02)
        if actual is None or abs(actual - duration) > tolerance:
            raise AudioOutputError(
                "the rendered audio duration does not match the composition",
                details={"expected": duration, "actual": actual},
            )
        tags = dict(probe.tags)
        for item in streams:
            for key, value in item.tags.items():
                tags.setdefault(key, value)
        for key, expected in metadata.as_tags().items():
            actual_value = tags.get(key.lower())
            if actual_value != expected:
                raise AudioOutputError(
                    "the rendered audio is missing its metadata",
                    details={
                        "tag": key,
                        "expected": expected,
                        "actual": actual_value,
                    },
                )

    def render(
        self,
        composition: Composition | Mapping[str, Any],
        output: str | Path = "out.mp3",
        **command_options: Any,
    ) -> AudioRenderResult:
        """Render *composition* to *output* and verify it with ffprobe."""
        value = _composition(composition)
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        options = dict(command_options)
        options.pop("executable", None)
        fmt = audio_output_format(destination, options.get("output_format"))
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}-",
            suffix=f".{fmt}",
            dir=destination.parent,
        )
        os.close(handle)
        temporary = Path(temporary_name)
        try:
            command = build_audio_command(
                value,
                temporary,
                executable=self.ffmpeg_executable,
                **options,
            )
            completed = self._run(command.argv)
            if completed.returncode != 0:
                raise AudioOutputError(
                    "ffmpeg could not render the audio",
                    details={
                        "returncode": completed.returncode,
                        "stderr": _text(completed.stderr)[:500],
                    },
                )
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise AudioOutputError("ffmpeg produced no output")
            from aivia.core.media_probe import run_ffprobe

            try:
                probe = run_ffprobe(
                    temporary,
                    executable=self.ffprobe_executable,
                    timeout_seconds=self.ffprobe_timeout_seconds,
                    runner=self.ffprobe_runner,
                )
            except ValidationError as exc:
                raise AudioOutputError(
                    "the rendered audio could not be inspected",
                    details={"error": exc.message},
                ) from exc
            codec = command[command.index("-c:a") + 1]
            duration = float(command[command.index("-t") + 1])
            metadata = AudioMetadata.from_value(options.get("metadata"))
            self._verify(
                duration=duration,
                codec=codec,
                metadata=metadata,
                probe=probe,
            )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return AudioRenderResult(
            output=destination,
            command=command,
            probe=probe,
            format=fmt,
            codec=codec,
            metadata=metadata,
        )


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def render_audio(
    composition: Composition | Mapping[str, Any],
    output: str | Path = "out.mp3",
    *,
    ffmpeg_executable: str = "ffmpeg",
    ffprobe_executable: str = "ffprobe",
    timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
    ffprobe_timeout_seconds: float = DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
    ffmpeg_runner: AudioFFmpegRunner | None = None,
    ffprobe_runner: FFprobeRunner | None = None,
    **command_options: Any,
) -> AudioRenderResult:
    """Render *composition* to an audio-only file and verify it."""
    renderer = AudioRenderer(
        ffmpeg_executable=ffmpeg_executable,
        ffprobe_executable=ffprobe_executable,
        timeout_seconds=timeout_seconds,
        ffprobe_timeout_seconds=ffprobe_timeout_seconds,
        ffmpeg_runner=ffmpeg_runner,
        ffprobe_runner=ffprobe_runner,
    )
    return renderer.render(composition, output, **command_options)


export_audio = render_audio

"""Resolve the output specification from configuration and apply it.

The video configuration owns the defaults for resolution, frame rate, codec,
bitrate, loudness and the duration bounds. Platforms and content profiles may
override any of those values. This module turns the merged configuration into
one immutable :class:`OutputProfile` and applies it to a composition before
rendering, so the renderer never reads configuration on its own.

Resolution and frame rate are canvas properties; the configured duration range
clamps the composition duration (a short composition is extended, a long one is
truncated) and the loudness target becomes the composition's target. Codec and
bitrate values are exposed as ffmpeg command options.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aivia.core.config.video import (
    DEFAULT_AUDIO_CODEC,
    DEFAULT_VIDEO_CODEC,
    VideoConfig,
)

from .errors import RenderError
from .schema import Composition

__all__ = [
    "AspectRatioWarning",
    "OutputProfile",
    "OutputProfileError",
    "OutputProfileWarning",
    "apply_output_profile",
    "apply_video_config",
    "aspect_ratio_changed",
    "ffmpeg_options",
    "has_absolute_layer_geometry",
    "output_profile",
    "output_profile_for",
    "resolve_output_profile",
    "validate_output_container",
    "warn_aspect_ratio_change",
]


class OutputProfileError(RenderError):
    """A configured output profile could not be resolved or applied."""

    code = "output_profile"


class OutputProfileWarning(UserWarning):
    """Advisory warning about an output profile applied to a composition."""


class AspectRatioWarning(OutputProfileWarning):
    """An output profile changes the aspect ratio of pixel-positioned layers."""


_ABSOLUTE_GEOMETRY_FIELDS: tuple[str, ...] = (
    "x",
    "y",
    "width",
    "height",
    "position",
    "size",
)

_ABSOLUTE_GEOMETRY_ALIASES: tuple[str, ...] = (
    "x",
    "left",
    "y",
    "top",
    "width",
    "w",
    "height",
    "h",
    "position",
    "pos",
    "offset",
    "size",
    "dimensions",
)


_CONTAINER_ALIASES: dict[str, str] = {
    "m2ts": "ts",
    "m4v": "mp4",
    "mts": "ts",
    "mpg": "mpeg",
}

_CONTAINER_CODECS: dict[str, frozenset[str] | None] = {
    "3gp": frozenset({"h263", "h264", "mpeg4"}),
    "avi": frozenset({"ffv1", "h264", "huffyuv", "mjpeg", "mpeg4", "rawvideo"}),
    "flv": frozenset({"flv1", "h264"}),
    "gif": frozenset({"gif"}),
    "mkv": None,
    "mov": frozenset({"av1", "dnxhd", "h264", "hevc", "mpeg4", "prores"}),
    "mp4": frozenset({"av1", "h264", "hevc", "mpeg4"}),
    "mpeg": frozenset({"mpeg1video", "mpeg2video"}),
    "ogv": frozenset({"theora", "vp8", "vp9"}),
    "ts": frozenset({"h264", "hevc", "mpeg2video"}),
    "webm": frozenset({"av1", "vp8", "vp9"}),
    "webp": frozenset({"webp"}),
    "wmv": frozenset({"mpeg4", "vc1", "wmv1", "wmv2"}),
}

_CODEC_FAMILIES: dict[str, str] = {
    "avc": "h264",
    "avc1": "h264",
    "dnxhd": "dnxhd",
    "ffv1": "ffv1",
    "ffvhuff": "huffyuv",
    "flv": "flv1",
    "flv1": "flv1",
    "gif": "gif",
    "h263": "h263",
    "h264": "h264",
    "h265": "hevc",
    "hevc": "hevc",
    "huffyuv": "huffyuv",
    "libaom-av1": "av1",
    "librav1e": "av1",
    "libsvtav1": "av1",
    "libtheora": "theora",
    "libvorbis": "vorbis",
    "libvpx": "vp8",
    "libvpx-vp8": "vp8",
    "libvpx-vp9": "vp9",
    "libwebp": "webp",
    "libwebp_anim": "webp",
    "libx264": "h264",
    "libx265": "hevc",
    "libxvid": "mpeg4",
    "mjpeg": "mjpeg",
    "mpeg1video": "mpeg1video",
    "mpeg2video": "mpeg2video",
    "mpeg4": "mpeg4",
    "prores": "prores",
    "rawvideo": "rawvideo",
    "theora": "theora",
    "vc1": "vc1",
    "vp8": "vp8",
    "vp9": "vp9",
    "webp": "webp",
    "wmv1": "wmv1",
    "wmv2": "wmv2",
    "xvid": "mpeg4",
}

_CODEC_PREFIXES: tuple[tuple[str, str], ...] = (
    ("libx264", "h264"),
    ("libx265", "hevc"),
    ("libvpx-vp9", "vp9"),
    ("libvpx-vp8", "vp8"),
    ("libaom-av1", "av1"),
    ("libsvtav1", "av1"),
    ("librav1e", "av1"),
    ("libtheora", "theora"),
    ("libwebp", "webp"),
    ("libxvid", "mpeg4"),
    ("dnx", "dnxhd"),
    ("h264", "h264"),
    ("h265", "hevc"),
    ("hevc", "hevc"),
    ("vp8", "vp8"),
    ("vp9", "vp9"),
    ("av1", "av1"),
    ("prores", "prores"),
    ("mpeg1", "mpeg1video"),
    ("mpeg2", "mpeg2video"),
    ("mpeg4", "mpeg4"),
    ("mjpeg", "mjpeg"),
    ("wmv", "wmv2"),
)


def _video_codec_family(codec: str) -> str:
    name = codec.strip().lower()
    if name in _CODEC_FAMILIES:
        return _CODEC_FAMILIES[name]
    for prefix, family in _CODEC_PREFIXES:
        if name.startswith(prefix):
            return family
    return name


def _video_container(output: str | Path) -> str:
    suffix = Path(output).suffix.lower().lstrip(".")
    if not suffix:
        raise OutputProfileError(
            f"cannot infer a video container from output {str(output)!r}; "
            "use an extension such as .mp4, .mov, .mkv or .webm"
        )
    container = _CONTAINER_ALIASES.get(suffix, suffix)
    if container not in _CONTAINER_CODECS:
        allowed = ", ".join(f".{name}" for name in sorted(_CONTAINER_CODECS))
        raise OutputProfileError(
            f"unsupported video output container {suffix!r}; expected one of {allowed}"
        )
    return container


def validate_output_container(
    output: str | Path,
    codec: str | OutputProfile,
) -> None:
    """Reject a video container that cannot carry *codec*.

    *codec* may be an encoder name or a resolved :class:`OutputProfile`. The
    check is purely textual: it inspects the output suffix and the configured
    encoder so an incompatible pair fails before ffmpeg is invoked.
    """
    if isinstance(codec, OutputProfile):
        codec = codec.codec
    if not isinstance(codec, str) or not codec.strip():
        raise OutputProfileError("codec must be a non-empty string")
    container = _video_container(output)
    allowed = _CONTAINER_CODECS[container]
    if allowed is None:
        return
    family = _video_codec_family(codec)
    if family not in allowed:
        expected = ", ".join(sorted(allowed))
        raise OutputProfileError(
            f"output container '.{container}' cannot carry video codec "
            f"{codec!r}; expected one of {expected}"
        )


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OutputProfileError(f"{name} must be a positive integer")
    if value <= 0:
        raise OutputProfileError(f"{name} must be a positive integer")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OutputProfileError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise OutputProfileError(f"{name} must be finite")
    return number


@dataclass(frozen=True, slots=True)
class OutputProfile:
    """The concrete output settings for one render."""

    width: int
    height: int
    fps: int
    codec: str
    bitrate: int | None = None
    audio_codec: str = "aac"
    audio_bitrate: int | None = None
    loudness_lufs: float = -14.0
    min_duration_seconds: float = 20.0
    max_duration_seconds: float = 45.0

    def __post_init__(self) -> None:
        _positive_integer(self.width, "width")
        _positive_integer(self.height, "height")
        _positive_integer(self.fps, "fps")
        if not isinstance(self.codec, str) or not self.codec.strip():
            raise OutputProfileError("codec must be a non-empty string")
        if not isinstance(self.audio_codec, str) or not self.audio_codec.strip():
            raise OutputProfileError("audio_codec must be a non-empty string")
        if self.bitrate is not None:
            _positive_integer(self.bitrate, "bitrate")
        if self.audio_bitrate is not None:
            _positive_integer(self.audio_bitrate, "audio_bitrate")
        _finite(self.loudness_lufs, "loudness_lufs")
        minimum = _finite(self.min_duration_seconds, "min_duration_seconds")
        maximum = _finite(self.max_duration_seconds, "max_duration_seconds")
        if minimum <= 0.0 or maximum <= 0.0:
            raise OutputProfileError("duration bounds must be positive")
        if minimum > maximum:
            raise OutputProfileError("duration min must not exceed max")

    @classmethod
    def from_config(
        cls,
        video: VideoConfig | Mapping[str, Any] | Any | None = None,
        *,
        platform: str | None = None,
        profile: str | None = None,
    ) -> OutputProfile:
        """Resolve *video* with the named platform and profile overrides."""
        return resolve_output_profile(video, platform=platform, profile=profile)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> OutputProfile:
        """Build a profile from a plain mapping."""
        return cls(**dict(values))

    @property
    def resolution(self) -> str:
        """The profile as a ``WIDTHxHEIGHT`` string."""
        return f"{self.width}x{self.height}"

    @property
    def aspect_ratio(self) -> float:
        """The width divided by the height."""
        return self.width / self.height

    @property
    def video_codec(self) -> str:
        """Alias for :attr:`codec`."""
        return self.codec

    @property
    def video_bitrate(self) -> int | None:
        """Alias for :attr:`bitrate`."""
        return self.bitrate

    @property
    def bitrate_kbps(self) -> int | None:
        """Alias for :attr:`bitrate` in kilobits per second."""
        return self.bitrate

    @property
    def audio_bitrate_kbps(self) -> int | None:
        """Alias for :attr:`audio_bitrate` in kilobits per second."""
        return self.audio_bitrate

    @property
    def duration_bounds(self) -> tuple[float, float]:
        """The inclusive duration range in seconds."""
        return self.min_duration_seconds, self.max_duration_seconds

    def clamp_duration(self, duration_seconds: float) -> float:
        """Clamp *duration_seconds* to the configured range."""
        value = _finite(duration_seconds, "duration_seconds")
        if value <= 0.0:
            raise OutputProfileError("duration_seconds must be positive")
        return min(max(value, self.min_duration_seconds), self.max_duration_seconds)

    def ffmpeg_options(self) -> dict[str, Any]:
        """Return the encoder options this profile contributes to a command."""
        return ffmpeg_options(self)


def _as_video_config(value: Any) -> VideoConfig:
    if value is None:
        return VideoConfig()
    if isinstance(value, VideoConfig):
        return value
    if isinstance(value, Mapping):
        if "video" in value:
            nested = value["video"]
            if nested is None:
                return VideoConfig()
            return _as_video_config(nested)
        return VideoConfig.from_mapping(value)
    nested = getattr(value, "video", None)
    if nested is not None and nested is not value:
        return _as_video_config(nested)
    if all(
        hasattr(value, name)
        for name in ("resolution", "fps", "duration_seconds", "loudness_lufs")
    ):
        return VideoConfig.from_mapping(
            {
                "resolution": value.resolution,
                "fps": value.fps,
                "duration_seconds": value.duration_seconds,
                "loudness_lufs": value.loudness_lufs,
                "codec": getattr(value, "codec", None) or DEFAULT_VIDEO_CODEC,
                "bitrate": getattr(value, "bitrate", None),
                "audio_codec": getattr(value, "audio_codec", None)
                or DEFAULT_AUDIO_CODEC,
                "audio_bitrate": getattr(value, "audio_bitrate", None),
            }
        )
    return VideoConfig.from_mapping({})


def resolve_output_profile(
    video: VideoConfig | Mapping[str, Any] | Any | None = None,
    *,
    platform: str | None = None,
    profile: str | None = None,
) -> OutputProfile:
    """Merge the video configuration with platform and profile overrides.

    *video* may be a :class:`~aivia.core.config.video.VideoConfig`, a root
    configuration object with a ``video`` section, or a mapping in either
    shape. Platform overrides apply first and a selected profile wins last.
    """
    config = _as_video_config(video)
    resolved = config.merged(platform=platform, profile=profile)
    return OutputProfile(
        width=resolved.width,
        height=resolved.height,
        fps=resolved.fps,
        codec=resolved.codec,
        bitrate=resolved.bitrate,
        audio_codec=resolved.audio_codec,
        audio_bitrate=resolved.audio_bitrate,
        loudness_lufs=resolved.loudness_lufs,
        min_duration_seconds=float(resolved.duration_seconds.min),
        max_duration_seconds=float(resolved.duration_seconds.max),
    )


def output_profile(
    video: VideoConfig | Mapping[str, Any] | Any | None = None,
    *,
    platform: str | None = None,
    profile: str | None = None,
) -> OutputProfile:
    """Alias for :func:`resolve_output_profile`."""
    return resolve_output_profile(video, platform=platform, profile=profile)


def output_profile_for(
    video: VideoConfig | Mapping[str, Any] | Any | None = None,
    *,
    platform: str | None = None,
    profile: str | None = None,
) -> OutputProfile:
    """Alias for :func:`resolve_output_profile`."""
    return resolve_output_profile(video, platform=platform, profile=profile)


def has_absolute_layer_geometry(
    composition: Composition | Mapping[str, Any],
) -> bool:
    """Whether any visible layer is placed with absolute pixel geometry.

    A layer counts when it carries an explicit ``x``, ``y``, ``width``,
    ``height``, ``position`` or ``size`` value, including the aliases accepted
    by the schema, or when it animates its position in canvas pixels. Anchored
    overlays that only use an anchor and a margin are relative to the canvas
    and do not count.
    """
    value = (
        composition
        if isinstance(composition, Composition)
        else Composition.model_validate(dict(composition))
    )
    for layer in value.layers:
        if not layer.visible:
            continue
        if any(
            getattr(layer, name, None) is not None for name in _ABSOLUTE_GEOMETRY_FIELDS
        ):
            return True
        extras = layer.model_extra or {}
        if any(extras.get(name) is not None for name in _ABSOLUTE_GEOMETRY_ALIASES):
            return True
        if _has_pixel_position_keyframes(layer, extras):
            return True
    return False


def _has_pixel_position_keyframes(
    layer: Any,
    extras: Mapping[str, Any],
) -> bool:
    for name in ("position_keyframes", "position_animation"):
        if getattr(layer, name, None) is not None or extras.get(name) is not None:
            return True
    keyframes = getattr(layer, "keyframes", None)
    if keyframes is None:
        keyframes = extras.get("keyframes")
    if keyframes is None:
        return False
    if isinstance(keyframes, Mapping):
        return keyframes.get("position") is not None
    return getattr(keyframes, "position", None) is not None


def _same_aspect_ratio(
    first_width: int,
    first_height: int,
    second_width: int,
    second_height: int,
) -> bool:
    return math.isclose(
        first_width * second_height,
        second_width * first_height,
        rel_tol=1e-6,
        abs_tol=0.0,
    )


def aspect_ratio_changed(
    composition: Composition | Mapping[str, Any],
    profile: OutputProfile | Mapping[str, Any],
) -> bool:
    """Whether *profile* changes the canvas aspect ratio of *composition*.

    A profile that changes only the resolution but keeps the aspect ratio
    returns false. Layer geometry is checked separately by
    :func:`warn_aspect_ratio_change`.
    """
    value = (
        composition
        if isinstance(composition, Composition)
        else Composition.model_validate(dict(composition))
    )
    resolved = (
        profile
        if isinstance(profile, OutputProfile)
        else OutputProfile.from_mapping(profile)
    )
    return not _same_aspect_ratio(
        value.width, value.height, resolved.width, resolved.height
    )


def warn_aspect_ratio_change(
    composition: Composition | Mapping[str, Any],
    profile: OutputProfile | Mapping[str, Any],
    *,
    stacklevel: int = 2,
) -> bool:
    """Emit :class:`AspectRatioWarning` when applying *profile* distorts layers."""
    value = (
        composition
        if isinstance(composition, Composition)
        else Composition.model_validate(dict(composition))
    )
    resolved = (
        profile
        if isinstance(profile, OutputProfile)
        else OutputProfile.from_mapping(profile)
    )
    if not aspect_ratio_changed(value, resolved):
        return False
    if not has_absolute_layer_geometry(value):
        return False
    warnings.warn(
        f"output profile changes the aspect ratio from "
        f"{value.width}x{value.height} to {resolved.width}x{resolved.height}; "
        "absolute pixel layer geometry may be distorted",
        AspectRatioWarning,
        stacklevel=stacklevel,
    )
    return True


def apply_output_profile(
    composition: Composition | Mapping[str, Any],
    profile: OutputProfile | Mapping[str, Any],
) -> Composition:
    """Return *composition* with the profile's canvas, timing and loudness."""
    value = (
        composition
        if isinstance(composition, Composition)
        else Composition.model_validate(dict(composition))
    )
    resolved = (
        profile
        if isinstance(profile, OutputProfile)
        else OutputProfile.from_mapping(profile)
    )
    warn_aspect_ratio_change(value, resolved, stacklevel=3)
    data = value.model_dump(by_alias=False)
    data["canvas"] = {
        **data["canvas"],
        "width": resolved.width,
        "height": resolved.height,
    }
    data["fps"] = resolved.fps
    data["duration_seconds"] = resolved.clamp_duration(value.duration_seconds)
    data["loudness_lufs"] = resolved.loudness_lufs
    return Composition.model_validate(data)


def apply_video_config(
    composition: Composition | Mapping[str, Any],
    video: VideoConfig | Mapping[str, Any] | Any | None = None,
    *,
    platform: str | None = None,
    profile: str | None = None,
) -> tuple[Composition, OutputProfile]:
    """Resolve the configuration and return the adjusted composition."""
    resolved = resolve_output_profile(video, platform=platform, profile=profile)
    return apply_output_profile(composition, resolved), resolved


def ffmpeg_options(profile: OutputProfile) -> dict[str, Any]:
    """Return the encoder options for *profile* understood by the builder."""
    options: dict[str, Any] = {
        "video_codec": profile.codec,
        "audio_codec": profile.audio_codec,
        "loudness_lufs": profile.loudness_lufs,
    }
    if profile.bitrate is not None:
        options["bitrate"] = profile.bitrate
    if profile.audio_bitrate is not None:
        options["audio_bitrate"] = profile.audio_bitrate
    return options

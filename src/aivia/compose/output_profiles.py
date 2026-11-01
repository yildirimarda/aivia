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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from aivia.core.config.video import (
    DEFAULT_AUDIO_CODEC,
    DEFAULT_VIDEO_CODEC,
    VideoConfig,
)

from .errors import RenderError
from .schema import Composition

__all__ = [
    "OutputProfile",
    "OutputProfileError",
    "apply_output_profile",
    "apply_video_config",
    "ffmpeg_options",
    "output_profile",
    "output_profile_for",
    "resolve_output_profile",
]


class OutputProfileError(RenderError):
    """A configured output profile could not be resolved or applied."""

    code = "output_profile"


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

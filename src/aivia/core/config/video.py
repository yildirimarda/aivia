"""Video specification: resolution, frame rate, duration bounds and overrides.

A video section carries the defaults used by every format. Platforms and
content profiles may narrow those defaults without repeating the whole
section: their partial mappings are deep-merged on top of the base values, so
an override of ``duration_seconds.max`` keeps the configured minimum.

Resolution is written as ``WIDTHxHEIGHT`` (the multiplication sign and ``*``
are accepted too) and parsed into positive pixel dimensions. Frame rates and
durations must be positive; a duration range must be ordered.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any, Self

from pydantic import Field, field_validator, model_validator

from aivia.core.config.base import ConfigModel
from aivia.core.errors import ConfigError

__all__ = [
    "DEFAULT_AUDIO_CODEC",
    "DEFAULT_VIDEO_CODEC",
    "DurationRangeConfig",
    "DurationRangeOverrideConfig",
    "PlatformVideoConfig",
    "ProfileVideoConfig",
    "Resolution",
    "VideoConfig",
    "VideoOverride",
    "VideoOverridesConfig",
    "VideoSpec",
    "VideoSpecConfig",
    "deep_merge",
    "merge_video_specs",
    "parse_resolution",
    "resolve_video",
]

_RESOLUTION = re.compile(r"^(?P<width>\d+)\s*[xX*\u00d7]\s*(?P<height>\d+)$")

DEFAULT_VIDEO_CODEC = "libx264"
"""The encoder used when neither a platform nor a profile names one."""

DEFAULT_AUDIO_CODEC = "aac"
"""The audio encoder used when neither a platform nor a profile names one."""

_OUTPUT_ALIASES: dict[str, str] = {
    "video_codec": "codec",
    "video_bitrate": "bitrate",
    "bitrate_kbps": "bitrate",
    "video_bitrate_kbps": "bitrate",
    "audio_bitrate_kbps": "audio_bitrate",
}


def _accept_output_aliases(data: Any) -> Any:
    """Rename the long output-field aliases onto their canonical names."""
    if not isinstance(data, Mapping):
        return data
    values = dict(data)
    for alias, canonical in _OUTPUT_ALIASES.items():
        if alias in values:
            values.setdefault(canonical, values.pop(alias))
    return values


class Resolution(str):
    """A validated ``WIDTHxHEIGHT`` string with parsed pixel dimensions."""

    width: int
    height: int

    def __new__(cls, value: str) -> Self:
        if not isinstance(value, str):
            raise ValueError("resolution must be a string such as 1080x1920")
        text = value.strip()
        parsed = _RESOLUTION.fullmatch(text)
        if parsed is None:
            raise ValueError("resolution must be WIDTHxHEIGHT, for example 1080x1920")
        width = int(parsed.group("width"))
        height = int(parsed.group("height"))
        if width <= 0 or height <= 0:
            raise ValueError("resolution dimensions must be positive")
        resolution = super().__new__(cls, text)
        resolution.width = width
        resolution.height = height
        return resolution

    @property
    def aspect_ratio(self) -> float:
        """Return the width divided by the height."""
        return self.width / self.height


def parse_resolution(value: str) -> tuple[int, int]:
    """Return ``(width, height)`` for a ``WIDTHxHEIGHT`` string."""
    resolution = Resolution(value)
    return resolution.width, resolution.height


def _as_resolution(value: str | None) -> Resolution | None:
    return None if value is None else Resolution(value)


class DurationRangeConfig(ConfigModel):
    """Inclusive lower and upper bounds for a generated video in seconds."""

    min: int = Field(default=20, gt=0)
    max: int = Field(default=45, gt=0)

    @model_validator(mode="after")
    def _ordered_bounds(self) -> Self:
        if self.min > self.max:
            raise ValueError("duration min must not exceed max")
        return self


class DurationRangeOverrideConfig(ConfigModel):
    """A partial duration range used by platform and profile overrides."""

    min: int | None = Field(default=None, gt=0)
    max: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _ordered_bounds(self) -> Self:
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("duration min must not exceed max")
        return self


class VideoOverridesConfig(ConfigModel):
    """Partial video settings applied on top of the base video section."""

    resolution: str | None = None
    fps: int | None = Field(default=None, gt=0)
    duration_seconds: DurationRangeOverrideConfig | None = None
    loudness_lufs: float | None = None
    codec: str | None = None
    bitrate: int | None = Field(default=None, gt=0)
    audio_codec: str | None = None
    audio_bitrate: int | None = Field(default=None, gt=0)

    @field_validator("resolution", mode="after")
    @classmethod
    def _validate_resolution(cls, value: str | None) -> Resolution | None:
        return _as_resolution(value)

    @field_validator("codec", "audio_codec", mode="after")
    @classmethod
    def _validate_codec(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        if not text:
            raise ValueError("codec must be a non-empty string")
        return text

    @model_validator(mode="before")
    @classmethod
    def _accept_duration_alias(cls, data: Any) -> Any:
        values = _accept_output_aliases(data)
        if isinstance(values, Mapping) and "duration" in values:
            values = dict(values)
            duration = values.pop("duration")
            values.setdefault("duration_seconds", duration)
        return values

    @property
    def video_codec(self) -> str | None:
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


class VideoConfig(ConfigModel):
    """Video defaults shared by formats, profiles and platform overrides."""

    resolution: str = "1080x1920"
    fps: int = Field(default=30, gt=0)
    duration_seconds: DurationRangeConfig = Field(default_factory=DurationRangeConfig)
    loudness_lufs: float = -14.0
    codec: str = DEFAULT_VIDEO_CODEC
    bitrate: int | None = Field(default=None, gt=0)
    audio_codec: str = DEFAULT_AUDIO_CODEC
    audio_bitrate: int | None = Field(default=None, gt=0)
    platforms: dict[str, VideoOverridesConfig] = Field(default_factory=dict)
    profiles: dict[str, VideoOverridesConfig] = Field(default_factory=dict)

    @field_validator("resolution", mode="after")
    @classmethod
    def _validate_resolution(cls, value: str) -> Resolution:
        resolution = _as_resolution(value)
        if resolution is None:
            raise ValueError("resolution must be a string such as 1080x1920")
        return resolution

    @field_validator("codec", "audio_codec", mode="after")
    @classmethod
    def _validate_codec(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("codec must be a non-empty string")
        return text

    @model_validator(mode="before")
    @classmethod
    def _accept_override_aliases(cls, data: Any) -> Any:
        values = _accept_output_aliases(data)
        if not isinstance(values, Mapping):
            return values
        values = dict(values)
        for alias, canonical in (
            ("platform_overrides", "platforms"),
            ("profile_overrides", "profiles"),
        ):
            if alias in values:
                overrides = values.pop(alias)
                values.setdefault(canonical, overrides)
        return values

    @property
    def width(self) -> int:
        """The configured horizontal pixel count."""
        return Resolution(self.resolution).width

    @property
    def height(self) -> int:
        """The configured vertical pixel count."""
        return Resolution(self.resolution).height

    @property
    def aspect_ratio(self) -> float:
        """The configured width divided by height."""
        return Resolution(self.resolution).aspect_ratio

    @property
    def platform_overrides(self) -> dict[str, VideoOverridesConfig]:
        """Alias for :attr:`platforms` used by callers that prefer the long name."""
        return self.platforms

    @property
    def profile_overrides(self) -> dict[str, VideoOverridesConfig]:
        """Alias for :attr:`profiles` used by callers that prefer the long name."""
        return self.profiles

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

    def for_platform(self, platform: str) -> VideoConfig:
        """Return the video spec with *platform* overrides deep-merged in."""
        return self.merged(platform=platform)

    def for_profile(self, profile: str) -> VideoConfig:
        """Return the video spec with *profile* overrides deep-merged in."""
        return self.merged(profile=profile)

    def resolved_for(
        self,
        *,
        platform: str | None = None,
        profile: str | None = None,
    ) -> VideoConfig:
        """Alias for :meth:`merged` with explicit keyword arguments."""
        return self.merged(platform=platform, profile=profile)

    def resolve(
        self,
        *,
        platform: str | None = None,
        profile: str | None = None,
    ) -> VideoConfig:
        """Alias for :meth:`merged` with explicit keyword arguments."""
        return self.merged(platform=platform, profile=profile)

    def merged(
        self,
        *,
        platform: str | None = None,
        profile: str | None = None,
    ) -> VideoConfig:
        """Deep-merge the base spec with platform and then profile overrides.

        Platform overrides apply first, so a selected profile can make the
        final decision for its own content. Overrides never mutate this spec.
        """
        values = self.model_dump(exclude={"platforms", "profiles"})
        if platform is not None:
            values = deep_merge(values, self._platform_values(platform))
        if profile is not None:
            values = deep_merge(values, self._profile_values(profile))
        return VideoConfig.from_mapping(values)

    def _platform_values(self, platform: str) -> dict[str, Any]:
        try:
            override = self.platforms[platform]
        except KeyError:
            available = ", ".join(sorted(self.platforms)) or "none"
            raise ConfigError(
                f"unknown video platform: {platform} (available: {available})",
                details={"platform": platform, "available": list(self.platforms)},
            ) from None
        return override.model_dump(exclude_none=True)

    def _profile_values(self, profile: str) -> dict[str, Any]:
        try:
            override = self.profiles[profile]
        except KeyError:
            available = ", ".join(sorted(self.profiles)) or "none"
            raise ConfigError(
                f"unknown video profile: {profile} (available: {available})",
                details={"profile": profile, "available": list(self.profiles)},
            ) from None
        return override.model_dump(exclude_none=True)


def deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a recursive merge of *override* onto *base*.

    Nested mappings merge key by key; every other value replaces the base and
    inputs are never mutated.
    """
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(value, Mapping) and isinstance(existing, Mapping):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def merge_video_specs(
    base: VideoConfig,
    *overrides: Mapping[str, Any],
) -> VideoConfig:
    """Deep-merge raw override mappings onto *base* and validate the result."""
    values: dict[str, Any] = base.model_dump(exclude={"platforms", "profiles"})
    for override in overrides:
        values = deep_merge(values, override)
    return VideoConfig.from_mapping(values)


def resolve_video(
    video: VideoConfig,
    *,
    platform: str | None = None,
    profile: str | None = None,
) -> VideoConfig:
    """Return *video* with the named platform and profile overrides applied."""
    return video.merged(platform=platform, profile=profile)


VideoSpecConfig = VideoConfig
VideoSpec = VideoConfig
VideoOverride = VideoOverridesConfig
PlatformVideoConfig = VideoOverridesConfig
ProfileVideoConfig = VideoOverridesConfig

"""Platform publishing rules loaded from ``platforms.yaml``.

Durations, resolutions, aspect ratios, size limits, caption limits, safe areas
and disclosure requirements change independently of a release, so they live in
a data file instead of code. The packaged file is the fallback; a user copy
beside the config wins. The loader validates the file and exposes typed lookups
by platform and media type.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from aivia.core import paths
from aivia.core.errors import ConfigError
from aivia.core.models.asset import Asset

__all__ = [
    "DEFAULT_PLATFORMS_FILENAME",
    "DEFAULT_PLATFORM_DATA_FILENAME",
    "PLATFORMS_FILENAME",
    "PLATFORMS_SCHEMA_VERSION",
    "PLATFORM_DATA_FILENAME",
    "PLATFORM_SCHEMA_VERSION",
    "CaptionLimits",
    "ChapterRules",
    "DisclosureRules",
    "DurationCollection",
    "DurationLimits",
    "MediaLimits",
    "MediaLimitsCollection",
    "Platform",
    "PlatformCaptionLimits",
    "PlatformCatalog",
    "PlatformData",
    "PlatformDisclosure",
    "PlatformDuration",
    "PlatformInfo",
    "PlatformLimits",
    "PlatformRegistry",
    "PlatformSafeArea",
    "PlatformSpec",
    "SafeArea",
    "default_platforms_path",
    "get_platform",
    "load_platform_catalog",
    "load_platform_data",
    "load_platforms",
    "load_platforms_file",
    "platform_catalog",
    "platform_data_path",
    "platform_names",
    "platforms_path",
    "read_platforms",
    "resolve_platform_data_path",
    "resolve_platforms_path",
]

PLATFORMS_FILENAME = "platforms.yaml"
DEFAULT_PLATFORMS_FILENAME = PLATFORMS_FILENAME
PLATFORM_DATA_FILENAME = PLATFORMS_FILENAME
DEFAULT_PLATFORM_DATA_FILENAME = PLATFORMS_FILENAME
PLATFORMS_SCHEMA_VERSION = 1
PLATFORM_SCHEMA_VERSION = PLATFORMS_SCHEMA_VERSION

_RESOLUTION = re.compile(r"^(?P<width>\d+)x(?P<height>\d+)$")
_ASPECT_RATIO = re.compile(r"^(?P<width>\d+(?:\.\d+)?):(?P<height>\d+(?:\.\d+)?)$")

_PLATFORM_ALIASES = {
    "instagram_reels": "instagram",
    "instagram-reels": "instagram",
    "reels": "instagram",
    "youtube_shorts": "youtube",
    "youtube-shorts": "youtube",
    "shorts": "youtube",
    "twitter": "x",
}

_TOP_LEVEL_KEYS = frozenset({"schema_version", "platforms", "defaults", "version"})
_DURATION_KEYS = frozenset(
    {
        "min_seconds",
        "min",
        "minimum",
        "min_duration",
        "min_duration_seconds",
        "max_seconds",
        "max",
        "maximum",
        "max_duration",
        "max_duration_seconds",
        "duration",
    }
)
_CAPTION_KEYS = frozenset(
    {
        "max_length",
        "max_chars",
        "max_characters",
        "length",
        "caption",
        "max_caption_length",
        "max_title_length",
        "title",
        "max_title",
        "max_description_length",
        "description",
        "max_description",
        "max_hashtags",
        "hashtags",
        "tags",
        "max_mentions",
        "mentions",
    }
)
_SAFE_AREA_KEYS = frozenset(
    {
        "top",
        "bottom",
        "left",
        "right",
        "top_percent",
        "bottom_percent",
        "left_percent",
        "right_percent",
    }
)
_DISCLOSURE_KEYS = frozenset(
    {
        "required",
        "requires_disclosure",
        "required_flag",
        "text",
        "disclosure_text",
        "label_text",
        "on_screen_label",
        "label",
        "on_screen",
        "metadata_field",
        "field",
        "metadata",
        "metadata_supported",
        "url",
    }
)
_CHAPTER_RULES_KEYS = frozenset(
    {
        "enabled",
        "max_chapters",
        "max_chapter_count",
        "min_chapter_duration_seconds",
        "min_chapter_duration",
        "min_duration_seconds",
        "max_chapter_title_length",
        "max_title_length",
        "chapter_title_length",
        "separator",
        "chapter_separator",
    }
)
_MEDIA_LIMIT_KEYS = frozenset(
    {
        "media_type",
        "type",
        "duration",
        "durations",
        "duration_limits",
        "min_seconds",
        "min",
        "minimum",
        "min_duration",
        "min_duration_seconds",
        "max_seconds",
        "max",
        "maximum",
        "max_duration",
        "max_duration_seconds",
        "resolutions",
        "resolution",
        "dimensions",
        "aspect_ratios",
        "aspect_ratio",
        "aspects",
        "size_limits",
        "size_limit",
        "max_file_size_bytes",
        "max_size_bytes",
        "max_file_size",
        "max_bytes",
        "audio_required",
        "requires_audio",
        "loudness_lufs",
        "loudness",
        "loudness_tolerance_lufs",
        "frame_rates",
        "fps",
    }
)
_PLATFORM_KEYS = frozenset(
    {
        "name",
        "display_name",
        "title",
        "label",
        "media_types",
        "media_type",
        "types",
        "output_types",
        "limits",
        "media_limits",
        "media",
        "per_media_type",
        "specs",
        "duration",
        "durations",
        "duration_limits",
        "resolutions",
        "resolution",
        "dimensions",
        "aspect_ratios",
        "aspect_ratio",
        "aspects",
        "size_limits",
        "size_limit",
        "max_file_size_bytes",
        "max_size_bytes",
        "max_file_size",
        "max_bytes",
        "caption_limits",
        "captions",
        "caption",
        "caption_limit",
        "text_limits",
        "safe_area",
        "safe_areas",
        "margins",
        "disclosure",
        "disclosure_rules",
        "audio_required",
        "requires_audio",
        "chapter_rules",
        "loudness_lufs",
        "loudness",
        "loudness_tolerance_lufs",
        "frame_rates",
        "fps",
        "aliases",
        "alias",
        "enabled",
    }
)


@dataclass(frozen=True, slots=True)
class DurationLimits:
    """The allowed duration of one media type, in seconds."""

    min_seconds: float | None = None
    max_seconds: float | None = None

    @property
    def min(self) -> float | None:
        """Alias for :attr:`min_seconds`."""
        return self.min_seconds

    @property
    def max(self) -> float | None:
        """Alias for :attr:`max_seconds`."""
        return self.max_seconds

    @property
    def minimum(self) -> float | None:
        """Alias for :attr:`min_seconds`."""
        return self.min_seconds

    @property
    def maximum(self) -> float | None:
        """Alias for :attr:`max_seconds`."""
        return self.max_seconds

    @property
    def min_duration_seconds(self) -> float | None:
        """Alias for :attr:`min_seconds`."""
        return self.min_seconds

    @property
    def max_duration_seconds(self) -> float | None:
        """Alias for :attr:`max_seconds`."""
        return self.max_seconds

    @property
    def min_duration(self) -> float | None:
        """Alias for :attr:`min_seconds`."""
        return self.min_seconds

    @property
    def max_duration(self) -> float | None:
        """Alias for :attr:`max_seconds`."""
        return self.max_seconds

    def allows(self, seconds: float) -> bool:
        """Whether *seconds* falls inside this range."""
        if self.min_seconds is not None and seconds < self.min_seconds:
            return False
        return self.max_seconds is None or seconds <= self.max_seconds

    def __contains__(self, value: object) -> bool:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return self.allows(float(value))

    def __getitem__(self, key: str) -> float | None:
        aliases = {
            "min_seconds": self.min_seconds,
            "min": self.min_seconds,
            "minimum": self.min_seconds,
            "min_duration": self.min_seconds,
            "min_duration_seconds": self.min_seconds,
            "max_seconds": self.max_seconds,
            "max": self.max_seconds,
            "maximum": self.max_seconds,
            "max_duration": self.max_seconds,
            "max_duration_seconds": self.max_seconds,
        }
        try:
            return aliases[key]
        except KeyError as exc:
            raise KeyError(key) from exc


@dataclass(frozen=True, slots=True)
class CaptionLimits:
    """Text limits for one platform's caption, title, description and tags."""

    max_length: int | None = None
    max_title_length: int | None = None
    max_description_length: int | None = None
    max_hashtags: int | None = None
    max_mentions: int | None = None

    @property
    def max_chars(self) -> int | None:
        """Alias for :attr:`max_length`."""
        return self.max_length

    @property
    def max_characters(self) -> int | None:
        """Alias for :attr:`max_length`."""
        return self.max_length

    @property
    def caption(self) -> int | None:
        """Alias for :attr:`max_length`."""
        return self.max_length

    @property
    def title(self) -> int | None:
        """Alias for :attr:`max_title_length`."""
        return self.max_title_length

    @property
    def description(self) -> int | None:
        """Alias for :attr:`max_description_length`."""
        return self.max_description_length

    @property
    def hashtags(self) -> int | None:
        """Alias for :attr:`max_hashtags`."""
        return self.max_hashtags

    def __getitem__(self, key: str) -> int | None:
        aliases = {
            "max_length": self.max_length,
            "max_chars": self.max_length,
            "max_characters": self.max_length,
            "length": self.max_length,
            "caption": self.max_length,
            "max_caption_length": self.max_length,
            "max_title_length": self.max_title_length,
            "title": self.max_title_length,
            "max_title": self.max_title_length,
            "max_description_length": self.max_description_length,
            "description": self.max_description_length,
            "max_description": self.max_description_length,
            "max_hashtags": self.max_hashtags,
            "hashtags": self.max_hashtags,
            "tags": self.max_hashtags,
            "max_mentions": self.max_mentions,
            "mentions": self.max_mentions,
        }
        try:
            return aliases[key]
        except KeyError as exc:
            raise KeyError(key) from exc


@dataclass(frozen=True, slots=True)
class SafeArea:
    """Margins kept clear of platform chrome, as fractions of the canvas."""

    top: float = 0.0
    bottom: float = 0.0
    left: float = 0.0
    right: float = 0.0

    @property
    def horizontal(self) -> float:
        """The combined left and right margin fraction."""
        return self.left + self.right

    @property
    def vertical(self) -> float:
        """The combined top and bottom margin fraction."""
        return self.top + self.bottom

    @property
    def margins(self) -> tuple[float, float, float, float]:
        """Return ``(top, right, bottom, left)`` in CSS order."""
        return (self.top, self.right, self.bottom, self.left)

    def as_dict(self) -> dict[str, float]:
        """Return the margins as a mapping."""
        return {
            "top": self.top,
            "bottom": self.bottom,
            "left": self.left,
            "right": self.right,
        }

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Convert the fractions to ``(top, right, bottom, left)`` pixels."""
        return (
            round(self.top * height),
            round(self.right * width),
            round(self.bottom * height),
            round(self.left * width),
        )

    def contains(
        self,
        x: float,
        y: float,
        *,
        width: int,
        height: int,
    ) -> bool:
        """Whether a point lies inside the safe area of a canvas."""
        return (
            self.left * width <= x <= (1 - self.right) * width
            and self.top * height <= y <= (1 - self.bottom) * height
        )

    def __getitem__(self, key: str) -> float:
        aliases = {
            "top": self.top,
            "bottom": self.bottom,
            "left": self.left,
            "right": self.right,
            "horizontal": self.horizontal,
            "vertical": self.vertical,
        }
        try:
            return aliases[key]
        except KeyError as exc:
            raise KeyError(key) from exc


@dataclass(frozen=True, slots=True)
class DisclosureRules:
    """How a platform wants AI-generated content disclosed."""

    required: bool = False
    text: str = ""
    on_screen_label: str = ""
    metadata_field: str = ""
    metadata: bool = False
    url: str = ""

    @property
    def requires_disclosure(self) -> bool:
        """Alias for :attr:`required`."""
        return self.required

    @property
    def disclosure_text(self) -> str:
        """Alias for :attr:`text`."""
        return self.text

    @property
    def label(self) -> str:
        """Alias for :attr:`on_screen_label`."""
        return self.on_screen_label

    @property
    def on_screen(self) -> str:
        """Alias for :attr:`on_screen_label`."""
        return self.on_screen_label

    @property
    def field(self) -> str:
        """Alias for :attr:`metadata_field`."""
        return self.metadata_field

    @property
    def metadata_supported(self) -> bool:
        """Alias for :attr:`metadata`."""
        return self.metadata

    def __getitem__(self, key: str) -> Any:
        aliases: dict[str, Any] = {
            "required": self.required,
            "requires_disclosure": self.required,
            "text": self.text,
            "disclosure_text": self.text,
            "on_screen_label": self.on_screen_label,
            "label": self.on_screen_label,
            "on_screen": self.on_screen_label,
            "metadata_field": self.metadata_field,
            "field": self.metadata_field,
            "metadata": self.metadata,
            "metadata_supported": self.metadata,
            "url": self.url,
        }
        try:
            return aliases[key]
        except KeyError as exc:
            raise KeyError(key) from exc


@dataclass(frozen=True, slots=True)
class ChapterRules:
    """Rules for splitting a transcript into description chapters.

    ``enabled`` turns chapter generation on or off for the platform.
    ``max_chapters`` caps the number of chapters. ``min_chapter_duration_seconds``
    keeps chapters from being too short to be useful. ``max_chapter_title_length``
    bounds each chapter title and ``separator`` joins chapter lines in a
    description.
    """

    enabled: bool = True
    max_chapters: int | None = 25
    min_chapter_duration_seconds: float | None = 30.0
    max_chapter_title_length: int | None = 100
    separator: str = "\n"


@dataclass(frozen=True, slots=True)
class MediaLimits:
    """The limits that apply to one media type on one platform."""

    media_type: str
    duration: DurationLimits = field(default_factory=DurationLimits)
    resolutions: tuple[str, ...] = ()
    aspect_ratios: tuple[str, ...] = ()
    max_file_size_bytes: int | None = None
    audio_required: bool = False
    loudness_lufs: float | None = None
    loudness_tolerance_lufs: float | None = None

    @property
    def type(self) -> str:
        """Alias for :attr:`media_type`."""
        return self.media_type

    @property
    def min_duration_seconds(self) -> float | None:
        """The minimum duration, in seconds."""
        return self.duration.min_seconds

    @property
    def max_duration_seconds(self) -> float | None:
        """The maximum duration, in seconds."""
        return self.duration.max_seconds

    @property
    def min_duration(self) -> float | None:
        """Alias for :attr:`min_duration_seconds`."""
        return self.min_duration_seconds

    @property
    def max_duration(self) -> float | None:
        """Alias for :attr:`max_duration_seconds`."""
        return self.max_duration_seconds

    @property
    def resolution(self) -> str | None:
        """The first declared resolution."""
        return self.resolutions[0] if self.resolutions else None

    @property
    def aspect_ratio(self) -> str | None:
        """The first declared aspect ratio."""
        return self.aspect_ratios[0] if self.aspect_ratios else None

    @property
    def max_size_bytes(self) -> int | None:
        """Alias for :attr:`max_file_size_bytes`."""
        return self.max_file_size_bytes

    @property
    def max_file_size(self) -> int | None:
        """Alias for :attr:`max_file_size_bytes`."""
        return self.max_file_size_bytes

    @property
    def dimensions(self) -> tuple[str, ...]:
        """Alias for :attr:`resolutions`."""
        return self.resolutions

    @property
    def requires_audio(self) -> bool:
        """Alias for :attr:`audio_required`."""
        return self.audio_required

    @property
    def loudness(self) -> float | None:
        """Alias for :attr:`loudness_lufs`."""
        return self.loudness_lufs

    @property
    def loudness_tolerance(self) -> float | None:
        """Alias for :attr:`loudness_tolerance_lufs`."""
        return self.loudness_tolerance_lufs

    def supports_resolution(self, resolution: str) -> bool:
        """Whether *resolution* is listed for this media type."""
        return resolution in self.resolutions

    def supports_aspect_ratio(self, aspect_ratio: str) -> bool:
        """Whether *aspect_ratio* is listed for this media type."""
        return aspect_ratio in self.aspect_ratios

    def __getitem__(self, key: str) -> Any:
        aliases: dict[str, Any] = {
            "media_type": self.media_type,
            "type": self.media_type,
            "duration": self.duration,
            "min_duration_seconds": self.min_duration_seconds,
            "max_duration_seconds": self.max_duration_seconds,
            "resolutions": self.resolutions,
            "resolution": self.resolutions,
            "dimensions": self.resolutions,
            "aspect_ratios": self.aspect_ratios,
            "aspect_ratio": self.aspect_ratios,
            "max_file_size_bytes": self.max_file_size_bytes,
            "max_size_bytes": self.max_file_size_bytes,
            "max_file_size": self.max_file_size_bytes,
            "audio_required": self.audio_required,
            "requires_audio": self.audio_required,
            "loudness_lufs": self.loudness_lufs,
            "loudness": self.loudness_lufs,
            "loudness_tolerance_lufs": self.loudness_tolerance_lufs,
            "loudness_tolerance": self.loudness_tolerance_lufs,
        }
        try:
            return aliases[key]
        except KeyError as exc:
            raise KeyError(key) from exc


class MediaLimitsCollection(Mapping[str, MediaLimits]):
    """Media-type limits keyed by type, with the primary limit as a shortcut."""

    def __init__(self, limits: Mapping[str, MediaLimits] | None = None) -> None:
        self._limits: dict[str, MediaLimits] = dict(limits or {})

    def __getitem__(self, key: str) -> MediaLimits:
        try:
            return self._limits[_normalise_name(key)]
        except KeyError as exc:
            available = ", ".join(sorted(self._limits)) or "none"
            raise KeyError(f"{key} (available: {available})") from exc

    def __iter__(self) -> Iterator[str]:
        return iter(self._limits)

    def __len__(self) -> int:
        return len(self._limits)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.primary, name)

    @property
    def primary(self) -> MediaLimits:
        """The first declared media type, used for flat platform shortcuts."""
        if not self._limits:
            return MediaLimits("")
        return next(iter(self._limits.values()))

    def for_media_type(self, media_type: str) -> MediaLimits | None:
        """Return the limits for *media_type*, or ``None`` when unsupported."""
        return self._limits.get(_normalise_name(media_type))

    def media_types(self) -> tuple[str, ...]:
        """Return the declared media types in order."""
        return tuple(self._limits)


class DurationCollection(Mapping[str, DurationLimits]):
    """Per-media duration ranges with the primary range as a shortcut."""

    def __init__(self, limits: Mapping[str, MediaLimits] | None = None) -> None:
        self._limits: dict[str, MediaLimits] = dict(limits or {})

    def __getitem__(self, key: str) -> DurationLimits:
        try:
            return self._limits[_normalise_name(key)].duration
        except KeyError as exc:
            available = ", ".join(sorted(self._limits)) or "none"
            raise KeyError(f"{key} (available: {available})") from exc

    def __iter__(self) -> Iterator[str]:
        return iter(self._limits)

    def __len__(self) -> int:
        return len(self._limits)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.primary, name)

    @property
    def primary(self) -> DurationLimits:
        """The duration range of the first declared media type."""
        if not self._limits:
            return DurationLimits()
        return next(iter(self._limits.values())).duration


@dataclass(frozen=True, slots=True)
class Platform:
    """The publishing rules for one platform."""

    id: str
    name: str = ""
    media_types: tuple[str, ...] = ()
    limits: MediaLimitsCollection = field(default_factory=MediaLimitsCollection)
    caption_limits: CaptionLimits = field(default_factory=CaptionLimits)
    safe_area: SafeArea = field(default_factory=SafeArea)
    disclosure: DisclosureRules = field(default_factory=DisclosureRules)
    chapter_rules: ChapterRules = field(default_factory=ChapterRules)
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.media_types and len(self.limits):
            object.__setattr__(self, "media_types", self.limits.media_types())
        if self.media_types and not len(self.limits):
            object.__setattr__(
                self,
                "limits",
                MediaLimitsCollection(
                    {
                        media_type: MediaLimits(media_type)
                        for media_type in self.media_types
                    }
                ),
            )

    @property
    def key(self) -> str:
        """Alias for :attr:`id`."""
        return self.id

    @property
    def platform(self) -> str:
        """Alias for :attr:`id`."""
        return self.id

    @property
    def display_name(self) -> str:
        """The human-readable name, falling back to the id."""
        return self.name or self.id

    @property
    def title(self) -> str:
        """Alias for :attr:`display_name`."""
        return self.display_name

    @property
    def captions(self) -> CaptionLimits:
        """Alias for :attr:`caption_limits`."""
        return self.caption_limits

    @property
    def caption(self) -> CaptionLimits:
        """Alias for :attr:`caption_limits`."""
        return self.caption_limits

    @property
    def safe_areas(self) -> SafeArea:
        """Alias for :attr:`safe_area`."""
        return self.safe_area

    @property
    def disclosure_rules(self) -> DisclosureRules:
        """Alias for :attr:`disclosure`."""
        return self.disclosure

    @property
    def durations(self) -> DurationCollection:
        """Duration ranges keyed by media type."""
        return DurationCollection(self.limits)

    @property
    def duration(self) -> DurationLimits:
        """The primary media type's duration range."""
        return self.limits.primary.duration

    @property
    def resolutions(self) -> tuple[str, ...]:
        """The primary media type's resolutions."""
        return self.limits.primary.resolutions

    @property
    def aspect_ratios(self) -> tuple[str, ...]:
        """The primary media type's aspect ratios."""
        return self.limits.primary.aspect_ratios

    @property
    def max_file_size_bytes(self) -> int | None:
        """The primary media type's maximum file size."""
        return self.limits.primary.max_file_size_bytes

    @property
    def max_size_bytes(self) -> int | None:
        """Alias for :attr:`max_file_size_bytes`."""
        return self.max_file_size_bytes

    @property
    def min_duration_seconds(self) -> float | None:
        """The primary media type's minimum duration."""
        return self.duration.min_seconds

    @property
    def max_duration_seconds(self) -> float | None:
        """The primary media type's maximum duration."""
        return self.duration.max_seconds

    @property
    def min_duration(self) -> float | None:
        """Alias for :attr:`min_duration_seconds`."""
        return self.min_duration_seconds

    @property
    def max_duration(self) -> float | None:
        """Alias for :attr:`max_duration_seconds`."""
        return self.max_duration_seconds

    @property
    def resolution(self) -> str | None:
        """The primary media type's first resolution."""
        return self.resolutions[0] if self.resolutions else None

    @property
    def aspect_ratio(self) -> str | None:
        """The primary media type's first aspect ratio."""
        return self.aspect_ratios[0] if self.aspect_ratios else None

    @property
    def max_caption_length(self) -> int | None:
        """The caption character limit."""
        return self.caption_limits.max_length

    @property
    def max_title_length(self) -> int | None:
        """The title character limit."""
        return self.caption_limits.max_title_length

    @property
    def max_description_length(self) -> int | None:
        """The description character limit."""
        return self.caption_limits.max_description_length

    @property
    def max_hashtags(self) -> int | None:
        """The hashtag limit."""
        return self.caption_limits.max_hashtags

    @property
    def max_mentions(self) -> int | None:
        """The mention limit."""
        return self.caption_limits.max_mentions

    @property
    def disclosure_required(self) -> bool:
        """Whether the platform requires an AI disclosure."""
        return self.disclosure.required

    @property
    def disclosure_text(self) -> str:
        """The disclosure text configured for the platform."""
        return self.disclosure.text

    @property
    def on_screen_label(self) -> str:
        """The optional on-screen disclosure label."""
        return self.disclosure.on_screen_label

    @property
    def media(self) -> MediaLimitsCollection:
        """Alias for :attr:`limits`."""
        return self.limits

    @property
    def audio_required(self) -> bool:
        """Whether the primary media type requires an audio track."""
        return self.limits.primary.audio_required

    @property
    def loudness_lufs(self) -> float | None:
        """The primary media type's loudness target."""
        return self.limits.primary.loudness_lufs

    @property
    def loudness_tolerance_lufs(self) -> float | None:
        """The primary media type's loudness tolerance."""
        return self.limits.primary.loudness_tolerance_lufs

    def supports(self, media_type: str) -> bool:
        """Whether this platform supports *media_type*."""
        return self.limits.for_media_type(media_type) is not None

    def supports_media_type(self, media_type: str) -> bool:
        """Alias for :meth:`supports`."""
        return self.supports(media_type)

    def limit(self, media_type: str | None = None) -> MediaLimits | None:
        """Return the limits for *media_type*, or the primary limits."""
        if media_type is None:
            return self.limits.primary
        return self.limits.for_media_type(media_type)

    def for_media_type(self, media_type: str) -> MediaLimits | None:
        """Alias for :meth:`limit`."""
        return self.limit(media_type)

    def choose_outputs(self, outputs: Sequence[Asset]) -> dict[str, Asset | None]:
        """Choose the best run output for each media type this platform supports.

        Returns a mapping of platform media type to the matching asset, or
        ``None`` when no output fits that media type's declared limits. Outputs
        are inspected in order and the first fit wins, so callers can rank
        preferred outputs earlier.
        """
        chosen: dict[str, Asset | None] = {}
        for media_type in self.media_types:
            limits = self.limit(media_type)
            chosen[media_type] = (
                None if limits is None else _pick_output(media_type, limits, outputs)
            )
        return chosen

    def __getitem__(self, key: str) -> Any:
        aliases: dict[str, Any] = {
            "id": self.id,
            "key": self.id,
            "platform": self.id,
            "name": self.display_name,
            "display_name": self.display_name,
            "media_types": self.media_types,
            "limits": self.limits,
            "duration": self.duration,
            "durations": self.durations,
            "resolutions": self.resolutions,
            "resolution": self.resolutions,
            "aspect_ratios": self.aspect_ratios,
            "aspect_ratio": self.aspect_ratios,
            "max_file_size_bytes": self.max_file_size_bytes,
            "max_size_bytes": self.max_file_size_bytes,
            "caption_limits": self.caption_limits,
            "captions": self.caption_limits,
            "caption": self.caption_limits,
            "safe_area": self.safe_area,
            "safe_areas": self.safe_area,
            "disclosure": self.disclosure,
            "disclosure_rules": self.disclosure,
            "chapter_rules": self.chapter_rules,
            "loudness_lufs": self.loudness_lufs,
            "loudness_tolerance_lufs": self.loudness_tolerance_lufs,
            "aliases": self.aliases,
        }
        try:
            return aliases[key]
        except KeyError as exc:
            raise KeyError(key) from exc


_IMAGE_KINDS = frozenset({"image", "carousel", "web_card"})
_AUDIO_KINDS = frozenset({"audio", "music", "voice", "sfx"})


def _broad_kind(media_type: str) -> str:
    """Collapse platform media types and asset kinds to video/image/audio."""
    key = _normalise_name(media_type)
    if key == "video":
        return "video"
    if key in _IMAGE_KINDS:
        return "image"
    if key in _AUDIO_KINDS:
        return "audio"
    return key


def _resolution_key(width: int | None, height: int | None) -> str | None:
    if width is None or height is None:
        return None
    return f"{width}x{height}"


def _aspect_ratio(width: int | None, height: int | None) -> float | None:
    if not width or not height:
        return None
    return width / height


def _parse_aspect_ratio(text: str) -> float | None:
    try:
        width_text, height_text = str(text).split(":")
        return float(width_text) / float(height_text)
    except (ValueError, ZeroDivisionError):
        return None


def _ratios_close(first: float, second: float | None, tolerance: float = 1e-6) -> bool:
    return second is not None and abs(first - second) <= tolerance


def _fits_limits(asset: Asset, limits: MediaLimits) -> bool:
    analysis = asset.analysis
    width = analysis.width
    height = analysis.height

    if limits.resolutions:
        key = _resolution_key(width, height)
        if key is None or key not in limits.resolutions:
            return False
    elif limits.aspect_ratios:
        ratio = _aspect_ratio(width, height)
        if ratio is None or not any(
            _ratios_close(ratio, _parse_aspect_ratio(ratio_text))
            for ratio_text in limits.aspect_ratios
        ):
            return False

    duration = analysis.duration_seconds
    if duration is not None:
        if (
            limits.duration.min_seconds is not None
            and duration < limits.duration.min_seconds
        ):
            return False
        if (
            limits.duration.max_seconds is not None
            and duration > limits.duration.max_seconds
        ):
            return False

    return not (limits.audio_required and analysis.has_audio is False)


def _pick_output(
    media_type: str, limits: MediaLimits, outputs: Sequence[Asset]
) -> Asset | None:
    target = _broad_kind(media_type)
    for asset in outputs:
        if _broad_kind(asset.media_type.value) != target:
            continue
        if _fits_limits(asset, limits):
            return asset
    return None


class PlatformCatalog(Mapping[str, Platform]):
    """The platforms described by one ``platforms.yaml`` file."""

    def __init__(
        self,
        platforms: Mapping[str, Platform] | Iterable[Platform],
        *,
        source: Path | None = None,
        schema_version: int = PLATFORMS_SCHEMA_VERSION,
        defaults: Mapping[str, Any] | None = None,
    ) -> None:
        entries: dict[str, Platform] = {}
        items = platforms.values() if isinstance(platforms, Mapping) else platforms
        for platform in items:
            key = _normalise_name(platform.id)
            if key in entries:
                raise ConfigError(f"duplicate platform id: {platform.id}")
            entries[key] = platform
        self._platforms = entries
        self.source = source
        self.schema_version = schema_version
        self.defaults: dict[str, Any] = dict(defaults or {})

    def __getitem__(self, key: str) -> Platform:
        try:
            return self._platforms[_lookup_name(key)]
        except KeyError as exc:
            available = ", ".join(self.ids()) or "none"
            raise KeyError(f"{key} (available: {available})") from exc

    def __iter__(self) -> Iterator[str]:
        return iter(self.ids())

    def __len__(self) -> int:
        return len(self._platforms)

    def get(self, key: str, default: Any = None) -> Any:
        """Return the platform named *key*, or *default* when unknown."""
        return self._platforms.get(_lookup_name(key), default)

    def ids(self) -> tuple[str, ...]:
        """Return every platform id in deterministic order."""
        return tuple(sorted(self._platforms))

    def names(self) -> tuple[str, ...]:
        """Alias for :meth:`ids`."""
        return self.ids()

    def entries(self) -> dict[str, Platform]:
        """Return a copy of the id-to-platform mapping."""
        return dict(self._platforms)

    @property
    def platforms(self) -> dict[str, Platform]:
        """Return a copy of the id-to-platform mapping."""
        return self.entries()

    def media_types(self) -> tuple[str, ...]:
        """Return every media type any platform supports."""
        types: set[str] = set()
        for platform in self._platforms.values():
            types.update(platform.media_types)
        return tuple(sorted(types))

    def require(self, key: str) -> Platform:
        """Return the platform named *key* or raise a clear configuration error."""
        platform = self._platforms.get(_lookup_name(key))
        if platform is None:
            available = ", ".join(self.ids()) or "none"
            raise ConfigError(
                f"unknown platform: {key} (available: {available})",
                details={"platform": key, "available": list(self.ids())},
            )
        return platform

    def for_platform(self, key: str) -> Platform:
        """Alias for :meth:`require`."""
        return self.require(key)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and _lookup_name(key) in self._platforms

    @classmethod
    def from_mapping(
        cls,
        payload: Any,
        *,
        source: Path | None = None,
    ) -> PlatformCatalog:
        """Build a catalog from a parsed ``platforms.yaml`` payload."""
        if not isinstance(payload, Mapping):
            raise ConfigError(_error("platform data must be a mapping", source))

        unknown = sorted(str(key) for key in payload if key not in _TOP_LEVEL_KEYS)
        if unknown:
            raise ConfigError(
                _error(f"unknown platform data key: {', '.join(unknown)}", source)
            )

        version = payload.get("schema_version", PLATFORMS_SCHEMA_VERSION)
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise ConfigError(
                _error(
                    "platform data schema_version must be a positive integer, "
                    f"got {version!r}",
                    source,
                )
            )
        if version > PLATFORMS_SCHEMA_VERSION:
            raise ConfigError(
                _error(
                    f"platform data schema version {version} is newer than supported "
                    f"version {PLATFORMS_SCHEMA_VERSION}; upgrade aivia",
                    source,
                )
            )

        defaults_raw = payload.get("defaults", {})
        if defaults_raw is None:
            defaults_raw = {}
        if not isinstance(defaults_raw, Mapping):
            raise ConfigError(
                _error("platform data defaults must be a mapping", source)
            )

        raw = payload.get("platforms", {})
        if raw is None:
            raw = {}
        platforms = _parse_platforms(raw, defaults=defaults_raw, source=source)
        return cls(
            platforms,
            source=source,
            schema_version=version,
            defaults=dict(defaults_raw),
        )

    @classmethod
    def load(cls, path: str | Path) -> PlatformCatalog:
        """Load a catalog from *path*."""
        target = Path(path).expanduser()
        try:
            text = target.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"cannot read platform data {target}: {exc}") from exc
        try:
            payload = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in platform data {target}: {exc}") from exc
        return cls.from_mapping(payload, source=target)

    @classmethod
    def load_default(cls) -> PlatformCatalog:
        """Load the packaged default platform data."""
        return cls.load(default_platforms_path())


def _error(message: str, source: Path | None) -> str:
    return f"{message} in {source}" if source is not None else message


def _normalise_name(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _lookup_name(value: str) -> str:
    key = _normalise_name(value)
    return _PLATFORM_ALIASES.get(key, key)


def _first(data: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _parse_platforms(
    raw: Any,
    *,
    defaults: Mapping[str, Any],
    source: Path | None,
) -> list[Platform]:
    if isinstance(raw, Mapping):
        return [
            _parse_platform(str(name), entry, defaults=defaults, source=source)
            for name, entry in raw.items()
        ]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        platforms: list[Platform] = []
        for entry in raw:
            if not isinstance(entry, Mapping):
                raise ConfigError(_error("platform entry must be a mapping", source))
            values = dict(entry)
            platform_id = values.pop("id", values.pop("key", values.pop("name", None)))
            if not isinstance(platform_id, str) or not platform_id.strip():
                raise ConfigError(
                    _error("platform entry requires a non-empty id", source)
                )
            platforms.append(
                _parse_platform(
                    platform_id,
                    values,
                    defaults=defaults,
                    source=source,
                )
            )
        return platforms
    raise ConfigError(
        _error("platform data platforms must be a mapping or list", source)
    )


def _parse_platform(
    platform_id: str,
    raw: Any,
    *,
    defaults: Mapping[str, Any],
    source: Path | None,
) -> Platform:
    key = _normalise_name(platform_id)
    if not key:
        raise ConfigError(_error("platform id must be non-empty", source))
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ConfigError(
            _error(f"platform {key!r} settings must be a mapping", source)
        )
    data = _deep_merge(defaults, raw) if defaults else dict(raw)

    unknown = sorted(str(name) for name in data if name not in _PLATFORM_KEYS)
    if unknown:
        raise ConfigError(
            _error(
                f"platform {key!r} has unknown key: {', '.join(unknown)}",
                source,
            )
        )

    name = (
        _optional_string(
            _first(data, "name", "display_name", "title", "label"),
            field=f"platform {key!r} name",
            source=source,
        )
        or key
    )
    media_types = _string_tuple(
        _first(data, "media_types", "media_type", "types", "output_types"),
        field=f"platform {key!r} media_types",
        source=source,
    )
    aliases = _string_tuple(
        _first(data, "aliases", "alias"),
        field=f"platform {key!r} aliases",
        source=source,
    )

    limits_raw = _first(
        data,
        "limits",
        "media_limits",
        "media",
        "per_media_type",
        "specs",
    )
    limits: dict[str, MediaLimits] = {}
    if limits_raw is not None:
        if not isinstance(limits_raw, Mapping):
            raise ConfigError(
                _error(f"platform {key!r} limits must be a mapping", source)
            )
        if _looks_like_media_limits(limits_raw):
            media_type = _optional_string(
                limits_raw.get("media_type"),
                field="media type",
                source=source,
            ) or (media_types[0] if media_types else "video")
            limits[_normalise_name(media_type)] = _parse_media_limits(
                media_type,
                limits_raw,
                field=f"platform {key!r} limits.{media_type}",
                source=source,
            )
        else:
            for media_type, spec in limits_raw.items():
                if not isinstance(spec, Mapping):
                    raise ConfigError(
                        _error(
                            f"platform {key!r} limits.{media_type} must be a mapping",
                            source,
                        )
                    )
                limits[_normalise_name(str(media_type))] = _parse_media_limits(
                    str(media_type),
                    spec,
                    field=f"platform {key!r} limits.{media_type}",
                    source=source,
                )

    durations_raw = _first(data, "duration", "durations", "duration_limits")
    if durations_raw is not None:
        if _looks_like_duration(durations_raw):
            media_type = media_types[0] if media_types else "video"
            limits[_normalise_name(media_type)] = _merge_media_limits(
                limits.get(_normalise_name(media_type), MediaLimits(media_type)),
                duration=_parse_duration(
                    durations_raw,
                    field=f"platform {key!r} duration",
                    source=source,
                ),
            )
        elif isinstance(durations_raw, Mapping):
            for media_type, spec in durations_raw.items():
                limits[_normalise_name(str(media_type))] = _merge_media_limits(
                    limits.get(
                        _normalise_name(str(media_type)),
                        MediaLimits(str(media_type)),
                    ),
                    duration=_parse_duration(
                        spec,
                        field=f"platform {key!r} duration.{media_type}",
                        source=source,
                    ),
                )
        else:
            raise ConfigError(
                _error(f"platform {key!r} durations must be a mapping", source)
            )

    flat = {
        "resolutions": _first(data, "resolutions", "resolution", "dimensions"),
        "aspect_ratios": _first(data, "aspect_ratios", "aspect_ratio", "aspects"),
        "max_file_size_bytes": _first(
            data,
            "size_limits",
            "size_limit",
            "max_file_size_bytes",
            "max_size_bytes",
            "max_file_size",
            "max_bytes",
        ),
        "audio_required": _first(data, "audio_required", "requires_audio"),
        "loudness_lufs": _first(data, "loudness_lufs", "loudness"),
        "loudness_tolerance_lufs": _first(
            data, "loudness_tolerance_lufs", "loudness_tolerance"
        ),
    }
    if any(value is not None for value in flat.values()):
        media_type = media_types[0] if media_types else "video"
        limits[_normalise_name(media_type)] = _merge_media_limits(
            limits.get(_normalise_name(media_type), MediaLimits(media_type)),
            resolutions=_parse_resolutions(
                flat["resolutions"],
                field=f"platform {key!r} resolutions",
                source=source,
            ),
            aspect_ratios=_parse_aspect_ratios(
                flat["aspect_ratios"],
                field=f"platform {key!r} aspect_ratios",
                source=source,
            ),
            max_file_size_bytes=_parse_size(
                flat["max_file_size_bytes"],
                field=f"platform {key!r} max_file_size_bytes",
                source=source,
            ),
            audio_required=_parse_bool(
                flat["audio_required"],
                field=f"platform {key!r} audio_required",
                source=source,
            ),
            loudness_lufs=_parse_number(
                flat["loudness_lufs"],
                field=f"platform {key!r} loudness_lufs",
                source=source,
            ),
            loudness_tolerance_lufs=_parse_number(
                flat["loudness_tolerance_lufs"],
                field=f"platform {key!r} loudness_tolerance_lufs",
                source=source,
            ),
        )

    if not limits:
        for media_type in media_types or ("video",):
            limits[_normalise_name(media_type)] = MediaLimits(media_type)

    caption_limits = _parse_caption_limits(
        _first(
            data,
            "caption_limits",
            "captions",
            "caption",
            "caption_limit",
            "text_limits",
        ),
        field=f"platform {key!r} caption_limits",
        source=source,
    )
    safe_area = _parse_safe_area(
        _first(data, "safe_area", "safe_areas", "margins"),
        field=f"platform {key!r} safe_area",
        source=source,
    )
    disclosure = _parse_disclosure(
        _first(data, "disclosure", "disclosure_rules"),
        field=f"platform {key!r} disclosure",
        source=source,
    )
    chapter_rules = _parse_chapter_rules(
        _first(data, "chapter_rules", "chapters"),
        field=f"platform {key!r} chapter_rules",
        source=source,
    )

    return Platform(
        id=key,
        name=name,
        media_types=media_types or tuple(limits),
        limits=MediaLimitsCollection(limits),
        caption_limits=caption_limits,
        safe_area=safe_area,
        disclosure=disclosure,
        chapter_rules=chapter_rules,
        aliases=aliases,
    )


def _looks_like_media_limits(value: Mapping[str, Any]) -> bool:
    return bool({str(key) for key in value} & _MEDIA_LIMIT_KEYS)


def _looks_like_duration(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    return isinstance(value, Mapping) and bool(
        {str(key) for key in value} & _DURATION_KEYS
    )


def _merge_media_limits(
    limits: MediaLimits,
    *,
    duration: DurationLimits | None = None,
    resolutions: tuple[str, ...] | None = None,
    aspect_ratios: tuple[str, ...] | None = None,
    max_file_size_bytes: int | None = None,
    audio_required: bool | None = None,
    loudness_lufs: float | None = None,
    loudness_tolerance_lufs: float | None = None,
) -> MediaLimits:
    return MediaLimits(
        media_type=limits.media_type,
        duration=duration if duration is not None else limits.duration,
        resolutions=(resolutions if resolutions is not None else limits.resolutions),
        aspect_ratios=(
            aspect_ratios if aspect_ratios is not None else limits.aspect_ratios
        ),
        max_file_size_bytes=(
            max_file_size_bytes
            if max_file_size_bytes is not None
            else limits.max_file_size_bytes
        ),
        audio_required=(
            audio_required if audio_required is not None else limits.audio_required
        ),
        loudness_lufs=(
            loudness_lufs if loudness_lufs is not None else limits.loudness_lufs
        ),
        loudness_tolerance_lufs=(
            loudness_tolerance_lufs
            if loudness_tolerance_lufs is not None
            else limits.loudness_tolerance_lufs
        ),
    )


def _parse_media_limits(
    media_type: str,
    raw: Mapping[str, Any],
    *,
    field: str,
    source: Path | None,
) -> MediaLimits:
    unknown = sorted(str(name) for name in raw if name not in _MEDIA_LIMIT_KEYS)
    if unknown:
        raise ConfigError(
            _error(f"{field} has unknown key: {', '.join(unknown)}", source)
        )
    declared = _optional_string(
        _first(raw, "media_type", "type"),
        field=f"{field} media_type",
        source=source,
    )
    duration = _parse_duration(
        _first(raw, "duration", "durations", "duration_limits"),
        field=f"{field} duration",
        source=source,
    )
    if duration.min_seconds is None and duration.max_seconds is None:
        duration = _parse_duration(
            {
                "min": _first(
                    raw,
                    "min_seconds",
                    "min",
                    "minimum",
                    "min_duration",
                    "min_duration_seconds",
                ),
                "max": _first(
                    raw,
                    "max_seconds",
                    "max",
                    "maximum",
                    "max_duration",
                    "max_duration_seconds",
                ),
            },
            field=f"{field} duration",
            source=source,
        )
    return MediaLimits(
        media_type=_normalise_name(declared or media_type),
        duration=duration,
        resolutions=_parse_resolutions(
            _first(raw, "resolutions", "resolution", "dimensions"),
            field=f"{field} resolutions",
            source=source,
        ),
        aspect_ratios=_parse_aspect_ratios(
            _first(raw, "aspect_ratios", "aspect_ratio", "aspects"),
            field=f"{field} aspect_ratios",
            source=source,
        ),
        max_file_size_bytes=_parse_size(
            _first(
                raw,
                "size_limits",
                "size_limit",
                "max_file_size_bytes",
                "max_size_bytes",
                "max_file_size",
                "max_bytes",
            ),
            field=f"{field} max_file_size_bytes",
            source=source,
        ),
        audio_required=_parse_bool(
            _first(raw, "audio_required", "requires_audio"),
            field=f"{field} audio_required",
            source=source,
        ),
        loudness_lufs=_parse_number(
            _first(raw, "loudness_lufs", "loudness"),
            field=f"{field} loudness_lufs",
            source=source,
        ),
        loudness_tolerance_lufs=_parse_number(
            _first(raw, "loudness_tolerance_lufs", "loudness_tolerance"),
            field=f"{field} loudness_tolerance_lufs",
            source=source,
        ),
    )


def _parse_duration(
    value: Any,
    *,
    field: str,
    source: Path | None,
) -> DurationLimits:
    if value is None:
        return DurationLimits()
    if isinstance(value, Mapping):
        if "duration" in value and len(value) == 1:
            return _parse_duration(value["duration"], field=field, source=source)
        unknown = sorted(str(key) for key in value if key not in _DURATION_KEYS)
        if unknown:
            raise ConfigError(
                _error(f"{field} has unknown key: {', '.join(unknown)}", source)
            )
        minimum = _parse_number(
            _first(
                value,
                "min_seconds",
                "min",
                "minimum",
                "min_duration",
                "min_duration_seconds",
            ),
            field=f"{field} min",
            source=source,
        )
        maximum = _parse_number(
            _first(
                value,
                "max_seconds",
                "max",
                "maximum",
                "max_duration",
                "max_duration_seconds",
            ),
            field=f"{field} max",
            source=source,
        )
        _check_duration_range(minimum, maximum, field=field, source=source)
        return DurationLimits(min_seconds=minimum, max_seconds=maximum)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ConfigError(
                _error(f"{field} must have a minimum and maximum", source)
            )
        minimum = _parse_number(value[0], field=f"{field} min", source=source)
        maximum = _parse_number(value[1], field=f"{field} max", source=source)
        _check_duration_range(minimum, maximum, field=field, source=source)
        return DurationLimits(min_seconds=minimum, max_seconds=maximum)
    maximum = _parse_number(value, field=field, source=source)
    _check_duration_range(None, maximum, field=field, source=source)
    return DurationLimits(max_seconds=maximum)


def _check_duration_range(
    minimum: float | None,
    maximum: float | None,
    *,
    field: str,
    source: Path | None,
) -> None:
    if minimum is not None and minimum < 0:
        raise ConfigError(_error(f"{field} minimum must not be negative", source))
    if maximum is not None and maximum < 0:
        raise ConfigError(_error(f"{field} maximum must not be negative", source))
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ConfigError(
            _error(f"{field} minimum must not exceed its maximum", source)
        )


def _parse_caption_limits(
    value: Any,
    *,
    field: str,
    source: Path | None,
) -> CaptionLimits:
    if value is None:
        return CaptionLimits()
    if not isinstance(value, Mapping):
        raise ConfigError(_error(f"{field} must be a mapping", source))
    unknown = sorted(str(key) for key in value if key not in _CAPTION_KEYS)
    if unknown:
        raise ConfigError(
            _error(f"{field} has unknown key: {', '.join(unknown)}", source)
        )
    return CaptionLimits(
        max_length=_parse_int(
            _first(
                value,
                "max_length",
                "max_chars",
                "max_characters",
                "length",
                "caption",
                "max_caption_length",
            ),
            field=f"{field} max_length",
            source=source,
        ),
        max_title_length=_parse_int(
            _first(value, "max_title_length", "title", "max_title"),
            field=f"{field} max_title_length",
            source=source,
        ),
        max_description_length=_parse_int(
            _first(value, "max_description_length", "description", "max_description"),
            field=f"{field} max_description_length",
            source=source,
        ),
        max_hashtags=_parse_int(
            _first(value, "max_hashtags", "hashtags", "tags"),
            field=f"{field} max_hashtags",
            source=source,
        ),
        max_mentions=_parse_int(
            _first(value, "max_mentions", "mentions"),
            field=f"{field} max_mentions",
            source=source,
        ),
    )


def _parse_safe_area(
    value: Any,
    *,
    field: str,
    source: Path | None,
) -> SafeArea:
    if value is None:
        return SafeArea()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 4:
            raise ConfigError(
                _error(f"{field} must list top, right, bottom and left", source)
            )
        top, right, bottom, left = (
            _parse_fraction(item, field=f"{field} margin", source=source)
            for item in value
        )
        return SafeArea(top=top, bottom=bottom, left=left, right=right)
    if not isinstance(value, Mapping):
        raise ConfigError(_error(f"{field} must be a mapping", source))
    unknown = sorted(str(key) for key in value if key not in _SAFE_AREA_KEYS)
    if unknown:
        raise ConfigError(
            _error(f"{field} has unknown key: {', '.join(unknown)}", source)
        )
    return SafeArea(
        top=_parse_fraction(
            _first(value, "top", "top_percent"),
            field=f"{field} top",
            source=source,
        ),
        bottom=_parse_fraction(
            _first(value, "bottom", "bottom_percent"),
            field=f"{field} bottom",
            source=source,
        ),
        left=_parse_fraction(
            _first(value, "left", "left_percent"),
            field=f"{field} left",
            source=source,
        ),
        right=_parse_fraction(
            _first(value, "right", "right_percent"),
            field=f"{field} right",
            source=source,
        ),
    )


def _parse_disclosure(
    value: Any,
    *,
    field: str,
    source: Path | None,
) -> DisclosureRules:
    if value is None:
        return DisclosureRules()
    if isinstance(value, bool):
        return DisclosureRules(required=value)
    if not isinstance(value, Mapping):
        raise ConfigError(_error(f"{field} must be a mapping or boolean", source))
    unknown = sorted(str(key) for key in value if key not in _DISCLOSURE_KEYS)
    if unknown:
        raise ConfigError(
            _error(f"{field} has unknown key: {', '.join(unknown)}", source)
        )
    return DisclosureRules(
        required=_parse_bool(
            _first(value, "required", "requires_disclosure", "required_flag"),
            field=f"{field} required",
            source=source,
        ),
        text=_optional_string(
            _first(value, "text", "disclosure_text", "label_text"),
            field=f"{field} text",
            source=source,
        )
        or "",
        on_screen_label=_optional_string(
            _first(value, "on_screen_label", "label", "on_screen"),
            field=f"{field} on_screen_label",
            source=source,
        )
        or "",
        metadata_field=_optional_string(
            _first(value, "metadata_field", "field"),
            field=f"{field} metadata_field",
            source=source,
        )
        or "",
        metadata=_parse_bool(
            _first(value, "metadata", "metadata_supported"),
            field=f"{field} metadata",
            source=source,
        ),
        url=_optional_string(
            value.get("url"),
            field=f"{field} url",
            source=source,
        )
        or "",
    )


def _parse_chapter_rules(
    value: Any,
    *,
    field: str,
    source: Path | None,
) -> ChapterRules:
    if value is None:
        return ChapterRules()
    if isinstance(value, bool):
        return ChapterRules(enabled=value)
    if not isinstance(value, Mapping):
        raise ConfigError(_error(f"{field} must be a mapping or boolean", source))
    unknown = sorted(str(key) for key in value if key not in _CHAPTER_RULES_KEYS)
    if unknown:
        raise ConfigError(
            _error(f"{field} has unknown key: {', '.join(unknown)}", source)
        )
    enabled_raw = _first(value, "enabled")
    enabled = (
        _parse_bool(enabled_raw, field=f"{field} enabled", source=source)
        if enabled_raw is not None
        else True
    )
    return ChapterRules(
        enabled=enabled,
        max_chapters=_parse_int(
            _first(value, "max_chapters", "max_chapter_count"),
            field=f"{field} max_chapters",
            source=source,
        ),
        min_chapter_duration_seconds=_parse_number(
            _first(
                value,
                "min_chapter_duration_seconds",
                "min_chapter_duration",
                "min_duration_seconds",
            ),
            field=f"{field} min_chapter_duration_seconds",
            source=source,
        ),
        max_chapter_title_length=_parse_int(
            _first(
                value,
                "max_chapter_title_length",
                "max_title_length",
                "chapter_title_length",
            ),
            field=f"{field} max_chapter_title_length",
            source=source,
        ),
        separator=_parse_chapter_separator(
            _first(value, "separator", "chapter_separator"),
            field=f"{field} separator",
            source=source,
        ),
    )


def _parse_chapter_separator(value: Any, *, field: str, source: Path | None) -> str:
    if value is None:
        return "\n"
    if not isinstance(value, str):
        raise ConfigError(_error(f"{field} must be a string", source))
    return value or "\n"


def _parse_resolutions(
    value: Any,
    *,
    field: str,
    source: Path | None,
) -> tuple[str, ...]:
    items = _as_items(value, field=field, source=source)
    resolutions: list[str] = []
    for item in items:
        if isinstance(item, Mapping):
            width = _parse_int(item.get("width"), field=f"{field} width", source=source)
            height = _parse_int(
                item.get("height"), field=f"{field} height", source=source
            )
            if not width or not height:
                raise ConfigError(
                    _error(f"{field} entries need positive width and height", source)
                )
            resolution = f"{width}x{height}"
        elif isinstance(item, str):
            resolution = item.strip()
        else:
            raise ConfigError(
                _error(f"{field} entries must be strings or mappings", source)
            )
        match = _RESOLUTION.fullmatch(resolution)
        if (
            match is None
            or int(match.group("width")) <= 0
            or int(match.group("height")) <= 0
        ):
            raise ConfigError(
                _error(
                    f"{field} entry must look like 1080x1920: {resolution!r}",
                    source,
                )
            )
        if resolution not in resolutions:
            resolutions.append(resolution)
    return tuple(resolutions)


def _parse_aspect_ratios(
    value: Any,
    *,
    field: str,
    source: Path | None,
) -> tuple[str, ...]:
    items = _as_items(value, field=field, source=source)
    ratios: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise ConfigError(_error(f"{field} entries must be strings", source))
        ratio = item.strip()
        match = _ASPECT_RATIO.fullmatch(ratio)
        if (
            match is None
            or float(match.group("width")) <= 0
            or float(match.group("height")) <= 0
        ):
            raise ConfigError(
                _error(f"{field} entry must look like 9:16: {ratio!r}", source)
            )
        if ratio not in ratios:
            ratios.append(ratio)
    return tuple(ratios)


def _as_items(value: Any, *, field: str, source: Path | None) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return list(value)
    raise ConfigError(_error(f"{field} must be a string or list", source))


def _string_tuple(value: Any, *, field: str, source: Path | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = list(value)
    else:
        raise ConfigError(_error(f"{field} must be a string or list", source))
    strings: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(
                _error(f"{field} entries must be non-empty strings", source)
            )
        name = item.strip()
        if name not in strings:
            strings.append(name)
    return tuple(strings)


def _optional_string(value: Any, *, field: str, source: Path | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(_error(f"{field} must be a non-empty string", source))
    return value.strip()


def _parse_bool(value: Any, *, field: str, source: Path | None) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ConfigError(_error(f"{field} must be a boolean", source))
    return value


def _parse_int(value: Any, *, field: str, source: Path | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(_error(f"{field} must be an integer", source))
    if value < 0:
        raise ConfigError(_error(f"{field} must not be negative", source))
    return value


def _parse_number(value: Any, *, field: str, source: Path | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(_error(f"{field} must be a number", source))
    return float(value)


def _parse_size(value: Any, *, field: str, source: Path | None) -> int | None:
    if isinstance(value, Mapping):
        allowed = {"bytes", "max", "max_bytes", "max_size_bytes", "max_file_size"}
        unknown = sorted(str(key) for key in value if key not in allowed)
        if unknown:
            raise ConfigError(
                _error(f"{field} has unknown key: {', '.join(unknown)}", source)
            )
        return _parse_size(
            _first(
                value,
                "bytes",
                "max",
                "max_bytes",
                "max_size_bytes",
                "max_file_size",
            ),
            field=field,
            source=source,
        )
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(_error(f"{field} must be an integer", source))
    if value < 0:
        raise ConfigError(_error(f"{field} must not be negative", source))
    return value


def _parse_fraction(value: Any, *, field: str, source: Path | None) -> float:
    number = _parse_number(value, field=field, source=source)
    if number is None:
        return 0.0
    if not 0 <= number <= 1:
        raise ConfigError(_error(f"{field} must be between 0 and 1", source))
    return number


def _configured_platforms_file(platforms_file: Any) -> str | None:
    if platforms_file is None:
        return None
    if isinstance(platforms_file, str):
        path = platforms_file.strip()
        return path or None
    if isinstance(platforms_file, Path):
        return str(platforms_file)
    if isinstance(platforms_file, Mapping):
        if "platforms_file" in platforms_file:
            return _configured_platforms_file(platforms_file["platforms_file"])
        for key in ("path", "file", "filename", "platforms", "platform_data"):
            if key in platforms_file:
                return _configured_platforms_file(platforms_file[key])
        return None
    nested = getattr(platforms_file, "platforms_file", None)
    if nested is not None and nested is not platforms_file:
        return _configured_platforms_file(nested)
    for name in ("path", "file", "filename"):
        value = getattr(platforms_file, name, None)
        if isinstance(value, (str, Path)):
            path = str(value).strip()
            return path or None
    return None


def resolve_platforms_path(
    platforms_file: Any = None,
    *,
    config_dir: str | Path | None = None,
) -> Path:
    """Return the platform data path selected by a file section or default.

    A relative path is resolved against the config directory, never the current
    working directory. The default file name prefers the user's editable copy
    and otherwise falls back to the packaged file.
    """
    base = paths.config_dir() if config_dir is None else Path(config_dir).expanduser()
    configured = _configured_platforms_file(platforms_file) or PLATFORMS_FILENAME
    target = Path(configured).expanduser()
    if target.is_absolute():
        return target
    candidate = base / target
    if target == Path(PLATFORMS_FILENAME):
        if candidate.is_file():
            return candidate
        return paths.package_defaults_dir() / PLATFORMS_FILENAME
    return candidate


resolve_platform_data_path = resolve_platforms_path
platforms_path = resolve_platforms_path
platform_data_path = resolve_platforms_path


def default_platforms_path() -> Path:
    """Return the packaged platform data file."""
    return paths.package_defaults_dir() / PLATFORMS_FILENAME


def load_platforms(
    path: str | Path | Any | None = None,
    *,
    config_dir: str | Path | None = None,
    platforms_file: Any = None,
    config: Any = None,
) -> PlatformCatalog:
    """Load platform data from *path*, a file section or the default."""
    if (
        path is not None
        and not isinstance(path, (str, Path))
        and platforms_file is None
        and config is None
    ):
        platforms_file = path
        path = None
    if path is None:
        target = resolve_platforms_path(
            platforms_file if platforms_file is not None else config,
            config_dir=config_dir,
        )
    else:
        target = Path(path).expanduser()
        if target.is_dir():
            target = target / PLATFORMS_FILENAME
    return PlatformCatalog.load(target)


load_platform_data = load_platforms
load_platform_catalog = load_platforms
load_platforms_file = load_platforms
platform_catalog = load_platforms
read_platforms = load_platforms


def get_platform(
    name: str,
    *,
    catalog: PlatformCatalog | None = None,
    config_dir: str | Path | None = None,
) -> Platform:
    """Return one platform from *catalog* or the active platform data."""
    resolved = catalog if catalog is not None else load_platforms(config_dir=config_dir)
    return resolved.require(name)


def platform_names(*, config_dir: str | Path | None = None) -> tuple[str, ...]:
    """Return the ids of every platform in the active platform data."""
    return load_platforms(config_dir=config_dir).ids()


PlatformInfo = Platform
PlatformData = Platform
PlatformSpec = Platform
PlatformRegistry = PlatformCatalog
PlatformLimits = MediaLimits
PlatformDuration = DurationLimits
PlatformCaptionLimits = CaptionLimits
PlatformSafeArea = SafeArea
PlatformDisclosure = DisclosureRules

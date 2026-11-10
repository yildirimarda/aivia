"""A tagged local background library with static, animated and AI loop modes.

A run often needs a background image or loop that is already on disk, not
another paid generation. The library turns a manifest of local backgrounds
into deterministic indexes, so a later stage can ask for a mode, a set of
tags and a duration without scanning every file or calling a provider.

Modes:

* ``static`` — a still image used as a backdrop. No seam check applies.
* ``animated`` — an animated image (GIF or WebP) played as a backdrop.
* ``ai_loop`` — an AI generated seamless loop whose first and last frames are
  verified to match above a configured similarity threshold.

Selection is pure code. Tags, mode, duration and license are filtered first
and the surviving entries are ranked by how well they match the request. The
same library and the same filters always return the same background, which
keeps a composition reproducible offline. A manifest may be YAML or JSON;
relative background paths are resolved against the manifest directory.

Loop seamlessness is verified with :func:`aivia.compose.loop_outputs.measure_loop_seam`,
which compares the first and last frames of an animated container. Entries in
the animated and ai_loop modes are only considered verified once their seam
has been measured and meets the threshold.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
)

from aivia.core.config.media import DEFAULT_MEDIA_LOOP_SIMILARITY_THRESHOLD
from aivia.core.errors import ConfigError, ValidationError
from aivia.core.models.asset import AssetLicense
from aivia.media.image_utils import ImageFormat

__all__ = [
    "BACKGROUND_LIBRARY_SCHEMA_VERSION",
    "BackgroundLibrary",
    "BackgroundLibraryEntry",
    "BackgroundLibraryError",
    "BackgroundLibraryMatch",
    "BackgroundMode",
    "background_library",
    "build_background_library",
    "load_background_library",
    "select_background",
]

BACKGROUND_LIBRARY_SCHEMA_VERSION = 1
"""The manifest schema version understood by this release."""

_MIN_ENTRY_COUNT = 1
_ZERO = 0.0


class BackgroundMode(StrEnum):
    """The kind of a background entry."""

    STATIC = "static"
    """A still image backdrop."""

    ANIMATED = "animated"
    """An animated image backdrop (GIF or WebP)."""

    AI_LOOP = "ai_loop"
    """An AI generated seamless loop backdrop."""

    @classmethod
    def loop_modes(cls) -> frozenset[BackgroundMode]:
        """The modes whose seam is verified before use."""
        return frozenset({cls.ANIMATED, cls.AI_LOOP})


_ENTRY_KEYS = frozenset(
    {
        "id",
        "key",
        "slug",
        "title",
        "name",
        "label",
        "path",
        "file",
        "source",
        "url",
        "duration_seconds",
        "duration",
        "length_seconds",
        "mode",
        "tags",
        "labels",
        "keywords",
        "license",
        "licence",
        "rights",
        "seam_similarity",
        "seam_verified",
        "format",
        "metadata",
    }
)
_RESERVED_KEYS = frozenset(
    {
        "schema_version",
        "version",
        "loop_similarity_threshold",
        "root",
        "backgrounds",
        "entries",
        "library",
    }
)


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{label} must be non-empty")
    return text


def _optional_text(value: Any, *, label: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value.strip()


def _normalise_tags(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items: Iterable[Any] = value.split(",")
    elif isinstance(value, (Mapping, bytes, bytearray)):
        raise ValueError("tags must be a string or a sequence of strings")
    elif isinstance(value, Iterable):
        items = value
    else:
        raise ValueError("tags must be a string or a sequence of strings")
    tags: list[str] = []
    for item in items:
        text = _text(item, label="tag").casefold()
        tags.append(text)
    return tuple(sorted(dict.fromkeys(tags)))


def _license_key(value: AssetLicense) -> str:
    name = value.name.strip().casefold()
    return name or value.url.strip().casefold()


def _license_keys(
    value: str | AssetLicense | Iterable[str | AssetLicense] | None,
) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, AssetLicense):
        keys = [_license_key(value)]
    elif isinstance(value, str):
        keys = [part.strip().casefold() for part in value.split(",")]
    elif isinstance(value, (Mapping, bytes, bytearray)):
        raise TypeError("license must be a string, a license or an iterable")
    elif isinstance(value, Iterable):
        keys = []
        for item in value:
            if isinstance(item, AssetLicense):
                keys.append(_license_key(item))
            elif isinstance(item, str):
                keys.append(item.strip().casefold())
            else:
                raise TypeError("license entries must be strings or licenses")
    else:
        raise TypeError("license must be a string, a license or an iterable")
    return frozenset(key for key in keys if key)


def _finite(
    value: Any,
    *,
    label: str,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if positive and number <= 0:
        raise ValueError(f"{label} must be positive")
    if not positive and number < 0:
        raise ValueError(f"{label} must be non-negative")
    return number


def _optional_label(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    text = value.strip().casefold()
    return text or None


def _coerce_mode(value: Any) -> BackgroundMode:
    if isinstance(value, BackgroundMode):
        return value
    if not isinstance(value, str):
        raise ValueError("mode must be a string")
    try:
        return BackgroundMode(value.strip().lower())
    except ValueError as exc:
        raise ValueError(
            "unsupported background mode "
            f"{value!r}; expected static, animated or ai_loop"
        ) from exc


class BackgroundLibraryEntry(BaseModel):
    """One local background and the metadata used to index and select it.

    ``path`` is a local file path or a ``file://`` URL. Tags and mode are
    case-normalised so a selection is not sensitive to spelling case.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        allow_inf_nan=False,
    )

    id: str = Field(validation_alias=AliasChoices("id", "key", "slug"))
    title: str = Field(validation_alias=AliasChoices("title", "name", "label"))
    path: str = Field(validation_alias=AliasChoices("path", "file", "source", "url"))
    mode: BackgroundMode = Field(validation_alias=AliasChoices("mode", "kind", "type"))
    duration_seconds: float | None = Field(
        default=None,
        validation_alias=AliasChoices("duration_seconds", "duration", "length_seconds"),
    )
    seam_similarity: float | None = Field(
        default=None,
        validation_alias=AliasChoices("seam_similarity", "seam"),
    )
    seam_verified: bool = Field(
        default=False,
        validation_alias=AliasChoices("seam_verified", "seamless", "verified"),
    )
    tags: tuple[str, ...] = Field(
        default_factory=tuple,
        validation_alias=AliasChoices("tags", "labels", "keywords"),
    )
    license: AssetLicense = Field(
        validation_alias=AliasChoices("license", "licence", "rights")
    )
    format: ImageFormat | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "title", "path", mode="before")
    @classmethod
    def _required_text(cls, value: Any, info: ValidationInfo) -> str:
        return _text(value, label=info.field_name or "value")

    @field_validator("mode", mode="before")
    @classmethod
    def _normalise_mode(cls, value: Any) -> BackgroundMode:
        return _coerce_mode(value)

    @field_validator("tags", mode="before")
    @classmethod
    def _normalise_tags(cls, value: Any) -> tuple[str, ...]:
        return _normalise_tags(value)

    @field_validator("license", mode="before")
    @classmethod
    def _coerce_license(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("license is required")
        if isinstance(value, str):
            return {"name": _text(value, label="license")}
        return value

    @field_validator("format", mode="before")
    @classmethod
    def _coerce_format(cls, value: Any) -> Any:
        if value is None or isinstance(value, ImageFormat):
            return value
        if not isinstance(value, str):
            raise ValueError("format must be an image format")
        text = value.strip().lower().lstrip(".")
        if not text:
            return None
        try:
            return ImageFormat(text)
        except ValueError as exc:
            raise ValueError(f"unsupported image format: {value}") from exc

    @field_validator("path")
    @classmethod
    def _local_path(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme and parsed.scheme.lower() != "file":
            raise ValueError("background library path must be local")
        return value

    @field_validator("duration_seconds", mode="before")
    @classmethod
    def _coerce_duration(cls, value: Any) -> float | None:
        if value is None:
            return None
        return _finite(value, label="duration_seconds", positive=True)

    @field_validator("seam_similarity", mode="before")
    @classmethod
    def _coerce_seam_similarity(cls, value: Any) -> float | None:
        if value is None:
            return None
        return _finite(value, label="seam_similarity")

    @property
    def tag_set(self) -> frozenset[str]:
        """The tags as a set for fast intersection."""
        return frozenset(self.tags)

    @property
    def license_name(self) -> str:
        """The license name, falling back to its terms URL."""
        return self.license.name or self.license.url

    @property
    def license_key(self) -> str:
        """The normalized key used in the license index."""
        return _license_key(self.license)

    @property
    def is_commercial(self) -> bool:
        """Whether the license explicitly allows commercial use."""
        return self.license.is_commercial

    @property
    def is_cleared(self) -> bool:
        """Whether the license is known and allows commercial use."""
        return self.license.is_commercial and self.license.is_known

    @property
    def duration(self) -> float | None:
        """Alias for :attr:`duration_seconds`."""
        return self.duration_seconds

    @property
    def is_loop(self) -> bool:
        """Whether the entry is an animated or AI loop background."""
        return self.mode in BackgroundMode.loop_modes()

    @property
    def is_relative_path(self) -> bool:
        """Whether :attr:`path` is a relative local path."""
        parsed = urlsplit(self.path)
        if parsed.scheme.lower() == "file":
            return False
        return not Path(self.path).is_absolute()

    @property
    def image_format(self) -> ImageFormat | None:
        """The declared format or the one implied by the file extension."""
        if self.format is not None:
            return self.format
        suffix = Path(urlsplit(self.path).path).suffix.lower().lstrip(".")
        try:
            return ImageFormat(suffix)
        except ValueError:
            return None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for logs and manifests."""
        return {
            "id": self.id,
            "title": self.title,
            "path": self.path,
            "mode": self.mode.value,
            "duration_seconds": self.duration_seconds,
            "seam_similarity": self.seam_similarity,
            "seam_verified": self.seam_verified,
            "tags": list(self.tags),
            "license": self.license.model_dump(mode="json"),
            "format": None if self.format is None else self.format.value,
        }


@dataclass(frozen=True, slots=True)
class BackgroundLibraryMatch:
    """One entry that passed every filter, with its deterministic rank score."""

    entry: BackgroundLibraryEntry
    score: float = _ZERO
    matched_tags: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    duration_distance_seconds: float = 0.0
    seam_similarity: float | None = None
    seam_verified: bool = False

    @property
    def id(self) -> str:
        """The entry id."""
        return self.entry.id

    @property
    def title(self) -> str:
        """The entry title."""
        return self.entry.title

    @property
    def path(self) -> str:
        """The entry local path."""
        return self.entry.path

    @property
    def mode(self) -> BackgroundMode:
        """The entry mode."""
        return self.entry.mode

    @property
    def duration_seconds(self) -> float | None:
        """The entry duration."""
        return self.entry.duration_seconds

    @property
    def license(self) -> AssetLicense:
        """The entry license."""
        return self.entry.license

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for logs."""
        return {
            "id": self.entry.id,
            "score": self.score,
            "matched_tags": list(self.matched_tags),
            "reasons": list(self.reasons),
            "duration_distance_seconds": self.duration_distance_seconds,
            "seam_similarity": self.seam_similarity,
            "seam_verified": self.seam_verified,
            "entry": self.entry.to_dict(),
        }


class BackgroundLibraryError(ValidationError):
    """No local background matched the requested filters."""

    code = "background_library"


def _freeze_index(
    index: Mapping[Any, Iterable[BackgroundLibraryEntry]],
) -> dict[Any, tuple[BackgroundLibraryEntry, ...]]:
    return {
        key: tuple(sorted(values, key=lambda entry: entry.id))
        for key, values in sorted(index.items(), key=lambda item: str(item[0]))
    }


class BackgroundLibrary:
    """A deterministic local background library built from manifest entries.

    ``by_id``, ``by_mode`` and ``by_tag`` expose the indexes. Every index is
    sorted by entry id, so listing or selecting never depends on input order.
    """

    def __init__(
        self,
        entries: Iterable[BackgroundLibraryEntry | Mapping[str, Any]] = (),
        *,
        loop_similarity_threshold: float = DEFAULT_MEDIA_LOOP_SIMILARITY_THRESHOLD,
    ) -> None:
        level = _finite(
            loop_similarity_threshold,
            label="loop_similarity_threshold",
        )
        if level < 0.0 or level > 1.0:
            raise ValueError("loop_similarity_threshold must be between 0 and 1")
        self._loop_similarity_threshold = level
        self._seam_records: dict[str, float] = {}
        if isinstance(entries, BackgroundLibrary):
            entries = entries.entries
        by_id: dict[str, BackgroundLibraryEntry] = {}
        for value in entries:
            entry = (
                value
                if isinstance(value, BackgroundLibraryEntry)
                else BackgroundLibraryEntry.model_validate(value)
            )
            if entry.id in by_id:
                raise ConfigError(
                    f"duplicate background library entry id: {entry.id}",
                    details={"id": entry.id},
                )
            by_id[entry.id] = entry
        self._entries = tuple(sorted(by_id.values(), key=lambda entry: entry.id))
        self._build_indexes()

    def _build_indexes(self) -> None:
        modes: dict[BackgroundMode, list[BackgroundLibraryEntry]] = defaultdict(list)
        tags: dict[str, list[BackgroundLibraryEntry]] = defaultdict(list)
        for entry in self._entries:
            modes[entry.mode].append(entry)
            for tag in entry.tags:
                tags[tag].append(entry)
        self._by_mode = _freeze_index(modes)
        self._by_tag = _freeze_index(tags)

    @classmethod
    def from_entries(
        cls,
        entries: Iterable[BackgroundLibraryEntry | Mapping[str, Any]],
        *,
        loop_similarity_threshold: float = DEFAULT_MEDIA_LOOP_SIMILARITY_THRESHOLD,
    ) -> BackgroundLibrary:
        """Build a library from entry models or mappings."""
        if isinstance(entries, BackgroundLibrary):
            return entries
        return cls(entries, loop_similarity_threshold=loop_similarity_threshold)

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        root: str | Path | None = None,
        loop_similarity_threshold: float = DEFAULT_MEDIA_LOOP_SIMILARITY_THRESHOLD,
    ) -> BackgroundLibrary:
        """Build a library from a manifest mapping, list or existing library."""
        if isinstance(value, BackgroundLibrary):
            return value
        base = None if root is None else Path(root).expanduser().resolve()
        raw = _raw_entries(value)
        entries = [_coerce_entry(item, root=base) for item in raw]
        return cls(
            entries,
            loop_similarity_threshold=loop_similarity_threshold,
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        root: str | Path | None = None,
        loop_similarity_threshold: float = DEFAULT_MEDIA_LOOP_SIMILARITY_THRESHOLD,
    ) -> BackgroundLibrary:
        """Load a YAML or JSON manifest and resolve its relative background paths."""
        manifest = Path(path).expanduser()
        if not manifest.is_file():
            raise ConfigError(
                f"background library manifest not found: {manifest}",
                details={"path": str(manifest)},
            )
        text = manifest.read_text(encoding="utf-8")
        suffix = manifest.suffix.lower()
        try:
            if suffix in {".yaml", ".yml"}:
                payload: object = yaml.safe_load(text)
            elif suffix == ".json":
                payload = json.loads(text)
            else:
                raise ConfigError(
                    f"unsupported background library manifest extension: {suffix}",
                    details={"path": str(manifest)},
                )
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"invalid background library manifest: {manifest}",
                details={"path": str(manifest), "error": str(exc)},
            ) from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"invalid background library manifest: {manifest}",
                details={"path": str(manifest), "error": str(exc)},
            ) from exc
        if payload is None:
            payload = []
        base = manifest.parent if root is None else Path(root).expanduser().resolve()
        return cls.from_mapping(
            payload,
            root=base,
            loop_similarity_threshold=loop_similarity_threshold,
        )

    @property
    def entries(self) -> tuple[BackgroundLibraryEntry, ...]:
        """Every entry, sorted by id."""
        return self._entries

    @property
    def ids(self) -> tuple[str, ...]:
        """Every entry id, sorted."""
        return tuple(entry.id for entry in self._entries)

    @property
    def modes(self) -> tuple[BackgroundMode, ...]:
        """Every indexed mode, sorted by value."""
        return tuple(sorted(self._by_mode, key=lambda mode: mode.value))

    @property
    def tags(self) -> tuple[str, ...]:
        """Every indexed tag, sorted."""
        return tuple(sorted(self._by_tag))

    @property
    def loop_similarity_threshold(self) -> float:
        """The seam similarity required for a loop to be considered verified."""
        return self._loop_similarity_threshold

    @property
    def by_mode(self) -> dict[BackgroundMode, tuple[BackgroundLibraryEntry, ...]]:
        """Mode to entries."""
        return dict(self._by_mode)

    @property
    def by_tag(self) -> dict[str, tuple[BackgroundLibraryEntry, ...]]:
        """Tag to entries."""
        return dict(self._by_tag)

    def get(self, entry_id: str) -> BackgroundLibraryEntry | None:
        """Return the entry with *entry_id*, or ``None``."""
        if not isinstance(entry_id, str):
            raise TypeError("entry_id must be a string")
        for entry in self._entries:
            if entry.id == entry_id:
                return entry
        return None

    def find(
        self,
        tags: str | Iterable[str] | None = None,
        *,
        mode: BackgroundMode | str | None = None,
        duration_seconds: float | None = None,
        duration_tolerance_seconds: float = 0.0,
        min_duration_seconds: float | None = None,
        max_duration_seconds: float | None = None,
        license: str | AssetLicense | Iterable[str | AssetLicense] | None = None,
        commercial_only: bool = False,
        match_all_tags: bool = False,
        verified_only: bool = False,
        min_seam_similarity: float | None = None,
        limit: int | None = None,
    ) -> tuple[BackgroundLibraryMatch, ...]:
        """Return every matching entry ranked from best to worst.

        ``tags`` matches any requested tag by default; pass
        ``match_all_tags=True`` to require all of them. ``mode`` restricts
        results to static, animated or ai_loop backgrounds. A requested
        duration is matched within its tolerance, while explicit minimum and
        maximum bounds are always applied. ``verified_only`` keeps only loop
        backgrounds whose seam meets the library threshold.
        """
        requested_tags = frozenset(_normalise_tags(tags))
        mode_key = _coerce_mode(mode) if mode is not None else None
        license_keys = _license_keys(license)
        duration_target = (
            None
            if duration_seconds is None
            else _finite(duration_seconds, label="duration_seconds", positive=True)
        )
        duration_tolerance = _finite(
            duration_tolerance_seconds,
            label="duration_tolerance_seconds",
        )
        min_duration = (
            None
            if min_duration_seconds is None
            else _finite(min_duration_seconds, label="min_duration_seconds")
        )
        max_duration = (
            None
            if max_duration_seconds is None
            else _finite(max_duration_seconds, label="max_duration_seconds")
        )
        min_seam = (
            None
            if min_seam_similarity is None
            else _finite(min_seam_similarity, label="min_seam_similarity")
        )
        if (
            min_duration is not None
            and max_duration is not None
            and min_duration > max_duration
        ):
            raise ValueError(
                "min_duration_seconds must not exceed max_duration_seconds"
            )
        if min_seam is not None and not 0.0 <= min_seam <= 1.0:
            raise ValueError("min_seam_similarity must be between 0 and 1")
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int):
                raise TypeError("limit must be an int")
            if limit < _MIN_ENTRY_COUNT:
                raise ValueError("limit must be positive")

        matches: list[BackgroundLibraryMatch] = []
        for entry in self._entries:
            matched_tags = requested_tags & entry.tag_set
            if requested_tags:
                if match_all_tags and matched_tags != requested_tags:
                    continue
                if not match_all_tags and not matched_tags:
                    continue
            if mode_key is not None and entry.mode != mode_key:
                continue
            if license_keys and entry.license_key not in license_keys:
                continue
            if commercial_only and not entry.is_cleared:
                continue
            if duration_target is not None and entry.duration_seconds is None:
                continue
            if entry.duration_seconds is not None:
                if duration_target is not None:
                    gap = abs(entry.duration_seconds - duration_target)
                    if gap > duration_tolerance:
                        continue
                if min_duration is not None and entry.duration_seconds < min_duration:
                    continue
                if max_duration is not None and entry.duration_seconds > max_duration:
                    continue
            seam_similarity = entry.seam_similarity
            seam_verified = entry.seam_verified
            measured_similarity = self._seam_records.get(entry.id)
            if measured_similarity is not None:
                seam_similarity = measured_similarity
            if verified_only or min_seam is not None:
                if not entry.is_loop:
                    continue
                effective_verified = seam_verified or (
                    seam_similarity is not None
                    and seam_similarity >= self._loop_similarity_threshold
                )
                if verified_only and not effective_verified:
                    continue
                if min_seam is not None and (
                    seam_similarity is None or seam_similarity < min_seam
                ):
                    continue

            rank = 0.0
            reasons: list[str] = []
            if mode_key is not None:
                rank += 1.0
                reasons.append(f"mode:{mode_key.value}")
            for tag in sorted(matched_tags):
                rank += 0.5
                reasons.append(f"tag:{tag}")
            if duration_target is not None and entry.duration_seconds is not None:
                gap = abs(entry.duration_seconds - duration_target)
                closeness = (
                    1.0 if not duration_tolerance else 1.0 - gap / duration_tolerance
                )
                rank += max(0.0, closeness)
                reasons.append("duration")
            if license_keys:
                rank += 0.5
                reasons.append("license")
            if entry.is_cleared:
                rank += 0.25
                reasons.append("commercial")
            matches.append(
                BackgroundLibraryMatch(
                    entry=entry,
                    score=rank,
                    matched_tags=tuple(sorted(matched_tags)),
                    reasons=tuple(reasons),
                    duration_distance_seconds=(
                        abs(entry.duration_seconds - duration_target)
                        if (
                            duration_target is not None
                            and entry.duration_seconds is not None
                        )
                        else 0.0
                    ),
                    seam_similarity=seam_similarity,
                    seam_verified=seam_verified,
                )
            )

        matches.sort(
            key=lambda match: (
                -match.score,
                match.duration_distance_seconds,
                match.entry.id,
            )
        )
        selected = matches if limit is None else matches[:limit]
        return tuple(selected)

    def select(
        self,
        tags: str | Iterable[str] | None = None,
        **filters: Any,
    ) -> BackgroundLibraryEntry:
        """Return the best matching entry or raise :class:`BackgroundLibraryError`."""
        matches = self.find(tags, **filters)
        if not matches:
            raise BackgroundLibraryError(
                "no local background matched the requested filters",
                details={
                    "tags": list(_normalise_tags(tags)),
                    "count": len(self._entries),
                    "modes": [mode.value for mode in self.modes],
                    "tags_index": list(self.tags),
                },
            )
        return matches[0].entry

    def select_match(
        self,
        tags: str | Iterable[str] | None = None,
        **filters: Any,
    ) -> BackgroundLibraryMatch:
        """Return the best :class:`BackgroundLibraryMatch` or raise."""
        matches = self.find(tags, **filters)
        if not matches:
            raise BackgroundLibraryError(
                "no local background matched the requested filters",
                details={
                    "tags": list(_normalise_tags(tags)),
                    "count": len(self._entries),
                },
            )
        return matches[0]

    def select_many(
        self,
        tags: str | Iterable[str] | None = None,
        **filters: Any,
    ) -> tuple[BackgroundLibraryEntry, ...]:
        """Return every matching entry in rank order."""
        return tuple(match.entry for match in self.find(tags, **filters))

    def for_duration(
        self,
        duration_seconds: float,
        tags: str | Iterable[str] | None = None,
        **filters: Any,
    ) -> BackgroundLibraryEntry:
        """Select a background that fits *duration_seconds*."""
        filters["duration_seconds"] = duration_seconds
        return self.select(tags, **filters)

    def verify_seam(self, entry_id: str) -> BackgroundLibraryMatch:
        """Measure the seam of a loop background and record the result.

        Static backgrounds have no loop seam and are trivially verified.
        Animated and ai_loop backgrounds are measured against the library
        threshold using :func:`measure_loop_seam`.
        """
        entry = self.get(entry_id)
        if entry is None:
            raise BackgroundLibraryError(
                f"no background library entry with id {entry_id!r}",
                details={"id": entry_id},
            )
        if entry.mode is BackgroundMode.STATIC:
            return BackgroundLibraryMatch(
                entry=entry,
                score=_ZERO,
                seam_similarity=None,
                seam_verified=True,
            )
        from aivia.compose.loop_outputs import measure_loop_seam

        try:
            similarity = measure_loop_seam(entry.path)
        except Exception as exc:
            raise BackgroundLibraryError(
                f"could not measure the seam of background {entry_id!r}",
                details={"id": entry_id, "error": str(exc)},
            ) from exc
        verified = similarity >= self._loop_similarity_threshold
        self._seam_records[entry_id] = similarity
        return BackgroundLibraryMatch(
            entry=entry,
            score=similarity,
            seam_similarity=similarity,
            seam_verified=verified,
        )

    matches = find
    choose = select

    def __contains__(self, value: object) -> bool:
        if isinstance(value, BackgroundLibraryEntry):
            return value.id in self.ids
        return isinstance(value, str) and any(
            entry.id == value for entry in self._entries
        )

    def __iter__(self) -> Iterator[BackgroundLibraryEntry]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"BackgroundLibrary(entries={len(self._entries)})"


def _as_entry_list(value: object) -> list[object]:
    if isinstance(value, BackgroundLibrary):
        return list(value.entries)
    if isinstance(value, Mapping):
        return _mapping_entries(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    raise ConfigError("background library entries must be a mapping or a sequence")


def _mapping_entries(value: Mapping[Any, Any]) -> list[object]:
    if any(key in value for key in ("path", "file", "source", "url")) and any(
        key in value for key in ("title", "name", "label")
    ):
        return [value]
    entries: list[object] = []
    for key, item in value.items():
        if not isinstance(key, str) or key in _RESERVED_KEYS:
            continue
        if isinstance(item, BackgroundLibraryEntry):
            entries.append(item)
        elif isinstance(item, Mapping):
            data = dict(item)
            data.setdefault("id", key)
            entries.append(data)
        else:
            raise ConfigError(
                f"background library entry {key!r} must be a mapping",
                details={"id": key},
            )
    return entries


def _raw_entries(value: object) -> list[object]:
    if isinstance(value, BackgroundLibrary):
        return list(value.entries)
    if isinstance(value, Mapping):
        for key in ("backgrounds", "entries", "library"):
            if key in value:
                return _as_entry_list(value[key])
        if any(key in value for key in _ENTRY_KEYS):
            return [value]
        return _mapping_entries(value)
    return _as_entry_list(value)


def _coerce_entry(
    value: object,
    *,
    root: Path | None,
) -> BackgroundLibraryEntry:
    if isinstance(value, BackgroundLibraryEntry):
        entry = value
    elif isinstance(value, Mapping):
        entry = BackgroundLibraryEntry.model_validate(dict(value))
    else:
        raise ConfigError("background library entry must be a mapping or an entry")
    if root is not None and entry.is_relative_path:
        entry = entry.model_copy(update={"path": str((root / entry.path).resolve())})
    return entry


def load_background_library(
    source: str | Path | Mapping[str, Any] | Iterable[Any] | BackgroundLibrary,
    *,
    root: str | Path | None = None,
    loop_similarity_threshold: float = DEFAULT_MEDIA_LOOP_SIMILARITY_THRESHOLD,
) -> BackgroundLibrary:
    """Load a library from a manifest path, a mapping or an existing library."""
    if isinstance(source, BackgroundLibrary):
        return source
    if isinstance(source, (str, Path)):
        return BackgroundLibrary.from_file(
            source,
            root=root,
            loop_similarity_threshold=loop_similarity_threshold,
        )
    return BackgroundLibrary.from_mapping(
        source,
        root=root,
        loop_similarity_threshold=loop_similarity_threshold,
    )


background_library = load_background_library
build_background_library = load_background_library


def select_background(
    library: str | Path | Mapping[str, Any] | Iterable[Any] | BackgroundLibrary,
    tags: str | Iterable[str] | None = None,
    **filters: Any,
) -> BackgroundLibraryEntry:
    """Load *library* and select the best matching local background."""
    return load_background_library(library).select(tags, **filters)

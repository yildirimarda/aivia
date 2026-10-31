"""A local royalty free music library indexed by mood, tempo, genre and license.

A run often needs a background track that is already on disk, not another paid
generation. The library turns a manifest of local tracks into deterministic
indexes, so a later stage can ask for a mood, a genre, a tempo and a duration
without scanning every file or calling a provider.

Selection is pure code. Tags, mood, genre, license, tempo and duration are
filtered first and the surviving entries are ranked by how well they match the
request. The same library and the same filters always return the same track,
which keeps a composition reproducible offline. A manifest may be YAML or JSON;
relative track paths are resolved against the manifest directory.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
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

from aivia.core.config.media import (
    DEFAULT_MUSIC_DURATION_TOLERANCE_SECONDS,
    MIN_MUSIC_LIBRARY_LIMIT,
    MUSIC_LIBRARY_DURATION_WEIGHT,
    MUSIC_LIBRARY_GENRE_WEIGHT,
    MUSIC_LIBRARY_LICENSE_WEIGHT,
    MUSIC_LIBRARY_MOOD_WEIGHT,
    MUSIC_LIBRARY_TAG_WEIGHT,
    MUSIC_LIBRARY_TEMPO_WEIGHT,
)
from aivia.core.config.media import (
    DEFAULT_MUSIC_LIBRARY_TEMPO_BUCKET_BPM as DEFAULT_TEMPO_BUCKET_BPM,
)
from aivia.core.config.media import (
    DEFAULT_MUSIC_LIBRARY_TEMPO_TOLERANCE_BPM as DEFAULT_TEMPO_TOLERANCE_BPM,
)
from aivia.core.errors import ConfigError, ValidationError
from aivia.core.models.asset import AssetLicense
from aivia.providers.protocols.audio import AudioFormat

__all__ = [
    "DEFAULT_TEMPO_BUCKET_BPM",
    "DEFAULT_TEMPO_TOLERANCE_BPM",
    "LIBRARY_SCHEMA_VERSION",
    "MusicLibrary",
    "MusicLibraryEntry",
    "MusicLibraryError",
    "MusicLibraryMatch",
    "MusicLibraryTrack",
    "MusicSelection",
    "build_music_library",
    "load_music_library",
    "music_library",
    "select_music",
]

LIBRARY_SCHEMA_VERSION = 1
"""The manifest schema version understood by this release."""

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
        "mood",
        "feel",
        "genre",
        "style",
        "tempo_bpm",
        "tempo",
        "bpm",
        "license",
        "licence",
        "rights",
        "tags",
        "labels",
        "keywords",
        "artist",
        "author",
        "composer",
        "format",
        "metadata",
    }
)
_RESERVED_KEYS = frozenset(
    {
        "schema_version",
        "version",
        "tempo_bucket_bpm",
        "root",
        "tracks",
        "entries",
        "music",
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


class MusicLibraryEntry(BaseModel):
    """One local track and the metadata used to index and select it.

    ``path`` is a local file path or a ``file://`` URL. ``license`` accepts the
    same shapes as :class:`~aivia.core.models.asset.AssetLicense`, including a
    plain license name. Mood, genre and tags are case-folded so a selection is
    not sensitive to spelling case.
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
    duration_seconds: float = Field(
        gt=0.0,
        validation_alias=AliasChoices(
            "duration_seconds",
            "duration",
            "length_seconds",
        ),
    )
    mood: str = Field(validation_alias=AliasChoices("mood", "feel"))
    genre: str = Field(validation_alias=AliasChoices("genre", "style"))
    tempo_bpm: float = Field(
        gt=0.0,
        validation_alias=AliasChoices("tempo_bpm", "tempo", "bpm"),
    )
    license: AssetLicense = Field(
        validation_alias=AliasChoices("license", "licence", "rights")
    )
    tags: tuple[str, ...] = Field(
        default_factory=tuple,
        validation_alias=AliasChoices("tags", "labels", "keywords"),
    )
    artist: str = Field(
        default="",
        validation_alias=AliasChoices("artist", "author", "composer"),
    )
    format: AudioFormat | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "title", "path", mode="before")
    @classmethod
    def _required_text(cls, value: Any, info: ValidationInfo) -> str:
        return _text(value, label=info.field_name or "value")

    @field_validator("mood", "genre", mode="before")
    @classmethod
    def _normalise_label(cls, value: Any, info: ValidationInfo) -> str:
        return _text(value, label=info.field_name or "value").casefold()

    @field_validator("tags", mode="before")
    @classmethod
    def _normalise_tags(cls, value: Any) -> tuple[str, ...]:
        return _normalise_tags(value)

    @field_validator("artist", mode="before")
    @classmethod
    def _normalise_artist(cls, value: Any) -> str:
        return _optional_text(value, label="artist")

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
        if value is None or isinstance(value, AudioFormat):
            return value
        if not isinstance(value, str):
            raise ValueError("format must be an audio format")
        text = value.strip().lower().lstrip(".")
        if not text:
            return None
        try:
            return AudioFormat(text)
        except ValueError as exc:
            raise ValueError(f"unsupported audio format: {value}") from exc

    @field_validator("path")
    @classmethod
    def _local_path(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme and parsed.scheme.lower() != "file":
            raise ValueError("music library path must be local")
        return value

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
    def duration(self) -> float:
        """Alias for :attr:`duration_seconds`."""
        return self.duration_seconds

    @property
    def tempo(self) -> float:
        """Alias for :attr:`tempo_bpm`."""
        return self.tempo_bpm

    @property
    def is_relative_path(self) -> bool:
        """Whether :attr:`path` is a relative local path."""
        parsed = urlsplit(self.path)
        if parsed.scheme.lower() == "file":
            return False
        return not Path(self.path).is_absolute()

    @property
    def audio_format(self) -> AudioFormat | None:
        """The declared format or the one implied by the file extension."""
        if self.format is not None:
            return self.format
        suffix = Path(urlsplit(self.path).path).suffix.lower().lstrip(".")
        try:
            return AudioFormat(suffix)
        except ValueError:
            return None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for logs and manifests."""
        return {
            "id": self.id,
            "title": self.title,
            "path": self.path,
            "duration_seconds": self.duration_seconds,
            "mood": self.mood,
            "genre": self.genre,
            "tempo_bpm": self.tempo_bpm,
            "tags": list(self.tags),
            "artist": self.artist,
            "format": None if self.format is None else self.format.value,
            "license": self.license.model_dump(mode="json"),
        }


@dataclass(frozen=True, slots=True)
class MusicLibraryMatch:
    """One entry that passed every filter, with its deterministic rank score."""

    entry: MusicLibraryEntry
    score: float = 0.0
    matched_tags: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    duration_distance_seconds: float = 0.0
    tempo_distance_bpm: float = 0.0

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
    def duration_seconds(self) -> float:
        """The entry duration."""
        return self.entry.duration_seconds

    @property
    def tempo_bpm(self) -> float:
        """The entry tempo."""
        return self.entry.tempo_bpm

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
            "tempo_distance_bpm": self.tempo_distance_bpm,
            "entry": self.entry.to_dict(),
        }


class MusicLibraryError(ValidationError):
    """No local track matched the requested filters."""

    code = "music_library"


def _freeze_index(
    index: Mapping[Any, Iterable[MusicLibraryEntry]],
) -> dict[Any, tuple[MusicLibraryEntry, ...]]:
    return {
        key: tuple(sorted(values, key=lambda entry: entry.id))
        for key, values in sorted(index.items(), key=lambda item: str(item[0]))
    }


class MusicLibrary:
    """A deterministic local music library built from manifest entries.

    ``by_mood``, ``by_genre``, ``by_license``, ``by_tempo``, ``by_tempo_bucket``
    and ``by_tag`` expose the indexes. Every index is sorted by entry id, so
    listing or selecting never depends on input order.
    """

    def __init__(
        self,
        entries: Iterable[MusicLibraryEntry | Mapping[str, Any]] = (),
        *,
        tempo_bucket_bpm: float = DEFAULT_TEMPO_BUCKET_BPM,
    ) -> None:
        width = _finite(tempo_bucket_bpm, label="tempo_bucket_bpm", positive=True)
        if isinstance(entries, MusicLibrary):
            entries = entries.entries
        self._tempo_bucket_bpm = width
        by_id: dict[str, MusicLibraryEntry] = {}
        for value in entries:
            entry = (
                value
                if isinstance(value, MusicLibraryEntry)
                else MusicLibraryEntry.model_validate(value)
            )
            if entry.id in by_id:
                raise ConfigError(
                    f"duplicate music library entry id: {entry.id}",
                    details={"id": entry.id},
                )
            by_id[entry.id] = entry
        self._entries = tuple(sorted(by_id.values(), key=lambda entry: entry.id))
        self._build_indexes()

    def _build_indexes(self) -> None:
        moods: dict[str, list[MusicLibraryEntry]] = defaultdict(list)
        genres: dict[str, list[MusicLibraryEntry]] = defaultdict(list)
        licenses: dict[str, list[MusicLibraryEntry]] = defaultdict(list)
        tempos: dict[int, list[MusicLibraryEntry]] = defaultdict(list)
        buckets: dict[float, list[MusicLibraryEntry]] = defaultdict(list)
        tags: dict[str, list[MusicLibraryEntry]] = defaultdict(list)
        for entry in self._entries:
            moods[entry.mood].append(entry)
            genres[entry.genre].append(entry)
            licenses[entry.license_key].append(entry)
            tempos[round(entry.tempo_bpm)].append(entry)
            bucket = math.floor(entry.tempo_bpm / self._tempo_bucket_bpm)
            buckets[round(bucket * self._tempo_bucket_bpm, 6)].append(entry)
            for tag in entry.tags:
                tags[tag].append(entry)
        self._by_mood = _freeze_index(moods)
        self._by_genre = _freeze_index(genres)
        self._by_license = _freeze_index(licenses)
        self._by_tempo = _freeze_index(tempos)
        self._by_tempo_bucket = _freeze_index(buckets)
        self._by_tag = _freeze_index(tags)

    @classmethod
    def from_entries(
        cls,
        entries: Iterable[MusicLibraryEntry | Mapping[str, Any]],
        *,
        tempo_bucket_bpm: float = DEFAULT_TEMPO_BUCKET_BPM,
    ) -> MusicLibrary:
        """Build a library from entry models or mappings."""
        if isinstance(entries, MusicLibrary):
            return entries
        return cls(entries, tempo_bucket_bpm=tempo_bucket_bpm)

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        root: str | Path | None = None,
        tempo_bucket_bpm: float = DEFAULT_TEMPO_BUCKET_BPM,
    ) -> MusicLibrary:
        """Build a library from a manifest mapping, list or existing library."""
        if isinstance(value, MusicLibrary):
            return value
        base = None if root is None else Path(root).expanduser().resolve()
        raw = _raw_entries(value)
        entries = [_coerce_entry(item, root=base) for item in raw]
        return cls(entries, tempo_bucket_bpm=tempo_bucket_bpm)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        root: str | Path | None = None,
        tempo_bucket_bpm: float = DEFAULT_TEMPO_BUCKET_BPM,
    ) -> MusicLibrary:
        """Load a YAML or JSON manifest and resolve its relative track paths."""
        manifest = Path(path).expanduser()
        if not manifest.is_file():
            raise ConfigError(
                f"music library manifest not found: {manifest}",
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
                    f"unsupported music library manifest extension: {suffix}",
                    details={"path": str(manifest)},
                )
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"invalid music library manifest: {manifest}",
                details={"path": str(manifest), "error": str(exc)},
            ) from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"invalid music library manifest: {manifest}",
                details={"path": str(manifest), "error": str(exc)},
            ) from exc
        if payload is None:
            payload = []
        base = manifest.parent if root is None else Path(root).expanduser().resolve()
        return cls.from_mapping(
            payload,
            root=base,
            tempo_bucket_bpm=tempo_bucket_bpm,
        )

    @property
    def entries(self) -> tuple[MusicLibraryEntry, ...]:
        """Every entry, sorted by id."""
        return self._entries

    @property
    def ids(self) -> tuple[str, ...]:
        """Every entry id, sorted."""
        return tuple(entry.id for entry in self._entries)

    @property
    def moods(self) -> tuple[str, ...]:
        """Every indexed mood, sorted."""
        return tuple(sorted(self._by_mood))

    @property
    def genres(self) -> tuple[str, ...]:
        """Every indexed genre, sorted."""
        return tuple(sorted(self._by_genre))

    @property
    def licenses(self) -> tuple[str, ...]:
        """Every indexed license key, sorted."""
        return tuple(sorted(self._by_license))

    @property
    def tags(self) -> tuple[str, ...]:
        """Every indexed tag, sorted."""
        return tuple(sorted(self._by_tag))

    @property
    def tempos(self) -> tuple[float, ...]:
        """Every entry tempo in ascending order."""
        return tuple(sorted(entry.tempo_bpm for entry in self._entries))

    @property
    def by_mood(self) -> dict[str, tuple[MusicLibraryEntry, ...]]:
        """Mood to entries."""
        return dict(self._by_mood)

    @property
    def by_genre(self) -> dict[str, tuple[MusicLibraryEntry, ...]]:
        """Genre to entries."""
        return dict(self._by_genre)

    @property
    def by_license(self) -> dict[str, tuple[MusicLibraryEntry, ...]]:
        """Normalized license key to entries."""
        return dict(self._by_license)

    @property
    def by_tempo(self) -> dict[int, tuple[MusicLibraryEntry, ...]]:
        """Rounded BPM to entries."""
        return dict(self._by_tempo)

    @property
    def by_tempo_bucket(self) -> dict[float, tuple[MusicLibraryEntry, ...]]:
        """Tempo bucket start to entries."""
        return dict(self._by_tempo_bucket)

    @property
    def by_tag(self) -> dict[str, tuple[MusicLibraryEntry, ...]]:
        """Tag to entries."""
        return dict(self._by_tag)

    @property
    def tempo_bucket_bpm(self) -> float:
        """The width of one tempo bucket."""
        return self._tempo_bucket_bpm

    def get(self, entry_id: str) -> MusicLibraryEntry | None:
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
        mood: str | None = None,
        genre: str | None = None,
        tempo_bpm: float | None = None,
        tempo_tolerance_bpm: float = DEFAULT_TEMPO_TOLERANCE_BPM,
        min_tempo_bpm: float | None = None,
        max_tempo_bpm: float | None = None,
        duration_seconds: float | None = None,
        duration_tolerance_seconds: float = DEFAULT_MUSIC_DURATION_TOLERANCE_SECONDS,
        min_duration_seconds: float | None = None,
        max_duration_seconds: float | None = None,
        license: str | AssetLicense | Iterable[str | AssetLicense] | None = None,
        commercial_only: bool = False,
        match_all_tags: bool = False,
        limit: int | None = None,
    ) -> tuple[MusicLibraryMatch, ...]:
        """Return every matching entry ranked from best to worst.

        ``tags`` matches any requested tag by default; pass
        ``match_all_tags=True`` to require all of them. A requested duration
        and tempo are matched within their tolerance, while explicit minimum
        and maximum bounds are always applied. ``commercial_only`` keeps only
        entries whose license is known and explicitly allows commercial use.
        """
        requested_tags = frozenset(_normalise_tags(tags))
        mood_key = _optional_label(mood, label="mood")
        genre_key = _optional_label(genre, label="genre")
        license_keys = _license_keys(license)
        tempo_target = (
            None
            if tempo_bpm is None
            else _finite(tempo_bpm, label="tempo_bpm", positive=True)
        )
        tempo_tolerance = _finite(
            tempo_tolerance_bpm,
            label="tempo_tolerance_bpm",
        )
        min_tempo = (
            None
            if min_tempo_bpm is None
            else _finite(min_tempo_bpm, label="min_tempo_bpm", positive=True)
        )
        max_tempo = (
            None
            if max_tempo_bpm is None
            else _finite(max_tempo_bpm, label="max_tempo_bpm", positive=True)
        )
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
        if min_tempo is not None and max_tempo is not None and min_tempo > max_tempo:
            raise ValueError("min_tempo_bpm must not exceed max_tempo_bpm")
        if (
            min_duration is not None
            and max_duration is not None
            and min_duration > max_duration
        ):
            raise ValueError(
                "min_duration_seconds must not exceed max_duration_seconds"
            )
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int):
                raise TypeError("limit must be an int")
            if limit < MIN_MUSIC_LIBRARY_LIMIT:
                raise ValueError("limit must be positive")

        matches: list[MusicLibraryMatch] = []
        for entry in self._entries:
            matched_tags = requested_tags & entry.tag_set
            if requested_tags:
                if match_all_tags and matched_tags != requested_tags:
                    continue
                if not match_all_tags and not matched_tags:
                    continue
            if mood_key is not None and entry.mood != mood_key:
                continue
            if genre_key is not None and entry.genre != genre_key:
                continue
            if license_keys and entry.license_key not in license_keys:
                continue
            if commercial_only and not entry.is_cleared:
                continue
            tempo_distance = 0.0
            if tempo_target is not None:
                tempo_distance = abs(entry.tempo_bpm - tempo_target)
                if tempo_distance > tempo_tolerance:
                    continue
            if min_tempo is not None and entry.tempo_bpm < min_tempo:
                continue
            if max_tempo is not None and entry.tempo_bpm > max_tempo:
                continue
            gap = 0.0
            if duration_target is not None:
                gap = abs(entry.duration_seconds - duration_target)
                if gap > duration_tolerance:
                    continue
            if min_duration is not None and entry.duration_seconds < min_duration:
                continue
            if max_duration is not None and entry.duration_seconds > max_duration:
                continue

            rank = 0.0
            reasons: list[str] = []
            if mood_key is not None:
                rank += MUSIC_LIBRARY_MOOD_WEIGHT
                reasons.append("mood")
            if genre_key is not None:
                rank += MUSIC_LIBRARY_GENRE_WEIGHT
                reasons.append("genre")
            for tag in sorted(matched_tags):
                rank += MUSIC_LIBRARY_TAG_WEIGHT
                reasons.append(f"tag:{tag}")
            if tempo_target is not None:
                closeness = (
                    1.0
                    if not tempo_tolerance
                    else 1.0 - tempo_distance / tempo_tolerance
                )
                rank += MUSIC_LIBRARY_TEMPO_WEIGHT * max(0.0, closeness)
                reasons.append("tempo")
            if duration_target is not None:
                closeness = (
                    1.0 if not duration_tolerance else 1.0 - gap / duration_tolerance
                )
                rank += MUSIC_LIBRARY_DURATION_WEIGHT * max(0.0, closeness)
                reasons.append("duration")
            if license_keys:
                rank += MUSIC_LIBRARY_LICENSE_WEIGHT
                reasons.append("license")
            matches.append(
                MusicLibraryMatch(
                    entry=entry,
                    score=rank,
                    matched_tags=tuple(sorted(matched_tags)),
                    reasons=tuple(reasons),
                    duration_distance_seconds=gap,
                    tempo_distance_bpm=tempo_distance,
                )
            )

        matches.sort(
            key=lambda match: (
                -match.score,
                match.duration_distance_seconds,
                match.tempo_distance_bpm,
                match.entry.id,
            )
        )
        selected = matches if limit is None else matches[:limit]
        return tuple(selected)

    def select(
        self,
        tags: str | Iterable[str] | None = None,
        **filters: Any,
    ) -> MusicLibraryEntry:
        """Return the best matching entry or raise :class:`MusicLibraryError`."""
        matches = self.find(tags, **filters)
        if not matches:
            raise MusicLibraryError(
                "no local music track matched the requested filters",
                details={
                    "tags": list(_normalise_tags(tags)),
                    "count": len(self._entries),
                    "moods": list(self.moods),
                    "genres": list(self.genres),
                    "licenses": list(self.licenses),
                    "tags_index": list(self.tags),
                },
            )
        return matches[0].entry

    def select_match(
        self,
        tags: str | Iterable[str] | None = None,
        **filters: Any,
    ) -> MusicLibraryMatch:
        """Return the best :class:`MusicLibraryMatch` or raise."""
        matches = self.find(tags, **filters)
        if not matches:
            raise MusicLibraryError(
                "no local music track matched the requested filters",
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
    ) -> tuple[MusicLibraryEntry, ...]:
        """Return every matching entry in rank order."""
        return tuple(match.entry for match in self.find(tags, **filters))

    def for_duration(
        self,
        duration_seconds: float,
        tags: str | Iterable[str] | None = None,
        **filters: Any,
    ) -> MusicLibraryEntry:
        """Select a track that fits *duration_seconds*."""
        filters["duration_seconds"] = duration_seconds
        return self.select(tags, **filters)

    matches = find
    choose = select

    def __contains__(self, value: object) -> bool:
        if isinstance(value, MusicLibraryEntry):
            return value.id in self.ids
        return isinstance(value, str) and any(
            entry.id == value for entry in self._entries
        )

    def __iter__(self) -> Iterator[MusicLibraryEntry]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"MusicLibrary(entries={len(self._entries)})"


MusicLibraryTrack = MusicLibraryEntry
"""Alias for one local track in the library."""

MusicSelection = MusicLibraryMatch
"""Alias for one ranked library selection."""


def _as_entry_list(value: object) -> list[object]:
    if isinstance(value, MusicLibrary):
        return list(value.entries)
    if isinstance(value, Mapping):
        return _mapping_entries(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    raise ConfigError("music library entries must be a mapping or a sequence")


def _mapping_entries(value: Mapping[Any, Any]) -> list[object]:
    if any(key in value for key in ("path", "file", "source", "url")) and any(
        key in value for key in ("title", "name", "label")
    ):
        return [value]
    entries: list[object] = []
    for key, item in value.items():
        if not isinstance(key, str) or key in _RESERVED_KEYS:
            continue
        if isinstance(item, MusicLibraryEntry):
            entries.append(item)
        elif isinstance(item, Mapping):
            data = dict(item)
            data.setdefault("id", key)
            entries.append(data)
        else:
            raise ConfigError(
                f"music library entry {key!r} must be a mapping",
                details={"id": key},
            )
    return entries


def _raw_entries(value: object) -> list[object]:
    if isinstance(value, MusicLibrary):
        return list(value.entries)
    if isinstance(value, Mapping):
        for key in ("tracks", "entries", "music"):
            if key in value:
                return _as_entry_list(value[key])
        if "library" in value:
            return _raw_entries(value["library"])
        if any(key in value for key in _ENTRY_KEYS):
            return [value]
        return _mapping_entries(value)
    return _as_entry_list(value)


def _coerce_entry(
    value: object,
    *,
    root: Path | None,
) -> MusicLibraryEntry:
    if isinstance(value, MusicLibraryEntry):
        entry = value
    elif isinstance(value, Mapping):
        entry = MusicLibraryEntry.model_validate(dict(value))
    else:
        raise ConfigError("music library entry must be a mapping or an entry")
    if root is not None and entry.is_relative_path:
        entry = entry.model_copy(update={"path": str((root / entry.path).resolve())})
    return entry


def load_music_library(
    source: str | Path | Mapping[str, Any] | Iterable[Any] | MusicLibrary,
    *,
    root: str | Path | None = None,
    tempo_bucket_bpm: float = DEFAULT_TEMPO_BUCKET_BPM,
) -> MusicLibrary:
    """Load a library from a manifest path, a mapping or an existing library."""
    if isinstance(source, MusicLibrary):
        return source
    if isinstance(source, (str, Path)):
        return MusicLibrary.from_file(
            source,
            root=root,
            tempo_bucket_bpm=tempo_bucket_bpm,
        )
    return MusicLibrary.from_mapping(
        source,
        root=root,
        tempo_bucket_bpm=tempo_bucket_bpm,
    )


music_library = load_music_library
build_music_library = load_music_library


def select_music(
    library: str | Path | Mapping[str, Any] | Iterable[Any] | MusicLibrary,
    tags: str | Iterable[str] | None = None,
    **filters: Any,
) -> MusicLibraryEntry:
    """Load *library* and select the best matching local track."""
    return load_music_library(library).select(tags, **filters)

"""Choose a background music bed and mark it for voice ducking.

The stage asks for a mood, not a vendor. When a local royalty free library is
configured it selects the best ranked track by mood, genre, tags, tempo and
duration. When no local track matches, or when the caller explicitly asks for a
provider, the request goes through the single screened, cached and budgeted
music service.

Every choice keeps its license, attribution and provenance. The license is
written on the timeline music track and in a small ``music.json`` manifest, so
a later publish or delivery gate can decide whether the bed is cleared without
guessing. The chosen track is always marked as duckable; when the timeline
already carries voice timings the stage also verifies that the generated mix
graph applies sidechain compression with the configured ducking settings.

Selection is deterministic. The same library, mood and duration return the
same track, and a provider result is recorded exactly as it was returned.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel

from aivia.audio.song_candidates import (
    SongCandidateSelection,
    lyrics_text,
    select_song_candidate,
)
from aivia.compose.schema import AudioTrackRole as TimelineTrackRole
from aivia.compose.timeline import Timeline, TimelineBeat, TimelineMusic
from aivia.core.config.media import (
    DEFAULT_MUSIC_BACKGROUND_FADE_IN_SECONDS,
    DEFAULT_MUSIC_BACKGROUND_FADE_OUT_SECONDS,
    DEFAULT_MUSIC_BACKGROUND_VOLUME_DB,
    DEFAULT_MUSIC_MIN_DURATION_SECONDS,
)
from aivia.core.errors import ConfigError
from aivia.core.models.asset import AssetLicense
from aivia.core.models.base import canonical_json
from aivia.core.models.run import StageResult, StageState
from aivia.core.stages.base import Stage, StageContext
from aivia.media.audio_mix import (
    AudioMixSettings,
    AudioTrack,
    AudioTrackRole,
    DuckingSettings,
    DuckingVerification,
    build_mix_filtergraph,
    verify_ducking,
)
from aivia.media.music_library import (
    MusicLibrary,
    MusicLibraryEntry,
    MusicLibraryError,
    MusicLibraryMatch,
    load_music_library,
)
from aivia.providers.protocols.audio import AudioFormat
from aivia.providers.protocols.music import (
    CAPABILITY_MUSIC_GEN,
    MusicRequest,
    MusicResponse,
    MusicTrack,
)
from aivia.writing.length import estimate_duration
from aivia.writing.schema import Script

__all__ = [
    "DEFAULT_MUSIC_FILENAME",
    "DEFAULT_TIMELINE_FILENAME",
    "MUSIC_STAGE_NAME",
    "SOURCE_MODES",
    "BackgroundMusic",
    "MusicChoice",
    "MusicSelectStage",
    "MusicSelection",
    "MusicStage",
    "SelectedMusic",
    "apply_ducking",
    "build_music_prompt",
    "build_music_prompt_from_profile",
    "build_music_stage",
    "build_profile_music_prompt",
    "music_prompt",
    "music_prompt_for_profile",
    "music_prompt_from_profile",
    "music_stage",
    "profile_music_prompt",
    "run_music_stage",
    "select_library_music",
    "select_music",
    "verify_timeline_ducking",
]

MUSIC_STAGE_NAME = "music"
"""The recipe stage name this implementation serves."""

DEFAULT_TIMELINE_FILENAME = "timeline.json"
"""The timeline file the voice stage writes and this stage updates."""

DEFAULT_MUSIC_FILENAME = "music.json"
"""The manifest that records the chosen track, its license and ducking."""

SOURCE_MODES = ("auto", "library", "provider")
"""Where a background track may come from."""

_SOURCE_ALIASES = {
    "ai": "provider",
    "auto": "auto",
    "fallback": "auto",
    "generate": "provider",
    "generated": "provider",
    "local": "library",
    "library": "library",
    "library_only": "library",
    "provider": "provider",
}
_EPSILON = 1e-9
_LICENSE_KEYS = ("license", "licence", "rights")


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _tags(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items: Iterable[Any] = value.split(",")
    elif isinstance(value, (Mapping, bytes, bytearray)):
        raise ConfigError("tags must be a string or a sequence of strings")
    elif isinstance(value, Iterable):
        items = value
    else:
        raise ConfigError("tags must be a string or a sequence of strings")
    result: list[str] = []
    for item in items:
        text = _text(item).casefold()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _finite(
    value: Any,
    name: str,
    *,
    non_negative: bool = False,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number", details={"value": repr(value)})
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{name} must be finite", details={"value": repr(value)})
    if non_negative and number < 0.0:
        raise ConfigError(
            f"{name} must not be negative", details={"value": repr(value)}
        )
    if positive and number <= 0.0:
        raise ConfigError(f"{name} must be positive", details={"value": repr(value)})
    return number


def _optional_finite(
    value: Any,
    name: str,
    *,
    non_negative: bool = False,
    positive: bool = False,
) -> float | None:
    if value is None:
        return None
    return _finite(value, name, non_negative=non_negative, positive=positive)


def _section(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _config_value(context: StageContext, *names: str) -> Any:
    section = _section(context.config, "music")
    for name in names:
        raw = _field(section, name)
        if raw is not None:
            return raw
    return None


def _source_mode(value: Any) -> str:
    text = _text(value).lower().replace("-", "_") or "auto"
    mode = _SOURCE_ALIASES.get(text, text)
    if mode not in SOURCE_MODES:
        raise ConfigError(
            f"unknown background music source: {value!r}",
            details={"sources": list(SOURCE_MODES)},
        )
    return mode


def _coerce_license(value: Any) -> AssetLicense:
    if isinstance(value, AssetLicense):
        return value
    if isinstance(value, BaseModel):
        return AssetLicense.model_validate(value.model_dump())
    if isinstance(value, str):
        text = value.strip()
        return AssetLicense(name=text) if text else AssetLicense()
    if isinstance(value, Mapping):
        return AssetLicense.model_validate(dict(value))
    return AssetLicense()


def _license_of(value: Any) -> AssetLicense:
    if isinstance(value, Mapping):
        for key in _LICENSE_KEYS:
            raw = value.get(key)
            if raw is not None:
                return _coerce_license(raw)
        raw = value.get("licenses")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and raw:
            return _coerce_license(raw[0])
    for source in _sources(value):
        raw = getattr(source, "license", None)
        if raw is not None:
            return _coerce_license(raw)
        raw = getattr(source, "licenses", None)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and raw:
            return _coerce_license(raw[0])
        asset = getattr(source, "asset", None)
        if asset is not None:
            raw = getattr(asset, "license", None)
            if raw is not None:
                return _coerce_license(raw)
    return AssetLicense()


def _sources(value: Any) -> tuple[Any, ...]:
    result: list[Any] = []
    for candidate in (
        value,
        getattr(value, "outcome", None),
        getattr(value, "result", None),
        getattr(value, "response", None),
        getattr(value, "asset", None),
        getattr(value, "track", None),
    ):
        if candidate is not None and all(candidate is not item for item in result):
            result.append(candidate)
    return tuple(result)


def _local_path(value: str) -> str:
    """Return a local filesystem path for a ``file://`` URL, else *value*."""
    parsed = urlsplit(value)
    if parsed.scheme.lower() == "file":
        return str(Path(unquote(parsed.path)))
    return value


def _path_of(value: Any) -> str:
    if isinstance(value, (str, Path)):
        return _local_path(str(value))
    if isinstance(value, Mapping):
        for name in ("path", "source", "url", "file"):
            raw = value.get(name)
            if isinstance(raw, str) and raw.strip():
                return _local_path(raw.strip())
        paths = value.get("paths")
        if isinstance(paths, Sequence) and not isinstance(paths, (str, bytes)):
            for item in paths:
                if isinstance(item, str) and item.strip():
                    return _local_path(item.strip())
        return ""
    for source in _sources(value):
        for name in ("path", "source", "url", "file"):
            raw = getattr(source, name, None)
            if isinstance(raw, str) and raw.strip():
                return _local_path(raw.strip())
        paths = getattr(source, "paths", None)
        if isinstance(paths, Sequence) and not isinstance(paths, (str, bytes)):
            for item in paths:
                if isinstance(item, str) and item.strip():
                    return _local_path(item.strip())
    return ""


def _data_of(value: Any) -> bytes | None:
    if isinstance(value, Mapping):
        raw = value.get("data") or value.get("audio")
        return bytes(raw) if isinstance(raw, (bytes, bytearray, memoryview)) else None
    for source in _sources(value):
        for name in ("data", "audio"):
            raw = getattr(source, name, None)
            if isinstance(raw, (bytes, bytearray, memoryview)):
                return bytes(raw)
    return None


def _format_of(value: Any) -> str:
    for source in _sources(value):
        raw = getattr(source, "format", None)
        if isinstance(raw, AudioFormat):
            return raw.value
        if isinstance(raw, str) and raw.strip():
            return raw.strip().lower().lstrip(".")
    path = _path_of(value)
    suffix = Path(path).suffix.lower().lstrip(".") if path else ""
    return suffix or "mp3"


def _duration_of(value: Any) -> float | None:
    if isinstance(value, Mapping):
        for name in ("duration_seconds", "duration"):
            raw = value.get(name)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                number = float(raw)
                if math.isfinite(number) and number >= 0.0:
                    return number
    for source in _sources(value):
        for name in ("duration_seconds", "duration"):
            raw = getattr(source, name, None)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                number = float(raw)
                if math.isfinite(number) and number >= 0.0:
                    return number
    return None


def _cost_of(value: Any) -> float:
    if isinstance(value, Mapping):
        for name in ("cost_usd", "amount_usd", "total_cost_usd"):
            raw = value.get(name)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                return max(0.0, float(raw))
        return 0.0
    for name in ("cost_usd", "amount_usd", "total_cost_usd"):
        raw = getattr(value, name, None)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return max(0.0, float(raw))
    return 0.0


def _provider_of(value: Any) -> str:
    for source in _sources(value):
        raw = getattr(source, "provider", None)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return ""


def _model_of(value: Any) -> str:
    for source in _sources(value):
        raw = getattr(source, "model", None)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return ""


def _title_of(value: Any) -> str:
    for source in _sources(value):
        raw = getattr(source, "title", None)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return ""


@dataclass(frozen=True, slots=True)
class BackgroundMusic:
    """One selected background track with its rights and provenance."""

    id: str
    source: str
    license: AssetLicense
    duration_seconds: float
    origin: str
    title: str = ""
    mood: str = ""
    genre: str = ""
    tempo_bpm: float | None = None
    provider: str = ""
    model: str = ""
    score: float = 0.0
    cost_usd: float = 0.0
    prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> str:
        """Alias for :attr:`source`."""
        return self.source

    @property
    def license_name(self) -> str:
        """The license name, falling back to its terms URL."""
        return self.license.name or self.license.url

    @property
    def attribution(self) -> str:
        """The exact credit line the license requires."""
        return self.license.attribution

    @property
    def is_commercial(self) -> bool:
        """Whether the license explicitly allows commercial use."""
        return self.license.is_commercial

    @property
    def is_cleared(self) -> bool:
        """Whether the license is known and allows commercial use."""
        return self.license.is_commercial and self.license.is_known

    @property
    def requires_attribution(self) -> bool:
        """Whether the license requires a credit line."""
        return self.license.requires_attribution

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for manifests and logs."""
        return {
            "id": self.id,
            "source": self.source,
            "origin": self.origin,
            "title": self.title,
            "mood": self.mood,
            "genre": self.genre,
            "tempo_bpm": self.tempo_bpm,
            "duration_seconds": self.duration_seconds,
            "provider": self.provider,
            "model": self.model,
            "score": self.score,
            "cost_usd": self.cost_usd,
            "prompt": self.prompt,
            "license": self.license.model_dump(mode="json"),
            "commercial_cleared": self.is_cleared,
            "metadata": dict(self.metadata),
        }


MusicChoice = BackgroundMusic
"""Alias for :class:`BackgroundMusic`."""

SelectedMusic = BackgroundMusic
"""Alias for :class:`BackgroundMusic`."""

MusicSelection = BackgroundMusic
"""Alias for :class:`BackgroundMusic`."""


_PROFILE_MUSIC_FIELDS: dict[str, tuple[str, ...]] = {
    "prompt": ("prompt", "music_prompt"),
    "mood": ("mood", "tone"),
    "genre": ("genre",),
    "tags": ("tags", "keywords"),
    "style": ("style", "style_bible"),
    "brief": ("brief", "instruction"),
    "niche": ("niche", "subject"),
    "instrumental": ("instrumental",),
}
"""Profile fields that may steer a generated music prompt.

A content profile has no music section of its own, so its tone, brief, niche,
keywords and style bible are used as the deterministic style inputs. A profile
may also carry a nested ``music`` mapping, either directly or under
``overrides``, whose values take precedence over the flat fields.
"""


def _profile_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, BaseModel):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, Mapping) else None
    return None


def _profile_music_values(profile: Any) -> dict[str, Any]:
    """Return the music inputs a profile supplies, without calling a model."""
    if profile is None or isinstance(profile, str):
        return {}
    mapping = _profile_mapping(profile)
    if mapping is None:
        return {}

    values: dict[str, Any] = {}
    for container in (mapping, mapping.get("overrides"), mapping.get("config")):
        nested = _profile_mapping(container)
        if nested is None:
            continue
        music = nested.get("music")
        music_values = _profile_mapping(music)
        if music_values is None:
            continue
        for name in _PROFILE_MUSIC_FIELDS:
            raw = music_values.get(name)
            if raw is not None:
                values.setdefault(name, raw)

    for name, aliases in _PROFILE_MUSIC_FIELDS.items():
        for alias in aliases:
            raw = mapping.get(alias)
            if raw is not None:
                values.setdefault(name, raw)
                break
    return values


def _first_text(*values: Any) -> str:
    for value in values:
        text = _text(value)
        if text:
            return text
    return ""


def _first_tags(*values: Any) -> tuple[str, ...]:
    for value in values:
        if value is None:
            continue
        tags = _tags(value)
        if tags:
            return tags
    return ()


def build_music_prompt(
    profile: Any = None,
    *,
    prompt: Any = None,
    mood: Any = None,
    genre: Any = None,
    tags: Any = None,
    instrumental: bool | None = None,
    brief: Any = None,
    niche: Any = None,
    style: Any = None,
) -> str:
    """Build a deterministic provider prompt from a profile and style filters.

    An explicit *prompt* always wins. Otherwise explicit mood, genre, tags,
    brief, niche and style values override the matching content profile
    settings, and the profile fills whatever is still missing. The result is a
    stable sentence with no provider name, model id or price in it.
    """
    explicit = _text(prompt)
    if explicit:
        return explicit
    if isinstance(profile, str):
        if _text(profile):
            return profile.strip()
        profile = None

    profile_values = _profile_music_values(profile)
    profile_prompt = _text(profile_values.get("prompt"))
    if profile_prompt and not any(
        (
            _text(mood),
            _text(genre),
            _tags(tags),
            _text(brief),
            _text(niche),
            _text(style),
        )
    ):
        return profile_prompt

    profile_instrumental = profile_values.get("instrumental")
    if instrumental is None:
        instrumental = (
            profile_instrumental if isinstance(profile_instrumental, bool) else True
        )

    mood_text = _first_text(mood, profile_values.get("mood"))
    genre_text = _first_text(genre, profile_values.get("genre"))
    tag_values = _first_tags(tags, profile_values.get("tags"))
    brief_text = _first_text(brief, profile_values.get("brief"))
    niche_text = _first_text(niche, profile_values.get("niche"))
    style_text = _first_text(style, profile_values.get("style"))

    parts = ["Instrumental background music" if instrumental else "Background music"]
    if mood_text:
        parts.append(f"{mood_text} mood")
    if genre_text:
        parts.append(f"{genre_text} genre")
    if tag_values:
        parts.append("style: " + ", ".join(tag_values))
    if style_text and style_text.casefold() not in {
        genre_text.casefold(),
        mood_text.casefold(),
    }:
        parts.append(f"inspired by {style_text}")
    if niche_text:
        parts.append(f"theme: {niche_text}")
    if brief_text:
        parts.append(f"brief: {brief_text}")
    return ", ".join(parts)


def build_profile_music_prompt(profile: Any, **values: Any) -> str:
    """Build a music prompt from *profile*, letting explicit values win."""
    return build_music_prompt(profile, **values)


build_music_prompt_from_profile = build_profile_music_prompt
music_prompt_for_profile = build_profile_music_prompt
music_prompt_from_profile = build_profile_music_prompt
profile_music_prompt = build_profile_music_prompt
music_prompt = build_profile_music_prompt


def _from_library_match(match: MusicLibraryMatch) -> BackgroundMusic:
    entry: MusicLibraryEntry = match.entry
    return BackgroundMusic(
        id=entry.id,
        source=entry.path,
        license=entry.license,
        duration_seconds=entry.duration_seconds,
        origin="library",
        title=entry.title,
        mood=entry.mood,
        genre=entry.genre,
        tempo_bpm=entry.tempo_bpm,
        score=match.score,
        metadata={
            "artist": entry.artist,
            "tags": list(entry.tags),
            "format": None if entry.format is None else entry.format.value,
            "reasons": list(match.reasons),
            "matched_tags": list(match.matched_tags),
        },
    )


def select_library_music(
    library: str | Path | Mapping[str, Any] | Iterable[Any] | MusicLibrary,
    *,
    mood: Any = None,
    genre: Any = None,
    tags: Any = None,
    tempo_bpm: float | None = None,
    duration_seconds: float | None = None,
    match_duration: bool = False,
    commercial_only: bool = False,
    license: Any = None,
) -> BackgroundMusic:
    """Select the best local track for the requested mood and style.

    The library is loaded from a manifest path, a mapping, a list of entries or
    an existing :class:`MusicLibrary`. Duration only filters when
    ``match_duration`` is true because a shorter bed may be looped.
    """
    value = load_music_library(library)
    filters: dict[str, Any] = {
        "mood": _text(mood) or None,
        "genre": _text(genre) or None,
        "commercial_only": bool(commercial_only),
    }
    if tempo_bpm is not None:
        filters["tempo_bpm"] = _finite(tempo_bpm, "tempo_bpm", positive=True)
    if match_duration and duration_seconds is not None:
        filters["duration_seconds"] = _finite(
            duration_seconds, "duration_seconds", positive=True
        )
    if license is not None:
        filters["license"] = license
    match = value.select_match(_tags(tags), **filters)
    return _from_library_match(match)


def apply_ducking(
    timeline: Timeline,
    ducking: DuckingSettings | Mapping[str, Any] | None = None,
    *,
    enabled: bool = True,
) -> DuckingSettings:
    """Mark every music track in *timeline* as duckable and return the settings."""
    settings = DuckingSettings.from_value(ducking)
    for track in timeline.music:
        track.duck = bool(enabled and settings.enabled)
    for language in timeline.languages:
        for track in language.music:
            track.duck = bool(enabled and settings.enabled)
    return settings


def verify_timeline_ducking(
    timeline: Timeline,
    ducking: DuckingSettings | Mapping[str, Any] | None = None,
) -> DuckingVerification:
    """Verify that the timeline's voice and music tracks form a ducking graph."""
    settings = DuckingSettings.from_value(ducking)
    voices: list[AudioTrack] = []
    backgrounds: list[AudioTrack] = []
    for language in timeline.resolved_languages():
        for voice in language.voices:
            if not voice.enabled:
                continue
            duration = voice.duration_seconds
            if duration is None and voice.end_seconds is not None:
                duration = max(0.0, voice.end_seconds - (voice.start_seconds or 0.0))
            voices.append(
                AudioTrack(
                    source=voice.source,
                    role=AudioTrackRole.VOICE,
                    offset_seconds=voice.start_seconds or 0.0,
                    duration_seconds=duration,
                )
            )
    for track in timeline.music:
        if not track.enabled:
            continue
        backgrounds.append(
            AudioTrack(
                source=track.source,
                role=AudioTrackRole(track.role.value),
                duck=track.duck,
                volume_db=track.volume_db,
                offset_seconds=track.start_seconds,
                duration_seconds=track.duration_seconds,
                fade_in_seconds=track.fade_in_seconds,
                fade_out_seconds=track.fade_out_seconds,
                loop=track.loop,
            )
        )
    if not voices or not backgrounds:
        duckable = bool(backgrounds) and any(
            track.duck is not False for track in backgrounds
        )
        return DuckingVerification(
            applied=duckable,
            verified=False,
            voice_tracks=len(voices),
            background_tracks=len(backgrounds),
            reason="ducking needs both a voice and a background track",
        )
    graph = build_mix_filtergraph(
        [*voices, *backgrounds],
        AudioMixSettings(
            ducking=settings,
            duration_seconds=max(_timeline_duration(timeline), _EPSILON),
        ),
    )
    return verify_ducking(
        graph,
        settings,
        voice_tracks=len(voices),
        background_tracks=len(backgrounds),
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical_json(value, indent=2) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _timeline_duration(timeline: Timeline) -> float:
    if timeline.duration_seconds is not None:
        return float(timeline.duration_seconds)
    ends: list[float] = []
    for beat in timeline.beats:
        if not beat.enabled:
            continue
        if beat.end_seconds is not None:
            ends.append(float(beat.end_seconds))
        elif beat.duration_seconds is not None:
            ends.append(beat.start_seconds + float(beat.duration_seconds))
    for language in timeline.resolved_languages():
        for voice in language.voices:
            if not voice.enabled:
                continue
            if voice.end_seconds is not None:
                ends.append(float(voice.end_seconds))
            elif voice.duration_seconds is not None:
                ends.append(
                    (voice.start_seconds or 0.0) + float(voice.duration_seconds)
                )
        if language.subtitles is not None:
            ends.append(float(language.subtitles.end_seconds))
    for track in timeline.music:
        if not track.enabled:
            continue
        if track.end_seconds is not None:
            ends.append(float(track.end_seconds))
        elif track.duration_seconds is not None:
            ends.append(track.start_seconds + float(track.duration_seconds))
    return max(ends, default=0.0)


class MusicSelectStage(Stage):
    """Select a background music bed and write it into the timeline.

    ``options`` may carry ``source`` (``auto``, ``library`` or ``provider``),
    ``library``/``library_path``, ``mood``, ``genre``, ``tags``, ``tempo_bpm``,
    ``duration_seconds``, ``match_duration``, ``commercial_only``, ``prompt``,
    ``instrumental``, ``lyrics``, ``lyrics_from_script``, ``script_language``,
    ``profile``, ``volume_db``, ``fade_in_seconds``, ``fade_out_seconds``,
    ``loop``, ``duck``, ``ducking``, ``timeline_filename`` and
    ``music_filename``. A caller may inject a ``music`` callable/service
    instead of calling the configured provider.
    """

    name = MUSIC_STAGE_NAME
    artifacts: tuple[str, ...] = (
        DEFAULT_TIMELINE_FILENAME,
        DEFAULT_MUSIC_FILENAME,
    )

    def __init__(
        self,
        name: str | None = None,
        *,
        options: Mapping[str, Any] | None = None,
        library: Any = None,
        source: str | None = None,
        strategy: str | None = None,
        mood: str | None = None,
        genre: str | None = None,
        tags: Any = None,
        tempo_bpm: float | None = None,
        duration_seconds: float | None = None,
        match_duration: bool | None = None,
        commercial_only: bool | None = None,
        prompt: str | None = None,
        instrumental: bool | None = None,
        lyrics: Any = None,
        lyrics_from_script: bool | None = None,
        script_language: str | None = None,
        profile: Any = None,
        music: Any = None,
        generator: Any = None,
        ducking: DuckingSettings | Mapping[str, Any] | None = None,
        volume_db: float | None = None,
        fade_in_seconds: float | None = None,
        fade_out_seconds: float | None = None,
        timeline_filename: str | None = None,
        music_filename: str | None = None,
        **kwargs: Any,
    ) -> None:
        merged = dict(options or {})
        explicit = {
            "library": library,
            "source": source,
            "strategy": strategy,
            "mood": mood,
            "genre": genre,
            "tags": tags,
            "tempo_bpm": tempo_bpm,
            "duration_seconds": duration_seconds,
            "match_duration": match_duration,
            "commercial_only": commercial_only,
            "prompt": prompt,
            "instrumental": instrumental,
            "lyrics": lyrics,
            "lyrics_from_script": lyrics_from_script,
            "script_language": script_language,
            "profile": profile,
            "music": music,
            "generator": generator,
            "ducking": ducking,
            "volume_db": volume_db,
            "fade_in_seconds": fade_in_seconds,
            "fade_out_seconds": fade_out_seconds,
            "timeline_filename": timeline_filename,
            "music_filename": music_filename,
        }
        for key, value in explicit.items():
            if value is not None:
                merged[key] = value
        merged.update(kwargs)
        timeline_name = str(
            merged.get("timeline_filename") or DEFAULT_TIMELINE_FILENAME
        )
        music_name = str(merged.get("music_filename") or DEFAULT_MUSIC_FILENAME)
        super().__init__(
            name=name or MUSIC_STAGE_NAME,
            options=merged,
            artifacts=(timeline_name, music_name),
        )
        self.library = merged.get("library")
        self.source = _source_mode(
            merged.get("source") or merged.get("strategy") or merged.get("mode")
        )
        self.mood = _text(merged.get("mood"))
        self.genre = _text(merged.get("genre"))
        self.tags = merged.get("tags")
        self.tempo_bpm = _optional_finite(
            merged.get("tempo_bpm"), "tempo_bpm", positive=True
        )
        self.duration_seconds = _optional_finite(
            merged.get("duration_seconds"), "duration_seconds", positive=True
        )
        self.match_duration = bool(merged.get("match_duration", False))
        self.commercial_only = bool(merged.get("commercial_only", False))
        self.prompt = _text(merged.get("prompt"))
        self.instrumental = bool(merged.get("instrumental", True))
        self.lyrics = merged.get("lyrics")
        self.lyrics_from_script = bool(merged.get("lyrics_from_script", False))
        self.script_language = _text(
            merged.get("script_language") or merged.get("language")
        )
        self.profile = merged.get("profile")
        self.music = merged.get("music") or merged.get("generator")
        self.duck = bool(merged.get("duck", True))
        self.ducking = DuckingSettings.from_value(merged.get("ducking"))
        self.volume_db = _finite(
            merged.get("volume_db", DEFAULT_MUSIC_BACKGROUND_VOLUME_DB),
            "volume_db",
        )
        self.fade_in_seconds = _finite(
            merged.get("fade_in_seconds", DEFAULT_MUSIC_BACKGROUND_FADE_IN_SECONDS),
            "fade_in_seconds",
            non_negative=True,
        )
        self.fade_out_seconds = _finite(
            merged.get("fade_out_seconds", DEFAULT_MUSIC_BACKGROUND_FADE_OUT_SECONDS),
            "fade_out_seconds",
            non_negative=True,
        )
        self.loop = bool(merged.get("loop", True))
        self.start_seconds = _finite(
            merged.get("start_seconds", 0.0),
            "start_seconds",
            non_negative=True,
        )
        self.track_id = _text(merged.get("id")) or "music"
        self.replace = bool(merged.get("replace", True))
        self.format = merged.get("format")
        self.count = int(merged.get("count", 1))
        self.prefilter = bool(merged.get("prefilter", self.count > 1))
        self.candidate_config = merged.get("candidate_config") or merged.get(
            "song_candidates"
        )
        self.analyzer = merged.get("analyzer")
        self.ffmpeg_runner = merged.get("ffmpeg_runner")
        self.ffprobe_runner = merged.get("ffprobe_runner")
        self.timeline_filename = timeline_name
        self.music_filename = music_name

    def _library_source(self, context: StageContext) -> Any:
        if self.library is not None:
            return self.library
        section = _section(context.config, "music")
        for name in (
            "library",
            "library_path",
            "library_file",
            "manifest",
            "music_library",
        ):
            raw = _field(section, name)
            if isinstance(raw, str):
                if raw.strip():
                    return raw
                continue
            if raw is not None:
                return raw
        return None

    def _value(self, context: StageContext, own: Any, *names: str) -> Any:
        if own not in (None, ""):
            return own
        return _config_value(context, *names)

    def _profiles(self, context: StageContext) -> Mapping[str, Any]:
        """Return the configured content profiles, however they are nested."""
        section = _section(context.config, "profiles")
        if isinstance(section, Mapping):
            nested = section.get("profiles")
            if isinstance(nested, Mapping):
                return nested
            return section
        nested = _field(section, "profiles")
        return nested if isinstance(nested, Mapping) else {}

    def _profile_name(self, context: StageContext) -> str:
        own = self.profile
        if isinstance(own, str) and own.strip():
            return own.strip()
        run = context.run
        name = _text(getattr(run, "profile", "")) if run is not None else ""
        if name:
            return name
        return _text(_config_value(context, "profile", "profile_name"))

    def _profile(self, context: StageContext) -> tuple[Any, str]:
        """Return the active content profile and its name, when one is selected."""
        own = self.profile
        name = self._profile_name(context)
        if own not in (None, "", {}):
            if isinstance(own, str):
                return self._profiles(context).get(name), name
            return own, name
        if not name:
            return None, ""
        return self._profiles(context).get(name), name

    def _script_paths(self, context: StageContext) -> list[Path]:
        """Return the written script paths for the requested language first."""
        language = _text(self.script_language).replace("_", "-").lower()
        names: list[str] = []
        if language:
            names.extend(
                (
                    f"script.{language}.json",
                    f"script_{language}.json",
                )
            )
        names.append("script.json")
        paths = [context.run_dir / name for name in names]
        paths.extend(sorted(context.run_dir.glob("script.*.json")))
        seen: set[Path] = set()
        result: list[Path] = []
        for path in paths:
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            result.append(path)
        return result

    def _script(self, context: StageContext) -> Script | None:
        """Read the written script the music stage should seed its timeline from."""
        if not self.lyrics_from_script:
            return None
        for path in self._script_paths(context):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(payload, Mapping):
                continue
            try:
                return Script.model_validate(dict(payload))
            except ValueError:
                continue
        return None

    def _script_lyrics(self, context: StageContext) -> str:
        """Read lyric text from a written script in the run directory."""
        if not self.lyrics_from_script:
            return ""
        for path in self._script_paths(context):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            script = None
            if isinstance(payload, Mapping):
                try:
                    script = Script.model_validate(dict(payload))
                except ValueError:
                    script = None
            text = lyrics_text(script if script is not None else payload)
            if text:
                return text
        return ""

    def _seed_timeline_from_lyrics(
        self,
        context: StageContext,
        timeline: Timeline,
    ) -> Timeline:
        """Seed an empty song timeline with timed beats from the written lyrics.

        A song recipe has no voice stage to measure a duration, so the music
        stage derives one from the lyric lines before asking for a track. The
        align stage later replaces these estimated timings with the measured
        song timings.
        """
        if not self.lyrics_from_script:
            return timeline
        if (
            timeline.duration_seconds is not None
            or timeline.beats
            or timeline.languages
            or timeline.music
        ):
            return timeline
        script = self._script(context)
        if script is None or not script.beats:
            return timeline
        language = _text(script.language) or self.script_language or None
        beats: list[TimelineBeat] = []
        cursor = 0.0
        for position, source in enumerate(script.beats):
            text = _text(source.text)
            if not text:
                continue
            length = source.duration_seconds
            if length is None:
                length = estimate_duration(text, language=language)
            length = max(float(length), _EPSILON)
            beats.append(
                TimelineBeat(
                    id=f"lyric_{position}",
                    index=position,
                    start_seconds=cursor,
                    end_seconds=cursor + length,
                    duration_seconds=length,
                    text=text,
                )
            )
            cursor += length
        if not beats:
            return timeline
        return timeline.model_copy(update={"beats": beats, "duration_seconds": cursor})

    def _lyrics(self, context: StageContext) -> str:
        """Resolve explicit lyrics, then lyrics written into the run script."""
        return lyrics_text(self.lyrics) or self._script_lyrics(context)

    def _music_inputs(self, context: StageContext) -> dict[str, Any]:
        """Resolve the prompt and style filters from explicit options and profile."""
        profile, profile_name = self._profile(context)
        lyrics = self._lyrics(context)
        profile_values = _profile_music_values(profile)
        mood = _first_text(
            self._value(context, self.mood, "mood"),
            profile_values.get("mood"),
        )
        genre = _first_text(
            self._value(context, self.genre, "genre"),
            profile_values.get("genre"),
        )
        tags = _first_tags(
            self._value(context, self.tags, "tags"),
            profile_values.get("tags"),
        )
        prompt = build_music_prompt(
            profile,
            prompt=self._value(context, self.prompt, "prompt"),
            mood=self._value(context, self.mood, "mood"),
            genre=self._value(context, self.genre, "genre"),
            tags=self._value(context, self.tags, "tags"),
            instrumental=self.instrumental and not lyrics,
        )
        return {
            "prompt": prompt,
            "mood": mood,
            "genre": genre,
            "tags": tags,
            "lyrics": lyrics,
            "profile": profile_name,
        }

    def _load_timeline(self, context: StageContext) -> Timeline:
        path = context.run_dir / self.timeline_filename
        if not path.is_file():
            return Timeline(id=context.run.id if context.run is not None else "")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigError(
                f"could not read the timeline: {path}",
                details={"path": str(path), "error": str(exc)},
            ) from exc
        return Timeline.model_validate(payload)

    def _target_duration(self, timeline: Timeline) -> float:
        duration = self.duration_seconds
        if duration is None:
            duration = _timeline_duration(timeline)
        if duration is None or duration <= DEFAULT_MUSIC_MIN_DURATION_SECONDS:
            raise ConfigError(
                "music stage needs a positive duration or a timeline with content"
            )
        return duration

    async def _call_music(
        self,
        request: MusicRequest,
        context: StageContext,
        *,
        run_id: str,
    ) -> Any:
        target = self.music
        if target is None:
            if context.router is None:
                raise ConfigError(
                    "music stage requires a music provider or an injected "
                    "music service",
                    details={"stage": self.name},
                )
            return await context.media.music(
                request,
                run_id=run_id,
                operation=CAPABILITY_MUSIC_GEN,
            )
        method = target
        if not callable(target):
            for name in ("generate", "music", "generate_music"):
                candidate = getattr(target, name, None)
                if callable(candidate):
                    method = candidate
                    break
        if not callable(method):
            raise ConfigError(
                "music must be callable or expose a generate method",
                details={"value": type(target).__name__},
            )
        value = method(request)
        if inspect.isawaitable(value):
            value = await value
        return value

    def _provider_request(
        self,
        context: StageContext,
        duration: float,
    ) -> MusicRequest:
        inputs = self._music_inputs(context)
        prompt = inputs["prompt"]
        mood = inputs["mood"]
        genre = inputs["genre"]
        tags = inputs["tags"]
        metadata: dict[str, Any] = {
            "stage": self.name,
            "prompt": prompt,
            "mood": mood,
            "genre": genre,
            "tags": list(tags),
        }
        if inputs["profile"]:
            metadata["profile"] = inputs["profile"]
        lyrics = inputs["lyrics"]
        values: dict[str, Any] = {
            "prompt": prompt,
            "instrumental": self.instrumental and not lyrics,
            "duration_seconds": duration,
            "count": self.count,
            "metadata": metadata,
        }
        if lyrics:
            values["lyrics"] = lyrics
            metadata["lyrics"] = True
        tempo = self.tempo_bpm
        if tempo is None:
            tempo = _optional_finite(
                _config_value(context, "tempo_bpm", "tempo", "bpm"),
                "tempo_bpm",
                positive=True,
            )
        if tempo is not None:
            values["tempo_bpm"] = tempo
        if self.format is not None:
            if isinstance(self.format, AudioFormat):
                values["format"] = self.format
            else:
                try:
                    values["format"] = AudioFormat(
                        str(self.format).strip().lower().lstrip(".")
                    )
                except ValueError as exc:
                    raise ConfigError(
                        f"unknown audio format: {self.format!r}",
                        details={"formats": [item.value for item in AudioFormat]},
                    ) from exc
        return MusicRequest(**values)

    def _provider_choice(
        self,
        value: Any,
        context: StageContext,
        *,
        duration: float,
        mood: str,
        genre: str,
        prompt: str = "",
        tags: Sequence[str] = (),
        profile: str = "",
    ) -> BackgroundMusic:
        source = _path_of(value)
        if not source:
            payload = _data_of(value)
            if payload is None:
                raise ConfigError("music result has no stored audio path or data")
            suffix = _format_of(value)
            path = context.run_dir / "music" / f"{self.track_id}.{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            source = str(path)
        license = _license_of(value)
        measured = _duration_of(value)
        metadata: dict[str, Any] = {
            "source": "provider",
            "prompt": prompt,
            "tags": list(tags),
        }
        if profile:
            metadata["profile"] = profile
        return BackgroundMusic(
            id=self.track_id,
            source=source,
            license=license,
            duration_seconds=duration if measured is None else measured,
            origin="provider",
            title=_title_of(value) or self.track_id,
            mood=mood,
            genre=genre,
            tempo_bpm=self.tempo_bpm,
            provider=_provider_of(value),
            model=_model_of(value),
            cost_usd=_cost_of(value),
            prompt=prompt,
            metadata=metadata,
        )

    def _response_of(self, value: Any) -> MusicResponse | None:
        """Normalise a provider result into a response of song candidates."""
        if isinstance(value, MusicResponse):
            return value
        response = getattr(value, "response", None)
        if isinstance(response, MusicResponse):
            return response
        if isinstance(value, MusicTrack):
            return MusicResponse(tracks=[value])
        tracks = getattr(value, "tracks", None)
        if isinstance(tracks, Sequence) and not isinstance(tracks, (str, bytes)):
            items = list(tracks)
            if items and all(isinstance(item, MusicTrack) for item in items):
                return MusicResponse(
                    tracks=items,
                    provider=_provider_of(value),
                    model=_model_of(value),
                )
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            items = list(value)
            if items and all(isinstance(item, MusicTrack) for item in items):
                return MusicResponse(tracks=items)
        return None

    def _select_provider_candidate(
        self,
        result: Any,
        context: StageContext,
        *,
        duration: float,
        mood: str,
        genre: str,
        prompt: str,
        tags: Sequence[str],
        profile: str,
    ) -> BackgroundMusic:
        """Prefilter generated song candidates and keep the chosen one."""
        response = self._response_of(result)
        if response is None:
            return self._provider_choice(
                result,
                context,
                duration=duration,
                mood=mood,
                genre=genre,
                prompt=prompt,
                tags=tags,
                profile=profile,
            )
        selection: SongCandidateSelection = select_song_candidate(
            response,
            self.candidate_config,
            lyrics=self._lyrics(context),
            analyzer=self.analyzer,
            ffmpeg_runner=self.ffmpeg_runner,
            ffprobe_runner=self.ffprobe_runner,
        )
        if not selection.ok or selection.track is None:
            raise ConfigError(
                "no song candidate passed the prefilter",
                details={
                    "candidates": [
                        candidate.to_dict() for candidate in selection.candidates
                    ]
                },
            )
        choice = self._provider_choice(
            selection.track,
            context,
            duration=duration,
            mood=mood,
            genre=genre,
            prompt=prompt,
            tags=tags,
            profile=profile,
        )
        metadata = dict(choice.metadata)
        metadata["chosen_index"] = selection.chosen_index
        metadata["candidate_count"] = len(selection.candidates)
        metadata["candidates"] = [
            candidate.to_dict() for candidate in selection.candidates
        ]
        license = _license_of(result)
        return replace(
            choice,
            license=license if license.is_known else choice.license,
            provider=_provider_of(result) or choice.provider,
            model=_model_of(result) or choice.model,
            cost_usd=_cost_of(result),
            metadata=metadata,
        )

    async def _select(
        self,
        context: StageContext,
        *,
        duration: float,
        run_id: str,
    ) -> BackgroundMusic:
        inputs = self._music_inputs(context)
        mood = inputs["mood"]
        genre = inputs["genre"]
        tags = inputs["tags"]
        prompt = inputs["prompt"]
        profile_name = inputs["profile"]
        library_source = self._library_source(context)
        library: MusicLibrary | None = None
        if library_source is not None:
            try:
                library = load_music_library(library_source)
            except (ConfigError, ValueError, TypeError) as exc:
                raise ConfigError(
                    "could not load the music library",
                    details={"error": str(exc)},
                ) from exc

        if self.source in ("auto", "library") and library is not None:
            try:
                choice = select_library_music(
                    library,
                    mood=mood,
                    genre=genre,
                    tags=tags,
                    tempo_bpm=self.tempo_bpm,
                    duration_seconds=duration,
                    match_duration=self.match_duration,
                    commercial_only=self.commercial_only,
                )
                metadata = dict(choice.metadata)
                metadata["prompt"] = prompt
                metadata["tags"] = list(tags)
                if profile_name:
                    metadata["profile"] = profile_name
                return replace(choice, prompt=prompt, metadata=metadata)
            except MusicLibraryError as exc:
                if self.source == "library":
                    raise ConfigError(
                        "no local music track matched the requested mood",
                        details={
                            "mood": mood,
                            "genre": genre,
                            "tags": list(_tags(tags)),
                            "error": exc.message,
                        },
                    ) from exc

        if self.source == "library":
            raise ConfigError(
                "music stage needs a local library when source is library"
            )

        request = self._provider_request(context, duration)
        result = await self._call_music(request, context, run_id=run_id)
        if self.prefilter or self.count > 1:
            return self._select_provider_candidate(
                result,
                context,
                duration=duration,
                mood=mood,
                genre=genre,
                prompt=prompt,
                tags=tags,
                profile=profile_name,
            )
        return self._provider_choice(
            result,
            context,
            duration=duration,
            mood=mood,
            genre=genre,
            prompt=prompt,
            tags=tags,
            profile=profile_name,
        )

    def _timeline_track(
        self,
        choice: BackgroundMusic,
        duration: float,
    ) -> TimelineMusic:
        start = min(self.start_seconds, max(0.0, duration - _EPSILON))
        length = max(_EPSILON, duration - start)
        fade_in = min(self.fade_in_seconds, length)
        fade_out = min(self.fade_out_seconds, length)
        settings = self.ducking
        return TimelineMusic(
            id=choice.id or self.track_id,
            source=choice.source,
            role=TimelineTrackRole.MUSIC,
            start_seconds=start,
            end_seconds=start + length,
            duration_seconds=length,
            volume_db=self.volume_db,
            fade_in_seconds=fade_in,
            fade_out_seconds=fade_out,
            loop=self.loop,
            duck=bool(self.duck and settings.enabled),
            license=choice.license,
            metadata={
                "origin": choice.origin,
                "title": choice.title,
                "mood": choice.mood,
                "genre": choice.genre,
                "tempo_bpm": choice.tempo_bpm,
                "provider": choice.provider,
                "model": choice.model,
                "cost_usd": choice.cost_usd,
                "prompt": choice.prompt,
                "license": choice.license.model_dump(mode="json"),
                "license_name": choice.license_name,
                "attribution": choice.attribution,
                "commercial_cleared": choice.is_cleared,
                **choice.metadata,
            },
        )

    async def run(self, context: StageContext) -> StageResult:
        timeline = self._load_timeline(context)
        timeline = self._seed_timeline_from_lyrics(context, timeline)
        duration = self._target_duration(timeline)
        run_id = context.run.id if context.run is not None else ""
        choice = await self._select(context, duration=duration, run_id=run_id)
        track = self._timeline_track(choice, duration)

        if self.replace:
            timeline.music = []
            for language in timeline.languages:
                language.music = []
        else:
            timeline.music = [item for item in timeline.music if item.id != track.id]
        timeline.music.append(track)
        if timeline.duration_seconds is None:
            timeline.duration_seconds = duration

        settings = self.ducking
        verification = verify_timeline_ducking(timeline, settings)
        timeline.metadata["music"] = {
            "source": choice.origin,
            "id": track.id,
            "license": choice.license.model_dump(mode="json"),
            "commercial_cleared": choice.is_cleared,
            "attribution": choice.attribution,
            "ducking": settings.model_dump(mode="json"),
        }
        _write_json(
            context.run_dir / self.timeline_filename,
            timeline.model_dump(mode="json"),
        )

        ducking_payload = {
            **settings.model_dump(mode="json"),
            "applied": bool(track.duck),
            "verified": verification.verified,
            "voice_tracks": verification.voice_tracks,
            "background_tracks": verification.background_tracks,
            "reason": verification.reason,
        }
        manifest = {
            "version": 1,
            "stage": self.name,
            "source": choice.origin,
            "timeline": self.timeline_filename,
            "selected": choice.to_dict(),
            "tracks": [track.model_dump(mode="json")],
            "licenses": [choice.license.model_dump(mode="json")],
            "license": choice.license.model_dump(mode="json"),
            "attribution": choice.attribution,
            "commercial_cleared": choice.is_cleared,
            "requires_attribution": choice.requires_attribution,
            "ducking": ducking_payload,
            "cost_usd": choice.cost_usd,
            "provider": choice.provider,
            "model": choice.model,
            "prompt": choice.prompt,
            "profile": choice.metadata.get("profile", ""),
            "chosen_index": choice.metadata.get("chosen_index"),
            "candidate_count": choice.metadata.get("candidate_count", 1),
            "candidates": choice.metadata.get("candidates", []),
        }
        _write_json(context.run_dir / self.music_filename, manifest)
        self.artifacts = (self.timeline_filename, self.music_filename)

        metadata = {
            "source": choice.origin,
            "track": track.id,
            "path": choice.source,
            "duration_seconds": duration,
            "mood": choice.mood,
            "genre": choice.genre,
            "license": choice.license.model_dump(mode="json"),
            "licenses": [choice.license.model_dump(mode="json")],
            "license_name": choice.license_name,
            "commercial_cleared": choice.is_cleared,
            "attribution": choice.attribution,
            "requires_attribution": choice.requires_attribution,
            "ducking": ducking_payload,
            "ducking_applied": bool(track.duck),
            "ducking_verified": verification.verified,
            "cost_usd": choice.cost_usd,
            "provider": choice.provider,
            "model": choice.model,
            "prompt": choice.prompt,
            "profile": choice.metadata.get("profile", ""),
            "calls": 1 if choice.origin == "provider" else 0,
            "chosen_index": choice.metadata.get("chosen_index"),
            "candidate_count": choice.metadata.get("candidate_count", 1),
            "candidates": choice.metadata.get("candidates", []),
            "timeline": self.timeline_filename,
            "manifest": self.music_filename,
        }
        return StageResult(
            run_id=run_id,
            stage=self.name,
            status=StageState.SUCCEEDED,
            artifacts=[self.timeline_filename, self.music_filename],
            cost_usd=choice.cost_usd,
            metadata=metadata,
        )


MusicStage = MusicSelectStage
"""Alias for :class:`MusicSelectStage`."""

music_stage = MusicSelectStage


def build_music_stage(
    *,
    options: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> MusicSelectStage:
    """Build a :class:`MusicSelectStage` from recipe options and overrides."""
    return MusicSelectStage(options=options, **kwargs)


async def run_music_stage(
    context: StageContext,
    *,
    stage: MusicSelectStage | None = None,
    **options: Any,
) -> StageResult:
    """Build a stage unless one is supplied, then run it asynchronously."""
    if stage is not None and options:
        raise ConfigError("pass either a stage or options, not both")
    active = stage if stage is not None else MusicSelectStage(**options)
    return await active.run(context)


select_music = select_library_music
"""Alias for :func:`select_library_music`."""

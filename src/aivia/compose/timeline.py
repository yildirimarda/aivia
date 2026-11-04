"""Turn a timed timeline into one composition per language and platform.

A timeline is the hand-off between the audio stages and the renderer. It keeps
the shared visual beats and assets, the music bed, and one entry per language
with that language's voice timings and subtitles. Rendering then happens on a
plain :class:`~aivia.compose.schema.Composition`, so this module is pure data
transformation: no file is read, no provider is called and no binary runs.

The timeline duration is exact. It is the end of the last beat, voice or
subtitle cue, or an explicit ``duration_seconds`` when the caller pinned one.
Platforms and profiles may change the canvas, frame rate and loudness, but they
never silently clamp the duration. A caller that needs platform bounds can
validate the resulting composition afterwards.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Self

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from aivia.compose.errors import RenderError
from aivia.compose.output_profiles import resolve_output_profile
from aivia.compose.schema import (
    AudioTrack,
    AudioTrackRole,
    Canvas,
    Composition,
    ImageLayer,
    VideoLayer,
)
from aivia.compose.subtitles import SubtitleTrack, SubtitleWord
from aivia.core.models.asset import AssetLicense

__all__ = [
    "Timeline",
    "TimelineAsset",
    "TimelineBeat",
    "TimelineError",
    "TimelineLanguage",
    "TimelineMusic",
    "TimelineVoice",
    "compositions_by_language_and_platform",
    "compositions_from_timeline",
    "timeline_to_compositions",
]

_EPSILON = 1e-9
_VIDEO_KINDS = frozenset({"video", "clip", "motion", "animation", "stock_video"})
_IMAGE_KINDS = frozenset(
    {"image", "photo", "picture", "still", "stock_image", "visual"}
)
_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".ogv"})
_SLUG = re.compile(r"[^a-z0-9]+")


class TimelineError(RenderError):
    """A timeline cannot be converted into a composition."""

    code = "timeline"


def _finite(
    value: Any,
    name: str,
    *,
    non_negative: bool = False,
    positive: bool = False,
) -> Any:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if non_negative and number < 0.0:
        raise ValueError(f"{name} must not be negative")
    if positive and number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _text(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _language_key(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("language must be a string")
    return value.strip().replace("_", "-").lower()


def _as_list(value: Any) -> Any:
    if value is None:
        return []
    if isinstance(value, (str, Mapping, BaseModel)):
        return [value]
    return value


def _unique(values: Iterable[str], kind: str) -> None:
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        if value in seen:
            raise ValueError(f"duplicate {kind} id {value!r}")
        seen.add(value)


def _slug(value: str) -> str:
    return _SLUG.sub("_", value.strip().lower()).strip("_")


class _TimelineModel(BaseModel):
    """Base for timeline values with readable aliases and mutable fields."""

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        validate_assignment=True,
    )


class TimelineAsset(_TimelineModel):
    """A visual asset that one or more beats can reference."""

    id: str = Field(
        default="",
        validation_alias=AliasChoices("id", "asset_id", "name"),
    )
    source: str = Field(
        min_length=1,
        validation_alias=AliasChoices(
            "source",
            "path",
            "file",
            "url",
            "uri",
            "asset",
        ),
    )
    kind: str = Field(
        default="image",
        validation_alias=AliasChoices("kind", "type", "media_type", "asset_kind"),
    )
    beat: str = Field(
        default="",
        validation_alias=AliasChoices("beat", "beat_id", "for_beat"),
    )
    fit: str = Field(default="", validation_alias=AliasChoices("fit", "fit_mode"))
    enabled: bool = Field(
        default=True, validation_alias=AliasChoices("enabled", "visible")
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_string_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"source": data}
        return data

    @field_validator("id", "beat", "kind", "fit", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _text(value)

    @property
    def is_video(self) -> bool:
        """Whether this asset should become a video layer."""
        return _is_video(self.source, self.kind)


class TimelineBeat(_TimelineModel):
    """One timed visual beat shared by every language."""

    id: str = Field(
        default="",
        validation_alias=AliasChoices("id", "beat_id", "name"),
    )
    index: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices(
            "index",
            "order",
            "number",
            "position",
            "sequence",
        ),
    )
    start_seconds: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("start_seconds", "start", "at", "offset_seconds"),
    )
    end_seconds: float | None = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices("end_seconds", "end", "stop"),
    )
    duration_seconds: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices(
            "duration_seconds",
            "duration",
            "length_seconds",
        ),
    )
    text: str = Field(
        default="",
        validation_alias=AliasChoices("text", "narration", "line", "content"),
    )
    asset: str = Field(
        default="",
        validation_alias=AliasChoices(
            "asset",
            "asset_id",
            "source",
            "visual",
            "visual_source",
        ),
    )
    asset_kind: str = Field(
        default="",
        validation_alias=AliasChoices(
            "asset_kind",
            "kind",
            "type",
            "media_type",
            "visual_type",
        ),
    )
    fit: str = Field(default="", validation_alias=AliasChoices("fit", "fit_mode"))
    z_index: int = Field(
        default=0,
        validation_alias=AliasChoices("z_index", "z", "order_index"),
    )
    enabled: bool = Field(
        default=True, validation_alias=AliasChoices("enabled", "visible")
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_string_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"text": data}
        return data

    @field_validator("id", "text", "asset", "asset_kind", "fit", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _text(value)

    @field_validator("start_seconds", "end_seconds", "duration_seconds", mode="before")
    @classmethod
    def _validate_seconds(cls, value: Any) -> Any:
        return _finite(
            value,
            "beat time",
            non_negative=value is not None,
        )

    @model_validator(mode="after")
    def _resolve_span(self) -> Self:
        if self.end_seconds is None and self.duration_seconds is not None:
            object.__setattr__(
                self,
                "end_seconds",
                self.start_seconds + self.duration_seconds,
            )
        if self.end_seconds is not None and self.duration_seconds is None:
            object.__setattr__(
                self,
                "duration_seconds",
                self.end_seconds - self.start_seconds,
            )
        if self.end_seconds is not None and self.end_seconds <= self.start_seconds:
            raise ValueError("beat end_seconds must be greater than start_seconds")
        return self

    @property
    def order(self) -> int | None:
        """Alias for :attr:`index`."""
        return self.index

    @property
    def end(self) -> float | None:
        """Alias for :attr:`end_seconds`."""
        return self.end_seconds


class TimelineVoice(_TimelineModel):
    """One measured voice segment, optionally attached to a beat."""

    id: str = Field(
        default="",
        validation_alias=AliasChoices("id", "voice_id", "name"),
    )
    beat: str = Field(
        default="",
        validation_alias=AliasChoices("beat", "beat_id", "segment", "segment_id"),
    )
    source: str = Field(
        min_length=1,
        validation_alias=AliasChoices(
            "source",
            "path",
            "file",
            "audio",
            "asset",
            "asset_id",
        ),
    )
    start_seconds: float | None = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices("start_seconds", "start", "at", "offset_seconds"),
    )
    end_seconds: float | None = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices("end_seconds", "end", "stop"),
    )
    duration_seconds: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices(
            "duration_seconds",
            "duration",
            "length_seconds",
        ),
    )
    volume_db: float = Field(
        default=0.0,
        validation_alias=AliasChoices("volume_db", "volume", "gain_db", "gain"),
    )
    words: list[SubtitleWord] = Field(
        default_factory=list,
        validation_alias=AliasChoices("words", "timed_words", "word_timings"),
    )
    speaker: str = Field(default="", validation_alias=AliasChoices("speaker", "who"))
    voice: str = Field(
        default="",
        validation_alias=AliasChoices("voice", "voice_id", "voice_name"),
    )
    language: str = Field(
        default="",
        validation_alias=AliasChoices("language", "lang", "locale"),
    )
    enabled: bool = Field(
        default=True, validation_alias=AliasChoices("enabled", "visible")
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_string_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"source": data}
        return data

    @field_validator(
        "id", "beat", "source", "speaker", "voice", "language", mode="before"
    )
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _text(value)

    @field_validator("language", mode="after")
    @classmethod
    def _normalize_language(cls, value: str) -> str:
        return _language_key(value)

    @field_validator("start_seconds", "end_seconds", "duration_seconds", mode="before")
    @classmethod
    def _validate_seconds(cls, value: Any) -> Any:
        return _finite(
            value,
            "voice time",
            non_negative=value is not None,
        )

    @field_validator("volume_db", mode="before")
    @classmethod
    def _validate_volume(cls, value: Any) -> Any:
        return _finite(value, "volume_db")

    @field_validator("words", mode="before")
    @classmethod
    def _coerce_words(cls, value: Any) -> Any:
        return [] if value is None else value

    @model_validator(mode="after")
    def _resolve_span(self) -> Self:
        if self.end_seconds is None and self.duration_seconds is not None:
            start = self.start_seconds or 0.0
            object.__setattr__(self, "end_seconds", start + self.duration_seconds)
        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds < self.start_seconds
        ):
            raise ValueError("voice end_seconds must not precede start_seconds")
        if (
            self.end_seconds is not None
            and self.start_seconds is not None
            and self.duration_seconds is None
        ):
            object.__setattr__(
                self,
                "duration_seconds",
                self.end_seconds - self.start_seconds,
            )
        return self

    @property
    def end(self) -> float | None:
        """Alias for :attr:`end_seconds`."""
        return self.end_seconds


class TimelineMusic(_TimelineModel):
    """One music or background track for the composition."""

    id: str = Field(
        default="",
        validation_alias=AliasChoices("id", "track_id", "name"),
    )
    source: str = Field(
        min_length=1,
        validation_alias=AliasChoices(
            "source",
            "path",
            "file",
            "audio",
            "asset",
            "asset_id",
        ),
    )
    role: AudioTrackRole = Field(
        default=AudioTrackRole.MUSIC,
        validation_alias=AliasChoices("role", "kind", "type"),
    )
    start_seconds: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("start_seconds", "start", "at", "offset_seconds"),
    )
    end_seconds: float | None = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices("end_seconds", "end", "stop"),
    )
    duration_seconds: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices(
            "duration_seconds",
            "duration",
            "length_seconds",
        ),
    )
    volume_db: float = Field(
        default=0.0,
        validation_alias=AliasChoices("volume_db", "volume", "gain_db", "gain"),
    )
    fade_in_seconds: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("fade_in_seconds", "fade_in"),
    )
    fade_out_seconds: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("fade_out_seconds", "fade_out"),
    )
    loop: bool = Field(default=False, validation_alias=AliasChoices("loop", "looped"))
    duck: bool | None = Field(
        default=None,
        validation_alias=AliasChoices("duck", "ducked", "allow_ducking"),
    )
    license: AssetLicense | None = Field(
        default=None,
        validation_alias=AliasChoices("license", "licence", "rights"),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = Field(
        default=True, validation_alias=AliasChoices("enabled", "visible")
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_string_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"source": data}
        return data

    @field_validator("id", "source", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _text(value)

    @field_validator(
        "start_seconds",
        "end_seconds",
        "duration_seconds",
        "fade_in_seconds",
        "fade_out_seconds",
        mode="before",
    )
    @classmethod
    def _validate_seconds(cls, value: Any) -> Any:
        return _finite(
            value,
            "music time",
            non_negative=value is not None,
        )

    @field_validator("volume_db", mode="before")
    @classmethod
    def _validate_volume(cls, value: Any) -> Any:
        return _finite(value, "volume_db")

    @field_validator("license", mode="before")
    @classmethod
    def _coerce_license(cls, value: Any) -> Any:
        if value is None or isinstance(value, AssetLicense):
            return value
        if isinstance(value, str):
            text = value.strip()
            return {"name": text} if text else None
        if isinstance(value, BaseModel):
            return value.model_dump()
        return value

    @field_validator("metadata", mode="before")
    @classmethod
    def _coerce_metadata(cls, value: Any) -> Any:
        return {} if value is None else value

    @model_validator(mode="after")
    def _resolve_span(self) -> Self:
        if self.end_seconds is None and self.duration_seconds is not None:
            object.__setattr__(
                self,
                "end_seconds",
                self.start_seconds + self.duration_seconds,
            )
        if self.end_seconds is not None and self.duration_seconds is None:
            object.__setattr__(
                self,
                "duration_seconds",
                self.end_seconds - self.start_seconds,
            )
        if self.end_seconds is not None and self.end_seconds <= self.start_seconds:
            raise ValueError("music end_seconds must be greater than start_seconds")
        if (
            self.duration_seconds is not None
            and self.fade_out_seconds > self.duration_seconds
        ):
            raise ValueError("fade_out_seconds must not exceed duration_seconds")
        return self

    @property
    def end(self) -> float | None:
        """Alias for :attr:`end_seconds`."""
        return self.end_seconds


class TimelineLanguage(_TimelineModel):
    """Voice timings and subtitles for one language."""

    language: str = Field(
        default="",
        validation_alias=AliasChoices("language", "lang", "locale", "code"),
    )
    voices: list[TimelineVoice] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "voices",
            "voice",
            "voice_timings",
            "voice_tracks",
        ),
    )
    subtitles: SubtitleTrack | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "subtitles",
            "subtitle",
            "subtitle_track",
            "captions",
            "caption",
        ),
    )
    music: list[TimelineMusic] = Field(
        default_factory=list,
        validation_alias=AliasChoices("music", "tracks", "audio_tracks"),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_shorthands(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"language": data}
        if isinstance(data, (list, tuple)):
            return {"voices": list(data)}
        return data

    @field_validator("language", mode="before")
    @classmethod
    def _strip_language(cls, value: Any) -> Any:
        return _language_key(value)

    @field_validator("voices", "music", mode="before")
    @classmethod
    def _coerce_tracks(cls, value: Any) -> Any:
        return _as_list(value)

    @property
    def lang(self) -> str:
        """Alias for :attr:`language`."""
        return self.language


class Timeline(_TimelineModel):
    """A timed timeline with shared visuals and per-language audio.

    ``beats`` and ``assets`` are shared by every language. ``languages`` holds
    the measured voice timings and subtitle track for each language; a timeline
    without languages may instead use the top-level ``language``, ``voices``
    and ``subtitles`` fields for a single language. ``music`` is shared unless a
    language supplies its own.
    """

    id: str = Field(
        default="",
        validation_alias=AliasChoices("id", "timeline_id", "name"),
    )
    duration_seconds: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices(
            "duration_seconds",
            "duration",
            "length_seconds",
        ),
    )
    language: str = Field(
        default="",
        validation_alias=AliasChoices("language", "lang", "locale"),
    )
    voices: list[TimelineVoice] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "voices",
            "voice",
            "voice_timings",
            "voice_tracks",
        ),
    )
    subtitles: SubtitleTrack | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "subtitles",
            "subtitle",
            "subtitle_track",
            "captions",
            "caption",
        ),
    )
    beats: list[TimelineBeat] = Field(
        default_factory=list,
        validation_alias=AliasChoices("beats", "segments", "scenes", "units"),
    )
    assets: list[TimelineAsset] = Field(
        default_factory=list,
        validation_alias=AliasChoices("assets", "visuals", "media"),
    )
    music: list[TimelineMusic] = Field(
        default_factory=list,
        validation_alias=AliasChoices("music", "tracks", "audio_tracks"),
    )
    languages: list[TimelineLanguage] = Field(
        default_factory=list,
        validation_alias=AliasChoices("languages", "language_tracks", "variants"),
    )
    platforms: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("platforms", "targets", "output_platforms"),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_shorthands(cls, data: Any) -> Any:
        if isinstance(data, BaseModel):
            return data.model_dump()
        if not isinstance(data, Mapping):
            return data
        values = dict(data)

        languages = values.get("languages")
        if isinstance(languages, Mapping):
            entries: list[Any] = []
            for code, raw in languages.items():
                if isinstance(raw, Mapping):
                    item = dict(raw)
                    item.setdefault("language", code)
                elif isinstance(raw, (list, tuple)):
                    item = {"language": code, "voices": list(raw)}
                elif isinstance(raw, str):
                    item = {"language": code, "voices": [{"source": raw}]}
                elif raw is None:
                    item = {"language": code}
                else:
                    item = {"language": code, "voices": raw}
                entries.append(item)
            values["languages"] = entries
        elif isinstance(languages, str):
            values["languages"] = [languages]

        assets = values.get("assets")
        if isinstance(assets, Mapping):
            entries = []
            for asset_id, raw in assets.items():
                if isinstance(raw, Mapping):
                    item = dict(raw)
                    item.setdefault("id", asset_id)
                else:
                    item = {"id": asset_id, "source": raw}
                entries.append(item)
            values["assets"] = entries

        beats = values.get("beats")
        if isinstance(beats, Mapping):
            entries = []
            for beat_id, raw in beats.items():
                if isinstance(raw, Mapping):
                    item = dict(raw)
                    item.setdefault("id", beat_id)
                else:
                    item = {"id": beat_id, "asset": raw}
                entries.append(item)
            values["beats"] = entries

        for key in ("music", "voices"):
            raw = values.get(key)
            if isinstance(raw, (str, Mapping)):
                values[key] = [raw]
        if isinstance(values.get("platforms"), str):
            values["platforms"] = [values["platforms"]]
        return values

    @field_validator("id", "language", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _text(value)

    @field_validator("language", mode="after")
    @classmethod
    def _normalize_language(cls, value: str) -> str:
        return _language_key(value)

    @field_validator("duration_seconds", mode="before")
    @classmethod
    def _validate_duration(cls, value: Any) -> Any:
        return _finite(value, "duration_seconds", positive=value is not None)

    @field_validator(
        "beats",
        "assets",
        "music",
        "languages",
        "platforms",
        mode="before",
    )
    @classmethod
    def _coerce_items(cls, value: Any) -> Any:
        return _as_list(value)

    @field_validator("platforms", mode="after")
    @classmethod
    def _normalize_platforms(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            name = value.strip() if isinstance(value, str) else value
            if name and name not in result:
                result.append(name)
        return result

    @model_validator(mode="after")
    def _assign_indexes_and_check_references(self) -> Self:
        for position, beat in enumerate(self.beats):
            if beat.index is None:
                beat.index = position

        _unique((beat.id for beat in self.beats), "beat")
        _unique((asset.id for asset in self.assets), "asset")
        _unique((voice.id for voice in self.voices), "voice")
        _unique((track.id for track in self.music), "music track")
        _unique((item.language for item in self.languages), "language")

        beat_ids = {beat.id for beat in self.beats if beat.id}
        voices = list(self.voices)
        for language in self.languages:
            voices.extend(language.voices)
            _unique((voice.id for voice in language.voices), "voice")
            _unique((track.id for track in language.music), "music track")
        for voice in voices:
            if voice.beat and voice.beat not in beat_ids:
                raise ValueError(f"voice references unknown beat {voice.beat!r}")
        for asset in self.assets:
            if asset.beat and asset.beat not in beat_ids:
                raise ValueError(f"asset references unknown beat {asset.beat!r}")
        return self

    def resolved_languages(self) -> list[TimelineLanguage]:
        """Return the language entries, folding top-level single-language fields."""
        languages = list(self.languages)
        has_default = (
            bool(self.voices) or self.subtitles is not None or bool(self.language)
        )
        if not has_default:
            return languages or [TimelineLanguage()]
        code = _language_key(self.language)
        for position, language in enumerate(languages):
            if _language_key(language.language) != code:
                continue
            languages[position] = language.model_copy(
                update={
                    "voices": [*language.voices, *self.voices],
                    "subtitles": language.subtitles or self.subtitles,
                }
            )
            return languages
        languages.append(
            TimelineLanguage(
                language=self.language,
                voices=list(self.voices),
                subtitles=self.subtitles,
            )
        )
        return languages

    def beat(self, beat_id: str) -> TimelineBeat | None:
        """Return the beat with *beat_id*, or ``None`` when absent."""
        for beat in self.beats:
            if beat.id == beat_id:
                return beat
        return None

    def asset(self, asset_id: str) -> TimelineAsset | None:
        """Return the asset with *asset_id*, or ``None`` when absent."""
        for asset in self.assets:
            if asset.id == asset_id:
                return asset
        return None


def _is_video(source: str, kind: str) -> bool:
    normalized = kind.strip().lower().replace("-", "_")
    if normalized in _VIDEO_KINDS:
        return True
    if normalized in _IMAGE_KINDS:
        return False
    lowered = source.lower().split("?", 1)[0]
    return any(lowered.endswith(extension) for extension in _VIDEO_EXTENSIONS)


def _layer_id(beat: TimelineBeat, position: int, used: set[str]) -> str:
    base = beat.id.strip() or f"beat_{position}"
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _track_id(
    base: str,
    fallback: str,
    used: set[str],
) -> str:
    candidate = base.strip() or fallback
    if candidate not in used:
        used.add(candidate)
        return candidate
    suffix = 2
    original = candidate
    while candidate in used:
        candidate = f"{original}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _voice_start(voice: TimelineVoice, beat: TimelineBeat | None) -> float:
    if voice.start_seconds is not None:
        return voice.start_seconds
    if beat is not None:
        return beat.start_seconds
    return 0.0


def _beat_explicit_end(beat: TimelineBeat) -> float | None:
    if beat.end_seconds is not None:
        return beat.end_seconds
    if beat.duration_seconds is not None:
        return beat.start_seconds + beat.duration_seconds
    return None


def _language_duration(
    timeline: Timeline,
    language: TimelineLanguage,
    beats: Sequence[TimelineBeat],
    beats_by_id: Mapping[str, TimelineBeat],
) -> float:
    voice_ends: list[float] = []
    voice_ends_by_beat: dict[str, list[float]] = {}
    for voice in language.voices:
        if not voice.enabled:
            continue
        beat = beats_by_id.get(voice.beat)
        start = _voice_start(voice, beat)
        end = voice.end_seconds
        if end is None and voice.duration_seconds is not None:
            end = start + voice.duration_seconds
        if end is None:
            continue
        voice_ends.append(end)
        if voice.beat:
            voice_ends_by_beat.setdefault(voice.beat, []).append(end)

    beat_ends: list[float] = []
    for position, beat in enumerate(beats):
        end = _beat_explicit_end(beat)
        if end is None:
            candidates = voice_ends_by_beat.get(beat.id, [])
            if candidates:
                end = max(candidates)
            elif position + 1 < len(beats):
                end = beats[position + 1].start_seconds
        if end is not None:
            beat_ends.append(end)

    ends = [*beat_ends, *voice_ends]
    if language.subtitles is not None:
        ends.append(language.subtitles.end_seconds)
    content_end = max(ends) if ends else 0.0

    if timeline.duration_seconds is not None:
        if content_end > timeline.duration_seconds + _EPSILON:
            raise TimelineError(
                f"timeline duration {timeline.duration_seconds:g}s is shorter "
                f"than its content ending at {content_end:g}s"
            )
        return timeline.duration_seconds

    if content_end <= 0.0:
        music = language.music or timeline.music
        music_ends = [
            track.end_seconds
            for track in music
            if track.enabled and track.end_seconds is not None
        ]
        content_end = max(music_ends, default=0.0)
    if content_end <= 0.0:
        raise TimelineError("timeline has no beats, voices, subtitles or music")
    return content_end


def _beat_end(
    beat: TimelineBeat,
    position: int,
    beats: Sequence[TimelineBeat],
    language: TimelineLanguage,
    duration: float,
) -> float:
    end = _beat_explicit_end(beat)
    if end is None:
        voice_ends = [
            voice.end_seconds
            for voice in language.voices
            if voice.enabled and voice.beat == beat.id and voice.end_seconds is not None
        ]
        if voice_ends:
            end = max(voice_ends)
        elif position + 1 < len(beats):
            end = beats[position + 1].start_seconds
        else:
            end = duration
    return min(end, duration)


def _beat_layers(
    timeline: Timeline,
    language: TimelineLanguage,
    beats: Sequence[TimelineBeat],
    duration: float,
) -> list[ImageLayer | VideoLayer]:
    assets_by_id = {asset.id: asset for asset in timeline.assets if asset.id}
    assets_by_beat = {
        asset.beat: asset for asset in timeline.assets if asset.beat and asset.enabled
    }
    layers: list[ImageLayer | VideoLayer] = []
    used: set[str] = set()
    for position, beat in enumerate(beats):
        if not beat.enabled:
            continue
        end = _beat_end(beat, position, beats, language, duration)
        if end <= beat.start_seconds:
            continue
        asset = assets_by_id.get(beat.asset) if beat.asset else None
        if asset is None and not beat.asset:
            asset = assets_by_beat.get(beat.id)
        source = asset.source if asset is not None else beat.asset
        if not source:
            continue
        kind = beat.asset_kind or (asset.kind if asset is not None else "")
        fit = beat.fit or (asset.fit if asset is not None else "")
        values: dict[str, Any] = {
            "id": _layer_id(beat, position, used),
            "source": source,
            "start_seconds": beat.start_seconds,
            "end_seconds": end,
            "z_index": (
                beat.z_index if "z_index" in beat.model_fields_set else position
            ),
        }
        if fit:
            values["fit"] = fit
        if _is_video(source, kind):
            values["muted"] = True
            layers.append(VideoLayer.model_validate(values))
        else:
            layers.append(ImageLayer.model_validate(values))
    return layers


def _voice_tracks(
    language: TimelineLanguage,
    beats_by_id: Mapping[str, TimelineBeat],
    duration: float,
) -> list[AudioTrack]:
    tracks: list[AudioTrack] = []
    used: set[str] = set()
    for position, voice in enumerate(language.voices):
        if not voice.enabled:
            continue
        beat = beats_by_id.get(voice.beat)
        start = _voice_start(voice, beat)
        if start >= duration - _EPSILON:
            raise TimelineError(
                f"voice {voice.id or position!r} starts at {start:g}s, at or "
                f"after the composition duration of {duration:g}s"
            )
        end = voice.end_seconds
        if end is None and voice.duration_seconds is not None:
            end = start + voice.duration_seconds
        if end is None and beat is not None:
            end = _beat_explicit_end(beat)
        if end is None:
            end = duration
        end = min(end, duration)
        if end <= start:
            raise TimelineError(
                f"voice {voice.id or position!r} ends at {end:g}s, before its "
                f"start at {start:g}s"
            )
        tracks.append(
            AudioTrack(
                id=_track_id(
                    voice.id,
                    f"voice_{_slug(language.language) or 'default'}_{position}",
                    used,
                ),
                source=voice.source,
                role=AudioTrackRole.VOICE,
                start_seconds=start,
                duration_seconds=end - start,
                volume_db=voice.volume_db,
            )
        )
    return tracks


def _music_tracks(
    tracks: Sequence[TimelineMusic],
    duration: float,
) -> list[AudioTrack]:
    result: list[AudioTrack] = []
    used: set[str] = set()
    for position, track in enumerate(tracks):
        if not track.enabled:
            continue
        start = track.start_seconds
        if start >= duration - _EPSILON:
            continue
        end = track.end_seconds
        if end is None and track.duration_seconds is not None:
            end = start + track.duration_seconds
        if end is None:
            end = duration
        end = min(end, duration)
        if end <= start:
            continue
        length = end - start
        result.append(
            AudioTrack(
                id=_track_id(track.id, f"music_{position}", used),
                source=track.source,
                role=track.role,
                start_seconds=start,
                duration_seconds=length,
                volume_db=track.volume_db,
                fade_in_seconds=min(track.fade_in_seconds, length),
                fade_out_seconds=min(track.fade_out_seconds, length),
                loop=track.loop,
                duck=track.duck,
            )
        )
    return result


def _subtitle_track(
    language: TimelineLanguage,
) -> SubtitleTrack | None:
    if language.subtitles is not None:
        if language.subtitles.language == language.language:
            return language.subtitles
        return language.subtitles.model_copy(update={"language": language.language})
    words = [word for voice in language.voices if voice.enabled for word in voice.words]
    if not words:
        return None
    return SubtitleTrack.from_words(words, language=language.language)


def _composition_id(
    timeline: Timeline,
    language: str,
    platform: str,
    used: set[str],
) -> str:
    base = timeline.id.strip() or "timeline"
    parts = [base]
    if language:
        parts.append(_slug(language) or "default")
    if platform:
        parts.append(_slug(platform) or "default")
    candidate = "_".join(part for part in parts if part)
    if candidate not in used:
        used.add(candidate)
        return candidate
    suffix = 2
    original = candidate
    while candidate in used:
        candidate = f"{original}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def timeline_to_compositions(
    timeline: Timeline | Mapping[str, Any],
    *,
    video: Any | None = None,
    platforms: Sequence[str | None] | None = None,
    profile: str | None = None,
) -> list[Composition]:
    """Convert *timeline* into one composition per language and platform.

    *video* is the video configuration used to resolve canvas, frame rate,
    codec and loudness for each platform. *platforms* overrides the timeline's
    own ``platforms`` list; an empty or missing list yields one composition per
    language using the default profile. The duration is never clamped: it is
    the exact end of the language's beats, voices and subtitles.
    """
    value = (
        timeline
        if isinstance(timeline, Timeline)
        else Timeline.model_validate(dict(timeline))
    )
    targets: list[str | None] = list(platforms) if platforms is not None else []
    if not targets:
        targets = list(value.platforms) or [None]

    languages = value.resolved_languages()
    beats_by_id = {beat.id: beat for beat in value.beats if beat.id}
    compositions: list[Composition] = []
    used_ids: set[str] = set()

    for language in languages:
        duration = _language_duration(value, language, value.beats, beats_by_id)
        layers = _beat_layers(value, language, value.beats, duration)
        voice_tracks = _voice_tracks(language, beats_by_id, duration)
        music = language.music or value.music
        tracks = [*voice_tracks, *_music_tracks(music, duration)]
        subtitles = _subtitle_track(language)

        for platform in targets:
            name = platform.strip() if isinstance(platform, str) else ""
            resolved = resolve_output_profile(
                video,
                platform=name or None,
                profile=profile,
            )
            composition = Composition(
                id=_composition_id(value, language.language, name, used_ids),
                canvas=Canvas(
                    width=resolved.width,
                    height=resolved.height,
                ),
                duration_seconds=duration,
                fps=resolved.fps,
                layers=list(layers),
                audio_tracks=list(tracks),
                subtitles=subtitles,
                platform=name,
                language=language.language,
                loudness_lufs=resolved.loudness_lufs,
            )
            compositions.append(composition)
    return compositions


def compositions_from_timeline(
    timeline: Timeline | Mapping[str, Any],
    **kwargs: Any,
) -> list[Composition]:
    """Alias for :func:`timeline_to_compositions`."""
    return timeline_to_compositions(timeline, **kwargs)


def compositions_by_language_and_platform(
    compositions: Iterable[Composition],
) -> dict[tuple[str, str], Composition]:
    """Index compositions by their ``(language, platform)`` pair."""
    return {
        (composition.language, composition.platform): composition
        for composition in compositions
    }

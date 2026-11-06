"""Generate several song candidates, prefilter them and record the chosen one.

A song is a vocal music track. The deterministic audio prefilter that already
serves background music (duration, silence, clipping and loudness) applies to a
song unchanged, so this module reuses :mod:`aivia.media.music_candidates`
instead of running a second analyser. It adds the song specific rules:

* a vocal request must carry lyrics and must not be instrumental;
* when the request asked for lyrics, a candidate that came back without them is
  rejected by the prefilter;
* the chosen index is kept on the selection and can be written to a small
  manifest beside the generated audio.

Generation accepts any object with an async ``generate`` method, including a
provider in a unit test and the music service in normal operation. The helper
never owns caching, budget checks, licensing or cost recording; those stay in
the music service. Parsing, filtering and selection are deterministic and
offline.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from aivia.core.errors import ConfigError
from aivia.media.music_candidates import (
    MusicCandidate,
    MusicCandidateAnalysis,
    MusicCandidateConfig,
    MusicCandidateError,
    MusicCandidateSelection,
    MusicCandidateSelector,
    analyze_music_candidate,
    generate_music_candidates,
    music_candidate_config,
    parse_music_analysis,
    score_music_candidate,
    select_music_candidate,
)
from aivia.providers.protocols.music import MusicRequest
from aivia.writing.lyrics import extract_lyrics

__all__ = [
    "DEFAULT_SONG_CANDIDATE_COUNT",
    "SONG_CANDIDATE_OPERATION",
    "SongCandidate",
    "SongCandidateAnalysis",
    "SongCandidateConfig",
    "SongCandidateError",
    "SongCandidateFilter",
    "SongCandidateGenerator",
    "SongCandidatePrefilter",
    "SongCandidateResult",
    "SongCandidateSelection",
    "SongCandidateSelector",
    "analyze_song_candidate",
    "filter_song_candidates",
    "generate_song_candidates",
    "generate_song_candidates_sync",
    "lyrics_text",
    "parse_song_analysis",
    "prefilter_song_candidates",
    "score_song_candidate",
    "select_song_candidate",
    "select_song_candidates",
    "select_song_track",
    "song_candidate_config",
    "song_manifest",
    "write_song_manifest",
]

DEFAULT_SONG_CANDIDATE_COUNT = 3
"""How many alternatives a song generation asks for when the caller is silent."""

SONG_CANDIDATE_OPERATION = "song_generation"
"""The operation name recorded on cost events for a song generation call."""

SongCandidateAnalysis = MusicCandidateAnalysis
"""A song reuses the measured music candidate analysis."""

SongCandidate = MusicCandidate
"""A song candidate reuses the analysed music candidate record."""

SongCandidateConfig = MusicCandidateConfig
"""The audio thresholds are identical for songs and background music."""

SongCandidateError = MusicCandidateError
"""No generated song candidate passed the deterministic prefilter."""

SongCandidateResult = MusicCandidateSelection
SongCandidateSelection = MusicCandidateSelection
"""A selection records every song candidate and the chosen index."""

_SONG_SECTIONS = ("song_candidates", "songs", "song")


def lyrics_text(value: Any = None) -> str:
    """Return lyric text from a string, payload, script or mapping.

    A plain string is returned stripped. Anything else is passed through the
    deterministic lyrics parser so a script or payload keeps its line order.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    payload = extract_lyrics(value)
    return "\n".join(
        line for section in payload.sections for line in section.lines
    ).strip()


def song_candidate_config(
    value: SongCandidateConfig | Mapping[str, Any] | BaseModel | None = None,
    **overrides: Any,
) -> SongCandidateConfig:
    """Resolve a song candidate config, reading the song sections first."""
    if isinstance(value, Mapping):
        for name in _SONG_SECTIONS:
            nested = value.get(name)
            if isinstance(nested, Mapping):
                return music_candidate_config(nested, **overrides)
    return music_candidate_config(value, **overrides)


def _with_lyrics(
    selection: SongCandidateSelection,
    require_lyrics: bool,
) -> SongCandidateSelection:
    """Reject passing candidates that came back without the requested lyrics."""
    if not require_lyrics:
        return selection
    candidates: list[MusicCandidate] = []
    for candidate in selection.candidates:
        has_lyrics = bool(candidate.track is not None and candidate.track.lyrics)
        if candidate.passed and not has_lyrics:
            candidates.append(
                MusicCandidate(
                    analysis=candidate.analysis,
                    track=candidate.track,
                    passed=False,
                    reasons=(*candidate.reasons, "missing lyrics"),
                    score=candidate.score,
                )
            )
        else:
            candidates.append(candidate)
    accepted = [candidate for candidate in candidates if candidate.passed]
    chosen_index: int | None = None
    if accepted:
        prefer = selection.config.prefer if selection.config is not None else "first"
        if prefer == "quality":
            chosen = max(accepted, key=lambda item: (item.score, -item.index))
        else:
            chosen = accepted[0]
        chosen_index = chosen.index
    return MusicCandidateSelection(
        candidates=tuple(candidates),
        chosen_index=chosen_index,
        config=selection.config,
        response=selection.response,
    )


def _request_lyrics(
    request: MusicRequest | None,
    lyrics: Any,
) -> tuple[str, bool]:
    if lyrics is not None:
        text = lyrics_text(lyrics)
    else:
        text = _text(request.lyrics) if request is not None else ""
    return text, bool(text)


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _vocal_request(
    request: MusicRequest,
    *,
    lyrics: Any,
    count: int | None,
) -> tuple[MusicRequest, bool]:
    if not isinstance(request, MusicRequest):
        raise TypeError("request must be a MusicRequest")
    text, requested = _request_lyrics(request, lyrics)
    if request.instrumental and text:
        raise ConfigError(
            "a song request must not be instrumental when it carries lyrics",
            details={"lyrics": len(text)},
        )
    active = request
    if text:
        active = active.model_copy(update={"lyrics": text, "instrumental": False})
    if count is not None:
        active = active.model_copy(update={"count": count})
    return active, requested


async def generate_song_candidates(
    generator: Any,
    request: MusicRequest,
    *,
    lyrics: Any = None,
    config: SongCandidateConfig | Mapping[str, Any] | BaseModel | None = None,
    count: int | None = None,
    require_lyrics: bool | None = None,
    analyzer: Callable[..., Any] | None = None,
    ffmpeg_runner: Callable[[Sequence[str]], Any] | None = None,
    ffprobe_runner: Callable[[Sequence[str]], Any] | None = None,
    ffmpeg_executable: str = "ffmpeg",
    ffprobe_executable: str = "ffprobe",
    timeout_seconds: float | None = None,
    ffprobe_timeout_seconds: float | None = None,
) -> SongCandidateSelection:
    """Ask *generator* for several songs, prefilter them and choose one.

    A lyric string, payload or script is copied onto the request. When the
    request or *lyrics* carries text the call is vocal, so a candidate without
    lyrics is rejected unless ``require_lyrics`` is explicitly false.
    """
    active, requested = _vocal_request(request, lyrics=lyrics, count=count)
    settings = song_candidate_config(config)
    if settings.target_duration_seconds is None and active.duration_seconds is not None:
        settings = settings.model_copy(
            update={"target_duration_seconds": active.duration_seconds}
        )
    values: dict[str, Any] = {
        "config": settings,
        "analyzer": analyzer,
        "ffmpeg_runner": ffmpeg_runner,
        "ffprobe_runner": ffprobe_runner,
        "ffmpeg_executable": ffmpeg_executable,
        "ffprobe_executable": ffprobe_executable,
    }
    if timeout_seconds is not None:
        values["timeout_seconds"] = timeout_seconds
    if ffprobe_timeout_seconds is not None:
        values["ffprobe_timeout_seconds"] = ffprobe_timeout_seconds
    selection = await generate_music_candidates(generator, active, **values)
    return _with_lyrics(
        selection,
        requested if require_lyrics is None else bool(require_lyrics),
    )


def generate_song_candidates_sync(
    generator: Any,
    request: MusicRequest,
    **values: Any,
) -> SongCandidateSelection:
    """Run :func:`generate_song_candidates` from synchronous code."""
    return asyncio.run(generate_song_candidates(generator, request, **values))


def select_song_candidate(
    candidates: Any,
    config: SongCandidateConfig | Mapping[str, Any] | BaseModel | None = None,
    *,
    request: MusicRequest | None = None,
    lyrics: Any = None,
    require_lyrics: bool | None = None,
    analyzer: Callable[..., Any] | None = None,
    ffmpeg_runner: Callable[[Sequence[str]], Any] | None = None,
    ffprobe_runner: Callable[[Sequence[str]], Any] | None = None,
    ffmpeg_executable: str = "ffmpeg",
    ffprobe_executable: str = "ffprobe",
    timeout_seconds: float | None = None,
    ffprobe_timeout_seconds: float | None = None,
) -> SongCandidateSelection:
    """Analyse, prefilter and choose one song candidate, recording its index."""
    settings = song_candidate_config(config)
    values: dict[str, Any] = {
        "analyzer": analyzer,
        "ffmpeg_runner": ffmpeg_runner,
        "ffprobe_runner": ffprobe_runner,
        "ffmpeg_executable": ffmpeg_executable,
        "ffprobe_executable": ffprobe_executable,
    }
    if timeout_seconds is not None:
        values["timeout_seconds"] = timeout_seconds
    if ffprobe_timeout_seconds is not None:
        values["ffprobe_timeout_seconds"] = ffprobe_timeout_seconds
    selection = select_music_candidate(candidates, settings, **values)
    _, requested = _request_lyrics(request, lyrics)
    return _with_lyrics(
        selection,
        requested if require_lyrics is None else bool(require_lyrics),
    )


def prefilter_song_candidates(
    candidates: Any,
    config: SongCandidateConfig | Mapping[str, Any] | BaseModel | None = None,
    **values: Any,
) -> tuple[SongCandidate, ...]:
    """Return only the song candidates that pass every deterministic filter."""
    return select_song_candidate(candidates, config, **values).accepted


filter_song_candidates = prefilter_song_candidates
"""Alias for :func:`prefilter_song_candidates`."""

select_song_candidates = select_song_candidate
select_song_track = select_song_candidate
"""Aliases for :func:`select_song_candidate`."""

analyze_song_candidate = analyze_music_candidate
"""Alias for the shared audio analysis."""

score_song_candidate = score_music_candidate
"""Alias for the shared candidate score."""

parse_song_analysis = parse_music_analysis
"""Alias for the shared ffmpeg analysis parser."""


def song_manifest(
    selection: SongCandidateSelection,
    *,
    request: MusicRequest | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable record of the chosen song candidate."""
    chosen = selection.chosen
    return {
        "version": 1,
        "chosen_index": selection.chosen_index,
        "best_index": selection.best_index,
        "lyrics_requested": bool(request is not None and request.lyrics),
        "chosen": None if chosen is None else chosen.to_dict(),
        "candidates": [candidate.to_dict() for candidate in selection.candidates],
    }


def write_song_manifest(
    selection: SongCandidateSelection,
    path: str | Path,
    *,
    request: MusicRequest | None = None,
) -> Path:
    """Write :func:`song_manifest` as canonical JSON and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(song_manifest(selection, request=request), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target


class SongCandidateSelector(MusicCandidateSelector):
    """Reusable song analyser and prefilter with one set of thresholds."""

    def __init__(
        self,
        config: SongCandidateConfig | Mapping[str, Any] | BaseModel | None = None,
        *,
        require_lyrics: bool = False,
        **values: Any,
    ) -> None:
        super().__init__(song_candidate_config(config), **values)
        self._require_lyrics = bool(require_lyrics)

    def select(
        self,
        candidates: Any,
        *,
        request: MusicRequest | None = None,
        lyrics: Any = None,
        require_lyrics: bool | None = None,
    ) -> SongCandidateSelection:
        """Analyse, prefilter and choose one song candidate."""
        selection = super().select(candidates)
        _, requested = _request_lyrics(request, lyrics)
        required = (
            requested or self._require_lyrics
            if require_lyrics is None
            else bool(require_lyrics)
        )
        return _with_lyrics(selection, required)

    def prefilter(
        self,
        candidates: Any,
        *,
        request: MusicRequest | None = None,
        lyrics: Any = None,
        require_lyrics: bool | None = None,
    ) -> tuple[SongCandidate, ...]:
        """Return only the song candidates that pass every filter."""
        return self.select(
            candidates,
            request=request,
            lyrics=lyrics,
            require_lyrics=require_lyrics,
        ).accepted

    filter = prefilter

    async def generate(
        self,
        generator: Any,
        request: MusicRequest,
        *,
        lyrics: Any = None,
        count: int | None = None,
        require_lyrics: bool | None = None,
    ) -> SongCandidateSelection:
        """Generate song candidates with *generator* and select one."""
        active, requested = _vocal_request(request, lyrics=lyrics, count=count)
        selection = await super().generate(generator, active)
        required = (
            requested or self._require_lyrics
            if require_lyrics is None
            else bool(require_lyrics)
        )
        return _with_lyrics(selection, required)

    def generate_sync(
        self,
        generator: Any,
        request: MusicRequest,
        *,
        lyrics: Any = None,
        count: int | None = None,
        require_lyrics: bool | None = None,
    ) -> SongCandidateSelection:
        """Run :meth:`generate` from synchronous code."""
        return asyncio.run(
            self.generate(
                generator,
                request,
                lyrics=lyrics,
                count=count,
                require_lyrics=require_lyrics,
            )
        )


SongCandidateFilter = SongCandidateSelector
SongCandidatePrefilter = SongCandidateSelector
SongCandidateGenerator = SongCandidateSelector
"""Aliases for :class:`SongCandidateSelector`."""

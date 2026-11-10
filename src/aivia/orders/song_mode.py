"""Batch naming: one song per name, or a single song shared across every name.

Batch naming turns a list of recipient names into a personalized song for each
name. In ``per_name`` mode every name triggers its own song generation, so two
different names can receive different tracks. In ``shared`` mode a single song
is generated once and attached to every name, so every name hears the same
track. The two modes differ in how many times the generator is called and in
whether the chosen track is identical across names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from aivia.audio import generate_song_candidates
from aivia.core.errors import ConfigError
from aivia.providers.protocols.music import MusicRequest, MusicTrack

__all__ = [
    "NamedSong",
    "SongMode",
    "generate_batch_songs",
]

SongMode = Literal["per_name", "shared"]
"""The batch naming song mode.

``per_name`` generates one song per name. ``shared`` generates one song once
and attaches it to every name.
"""


@dataclass(frozen=True, slots=True)
class NamedSong:
    """One recipient name paired with the song chosen for it."""

    name: str
    """The recipient name."""
    chosen_index: int | None
    """Index of the chosen candidate, or ``None`` when none passed."""
    track: MusicTrack | None
    """The chosen track, or ``None`` when no candidate passed."""


def _resolve_names(names: list[str]) -> list[str]:
    resolved: list[str] = []
    for name in names:
        if not isinstance(name, str):
            raise TypeError("batch names must be strings")
        text = name.strip()
        if not text:
            raise ConfigError("batch names must not be empty")
        resolved.append(text)
    return resolved


async def _song_for_name(
    generator: Any,
    name: str,
    *,
    count: int,
    config: Any,
    ffmpeg_runner: Any,
    ffprobe_runner: Any,
    require_lyrics: bool,
) -> tuple[int | None, MusicTrack | None]:
    request = MusicRequest(
        prompt=f"A personalized song for {name}",
        lyrics=name,
        count=count,
    )
    selection = await generate_song_candidates(
        generator,
        request,
        config=config,
        count=count,
        ffmpeg_runner=ffmpeg_runner,
        ffprobe_runner=ffprobe_runner,
        require_lyrics=require_lyrics,
    )
    chosen = selection.chosen
    track = chosen.track if chosen is not None else None
    return selection.chosen_index, track


async def generate_batch_songs(
    names: list[str],
    generator: Any,
    *,
    mode: SongMode = "shared",
    count: int = 3,
    config: Any = None,
    ffmpeg_runner: Any = None,
    ffprobe_runner: Any = None,
    require_lyrics: bool = True,
) -> list[NamedSong]:
    """Generate a song for every name in *names*.

    ``per_name`` calls the generator once per name so each name receives its
    own chosen track. ``shared`` calls the generator once for the first name
    and attaches that same track to every name.
    """
    resolved = _resolve_names(names)
    if not resolved:
        raise ConfigError("batch names must not be empty")

    if mode not in {"per_name", "shared"}:
        raise ConfigError(
            f"unknown song mode: {mode!r} (expected 'per_name' or 'shared')"
        )

    results: list[NamedSong] = []
    if mode == "shared":
        index, track = await _song_for_name(
            generator,
            resolved[0],
            count=count,
            config=config,
            ffmpeg_runner=ffmpeg_runner,
            ffprobe_runner=ffprobe_runner,
            require_lyrics=require_lyrics,
        )
        for name in resolved:
            results.append(NamedSong(name=name, chosen_index=index, track=track))
    else:
        for name in resolved:
            index, track = await _song_for_name(
                generator,
                name,
                count=count,
                config=config,
                ffmpeg_runner=ffmpeg_runner,
                ffprobe_runner=ffprobe_runner,
                require_lyrics=require_lyrics,
            )
            results.append(NamedSong(name=name, chosen_index=index, track=track))
    return results

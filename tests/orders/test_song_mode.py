"""Batch naming: one song per name, or a single song shared across every name.

The two song modes differ in how many times the generator is called and in
whether every name receives the same track. ``per_name`` gives each name its
own chosen track; ``shared`` generates one song once and attaches it to all
names. These tests assert that difference.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from typing import Any

import pytest

from aivia.core.errors import ConfigError
from aivia.orders import NamedSong, SongMode, generate_batch_songs
from aivia.providers.protocols.audio import AudioFormat
from aivia.providers.protocols.music import MusicRequest, MusicResponse, MusicTrack

GENERATION_CONFIG: dict[str, Any] = {
    "song_candidates": {
        "target_duration_seconds": 10.0,
        "duration_tolerance_seconds": 5.0,
        "max_silence_ratio": 1.0,
        "max_clipping_ratio": 1.0,
        "min_loudness_lufs": -60.0,
        "max_loudness_lufs": -1.0,
    }
}


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _track(
    data: bytes,
    *,
    duration: float = 10.0,
    lyrics: str | None = "la la",
) -> MusicTrack:
    return MusicTrack(
        data=data,
        format=AudioFormat.MP3,
        duration_seconds=duration,
        lyrics=lyrics,
        title=data.decode("utf-8", "replace"),
    )


class NameSongGenerator:
    """Return one track whose title carries the request lyrics and record calls."""

    def __init__(self) -> None:
        self.requests: list[MusicRequest] = []

    async def generate(self, request: MusicRequest) -> MusicResponse:
        self.requests.append(request)
        lyrics = request.lyrics or "shared"
        return MusicResponse(
            tracks=[_track(f"song-for-{lyrics}".encode(), lyrics=lyrics)],
            model="fake-song",
            provider="fake",
        )


def probe_runner(duration: float = 10.0) -> Any:
    """Return ffprobe output reporting a fixed duration for any input."""

    def runner(argv: list[str]) -> subprocess.CompletedProcess[bytes]:
        payload = {
            "streams": [
                {
                    "index": 0,
                    "codec_type": "audio",
                    "codec_name": "mp3",
                    "duration": f"{duration:.3f}",
                    "sample_rate": "44100",
                    "channels": 1,
                }
            ],
            "format": {
                "format_name": "mp3",
                "format_long_name": "MP2/3 (MPEG audio layer 2/3)",
                "duration": f"{duration:.3f}",
                "size": "12345",
            },
        }
        return subprocess.CompletedProcess(
            list(argv), 0, stdout=json.dumps(payload).encode(), stderr=b""
        )

    return runner


def analysis_runner() -> Any:
    """Return ffmpeg output reporting clean, loud, non-clipped audio."""

    def runner(argv: list[str]) -> subprocess.CompletedProcess[bytes]:
        text = (
            "[ebur128] Summary: I: -14.000 LUFS\n"
            "[ebur128] True peak: Peak: -1.000 dBFS\n"
            "[astats] Channel: 1\n"
            "[astats] Peak level dB: -1.000\n"
            "[astats] Peak count: 0\n"
            "[astats] Number of samples: 44100\n"
        )
        return subprocess.CompletedProcess(
            list(argv), 0, stdout=b"", stderr=text.encode()
        )

    return runner


def _results(
    names: list[str],
    *,
    mode: SongMode = "shared",
    generator: NameSongGenerator | None = None,
) -> tuple[list[NamedSong], NameSongGenerator]:
    generator = generator or NameSongGenerator()
    results = _run(
        generate_batch_songs(
            names,
            generator,
            mode=mode,
            config=GENERATION_CONFIG,
            ffmpeg_runner=analysis_runner(),
            ffprobe_runner=probe_runner(),
        )
    )
    return results, generator


def test_per_name_generates_one_song_per_name() -> None:
    results, generator = _results(["Ava", "Ben"], mode="per_name")

    assert [song.name for song in results] == ["Ava", "Ben"]
    assert results[0].track is not None
    assert results[1].track is not None
    assert results[0].track.data == b"song-for-Ava"
    assert results[1].track.data == b"song-for-Ben"
    assert results[0].track.data != results[1].track.data
    assert results[0].chosen_index == 0
    assert results[1].chosen_index == 0
    # One generation per name.
    assert len(generator.requests) == 2
    assert generator.requests[0].lyrics == "Ava"
    assert generator.requests[1].lyrics == "Ben"


def test_shared_generates_one_song_for_all_names() -> None:
    results, generator = _results(["Ava", "Ben"], mode="shared")

    assert [song.name for song in results] == ["Ava", "Ben"]
    assert results[0].track is not None
    assert results[1].track is not None
    # Both names hear the same track (the first name's song).
    assert results[0].track.data == b"song-for-Ava"
    assert results[1].track.data == b"song-for-Ava"
    assert results[0].track == results[1].track
    assert results[0].chosen_index == results[1].chosen_index
    # A single generation serves every name.
    assert len(generator.requests) == 1
    assert generator.requests[0].lyrics == "Ava"


def test_per_name_and_shared_produce_a_measurable_difference() -> None:
    per_name, per_gen = _results(["Ava", "Ben"], mode="per_name")
    shared, shared_gen = _results(["Ava", "Ben"], mode="shared")

    # per_name calls the generator once per name; shared calls it once.
    assert len(per_gen.requests) == 2
    assert len(shared_gen.requests) == 1
    # per_name gives each name a distinct track; shared gives all names one.
    assert per_name[0].track is not None
    assert per_name[1].track is not None
    assert shared[0].track is not None
    assert shared[1].track is not None
    assert per_name[0].track.data != per_name[1].track.data
    assert shared[0].track.data == shared[1].track.data


def test_single_name_is_identical_in_both_modes() -> None:
    per_name, _ = _results(["Ava"], mode="per_name")
    shared, _ = _results(["Ava"], mode="shared")

    assert per_name[0].track is not None
    assert shared[0].track is not None
    assert per_name[0].track.data == shared[0].track.data == b"song-for-Ava"


def test_unknown_song_mode_is_rejected() -> None:
    generator = NameSongGenerator()
    with pytest.raises(ConfigError, match="unknown song mode"):
        _run(
            generate_batch_songs(
                ["Ava"],
                generator,
                mode="shuffle",  # type: ignore[arg-type]
                config=GENERATION_CONFIG,
                ffmpeg_runner=analysis_runner(),
                ffprobe_runner=probe_runner(),
            )
        )
    assert generator.requests == []


def test_empty_batch_is_rejected() -> None:
    generator = NameSongGenerator()
    with pytest.raises(ConfigError, match="batch names must not be empty"):
        _run(
            generate_batch_songs(
                [],
                generator,
                config=GENERATION_CONFIG,
                ffmpeg_runner=analysis_runner(),
                ffprobe_runner=probe_runner(),
            )
        )
    assert generator.requests == []


def test_empty_or_blank_names_are_rejected() -> None:
    generator = NameSongGenerator()
    for names in (["", "Ben"], ["Ava", "   "], ["\t\n"]):
        with pytest.raises(ConfigError, match="batch names must not be empty"):
            _run(
                generate_batch_songs(
                    names,
                    generator,
                    config=GENERATION_CONFIG,
                    ffmpeg_runner=analysis_runner(),
                    ffprobe_runner=probe_runner(),
                )
            )
    assert generator.requests == []


def test_non_string_names_are_rejected() -> None:
    generator = NameSongGenerator()
    with pytest.raises(TypeError, match="batch names must be strings"):
        _run(
            generate_batch_songs(
                ["Ava", 123],  # type: ignore[list-item]
                generator,
                config=GENERATION_CONFIG,
                ffmpeg_runner=analysis_runner(),
                ffprobe_runner=probe_runner(),
            )
        )
    assert generator.requests == []


def test_default_mode_is_shared() -> None:
    results, generator = _results(["Ava", "Ben"])  # mode defaults to shared

    assert results[0].track is not None
    assert results[1].track is not None
    assert results[0].track.data == results[1].track.data
    assert len(generator.requests) == 1

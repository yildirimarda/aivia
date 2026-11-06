"""Song candidates are generated, prefiltered and selected by index."""

from __future__ import annotations

import asyncio
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from aivia.audio import (
    MusicStage,
    SongCandidateAnalysis,
    SongCandidateConfig,
    SongCandidateSelector,
    generate_song_candidates,
    lyrics_text,
    prefilter_song_candidates,
    select_song_candidate,
    song_candidate_config,
    song_manifest,
    write_song_manifest,
)
from aivia.core.errors import ConfigError
from aivia.providers.protocols.audio import AudioFormat
from aivia.providers.protocols.music import (
    MusicRequest,
    MusicResponse,
    MusicTrack,
)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _source_path(argv: Sequence[str]) -> Path:
    if "-i" in argv:
        return Path(argv[list(argv).index("-i") + 1])
    return Path(argv[-1])


def _probe_json(duration: float) -> str:
    return json.dumps(
        {
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
    )


def probe_runner(
    durations: Mapping[bytes, float] | Sequence[float] | float,
) -> Any:
    if isinstance(durations, Mapping):
        remaining: list[float] = []
    elif isinstance(durations, (int, float)):
        remaining = [float(durations)]
    else:
        remaining = list(durations)

    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        data = _source_path(argv).read_bytes()
        if isinstance(durations, Mapping):
            duration = durations.get(data, 10.0)
        else:
            duration = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return subprocess.CompletedProcess(
            list(argv), 0, stdout=_probe_json(duration), stderr=""
        )

    return runner


def analysis_text(
    *,
    silence_seconds: float = 0.0,
    lufs: float = -14.0,
    true_peak_db: float = -1.0,
    peak_db: float = -1.0,
    clipped_samples: int = 0,
    total_samples: int = 44_100,
) -> str:
    lines: list[str] = []
    if silence_seconds > 0:
        lines.extend(
            [
                "[silencedetect] silence_start: 0.000",
                f"[silencedetect] silence_end: {silence_seconds:.3f} "
                f"| silence_duration: {silence_seconds:.3f}",
            ]
        )
    lines.extend(
        [
            f"[ebur128] Summary: I: {lufs:.3f} LUFS",
            f"[ebur128] True peak: Peak: {true_peak_db:.3f} dBFS",
            "[astats] Channel: 1",
            f"[astats] Peak level dB: {peak_db:.3f}",
            f"[astats] Peak count: {clipped_samples}",
            f"[astats] Number of samples: {total_samples}",
        ]
    )
    return "\n".join(lines)


def analysis_runner(
    metrics: Mapping[bytes, Mapping[str, Any]] | None = None,
) -> Any:
    values = dict(metrics or {})

    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
        data = _source_path(argv).read_bytes()
        text = analysis_text(**dict(values.get(data, {})))
        return subprocess.CompletedProcess(
            list(argv), 0, stdout=b"", stderr=text.encode()
        )

    return runner


def _analysis(track: MusicTrack, index: int = 0) -> SongCandidateAnalysis:
    duration = track.duration_seconds or 10.0
    silence = 8.0 if track.data == b"silent" else 0.0
    return SongCandidateAnalysis(
        index=index,
        duration_seconds=duration,
        silence_seconds=silence,
        silence_ratio=silence / duration,
        integrated_loudness_lufs=-14.0,
    )


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


class FakeSongGenerator:
    """Return the configured tracks and record every request."""

    def __init__(self, tracks: Sequence[MusicTrack]) -> None:
        self.tracks = list(tracks)
        self.requests: list[MusicRequest] = []

    async def generate(self, request: MusicRequest) -> MusicResponse:
        self.requests.append(request)
        return MusicResponse(
            tracks=self.tracks,
            model="fake-song",
            provider="fake",
        )


def test_generate_song_candidates_prefilters_and_records_the_chosen_index() -> None:
    generator = FakeSongGenerator(
        [_track(b"silent"), _track(b"good"), _track(b"also-good")]
    )

    selection = _run(
        generate_song_candidates(
            generator,
            MusicRequest(
                prompt="a dreamy song",
                count=3,
                duration_seconds=10.0,
                lyrics="la la la",
            ),
            config={"song_candidates": {"duration_tolerance_seconds": 1.0}},
            ffmpeg_runner=analysis_runner({b"silent": {"silence_seconds": 8.0}}),
            ffprobe_runner=probe_runner(10.0),
        )
    )

    assert generator.requests[0].count == 3
    assert generator.requests[0].lyrics == "la la la"
    assert generator.requests[0].instrumental is False
    assert selection.chosen_index == 1
    assert selection.index == 1
    assert selection.track is not None
    assert selection.track.data == b"good"
    assert [candidate.index for candidate in selection.accepted] == [1, 2]
    assert [candidate.index for candidate in selection.rejected] == [0]
    assert "silence" in selection.rejected[0].reason


def test_a_candidate_without_lyrics_is_rejected_when_lyrics_are_requested() -> None:
    candidates = [
        _track(b"no-lyrics", lyrics=None),
        _track(b"with-lyrics", lyrics="la la"),
    ]

    selection = select_song_candidate(
        candidates,
        {"target_duration_seconds": 10.0, "duration_tolerance_seconds": 1.0},
        lyrics="la la",
        analyzer=_analysis,
    )

    assert selection.chosen_index == 1
    assert selection.rejected[0].index == 0
    assert "missing lyrics" in selection.rejected[0].reason


def test_a_vocal_song_request_must_not_be_instrumental() -> None:
    generator = FakeSongGenerator([_track(b"good")])

    with pytest.raises(ConfigError, match="instrumental"):
        _run(
            generate_song_candidates(
                generator,
                MusicRequest(prompt="a song", instrumental=True),
                lyrics="la la",
            )
        )

    assert generator.requests == []


def test_song_lyrics_are_read_from_a_payload_or_script() -> None:
    payload = {
        "sections": [
            {"name": "verse", "lines": ["first line", "second line"]},
            {"name": "chorus", "lines": ["hook line"]},
        ]
    }

    assert lyrics_text(payload) == "first line\nsecond line\nhook line"
    assert lyrics_text("  plain lyrics  ") == "plain lyrics"


def test_song_manifest_records_the_chosen_index(tmp_path: Path) -> None:
    candidates = [_track(b"silent"), _track(b"good")]
    selection = select_song_candidate(
        candidates,
        {"target_duration_seconds": 10.0, "duration_tolerance_seconds": 1.0},
        request=MusicRequest(prompt="a song", lyrics="la la"),
        analyzer=_analysis,
    )

    request = MusicRequest(prompt="a song", lyrics="la la")
    path = write_song_manifest(selection, tmp_path / "songs.json", request=request)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert song_manifest(selection)["chosen_index"] == 1
    assert payload["chosen_index"] == 1
    assert payload["chosen"]["index"] == 1
    assert payload["lyrics_requested"] is True
    assert [item["index"] for item in payload["candidates"]] == [0, 1]


def test_song_candidate_config_reads_the_song_section() -> None:
    config = song_candidate_config(
        {
            "song_candidates": {
                "target_duration": 12.0,
                "duration_tolerance": 1.0,
                "max_silence": 0.1,
            }
        }
    )

    assert isinstance(config, SongCandidateConfig)
    assert config.target_duration_seconds == pytest.approx(12.0)
    assert config.max_silence_ratio == pytest.approx(0.1)
    assert config.duration_bounds == pytest.approx((11.0, 13.0))


def test_prefilter_returns_only_accepted_song_candidates() -> None:
    candidates = [_track(b"silent"), _track(b"good")]

    accepted = prefilter_song_candidates(
        candidates,
        {"target_duration_seconds": 10.0, "duration_tolerance_seconds": 1.0},
        analyzer=_analysis,
    )

    assert [candidate.index for candidate in accepted] == [1]


def test_a_song_selector_reuses_its_thresholds() -> None:
    selector = SongCandidateSelector(
        {"target_duration_seconds": 10.0, "duration_tolerance_seconds": 1.0},
        analyzer=_analysis,
        require_lyrics=True,
    )

    selection = selector.select([_track(b"silent"), _track(b"good")])

    assert selector.config.target_duration_seconds == 10.0
    assert selection.chosen_index == 1
    assert selector.prefilter([_track(b"good")])[0].index == 0


def test_the_music_stage_records_the_chosen_song_candidate(
    tmp_path: Path,
) -> None:
    generator = FakeSongGenerator(
        [
            _track(b"silent", duration=5.0),
            _track(b"good", duration=5.0),
            _track(b"also-good", duration=5.0),
        ]
    )
    stage = MusicStage(
        options={
            "source": "provider",
            "duration_seconds": 5.0,
            "count": 3,
            "lyrics": "la la la",
            "instrumental": False,
            "candidate_config": {
                "target_duration_seconds": 5.0,
                "duration_tolerance_seconds": 1.0,
            },
            "analyzer": _analysis,
        },
        music=generator,
    )

    result = _run(stage.run(_context(tmp_path)))

    assert generator.requests[0].count == 3
    assert generator.requests[0].lyrics == "la la la"
    assert result.metadata["chosen_index"] == 1
    assert result.metadata["candidate_count"] == 3
    assert (tmp_path / "music" / "music.mp3").read_bytes() == b"good"

    manifest = json.loads((tmp_path / "music.json").read_text(encoding="utf-8"))
    assert manifest["chosen_index"] == 1
    assert manifest["candidate_count"] == 3
    assert [item["index"] for item in manifest["candidates"]] == [0, 1, 2]
    assert manifest["selected"]["source"] == str(tmp_path / "music" / "music.mp3")


def _context(run_dir: Path) -> Any:
    from aivia.core.stages import StageContext

    return StageContext(run_dir=run_dir, config={})

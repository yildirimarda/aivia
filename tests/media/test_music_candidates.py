"""Music candidates are generated, filtered deterministically and indexed."""

from __future__ import annotations

import asyncio
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from aivia.core.errors import ValidationError
from aivia.media.music_candidates import (
    MusicCandidateAnalysis,
    MusicCandidateConfig,
    MusicCandidateError,
    MusicCandidateFilter,
    MusicCandidateGenerator,
    MusicCandidatePrefilter,
    MusicCandidateSelection,
    MusicCandidateSelector,
    analyze_music_candidate,
    filter_music_candidates,
    generate_music_candidates,
    music_candidate_config,
    parse_music_analysis,
    prefilter_music_candidates,
    score_music_candidate,
    select_music_candidate,
    select_music_candidates,
    select_music_track,
)
from aivia.providers.protocols import (
    AudioFormat,
    MusicRequest,
    MusicResponse,
    MusicTrack,
)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


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


def _source_path(argv: Sequence[str]) -> Path:
    if "-i" in argv:
        return Path(argv[list(argv).index("-i") + 1])
    return Path(argv[-1])


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
        path = _source_path(argv)
        data = path.read_bytes()
        if isinstance(durations, Mapping):
            duration = durations.get(data, 10.0)
        else:
            duration = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return subprocess.CompletedProcess(
            list(argv),
            0,
            stdout=_probe_json(duration),
            stderr="",
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
    *,
    calls: list[list[str]] | None = None,
) -> Any:
    values = dict(metrics or {})

    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
        if calls is not None:
            calls.append(list(argv))
        path = _source_path(argv)
        text = analysis_text(**dict(values.get(path.read_bytes(), {})))
        return subprocess.CompletedProcess(
            list(argv),
            0,
            stdout=b"",
            stderr=text.encode(),
        )

    return runner


def _track(data: bytes, *, duration: float = 10.0) -> MusicTrack:
    return MusicTrack(
        data=data,
        format=AudioFormat.MP3,
        duration_seconds=duration,
        seed=int.from_bytes(data[:1], "big"),
    )


def _metrics(
    duration: float,
    *,
    silence: float = 0.0,
    lufs: float | None = -14.0,
    clipping: float = 0.0,
    true_peak: float = -1.0,
) -> MusicCandidateAnalysis:
    return MusicCandidateAnalysis(
        duration_seconds=duration,
        silence_seconds=silence,
        silence_ratio=silence / duration if duration else 0.0,
        clipping_ratio=clipping,
        integrated_loudness_lufs=lufs,
        true_peak_db=true_peak,
    )


class FakeMusicProvider:
    """Return one track per requested candidate with deterministic bytes."""

    def __init__(self, duration: float = 10.0) -> None:
        self.duration = duration
        self.requests: list[MusicRequest] = []

    async def generate(self, request: MusicRequest) -> MusicResponse:
        self.requests.append(request)
        tracks = [
            _track(f"track-{index}".encode(), duration=self.duration)
            for index in range(request.count)
        ]
        return MusicResponse(
            tracks=tracks,
            model="fake-music",
            provider="fake",
            metadata={"request_count": request.count},
        )


def test_analysis_parses_duration_silence_clipping_and_loudness() -> None:
    data = b"candidate-audio"
    text = analysis_text(
        silence_seconds=2.0,
        lufs=-14.5,
        true_peak_db=-0.2,
        peak_db=0.0,
        clipped_samples=100,
        total_samples=1_000,
    )
    analysis = analyze_music_candidate(
        data,
        index=2,
        duration_seconds=10.0,
        format=AudioFormat.MP3,
        ffmpeg_runner=lambda argv: subprocess.CompletedProcess(
            list(argv), 0, stdout=b"", stderr=text.encode()
        ),
    )

    assert analysis.index == 2
    assert analysis.duration_seconds == pytest.approx(10.0)
    assert analysis.silence_seconds == pytest.approx(2.0)
    assert analysis.silence_ratio == pytest.approx(0.2)
    assert analysis.clipping_ratio == pytest.approx(0.1)
    assert analysis.clipped_samples == 100
    assert analysis.integrated_loudness_lufs == pytest.approx(-14.5)
    assert analysis.lufs == pytest.approx(-14.5)
    assert analysis.true_peak_db == pytest.approx(-0.2)
    assert analysis.peak_db == pytest.approx(0.0)
    assert analysis.is_silent is True
    assert analysis.is_clipped is True


def test_an_unclosed_silence_stretch_runs_to_the_end_of_the_track() -> None:
    text = "\n".join(
        (
            "silence_start: 1.000",
            "silence_end: 2.000 | silence_duration: 1.000",
            "silence_start: 8.000",
            "I: -14.000 LUFS",
        )
    )

    analysis = parse_music_analysis(text, 10.0, index=1)

    assert analysis.silence_intervals == ((1.0, 2.0), (8.0, 10.0))
    assert analysis.silence_seconds == pytest.approx(3.0)
    assert analysis.silence_ratio == pytest.approx(0.3)


def test_the_first_candidate_that_passes_every_filter_is_chosen() -> None:
    provider = FakeMusicProvider()
    metrics = {
        b"track-0": {"silence_seconds": 8.0},
        b"track-1": {},
        b"track-2": {},
    }

    selection = _run(
        generate_music_candidates(
            provider,
            MusicRequest(prompt="calm loop", count=3, duration_seconds=10.0),
            config=MusicCandidateConfig(duration_tolerance_seconds=1.0),
            ffmpeg_runner=analysis_runner(metrics),
            ffprobe_runner=probe_runner(10.0),
        )
    )

    assert provider.requests[0].count == 3
    assert isinstance(selection, MusicCandidateSelection)
    assert selection.chosen_index == 1
    assert selection.index == 1
    assert selection.track is not None
    assert selection.track.data == b"track-1"
    assert [item.index for item in selection.accepted] == [1, 2]
    assert [item.index for item in selection.rejected] == [0]
    assert "silence" in selection.rejected[0].reason
    assert selection.response is not None
    assert selection.response.model == "fake-music"


def test_quality_preference_picks_the_best_accepted_candidate() -> None:
    candidates = [
        _metrics(10.0, lufs=-25.0),
        _metrics(10.0, lufs=-14.0),
        _metrics(10.0, lufs=-9.0),
    ]

    selection = select_music_candidate(
        candidates,
        MusicCandidateConfig(prefer="quality"),
    )

    assert selection.chosen_index == 1
    assert selection.best_index == 1
    assert score_music_candidate(candidates[1]) > score_music_candidate(candidates[0])


@pytest.mark.parametrize(
    ("analysis", "reason"),
    [
        (_metrics(2.0), "duration too short"),
        (_metrics(20.0), "duration too long"),
        (_metrics(10.0, silence=5.0), "too much silence"),
        (_metrics(10.0, clipping=0.2), "clipping"),
        (_metrics(10.0, true_peak=0.0), "clipping"),
        (_metrics(10.0, lufs=-50.0), "too quiet"),
        (_metrics(10.0, lufs=-3.0), "too loud"),
        (_metrics(10.0, lufs=None), "loudness unavailable"),
    ],
)
def test_each_filter_rejects_an_unusable_candidate(
    analysis: MusicCandidateAnalysis,
    reason: str,
) -> None:
    config = MusicCandidateConfig(
        target_duration_seconds=10.0,
        duration_tolerance_seconds=1.0,
    )

    selection = select_music_candidate([analysis], config)

    assert selection.failed is True
    assert selection.chosen_index is None
    assert selection.rejected[0].rejected is True
    assert reason in selection.rejected[0].reason


def test_prefilter_returns_only_accepted_candidates_and_aliases_agree() -> None:
    candidates = [
        _metrics(10.0),
        _metrics(2.0),
        _metrics(10.0, silence=9.0),
    ]

    config = MusicCandidateConfig(
        target_duration_seconds=10.0,
        duration_tolerance_seconds=1.0,
    )
    accepted = prefilter_music_candidates(candidates, config)
    filtered = filter_music_candidates(candidates, config)

    assert [item.index for item in accepted] == [0]
    assert [item.index for item in filtered] == [0]
    assert select_music_candidates is select_music_candidate
    assert select_music_track is select_music_candidate
    assert MusicCandidateFilter is MusicCandidateSelector
    assert MusicCandidatePrefilter is MusicCandidateSelector
    assert MusicCandidateGenerator is MusicCandidateSelector


def test_no_passing_candidate_is_recorded_and_require_raises() -> None:
    selection = select_music_candidate(
        [_metrics(1.0), _metrics(2.0)],
        MusicCandidateConfig(target_duration_seconds=10.0),
    )

    assert selection.ok is False
    assert selection.failed is True
    assert selection.chosen is None
    assert selection.track is None
    assert selection.analysis is None
    assert selection.reason == "duration too short"

    with pytest.raises(MusicCandidateError, match="prefilter") as error:
        selection.require()
    assert error.value.details["candidates"]


def test_a_remote_track_is_rejected_before_analysis() -> None:
    track = MusicTrack(url="https://cdn.example.com/song.mp3")

    with pytest.raises(ValidationError, match="inline audio data"):
        analyze_music_candidate(track)


def test_a_selector_reuses_its_thresholds_and_runner() -> None:
    selector = MusicCandidateSelector(
        MusicCandidateConfig(target_duration_seconds=10.0),
        ffmpeg_runner=analysis_runner({b"ok": {}}),
        ffprobe_runner=probe_runner({b"ok": 10.0, b"bad": 1.0}),
    )

    selection = selector.select([_track(b"bad"), _track(b"ok")])
    analyzed = selector.analyze(b"ok", duration_seconds=10.0)

    assert selector.config.target_duration_seconds == 10.0
    assert selector.thresholds is selector.config
    assert selection.chosen_index == 1
    assert selector.prefilter([_track(b"ok")])[0].index == 0
    assert analyzed.duration_seconds == pytest.approx(10.0)


def test_config_aliases_and_invalid_thresholds() -> None:
    config = music_candidate_config(
        {
            "music_candidates": {
                "target_duration": 12.0,
                "duration_tolerance": 1.5,
                "max_silence": 0.1,
                "min_lufs": -20.0,
            }
        }
    )

    assert config.target_duration_seconds == pytest.approx(12.0)
    assert config.duration_tolerance_seconds == pytest.approx(1.5)
    assert config.max_silence_ratio == pytest.approx(0.1)
    assert config.min_loudness_lufs == pytest.approx(-20.0)
    assert config.duration_bounds == pytest.approx((10.5, 13.5))
    assert music_candidate_config(None, max_lufs=-8.0).max_loudness_lufs == -8.0

    with pytest.raises(ValueError, match="min_duration_seconds"):
        MusicCandidateConfig(min_duration_seconds=5.0, max_duration_seconds=2.0)
    with pytest.raises(ValueError, match="min_loudness_lufs"):
        MusicCandidateConfig(min_loudness_lufs=-5.0, max_loudness_lufs=-10.0)
    with pytest.raises(TypeError, match="number"):
        MusicCandidateConfig(max_silence_ratio="lots")  # type: ignore[arg-type]


def test_a_failing_ffmpeg_analysis_raises_a_validation_error() -> None:
    def failing(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            list(argv), 1, stdout="", stderr="decoder failed"
        )

    with pytest.raises(ValidationError, match="analyse"):
        analyze_music_candidate(
            b"broken",
            duration_seconds=10.0,
            ffmpeg_runner=failing,
        )

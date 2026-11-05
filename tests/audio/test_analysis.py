"""The audio analysis utility reports tempo, silence and loudness."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from aivia.audio import (
    AudioAnalysis,
    AudioAnalysisReport,
    AudioAnalyzer,
    SilenceMap,
    SilenceSpan,
    analyze_audio,
    parse_audio_analysis,
)

AUDIO = b"ID3" + b"\x00" * 64


def probe_json(duration: float) -> str:
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
    duration: float,
    *,
    calls: list[list[str]] | None = None,
) -> Callable[[Sequence[str]], subprocess.CompletedProcess[str]]:
    def run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if calls is not None:
            calls.append(list(argv))
        return subprocess.CompletedProcess(
            list(argv),
            0,
            stdout=probe_json(duration),
            stderr="",
        )

    return run


def analysis_text(
    *,
    silence: Sequence[tuple[float, float]] = (),
    lufs: float = -14.0,
    true_peak: float = -1.0,
    mean_volume: float = -20.0,
    tempo: float | None = None,
) -> str:
    lines: list[str] = []
    for start, end in silence:
        lines.append(f"[silencedetect @ 0x1] silence_start: {start}")
        lines.append(
            f"[silencedetect @ 0x1] silence_end: {end} "
            f"| silence_duration: {end - start}"
        )
    lines.extend(
        (
            "[Parsed_ebur128_0 @ 0x1] Summary:",
            "  Integrated loudness:",
            f"    I: {lufs:.1f} LUFS",
            f"    Peak: {true_peak:.1f} dBFS",
            "[Parsed_astats_1 @ 0x1] Channel: 1",
            f"Peak level dB: {true_peak:.1f}",
            f"max_volume: {true_peak:.1f} dB",
            f"mean_volume: {mean_volume:.1f} dB",
        )
    )
    if tempo is not None:
        lines.append(f"[mp3 @ 0x1] tempo: {tempo:.1f} bpm")
    return "\n".join(lines)


def audio_runner(
    text: str,
    *,
    calls: list[list[str]] | None = None,
) -> Callable[[Sequence[str]], subprocess.CompletedProcess[str]]:
    def run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if calls is not None:
            calls.append(list(argv))
        return subprocess.CompletedProcess(list(argv), 0, stdout=text, stderr="")

    return run


def source_file(tmp_path: Path, name: str = "source.mp3") -> Path:
    path = tmp_path / name
    path.write_bytes(AUDIO)
    return path


def test_parse_reports_tempo_silence_map_and_loudness() -> None:
    text = analysis_text(
        silence=((1.0, 2.5), (7.0, 8.0)),
        lufs=-13.5,
        true_peak=-0.8,
        mean_volume=-19.0,
        tempo=128.0,
    )

    analysis = parse_audio_analysis(text, 10.0)

    assert isinstance(analysis, AudioAnalysis)
    assert isinstance(analysis, AudioAnalysisReport)
    assert analysis.duration_seconds == 10.0
    assert analysis.tempo_bpm == 128.0
    assert analysis.tempo == 128.0
    assert analysis.silence_seconds == pytest.approx(2.5)
    assert analysis.silence_ratio == pytest.approx(0.25)
    assert analysis.silence_intervals == ((1.0, 2.5), (7.0, 8.0))
    assert analysis.integrated_loudness_lufs == -13.5
    assert analysis.loudness_lufs == -13.5
    assert analysis.true_peak_db == -0.8
    assert analysis.peak == -0.8
    assert analysis.mean_volume_db == -19.0
    assert analysis.metadata["tempo_source"] == "reported"


def test_the_silence_map_exposes_spans_totals_and_lookups() -> None:
    text = analysis_text(silence=((1.0, 2.5), (7.0, 8.0)), tempo=128.0)

    silence_map = parse_audio_analysis(text, 10.0).silence_map

    assert isinstance(silence_map, SilenceMap)
    assert silence_map.spans == (SilenceSpan(1.0, 2.5), SilenceSpan(7.0, 8.0))
    assert silence_map.intervals == ((1.0, 2.5), (7.0, 8.0))
    assert silence_map.total_seconds == pytest.approx(2.5)
    assert silence_map.seconds == pytest.approx(2.5)
    assert silence_map.ratio == pytest.approx(0.25)
    assert silence_map.is_empty is False
    assert silence_map.has_silence is True
    assert silence_map.contains(1.5) is True
    assert silence_map.contains(2.5) is False
    assert silence_map.to_list() == [[1.0, 2.5], [7.0, 8.0]]
    assert len(silence_map) == 2
    assert [span.start_seconds for span in silence_map] == [1.0, 7.0]


def test_the_silence_map_sorts_and_clamps_intervals() -> None:
    silence_map = SilenceMap.from_intervals(
        ((7.0, 12.0), (1.0, 2.0), (9.0, 9.5)),
        duration_seconds=10.0,
    )

    assert silence_map.intervals == ((1.0, 2.0), (7.0, 10.0), (9.0, 9.5))
    assert silence_map.total_seconds == pytest.approx(4.5)
    assert silence_map.ratio == pytest.approx(0.45)


def test_analyze_audio_uses_the_media_analysis_and_keeps_the_source(
    tmp_path: Path,
) -> None:
    source = source_file(tmp_path)
    before = source.read_bytes()
    probe_calls: list[list[str]] = []
    calls: list[list[str]] = []
    text = analysis_text(silence=((0.0, 1.0),), tempo=96.0)

    analysis = analyze_audio(
        source,
        ffprobe_runner=probe_runner(12.5, calls=probe_calls),
        ffmpeg_runner=audio_runner(text, calls=calls),
    )

    assert analysis.duration_seconds == 12.5
    assert analysis.tempo_bpm == 96.0
    assert analysis.silence_map.intervals == ((0.0, 1.0),)
    assert analysis.integrated_loudness_lufs == -14.0
    assert source.read_bytes() == before
    assert len(probe_calls) == 1
    assert len(calls) == 1
    filters = calls[0][calls[0].index("-af") + 1]
    assert "silencedetect=" in filters
    assert "ebur128=peak=true" in filters
    assert "astats=metadata=0:reset=0" in filters


def test_the_analyzer_facade_binds_defaults_and_accepts_overrides(
    tmp_path: Path,
) -> None:
    source = source_file(tmp_path)
    calls: list[list[str]] = []
    analyzer = AudioAnalyzer(
        silence_noise_db=-40.0,
        silence_min_seconds=0.25,
        min_bpm=60.0,
        max_bpm=180.0,
    )

    analysis = analyzer.analyze(
        source,
        duration_seconds=4.0,
        ffmpeg_runner=audio_runner(analysis_text(tempo=90.0), calls=calls),
    )

    assert analysis.tempo_bpm == 90.0
    filters = calls[0][calls[0].index("-af") + 1]
    assert "noise=-40dB" in filters
    assert "d=0.25" in filters

    analyzer.analyze(
        source,
        duration_seconds=4.0,
        silence_noise_db=-30.0,
        ffmpeg_runner=audio_runner(analysis_text(), calls=calls),
    )
    filters = calls[1][calls[1].index("-af") + 1]
    assert "noise=-30dB" in filters


def test_invalid_values_are_rejected() -> None:
    with pytest.raises(ValueError):
        AudioAnalysis(tempo_bpm=-1.0)
    with pytest.raises(ValueError):
        AudioAnalysis(duration_seconds=-1.0)
    with pytest.raises(ValueError):
        AudioAnalysis(silence_ratio=1.5)
    with pytest.raises(ValueError):
        AudioAnalyzer(min_bpm=180.0, max_bpm=60.0)
    with pytest.raises(ValueError):
        SilenceMap(spans=(SilenceSpan(2.0, 1.0),))

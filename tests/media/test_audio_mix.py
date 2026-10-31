"""Audio analysis and multi-track mixing with verified voice ducking."""

from __future__ import annotations

import array
import json
import math
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from aivia.core.errors import ValidationError
from aivia.media.audio_mix import (
    AudioAnalysis,
    AudioMixSettings,
    AudioTrack,
    DuckingSettings,
    analyze_audio,
    build_mix_filtergraph,
    estimate_tempo,
    estimate_tempo_from_envelope,
    mix_audio,
    parse_audio_analysis,
    verify_ducking,
)

AUDIO = b"ID3" + b"\x00" * 32
MP3 = b"ID3" + b"\x00" * 64


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
        length = end - start
        lines.extend(
            [
                f"[silencedetect] silence_start: {start:.3f}",
                f"[silencedetect] silence_end: {end:.3f} "
                f"| silence_duration: {length:.3f}",
            ]
        )
    lines.extend(
        [
            f"[ebur128] Summary: I: {lufs:.3f} LUFS",
            f"[ebur128] True peak: Peak: {true_peak:.3f} dBFS",
            "[astats] Channel: 1",
            f"[astats] Peak level dB: {true_peak:.3f}",
            f"[astats] max_volume: {true_peak:.3f} dB",
            f"[astats] mean_volume: {mean_volume:.3f} dB",
        ]
    )
    if tempo is not None:
        lines.append(f"tempo: {tempo:.3f} bpm")
    return "\n".join(lines)


def audio_runner(
    text: str,
    *,
    pcm: bytes = b"",
    calls: list[list[str]] | None = None,
) -> Callable[[Sequence[str]], subprocess.CompletedProcess[bytes]]:
    def run(argv: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
        if calls is not None:
            calls.append(list(argv))
        if "s16le" in argv:
            return subprocess.CompletedProcess(list(argv), 0, stdout=pcm, stderr=b"")
        return subprocess.CompletedProcess(
            list(argv),
            0,
            stdout=b"",
            stderr=text.encode(),
        )

    return run


def mix_runner(
    *,
    payload: bytes = MP3,
    calls: list[list[str]] | None = None,
) -> Callable[[Sequence[str]], subprocess.CompletedProcess[bytes]]:
    def run(argv: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
        if calls is not None:
            calls.append(list(argv))
        Path(argv[-1]).write_bytes(payload)
        return subprocess.CompletedProcess(list(argv), 0, stdout=b"", stderr=b"")

    return run


def pcm_click_track(
    bpm: float,
    *,
    seconds: float = 6.0,
    sample_rate: int = 8_000,
) -> bytes:
    total = int(seconds * sample_rate)
    samples = array.array("h", [0]) * total
    interval = 60.0 / bpm
    click_length = int(0.04 * sample_rate)
    time = 0.0
    while time < seconds:
        start = int(time * sample_rate)
        for offset in range(click_length):
            index = start + offset
            if index >= total:
                break
            samples[index] = 12_000
        time += interval
    return samples.tobytes()


def source_file(tmp_path: Path, name: str = "source.mp3") -> Path:
    path = tmp_path / name
    path.write_bytes(AUDIO)
    return path


def option(argv: Sequence[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_parse_analysis_reports_tempo_silence_and_loudness() -> None:
    text = analysis_text(
        silence=((1.0, 2.5), (7.0, 8.0)),
        lufs=-13.5,
        true_peak=-0.8,
        mean_volume=-19.0,
        tempo=128.0,
    )
    analysis = parse_audio_analysis(text, 10.0)

    assert isinstance(analysis, AudioAnalysis)
    assert analysis.duration_seconds == 10.0
    assert analysis.tempo_bpm == 128.0
    assert analysis.tempo == 128.0
    assert analysis.silence_seconds == pytest.approx(2.5)
    assert analysis.silence_ratio == pytest.approx(0.25)
    assert analysis.silence_intervals == ((1.0, 2.5), (7.0, 8.0))
    assert analysis.integrated_loudness_lufs == -13.5
    assert analysis.loudness_lufs == -13.5
    assert analysis.true_peak_db == -0.8
    assert analysis.mean_volume_db == -19.0
    assert analysis.metadata["tempo_source"] == "reported"
    assert analysis.to_dict()["tempo_bpm"] == 128.0


def test_parse_analysis_estimates_tempo_from_the_loudness_envelope() -> None:
    frames = ["[ebur128] t: 0.100 TARGET:-23 LUFS M: 0.000"]
    for index in range(1, 100):
        level = 1.0 if index % 5 == 0 else 0.0
        frames.append(f"[ebur128] t: {index / 10:.3f} TARGET:-23 LUFS M: {level:.3f}")
    analysis = parse_audio_analysis("\n".join(frames), 10.0)

    assert analysis.tempo_bpm is not None
    assert analysis.tempo_bpm == pytest.approx(120.0, abs=2.0)
    assert analysis.metadata["tempo_source"] == "estimated"


def test_estimate_tempo_from_envelope_and_pcm() -> None:
    envelope = [0.0] * 100
    for index in range(0, 100, 5):
        envelope[index] = 1.0
    assert estimate_tempo_from_envelope(envelope, frames_per_second=10.0) == 120.0

    pcm = pcm_click_track(120.0)
    assert estimate_tempo(pcm) == pytest.approx(120.0, abs=2.0)
    assert estimate_tempo(b"") is None


def test_analyze_audio_uses_ffmpeg_filters_and_keeps_the_source(
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
    assert analysis.silence_seconds == 1.0
    assert analysis.integrated_loudness_lufs == -14.0
    assert source.read_bytes() == before
    assert len(probe_calls) == 1
    assert len(calls) == 1
    filters = option(calls[0], "-af")
    assert "silencedetect=" in filters
    assert "ebur128=peak=true" in filters
    assert "astats=metadata=0:reset=0" in filters
    assert "s16le" not in calls[0]


def test_analyze_audio_decodes_pcm_when_no_tempo_is_reported(
    tmp_path: Path,
) -> None:
    source = source_file(tmp_path)
    calls: list[list[str]] = []
    analysis = analyze_audio(
        source,
        duration_seconds=6.0,
        ffmpeg_runner=audio_runner(
            analysis_text(),
            pcm=pcm_click_track(120.0),
            calls=calls,
        ),
    )

    assert analysis.tempo_bpm == pytest.approx(120.0, abs=2.0)
    assert analysis.metadata["tempo_source"] == "decoded"
    assert len(calls) == 2
    assert "s16le" in calls[1]
    assert option(calls[1], "-ar") == "8000"


def test_analyze_audio_honours_an_explicit_tempo_without_decoding(
    tmp_path: Path,
) -> None:
    source = source_file(tmp_path)
    calls: list[list[str]] = []
    analysis = analyze_audio(
        source,
        duration_seconds=4.0,
        tempo_bpm=90.0,
        ffmpeg_runner=audio_runner(analysis_text(), calls=calls),
    )

    assert analysis.tempo_bpm == 90.0
    assert analysis.metadata["tempo_source"] == "provided"
    assert len(calls) == 1


def test_build_graph_applies_gain_offsets_fades_and_sidechain_ducking() -> None:
    tracks = [
        AudioTrack(source="voice.mp3", is_voice=True),
        AudioTrack(
            source="music.mp3",
            volume_db=-6.0,
            offset_seconds=0.5,
            fade_in_seconds=1.0,
            fade_out_seconds=2.0,
            duration_seconds=10.0,
        ),
    ]
    graph = build_mix_filtergraph(
        tracks,
        AudioMixSettings(duration_seconds=12.0, normalize_loudness=True),
    )

    assert "volume=-6.000000dB" in graph
    assert "adelay=500:all=1" in graph
    assert "afade=t=in:st=0:d=1.000000" in graph
    assert "afade=t=out:st=8.000000:d=2.000000" in graph
    assert "sidechaincompress=" in graph
    assert "asplit=2[voice_sc][voice_out]" in graph
    assert "amix=inputs=2" in graph
    assert "loudnorm=I=-14.000000:TP=-1.500000:LRA=11.000000" in graph
    assert graph.endswith("[out]")

    report = verify_ducking(graph, voice_tracks=1, background_tracks=1)
    assert report.applied is True
    assert report.verified is True
    assert report.ratio == 8.0
    report.require()


def test_verify_ducking_rejects_a_graph_without_a_sidechain() -> None:
    graph = build_mix_filtergraph(
        [AudioTrack(source="voice.mp3", is_voice=True)],
        AudioMixSettings(),
    )
    report = verify_ducking(graph, voice_tracks=1, background_tracks=1)

    assert report.applied is False
    assert report.verified is False
    with pytest.raises(ValidationError):
        report.require()


def test_disabled_ducking_and_a_voice_only_mix_need_no_sidechain() -> None:
    voice = AudioTrack(source="voice.mp3", is_voice=True)
    music = AudioTrack(source="music.mp3")
    disabled = build_mix_filtergraph(
        [voice, music],
        AudioMixSettings(ducking=DuckingSettings(enabled=False)),
    )
    assert "sidechaincompress" not in disabled
    assert verify_ducking(
        disabled,
        DuckingSettings(enabled=False),
        voice_tracks=1,
        background_tracks=1,
    ).verified

    only_voice = build_mix_filtergraph([voice], AudioMixSettings())
    assert "sidechaincompress" not in only_voice
    assert verify_ducking(
        only_voice,
        voice_tracks=1,
        background_tracks=0,
    ).verified


def test_mix_audio_writes_a_new_file_and_preserves_the_inputs(
    tmp_path: Path,
) -> None:
    voice = source_file(tmp_path, "voice.mp3")
    music = source_file(tmp_path, "music.mp3")
    before_voice = voice.read_bytes()
    before_music = music.read_bytes()
    destination = tmp_path / "mix.mp3"
    calls: list[list[str]] = []

    result = mix_audio(
        [
            AudioTrack(source=voice, is_voice=True),
            AudioTrack(source=music, volume_db=-3.0),
        ],
        destination,
        duration_seconds=5.0,
        normalize_loudness=True,
        ffmpeg_runner=mix_runner(calls=calls),
    )

    assert destination.is_file()
    assert destination.read_bytes() == MP3
    assert voice.read_bytes() == before_voice
    assert music.read_bytes() == before_music
    assert result.path == destination
    assert result.ducking.verified is True
    assert result.ducking.applied is True
    assert len(result.tracks) == 2
    assert result.to_dict()["tracks"] == 2
    assert len(calls) == 1
    argv = calls[0]
    assert argv.count("-i") == 2
    assert Path(argv[argv.index("-i") + 1]) == voice
    assert Path(argv[argv.index("-i", argv.index("-i") + 1) + 1]) == music
    assert "-filter_complex" in argv
    assert option(argv, "-map") == "[out]"
    assert option(argv, "-c:a") == "libmp3lame"
    assert option(argv, "-ar") == "48000"
    assert option(argv, "-ac") == "2"
    assert option(argv, "-filter_complex") == result.filtergraph


def test_mix_audio_accepts_inline_bytes_and_loops_a_bed(tmp_path: Path) -> None:
    destination = tmp_path / "mix.wav"
    calls: list[list[str]] = []
    result = mix_audio(
        [
            AudioTrack(source=AUDIO, is_voice=True),
            AudioTrack(source=AUDIO, loop=True),
        ],
        destination,
        duration_seconds=3.0,
        output_format="wav",
        ffmpeg_runner=mix_runner(payload=b"RIFF" + b"\x00" * 16, calls=calls),
    )

    assert destination.is_file()
    assert result.settings.format is not None
    assert calls[0].count("-stream_loop") == 1
    inputs = [
        Path(value)
        for index, value in enumerate(calls[0])
        if index > 0 and calls[0][index - 1] == "-i"
    ]
    assert len(inputs) == 2
    assert len(set(inputs)) == 2
    assert all(path.suffix == ".mp3" for path in inputs)
    assert option(calls[0], "-c:a") == "pcm_s16le"


def test_mix_audio_rejects_invalid_track_sets_and_destinations(
    tmp_path: Path,
) -> None:
    source = source_file(tmp_path)
    with pytest.raises(ValidationError):
        build_mix_filtergraph([])
    with pytest.raises(ValidationError):
        build_mix_filtergraph([AudioTrack(source=source, loop=True)])
    with pytest.raises(ValidationError):
        mix_audio([], tmp_path / "mix.mp3")
    with pytest.raises(ValidationError):
        mix_audio(
            [AudioTrack(source=source)],
            source,
            ffmpeg_runner=mix_runner(),
        )
    with pytest.raises(ValidationError):
        mix_audio(
            [AudioTrack(source=source)],
            tmp_path / "no-extension",
            ffmpeg_runner=mix_runner(),
        )
    with pytest.raises(ValueError):
        AudioTrack(source=source, offset_seconds=-1.0)


def test_mix_audio_raises_when_ffmpeg_fails(tmp_path: Path) -> None:
    source = source_file(tmp_path)

    def failing(argv: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            list(argv),
            1,
            stdout=b"",
            stderr=b"bad filter",
        )

    with pytest.raises(ValidationError):
        mix_audio(
            [AudioTrack(source=source)],
            tmp_path / "mix.mp3",
            ffmpeg_runner=failing,
        )


def test_tempo_estimator_ignores_flat_and_tiny_inputs() -> None:
    assert estimate_tempo_from_envelope([0.0] * 100) is None
    assert estimate_tempo_from_envelope([1.0, 2.0]) is None
    assert estimate_tempo_from_envelope([1.0] * 100) is None
    assert estimate_tempo(pcm_click_track(120.0, seconds=0.01)) is None
    assert math.isfinite(estimate_tempo(pcm_click_track(120.0)) or 0.0)

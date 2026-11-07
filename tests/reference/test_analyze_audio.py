"""An audio reference is analyzed for musical traits without copying it."""

from __future__ import annotations

import asyncio
import json
import subprocess
from collections.abc import Coroutine, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from aivia.core.errors import ValidationError
from aivia.providers.protocols.llm import (
    LLMRequest,
    LLMResponse,
    LLMUsage,
    TextPart,
)
from aivia.reference import (
    DEFAULT_REFERENCE_AUDIO_PROMPT,
    AudioReferenceAnalyzer,
    ReferenceAnalysis,
    ReferenceKind,
    ReferenceSpec,
    analyze_audio_reference,
    analyze_audio_reference_offline,
    analyze_audio_reference_sync,
    build_audio_reference_request,
    parse_audio_reference_analysis,
    sample_audio_reference,
)

MP3 = b"ID3" + b"\x00" * 64


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _probe_json(
    *,
    duration: str = "12.5",
    sample_rate: str = "44100",
    channels: int = 2,
    audio: bool = True,
) -> str:
    streams: list[dict[str, Any]] = []
    if audio:
        streams.append(
            {
                "index": 0,
                "codec_type": "audio",
                "codec_name": "mp3",
                "duration": duration,
                "sample_rate": sample_rate,
                "channels": channels,
            }
        )
    else:
        streams.append(
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 640,
                "height": 360,
                "duration": duration,
            }
        )
    return json.dumps(
        {
            "streams": streams,
            "format": {
                "format_name": "mp3",
                "format_long_name": "MP2/3 (MPEG audio layer 2/3)",
                "duration": duration,
                "size": "12345",
            },
        }
    )


def _ffprobe_runner(payload: str) -> Any:
    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            list(argv),
            0,
            stdout=payload,
            stderr="",
        )

    return runner


def _analysis_text(
    *,
    tempo: float = 96.0,
    lufs: float = -14.0,
    silence: Sequence[tuple[float, float]] = ((0.0, 1.0),),
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
            "[ebur128] True peak: Peak: -1.000 dBFS",
            "[astats] Peak level dB: -1.000",
            "[astats] max_volume: -1.000 dB",
            "[astats] mean_volume: -20.000 dB",
            f"tempo: {tempo:.3f} bpm",
        ]
    )
    return "\n".join(lines)


def _ffmpeg_runner(
    *,
    text: str | None = None,
    pcm: bytes = b"",
    calls: list[list[str]] | None = None,
) -> Any:
    payload = _analysis_text() if text is None else text

    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[Any]:
        if calls is not None:
            calls.append(list(argv))
        if "s16le" in argv:
            return subprocess.CompletedProcess(
                list(argv),
                0,
                stdout=pcm,
                stderr=b"",
            )
        return subprocess.CompletedProcess(
            list(argv),
            0,
            stdout=b"",
            stderr=payload.encode(),
        )

    return runner


class FakeProvider:
    """An async LLM provider that returns one canned audio payload."""

    def __init__(self, payload: Mapping[str, Any] | str, *, name: str = "acme") -> None:
        self.payload = payload
        self.name = name
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        content = (
            self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        )
        return LLMResponse(
            content=content,
            provider=self.name,
            model="audio-model",
            usage=LLMUsage(input_tokens=10, output_tokens=5),
        )


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "music": {
            "genre": "lo-fi",
            "moods": ["calm", "nostalgic"],
            "tempo": 92,
            "key": "C",
            "scale": "minor",
            "instrumentation": ["piano", "vinyl crackle", "soft drums"],
            "vocal_type": "female",
            "energy": 0.3,
            "valence": 0.6,
        },
        "structure": {
            "sections": [
                {"name": "intro", "start_seconds": 0.0, "end_seconds": 4.0},
                {"name": "verse", "start_seconds": 4.0, "end_seconds": 20.0},
                {"name": "chorus", "start_seconds": 20.0, "end_seconds": 32.0},
            ],
            "hook": "soft vocal hook",
            "narrative_arc": "calm build to a warm chorus",
        },
        "subject": "Midnight Study by Acme",
        "reproducing_prompt": (
            "A lo-fi track with piano, vinyl crackle and soft drums at 92 bpm, "
            "calm and nostalgic, with a female vocal."
        ),
        "lyrics": "verbatim lyric line that must never be stored",
        "melody": {"notes": [60, 62, 64]},
        "detected_text": "another verbatim line",
        "confidence": 0.82,
    }
    payload.update(overrides)
    return payload


def test_offline_analysis_reads_duration_tempo_loudness_and_stream_shape() -> None:
    analysis = analyze_audio_reference_offline(
        MP3,
        ffprobe_runner=_ffprobe_runner(_probe_json()),
        ffmpeg_runner=_ffmpeg_runner(),
    )

    assert isinstance(analysis, ReferenceAnalysis)
    assert analysis.kind is ReferenceKind.AUDIO
    assert analysis.duration_seconds == 12.5
    assert analysis.music.bpm == 96.0
    assert analysis.music.tempo == 96.0
    assert analysis.music.loudness_lufs == -14.0
    assert analysis.structure.total_duration_seconds == 12.5
    assert analysis.detected_text == ""
    assert analysis.reproducing_prompt == ""


def test_sample_audio_reference_exposes_measured_stream_shape() -> None:
    sample = sample_audio_reference(
        MP3,
        ffprobe_runner=_ffprobe_runner(_probe_json()),
        ffmpeg_runner=_ffmpeg_runner(),
    )

    assert sample.duration_seconds == 12.5
    assert sample.sample_rate == 44100
    assert sample.channels == 2
    assert sample.codec_name == "mp3"
    assert sample.tempo_bpm == 96.0
    assert sample.loudness_lufs == -14.0
    assert sample.silence_seconds == 1.0
    assert sample.has_audio is True


def test_analysis_maps_genre_tempo_mood_instrumentation_vocal_type_and_structure() -> (
    None
):
    provider = FakeProvider(_payload())

    analysis = _run(
        analyze_audio_reference(
            MP3,
            provider=provider,
            ffprobe_runner=_ffprobe_runner(_probe_json()),
            ffmpeg_runner=_ffmpeg_runner(),
        )
    )

    assert analysis.kind is ReferenceKind.AUDIO
    assert analysis.music.genre == "lo-fi"
    assert analysis.music.genres == ["lo-fi"]
    assert analysis.music.moods == ["calm", "nostalgic"]
    assert analysis.music.bpm == 96.0
    assert analysis.music.key == "C"
    assert analysis.music.scale == "minor"
    assert analysis.music.instruments == [
        "piano",
        "vinyl crackle",
        "soft drums",
    ]
    assert analysis.music.instrumentation == analysis.music.instruments
    assert analysis.music.vocals == "female"
    assert analysis.music.vocal_type == "female"
    assert analysis.music.energy == 0.3
    assert analysis.music.valence == 0.6
    assert analysis.structure.scene_count == 3
    assert [scene.description for scene in analysis.structure.scenes] == [
        "intro",
        "verse",
        "chorus",
    ]
    assert analysis.structure.beats == ["intro", "verse", "chorus"]
    assert analysis.structure.hook == "soft vocal hook"
    assert analysis.structure.narrative_arc == "calm build to a warm chorus"
    assert analysis.duration_seconds == 12.5
    assert analysis.provider == "acme"
    assert analysis.model == "audio-model"


def test_analysis_stores_no_lyrics_melody_or_transcript() -> None:
    provider = FakeProvider(_payload())

    analysis = _run(
        analyze_audio_reference(
            MP3,
            provider=provider,
            ffprobe_runner=_ffprobe_runner(_probe_json()),
            ffmpeg_runner=_ffmpeg_runner(),
        )
    )

    assert analysis.detected_text == ""
    assert all(scene.text == "" for scene in analysis.structure.scenes)
    dumped = analysis.to_dict()
    serialized = json.dumps(dumped).lower()
    assert "verbatim lyric line" not in serialized
    assert "another verbatim line" not in serialized
    assert "lyrics" not in dumped
    assert "melody" not in dumped
    assert "notes" not in dumped
    assert not any(
        key in (analysis.model_extra or {})
        for key in ("lyrics", "melody", "transcript", "notes")
    )


def test_request_carries_measured_summary_and_schema() -> None:
    provider = FakeProvider(_payload())

    _run(
        analyze_audio_reference(
            MP3,
            provider=provider,
            model="audio-model",
            ffprobe_runner=_ffprobe_runner(_probe_json()),
            ffmpeg_runner=_ffmpeg_runner(),
        )
    )

    request = provider.requests[0]
    assert request.model == "audio-model"
    assert request.role == "reference_analysis"
    assert request.response_schema is not None
    assert {"music", "structure"} <= set(request.response_schema["properties"])
    assert request.metadata["reference_kind"] == "audio"
    assert request.metadata["duration_seconds"] == 12.5
    assert request.metadata["tempo_bpm"] == 96.0
    assert request.metadata["sample_rate"] == 44100
    assert request.metadata["channels"] == 2
    content = request.messages[0].content
    assert isinstance(content, list)
    text = next(part for part in content if isinstance(part, TextPart)).text
    assert "Measured:" in text
    assert "duration 12.50s" in text
    assert "tempo 96.0 bpm" in text
    assert "44100 Hz" in text
    assert "2 channels" in text


def test_build_request_can_be_inspected_without_a_provider() -> None:
    measured = ReferenceAnalysis.model_validate(
        {
            "kind": ReferenceKind.AUDIO,
            "duration_seconds": 8.0,
            "music": {"genre": "ambient", "bpm": 70.0},
        }
    )

    request = build_audio_reference_request(measured, model="audio-model")

    assert isinstance(request, LLMRequest)
    assert request.role == "reference_analysis"
    assert request.model == "audio-model"
    assert request.response_schema is not None
    content = request.messages[0].content
    assert isinstance(content, list)
    text = next(part for part in content if isinstance(part, TextPart)).text
    assert "ambient" not in text
    assert "duration 8.00s" in text
    assert "tempo 70.0 bpm" in text


def test_parse_merges_measured_tempo_and_strips_verbatim() -> None:
    analysis = parse_audio_reference_analysis(
        _payload(),
        source=MP3,
        reference_id="track-1",
        provider="acme",
        model="audio-model",
        ffprobe_runner=_ffprobe_runner(_probe_json()),
        ffmpeg_runner=_ffmpeg_runner(),
    )

    assert analysis.reference_id == "track-1"
    assert analysis.kind is ReferenceKind.AUDIO
    assert analysis.music.bpm == 96.0
    assert analysis.duration_seconds == 12.5
    assert analysis.structure.beats == ["intro", "verse", "chorus"]
    assert analysis.detected_text == ""
    assert "verbatim lyric line" not in analysis.to_json()
    assert "melody" not in analysis.to_dict()


def test_top_level_music_aliases_are_folded_into_the_music_section() -> None:
    analysis = parse_audio_reference_analysis(
        {
            "genre": "synthwave",
            "mood": "driving",
            "tempo": 118,
            "instrumentation": ["analog synth", "drum machine"],
            "vocal_type": "instrumental",
            "structure": {
                "sections": [
                    {"section": "intro", "start": 0, "duration": 5},
                    {"section": "outro", "start": 30, "duration": 10},
                ]
            },
        }
    )

    assert analysis.music.genre == "synthwave"
    assert analysis.music.moods == ["driving"]
    assert analysis.music.bpm == 118.0
    assert analysis.music.instruments == ["analog synth", "drum machine"]
    assert analysis.music.vocals == "instrumental"
    assert analysis.music.is_instrumental is True
    assert analysis.structure.beats == ["intro", "outro"]


def test_analysis_rejects_a_reference_that_is_not_audio(tmp_path: Path) -> None:
    path = tmp_path / "look.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")

    with pytest.raises(ValidationError, match="not an audio"):
        analyze_audio_reference_offline(
            ReferenceSpec(kind=ReferenceKind.IMAGE, path=str(path))
        )


def test_analysis_rejects_a_remote_reference_before_any_download() -> None:
    with pytest.raises(ValidationError, match="downloaded"):
        analyze_audio_reference_offline(
            ReferenceSpec(kind=ReferenceKind.AUDIO, url="https://example.test/a.mp3")
        )


def test_analysis_rejects_a_file_with_no_audio_stream() -> None:
    with pytest.raises(ValidationError, match="no audio stream"):
        analyze_audio_reference_offline(
            MP3,
            ffprobe_runner=_ffprobe_runner(_probe_json(audio=False)),
            ffmpeg_runner=_ffmpeg_runner(),
        )


def test_sync_wrapper_and_reusable_analyzer_share_the_result() -> None:
    provider = FakeProvider(_payload())

    direct = analyze_audio_reference_sync(
        MP3,
        provider=provider,
        ffprobe_runner=_ffprobe_runner(_probe_json()),
        ffmpeg_runner=_ffmpeg_runner(),
    )
    reusable = _run(
        AudioReferenceAnalyzer(
            provider,
            ffprobe_runner=_ffprobe_runner(_probe_json()),
            ffmpeg_runner=_ffmpeg_runner(),
        ).analyze(MP3)
    )

    assert direct.reproducing_prompt == reusable.reproducing_prompt
    assert reusable.provider == "acme"
    assert reusable.music.genre == "lo-fi"


def test_the_default_prompt_names_every_required_section_and_forbids_verbatim() -> None:
    prompt = DEFAULT_REFERENCE_AUDIO_PROMPT.lower()

    for word in (
        "genre",
        "tempo",
        "mood",
        "instrumentation",
        "vocal type",
        "structure",
        "lyrics",
        "melody",
        "never",
    ):
        assert word in prompt

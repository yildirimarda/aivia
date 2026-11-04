"""The voice stage voices scripts and writes measured durations to a timeline."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from aivia.audio import VoiceStage, VoiceStageResult
from aivia.compose import Timeline, compositions_from_timeline
from aivia.core.errors import ConfigError
from aivia.core.stages import StageContext
from aivia.providers.protocols.tts import TTSRequest


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


class FakeSpeech:
    """Return a measured duration and a fake stored path for each request."""

    def __init__(
        self,
        *,
        durations: dict[str, float] | None = None,
        language_durations: dict[str, float] | None = None,
        words: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.durations = durations or {}
        self.language_durations = language_durations or {}
        self.words = words or {}
        self.requests: list[TTSRequest] = []

    async def __call__(self, request: TTSRequest) -> SimpleNamespace:
        self.requests.append(request)
        duration = self.durations.get(request.text)
        if duration is None and request.language is not None:
            duration = self.language_durations.get(request.language)
        if duration is None:
            duration = 1.0
        slug = f"{len(self.requests):02d}"
        return SimpleNamespace(
            duration_seconds=duration,
            path=f"audio/{slug}.mp3",
            cost_usd=0.01,
            provider="fake",
            model="fake-tts",
            word_timestamps=self.words.get(request.text, []),
        )


def _script(
    *beats: dict[str, Any],
    language: str = "en",
    hook: str = "",
    outro: str = "",
) -> dict[str, Any]:
    return {
        "language": language,
        "mode": "facts",
        "hook": hook,
        "outro": outro,
        "beats": list(beats),
    }


def _write_script(run_dir: Path, value: dict[str, Any]) -> Path:
    path = run_dir / f"script.{value['language']}.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _context(run_dir: Path, *, config: Any = None) -> StageContext:
    return StageContext(run_dir=run_dir, config=config or {})


def test_beat_mode_voices_each_beat_with_gaps_and_real_durations(
    tmp_path: Path,
) -> None:
    _write_script(
        tmp_path,
        _script(
            {"text": "First line."},
            {"text": "Second line."},
            {"text": "Third line."},
            hook="Hook.",
        ),
    )
    speech = FakeSpeech(
        durations={"First line.": 1.0, "Second line.": 2.0, "Third line.": 1.5}
    )
    stage = VoiceStage(
        options={"mode": "beat", "gap_seconds": 0.5, "voice": "narrator"},
        speech=speech,
    )

    result = cast(VoiceStageResult, _run(stage.run(_context(tmp_path))))

    assert result.stage == "voice"
    assert result.artifacts == ["timeline.json"]
    assert result.metadata["mode"] == "beat"
    assert result.metadata["languages"] == ["en"]
    assert result.metadata["calls"] == 3
    assert result.durations["en"] == pytest.approx(5.5)
    assert result.cost_usd == pytest.approx(0.03)
    assert [request.text for request in speech.requests] == [
        "First line.",
        "Second line.",
        "Third line.",
    ]
    assert all(request.voice == "narrator" for request in speech.requests)
    assert all(request.language == "en" for request in speech.requests)

    payload = json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    timeline = Timeline.model_validate(payload)
    assert [beat.id for beat in timeline.beats] == ["beat_0", "beat_1", "beat_2"]
    assert [beat.start_seconds for beat in timeline.beats] == pytest.approx(
        [0.0, 1.5, 4.0]
    )
    assert [beat.text for beat in timeline.beats] == [
        "First line.",
        "Second line.",
        "Third line.",
    ]

    language = timeline.languages[0]
    assert language.language == "en"
    assert [voice.start_seconds for voice in language.voices] == pytest.approx(
        [0.0, 1.5, 4.0]
    )
    assert [voice.end_seconds for voice in language.voices] == pytest.approx(
        [1.0, 3.5, 5.5]
    )
    assert [voice.source for voice in language.voices] == [
        "audio/01.mp3",
        "audio/02.mp3",
        "audio/03.mp3",
    ]
    assert [voice.beat for voice in language.voices] == [
        "beat_0",
        "beat_1",
        "beat_2",
    ]

    compositions = compositions_from_timeline(timeline)
    assert len(compositions) == 1
    assert compositions[0].duration_seconds == pytest.approx(5.5)
    assert [
        (track.start_seconds, track.duration_seconds)
        for track in compositions[0].audio_tracks
    ] == pytest.approx([(0.0, 1.0), (1.5, 2.0), (4.0, 1.5)])


def test_script_mode_makes_one_call_per_language_and_records_measured_duration(
    tmp_path: Path,
) -> None:
    _write_script(
        tmp_path,
        _script({"text": "First line."}, {"text": "Second line."}, hook="Hook."),
    )
    _write_script(
        tmp_path,
        _script(
            {"text": "Erste Zeile."},
            {"text": "Zweite Zeile."},
            language="de",
            hook="Einstieg.",
        ),
    )
    speech = FakeSpeech(language_durations={"en": 4.25, "de": 6.5})
    stage = VoiceStage(
        options={"mode": "script", "languages": ["en", "de"]},
        speech=speech,
    )

    result = cast(VoiceStageResult, _run(stage.run(_context(tmp_path))))

    assert result.metadata["mode"] == "script"
    assert result.metadata["languages"] == ["en", "de"]
    assert result.metadata["calls"] == 2
    assert result.durations == pytest.approx({"en": 4.25, "de": 6.5})
    assert len(speech.requests) == 2
    assert speech.requests[0].text == "Hook.\n\nFirst line.\n\nSecond line."
    assert speech.requests[1].text == "Einstieg.\n\nErste Zeile.\n\nZweite Zeile."

    timeline = Timeline.model_validate(
        json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    )
    assert [language.language for language in timeline.languages] == ["en", "de"]
    for language, expected in zip(timeline.languages, (4.25, 6.5), strict=True):
        assert len(language.voices) == 1
        voice = language.voices[0]
        assert voice.start_seconds == 0.0
        assert voice.end_seconds == pytest.approx(expected)
        assert voice.duration_seconds == pytest.approx(expected)
        assert voice.beat == ""

    assert [beat.start_seconds for beat in timeline.beats] == pytest.approx(
        [0.0, 2.125]
    )
    assert compositions_from_timeline(timeline)[0].duration_seconds == pytest.approx(
        4.25
    )


def test_script_mode_uses_explicit_beat_durations_for_beat_starts(
    tmp_path: Path,
) -> None:
    _write_script(
        tmp_path,
        _script(
            {"text": "Short.", "duration_seconds": 1.0},
            {"text": "Long.", "duration_seconds": 3.0},
        ),
    )
    stage = VoiceStage(
        options={"mode": "script"},
        speech=FakeSpeech(language_durations={"en": 5.0}),
    )

    _run(stage.run(_context(tmp_path)))

    timeline = Timeline.model_validate(
        json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    )
    assert [beat.start_seconds for beat in timeline.beats] == pytest.approx([0.0, 1.0])


def test_word_timings_are_shifted_by_the_voice_start(tmp_path: Path) -> None:
    _write_script(
        tmp_path,
        _script({"text": "First."}, {"text": "Second."}),
    )
    speech = FakeSpeech(
        durations={"First.": 1.0, "Second.": 1.0},
        words={
            "Second.": [
                {"word": "Second", "start": 0.1, "end": 0.4},
            ]
        },
    )
    stage = VoiceStage(options={"gap_seconds": 0.5}, speech=speech)

    _run(stage.run(_context(tmp_path)))

    timeline = Timeline.model_validate(
        json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    )
    second = timeline.languages[0].voices[1]
    assert second.start_seconds == pytest.approx(1.5)
    assert len(second.words) == 1
    assert second.words[0].word == "Second"
    assert second.words[0].start_seconds == pytest.approx(1.6)
    assert second.words[0].end_seconds == pytest.approx(1.9)


def test_stage_rejects_a_missing_script_before_spending(tmp_path: Path) -> None:
    speech = FakeSpeech()
    stage = VoiceStage(options={"languages": ["en"]}, speech=speech)

    with pytest.raises(ConfigError, match="needs a script"):
        _run(stage.run(_context(tmp_path)))

    assert speech.requests == []


def test_stage_rejects_an_unknown_mode_and_a_negative_gap() -> None:
    with pytest.raises(ConfigError, match="unknown voice synthesis mode"):
        VoiceStage(options={"mode": "karaoke"})

    with pytest.raises(ConfigError, match="gap_seconds"):
        VoiceStage(options={"gap_seconds": -1})


def test_stage_reads_only_the_requested_language(tmp_path: Path) -> None:
    _write_script(tmp_path, _script({"text": "Hello."}, language="en"))
    _write_script(tmp_path, _script({"text": "Hallo."}, language="de"))
    speech = FakeSpeech()

    stage = VoiceStage(options={"languages": ["de"]}, speech=speech)
    result = _run(stage.run(_context(tmp_path)))

    assert result.metadata["languages"] == ["de"]
    assert [request.text for request in speech.requests] == ["Hallo."]
    timeline = Timeline.model_validate(
        json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    )
    assert [language.language for language in timeline.languages] == ["de"]

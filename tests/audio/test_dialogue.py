"""Dialogue lines are voiced per speaker with correct narration timing."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from aivia.audio import VoiceStage, VoiceStageResult
from aivia.compose import Timeline
from aivia.core.stages import StageContext
from aivia.providers.protocols.tts import TTSRequest


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


class FakeSpeech:
    """Return a measured duration and a fake stored path for each request."""

    def __init__(
        self,
        durations: dict[str, float] | None = None,
        words: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.durations = durations or {}
        self.words = words or {}
        self.requests: list[TTSRequest] = []

    async def __call__(self, request: TTSRequest) -> SimpleNamespace:
        self.requests.append(request)
        slug = f"{len(self.requests):02d}"
        return SimpleNamespace(
            duration_seconds=self.durations.get(request.text, 1.0),
            path=f"audio/{slug}.mp3",
            cost_usd=0.01,
            provider="fake",
            model="fake-tts",
            word_timestamps=self.words.get(request.text, []),
        )


def _script(*beats: dict[str, Any], language: str = "en") -> dict[str, Any]:
    return {
        "language": language,
        "mode": "facts",
        "beats": list(beats),
    }


def _write_script(run_dir: Path, value: dict[str, Any]) -> Path:
    path = run_dir / f"script.{value['language']}.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _context(run_dir: Path, *, config: Any = None) -> StageContext:
    return StageContext(run_dir=run_dir, config=config or {})


def _timeline(run_dir: Path) -> Timeline:
    return Timeline.model_validate(
        json.loads((run_dir / "timeline.json").read_text(encoding="utf-8"))
    )


def test_narrator_and_character_lines_use_distinct_voices_without_duplication(
    tmp_path: Path,
) -> None:
    _write_script(
        tmp_path,
        _script(
            {
                "text": "Narration.",
                "dialogue": [
                    {"speaker": "narrator", "text": "Narration."},
                    {"speaker": "hero", "text": "Hello."},
                ],
            },
            {
                "dialogue": [
                    {"speaker": "hero", "text": "Again."},
                ],
            },
        ),
    )
    speech = FakeSpeech({"Narration.": 1.0, "Hello.": 2.0, "Again.": 1.0})
    stage = VoiceStage(
        options={
            "gap_seconds": 0.5,
            "voice": "narrator-voice",
            "speaker_voices": {"hero": "hero-voice"},
        },
        speech=speech,
    )

    result = cast(VoiceStageResult, _run(stage.run(_context(tmp_path))))

    assert [request.text for request in speech.requests] == [
        "Narration.",
        "Hello.",
        "Again.",
    ]
    assert [request.voice for request in speech.requests] == [
        "narrator-voice",
        "hero-voice",
        "hero-voice",
    ]
    assert [request.metadata["speaker"] for request in speech.requests] == [
        "narrator",
        "hero",
        "hero",
    ]
    assert result.metadata["calls"] == 3
    assert result.durations["en"] == pytest.approx(5.0)
    assert result.cost_usd == pytest.approx(0.03)

    timeline = _timeline(tmp_path)
    assert [beat.start_seconds for beat in timeline.beats] == pytest.approx([0.0, 4.0])
    language = timeline.languages[0]
    assert [voice.speaker for voice in language.voices] == [
        "narrator",
        "hero",
        "hero",
    ]
    assert [voice.voice for voice in language.voices] == [
        "narrator-voice",
        "hero-voice",
        "hero-voice",
    ]
    assert [voice.start_seconds for voice in language.voices] == pytest.approx(
        [0.0, 1.5, 4.0]
    )
    assert [voice.end_seconds for voice in language.voices] == pytest.approx(
        [1.0, 3.5, 5.0]
    )
    assert [voice.beat for voice in language.voices] == [
        "beat_0",
        "beat_0",
        "beat_1",
    ]
    assert [voice.id for voice in language.voices] == [
        "en_beat_0_0",
        "en_beat_0_1",
        "en_beat_1",
    ]


def test_a_line_voice_overrides_the_speaker_mapping(tmp_path: Path) -> None:
    _write_script(
        tmp_path,
        _script(
            {
                "dialogue": [
                    {"speaker": "hero", "text": "Normal."},
                    {"speaker": "hero", "text": "Whisper.", "voice": "soft-voice"},
                ]
            }
        ),
    )
    speech = FakeSpeech()
    stage = VoiceStage(
        options={
            "voice": "narrator-voice",
            "speaker_voices": {"HERO": "hero-voice"},
        },
        speech=speech,
    )

    _run(stage.run(_context(tmp_path)))

    assert [request.voice for request in speech.requests] == [
        "hero-voice",
        "soft-voice",
    ]


def test_an_unmapped_character_falls_back_to_its_speaker_alias(
    tmp_path: Path,
) -> None:
    _write_script(
        tmp_path,
        _script(
            {
                "text": "Narration.",
                "dialogue": [{"speaker": "guide", "text": "A line."}],
            }
        ),
    )
    speech = FakeSpeech()
    stage = VoiceStage(options={"voice": "narrator-voice"}, speech=speech)

    _run(stage.run(_context(tmp_path)))

    assert [request.voice for request in speech.requests] == [
        "narrator-voice",
        "guide",
    ]


def test_speaker_voices_are_read_from_the_configured_voice_section(
    tmp_path: Path,
) -> None:
    _write_script(
        tmp_path,
        _script(
            {
                "text": "Narration.",
                "dialogue": [{"speaker": "guide", "text": "A line."}],
            }
        ),
    )
    speech = FakeSpeech()
    stage = VoiceStage(speech=speech)
    config = {
        "voices": {
            "default": "configured-narrator",
            "speakers": {"guide": "configured-guide"},
        }
    }

    _run(stage.run(_context(tmp_path, config=config)))

    assert [request.voice for request in speech.requests] == [
        "configured-narrator",
        "configured-guide",
    ]


def test_script_level_dialogue_is_voiced_when_there_are_no_beats(
    tmp_path: Path,
) -> None:
    _write_script(
        tmp_path,
        {
            "language": "en",
            "mode": "story",
            "dialogue_lines": [
                {"speaker": "narrator", "text": "Once."},
                {"speaker": "bard", "text": "Listen."},
            ],
        },
    )
    speech = FakeSpeech({"Once.": 1.0, "Listen.": 2.0})
    stage = VoiceStage(
        options={"voice": "narrator-voice", "speaker_voices": {"bard": "bard-voice"}},
        speech=speech,
    )

    result = cast(VoiceStageResult, _run(stage.run(_context(tmp_path))))

    assert [request.text for request in speech.requests] == ["Once.", "Listen."]
    assert [request.voice for request in speech.requests] == [
        "narrator-voice",
        "bard-voice",
    ]
    assert result.durations["en"] == pytest.approx(3.0)
    language = _timeline(tmp_path).languages[0]
    assert [voice.speaker for voice in language.voices] == ["narrator", "bard"]
    assert [voice.beat for voice in language.voices] == ["", ""]
    assert [voice.id for voice in language.voices] == [
        "en_dialogue_0",
        "en_dialogue_1",
    ]


def test_explicit_line_timing_is_kept_and_words_are_shifted(
    tmp_path: Path,
) -> None:
    _write_script(
        tmp_path,
        _script(
            {
                "dialogue": [
                    {
                        "speaker": "hero",
                        "text": "Delayed.",
                        "start": 2.0,
                        "duration": 1.5,
                    }
                ]
            }
        ),
    )
    speech = FakeSpeech(
        words={"Delayed.": [{"word": "Delayed", "start": 0.25, "end": 0.75}]}
    )
    stage = VoiceStage(speech=speech)

    _run(stage.run(_context(tmp_path)))

    voice = _timeline(tmp_path).languages[0].voices[0]
    assert voice.start_seconds == pytest.approx(2.0)
    assert voice.end_seconds == pytest.approx(3.5)
    assert voice.words[0].start_seconds == pytest.approx(2.25)
    assert voice.words[0].end_seconds == pytest.approx(2.75)

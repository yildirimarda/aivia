"""The voice stage voices scripts and writes measured durations to a timeline."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from sqlalchemy import Engine

from aivia.audio import VoiceStage, VoiceStageResult
from aivia.compose import Timeline, compositions_from_timeline
from aivia.core.cost import CostTracker
from aivia.core.db import (
    RunRepository,
    StageRunRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.errors import ConfigError, ValidationError
from aivia.core.models import Run, StageState
from aivia.core.orchestrator import Orchestrator
from aivia.core.resume import compute_input_hash
from aivia.core.run_state import RunStateMachine
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


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


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


def test_stage_declares_each_language_script_file_as_an_input() -> None:
    stage = VoiceStage(options={"languages": ["en", "de"]})

    assert stage.inputs == ("script.en.json", "script.de.json")


def test_stage_declares_script_inputs_using_the_configured_pattern() -> None:
    stage = VoiceStage(
        options={
            "languages": ["en"],
            "script_filename_pattern": "scripts/{code}/script.json",
        }
    )

    assert stage.inputs == ("scripts/en/script.json",)


def test_stage_declares_no_script_inputs_when_scripts_are_injected() -> None:
    stage = VoiceStage(
        scripts={"en": _script({"text": "Hello."})},
        options={"languages": ["en"]},
    )

    assert stage.inputs == ()


def test_editing_a_language_script_changes_the_stage_input_hash(
    tmp_path: Path,
) -> None:
    _write_script(tmp_path, _script({"text": "First."}, language="en"))
    _write_script(tmp_path, _script({"text": "Erste."}, language="de"))
    stage = VoiceStage(options={"languages": ["en", "de"]})
    original = compute_input_hash(stage, base_dir=tmp_path)

    _write_script(tmp_path, _script({"text": "Changed."}, language="de"))
    assert compute_input_hash(stage, base_dir=tmp_path) != original

    _write_script(tmp_path, _script({"text": "Erste."}, language="de"))
    assert compute_input_hash(stage, base_dir=tmp_path) == original


def test_editing_a_discovered_script_changes_the_stage_input_hash(
    tmp_path: Path,
) -> None:
    _write_script(tmp_path, _script({"text": "First."}, language="en"))
    stage = VoiceStage()
    original = compute_input_hash(stage, base_dir=tmp_path)

    _write_script(tmp_path, _script({"text": "Changed."}, language="en"))

    assert compute_input_hash(stage, base_dir=tmp_path) != original


def test_editing_a_script_reruns_the_voice_stage_on_resume(
    tmp_path: Path,
    engine: Engine,
) -> None:
    _write_script(tmp_path, _script({"text": "First."}, language="en"))
    machine = RunStateMachine(RunRepository(engine))
    run = machine.create(Run())
    speech = FakeSpeech()
    stage = VoiceStage(speech=speech)

    def execute() -> Any:
        return Orchestrator(
            stages=[stage],
            run=run.id,
            repository=StageRunRepository(engine),
            run_dir=tmp_path,
            config={},
            cost=CostTracker(engine),
            raise_on_failure=False,
        ).execute_sync()

    first = execute()
    stored = StageRunRepository(engine).get(run.id, "voice")
    assert first.ok is True
    assert len(speech.requests) == 1
    assert stored is not None

    _write_script(tmp_path, _script({"text": "Changed."}, language="en"))
    second = execute()

    assert second.ok is True
    assert len(speech.requests) == 2
    assert second.stages[0].status is StageState.SUCCEEDED
    assert second.stages[0].input_hash != stored.input_hash

    third = execute()

    assert third.ok is True
    assert len(speech.requests) == 2
    assert third.stages[0].status is StageState.SKIPPED


def test_stage_runs_audio_quality_checks_and_records_the_report(
    tmp_path: Path,
) -> None:
    _write_script(
        tmp_path,
        _script({"text": "First line."}, {"text": "Second line."}),
    )
    speech = FakeSpeech(
        durations={"First line.": 1.0, "Second line.": 1.0},
    )
    stage = VoiceStage(options={"gap_seconds": 0.5}, speech=speech)

    result = cast(VoiceStageResult, _run(stage.run(_context(tmp_path))))

    quality = result.metadata["quality"]
    assert quality == result.metadata["audio_quality"]
    assert quality["ok"] is True
    assert quality["passed"] is True
    assert quality["segments"] == 2
    assert quality["languages"] == ["en"]
    assert quality["findings"] == []
    assert result.metadata["quality_ok"] is True
    assert result.metadata["quality_findings"] == []
    assert result.quality == quality
    assert result.quality_ok is True
    assert result.quality_findings == []


def test_stage_records_duplication_findings_but_still_succeeds(
    tmp_path: Path,
) -> None:
    _write_script(tmp_path, _script({"text": "The the same."}))
    speech = FakeSpeech(durations={"The the same.": 1.0})
    stage = VoiceStage(speech=speech)

    result = cast(VoiceStageResult, _run(stage.run(_context(tmp_path))))

    quality = result.metadata["quality"]
    assert quality["ok"] is False
    assert quality["passed"] is True
    assert [finding["code"] for finding in quality["findings"]] == ["duplication"]
    assert "repeats the word 'the'" in quality["findings"][0]["message"]
    assert result.metadata["quality_ok"] is True
    assert result.metadata["quality_findings"] == quality["findings"]
    assert result.status is StageState.SUCCEEDED


def test_stage_records_silence_findings_from_word_timings(
    tmp_path: Path,
) -> None:
    _write_script(tmp_path, _script({"text": "Hello world."}))
    speech = FakeSpeech(
        durations={"Hello world.": 3.0},
        words={
            "Hello world.": [
                {"word": "Hello", "start": 1.0, "end": 1.5},
                {"word": "world", "start": 1.5, "end": 3.0},
            ]
        },
    )
    stage = VoiceStage(speech=speech)

    result = cast(VoiceStageResult, _run(stage.run(_context(tmp_path))))

    findings = result.metadata["quality_findings"]
    assert [finding["code"] for finding in findings] == ["silence"]
    assert "starts with 1.00s of silence" in findings[0]["message"]
    assert findings[0]["severity"] == "warning"
    assert result.metadata["quality"]["passed"] is True
    assert result.quality_ok is True


def test_stage_quality_thresholds_can_be_configured_or_disabled(
    tmp_path: Path,
) -> None:
    _write_script(tmp_path, _script({"text": "Hello world."}))
    speech = FakeSpeech(
        durations={"Hello world.": 3.0},
        words={
            "Hello world.": [
                {"word": "Hello", "start": 1.0, "end": 1.5},
                {"word": "world", "start": 1.5, "end": 3.0},
            ]
        },
    )

    relaxed = VoiceStage(
        options={"quality": {"max_leading_silence_seconds": 5.0}},
        speech=speech,
    )
    relaxed_result = cast(
        VoiceStageResult,
        _run(relaxed.run(_context(tmp_path))),
    )
    assert relaxed_result.metadata["quality"]["ok"] is True

    disabled = VoiceStage(options={"quality": False}, speech=speech)
    disabled_result = cast(
        VoiceStageResult,
        _run(disabled.run(_context(tmp_path))),
    )
    assert "quality" not in disabled_result.metadata
    assert disabled_result.quality == {}


def test_config_audio_qa_thresholds_are_stage_defaults(tmp_path: Path) -> None:
    _write_script(tmp_path, _script({"text": "Hello world."}))
    speech = FakeSpeech(
        durations={"Hello world.": 3.0},
        words={
            "Hello world.": [
                {"word": "Hello", "start": 1.0, "end": 1.5},
                {"word": "world", "start": 1.5, "end": 3.0},
            ]
        },
    )
    config = {"audio_qa": {"max_leading_silence_seconds": 5.0}}
    stage = VoiceStage(speech=speech)

    result = cast(
        VoiceStageResult,
        _run(stage.run(_context(tmp_path, config=config))),
    )

    assert result.metadata["quality"]["findings"] == []
    assert result.quality_ok is True


def test_config_audio_qa_pronunciations_are_stage_defaults(tmp_path: Path) -> None:
    _write_script(tmp_path, _script({"text": "nuclear energy is safe."}))
    speech = FakeSpeech(durations={"nuclear energy is safe.": 2.0})
    config = {"audio_qa": {"pronunciations": {"nuclear": ["nucular"]}}}
    stage = VoiceStage(
        options={
            "quality": {
                "transcripts": {"en_beat_0": "nucular energy is safe."},
            }
        },
        speech=speech,
    )

    result = cast(
        VoiceStageResult,
        _run(stage.run(_context(tmp_path, config=config))),
    )

    assert result.metadata["quality"]["findings"] == []
    assert result.quality_ok is True


def test_config_audio_qa_failure_policy_fails_the_stage_on_errors(
    tmp_path: Path,
) -> None:
    _write_script(tmp_path, _script({"text": "Hello world."}))
    speech = FakeSpeech(
        durations={"Hello world.": 3.0},
        words={"Hello world.": [{"word": "Hello", "start": 0.0, "end": 0.1}]},
    )
    config = {"audio_qa": {"fail_on_error": True}}
    stage = VoiceStage(speech=speech)

    with pytest.raises(ValidationError) as caught:
        _run(stage.run(_context(tmp_path, config=config)))

    details = caught.value.details
    assert details["quality"]["passed"] is False
    assert "silence" in {finding["code"] for finding in details["findings"]}


def test_config_audio_qa_failure_policy_can_report_instead_of_failing(
    tmp_path: Path,
) -> None:
    _write_script(tmp_path, _script({"text": "Hello world."}))
    speech = FakeSpeech(
        durations={"Hello world.": 3.0},
        words={"Hello world.": [{"word": "Hello", "start": 0.0, "end": 0.1}]},
    )
    config = {"audio_qa": {"fail_on_error": False}}
    stage = VoiceStage(speech=speech)

    result = cast(
        VoiceStageResult,
        _run(stage.run(_context(tmp_path, config=config))),
    )

    assert result.metadata["quality"]["passed"] is False
    assert result.quality_ok is False
    assert result.status is StageState.SUCCEEDED


def test_stage_quality_options_override_audio_qa_config_defaults(
    tmp_path: Path,
) -> None:
    _write_script(tmp_path, _script({"text": "Hello world."}))
    speech = FakeSpeech(
        durations={"Hello world.": 3.0},
        words={
            "Hello world.": [
                {"word": "Hello", "start": 1.0, "end": 1.5},
                {"word": "world", "start": 1.5, "end": 3.0},
            ]
        },
    )
    config = {"audio_qa": {"enabled": False, "max_leading_silence_seconds": 5.0}}
    stage = VoiceStage(
        options={
            "quality": {
                "enabled": True,
                "max_leading_silence_seconds": 0.1,
            }
        },
        speech=speech,
    )

    result = cast(
        VoiceStageResult,
        _run(stage.run(_context(tmp_path, config=config))),
    )

    assert [finding["code"] for finding in result.quality_findings] == ["silence"]

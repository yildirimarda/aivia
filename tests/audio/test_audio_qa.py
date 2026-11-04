"""Audio quality checks report silence, duplication, truncation and wording."""

from __future__ import annotations

import pytest

from aivia.audio import (
    AudioQAFinding,
    AudioQAReport,
    AudioQualityChecker,
    AudioSegment,
    Severity,
    SilenceSpan,
    TimedWord,
    check_audio_quality,
    segments_from_timeline,
)
from aivia.compose import Timeline


def _segment(**overrides: object) -> AudioSegment:
    values: dict[str, object] = {
        "id": "voice",
        "text": "hello world",
        "language": "en",
        "duration_seconds": 0.8,
        "words": (TimedWord("hello", 0.0, 0.4), TimedWord("world", 0.4, 0.8)),
        "transcript": "hello world",
    }
    values.update(overrides)
    return AudioSegment(**values)  # type: ignore[arg-type]


def test_a_clean_segment_has_no_findings() -> None:
    report = check_audio_quality([_segment()])

    assert isinstance(report, AudioQAReport)
    assert report.ok is True
    assert report.passed is True
    assert report.findings == []
    assert report.segments == 1
    assert report.languages == ("en",)
    assert report.to_dict()["ok"] is True


def test_long_leading_trailing_and_internal_silence_is_reported() -> None:
    segment = _segment(
        text="one two three",
        duration_seconds=10.0,
        transcript="",
        words=(
            TimedWord("one", 2.0, 2.5),
            TimedWord("two", 5.0, 5.5),
            TimedWord("three", 8.0, 8.5),
        ),
    )

    report = check_audio_quality([segment])

    assert report.silence
    assert all(finding.code == "silence" for finding in report.silence)
    assert all(finding.severity is Severity.WARNING for finding in report.silence)
    assert any("starts with 2.00s" in finding.message for finding in report.silence)
    assert any("ends with 1.50s" in finding.message for finding in report.silence)
    assert sum("internal pause" in finding.message for finding in report.silence) == 2
    assert any("85% silent" in finding.message for finding in report.silence)


def test_an_almost_silent_segment_is_an_error() -> None:
    segment = _segment(
        text="hello",
        duration_seconds=10.0,
        words=(),
        transcript="",
        silences=(SilenceSpan(0.0, 10.0),),
    )

    report = check_audio_quality([segment])

    assert report.passed is False
    assert report.ok is False
    assert [finding.code for finding in report.errors] == ["silence"]
    assert "almost entirely silent" in report.errors[0].message


def test_explicit_silence_spans_are_used_without_word_timings() -> None:
    segment = _segment(
        duration_seconds=4.0,
        words=(),
        transcript="",
        silences=(SilenceSpan(0.4, 1.6),),
    )

    report = check_audio_quality([segment])

    assert len(report.silence) == 1
    assert "1.20s internal pause" in report.silence[0].message


def test_repeated_words_and_sentences_are_duplicates() -> None:
    words = check_audio_quality([AudioSegment(text="The the same.")])
    sentences = check_audio_quality([AudioSegment(text="We go. We go.")])

    assert [finding.code for finding in words] == ["duplication"]
    assert "repeats the word 'the'" in words[0].message
    assert [finding.code for finding in sentences] == ["duplication"]
    assert "repeats the sentence 'We go.'" in sentences[0].message


def test_two_segments_with_the_same_text_are_duplicates() -> None:
    first = AudioSegment(id="first", text="Same line.", language="en")
    second = AudioSegment(id="second", text="Same line.", language="en")

    report = check_audio_quality([first, second])

    assert len(report.duplication) == 1
    assert report.duplication[0].severity is Severity.WARNING
    assert "repeats the text of first" in report.duplication[0].message


def test_a_transcript_missing_the_tail_is_truncation() -> None:
    segment = _segment(
        text="one two three four",
        transcript="one two",
        duration_seconds=2.0,
        words=(),
    )

    report = check_audio_quality([segment])

    assert len(report.truncation) == 1
    finding = report.truncation[0]
    assert finding.severity is Severity.ERROR
    assert "ends after 2 of 4 expected words" in finding.message
    assert report.mispronunciation == []


def test_word_timing_past_the_segment_end_is_truncation() -> None:
    segment = _segment(
        text="hello world",
        transcript="",
        end_seconds=2.0,
        duration_seconds=2.0,
        words=(TimedWord("hello", 0.0, 0.5), TimedWord("world", 0.6, 3.0)),
    )

    report = check_audio_quality([segment])

    assert len(report.truncation) == 1
    assert report.truncation[0].severity is Severity.ERROR
    assert "past the 2.00s segment end" in report.truncation[0].message


def test_a_duration_far_shorter_than_the_text_is_truncation() -> None:
    segment = AudioSegment(
        id="short",
        text="one two three four five six seven eight nine ten",
        duration_seconds=1.0,
    )

    report = check_audio_quality([segment])

    assert len(report.truncation) == 1
    assert report.truncation[0].severity is Severity.ERROR
    assert "need about 4.00s" in report.truncation[0].message


def test_a_trailing_ellipsis_is_a_truncation_warning() -> None:
    segment = _segment(
        text="A story cut off\u2026",
        transcript="A story cut off",
        words=(),
    )

    report = check_audio_quality([segment])

    assert len(report.truncation) == 1
    assert report.truncation[0].severity is Severity.WARNING
    assert "ellipsis" in report.truncation[0].message


def test_a_transcript_mismatch_is_a_mispronunciation_warning() -> None:
    segment = _segment(
        text="nuclear energy is safe",
        transcript="nucular energy is safe",
        words=(),
    )

    report = check_audio_quality([segment])

    assert len(report.mispronunciation) == 1
    finding = report.mispronunciation[0]
    assert finding.severity is Severity.WARNING
    assert finding.expected == "nuclear"
    assert finding.actual == "nucular"
    assert "may mispronounce 'nuclear' as 'nucular'" in finding.message


def test_an_accepted_pronunciation_suppresses_the_finding() -> None:
    segment = _segment(
        text="nuclear energy is safe",
        transcript="nucular energy is safe",
        words=(),
    )

    report = check_audio_quality(
        [segment],
        pronunciations={"nuclear": ["nucular"]},
    )

    assert report.mispronunciation == []
    assert report.ok is True


def test_many_wrong_words_raise_the_severity_to_error() -> None:
    segment = _segment(
        text="alpha beta gamma delta",
        transcript="one two three four",
        words=(),
    )

    report = check_audio_quality([segment])

    assert len(report.mispronunciation) == 4
    assert all(
        finding.severity is Severity.ERROR for finding in report.mispronunciation
    )


def test_a_low_confidence_word_is_a_mispronunciation_warning() -> None:
    segment = _segment(
        text="hello world",
        transcript="",
        words=(
            TimedWord("hello", 0.0, 0.4, confidence=0.2),
            TimedWord("world", 0.4, 0.8, confidence=0.9),
        ),
    )

    report = check_audio_quality([segment])

    assert len(report.mispronunciation) == 1
    assert report.mispronunciation[0].severity is Severity.WARNING
    assert "low confidence (0.20) on 'hello'" in report.mispronunciation[0].message


def test_the_report_groups_findings_and_serializes() -> None:
    silence = _segment(
        id="silent",
        text="one two",
        duration_seconds=10.0,
        transcript="",
        words=(TimedWord("one", 5.0, 5.5), TimedWord("two", 6.0, 6.5)),
    )
    truncated = AudioSegment(
        id="short",
        text="one two three four",
        transcript="one",
        duration_seconds=1.0,
    )
    misheard = AudioSegment(
        id="misheard",
        text="nuclear energy is safe",
        transcript="nucular energy is safe",
    )

    report = check_audio_quality([silence, truncated, misheard])

    assert report.ok is False
    assert report.passed is False
    assert report.by_code("silence")
    assert report.by_code("truncation")
    assert report.by_code("mispronunciation")
    assert report.errors
    assert report.warnings
    payload = report.to_dict()
    assert payload["segments"] == 3
    assert set(payload["languages"]) == {"en"}
    assert isinstance(report[0], AudioQAFinding)
    assert payload["findings"][0]["severity"] in {"warning", "error"}


def _timeline(voices: list[dict[str, object]]) -> Timeline:
    return Timeline.model_validate(
        {
            "beats": [{"id": "beat_0", "text": "Hello world"}],
            "languages": [{"language": "en", "voices": voices}],
        }
    )


def test_timeline_segments_use_script_text_and_a_transcript() -> None:
    timeline = _timeline(
        [
            {
                "id": "v1",
                "beat": "beat_0",
                "source": "audio/v1.mp3",
                "start": 0.0,
                "end": 1.0,
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.4},
                    {"word": "world", "start": 0.4, "end": 0.8},
                ],
            }
        ]
    )
    scripts = {"en": {"beats": [{"text": "Hello world"}]}}

    segments = segments_from_timeline(
        timeline,
        scripts=scripts,
        transcripts={"v1": "Hello world"},
    )
    report = check_audio_quality(
        timeline,
        scripts=scripts,
        transcripts={"v1": "Hello world"},
    )

    assert len(segments) == 1
    assert segments[0].text == "Hello world"
    assert segments[0].transcript == "Hello world"
    assert [word.word for word in segments[0].words] == ["Hello", "world"]
    assert report.ok is True


def test_timeline_dialogue_maps_each_voice_to_its_speaker_line() -> None:
    timeline = _timeline(
        [
            {
                "id": "v1",
                "beat": "beat_0",
                "speaker": "narrator",
                "source": "audio/v1.mp3",
                "start": 0.0,
                "end": 1.0,
            },
            {
                "id": "v2",
                "beat": "beat_0",
                "speaker": "hero",
                "source": "audio/v2.mp3",
                "start": 1.0,
                "end": 2.0,
            },
        ]
    )
    scripts = {
        "en": {
            "beats": [
                {
                    "text": "Narration.",
                    "dialogue": [
                        {"speaker": "narrator", "text": "Narration."},
                        {"speaker": "hero", "text": "Hello."},
                    ],
                }
            ]
        }
    }

    segments = segments_from_timeline(timeline, scripts=scripts)

    assert [segment.text for segment in segments] == ["Narration.", "Hello."]


def test_thresholds_and_word_spans_are_validated() -> None:
    with pytest.raises(ValueError):
        AudioQualityChecker(max_silence_ratio=2.0)
    with pytest.raises(ValueError):
        TimedWord("word", start_seconds=2.0, end_seconds=1.0)
    with pytest.raises(ValueError):
        SilenceSpan(2.0, 1.0)

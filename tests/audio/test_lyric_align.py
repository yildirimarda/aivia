"""Written lyrics are aligned to a sung transcript with line and word timing."""

from __future__ import annotations

import asyncio

import pytest

from aivia.audio import (
    LyricAligner,
    LyricAlignment,
    LyricLine,
    LyricMismatch,
    LyricWord,
    align_lyrics,
    align_lyrics_audio,
    find_lyric_mismatches,
    lyric_word_similarity,
)
from aivia.core.errors import ValidationError
from aivia.providers.protocols import (
    STTRequest,
    STTResponse,
    WordTimestamp,
)
from aivia.writing import LyricsPayload, LyricsSection

AUDIO = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 20


def _word(text: str, start: float, end: float) -> WordTimestamp:
    return WordTimestamp(word=text, start_seconds=start, end_seconds=end)


def _response(*words: WordTimestamp, language: str = "en") -> STTResponse:
    return STTResponse(
        text=" ".join(word.word for word in words),
        language=language,
        duration_seconds=max((word.end_seconds for word in words), default=0.0),
        words=list(words),
        model="song-stt",
        provider="local-stt",
    )


def test_exact_lyrics_receive_line_and_word_timings() -> None:
    lyrics = "[Verse 1]\nHello world\nWe sing tonight\n\n[Chorus]\nCarry me home"
    transcript = _response(
        _word("Hello", 0.0, 0.4),
        _word("world", 0.5, 1.0),
        _word("We", 1.2, 1.3),
        _word("sing", 1.3, 1.7),
        _word("tonight", 1.7, 2.3),
        _word("Carry", 2.5, 2.9),
        _word("me", 2.9, 3.1),
        _word("home", 3.1, 3.6),
    )

    result = align_lyrics(lyrics, transcript)

    assert isinstance(result, LyricAlignment)
    assert [line.text for line in result.lines] == [
        "Hello world",
        "We sing tonight",
        "Carry me home",
    ]
    assert [line.section for line in result.lines] == [
        "verse 1",
        "verse 1",
        "chorus",
    ]
    assert [line.start_seconds for line in result.lines] == [0.0, 1.2, 2.5]
    assert [line.end_seconds for line in result.lines] == [1.0, 2.3, 3.6]
    assert result.lines[0].words[1].word == "world"
    assert result.lines[0].words[1].start_seconds == 0.5
    assert result.lines[0].words[1].end_seconds == 1.0
    assert all(isinstance(word, LyricWord) for word in result.words)
    assert [word.word for word in result.words] == [
        "Hello",
        "world",
        "We",
        "sing",
        "tonight",
        "Carry",
        "me",
        "home",
    ]
    assert result.coverage == 1.0
    assert result.line_coverage == 1.0
    assert result.mismatches == []
    assert result.ok is True
    assert result.passed is True
    assert result.has_timings is True
    assert result.language == "en"
    assert result.provider == "local-stt"
    assert result.model == "song-stt"
    assert result.duration_seconds == 3.6
    assert [word.word for word in result.word_timestamps] == [
        "Hello",
        "world",
        "We",
        "sing",
        "tonight",
        "Carry",
        "me",
        "home",
    ]
    assert [line.text for line in result.line_timestamps] == [
        "Hello world",
        "We sing tonight",
        "Carry me home",
    ]


def test_a_text_only_transcript_aligns_words_but_has_no_timings() -> None:
    result = align_lyrics("Hello world", {"text": "Hello world"})

    assert result.coverage == 1.0
    assert result.mismatches[0].code == "untimed_line"
    assert result.aligned is False
    assert result.has_timings is False
    assert result.word_timestamps == []
    assert result.line_timestamps == []


def test_sung_variants_and_adlibs_do_not_shift_later_lines() -> None:
    lyrics = "I love you\nWe sing tonight"
    transcript = _response(
        _word("I", 0.0, 0.2),
        _word("lo-o-ove", 0.2, 0.8),
        _word("you", 0.9, 1.2),
        _word("oh", 1.3, 1.4),
        _word("We", 1.5, 1.6),
        _word("si-i-ing", 1.6, 2.0),
        _word("toniiight", 2.0, 2.6),
    )

    result = align_lyrics(lyrics, transcript)

    assert lyric_word_similarity("lo-o-ove", "love") == 1.0
    assert lyric_word_similarity("toniiight", "tonight") == 1.0
    assert [word.matched for word in result.words] == [True] * 6
    assert result.words[1].transcript_word == "lo-o-ove"
    assert result.words[1].source == "exact"
    assert result.lines[0].start_seconds == 0.0
    assert result.lines[0].end_seconds == 1.2
    assert result.lines[1].start_seconds == 1.5
    assert result.lines[1].end_seconds == 2.6
    assert result.coverage == 1.0
    assert result.mismatches == []


def test_a_split_sung_word_is_merged_without_becoming_extra() -> None:
    lyrics = "I love you"
    transcript = _response(
        _word("I", 0.0, 0.2),
        _word("lo", 0.2, 0.5),
        _word("ove", 0.5, 0.9),
        _word("you", 0.9, 1.2),
    )

    result = align_lyrics(lyrics, transcript)

    assert [word.matched for word in result.words] == [True, True, True]
    assert result.words[1].transcript_word == "lo ove"
    assert result.words[1].start_seconds == 0.2
    assert result.words[1].end_seconds == 0.9
    assert result.words[1].metadata["merged"] is True
    assert result.mismatches == []


def test_a_repeated_chorus_maps_to_each_occurrence_in_order() -> None:
    lyrics = "[Chorus]\nHold on\n\n[Verse]\nNew day\n\n[Chorus]\nHold on"
    transcript = _response(
        _word("Hold", 0.0, 0.5),
        _word("on", 0.5, 1.0),
        _word("New", 2.0, 2.3),
        _word("day", 2.3, 2.7),
        _word("Hold", 3.0, 3.5),
        _word("on", 3.5, 4.0),
    )

    result = align_lyrics(lyrics, transcript)

    assert [line.text for line in result.lines] == [
        "Hold on",
        "New day",
        "Hold on",
    ]
    assert [line.start_seconds for line in result.lines] == [0.0, 2.0, 3.0]
    assert [line.end_seconds for line in result.lines] == [1.0, 2.7, 4.0]
    assert [line.section for line in result.lines] == ["chorus", "verse", "chorus"]
    assert result.mismatches == []


def test_missing_and_extra_words_are_reported_with_line_context() -> None:
    lyrics = "Hello brave world"
    transcript = _response(
        _word("Hello", 0.0, 0.4),
        _word("world", 0.8, 1.2),
        _word("loud", 1.3, 1.5),
    )

    result = align_lyrics(lyrics, transcript)

    assert [word.matched for word in result.words] == [True, False, True]
    assert result.words[1].start_seconds is None
    assert result.words[1].source == "unmatched"
    assert result.coverage == pytest.approx(2 / 3)
    assert result.lines[0].start_seconds == 0.0
    assert result.lines[0].end_seconds == 1.2

    missing = result.by_code("missing_word")
    extra = result.by_code("extra_word")
    assert len(missing) == 1
    assert missing[0].kind == "missing"
    assert missing[0].expected == "brave"
    assert missing[0].line_index == 0
    assert missing[0].line_word_index == 1
    assert len(extra) == 1
    assert extra[0].kind == "extra"
    assert extra[0].actual == "loud"
    assert extra[0].line_index == 0
    assert extra[0].severity.value == "info"
    assert result.passed is True


def test_fuzzy_substitutions_are_reported_and_threshold_is_configurable() -> None:
    transcript = _response(_word("Smith", 0.0, 0.5))

    matched = align_lyrics("Smyth", transcript)
    rejected = align_lyrics("Smyth", transcript, threshold=0.95)

    assert matched.words[0].matched is True
    assert matched.words[0].similarity == pytest.approx(0.8)
    assert matched.words[0].source == "fuzzy"
    substitution = matched.substitutions[0]
    assert isinstance(substitution, LyricMismatch)
    assert substitution.code == "mispronounced_word"
    assert substitution.expected == "Smyth"
    assert substitution.actual == "Smith"
    assert rejected.words[0].matched is False
    assert rejected.by_code("missing_word")[0].expected == "Smyth"


def test_low_confidence_words_are_reported_when_a_threshold_is_set() -> None:
    transcript = _response(
        WordTimestamp(word="Hello", start_seconds=0.0, end_seconds=0.4, confidence=0.2),
        WordTimestamp(word="world", start_seconds=0.4, end_seconds=0.8, confidence=0.9),
    )

    default = align_lyrics("Hello world", transcript)
    reported = align_lyrics("Hello world", transcript, min_confidence=0.5)
    reusable = LyricAligner(min_confidence=0.5)("Hello world", transcript)

    assert default.by_code("low_confidence") == []
    low = reported.by_code("low_confidence")
    assert len(low) == 1
    assert low[0].code == "low_confidence_word"
    assert low[0].expected == "Hello"
    assert low[0].actual == "Hello"
    assert low[0].line_index == 0
    assert low[0].line_word_index == 0
    assert low[0].metadata == {"confidence": 0.2, "min_confidence": 0.5}
    assert reported.passed is True
    assert [item.code for item in reusable.mismatches] == ["low_confidence_word"]


def test_a_line_below_the_coverage_threshold_is_reported_as_weak() -> None:
    transcript = _response(_word("Hello", 0.0, 0.4))

    default = align_lyrics("Hello world again", transcript)
    strict = align_lyrics("Hello world again", transcript, line_threshold=0.2)
    reusable = LyricAligner(line_threshold=0.5)("Hello world again", transcript)

    assert default.lines[0].coverage == pytest.approx(1 / 3)
    weak = default.by_code("weak_line")
    assert len(weak) == 1
    assert weak[0].kind == "line"
    assert weak[0].expected == "Hello world again"
    assert weak[0].metadata == {
        "coverage": pytest.approx(1 / 3),
        "line_threshold": 0.5,
    }
    assert default.weak_lines == default.lines
    assert default.line_mismatches == weak
    assert default.passed is True
    assert strict.by_code("weak_line") == []
    assert reusable.by_code("weak_line") == weak


def test_a_line_with_no_matches_is_untimed_and_can_be_interpolated() -> None:
    lyrics = "Hello world\nSilent line\nFinal words"
    transcript = _response(
        _word("Hello", 0.0, 0.5),
        _word("world", 0.5, 1.0),
        _word("Final", 2.0, 2.4),
        _word("words", 2.4, 2.8),
    )

    plain = align_lyrics(lyrics, transcript)
    interpolated = align_lyrics(lyrics, transcript, interpolate=True)

    assert plain.lines[1].start_seconds is None
    assert plain.lines[1].end_seconds is None
    assert plain.by_code("untimed_line")[0].expected == "Silent line"
    silent = interpolated.lines[1]
    assert silent.start_seconds == pytest.approx(1.0)
    assert silent.end_seconds == pytest.approx(2.0)
    assert silent.aligned is True
    assert all(word.interpolated for word in silent.words)
    assert all(word.matched is False for word in silent.words)
    assert not interpolated.by_code("untimed_line")


def test_lyrics_payload_and_mapping_transcripts_are_supported() -> None:
    lyrics = LyricsPayload(
        sections=[
            LyricsSection(name="verse", lines=["Hello world"]),
            LyricsSection(name="chorus", lines=["Sing it back"]),
        ]
    )
    transcript = {
        "text": "Hello world Sing it back",
        "language": "en",
        "words": [
            {"word": "Hello", "start": 0.0, "end": 0.4},
            {"word": "world", "start": 0.4, "end": 0.8},
            {"word": "Sing", "start": 1.0, "end": 1.3},
            {"word": "it", "start": 1.3, "end": 1.4},
            {"word": "back", "start": 1.4, "end": 1.8},
        ],
    }

    result = align_lyrics(lyrics, transcript)

    assert [line.section for line in result.lines] == ["verse", "chorus"]
    assert result.lines[1].start_seconds == 1.0
    assert result.lines[1].end_seconds == 1.8
    assert result.mismatches == []


def test_a_script_like_beat_list_is_accepted_when_payload_is_not_typed() -> None:
    lyrics = {
        "mode": "lyrics",
        "beats": [
            {"text": "First line"},
            {"text": "Second line"},
        ],
    }
    transcript = _response(
        _word("First", 0.0, 0.4),
        _word("line", 0.4, 0.8),
        _word("Second", 1.0, 1.4),
        _word("line", 1.4, 1.8),
    )

    result = align_lyrics(lyrics, transcript)

    assert [line.text for line in result.lines] == ["First line", "Second line"]
    assert result.coverage == 1.0


class FakeSTT:
    """A minimal speech to text provider that records the request."""

    name = "fake-stt"

    def __init__(self, response: STTResponse) -> None:
        self.response = response
        self.requests: list[STTRequest] = []

    async def transcribe(self, request: STTRequest) -> STTResponse:
        self.requests.append(request)
        return self.response


def test_align_lyrics_audio_transcribes_then_aligns() -> None:
    provider = FakeSTT(_response(_word("Hello", 0.0, 0.4), _word("world", 0.4, 0.8)))

    result = asyncio.run(
        align_lyrics_audio(
            "Hello world",
            AUDIO,
            provider,
            language="en",
            model="tiny",
        )
    )

    assert result.coverage == 1.0
    assert result.lines[0].start_seconds == 0.0
    assert result.lines[0].end_seconds == 0.8
    request = provider.requests[0]
    assert request.audio == AUDIO
    assert request.language == "en"
    assert request.model == "tiny"
    assert request.word_timestamps is True


def test_empty_lyrics_and_missing_transcripts_are_rejected() -> None:
    with pytest.raises(ValidationError):
        align_lyrics("   ", _response(_word("Hello", 0.0, 0.4)))
    with pytest.raises(ValidationError):
        align_lyrics("Hello", None)


def test_reusable_aligner_and_mismatch_helper_share_the_same_policy() -> None:
    transcript = _response(_word("Hello", 0.0, 0.4))
    aligner = LyricAligner(threshold=0.95)

    result = aligner("Hello world", transcript)
    mismatches = find_lyric_mismatches("Hello world", transcript, threshold=0.95)

    assert result.by_code("missing_word")[0].expected == "world"
    assert [mismatch.expected for mismatch in mismatches] == ["world"]
    assert aligner.threshold == 0.95
    assert isinstance(result.lines[0], LyricLine)

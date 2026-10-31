"""Known text is aligned to transcript timing and expected terms are verified."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from aivia.core.config.orders import NameVerifyConfig, OrdersConfig
from aivia.core.errors import ValidationError
from aivia.media.align_verify import (
    DEFAULT_ALIGNMENT_THRESHOLD,
    DEFAULT_TERM_THRESHOLD,
    AlignedWord,
    AlignmentResult,
    ForcedAligner,
    TermMatch,
    TermVerification,
    TermVerifier,
    align_audio,
    align_text,
    align_text_to_audio,
    align_word_timestamps,
    aligned_word_timestamps,
    check_terms,
    forced_align,
    phrase_similarity,
    term_matches,
    terms_match,
    verification_threshold,
    verify_term,
    verify_terms,
    word_similarity,
)
from aivia.providers.protocols import (
    STTRequest,
    STTResponse,
    TranscriptSegment,
    WordTimestamp,
)

AUDIO = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 20


def _word(text: str, start: float, end: float) -> WordTimestamp:
    return WordTimestamp(word=text, start_seconds=start, end_seconds=end)


def _response(*words: WordTimestamp) -> STTResponse:
    text = " ".join(word.word for word in words)
    return STTResponse(
        text=text,
        language="en",
        duration_seconds=max((word.end_seconds for word in words), default=0.0),
        words=list(words),
        model="tiny",
        provider="local-stt",
    )


def test_exact_script_words_receive_the_transcript_timings() -> None:
    transcript = _response(
        _word("Hello", 0.0, 0.4),
        _word("new", 0.5, 0.8),
        _word("world", 0.8, 1.2),
    )

    result = align_text("Hello brave new world", transcript)

    assert isinstance(result, AlignmentResult)
    assert [word.word for word in result.words] == [
        "Hello",
        "brave",
        "new",
        "world",
    ]
    assert [word.matched for word in result.words] == [True, False, True, True]
    brave = result.words[1]
    assert brave.start_seconds is None
    assert brave.end_seconds is None
    assert brave.source == "unmatched"
    assert result.words[2].start_seconds == 0.5
    assert result.words[2].end_seconds == 0.8
    assert result.words[0].exact is True
    assert result.coverage == 0.75
    assert result.has_timings is True
    assert [word.word for word in result.word_timestamps] == [
        "Hello",
        "new",
        "world",
    ]
    assert result.language == "en"
    assert result.provider == "local-stt"
    assert result.model == "tiny"
    assert result.duration_seconds == 1.2
    assert result.metadata == {"matched": 3, "unmatched": 1}


def test_fuzzy_matching_is_thresholded_and_configurable() -> None:
    transcript = _response(_word("Smith", 0.0, 0.5))

    matched = align_text("Smyth", transcript)
    rejected = align_text("Smyth", transcript, threshold=0.95)

    assert matched.words[0].matched is True
    assert matched.words[0].similarity == pytest.approx(0.8)
    assert matched.words[0].transcript_word == "Smith"
    assert matched.words[0].source == "fuzzy"
    assert rejected.words[0].matched is False
    assert rejected.words[0].start_seconds is None
    assert word_similarity("Café", "cafe") == 1.0
    assert phrase_similarity("Ada Lovelace", "ada loveless") > 0.8


def test_extra_transcript_words_do_not_shift_the_script_timing() -> None:
    transcript = _response(
        _word("um", 0.0, 0.1),
        _word("Hello", 0.1, 0.4),
        _word("brave", 0.4, 0.6),
        _word("world", 0.6, 1.0),
    )

    result = align_text("Hello world", transcript)

    assert [word.word for word in result.words] == ["Hello", "world"]
    assert result.words[0].start_seconds == 0.1
    assert result.words[1].start_seconds == 0.6
    assert result.words[1].end_seconds == 1.0
    assert result.coverage == 1.0


def test_missing_words_are_only_interpolated_when_requested() -> None:
    transcript = _response(
        _word("Hello", 0.0, 0.4),
        _word("world", 1.0, 1.4),
    )

    plain = align_text("Hello brave world", transcript)
    interpolated = align_text("Hello brave world", transcript, interpolate=True)

    assert plain.words[1].start_seconds is None
    assert plain.words[1].interpolated is False
    brave = interpolated.words[1]
    assert brave.start_seconds == pytest.approx(0.4)
    assert brave.end_seconds == pytest.approx(1.0)
    assert brave.interpolated is True
    assert brave.matched is False
    assert brave.source == "interpolated"
    assert [word.word for word in interpolated.word_timestamps] == [
        "Hello",
        "brave",
        "world",
    ]


def test_segments_and_plain_text_are_accepted_as_transcript_sources() -> None:
    words = [_word("Hello", 0.0, 0.4), _word("world", 0.4, 0.8)]
    response = STTResponse(
        text="Hello world",
        segments=[
            TranscriptSegment(
                text="Hello world",
                start_seconds=0.0,
                end_seconds=0.8,
                words=words,
            )
        ],
    )

    segmented = align_text("Hello world", response)
    plain = align_text("Hello world", "hello world")

    assert [word.start_seconds for word in segmented.word_timestamps] == [0.0, 0.4]
    assert all(word.matched for word in plain.words)
    assert plain.has_timings is False
    assert plain.words[0].start_seconds is None


def test_word_mappings_and_mapping_transcripts_are_supported() -> None:
    transcript: Mapping[str, Any] = {
        "text": "Hello world",
        "language": "en",
        "words": [
            {"word": "Hello", "start": 0.0, "end": 0.4},
            {"word": "world", "start": 0.4, "end": 0.8},
        ],
    }

    result = align_text("Hello world", transcript)

    assert result.language == "en"
    assert [word.start_seconds for word in result.word_timestamps] == [0.0, 0.4]


class FakeSTT:
    """A minimal speech to text provider that records the request."""

    name = "fake-stt"

    def __init__(self, response: STTResponse) -> None:
        self.response = response
        self.requests: list[STTRequest] = []

    async def transcribe(self, request: STTRequest) -> STTResponse:
        self.requests.append(request)
        return self.response


def test_align_audio_transcribes_with_word_timestamps_then_aligns() -> None:
    provider = FakeSTT(_response(_word("Hello", 0.0, 0.4), _word("world", 0.4, 0.8)))

    result = asyncio.run(
        align_audio(
            "Hello world",
            AUDIO,
            provider,
            language="en",
            model="tiny",
        )
    )

    request = provider.requests[0]
    assert request.audio == AUDIO
    assert request.language == "en"
    assert request.model == "tiny"
    assert request.word_timestamps is True
    assert [word.word for word in result.word_timestamps] == ["Hello", "world"]
    assert result.provider == "local-stt"


def test_forced_alignment_can_transcribe_audio_or_use_a_transcript() -> None:
    provider = FakeSTT(_response(_word("Hello", 0.0, 0.4)))

    from_audio = asyncio.run(
        forced_align("Hello", AUDIO, provider=provider)  # type: ignore[arg-type]
    )
    from_transcript = forced_align(
        "Hello",
        transcript=_response(_word("Hello", 1.0, 1.4)),
    )

    assert from_audio.words[0].start_seconds == 0.0
    assert from_transcript.words[0].start_seconds == 1.0
    assert align_text_to_audio is forced_align
    assert align_word_timestamps("Hello", from_transcript.transcript)[0].word == (
        "Hello"
    )
    assert (
        aligned_word_timestamps("Hello", from_transcript.transcript)[0].start_seconds
        == 1.0
    )


def test_term_verification_matches_fuzzy_terms_with_their_timing() -> None:
    transcript = _response(
        _word("Ada", 1.0, 1.2),
        _word("Lovelace", 1.2, 1.8),
        _word("and", 1.8, 1.9),
        _word("Grace", 1.9, 2.1),
        _word("Hooper", 2.1, 2.6),
    )

    verification = verify_terms(
        ["Ada Lovelace", "Grace Hopper", "Alan Turing"],
        transcript,
    )

    assert isinstance(verification, TermVerification)
    assert [match.term for match in verification] == [
        "Ada Lovelace",
        "Grace Hopper",
        "Alan Turing",
    ]
    assert verification.matched_terms == ("Ada Lovelace", "Grace Hopper")
    assert verification.missing_terms == ("Alan Turing",)
    assert verification.all_matched is False
    assert verification.verified is False
    assert verification.as_mapping() == {
        "Ada Lovelace": True,
        "Grace Hopper": True,
        "Alan Turing": False,
    }
    ada = verification["Ada Lovelace"]
    assert isinstance(ada, TermMatch)
    assert ada.matched is True
    assert ada.matched_text == "Ada Lovelace"
    assert ada.start_seconds == 1.0
    assert ada.end_seconds == 1.8
    assert ada.words[0].word == "Ada"
    assert verification.scores["Grace Hopper"] > DEFAULT_TERM_THRESHOLD
    assert verification.score == min(match.score for match in verification.terms)
    assert bool(verification) is False
    assert verification == False  # noqa: E712


def test_single_terms_plain_text_and_boolean_helpers() -> None:
    match = verify_term("ada", "We met ADA at the conference")
    text_only = verify_terms(
        ["ada", "turing"],
        STTResponse(text="We met ADA at the conference"),
    )

    assert match.matched is True
    assert match.score == 1.0
    assert match.start_seconds is None
    assert text_only.matched_terms == ("ada",)
    assert text_only.missing_terms == ("turing",)
    assert term_matches("ada", text="We met ADA") is True
    assert check_terms(["ada", "conference"], "We met ADA at the conference") is True
    assert terms_match(["ada", "turing"], "We met ADA") == {
        "ada": True,
        "turing": False,
    }
    assert word_similarity("", "") == 1.0
    assert bool(match) is True
    assert match == True  # noqa: E712


def test_term_verification_reads_the_configured_threshold() -> None:
    assert verification_threshold(None) == DEFAULT_TERM_THRESHOLD
    assert verification_threshold(NameVerifyConfig(threshold=0.9)) == 0.9
    assert verification_threshold({"orders": {"name_verify": {"threshold": 0.7}}}) == (
        0.7
    )
    assert (
        verification_threshold(
            OrdersConfig.from_mapping({"name_verify": {"threshold": 0.85}})
        )
        == 0.85
    )
    assert verification_threshold(0.75) == 0.75

    transcript = _response(_word("Smyth", 0.0, 0.5))
    with_config = verify_term(
        "Smith",
        transcript,
        config={"orders": {"name_verify": {"threshold": 0.95}}},
    )
    without_config = verify_term(
        "Smith",
        transcript,
        config=NameVerifyConfig(threshold=0.7),
    )

    assert with_config.matched is False
    assert without_config.matched is True
    assert DEFAULT_ALIGNMENT_THRESHOLD == 0.8


def test_reusable_aligner_and_verifier_keep_their_policy() -> None:
    transcript = _response(_word("Hello", 0.0, 0.4), _word("world", 0.4, 0.8))
    aligner = ForcedAligner(threshold=0.9, interpolate=True)
    verifier = TermVerifier(config=NameVerifyConfig(threshold=0.9))

    aligned = aligner("Hello brave world", transcript)

    assert aligner.threshold == 0.9
    assert aligner.interpolate is True
    assert aligned.words[1].interpolated is True
    assert verifier.threshold == 0.9
    assert verifier.matches("Hello", transcript) is True
    assert verifier.term("Hello", transcript).matched is True
    assert verifier.score("Hello", transcript) == 1.0
    assert verifier("Hello", transcript).all_matched is True


def test_require_all_and_invalid_inputs_raise_validation_errors() -> None:
    transcript = _response(_word("Hello", 0.0, 0.4))

    with pytest.raises(ValidationError, match="not found"):
        verify_terms(["Hello", "missing"], transcript, require_all=True)
    with pytest.raises(ValidationError, match="non-empty"):
        align_text("", transcript)
    with pytest.raises(ValidationError, match="between 0 and 1"):
        align_text("Hello", transcript, threshold=1.5)
    with pytest.raises(ValidationError, match="between 0 and 1"):
        verify_term("Hello", transcript, threshold=-0.1)
    with pytest.raises(ValidationError, match="non-empty"):
        verify_term("", transcript)
    with pytest.raises(ValidationError, match="transcript"):
        align_text("Hello", object())
    with pytest.raises(ValidationError, match="non-empty bytes"):
        asyncio.run(align_audio("Hello", b"", FakeSTT(transcript)))
    with pytest.raises(ValidationError, match="ordered"):
        align_text(
            "Hello",
            [_word("world", 1.0, 1.2), _word("Hello", 0.0, 0.4)],
        )
    with pytest.raises(ValidationError, match="between 0 and 1"):
        verification_threshold({"orders": {"name_verify": {"threshold": 2}}})


def test_aligned_word_aliases_and_validation() -> None:
    word = AlignedWord.model_validate(
        {
            "text": "Hello",
            "start": 0.0,
            "end": 0.4,
            "score": 1.0,
            "matched": True,
            "matched_word": "hello",
        }
    )

    assert word.word == "Hello"
    assert word.start == 0.0
    assert word.end == 0.4
    assert word.score == 1.0
    assert word.matched_word == "hello"
    assert word.duration_seconds == 0.4
    assert word.aligned is True

    with pytest.raises(ValueError, match="end_seconds"):
        AlignedWord(word="Hello", start_seconds=1.0, end_seconds=0.5)

"""Phonetic hints are passed to speech and music, and the name is verified.

The recipient's name is synthesized as speech and set to music with phonetic
hints, then transcribed. When the name does not survive transcription within
the configured threshold, speech and music are regenerated, up to the retry
limit.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from aivia.core.config.orders import NameVerifyConfig
from aivia.media.voices import PronunciationHint
from aivia.orders import NamePronunciationError, verify_name_pronunciation
from aivia.providers.protocols.music import MusicRequest, MusicResponse, MusicTrack
from aivia.providers.protocols.stt import STTResponse
from aivia.providers.protocols.tts import TTSRequest, TTSResponse


def _make_fakes() -> dict[str, Any]:
    speech_requests: list[TTSRequest] = []
    music_requests: list[MusicRequest] = []

    async def synthesize(request: TTSRequest) -> TTSResponse:
        speech_requests.append(request)
        return TTSResponse(audio=b"synthetic-speech")

    async def generate(request: MusicRequest) -> MusicResponse:
        music_requests.append(request)
        return MusicResponse(tracks=[MusicTrack(data=b"synthetic-music")])

    async def transcribe(audio: bytes) -> STTResponse:
        return STTResponse(text="")

    return {
        "speech_requests": speech_requests,
        "music_requests": music_requests,
        "synthesize": synthesize,
        "generate": generate,
        "transcribe": transcribe,
    }


def test_name_that_transcribes_cleanly_is_verified_on_first_attempt() -> None:
    fakes = _make_fakes()
    transcriptions = 0

    async def transcribe(audio: bytes) -> STTResponse:
        nonlocal transcriptions
        transcriptions += 1
        return STTResponse(text="Hello Ava, welcome to the show")

    result = asyncio.run(
        verify_name_pronunciation(
            name="Ava",
            hints={"Ava": "AY-vuh"},
            synthesize_speech=fakes["synthesize"],
            generate_music=fakes["generate"],
            transcribe=transcribe,
        )
    )

    assert result.verified is True
    assert result.attempts == 1
    assert result.regenerated == 0
    assert result.score == 1.0
    assert result.transcript == "Hello Ava, welcome to the show"
    assert result.reason is None
    assert fakes["speech_requests"][0].pronunciation == {"Ava": "AY-vuh"}
    assert fakes["music_requests"][0].pronunciation == {"Ava": "AY-vuh"}
    assert len(fakes["speech_requests"]) == 1
    assert len(fakes["music_requests"]) == 1
    assert transcriptions == 1


def test_hints_are_passed_to_speech_and_music_requests() -> None:
    fakes = _make_fakes()

    async def transcribe(audio: bytes) -> STTResponse:
        return STTResponse(text="Ava")

    result = asyncio.run(
        verify_name_pronunciation(
            name="Ava",
            hints={"Ava": "AY-vuh", "Song": "S ONG"},
            synthesize_speech=fakes["synthesize"],
            generate_music=fakes["generate"],
            transcribe=transcribe,
        )
    )

    assert result.hints == {"Ava": "AY-vuh", "Song": "S ONG"}
    assert result.speech_request.pronunciation == {"Ava": "AY-vuh", "Song": "S ONG"}
    assert result.music_request.pronunciation == {"Ava": "AY-vuh", "Song": "S ONG"}
    assert result.verified is True


def test_hints_accept_pronunciation_hint_objects() -> None:
    fakes = _make_fakes()

    async def transcribe(audio: bytes) -> STTResponse:
        return STTResponse(text="Siobahn")

    result = asyncio.run(
        verify_name_pronunciation(
            name="Siobahn",
            hints=[PronunciationHint(word="Siobahn", phonetic="shiv-awn")],
            synthesize_speech=fakes["synthesize"],
            generate_music=fakes["generate"],
            transcribe=transcribe,
        )
    )

    assert result.hints == {"Siobahn": "shiv-awn"}
    assert result.speech_request.pronunciation == {"Siobahn": "shiv-awn"}
    assert result.music_request.pronunciation == {"Siobahn": "shiv-awn"}
    assert result.verified is True


def test_hints_are_normalized_and_empty_hints_are_allowed() -> None:
    fakes = _make_fakes()

    async def transcribe(audio: bytes) -> STTResponse:
        return STTResponse(text="Ava")

    result = asyncio.run(
        verify_name_pronunciation(
            name="Ava",
            hints=None,
            synthesize_speech=fakes["synthesize"],
            generate_music=fakes["generate"],
            transcribe=transcribe,
        )
    )

    assert result.hints == {}
    assert result.speech_request.pronunciation == {}
    assert result.music_request.pronunciation == {}
    assert result.verified is True


def test_name_is_regenerated_up_to_the_retry_limit() -> None:
    fakes = _make_fakes()
    transcripts = iter(["the wrong person", "Ava is here"])

    async def transcribe(audio: bytes) -> STTResponse:
        return STTResponse(text=next(transcripts))

    result = asyncio.run(
        verify_name_pronunciation(
            name="Ava",
            hints={"Ava": "AY-vuh"},
            config=NameVerifyConfig(retries=1, threshold=0.8),
            synthesize_speech=fakes["synthesize"],
            generate_music=fakes["generate"],
            transcribe=transcribe,
        )
    )

    assert result.verified is True
    assert result.attempts == 2
    assert result.regenerated == 1
    assert result.score == 1.0
    assert len(fakes["speech_requests"]) == 2
    assert len(fakes["music_requests"]) == 2
    assert all(
        request.pronunciation == {"Ava": "AY-vuh"}
        for request in (*fakes["speech_requests"], *fakes["music_requests"])
    )


def test_name_never_matching_is_regenerated_then_refused() -> None:
    fakes = _make_fakes()

    async def transcribe(audio: bytes) -> STTResponse:
        return STTResponse(text="someone entirely different")

    with pytest.raises(NamePronunciationError, match="below the threshold"):
        asyncio.run(
            verify_name_pronunciation(
                name="Ava",
                hints={"Ava": "AY-vuh"},
                config=NameVerifyConfig(retries=1, threshold=0.8),
                synthesize_speech=fakes["synthesize"],
                generate_music=fakes["generate"],
                transcribe=transcribe,
            )
        )

    assert len(fakes["speech_requests"]) == 2
    assert len(fakes["music_requests"]) == 2


def test_zero_retries_fails_on_first_mismatch() -> None:
    fakes = _make_fakes()

    async def transcribe(audio: bytes) -> STTResponse:
        return STTResponse(text="the wrong person")

    with pytest.raises(NamePronunciationError):
        asyncio.run(
            verify_name_pronunciation(
                name="Ava",
                hints={"Ava": "AY-vuh"},
                config=NameVerifyConfig(retries=0, threshold=0.8),
                synthesize_speech=fakes["synthesize"],
                generate_music=fakes["generate"],
                transcribe=transcribe,
            )
        )

    assert len(fakes["speech_requests"]) == 1
    assert len(fakes["music_requests"]) == 1


def test_partial_name_match_below_threshold_is_not_verified() -> None:
    fakes = _make_fakes()

    async def transcribe(audio: bytes) -> STTResponse:
        return STTResponse(text="Ava")

    with pytest.raises(NamePronunciationError):
        asyncio.run(
            verify_name_pronunciation(
                name="Ava Marie",
                hints={"Ava": "AY-vuh", "Marie": "muh-REE"},
                config=NameVerifyConfig(retries=0, threshold=0.8),
                synthesize_speech=fakes["synthesize"],
                generate_music=fakes["generate"],
                transcribe=transcribe,
            )
        )


def test_disabled_verification_returns_without_calling_providers() -> None:
    fakes = _make_fakes()
    calls = {"transcribe": 0}

    async def transcribe(audio: bytes) -> STTResponse:
        calls["transcribe"] += 1
        return STTResponse(text="should not be called")

    result = asyncio.run(
        verify_name_pronunciation(
            name="Ava",
            hints={"Ava": "AY-vuh"},
            config=NameVerifyConfig(enabled=False),
            synthesize_speech=fakes["synthesize"],
            generate_music=fakes["generate"],
            transcribe=transcribe,
        )
    )

    assert result.verified is True
    assert result.attempts == 0
    assert result.regenerated == 0
    assert result.score == 0.0
    assert result.transcript == ""
    assert calls["transcribe"] == 0
    assert result.speech_request.pronunciation == {"Ava": "AY-vuh"}
    assert result.music_request.pronunciation == {"Ava": "AY-vuh"}


def test_empty_name_is_rejected() -> None:
    fakes = _make_fakes()

    async def transcribe(audio: bytes) -> STTResponse:
        return STTResponse(text="Ava")

    for name in ("", "   ", "\t\n"):
        with pytest.raises(ValueError, match="name must be non-empty"):
            asyncio.run(
                verify_name_pronunciation(
                    name=name,
                    synthesize_speech=fakes["synthesize"],
                    generate_music=fakes["generate"],
                    transcribe=transcribe,
                )
            )


def test_language_is_recorded_on_the_result() -> None:
    fakes = _make_fakes()

    async def transcribe(audio: bytes) -> STTResponse:
        return STTResponse(text="Ava")

    result = asyncio.run(
        verify_name_pronunciation(
            name="Ava",
            hints={"Ava": "AY-vuh"},
            language="de-AT",
            synthesize_speech=fakes["synthesize"],
            generate_music=fakes["generate"],
            transcribe=transcribe,
        )
    )

    assert result.language == "de-at"
    assert result.speech_request.language == "de-at"

"""Local faster-whisper speech to text adapter."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from aivia.core.errors import InvalidRequestError, ValidationError
from aivia.providers import provider_registry
from aivia.providers.protocols import (
    CAPABILITY_STT,
    STTRequest,
    STTResponse,
    TranscriptSegment,
    WordTimestamp,
)
from aivia.providers.stt import (
    DEFAULT_MODEL,
    FasterWhisper,
    FasterWhisperProvider,
    FasterWhisperSTT,
    LocalSTT,
    LocalSTTProvider,
    STTCostCalculator,
    WhisperLocal,
    WhisperLocalSTT,
)

AUDIO = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 20


def _response() -> STTResponse:
    words = [
        WordTimestamp(word="Hello", start_seconds=0.0, end_seconds=0.4),
        WordTimestamp(word="world", start_seconds=0.4, end_seconds=0.9),
    ]
    return STTResponse(
        text="Hello world",
        language="en",
        duration_seconds=12.0,
        segments=[
            TranscriptSegment(
                text="Hello world",
                start_seconds=0.0,
                end_seconds=1.0,
                words=words,
            )
        ],
        words=words,
        model="tiny",
        provider="local-stt",
    )


def test_metadata_and_registration_describe_the_provider() -> None:
    provider = LocalSTTProvider(transcriber=lambda audio, request, model: _response())

    assert provider.name == "local-stt"
    assert provider.capabilities == (CAPABILITY_STT,)
    assert provider.default_model == DEFAULT_MODEL
    assert provider.model == DEFAULT_MODEL
    assert provider.device == "auto"
    assert provider.compute_type == "default"
    assert provider.transcriber is not None
    assert FasterWhisper is LocalSTTProvider
    assert FasterWhisperSTT is LocalSTTProvider
    assert FasterWhisperProvider is LocalSTTProvider
    assert LocalSTT is LocalSTTProvider
    assert WhisperLocal is LocalSTTProvider
    assert WhisperLocalSTT is LocalSTTProvider
    assert provider_registry.resolve("local-stt") is LocalSTTProvider
    assert provider_registry.resolve("faster-whisper") is LocalSTTProvider
    assert provider_registry.resolve("local-whisper") is LocalSTTProvider


def test_an_injected_transcriber_receives_audio_request_and_model() -> None:
    calls: list[tuple[bytes, STTRequest, str]] = []

    def transcriber(audio: bytes, request: STTRequest, model: str) -> STTResponse:
        calls.append((audio, request, model))
        return _response()

    provider = LocalSTTProvider(transcriber=transcriber, default_model="tiny")
    request = STTRequest(
        audio=AUDIO,
        language="en",
        prompt="A greeting",
        word_timestamps=True,
    )

    response = asyncio.run(provider.transcribe(request))

    assert calls == [(AUDIO, request, "tiny")]
    assert response.text == "Hello world"
    assert response.duration_seconds == 12.0
    assert response.provider == "local-stt"
    assert response.model == "tiny"
    assert [word.word for word in response.all_words] == ["Hello", "world"]
    assert response.has_word_timestamps is True
    assert STTCostCalculator().cost_for(provider, 2.0).amount_usd == 0.0


def test_an_async_transcriber_and_a_mapping_result_are_supported() -> None:
    async def transcriber(
        audio: bytes,
        request: STTRequest,
        model: str,
    ) -> Mapping[str, Any]:
        return {
            "text": "mapped transcript",
            "language": "de",
            "duration_seconds": 60.0,
            "words": [
                {"word": "mapped", "start": 0.0, "end": 0.5},
                {"word": "transcript", "start": 0.5, "end": 1.0},
            ],
        }

    provider = LocalSTTProvider(transcriber=transcriber, model="small")

    response = asyncio.run(provider.transcribe(STTRequest(audio=AUDIO)))

    assert response.text == "mapped transcript"
    assert response.provider == "local-stt"
    assert response.model == "small"
    assert response.language == "de"
    assert response.duration_seconds == 60.0
    assert [word.word for word in response.words] == ["mapped", "transcript"]


def test_word_timestamps_can_be_disabled() -> None:
    def transcriber(audio: bytes, request: STTRequest, model: str) -> STTResponse:
        return _response()

    provider = LocalSTTProvider(transcriber=transcriber)

    response = asyncio.run(
        provider.transcribe(STTRequest(audio=AUDIO, word_timestamps=False))
    )

    assert response.words == []
    assert response.segments[0].words == []
    assert response.has_word_timestamps is False


def test_diarization_is_rejected() -> None:
    with pytest.raises(InvalidRequestError, match="diarization"):
        asyncio.run(
            LocalSTTProvider(
                transcriber=lambda audio, request, model: _response()
            ).transcribe(STTRequest(audio=AUDIO, diarize=True))
        )


def test_an_invalid_transcriber_result_is_rejected() -> None:
    def transcriber(audio: bytes, request: STTRequest, model: str) -> Any:
        return "not a result"

    provider = LocalSTTProvider(transcriber=transcriber)

    with pytest.raises(ValidationError, match="invalid result"):
        asyncio.run(provider.transcribe(STTRequest(audio=AUDIO)))

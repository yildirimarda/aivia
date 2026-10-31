"""OpenAI speech to text adapter."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from aivia.core.errors import InvalidRequestError, ValidationError
from aivia.providers import provider_registry
from aivia.providers.protocols import CAPABILITY_STT, STTRequest
from aivia.providers.stt import (
    OPENAI_DEFAULT_BASE_URL,
    OPENAI_STT_BASE_URL,
    OpenAISpeechToText,
    OpenAISpeechToTextProvider,
    OpenAISTT,
    OpenAISTTProvider,
    OpenAITranscription,
    OpenAITranscriptionProvider,
    STTCostCalculator,
    Whisper,
    WhisperAPI,
    WhisperAPIProvider,
    WhisperSTT,
    WhisperSTTProvider,
)

API_KEY = "test-key"
MODEL = "whisper-1"
AUDIO = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 20

TRANSCRIPT: dict[str, Any] = {
    "task": "transcribe",
    "language": "english",
    "duration": 90.0,
    "text": "Hello there friend",
    "words": [
        {"word": "Hello", "start": 0.0, "end": 0.4},
        {"word": "there", "start": 0.4, "end": 0.8},
        {"word": "friend", "start": 0.8, "end": 1.2},
    ],
    "segments": [
        {
            "id": 0,
            "start": 0.0,
            "end": 1.2,
            "text": "Hello there friend",
        }
    ],
}


def _provider(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> OpenAISTTProvider:
    options: dict[str, Any] = {
        "api_key": API_KEY,
        "default_model": MODEL,
        "transport": httpx.MockTransport(handler),
    }
    options.update(kwargs)
    return OpenAISTTProvider(**options)


def _capture(
    response: httpx.Response,
) -> tuple[list[httpx.Request], Callable[[httpx.Request], httpx.Response]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response

    return requests, handler


def test_metadata_and_registration_describe_the_provider() -> None:
    provider = _provider(lambda request: httpx.Response(200, json=TRANSCRIPT))

    assert provider.name == "openai-stt"
    assert provider.capabilities == (CAPABILITY_STT,)
    assert provider.base_url == OPENAI_DEFAULT_BASE_URL
    assert provider.api_url == OPENAI_DEFAULT_BASE_URL
    assert provider.default_model == MODEL
    assert provider.model == MODEL
    assert provider.endpoint == (f"{OPENAI_DEFAULT_BASE_URL}/v1/audio/transcriptions")
    assert OPENAI_STT_BASE_URL == OPENAI_DEFAULT_BASE_URL
    assert OpenAISTT is OpenAISTTProvider
    assert OpenAITranscription is OpenAISTTProvider
    assert OpenAITranscriptionProvider is OpenAISTTProvider
    assert OpenAISpeechToText is OpenAISTTProvider
    assert OpenAISpeechToTextProvider is OpenAISTTProvider
    assert Whisper is OpenAISTTProvider
    assert WhisperSTT is OpenAISTTProvider
    assert WhisperSTTProvider is OpenAISTTProvider
    assert WhisperAPI is OpenAISTTProvider
    assert WhisperAPIProvider is OpenAISTTProvider
    assert provider_registry.resolve("openai-stt") is OpenAISTTProvider
    assert provider_registry.resolve("openai-whisper") is OpenAISTTProvider
    assert provider_registry.resolve("whisper") is OpenAISTTProvider


def test_transcribe_uploads_audio_and_returns_word_timing() -> None:
    requests, handler = _capture(
        httpx.Response(
            200,
            json=TRANSCRIPT,
            headers={"x-request-id": "req-9"},
        )
    )
    provider = _provider(handler)
    request = STTRequest(
        audio=AUDIO,
        filename="clip.wav",
        media_type="audio/wav",
        language="en",
        prompt="A greeting",
        temperature=0.2,
        word_timestamps=True,
    )

    response = asyncio.run(provider.transcribe(request))

    submitted = requests[0]
    assert str(submitted.url) == (f"{OPENAI_DEFAULT_BASE_URL}/v1/audio/transcriptions")
    assert submitted.headers["authorization"] == f"Bearer {API_KEY}"
    assert submitted.headers["content-type"].startswith("multipart/form-data")
    content = submitted.content
    assert b'name="model"' in content
    assert MODEL.encode() in content
    assert b'name="response_format"' in content
    assert b"verbose_json" in content
    assert b'name="timestamp_granularities[]"' in content
    assert b"word" in content
    assert b"segment" in content
    assert b'name="language"' in content
    assert b'name="prompt"' in content
    assert b'name="temperature"' in content
    assert b'filename="clip.wav"' in content
    assert b"audio/wav" in content
    assert response.text == "Hello there friend"
    assert response.language == "english"
    assert response.duration_seconds == 90.0
    assert response.model == MODEL
    assert response.provider == "openai-stt"
    assert response.request_id == "req-9"
    assert [word.word for word in response.words] == ["Hello", "there", "friend"]
    assert [segment.text for segment in response.segments] == ["Hello there friend"]
    assert response.metadata["words"] == 3
    assert response.metadata["duration_seconds"] == 90.0
    assert STTCostCalculator().cost_for(provider, 1.5).amount_usd == pytest.approx(
        0.009
    )


def test_word_timestamps_can_be_disabled() -> None:
    requests, handler = _capture(httpx.Response(200, json=TRANSCRIPT))
    provider = _provider(handler)

    response = asyncio.run(
        provider.transcribe(STTRequest(audio=AUDIO, word_timestamps=False))
    )

    content = requests[0].content
    assert b'name="timestamp_granularities[]"' in content
    assert b"word" not in content
    assert response.words == []
    assert response.segments[0].words == []
    assert response.has_word_timestamps is False


def test_an_audio_url_is_fetched_before_the_transcription() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, content=AUDIO)
        return httpx.Response(200, json=TRANSCRIPT)

    provider = _provider(handler)

    response = asyncio.run(
        provider.transcribe(
            STTRequest(
                audio_url="https://example.com/audio/clip.wav",
                media_type="audio/wav",
            )
        )
    )

    assert [request.method for request in requests] == ["GET", "POST"]
    assert requests[0].url.path == "/audio/clip.wav"
    assert b'filename="audio.wav"' in requests[1].content
    assert response.text == "Hello there friend"


def test_a_missing_model_or_diarization_is_rejected() -> None:
    provider = _provider(
        lambda request: httpx.Response(200, json=TRANSCRIPT),
        default_model=None,
    )

    with pytest.raises(InvalidRequestError, match="model"):
        asyncio.run(provider.transcribe(STTRequest(audio=AUDIO)))
    with pytest.raises(InvalidRequestError, match="diarization"):
        asyncio.run(
            _provider(lambda request: httpx.Response(200, json=TRANSCRIPT)).transcribe(
                STTRequest(audio=AUDIO, diarize=True)
            )
        )


def test_non_json_and_empty_transcripts_are_rejected() -> None:
    provider = _provider(lambda request: httpx.Response(200, content=b"nope"))

    with pytest.raises(ValidationError, match="non-JSON"):
        asyncio.run(provider.transcribe(STTRequest(audio=AUDIO)))

    empty = _provider(
        lambda request: httpx.Response(
            200,
            json={"text": "", "duration": 1.0, "words": [], "segments": []},
        )
    )
    with pytest.raises(ValidationError, match="empty transcript"):
        asyncio.run(empty.transcribe(STTRequest(audio=AUDIO)))

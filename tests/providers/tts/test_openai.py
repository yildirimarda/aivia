"""OpenAI text to speech adapter."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from aivia.core.errors import InvalidRequestError
from aivia.providers import provider_registry
from aivia.providers.protocols import CAPABILITY_TTS, AudioFormat, TTSRequest
from aivia.providers.tts import (
    OPENAI_BASE_URL,
    OPENAI_DEFAULT_BASE_URL,
    OpenAI,
    OpenAISpeech,
    OpenAISpeechProvider,
    OpenAITTS,
    OpenAITTSProvider,
)
from aivia.providers.tts.openai import DEFAULT_BASE_URL

API_KEY = "test-key"
MODEL = "gpt-4o-mini-tts"
VOICE = "alloy"
MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00audio"


def _provider(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> OpenAITTSProvider:
    options: dict[str, Any] = {
        "api_key": API_KEY,
        "default_model": MODEL,
        "voice_map": {"host": VOICE},
        "transport": httpx.MockTransport(handler),
    }
    options.update(kwargs)
    return OpenAITTSProvider(**options)


def _capture(
    response: httpx.Response,
) -> tuple[list[httpx.Request], Callable[[httpx.Request], httpx.Response]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response

    return requests, handler


def test_metadata_and_registration_describe_the_provider() -> None:
    provider = _provider(lambda request: httpx.Response(200, content=MP3))

    assert provider.name == "openai-tts"
    assert provider.capabilities == (CAPABILITY_TTS,)
    assert provider.base_url == DEFAULT_BASE_URL
    assert provider.api_url == DEFAULT_BASE_URL
    assert provider.default_model == MODEL
    assert provider.model == MODEL
    assert provider.voice_map.resolve("host") == VOICE
    assert provider.endpoint == f"{DEFAULT_BASE_URL}/v1/audio/speech"
    assert OPENAI_DEFAULT_BASE_URL == DEFAULT_BASE_URL
    assert OPENAI_BASE_URL == DEFAULT_BASE_URL
    assert OpenAI is OpenAITTSProvider
    assert OpenAITTS is OpenAITTSProvider
    assert OpenAISpeech is OpenAITTSProvider
    assert OpenAISpeechProvider is OpenAITTSProvider
    assert provider_registry.resolve("openai-tts") is OpenAITTSProvider
    assert provider_registry.resolve("openai_speech") is OpenAITTSProvider


def test_synthesize_submits_the_speech_payload_and_returns_audio() -> None:
    requests, handler = _capture(
        httpx.Response(200, content=MP3, headers={"x-request-id": "req-2"})
    )
    provider = _provider(handler)
    request = TTSRequest(
        text="  Hello there  ",
        voice="host",
        format=AudioFormat.MP3,
        speed=1.25,
        instructions="Speak calmly",
    )

    response = asyncio.run(provider.synthesize(request))

    submitted = requests[0]
    assert str(submitted.url) == f"{DEFAULT_BASE_URL}/v1/audio/speech"
    assert submitted.headers["authorization"] == f"Bearer {API_KEY}"
    assert json.loads(submitted.content) == {
        "model": MODEL,
        "input": "Hello there",
        "voice": VOICE,
        "response_format": "mp3",
        "speed": 1.25,
        "instructions": "Speak calmly",
    }
    assert response.audio == MP3
    assert response.format is AudioFormat.MP3
    assert response.voice == VOICE
    assert response.model == MODEL
    assert response.provider == "openai-tts"
    assert response.request_id == "req-2"
    assert response.metadata["characters"] == len("Hello there")


def test_wav_format_is_requested_and_detected() -> None:
    wav = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 20
    requests, handler = _capture(httpx.Response(200, content=wav))
    provider = _provider(handler)

    response = asyncio.run(
        provider.synthesize(TTSRequest(text="hi", voice=VOICE, format=AudioFormat.WAV))
    )

    assert json.loads(requests[0].content)["response_format"] == "wav"
    assert response.format is AudioFormat.WAV


def test_opus_is_requested_for_ogg_and_m4a_maps_to_aac() -> None:
    requests, handler = _capture(httpx.Response(200, content=MP3))
    provider = _provider(handler)

    asyncio.run(
        provider.synthesize(TTSRequest(text="hi", voice=VOICE, format=AudioFormat.OGG))
    )
    asyncio.run(
        provider.synthesize(TTSRequest(text="hi", voice=VOICE, format=AudioFormat.M4A))
    )

    assert json.loads(requests[0].content)["response_format"] == "opus"
    assert json.loads(requests[1].content)["response_format"] == "aac"


def test_a_missing_model_is_rejected() -> None:
    provider = _provider(
        lambda request: httpx.Response(200, content=MP3),
        default_model=None,
    )

    with pytest.raises(InvalidRequestError, match="model"):
        asyncio.run(provider.synthesize(TTSRequest(text="hi", voice=VOICE)))

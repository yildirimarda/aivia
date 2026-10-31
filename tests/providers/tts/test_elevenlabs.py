"""ElevenLabs text to speech adapter."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from aivia.core.errors import InvalidRequestError
from aivia.providers import provider_registry
from aivia.providers.protocols import CAPABILITY_TTS, AudioFormat, TTSRequest
from aivia.providers.tts import (
    ELEVENLABS_BASE_URL,
    ELEVENLABS_DEFAULT_BASE_URL,
    ElevenLabs,
    ElevenLabsProvider,
    ElevenLabsTTS,
    ElevenLabsTTSProvider,
)
from aivia.providers.tts.elevenlabs import DEFAULT_BASE_URL

API_KEY = "test-key"
MODEL = "eleven_multilingual_v2"
VOICE = "voice-1"
MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00audio"
PCM = b"\x01\x02" * 64


def _provider(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> ElevenLabsTTSProvider:
    options: dict[str, Any] = {
        "api_key": API_KEY,
        "default_model": MODEL,
        "voice_map": {"narrator": VOICE},
        "transport": httpx.MockTransport(handler),
    }
    options.update(kwargs)
    return ElevenLabsTTSProvider(**options)


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

    assert provider.name == "elevenlabs-tts"
    assert provider.capabilities == (CAPABILITY_TTS,)
    assert provider.base_url == DEFAULT_BASE_URL
    assert provider.api_url == DEFAULT_BASE_URL
    assert provider.default_model == MODEL
    assert provider.model == MODEL
    assert provider.voice_map.resolve("narrator") == VOICE
    assert provider.endpoint_for(VOICE) == (
        f"{DEFAULT_BASE_URL}/v1/text-to-speech/{VOICE}"
    )
    assert provider.endpoint_for(VOICE, timestamps=True) == (
        f"{DEFAULT_BASE_URL}/v1/text-to-speech/{VOICE}/with-timestamps"
    )
    assert ELEVENLABS_DEFAULT_BASE_URL == DEFAULT_BASE_URL
    assert ELEVENLABS_BASE_URL == DEFAULT_BASE_URL
    assert ElevenLabs is ElevenLabsTTSProvider
    assert ElevenLabsTTS is ElevenLabsTTSProvider
    assert ElevenLabsProvider is ElevenLabsTTSProvider
    assert provider_registry.resolve("elevenlabs-tts") is ElevenLabsTTSProvider
    assert provider_registry.resolve("elevenlabs") is ElevenLabsTTSProvider


def test_synthesize_submits_text_voice_and_model_and_returns_audio() -> None:
    requests, handler = _capture(
        httpx.Response(200, content=MP3, headers={"x-request-id": "req-1"})
    )
    provider = _provider(handler)
    request = TTSRequest(
        text="  Hello world  ",
        voice="narrator",
        language="en",
        format=AudioFormat.MP3,
        speed=1.2,
        metadata={"stability": 0.5, "use_speaker_boost": True},
    )

    response = asyncio.run(provider.synthesize(request))

    submitted = requests[0]
    assert str(submitted.url).startswith(
        f"{DEFAULT_BASE_URL}/v1/text-to-speech/{VOICE}?"
    )
    assert submitted.url.params["output_format"] == "mp3_44100_128"
    assert submitted.headers["xi-api-key"] == API_KEY
    assert submitted.headers["accept"] == "audio/mpeg"
    assert json.loads(submitted.content) == {
        "text": "Hello world",
        "model_id": MODEL,
        "language_code": "en",
        "voice_settings": {
            "stability": 0.5,
            "use_speaker_boost": True,
            "speed": 1.2,
        },
    }
    assert response.audio == MP3
    assert response.format is AudioFormat.MP3
    assert response.voice == VOICE
    assert response.model == MODEL
    assert response.provider == "elevenlabs-tts"
    assert response.request_id == "req-1"
    assert response.metadata["characters"] == len("Hello world")


def test_wav_output_wraps_pcm_in_a_wave_container() -> None:
    requests, handler = _capture(httpx.Response(200, content=PCM))
    provider = _provider(handler)

    response = asyncio.run(
        provider.synthesize(
            TTSRequest(
                text="hello",
                voice=VOICE,
                format=AudioFormat.WAV,
                sample_rate=16000,
            )
        )
    )

    assert requests[0].url.params["output_format"] == "pcm_16000"
    assert response.format is AudioFormat.WAV
    assert response.sample_rate == 16000
    assert response.audio.startswith(b"RIFF")
    assert response.audio[8:12] == b"WAVE"


def test_word_timestamps_are_converted_from_character_alignment() -> None:
    alignment = {
        "characters": list("Hi all"),
        "character_start_times_seconds": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "character_end_times_seconds": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    }
    body = {
        "audio_base64": base64.b64encode(MP3).decode("ascii"),
        "alignment": alignment,
    }
    requests, handler = _capture(httpx.Response(200, json=body))
    provider = _provider(handler)

    response = asyncio.run(
        provider.synthesize(
            TTSRequest(
                text="Hi all",
                voice=VOICE,
                format=AudioFormat.MP3,
                word_timestamps=True,
            )
        )
    )

    assert requests[0].url.path.endswith("/with-timestamps")
    assert [word.word for word in response.word_timestamps] == ["Hi", "all"]
    assert response.word_timestamps[0].start_seconds == 0.0
    assert response.word_timestamps[1].start_seconds == 0.3


def test_a_missing_model_or_unsupported_format_is_rejected() -> None:
    provider = _provider(
        lambda request: httpx.Response(200, content=MP3),
        default_model=None,
    )
    with pytest.raises(InvalidRequestError, match="model"):
        asyncio.run(provider.synthesize(TTSRequest(text="hi", voice=VOICE)))

    provider = _provider(lambda request: httpx.Response(200, content=MP3))
    with pytest.raises(InvalidRequestError, match="does not support"):
        asyncio.run(
            provider.synthesize(
                TTSRequest(text="hi", voice=VOICE, format=AudioFormat.FLAC)
            )
        )

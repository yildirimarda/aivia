"""Google Cloud text to speech adapter."""

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
    GOOGLE_API_URL,
    GOOGLE_BASE_URL,
    GOOGLE_DEFAULT_BASE_URL,
    GOOGLE_TTS_API_URL,
    Google,
    GoogleCloudTTS,
    GoogleCloudTTSProvider,
    GoogleProvider,
    GoogleTTS,
    GoogleTTSProvider,
)
from aivia.providers.tts.google import DEFAULT_API_VERSION, DEFAULT_BASE_URL

API_KEY = "test-key"
VOICE = "en-US-Neural2-C"
MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00audio"
PCM = b"\x01\x02" * 64


def _provider(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> GoogleTTSProvider:
    options: dict[str, Any] = {
        "api_key": API_KEY,
        "voice_map": {"narrator": VOICE},
        "transport": httpx.MockTransport(handler),
    }
    options.update(kwargs)
    return GoogleTTSProvider(**options)


def _capture(
    response: httpx.Response,
) -> tuple[list[httpx.Request], Callable[[httpx.Request], httpx.Response]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response

    return requests, handler


def _body(audio: bytes) -> dict[str, str]:
    return {"audioContent": base64.b64encode(audio).decode("ascii")}


def test_metadata_and_registration_describe_the_provider() -> None:
    provider = _provider(lambda request: httpx.Response(200, json=_body(MP3)))

    assert provider.name == "google-tts"
    assert provider.capabilities == (CAPABILITY_TTS,)
    assert provider.base_url == DEFAULT_BASE_URL
    assert provider.api_url == DEFAULT_BASE_URL
    assert provider.api_version == DEFAULT_API_VERSION
    assert provider.voice_map.resolve("narrator") == VOICE
    assert provider.endpoint == (
        f"{DEFAULT_BASE_URL}/{DEFAULT_API_VERSION}/text:synthesize"
    )
    assert GOOGLE_DEFAULT_BASE_URL == DEFAULT_BASE_URL
    assert GOOGLE_API_URL == DEFAULT_BASE_URL
    assert GOOGLE_BASE_URL == DEFAULT_BASE_URL
    assert GOOGLE_TTS_API_URL == DEFAULT_BASE_URL
    assert Google is GoogleTTSProvider
    assert GoogleTTS is GoogleTTSProvider
    assert GoogleCloudTTS is GoogleTTSProvider
    assert GoogleCloudTTSProvider is GoogleTTSProvider
    assert GoogleProvider is GoogleTTSProvider
    assert provider_registry.resolve("google-tts") is GoogleTTSProvider
    assert provider_registry.resolve("google_tts") is GoogleTTSProvider


def test_synthesize_infers_language_and_wraps_pcm_for_wav() -> None:
    requests, handler = _capture(
        httpx.Response(
            200,
            json=_body(PCM),
            headers={"x-goog-request-id": "req-3"},
        )
    )
    provider = _provider(handler)

    response = asyncio.run(
        provider.synthesize(
            TTSRequest(
                text="  Hello world  ",
                voice="narrator",
                format=AudioFormat.WAV,
                sample_rate=16000,
                speed=0.9,
            )
        )
    )

    submitted = requests[0]
    assert str(submitted.url) == (
        f"{DEFAULT_BASE_URL}/{DEFAULT_API_VERSION}/text:synthesize"
    )
    assert submitted.headers["x-goog-api-key"] == API_KEY
    assert json.loads(submitted.content) == {
        "input": {"text": "Hello world"},
        "voice": {"languageCode": "en-US", "name": VOICE},
        "audioConfig": {
            "audioEncoding": "LINEAR16",
            "speakingRate": 0.9,
            "sampleRateHertz": 16000,
        },
    }
    assert response.format is AudioFormat.WAV
    assert response.sample_rate == 16000
    assert response.audio.startswith(b"RIFF")
    assert response.voice == VOICE
    assert response.provider == "google-tts"
    assert response.request_id == "req-3"
    assert response.metadata["characters"] == len("Hello world")
    assert response.metadata["language"] == "en-US"


def test_mp3_and_ogg_encodings_are_selected() -> None:
    requests, handler = _capture(httpx.Response(200, json=_body(MP3)))
    provider = _provider(handler)

    asyncio.run(
        provider.synthesize(
            TTSRequest(
                text="hi",
                voice=VOICE,
                language="de-DE",
                format=AudioFormat.MP3,
            )
        )
    )
    asyncio.run(
        provider.synthesize(TTSRequest(text="hi", voice=VOICE, format=AudioFormat.OGG))
    )

    first = json.loads(requests[0].content)
    second = json.loads(requests[1].content)
    assert first["audioConfig"]["audioEncoding"] == "MP3"
    assert first["voice"]["languageCode"] == "de-DE"
    assert second["audioConfig"]["audioEncoding"] == "OGG_OPUS"


def test_a_voice_without_a_language_and_no_language_request_is_rejected() -> None:
    provider = _provider(lambda request: httpx.Response(200, json=_body(MP3)))

    with pytest.raises(InvalidRequestError, match="language"):
        asyncio.run(
            provider.synthesize(
                TTSRequest(text="hi", voice="custom-voice", format=AudioFormat.MP3)
            )
        )


def test_an_unsupported_format_is_rejected() -> None:
    provider = _provider(lambda request: httpx.Response(200, json=_body(MP3)))

    with pytest.raises(InvalidRequestError, match="does not support"):
        asyncio.run(
            provider.synthesize(
                TTSRequest(text="hi", voice=VOICE, format=AudioFormat.FLAC)
            )
        )

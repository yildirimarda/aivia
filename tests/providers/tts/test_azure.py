"""Azure Speech text to speech adapter."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from aivia.core.errors import ConfigError, InvalidRequestError
from aivia.providers import provider_registry
from aivia.providers.protocols import CAPABILITY_TTS, AudioFormat, TTSRequest
from aivia.providers.tts import (
    AZURE_TTS_API_VERSION,
    Azure,
    AzureProvider,
    AzureSpeech,
    AzureSpeechProvider,
    AzureTTS,
    AzureTTSProvider,
)

API_KEY = "test-key"
REGION = "westeurope"
VOICE = "en-US-JennyNeural"
MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00audio"
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 20


def _provider(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> AzureTTSProvider:
    options: dict[str, Any] = {
        "api_key": API_KEY,
        "region": REGION,
        "voice_map": {"host": VOICE},
        "transport": httpx.MockTransport(handler),
    }
    options.update(kwargs)
    return AzureTTSProvider(**options)


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

    assert provider.name == "azure-tts"
    assert provider.capabilities == (CAPABILITY_TTS,)
    assert provider.base_url == f"https://{REGION}.tts.speech.microsoft.com"
    assert provider.api_url == provider.base_url
    assert provider.voice_map.resolve("host") == VOICE
    assert provider.endpoint == f"{provider.base_url}/cognitiveservices/v1"
    assert AZURE_TTS_API_VERSION == "1.0"
    assert Azure is AzureTTSProvider
    assert AzureTTS is AzureTTSProvider
    assert AzureSpeech is AzureTTSProvider
    assert AzureSpeechProvider is AzureTTSProvider
    assert AzureProvider is AzureTTSProvider
    assert provider_registry.resolve("azure-tts") is AzureTTSProvider
    assert provider_registry.resolve("azure_speech") is AzureTTSProvider


def test_synthesize_builds_ssml_and_returns_audio() -> None:
    requests, handler = _capture(
        httpx.Response(200, content=MP3, headers={"x-requestid": "req-4"})
    )
    provider = _provider(handler)
    request = TTSRequest(
        text="Tom & Jerry <3",
        voice="host",
        format=AudioFormat.MP3,
        speed=1.5,
    )

    response = asyncio.run(provider.synthesize(request))

    submitted = requests[0]
    assert str(submitted.url) == f"{provider.base_url}/cognitiveservices/v1"
    assert submitted.headers["ocp-apim-subscription-key"] == API_KEY
    assert submitted.headers["content-type"] == "application/ssml+xml"
    assert submitted.headers["x-microsoft-outputformat"] == (
        "audio-24khz-96kbitrate-mono-mp3"
    )
    ssml = submitted.content.decode("utf-8")
    assert 'xml:lang="en-US"' in ssml
    assert f'<voice name="{VOICE}">' in ssml
    assert 'rate="+50%"' in ssml
    assert "Tom &amp; Jerry &lt;3" in ssml
    assert response.audio == MP3
    assert response.format is AudioFormat.MP3
    assert response.voice == VOICE
    assert response.provider == "azure-tts"
    assert response.request_id == "req-4"
    assert response.metadata["characters"] == len("Tom & Jerry <3")
    assert response.metadata["language"] == "en-US"


def test_wav_output_and_an_explicit_language_are_used() -> None:
    requests, handler = _capture(httpx.Response(200, content=WAV))
    provider = _provider(handler)

    response = asyncio.run(
        provider.synthesize(
            TTSRequest(
                text="hallo",
                voice=VOICE,
                language="de-DE",
                format=AudioFormat.WAV,
            )
        )
    )

    assert requests[0].headers["x-microsoft-outputformat"] == (
        "riff-24khz-16bit-mono-pcm"
    )
    assert 'xml:lang="de-DE"' in requests[0].content.decode("utf-8")
    assert response.format is AudioFormat.WAV
    assert response.sample_rate == 24000


def test_a_base_url_can_replace_the_region() -> None:
    provider = _provider(
        lambda request: httpx.Response(200, content=MP3),
        region=None,
        base_url="https://speech.example.test",
    )

    assert provider.base_url == "https://speech.example.test"


def test_a_missing_region_and_base_url_is_rejected() -> None:
    with pytest.raises(ConfigError, match="region or base_url"):
        AzureTTSProvider(api_key=API_KEY)


def test_an_invalid_region_is_rejected() -> None:
    with pytest.raises(ConfigError, match="invalid Azure region"):
        AzureTTSProvider(api_key=API_KEY, region="../bad")


def test_an_unsupported_format_is_rejected() -> None:
    provider = _provider(lambda request: httpx.Response(200, content=MP3))

    with pytest.raises(InvalidRequestError, match="does not support"):
        asyncio.run(
            provider.synthesize(
                TTSRequest(text="hi", voice=VOICE, format=AudioFormat.FLAC)
            )
        )

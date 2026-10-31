"""Stable Audio instrumental adapter and its licensing metadata."""

from __future__ import annotations

import asyncio
import io
import wave
from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest

from aivia.core.errors import ConfigError, InvalidRequestError, ValidationError
from aivia.providers import provider_registry
from aivia.providers.music import (
    STABLE_AUDIO_BASE_URL,
    STABLE_AUDIO_DEFAULT_BASE_URL,
    StableAudio,
    StableAudioMusic,
    StableAudioMusicProvider,
    StableAudioProvider,
)
from aivia.providers.music.stable_audio import DEFAULT_BASE_URL, DEFAULT_MODEL
from aivia.providers.protocols import CAPABILITY_MUSIC_GEN, AudioFormat, MusicRequest

API_KEY = "test-key"


def _wav() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8_000)
        writer.writeframes(b"\x00\x00" * 8_000)
    return buffer.getvalue()


WAV = _wav()


def _provider(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> StableAudioMusicProvider:
    options: dict[str, Any] = {
        "api_key": API_KEY,
        "transport": httpx.MockTransport(handler),
    }
    options.update(kwargs)
    return StableAudioMusicProvider(**options)


def _capture(
    response: httpx.Response,
) -> tuple[list[httpx.Request], Callable[[httpx.Request], httpx.Response]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response

    return requests, handler


def test_metadata_and_registration_record_commercial_use() -> None:
    provider = _provider(lambda request: httpx.Response(200, content=WAV))

    assert provider.name == "stable-audio"
    assert provider.capabilities == (CAPABILITY_MUSIC_GEN,)
    assert provider.base_url == DEFAULT_BASE_URL
    assert provider.api_url == DEFAULT_BASE_URL
    assert provider.default_model == DEFAULT_MODEL
    assert provider.model == DEFAULT_MODEL
    assert provider.endpoint == (
        f"{DEFAULT_BASE_URL}/v2beta/audio/{DEFAULT_MODEL}/text-to-audio"
    )
    assert provider.metadata.is_commercial is True
    assert "commercial" in provider.metadata.commercial_use_notes.lower()
    assert STABLE_AUDIO_DEFAULT_BASE_URL == DEFAULT_BASE_URL
    assert STABLE_AUDIO_BASE_URL == DEFAULT_BASE_URL
    assert StableAudio is StableAudioMusicProvider
    assert StableAudioMusic is StableAudioMusicProvider
    assert StableAudioProvider is StableAudioMusicProvider
    assert provider_registry.resolve("stable-audio") is StableAudioMusicProvider
    assert provider_registry.resolve("stable_audio") is StableAudioMusicProvider


def test_an_instrumental_request_sends_the_form_and_returns_a_track() -> None:
    requests, handler = _capture(
        httpx.Response(200, content=WAV, headers={"x-request-id": "req-2"})
    )
    provider = _provider(handler)
    request = MusicRequest(
        prompt="  a calm piano loop  ",
        instrumental=True,
        duration_seconds=12.0,
        format=AudioFormat.WAV,
        sample_rate=8_000,
        seed=7,
    )

    response = asyncio.run(provider.generate(request))

    submitted = requests[0]
    assert str(submitted.url) == provider.endpoint
    assert submitted.headers["authorization"] == f"Bearer {API_KEY}"
    assert submitted.headers["accept"] == "audio/*"
    assert parse_qs(submitted.content.decode("utf-8")) == {
        "prompt": ["a calm piano loop"],
        "duration": ["12"],
        "output_format": ["wav"],
        "model": [DEFAULT_MODEL],
        "seed": ["7"],
    }
    assert response.model == DEFAULT_MODEL
    assert response.provider == "stable-audio"
    assert response.request_id == "req-2"
    assert response.metadata["commercial_use"] is True
    assert response.metadata["instrumental"] is True
    assert response.first.data == WAV
    assert response.first.format is AudioFormat.WAV
    assert response.first.duration_seconds == 12.0
    assert response.first.lyrics is None
    assert response.first.seed == 7
    assert response.first.metadata["commercial_use"] is True


def test_a_request_model_changes_the_endpoint_and_form() -> None:
    requests, handler = _capture(httpx.Response(200, content=WAV))
    provider = _provider(handler)

    asyncio.run(
        provider.generate(
            MusicRequest(
                prompt="calm piano",
                instrumental=True,
                model="stable-audio-3",
            )
        )
    )

    assert str(requests[0].url).endswith("/v2beta/audio/stable-audio-3/text-to-audio")
    assert parse_qs(requests[0].content.decode("utf-8"))["model"] == ["stable-audio-3"]


def test_lyrics_and_unsupported_formats_are_rejected() -> None:
    provider = _provider(lambda request: httpx.Response(200, content=WAV))

    with pytest.raises(InvalidRequestError, match="instrumental"):
        asyncio.run(provider.generate(MusicRequest(prompt="x", lyrics="la la la")))
    with pytest.raises(InvalidRequestError, match="does not support"):
        asyncio.run(
            provider.generate(
                MusicRequest(prompt="x", instrumental=True, format=AudioFormat.OGG)
            )
        )


def test_duration_outside_the_supported_range_is_rejected() -> None:
    provider = _provider(lambda request: httpx.Response(200, content=WAV))

    with pytest.raises(InvalidRequestError, match="between 1 and 190"):
        asyncio.run(
            provider.generate(
                MusicRequest(prompt="x", instrumental=True, duration_seconds=0.5)
            )
        )
    with pytest.raises(InvalidRequestError, match="between 1 and 190"):
        asyncio.run(
            provider.generate(
                MusicRequest(prompt="x", instrumental=True, duration_seconds=200.0)
            )
        )


def test_empty_audio_and_wrong_request_type_are_rejected() -> None:
    provider = _provider(lambda request: httpx.Response(200, content=b""))

    with pytest.raises(ValidationError, match="empty audio"):
        asyncio.run(provider.generate(MusicRequest(prompt="x", instrumental=True)))
    with pytest.raises(TypeError, match="MusicRequest"):
        asyncio.run(provider.generate("not a request"))  # type: ignore[arg-type]


def test_credentials_and_base_url_are_validated() -> None:
    with pytest.raises(ConfigError, match="API key"):
        StableAudioMusicProvider()
    with pytest.raises(ConfigError, match="http"):
        StableAudioMusicProvider(api_key=API_KEY, base_url="not-a-url")

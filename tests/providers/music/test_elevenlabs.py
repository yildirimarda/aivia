"""ElevenLabs music adapter: vocals, instrumentals and licensing metadata."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from aivia.core.errors import ConfigError, InvalidRequestError, ValidationError
from aivia.providers import provider_registry
from aivia.providers.music import (
    ELEVENLABS_MUSIC_BASE_URL,
    ELEVENLABS_MUSIC_DEFAULT_BASE_URL,
    ElevenLabsMusic,
    ElevenLabsMusicGen,
    ElevenLabsMusicGenProvider,
    ElevenLabsMusicProvider,
)
from aivia.providers.music.elevenlabs import DEFAULT_BASE_URL
from aivia.providers.protocols import CAPABILITY_MUSIC_GEN, AudioFormat, MusicRequest

API_KEY = "test-key"
MODEL = "music-model-1"
MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00audio"


def _provider(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> ElevenLabsMusicProvider:
    options: dict[str, Any] = {
        "api_key": API_KEY,
        "default_model": MODEL,
        "transport": httpx.MockTransport(handler),
    }
    options.update(kwargs)
    return ElevenLabsMusicProvider(**options)


def _capture(
    response: httpx.Response,
) -> tuple[list[httpx.Request], Callable[[httpx.Request], httpx.Response]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response

    return requests, handler


def test_metadata_and_registration_record_commercial_use() -> None:
    provider = _provider(lambda request: httpx.Response(200, content=MP3))

    assert provider.name == "elevenlabs-music"
    assert provider.capabilities == (CAPABILITY_MUSIC_GEN,)
    assert provider.base_url == DEFAULT_BASE_URL
    assert provider.api_url == DEFAULT_BASE_URL
    assert provider.default_model == MODEL
    assert provider.model == MODEL
    assert provider.endpoint == f"{DEFAULT_BASE_URL}/v1/music"
    assert provider.metadata.is_commercial is True
    assert "commercial" in provider.metadata.commercial_use_notes.lower()
    assert ELEVENLABS_MUSIC_DEFAULT_BASE_URL == DEFAULT_BASE_URL
    assert ELEVENLABS_MUSIC_BASE_URL == DEFAULT_BASE_URL
    assert ElevenLabsMusic is ElevenLabsMusicProvider
    assert ElevenLabsMusicGen is ElevenLabsMusicProvider
    assert ElevenLabsMusicGenProvider is ElevenLabsMusicProvider
    assert provider_registry.resolve("elevenlabs-music") is ElevenLabsMusicProvider
    assert provider_registry.resolve("elevenlabs_music") is ElevenLabsMusicProvider


def test_a_vocal_request_sends_lyrics_and_returns_tracks() -> None:
    requests, handler = _capture(
        httpx.Response(200, content=MP3, headers={"x-request-id": "req-1"})
    )
    provider = _provider(handler)
    request = MusicRequest(
        prompt="  a warm synth pop song  ",
        lyrics="  verse one  ",
        duration_seconds=45.0,
        format=AudioFormat.MP3,
        sample_rate=48_000,
        seed=5,
        count=2,
    )

    response = asyncio.run(provider.generate(request))

    assert len(requests) == 2
    submitted = requests[0]
    assert str(submitted.url).startswith(f"{DEFAULT_BASE_URL}/v1/music?")
    assert submitted.url.params["output_format"] == "mp3_44100_128"
    assert submitted.headers["xi-api-key"] == API_KEY
    assert submitted.headers["accept"] == "audio/*"
    assert json.loads(submitted.content) == {
        "prompt": "a warm synth pop song",
        "lyrics": "verse one",
        "music_length_ms": 45000,
        "model_id": MODEL,
    }
    assert response.count == 2
    assert response.model == MODEL
    assert response.provider == "elevenlabs-music"
    assert response.request_id == "req-1"
    assert response.has_lyrics is True
    assert response.metadata["commercial_use"] is True
    assert response.first.data == MP3
    assert response.first.format is AudioFormat.MP3
    assert response.first.duration_seconds == 45.0
    assert response.first.sample_rate == 48_000
    assert response.first.lyrics == "verse one"
    assert [track.seed for track in response.tracks] == [5, 6]
    assert response.first.metadata["commercial_use"] is True


def test_an_instrumental_request_forces_instrumental_and_omits_lyrics() -> None:
    requests, handler = _capture(httpx.Response(200, content=MP3))
    provider = _provider(handler)
    request = MusicRequest(prompt="calm piano", instrumental=True)

    response = asyncio.run(provider.generate(request))

    payload = json.loads(requests[0].content)
    assert payload == {
        "prompt": "calm piano",
        "force_instrumental": True,
        "model_id": MODEL,
    }
    assert "lyrics" not in payload
    assert response.has_lyrics is False
    assert response.first.lyrics is None
    assert response.metadata["instrumental"] is True


def test_a_request_model_overrides_the_default_model() -> None:
    requests, handler = _capture(httpx.Response(200, content=MP3))
    provider = _provider(handler)

    asyncio.run(
        provider.generate(
            MusicRequest(prompt="calm piano", instrumental=True, model="other-model")
        )
    )

    assert json.loads(requests[0].content)["model_id"] == "other-model"


def test_unsupported_output_format_and_duration_are_rejected() -> None:
    provider = _provider(lambda request: httpx.Response(200, content=MP3))

    with pytest.raises(InvalidRequestError, match="does not support"):
        asyncio.run(
            provider.generate(
                MusicRequest(prompt="x", instrumental=True, format=AudioFormat.FLAC)
            )
        )
    with pytest.raises(InvalidRequestError, match="between 10 and 300"):
        asyncio.run(
            provider.generate(
                MusicRequest(prompt="x", instrumental=True, duration_seconds=2.0)
            )
        )
    with pytest.raises(InvalidRequestError, match="between 10 and 300"):
        asyncio.run(
            provider.generate(
                MusicRequest(prompt="x", instrumental=True, duration_seconds=400.0)
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
        ElevenLabsMusicProvider()
    with pytest.raises(ConfigError, match="http"):
        ElevenLabsMusicProvider(api_key=API_KEY, base_url="not-a-url")

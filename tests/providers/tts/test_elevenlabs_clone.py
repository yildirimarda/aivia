"""ElevenLabs voice cloning is opt-in and only touches the vendor when enabled."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from aivia.core.errors import InvalidRequestError, ValidationError
from aivia.providers import (
    CAPABILITY_VOICE_CLONE as TOP_LEVEL_VOICE_CLONE,
)
from aivia.providers.protocols import (
    CAPABILITY_TTS,
    CAPABILITY_VOICE_CLONE,
    CAPABILITY_VOICE_CLONING,
    VoiceCloneRequest,
)
from aivia.providers.tts import (
    CAPABILITY_VOICE_CLONE as TTS_VOICE_CLONE,
)
from aivia.providers.tts import (
    ELEVENLABS_DEFAULT_BASE_URL,
    ElevenLabsTTSProvider,
    VoiceCloneResponse,
)

API_KEY = "test-key"
MODEL = "eleven_multilingual_v2"
VOICE_ID = "cloned-voice-1"
SAMPLE = b"ID3\x04\x00\x00\x00\x00\x00\x00sample"
CLONE_URL = f"{ELEVENLABS_DEFAULT_BASE_URL}/v1/voices/add"
VOICE_URL = f"{ELEVENLABS_DEFAULT_BASE_URL}/v1/voices/{VOICE_ID}"


def _provider(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> ElevenLabsTTSProvider:
    options: dict[str, Any] = {
        "api_key": API_KEY,
        "default_model": MODEL,
        "transport": httpx.MockTransport(handler),
        "voice_clone": True,
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


def test_voice_clone_is_a_capability_flag_that_defaults_to_off() -> None:
    plain = _provider(
        lambda request: httpx.Response(200, json={}),
        voice_clone=None,
    )
    enabled = _provider(lambda request: httpx.Response(200, json={}))

    assert CAPABILITY_VOICE_CLONE == "voice_clone"
    assert CAPABILITY_VOICE_CLONING == CAPABILITY_VOICE_CLONE
    assert TOP_LEVEL_VOICE_CLONE == CAPABILITY_VOICE_CLONE
    assert TTS_VOICE_CLONE == CAPABILITY_VOICE_CLONE
    assert plain.capabilities == (CAPABILITY_TTS,)
    assert plain.voice_clone_enabled is False
    assert not plain.supports(CAPABILITY_VOICE_CLONE)
    assert enabled.capabilities == (CAPABILITY_TTS, CAPABILITY_VOICE_CLONE)
    assert enabled.voice_clone_enabled is True
    assert enabled.voice_cloning_enabled is True
    assert enabled.supports_voice_clone is True
    assert enabled.supports(CAPABILITY_VOICE_CLONE)


def test_any_voice_clone_flag_alias_enables_the_capability() -> None:
    for flag in (
        "voice_clone",
        "voice_cloning",
        "enable_voice_clone",
        "enable_voice_cloning",
        "allow_voice_cloning",
    ):
        provider = ElevenLabsTTSProvider(
            api_key=API_KEY,
            default_model=MODEL,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={})
            ),
            **{flag: True},  # type: ignore[arg-type]
        )
        assert provider.supports(CAPABILITY_VOICE_CLONE)


def test_create_voice_requires_the_capability_and_sends_nothing() -> None:
    requests, handler = _capture(httpx.Response(200, json={"voice_id": VOICE_ID}))
    provider = _provider(handler, voice_clone=False)

    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(provider.create_voice(name="Ada", samples=[SAMPLE]))

    assert caught.value.details["required_capability"] == CAPABILITY_VOICE_CLONE
    assert requests == []


def test_create_voice_uploads_samples_and_returns_the_provider_voice_id() -> None:
    requests, handler = _capture(
        httpx.Response(
            200,
            json={"voice_id": VOICE_ID},
            headers={"x-request-id": "req-clone-1"},
        )
    )
    provider = _provider(handler)

    response = asyncio.run(
        provider.create_voice(
            name="  Ada  ",
            description="  warm narrator  ",
            samples=[
                b"first-sample",
                {
                    "data": b"second-sample",
                    "media_type": "audio/wav",
                    "filename": "second.wav",
                },
            ],
            owner="ada@example.test",
            consent={"granted": True, "reference": "contract-1"},
            language="EN",
            labels={"accent": "british"},
            metadata={"source": "studio"},
        )
    )

    submitted = requests[0]
    assert submitted.method == "POST"
    assert str(submitted.url) == CLONE_URL
    assert submitted.headers["xi-api-key"] == API_KEY
    assert submitted.headers["content-type"].startswith("multipart/form-data")
    content = submitted.content
    assert b'name="name"' in content
    assert b"Ada" in content
    assert b'name="description"' in content
    assert b"warm narrator" in content
    assert b'name="files"' in content
    assert b"first-sample" in content
    assert b"second-sample" in content
    assert b'filename="second.wav"' in content
    assert b'name="labels"' in content
    assert json.dumps({"accent": "british"}, sort_keys=True).encode() in content
    assert isinstance(response, VoiceCloneResponse)
    assert response.voice_id == VOICE_ID
    assert response.provider_voice_id == VOICE_ID
    assert response.id == VOICE_ID
    assert response == VOICE_ID
    assert response.provider == "elevenlabs-tts"
    assert response.name == "Ada"
    assert response.owner == "ada@example.test"
    assert response.language == "en"
    assert response.consent == {"granted": True, "reference": "contract-1"}
    assert response.request_id == "req-clone-1"
    assert response.metadata == {"source": "studio"}


def test_create_voice_accepts_a_request_model_and_defaults_the_filename() -> None:
    requests, handler = _capture(httpx.Response(200, json={"voice_id": VOICE_ID}))
    provider = _provider(handler)
    request = VoiceCloneRequest.model_validate(
        {
            "name": "Narrator",
            "samples": [SAMPLE],
            "labels": ["accent:american", "tone:warm"],
        }
    )

    response = asyncio.run(provider.clone_voice(request))

    content = requests[0].content
    assert b'filename="sample-1.wav"' in content
    assert response.voice_id == VOICE_ID
    assert response.name == "Narrator"


def test_create_voice_rejects_a_missing_sample_or_voice_id() -> None:
    _requests, handler = _capture(httpx.Response(200, json={}))
    provider = _provider(handler)

    with pytest.raises(InvalidRequestError, match="audio sample"):
        asyncio.run(provider.create_voice(name="Ada"))

    _requests, handler = _capture(httpx.Response(200, json={"status": "ok"}))
    provider = _provider(handler)
    with pytest.raises(ValidationError, match="no voice id"):
        asyncio.run(provider.create_voice(name="Ada", samples=[SAMPLE]))


def test_delete_voice_requires_the_capability_and_sends_nothing() -> None:
    requests, handler = _capture(httpx.Response(200, json={"status": "ok"}))
    provider = _provider(handler, voice_clone=False)

    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(provider.delete_voice(VOICE_ID))

    assert caught.value.details["required_capability"] == CAPABILITY_VOICE_CLONE
    assert requests == []


def test_delete_voice_sends_a_delete_and_reports_whether_it_existed() -> None:
    requests, handler = _capture(httpx.Response(200, json={"status": "ok"}))
    provider = _provider(handler)

    assert asyncio.run(provider.delete_voice(VOICE_ID)) is True
    submitted = requests[0]
    assert submitted.method == "DELETE"
    assert str(submitted.url) == VOICE_URL
    assert submitted.headers["xi-api-key"] == API_KEY

    requests, handler = _capture(httpx.Response(404, json={"detail": "not found"}))
    provider = _provider(handler)
    assert asyncio.run(provider.delete_voice(provider_voice_id=VOICE_ID)) is False
    assert requests[0].method == "DELETE"


def test_delete_voice_accepts_a_response_model_or_mapping() -> None:
    requests, handler = _capture(httpx.Response(200, json={"status": "ok"}))
    provider = _provider(handler)

    assert asyncio.run(
        provider.delete_clone(VoiceCloneResponse(voice_id=VOICE_ID))
    ) is True
    assert str(requests[0].url) == VOICE_URL

    requests, handler = _capture(httpx.Response(200, json={"status": "ok"}))
    provider = _provider(handler)
    assert asyncio.run(provider.delete_voice({"provider_voice_id": VOICE_ID})) is True
    assert str(requests[0].url) == VOICE_URL

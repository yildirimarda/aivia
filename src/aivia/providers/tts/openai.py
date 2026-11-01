"""OpenAI text to speech adapter.

OpenAI exposes speech synthesis through ``POST /v1/audio/speech``. The endpoint
returns encoded audio bytes directly, so the adapter maps the request format,
speed and optional instructions onto the JSON body and wraps the response in the
shared :class:`~aivia.providers.protocols.tts.TTSResponse`. Voice aliases are
resolved through a :class:`~aivia.providers.tts.voice.VoiceMap`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import InvalidRequestError, ValidationError
from aivia.core.models.provider import ProviderMeta, ProviderPrice
from aivia.core.secrets import SecretsLoader
from aivia.providers.base import Provider
from aivia.providers.protocols.audio import AudioFormat
from aivia.providers.protocols.tts import CAPABILITY_TTS, TTSRequest, TTSResponse
from aivia.providers.registry import register_provider
from aivia.providers.tts.common import (
    as_mapping,
    as_text,
    resolve_api_key,
    sniff_audio_format,
    validate_base_url,
)
from aivia.providers.tts.voice import VoiceMap

__all__ = [
    "DEFAULT_BASE_URL",
    "OPENAI_BASE_URL",
    "OPENAI_DEFAULT_BASE_URL",
    "OpenAI",
    "OpenAISpeech",
    "OpenAISpeechProvider",
    "OpenAITTS",
    "OpenAITTSProvider",
]

DEFAULT_BASE_URL = "https://api.openai.com"
"""The OpenAI API root, without a trailing slash."""

OPENAI_DEFAULT_BASE_URL = DEFAULT_BASE_URL
"""Alias that names the OpenAI API root explicitly."""

OPENAI_BASE_URL = DEFAULT_BASE_URL
"""Alias for callers that prefer the short base URL name."""

_DEFAULT_METADATA = ProviderMeta(
    name="openai-tts",
    capabilities=[CAPABILITY_TTS],
    supported_options={
        CAPABILITY_TTS: [
            "text",
            "model",
            "format",
            "sample_rate",
            "speed",
            "instructions",
        ]
    },
    pricing={
        CAPABILITY_TTS: ProviderPrice(amount_usd=0.000015, unit="character"),
    },
)

_RESPONSE_FORMATS: dict[AudioFormat, str] = {
    AudioFormat.MP3: "mp3",
    AudioFormat.WAV: "wav",
    AudioFormat.OGG: "opus",
    AudioFormat.OPUS: "opus",
    AudioFormat.AAC: "aac",
    AudioFormat.FLAC: "flac",
    AudioFormat.PCM: "pcm",
    AudioFormat.M4A: "aac",
}


def _payload(request: TTSRequest, model: str, response_format: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "input": request.text,
        "voice": request.voice,
        "response_format": response_format,
    }
    if request.speed != 1.0:
        payload["speed"] = request.speed
    if request.instructions is not None:
        payload["instructions"] = request.instructions
    return payload


def _request_id(response: httpx.Response, body: Mapping[str, Any] | None) -> str | None:
    header = response.headers.get("x-request-id")
    if header:
        return as_text(header) or None
    if body is not None:
        return as_text(body.get("id")) or None
    return None


class OpenAITTSProvider(Provider):
    """A text to speech provider backed by the OpenAI audio API.

    ``api_key`` is an already resolved credential while ``api_key_ref`` names a
    secret resolved through ``secrets``. ``default_model`` supplies the speech
    model id when a request does not name one. ``voice_map`` maps logical voice
    aliases to OpenAI voice ids. ``base_url`` may point at a local test server
    and ``transport`` injects an ``httpx`` mock transport for the offline suite.
    """

    default_metadata = _DEFAULT_METADATA

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_ref: str | None = None,
        secrets: SecretsLoader | None = None,
        base_url: str = DEFAULT_BASE_URL,
        api_url: str | None = None,
        default_model: str | None = None,
        model: str | None = None,
        voice_map: Mapping[str, str] | VoiceMap | None = None,
        default_voice: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        name: str | None = None,
        health_timeout_seconds: float | None = None,
        close_timeout_seconds: float | None = None,
        http_config: HttpClientConfig | None = None,
    ) -> None:
        super().__init__(
            name=name,
            health_timeout_seconds=health_timeout_seconds,
            close_timeout_seconds=close_timeout_seconds,
            http_config=http_config,
        )
        if transport is not None and not isinstance(
            transport, httpx.AsyncBaseTransport
        ):
            raise TypeError("transport must be an httpx.AsyncBaseTransport")
        selected = model if model is not None else default_model
        if selected is not None:
            if not isinstance(selected, str):
                raise TypeError("default_model must be a string or None")
            selected = selected.strip() or None
        self._default_model = selected
        self._api_key = resolve_api_key(
            api_key,
            api_key_ref,
            secrets,
            provider="OpenAI",
        )
        self._base_url = validate_base_url(api_url or base_url)
        self._voices = VoiceMap.from_value(voice_map, default=default_voice)
        if transport is not None:
            self._http = self.create_http_client(transport=transport)

    @property
    def base_url(self) -> str:
        """The configured OpenAI API root, without a trailing slash."""
        return self._base_url

    @property
    def api_url(self) -> str:
        """Alias for :attr:`base_url`."""
        return self._base_url

    @property
    def default_model(self) -> str | None:
        """The model id used when a request does not name one."""
        return self._default_model

    @property
    def model(self) -> str | None:
        """Alias for :attr:`default_model`."""
        return self._default_model

    @property
    def voice_map(self) -> VoiceMap:
        """The alias map used to resolve request voices."""
        return self._voices

    @property
    def endpoint(self) -> str:
        """The speech synthesis endpoint derived from :attr:`base_url`."""
        return f"{self._base_url}/v1/audio/speech"

    def _headers(self) -> dict[str, str]:
        return {
            "authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
            "accept": "audio/*",
        }

    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """Synthesize *request* through OpenAI and return its audio."""
        if not isinstance(request, TTSRequest):
            raise TypeError("request must be a TTSRequest")
        model = request.model or self._default_model
        if model is None or not model.strip():
            raise InvalidRequestError(
                "OpenAI text to speech needs a model id",
                details={"provider": self.name},
            )
        model = model.strip()
        voice = self._voices.resolve(request.voice)
        response_format = _RESPONSE_FORMATS.get(request.format)
        if response_format is None:
            raise InvalidRequestError(
                f"OpenAI does not support {request.format.value} output",
                details={"provider": self.name, "format": request.format.value},
            )
        resolved = request.model_copy(update={"voice": voice})
        response = await self.http.post(
            self.endpoint,
            headers=self._headers(),
            json=_payload(resolved, model, response_format),
            timeout=request.timeout_seconds,
            raise_for_status=True,
        )
        audio = response.content
        if not audio:
            raise ValidationError(
                "OpenAI returned empty audio",
                details={"provider": self.name},
            )
        actual_format = sniff_audio_format(audio, request.format)
        return TTSResponse(
            audio=audio,
            format=actual_format,
            sample_rate=request.sample_rate,
            voice=voice,
            model=model,
            provider=self.name,
            request_id=_request_id(response, as_mapping(_json(response))),
            metadata={"characters": len(request.text)},
        )


def _json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


OpenAI = OpenAITTSProvider
"""Alias for :class:`OpenAITTSProvider`."""

OpenAITTS = OpenAITTSProvider
"""Alias for :class:`OpenAITTSProvider`."""

OpenAISpeech = OpenAITTSProvider
"""Alias for :class:`OpenAITTSProvider`."""

OpenAISpeechProvider = OpenAITTSProvider
"""Alias for :class:`OpenAITTSProvider`."""

register_provider("openai-tts", OpenAITTSProvider, module=__name__)
register_provider("openai_speech", OpenAITTSProvider, module=__name__)
register_provider("openai-speech", OpenAITTSProvider, module=__name__)

"""ElevenLabs text to speech adapter.

ElevenLabs synthesizes speech with ``POST /v1/text-to-speech/{voice_id}`` and
returns encoded audio. The ``/with-timestamps`` variant returns the audio and a
character alignment, which this adapter converts into word timestamps. Voice
aliases are resolved through a :class:`~aivia.providers.tts.voice.VoiceMap`
before the request is built; the adapter never hard-codes a voice id.
"""

from __future__ import annotations

import math
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
    decode_base64_audio,
    pcm_to_wav,
    resolve_api_key,
    sniff_audio_format,
    validate_base_url,
    word_timestamps_from_alignment,
)
from aivia.providers.tts.voice import VoiceMap

__all__ = [
    "DEFAULT_BASE_URL",
    "ELEVENLABS_BASE_URL",
    "ELEVENLABS_DEFAULT_BASE_URL",
    "ElevenLabs",
    "ElevenLabsProvider",
    "ElevenLabsTTS",
    "ElevenLabsTTSProvider",
]

DEFAULT_BASE_URL = "https://api.elevenlabs.io"
"""The ElevenLabs API root, without a trailing slash."""

ELEVENLABS_DEFAULT_BASE_URL = DEFAULT_BASE_URL
"""Alias that names the ElevenLabs API root explicitly."""

ELEVENLABS_BASE_URL = DEFAULT_BASE_URL
"""Alias for callers that prefer the short base URL name."""

_DEFAULT_METADATA = ProviderMeta(
    name="elevenlabs-tts",
    capabilities=[CAPABILITY_TTS],
    pricing={
        CAPABILITY_TTS: ProviderPrice(amount_usd=0.00015, unit="character"),
    },
)

_PCM_RATES = (16000, 22050, 24000, 44100)
_MP3_RATE = 44100
_OPUS_RATE = 48000


def _nearest_rate(value: int | None, allowed: tuple[int, ...]) -> int:
    if value is None:
        return allowed[0]
    return min(allowed, key=lambda rate: abs(rate - value))


def _output_format(
    audio_format: AudioFormat,
    sample_rate: int | None,
) -> tuple[str, AudioFormat, int]:
    if audio_format is AudioFormat.MP3:
        return f"mp3_{_MP3_RATE}_128", AudioFormat.MP3, _MP3_RATE
    if audio_format in {AudioFormat.OGG, AudioFormat.OPUS}:
        return f"opus_{_OPUS_RATE}_32", AudioFormat.OGG, _OPUS_RATE
    if audio_format in {AudioFormat.PCM, AudioFormat.WAV}:
        rate = _nearest_rate(sample_rate, _PCM_RATES)
        return f"pcm_{rate}", AudioFormat.PCM, rate
    raise InvalidRequestError(
        f"ElevenLabs does not support {audio_format.value} output",
        details={"provider": "elevenlabs-tts", "format": audio_format.value},
    )


def _voice_settings(request: TTSRequest) -> dict[str, float | bool]:
    settings: dict[str, float | bool] = {}
    for key in ("stability", "similarity_boost", "style"):
        value = request.metadata.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number) and 0.0 <= number <= 1.0:
                settings[key] = number
    boost = request.metadata.get("use_speaker_boost")
    if isinstance(boost, bool):
        settings["use_speaker_boost"] = boost
    if request.speed != 1.0:
        settings["speed"] = request.speed
    return settings


def _payload(request: TTSRequest, model: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "text": request.text,
        "model_id": model,
    }
    if request.language is not None:
        payload["language_code"] = request.language
    settings = _voice_settings(request)
    if settings:
        payload["voice_settings"] = settings
    seed = request.metadata.get("seed")
    if isinstance(seed, int) and not isinstance(seed, bool):
        payload["seed"] = seed
    return payload


def _alignment(body: Mapping[str, Any], text: str) -> list[Any]:
    alignment = as_mapping(body.get("alignment"))
    if alignment is None:
        return []
    characters = alignment.get("characters")
    starts = alignment.get("character_start_times_seconds")
    ends = alignment.get("character_end_times_seconds")
    if not isinstance(characters, list) or not isinstance(starts, list):
        return []
    if not isinstance(ends, list):
        return []
    return word_timestamps_from_alignment(
        text,
        [str(character) for character in characters],
        [float(value) for value in starts],
        [float(value) for value in ends],
    )


class ElevenLabsTTSProvider(Provider):
    """A text to speech provider backed by the ElevenLabs API.

    ``api_key`` is an already resolved credential while ``api_key_ref`` names a
    secret resolved through ``secrets``. ``default_model`` supplies the model id
    when a request does not name one. ``voice_map`` maps logical voice aliases
    to ElevenLabs voice ids. ``base_url`` may point at a local test server and
    ``transport`` injects an ``httpx`` mock transport for the offline suite.
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
            provider="ElevenLabs",
        )
        self._base_url = validate_base_url(api_url or base_url)
        self._voices = VoiceMap.from_value(voice_map, default=default_voice)
        if transport is not None:
            self._http = self.create_http_client(transport=transport)

    @property
    def base_url(self) -> str:
        """The configured ElevenLabs API root, without a trailing slash."""
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

    def endpoint_for(self, voice: str, *, timestamps: bool = False) -> str:
        """Return the synthesis endpoint for *voice*."""
        suffix = "/with-timestamps" if timestamps else ""
        return f"{self._base_url}/v1/text-to-speech/{voice}{suffix}"

    def _headers(self, *, timestamps: bool) -> dict[str, str]:
        return {
            "xi-api-key": self._api_key,
            "content-type": "application/json",
            "accept": "application/json" if timestamps else "audio/mpeg",
        }

    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """Synthesize *request* through ElevenLabs and return its audio."""
        if not isinstance(request, TTSRequest):
            raise TypeError("request must be a TTSRequest")
        model = request.model or self._default_model
        if model is None or not model.strip():
            raise InvalidRequestError(
                "ElevenLabs text to speech needs a model id",
                details={"provider": self.name},
            )
        model = model.strip()
        voice = self._voices.resolve(request.voice)
        output_format, actual_format, sample_rate = _output_format(
            request.format,
            request.sample_rate,
        )
        payload = _payload(request, model)
        response = await self.http.post(
            self.endpoint_for(voice, timestamps=request.word_timestamps),
            headers=self._headers(timestamps=request.word_timestamps),
            params={"output_format": output_format},
            json=payload,
            timeout=request.timeout_seconds,
            raise_for_status=True,
        )

        timestamps = []
        if request.word_timestamps:
            body = as_mapping(response.json())
            if body is None:
                raise ValidationError(
                    "ElevenLabs returned a non-JSON timestamp response",
                    details={"provider": self.name},
                )
            audio = decode_base64_audio(
                as_text(body.get("audio_base64")),
                provider=self.name,
            )
            timestamps = _alignment(body, request.text)
        else:
            audio = response.content
            if not audio:
                raise ValidationError(
                    "ElevenLabs returned empty audio",
                    details={"provider": self.name},
                )

        if actual_format is AudioFormat.PCM and request.format is AudioFormat.WAV:
            audio = pcm_to_wav(audio, sample_rate=sample_rate)
            actual_format = AudioFormat.WAV
        else:
            actual_format = sniff_audio_format(audio, actual_format)

        return TTSResponse(
            audio=audio,
            format=actual_format,
            sample_rate=sample_rate,
            voice=voice,
            model=model,
            provider=self.name,
            word_timestamps=timestamps,
            request_id=response.headers.get("x-request-id") or None,
            metadata={"characters": len(request.text)},
        )


ElevenLabs = ElevenLabsTTSProvider
"""Alias for :class:`ElevenLabsTTSProvider`."""

ElevenLabsProvider = ElevenLabsTTSProvider
"""Alias for :class:`ElevenLabsTTSProvider`."""

ElevenLabsTTS = ElevenLabsTTSProvider
"""Alias for :class:`ElevenLabsTTSProvider`."""

register_provider("elevenlabs-tts", ElevenLabsTTSProvider, module=__name__)
register_provider("elevenlabs", ElevenLabsTTSProvider, module=__name__)
register_provider("elevenlabs_tts", ElevenLabsTTSProvider, module=__name__)

"""Google Cloud text to speech adapter.

Google's Text-to-Speech API accepts a JSON body at ``/v1/text:synthesize`` and
returns base64 audio in ``audioContent``. The adapter maps the requested
container to a Google audio encoding and wraps raw ``LINEAR16`` PCM in a WAV
container when the caller asked for WAV. Voice aliases are resolved through a
:class:`~aivia.providers.tts.voice.VoiceMap`.
"""

from __future__ import annotations

import re
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
    format_language,
    pcm_to_wav,
    phoneme_ssml,
    resolve_api_key,
    validate_base_url,
)
from aivia.providers.tts.voice import VoiceMap

__all__ = [
    "DEFAULT_API_VERSION",
    "DEFAULT_BASE_URL",
    "GOOGLE_API_URL",
    "GOOGLE_BASE_URL",
    "GOOGLE_DEFAULT_BASE_URL",
    "GOOGLE_TTS_API_URL",
    "Google",
    "GoogleCloudTTS",
    "GoogleCloudTTSProvider",
    "GoogleProvider",
    "GoogleTTS",
    "GoogleTTSProvider",
]

DEFAULT_BASE_URL = "https://texttospeech.googleapis.com"
"""The Google Cloud Text-to-Speech API root."""

DEFAULT_API_VERSION = "v1"
"""The Text-to-Speech API version used in request paths."""

GOOGLE_DEFAULT_BASE_URL = DEFAULT_BASE_URL
"""Alias that names the Google API root explicitly."""

GOOGLE_API_URL = DEFAULT_BASE_URL
"""Alias for callers that describe the root as the API URL."""

GOOGLE_BASE_URL = DEFAULT_BASE_URL
"""Alias for callers that prefer the short base URL name."""

GOOGLE_TTS_API_URL = DEFAULT_BASE_URL
"""Alias that names the Google text to speech API root explicitly."""

_DEFAULT_METADATA = ProviderMeta(
    name="google-tts",
    capabilities=[CAPABILITY_TTS],
    supported_options={
        CAPABILITY_TTS: [
            "pronunciation",
            "text",
            "model",
            "language",
            "format",
            "sample_rate",
            "speed",
        ]
    },
    pricing={
        CAPABILITY_TTS: ProviderPrice(amount_usd=0.000016, unit="character"),
    },
)

_DEFAULT_SAMPLE_RATE = 24000
_LANGUAGE = re.compile(r"^([a-z]{2,3})-([A-Z]{2})(?:-|$)")


def _encoding(audio_format: AudioFormat) -> tuple[str, AudioFormat]:
    if audio_format is AudioFormat.MP3:
        return "MP3", AudioFormat.MP3
    if audio_format in {AudioFormat.OGG, AudioFormat.OPUS}:
        return "OGG_OPUS", AudioFormat.OGG
    if audio_format in {AudioFormat.WAV, AudioFormat.PCM}:
        return "LINEAR16", AudioFormat.PCM
    raise InvalidRequestError(
        f"Google text to speech does not support {audio_format.value} output",
        details={"provider": "google-tts", "format": audio_format.value},
    )


def _language(request: TTSRequest, voice: str) -> str:
    if request.language is not None:
        return format_language(request.language)
    match = _LANGUAGE.match(voice)
    if match is not None:
        return f"{match.group(1)}-{match.group(2)}"
    raise InvalidRequestError(
        "Google text to speech needs a language code when the voice does not name one",
        details={"provider": "google-tts"},
    )


def _payload(
    request: TTSRequest,
    voice: str,
    language: str,
    encoding: str,
    sample_rate: int,
) -> dict[str, Any]:
    audio_config: dict[str, Any] = {
        "audioEncoding": encoding,
        "speakingRate": request.speed,
        "sampleRateHertz": sample_rate,
    }
    voice_config: dict[str, Any] = {"languageCode": language, "name": voice}
    if request.model is not None:
        voice_config["modelName"] = request.model
    if request.pronunciation:
        spoken = phoneme_ssml(request.text, request.pronunciation)
        speech_input: dict[str, str] = {"ssml": f"<speak>{spoken}</speak>"}
    else:
        speech_input = {"text": request.text}
    return {
        "input": speech_input,
        "voice": voice_config,
        "audioConfig": audio_config,
    }


class GoogleTTSProvider(Provider):
    """A text to speech provider backed by Google Cloud Text-to-Speech.

    ``api_key`` is an already resolved credential while ``api_key_ref`` names a
    secret resolved through ``secrets``. ``voice_map`` maps logical voice
    aliases to Google voice names. ``base_url`` may point at a local test server
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
        api_version: str = DEFAULT_API_VERSION,
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
        if not isinstance(api_version, str) or not api_version.strip():
            raise TypeError("api_version must be a non-empty string")
        self._api_version = api_version.strip()
        self._api_key = resolve_api_key(
            api_key,
            api_key_ref,
            secrets,
            provider="Google",
        )
        self._base_url = validate_base_url(api_url or base_url)
        self._voices = VoiceMap.from_value(voice_map, default=default_voice)
        if transport is not None:
            self._http = self.create_http_client(transport=transport)

    @property
    def base_url(self) -> str:
        """The configured Google API root, without a trailing slash."""
        return self._base_url

    @property
    def api_url(self) -> str:
        """Alias for :attr:`base_url`."""
        return self._base_url

    @property
    def api_version(self) -> str:
        """The configured Text-to-Speech API version."""
        return self._api_version

    @property
    def voice_map(self) -> VoiceMap:
        """The alias map used to resolve request voices."""
        return self._voices

    @property
    def endpoint(self) -> str:
        """The synthesis endpoint derived from :attr:`base_url`."""
        return f"{self._base_url}/{self._api_version}/text:synthesize"

    def _headers(self) -> dict[str, str]:
        return {
            "x-goog-api-key": self._api_key,
            "content-type": "application/json",
            "accept": "application/json",
        }

    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """Synthesize *request* through Google and return its audio."""
        if not isinstance(request, TTSRequest):
            raise TypeError("request must be a TTSRequest")
        voice = self._voices.resolve(request.voice)
        language = _language(request, voice)
        encoding, actual_format = _encoding(request.format)
        sample_rate = request.sample_rate or _DEFAULT_SAMPLE_RATE
        response = await self.http.post(
            self.endpoint,
            headers=self._headers(),
            json=_payload(request, voice, language, encoding, sample_rate),
            timeout=request.timeout_seconds,
            raise_for_status=True,
        )
        body = as_mapping(_json(response))
        if body is None:
            raise ValidationError(
                "Google text to speech returned a non-JSON response",
                details={"provider": self.name},
            )
        audio = decode_base64_audio(
            as_text(body.get("audioContent")),
            provider=self.name,
        )
        if actual_format is AudioFormat.PCM and request.format is AudioFormat.WAV:
            audio = pcm_to_wav(audio, sample_rate=sample_rate)
            actual_format = AudioFormat.WAV
        return TTSResponse(
            audio=audio,
            format=actual_format,
            sample_rate=sample_rate,
            voice=voice,
            model=request.model or "",
            provider=self.name,
            request_id=response.headers.get("x-goog-request-id")
            or response.headers.get("x-request-id")
            or None,
            metadata={
                "characters": len(request.text),
                "language": language,
            },
        )


def _json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


Google = GoogleTTSProvider
"""Alias for :class:`GoogleTTSProvider`."""

GoogleTTS = GoogleTTSProvider
"""Alias for :class:`GoogleTTSProvider`."""

GoogleCloudTTS = GoogleTTSProvider
"""Alias for :class:`GoogleTTSProvider`."""

GoogleCloudTTSProvider = GoogleTTSProvider
"""Alias for :class:`GoogleTTSProvider`."""

GoogleProvider = GoogleTTSProvider
"""Alias for :class:`GoogleTTSProvider`."""

register_provider("google-tts", GoogleTTSProvider, module=__name__)
register_provider("google_tts", GoogleTTSProvider, module=__name__)
register_provider("google-speech", GoogleTTSProvider, module=__name__)

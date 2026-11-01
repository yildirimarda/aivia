"""Azure Speech text to speech adapter.

Azure's REST endpoint accepts SSML at ``/cognitiveservices/v1`` and returns
encoded audio directly. The adapter builds a minimal SSML document with the
resolved voice, language and speaking rate, selects an output format matching
the requested container and wraps the response in the shared
:class:`~aivia.providers.protocols.tts.TTSResponse`. Voice aliases are resolved
through a :class:`~aivia.providers.tts.voice.VoiceMap`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from xml.sax.saxutils import escape

import httpx

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import ConfigError, InvalidRequestError, ValidationError
from aivia.core.models.provider import ProviderMeta, ProviderPrice
from aivia.core.secrets import SecretsLoader
from aivia.providers.base import Provider
from aivia.providers.protocols.audio import AudioFormat
from aivia.providers.protocols.tts import CAPABILITY_TTS, TTSRequest, TTSResponse
from aivia.providers.registry import register_provider
from aivia.providers.tts.common import (
    as_text,
    format_language,
    phoneme_ssml,
    resolve_api_key,
    sniff_audio_format,
    validate_base_url,
)
from aivia.providers.tts.voice import VoiceMap

__all__ = [
    "AZURE_TTS_API_VERSION",
    "DEFAULT_API_VERSION",
    "Azure",
    "AzureProvider",
    "AzureSpeech",
    "AzureSpeechProvider",
    "AzureTTS",
    "AzureTTSProvider",
]

DEFAULT_API_VERSION = "1.0"
"""The SSML version emitted in the request document."""

AZURE_TTS_API_VERSION = DEFAULT_API_VERSION
"""Alias that names the SSML version explicitly."""

_DEFAULT_METADATA = ProviderMeta(
    name="azure-tts",
    capabilities=[CAPABILITY_TTS],
    supported_options={
        CAPABILITY_TTS: [
            "pronunciation",
            "text",
            "model",
            "language",
            "format",
            "speed",
        ]
    },
    pricing={
        CAPABILITY_TTS: ProviderPrice(amount_usd=0.000016, unit="character"),
    },
)

_REGION = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_LANGUAGE = re.compile(r"^([a-z]{2,3})-([A-Z]{2})")

_OUTPUT_FORMATS: dict[AudioFormat, tuple[str, AudioFormat, int]] = {
    AudioFormat.MP3: ("audio-24khz-96kbitrate-mono-mp3", AudioFormat.MP3, 24000),
    AudioFormat.WAV: ("riff-24khz-16bit-mono-pcm", AudioFormat.WAV, 24000),
    AudioFormat.OGG: ("ogg-24khz-16bit-mono-opus", AudioFormat.OGG, 24000),
    AudioFormat.OPUS: ("ogg-24khz-16bit-mono-opus", AudioFormat.OGG, 24000),
    AudioFormat.PCM: ("raw-24khz-16bit-mono-pcm", AudioFormat.PCM, 24000),
}


def _region(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("region must be a string")
    text = value.strip().lower()
    if not _REGION.fullmatch(text):
        raise ConfigError(
            f"invalid Azure region: {value!r}",
            details={"field": "region"},
        )
    return text


def _base_url(region: str | None, base_url: str | None, api_url: str | None) -> str:
    selected = api_url or base_url
    if selected is not None:
        return validate_base_url(selected)
    if region is None:
        raise ConfigError(
            "Azure text to speech needs a region or base_url",
            details={"field": "region"},
        )
    return f"https://{_region(region)}.tts.speech.microsoft.com"


def _language(request: TTSRequest, voice: str) -> str:
    if request.language is not None:
        return format_language(request.language)
    match = _LANGUAGE.match(voice)
    if match is not None:
        return f"{match.group(1)}-{match.group(2)}"
    raise InvalidRequestError(
        "Azure text to speech needs a language code when the voice does not name one",
        details={"provider": "azure-tts"},
    )


def _rate(speed: float) -> str | None:
    if speed == 1.0:
        return None
    percentage = round((speed - 1.0) * 100)
    percentage = max(-50, min(200, percentage))
    return f"{percentage:+d}%"


def _ssml(request: TTSRequest, voice: str, language: str) -> str:
    text = phoneme_ssml(request.text, request.pronunciation)
    rate = _rate(request.speed)
    if rate is None:
        prosody = f"<prosody>{text}</prosody>"
    else:
        prosody = f'<prosody rate="{rate}">{text}</prosody>'
    return (
        f'<speak version="{DEFAULT_API_VERSION}" '
        'xmlns="http://www.w3.org/2001/10/synthesis" '
        f'xml:lang="{language}">'
        f'<voice name="{escape(voice)}">{prosody}</voice>'
        "</speak>"
    )


class AzureTTSProvider(Provider):
    """A text to speech provider backed by Azure Speech Services.

    ``api_key`` is an already resolved credential while ``api_key_ref`` names a
    secret resolved through ``secrets``. Either ``region`` or ``base_url`` must
    identify the service. ``voice_map`` maps logical voice aliases to Azure
    voice names. ``transport`` injects an ``httpx`` mock transport for the
    offline suite.
    """

    default_metadata = _DEFAULT_METADATA

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_ref: str | None = None,
        secrets: SecretsLoader | None = None,
        region: str | None = None,
        base_url: str | None = None,
        api_url: str | None = None,
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
        self._api_key = resolve_api_key(
            api_key,
            api_key_ref,
            secrets,
            provider="Azure",
        )
        self._base_url = _base_url(region, base_url, api_url)
        self._voices = VoiceMap.from_value(voice_map, default=default_voice)
        if transport is not None:
            self._http = self.create_http_client(transport=transport)

    @property
    def base_url(self) -> str:
        """The configured Azure service root, without a trailing slash."""
        return self._base_url

    @property
    def api_url(self) -> str:
        """Alias for :attr:`base_url`."""
        return self._base_url

    @property
    def voice_map(self) -> VoiceMap:
        """The alias map used to resolve request voices."""
        return self._voices

    @property
    def endpoint(self) -> str:
        """The synthesis endpoint derived from :attr:`base_url`."""
        return f"{self._base_url}/cognitiveservices/v1"

    def _headers(self, output_format: str) -> dict[str, str]:
        return {
            "ocp-apim-subscription-key": self._api_key,
            "content-type": "application/ssml+xml",
            "x-microsoft-outputformat": output_format,
            "accept": "audio/*",
            "user-agent": "aivia",
        }

    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """Synthesize *request* through Azure and return its audio."""
        if not isinstance(request, TTSRequest):
            raise TypeError("request must be a TTSRequest")
        voice = self._voices.resolve(request.voice)
        language = _language(request, voice)
        selected = _OUTPUT_FORMATS.get(request.format)
        if selected is None:
            raise InvalidRequestError(
                f"Azure text to speech does not support {request.format.value} output",
                details={"provider": self.name, "format": request.format.value},
            )
        output_format, actual_format, sample_rate = selected
        response = await self.http.post(
            self.endpoint,
            headers=self._headers(output_format),
            content=_ssml(request, voice, language),
            timeout=request.timeout_seconds,
            raise_for_status=True,
        )
        audio = response.content
        if not audio:
            raise ValidationError(
                "Azure returned empty audio",
                details={"provider": self.name},
            )
        actual_format = sniff_audio_format(audio, actual_format)
        request_id = (
            response.headers.get("x-requestid")
            or response.headers.get("x-microsoft-request-id")
            or None
        )
        return TTSResponse(
            audio=audio,
            format=actual_format,
            sample_rate=sample_rate,
            voice=voice,
            model=request.model or "",
            provider=self.name,
            request_id=as_text(request_id) or None,
            metadata={
                "characters": len(request.text),
                "language": language,
            },
        )


Azure = AzureTTSProvider
"""Alias for :class:`AzureTTSProvider`."""

AzureTTS = AzureTTSProvider
"""Alias for :class:`AzureTTSProvider`."""

AzureSpeech = AzureTTSProvider
"""Alias for :class:`AzureTTSProvider`."""

AzureSpeechProvider = AzureTTSProvider
"""Alias for :class:`AzureTTSProvider`."""

AzureProvider = AzureTTSProvider
"""Alias for :class:`AzureTTSProvider`."""

register_provider("azure-tts", AzureTTSProvider, module=__name__)
register_provider("azure_speech", AzureTTSProvider, module=__name__)
register_provider("azure", AzureTTSProvider, module=__name__)

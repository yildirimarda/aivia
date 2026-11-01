"""ElevenLabs music adapter for vocal songs and instrumentals.

ElevenLabs exposes music generation through ``POST /v1/music``. The endpoint
returns encoded audio, so the adapter maps the request prompt, optional lyrics,
duration and instrumental flag onto the JSON body and wraps the response in the
shared :class:`~aivia.providers.protocols.music.MusicResponse`. A vocal request
carries lyrics and omits the instrumental flag; an instrumental request sets
``force_instrumental`` and never sends lyrics.

The adapter records the provider's licensing terms and commercial-use notes on
its metadata so a later license gate can decide whether a track may be
published or delivered.
"""

from __future__ import annotations

from typing import Any

import httpx

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import InvalidRequestError
from aivia.core.models.asset import AssetLicense
from aivia.core.models.provider import ProviderMeta
from aivia.core.secrets import SecretsLoader
from aivia.providers.base import Provider
from aivia.providers.music.common import (
    as_text,
    resolve_api_key,
    sniff_audio_format,
    validate_audio,
    validate_base_url,
)
from aivia.providers.protocols.audio import AudioFormat
from aivia.providers.protocols.music import (
    CAPABILITY_MUSIC_GEN,
    MusicRequest,
    MusicResponse,
    MusicTrack,
)
from aivia.providers.registry import register_provider

__all__ = [
    "DEFAULT_BASE_URL",
    "ELEVENLABS_MUSIC_BASE_URL",
    "ELEVENLABS_MUSIC_DEFAULT_BASE_URL",
    "ElevenLabsMusic",
    "ElevenLabsMusicGen",
    "ElevenLabsMusicGenProvider",
    "ElevenLabsMusicProvider",
]

DEFAULT_BASE_URL = "https://api.elevenlabs.io"
"""The ElevenLabs API root, without a trailing slash."""

ELEVENLABS_MUSIC_DEFAULT_BASE_URL = DEFAULT_BASE_URL
"""Alias that names the music adapter's default API root."""

ELEVENLABS_MUSIC_BASE_URL = DEFAULT_BASE_URL
"""Alias for callers that prefer the short base URL name."""

LICENSE_NAME = "ElevenLabs Music Terms"
LICENSE_URL = "https://elevenlabs.io/terms-of-use"

MIN_DURATION_MS = 10_000
MAX_DURATION_MS = 300_000

_DEFAULT_METADATA = ProviderMeta(
    name="elevenlabs-music",
    capabilities=[CAPABILITY_MUSIC_GEN],
    supported_options={
        CAPABILITY_MUSIC_GEN: [
            "vocals",
            "instrumental",
            "mp3",
            "ogg",
            "opus",
            "pcm",
            "prompt",
            "lyrics",
            "duration",
            "model",
            "sample_rate",
            "seed",
            "count",
            "format",
        ]
    },
    license=AssetLicense(
        name=LICENSE_NAME,
        url=LICENSE_URL,
        commercial_use=True,
        attribution_required=False,
    ),
    commercial_use_notes=(
        "Commercial use follows the ElevenLabs plan and terms; free plans may "
        "require attribution. Review the current terms before publishing."
    ),
)

_OUTPUT_FORMATS: dict[AudioFormat, str] = {
    AudioFormat.MP3: "mp3_44100_128",
    AudioFormat.OGG: "opus_48000_64",
    AudioFormat.OPUS: "opus_48000_64",
    AudioFormat.PCM: "pcm_44100",
}


def _payload(request: MusicRequest, model: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"prompt": request.prompt}
    if request.lyrics is not None:
        payload["lyrics"] = request.lyrics
    if request.instrumental:
        payload["force_instrumental"] = True
    if request.duration_seconds is not None:
        milliseconds = round(request.duration_seconds * 1000)
        if not MIN_DURATION_MS <= milliseconds <= MAX_DURATION_MS:
            raise InvalidRequestError(
                "ElevenLabs music duration must be between 10 and 300 seconds",
                details={
                    "provider": "elevenlabs-music",
                    "duration_seconds": request.duration_seconds,
                },
            )
        payload["music_length_ms"] = milliseconds
    if model:
        payload["model_id"] = model
    return payload


def _request_id(response: httpx.Response) -> str | None:
    return as_text(response.headers.get("x-request-id")) or None


class ElevenLabsMusicProvider(Provider):
    """A music provider backed by the ElevenLabs music API.

    ``api_key`` is an already resolved credential while ``api_key_ref`` names a
    secret resolved through ``secrets``. ``base_url`` may point at a local test
    server and ``transport`` injects an ``httpx`` mock transport for the offline
    suite. A request may name a model; otherwise ``default_model`` is used and
    an empty value lets the provider choose its default.
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
            provider="ElevenLabs music",
        )
        self._base_url = validate_base_url(api_url or base_url)
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
    def endpoint(self) -> str:
        """The music generation endpoint derived from :attr:`base_url`."""
        return f"{self._base_url}/v1/music"

    def _headers(self) -> dict[str, str]:
        return {
            "xi-api-key": self._api_key,
            "content-type": "application/json",
            "accept": "audio/*",
        }

    async def generate(self, request: MusicRequest) -> MusicResponse:
        """Generate *request* through ElevenLabs and return its tracks."""
        if not isinstance(request, MusicRequest):
            raise TypeError("request must be a MusicRequest")
        output_format = _OUTPUT_FORMATS.get(request.format)
        if output_format is None:
            raise InvalidRequestError(
                f"ElevenLabs music does not support {request.format.value} output",
                details={
                    "provider": self.name,
                    "format": request.format.value,
                },
            )
        model = request.model or self._default_model
        payload = _payload(request, model)
        tracks: list[MusicTrack] = []
        request_ids: list[str] = []
        for index in range(request.count):
            response = await self.http.post(
                self.endpoint,
                headers=self._headers(),
                params={"output_format": output_format},
                json=payload,
                timeout=request.timeout_seconds,
                raise_for_status=True,
            )
            audio = validate_audio(response.content, provider=self.name)
            request_id = _request_id(response)
            if request_id is not None:
                request_ids.append(request_id)
            tracks.append(
                MusicTrack(
                    data=audio,
                    format=sniff_audio_format(audio, request.format),
                    duration_seconds=request.duration_seconds,
                    sample_rate=request.sample_rate,
                    seed=None if request.seed is None else request.seed + index,
                    title=request.prompt,
                    lyrics=None if request.instrumental else request.lyrics,
                    metadata={
                        "license": LICENSE_NAME,
                        "license_url": LICENSE_URL,
                        "commercial_use": True,
                        "request_id": request_id,
                    },
                )
            )
        return MusicResponse(
            tracks=tracks,
            model=model or "",
            provider=self.name,
            request_id=request_ids[0] if request_ids else None,
            metadata={
                "license": LICENSE_NAME,
                "license_url": LICENSE_URL,
                "commercial_use": True,
                "instrumental": request.instrumental,
            },
        )


ElevenLabsMusic = ElevenLabsMusicProvider
"""Alias for :class:`ElevenLabsMusicProvider`."""

ElevenLabsMusicGen = ElevenLabsMusicProvider
"""Alias for :class:`ElevenLabsMusicProvider`."""

ElevenLabsMusicGenProvider = ElevenLabsMusicProvider
"""Alias for :class:`ElevenLabsMusicProvider`."""

register_provider("elevenlabs-music", ElevenLabsMusicProvider, module=__name__)
register_provider("elevenlabs_music", ElevenLabsMusicProvider, module=__name__)
register_provider("elevenlabs-music-gen", ElevenLabsMusicProvider, module=__name__)

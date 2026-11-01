"""Stable Audio instrumental music adapter.

Stable Audio exposes text to audio through the Stability AI audio API. The
endpoint returns encoded audio, so the adapter maps the prompt, duration, seed
and output format onto a form body and wraps the response in the shared
:class:`~aivia.providers.protocols.music.MusicResponse`.

Stable Audio generates instrumental music only. A request that carries lyrics
is rejected instead of silently dropping them, because dropping a vocal brief
would change what the caller asked for. The adapter records the provider's
licensing terms and commercial-use notes on its metadata.
"""

from __future__ import annotations

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
    "DEFAULT_MODEL",
    "STABLE_AUDIO_BASE_URL",
    "STABLE_AUDIO_DEFAULT_BASE_URL",
    "StableAudio",
    "StableAudioMusic",
    "StableAudioMusicProvider",
    "StableAudioProvider",
]

DEFAULT_BASE_URL = "https://api.stability.ai"
"""The Stability AI API root, without a trailing slash."""

STABLE_AUDIO_DEFAULT_BASE_URL = DEFAULT_BASE_URL
"""Alias that names the music adapter's default API root."""

STABLE_AUDIO_BASE_URL = DEFAULT_BASE_URL
"""Alias for callers that prefer the short base URL name."""

DEFAULT_MODEL = "stable-audio-2"
"""The model named in the default text to audio endpoint."""

LICENSE_NAME = "Stability AI Community License"
LICENSE_URL = "https://stability.ai/license"

MIN_DURATION_SECONDS = 1.0
MAX_DURATION_SECONDS = 190.0

_DEFAULT_METADATA = ProviderMeta(
    name="stable-audio",
    capabilities=[CAPABILITY_MUSIC_GEN],
    supported_options={
        CAPABILITY_MUSIC_GEN: [
            "instrumental",
            "mp3",
            "wav",
            "prompt",
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
        "Commercial use is allowed below the revenue threshold in the "
        "Stability AI Community License; review the current license before "
        "publishing."
    ),
)

_OUTPUT_FORMATS: dict[AudioFormat, str] = {
    AudioFormat.MP3: "mp3",
    AudioFormat.WAV: "wav",
    AudioFormat.PCM: "wav",
}


def _duration(request: MusicRequest) -> float:
    duration = request.duration_seconds or 30.0
    if not MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS:
        raise InvalidRequestError(
            "Stable Audio duration must be between 1 and 190 seconds",
            details={
                "provider": "stable-audio",
                "duration_seconds": duration,
            },
        )
    return duration


def _payload(
    request: MusicRequest,
    model: str,
    output_format: str,
) -> dict[str, str]:
    payload: dict[str, str] = {
        "prompt": request.prompt,
        "duration": f"{_duration(request):g}",
        "output_format": output_format,
        "model": model,
    }
    if request.seed is not None:
        payload["seed"] = str(request.seed)
    return payload


def _request_id(response: httpx.Response) -> str | None:
    return as_text(response.headers.get("x-request-id")) or None


class StableAudioMusicProvider(Provider):
    """An instrumental music provider backed by Stable Audio.

    ``api_key`` is an already resolved credential while ``api_key_ref`` names a
    secret resolved through ``secrets``. ``base_url`` may point at a local test
    server and ``transport`` injects an ``httpx`` mock transport for the offline
    suite. ``default_model`` names the model used when a request does not.
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
        self._default_model = selected or DEFAULT_MODEL
        self._api_key = resolve_api_key(
            api_key,
            api_key_ref,
            secrets,
            provider="Stable Audio",
        )
        self._base_url = validate_base_url(api_url or base_url)
        if transport is not None:
            self._http = self.create_http_client(transport=transport)

    @property
    def base_url(self) -> str:
        """The configured Stability AI API root, without a trailing slash."""
        return self._base_url

    @property
    def api_url(self) -> str:
        """Alias for :attr:`base_url`."""
        return self._base_url

    @property
    def default_model(self) -> str:
        """The model id used when a request does not name one."""
        return self._default_model

    @property
    def model(self) -> str:
        """Alias for :attr:`default_model`."""
        return self._default_model

    @property
    def endpoint(self) -> str:
        """The text to audio endpoint derived from :attr:`base_url`."""
        return self.endpoint_for()

    def endpoint_for(self, model: str | None = None) -> str:
        """The text to audio endpoint for *model*, or the default model."""
        selected = model or self._default_model
        return f"{self._base_url}/v2beta/audio/{selected}/text-to-audio"

    def _headers(self) -> dict[str, str]:
        return {
            "authorization": f"Bearer {self._api_key}",
            "accept": "audio/*",
        }

    async def generate(self, request: MusicRequest) -> MusicResponse:
        """Generate instrumental *request* through Stable Audio."""
        if not isinstance(request, MusicRequest):
            raise TypeError("request must be a MusicRequest")
        if request.lyrics is not None:
            raise InvalidRequestError(
                "Stable Audio generates instrumental music only",
                details={
                    "provider": self.name,
                    "field": "lyrics",
                },
            )
        output_format = _OUTPUT_FORMATS.get(request.format)
        if output_format is None:
            raise InvalidRequestError(
                f"Stable Audio does not support {request.format.value} output",
                details={
                    "provider": self.name,
                    "format": request.format.value,
                },
            )
        model = request.model or self._default_model
        payload = _payload(request, model, output_format)
        tracks: list[MusicTrack] = []
        request_ids: list[str] = []
        for index in range(request.count):
            response = await self.http.post(
                self.endpoint_for(model),
                headers=self._headers(),
                data=payload,
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
                    lyrics=None,
                    metadata={
                        "license": LICENSE_NAME,
                        "license_url": LICENSE_URL,
                        "commercial_use": True,
                        "instrumental": True,
                        "request_id": request_id,
                    },
                )
            )
        return MusicResponse(
            tracks=tracks,
            model=model,
            provider=self.name,
            request_id=request_ids[0] if request_ids else None,
            metadata={
                "license": LICENSE_NAME,
                "license_url": LICENSE_URL,
                "commercial_use": True,
                "instrumental": True,
            },
        )


StableAudio = StableAudioMusicProvider
"""Alias for :class:`StableAudioMusicProvider`."""

StableAudioMusic = StableAudioMusicProvider
"""Alias for :class:`StableAudioMusicProvider`."""

StableAudioProvider = StableAudioMusicProvider
"""Alias for :class:`StableAudioMusicProvider`."""

register_provider("stable-audio", StableAudioMusicProvider, module=__name__)
register_provider("stable_audio", StableAudioMusicProvider, module=__name__)
register_provider("stableaudio", StableAudioMusicProvider, module=__name__)

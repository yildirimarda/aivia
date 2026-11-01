"""ElevenLabs text to speech adapter.

ElevenLabs synthesizes speech with ``POST /v1/text-to-speech/{voice_id}`` and
returns encoded audio. The ``/with-timestamps`` variant returns the audio and a
character alignment, which this adapter converts into word timestamps. Voice
aliases are resolved through a :class:`~aivia.providers.tts.voice.VoiceMap`
before the request is built; the adapter never hard-codes a voice id.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

import httpx

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import InvalidRequestError, ValidationError
from aivia.core.http import raise_for_status
from aivia.core.models.provider import ProviderMeta, ProviderPrice
from aivia.core.secrets import SecretsLoader
from aivia.providers.base import Provider
from aivia.providers.protocols.audio import AudioFormat
from aivia.providers.protocols.tts import CAPABILITY_TTS, TTSRequest, TTSResponse
from aivia.providers.protocols.voice_clone import (
    CAPABILITY_VOICE_CLONE,
    VoiceCloneRequest,
    VoiceCloneResponse,
)
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
    "CAPABILITY_VOICE_CLONE",
    "DEFAULT_BASE_URL",
    "ELEVENLABS_BASE_URL",
    "ELEVENLABS_DEFAULT_BASE_URL",
    "ElevenLabs",
    "ElevenLabsProvider",
    "ElevenLabsTTS",
    "ElevenLabsTTSProvider",
    "VoiceCloneRequest",
    "VoiceCloneResponse",
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


_AUDIO_EXTENSIONS = {
    "audio/flac": "flac",
    "audio/m4a": "m4a",
    "audio/mp3": "mp3",
    "audio/mp4": "m4a",
    "audio/mpeg": "mp3",
    "audio/ogg": "ogg",
    "audio/opus": "opus",
    "audio/wav": "wav",
    "audio/webm": "webm",
    "audio/x-m4a": "m4a",
    "audio/x-wav": "wav",
}


def _voice_clone_enabled(*values: bool | None) -> bool:
    selected = [value for value in values if value is not None]
    for value in selected:
        if not isinstance(value, bool):
            raise TypeError("voice cloning flags must be booleans or None")
    return any(selected)


def _with_voice_clone_capability(
    metadata: ProviderMeta | Mapping[str, Any] | None,
    requested: bool,
) -> tuple[ProviderMeta | Mapping[str, Any] | None, bool]:
    if metadata is None:
        base: ProviderMeta | None = None
    elif isinstance(metadata, ProviderMeta):
        base = metadata
    else:
        base = ProviderMeta.model_validate(metadata)
    declared = base is not None and base.supports(CAPABILITY_VOICE_CLONE)
    enabled = requested or declared
    if not enabled:
        return metadata, False
    if base is None:
        base = ProviderMeta()
    if not declared:
        base = base.model_copy(
            update={"capabilities": [*base.capabilities, CAPABILITY_VOICE_CLONE]}
        )
    return base, True


def _clone_request_data(
    request: VoiceCloneRequest | Mapping[str, Any] | str | None,
    samples: Any,
    values: Mapping[str, Any],
) -> dict[str, Any]:
    if request is None:
        data: dict[str, Any] = {}
    elif isinstance(request, VoiceCloneRequest):
        data = request.model_dump(mode="python")
    elif isinstance(request, Mapping):
        data = dict(request)
    elif isinstance(request, str):
        data = {"name": request}
    else:
        raise TypeError(
            "request must be a VoiceCloneRequest, mapping, string or None"
        )
    if samples is not None and not any(
        key in data for key in ("samples", "files", "audio", "audio_samples")
    ):
        data["samples"] = samples
    data.update(values)
    return data


def _audio_filename(index: int, media_type: str | None) -> str:
    extension = _AUDIO_EXTENSIONS.get((media_type or "").lower(), "wav")
    return f"sample-{index + 1}.{extension}"


def _clone_identifier(
    value: Any,
    provider_voice_id: str | None,
    values: dict[str, Any],
) -> str:
    if value is None:
        value = (
            provider_voice_id
            or values.pop("id", None)
            or values.pop("voice", None)
            or values.pop("clone_id", None)
        )
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, Mapping):
        text = ""
        for key in ("voice_id", "provider_voice_id", "id", "clone_id", "voice"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                text = candidate.strip()
                break
    else:
        text = ""
        for key in ("voice_id", "provider_voice_id", "id", "clone_id", "voice"):
            candidate = getattr(value, key, None)
            if isinstance(candidate, str) and candidate.strip():
                text = candidate.strip()
                break
    if not text:
        raise InvalidRequestError(
            "ElevenLabs voice deletion needs a provider voice id",
            details={"provider": "elevenlabs-tts", "field": "voice_id"},
        )
    return text


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
        voice_clone: bool | None = None,
        voice_cloning: bool | None = None,
        enable_voice_clone: bool | None = None,
        enable_voice_cloning: bool | None = None,
        allow_voice_cloning: bool | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        name: str | None = None,
        health_timeout_seconds: float | None = None,
        close_timeout_seconds: float | None = None,
        http_config: HttpClientConfig | None = None,
    ) -> None:
        metadata, self._voice_clone_enabled = _with_voice_clone_capability(
            type(self).default_metadata,
            _voice_clone_enabled(
                voice_clone,
                voice_cloning,
                enable_voice_clone,
                enable_voice_cloning,
                allow_voice_cloning,
            ),
        )
        super().__init__(
            metadata=metadata,
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

    @property
    def voice_clone_enabled(self) -> bool:
        """Whether this instance declares the ``voice_clone`` capability."""
        return self._voice_clone_enabled

    @property
    def voice_cloning_enabled(self) -> bool:
        """Alias for :attr:`voice_clone_enabled`."""
        return self._voice_clone_enabled

    @property
    def voice_clone(self) -> bool:
        """Alias for :attr:`voice_clone_enabled`."""
        return self._voice_clone_enabled

    @property
    def voice_cloning(self) -> bool:
        """Alias for :attr:`voice_clone_enabled`."""
        return self._voice_clone_enabled

    @property
    def supports_voice_clone(self) -> bool:
        """Alias for :attr:`voice_clone_enabled`."""
        return self._voice_clone_enabled

    @property
    def supports_voice_cloning(self) -> bool:
        """Alias for :attr:`voice_clone_enabled`."""
        return self._voice_clone_enabled

    def _require_voice_clone(self) -> None:
        if not self.supports(CAPABILITY_VOICE_CLONE):
            raise InvalidRequestError(
                "ElevenLabs voice cloning is not enabled",
                details={
                    "provider": self.name,
                    "required_capability": CAPABILITY_VOICE_CLONE,
                },
            )

    def _voice_headers(self) -> dict[str, str]:
        return {
            "xi-api-key": self._api_key,
            "accept": "application/json",
        }

    def clone_endpoint(self) -> str:
        """The endpoint that creates a cloned voice."""
        return f"{self._base_url}/v1/voices/add"

    def voice_endpoint(self, voice_id: str) -> str:
        """The endpoint for one provider voice."""
        return f"{self._base_url}/v1/voices/{voice_id}"

    async def create_voice(
        self,
        request: VoiceCloneRequest | Mapping[str, Any] | str | None = None,
        samples: Any = None,
        **values: Any,
    ) -> VoiceCloneResponse:
        """Create a cloned voice and return its provider voice id."""
        self._require_voice_clone()
        clone = VoiceCloneRequest.model_validate(
            _clone_request_data(request, samples, values)
        )
        if not clone.samples:
            raise InvalidRequestError(
                "ElevenLabs voice cloning needs at least one audio sample",
                details={"provider": self.name, "field": "samples"},
            )
        files: list[tuple[str, tuple[str, bytes, str]]] = []
        for index, sample in enumerate(clone.samples):
            if sample.data is None:
                if sample.url is None:
                    raise ValidationError(
                        "ElevenLabs voice cloning needs inline audio or a URL",
                        details={"provider": self.name, "index": index},
                    )
                downloaded = await self.http.get(
                    sample.url,
                    headers=self._voice_headers(),
                    timeout=clone.timeout_seconds,
                    raise_for_status=True,
                )
                audio = downloaded.content
                media_type = (
                    sample.media_type
                    or as_text(downloaded.headers.get("content-type"))
                    or "application/octet-stream"
                )
            else:
                audio = sample.data
                media_type = sample.media_type or "application/octet-stream"
            if not audio:
                raise ValidationError(
                    "ElevenLabs voice cloning received an empty audio sample",
                    details={"provider": self.name, "index": index},
                )
            filename = sample.filename or _audio_filename(index, media_type)
            files.append(("files", (filename, audio, media_type)))
        form: dict[str, Any] = {"name": clone.name}
        if clone.description:
            form["description"] = clone.description
        if clone.labels:
            form["labels"] = json.dumps(clone.labels, sort_keys=True)
        response = await self.http.post(
            self.clone_endpoint(),
            headers=self._voice_headers(),
            data=form,
            files=files,
            timeout=clone.timeout_seconds,
            raise_for_status=True,
        )
        body = as_mapping(response.json())
        if body is None:
            raise ValidationError(
                "ElevenLabs returned a non-JSON voice clone response",
                details={"provider": self.name},
            )
        voice_id = as_text(body.get("voice_id")) or as_text(body.get("id"))
        if not voice_id:
            raise ValidationError(
                "ElevenLabs returned no voice id for the clone",
                details={"provider": self.name},
            )
        return VoiceCloneResponse(
            voice_id=voice_id,
            provider=self.name,
            name=clone.name,
            owner=clone.owner,
            language=clone.language,
            consent=clone.consent,
            request_id=response.headers.get("x-request-id") or None,
            metadata=clone.metadata,
        )

    async def delete_voice(
        self,
        voice_id: Any = None,
        *,
        provider_voice_id: str | None = None,
        timeout_seconds: float | None = None,
        **values: Any,
    ) -> bool:
        """Delete a provider voice and report whether it existed."""
        self._require_voice_clone()
        identifier = _clone_identifier(voice_id, provider_voice_id, values)
        response = await self.http.delete(
            self.voice_endpoint(identifier),
            headers=self._voice_headers(),
            timeout=timeout_seconds,
            raise_for_status=False,
        )
        if response.status_code == 404:
            return False
        await raise_for_status(response)
        return True

    clone_voice = create_voice
    create_clone = create_voice
    create_voice_clone = create_voice
    delete_clone = delete_voice
    delete_voice_clone = delete_voice
    remove_voice = delete_voice

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

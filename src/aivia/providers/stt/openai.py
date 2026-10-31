"""OpenAI speech to text adapter.

OpenAI exposes transcription through ``POST /v1/audio/transcriptions``. The
adapter uploads the audio as multipart form data, asks for ``verbose_json`` and
converts the reported words and segments into the shared
:class:`~aivia.providers.protocols.stt.STTResponse`. Word timing is requested
only when the caller asks for it.
"""

from __future__ import annotations

from typing import Any

import httpx

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import InvalidRequestError, ValidationError
from aivia.core.models.provider import ProviderMeta, ProviderPrice
from aivia.core.secrets import SecretsLoader
from aivia.providers.base import Provider
from aivia.providers.protocols.stt import (
    CAPABILITY_STT,
    STTRequest,
    STTResponse,
)
from aivia.providers.registry import register_provider
from aivia.providers.stt.common import (
    audio_filename,
    load_audio,
    segments_from_values,
    words_from_values,
)
from aivia.providers.tts.common import (
    as_mapping,
    as_text,
    resolve_api_key,
    validate_base_url,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "OPENAI_DEFAULT_BASE_URL",
    "OPENAI_STT_BASE_URL",
    "OpenAISTT",
    "OpenAISTTProvider",
    "OpenAISpeechToText",
    "OpenAISpeechToTextProvider",
    "OpenAITranscription",
    "OpenAITranscriptionProvider",
    "Whisper",
    "WhisperAPI",
    "WhisperAPIProvider",
    "WhisperSTT",
    "WhisperSTTProvider",
]

DEFAULT_BASE_URL = "https://api.openai.com"
"""The OpenAI API root, without a trailing slash."""

OPENAI_DEFAULT_BASE_URL = DEFAULT_BASE_URL
"""Alias that names the OpenAI API root explicitly."""

OPENAI_STT_BASE_URL = DEFAULT_BASE_URL
"""Alias for callers that prefer the short speech to text base URL name."""

_DEFAULT_METADATA = ProviderMeta(
    name="openai-stt",
    capabilities=[CAPABILITY_STT],
    pricing={CAPABILITY_STT: ProviderPrice(amount_usd=0.006, unit="minute")},
)


def _json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


class OpenAISTTProvider(Provider):
    """A speech to text provider backed by the OpenAI transcription API.

    ``api_key`` is an already resolved credential while ``api_key_ref`` names a
    secret resolved through ``secrets``. ``default_model`` supplies the model id
    when a request does not name one. ``base_url`` may point at a local test
    server and ``transport`` injects an ``httpx`` mock transport for the offline
    suite.
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
            provider="OpenAI",
        )
        self._base_url = validate_base_url(api_url or base_url)
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
    def endpoint(self) -> str:
        """The transcription endpoint derived from :attr:`base_url`."""
        return f"{self._base_url}/v1/audio/transcriptions"

    def _headers(self) -> dict[str, str]:
        return {
            "authorization": f"Bearer {self._api_key}",
            "accept": "application/json",
        }

    async def transcribe(self, request: STTRequest) -> STTResponse:
        """Transcribe *request* through OpenAI and return its timing."""
        if not isinstance(request, STTRequest):
            raise TypeError("request must be a STTRequest")
        if request.diarize:
            raise InvalidRequestError(
                "OpenAI speech to text does not support diarization",
                details={"provider": self.name},
            )
        model = request.model or self._default_model
        if model is None or not model.strip():
            raise InvalidRequestError(
                "OpenAI speech to text needs a model id",
                details={"provider": self.name},
            )
        model = model.strip()
        audio = await load_audio(self, request, self.http)
        filename = audio_filename(request, audio)
        data: dict[str, Any] = {
            "model": model,
            "response_format": "verbose_json",
            "timestamp_granularities[]": (
                ["word", "segment"] if request.word_timestamps else ["segment"]
            ),
        }
        if request.language is not None:
            data["language"] = request.language
        if request.prompt is not None:
            data["prompt"] = request.prompt
        if request.temperature is not None:
            data["temperature"] = str(request.temperature)
        files = {
            "file": (
                filename,
                audio,
                request.media_type or "application/octet-stream",
            )
        }
        response = await self.http.post(
            self.endpoint,
            headers=self._headers(),
            data=data,
            files=files,
            timeout=request.timeout_seconds,
            raise_for_status=True,
        )
        body = as_mapping(_json(response))
        if body is None:
            raise ValidationError(
                "OpenAI speech to text returned a non-JSON response",
                details={"provider": self.name},
            )
        words = words_from_values(body.get("words")) if request.word_timestamps else []
        segments = segments_from_values(
            body.get("segments"),
            keep_words=request.word_timestamps,
        )
        text = as_text(body.get("text"))
        if not text and not segments and not words:
            raise ValidationError(
                "OpenAI speech to text returned an empty transcript",
                details={"provider": self.name},
            )
        duration = body.get("duration")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            duration = None
        else:
            duration = max(0.0, float(duration))
        language = as_text(body.get("language")) or request.language
        metadata: dict[str, Any] = {"words": len(words)}
        if duration is not None:
            metadata["duration_seconds"] = duration
        return STTResponse(
            text=text,
            language=language,
            duration_seconds=duration,
            segments=segments,
            words=words,
            model=model,
            provider=self.name,
            request_id=response.headers.get("x-request-id") or None,
            metadata=metadata,
        )


OpenAISTT = OpenAISTTProvider
"""Alias for :class:`OpenAISTTProvider`."""

OpenAITranscription = OpenAISTTProvider
"""Alias for :class:`OpenAITranscriptionProvider`."""

OpenAITranscriptionProvider = OpenAISTTProvider
"""Alias for :class:`OpenAISTTProvider`."""

OpenAISpeechToText = OpenAISTTProvider
"""Alias for :class:`OpenAISpeechToTextProvider`."""

OpenAISpeechToTextProvider = OpenAISTTProvider
"""Alias for :class:`OpenAISTTProvider`."""

Whisper = OpenAISTTProvider
"""Alias for :class:`OpenAISTTProvider`."""

WhisperSTT = OpenAISTTProvider
"""Alias for :class:`OpenAISTTProvider`."""

WhisperSTTProvider = OpenAISTTProvider
"""Alias for :class:`OpenAISTTProvider`."""

WhisperAPI = OpenAISTTProvider
"""Alias for :class:`OpenAISTTProvider`."""

WhisperAPIProvider = OpenAISTTProvider
"""Alias for :class:`OpenAISTTProvider`."""

register_provider("openai-stt", OpenAISTTProvider, module=__name__)
register_provider("openai_stt", OpenAISTTProvider, module=__name__)
register_provider("openai-transcription", OpenAISTTProvider, module=__name__)
register_provider("openai-whisper", OpenAISTTProvider, module=__name__)
register_provider("whisper", OpenAISTTProvider, module=__name__)
register_provider("whisper-api", OpenAISTTProvider, module=__name__)

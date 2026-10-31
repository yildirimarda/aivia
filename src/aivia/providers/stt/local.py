"""Local speech to text using faster-whisper or an injected transcriber.

A local model runs on the machine and costs nothing, so it is the natural
fallback when no hosted transcription service is configured. The adapter never
needs the optional ``faster-whisper`` dependency in the offline suite: tests
inject a ``transcriber`` callable with the same result shape. When no callable
is supplied, the model is imported and loaded lazily on first use.
"""

from __future__ import annotations

import asyncio
import inspect
import io
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import ConfigError, InvalidRequestError, ValidationError
from aivia.core.models.provider import ProviderMeta, ProviderPrice
from aivia.providers.base import Provider
from aivia.providers.protocols.audio import TranscriptSegment, WordTimestamp
from aivia.providers.protocols.stt import (
    CAPABILITY_STT,
    STTRequest,
    STTResponse,
)
from aivia.providers.registry import register_provider
from aivia.providers.stt.common import (
    load_audio,
    segments_from_values,
    words_from_values,
)

__all__ = [
    "DEFAULT_MODEL",
    "FasterWhisper",
    "FasterWhisperProvider",
    "FasterWhisperSTT",
    "LocalSTT",
    "LocalSTTProvider",
    "Transcriber",
    "WhisperLocal",
    "WhisperLocalSTT",
]

DEFAULT_MODEL = "base"
"""The default faster-whisper model size or path."""

TranscriberResult = STTResponse | Mapping[str, Any]
Transcriber = Callable[
    [bytes, STTRequest, str],
    TranscriberResult | Awaitable[TranscriberResult],
]
"""A callable taking audio, the request and the model name."""

_DEFAULT_METADATA = ProviderMeta(
    name="local-stt",
    capabilities=[CAPABILITY_STT],
    pricing={CAPABILITY_STT: ProviderPrice(amount_usd=0.0, unit="minute")},
)


def _optional_text(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string or None")
    return value.strip() or None


class LocalSTTProvider(Provider):
    """A speech to text provider that runs faster-whisper locally.

    ``transcriber`` replaces model execution entirely, which is how tests stay
    offline. ``model`` or ``default_model`` names the faster-whisper model size
    or path. ``device`` and ``compute_type`` are passed to faster-whisper when a
    model is loaded, and ``download_root`` controls where model files are kept.
    """

    default_metadata = _DEFAULT_METADATA

    def __init__(
        self,
        *,
        transcriber: Transcriber | None = None,
        default_model: str | None = None,
        model: str | None = None,
        device: str = "auto",
        compute_type: str = "default",
        download_root: str | None = None,
        vad_filter: bool = False,
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
        if transcriber is not None and not callable(transcriber):
            raise TypeError("transcriber must be callable or None")
        self._transcriber = transcriber
        selected = model if model is not None else default_model
        self._default_model = _optional_text(selected, field="model") or DEFAULT_MODEL
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty string")
        if not isinstance(compute_type, str) or not compute_type.strip():
            raise ValueError("compute_type must be a non-empty string")
        self._device = device.strip()
        self._compute_type = compute_type.strip()
        self._download_root = _optional_text(download_root, field="download_root")
        if not isinstance(vad_filter, bool):
            raise TypeError("vad_filter must be a bool")
        self._vad_filter = vad_filter
        self._models: dict[str, Any] = {}

    @property
    def transcriber(self) -> Transcriber | None:
        """The injected transcriber, or ``None`` for faster-whisper."""
        return self._transcriber

    @property
    def default_model(self) -> str | None:
        """The model size or path used when a request does not name one."""
        return self._default_model

    @property
    def model(self) -> str | None:
        """Alias for :attr:`default_model`."""
        return self._default_model

    @property
    def device(self) -> str:
        """The device passed to faster-whisper."""
        return self._device

    @property
    def compute_type(self) -> str:
        """The compute type passed to faster-whisper."""
        return self._compute_type

    def _load_model(self, model_name: str) -> Any:
        cached = self._models.get(model_name)
        if cached is not None:
            return cached
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ConfigError(
                "local speech to text needs faster-whisper; install it or pass "
                "a transcriber callable",
                details={"provider": self.name},
            ) from exc
        options: dict[str, Any] = {
            "device": self._device,
            "compute_type": self._compute_type,
        }
        if self._download_root is not None:
            options["download_root"] = self._download_root
        model = WhisperModel(model_name, **options)
        self._models[model_name] = model
        return model

    def _faster_whisper_transcribe(
        self,
        audio: bytes,
        request: STTRequest,
        model_name: str,
    ) -> STTResponse:
        model = self._load_model(model_name)
        segments_iter, info = model.transcribe(
            io.BytesIO(audio),
            language=request.language,
            initial_prompt=request.prompt,
            temperature=request.temperature if request.temperature is not None else 0.0,
            word_timestamps=request.word_timestamps,
            vad_filter=self._vad_filter,
        )
        words: list[WordTimestamp] = []
        segments: list[TranscriptSegment] = []
        for raw_segment in segments_iter:
            start = max(0.0, float(raw_segment.start))
            end = max(start, float(raw_segment.end))
            segment_words: list[WordTimestamp] = []
            if request.word_timestamps:
                for raw_word in raw_segment.words or ():
                    text = (raw_word.word or "").strip()
                    if not text:
                        continue
                    word_start = max(0.0, float(raw_word.start))
                    word_end = max(word_start, float(raw_word.end))
                    probability = getattr(raw_word, "probability", None)
                    confidence = (
                        min(1.0, max(0.0, float(probability)))
                        if isinstance(probability, (int, float))
                        and not isinstance(probability, bool)
                        else None
                    )
                    word = WordTimestamp(
                        word=text,
                        start_seconds=word_start,
                        end_seconds=word_end,
                        confidence=confidence,
                    )
                    segment_words.append(word)
                    words.append(word)
            text = (raw_segment.text or "").strip()
            if text:
                segments.append(
                    TranscriptSegment(
                        text=text,
                        start_seconds=start,
                        end_seconds=end,
                        words=segment_words,
                    )
                )
        text = " ".join(segment.text for segment in segments).strip()
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        if duration <= 0 and segments:
            duration = max(segment.end_seconds for segment in segments)
        language = getattr(info, "language", None) or request.language
        return STTResponse(
            text=text,
            language=language,
            duration_seconds=duration if duration > 0 else None,
            segments=segments,
            words=words,
            model=model_name,
            provider=self.name,
            metadata={
                "backend": "faster-whisper",
                "duration_seconds": duration,
                "words": len(words),
            },
        )

    async def transcribe(self, request: STTRequest) -> STTResponse:
        """Transcribe *request* with a local model and return its timing."""
        if not isinstance(request, STTRequest):
            raise TypeError("request must be a STTRequest")
        if request.diarize:
            raise InvalidRequestError(
                "local speech to text does not support diarization",
                details={"provider": self.name},
            )
        model = request.model or self._default_model
        if model is None or not model.strip():
            raise InvalidRequestError(
                "local speech to text needs a model name",
                details={"provider": self.name},
            )
        model = model.strip()
        audio = await load_audio(self, request, self.http)
        if self._transcriber is not None:
            result = await asyncio.to_thread(
                self._transcriber,
                audio,
                request,
                model,
            )
            if inspect.isawaitable(result):
                result = await result
        else:
            result = await asyncio.to_thread(
                self._faster_whisper_transcribe,
                audio,
                request,
                model,
            )
        if isinstance(result, STTResponse):
            response = result
        elif isinstance(result, Mapping):
            values = dict(result)
            if "words" in values:
                values["words"] = words_from_values(values["words"])
            if "segments" in values:
                values["segments"] = segments_from_values(
                    values["segments"],
                    keep_words=request.word_timestamps,
                )
            values.setdefault("provider", self.name)
            values.setdefault("model", model)
            response = STTResponse.model_validate(values)
        else:
            raise ValidationError(
                "local speech to text returned an invalid result",
                details={"provider": self.name},
            )
        updates: dict[str, Any] = {}
        if not response.provider:
            updates["provider"] = self.name
        if not response.model:
            updates["model"] = model
        if not request.word_timestamps and (
            response.words or any(segment.words for segment in response.segments)
        ):
            updates["words"] = []
            updates["segments"] = [
                segment.model_copy(update={"words": []})
                for segment in response.segments
            ]
        return response.model_copy(update=updates) if updates else response


FasterWhisper = LocalSTTProvider
"""Alias for :class:`LocalSTTProvider`."""

FasterWhisperSTT = LocalSTTProvider
"""Alias for :class:`LocalSTTProvider`."""

FasterWhisperProvider = LocalSTTProvider
"""Alias for :class:`LocalSTTProvider`."""

LocalSTT = LocalSTTProvider
"""Alias for :class:`LocalSTTProvider`."""

WhisperLocal = LocalSTTProvider
"""Alias for :class:`LocalSTTProvider`."""

WhisperLocalSTT = LocalSTTProvider
"""Alias for :class:`LocalSTTProvider`."""

register_provider("local-stt", LocalSTTProvider, module=__name__)
register_provider("local_stt", LocalSTTProvider, module=__name__)
register_provider("faster-whisper", LocalSTTProvider, module=__name__)
register_provider("faster_whisper", LocalSTTProvider, module=__name__)
register_provider("local-whisper", LocalSTTProvider, module=__name__)

"""Shared helpers for the speech to text adapters.

Every adapter receives either inline audio or an audio URL and must return the
same transcript shape. These helpers load the audio once, name the upload, and
convert provider JSON into the validated word and segment models. A provider
that reports timings with different key names is normalized here rather than in
each adapter.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from aivia.core.errors import ValidationError
from aivia.core.http import HttpClient
from aivia.providers.base import Provider
from aivia.providers.protocols.audio import TranscriptSegment, WordTimestamp
from aivia.providers.protocols.stt import STTRequest, STTResponse
from aivia.providers.tts.common import as_mapping, as_text

__all__ = [
    "audio_filename",
    "load_audio",
    "response_duration_seconds",
    "segments_from_values",
    "words_from_values",
]

_MEDIA_EXTENSIONS: dict[str, str] = {
    "audio/aac": "aac",
    "audio/flac": "flac",
    "audio/m4a": "m4a",
    "audio/mp3": "mp3",
    "audio/mp4": "mp4",
    "audio/mpeg": "mp3",
    "audio/mpga": "mpga",
    "audio/ogg": "ogg",
    "audio/opus": "opus",
    "audio/wav": "wav",
    "audio/wave": "wav",
    "audio/webm": "webm",
    "audio/x-m4a": "m4a",
    "audio/x-wav": "wav",
    "video/mp4": "mp4",
    "video/mpeg": "mpeg",
    "video/webm": "webm",
}


def audio_filename(request: STTRequest, audio: bytes) -> str:
    """Return an upload filename for *request* or derive one from its media type."""
    if request.filename:
        return request.filename
    media_type = request.media_type or ""
    extension = _MEDIA_EXTENSIONS.get(media_type, "wav")
    return f"audio.{extension}"


async def load_audio(
    provider: Provider,
    request: STTRequest,
    client: HttpClient,
) -> bytes:
    """Return the inline audio, or fetch and validate the request audio URL."""
    if request.audio is not None:
        return request.audio
    if request.audio_url is None:
        raise ValidationError(
            "speech to text request has no audio source",
            details={"provider": provider.name},
        )
    response = await client.get(
        request.audio_url,
        timeout=request.timeout_seconds,
        raise_for_status=True,
    )
    audio = response.content
    if not audio:
        raise ValidationError(
            f"{provider.name} fetched empty audio",
            details={"provider": provider.name},
        )
    return audio


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _value(raw: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in raw:
            return raw[key]
    return None


def _word(raw: Any) -> WordTimestamp | None:
    mapping = as_mapping(raw)
    if mapping is None:
        return None
    text = as_text(_value(mapping, "word", "text"))
    start = _number(_value(mapping, "start", "start_seconds", "start_time"))
    end = _number(_value(mapping, "end", "end_seconds", "end_time"))
    if not text or start is None:
        return None
    if end is None or end < start:
        end = start
    confidence = _number(_value(mapping, "confidence", "probability", "score"))
    if confidence is not None:
        confidence = min(1.0, max(0.0, confidence))
    speaker = as_text(_value(mapping, "speaker", "speaker_id")) or None
    return WordTimestamp(
        word=text,
        start_seconds=max(0.0, start),
        end_seconds=max(0.0, end),
        confidence=confidence,
        speaker=speaker,
    )


def words_from_values(values: Any) -> list[WordTimestamp]:
    """Convert a provider word list into ordered word timestamps."""
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
        return []
    words = [word for raw in values if (word := _word(raw)) is not None]
    return sorted(words, key=lambda word: word.start_seconds)


def _segment(raw: Any, *, keep_words: bool) -> TranscriptSegment | None:
    mapping = as_mapping(raw)
    if mapping is None:
        return None
    text = as_text(_value(mapping, "text", "transcript"))
    start = _number(_value(mapping, "start", "start_seconds", "start_time"))
    end = _number(_value(mapping, "end", "end_seconds", "end_time"))
    if not text or start is None:
        return None
    if end is None or end < start:
        end = start
    words = words_from_values(_value(mapping, "words", "tokens")) if keep_words else []
    confidence = _number(_value(mapping, "confidence", "score"))
    if confidence is not None:
        confidence = min(1.0, max(0.0, confidence))
    speaker = as_text(_value(mapping, "speaker", "speaker_id")) or None
    return TranscriptSegment(
        text=text,
        start_seconds=max(0.0, start),
        end_seconds=max(0.0, end),
        words=words,
        confidence=confidence,
        speaker=speaker,
    )


def segments_from_values(
    values: Any,
    *,
    keep_words: bool = True,
) -> list[TranscriptSegment]:
    """Convert a provider segment list into ordered transcript segments."""
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
        return []
    segments = [
        segment
        for raw in values
        if (segment := _segment(raw, keep_words=keep_words)) is not None
    ]
    return sorted(segments, key=lambda segment: segment.start_seconds)


def response_duration_seconds(response: STTResponse) -> float | None:
    """Return the audio duration reported by *response* or inferred from timing."""
    if response.duration_seconds is not None:
        return response.duration_seconds
    reported = _number(response.metadata.get("duration_seconds"))
    if reported is not None:
        return max(0.0, reported)
    candidates = [word.end_seconds for word in response.all_words]
    candidates.extend(segment.end_seconds for segment in response.segments)
    if candidates:
        return max(candidates)
    return None

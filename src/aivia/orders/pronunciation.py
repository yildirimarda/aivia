"""Verify a spoken name in generated speech and music.

The recipient's name is synthesized as speech and set to music, both carrying
the phonetic hints, then the spoken result is transcribed. When the
transcription does not carry the name well enough, speech and music are
regenerated with the hints, up to the configured retry limit.
"""
# config-coverage: allow-file — attempts and score are local loop state
# (attempt counter and best name-to-transcript score), not configuration
# thresholds that belong in config or data files.

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from aivia.core.config.orders import NameVerifyConfig
from aivia.core.errors import ConfigError
from aivia.media.voices import apply_pronunciation, normalize_pronunciation
from aivia.providers.protocols.music import MusicRequest, MusicResponse
from aivia.providers.protocols.stt import STTResponse
from aivia.providers.protocols.tts import TTSRequest, TTSResponse

__all__ = [
    "NamePronunciationError",
    "NamePronunciationResult",
    "verify_name_pronunciation",
]


class NamePronunciationError(ConfigError):
    """The name could not be verified within the retry limit."""

    code = "name_pronunciation"
    retryable = False


@dataclass(frozen=True, slots=True)
class NamePronunciationResult:
    """The outcome of one name pronunciation verification."""

    name: str
    """The recipient name that was verified."""
    language: str
    """The language the name was spoken in."""
    hints: dict[str, str]
    """The resolved word to phonetic hint mapping."""
    attempts: int
    """Total synthesis attempts made, including the first."""
    regenerated: int
    """How many times speech and music were regenerated after a mismatch."""
    verified: bool
    """Whether the name survived transcription within the threshold."""
    score: float
    """Best name-to-transcript score in ``[0.0, 1.0]``."""
    speech_request: TTSRequest
    """The last speech request sent to the provider, hints included."""
    music_request: MusicRequest
    """The last music request sent to the provider, hints included."""
    transcript: str
    """The best transcription returned for the spoken name."""
    reason: str | None = None
    """Why verification failed, or ``None`` when it passed."""


def _require_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _language_key(value: str) -> str:
    return value.strip().replace("_", "-").lower()


def _words(text: str) -> list[str]:
    return [
        word.strip(".,!?;:\"'()[]{}").casefold()
        for word in text.split()
        if word.strip(".,!?;:\"'()[]{}")
    ]


def _name_score(transcript: str, name: str) -> float:
    """Return the fraction of *name* words present in *transcript*."""
    name_words = _words(name)
    if not name_words:
        return 1.0
    transcript_words = set(_words(transcript))
    matched = sum(1 for word in name_words if word in transcript_words)
    return matched / len(name_words)


def _speech_request(name: str, hints: dict[str, str], language: str) -> TTSRequest:
    request = TTSRequest(
        text=name,
        voice="default",
        language=language or None,
        pronunciation=dict(hints),
    )
    return apply_pronunciation(request, supported=True)


def _music_request(name: str, hints: dict[str, str]) -> MusicRequest:
    return MusicRequest(
        prompt=f"A personalized song for {name}",
        lyrics=name,
        pronunciation=dict(hints),
    )


async def verify_name_pronunciation(
    *,
    name: str,
    hints: object | None = None,
    language: str = "en",
    config: NameVerifyConfig | None = None,
    synthesize_speech: Callable[[TTSRequest], Awaitable[TTSResponse]],
    generate_music: Callable[[MusicRequest], Awaitable[MusicResponse]],
    transcribe: Callable[[bytes], Awaitable[STTResponse]],
) -> NamePronunciationResult:
    """Verify *name* survives synthesis and transcription.

    Builds a TTS request and a music request both carrying the phonetic hints,
    synthesizes them, transcribes the spoken name and checks the name is still
    present. On a mismatch the speech and music are regenerated, up to
    ``config.retries`` regenerations. When the limit is exhausted without a
    match, :class:`NamePronunciationError` is raised.
    """
    resolved_name = _require_text(name, field="name")
    resolved_hints = normalize_pronunciation(hints)
    resolved_config = config if config is not None else NameVerifyConfig()
    resolved_language = _language_key(language)

    speech_request = _speech_request(resolved_name, resolved_hints, resolved_language)
    music_request = _music_request(resolved_name, resolved_hints)

    if not resolved_config.enabled:
        return NamePronunciationResult(
            name=resolved_name,
            language=resolved_language,
            hints=dict(resolved_hints),
            attempts=0,
            regenerated=0,
            verified=True,
            score=0.0,
            speech_request=speech_request,
            music_request=music_request,
            transcript="",
            reason=None,
        )

    max_regenerations = resolved_config.retries
    threshold = resolved_config.threshold
    regenerated = 0
    attempts = 1
    score = 0.0
    transcript = ""
    reason: str | None = None

    while True:
        speech = await synthesize_speech(speech_request)
        await generate_music(music_request)
        transcript_response = await transcribe(speech.audio)
        transcript = transcript_response.text
        score = _name_score(transcript, resolved_name)
        if score >= threshold:
            return NamePronunciationResult(
                name=resolved_name,
                language=resolved_language,
                hints=dict(resolved_hints),
                attempts=attempts,
                regenerated=regenerated,
                verified=True,
                score=score,
                speech_request=speech_request,
                music_request=music_request,
                transcript=transcript,
                reason=None,
            )
        if regenerated >= max_regenerations:
            reason = (
                f"name {resolved_name!r} scored {score:.3f} below the "
                f"threshold {threshold:g} after {max_regenerations} "
                f"regeneration(s)"
            )
            raise NamePronunciationError(
                reason,
                details={
                    "name": resolved_name,
                    "language": resolved_language,
                    "hints": dict(resolved_hints),
                    "attempts": attempts,
                    "regenerated": regenerated,
                    "score": score,
                    "threshold": threshold,
                    "transcript": transcript,
                },
            )
        regenerated += 1
        attempts += 1

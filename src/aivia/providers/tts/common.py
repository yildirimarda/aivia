"""Shared helpers for the text to speech adapters.

The helpers here are deliberately small: URL and credential validation, safe
JSON access, audio format sniffing and the alignment conversion used when a
provider reports character timings. Keeping them in one module means every
adapter rejects bad configuration and empty audio the same way.
"""

from __future__ import annotations

import base64
import binascii
import io
import re
import wave
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from aivia.core.errors import ConfigError, ValidationError
from aivia.core.secrets import SecretsLoader
from aivia.providers.protocols.audio import AudioFormat, WordTimestamp

__all__ = [
    "as_mapping",
    "as_text",
    "decode_base64_audio",
    "format_language",
    "pcm_to_wav",
    "phoneme_ssml",
    "resolve_api_key",
    "sniff_audio_format",
    "validate_base_url",
    "word_timestamps_from_alignment",
]

_WORD = re.compile(r"\S+")
_LANGUAGE = re.compile(r"^([a-z]{2,3})-([a-z]{2})$")


def validate_base_url(value: str, *, field: str = "base_url") -> str:
    """Return *value* as an absolute, credential-free http(s) URL."""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    text = value.strip().rstrip("/")
    parsed = urlsplit(text)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ConfigError(
            f"{field} must be an absolute http(s) URL: {value!r}",
            details={"field": field},
        )
    if parsed.username is not None or parsed.password is not None:
        raise ConfigError(
            f"{field} must not contain credentials",
            details={"field": field},
        )
    return text


def resolve_api_key(
    api_key: str | None,
    api_key_ref: str | None,
    secrets: SecretsLoader | None,
    *,
    provider: str,
) -> str:
    """Resolve an API key from a literal value or a secret reference."""
    if secrets is not None and not isinstance(secrets, SecretsLoader):
        raise TypeError("secrets must be a SecretsLoader")
    if api_key is not None:
        if not isinstance(api_key, str):
            raise TypeError("api_key must be a string")
        text = api_key.strip()
        if not text:
            raise ConfigError(f"{provider} API key must be non-empty")
        return text
    if api_key_ref is None or not isinstance(api_key_ref, str):
        raise ConfigError(
            f"{provider} requires an API key or api_key_ref",
            details={"field": "api_key_ref"},
        )
    reference = api_key_ref.strip()
    if not reference:
        raise ConfigError(
            f"{provider} requires an API key or api_key_ref",
            details={"field": "api_key_ref"},
        )
    loader = secrets or SecretsLoader()
    return loader.get(reference)


def format_language(value: str) -> str:
    """Return a BCP 47 language tag with an upper-case region subtag."""
    text = value.strip()
    match = _LANGUAGE.fullmatch(text)
    if match is None:
        return text
    return f"{match.group(1)}-{match.group(2).upper()}"


def as_mapping(value: Any) -> Mapping[str, Any] | None:
    """Return *value* when it is a mapping, otherwise ``None``."""
    return value if isinstance(value, Mapping) else None


def as_text(value: Any) -> str:
    """Return a stripped string for *value*, or an empty string."""
    return value.strip() if isinstance(value, str) else ""


def decode_base64_audio(value: str, *, provider: str) -> bytes:
    """Decode a base64 audio payload, accepting a data URL prefix."""
    text = value.strip()
    if text.startswith("data:"):
        _, _, text = text.partition(",")
    try:
        data = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValidationError(
            f"{provider} returned invalid base64 audio",
            details={"provider": provider},
        ) from exc
    if not data:
        raise ValidationError(
            f"{provider} returned empty audio",
            details={"provider": provider},
        )
    return data


def sniff_audio_format(data: bytes, fallback: AudioFormat) -> AudioFormat:
    """Detect a common audio container from its leading bytes."""
    if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        return AudioFormat.WAV
    if data.startswith(b"OggS"):
        return AudioFormat.OGG
    if data.startswith(b"fLaC"):
        return AudioFormat.FLAC
    if data.startswith(b"ADIF") or (
        len(data) > 1 and data[0] == 0xFF and data[1] & 0xF6 == 0xF0
    ):
        return AudioFormat.AAC
    if data.startswith(b"ID3") or (
        len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0
    ):
        return AudioFormat.MP3
    if data.startswith(b"ftyp") or data[4:8] == b"ftyp":
        return AudioFormat.M4A
    return fallback


def pcm_to_wav(
    data: bytes,
    *,
    sample_rate: int,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Wrap raw little-endian PCM bytes in a RIFF/WAVE container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(sample_width)
        target.setframerate(sample_rate)
        target.writeframes(data)
    return buffer.getvalue()


def phoneme_ssml(
    text: str,
    pronunciation: Mapping[str, str] | None = None,
    *,
    alphabet: str = "ipa",
) -> str:
    """Render *text* as escaped SSML with optional phoneme replacements.

    The text is escaped before markup is inserted, so a pronunciation hint can
    never inject SSML. Longer hints are matched first and matching is
    case-insensitive at word boundaries.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if pronunciation is None:
        hints: dict[str, str] = {}
    elif isinstance(pronunciation, Mapping):
        hints = {}
        for raw_key, raw_value in pronunciation.items():
            key = as_text(raw_key)
            value = as_text(raw_value)
            if not key or not value:
                raise ValueError("pronunciation hints must be non-empty")
            hints[key] = value
    else:
        raise TypeError("pronunciation must be a mapping")
    if not hints:
        return _escape_xml(text)
    words = sorted(hints, key=len, reverse=True)
    pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(word) for word in words) + r")(?!\w)",
        re.IGNORECASE,
    )
    folded = {key.casefold(): value for key, value in hints.items()}
    pieces: list[str] = []
    cursor = 0
    for match in pattern.finditer(text):
        pieces.append(_escape_xml(text[cursor : match.start()]))
        word = match.group()
        phonetic = hints.get(word) or folded.get(word.casefold(), "")
        pieces.append(
            f'<phoneme alphabet="{_escape_attribute(alphabet)}" '
            f'ph="{_escape_attribute(phonetic)}">'
            f"{_escape_xml(word)}</phoneme>"
        )
        cursor = match.end()
    pieces.append(_escape_xml(text[cursor:]))
    return "".join(pieces)


def _escape_xml(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _escape_attribute(value: str) -> str:
    return _escape_xml(value).replace('"', "&quot;")


def word_timestamps_from_alignment(
    text: str,
    characters: Sequence[str],
    starts: Sequence[float],
    ends: Sequence[float],
) -> list[WordTimestamp]:
    """Convert character alignment into ordered word timestamps.

    Providers such as ElevenLabs report timing per character. Words are found
    in the original text and matched against the aligned character stream. A
    mismatch in normalization falls back to skipping that word rather than
    inventing a timing.
    """
    if not characters or len(characters) != len(starts) or len(characters) != len(ends):
        return []
    aligned = "".join(characters)
    folded = aligned.casefold()
    cursor = 0
    words: list[WordTimestamp] = []
    for match in _WORD.finditer(text):
        word = match.group()
        index = aligned.find(word, cursor)
        if index < 0:
            index = folded.find(word.casefold(), cursor)
        if index < 0:
            continue
        last = min(index + len(word) - 1, len(starts) - 1)
        words.append(
            WordTimestamp(
                word=word,
                start_seconds=starts[index],
                end_seconds=ends[last],
            )
        )
        cursor = index + len(word)
    return words

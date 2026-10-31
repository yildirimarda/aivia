"""Shared helpers for the music generation adapters.

The helpers here are deliberately small: URL and credential validation, safe
JSON access and audio container sniffing. Keeping them in one module means
every music adapter rejects bad configuration and empty audio the same way.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from aivia.core.errors import ConfigError, ValidationError
from aivia.core.secrets import SecretsLoader
from aivia.providers.protocols.audio import AudioFormat

__all__ = [
    "as_mapping",
    "as_text",
    "resolve_api_key",
    "sniff_audio_format",
    "validate_audio",
    "validate_base_url",
]


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


def as_mapping(value: Any) -> Mapping[str, Any] | None:
    """Return *value* when it is a mapping, otherwise ``None``."""
    return value if isinstance(value, Mapping) else None


def as_text(value: Any) -> str:
    """Return a stripped string for *value*, or an empty string."""
    return value.strip() if isinstance(value, str) else ""


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


def validate_audio(data: bytes, *, provider: str) -> bytes:
    """Return *data* or raise when a provider returned an empty track."""
    if not data:
        raise ValidationError(
            f"{provider} returned empty audio",
            details={"provider": provider},
        )
    return data

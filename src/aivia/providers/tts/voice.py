"""Map logical voice names to provider voice ids.

Configuration and content profiles name voices by a stable, human readable
alias such as ``narrator`` or ``host_de``. A provider only understands its own
voice ids, so every adapter resolves the requested voice through a
:class:`VoiceMap` before it builds a request. An alias missing from the map is
passed through unchanged, which lets an operator use a provider id directly
without maintaining an identity mapping.
"""

from __future__ import annotations

from collections.abc import Mapping

from aivia.core.errors import InvalidRequestError

__all__ = ["VoiceMap", "VoiceMapping"]


def _voice(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{label} must be non-empty")
    return text


class VoiceMap:
    """A case-insensitive alias map from logical names to provider voice ids.

    The mapping is copied on construction and never mutated. Keys and values
    are stripped; an empty key or value is rejected because it would silently
    make every unmapped voice resolve to nothing.
    """

    def __init__(
        self,
        mapping: Mapping[str, str] | None = None,
        *,
        default: str | None = None,
    ) -> None:
        if mapping is None:
            mapping = {}
        if not isinstance(mapping, Mapping):
            raise TypeError("voice mapping must be a mapping")
        voices: dict[str, str] = {}
        for raw_key, raw_value in mapping.items():
            key = _voice(raw_key, label="voice alias")
            value = _voice(raw_value, label="voice id")
            folded = key.casefold()
            if folded in voices:
                raise ValueError(f"duplicate voice alias: {key}")
            voices[folded] = value
        self._voices = voices
        self._default = (
            _voice(default, label="default voice") if default is not None else None
        )

    @classmethod
    def from_value(
        cls,
        value: VoiceMap | Mapping[str, str] | None,
        *,
        default: str | None = None,
    ) -> VoiceMap:
        """Return *value* as a :class:`VoiceMap`, copying a mapping once."""
        if isinstance(value, VoiceMap):
            return value
        return cls(value, default=default)

    @property
    def voices(self) -> dict[str, str]:
        """A copy of the alias to voice id mapping."""
        return dict(self._voices)

    @property
    def default(self) -> str | None:
        """The fallback used when the requested voice is empty."""
        return self._default

    def resolve(self, voice: str) -> str:
        """Return the provider voice id for *voice*.

        An alias is matched case-insensitively. An unknown, non-empty voice is
        passed through unchanged so a caller can name a provider id directly.
        """
        if not isinstance(voice, str):
            raise TypeError("voice must be a string")
        requested = voice.strip()
        if not requested:
            if self._default is not None:
                return self._default
            raise InvalidRequestError("text to speech request needs a voice")
        return self._voices.get(requested.casefold(), requested)

    def __contains__(self, voice: object) -> bool:
        return isinstance(voice, str) and voice.strip().casefold() in self._voices

    def __len__(self) -> int:
        return len(self._voices)


VoiceMapping = VoiceMap
"""Alias for :class:`VoiceMap`."""

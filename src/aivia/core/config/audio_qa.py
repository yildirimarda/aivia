"""Audio quality assurance: thresholds, pronunciations and failure policy.

The audio QA checks compare synthesized speech against the expected script and
report silence, duplication, truncation and mispronunciation findings. This
section exposes the deterministic thresholds used by those checks, the accepted
pronunciations for terms that a transcript may spell differently, and whether a
finding severe enough to be an error should fail the stage.

Defaults mirror the deterministic checker so an empty section behaves exactly
like the module defaults. Thresholds may be written flat or under a
``thresholds`` mapping; both forms validate the same fields.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import Field, field_validator, model_validator

from aivia.core.config.base import ConfigModel
from aivia.core.config.media import (
    DEFAULT_AUDIO_QA_EXPECTED_WORDS_PER_SECOND,
    DEFAULT_AUDIO_QA_MAX_INTERNAL_SILENCE_SECONDS,
    DEFAULT_AUDIO_QA_MAX_LEADING_SILENCE_SECONDS,
    DEFAULT_AUDIO_QA_MAX_MISPRONUNCIATION_RATIO,
    DEFAULT_AUDIO_QA_MAX_SILENCE_RATIO,
    DEFAULT_AUDIO_QA_MAX_TRAILING_SILENCE_SECONDS,
    DEFAULT_AUDIO_QA_MIN_DURATION_RATIO,
    DEFAULT_AUDIO_QA_MIN_WORD_CONFIDENCE,
    DEFAULT_AUDIO_QA_SILENT_RATIO,
)

__all__ = [
    "AudioQA",
    "AudioQAConfig",
    "AudioQa",
    "AudioQaConfig",
    "AudioQualityConfig",
]

_AUDIO_QA_ALIASES: dict[str, tuple[str, ...]] = {
    "enabled": ("enable", "active"),
    "max_internal_silence_seconds": (
        "max_internal_silence",
        "max_internal_pause_seconds",
    ),
    "max_leading_silence_seconds": ("max_leading_silence",),
    "max_trailing_silence_seconds": ("max_trailing_silence",),
    "max_silence_ratio": ("silence_ratio",),
    "silent_ratio": ("almost_silent_ratio",),
    "min_word_confidence": ("min_confidence",),
    "max_mispronunciation_ratio": (
        "mispronunciation_ratio",
        "max_mispronunciation",
    ),
    "expected_words_per_second": ("words_per_second", "wps"),
    "min_duration_ratio": ("duration_ratio",),
    "pronunciations": (
        "pronunciation",
        "pronunciation_hints",
        "pronunciation_map",
    ),
    "fail_on_error": ("fail_on_errors", "raise_on_error", "strict"),
}

_DURATION_FIELDS = (
    "max_internal_silence_seconds",
    "max_leading_silence_seconds",
    "max_trailing_silence_seconds",
)
_RATIO_FIELDS = (
    "max_silence_ratio",
    "silent_ratio",
    "min_word_confidence",
    "max_mispronunciation_ratio",
    "min_duration_ratio",
)
_THRESHOLD_FIELDS = (
    *_DURATION_FIELDS,
    *_RATIO_FIELDS,
    "expected_words_per_second",
)


def _accept_aliases(data: Any, aliases: Mapping[str, tuple[str, ...]]) -> Any:
    if not isinstance(data, Mapping):
        return data
    values = dict(data)
    for canonical, names in aliases.items():
        for name in names:
            if name in values:
                value = values.pop(name)
                values.setdefault(canonical, value)
    return values


def _flatten_thresholds(values: dict[str, Any]) -> dict[str, Any]:
    nested = values.pop("thresholds", None)
    if nested is None:
        return values
    if not isinstance(nested, Mapping):
        raise ValueError("thresholds must be a mapping")
    for name, value in nested.items():
        values.setdefault(name, value)
    return values


def _normalise_pronunciations(value: Any) -> dict[str, list[str]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("pronunciations must be a mapping")
    pronunciations: dict[str, list[str]] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise ValueError("pronunciation keys must be non-empty strings")
        if isinstance(raw_value, str):
            entries: Sequence[Any] = (raw_value,)
        elif isinstance(raw_value, Sequence) and not isinstance(
            raw_value, (bytes, bytearray)
        ):
            entries = raw_value
        else:
            raise ValueError(
                "pronunciation values must be a string or a sequence of strings"
            )
        accepted: list[str] = []
        for entry in entries:
            if not isinstance(entry, str):
                raise ValueError("pronunciation entries must be strings")
            text = entry.strip()
            if text and text not in accepted:
                accepted.append(text)
        if not accepted:
            raise ValueError(f"pronunciation for {raw_key.strip()!r} must not be empty")
        pronunciations[raw_key.strip()] = accepted
    return pronunciations


class AudioQAConfig(ConfigModel):
    """Thresholds, accepted pronunciations and failure policy for audio QA."""

    enabled: bool = True
    max_internal_silence_seconds: float = Field(
        default=DEFAULT_AUDIO_QA_MAX_INTERNAL_SILENCE_SECONDS,
        ge=0,
    )
    max_leading_silence_seconds: float = Field(
        default=DEFAULT_AUDIO_QA_MAX_LEADING_SILENCE_SECONDS,
        ge=0,
    )
    max_trailing_silence_seconds: float = Field(
        default=DEFAULT_AUDIO_QA_MAX_TRAILING_SILENCE_SECONDS,
        ge=0,
    )
    max_silence_ratio: float = Field(
        default=DEFAULT_AUDIO_QA_MAX_SILENCE_RATIO,
        ge=0,
        le=1,
    )
    silent_ratio: float = Field(
        default=DEFAULT_AUDIO_QA_SILENT_RATIO,
        ge=0,
        le=1,
    )
    min_word_confidence: float = Field(
        default=DEFAULT_AUDIO_QA_MIN_WORD_CONFIDENCE,
        ge=0,
        le=1,
    )
    max_mispronunciation_ratio: float = Field(
        default=DEFAULT_AUDIO_QA_MAX_MISPRONUNCIATION_RATIO,
        ge=0,
        le=1,
    )
    expected_words_per_second: float = Field(
        default=DEFAULT_AUDIO_QA_EXPECTED_WORDS_PER_SECOND,
        gt=0,
    )
    min_duration_ratio: float = Field(
        default=DEFAULT_AUDIO_QA_MIN_DURATION_RATIO,
        ge=0,
        le=1,
    )
    pronunciations: dict[str, list[str]] = Field(default_factory=dict)
    fail_on_error: bool = True

    @model_validator(mode="before")
    @classmethod
    def _normalise_input(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = _flatten_thresholds(_accept_aliases(dict(data), _AUDIO_QA_ALIASES))
        if "pronunciations" in values:
            values["pronunciations"] = _normalise_pronunciations(
                values["pronunciations"]
            )
        return values

    @field_validator(*_DURATION_FIELDS)
    @classmethod
    def _finite_duration(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("silence duration must be finite")
        return value

    @field_validator(*_RATIO_FIELDS)
    @classmethod
    def _finite_ratio(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("ratio must be finite")
        return value

    @field_validator("expected_words_per_second")
    @classmethod
    def _finite_rate(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("expected_words_per_second must be finite")
        return value

    @property
    def thresholds(self) -> dict[str, float]:
        """The configured checker thresholds keyed by checker argument."""
        return {name: getattr(self, name) for name in _THRESHOLD_FIELDS}

    @property
    def pronunciation(self) -> dict[str, list[str]]:
        """Alias for :attr:`pronunciations`."""
        return self.pronunciations

    @property
    def fail_on_errors(self) -> bool:
        """Alias for :attr:`fail_on_error`."""
        return self.fail_on_error

    @property
    def strict(self) -> bool:
        """Alias for :attr:`fail_on_error`."""
        return self.fail_on_error


AudioQaConfig = AudioQAConfig
AudioQualityConfig = AudioQAConfig
AudioQA = AudioQAConfig
AudioQa = AudioQAConfig

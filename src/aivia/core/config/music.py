"""Background music defaults: source, library, mood, gain and ducking.

A run asks for a mood rather than a vendor. ``source`` chooses whether a bed
comes from the local royalty free ``library``, from a paid ``provider`` or from
``auto`` (the library first, then the provider). ``library_path`` points at the
library manifest, ``mood`` is the default style request, and ``volume_db`` is
the gain applied before ducking.

``ducking`` holds the sidechain settings used when a voice is present: the
threshold where compression starts, the compression ratio, and the attack and
release times. The values mirror the deterministic audio mixer so the music
stage can pass them through unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import AliasChoices, Field, field_validator, model_validator

from aivia.core.config.base import ConfigModel
from aivia.core.config.media import (
    DEFAULT_AUDIO_DUCKING_ATTACK_MS,
    DEFAULT_AUDIO_DUCKING_RATIO,
    DEFAULT_AUDIO_DUCKING_RELEASE_MS,
    DEFAULT_AUDIO_DUCKING_THRESHOLD_DB,
    DEFAULT_MUSIC_BACKGROUND_VOLUME_DB,
)

__all__ = [
    "MusicConfig",
    "MusicDucking",
    "MusicDuckingConfig",
    "MusicSource",
    "MusicSourceMode",
]


class MusicSource(StrEnum):
    """Where a background music bed may come from."""

    AUTO = "auto"
    LIBRARY = "library"
    PROVIDER = "provider"


MusicSourceMode = MusicSource

_SOURCE_ALIASES = {
    "ai": MusicSource.PROVIDER,
    "auto": MusicSource.AUTO,
    "fallback": MusicSource.AUTO,
    "generate": MusicSource.PROVIDER,
    "generated": MusicSource.PROVIDER,
    "local": MusicSource.LIBRARY,
    "library": MusicSource.LIBRARY,
    "library_only": MusicSource.LIBRARY,
    "provider": MusicSource.PROVIDER,
}

_MUSIC_ALIASES: dict[str, tuple[str, ...]] = {
    "source": ("source_mode", "mode"),
    "library_path": ("library", "library_file", "manifest", "music_library"),
    "volume_db": ("volume", "gain_db", "background_volume_db"),
    "ducking": ("duck", "ducking_settings"),
}

_DUCKING_ALIASES: dict[str, tuple[str, ...]] = {
    "enabled": ("on", "duck"),
    "threshold_db": ("threshold", "duck_threshold_db"),
    "ratio": ("duck_ratio",),
    "attack_ms": ("attack",),
    "release_ms": ("release",),
}


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


def _finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


class MusicDuckingConfig(ConfigModel):
    """Sidechain compression that lowers music while a voice is present."""

    enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("enabled", "on", "duck"),
    )
    threshold_db: float = Field(
        default=DEFAULT_AUDIO_DUCKING_THRESHOLD_DB,
        validation_alias=AliasChoices(
            "threshold_db",
            "threshold",
            "duck_threshold_db",
        ),
    )
    ratio: float = Field(
        default=DEFAULT_AUDIO_DUCKING_RATIO,
        validation_alias=AliasChoices("ratio", "duck_ratio"),
    )
    attack_ms: float = Field(
        default=DEFAULT_AUDIO_DUCKING_ATTACK_MS,
        validation_alias=AliasChoices("attack_ms", "attack"),
    )
    release_ms: float = Field(
        default=DEFAULT_AUDIO_DUCKING_RELEASE_MS,
        validation_alias=AliasChoices("release_ms", "release"),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        return _accept_aliases(data, _DUCKING_ALIASES)

    @field_validator("threshold_db", "ratio", "attack_ms", "release_ms")
    @classmethod
    def _finite_values(cls, value: float) -> float:
        return _finite(value, "music ducking value")

    @model_validator(mode="after")
    def _validate_ranges(self) -> MusicDuckingConfig:
        if self.threshold_db > 0:
            raise ValueError("threshold_db must not be positive")
        if self.ratio < 1:
            raise ValueError("ratio must be at least 1")
        if self.attack_ms < 0 or self.release_ms < 0:
            raise ValueError("attack_ms and release_ms must not be negative")
        return self

    @property
    def threshold(self) -> float:
        """Alias for :attr:`threshold_db`."""
        return self.threshold_db


MusicDucking = MusicDuckingConfig


class MusicConfig(ConfigModel):
    """Default music source, local library, mood, gain and ducking settings."""

    source: MusicSource = MusicSource.AUTO
    library_path: str = ""
    mood: str = ""
    volume_db: float = DEFAULT_MUSIC_BACKGROUND_VOLUME_DB
    ducking: MusicDuckingConfig = Field(default_factory=MusicDuckingConfig)

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = _accept_aliases(dict(data), _MUSIC_ALIASES)
        ducking = values.get("ducking")
        if isinstance(ducking, bool):
            values["ducking"] = {"enabled": ducking}
        elif isinstance(ducking, str):
            key = ducking.strip().lower()
            if key in {"on", "yes", "true", "enable", "enabled"}:
                values["ducking"] = {"enabled": True}
            elif key in {"off", "no", "false", "disable", "disabled"}:
                values["ducking"] = {"enabled": False}
        if values.get("ducking") is None:
            values["ducking"] = {}
        return values

    @field_validator("source", mode="before")
    @classmethod
    def _normalise_source(cls, value: Any) -> Any:
        if isinstance(value, MusicSource):
            return value
        if isinstance(value, str):
            key = value.strip().lower().replace("-", "_").replace(" ", "_")
            mode = _SOURCE_ALIASES.get(key)
            if mode is not None:
                return mode
        return value

    @field_validator("library_path", "mood")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("volume_db")
    @classmethod
    def _finite_volume(cls, value: float) -> float:
        return _finite(value, "volume_db")

    @property
    def library(self) -> str:
        """Alias for :attr:`library_path`."""
        return self.library_path

    @property
    def volume(self) -> float:
        """Alias for :attr:`volume_db`."""
        return self.volume_db

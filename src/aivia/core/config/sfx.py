"""Sound effect defaults: rule files, gain and placement.

A run asks for a rule set rather than a vendor. ``rule_files`` lists the
YAML/JSON files that describe which effects to place, ``volume_db`` is the gain
applied when a rule does not set its own, and ``placement`` is where an effect
sits relative to a beat (start, center, end, matched word, explicit offset or
absolute timeline position). The values mirror the deterministic sound effect
stage so a later command can pass them through unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import AliasChoices, Field, field_validator, model_validator

from aivia.core.config.base import ConfigModel
from aivia.core.config.media import DEFAULT_SFX_VOLUME_DB

__all__ = [
    "SfxConfig",
    "SfxPlacement",
    "SfxPlacementMode",
    "SfxPosition",
]


class SfxPlacement(StrEnum):
    """Where an effect is placed relative to a beat or the timeline."""

    START = "start"
    CENTER = "center"
    END = "end"
    WORD = "word"
    OFFSET = "offset"
    ABSOLUTE = "absolute"


SfxPlacementMode = SfxPlacement
SfxPosition = SfxPlacement

_PLACEMENT_ALIASES = {
    "absolute": SfxPlacement.ABSOLUTE,
    "after": SfxPlacement.END,
    "anchor": SfxPlacement.WORD,
    "at_end": SfxPlacement.END,
    "at_start": SfxPlacement.START,
    "begin": SfxPlacement.START,
    "beginning": SfxPlacement.START,
    "beat_end": SfxPlacement.END,
    "beat_start": SfxPlacement.START,
    "before": SfxPlacement.START,
    "center": SfxPlacement.CENTER,
    "centre": SfxPlacement.CENTER,
    "close": SfxPlacement.END,
    "delay": SfxPlacement.OFFSET,
    "end": SfxPlacement.END,
    "finish": SfxPlacement.END,
    "global": SfxPlacement.ABSOLUTE,
    "intro": SfxPlacement.START,
    "keyword": SfxPlacement.WORD,
    "match": SfxPlacement.WORD,
    "mid": SfxPlacement.CENTER,
    "middle": SfxPlacement.CENTER,
    "offset": SfxPlacement.OFFSET,
    "open": SfxPlacement.START,
    "outro": SfxPlacement.END,
    "relative": SfxPlacement.OFFSET,
    "start": SfxPlacement.START,
    "timeline": SfxPlacement.ABSOLUTE,
    "word": SfxPlacement.WORD,
}

_SFX_ALIASES: dict[str, tuple[str, ...]] = {
    "rule_files": (
        "rules",
        "files",
        "rule_file",
        "rules_file",
        "rule_paths",
        "paths",
        "cues",
    ),
    "volume_db": (
        "volume",
        "gain_db",
        "gain",
        "default_volume_db",
        "default_volume",
    ),
    "placement": (
        "position",
        "place",
        "anchor",
        "default_placement",
        "default_position",
    ),
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


class SfxConfig(ConfigModel):
    """Default rule files, gain and placement for sound effects."""

    rule_files: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("rule_files", "rules", "files"),
    )
    volume_db: float = Field(
        default=DEFAULT_SFX_VOLUME_DB,
        validation_alias=AliasChoices(
            "volume_db",
            "volume",
            "gain_db",
            "gain",
            "default_volume_db",
            "default_volume",
        ),
    )
    placement: SfxPlacement = Field(
        default=SfxPlacement.START,
        validation_alias=AliasChoices(
            "placement",
            "position",
            "place",
            "anchor",
            "default_placement",
            "default_position",
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _normalise_aliases(cls, data: Any) -> Any:
        return _accept_aliases(data, _SFX_ALIASES)

    @field_validator("rule_files", mode="before")
    @classmethod
    def _normalise_rule_files(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            items: list[Any] = [value]
        elif isinstance(value, (Mapping, bytes, bytearray)):
            raise ValueError("rule_files must be a path or a sequence of paths")
        elif isinstance(value, Sequence):
            items = list(value)
        else:
            raise ValueError("rule_files must be a path or a sequence of paths")
        files: list[str] = []
        for item in items:
            if not isinstance(item, str):
                raise ValueError("rule_files entries must be paths")
            text = item.strip()
            if text and text not in files:
                files.append(text)
        return files

    @field_validator("placement", mode="before")
    @classmethod
    def _normalise_placement(cls, value: Any) -> Any:
        if isinstance(value, SfxPlacement):
            return value
        if isinstance(value, str):
            key = value.strip().lower().replace("-", "_").replace(" ", "_")
            placement = _PLACEMENT_ALIASES.get(key)
            if placement is not None:
                return placement
        return value

    @field_validator("volume_db")
    @classmethod
    def _finite_volume(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("volume_db must be finite")
        return value

    @property
    def rules(self) -> list[str]:
        """Alias for :attr:`rule_files`."""
        return self.rule_files

    @property
    def volume(self) -> float:
        """Alias for :attr:`volume_db`."""
        return self.volume_db

    @property
    def position(self) -> SfxPlacement:
        """Alias for :attr:`placement`."""
        return self.placement

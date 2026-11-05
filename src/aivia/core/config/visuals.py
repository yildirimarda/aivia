"""Visual assignment defaults: strategy and the AI video beat limit.

The visuals stage needs to know how each script beat should be illustrated and
how many beats may become AI generated video. ``strategy`` names the default
assignment (``hybrid`` mixes AI video and images, ``image`` uses images for
every beat and the other names select a single source). ``max_ai_video_beats``
is a hard cap: no assignment ever asks for more generated video than this.

The section is read by the visual assigner so a run can change its defaults
without passing options to every stage. An explicit stage option still wins.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Self

from pydantic import AliasChoices, Field, field_validator

from aivia.core.config.base import ConfigModel
from aivia.core.config.writing import (
    DEFAULT_MAX_AI_VIDEO_BEATS,
    DEFAULT_VISUAL_STRATEGY,
)
from aivia.core.errors import ConfigError

__all__ = [
    "DEFAULT_MAX_AI_VIDEO_BEATS",
    "DEFAULT_VISUAL_STRATEGY",
    "VISUALS_SECTION_KEY",
    "VisualStrategy",
    "VisualStrategyConfig",
    "VisualsConfig",
    "normalise_visual_strategy",
    "normalize_visual_strategy",
]

VISUALS_SECTION_KEY = "visuals"
"""The top-level config key that owns the visual assignment defaults."""

VISUALS_SECTION_ALIASES = ("visuals", "visual_strategy", "visual-strategy")
"""Accepted top-level keys when reading a config that predates the section."""

_STRATEGY_KEYS = (
    "strategy",
    "default",
    "default_strategy",
    "visual_strategy",
    "mode",
    "source",
)
_LIMIT_KEYS = (
    "max_ai_video_beats",
    "max_video_beats",
    "max_beats",
    "ai_video_beats",
    "max_ai_videos",
)


class VisualStrategy(StrEnum):
    """The ways a script's beats can be assigned a visual source."""

    HYBRID = "hybrid"
    IMAGE = "image"
    VIDEO = "video"
    STOCK = "stock"
    STILL = "still"
    TEXT = "text"
    NONE = "none"
    EXISTING = "existing"


VISUAL_STRATEGY_ALIASES: dict[str, VisualStrategy] = {
    "ai_image": VisualStrategy.IMAGE,
    "ai_video": VisualStrategy.VIDEO,
    "assets": VisualStrategy.EXISTING,
    "auto": VisualStrategy.HYBRID,
    "existing_assets": VisualStrategy.EXISTING,
    "image_gen": VisualStrategy.IMAGE,
    "images": VisualStrategy.IMAGE,
    "keep": VisualStrategy.EXISTING,
    "mix": VisualStrategy.HYBRID,
    "mixed": VisualStrategy.HYBRID,
    "provided": VisualStrategy.EXISTING,
    "provided_assets": VisualStrategy.EXISTING,
    "video_gen": VisualStrategy.VIDEO,
    "videos": VisualStrategy.VIDEO,
}


def _normalise_name(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("visual strategy must be a string")
    text = value.strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def normalise_visual_strategy(value: Any = None) -> VisualStrategy:
    """Return *value* as a :class:`VisualStrategy`.

    An empty value means the default. Common aliases such as ``ai_video`` or
    ``provided`` are accepted so callers can use the vocabulary of the config
    and estimate layers. Anything else fails with the allowed names.
    """
    if value is None or value == "":
        return VisualStrategy(DEFAULT_VISUAL_STRATEGY)
    if isinstance(value, VisualStrategy):
        return value
    name = _normalise_name(value)
    if name in VISUAL_STRATEGY_ALIASES:
        return VISUAL_STRATEGY_ALIASES[name]
    try:
        return VisualStrategy(name)
    except ValueError:
        allowed = ", ".join(strategy.value for strategy in VisualStrategy)
        raise ValueError(
            f"unknown visual strategy: {value!r} (allowed: {allowed})"
        ) from None


normalize_visual_strategy = normalise_visual_strategy


class VisualsConfig(ConfigModel):
    """Default visual strategy and the maximum number of AI video beats."""

    strategy: VisualStrategy = Field(
        default=VisualStrategy(DEFAULT_VISUAL_STRATEGY),
        validation_alias=AliasChoices(
            "strategy",
            "default",
            "default_strategy",
            "visual_strategy",
            "mode",
            "source",
        ),
    )
    max_ai_video_beats: int = Field(
        default=DEFAULT_MAX_AI_VIDEO_BEATS,
        ge=0,
        validation_alias=AliasChoices(
            "max_ai_video_beats",
            "max_video_beats",
            "max_beats",
            "ai_video_beats",
            "max_ai_videos",
        ),
    )

    @field_validator("strategy", mode="before")
    @classmethod
    def _validate_strategy(cls, value: Any) -> VisualStrategy:
        return normalise_visual_strategy(value)

    @property
    def default(self) -> VisualStrategy:
        """Alias for :attr:`strategy` used by the ``visual_strategy`` spelling."""
        return self.strategy

    @property
    def default_strategy(self) -> VisualStrategy:
        """Alias for :attr:`strategy`."""
        return self.strategy

    @property
    def visual_strategy(self) -> VisualStrategy:
        """Alias for :attr:`strategy`."""
        return self.strategy

    @property
    def max_video_beats(self) -> int:
        """Alias for :attr:`max_ai_video_beats`."""
        return self.max_ai_video_beats

    @classmethod
    def from_config(cls, config: Any = None) -> Self:
        """Read the visuals section from an app config, mapping or section."""
        if config is None:
            return cls()
        if isinstance(config, cls):
            return config

        section: Any = None
        if isinstance(config, Mapping):
            for key in VISUALS_SECTION_ALIASES:
                if key in config:
                    section = config[key]
                    break
            else:
                if any(name in config for name in (*_STRATEGY_KEYS, *_LIMIT_KEYS)):
                    section = config
                else:
                    return cls()
        else:
            for key in VISUALS_SECTION_ALIASES:
                section = getattr(config, key, None)
                if section is not None:
                    break
            else:
                if any(
                    hasattr(config, name) for name in (*_STRATEGY_KEYS, *_LIMIT_KEYS)
                ):
                    section = config
                else:
                    return cls()

        if section is None:
            return cls()
        if isinstance(section, cls):
            return section
        if isinstance(section, Mapping):
            return cls.from_mapping(section)
        if isinstance(section, ConfigModel):
            return cls.from_mapping(
                {
                    name: getattr(section, name)
                    for name in cls.model_fields
                    if hasattr(section, name)
                }
            )
        values = {
            name: getattr(section, name)
            for name in cls.model_fields
            if hasattr(section, name)
        }
        if not values:
            raise ConfigError(
                "visuals config must be a mapping or VisualsConfig",
                details={"section": type(section).__name__},
            )
        return cls.from_mapping(values)


VisualStrategyConfig = VisualsConfig

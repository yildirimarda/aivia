"""Research configuration: how long repeats are suppressed and how close is close.

The research module owns the ``research`` section. It is not part of the core
section set, so a configuration without the module still validates and a
``research`` key without the module fails with the usual unknown-section error.
Profiles tune the section through their normal ``overrides`` mapping, for
example ``profiles.news.overrides.research.dedupe_window_days: 7``.

``dedupe_window`` is a :class:`~datetime.timedelta`. A bare number in config is
read as a number of days, matching ``--dedupe-window`` and the topic modes, and
strings such as ``30d`` or ``2 weeks`` are accepted for readability.
``embedding_threshold`` is the cosine similarity at or above which two titles
are the same topic; ``None`` disables embedding matching while keeping the
normalized-title check.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from datetime import timedelta
from typing import Any, ClassVar, Self

from pydantic import Field, field_validator, model_validator

from aivia.core.config.base import ConfigModel
from aivia.core.config.reference import DEFAULT_TEXT_SIMILARITY_THRESHOLD
from aivia.core.config.registry import SectionRegistry, registry
from aivia.core.errors import ConfigError

__all__ = [
    "DEFAULT_DEDUPE_WINDOW",
    "DEFAULT_DEDUPE_WINDOW_DAYS",
    "DEFAULT_EMBEDDING_THRESHOLD",
    "RESEARCH_SECTION_KEY",
    "DedupeWindow",
    "ResearchConfig",
    "ResearchSettings",
    "ResearchSettingsConfig",
    "register_research_config",
    "register_research_section",
    "research_config",
]

RESEARCH_SECTION_KEY = "research"
"""The top-level config key owned by the research module."""

DEFAULT_DEDUPE_WINDOW_DAYS = 365.0
"""A full year, so an annually recurring topic is not repeated."""

DEFAULT_EMBEDDING_THRESHOLD = DEFAULT_TEXT_SIMILARITY_THRESHOLD
"""Cosine similarity at or above this marks two topics as the same story."""

_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "dedupe_window": (
        "dedupe_window_days",
        "window",
        "window_days",
        "repeat_window",
        "history_window",
    ),
    "embedding_threshold": (
        "similarity_threshold",
        "embedding_similarity_threshold",
        "dedupe_threshold",
        "cosine_threshold",
        "threshold",
    ),
}

_DURATION = re.compile(
    r"^(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*"
    r"(?P<unit>d|day|days|w|week|weeks|mo|month|months|y|year|years|"
    r"h|hour|hours|m|min|minute|minutes)?$",
    re.IGNORECASE,
)

_UNIT_DAYS = {
    "d": 1.0,
    "day": 1.0,
    "days": 1.0,
    "w": 7.0,
    "week": 7.0,
    "weeks": 7.0,
    "mo": 30.0,
    "month": 30.0,
    "months": 30.0,
    "y": 365.0,
    "year": 365.0,
    "years": 365.0,
    "h": 1.0 / 24.0,
    "hour": 1.0 / 24.0,
    "hours": 1.0 / 24.0,
    "m": 1.0 / 1440.0,
    "min": 1.0 / 1440.0,
    "minute": 1.0 / 1440.0,
    "minutes": 1.0 / 1440.0,
}


class DedupeWindow(timedelta):
    """A repeat-suppression window that also compares to a number of days.

    The config value is a real :class:`~datetime.timedelta`, but a bare number
    is the number of days. Comparing to either keeps the model ergonomic for
    callers that read ``dedupe_window == 30`` and for callers that read
    ``dedupe_window == timedelta(days=30)``.
    """

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return self.total_seconds() == float(other) * 86400.0
        return timedelta.__eq__(self, other)

    __hash__ = timedelta.__hash__

    def __float__(self) -> float:
        """Return the window in days."""
        return self.total_seconds() / 86400.0


DEFAULT_DEDUPE_WINDOW = DedupeWindow(days=DEFAULT_DEDUPE_WINDOW_DAYS)
"""The default repeat-suppression window."""


def _coerce_window(value: Any) -> DedupeWindow:
    if isinstance(value, DedupeWindow):
        return value
    if isinstance(value, timedelta):
        return DedupeWindow(seconds=value.total_seconds())
    if isinstance(value, bool):
        raise ConfigError("dedupe_window must be a number of days or a duration")
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ConfigError("dedupe_window must be finite")
        if number < 0:
            raise ConfigError("dedupe_window must not be negative")
        return DedupeWindow(days=number)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ConfigError("dedupe_window must not be empty")
        match = _DURATION.fullmatch(text)
        if match is None:
            raise ConfigError(
                "dedupe_window must be a number of days or a duration such as 30d"
            )
        number = float(match.group("value"))
        unit = (match.group("unit") or "d").lower()
        days = number * _UNIT_DAYS[unit]
        if not math.isfinite(days):
            raise ConfigError("dedupe_window must be finite")
        if days < 0:
            raise ConfigError("dedupe_window must not be negative")
        return DedupeWindow(days=days)
    raise ConfigError("dedupe_window must be a number of days or a duration")


def _coerce_threshold(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError("embedding_threshold must be a number or null")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError("embedding_threshold must be finite")
    if not 0.0 <= number <= 1.0:
        raise ConfigError("embedding_threshold must be between zero and one")
    return number


def _accept_aliases(data: Any) -> Any:
    if not isinstance(data, Mapping):
        return data
    values = dict(data)
    for canonical, aliases in _FIELD_ALIASES.items():
        if canonical in values:
            for alias in aliases:
                values.pop(alias, None)
            continue
        for alias in aliases:
            if alias in values:
                values[canonical] = values.pop(alias)
                break
    return values


class ResearchConfig(ConfigModel):
    """Repeat suppression settings owned by the research module."""

    dedupe_window: timedelta = Field(default=DEFAULT_DEDUPE_WINDOW)
    embedding_threshold: float | None = Field(
        default=DEFAULT_EMBEDDING_THRESHOLD,
        ge=0,
        le=1,
    )

    _section_key: ClassVar[str] = RESEARCH_SECTION_KEY

    @model_validator(mode="before")
    @classmethod
    def _accept_field_aliases(cls, data: Any) -> Any:
        return _accept_aliases(data)

    @field_validator("dedupe_window", mode="before")
    @classmethod
    def _validate_window(cls, value: Any) -> DedupeWindow:
        return _coerce_window(value)

    @field_validator("embedding_threshold", mode="before")
    @classmethod
    def _validate_threshold(cls, value: Any) -> float | None:
        return _coerce_threshold(value)

    @property
    def window(self) -> timedelta:
        """Alias for :attr:`dedupe_window`."""
        return self.dedupe_window

    @property
    def window_days(self) -> float:
        """The dedupe window expressed in days."""
        return self.dedupe_window.total_seconds() / 86400.0

    @property
    def dedupe_window_days(self) -> float:
        """Alias for :attr:`window_days`."""
        return self.window_days

    @property
    def dedupe_window_seconds(self) -> float:
        """The dedupe window expressed in seconds."""
        return self.dedupe_window.total_seconds()

    @property
    def threshold(self) -> float | None:
        """Alias for :attr:`embedding_threshold`."""
        return self.embedding_threshold

    @property
    def similarity_threshold(self) -> float | None:
        """Alias for :attr:`embedding_threshold`."""
        return self.embedding_threshold

    def dedupe_options(self) -> dict[str, Any]:
        """Return the keywords accepted by the dedupe stage."""
        return {
            "window": self.dedupe_window,
            "embedding_threshold": self.embedding_threshold,
        }

    def to_dedupe_options(self) -> dict[str, Any]:
        """Alias for :meth:`dedupe_options`."""
        return self.dedupe_options()

    @classmethod
    def from_config(cls, config: Any) -> Self:
        """Read the research section from an app config or a root mapping."""
        section = (
            config.get(RESEARCH_SECTION_KEY)
            if isinstance(config, Mapping)
            else getattr(config, RESEARCH_SECTION_KEY, None)
        )
        if section is None:
            return cls()
        if isinstance(section, cls):
            return section
        if isinstance(section, Mapping):
            return cls.from_mapping(section)
        raise ConfigError("research config must be a mapping or ResearchConfig")

    @classmethod
    def from_resolved(cls, resolved: Any) -> Self:
        """Read the research section from a resolved config object."""
        values = getattr(resolved, "values", resolved)
        return cls.from_config(values)


ResearchSettings = ResearchConfig
ResearchSettingsConfig = ResearchConfig


def register_research_section(
    sections: SectionRegistry | None = None,
) -> SectionRegistry:
    """Register the ``research`` section on *sections* and return the registry.

    The call is idempotent for :class:`ResearchConfig`. A key already owned by
    another model still fails, so two modules cannot silently claim ``research``.
    """
    target = registry if sections is None else sections
    existing = (
        target.get(RESEARCH_SECTION_KEY) if RESEARCH_SECTION_KEY in target else None
    )
    if existing is not None:
        if existing.model is ResearchConfig:
            return target
        raise ConfigError(
            f"config section already registered: {RESEARCH_SECTION_KEY}",
            details={
                "key": RESEARCH_SECTION_KEY,
                "existing": existing.model.__name__,
                "new": ResearchConfig.__name__,
            },
        )
    target.register(
        RESEARCH_SECTION_KEY,
        ResearchConfig,
        module="research",
    )
    return target


register_research_config = register_research_section


def research_config(config: Any) -> ResearchConfig:
    """Return the research section of *config*, or defaults when it is absent."""
    return ResearchConfig.from_config(config)

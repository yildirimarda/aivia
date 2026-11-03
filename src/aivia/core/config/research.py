"""Research configuration: repeat suppression and topic sensitivity rules.

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

``sensitivity`` holds overrides for the research sensitivity filter. Profiles
can add blocked or review terms, add whole rules, and disable a default rule by
name without restating the defaults. Flat ``sensitivity_*`` keys are accepted
as aliases for the nested mapping, so both of these are equivalent::

    research:
      sensitivity:
        blocked_terms: [lottery]
        disabled_rules: [tragedy]

    research:
      sensitivity_blocked_terms: [lottery]
      sensitivity_disabled_rules: [tragedy]

``max_topics`` caps how many topics a run may select; ``None`` means no cap.
``max_per_category`` is the diversity limit, so one category cannot crowd out
the rest; ``0`` or ``None`` disables the category limit. ``diversity`` is
accepted as a boolean or integer alias for ``max_per_category``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from datetime import timedelta
from typing import Any, ClassVar, Self

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

from aivia.core.config.base import ConfigModel
from aivia.core.config.reference import DEFAULT_TEXT_SIMILARITY_THRESHOLD
from aivia.core.config.registry import SectionRegistry, registry
from aivia.core.errors import ConfigError

__all__ = [
    "DEFAULT_DEDUPE_WINDOW",
    "DEFAULT_DEDUPE_WINDOW_DAYS",
    "DEFAULT_EMBEDDING_THRESHOLD",
    "DEFAULT_MAX_PER_CATEGORY",
    "DEFAULT_MAX_TOPICS",
    "DEFAULT_TOPICS_PER_RUN",
    "RESEARCH_SECTION_KEY",
    "DedupeWindow",
    "ResearchConfig",
    "ResearchSettings",
    "ResearchSettingsConfig",
    "SensitivityConfig",
    "SensitivityRuleConfig",
    "SensitivityRuleSettings",
    "SensitivitySettings",
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

DEFAULT_MAX_TOPICS: int | None = None
"""No per-run topic limit is applied unless the profile sets one."""

DEFAULT_TOPICS_PER_RUN = DEFAULT_MAX_TOPICS
"""Alias for :data:`DEFAULT_MAX_TOPICS`."""

DEFAULT_MAX_PER_CATEGORY: int | None = 1
"""At most one topic from each category is selected by default."""

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
    "max_topics": (
        "limit",
        "topics_per_run",
        "per_run",
        "topic_limit",
        "per_run_limit",
        "count",
        "size",
        "max_results",
        "max_count",
    ),
    "max_per_category": (
        "per_category",
        "diversity",
        "category_limit",
        "max_category",
        "max_per_category_count",
        "max_per_category_size",
        "category_cap",
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


def _coerce_max_topics(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError("max_topics must be a non-negative integer or null")
    if value < 0:
        raise ConfigError("max_topics must not be negative")
    return value


def _coerce_max_per_category(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return DEFAULT_MAX_PER_CATEGORY if value else None
    if not isinstance(value, int):
        raise ConfigError(
            "max_per_category must be a non-negative integer, boolean or null"
        )
    if value < 0:
        raise ConfigError("max_per_category must not be negative")
    return None if value == 0 else value


def _normalise_selection(data: Any) -> Any:
    if not isinstance(data, Mapping):
        return data
    values = dict(data)
    nested = values.pop("selection", None)
    if nested is None:
        return values
    if not isinstance(nested, Mapping):
        raise ConfigError("selection must be a mapping")
    merged = dict(nested)
    merged.update(values)
    return merged


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


_SENSITIVITY_ACTION_ALIASES: dict[str, str] = {
    "allow": "allow",
    "allowed": "allow",
    "pass": "allow",
    "ok": "allow",
    "review": "manual_review",
    "manual": "manual_review",
    "manual_review": "manual_review",
    "manualreview": "manual_review",
    "flag": "manual_review",
    "hold": "manual_review",
    "pending": "manual_review",
    "block": "blocked",
    "blocked": "blocked",
    "deny": "blocked",
    "reject": "blocked",
    "exclude": "blocked",
}

_SENSITIVITY_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "enabled": ("enable", "active", "screening"),
    "rules": (
        "rule",
        "overrides",
        "rule_overrides",
        "sensitivity_rules",
        "topic_rules",
        "custom_rules",
    ),
    "blocked_terms": (
        "blocked",
        "block_terms",
        "blocked_topics",
        "block_topics",
        "deny_terms",
        "blocked_keywords",
    ),
    "review_terms": (
        "review",
        "review_topics",
        "manual_review",
        "manual_review_terms",
        "manual_review_topics",
        "hold_terms",
    ),
    "disabled_rules": (
        "disabled",
        "disable_rules",
        "disabled_sensitivity_rules",
        "ignored_rules",
        "allowed_rules",
        "exclude_rules",
        "skip_rules",
    ),
}

_SENSITIVITY_ROOT_ALIASES: dict[str, tuple[str, ...]] = {
    "sensitivity_enabled": (
        "sensitivity_enable",
        "screen_sensitive_topics",
    ),
    "sensitivity_rules": (
        "rules",
        "rule",
        "sensitivity_rule_overrides",
        "rule_overrides",
        "topic_rules",
        "custom_rules",
    ),
    "sensitivity_blocked_terms": (
        "blocked_terms",
        "blocked_topics",
        "block_terms",
        "block_topics",
        "deny_terms",
        "blocked_keywords",
    ),
    "sensitivity_review_terms": (
        "review_terms",
        "review_topics",
        "manual_review_terms",
        "manual_review_topics",
        "hold_terms",
    ),
    "sensitivity_disabled_rules": (
        "disabled_rules",
        "disabled_sensitivity_rules",
        "disable_rules",
        "ignored_rules",
        "allowed_rules",
        "exclude_rules",
        "skip_rules",
    ),
}


def _clean_terms(value: Any, *, label: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw: Iterable[Any] = value.replace(";", ",").replace("\n", ",").split(",")
    elif isinstance(value, Iterable) and not isinstance(
        value, (bytes, bytearray, Mapping)
    ):
        raw = value
    else:
        raise ConfigError(f"{label} must be a string or a list of strings")
    terms: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            raise ConfigError(f"{label} must contain only strings")
        text = item.strip()
        if not text:
            continue
        key = text.casefold()
        if key not in seen:
            seen.add(key)
            terms.append(text)
    return terms


def _coerce_sensitivity_action(value: Any) -> str:
    if not isinstance(value, str):
        raise ConfigError("sensitivity action must be a string")
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return _SENSITIVITY_ACTION_ALIASES[key]
    except KeyError:
        allowed = ", ".join(sorted(set(_SENSITIVITY_ACTION_ALIASES.values())))
        raise ConfigError(
            f"unknown sensitivity action: {value!r} (allowed: {allowed})"
        ) from None


def _accept_field_aliases(
    data: Any,
    aliases: Mapping[str, tuple[str, ...]],
) -> Any:
    if not isinstance(data, Mapping):
        return data
    values = dict(data)
    for canonical, names in aliases.items():
        if canonical in values:
            for alias in names:
                values.pop(alias, None)
            continue
        for alias in names:
            if alias in values:
                values[canonical] = values.pop(alias)
                break
    return values


class SensitivityRuleConfig(ConfigModel):
    """One sensitivity rule override loaded from research configuration."""

    name: str = Field(
        min_length=1,
        validation_alias=AliasChoices("name", "id", "rule", "rule_id", "label"),
    )
    terms: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "terms",
            "keywords",
            "matches",
            "phrases",
            "blocked_terms",
        ),
    )
    action: str = Field(
        default="manual_review",
        validation_alias=AliasChoices("action", "decision", "outcome"),
    )
    category: str = ""
    reason: str = ""
    enabled: bool = True

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        name = value.strip()
        if not name:
            raise ValueError("sensitivity rule name must not be empty")
        return name

    @field_validator("terms", mode="before")
    @classmethod
    def _validate_terms(cls, value: Any) -> list[str]:
        terms = _clean_terms(value, label="sensitivity terms")
        if not terms:
            raise ValueError("sensitivity rule must have at least one term")
        return terms

    @field_validator("action", mode="before")
    @classmethod
    def _validate_action(cls, value: Any) -> str:
        return _coerce_sensitivity_action(value)

    @field_validator("category", "reason", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the rule."""
        return {
            "name": self.name,
            "terms": list(self.terms),
            "action": self.action,
            "category": self.category,
            "reason": self.reason,
            "enabled": self.enabled,
        }


class SensitivityConfig(ConfigModel):
    """Sensitivity overrides that add, replace or disable research rules."""

    enabled: bool = True
    rules: list[SensitivityRuleConfig] = Field(default_factory=list)
    blocked_terms: list[str] = Field(default_factory=list)
    review_terms: list[str] = Field(default_factory=list)
    disabled_rules: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_field_aliases(cls, data: Any) -> Any:
        return _accept_field_aliases(data, _SENSITIVITY_FIELD_ALIASES)

    @field_validator("blocked_terms", "review_terms", mode="before")
    @classmethod
    def _validate_terms(cls, value: Any) -> list[str]:
        return _clean_terms(value, label="sensitivity terms")

    @field_validator("disabled_rules", mode="before")
    @classmethod
    def _validate_disabled(cls, value: Any) -> list[str]:
        return _clean_terms(value, label="disabled sensitivity rule names")

    @field_validator("rules", mode="before")
    @classmethod
    def _coerce_rules(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, (Mapping, SensitivityRuleConfig)):
            return [value]
        return value

    @field_validator("rules")
    @classmethod
    def _unique_rules(
        cls, value: list[SensitivityRuleConfig]
    ) -> list[SensitivityRuleConfig]:
        seen: set[str] = set()
        for rule in value:
            key = rule.name.casefold()
            if key in seen:
                raise ValueError(f"duplicate sensitivity rule name: {rule.name!r}")
            seen.add(key)
        return value

    def options(self) -> dict[str, Any]:
        """Return the configured sensitivity override values."""
        return {
            "enabled": self.enabled,
            "rules": [rule.to_dict() for rule in self.rules],
            "blocked_terms": list(self.blocked_terms),
            "review_terms": list(self.review_terms),
            "disabled_rules": list(self.disabled_rules),
        }

    def stage_options(self, defaults: Iterable[Any] = ()) -> dict[str, Any]:
        """Return keyword arguments for a sensitivity stage.

        Unlike :meth:`options`, this resolves the overrides over *defaults* and
        omits ``disabled_rules`` because those rules have already been removed.
        """
        return {
            "enabled": self.enabled,
            "rules": [rule.to_dict() for rule in self.resolve_rules(defaults)],
            "blocked_terms": list(self.blocked_terms),
            "review_terms": list(self.review_terms),
        }

    def rule_overrides(self) -> dict[str, SensitivityRuleConfig]:
        """Return rule overrides keyed by their case-insensitive name."""
        return {rule.name.casefold(): rule for rule in self.rules}

    def resolve_rules(
        self,
        defaults: Iterable[Any] = (),
    ) -> tuple[SensitivityRuleConfig, ...]:
        """Merge overrides over *defaults*, dropping disabled rules.

        A configured rule replaces the default with the same name, keeping its
        position. New rules are appended in configuration order. A rule named
        in ``disabled_rules`` or an override with ``enabled: false`` is removed.
        """
        overrides = self.rule_overrides()
        disabled = {name.casefold() for name in self.disabled_rules}
        merged: list[SensitivityRuleConfig] = []
        seen: set[str] = set()
        for item in defaults:
            rule = _coerce_rule_config(item)
            key = rule.name.casefold()
            if key in disabled or key in seen:
                continue
            override = overrides.get(key)
            if override is not None:
                if override.enabled:
                    merged.append(override)
                seen.add(key)
                continue
            merged.append(rule)
            seen.add(key)
        for rule in self.rules:
            key = rule.name.casefold()
            if key in disabled or key in seen or not rule.enabled:
                continue
            merged.append(rule)
            seen.add(key)
        return tuple(merged)


SensitivityRuleSettings = SensitivityRuleConfig
"""Alias for :class:`SensitivityRuleConfig`."""

SensitivitySettings = SensitivityConfig
"""Alias for :class:`SensitivityConfig`."""


def _coerce_rule_config(value: Any) -> SensitivityRuleConfig:
    if isinstance(value, SensitivityRuleConfig):
        return value
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, Mapping):
        return SensitivityRuleConfig.model_validate(dict(value))
    raise ConfigError("sensitivity rule must be a mapping or model")


def _normalise_sensitivity(data: Any) -> Any:
    if not isinstance(data, Mapping):
        return data
    values = dict(data)
    nested = values.pop("sensitivity", None)
    overrides: dict[str, Any] = {}
    if nested is not None:
        if isinstance(nested, bool):
            overrides["enabled"] = nested
        elif isinstance(nested, Mapping):
            overrides.update(nested)
        else:
            raise ConfigError("sensitivity must be a mapping or boolean")
    for canonical, aliases in _SENSITIVITY_ROOT_ALIASES.items():
        key = canonical.removeprefix("sensitivity_")
        if canonical in values:
            overrides.setdefault(key, values.pop(canonical))
            for alias in aliases:
                values.pop(alias, None)
            continue
        for alias in aliases:
            if alias in values:
                overrides.setdefault(key, values.pop(alias))
                break
    if overrides or nested is not None:
        values["sensitivity"] = overrides
    return values


class ResearchConfig(ConfigModel):
    """Repeat suppression, selection limits and sensitivity settings."""

    dedupe_window: timedelta = Field(default=DEFAULT_DEDUPE_WINDOW)
    embedding_threshold: float | None = Field(
        default=DEFAULT_EMBEDDING_THRESHOLD,
        ge=0,
        le=1,
    )
    max_topics: int | None = Field(default=DEFAULT_MAX_TOPICS)
    max_per_category: int | None = Field(default=DEFAULT_MAX_PER_CATEGORY)
    sensitivity: SensitivityConfig = Field(default_factory=SensitivityConfig)

    _section_key: ClassVar[str] = RESEARCH_SECTION_KEY

    @model_validator(mode="before")
    @classmethod
    def _accept_field_aliases(cls, data: Any) -> Any:
        return _accept_aliases(_normalise_sensitivity(_normalise_selection(data)))

    @field_validator("dedupe_window", mode="before")
    @classmethod
    def _validate_window(cls, value: Any) -> DedupeWindow:
        return _coerce_window(value)

    @field_validator("embedding_threshold", mode="before")
    @classmethod
    def _validate_threshold(cls, value: Any) -> float | None:
        return _coerce_threshold(value)

    @field_validator("max_topics", mode="before")
    @classmethod
    def _validate_max_topics(cls, value: Any) -> int | None:
        return _coerce_max_topics(value)

    @field_validator("max_per_category", mode="before")
    @classmethod
    def _validate_max_per_category(cls, value: Any) -> int | None:
        return _coerce_max_per_category(value)

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

    @property
    def limit(self) -> int | None:
        """Alias for :attr:`max_topics`."""
        return self.max_topics

    @property
    def topics_per_run(self) -> int | None:
        """Alias for :attr:`max_topics`."""
        return self.max_topics

    @property
    def per_run(self) -> int | None:
        """Alias for :attr:`max_topics`."""
        return self.max_topics

    @property
    def topic_limit(self) -> int | None:
        """Alias for :attr:`max_topics`."""
        return self.max_topics

    @property
    def per_category(self) -> int | None:
        """Alias for :attr:`max_per_category`."""
        return self.max_per_category

    @property
    def diversity(self) -> int | None:
        """Alias for :attr:`max_per_category`."""
        return self.max_per_category

    @property
    def category_limit(self) -> int | None:
        """Alias for :attr:`max_per_category`."""
        return self.max_per_category

    @property
    def max_category(self) -> int | None:
        """Alias for :attr:`max_per_category`."""
        return self.max_per_category

    @property
    def sensitivity_enabled(self) -> bool:
        """Whether the sensitivity filter is active."""
        return self.sensitivity.enabled

    @property
    def sensitivity_rules(self) -> list[SensitivityRuleConfig]:
        """Sensitivity rule overrides and additions."""
        return self.sensitivity.rules

    @property
    def sensitivity_blocked_terms(self) -> list[str]:
        """Terms that add a blocked rule."""
        return self.sensitivity.blocked_terms

    @property
    def sensitivity_review_terms(self) -> list[str]:
        """Terms that add a manual-review rule."""
        return self.sensitivity.review_terms

    @property
    def sensitivity_disabled_rules(self) -> list[str]:
        """Names of default or configured rules to disable."""
        return self.sensitivity.disabled_rules

    def dedupe_options(self) -> dict[str, Any]:
        """Return the keywords accepted by the dedupe stage."""
        return {
            "window": self.dedupe_window,
            "embedding_threshold": self.embedding_threshold,
        }

    def to_dedupe_options(self) -> dict[str, Any]:
        """Alias for :meth:`dedupe_options`."""
        return self.dedupe_options()

    def selection_options(self) -> dict[str, Any]:
        """Return the keywords accepted by the selection stage."""
        return {
            "limit": self.max_topics,
            "max_per_category": self.max_per_category,
        }

    def to_selection_options(self) -> dict[str, Any]:
        """Alias for :meth:`selection_options`."""
        return self.selection_options()

    def selection_stage_options(self) -> dict[str, Any]:
        """Alias for :meth:`selection_options`."""
        return self.selection_options()

    def sensitivity_options(self) -> dict[str, Any]:
        """Return the keywords accepted by the sensitivity stage."""
        return self.sensitivity.options()

    def to_sensitivity_options(self) -> dict[str, Any]:
        """Alias for :meth:`sensitivity_options`."""
        return self.sensitivity_options()

    def sensitivity_rule_overrides(self) -> dict[str, SensitivityRuleConfig]:
        """Return sensitivity rule overrides keyed by their case-insensitive name."""
        return self.sensitivity.rule_overrides()

    def sensitivity_stage_options(
        self,
        defaults: Iterable[Any] = (),
    ) -> dict[str, Any]:
        """Return sensitivity stage keyword arguments resolved over *defaults*."""
        return self.sensitivity.stage_options(defaults)

    def resolve_sensitivity_rules(
        self,
        defaults: Iterable[Any] = (),
    ) -> tuple[SensitivityRuleConfig, ...]:
        """Merge sensitivity overrides over *defaults*, dropping disabled rules."""
        return self.sensitivity.resolve_rules(defaults)

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

"""Content profiles: reusable descriptions of what a run should produce.

A profile names the format recipe to use, how topics are found, what the
content is about and which reusable assets it binds. Topic modes are combined
by weight, so an explicit mix must be non-negative and sum to ``1.0``. Every
field has a default: an empty mapping is a valid profile, which lets a caller
supply the missing pieces at run time.

Profiles are keyed by name. Entries may be written directly under the section
(``profiles: {short: {...}}``) or nested under ``profiles``; both forms
validate to the same mapping.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator, model_validator

from aivia.core.config.base import ConfigModel
from aivia.core.errors import ConfigError

__all__ = [
    "ContentProfile",
    "ContentProfileConfig",
    "ContentProfilesConfig",
    "ExamplesConfig",
    "ProfileConfig",
    "ProfileExamplesConfig",
    "ProfileSourceConfig",
    "ProfilesConfig",
    "TopicMode",
    "TopicModeConfig",
    "TopicModesConfig",
]

_WEIGHT_SUM = 1.0
_WEIGHT_TOLERANCE = 1e-6

_PROFILE_ALIASES: dict[str, tuple[str, ...]] = {
    "format": ("format_name", "recipe"),
    "topic_modes": (
        "topic_mode",
        "modes",
        "topics",
        "topic_weights",
        "mode_weights",
    ),
    "brief": ("instruction", "steering"),
    "niche": ("niche_description", "subject"),
    "keywords": ("keyword", "terms"),
    "sources": ("source", "topic_sources"),
    "source_options": ("source_configs", "source_settings", "source_specs"),
    "tone": ("tone_of_voice", "voice_tone"),
    "style_bible": ("style", "style_guide", "style_bible_file"),
    "character": ("character_id", "persona"),
    "brand_kit": ("brand", "brand_id"),
    "accounts": ("account", "publish_accounts"),
    "overrides": ("config", "config_overrides", "settings"),
    "voice": ("voice_profile",),
    "visual_strategy": ("visuals",),
}

_EXAMPLE_ALIASES: dict[str, tuple[str, ...]] = {
    "liked": ("liked_examples", "like", "likes", "positive", "good"),
    "disliked": (
        "disliked_examples",
        "dislike",
        "dislikes",
        "negative",
        "bad",
    ),
}


class TopicMode(StrEnum):
    """The core ways a profile can discover topics."""

    ON_THIS_DAY = "on_this_day"
    NICHE = "niche"
    KEYWORDS = "keywords"
    FEEDS = "feeds"
    EVERGREEN_IDEAS = "evergreen_ideas"
    TRENDING = "trending"
    MANUAL = "manual"


_CORE_TOPIC_MODES = frozenset(mode.value for mode in TopicMode)


class TopicModeConfig(ConfigModel):
    """One topic mode with the share of the mix it receives."""

    mode: str = Field(min_length=1)
    weight: float = Field(default=1.0, ge=0)


TopicModesConfig = dict[str, float]


def _strip(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{label} must be non-empty")
    return text


def _normalise_names(
    values: Sequence[str],
    label: str,
    *,
    unique: bool = True,
) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for entry in values:
        name = _strip(entry, label)
        if unique and name in seen:
            raise ValueError(f"duplicate {label}: {name}")
        seen.add(name)
        names.append(name)
    return names


def _normalise_weights(
    weights: Mapping[str, float],
    label: str,
    *,
    require_sum: bool = True,
) -> dict[str, float]:
    normalised: dict[str, float] = {}
    for name, weight in weights.items():
        mode = _strip(name, f"{label} name")
        if mode in normalised:
            raise ValueError(f"duplicate {label}: {mode}")
        try:
            number = float(weight)
        except (TypeError, ValueError):
            raise ValueError(f"{label} weight for {mode} must be a number") from None
        if not math.isfinite(number):
            raise ValueError(f"{label} weight for {mode} must be finite")
        if number < 0:
            raise ValueError(f"{label} weight for {mode} must not be negative")
        normalised[mode] = number

    if require_sum and normalised:
        total = sum(normalised.values())
        if not math.isclose(
            total,
            _WEIGHT_SUM,
            rel_tol=0.0,
            abs_tol=_WEIGHT_TOLERANCE,
        ):
            raise ValueError(
                f"{label} weights must sum to {_WEIGHT_SUM:g}, got {total:g}"
            )
    return normalised


def _coerce_topic_modes(value: Any) -> Any:
    if value is None:
        return {}
    if isinstance(value, str):
        return {value: 1.0}
    if isinstance(value, Mapping):
        modes: dict[str, Any] = {}
        for name, entry in value.items():
            if isinstance(entry, Mapping):
                modes[name] = entry.get("weight", 1.0)
            else:
                modes[name] = entry
        return modes
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        entries = list(value)
        if not entries:
            return {}
        if all(isinstance(entry, str) for entry in entries):
            share = 1.0 / len(entries)
            return dict.fromkeys(entries, share)

        modes = {}
        explicit = False
        for entry in entries:
            if isinstance(entry, str):
                name, weight = entry, 1.0
            elif isinstance(entry, Mapping):
                name = entry.get("mode", entry.get("name", entry.get("id")))
                weight = entry.get("weight", 1.0)
                explicit = explicit or "weight" in entry
            else:
                raise ValueError("topic modes must be names or mappings")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("topic mode names must be non-empty")
            modes[name] = weight
        if modes and not explicit:
            share = 1.0 / len(modes)
            return dict.fromkeys(modes, share)
        return modes
    return value


class ExamplesConfig(ConfigModel):
    """Liked and disliked examples that steer topic and writing choices."""

    liked: list[str] = Field(default_factory=list)
    disliked: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            values = dict(data)
            for canonical, aliases in _EXAMPLE_ALIASES.items():
                for alias in aliases:
                    if alias in values:
                        value = values.pop(alias)
                        values.setdefault(canonical, value)
            return values
        if isinstance(data, str):
            return {"liked": [data]}
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            return {"liked": list(data)}
        return data

    @field_validator("liked", "disliked")
    @classmethod
    def _normalise_examples(cls, value: list[str]) -> list[str]:
        return _normalise_names(value, "example")


ProfileExamplesConfig = ExamplesConfig


def _normalise_language_tag(value: str) -> str:
    return _strip(value, "language").lower().replace("_", "-")


class ProfileSourceConfig(ConfigModel):
    """One topic source with a relative weight and an optional language filter.

    ``weight`` is relative rather than a share, so an unweighted list gives
    every source the same weight of ``1.0``. An empty ``languages`` list means
    the source is not restricted and accepts every profile language.
    """

    name: str = ""
    weight: float = Field(default=1.0, ge=0)
    languages: list[str] = Field(default_factory=list)
    enabled: bool = True

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"name": data}
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for alias in ("source", "type", "id"):
            if alias in values:
                values.setdefault("name", values.pop(alias))
        for alias in (
            "language",
            "lang",
            "langs",
            "locale",
            "locales",
            "language_filter",
            "language_filters",
        ):
            if alias in values:
                values.setdefault("languages", values.pop(alias))
        for alias in ("share", "priority"):
            if alias in values:
                values.setdefault("weight", values.pop(alias))
        return values

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not value:
            return ""
        return _strip(value, "source")

    @field_validator("weight")
    @classmethod
    def _validate_weight(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("source weight must be finite")
        return value

    @field_validator("languages", mode="before")
    @classmethod
    def _accept_languages(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("languages")
    @classmethod
    def _normalise_languages(cls, value: list[str]) -> list[str]:
        languages: list[str] = []
        for entry in value:
            language = _normalise_language_tag(entry)
            if language not in languages:
                languages.append(language)
        return languages


_NAME_KEYS = ("name", "source", "type", "id")


def _source_entry_options(entry: Any) -> dict[str, Any]:
    if entry is None:
        return {}
    if isinstance(entry, ProfileSourceConfig):
        return entry.model_dump()
    if isinstance(entry, bool):
        return {"enabled": entry}
    if isinstance(entry, (int, float)):
        return {"weight": entry}
    if isinstance(entry, str):
        try:
            return {"weight": float(entry)}
        except ValueError:
            return {"languages": [entry]}
    if isinstance(entry, Mapping):
        return dict(entry)
    if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
        return {"languages": list(entry)}
    raise ValueError("source entries must be names, numbers or mappings")


def _coerce_source_entries(value: Any) -> tuple[list[str], dict[str, Any]]:
    if value is None:
        return [], {}
    if isinstance(value, str):
        return [value], {}

    names: list[str] = []
    options: dict[str, Any] = {}

    def add(name: Any, entry: Any) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("source names must be non-empty")
        source = name.strip()
        if source in names:
            raise ValueError(f"duplicate source: {source}")
        names.append(source)
        settings = _source_entry_options(entry)
        if "name" in settings and settings["name"] != source:
            raise ValueError(f"conflicting source name: {source}")
        settings.pop("name", None)
        if settings:
            options[source] = settings

    if isinstance(value, Mapping):
        for name, entry in value.items():
            add(name, entry)
        return names, options

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for entry in value:
            if isinstance(entry, str):
                add(entry, None)
                continue
            if isinstance(entry, ProfileSourceConfig):
                settings = entry.model_dump()
            elif isinstance(entry, Mapping):
                settings = dict(entry)
            else:
                raise ValueError("source entries must be names or mappings")
            name = None
            for key in _NAME_KEYS:
                if key in settings:
                    name = settings.pop(key)
                    break
            if name is None:
                raise ValueError("source name must be non-empty")
            add(name, settings)
        return names, options

    raise ValueError("sources must be a name, a list or a mapping")


def _merge_source_options(
    values: dict[str, Any],
    options: Mapping[str, Any],
) -> None:
    merged: dict[str, Any] = {}
    existing = values.get("source_options")
    if existing is not None:
        if not isinstance(existing, Mapping):
            raise ValueError("source options must be a mapping")
        for name, entry in existing.items():
            merged[str(name)] = entry
    for name, entry in options.items():
        current = merged.get(name)
        if isinstance(current, Mapping) and isinstance(entry, Mapping):
            combined = dict(current)
            combined.update(entry)
            merged[name] = combined
        else:
            merged[name] = entry
    if merged:
        values["source_options"] = merged


class ContentProfileConfig(ConfigModel):
    """Everything a run needs to produce content for one audience or brand."""

    format: str = ""
    topic_modes: dict[str, float] = Field(default_factory=dict)
    brief: str = ""
    niche: str = ""
    keywords: list[str] = Field(default_factory=list)
    examples: ExamplesConfig = Field(default_factory=ExamplesConfig)
    sources: list[str] = Field(default_factory=list)
    source_options: dict[str, ProfileSourceConfig] = Field(default_factory=dict)
    tone: str = ""
    style_bible: str = ""
    character: str | None = None
    brand_kit: str | None = None
    accounts: list[str] = Field(default_factory=list)
    overrides: dict[str, Any] = Field(default_factory=dict)
    voice: str = ""
    languages: list[str] = Field(default_factory=list)
    visual_strategy: str = ""

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for canonical, aliases in _PROFILE_ALIASES.items():
            for alias in aliases:
                if alias in values:
                    value = values.pop(alias)
                    values.setdefault(canonical, value)

        if "topic_modes" in values:
            values["topic_modes"] = _coerce_topic_modes(values["topic_modes"])

        weights = values.pop("source_weights", None)
        languages = values.pop("source_languages", None)
        source_options: dict[str, Any] = {}
        if weights is not None:
            if not isinstance(weights, Mapping):
                raise ValueError("source weights must be a mapping")
            for name, weight in weights.items():
                source_options.setdefault(str(name), {})["weight"] = weight
        if languages is not None:
            if not isinstance(languages, Mapping):
                raise ValueError("source languages must be a mapping")
            for name, entries in languages.items():
                source_options.setdefault(str(name), {})["languages"] = entries

        if "sources" in values:
            names, embedded = _coerce_source_entries(values["sources"])
            values["sources"] = names
            source_options.update(embedded)
        _merge_source_options(values, source_options)

        if "liked_examples" in values or "disliked_examples" in values:
            examples = values.get("examples")
            if examples is None:
                examples = {}
            if not isinstance(examples, Mapping):
                raise ValueError("examples must be a mapping or a list")
            merged = dict(examples)
            for canonical in ("liked", "disliked"):
                alias = f"{canonical}_examples"
                if alias in values:
                    merged.setdefault(canonical, values.pop(alias))
            values["examples"] = merged

        for field in ("character", "brand_kit"):
            if field in values and values[field] == "":
                values[field] = None
        return values

    @field_validator("format")
    @classmethod
    def _validate_format(cls, value: str) -> str:
        return _strip(value, "format")

    @field_validator("topic_modes")
    @classmethod
    def _validate_topic_modes(cls, value: dict[str, float]) -> dict[str, float]:
        weights = _normalise_weights(value, "topic mode")
        for name in weights:
            if name not in _CORE_TOPIC_MODES and "." not in name:
                allowed = ", ".join(sorted(_CORE_TOPIC_MODES))
                raise ValueError(f"unknown topic mode: {name} (allowed: {allowed})")
        return weights

    @field_validator("brief", "niche", "tone", "style_bible", "voice")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("visual_strategy")
    @classmethod
    def _strip_strategy(cls, value: str) -> str:
        return value.strip()

    @field_validator("character", "brand_kit")
    @classmethod
    def _strip_reference(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        return text or None

    @field_validator("keywords")
    @classmethod
    def _normalise_keywords(cls, value: list[str]) -> list[str]:
        return _normalise_names(value, "keyword")

    @field_validator("sources")
    @classmethod
    def _normalise_sources(cls, value: list[str]) -> list[str]:
        return _normalise_names(value, "source")

    @field_validator("source_options")
    @classmethod
    def _name_source_options(
        cls, value: dict[str, ProfileSourceConfig]
    ) -> dict[str, ProfileSourceConfig]:
        for name, entry in value.items():
            if not entry.name:
                entry.name = name
        return value

    @field_validator("accounts")
    @classmethod
    def _normalise_accounts(cls, value: list[str]) -> list[str]:
        return _normalise_names(value, "account")

    @field_validator("languages")
    @classmethod
    def _normalise_languages(cls, value: list[str]) -> list[str]:
        return _normalise_names(value, "language")

    @field_validator("accounts", mode="before")
    @classmethod
    def _accept_mapping_names(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return list(value)
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("overrides", mode="before")
    @classmethod
    def _accept_overrides(cls, value: Any) -> Any:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("overrides must be a mapping")
        return dict(value)

    @property
    def modes(self) -> dict[str, float]:
        """Alias for :attr:`topic_modes`."""
        return self.topic_modes

    @property
    def format_name(self) -> str:
        """Alias for :attr:`format`."""
        return self.format

    @property
    def config_overrides(self) -> dict[str, Any]:
        """Alias for :attr:`overrides`."""
        return self.overrides

    @property
    def source_weights(self) -> dict[str, float]:
        """The configured weight for each source, defaulting to ``1.0``."""
        return {
            name: self.source_options[name].weight
            if name in self.source_options
            else 1.0
            for name in self.sources
        }

    @property
    def source_languages(self) -> dict[str, list[str]]:
        """The configured language filter for each source."""
        return {
            name: list(self.source_options[name].languages)
            if name in self.source_options
            else []
            for name in self.sources
        }

    @property
    def source_configs(self) -> dict[str, ProfileSourceConfig]:
        """Alias for :attr:`source_options`."""
        return self.source_options

    @property
    def source_settings(self) -> dict[str, ProfileSourceConfig]:
        """Alias for :attr:`source_options`."""
        return self.source_options


ContentProfile = ContentProfileConfig
ProfileConfig = ContentProfileConfig


class ProfilesConfig(ConfigModel):
    """Named content profiles.

    Profiles may be written directly under the section
    (``profiles: {short: {...}}``) or nested under ``profiles``; both forms
    validate to the same mapping.
    """

    profiles: dict[str, ContentProfileConfig] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_direct_mapping(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        if "profiles" in values:
            return values
        if "content_profiles" in values:
            profiles = values.pop("content_profiles")
            values.setdefault("profiles", profiles)
            return values
        return {"profiles": values}

    @field_validator("profiles")
    @classmethod
    def _normalise_names(
        cls, value: dict[str, ContentProfileConfig]
    ) -> dict[str, ContentProfileConfig]:
        profiles: dict[str, ContentProfileConfig] = {}
        for name, profile in value.items():
            key = _strip(name, "profile name")
            if key in profiles:
                raise ValueError(f"duplicate profile: {key}")
            profiles[key] = profile
        return profiles

    @property
    def names(self) -> tuple[str, ...]:
        """The configured profile names in mapping order."""
        return tuple(self.profiles)

    def get(self, name: str) -> ContentProfileConfig:
        """Return the named profile or fail with the available names."""
        try:
            return self.profiles[name]
        except KeyError:
            available = ", ".join(sorted(self.profiles)) or "none"
            raise ConfigError(
                f"unknown profile: {name} (available: {available})",
                details={"profile": name, "available": list(self.profiles)},
            ) from None

    def profile(self, name: str) -> ContentProfileConfig:
        """Alias for :meth:`get`."""
        return self.get(name)

    def for_profile(self, name: str) -> ContentProfileConfig:
        """Alias for :meth:`get`."""
        return self.get(name)


ContentProfilesConfig = ProfilesConfig

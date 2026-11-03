"""Resolve a content profile's topic sources into weighted selections.

A profile may list sources as plain names, as a mapping of name to weight, or
as richer entries carrying a weight and a language filter. Resolution turns
those forms into one ordered :class:`SourceSelection` and validates every name
against the source registry, so a typo fails with the unknown name and the
available sources instead of silently collecting nothing.

Weights are relative, not shares: an unweighted source defaults to ``1.0`` and
the collection stage decides how to use them. A per-source language filter is
intersected with the languages requested for the run; a source that supports
none of them is left out. A source with no filter supports every language.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from aivia.core.config.profile import (
    ContentProfileConfig,
    ProfilesConfig,
    ProfileSourceConfig,
    _coerce_source_entries,
)
from aivia.core.errors import ConfigError
from aivia.providers.sources import (
    SourceProviderBase,
    SourceRegistry,
    source_registry,
)

__all__ = [
    "DEFAULT_SOURCE_WEIGHT",
    "DEFAULT_WEIGHT",
    "ProfileSource",
    "ResolvedSource",
    "ResolvedSources",
    "SourceConfig",
    "SourceResolution",
    "SourceSelection",
    "SourceSelector",
    "SourceSpec",
    "parse_source_specs",
    "parse_sources",
    "resolve_profile_sources",
    "resolve_source_selection",
    "resolve_sources",
    "select_profile_sources",
    "select_sources",
    "source_selection",
]

DEFAULT_SOURCE_WEIGHT = ProfileSourceConfig.model_fields["weight"].default
"""The relative weight used when a profile names a source without one."""


def _normalise_language(value: str) -> str:
    if not isinstance(value, str):
        raise ConfigError("language must be a string")
    text = value.strip().lower().replace("_", "-")
    if not text:
        raise ConfigError("language must not be empty")
    return text


def _language_matches(requested: str, supported: str) -> bool:
    if requested == supported:
        return True
    return requested.split("-", 1)[0] == supported.split("-", 1)[0]


def _normalise_languages(value: Any, label: str = "language") -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        entries: Sequence[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        entries = value
    else:
        raise ConfigError(f"{label}s must be a string or a list of strings")
    languages: list[str] = []
    for entry in entries:
        language = _normalise_language(entry)
        if language not in languages:
            languages.append(language)
    return languages


SourceSpec = ProfileSourceConfig
"""A source name, relative weight, optional language filter and enabled flag."""

SourceConfig = SourceSpec
ProfileSource = SourceSpec
DEFAULT_WEIGHT = DEFAULT_SOURCE_WEIGHT


@dataclass(frozen=True, slots=True)
class ResolvedSource:
    """One profile source resolved to its provider class and effective filter."""

    name: str
    weight: float
    languages: tuple[str, ...]
    provider: type[SourceProviderBase]
    spec: SourceSpec

    @property
    def source(self) -> str:
        """Alias for :attr:`name`."""
        return self.name

    @property
    def enabled(self) -> bool:
        """Whether the profile enables this source."""
        return self.spec.enabled

    @property
    def provider_class(self) -> type[SourceProviderBase]:
        """Alias for :attr:`provider`."""
        return self.provider

    def supports(self, language: str) -> bool:
        """Whether this source accepts *language* under its effective filter."""
        return not self.languages or _normalise_language(language) in self.languages


class SourceSelection(list[ResolvedSource]):
    """An ordered list of resolved sources with convenience views."""

    @property
    def sources(self) -> list[ResolvedSource]:
        """Alias for the selection itself as a plain list."""
        return list(self)

    @property
    def resolved(self) -> list[ResolvedSource]:
        """Alias for the resolved sources as a plain list."""
        return list(self)

    @property
    def by_source(self) -> dict[str, ResolvedSource]:
        """Map each source name to its resolved entry."""
        return {source.name: source for source in self}

    @property
    def source_names(self) -> list[str]:
        """Alias for :attr:`names`."""
        return self.names

    @property
    def weight_map(self) -> dict[str, float]:
        """Alias for :attr:`weights`."""
        return self.weights

    @property
    def language_map(self) -> dict[str, list[str]]:
        """Alias for :attr:`languages`."""
        return self.languages

    @property
    def names(self) -> list[str]:
        """The resolved source names in profile order."""
        return [source.name for source in self]

    @property
    def weights(self) -> dict[str, float]:
        """The relative weight of each resolved source."""
        return {source.name: source.weight for source in self}

    @property
    def languages(self) -> dict[str, list[str]]:
        """The effective language filter of each resolved source."""
        return {source.name: list(source.languages) for source in self}

    @property
    def providers(self) -> dict[str, type[SourceProviderBase]]:
        """The provider class selected for each source."""
        return {source.name: source.provider for source in self}

    @property
    def total_weight(self) -> float:
        """The sum of every resolved relative weight."""
        return sum(source.weight for source in self)

    def by_name(self, name: str) -> ResolvedSource | None:
        """Return the resolved source called *name*, or ``None``."""
        for source in self:
            if source.name == name:
                return source
        return None

    def get(
        self,
        name: str,
        default: ResolvedSource | None = None,
    ) -> ResolvedSource | None:
        """Alias for :meth:`by_name` with a default."""
        found = self.by_name(name)
        return default if found is None else found

    def normalized_weights(self) -> dict[str, float]:
        """Return each weight as a share of the total, or zero when empty."""
        total = self.total_weight
        if total <= 0:
            return {source.name: 0.0 for source in self}
        return {source.name: source.weight / total for source in self}

    def items(self) -> list[tuple[str, float]]:
        """Return ``(name, weight)`` pairs in profile order."""
        return [(source.name, source.weight) for source in self]


def _profile_parts(
    profile: Any,
) -> tuple[Any, dict[str, Any], list[str]]:
    if profile is None:
        return [], {}, []
    if isinstance(profile, ContentProfileConfig):
        profile_options = {
            name: entry.model_dump() for name, entry in profile.source_options.items()
        }
        return list(profile.sources), profile_options, list(profile.languages)
    if isinstance(profile, ProfilesConfig):
        raise ConfigError(
            "a named profile is required when resolving a profiles section",
            details={"available": list(profile.names)},
        )
    if isinstance(profile, Mapping):
        values = dict(profile)
        profile_keys = {
            "sources",
            "source",
            "topic_sources",
            "source_options",
            "source_configs",
            "source_settings",
            "source_weights",
            "source_languages",
            "languages",
        }
        if not profile_keys.intersection(values):
            return values, {}, []
        sources = values.get(
            "sources",
            values.get("source", values.get("topic_sources", [])),
        )
        options: dict[str, Any] = {}
        for alias in ("source_options", "source_configs", "source_settings"):
            if alias in values and values[alias] is not None:
                options.update(dict(values[alias]))
        weights = values.get("source_weights")
        if weights is not None:
            if not isinstance(weights, Mapping):
                raise ConfigError("source weights must be a mapping")
            for name, weight in weights.items():
                options.setdefault(str(name), {})
                if isinstance(options[str(name)], Mapping):
                    options[str(name)] = {**options[str(name)], "weight": weight}
                else:
                    options[str(name)] = {"weight": weight}
        languages = values.get("source_languages")
        if languages is not None:
            if not isinstance(languages, Mapping):
                raise ConfigError("source languages must be a mapping")
            for name, entries in languages.items():
                options.setdefault(str(name), {})
                if isinstance(options[str(name)], Mapping):
                    options[str(name)] = {**options[str(name)], "languages": entries}
                else:
                    options[str(name)] = {"languages": entries}
        return sources, options, _normalise_languages(values.get("languages"))

    sources = getattr(profile, "sources", None)
    if sources is not None:
        options = getattr(profile, "source_options", {})
        return sources, dict(options), list(getattr(profile, "languages", []))

    if isinstance(profile, str) or (
        isinstance(profile, Sequence) and not isinstance(profile, (bytes, bytearray))
    ):
        return profile, {}, []

    raise ConfigError("profile must be a content profile or a mapping")


def parse_source_specs(
    sources: Any,
    *,
    options: Mapping[str, Any] | None = None,
) -> list[SourceSpec]:
    """Validate *sources* into ordered :class:`SourceSpec` records.

    Accepted forms are a single name, a list of names or mappings, or a mapping
    of name to a weight, language list or option mapping. Explicit *options*
    override values embedded in *sources*.
    """
    try:
        names, embedded = _coerce_source_entries(sources)
    except (TypeError, ValueError) as error:
        raise ConfigError(str(error)) from error

    specs: list[SourceSpec] = []
    for name in names:
        values: dict[str, Any] = {}
        embedded_entry = embedded.get(name)
        if isinstance(embedded_entry, Mapping):
            values.update(embedded_entry)
        explicit = (options or {}).get(name)
        if isinstance(explicit, ProfileSourceConfig):
            values.update(explicit.model_dump())
        elif isinstance(explicit, Mapping):
            values.update(explicit)
        elif (
            isinstance(explicit, (int, float)) and not isinstance(explicit, bool)
        ) or explicit is not None:
            values["weight"] = explicit
        values["name"] = name
        try:
            specs.append(SourceSpec.from_mapping(values))
        except ConfigError as error:
            raise ConfigError(
                f"invalid source {name!r}: {error}",
                details={"source": name},
            ) from error
    return specs


def _requested_languages(
    languages: Sequence[str] | str | None,
    language: str | None,
    profile_languages: Sequence[str],
) -> list[str]:
    requested = _normalise_languages(languages)
    if language is not None:
        requested.append(_normalise_language(language))
    if not requested:
        requested = _normalise_languages(profile_languages)
    deduplicated: list[str] = []
    for entry in requested:
        if entry not in deduplicated:
            deduplicated.append(entry)
    return deduplicated


def _available_names(registry: Any) -> list[str]:
    for method in ("source_names", "names", "keys"):
        getter = getattr(registry, method, None)
        if getter is not None:
            return list(getter())
    return []


def _resolve_provider(registry: Any, name: str) -> type[SourceProviderBase]:
    resolve = getattr(registry, "resolve_source", None)
    if resolve is None:
        resolve = registry.resolve
    try:
        return cast(type[SourceProviderBase], resolve(name))
    except (ConfigError, KeyError) as error:
        available = ", ".join(_available_names(registry)) or "none"
        raise ConfigError(
            f"unknown source: {name} (available: {available})",
            details={"source": name, "available": _available_names(registry)},
        ) from error


def resolve_sources(
    profile: (
        ContentProfileConfig
        | ProfilesConfig
        | Mapping[str, Any]
        | Sequence[str]
        | str
        | None
    ) = None,
    profile_name: str | None = None,
    *,
    sources: Any = None,
    languages: Sequence[str] | str | None = None,
    language: str | None = None,
    registry: SourceRegistry | None = None,
) -> SourceSelection:
    """Resolve *profile* sources with weights and language filters.

    ``languages`` or the singular ``language`` narrows the run and overrides the
    profile's own language list. Unknown source names raise
    :class:`~aivia.core.errors.ConfigError` with the available names. Disabled
    sources and sources whose weight is zero are omitted after validation.
    """
    if sources is not None:
        if profile is not None:
            raise ConfigError("pass either a profile or sources, not both")
        profile = {"sources": sources}

    if profile is not None and not isinstance(
        profile,
        (
            ContentProfileConfig,
            ProfilesConfig,
            Mapping,
            str,
            Sequence,
        ),
    ):
        profiles = getattr(profile, "profiles", None)
        if profiles is not None:
            profile = profiles

    if isinstance(profile, ProfilesConfig):
        if profile_name is None:
            raise ConfigError(
                "profile_name is required when resolving a profiles section",
                details={"available": list(profile.names)},
            )
        profile = profile.get(profile_name)

    sources_value, options, profile_languages = _profile_parts(profile)
    specs = parse_source_specs(sources_value, options=options)
    requested = _requested_languages(languages, language, profile_languages)
    active_registry = registry if registry is not None else source_registry

    selection = SourceSelection()
    for spec in specs:
        provider = _resolve_provider(active_registry, spec.name)
        if not spec.enabled or not spec.weight:
            continue
        supported = tuple(spec.languages)
        if supported and requested:
            effective = tuple(
                entry
                for entry in requested
                if any(_language_matches(entry, tag) for tag in supported)
            )
            if not effective:
                continue
        elif supported:
            effective = supported
        else:
            effective = tuple(requested)
        selection.append(
            ResolvedSource(
                name=spec.name,
                weight=spec.weight,
                languages=effective,
                provider=provider,
                spec=spec,
            )
        )
    return selection


resolve_profile_sources = resolve_sources
"""Alias for :func:`resolve_sources`."""

resolve_source_selection = resolve_sources
"""Alias for :func:`resolve_sources`."""

select_profile_sources = resolve_sources
"""Alias for :func:`resolve_sources`."""

parse_sources = parse_source_specs
"""Alias for :func:`parse_source_specs`."""

ResolvedSources = SourceSelection
SourceResolution = SourceSelection

select_sources = resolve_sources
"""Alias for :func:`resolve_sources`."""

source_selection = resolve_sources
"""Alias for :func:`resolve_sources`."""


class SourceSelector:
    """A reusable resolver bound to one source registry."""

    def __init__(self, registry: SourceRegistry | None = None) -> None:
        self.registry = registry if registry is not None else source_registry

    def resolve(
        self,
        profile: Any = None,
        profile_name: str | None = None,
        *,
        sources: Any = None,
        languages: Sequence[str] | str | None = None,
        language: str | None = None,
    ) -> SourceSelection:
        """Resolve *profile* against this selector's registry."""
        return resolve_sources(
            profile,
            profile_name,
            sources=sources,
            languages=languages,
            language=language,
            registry=self.registry,
        )

    __call__ = resolve

    def for_profile(
        self,
        profiles: ProfilesConfig,
        profile_name: str,
        *,
        languages: Sequence[str] | str | None = None,
        language: str | None = None,
    ) -> SourceSelection:
        """Resolve one named profile from a profiles section."""
        return self.resolve(
            profiles,
            profile_name,
            languages=languages,
            language=language,
        )

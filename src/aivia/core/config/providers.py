"""Provider instance configuration for router chains.

A capability chain names the providers it may use, but a name alone does not
say which adapter implements it or how to construct it. This section maps each
name to the provider ``type`` (a registered type or a dotted import path), the
adapter ``options`` and ``secret_refs``. A secret reference is a name resolved
by the secrets loader, never a secret value, so the configuration stays safe to
copy and commit.

Entries may be written directly under the section (``providers: {fal: {...}}``)
or nested under ``providers``. A bare string is shorthand for an entry that
only sets ``type``. ``options`` accepts arbitrary adapter values; ``secret_refs``
maps an option name to the reference that supplies it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import Field, field_validator, model_validator

from aivia.core.config.base import ConfigModel
from aivia.core.errors import ConfigError

__all__ = [
    "ProviderConfig",
    "ProvidersConfig",
]

_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "type": ("kind", "provider_type", "adapter"),
    "options": ("settings", "params", "parameters", "config"),
    "secret_refs": (
        "secrets",
        "secret_references",
        "secret_reference",
        "credentials",
    ),
}

_GLOBAL_KEYS = frozenset({"enabled"})
_PROVIDER_ALIASES = ("providers", "items", "definitions", "instances")
_WHITESPACE = frozenset(" \t\r\n")


def _text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{label} must be non-empty")
    if any(character in _WHITESPACE for character in text):
        raise ValueError(f"{label} must not contain whitespace: {text!r}")
    return text


def _options(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("provider options must be a mapping")
    options: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError("provider option names must be strings")
        name = key.strip()
        if not name:
            raise ValueError("provider option names must be non-empty")
        if name in options:
            raise ValueError(f"duplicate provider option: {name}")
        options[name] = item
    return options


def _secret_refs(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, str):
        reference = _text(value, "secret reference")
        return {reference: reference}
    if isinstance(value, Mapping):
        entries: Iterable[tuple[Any, Any]] = value.items()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        entries = ((entry, entry) for entry in value)
    else:
        raise ValueError("provider secret references must be a mapping or list")
    refs: dict[str, str] = {}
    for key, reference in entries:
        name = _text(key, "secret option name")
        target = _text(reference, "secret reference")
        if name in refs:
            raise ValueError(f"duplicate secret option: {name}")
        refs[name] = target
    return refs


class ProviderConfig(ConfigModel):
    """One named provider instance used by router chains.

    ``type`` names a registered provider type or a dotted import path.
    ``options`` are passed to the adapter unchanged. ``secret_refs`` maps an
    option name to the secret reference that supplies its value; the reference
    is resolved through the secrets loader, never stored here.
    """

    type: str = ""
    enabled: bool = True
    options: dict[str, Any] = Field(default_factory=dict)
    secret_refs: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_shorthand_and_aliases(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"type": data}
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for canonical, aliases in _FIELD_ALIASES.items():
            for alias in aliases:
                if alias in values:
                    value = values.pop(alias)
                    values.setdefault(canonical, value)
        return values

    @field_validator("type")
    @classmethod
    def _strip_type(cls, value: str) -> str:
        return value.strip()

    @field_validator("options", mode="before")
    @classmethod
    def _validate_options(cls, value: Any) -> dict[str, Any]:
        return _options(value)

    @field_validator("secret_refs", mode="before")
    @classmethod
    def _validate_secret_refs(cls, value: Any) -> dict[str, str]:
        return _secret_refs(value)

    @model_validator(mode="after")
    def _require_type_when_configured(self) -> ProviderConfig:
        if self.type:
            return self
        if self.model_fields_set:
            raise ValueError("provider type must be non-empty")
        return self

    @property
    def kind(self) -> str:
        """Alias for :attr:`type`."""
        return self.type

    @property
    def provider_type(self) -> str:
        """Alias for :attr:`type`."""
        return self.type

    @property
    def settings(self) -> dict[str, Any]:
        """Alias for :attr:`options`."""
        return self.options

    @property
    def params(self) -> dict[str, Any]:
        """Alias for :attr:`options`."""
        return self.options

    @property
    def secrets(self) -> dict[str, str]:
        """Alias for :attr:`secret_refs`."""
        return self.secret_refs

    @property
    def secret_references(self) -> dict[str, str]:
        """Alias for :attr:`secret_refs`."""
        return self.secret_refs

    @property
    def requires_secrets(self) -> bool:
        """Whether this provider declares any secret reference."""
        return bool(self.secret_refs)


class ProvidersConfig(ConfigModel):
    """Named provider instances keyed by the names router chains use.

    Entries may be written directly under the section (``providers: {fal: {...}}``)
    or nested under ``providers``. Every entry must name a non-empty ``type``.
    """

    enabled: bool = True
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_direct_mapping(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for alias in _PROVIDER_ALIASES[1:]:
            if alias in values:
                values.setdefault("providers", values.pop(alias))

        provided = values.get("providers")
        if provided is None:
            entries: dict[str, Any] = {}
        elif isinstance(provided, Mapping):
            entries = dict(provided)
        else:
            return values

        for key in list(values):
            if key == "providers" or key in _GLOBAL_KEYS:
                continue
            entries.setdefault(key, values.pop(key))
        values["providers"] = entries
        return values

    @field_validator("providers")
    @classmethod
    def _validate_providers(
        cls,
        value: dict[str, ProviderConfig],
    ) -> dict[str, ProviderConfig]:
        providers: dict[str, ProviderConfig] = {}
        for name, provider in value.items():
            key = _text(name, "provider name")
            if key in providers:
                raise ValueError(f"duplicate provider: {key}")
            if not provider.type:
                raise ValueError(f"provider {key} requires a non-empty type")
            providers[key] = provider
        return providers

    def names(self) -> tuple[str, ...]:
        """Return the configured provider names in insertion order."""
        return tuple(self.providers)

    def provider(self, name: str) -> ProviderConfig:
        """Return the provider named *name* or raise :class:`ConfigError`."""
        key = name.strip()
        try:
            return self.providers[key]
        except KeyError:
            available = ", ".join(self.names()) or "none"
            raise ConfigError(
                f"unknown provider: {key} (configured: {available})",
                details={"provider": key, "available": list(self.names())},
            ) from None

    def get(self, name: str, default: Any = None) -> Any:
        """Return the provider named *name* or *default* when absent."""
        return self.providers.get(name.strip(), default)

    def enabled_providers(self) -> dict[str, ProviderConfig]:
        """Return only the providers that are enabled."""
        return {
            name: provider
            for name, provider in self.providers.items()
            if provider.enabled
        }

    def type_for(self, name: str) -> str:
        """Return the adapter type configured for *name*."""
        return self.provider(name).type

    def options_for(self, name: str) -> dict[str, Any]:
        """Return the adapter options configured for *name*."""
        return self.provider(name).options

    def secret_refs_for(self, name: str) -> dict[str, str]:
        """Return the secret references configured for *name*."""
        return self.provider(name).secret_refs

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.strip() in self.providers

    def __len__(self) -> int:
        return len(self.providers)

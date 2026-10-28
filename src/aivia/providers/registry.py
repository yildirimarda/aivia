"""Registry mapping provider type strings to provider classes.

Configuration names a provider by its ``type``. Built-in adapters register
themselves with the :func:`register_provider` decorator, while a custom adapter
is named by a dotted path such as ``my_package.providers:MyProvider`` and
imported lazily. Registration rejects duplicate names, and resolving a type
that is neither registered nor loadable fails with the type and the registered
names, so a typo never silently selects the wrong provider.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import overload

from aivia.core.errors import ConfigError
from aivia.providers.base import Provider

__all__ = [
    "ProviderRegistration",
    "ProviderRegistry",
    "provider_registry",
    "register_provider",
    "registry",
    "resolve_provider",
]


def _validate_name(name: object) -> str:
    if not isinstance(name, str):
        raise TypeError("provider type must be a string")
    stripped = name.strip()
    if not stripped:
        raise ConfigError("provider type must not be empty")
    if any(character.isspace() for character in stripped):
        raise ConfigError(f"invalid provider type: {name!r}")
    return stripped


def _validate_provider(
    name: str,
    provider: object,
    *,
    source: str = "provider",
) -> type[Provider]:
    if not isinstance(provider, type) or not issubclass(provider, Provider):
        raise TypeError(
            f"{source} {name!r} must be a Provider subclass, "
            f"not {type(provider).__name__}"
        )
    return provider


def _is_dotted_path(value: str) -> bool:
    return ":" in value or "." in value


def _split_path(value: str) -> tuple[str, str]:
    if ":" in value:
        module_name, _, attribute = value.partition(":")
    else:
        module_name, _, attribute = value.rpartition(".")
    if not module_name or not attribute:
        raise ConfigError(
            f"invalid provider type path: {value!r} "
            "(expected 'module:Class' or 'module.Class')"
        )
    return module_name, attribute


@dataclass(frozen=True, slots=True)
class ProviderRegistration:
    """One provider class registered under a config ``type`` name."""

    name: str
    provider: type[Provider]
    module: str | None = None


class ProviderRegistry:
    """An ordered, collision-checked collection of provider classes."""

    def __init__(self) -> None:
        self._providers: dict[str, ProviderRegistration] = {}
        self._loaded: dict[str, type[Provider]] = {}

    @overload
    def register(
        self,
        name: str,
        provider: type[Provider],
        *,
        module: str | None = None,
    ) -> ProviderRegistration: ...

    @overload
    def register(
        self,
        name: str,
        provider: None = None,
        *,
        module: str | None = None,
    ) -> Callable[[type[Provider]], type[Provider]]: ...

    def register(
        self,
        name: str,
        provider: type[Provider] | None = None,
        *,
        module: str | None = None,
    ) -> ProviderRegistration | Callable[[type[Provider]], type[Provider]]:
        """Register *provider* under *name* or return a registration decorator.

        A duplicate name is rejected even when it names the same class, because
        two owners for one config type would make selection ambiguous.
        """
        validated_name = _validate_name(name)
        if provider is None:

            def decorator(cls: type[Provider]) -> type[Provider]:
                self.register(validated_name, cls, module=module)
                return cls

            return decorator

        validated_provider = _validate_provider(validated_name, provider)
        existing = self._providers.get(validated_name)
        if existing is not None:
            raise ConfigError(
                f"provider type already registered: {validated_name}",
                details={
                    "type": validated_name,
                    "existing": existing.provider.__name__,
                    "new": validated_provider.__name__,
                },
            )
        registration = ProviderRegistration(
            name=validated_name,
            provider=validated_provider,
            module=module,
        )
        self._providers[validated_name] = registration
        return registration

    def register_provider(
        self,
        name: str,
        provider: type[Provider],
        *,
        module: str | None = None,
    ) -> ProviderRegistration:
        """Register a provider from an explicit class."""
        return self.register(name, provider, module=module)

    def get(self, name: str) -> type[Provider]:
        """Return the class registered under *name*."""
        validated_name = _validate_name(name)
        registration = self._providers.get(validated_name)
        if registration is None:
            raise KeyError(f"unknown provider: {validated_name}")
        return registration.provider

    def load(self, path: str) -> type[Provider]:
        """Import and return a provider class from a dotted *path*."""
        validated_path = _validate_name(path)
        cached = self._loaded.get(validated_path)
        if cached is not None:
            return cached

        module_name, attribute = _split_path(validated_path)
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ConfigError(
                f"cannot load provider type {validated_path!r}: {exc}",
                details={"type": validated_path, "module": module_name},
            ) from exc

        target: object = module
        for part in attribute.split("."):
            try:
                target = getattr(target, part)
            except AttributeError:
                raise ConfigError(
                    f"cannot load provider type {validated_path!r}: "
                    f"{module_name!r} has no attribute {attribute!r}",
                    details={
                        "type": validated_path,
                        "module": module_name,
                        "attribute": attribute,
                    },
                ) from None

        try:
            provider = _validate_provider(
                validated_path,
                target,
                source="provider type",
            )
        except TypeError as exc:
            raise ConfigError(
                f"cannot load provider type {validated_path!r}: {exc}",
                details={"type": validated_path, "module": module_name},
            ) from exc
        self._loaded[validated_path] = provider
        return provider

    def resolve(self, name: str) -> type[Provider]:
        """Return the class for a registered *name* or a dotted path."""
        validated_name = _validate_name(name)
        registration = self._providers.get(validated_name)
        if registration is not None:
            return registration.provider
        if _is_dotted_path(validated_name):
            return self.load(validated_name)
        known = ", ".join(self.names()) or "none"
        raise ConfigError(
            f"unknown provider type: {validated_name} (registered: {known})",
            details={
                "type": validated_name,
                "available": list(self.names()),
            },
        )

    def names(self) -> tuple[str, ...]:
        """Return the registered provider types in sorted order."""
        return tuple(sorted(self._providers))

    def keys(self) -> tuple[str, ...]:
        """Alias for :meth:`names`."""
        return self.names()

    def registrations(self) -> tuple[ProviderRegistration, ...]:
        """Return the registered entries in sorted name order."""
        return tuple(self._providers[name] for name in self.names())

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return name.strip() in self._providers

    def __iter__(self) -> Iterator[ProviderRegistration]:
        return iter(self.registrations())

    def __len__(self) -> int:
        return len(self._providers)


registry = ProviderRegistry()
provider_registry = registry
register_provider = registry.register_provider
resolve_provider = registry.resolve

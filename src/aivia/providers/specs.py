"""Load declarative provider specs from providers.d and register them for chains.

Specs are YAML files installed into the ``providers.d`` directory (see
:mod:`aivia.cli.providers`). Each spec becomes a normal
:class:`~aivia.providers.base.Provider` through
:class:`~aivia.providers.http_spec.HttpSpecProvider`, so the router, budget
tracker and media service can use it exactly like a hand-written adapter. The
store loads every spec from the directory, keys it by its declared ``name``,
registers each name in the provider registry and builds one live instance per
spec so configured capability chains can reference declarative providers by
name.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, cast

from aivia.core.secrets import SecretsLoader
from aivia.providers.http_spec import (
    HttpProviderSpec,
    HttpSpecProvider,
    load_http_specs,
)
from aivia.providers.registry import (
    ProviderRegistry,
)
from aivia.providers.registry import (
    registry as provider_registry,
)

__all__ = [
    "PROVIDERS_DIR_NAME",
    "ProviderSpecStore",
    "build_providers_from_specs",
    "load_provider_specs",
    "provider_registry",
    "register_provider_specs",
]

PROVIDERS_DIR_NAME = "providers.d"


def _spec_provider_class(spec: HttpProviderSpec) -> type[HttpSpecProvider]:
    """Return an :class:`HttpSpecProvider` subclass bound to *spec*.

    The spec is baked into the constructor so the provider factory can build an
    instance from just the configured name, the same way it builds any other
    registered adapter. Declarative specs are self-contained: their base URL,
    headers, body templates and pricing all live in the spec, so constructor
    options beyond ``name`` and ``secrets`` are not expected.
    """

    class SpecProvider(HttpSpecProvider):
        def __init__(
            self,
            *,
            name: str | None = None,
            secrets: Mapping[str, str] | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(spec, name=name, secrets=secrets, **kwargs)

    SpecProvider.__name__ = f"HttpSpecProvider:{spec.name}"
    SpecProvider.__qualname__ = SpecProvider.__name__
    return SpecProvider


def _default_directory(
    environ: Mapping[str, str] | None = None,
) -> Path:
    from aivia.core.paths import resolve_paths

    return resolve_paths(environ=environ).config_dir / PROVIDERS_DIR_NAME


class ProviderSpecStore:
    """A collection of declarative provider specs loaded from a directory.

    ``directory`` defaults to the user ``providers.d`` directory next to the
    config file. Specs are loaded lazily by :meth:`load` and keyed by their
    declared ``name``.
    """

    def __init__(
        self,
        directory: str | Path | None = None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        if directory is None:
            directory = _default_directory(environ)
        self.directory = Path(directory).expanduser().resolve()
        self._specs: dict[str, HttpProviderSpec] = {}

    def load(self) -> dict[str, HttpProviderSpec]:
        """Read and validate every spec in the directory, keyed by name.

        A missing directory yields an empty store. Invalid specs raise
        :class:`~aivia.providers.http_spec.HttpSpecError`.
        """
        self._specs.clear()
        if not self.directory.is_dir():
            return dict(self._specs)
        for spec in load_http_specs(self.directory):
            self._specs[spec.name] = spec
        return dict(self._specs)

    reload = load

    def names(self) -> tuple[str, ...]:
        """Return every loaded spec name in sorted order."""
        return tuple(sorted(self._specs))

    def get(self, name: str) -> HttpProviderSpec | None:
        """Return the spec for *name*, or ``None`` when it is not loaded."""
        return self._specs.get(name.strip())

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return name.strip() in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def register(self, registry: ProviderRegistry | None = None) -> ProviderRegistry:
        """Register every loaded spec as a provider type in *registry*.

        Each spec name becomes resolvable through
        :meth:`ProviderRegistry.resolve` and constructible through the provider
        factory, so configured chains can reference declarative providers by
        name.
        """
        target = registry if registry is not None else provider_registry
        for name in self.names():
            spec = self._specs[name]
            provider_class = _spec_provider_class(spec)
            target.register(
                name,
                provider_class,
                module=str(self.directory / f"{spec.name}.yaml"),
            )
        return target

    def build(
        self,
        registry: ProviderRegistry | None = None,
        secrets: Mapping[str, str] | SecretsLoader | None = None,
        **overrides: Any,
    ) -> dict[str, HttpSpecProvider]:
        """Build one provider instance per loaded spec, keyed by name.

        Instances are resolved from *registry* (defaulting to the shared
        registry) after :meth:`register` has run, so each configured chain
        receives a live adapter.
        """
        target = registry if registry is not None else provider_registry
        providers: dict[str, HttpSpecProvider] = {}
        for name in self.names():
            provider_class = cast(type[HttpSpecProvider], target.resolve(name))
            providers[name] = provider_class(name=name, secrets=secrets, **overrides)
        return providers


def load_provider_specs(
    directory: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> ProviderSpecStore:
    """Load every spec in *directory* into a fresh :class:`ProviderSpecStore`."""
    store = ProviderSpecStore(directory=directory, environ=environ)
    store.load()
    return store


def register_provider_specs(
    directory: str | Path | None = None,
    *,
    registry: ProviderRegistry | None = None,
    environ: Mapping[str, str] | None = None,
) -> ProviderRegistry:
    """Load specs from *directory* and register them in *registry*."""
    store = load_provider_specs(directory=directory, environ=environ)
    return store.register(registry=registry)


def build_providers_from_specs(
    directory: str | Path | None = None,
    *,
    registry: ProviderRegistry | None = None,
    secrets: Mapping[str, str] | None = None,
    environ: Mapping[str, str] | None = None,
    **overrides: Any,
) -> dict[str, HttpSpecProvider]:
    """Load specs from *directory*, register them and build one instance each."""
    store = load_provider_specs(directory=directory, environ=environ)
    store.register(registry=registry)
    return store.build(registry=registry, secrets=secrets, **overrides)

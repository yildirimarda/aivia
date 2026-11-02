"""Base class and registry for topic source providers.

Sources are capability providers like any other adapter: they carry validated
metadata, a health report and an idempotent close. The source registry maps a
config or profile name such as ``wikipedia`` to the class that implements it.
It is separate from the media provider registry because sources are selected by
the research module, not by the paid-call router.

A source adapter subclasses :class:`SourceProviderBase` and implements
:meth:`discover`. The base class adds a ``fetch`` alias so callers can use the
verb that reads naturally for their source.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import overload

from aivia.providers.base import Provider
from aivia.providers.protocols.source import (
    CAPABILITY_SOURCE,
    SourceCandidate,
    SourceProvider,
    SourceRequest,
    SourceResponse,
    SourceResult,
)
from aivia.providers.registry import (
    ProviderRegistration,
    ProviderRegistry,
)

__all__ = [
    "CAPABILITY_SOURCE",
    "Source",
    "SourceCandidate",
    "SourceProvider",
    "SourceProviderBase",
    "SourceRegistry",
    "SourceRequest",
    "SourceResponse",
    "SourceResult",
    "register_source",
    "resolve_source",
    "source_registry",
]


class SourceProviderBase(Provider):
    """Convenience base for a topic source adapter.

    Subclasses implement :meth:`discover`. :meth:`fetch` delegates to it so a
    caller may use either verb, and both return the same validated response.
    """

    async def discover(self, request: SourceRequest) -> SourceResponse:
        """Discover topic candidates for *request*."""
        raise NotImplementedError

    async def fetch(self, request: SourceRequest) -> SourceResponse:
        """Alias for :meth:`discover`."""
        return await self.discover(request)

    async def collect(self, request: SourceRequest) -> list[SourceCandidate]:
        """Return only the candidates from :meth:`discover`."""
        response = await self.discover(request)
        return list(response.candidates)


Source = SourceProviderBase
"""Alias for :class:`SourceProviderBase`."""


class SourceRegistry(ProviderRegistry):
    """An ordered, collision-checked collection of source provider classes.

    It inherits the provider registry's duplicate detection and dotted-path
    loading, so a custom source can be named by ``module:Class`` just like a
    media provider. ``resolve`` and ``resolve_source`` both fail with the
    unknown name and the registered names, which keeps a profile typo visible.
    """

    @overload
    def register_source(
        self,
        name: str,
        provider: type[Provider],
        *,
        module: str | None = None,
    ) -> ProviderRegistration: ...

    @overload
    def register_source(
        self,
        name: str,
        provider: None = None,
        *,
        module: str | None = None,
    ) -> Callable[[type[Provider]], type[Provider]]: ...

    def register_source(
        self,
        name: str,
        provider: type[Provider] | None = None,
        *,
        module: str | None = None,
    ) -> ProviderRegistration | Callable[[type[Provider]], type[Provider]]:
        """Register *provider* under *name* or return a registration decorator."""
        return self.register(name, provider, module=module)

    def get_source(self, name: str) -> type[Provider]:
        """Return the source class registered under *name*."""
        return self.get(name)

    def resolve_source(self, name: str) -> type[Provider]:
        """Return the source class for a registered name or dotted path."""
        return self.resolve(name)

    def source_names(self) -> tuple[str, ...]:
        """Return the registered source names in sorted order."""
        return self.names()


source_registry = SourceRegistry()
"""The shared registry of built-in and custom topic sources."""

register_source = source_registry.register_source
"""Register a source on the shared :data:`source_registry`."""

resolve_source = source_registry.resolve_source
"""Resolve a source name or dotted path through the shared registry."""

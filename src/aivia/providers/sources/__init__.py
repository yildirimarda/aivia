"""Topic source adapters, their base class and the source registry.

A source discovers topic candidates from an external feed. Importing this
package exposes the protocol, the registry and the deterministic mock used by
offline tests and development.
"""

from aivia.providers.sources.base import (
    CAPABILITY_SOURCE,
    Source,
    SourceCandidate,
    SourceProvider,
    SourceProviderBase,
    SourceRegistry,
    SourceRequest,
    SourceResponse,
    SourceResult,
    register_source,
    resolve_source,
    source_registry,
)
from aivia.providers.sources.mock import MockSource, MockSourceProvider

__all__ = [
    "CAPABILITY_SOURCE",
    "MockSource",
    "MockSourceProvider",
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

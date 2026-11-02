"""Deterministic topic source mock.

The implementation lives with the other mocks so every capability has one
mock provider. This module re-exports it for callers that group sources under
``aivia.providers.sources`` and registers it in the source registry.
"""

from __future__ import annotations

from aivia.providers.mock import MockSource, MockSourceProvider
from aivia.providers.sources.base import register_source, source_registry

__all__ = ["MockSource", "MockSourceProvider"]


def register_mock_sources() -> None:
    """Register the mock source on the shared source registry."""
    for name in ("mock-source", "mock"):
        if name not in source_registry:
            register_source(name, MockSourceProvider, module=__name__)


register_mock_sources()

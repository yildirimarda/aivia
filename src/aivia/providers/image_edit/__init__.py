"""Image edit adapters.

Importing this package gives callers the built-in image edit providers and
their aliases. Each adapter registers itself with the provider registry.
"""

from aivia.providers.image_edit.aggregator import (
    AGGREGATOR_DEFAULT_BASE_URL,
    SUPPORTED_OPERATIONS,
    Aggregator,
    AggregatorImageEdit,
    AggregatorImageEditProvider,
    AggregatorProvider,
)

__all__ = [
    "AGGREGATOR_DEFAULT_BASE_URL",
    "SUPPORTED_OPERATIONS",
    "Aggregator",
    "AggregatorImageEdit",
    "AggregatorImageEditProvider",
    "AggregatorProvider",
]

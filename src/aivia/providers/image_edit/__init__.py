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
from aivia.providers.image_edit.rembg import (
    DEFAULT_MODEL,
    DEFAULT_TOLERANCE,
    LocalBackgroundRemoval,
    LocalBackgroundRemovalProvider,
    LocalBackgroundRemover,
    Rembg,
    RembgImageEdit,
    RembgImageEditProvider,
    RembgProvider,
    Remover,
    remove_background,
)

__all__ = [
    "AGGREGATOR_DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_TOLERANCE",
    "SUPPORTED_OPERATIONS",
    "Aggregator",
    "AggregatorImageEdit",
    "AggregatorImageEditProvider",
    "AggregatorProvider",
    "LocalBackgroundRemoval",
    "LocalBackgroundRemovalProvider",
    "LocalBackgroundRemover",
    "Rembg",
    "RembgImageEdit",
    "RembgImageEditProvider",
    "RembgProvider",
    "Remover",
    "remove_background",
]

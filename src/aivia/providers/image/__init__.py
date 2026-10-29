"""Image generation adapters.

Importing this package gives callers the built-in image providers and their
aliases. Each adapter registers itself with the provider registry.
"""

from aivia.providers.image.fal import (
    DEFAULT_BASE_URL,
    DEFAULT_QUEUE_URL,
    FAL_DEFAULT_BASE_URL,
    FAL_QUEUE_URL,
    Fal,
    FalImageProvider,
    FalProvider,
)
from aivia.providers.image.replicate import (
    REPLICATE_API_URL,
    REPLICATE_BASE_URL,
    REPLICATE_DEFAULT_BASE_URL,
    Replicate,
    ReplicateImageProvider,
    ReplicateProvider,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_QUEUE_URL",
    "FAL_DEFAULT_BASE_URL",
    "FAL_QUEUE_URL",
    "REPLICATE_API_URL",
    "REPLICATE_BASE_URL",
    "REPLICATE_DEFAULT_BASE_URL",
    "Fal",
    "FalImageProvider",
    "FalProvider",
    "Replicate",
    "ReplicateImageProvider",
    "ReplicateProvider",
]

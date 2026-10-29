"""fal image generation adapter.

Importing this package gives callers the fal image provider and its aliases.
The adapter registers itself with the provider registry under ``fal``.
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

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_QUEUE_URL",
    "FAL_DEFAULT_BASE_URL",
    "FAL_QUEUE_URL",
    "Fal",
    "FalImageProvider",
    "FalProvider",
]

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
from aivia.providers.image_edit.remote_ops import (
    DEFAULT_SCALES,
    DEFAULT_UPSCALE_FACTORS,
    REMOTE_OPS_DEFAULT_BASE_URL,
    SUPPORTED_UPSCALE_FACTORS,
    RemoteBackgroundRemovalAndUpscale,
    RemoteBackgroundRemovalAndUpscaleProvider,
    RemoteImageEdit,
    RemoteImageEditProvider,
    RemoteImageOps,
    RemoteImageOpsProvider,
    RemoteOps,
    RemoteOpsImageEditProvider,
    RemoteOpsProvider,
)

__all__ = [
    "AGGREGATOR_DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_SCALES",
    "DEFAULT_TOLERANCE",
    "DEFAULT_UPSCALE_FACTORS",
    "REMOTE_OPS_DEFAULT_BASE_URL",
    "SUPPORTED_OPERATIONS",
    "SUPPORTED_UPSCALE_FACTORS",
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
    "RemoteBackgroundRemovalAndUpscale",
    "RemoteBackgroundRemovalAndUpscaleProvider",
    "RemoteImageEdit",
    "RemoteImageEditProvider",
    "RemoteImageOps",
    "RemoteImageOpsProvider",
    "RemoteOps",
    "RemoteOpsImageEditProvider",
    "RemoteOpsProvider",
    "Remover",
    "remove_background",
]

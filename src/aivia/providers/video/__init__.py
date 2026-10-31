"""Video generation adapters.

Importing this package gives callers the built-in video providers and their
aliases. Each adapter registers itself with the provider registry.
"""

from aivia.providers.protocols.video import (
    CAPABILITY_VIDEO_FIRST_FRAME,
    CAPABILITY_VIDEO_GEN,
    CAPABILITY_VIDEO_IMAGE_INPUT,
)
from aivia.providers.video.fal import (
    DEFAULT_BASE_URL,
    DEFAULT_QUEUE_URL,
    FAL_DEFAULT_BASE_URL,
    FAL_QUEUE_URL,
    Fal,
    FalVideo,
    FalVideoGen,
    FalVideoGenProvider,
    FalVideoProvider,
)

__all__ = [
    "CAPABILITY_VIDEO_FIRST_FRAME",
    "CAPABILITY_VIDEO_GEN",
    "CAPABILITY_VIDEO_IMAGE_INPUT",
    "DEFAULT_BASE_URL",
    "DEFAULT_QUEUE_URL",
    "FAL_DEFAULT_BASE_URL",
    "FAL_QUEUE_URL",
    "Fal",
    "FalVideo",
    "FalVideoGen",
    "FalVideoGenProvider",
    "FalVideoProvider",
]

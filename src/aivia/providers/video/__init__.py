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
from aivia.providers.video.replicate import (
    REPLICATE_API_URL,
    REPLICATE_BASE_URL,
    REPLICATE_DEFAULT_BASE_URL,
    REPLICATE_VIDEO_API_URL,
    REPLICATE_VIDEO_BASE_URL,
    REPLICATE_VIDEO_DEFAULT_BASE_URL,
    ReplicateVideo,
    ReplicateVideoGen,
    ReplicateVideoGenProvider,
    ReplicateVideoProvider,
)
from aivia.providers.video.veo import (
    GOOGLE_API_URL,
    GOOGLE_DEFAULT_BASE_URL,
    GOOGLE_VEO_API_URL,
    GOOGLE_VEO_BASE_URL,
    VEO_API_URL,
    VEO_DEFAULT_BASE_URL,
    Veo,
    VeoProvider,
    VeoVideo,
    VeoVideoGen,
    VeoVideoGenProvider,
    VeoVideoProvider,
)

__all__ = [
    "CAPABILITY_VIDEO_FIRST_FRAME",
    "CAPABILITY_VIDEO_GEN",
    "CAPABILITY_VIDEO_IMAGE_INPUT",
    "DEFAULT_BASE_URL",
    "DEFAULT_QUEUE_URL",
    "FAL_DEFAULT_BASE_URL",
    "FAL_QUEUE_URL",
    "GOOGLE_API_URL",
    "GOOGLE_DEFAULT_BASE_URL",
    "GOOGLE_VEO_API_URL",
    "GOOGLE_VEO_BASE_URL",
    "REPLICATE_API_URL",
    "REPLICATE_BASE_URL",
    "REPLICATE_DEFAULT_BASE_URL",
    "REPLICATE_VIDEO_API_URL",
    "REPLICATE_VIDEO_BASE_URL",
    "REPLICATE_VIDEO_DEFAULT_BASE_URL",
    "VEO_API_URL",
    "VEO_DEFAULT_BASE_URL",
    "Fal",
    "FalVideo",
    "FalVideoGen",
    "FalVideoGenProvider",
    "FalVideoProvider",
    "ReplicateVideo",
    "ReplicateVideoGen",
    "ReplicateVideoGenProvider",
    "ReplicateVideoProvider",
    "Veo",
    "VeoProvider",
    "VeoVideo",
    "VeoVideoGen",
    "VeoVideoGenProvider",
    "VeoVideoProvider",
]

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
from aivia.providers.image.google import (
    GOOGLE_API_URL,
    GOOGLE_BASE_URL,
    GOOGLE_DEFAULT_BASE_URL,
    GOOGLE_IMAGEN_API_URL,
    Google,
    GoogleImage,
    GoogleImagen,
    GoogleImagenProvider,
    GoogleImageProvider,
    GoogleImages,
    GoogleImagesProvider,
    GoogleProvider,
)
from aivia.providers.image.openai import (
    OPENAI_API_URL,
    OPENAI_BASE_URL,
    OPENAI_DEFAULT_BASE_URL,
    OpenAI,
    OpenAIImage,
    OpenAIImageProvider,
    OpenAIImages,
    OpenAIImagesProvider,
)
from aivia.providers.image.reference import (
    CAPABILITY_IMAGE_INPUT,
    ReferenceInputNotSupportedError,
    ensure_reference_support,
    filter_providers_for_reference_input,
    prepare_image_request,
    required_image_capabilities,
    supports_reference_images,
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
    "CAPABILITY_IMAGE_INPUT",
    "DEFAULT_BASE_URL",
    "DEFAULT_QUEUE_URL",
    "FAL_DEFAULT_BASE_URL",
    "FAL_QUEUE_URL",
    "GOOGLE_API_URL",
    "GOOGLE_BASE_URL",
    "GOOGLE_DEFAULT_BASE_URL",
    "GOOGLE_IMAGEN_API_URL",
    "OPENAI_API_URL",
    "OPENAI_BASE_URL",
    "OPENAI_DEFAULT_BASE_URL",
    "REPLICATE_API_URL",
    "REPLICATE_BASE_URL",
    "REPLICATE_DEFAULT_BASE_URL",
    "Fal",
    "FalImageProvider",
    "FalProvider",
    "Google",
    "GoogleImage",
    "GoogleImageProvider",
    "GoogleImagen",
    "GoogleImagenProvider",
    "GoogleImages",
    "GoogleImagesProvider",
    "GoogleProvider",
    "OpenAI",
    "OpenAIImage",
    "OpenAIImageProvider",
    "OpenAIImages",
    "OpenAIImagesProvider",
    "ReferenceInputNotSupportedError",
    "Replicate",
    "ReplicateImageProvider",
    "ReplicateProvider",
    "ensure_reference_support",
    "filter_providers_for_reference_input",
    "prepare_image_request",
    "required_image_capabilities",
    "supports_reference_images",
]

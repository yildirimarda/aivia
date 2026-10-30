"""Stock media adapters.

Importing this package gives callers the built-in Pexels and Pixabay adapters
and their aliases. Each adapter registers itself with the provider registry.
"""

from aivia.providers.stock.pexels import (
    MAX_PER_PAGE as PEXELS_MAX_PER_PAGE,
)
from aivia.providers.stock.pexels import (
    PEXELS_DEFAULT_BASE_URL,
    Pexels,
    PexelsProvider,
    PexelsStockProvider,
)
from aivia.providers.stock.pixabay import (
    MAX_PER_PAGE as PIXABAY_MAX_PER_PAGE,
)
from aivia.providers.stock.pixabay import (
    MIN_PER_PAGE as PIXABAY_MIN_PER_PAGE,
)
from aivia.providers.stock.pixabay import (
    PIXABAY_DEFAULT_BASE_URL,
    Pixabay,
    PixabayProvider,
    PixabayStockProvider,
)

__all__ = [
    "PEXELS_DEFAULT_BASE_URL",
    "PEXELS_MAX_PER_PAGE",
    "PIXABAY_DEFAULT_BASE_URL",
    "PIXABAY_MAX_PER_PAGE",
    "PIXABAY_MIN_PER_PAGE",
    "Pexels",
    "PexelsProvider",
    "PexelsStockProvider",
    "Pixabay",
    "PixabayProvider",
    "PixabayStockProvider",
]

"""Characters: recurring figures that stay consistent across outputs.

A character is a YAML file in the user data directory next to its reference
images and pose cutouts. :class:`~aivia.characters.schema.Character` is the
schema and :class:`~aivia.characters.loader.CharacterLibrary` loads and
validates one, naming the file and field in every error.
"""

from aivia.characters.brand_kit import (
    CURRENT_BRAND_KIT_SCHEMA_VERSION,
    BrandKit,
    BrandKitLibrary,
    load_brand_kit_file,
)
from aivia.characters.errors import (
    BrandKitError,
    BrandKitNotFoundError,
    BrandKitSchemaError,
    CharacterError,
    CharacterNotFoundError,
    CharacterSchemaError,
)
from aivia.characters.loader import CharacterLibrary, load_character_file
from aivia.characters.schema import (
    CURRENT_CHARACTER_SCHEMA_VERSION,
    Character,
)

__all__ = [
    "CURRENT_BRAND_KIT_SCHEMA_VERSION",
    "CURRENT_CHARACTER_SCHEMA_VERSION",
    "BrandKit",
    "BrandKitError",
    "BrandKitLibrary",
    "BrandKitNotFoundError",
    "BrandKitSchemaError",
    "Character",
    "CharacterError",
    "CharacterLibrary",
    "CharacterNotFoundError",
    "CharacterSchemaError",
    "load_brand_kit_file",
    "load_character_file",
]

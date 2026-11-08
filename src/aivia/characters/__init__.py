"""Characters: recurring figures that stay consistent across outputs.

A character is a YAML file in the user data directory next to its reference
images and pose cutouts. :class:`~aivia.characters.schema.Character` is the
schema and :class:`~aivia.characters.loader.CharacterLibrary` loads and
validates one, naming the file and field in every error.
"""

from aivia.characters.adapter import (
    AdapterUnusableError,
    TrainerAdapter,
)
from aivia.characters.brand_kit import (
    CURRENT_BRAND_KIT_SCHEMA_VERSION,
    BrandKit,
    BrandKitLibrary,
    load_brand_kit_file,
)
from aivia.characters.chain import (
    Asset,
    Scene,
    SceneResult,
    Strategy,
    run_strategy_chain,
)
from aivia.characters.composite import (
    Flip,
    PlaceCutoutError,
    idle_motion,
    place_cutout,
)
from aivia.characters.errors import (
    BrandKitError,
    BrandKitNotFoundError,
    BrandKitSchemaError,
    CharacterError,
    CharacterNotFoundError,
    CharacterSchemaError,
    PoseError,
    PoseNotFoundError,
    PoseSchemaError,
)
from aivia.characters.loader import CharacterLibrary, load_character_file
from aivia.characters.pose import (
    POSE_LIBRARY_SCHEMA_VERSION,
    Pose,
    PoseLibrary,
    cleanup_edges,
    load_pose_library,
)
from aivia.characters.roles import (
    CHARACTER_ROLES,
    CharacterCast,
    select_characters,
)
from aivia.characters.schema import (
    CURRENT_CHARACTER_SCHEMA_VERSION,
    Character,
)
from aivia.characters.select import select_character

__all__ = [
    "CHARACTER_ROLES",
    "CURRENT_BRAND_KIT_SCHEMA_VERSION",
    "CURRENT_CHARACTER_SCHEMA_VERSION",
    "POSE_LIBRARY_SCHEMA_VERSION",
    "AdapterUnusableError",
    "Asset",
    "BrandKit",
    "BrandKitError",
    "BrandKitLibrary",
    "BrandKitNotFoundError",
    "BrandKitSchemaError",
    "Character",
    "CharacterCast",
    "CharacterError",
    "CharacterLibrary",
    "CharacterNotFoundError",
    "CharacterSchemaError",
    "Flip",
    "PlaceCutoutError",
    "Pose",
    "PoseError",
    "PoseLibrary",
    "PoseNotFoundError",
    "PoseSchemaError",
    "Scene",
    "SceneResult",
    "Strategy",
    "TrainerAdapter",
    "cleanup_edges",
    "idle_motion",
    "load_brand_kit_file",
    "load_character_file",
    "load_pose_library",
    "place_cutout",
    "run_strategy_chain",
    "select_character",
    "select_characters",
]

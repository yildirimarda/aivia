"""Deterministic JSON Schema export for the registered configuration sections.

The schema is built from the section registry, so a module that registers its
own top-level key appears without the core knowing about it. Serialisation is
stable (sorted keys, fixed indentation, trailing newline) so the committed
artifact can be compared byte for byte and a stale copy fails CI.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from pydantic import Field, create_model

from aivia.core.config.base import ConfigModel
from aivia.core.config.migrations import CURRENT_SCHEMA_VERSION
from aivia.core.config.registry import SectionRegistry, registry
from aivia.core.config.sections import register_core_sections
from aivia.core.paths import package_resource

__all__ = [
    "SCHEMA_DIALECT",
    "SCHEMA_FILENAME",
    "SCHEMA_TITLE",
    "build_config_schema",
    "dump_config_schema",
    "packaged_schema_path",
]

SCHEMA_FILENAME = "config.schema.json"
SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_TITLE = "AIVIA configuration"


def build_config_schema(
    sections: SectionRegistry | None = None,
) -> dict[str, Any]:
    """Return the JSON Schema for every section registered in *sections*.

    A custom registry is filled with the core sections first, mirroring
    :func:`aivia.core.config.app.build_app_config`. ``schema_version`` is
    included because it is a valid root key even though it is not a section.
    """
    target = registry if sections is None else sections
    register_core_sections(target)
    fields: dict[str, Any] = {
        "schema_version": (int, Field(default=CURRENT_SCHEMA_VERSION, ge=1)),
    }
    fields.update({section.key: (section.model | None, None) for section in target})
    model = cast(
        type[ConfigModel],
        create_model("ConfigSchema", __base__=ConfigModel, **fields),
    )
    schema = model.model_json_schema()
    schema["$schema"] = SCHEMA_DIALECT
    schema["title"] = SCHEMA_TITLE
    return schema


def dump_config_schema(schema: Mapping[str, Any] | None = None) -> str:
    """Render *schema* (or the current schema) as stable JSON text."""
    payload = build_config_schema() if schema is None else dict(schema)
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )


def packaged_schema_path() -> Path:
    """Return the path of the committed schema inside the package."""
    return package_resource("data", SCHEMA_FILENAME)

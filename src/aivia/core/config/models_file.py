"""Location of the editable model catalog file.

The catalog defaults to ``models.yaml`` beside the user configuration. ``path``
may be absolute, start with ``~`` or be relative to the config directory, so a
custom catalog never depends on the current working directory. When the path is
the default file name, a missing user copy falls back to the packaged catalog.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

from pydantic import Field, field_validator, model_validator

from aivia.core.config.base import ConfigModel
from aivia.core.model_ids import CATALOG_FILENAME, resolve_model_catalog_path

__all__ = ["ModelsFileConfig", "ModelsFileSettings"]

_PATH_ALIASES = ("file", "filename", "catalog", "catalog_file", "models_file")


class ModelsFileConfig(ConfigModel):
    """The model catalog path, with a documented default of ``models.yaml``.

    A bare string is accepted as shorthand for ``{path: <string>}`` so the
    canonical ``models_file: models.yaml`` form in the design stays valid.
    """

    path: str = Field(default=CATALOG_FILENAME, min_length=1)

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | str,
        *,
        environ: Mapping[str, str] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> Self:
        """Validate a section mapping or the scalar path shorthand."""
        if isinstance(values, str):
            values = {"path": values}
        return super().from_mapping(values, environ=environ, context=context)

    @model_validator(mode="before")
    @classmethod
    def _accept_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"path": data}
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for alias in _PATH_ALIASES:
            if alias in values:
                values.setdefault("path", values.pop(alias))
        return values

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        path = value.strip()
        if not path:
            raise ValueError("models_file.path must not be empty")
        return path

    @property
    def filename(self) -> str:
        """Alias for :attr:`path` for callers that read it as a file name."""
        return self.path

    def resolve(self, config_dir: str | Path | None = None) -> Path:
        """Return the catalog path this section selects."""
        return resolve_model_catalog_path(self, config_dir=config_dir)

    def resolved_path(self, config_dir: str | Path | None = None) -> Path:
        """Alias for :meth:`resolve`."""
        return self.resolve(config_dir=config_dir)


ModelsFileSettings = ModelsFileConfig

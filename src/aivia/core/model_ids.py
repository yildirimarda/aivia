"""Known model ids and advisory checks for configured ids.

Vendors rename and retire models far more often than the application releases.
The catalog therefore lives in ``models.yaml`` as data. Configured ids are
checked against it without failing validation: an unknown id is reported with
the closest known id, and a deprecated id is reported with its replacement.
"""

from __future__ import annotations

import difflib
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from aivia.core import paths
from aivia.core.errors import ConfigError

__all__ = [
    "CATALOG_FILENAME",
    "CATALOG_SCHEMA_VERSION",
    "CLOSE_MATCH_CUTOFF",
    "ConfiguredModelId",
    "ModelCatalog",
    "ModelIdIssue",
    "ModelIdStatus",
    "ModelIdWarning",
    "ModelInfo",
    "configured_model_ids",
    "load_model_catalog",
    "model_catalog_path",
    "resolve_model_catalog_path",
    "validate_configured_model_ids",
    "validate_model_ids",
    "warn_configured_model_ids",
    "warn_model_ids",
]

CATALOG_FILENAME = "models.yaml"
CATALOG_SCHEMA_VERSION = 1
CLOSE_MATCH_CUTOFF = 0.6

_ENTRY_KEYS = frozenset({"capabilities", "provider", "deprecated", "replacement"})
_TOP_LEVEL_KEYS = frozenset({"schema_version", "models"})


class ModelIdWarning(UserWarning):
    """A configured model id is unknown or deprecated."""


class ModelIdStatus(StrEnum):
    """Why a configured model id needs attention."""

    UNKNOWN = "unknown"
    DEPRECATED = "deprecated"


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """One model id known to the catalog."""

    id: str
    capabilities: tuple[str, ...] = ()
    provider: str | None = None
    deprecated: bool = False
    replacement: str | None = None


@dataclass(frozen=True, slots=True)
class ModelIdIssue:
    """An advisory finding for one configured model id."""

    model: str
    status: ModelIdStatus
    path: str = ""
    closest: str | None = None
    replacement: str | None = None

    @property
    def message(self) -> str:
        """Return a human-readable warning that names the closest match."""
        if self.status is ModelIdStatus.DEPRECATED:
            text = f"model id {self.model!r} is deprecated"
            if self.replacement is not None:
                text += f"; use {self.replacement!r} instead"
        else:
            text = f"unknown model id {self.model!r}"
            if self.closest is not None:
                text += f"; closest match is {self.closest!r}"
        return f"{self.path}: {text}" if self.path else text

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class ConfiguredModelId:
    """A model id found in configuration, with the dotted path that uses it."""

    model: str
    path: str


class ModelCatalog:
    """The model ids known to one ``models.yaml`` file."""

    def __init__(
        self,
        models: Mapping[str, ModelInfo] | Iterable[ModelInfo],
        *,
        source: Path | None = None,
    ) -> None:
        entries: dict[str, ModelInfo] = {}
        items = models.values() if isinstance(models, Mapping) else models
        for info in items:
            if info.id in entries:
                raise ConfigError(f"duplicate model id in catalog: {info.id}")
            entries[info.id] = info
        self._models = entries
        self.source = source
        self._validate_replacements()

    def _validate_replacements(self) -> None:
        for info in self._models.values():
            if not info.deprecated:
                continue
            if info.replacement is None:
                raise ConfigError(
                    f"deprecated model id {info.id!r} must name a replacement"
                )
            if info.replacement == info.id:
                raise ConfigError(
                    f"deprecated model id {info.id!r} cannot replace itself"
                )
            if info.replacement not in self._models:
                raise ConfigError(
                    f"replacement {info.replacement!r} for deprecated model id "
                    f"{info.id!r} is not in the catalog"
                )

    def __len__(self) -> int:
        return len(self._models)

    def __iter__(self) -> Iterator[str]:
        return iter(self.ids())

    def __contains__(self, model_id: object) -> bool:
        return isinstance(model_id, str) and model_id.strip() in self._models

    def ids(self) -> tuple[str, ...]:
        """Return every known id in deterministic order."""
        return tuple(sorted(self._models))

    def entries(self) -> dict[str, ModelInfo]:
        """Return a copy of the id-to-entry mapping."""
        return dict(self._models)

    def get(self, model_id: str) -> ModelInfo | None:
        """Return the entry for *model_id*, or ``None`` when it is unknown."""
        return self._models.get(model_id.strip())

    def closest(self, model_id: str) -> str | None:
        """Return the closest known id, when it is similar enough."""
        matches = difflib.get_close_matches(
            model_id.strip(),
            self.ids(),
            n=1,
            cutoff=CLOSE_MATCH_CUTOFF,
        )
        return matches[0] if matches else None

    def check(self, model_id: str, *, path: str = "") -> ModelIdIssue | None:
        """Return why *model_id* needs attention, or ``None`` when it is fine."""
        key = model_id.strip()
        info = self._models.get(key)
        if info is None:
            return ModelIdIssue(
                model=key,
                status=ModelIdStatus.UNKNOWN,
                path=path,
                closest=self.closest(key),
            )
        if info.deprecated:
            return ModelIdIssue(
                model=key,
                status=ModelIdStatus.DEPRECATED,
                path=path,
                replacement=info.replacement,
            )
        return None

    @classmethod
    def from_mapping(
        cls,
        payload: Any,
        *,
        source: Path | None = None,
    ) -> ModelCatalog:
        """Build a catalog from a parsed ``models.yaml`` payload."""
        if not isinstance(payload, Mapping):
            raise ConfigError("model catalog must be a mapping")

        unknown = sorted(str(key) for key in payload if key not in _TOP_LEVEL_KEYS)
        if unknown:
            raise ConfigError(f"unknown model catalog key: {', '.join(unknown)}")

        version = payload.get("schema_version", CATALOG_SCHEMA_VERSION)
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise ConfigError(
                f"model catalog schema_version must be a positive integer, "
                f"got {version!r}"
            )
        if version > CATALOG_SCHEMA_VERSION:
            raise ConfigError(
                f"model catalog schema version {version} is newer than supported "
                f"version {CATALOG_SCHEMA_VERSION}; upgrade aivia"
            )

        raw = payload.get("models", {})
        infos = _parse_models(raw, source=source)
        return cls(infos, source=source)

    @classmethod
    def load(cls, path: str | Path) -> ModelCatalog:
        """Load a catalog from *path*."""
        target = Path(path)
        try:
            text = target.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"cannot read model catalog {target}: {exc}") from exc
        try:
            payload = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in model catalog {target}: {exc}") from exc
        return cls.from_mapping(payload, source=target)


def _parse_models(raw: Any, *, source: Path | None) -> list[ModelInfo]:
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        return [
            _parse_entry(str(model_id), entry, source=source)
            for model_id, entry in raw.items()
        ]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [_parse_list_entry(entry, source=source) for entry in raw]
    location = f" in {source}" if source is not None else ""
    raise ConfigError(f"model catalog models must be a mapping or list{location}")


def _parse_list_entry(entry: Any, *, source: Path | None) -> ModelInfo:
    if not isinstance(entry, Mapping):
        raise ConfigError(_entry_error("model entry must be a mapping", source))
    values = dict(entry)
    model_id = values.pop("id", None)
    if not isinstance(model_id, str) or not model_id.strip():
        raise ConfigError(_entry_error("model entry requires a non-empty id", source))
    return _parse_entry(model_id, values, source=source)


def _parse_entry(
    model_id: str,
    entry: Any,
    *,
    source: Path | None,
) -> ModelInfo:
    key = model_id.strip()
    if not key:
        raise ConfigError(_entry_error("model id must be non-empty", source))
    if entry is None:
        values: Mapping[str, Any] = {}
    elif isinstance(entry, Mapping):
        values = entry
    else:
        raise ConfigError(
            _entry_error(f"model {key!r} settings must be a mapping", source)
        )

    unknown = sorted(str(name) for name in values if name not in _ENTRY_KEYS)
    if unknown:
        raise ConfigError(
            _entry_error(
                f"model {key!r} has unknown key: {', '.join(unknown)}",
                source,
            )
        )

    capabilities = _string_tuple(
        values.get("capabilities"),
        field=f"model {key!r} capabilities",
        source=source,
    )
    provider = _optional_string(
        values.get("provider"),
        field=f"model {key!r} provider",
        source=source,
    )
    replacement = _optional_string(
        values.get("replacement"),
        field=f"model {key!r} replacement",
        source=source,
    )
    deprecated = values.get("deprecated", False)
    if not isinstance(deprecated, bool):
        raise ConfigError(
            _entry_error(f"model {key!r} deprecated must be a boolean", source)
        )

    return ModelInfo(
        id=key,
        capabilities=capabilities,
        provider=provider,
        deprecated=deprecated,
        replacement=replacement,
    )


def _string_tuple(
    value: Any,
    *,
    field: str,
    source: Path | None,
) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigError(_entry_error(f"{field} must be a list", source))
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(
                _entry_error(f"{field} entries must be non-empty strings", source)
            )
        name = item.strip()
        if name not in items:
            items.append(name)
    return tuple(items)


def _optional_string(value: Any, *, field: str, source: Path | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(_entry_error(f"{field} must be a non-empty string", source))
    return value.strip()


def _entry_error(message: str, source: Path | None) -> str:
    return f"{message} in {source}" if source is not None else message


def _configured_models_file(models_file: Any) -> str | None:
    """Extract a catalog path from a section, root config or mapping."""
    if models_file is None:
        return None
    if isinstance(models_file, str):
        path = models_file.strip()
        return path or None
    if isinstance(models_file, Path):
        return str(models_file)
    if isinstance(models_file, Mapping):
        if "models_file" in models_file:
            return _configured_models_file(models_file["models_file"])
        for key in ("path", "file", "filename", "catalog", "catalog_file"):
            if key in models_file:
                return _configured_models_file(models_file[key])
        return None
    nested = getattr(models_file, "models_file", None)
    if nested is not None and nested is not models_file:
        return _configured_models_file(nested)
    for name in ("path", "file", "filename", "catalog", "catalog_file"):
        value = getattr(models_file, name, None)
        if isinstance(value, (str, Path)):
            path = str(value).strip()
            return path or None
    return None


def resolve_model_catalog_path(
    models_file: Any = None,
    *,
    config_dir: str | Path | None = None,
) -> Path:
    """Return the catalog path selected by a ``models_file`` section.

    A relative path is resolved against the config directory, never the current
    working directory. The default file name keeps the existing behaviour: the
    user's editable copy wins when it exists, otherwise the packaged catalog is
    used.
    """
    base = paths.config_dir() if config_dir is None else Path(config_dir).expanduser()
    configured = _configured_models_file(models_file) or CATALOG_FILENAME
    target = Path(configured).expanduser()
    if target.is_absolute():
        return target
    candidate = base / target
    if target == Path(CATALOG_FILENAME):
        if candidate.is_file():
            return candidate
        return paths.package_defaults_dir() / CATALOG_FILENAME
    return candidate


def model_catalog_path(
    config_dir: str | Path | Any | None = None,
    models_file: Any = None,
    *,
    config: Any = None,
) -> Path:
    """Return the active catalog path, preferring the user's editable copy.

    The first argument may also be a ``models_file`` section or a root config,
    which makes ``model_catalog_path(config)`` read naturally.
    """
    if config is not None and models_file is None:
        models_file = config
    if config_dir is not None and not isinstance(config_dir, (str, Path)):
        if models_file is None:
            models_file = config_dir
            config_dir = None
        elif isinstance(models_file, (str, Path)):
            config_dir, models_file = models_file, config_dir
    elif models_file is None and isinstance(config_dir, (str, Path)):
        candidate = Path(config_dir)
        if candidate.name == CATALOG_FILENAME or candidate.suffix in {".yaml", ".yml"}:
            models_file = config_dir
            config_dir = None
    return resolve_model_catalog_path(
        models_file,
        config_dir=config_dir,
    )


def load_model_catalog(
    path: str | Path | Any | None = None,
    *,
    config_dir: str | Path | None = None,
    models_file: Any = None,
    config: Any = None,
) -> ModelCatalog:
    """Load the catalog from *path*, a ``models_file`` section or the default."""
    if (
        path is not None
        and not isinstance(path, (str, Path))
        and models_file is None
        and config is None
    ):
        models_file = path
        path = None
    target = (
        model_catalog_path(
            config_dir=config_dir,
            models_file=models_file,
            config=config,
        )
        if path is None
        else Path(path).expanduser()
    )
    return ModelCatalog.load(target)


def _llm_section(config: Any) -> Any:
    if isinstance(config, Mapping):
        if "llm" in config:
            return config.get("llm")
        if "roles" in config or "pricing" in config:
            return config
        return None
    llm = getattr(config, "llm", None)
    if llm is not None:
        return llm
    if hasattr(config, "roles") or hasattr(config, "pricing"):
        return config
    return None


def _field(section: Any, name: str) -> Any:
    if isinstance(section, Mapping):
        return section.get(name, {})
    return getattr(section, name, {})


def configured_model_ids(config: Any) -> tuple[ConfiguredModelId, ...]:
    """Return the model ids in *config* and the paths that use them."""
    llm = _llm_section(config)
    if llm is None:
        return ()

    uses: list[ConfiguredModelId] = []
    roles = _field(llm, "roles")
    if isinstance(roles, Mapping):
        for role in sorted(roles, key=str):
            model = roles[role]
            if isinstance(model, str) and model.strip():
                uses.append(
                    ConfiguredModelId(
                        model=model.strip(),
                        path=f"llm.roles.{role}",
                    )
                )

    pricing = _field(llm, "pricing")
    if isinstance(pricing, Mapping):
        for model in sorted(pricing, key=str):
            if isinstance(model, str) and model.strip():
                uses.append(
                    ConfiguredModelId(
                        model=model.strip(),
                        path=f"llm.pricing.{model.strip()}",
                    )
                )
    return tuple(uses)


def validate_model_ids(
    config: Any,
    *,
    catalog: ModelCatalog | None = None,
    config_dir: str | Path | None = None,
) -> list[ModelIdIssue]:
    """Return every unknown or deprecated model id configured in *config*."""
    resolved = (
        catalog
        if catalog is not None
        else load_model_catalog(models_file=config, config_dir=config_dir)
    )
    issues: list[ModelIdIssue] = []
    for use in configured_model_ids(config):
        issue = resolved.check(use.model, path=use.path)
        if issue is not None:
            issues.append(issue)
    return issues


def warn_model_ids(
    config: Any,
    *,
    catalog: ModelCatalog | None = None,
    config_dir: str | Path | None = None,
    stacklevel: int = 2,
) -> list[ModelIdIssue]:
    """Warn about every configured model id that needs attention."""
    issues = validate_model_ids(config, catalog=catalog, config_dir=config_dir)
    for issue in issues:
        warnings.warn(issue.message, ModelIdWarning, stacklevel=stacklevel)
    return issues


validate_configured_model_ids = validate_model_ids
warn_configured_model_ids = warn_model_ids

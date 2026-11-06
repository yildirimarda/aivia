"""Load named format recipes from the user directory and packaged defaults.

A format recipe may live in the configuration section or in its own file. The
packaged files are the source of truth for a fresh installation; a user file
with the same name overrides them, and an explicit ``formats`` configuration
section wins over both. Files are read in sorted path order so loading the same
directory twice is deterministic.

A file either holds one recipe (``stages`` at the top level) or a mapping of
names to recipes, optionally nested under ``formats``/``recipes``. The file
stem names a single recipe unless the payload supplies its own ``name``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml

from aivia.core import paths
from aivia.core.config.formats import FormatRecipeConfig, FormatsConfig
from aivia.core.errors import ConfigError

__all__ = [
    "FORMATS_DIR",
    "FORMAT_FILE_SUFFIXES",
    "FormatCatalog",
    "FormatStore",
    "RecipeCatalog",
    "discover_formats",
    "format_catalog",
    "formats_dir",
    "load_format",
    "load_format_directory",
    "load_format_file",
    "load_format_files",
    "load_formats",
    "load_packaged_formats",
    "load_recipe",
    "load_user_formats",
    "packaged_formats_dir",
    "resolve_format",
]

FORMATS_DIR = "formats"
FORMAT_FILE_SUFFIXES = frozenset({".yaml", ".yml", ".json"})

_RECIPE_KEYS = frozenset(
    {
        "capabilities",
        "description",
        "handler",
        "output",
        "output_type",
        "pipeline",
        "required_capabilities",
        "requires",
        "steps",
        "stages",
    }
)
_NAME_KEYS = ("name", "id")


def formats_dir(config_dir: str | Path | None = None) -> Path:
    """Return the user directory that holds format recipe files."""
    base = paths.config_dir() if config_dir is None else Path(config_dir)
    return Path(base).expanduser() / FORMATS_DIR


def packaged_formats_dir(defaults_dir: str | Path | None = None) -> Path:
    """Return the packaged directory that holds the default format files."""
    base = (
        paths.package_defaults_dir()
        if defaults_dir is None
        else Path(defaults_dir).expanduser()
    )
    return Path(base) / FORMATS_DIR


def _normalise_name(name: object) -> str:
    if not isinstance(name, str):
        raise ConfigError("format name must be a string")
    cleaned = name.strip()
    if not cleaned:
        raise ConfigError("format name must not be empty")
    return cleaned


def _recipe_from_mapping(
    values: Mapping[str, Any],
    *,
    name: str,
    source: Path | None,
) -> FormatRecipeConfig:
    payload = dict(values)
    for key in _NAME_KEYS:
        payload.pop(key, None)
    try:
        return FormatRecipeConfig.from_mapping(payload)
    except ConfigError as exc:
        where = f" in {source}" if source is not None else ""
        raise ConfigError(
            f"invalid format recipe {name!r}{where}: {exc}",
            details={"format": name, "source": str(source) if source else ""},
        ) from exc


def _recipes_from_payload(
    payload: Any,
    *,
    path: Path,
) -> dict[str, FormatRecipeConfig]:
    if not isinstance(payload, Mapping):
        raise ConfigError(
            f"format file must contain a mapping: {path}",
            details={"source": str(path)},
        )
    values = dict(payload)
    nested = None
    for key in ("formats", "recipes"):
        if key in values:
            nested = values.pop(key)
            break
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise ConfigError(
                f"format file {path} must map names to recipes under {key!r}",
                details={"source": str(path)},
            )
        values = dict(nested)
    elif _RECIPE_KEYS & set(values):
        name = next(
            (
                str(values[key]).strip()
                for key in _NAME_KEYS
                if isinstance(values.get(key), str) and values[key].strip()
            ),
            path.stem,
        )
        return {name: _recipe_from_mapping(values, name=name, source=path)}

    recipes: dict[str, FormatRecipeConfig] = {}
    for key, value in values.items():
        name = _normalise_name(key)
        if not isinstance(value, Mapping):
            raise ConfigError(
                f"format {name!r} in {path} must be a mapping",
                details={"format": name, "source": str(path)},
            )
        if name in recipes:
            raise ConfigError(
                f"duplicate format {name!r} in {path}",
                details={"format": name, "source": str(path)},
            )
        recipes[name] = _recipe_from_mapping(value, name=name, source=path)
    return recipes


def load_format_file(path: str | Path) -> dict[str, FormatRecipeConfig]:
    """Load every named recipe from one YAML or JSON format file."""
    target = Path(path).expanduser()
    if not target.is_file():
        raise ConfigError(
            f"format file not found: {target}",
            details={"source": str(target)},
        )
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read format file {target}: {exc}") from exc
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in format file {target}: {exc}") from exc
    if payload is None:
        raise ConfigError(
            f"format file is empty: {target}",
            details={"source": str(target)},
        )
    return _recipes_from_payload(payload, path=target)


def load_format_directory(
    directory: str | Path | None = None,
    *,
    required: bool = False,
    recursive: bool = False,
) -> dict[str, FormatRecipeConfig]:
    """Load every format file in *directory* in sorted path order.

    A missing default directory is empty. A caller-supplied directory that does
    not exist fails when *required* is true, otherwise it is also empty.
    """
    if directory is None:
        base = packaged_formats_dir()
        required = False
    else:
        base = Path(directory).expanduser()
    if not base.exists():
        if required:
            raise ConfigError(
                f"format directory does not exist: {base}",
                details={"path": str(base)},
            )
        return {}
    if not base.is_dir():
        raise ConfigError(
            f"format path is not a directory: {base}",
            details={"path": str(base)},
        )

    entries = base.rglob("*") if recursive else base.iterdir()
    files = sorted(
        (
            entry
            for entry in entries
            if entry.is_file()
            and not entry.name.startswith(".")
            and entry.suffix.lower() in FORMAT_FILE_SUFFIXES
        ),
        key=str,
    )
    recipes: dict[str, FormatRecipeConfig] = {}
    for file in files:
        for name, recipe in load_format_file(file).items():
            if name in recipes:
                raise ConfigError(
                    f"duplicate format {name!r} in {base}",
                    details={"format": name, "path": str(base)},
                )
            recipes[name] = recipe
    return recipes


def _recipes_from_config(
    config: Any,
    *,
    source: Path | None = None,
) -> dict[str, FormatRecipeConfig]:
    if config is None:
        return {}
    if isinstance(config, FormatCatalog):
        return dict(config.recipes)
    if isinstance(config, FormatRecipeConfig):
        raise ConfigError("a single recipe needs a name to be a catalog")
    if isinstance(config, FormatsConfig):
        return dict(config.formats)
    if isinstance(config, Mapping):
        values = dict(config)
        if "formats" in values and isinstance(values["formats"], Mapping):
            values = dict(values["formats"])
        elif "recipes" in values and isinstance(values["recipes"], Mapping):
            values = dict(values["recipes"])
        if not values:
            return {}
        if _RECIPE_KEYS & set(values):
            raise ConfigError("a single recipe needs a name to be a catalog")
        recipes: dict[str, FormatRecipeConfig] = {}
        for key, value in values.items():
            name = _normalise_name(key)
            if isinstance(value, FormatRecipeConfig):
                recipes[name] = value
            elif isinstance(value, Mapping):
                recipes[name] = _recipe_from_mapping(
                    value,
                    name=name,
                    source=source,
                )
            else:
                raise ConfigError(
                    f"format {name!r} must be a mapping",
                    details={"format": name},
                )
        return recipes
    section = getattr(config, "formats", None)
    if section is None:
        return {}
    if section is not config:
        return _recipes_from_config(section, source=source)
    raise ConfigError("cannot read formats from the supplied configuration")


class FormatCatalog:
    """Named format recipes with deterministic lookup and precedence."""

    def __init__(
        self,
        recipes: (
            Mapping[str, FormatRecipeConfig | Mapping[str, Any]]
            | Iterable[tuple[str, FormatRecipeConfig | Mapping[str, Any]]]
            | None
        ) = None,
        *,
        sources: Mapping[str, str] | None = None,
    ) -> None:
        self._recipes: dict[str, FormatRecipeConfig] = {}
        self._sources: dict[str, str] = dict(sources or {})
        if recipes is None:
            return
        items = recipes.items() if isinstance(recipes, Mapping) else recipes
        for name, recipe in items:
            self.register(name, recipe)

    @property
    def recipes(self) -> dict[str, FormatRecipeConfig]:
        """A copy of the name-to-recipe mapping in sorted name order."""
        return {name: self._recipes[name] for name in self.names()}

    @property
    def formats(self) -> dict[str, FormatRecipeConfig]:
        """Alias for :attr:`recipes`."""
        return self.recipes

    @property
    def sources(self) -> dict[str, str]:
        """A copy of the file each recipe came from, when known."""
        return dict(self._sources)

    def names(self) -> tuple[str, ...]:
        """Return the recipe names in sorted order."""
        return tuple(sorted(self._recipes))

    def keys(self) -> tuple[str, ...]:
        """Alias for :meth:`names`."""
        return self.names()

    def register(
        self,
        name: str,
        recipe: FormatRecipeConfig | Mapping[str, Any],
        *,
        source: str | Path | None = None,
        override: bool = False,
    ) -> FormatRecipeConfig:
        """Add *recipe* under *name*, optionally replacing an existing entry."""
        key = _normalise_name(name)
        if key in self._recipes and not override:
            raise ConfigError(
                f"format already registered: {key}",
                details={"format": key, "available": list(self.names())},
            )
        if isinstance(recipe, FormatRecipeConfig):
            value = recipe
        elif isinstance(recipe, Mapping):
            value = _recipe_from_mapping(
                recipe,
                name=key,
                source=Path(source).expanduser() if source is not None else None,
            )
        else:
            raise ConfigError(
                f"format {key!r} must be a recipe or mapping",
                details={"format": key},
            )
        self._recipes[key] = value
        if source is not None:
            self._sources[key] = str(source)
        return value

    def add(
        self,
        name: str,
        recipe: FormatRecipeConfig | Mapping[str, Any],
        *,
        source: str | Path | None = None,
    ) -> FormatRecipeConfig:
        """Alias for :meth:`register`."""
        return self.register(name, recipe, source=source)

    def update(
        self,
        recipes: Mapping[str, FormatRecipeConfig | Mapping[str, Any]],
        *,
        source: str | Path | None = None,
        override: bool = True,
    ) -> None:
        """Add every recipe in *recipes*, replacing entries by default."""
        for name, recipe in recipes.items():
            self.register(name, recipe, source=source, override=override)

    def merge(self, other: FormatCatalog, *, override: bool = True) -> FormatCatalog:
        """Merge *other* into this catalog and return this catalog."""
        for name in other.names():
            self.register(
                name,
                other._recipes[name],
                source=other._sources.get(name),
                override=override,
            )
        return self

    def get(self, name: str) -> FormatRecipeConfig:
        """Return the recipe named *name* or raise :class:`ConfigError`."""
        key = _normalise_name(name)
        recipe = self._recipes.get(key)
        if recipe is None:
            available = ", ".join(self.names()) or "none"
            raise ConfigError(
                f"unknown format: {key} (available: {available})",
                details={"format": key, "available": list(self.names())},
            )
        return recipe

    def find(self, name: str) -> FormatRecipeConfig | None:
        """Return the recipe named *name*, or ``None`` when it is unknown."""
        try:
            return self.get(name)
        except ConfigError:
            return None

    def resolve(self, name: str) -> FormatRecipeConfig:
        """Alias for :meth:`get`."""
        return self.get(name)

    def require(self, name: str) -> FormatRecipeConfig:
        """Alias for :meth:`get`."""
        return self.get(name)

    def load_file(
        self,
        path: str | Path,
        *,
        override: bool = True,
    ) -> dict[str, FormatRecipeConfig]:
        """Load one file and merge its recipes into this catalog."""
        loaded = load_format_file(path)
        self.update(loaded, source=path, override=override)
        return loaded

    def load_directory(
        self,
        directory: str | Path | None = None,
        *,
        required: bool = False,
        recursive: bool = False,
        override: bool = True,
    ) -> dict[str, FormatRecipeConfig]:
        """Load one directory and merge its recipes into this catalog."""
        loaded = load_format_directory(
            directory,
            required=required,
            recursive=recursive,
        )
        self.update(loaded, source=directory, override=override)
        return loaded

    def to_config(self) -> FormatsConfig:
        """Return the catalog as a validated ``formats`` configuration section."""
        return FormatsConfig(formats=dict(self._recipes))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.strip() in self._recipes

    def __getitem__(self, name: str) -> FormatRecipeConfig:
        return self.get(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._recipes)

    def __repr__(self) -> str:
        return f"FormatCatalog({', '.join(self.names())})"


FormatStore = FormatCatalog
RecipeCatalog = FormatCatalog


def load_packaged_formats(
    defaults_dir: str | Path | None = None,
    *,
    recursive: bool = False,
) -> FormatCatalog:
    """Load only the packaged format files."""
    catalog = FormatCatalog()
    catalog.load_directory(
        packaged_formats_dir(defaults_dir),
        recursive=recursive,
    )
    return catalog


def load_user_formats(
    config_dir: str | Path | None = None,
    *,
    recursive: bool = False,
) -> FormatCatalog:
    """Load only the user format files."""
    catalog = FormatCatalog()
    catalog.load_directory(
        formats_dir(config_dir),
        recursive=recursive,
    )
    return catalog


def load_formats(
    *directories: str | Path,
    config: Any = None,
    config_dir: str | Path | None = None,
    defaults_dir: str | Path | None = None,
    user_dir: str | Path | None = None,
    packaged_dir: str | Path | None = None,
    include_user: bool = True,
    include_packaged: bool = True,
    recursive: bool = False,
) -> FormatCatalog:
    """Load packaged, user, extra and configured format recipes.

    Precedence is lowest to highest: packaged files, user files, each extra
    directory in the order given, then an explicit configuration section. A
    missing default directory is empty; an explicit extra directory must exist.
    """
    catalog = FormatCatalog()
    if include_packaged:
        catalog.load_directory(
            packaged_dir or packaged_formats_dir(defaults_dir),
            recursive=recursive,
        )
    if include_user:
        catalog.load_directory(
            user_dir or formats_dir(config_dir),
            recursive=recursive,
        )
    for directory in directories:
        path = Path(directory).expanduser()
        if path.is_file():
            catalog.load_file(path, override=True)
            continue
        catalog.load_directory(
            path,
            required=True,
            recursive=recursive,
        )
    if config is not None:
        catalog.update(
            _recipes_from_config(config),
            source="<config>",
            override=True,
        )
    return catalog


format_catalog = load_formats


def resolve_format(
    name: str,
    *,
    catalog: FormatCatalog | None = None,
    config: Any = None,
    config_dir: str | Path | None = None,
    defaults_dir: str | Path | None = None,
    user_dir: str | Path | None = None,
    packaged_dir: str | Path | None = None,
    include_user: bool = True,
    include_packaged: bool = True,
    recursive: bool = False,
) -> FormatRecipeConfig:
    """Return the recipe named *name* from a catalog or the standard sources."""
    target = catalog or load_formats(
        config=config,
        config_dir=config_dir,
        defaults_dir=defaults_dir,
        user_dir=user_dir,
        packaged_dir=packaged_dir,
        include_user=include_user,
        include_packaged=include_packaged,
        recursive=recursive,
    )
    return target.get(name)


load_recipe = resolve_format


def load_format(
    name_or_path: str | Path,
    *,
    name: str | None = None,
    catalog: FormatCatalog | None = None,
    **options: Any,
) -> FormatRecipeConfig:
    """Load a recipe from a file path or resolve it by name.

    When a file holds several recipes, *name* selects one; otherwise the file
    stem or the only recipe in the file is used.
    """
    candidate = Path(name_or_path).expanduser()
    is_file = candidate.is_file() or (
        candidate.suffix.lower() in FORMAT_FILE_SUFFIXES and candidate.exists()
    )
    if not is_file:
        return resolve_format(str(name_or_path), catalog=catalog, **options)

    loaded = load_format_file(candidate)
    if name is not None:
        return FormatCatalog(loaded).get(name)
    if candidate.stem in loaded:
        return loaded[candidate.stem]
    if len(loaded) == 1:
        return next(iter(loaded.values()))
    available = ", ".join(sorted(loaded)) or "none"
    raise ConfigError(
        f"format file {candidate} contains several recipes; name one of: {available}",
        details={"source": str(candidate), "available": sorted(loaded)},
    )


def load_format_files(
    *paths: str | Path,
    config: Any = None,
    config_dir: str | Path | None = None,
    defaults_dir: str | Path | None = None,
    include_user: bool = True,
    include_packaged: bool = True,
    recursive: bool = False,
) -> FormatCatalog:
    """Alias for :func:`load_formats` for callers that name files explicitly."""
    return load_formats(
        *paths,
        config=config,
        config_dir=config_dir,
        defaults_dir=defaults_dir,
        include_user=include_user,
        include_packaged=include_packaged,
        recursive=recursive,
    )


def discover_formats(
    *directories: str | Path,
    config: Any = None,
    config_dir: str | Path | None = None,
    defaults_dir: str | Path | None = None,
    include_user: bool = True,
    include_packaged: bool = True,
    recursive: bool = False,
) -> tuple[str, ...]:
    """Return the names available from the standard sources in sorted order."""
    return load_formats(
        *directories,
        config=config,
        config_dir=config_dir,
        defaults_dir=defaults_dir,
        include_user=include_user,
        include_packaged=include_packaged,
        recursive=recursive,
    ).names()

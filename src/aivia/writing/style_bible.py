"""The style bible, loaded from user files with packaged defaults as fallback.

A style bible is the reusable guidance a writer and a media prompt share: the
voice to use, the structure to follow, the visual language to keep and the
things to avoid. It is data, not code, so a release can ship a usable default
while a user can shadow it with a file of their own.

User files live in ``<config>/styles`` and packaged files in
``<package>/data/defaults/styles``. A name without an extension is resolved
against both directories, preferring the user's copy, and every supported
extension is tried in turn. When neither copy exists the loader fails with a
message that names the bible and every directory it searched instead of
silently writing with no guidance.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Annotated, Any, cast

import yaml
from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from aivia.core import paths
from aivia.core.errors import ConfigError

__all__ = [
    "DEFAULT_STYLE_BIBLE",
    "DEFAULT_STYLE_BIBLE_FILENAME",
    "DEFAULT_STYLE_BIBLE_NAME",
    "PACKAGED_STYLE_BIBLE_DIR",
    "STYLE_BIBLE_DIR",
    "STYLE_BIBLE_DIRNAME",
    "STYLE_BIBLE_EXTENSIONS",
    "STYLE_BIBLE_FILENAME",
    "StyleBible",
    "StyleBibleEntries",
    "StyleBibleEntry",
    "StyleBibleError",
    "StyleBibleInvalidError",
    "StyleBibleLibrary",
    "StyleBibleNotFoundError",
    "StyleBibleSet",
    "build_style_bible",
    "default_style_bible",
    "default_style_bible_path",
    "load_default_style_bible",
    "load_style_bible",
    "load_style_bible_file",
    "package_style_bible_path",
    "read_style_bible",
    "resolve_style_bible_path",
    "style_bible",
    "style_bible_names",
    "style_bible_path",
]

STYLE_BIBLE_DIR = "styles"
STYLE_BIBLE_DIRNAME = STYLE_BIBLE_DIR
PACKAGED_STYLE_BIBLE_DIR = STYLE_BIBLE_DIR
STYLE_BIBLE_EXTENSIONS = (".yaml", ".yml", ".json", ".md", ".markdown", ".txt")
DEFAULT_STYLE_BIBLE_NAME = "default"
DEFAULT_STYLE_BIBLE_FILENAME = f"{DEFAULT_STYLE_BIBLE_NAME}.yaml"
STYLE_BIBLE_FILENAME = DEFAULT_STYLE_BIBLE_FILENAME

_NAME_KEYS = ("name", "id", "title", "key", "slug")
_DESCRIPTION_KEYS = ("description", "notes", "summary")
_SUFFIX_KEYS = ("prompt_suffix", "suffix", "prompt")
_TEXT_KEYS = ("text", "content", "body")
_ENTRY_LIST_KEYS = (
    "entries",
    "rules",
    "guidance",
    "sections",
    "items",
    "principles",
    "style",
)
_ENTRY_ID_KEYS = ("key", "id", "name", "slug")
_ENTRY_TEXT_KEYS = (
    "text",
    "value",
    "guidance",
    "content",
    "body",
    "instruction",
    "directive",
    "rule",
    "description",
)
_RESERVED_KEYS = frozenset(
    {
        "schema_version",
        "version",
        *_NAME_KEYS,
        *_DESCRIPTION_KEYS,
        *_SUFFIX_KEYS,
        *_TEXT_KEYS,
        *_ENTRY_LIST_KEYS,
    }
)

_WHITESPACE = re.compile(r"\s+", re.UNICODE)


class StyleBibleError(ConfigError):
    """A style bible is missing, unreadable or invalid."""

    code = "style_bible"


class StyleBibleNotFoundError(StyleBibleError):
    """No user or packaged style bible matches the requested name."""

    code = "style_bible_not_found"


class StyleBibleInvalidError(StyleBibleError):
    """A style bible file exists but its contents are not usable."""

    code = "style_bible_invalid"


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _normalise_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).casefold()).strip("-")


def _first(data: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _entry_from_value(key: str, value: Any) -> dict[str, Any] | None:
    """Turn one mapping value into an entry manifest."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        data = dict(value)
        if any(name in data for name in _ENTRY_TEXT_KEYS) or any(
            name in data for name in _ENTRY_ID_KEYS
        ):
            data.setdefault("key", key)
            return data
        return None
    if isinstance(value, (list, tuple)):
        text = _text(value)
        if not text:
            return None
        return {"key": key, "text": text}
    text = _text(value)
    if not text:
        return None
    return {"key": key, "text": text}


def _mapping_entries(value: Mapping[str, Any]) -> list[Any]:
    entries: list[Any] = []
    for key, item in value.items():
        if isinstance(item, Mapping):
            entry = _entry_from_value(str(key), item)
            if entry is not None:
                entries.append(entry)
                continue
            for child_key, child in item.items():
                child_entry = _entry_from_value(str(child_key), child)
                if child_entry is not None:
                    entries.append(child_entry)
            continue
        entry = _entry_from_value(str(key), item)
        if entry is not None:
            entries.append(entry)
    return entries


def _raw_entries(value: Any) -> list[Any]:
    if isinstance(value, StyleBible):
        return list(value.entries)
    if value is None:
        return []
    if isinstance(value, Mapping):
        return _mapping_entries(value)
    if isinstance(value, (str, bytes, bytearray)):
        text = _text(value)
        return [{"key": "style", "text": text}] if text else []
    if isinstance(value, Iterable):
        entries: list[Any] = []
        for index, item in enumerate(value, start=1):
            if isinstance(item, Mapping):
                entries.append(dict(item))
            elif isinstance(item, (str, bytes, bytearray)):
                text = _text(item)
                if text:
                    entries.append({"key": f"entry-{index}", "text": text})
            else:
                entry = _entry_from_value(f"entry-{index}", item)
                if entry is not None:
                    entries.append(entry)
        return entries
    raise StyleBibleInvalidError(
        "a style bible must be a mapping, a list or text",
        details={"value": repr(value)},
    )


def _bible_data(value: Any, *, name: str | None = None) -> dict[str, Any]:
    """Normalise a manifest into the fields :class:`StyleBible` expects."""
    if isinstance(value, StyleBible):
        return value.model_dump()
    if isinstance(value, Mapping):
        data = dict(value)
        resolved_name = _clean(_first(data, _NAME_KEYS) or name or "")
        description = _clean(_first(data, _DESCRIPTION_KEYS))
        suffix = _text(_first(data, _SUFFIX_KEYS))
        entries_value = _first(data, _ENTRY_LIST_KEYS)
        text_value = _first(data, _TEXT_KEYS)
        if entries_value is not None:
            entries = _raw_entries(entries_value)
        elif text_value is not None:
            text = _text(text_value)
            entries = [{"key": "style", "text": text}] if text else []
        else:
            entries = _mapping_entries(
                {key: item for key, item in data.items() if key not in _RESERVED_KEYS}
            )
        if not entries and suffix:
            entries = [{"key": "style", "text": suffix}]
        return {
            "name": resolved_name,
            "description": description,
            "prompt_suffix": suffix,
            "entries": entries,
            "schema_version": data.get("schema_version", data.get("version", 1)),
        }
    if isinstance(value, (str, bytes, bytearray)):
        text = _text(value)
        return {
            "name": _clean(name or ""),
            "description": "",
            "prompt_suffix": "",
            "entries": [{"key": "style", "text": text}] if text else [],
            "schema_version": 1,
        }
    if isinstance(value, Iterable):
        return {
            "name": _clean(name or ""),
            "description": "",
            "prompt_suffix": "",
            "entries": _raw_entries(value),
            "schema_version": 1,
        }
    raise StyleBibleInvalidError(
        "a style bible must be a mapping, a list or text",
        details={"value": repr(value)},
    )


class StyleBibleEntry(BaseModel):
    """One reusable rule in a style bible.

    An entry has a stable key, the guidance itself and an optional prompt
    suffix. The suffix lets a visual prompt carry the same rule in a shorter
    form without repeating the full explanation.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    key: str = Field(
        validation_alias=AliasChoices("key", "id", "name", "slug", "field"),
    )
    title: str = Field(
        default="",
        validation_alias=AliasChoices("title", "label", "heading"),
    )
    text: str = Field(
        validation_alias=AliasChoices(
            "text",
            "value",
            "guidance",
            "content",
            "body",
            "instruction",
            "directive",
            "rule",
            "description",
        ),
    )
    prompt_suffix: str = Field(
        default="",
        validation_alias=AliasChoices(
            "prompt_suffix",
            "suffix",
            "prompt",
            "prompt_fragment",
        ),
    )
    aliases: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @field_validator("key", mode="before")
    @classmethod
    def _validate_key(cls, value: Any) -> str:
        text = _clean(value)
        if not text:
            raise ValueError("style bible entry key must not be empty")
        return text

    @field_validator("title", "text", "prompt_suffix", mode="before")
    @classmethod
    def _clean_text(cls, value: Any, info: ValidationInfo) -> str:
        text = _text(value)
        if info.field_name == "text" and not text:
            raise ValueError("style bible entry text must not be empty")
        return text

    @field_validator("aliases", "tags", mode="before")
    @classmethod
    def _normalise_sequence(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        if isinstance(value, (bytes, bytearray, Mapping)):
            raise ValueError("style bible entry aliases must be a sequence")
        if isinstance(value, Iterable):
            return tuple(str(item).strip() for item in value if str(item).strip())
        raise ValueError("style bible entry aliases must be a sequence")

    @model_validator(mode="after")
    def _fill_title(self) -> StyleBibleEntry:
        if not self.title:
            self.title = self.key
        return self

    @property
    def id(self) -> str:
        """Alias for :attr:`key`."""
        return self.key

    @property
    def name(self) -> str:
        """The display name of this entry."""
        return self.title or self.key

    @property
    def value(self) -> str:
        """Alias for :attr:`text`."""
        return self.text

    @property
    def guidance(self) -> str:
        """Alias for :attr:`text`."""
        return self.text

    @property
    def suffix(self) -> str:
        """Alias for :attr:`prompt_suffix`."""
        return self.prompt_suffix

    def matches(self, value: Any) -> bool:
        """Whether *value* names this entry by key, title or alias."""
        key = _normalise_key(value)
        if not key:
            return False
        candidates = (self.key, self.title, *self.aliases, *self.tags)
        return key in {_normalise_key(candidate) for candidate in candidates}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this entry."""
        return {
            "key": self.key,
            "title": self.title,
            "text": self.text,
            "prompt_suffix": self.prompt_suffix,
            "aliases": list(self.aliases),
            "tags": list(self.tags),
        }


class StyleBibleEntries(tuple[StyleBibleEntry, ...]):
    """An ordered tuple of entries that also supports lookup by name."""

    def __new__(
        cls,
        entries: Iterable[StyleBibleEntry | Mapping[str, Any]] = (),
    ) -> StyleBibleEntries:
        items: list[StyleBibleEntry] = []
        for value in entries:
            if isinstance(value, StyleBibleEntry):
                items.append(value)
            elif isinstance(value, Mapping):
                items.append(StyleBibleEntry.model_validate(dict(value)))
            else:
                raise StyleBibleInvalidError(
                    "a style bible entry must be a mapping or a StyleBibleEntry",
                    details={"value": repr(value)},
                )
        return super().__new__(cls, tuple(items))

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, str):
            entry = self.get(index)
            if entry is None:
                raise KeyError(index)
            return entry
        return super().__getitem__(index)

    def get(self, value: Any) -> StyleBibleEntry | None:
        """Return the entry named by *value*, or ``None``."""
        for entry in self:
            if entry.matches(value):
                return entry
        return None

    def keys(self) -> tuple[str, ...]:
        """Return every entry key in order."""
        return tuple(entry.key for entry in self)

    def values(self) -> tuple[StyleBibleEntry, ...]:
        """Return every entry in order."""
        return tuple(self)

    def items(self) -> tuple[tuple[str, StyleBibleEntry], ...]:
        """Return ``(key, entry)`` pairs in order."""
        return tuple((entry.key, entry) for entry in self)

    @property
    def by_key(self) -> dict[str, StyleBibleEntry]:
        """Return a key index of every entry."""
        return {entry.key: entry for entry in self}


def _coerce_entries(value: Any) -> StyleBibleEntries:
    if isinstance(value, StyleBibleEntries):
        return value
    return StyleBibleEntries(_raw_entries(value))


class StyleBible(BaseModel):
    """A named set of style rules plus its source file."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    name: str = ""
    description: str = ""
    entries: Annotated[StyleBibleEntries, BeforeValidator(_coerce_entries)]
    prompt_suffix: str = ""
    schema_version: int = 1
    source: Path | None = Field(default=None, exclude=True)

    @field_validator("name", "description", "prompt_suffix", mode="before")
    @classmethod
    def _clean_text(cls, value: Any, info: ValidationInfo) -> str:
        if info.field_name == "prompt_suffix":
            return _text(value)
        return _clean(value)

    @model_validator(mode="after")
    def _require_content(self) -> StyleBible:
        if not self.name:
            self.name = DEFAULT_STYLE_BIBLE_NAME
        if not self.entries:
            raise ValueError("a style bible must define at least one entry")
        return self

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        name: str | None = None,
        source: str | Path | None = None,
    ) -> StyleBible:
        """Build a bible from a manifest mapping, list or text."""
        if isinstance(value, StyleBible):
            if source is not None:
                value = value.model_copy(update={"source": Path(source)})
            return value
        bible = cls.model_validate(_bible_data(value, name=name))
        if source is not None:
            bible.source = Path(source).expanduser()
        return bible

    @classmethod
    def from_file(cls, path: str | Path) -> StyleBible:
        """Load a YAML, JSON, Markdown or text style bible from *path*."""
        manifest = Path(path).expanduser()
        if not manifest.is_file():
            raise StyleBibleNotFoundError(
                f"style bible not found: {manifest}",
                details={"path": str(manifest)},
            )
        text = manifest.read_text(encoding="utf-8")
        suffix = manifest.suffix.lower()
        try:
            if suffix in {".yaml", ".yml"}:
                payload: object = yaml.safe_load(text)
            elif suffix == ".json":
                payload = json.loads(text)
            elif suffix in {".md", ".markdown", ".txt"}:
                payload = text
            else:
                raise StyleBibleInvalidError(
                    f"unsupported style bible extension: {suffix}",
                    details={"path": str(manifest)},
                )
        except (yaml.YAMLError, json.JSONDecodeError) as exc:
            raise StyleBibleInvalidError(
                f"invalid style bible: {manifest}",
                details={"path": str(manifest), "error": str(exc)},
            ) from exc
        if payload is None:
            raise StyleBibleInvalidError(
                f"empty style bible: {manifest}",
                details={"path": str(manifest)},
            )
        return cls.from_mapping(payload, name=manifest.stem, source=manifest)

    @classmethod
    def from_default(cls, name: str = DEFAULT_STYLE_BIBLE_NAME) -> StyleBible:
        """Load a bible shipped as package data."""
        return cls.from_file(default_style_bible_path(name))

    def get(self, value: Any) -> StyleBibleEntry | None:
        """Return an entry by key, title or alias."""
        return self.entries.get(value)

    def require(self, value: Any) -> StyleBibleEntry:
        """Return an entry or raise :class:`StyleBibleInvalidError`."""
        entry = self.get(value)
        if entry is None:
            raise StyleBibleInvalidError(
                f"unknown style bible entry: {value!r}",
                details={"entry": repr(value), "known": list(self.entries.keys())},
            )
        return entry

    entry = require

    @property
    def by_key(self) -> dict[str, StyleBibleEntry]:
        """Return a key index of every entry."""
        return self.entries.by_key

    def prompt_lines(self) -> tuple[str, ...]:
        """Return one readable line per entry for a model prompt."""
        return tuple(f"- {entry.title}: {entry.text}" for entry in self.entries)

    def suffixes(self) -> tuple[str, ...]:
        """Return every non-empty entry prompt suffix, then the bible suffix."""
        values = [entry.prompt_suffix for entry in self.entries if entry.prompt_suffix]
        if self.prompt_suffix:
            values.append(self.prompt_suffix)
        return tuple(dict.fromkeys(values))

    @property
    def suffix(self) -> str:
        """Return every prompt suffix joined into one string."""
        return " ".join(self.suffixes())

    @property
    def text(self) -> str:
        """Return the whole bible as readable prompt text."""
        return self.to_prompt()

    @property
    def body(self) -> str:
        """Alias for :attr:`text`."""
        return self.text

    def to_prompt(self) -> str:
        """Render the bible as deterministic prompt text."""
        lines: list[str] = []
        if self.description:
            lines.append(self.description)
        lines.extend(self.prompt_lines())
        if self.prompt_suffix:
            lines.append(self.prompt_suffix)
        return "\n".join(lines)

    def render(self) -> str:
        """Alias for :meth:`to_prompt`."""
        return self.to_prompt()

    as_prompt = render

    def to_prompt_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible prompt payload."""
        return {
            "name": self.name,
            "description": self.description,
            "entries": {entry.key: entry.text for entry in self.entries},
            "prompt_suffix": self.prompt_suffix,
            "suffixes": list(self.suffixes()),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this bible."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "entries": [entry.to_dict() for entry in self.entries],
            "prompt_suffix": self.prompt_suffix,
        }

    def __getitem__(self, value: Any) -> StyleBibleEntry:
        if isinstance(value, str):
            return self.require(value)
        return cast(StyleBibleEntry, self.entries[value])

    def __contains__(self, value: object) -> bool:
        return self.get(value) is not None

    def __iter__(self) -> Iterator[StyleBibleEntry]:  # type: ignore[override]
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


def _name_variants(name: str | Path) -> tuple[str, ...]:
    text = str(name).strip()
    if not text:
        return ()
    suffix = Path(text).suffix.lower()
    if suffix in STYLE_BIBLE_EXTENSIONS:
        return (text,)
    return tuple(f"{text}{extension}" for extension in STYLE_BIBLE_EXTENSIONS)


def _available_names(directory: Path) -> tuple[str, ...]:
    if not directory.is_dir():
        return ()
    names = {
        path.stem
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in STYLE_BIBLE_EXTENSIONS
    }
    return tuple(sorted(names))


def _normalise_source(source: str | None) -> str:
    value = (source or "auto").strip().lower()
    aliases = {
        "auto": "auto",
        "any": "auto",
        "user": "user",
        "default": "default",
        "packaged": "default",
        "package": "default",
    }
    return aliases.get(value, value)


class StyleBibleLibrary:
    """Style bible files in the user config directory with packaged fallbacks."""

    def __init__(
        self,
        *,
        config_dir: str | Path | None = None,
        defaults_dir: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        if config_dir is None:
            self.config_dir = paths.config_dir(environ)
        else:
            self.config_dir = Path(config_dir).expanduser().resolve()
        if defaults_dir is None:
            self.defaults_dir = paths.package_defaults_dir()
        else:
            self.defaults_dir = Path(defaults_dir).expanduser().resolve()
        self.user_dir = self.config_dir / STYLE_BIBLE_DIR
        self.packaged_dir = self.defaults_dir / STYLE_BIBLE_DIR

    def _user_path(self, relative: str) -> Path:
        return self.user_dir / relative

    def _default_path(self, relative: str) -> Path:
        return self.packaged_dir / relative

    def _direct_path(self, name: str | Path) -> Path | None:
        candidate = Path(name).expanduser()
        if candidate.is_absolute():
            return candidate
        if candidate.parent != Path("."):
            return self.config_dir / candidate
        return None

    def _paths(self, name: str | Path) -> tuple[Path | None, Path | None]:
        direct = self._direct_path(name)
        if direct is not None:
            if direct.is_file():
                return direct, None
            return None, None
        user = next(
            (
                path
                for variant in _name_variants(name)
                if (path := self._user_path(variant)).is_file()
            ),
            None,
        )
        packaged = next(
            (
                path
                for variant in _name_variants(name)
                if (path := self._default_path(variant)).is_file()
            ),
            None,
        )
        return user, packaged

    def names(self) -> tuple[str, ...]:
        """Return every bible name visible in the user and packaged directories."""
        return tuple(
            sorted(
                set(_available_names(self.user_dir))
                | set(_available_names(self.packaged_dir))
            )
        )

    def resolve(
        self,
        name: str | Path = DEFAULT_STYLE_BIBLE_NAME,
        source: str | None = None,
    ) -> Path:
        """Return the file selected by *name* and *source*.

        ``auto`` prefers the user copy and falls back to the packaged default.
        ``user`` and ``default`` require that copy. A missing bible raises
        :class:`StyleBibleNotFoundError` naming every directory searched.
        """
        requested = _normalise_source(source)
        if requested not in {"auto", "user", "default"}:
            raise StyleBibleError(
                f"unknown style bible source: {source!r}",
                details={"source": source},
            )
        direct = self._direct_path(name)
        user, packaged = self._paths(name)
        selected = {
            "auto": user or packaged,
            "user": user,
            "default": packaged,
        }[requested]
        if selected is not None:
            return selected

        searched: tuple[Path, ...] = (self.user_dir, self.packaged_dir)
        if direct is not None:
            searched = (direct,)
        available = self.names()
        hint = f"; available: {', '.join(available)}" if available else ""
        versions = [
            version
            for version, path in (("user", user), ("default", packaged))
            if path is not None
        ]
        if versions and requested != "auto":
            raise StyleBibleNotFoundError(
                f"style bible {name!s} has no {requested} version "
                f"(available: {', '.join(versions)})",
                details={
                    "style_bible": str(name),
                    "source": requested,
                    "available": versions,
                },
            )
        raise StyleBibleNotFoundError(
            f"style bible not found: {name!s} "
            f"(searched {', '.join(map(str, searched))}{hint})",
            details={
                "name": str(name),
                "style_bible": str(name),
                "source": requested,
                "searched": [str(path) for path in searched],
                "available": list(available),
            },
        )

    def load(
        self,
        name: str | Path = DEFAULT_STYLE_BIBLE_NAME,
        source: str | None = None,
    ) -> StyleBible:
        """Load the style bible selected by *name* and *source*."""
        return StyleBible.from_file(self.resolve(name, source))

    def __contains__(self, name: object) -> bool:
        try:
            self.resolve(str(name))
        except StyleBibleError:
            return False
        return True

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self.names())


def resolve_style_bible_path(
    name: str | Path = DEFAULT_STYLE_BIBLE_NAME,
    *,
    source: str | None = None,
    config_dir: str | Path | None = None,
    defaults_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return the user or packaged path selected for *name*."""
    library = StyleBibleLibrary(
        config_dir=config_dir,
        defaults_dir=defaults_dir,
        environ=environ,
    )
    return library.resolve(name, source)


def default_style_bible_path(
    name: str | Path = DEFAULT_STYLE_BIBLE_NAME,
    *,
    defaults_dir: str | Path | None = None,
) -> Path:
    """Return the packaged path for *name*."""
    directory = (
        paths.package_defaults_dir()
        if defaults_dir is None
        else Path(defaults_dir).expanduser().resolve()
    )
    variants = _name_variants(name)
    for variant in variants:
        candidate = directory / STYLE_BIBLE_DIR / variant
        if candidate.is_file():
            return candidate
    return (
        directory
        / STYLE_BIBLE_DIR
        / (variants[0] if variants else DEFAULT_STYLE_BIBLE_FILENAME)
    )


def load_style_bible(
    source: str | Path | Mapping[str, Any] | Iterable[Any] | StyleBible | None = None,
    *,
    name: str | Path | None = None,
    version: str | None = None,
    config_dir: str | Path | None = None,
    defaults_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> StyleBible:
    """Load a style bible from a name, file, mapping, list or existing bible.

    With no source and no name the packaged ``default`` bible is loaded. A
    string is treated as a name in the user and packaged ``styles`` directories
    unless it is an absolute path or contains a directory component.
    """
    if isinstance(source, StyleBible):
        return source
    library = StyleBibleLibrary(
        config_dir=config_dir,
        defaults_dir=defaults_dir,
        environ=environ,
    )
    if source is None:
        return library.load(name or DEFAULT_STYLE_BIBLE_NAME, version)
    if isinstance(source, (str, Path)):
        return library.load(source, version)
    if isinstance(source, Mapping) or (
        isinstance(source, Iterable) and not isinstance(source, (bytes, bytearray))
    ):
        return StyleBible.from_mapping(source, name=None if name is None else str(name))
    raise StyleBibleInvalidError(
        "a style bible must be a name, path, mapping, list or StyleBible",
        details={"value": repr(source)},
    )


style_bible = load_style_bible
build_style_bible = load_style_bible


def load_default_style_bible(name: str = DEFAULT_STYLE_BIBLE_NAME) -> StyleBible:
    """Load a style bible shipped with this release."""
    return StyleBible.from_default(name)


default_style_bible = load_default_style_bible


def style_bible_names(
    *,
    config_dir: str | Path | None = None,
    defaults_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return every style bible name visible to the loader."""
    return StyleBibleLibrary(
        config_dir=config_dir,
        defaults_dir=defaults_dir,
        environ=environ,
    ).names()


style_bible_path = resolve_style_bible_path
package_style_bible_path = default_style_bible_path
read_style_bible = load_style_bible
load_style_bible_file = StyleBible.from_file

StyleBibleSet = StyleBibleLibrary

DEFAULT_STYLE_BIBLE = load_default_style_bible()

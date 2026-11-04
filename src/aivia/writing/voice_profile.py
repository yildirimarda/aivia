"""Per profile writing voice guidance loaded from user files or packaged data.

A voice profile records how a content profile should sound before a script is
written: its persona, tone, audience, point of view, vocabulary and the phrases
it must avoid. It is data, not code, so a release ships an empty template while
a user can shadow it with a file of their own.

User files live in ``<config>/voices`` and packaged files in
``<package>/data/defaults/voices``. A name without an extension is resolved
against both directories, preferring the user's copy, and every supported
extension is tried in turn. An empty template is valid: it means the writer has
no extra guidance rather than that the profile is missing. An explicitly
requested profile that cannot be found fails with a message naming every
directory searched.
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
from aivia.core.config.profile import ContentProfileConfig
from aivia.core.errors import ConfigError
from aivia.writing.style_bible import (
    StyleBibleEntries,
    StyleBibleEntry,
    _coerce_entries,
)

__all__ = [
    "DEFAULT_VOICE_PROFILE",
    "DEFAULT_VOICE_PROFILE_FILENAME",
    "DEFAULT_VOICE_PROFILE_NAME",
    "PACKAGED_VOICE_PROFILE_DIR",
    "VOICE_PROFILE_DIR",
    "VOICE_PROFILE_DIRNAME",
    "VOICE_PROFILE_EXTENSIONS",
    "VOICE_PROFILE_FILENAME",
    "VOICE_PROFILE_TEMPLATE",
    "VOICE_PROFILE_TEMPLATE_FILENAME",
    "VOICE_PROFILE_TEMPLATE_NAME",
    "VoiceProfile",
    "VoiceProfileEntries",
    "VoiceProfileEntry",
    "VoiceProfileError",
    "VoiceProfileInvalidError",
    "VoiceProfileLibrary",
    "VoiceProfileNotFoundError",
    "VoiceProfileSet",
    "build_voice_profile",
    "default_voice_profile",
    "default_voice_profile_path",
    "load_default_voice_profile",
    "load_voice_profile",
    "load_voice_profile_file",
    "load_voice_profile_for_profile",
    "package_voice_profile_path",
    "read_voice_profile",
    "resolve_voice_profile_path",
    "voice_profile",
    "voice_profile_names",
    "voice_profile_path",
]

VOICE_PROFILE_DIR = "voices"
VOICE_PROFILE_DIRNAME = VOICE_PROFILE_DIR
PACKAGED_VOICE_PROFILE_DIR = VOICE_PROFILE_DIR
VOICE_PROFILE_EXTENSIONS = (".yaml", ".yml", ".json", ".md", ".markdown", ".txt")
DEFAULT_VOICE_PROFILE_NAME = "default"
DEFAULT_VOICE_PROFILE_FILENAME = f"{DEFAULT_VOICE_PROFILE_NAME}.yaml"
VOICE_PROFILE_FILENAME = DEFAULT_VOICE_PROFILE_FILENAME
VOICE_PROFILE_TEMPLATE_NAME = "template"
VOICE_PROFILE_TEMPLATE_FILENAME = f"{VOICE_PROFILE_TEMPLATE_NAME}.yaml"

_ALTERNATE_DIRS = ("voice_profiles",)

_NAME_KEYS = ("name", "id", "title", "key", "slug")
_DESCRIPTION_KEYS = ("description", "notes", "summary")
_SUFFIX_KEYS = ("prompt_suffix", "suffix", "prompt_fragment")
_ENTRY_LIST_KEYS = ("entries", "sections", "items")
_RESERVED_KEYS = frozenset(
    {
        "schema_version",
        "version",
        *_NAME_KEYS,
        *_DESCRIPTION_KEYS,
        *_SUFFIX_KEYS,
        *_ENTRY_LIST_KEYS,
    }
)

_WHITESPACE = re.compile(r"\s+", re.UNICODE)


class VoiceProfileError(ConfigError):
    """A voice profile is missing, unreadable or invalid."""

    code = "voice_profile"


class VoiceProfileNotFoundError(VoiceProfileError):
    """No user or packaged voice profile matches the requested name."""

    code = "voice_profile_not_found"


class VoiceProfileInvalidError(VoiceProfileError):
    """A voice profile file exists but its contents are not usable."""

    code = "voice_profile_invalid"


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return _WHITESPACE.sub(" ", str(value)).strip()


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _first(data: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _text_sequence(value: Any, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        parts = text.splitlines() if "\n" in text else [text]
        return tuple(part.strip() for part in parts if part.strip())
    if isinstance(value, (bytes, bytearray, Mapping)):
        raise ValueError(f"{field} must be a sequence of strings")
    if isinstance(value, Iterable):
        return tuple(str(item).strip() for item in value if str(item).strip())
    raise ValueError(f"{field} must be a sequence of strings")


def _coerce_text_sequence(value: Any, info: ValidationInfo) -> tuple[str, ...]:
    return _text_sequence(value, info.field_name or "value")


class VoiceProfile(BaseModel):
    """A named set of writing voice guidance plus its source file."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
        populate_by_name=True,
    )

    name: str = Field(
        default="",
        validation_alias=AliasChoices(*_NAME_KEYS),
    )
    description: str = Field(
        default="",
        validation_alias=AliasChoices(*_DESCRIPTION_KEYS),
    )
    language: str = Field(
        default="",
        validation_alias=AliasChoices("language", "lang", "locale"),
    )
    persona: str = Field(
        default="",
        validation_alias=AliasChoices("persona", "character", "identity"),
    )
    tone: str = Field(
        default="",
        validation_alias=AliasChoices("tone", "voice_tone", "tone_of_voice"),
    )
    audience: str = Field(
        default="",
        validation_alias=AliasChoices("audience", "reader", "readers"),
    )
    point_of_view: str = Field(
        default="",
        validation_alias=AliasChoices(
            "point_of_view", "pov", "perspective", "viewpoint"
        ),
    )
    tense: str = ""
    formality: str = Field(
        default="",
        validation_alias=AliasChoices("formality", "register"),
    )
    style: str = Field(
        default="",
        validation_alias=AliasChoices("style", "writing_style"),
    )
    voice: str = Field(
        default="",
        validation_alias=AliasChoices("voice", "voice_description"),
    )
    pacing: str = Field(
        default="",
        validation_alias=AliasChoices("pacing", "pace", "rhythm"),
    )
    sentence_style: str = Field(
        default="",
        validation_alias=AliasChoices("sentence_style", "sentences"),
    )
    vocabulary: tuple[str, ...] = Field(
        default=(),
        validation_alias=AliasChoices("vocabulary", "word_choice", "preferred_words"),
    )
    forbidden_words: tuple[str, ...] = Field(
        default=(),
        validation_alias=AliasChoices("forbidden_words", "banned_words", "avoid_words"),
    )
    forbidden_phrases: tuple[str, ...] = Field(
        default=(),
        validation_alias=AliasChoices(
            "forbidden_phrases", "banned_phrases", "avoid_phrases"
        ),
    )
    signature_phrases: tuple[str, ...] = Field(
        default=(),
        validation_alias=AliasChoices("signature_phrases", "catchphrases", "phrases"),
    )
    avoid: tuple[str, ...] = Field(
        default=(),
        validation_alias=AliasChoices("avoid", "avoid_list", "dont", "donts"),
    )
    rules: tuple[str, ...] = Field(
        default=(),
        validation_alias=AliasChoices("rules", "guidelines", "principles"),
    )
    examples: tuple[str, ...] = ()
    instructions: str = Field(
        default="",
        validation_alias=AliasChoices(
            "instructions",
            "instruction",
            "guidance",
            "text",
            "content",
            "body",
        ),
    )
    prompt_suffix: str = Field(
        default="",
        validation_alias=AliasChoices(*_SUFFIX_KEYS),
    )
    entries: Annotated[StyleBibleEntries, BeforeValidator(_coerce_entries)] = Field(
        default_factory=StyleBibleEntries
    )
    schema_version: int = Field(
        default=1,
        validation_alias=AliasChoices("schema_version", "version"),
    )
    source: Path | None = Field(default=None, exclude=True)

    @field_validator(
        "name",
        "description",
        "language",
        "persona",
        "tone",
        "audience",
        "point_of_view",
        "tense",
        "formality",
        "style",
        "voice",
        "pacing",
        "sentence_style",
        "instructions",
        "prompt_suffix",
        mode="before",
    )
    @classmethod
    def _clean_text(cls, value: Any, info: ValidationInfo) -> str:
        if info.field_name in {"instructions", "prompt_suffix"}:
            return _text(value)
        return _clean(value)

    @field_validator(
        "vocabulary",
        "forbidden_words",
        "forbidden_phrases",
        "signature_phrases",
        "avoid",
        "rules",
        "examples",
        mode="before",
    )
    @classmethod
    def _clean_sequence(cls, value: Any, info: ValidationInfo) -> tuple[str, ...]:
        return _coerce_text_sequence(value, info)

    @model_validator(mode="after")
    def _fill_name(self) -> VoiceProfile:
        if not self.name:
            self.name = DEFAULT_VOICE_PROFILE_NAME
        return self

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        name: str | None = None,
        source: str | Path | None = None,
    ) -> VoiceProfile:
        """Build a voice profile from a mapping, list, text or existing profile."""
        if isinstance(value, VoiceProfile):
            if source is not None:
                value = value.model_copy(update={"source": Path(source)})
            return value
        if value is None:
            data: dict[str, Any] = {}
        elif isinstance(value, Mapping):
            data = dict(value)
        elif isinstance(value, (str, bytes, bytearray)):
            data = {"instructions": _text(value)}
        elif isinstance(value, Iterable):
            data = {"rules": list(value)}
        else:
            raise VoiceProfileInvalidError(
                "a voice profile must be a mapping, a list or text",
                details={"value": repr(value)},
            )

        resolved_name = _clean(_first(data, _NAME_KEYS) or name or "")
        entries_value = _first(data, _ENTRY_LIST_KEYS)
        if entries_value is not None:
            data["entries"] = entries_value
        if resolved_name:
            data["name"] = resolved_name
        profile = cls.model_validate(data)
        if source is not None:
            profile.source = Path(source).expanduser()
        return profile

    @classmethod
    def from_file(cls, path: str | Path) -> VoiceProfile:
        """Load a YAML, JSON, Markdown or text voice profile from *path*."""
        manifest = Path(path).expanduser()
        if not manifest.is_file():
            raise VoiceProfileNotFoundError(
                f"voice profile not found: {manifest}",
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
                raise VoiceProfileInvalidError(
                    f"unsupported voice profile extension: {suffix}",
                    details={"path": str(manifest)},
                )
        except (yaml.YAMLError, json.JSONDecodeError) as exc:
            raise VoiceProfileInvalidError(
                f"invalid voice profile: {manifest}",
                details={"path": str(manifest), "error": str(exc)},
            ) from exc
        return cls.from_mapping(payload, name=manifest.stem, source=manifest)

    @classmethod
    def from_default(cls, name: str = DEFAULT_VOICE_PROFILE_NAME) -> VoiceProfile:
        """Load a voice profile shipped as package data."""
        return cls.from_file(default_voice_profile_path(name))

    @classmethod
    def from_profile(
        cls,
        profile: ContentProfileConfig | Mapping[str, Any] | str | None = None,
        *,
        config_dir: str | Path | None = None,
        defaults_dir: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> VoiceProfile:
        """Load the voice profile selected by a content profile."""
        return load_voice_profile_for_profile(
            profile,
            config_dir=config_dir,
            defaults_dir=defaults_dir,
            environ=environ,
        )

    def get(self, value: Any) -> StyleBibleEntry | None:
        """Return an entry by key, title or alias."""
        return self.entries.get(value)

    def require(self, value: Any) -> StyleBibleEntry:
        """Return an entry or raise :class:`VoiceProfileInvalidError`."""
        entry = self.get(value)
        if entry is None:
            raise VoiceProfileInvalidError(
                f"unknown voice profile entry: {value!r}",
                details={"entry": repr(value), "known": list(self.entries.keys())},
            )
        return entry

    entry = require

    @property
    def by_key(self) -> dict[str, StyleBibleEntry]:
        """Return a key index of every entry."""
        return self.entries.by_key

    @property
    def suffixes(self) -> tuple[str, ...]:
        """Return every non-empty entry prompt suffix, then the profile suffix."""
        values = [entry.prompt_suffix for entry in self.entries if entry.prompt_suffix]
        if self.prompt_suffix:
            values.append(self.prompt_suffix)
        return tuple(dict.fromkeys(values))

    @property
    def suffix(self) -> str:
        """Return every prompt suffix joined into one string."""
        return " ".join(self.suffixes)

    @property
    def text(self) -> str:
        """Return the whole profile as readable prompt text."""
        return self.to_prompt()

    @property
    def body(self) -> str:
        """Alias for :attr:`text`."""
        return self.text

    @property
    def is_empty(self) -> bool:
        """Whether the profile carries no guidance at all."""
        return not self.to_prompt()

    @property
    def guidance(self) -> str:
        """Alias for :attr:`instructions`."""
        return self.instructions

    @property
    def banned_words(self) -> tuple[str, ...]:
        """Alias for :attr:`forbidden_words`."""
        return self.forbidden_words

    @property
    def banned_phrases(self) -> tuple[str, ...]:
        """Alias for :attr:`forbidden_phrases`."""
        return self.forbidden_phrases

    @property
    def catchphrases(self) -> tuple[str, ...]:
        """Alias for :attr:`signature_phrases`."""
        return self.signature_phrases

    @property
    def pov(self) -> str:
        """Alias for :attr:`point_of_view`."""
        return self.point_of_view

    @property
    def pace(self) -> str:
        """Alias for :attr:`pacing`."""
        return self.pacing

    @property
    def rhythm(self) -> str:
        """Alias for :attr:`pacing`."""
        return self.pacing

    @property
    def writing_style(self) -> str:
        """Alias for :attr:`style`."""
        return self.style

    @property
    def word_choice(self) -> tuple[str, ...]:
        """Alias for :attr:`vocabulary`."""
        return self.vocabulary

    def _scalar_lines(self) -> list[str]:
        values = (
            ("Language", self.language),
            ("Persona", self.persona),
            ("Tone", self.tone),
            ("Audience", self.audience),
            ("Point of view", self.point_of_view),
            ("Tense", self.tense),
            ("Formality", self.formality),
            ("Style", self.style),
            ("Voice", self.voice),
            ("Pacing", self.pacing),
            ("Sentences", self.sentence_style),
        )
        return [f"{label}: {value}" for label, value in values if value]

    def _sequence_lines(self) -> list[str]:
        values = (
            ("Vocabulary", self.vocabulary),
            ("Forbidden words", self.forbidden_words),
            ("Forbidden phrases", self.forbidden_phrases),
            ("Signature phrases", self.signature_phrases),
            ("Avoid", self.avoid),
            ("Rules", self.rules),
            ("Examples", self.examples),
        )
        return [f"{label}: {', '.join(items)}" for label, items in values if items]

    def prompt_lines(self) -> tuple[str, ...]:
        """Return one readable line per guidance item for a model prompt."""
        return tuple(f"- {entry.title}: {entry.text}" for entry in self.entries)

    def to_prompt(self) -> str:
        """Render the profile as deterministic prompt text."""
        lines: list[str] = []
        if self.description:
            lines.append(self.description)
        if self.instructions:
            lines.append(self.instructions)
        lines.extend(self._scalar_lines())
        lines.extend(self._sequence_lines())
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
            "instructions": self.instructions,
            "fields": {
                "language": self.language,
                "persona": self.persona,
                "tone": self.tone,
                "audience": self.audience,
                "point_of_view": self.point_of_view,
                "style": self.style,
                "voice": self.voice,
                "pacing": self.pacing,
            },
            "entries": {entry.key: entry.text for entry in self.entries},
            "prompt_suffix": self.prompt_suffix,
            "suffixes": list(self.suffixes),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this profile."""
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "language": self.language,
            "persona": self.persona,
            "tone": self.tone,
            "audience": self.audience,
            "point_of_view": self.point_of_view,
            "tense": self.tense,
            "formality": self.formality,
            "style": self.style,
            "voice": self.voice,
            "pacing": self.pacing,
            "sentence_style": self.sentence_style,
            "vocabulary": list(self.vocabulary),
            "forbidden_words": list(self.forbidden_words),
            "forbidden_phrases": list(self.forbidden_phrases),
            "signature_phrases": list(self.signature_phrases),
            "avoid": list(self.avoid),
            "rules": list(self.rules),
            "examples": list(self.examples),
            "instructions": self.instructions,
            "prompt_suffix": self.prompt_suffix,
            "entries": [entry.to_dict() for entry in self.entries],
        }
        for key, value in (self.model_extra or {}).items():
            if key not in payload and key not in _RESERVED_KEYS:
                payload[key] = value
        return payload

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


VoiceProfileEntry = StyleBibleEntry
VoiceProfileEntries = StyleBibleEntries


def _name_variants(name: str | Path) -> tuple[str, ...]:
    text = str(name).strip()
    if not text:
        return ()
    suffix = Path(text).suffix.lower()
    if suffix in VOICE_PROFILE_EXTENSIONS:
        return (text,)
    return tuple(f"{text}{extension}" for extension in VOICE_PROFILE_EXTENSIONS)


def _available_names(directory: Path) -> tuple[str, ...]:
    if not directory.is_dir():
        return ()
    names = {
        path.stem
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VOICE_PROFILE_EXTENSIONS
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
        "template": "default",
        "empty": "default",
    }
    return aliases.get(value, value)


def _profile_voice_name(
    profile: ContentProfileConfig | Mapping[str, Any] | str | None,
) -> str:
    if profile is None:
        return DEFAULT_VOICE_PROFILE_NAME
    if isinstance(profile, ContentProfileConfig):
        return profile.voice.strip() or DEFAULT_VOICE_PROFILE_NAME
    if isinstance(profile, str):
        return profile.strip() or DEFAULT_VOICE_PROFILE_NAME
    if isinstance(profile, Mapping):
        value = profile.get("voice", profile.get("voice_profile", ""))
        text = str(value).strip() if value is not None else ""
        return text or DEFAULT_VOICE_PROFILE_NAME
    raise VoiceProfileInvalidError(
        "a profile must be a ContentProfileConfig, a mapping or a name",
        details={"value": repr(profile)},
    )


class VoiceProfileLibrary:
    """Voice profile files in the user config directory with packaged fallbacks."""

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
        self.user_dir = self.config_dir / VOICE_PROFILE_DIR
        self.packaged_dir = self.defaults_dir / VOICE_PROFILE_DIR

    def _user_dirs(self) -> tuple[Path, ...]:
        return (self.user_dir, *(self.config_dir / name for name in _ALTERNATE_DIRS))

    def _packaged_dirs(self) -> tuple[Path, ...]:
        return (
            self.packaged_dir,
            *(self.defaults_dir / name for name in _ALTERNATE_DIRS),
        )

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
                for directory in self._user_dirs()
                for variant in _name_variants(name)
                if (path := directory / variant).is_file()
            ),
            None,
        )
        packaged = next(
            (
                path
                for directory in self._packaged_dirs()
                for variant in _name_variants(name)
                if (path := directory / variant).is_file()
            ),
            None,
        )
        return user, packaged

    def _searched_dirs(self) -> tuple[Path, ...]:
        directories: list[Path] = [self.user_dir, self.packaged_dir]
        for directory in (*self._user_dirs()[1:], *self._packaged_dirs()[1:]):
            if directory.is_dir():
                directories.append(directory)
        return tuple(directories)

    def names(self) -> tuple[str, ...]:
        """Return every profile name visible in the user and packaged directories."""
        names: set[str] = set()
        for directory in (*self._user_dirs(), *self._packaged_dirs()):
            names.update(_available_names(directory))
        return tuple(sorted(names))

    def resolve(
        self,
        name: str | Path = DEFAULT_VOICE_PROFILE_NAME,
        source: str | None = None,
    ) -> Path:
        """Return the file selected by *name* and *source*.

        ``auto`` prefers the user copy and falls back to the packaged default.
        ``user`` and ``default`` require that copy. A missing profile raises
        :class:`VoiceProfileNotFoundError` naming every directory searched.
        """
        requested = _normalise_source(source)
        if requested not in {"auto", "user", "default"}:
            raise VoiceProfileError(
                f"unknown voice profile source: {source!r}",
                details={"source": source},
            )
        if isinstance(name, (ContentProfileConfig, Mapping)):
            name = _profile_voice_name(name)
        elif name is None or (isinstance(name, str) and not name.strip()):
            name = DEFAULT_VOICE_PROFILE_NAME
        direct = self._direct_path(name)
        user, packaged = self._paths(name)
        selected = {
            "auto": user or packaged,
            "user": user,
            "default": packaged,
        }[requested]
        if selected is not None:
            return selected

        searched: tuple[Path, ...] = self._searched_dirs()
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
            raise VoiceProfileNotFoundError(
                f"voice profile {name!s} has no {requested} version "
                f"(available: {', '.join(versions)})",
                details={
                    "voice_profile": str(name),
                    "source": requested,
                    "available": versions,
                },
            )
        raise VoiceProfileNotFoundError(
            f"voice profile not found: {name!s} "
            f"(searched {', '.join(map(str, searched))}{hint})",
            details={
                "name": str(name),
                "voice_profile": str(name),
                "source": requested,
                "searched": [str(path) for path in searched],
                "available": list(available),
            },
        )

    def load(
        self,
        name: str | Path = DEFAULT_VOICE_PROFILE_NAME,
        source: str | None = None,
    ) -> VoiceProfile:
        """Load the voice profile selected by *name* and *source*."""
        if isinstance(name, (ContentProfileConfig, Mapping)):
            return self.for_profile(name)
        return VoiceProfile.from_file(self.resolve(name, source))

    def for_profile(
        self,
        profile: ContentProfileConfig | Mapping[str, Any] | str | None,
    ) -> VoiceProfile:
        """Load the voice profile named by a content profile."""
        return self.load(_profile_voice_name(profile))

    def __contains__(self, name: object) -> bool:
        try:
            self.resolve(str(name))
        except VoiceProfileError:
            return False
        return True

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self.names())


VoiceProfileSet = VoiceProfileLibrary


def resolve_voice_profile_path(
    name: str | Path = DEFAULT_VOICE_PROFILE_NAME,
    *,
    source: str | None = None,
    config_dir: str | Path | None = None,
    defaults_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return the user or packaged path selected for *name*."""
    library = VoiceProfileLibrary(
        config_dir=config_dir,
        defaults_dir=defaults_dir,
        environ=environ,
    )
    return library.resolve(name, source)


def default_voice_profile_path(
    name: str | Path = DEFAULT_VOICE_PROFILE_NAME,
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
    for base in (
        directory / VOICE_PROFILE_DIR,
        *(directory / item for item in _ALTERNATE_DIRS),
    ):
        for variant in variants:
            candidate = base / variant
            if candidate.is_file():
                return candidate
    return (
        directory
        / VOICE_PROFILE_DIR
        / (variants[0] if variants else DEFAULT_VOICE_PROFILE_FILENAME)
    )


def load_voice_profile(
    source: (
        str
        | Path
        | Mapping[str, Any]
        | Iterable[Any]
        | VoiceProfile
        | ContentProfileConfig
        | None
    ) = None,
    *,
    name: str | Path | None = None,
    profile: ContentProfileConfig | Mapping[str, Any] | str | None = None,
    version: str | None = None,
    config_dir: str | Path | None = None,
    defaults_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> VoiceProfile:
    """Load a voice profile from a name, file, mapping, list or existing profile.

    With no source and no name the packaged ``default`` profile is loaded. A
    string is treated as a name in the user and packaged ``voices`` directories
    unless it is an absolute path or contains a directory component.
    """
    if isinstance(source, VoiceProfile):
        return source
    library = VoiceProfileLibrary(
        config_dir=config_dir,
        defaults_dir=defaults_dir,
        environ=environ,
    )
    if profile is not None:
        return library.for_profile(profile)
    if isinstance(source, ContentProfileConfig):
        return library.for_profile(source)
    if source is None:
        return library.load(name or DEFAULT_VOICE_PROFILE_NAME, version)
    if isinstance(source, (str, Path)):
        return library.load(source, version)
    if isinstance(source, Mapping) or (
        isinstance(source, Iterable) and not isinstance(source, (bytes, bytearray))
    ):
        return VoiceProfile.from_mapping(
            source, name=None if name is None else str(name)
        )
    raise VoiceProfileInvalidError(
        "a voice profile must be a name, path, mapping, list or VoiceProfile",
        details={"value": repr(source)},
    )


def load_voice_profile_for_profile(
    profile: ContentProfileConfig | Mapping[str, Any] | str | None = None,
    *,
    config_dir: str | Path | None = None,
    defaults_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> VoiceProfile:
    """Load the voice profile selected by *profile* before writing."""
    library = VoiceProfileLibrary(
        config_dir=config_dir,
        defaults_dir=defaults_dir,
        environ=environ,
    )
    return library.for_profile(profile)


voice_profile = load_voice_profile
build_voice_profile = load_voice_profile


def load_default_voice_profile(
    name: str = DEFAULT_VOICE_PROFILE_NAME,
) -> VoiceProfile:
    """Load a voice profile shipped with this release."""
    return VoiceProfile.from_default(name)


default_voice_profile = load_default_voice_profile


def voice_profile_names(
    *,
    config_dir: str | Path | None = None,
    defaults_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return every voice profile name visible to the loader."""
    return VoiceProfileLibrary(
        config_dir=config_dir,
        defaults_dir=defaults_dir,
        environ=environ,
    ).names()


voice_profile_path = resolve_voice_profile_path
package_voice_profile_path = default_voice_profile_path
read_voice_profile = load_voice_profile
load_voice_profile_file = VoiceProfile.from_file

DEFAULT_VOICE_PROFILE = load_default_voice_profile()
VOICE_PROFILE_TEMPLATE = DEFAULT_VOICE_PROFILE

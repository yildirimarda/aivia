"""Per language brand glossaries loaded from user files or packaged data.

A brand glossary records the terms a brand wants used, their preferred spelling
or translation, aliases, forms to avoid and an optional definition. It is data,
not code: a release ships an empty default, a user can shadow it with a file of
their own, and the same term set can be scoped to one target language.

User files live in ``<config>/glossary`` and packaged files in
``<package>/data/defaults/glossary``. A name without an extension is resolved
against both directories, preferring the user's copy, and every supported
extension is tried in turn. Loading a glossary for a language prefers a file
named after that language, then falls back to the named or default file and
filters its terms by language.

Applying a glossary rewrites the terms in scripts, captions and media prompts.
The rewrite is deterministic and always returns the violations it found in the
source text, so a caller can report what changed instead of silently editing
brand language.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import yaml
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from aivia.core import paths
from aivia.core.errors import ConfigError
from aivia.core.models.script import Beat, MediaPrompt, Script

if TYPE_CHECKING:
    from aivia.writing.captions import CaptionSet

__all__ = [
    "DEFAULT_GLOSSARY",
    "DEFAULT_GLOSSARY_FILENAME",
    "DEFAULT_GLOSSARY_NAME",
    "GLOSSARY_DIR",
    "GLOSSARY_DIRNAME",
    "GLOSSARY_EXTENSIONS",
    "GLOSSARY_FILENAME",
    "PACKAGED_GLOSSARY_DIR",
    "BrandGlossary",
    "Glossary",
    "GlossaryApplication",
    "GlossaryEntry",
    "GlossaryError",
    "GlossaryInvalidError",
    "GlossaryLibrary",
    "GlossaryNotFoundError",
    "GlossaryReport",
    "GlossarySet",
    "GlossaryTerm",
    "GlossaryTerms",
    "GlossaryViolation",
    "GlossaryWarning",
    "apply_glossary",
    "apply_glossary_to_captions",
    "apply_glossary_to_media_prompts",
    "apply_glossary_to_prompts",
    "apply_glossary_to_script",
    "build_glossary",
    "default_glossary",
    "default_glossary_path",
    "find_glossary_violations",
    "glossary",
    "glossary_names",
    "glossary_path",
    "glossary_terms",
    "glossary_violations",
    "load_default_glossary",
    "load_glossary",
    "load_glossary_file",
    "normalise_glossary",
    "normalize_glossary",
    "package_glossary_path",
    "read_glossary",
    "report_glossary_violations",
    "resolve_glossary_path",
    "warn_glossary_violations",
]

GLOSSARY_DIR = "glossary"
GLOSSARY_DIRNAME = GLOSSARY_DIR
PACKAGED_GLOSSARY_DIR = GLOSSARY_DIR
GLOSSARY_EXTENSIONS = (".yaml", ".yml", ".json", ".md", ".markdown", ".txt")
DEFAULT_GLOSSARY_NAME = "default"
DEFAULT_GLOSSARY_FILENAME = f"{DEFAULT_GLOSSARY_NAME}.yaml"
GLOSSARY_FILENAME = DEFAULT_GLOSSARY_FILENAME

_WHITESPACE = re.compile(r"\s+", re.UNICODE)
_LANGUAGE_FILE = re.compile(r"[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,8})?")


class GlossaryError(ConfigError):
    """A glossary is missing, unreadable or invalid."""

    code = "glossary"


class GlossaryNotFoundError(GlossaryError):
    """No user or packaged glossary matches the requested name."""

    code = "glossary_not_found"


class GlossaryInvalidError(GlossaryError):
    """A glossary file exists but its contents are not usable."""

    code = "glossary_invalid"


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return _WHITESPACE.sub(" ", str(value)).strip()


def _normalise_language(value: Any) -> str:
    return _clean(value).casefold().replace("_", "-")


def _same_language(left: str, right: str) -> bool:
    if not left or not right:
        return True
    first = _normalise_language(left)
    second = _normalise_language(right)
    return first == second or first.split("-", 1)[0] == second.split("-", 1)[0]


_TERM_KEYS = ("term", "key", "name", "source", "from")
_PREFERRED_KEYS = (
    "preferred",
    "replacement",
    "replace",
    "use",
    "instead",
    "target",
    "translation",
    "translated",
    "canonical",
    "render",
)
_DEFINITION_KEYS = ("definition", "description", "meaning", "gloss", "note", "notes")
_AVOID_TERM_KEYS = (
    "avoid",
    "forbidden",
    "banned",
    "incorrect",
    "wrong",
    "variants",
)
_ALIAS_KEYS = ("aliases", "alias", "synonyms", "also")
_LANGUAGE_KEYS = ("language", "lang", "locale")
_GLOSSARY_LIST_KEYS = (
    "terms",
    "entries",
    "glossary",
    "items",
    "definitions",
    "lexicon",
)


def _text_sequence(value: Any, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        parts = text.splitlines() if "\n" in text else text.split(",")
        return tuple(part.strip() for part in parts if part.strip())
    if isinstance(value, (bytes, bytearray, Mapping)):
        raise ValueError(f"{field} must be a sequence of strings")
    if isinstance(value, Iterable):
        return tuple(str(item).strip() for item in value if str(item).strip())
    raise ValueError(f"{field} must be a sequence of strings")


def _normalise_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).casefold()).strip("-")


def _looks_like_language_code(value: Any) -> bool:
    text = _clean(value)
    return bool(re.fullmatch(r"[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,8})?", text))


def _language_mapping(data: Mapping[str, Any]) -> dict[str, Any] | None:
    if not data or not all(isinstance(value, Mapping) for value in data.values()):
        return None
    if not all(_looks_like_language_code(key) for key in data):
        return None
    return dict(data)


def _terms_from_mapping(
    data: Mapping[str, Any],
    *,
    language: str = "",
) -> list[Any]:
    terms: list[Any] = []
    for key, value in data.items():
        if isinstance(value, Mapping):
            item = dict(value)
            item.setdefault("term", key)
            if language:
                item.setdefault("language", language)
            terms.append(item)
        elif isinstance(value, (list, tuple)):
            terms.append(
                {
                    "term": str(key),
                    "aliases": list(value),
                    "language": language,
                }
            )
        else:
            terms.append(
                {
                    "term": str(key),
                    "preferred": "" if value is None else str(value),
                    "language": language,
                }
            )
    return terms


class GlossaryTerm(BaseModel):
    """One glossary entry with a preferred form and optional definition.

    ``term`` is the spelling the writer used. ``preferred`` is the spelling,
    translation or brand form that prompts should use instead. ``avoid`` and
    ``aliases`` list additional forms that map to the same preferred form.
    A term with no preferred form is still useful when it carries a definition.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    term: str = Field(validation_alias=AliasChoices(*_TERM_KEYS))
    preferred: str = Field(default="", validation_alias=AliasChoices(*_PREFERRED_KEYS))
    definition: str = Field(
        default="",
        validation_alias=AliasChoices(*_DEFINITION_KEYS),
    )
    avoid: tuple[str, ...] = Field(
        default=(),
        validation_alias=AliasChoices(*_AVOID_TERM_KEYS),
    )
    aliases: tuple[str, ...] = Field(
        default=(),
        validation_alias=AliasChoices(*_ALIAS_KEYS),
    )
    language: str = Field(
        default="",
        validation_alias=AliasChoices(*_LANGUAGE_KEYS),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_string(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"term": value}
        return value

    @field_validator("term", mode="before")
    @classmethod
    def _validate_term(cls, value: Any) -> str:
        text = _clean(value)
        if not text:
            raise ValueError("glossary term must not be empty")
        return text

    @field_validator("preferred", "definition", "language", mode="before")
    @classmethod
    def _clean_text(cls, value: Any) -> str:
        return _clean(value)

    @field_validator("avoid", "aliases", mode="before")
    @classmethod
    def _normalise_sequence(cls, value: Any) -> tuple[str, ...]:
        return _text_sequence(value, "glossary variants")

    @property
    def key(self) -> str:
        """Alias for :attr:`term`."""
        return self.term

    @property
    def replacement(self) -> str:
        """Alias for :attr:`preferred`."""
        return self.preferred

    @property
    def target(self) -> str:
        """Alias for :attr:`preferred`."""
        return self.preferred

    @property
    def display(self) -> str:
        """The form a prompt should show."""
        return self.preferred or self.term

    @property
    def variants(self) -> tuple[str, ...]:
        """Every source form that should be rewritten to :attr:`display`."""
        if not self.preferred:
            return ()
        values = [self.term, *self.aliases, *self.avoid]
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            text = _clean(value)
            if not text or text == self.preferred:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            result.append(text)
        result.sort(key=len, reverse=True)
        return tuple(result)

    def matches(self, value: Any) -> bool:
        """Whether *value* names this term by term, preferred or alias."""
        text = _clean(value).casefold()
        if not text:
            return False
        candidates = (self.term, self.preferred, *self.aliases, *self.avoid)
        return text in {_clean(candidate).casefold() for candidate in candidates}

    def applies_to(self, language: str) -> bool:
        """Whether this term applies to *language*."""
        return _same_language(self.language, language)

    def apply(self, text: str) -> str:
        """Rewrite every source form in *text* to the preferred form."""
        result = text
        for variant in self.variants:
            pattern = re.compile(
                rf"(?<!\w){re.escape(variant)}(?!\w)",
                re.IGNORECASE,
            )
            result = pattern.sub(self.preferred, result)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this term."""
        return {
            "term": self.term,
            "preferred": self.preferred,
            "definition": self.definition,
            "avoid": list(self.avoid),
            "aliases": list(self.aliases),
            "language": self.language,
        }


GlossaryEntry = GlossaryTerm


class _BaseGlossary(BaseModel):
    """A named set of brand terms, optionally split by language."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    name: str = ""
    language: str = ""
    terms: list[GlossaryTerm] = Field(
        default_factory=list,
        validation_alias=AliasChoices(*_GLOSSARY_LIST_KEYS),
    )
    source: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _coerce_value(cls, value: Any) -> Any:
        if isinstance(value, _BaseGlossary):
            return value.model_dump()
        if value is None:
            return {"terms": []}
        if isinstance(value, (str, bytes, bytearray)):
            text = value.decode() if isinstance(value, (bytes, bytearray)) else value
            return {
                "terms": [
                    {"term": line.strip()} for line in text.splitlines() if line.strip()
                ]
            }
        if isinstance(value, GlossaryTerm):
            return {"terms": [value]}
        if isinstance(value, Mapping):
            data = dict(value)
            if any(key in data for key in _GLOSSARY_LIST_KEYS):
                return data
            if any(key in data for key in _TERM_KEYS):
                return {"terms": [data]}
            language_map = _language_mapping(data)
            if language_map is not None:
                terms: list[Any] = []
                for language, entries in language_map.items():
                    terms.extend(_terms_from_mapping(entries, language=language))
                return {"terms": terms}
            return {"terms": _terms_from_mapping(data)}
        if isinstance(value, Iterable):
            return {"terms": list(value)}
        raise ValueError("a glossary must be a mapping, a list or text")

    @field_validator("name", "language", mode="before")
    @classmethod
    def _clean_text(cls, value: Any) -> str:
        return _clean(value)

    @classmethod
    def from_value(
        cls,
        value: Any = None,
        *,
        language: str | None = None,
    ) -> Self:
        """Build a glossary from text, mappings, terms or an existing glossary."""
        glossary = value if isinstance(value, cls) else cls.model_validate(value)
        if language is not None:
            glossary = glossary.for_language(language)
        return glossary

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        name: str | None = None,
        source: str | Path | None = None,
        language: str | None = None,
    ) -> Self:
        """Build a glossary from a mapping, list, text or existing glossary."""
        if isinstance(value, str) and "\n" in value:
            try:
                parsed: Any = yaml.safe_load(value)
            except yaml.YAMLError:
                parsed = None
            if isinstance(parsed, (Mapping, list)):
                value = parsed
        glossary = cls.from_value(value, language=language)
        if name:
            glossary.name = name
        if source is not None:
            glossary.source = Path(source).expanduser()
        return glossary

    def for_language(self, language: str | None) -> Self:
        """Return a glossary restricted to terms that apply to *language*."""
        if not language:
            return self
        return self.model_copy(
            update={"terms": [term for term in self.terms if term.applies_to(language)]}
        )

    def apply(self, text: str, *, language: str | None = None) -> str:
        """Rewrite every matching term in *text* for *language*."""
        result = text
        for term in self.for_language(language).terms:
            result = term.apply(result)
        return result

    def matched(
        self,
        text: str,
        *,
        language: str | None = None,
    ) -> tuple[GlossaryTerm, ...]:
        """Return the terms that appear in *text*, in glossary order."""
        matched: list[GlossaryTerm] = []
        for term in self.for_language(language).terms:
            candidates = (term.term, term.preferred, *term.aliases, *term.avoid)
            if any(
                candidate
                and re.search(
                    rf"(?<!\w){re.escape(candidate)}(?!\w)",
                    text,
                    re.IGNORECASE,
                )
                for candidate in candidates
            ):
                matched.append(term)
        return tuple(matched)

    def definitions(
        self,
        text: str,
        *,
        language: str | None = None,
    ) -> tuple[str, ...]:
        """Return ``term: definition`` lines for matched terms with definitions."""
        lines: list[str] = []
        for term in self.matched(text, language=language):
            if term.definition:
                lines.append(f"{term.display}: {term.definition}")
        return tuple(dict.fromkeys(lines))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this glossary."""
        return {
            "name": self.name,
            "language": self.language,
            "terms": [term.to_dict() for term in self.terms],
        }

    def __len__(self) -> int:
        return len(self.terms)

    def __iter__(self) -> Iterator[GlossaryTerm]:  # type: ignore[override]
        return iter(self.terms)


@dataclass(frozen=True, slots=True)
class GlossaryViolation:
    """One source form that does not match the glossary's preferred form."""

    term: str
    found: str
    expected: str = ""
    language: str = ""
    path: str = ""
    message: str = ""
    code: str = "glossary"
    position: int = 0

    @property
    def variant(self) -> str:
        """Alias for :attr:`found`."""
        return self.found

    @property
    def preferred(self) -> str:
        """Alias for :attr:`expected`."""
        return self.expected

    @property
    def location(self) -> str:
        """Alias for :attr:`path`."""
        return self.path

    @property
    def field(self) -> str:
        """Alias for :attr:`path`."""
        return self.path

    def __str__(self) -> str:
        text = self.message or f"{self.found!r} should be {self.expected!r}"
        return f"{self.path}: {text}" if self.path else text

    def to_dict(self) -> dict[str, str]:
        """Return the violation as a JSON-compatible mapping."""
        return {
            "code": self.code,
            "term": self.term,
            "found": self.found,
            "expected": self.expected,
            "language": self.language,
            "path": self.path,
            "message": self.message or str(self),
            "position": str(self.position),
        }


class GlossaryReport:
    """The violations found in one text or artifact."""

    def __init__(
        self,
        violations: Iterable[GlossaryViolation] = (),
        *,
        language: str = "",
        name: str = "",
    ) -> None:
        self.violations: tuple[GlossaryViolation, ...] = tuple(violations)
        self.language = language
        self.name = name

    @property
    def passed(self) -> bool:
        """Whether no glossary violation was found."""
        return not self.violations

    @property
    def ok(self) -> bool:
        """Alias for :attr:`passed`."""
        return self.passed

    @property
    def found(self) -> tuple[str, ...]:
        """Every distinct offending form in first-seen order."""
        return tuple(dict.fromkeys(item.found for item in self.violations))

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-compatible mapping."""
        return {
            "name": self.name,
            "language": self.language,
            "passed": self.passed,
            "violations": [item.to_dict() for item in self.violations],
        }

    def __bool__(self) -> bool:
        return self.passed

    def __len__(self) -> int:
        return len(self.violations)

    def __iter__(self) -> Iterator[GlossaryViolation]:
        return iter(self.violations)


class GlossaryApplication(str):
    """A rewritten string that also reports the violations it replaced."""

    __slots__ = ("_glossary", "_language", "_violations")

    _glossary: Glossary | None
    _language: str
    _violations: tuple[GlossaryViolation, ...]

    def __new__(
        cls,
        text: str,
        *,
        glossary: Glossary | None = None,
        language: str = "",
        violations: Sequence[GlossaryViolation] = (),
    ) -> GlossaryApplication:
        result = super().__new__(cls, text)
        result._glossary = glossary
        result._language = language
        result._violations = tuple(violations)
        return result

    @property
    def text(self) -> str:
        """The rewritten text."""
        return str(self)

    @property
    def value(self) -> str:
        """Alias for :attr:`text`."""
        return str(self)

    @property
    def violations(self) -> tuple[GlossaryViolation, ...]:
        """The violations found before the rewrite."""
        return self._violations

    @property
    def glossary(self) -> Glossary | None:
        """The glossary that was applied."""
        return self._glossary

    @property
    def language(self) -> str:
        """The language the glossary was scoped to."""
        return self._language

    @property
    def changed(self) -> bool:
        """Whether the rewrite changed the text."""
        return bool(self._violations)

    @property
    def passed(self) -> bool:
        """Whether the source text already matched the glossary."""
        return not self._violations

    def to_dict(self) -> dict[str, Any]:
        """Return the application as a JSON-compatible mapping."""
        return {
            "text": str(self),
            "language": self._language,
            "changed": self.changed,
            "violations": [item.to_dict() for item in self._violations],
        }


class Glossary(_BaseGlossary):
    """A named set of brand terms with the file it was loaded from."""

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        language: str | None = None,
    ) -> Glossary:
        """Load a YAML, JSON, Markdown or text glossary from *path*."""
        manifest = Path(path).expanduser()
        if not manifest.is_file():
            raise GlossaryNotFoundError(
                f"glossary not found: {manifest}",
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
                raise GlossaryInvalidError(
                    f"unsupported glossary extension: {suffix}",
                    details={"path": str(manifest)},
                )
        except (yaml.YAMLError, json.JSONDecodeError) as exc:
            raise GlossaryInvalidError(
                f"invalid glossary: {manifest}",
                details={"path": str(manifest), "error": str(exc)},
            ) from exc
        if payload is None:
            raise GlossaryInvalidError(
                f"empty glossary: {manifest}",
                details={"path": str(manifest)},
            )
        glossary = cls.from_value(payload, language=language)
        if not glossary.name:
            glossary.name = manifest.stem
        glossary.source = manifest
        return glossary

    @classmethod
    def from_default(
        cls,
        name: str = DEFAULT_GLOSSARY_NAME,
        *,
        language: str | None = None,
    ) -> Glossary:
        """Load a glossary shipped as package data."""
        return cls.from_file(default_glossary_path(name), language=language)

    @property
    def languages(self) -> tuple[str, ...]:
        """Every language named by at least one term, in glossary order."""
        values = [term.language for term in self.terms if term.language]
        return tuple(dict.fromkeys(values))

    def terms_for(self, language: str | None = None) -> tuple[GlossaryTerm, ...]:
        """Return the terms that apply to *language*."""
        return tuple(self.for_language(language).terms)

    def violations(
        self,
        text: str,
        *,
        language: str | None = None,
        path: str = "",
    ) -> tuple[GlossaryViolation, ...]:
        """Return every source form in *text* that is not the preferred form."""
        found: list[GlossaryViolation] = []
        seen: set[tuple[str, str, int]] = set()
        for term in self.for_language(language).terms:
            expected = term.preferred or term.term
            forms = [term.term, *term.aliases, *term.avoid]
            for form in dict.fromkeys(item for item in forms if item):
                pattern = re.compile(
                    rf"(?<!\w){re.escape(form)}(?!\w)",
                    re.IGNORECASE,
                )
                for match in pattern.finditer(text):
                    if match.group(0) == expected:
                        continue
                    key = (term.term, form.casefold(), match.start())
                    if key in seen:
                        continue
                    seen.add(key)
                    found.append(
                        GlossaryViolation(
                            term=term.term,
                            found=match.group(0),
                            expected=expected,
                            language=term.language,
                            path=path,
                            message=f"{match.group(0)!r} should be {expected!r}",
                            position=match.start(),
                        )
                    )
        found.sort(key=lambda item: (item.path, item.position))
        return tuple(found)

    def report(
        self,
        text: str,
        *,
        language: str | None = None,
        path: str = "",
    ) -> GlossaryReport:
        """Return a report of the violations in *text*."""
        return GlossaryReport(
            self.violations(text, language=language, path=path),
            language=language or self.language,
            name=self.name,
        )

    def apply(self, text: str, *, language: str | None = None) -> GlossaryApplication:
        """Rewrite every matching term in *text* and report what changed."""
        active = self.for_language(language)
        violations = active.violations(text)
        result = text
        for term in active.terms:
            result = term.apply(result)
        return GlossaryApplication(
            result,
            glossary=active,
            language=language or "",
            violations=violations,
        )

    def apply_to_script(
        self,
        script: Script,
        *,
        language: str | None = None,
        include_media_prompts: bool = True,
    ) -> Script:
        """Return a copy of *script* with glossary terms rewritten."""
        return apply_glossary_to_script(
            script,
            self,
            language=language,
            include_media_prompts=include_media_prompts,
        )

    def apply_to_captions(
        self,
        captions: CaptionSet,
        *,
        language: str | None = None,
    ) -> CaptionSet:
        """Return a copy of *captions* with glossary terms rewritten."""
        return apply_glossary_to_captions(captions, self, language=language)

    def apply_to_media_prompts(
        self,
        value: Any,
        *,
        language: str | None = None,
    ) -> Any:
        """Return a copy of *value* with glossary terms in prompts rewritten."""
        return apply_glossary_to_media_prompts(value, self, language=language)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this glossary."""
        return {
            "name": self.name,
            "language": self.language,
            "terms": [term.to_dict() for term in self.terms],
        }


BrandGlossary = Glossary
GlossaryTerms = Glossary


def _name_variants(name: str | Path) -> tuple[str, ...]:
    text = str(name).strip()
    if not text:
        return ()
    suffix = Path(text).suffix.lower()
    if suffix in GLOSSARY_EXTENSIONS:
        return (text,)
    return tuple(f"{text}{extension}" for extension in GLOSSARY_EXTENSIONS)


def _available_names(directory: Path) -> tuple[str, ...]:
    if not directory.is_dir():
        return ()
    names = {
        path.stem
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in GLOSSARY_EXTENSIONS
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


def _is_default_name(name: str | Path | None) -> bool:
    if name is None:
        return True
    text = str(name).strip()
    return not text or text == DEFAULT_GLOSSARY_NAME


class GlossaryLibrary:
    """Glossary files in the user config directory with packaged fallbacks."""

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
        self.user_dir = self.config_dir / GLOSSARY_DIR
        self.packaged_dir = self.defaults_dir / GLOSSARY_DIR

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
        """Return every glossary name visible in the user and packaged dirs."""
        return tuple(
            sorted(
                set(_available_names(self.user_dir))
                | set(_available_names(self.packaged_dir))
            )
        )

    def resolve(
        self,
        name: str | Path = DEFAULT_GLOSSARY_NAME,
        source: str | None = None,
        language: str | None = None,
    ) -> Path:
        """Return the file selected by *name*, *source* and *language*."""
        requested = _normalise_source(source)
        if requested not in {"auto", "user", "default"}:
            raise GlossaryError(
                f"unknown glossary source: {source!r}",
                details={"source": source},
            )
        if _is_default_name(name) and language:
            language_name = _clean(language).replace("_", "-")
            user, packaged = self._paths(language_name)
            selected_language = {
                "auto": user or packaged,
                "user": user,
                "default": packaged,
            }[requested]
            if selected_language is not None:
                return selected_language
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
            raise GlossaryNotFoundError(
                f"glossary {name!s} has no {requested} version "
                f"(available: {', '.join(versions)})",
                details={
                    "glossary": str(name),
                    "source": requested,
                    "available": versions,
                },
            )
        raise GlossaryNotFoundError(
            f"glossary not found: {name!s} "
            f"(searched {', '.join(map(str, searched))}{hint})",
            details={
                "name": str(name),
                "glossary": str(name),
                "source": requested,
                "searched": [str(path) for path in searched],
                "available": list(available),
            },
        )

    def load(
        self,
        name: str | Path = DEFAULT_GLOSSARY_NAME,
        source: str | None = None,
        *,
        language: str | None = None,
    ) -> Glossary:
        """Load the glossary selected by *name*, *source* and *language*."""
        path = self.resolve(name, source, language)
        return Glossary.from_file(path, language=language)

    def __contains__(self, name: object) -> bool:
        try:
            self.resolve(str(name))
        except GlossaryError:
            return False
        return True

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self.names())


GlossarySet = GlossaryLibrary


def resolve_glossary_path(
    name: str | Path = DEFAULT_GLOSSARY_NAME,
    *,
    source: str | None = None,
    language: str | None = None,
    config_dir: str | Path | None = None,
    defaults_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return the user or packaged path selected for *name*."""
    library = GlossaryLibrary(
        config_dir=config_dir,
        defaults_dir=defaults_dir,
        environ=environ,
    )
    return library.resolve(name, source, language)


def default_glossary_path(
    name: str | Path = DEFAULT_GLOSSARY_NAME,
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
        candidate = directory / GLOSSARY_DIR / variant
        if candidate.is_file():
            return candidate
    return (
        directory
        / GLOSSARY_DIR
        / (variants[0] if variants else DEFAULT_GLOSSARY_FILENAME)
    )


def load_glossary(
    source: (
        str | Path | Mapping[str, Any] | Iterable[Any] | _BaseGlossary | None
    ) = None,
    *,
    name: str | Path | None = None,
    language: str | None = None,
    version: str | None = None,
    config_dir: str | Path | None = None,
    defaults_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Glossary:
    """Load a glossary from a name, file, mapping, list or existing glossary.

    With no source and no name the packaged ``default`` glossary is loaded. A
    string is treated as a name in the user and packaged ``glossary``
    directories unless it is an absolute path or contains a directory
    component. When *language* is given a file named after the language is
    preferred before the default file is filtered by language.
    """
    if isinstance(source, Glossary) and language is None:
        return source
    library = GlossaryLibrary(
        config_dir=config_dir,
        defaults_dir=defaults_dir,
        environ=environ,
    )
    if source is None:
        return library.load(
            name or DEFAULT_GLOSSARY_NAME,
            version,
            language=language,
        )
    if isinstance(source, str) and "\n" in source:
        return Glossary.from_mapping(source, language=language)
    if isinstance(source, (str, Path)):
        return library.load(source, version, language=language)
    if isinstance(source, Mapping) or (
        isinstance(source, Iterable) and not isinstance(source, (bytes, bytearray))
    ):
        return Glossary.from_value(source, language=language)
    if isinstance(source, _BaseGlossary):
        return Glossary.from_value(source, language=language)
    raise GlossaryInvalidError(
        "a glossary must be a name, path, mapping, list or Glossary",
        details={"value": repr(source)},
    )


glossary = load_glossary
build_glossary = load_glossary


def load_default_glossary(
    name: str = DEFAULT_GLOSSARY_NAME,
    *,
    language: str | None = None,
) -> Glossary:
    """Load a glossary shipped with this release."""
    return Glossary.from_default(name, language=language)


default_glossary = load_default_glossary


def glossary_names(
    *,
    config_dir: str | Path | None = None,
    defaults_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return every glossary name visible to the loader."""
    return GlossaryLibrary(
        config_dir=config_dir,
        defaults_dir=defaults_dir,
        environ=environ,
    ).names()


glossary_path = resolve_glossary_path
package_glossary_path = default_glossary_path
read_glossary = load_glossary
load_glossary_file = Glossary.from_file


def normalise_glossary(
    value: Any = None,
    *,
    language: str | None = None,
) -> Glossary:
    """Return *value* as a :class:`Glossary`, restricted to *language*."""
    if value is None:
        return Glossary()
    return Glossary.from_value(value, language=language)


normalize_glossary = normalise_glossary


def glossary_terms(value: Any = None) -> tuple[GlossaryTerm, ...]:
    """Return every term in *value* as a tuple."""
    return tuple(normalise_glossary(value).terms)


def apply_glossary(
    text: Any,
    glossary: Any = None,
    *,
    language: str | None = None,
    **kwargs: Any,
) -> Any:
    """Rewrite glossary terms in text, a script, captions or media prompts."""
    if isinstance(text, Script):
        return apply_glossary_to_script(
            text,
            glossary,
            language=language,
            **kwargs,
        )
    if _is_captions(text):
        return apply_glossary_to_captions(
            text,
            glossary,
            language=language,
            **kwargs,
        )
    if isinstance(text, (Beat, MediaPrompt, list, tuple)):
        return apply_glossary_to_media_prompts(
            text,
            glossary,
            language=language,
            **kwargs,
        )
    return normalise_glossary(glossary, language=language).apply(text)


def _apply_value(
    value: Any,
    active: Glossary,
    *,
    language: str | None,
) -> Any:
    if isinstance(value, str):
        return active.apply(value, language=language)
    if isinstance(value, Mapping):
        return {
            key: _apply_value(item, active, language=language)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_apply_value(item, active, language=language) for item in value]
    if isinstance(value, tuple):
        return tuple(_apply_value(item, active, language=language) for item in value)
    return value


def _text_violations(
    value: Any,
    active: Glossary,
    *,
    language: str | None,
    path: str = "",
) -> list[GlossaryViolation]:
    if isinstance(value, str):
        return list(active.violations(value, language=language, path=path))
    if isinstance(value, Mapping):
        found: list[GlossaryViolation] = []
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            found.extend(_text_violations(item, active, language=language, path=child))
        return found
    if isinstance(value, (list, tuple)):
        found = []
        for index, item in enumerate(value):
            child = f"{path}[{index}]" if path else f"[{index}]"
            found.extend(_text_violations(item, active, language=language, path=child))
        return found
    return []


def _active_glossary(
    value: Any,
    *,
    language: str | None = None,
    config_dir: str | Path | None = None,
    defaults_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Glossary:
    if isinstance(value, _BaseGlossary):
        return Glossary.from_value(value, language=language)
    if value is None:
        return Glossary()
    return load_glossary(
        value,
        language=language,
        config_dir=config_dir,
        defaults_dir=defaults_dir,
        environ=environ,
    )


def apply_glossary_to_script(
    script: Script,
    glossary: Any = None,
    *,
    language: str | None = None,
    include_media_prompts: bool = True,
    config_dir: str | Path | None = None,
    defaults_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Script:
    """Return a copy of *script* with glossary terms rewritten."""
    target = language or script.language or None
    active = _active_glossary(
        glossary,
        language=target,
        config_dir=config_dir,
        defaults_dir=defaults_dir,
        environ=environ,
    )
    result = script.model_copy(deep=True)
    result.title = str(active.apply(result.title, language=target))
    result.hook = str(active.apply(result.hook, language=target))
    result.outro = str(active.apply(result.outro, language=target))
    result.payload = _apply_value(result.payload, active, language=target)
    for beat in result.beats:
        beat.text = str(active.apply(beat.text, language=target))
        for line in beat.dialogue:
            line.text = str(active.apply(line.text, language=target))
        if include_media_prompts:
            for prompt in beat.media_prompts:
                prompt.prompt = str(active.apply(prompt.prompt, language=target))
                prompt.negative_prompt = str(
                    active.apply(prompt.negative_prompt, language=target)
                )
    for line in result.dialogue_lines:
        line.text = str(active.apply(line.text, language=target))
    return result


def apply_glossary_to_media_prompts(
    value: Any,
    glossary: Any = None,
    *,
    language: str | None = None,
    config_dir: str | Path | None = None,
    defaults_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Any:
    """Return a copy of *value* with glossary terms in prompts rewritten."""
    active = _active_glossary(
        glossary,
        language=language,
        config_dir=config_dir,
        defaults_dir=defaults_dir,
        environ=environ,
    )
    if isinstance(value, Script):
        script = value.model_copy(deep=True)
        for beat in script.beats:
            for prompt in beat.media_prompts:
                prompt.prompt = str(active.apply(prompt.prompt, language=language))
                prompt.negative_prompt = str(
                    active.apply(prompt.negative_prompt, language=language)
                )
        return script
    if isinstance(value, Beat):
        beat = value.model_copy(deep=True)
        for prompt in beat.media_prompts:
            prompt.prompt = str(active.apply(prompt.prompt, language=language))
            prompt.negative_prompt = str(
                active.apply(prompt.negative_prompt, language=language)
            )
        return beat
    if isinstance(value, MediaPrompt):
        prompt = value.model_copy(deep=True)
        prompt.prompt = str(active.apply(prompt.prompt, language=language))
        prompt.negative_prompt = str(
            active.apply(prompt.negative_prompt, language=language)
        )
        return prompt
    if isinstance(value, (list, tuple)):
        return type(value)(
            apply_glossary_to_media_prompts(
                item,
                active,
                language=language,
            )
            for item in value
        )
    raise GlossaryInvalidError(
        "media prompts must be a Script, Beat, MediaPrompt or a sequence",
        details={"value": type(value).__name__},
    )


apply_glossary_to_prompts = apply_glossary_to_media_prompts


def apply_glossary_to_captions(
    captions: Any,
    glossary: Any = None,
    *,
    language: str | None = None,
    config_dir: str | Path | None = None,
    defaults_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> CaptionSet:
    """Return a copy of *captions* with glossary terms rewritten."""
    target = language or getattr(captions, "language", "") or None
    active = _active_glossary(
        glossary,
        language=target,
        config_dir=config_dir,
        defaults_dir=defaults_dir,
        environ=environ,
    )
    result = captions.model_copy(deep=True)
    for name in ("title", "caption", "description", "source_line", "disclosure"):
        setattr(
            result,
            name,
            str(active.apply(str(getattr(result, name)), language=target)),
        )
    result.tags = [str(active.apply(tag, language=target)) for tag in result.tags]
    result.hashtags = [
        str(active.apply(tag, language=target)) for tag in result.hashtags
    ]
    violations = list(
        _text_violations(
            {
                "title": captions.title,
                "caption": captions.caption,
                "description": captions.description,
                "tags": captions.tags,
            },
            active,
            language=target,
        )
    )
    result.glossary_violations = [item.to_dict() for item in violations]
    return result  # type: ignore[no-any-return]


def glossary_violations(
    value: Any,
    glossary: Any = None,
    *,
    language: str | None = None,
    config_dir: str | Path | None = None,
    defaults_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[GlossaryViolation, ...]:
    """Return every glossary violation found in a text, script or captions."""
    target = language
    if target is None and isinstance(value, Script):
        target = value.language or None
    active = _active_glossary(
        glossary,
        language=target,
        config_dir=config_dir,
        defaults_dir=defaults_dir,
        environ=environ,
    )
    if isinstance(value, Script):
        parts: dict[str, Any] = {
            "title": value.title,
            "hook": value.hook,
            "outro": value.outro,
            "beats": [
                {
                    "text": beat.text,
                    "dialogue": [line.text for line in beat.dialogue],
                    "media_prompts": [
                        {
                            "prompt": prompt.prompt,
                            "negative_prompt": prompt.negative_prompt,
                        }
                        for prompt in beat.media_prompts
                    ],
                }
                for beat in value.beats
            ],
            "dialogue": [line.text for line in value.dialogue_lines],
            "payload": value.payload,
        }
        found = _text_violations(parts, active, language=target)
    elif _is_captions(value):
        found = _text_violations(
            {
                "title": value.title,
                "caption": value.caption,
                "description": value.description,
                "tags": value.tags,
            },
            active,
            language=target,
        )
    elif isinstance(value, MediaPrompt):
        found = _text_violations(
            {"prompt": value.prompt, "negative_prompt": value.negative_prompt},
            active,
            language=target,
        )
    elif isinstance(value, Beat):
        found = _text_violations(
            {
                "text": value.text,
                "dialogue": [line.text for line in value.dialogue],
                "media_prompts": [
                    {
                        "prompt": prompt.prompt,
                        "negative_prompt": prompt.negative_prompt,
                    }
                    for prompt in value.media_prompts
                ],
            },
            active,
            language=target,
        )
    else:
        found = _text_violations(value, active, language=target)
    return tuple(found)


def _is_captions(value: Any) -> bool:
    from aivia.writing.captions import CaptionSet

    return isinstance(value, CaptionSet)


find_glossary_violations = glossary_violations
report_glossary_violations = glossary_violations
check_glossary = glossary_violations


def warn_glossary_violations(
    value: Any,
    glossary: Any = None,
    *,
    language: str | None = None,
    stacklevel: int = 2,
    **kwargs: Any,
) -> tuple[GlossaryViolation, ...]:
    """Emit a warning for each glossary violation found in *value*."""
    import warnings

    violations = glossary_violations(value, glossary, language=language, **kwargs)
    for violation in violations:
        warnings.warn(str(violation), GlossaryWarning, stacklevel=stacklevel)
    return violations


class GlossaryWarning(UserWarning):
    """Advisory warning for a source form that differs from the glossary."""


DEFAULT_GLOSSARY = load_default_glossary()

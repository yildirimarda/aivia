"""Resolve bundled font families and their per-language fallback chains.

A composition names a family, a weight and a slant, but a renderer needs one
concrete font file. This module keeps that mapping in one place. Families are
read from the packaged font directory (or an explicit directory), so the same
configuration resolves to the same files on every machine.

Resolution is deliberately ordered:

* an explicit family or file always wins;
* a language's configured families are tried next;
* the configured fallback families and default family follow.

When bundled fonts are required, an unresolved family is an error. The manager
never silently substitutes a system font in that mode, because that would make
a render depend on the machine rather than on the pinned font set. A caller
that explicitly allows system fonts gets a system fallback instead.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from aivia.core.config.sections import RenderConfig
from aivia.core.paths import package_fonts_dir

from .errors import FontError
from .schema import TextStyle, TextWeight

__all__ = [
    "DEFAULT_FONT_FAMILY",
    "DEFAULT_RTL_FONT_FAMILY",
    "FONT_EXTENSIONS",
    "FontFace",
    "FontFamily",
    "FontManager",
    "FontNotFoundError",
    "FontResolutionError",
    "FontStyle",
    "UnknownFontError",
    "available_fonts",
    "font_families",
    "font_family_for_text",
    "font_manager",
    "font_path",
    "resolve_font",
    "system_font_path",
]

FONT_EXTENSIONS = frozenset({".ttf", ".otf", ".ttc"})
"""File suffixes the manager treats as font files."""

DEFAULT_FONT_FAMILY = RenderConfig().default_font
"""The family used when neither a layer nor the configuration names one."""

DEFAULT_RTL_FONT_FAMILY = "Amiri"
"""The bundled family used for Arabic text when no family is requested."""


class FontStyle(StrEnum):
    """The four faces a bundled family may provide."""

    REGULAR = "regular"
    BOLD = "bold"
    ITALIC = "italic"
    BOLDITALIC = "bolditalic"

    @property
    def bold(self) -> bool:
        """Whether this face uses the bold member."""
        return self in (FontStyle.BOLD, FontStyle.BOLDITALIC)

    @property
    def italic(self) -> bool:
        """Whether this face uses the italic member."""
        return self in (FontStyle.ITALIC, FontStyle.BOLDITALIC)

    @property
    def weight(self) -> TextWeight:
        """The text weight that selects this face."""
        return TextWeight.BOLD if self.bold else TextWeight.REGULAR

    @property
    def text_style(self) -> TextStyle:
        """The text slant that selects this face."""
        return TextStyle.ITALIC if self.italic else TextStyle.NORMAL

    @classmethod
    def _missing_(cls, value: Any) -> FontStyle | None:
        if not isinstance(value, str):
            return None
        key = _normalize_token(value)
        aliases = {
            "regular": cls.REGULAR,
            "normal": cls.REGULAR,
            "roman": cls.REGULAR,
            "book": cls.REGULAR,
            "bold": cls.BOLD,
            "italic": cls.ITALIC,
            "oblique": cls.ITALIC,
            "bolditalic": cls.BOLDITALIC,
            "italicbold": cls.BOLDITALIC,
        }
        return aliases.get(key)


_FACE_TOKENS = {
    "regular": FontStyle.REGULAR,
    "normal": FontStyle.REGULAR,
    "roman": FontStyle.REGULAR,
    "book": FontStyle.REGULAR,
    "bold": FontStyle.BOLD,
    "italic": FontStyle.ITALIC,
    "oblique": FontStyle.ITALIC,
    "bolditalic": FontStyle.BOLDITALIC,
    "italicbold": FontStyle.BOLDITALIC,
}

_SEPARATORS = re.compile(r"[^a-z0-9]+")
_SYSTEM_FONT_DEADLINE = 5

_LANGUAGE_NAMES = {
    "english": "en",
    "eng": "en",
    "german": "de",
    "deutsch": "de",
    "ger": "de",
    "deu": "de",
    "turkish": "tr",
    "türkçe": "tr",
    "turkce": "tr",
    "tur": "tr",
}


def font_family_for_text(
    text: str,
    family: str | Path | None = None,
) -> str | Path | None:
    """Return *family*, or the bundled Arabic family for unset Arabic text.

    An explicit family always wins. Text with no Arabic letters keeps the
    caller's value so the configured default applies.
    """
    if family is not None and str(family).strip():
        return family
    if any(unicodedata.bidirectional(character) == "AL" for character in text):
        return DEFAULT_RTL_FONT_FAMILY
    return family


def _normalize_token(value: str) -> str:
    """Fold a family or face name to a comparison key."""
    return _SEPARATORS.sub("", value.casefold())


def _language_key(value: str) -> str:
    """Normalize a language tag to a lower-case, hyphen-separated key."""
    key = value.strip().replace("_", "-").casefold()
    base = key.split("-", 1)[0]
    return _LANGUAGE_NAMES.get(key, _LANGUAGE_NAMES.get(base, key))


def _language_keys(value: str) -> tuple[str, ...]:
    key = _language_key(value)
    base = key.split("-", 1)[0]
    return (key,) if base == key or not base else (key, base)


def _face_style(value: Any) -> FontStyle | None:
    if value is None:
        return None
    if isinstance(value, FontStyle):
        return value
    if isinstance(value, TextStyle):
        return FontStyle.ITALIC if value is TextStyle.ITALIC else FontStyle.REGULAR
    if isinstance(value, TextWeight):
        return FontStyle.BOLD if value.is_bold else FontStyle.REGULAR
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        try:
            weight = TextWeight(str(value))
        except ValueError:
            return None
        return FontStyle.BOLD if weight.is_bold else FontStyle.REGULAR
    if isinstance(value, str):
        try:
            return FontStyle(value)
        except ValueError:
            pass
        try:
            weight = TextWeight(value)
        except ValueError:
            pass
        else:
            return FontStyle.BOLD if weight.is_bold else FontStyle.REGULAR
        try:
            text_style = TextStyle(value)
        except ValueError:
            return None
        return FontStyle.ITALIC if text_style is TextStyle.ITALIC else FontStyle.REGULAR
    return None


def _select_style(weight: Any = None, style: Any = None) -> FontStyle:
    """Combine a weight and a slant into one of the four bundled faces."""
    bold = False
    italic = False
    if weight is not None:
        selected = _face_style(weight)
        if selected is None:
            raise FontError(f"unsupported font weight {weight!r}")
        bold = bold or selected.bold
        italic = italic or selected.italic
    if style is not None:
        selected = _face_style(style)
        if selected is None:
            raise FontError(f"unsupported font style {style!r}")
        bold = bold or selected.bold
        italic = italic or selected.italic
    if bold and italic:
        return FontStyle.BOLDITALIC
    if bold:
        return FontStyle.BOLD
    if italic:
        return FontStyle.ITALIC
    return FontStyle.REGULAR


def _member_style(family: str, stem: str) -> FontStyle | None:
    """Return the face a font file provides inside its family directory."""
    stem_key = _normalize_token(stem)
    family_key = _normalize_token(family)
    remainder = (
        stem_key[len(family_key) :]
        if family_key and stem_key.startswith(family_key)
        else stem_key
    )
    if not remainder:
        return FontStyle.REGULAR
    for token in sorted(_FACE_TOKENS, key=len, reverse=True):
        if remainder.endswith(token):
            return _FACE_TOKENS[token]
    has_bold = "bold" in remainder
    has_italic = "italic" in remainder or "oblique" in remainder
    if has_bold and has_italic:
        return FontStyle.BOLDITALIC
    if has_bold:
        return FontStyle.BOLD
    if has_italic:
        return FontStyle.ITALIC
    return None


def _family_name(directory: str, files: list[Path]) -> str:
    """Derive the display name of a family from its font file names."""
    for path in files:
        for match in reversed(list(re.finditer(r"[A-Za-z0-9]+", path.stem))):
            if match.group().casefold() in _FACE_TOKENS:
                prefix = path.stem[: match.start()].rstrip("-_ ")
                if prefix:
                    return prefix
                break
    return directory


@dataclass(frozen=True, slots=True)
class FontFace:
    """One concrete font file selected for a family, weight and slant."""

    family: str
    path: Path
    style: FontStyle = FontStyle.REGULAR
    bundled: bool = True

    @property
    def name(self) -> str:
        """The file name of the selected face."""
        return self.path.name

    @property
    def suffix(self) -> str:
        """The file suffix of the selected face."""
        return self.path.suffix

    @property
    def weight(self) -> TextWeight:
        """The text weight that selects this face."""
        return self.style.weight

    @property
    def text_style(self) -> TextStyle:
        """The text slant that selects this face."""
        return self.style.text_style

    @property
    def bold(self) -> bool:
        """Whether this face is bold."""
        return self.style.bold

    @property
    def italic(self) -> bool:
        """Whether this face is italic."""
        return self.style.italic

    def is_file(self) -> bool:
        """Whether the selected font file exists."""
        return self.path.is_file()

    def exists(self) -> bool:
        """Alias for :meth:`is_file`."""
        return self.is_file()

    def __fspath__(self) -> str:
        return str(self.path)

    def __str__(self) -> str:
        return str(self.path)


@dataclass(frozen=True, slots=True)
class FontFamily:
    """A bundled family and the faces it provides."""

    name: str
    members: Mapping[FontStyle, Path]
    bundled: bool = True

    def face(self, style: FontStyle = FontStyle.REGULAR) -> FontFace | None:
        """Return the requested face, or ``None`` when the family lacks it."""
        path = self.members.get(style)
        if path is None:
            return None
        return FontFace(
            family=self.name,
            path=path,
            style=style,
            bundled=self.bundled,
        )

    def path(
        self,
        *,
        weight: Any = None,
        style: Any = None,
    ) -> Path | None:
        """Return the path of the face selected by *weight* and *style*."""
        face = self.face(_select_style(weight, style))
        return None if face is None else face.path

    @property
    def styles(self) -> tuple[FontStyle, ...]:
        """The faces this family provides, in canonical order."""
        return tuple(style for style in FontStyle if style in self.members)

    def __contains__(self, style: object) -> bool:
        return isinstance(style, FontStyle) and style in self.members

    def __iter__(self) -> Iterator[FontStyle]:
        return iter(self.styles)


def _scan_families(root: Path) -> dict[str, FontFamily]:
    families: dict[str, FontFamily] = {}
    if not root.is_dir():
        return families
    for directory in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
        if not directory.is_dir():
            continue
        members: dict[FontStyle, Path] = {}
        files = sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.casefold() in FONT_EXTENSIONS
            ),
            key=lambda item: item.name.casefold(),
        )
        for path in files:
            style = _member_style(directory.name, path.stem)
            if style is not None and style not in members:
                members[style] = path
        if not members and files:
            members[FontStyle.REGULAR] = files[0]
        if not members:
            continue
        name = _family_name(directory.name, files)
        key = _normalize_token(name)
        families.setdefault(
            key,
            FontFamily(
                name=name,
                members=MappingProxyType(dict(members)),
            ),
        )
    return families


def _chain(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values: Iterable[Any] = (value,)
    elif isinstance(value, Iterable):
        values = value
    else:
        raise TypeError("font fallback chain must be a string or an iterable")
    result: list[str] = []
    for item in values:
        if not isinstance(item, str):
            raise TypeError("font family names must be strings")
        text = item.strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _language_fonts(value: object | None) -> dict[str, tuple[str, ...]]:
    if value is None:
        return {}
    if hasattr(value, "languages"):
        raw = cast(Any, value).languages
    elif isinstance(value, Mapping):
        raw = value.get("languages", value)
    else:
        return {}
    if not isinstance(raw, Mapping):
        return {}
    fonts: dict[str, tuple[str, ...]] = {}
    for code, entry in raw.items():
        if not isinstance(code, str):
            continue
        values: list[str] = []
        if isinstance(entry, str) and entry.strip():
            values.append(entry.strip())
        elif isinstance(entry, Iterable) and not isinstance(entry, (str, Mapping)):
            values.extend(
                item.strip() for item in entry if isinstance(item, str) and item.strip()
            )
        for field in (
            "subtitle_font",
            "font",
            "font_family",
            "fonts",
            "fallback_fonts",
            "font_fallbacks",
        ):
            value = getattr(entry, field, None)
            if value is None and isinstance(entry, Mapping):
                value = entry.get(field)
            if isinstance(value, str) and value.strip():
                values.append(value.strip())
            elif isinstance(value, Iterable) and not isinstance(value, str):
                values.extend(
                    item.strip()
                    for item in value
                    if isinstance(item, str) and item.strip()
                )
        if values:
            fonts[_language_key(code)] = tuple(dict.fromkeys(values))
    return fonts


def _split_face_name(name: str) -> tuple[str, FontStyle] | None:
    key = _normalize_token(name)
    for token in sorted(_FACE_TOKENS, key=len, reverse=True):
        if key.endswith(token) and len(key) > len(token):
            return key[: -len(token)], _FACE_TOKENS[token]
    return None


class FontManager:
    """Resolve font families and faces from a bundled font directory.

    ``fonts_dir`` may be a directory, ``"package"`` for the packaged fonts, or
    a :class:`RenderConfig`. Passing a config copies its ``default_font``,
    ``fallback_fonts``, ``fonts_dir`` and ``require_bundled_fonts`` settings.
    ``languages`` may be a :class:`LanguagesConfig` whose ``subtitle_font``
    becomes the first family in each language's chain.
    """

    def __init__(
        self,
        fonts_dir: str | Path | RenderConfig | Mapping[str, Any] | None = None,
        *,
        default_family: str | None = None,
        default_font: str | None = None,
        fallback_families: Iterable[str] | str | None = None,
        fallbacks: Iterable[str] | str | None = None,
        language_fallbacks: Mapping[str, Iterable[str] | str] | None = None,
        language_fonts: Mapping[str, Iterable[str] | str] | None = None,
        languages: object | None = None,
        render: RenderConfig | Mapping[str, Any] | None = None,
        require_bundled: bool | None = None,
        require_bundled_fonts: bool | None = None,
        allow_system: bool | None = None,
        system_fallback: str | Path | Callable[[str], str | Path | None] | None = None,
        system_fonts: str | Path | Callable[[str], str | Path | None] | None = None,
    ) -> None:
        if isinstance(fonts_dir, (RenderConfig, Mapping)):
            render = fonts_dir
            fonts_dir = None
        elif fonts_dir is not None and hasattr(fonts_dir, "render"):
            app = cast(Any, fonts_dir)
            if languages is None:
                languages = getattr(app, "languages", None)
            render = app.render
            fonts_dir = None
        if default_family is None:
            default_family = default_font
        if fallback_families is None:
            fallback_families = fallbacks
        if language_fallbacks is None:
            language_fallbacks = language_fonts
        if render is not None and not isinstance(render, (RenderConfig, Mapping)):
            app = cast(Any, render)
            if languages is None:
                languages = getattr(app, "languages", None)
            render = getattr(app, "render", render)
        config = RenderConfig() if render is None else render
        if not isinstance(config, RenderConfig):
            config = RenderConfig.from_mapping(config)

        if fonts_dir is None and config.fonts_dir not in ("", "package"):
            fonts_dir = config.fonts_dir
        root = (
            package_fonts_dir()
            if fonts_dir is None
            or (isinstance(fonts_dir, str) and fonts_dir.strip() in ("", "package"))
            else Path(fonts_dir).expanduser()
        )

        family = config.default_font if default_family is None else default_family
        fallbacks = (
            config.fallback_fonts if fallback_families is None else fallback_families
        )
        if not isinstance(family, str) or not family.strip():
            raise FontError("default font family must not be empty")

        chains = _language_fonts(languages)
        if language_fallbacks is not None:
            for code, values in language_fallbacks.items():
                if not isinstance(code, str) or not code.strip():
                    raise FontError("language font fallback keys must not be empty")
                chains[_language_key(code)] = _chain(values)

        self._fonts_dir = root
        self._default_family = family.strip()
        self._fallback_families = _chain(fallbacks)
        self._language_fallbacks = chains
        if (
            allow_system is not None
            and require_bundled is None
            and require_bundled_fonts is None
        ):
            require_bundled = not allow_system
        self._require_bundled = (
            bool(require_bundled_fonts)
            if require_bundled_fonts is not None
            else (
                bool(require_bundled)
                if require_bundled is not None
                else config.require_bundled_fonts
            )
        )
        self._system_fallback = (
            system_fallback if system_fallback is not None else system_fonts
        )
        self._families = _scan_families(root)

    @classmethod
    def from_config(
        cls,
        render: object | None = None,
        languages: object | None = None,
        **kwargs: Any,
    ) -> FontManager:
        """Build a manager from render and language configuration."""
        if render is not None and not isinstance(render, (RenderConfig, Mapping)):
            app = cast(Any, render)
            if languages is None:
                languages = getattr(app, "languages", None)
            render = getattr(app, "render", render)
        return cls(render=render, languages=languages, **kwargs)  # type: ignore[arg-type]

    @property
    def fonts_dir(self) -> Path:
        """The directory scanned for bundled families."""
        return self._fonts_dir

    @property
    def default_family(self) -> str:
        """The family used when no layer or language names one."""
        return self._default_family

    @property
    def fallback_families(self) -> tuple[str, ...]:
        """The configured global fallback families."""
        return self._fallback_families

    @property
    def language_fallbacks(self) -> dict[str, tuple[str, ...]]:
        """A copy of the per-language family chains."""
        return dict(self._language_fallbacks)

    @property
    def require_bundled(self) -> bool:
        """Whether an unresolved family is an error instead of a system font."""
        return self._require_bundled

    @property
    def require_bundled_fonts(self) -> bool:
        """Alias for :attr:`require_bundled`."""
        return self._require_bundled

    @property
    def families(self) -> dict[str, FontFamily]:
        """A copy of the available families keyed by their display name."""
        return {family.name: family for family in self._families.values()}

    @property
    def available_families(self) -> tuple[str, ...]:
        """The available family names in sorted order."""
        return tuple(sorted(family.name for family in self._families.values()))

    def names(self) -> list[str]:
        """Return the available family names in sorted order."""
        return list(self.available_families)

    def list(self) -> list[FontFamily]:
        """Return the available families sorted by name."""
        return sorted(
            self._families.values(),
            key=lambda family: family.name.casefold(),
        )

    def family(self, name: str) -> FontFamily | None:
        """Return a bundled family by name, including a ``Family-Bold`` alias."""
        found = self._find_family(name)
        return None if found is None else found[0]

    def get(self, name: str) -> FontFamily | None:
        """Alias for :meth:`family`."""
        return self.family(name)

    def refresh(self) -> None:
        """Re-scan the font directory after files were added or removed."""
        self._families = _scan_families(self._fonts_dir)

    def _find_family(self, name: str) -> tuple[FontFamily, FontStyle | None] | None:
        family = self._families.get(_normalize_token(name))
        if family is not None:
            return family, None
        split = _split_face_name(name)
        if split is None:
            return None
        base, style = split
        family = self._families.get(base)
        if family is None:
            return None
        return family, style

    def _default_chain(self) -> tuple[str, ...]:
        chain: list[str] = []
        for family in (self._default_family, *self._fallback_families):
            if family not in chain:
                chain.append(family)
        return tuple(chain)

    def _language_chain(self, language: str) -> tuple[str, ...]:
        chain: list[str] = []
        for key in _language_keys(language):
            for family in self._language_fallbacks.get(key, ()):
                if family not in chain:
                    chain.append(family)
        for family in (*self._fallback_families, self._default_family):
            if family not in chain:
                chain.append(family)
        return tuple(chain)

    def candidates(
        self,
        family: str | Path | None = None,
        language: str | None = None,
        *,
        fallback: str | None = None,
    ) -> tuple[str, ...]:
        """Return the family names tried for *family* and *language*."""
        chain: list[str] = []
        if family is not None:
            text = str(family).strip()
            if text:
                chain.append(text)
        if language is not None and str(language).strip():
            chain.extend(self._language_chain(str(language)))
        elif not chain:
            chain.extend(self._default_chain())
        if fallback is not None and fallback.strip():
            chain.append(fallback.strip())
        if not chain:
            chain.extend(self._default_chain())
        result: list[str] = []
        for item in chain:
            if item not in result:
                result.append(item)
        return tuple(result)

    def select(
        self,
        family: str | Path | None = None,
        language: str | None = None,
        *,
        weight: Any = None,
        style: Any = None,
        fallback: str | None = None,
    ) -> FontFace:
        """Return the :class:`FontFace` chosen for a family and language.

        An explicit *family* is never silently replaced by a configured
        fallback unless a *language* or *fallback* supplies another candidate.
        """
        requested = _select_style(weight, style)
        candidates = self.candidates(
            family,
            language=language,
            fallback=fallback,
        )
        tried: list[str] = []
        for candidate in candidates:
            path = Path(candidate).expanduser()
            if path.is_file():
                return FontFace(
                    family=path.stem,
                    path=path,
                    style=requested,
                    bundled=False,
                )
            found = self._find_family(candidate)
            if found is None:
                tried.append(candidate)
                continue
            bundled, override = found
            face = bundled.face(override or requested)
            if face is None:
                tried.append(candidate)
                continue
            return face

        requested_family = (
            str(family).strip()
            if family is not None and str(family).strip()
            else (self._default_family if not candidates else candidates[0])
        )
        if self._require_bundled:
            if not self._families:
                raise FontError(
                    "bundled fonts are not available (system fallback is disabled)",
                    details={
                        "fonts_dir": str(self._fonts_dir),
                        "require_bundled": True,
                    },
                )
            raise FontError(
                f"bundled font family {requested_family!r} is not available "
                "(system fallback is disabled)",
                details={
                    "family": requested_family,
                    "language": language,
                    "style": requested,
                    "tried": tried,
                    "available": list(self.available_families),
                    "require_bundled": True,
                },
            )
        return self._system_face(
            requested_family,
            requested,
            language=language,
        )

    def resolve(
        self,
        family: str | Path | None = None,
        language: str | None = None,
        *,
        weight: Any = None,
        style: Any = None,
        fallback: str | None = None,
    ) -> Path:
        """Return the path of the font file chosen for a family and language."""
        return self.select(
            family,
            language=language,
            weight=weight,
            style=style,
            fallback=fallback,
        ).path

    def resolve_family(
        self,
        family: str | Path | None = None,
        language: str | None = None,
        *,
        weight: Any = None,
        style: Any = None,
        fallback: str | None = None,
    ) -> str:
        """Return the display name of the family chosen for a request."""
        return self.select(
            family,
            language=language,
            weight=weight,
            style=style,
            fallback=fallback,
        ).family

    def resolve_font(
        self,
        family: str | Path | None = None,
        language: str | None = None,
        *,
        weight: Any = None,
        style: Any = None,
        fallback: str | None = None,
    ) -> Path:
        """Alias for :meth:`resolve`."""
        return self.resolve(
            family,
            language=language,
            weight=weight,
            style=style,
            fallback=fallback,
        )

    def resolve_for_language(
        self,
        language: str | None = None,
        *,
        family: str | Path | None = None,
        weight: Any = None,
        style: Any = None,
        fallback: str | None = None,
    ) -> Path:
        """Resolve a family with *language* as the first argument."""
        return self.resolve(
            family,
            language=language,
            weight=weight,
            style=style,
            fallback=fallback,
        )

    resolve_language = resolve_for_language

    def font(
        self,
        family: str | Path | None = None,
        language: str | None = None,
        *,
        weight: Any = None,
        style: Any = None,
        fallback: str | None = None,
    ) -> Path:
        """Alias for :meth:`resolve`."""
        return self.resolve(
            family,
            language=language,
            weight=weight,
            style=style,
            fallback=fallback,
        )

    def path(
        self,
        family: str | Path | None = None,
        language: str | None = None,
        *,
        weight: Any = None,
        style: Any = None,
        fallback: str | None = None,
    ) -> Path:
        """Alias for :meth:`resolve`."""
        return self.resolve(
            family,
            language=language,
            weight=weight,
            style=style,
            fallback=fallback,
        )

    font_path = path

    def resolve_face(
        self,
        family: str | Path | None = None,
        language: str | None = None,
        *,
        weight: Any = None,
        style: Any = None,
        fallback: str | None = None,
    ) -> FontFace:
        """Alias for :meth:`select`."""
        return self.select(
            family,
            language=language,
            weight=weight,
            style=style,
            fallback=fallback,
        )

    select_face = resolve_face
    get_face = resolve_face

    def _system_face(
        self,
        family: str,
        style: FontStyle,
        *,
        language: str | None,
    ) -> FontFace:
        path = self._system_path(family)
        return FontFace(
            family=family,
            path=Path(path) if path is not None else Path(family),
            style=style,
            bundled=False,
        )

    def _system_path(self, family: str) -> str | Path | None:
        fallback = self._system_fallback
        if callable(fallback):
            return fallback(family)
        if fallback is not None:
            return fallback
        return system_font_path(family)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self._find_family(name) is not None

    def __len__(self) -> int:
        return len(self._families)

    def __iter__(self) -> Iterator[FontFamily]:
        return iter(self.list())


FontResolutionError = FontError
"""Alias for :class:`~aivia.compose.errors.FontError`."""

FontNotFoundError = FontError
"""Alias for :class:`~aivia.compose.errors.FontError`."""

UnknownFontError = FontError
"""Alias for :class:`~aivia.compose.errors.FontError`."""


@lru_cache(maxsize=32)
def _cached_system_font_path(family: str) -> str | None:
    executable = shutil.which("fc-match")
    if executable is not None:
        try:
            result = subprocess.run(
                [executable, "-f", "%{file}", family],
                capture_output=True,
                text=True,
                check=False,
                timeout=_SYSTEM_FONT_DEADLINE,
            )
        except (OSError, subprocess.SubprocessError):
            result = None
        if result is not None:
            candidate = Path(result.stdout.strip())
            if candidate.is_file():
                return str(candidate)
    key = _normalize_token(family)
    if not key:
        return None
    roots = (
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".fonts",
        Path.home() / ".local" / "share" / "fonts",
    )
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*"), key=lambda item: str(item)):
            if (
                path.is_file()
                and path.suffix.casefold() in FONT_EXTENSIONS
                and key in _normalize_token(path.stem)
            ):
                return str(path)
    return None


def system_font_path(family: str) -> str | None:
    """Return a system font file matching *family*, or ``None``.

    Fontconfig is used when it is installed; otherwise the usual system font
    directories are searched by file name. This never touches the network.
    """
    return _cached_system_font_path(family)


def font_manager(
    value: FontManager | str | Path | RenderConfig | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> FontManager:
    """Return *value* as a :class:`FontManager`, building one when needed."""
    if isinstance(value, FontManager):
        return value
    return FontManager(value, **kwargs)


def resolve_font(
    family: str | Path | FontManager | None = None,
    manager: FontManager | str | Path | RenderConfig | Mapping[str, Any] | None = None,
    *,
    language: str | None = None,
    weight: Any = None,
    style: Any = None,
    fallback: str | None = None,
    **kwargs: Any,
) -> Path:
    """Resolve a font path with a manager, family-first or manager-first."""
    if isinstance(family, FontManager):
        if isinstance(manager, (str, Path)) or manager is None:
            family, manager = manager, family
        else:
            manager = family
            family = None
    selected = font_manager(manager, **kwargs)
    return selected.resolve(
        family,
        language=language,
        weight=weight,
        style=style,
        fallback=fallback,
    )


font_path = resolve_font


def available_fonts(
    manager: FontManager | str | Path | RenderConfig | Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Return the family names available to *manager*."""
    return font_manager(manager).available_families


def font_families(
    manager: FontManager | str | Path | RenderConfig | Mapping[str, Any] | None = None,
) -> list[FontFamily]:
    """Return the families available to *manager*."""
    return font_manager(manager).list()

# ruff: noqa: RUF001, RUF002
"""Normalize language codes, detect a language and read its settings.

A language is named in many ways: ``en``, ``EN``, ``en_US``, ``en-US.UTF-8``
or ``English``. This module reduces all of them to one canonical base code so
callers never compare raw locale strings. The set of usable languages and their
settings still comes from :class:`~aivia.core.config.sections.LanguagesConfig`;
nothing here hard-codes a font, a speech rate or a locale.

Detection is deliberately cheap and offline. It scores distinctive characters
and common words for the languages the design ships (English, German and
Turkish) and returns ``None`` when the text is empty or ambiguous. A caller can
restrict detection to its configured languages, which keeps an unknown language
from being guessed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from aivia.core import languages as _languages
from aivia.core.config.sections import (
    LanguageConfig,
    LanguagesConfig,
    WordsPerSecondConfig,
)
from aivia.core.errors import ConfigError, ValidationError
from aivia.core.languages import TextDirection

__all__ = [
    "DEFAULT_LANGUAGE",
    "LanguageResolver",
    "LanguageSettings",
    "TextDirection",
    "detect_language",
    "hashtag_style",
    "is_rtl_language",
    "language_code",
    "language_direction",
    "language_locale",
    "language_name",
    "normalize_language_code",
    "normalize_languages",
    "resolve_language",
    "subtitle_font",
    "words_per_second",
]

DEFAULT_LANGUAGE = "en"
"""The canonical code used when no language is configured."""


_NAME_TO_CODE = {
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

_ISO_639_2_TO_CODE = {
    "eng": "en",
    "ger": "de",
    "deu": "de",
    "tur": "tr",
}

_ENCODING_SUFFIX = re.compile(r"\.[A-Za-z0-9_-]+$")
_CODE_PART = re.compile(r"[a-z]{2,3}")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

_TURKISH_MARKERS = ("ı", "ş", "ğ", "ç", "İ")
_GERMAN_MARKERS = ("ä", "ß")

_STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset(
        {
            "a",
            "about",
            "all",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "been",
            "but",
            "by",
            "can",
            "could",
            "did",
            "do",
            "does",
            "for",
            "from",
            "had",
            "has",
            "have",
            "he",
            "her",
            "here",
            "his",
            "how",
            "i",
            "in",
            "is",
            "it",
            "its",
            "me",
            "more",
            "most",
            "my",
            "no",
            "not",
            "of",
            "on",
            "or",
            "our",
            "she",
            "should",
            "that",
            "the",
            "their",
            "there",
            "these",
            "they",
            "this",
            "those",
            "to",
            "very",
            "was",
            "we",
            "were",
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "will",
            "with",
            "would",
            "yes",
            "you",
            "your",
        }
    ),
    "de": frozenset(
        {
            "aber",
            "als",
            "am",
            "an",
            "auch",
            "auf",
            "aus",
            "bei",
            "das",
            "dass",
            "dein",
            "dem",
            "den",
            "der",
            "des",
            "die",
            "dich",
            "diese",
            "dieser",
            "dieses",
            "du",
            "ein",
            "eine",
            "einem",
            "einen",
            "einer",
            "eines",
            "er",
            "es",
            "euer",
            "für",
            "hat",
            "habe",
            "haben",
            "hast",
            "hatte",
            "ich",
            "ihr",
            "im",
            "in",
            "ist",
            "kann",
            "kein",
            "keine",
            "können",
            "mit",
            "mich",
            "mehr",
            "mein",
            "muss",
            "müssen",
            "nach",
            "nicht",
            "noch",
            "nur",
            "oder",
            "sich",
            "sie",
            "sind",
            "soll",
            "sollen",
            "sehr",
            "sein",
            "über",
            "um",
            "und",
            "uns",
            "unser",
            "unter",
            "vor",
            "war",
            "waren",
            "was",
            "wann",
            "warum",
            "wer",
            "werden",
            "wie",
            "wird",
            "wurde",
            "wurden",
            "zu",
        }
    ),
    "tr": frozenset(
        {
            "ama",
            "ancak",
            "bazı",
            "ben",
            "benim",
            "bir",
            "biz",
            "bizim",
            "bu",
            "bütün",
            "çok",
            "da",
            "daha",
            "de",
            "değil",
            "en",
            "fakat",
            "gibi",
            "her",
            "için",
            "ile",
            "ise",
            "kadar",
            "ki",
            "kim",
            "mı",
            "mi",
            "mu",
            "mü",
            "nasıl",
            "ne",
            "neden",
            "nerede",
            "o",
            "olan",
            "olarak",
            "olduğu",
            "onlar",
            "onun",
            "önce",
            "sen",
            "senin",
            "siz",
            "sizin",
            "sonra",
            "şey",
            "şu",
            "tüm",
            "var",
            "ve",
            "veya",
            "ya",
            "yok",
        }
    ),
}


def normalize_language_code(value: str) -> str:
    """Return the canonical base code for a language, locale or name.

    Region, script, encoding and POSIX modifiers are dropped, so ``en_US``,
    ``en-US.UTF-8`` and ``English`` all become ``en``. A well-formed code that
    is not one of the known names is returned in lower case rather than
    rejected, because adding a language is a configuration change.
    """
    if not isinstance(value, str):
        raise ValidationError(
            "language code must be a string",
            details={"value": repr(value)},
        )
    text = value.strip()
    if not text:
        raise ValidationError("language code must not be empty")

    text = _ENCODING_SUFFIX.sub("", text)
    text = text.split("@", 1)[0]
    text = text.split("(", 1)[0].strip()
    lowered = text.replace("_", "-").casefold()
    if not lowered:
        raise ValidationError("language code must not be empty")

    if lowered in _NAME_TO_CODE:
        return _NAME_TO_CODE[lowered]

    primary = re.split(r"[-\s]+", lowered, maxsplit=1)[0]
    if primary in _NAME_TO_CODE:
        return _NAME_TO_CODE[primary]
    if primary in _ISO_639_2_TO_CODE:
        return _ISO_639_2_TO_CODE[primary]
    if not _CODE_PART.fullmatch(primary):
        raise ValidationError(
            f"not a language code: {value!r}",
            details={"value": value},
        )
    return primary


def language_code(value: str) -> str:
    """Alias for :func:`normalize_language_code`."""
    return normalize_language_code(value)


def normalize_languages(values: Iterable[str]) -> list[str]:
    """Normalize and de-duplicate language names, keeping their first order."""
    codes: list[str] = []
    for value in values:
        code = normalize_language_code(value)
        if code not in codes:
            codes.append(code)
    return codes


def language_direction(value: str) -> TextDirection:
    """Return the default writing direction for a language."""
    return _languages.language_direction(normalize_language_code(value))


def is_rtl_language(value: str) -> bool:
    """Whether a language is written right to left by default."""
    return _languages.is_rtl_language(normalize_language_code(value))


def _marker_score(text: str, markers: tuple[str, ...]) -> int:
    folded = text.casefold()
    score = sum(folded.count(marker) for marker in markers if marker.islower())
    score += sum(text.count(marker) for marker in markers if not marker.islower())
    return score


def _detection_score(text: str, code: str) -> int:
    markers = (
        _TURKISH_MARKERS if code == "tr" else _GERMAN_MARKERS if code == "de" else ()
    )
    words = _STOPWORDS.get(code, frozenset())
    tokens = _WORD.findall(text.casefold())
    stopword_hits = sum(1 for token in tokens if token in words)
    return stopword_hits + 3 * _marker_score(text, markers)


def detect_language(
    text: str,
    *,
    allowed: Iterable[str] | None = None,
) -> str | None:
    """Detect English, German or Turkish text, or return ``None``.

    Detection is deterministic and offline. It combines distinctive characters
    (for example ``ı`` and ``ß``) with common words. Empty, numeric or
    ambiguous text returns ``None`` instead of a guess. Passing *allowed*
    restricts the result to those languages.
    """
    if not isinstance(text, str):
        raise ValidationError("text must be a string")
    if not _WORD.search(text):
        return None

    candidates = (
        tuple(_STOPWORDS) if allowed is None else tuple(normalize_languages(allowed))
    )
    if not candidates:
        return None

    scores = {code: _detection_score(text, code) for code in candidates}
    best = max(scores.values())
    if best <= 0:
        return None
    winners = [code for code, score in scores.items() if score == best]
    return winners[0] if len(winners) == 1 else None


@dataclass(frozen=True, slots=True)
class LanguageSettings:
    """One configured language addressed by its canonical code."""

    code: str
    config: LanguageConfig

    @property
    def name(self) -> str:
        """The configured display name, falling back to the code."""
        return self.config.name or self.code

    @property
    def locale(self) -> str:
        """The configured locale, for example ``de_DE``."""
        return self.config.locale

    @property
    def words_per_second(self) -> WordsPerSecondConfig:
        """The configured inclusive speaking-rate bounds."""
        return self.config.words_per_second

    @property
    def min_words_per_second(self) -> float:
        """The lower speaking-rate bound."""
        return self.config.words_per_second.min

    @property
    def max_words_per_second(self) -> float:
        """The upper speaking-rate bound."""
        return self.config.words_per_second.max

    @property
    def subtitle_font(self) -> str:
        """The configured subtitle font family."""
        return self.config.subtitle_font

    @property
    def hashtag_style(self) -> str:
        """The configured hashtag casing style."""
        return self.config.hashtag_style

    @property
    def direction(self) -> TextDirection:
        """The default writing direction of this language."""
        return language_direction(self.code)


class LanguageResolver:
    """Look up configured languages and their per-language settings."""

    def __init__(
        self,
        config: LanguagesConfig | Mapping[str, Any] | None = None,
    ) -> None:
        if config is None:
            config = LanguagesConfig.from_mapping({DEFAULT_LANGUAGE: {}})
        elif not isinstance(config, LanguagesConfig):
            config = LanguagesConfig.from_mapping(config)
        self._config = config

        settings: dict[str, LanguageSettings] = {}
        aliases: dict[str, str] = {}
        for raw_code, language in config.languages.items():
            code = normalize_language_code(raw_code)
            if code in settings:
                raise ConfigError(
                    f"duplicate language code: {code}",
                    details={"code": code, "available": sorted(settings)},
                )
            item = LanguageSettings(code=code, config=language)
            settings[code] = item
            aliases[code] = code
            self._add_alias(aliases, language.name, code)
            self._add_alias(aliases, language.locale, code)

        self._settings = settings
        self._aliases = aliases
        self._default = normalize_language_code(config.default)
        if self._default not in settings:
            raise ConfigError(
                f"default language is not configured: {config.default}",
                details={
                    "default": config.default,
                    "available": sorted(settings),
                },
            )

    @staticmethod
    def _add_alias(aliases: dict[str, str], value: str, code: str) -> None:
        if not value:
            return
        try:
            normalized = normalize_language_code(value)
        except ValidationError:
            normalized = ""
        if normalized:
            aliases.setdefault(normalized, code)
        raw = value.strip().replace("_", "-").casefold()
        if raw:
            aliases.setdefault(raw, code)

    @property
    def default(self) -> LanguageSettings:
        """The language used when none is requested."""
        return self._settings[self._default]

    @property
    def codes(self) -> tuple[str, ...]:
        """Every configured canonical code, in configuration order."""
        return tuple(self._settings)

    @property
    def settings(self) -> tuple[LanguageSettings, ...]:
        """Every configured language, in configuration order."""
        return tuple(self._settings.values())

    def __len__(self) -> int:
        return len(self._settings)

    def __iter__(self) -> Iterator[LanguageSettings]:
        return iter(self._settings.values())

    def __contains__(self, value: object) -> bool:
        return isinstance(value, str) and self.get(value) is not None

    def get(self, value: str | None) -> LanguageSettings | None:
        """Return a language by code, locale or name, or ``None``."""
        if value is None:
            return None
        try:
            normalized = normalize_language_code(value)
        except ValidationError:
            return None
        code = self._aliases.get(normalized)
        if code is None:
            code = self._aliases.get(value.strip().replace("_", "-").casefold())
        return self._settings.get(code) if code is not None else None

    def require(self, value: str | None = None) -> LanguageSettings:
        """Return a language or raise :class:`ConfigError` with the choices."""
        if value is None or not str(value).strip():
            return self.default
        item = self.get(value)
        if item is None:
            available = ", ".join(self.codes)
            raise ConfigError(
                f"unknown language: {value} (available: {available})",
                details={"language": value, "available": list(self.codes)},
            )
        return item

    def resolve(self, value: str | None = None) -> LanguageSettings:
        """Resolve *value*, falling back to the configured default."""
        return self.require(value)

    def detect(self, text: str) -> LanguageSettings | None:
        """Detect a language from *text*, limited to the configured codes."""
        code = detect_language(text, allowed=self.codes)
        return self._settings.get(code) if code is not None else None

    def words_per_second(self, value: str | None = None) -> WordsPerSecondConfig:
        """Return the speaking-rate bounds for a language."""
        return self.require(value).words_per_second

    def subtitle_font(self, value: str | None = None) -> str:
        """Return the subtitle font family for a language."""
        return self.require(value).subtitle_font

    def hashtag_style(self, value: str | None = None) -> str:
        """Return the hashtag casing style for a language."""
        return self.require(value).hashtag_style

    def locale(self, value: str | None = None) -> str:
        """Return the locale for a language."""
        return self.require(value).locale

    def name(self, value: str | None = None) -> str:
        """Return the display name for a language."""
        return self.require(value).name

    def direction(self, value: str | None = None) -> TextDirection:
        """Return the writing direction for a language."""
        return self.require(value).direction


def resolve_language(
    config: LanguagesConfig | Mapping[str, Any],
    language: str | None = None,
) -> LanguageSettings:
    """Resolve *language* against *config*, falling back to its default."""
    return LanguageResolver(config).resolve(language)


def words_per_second(
    config: LanguagesConfig | Mapping[str, Any],
    language: str | None = None,
) -> WordsPerSecondConfig:
    """Return the configured speaking-rate bounds for a language."""
    return LanguageResolver(config).words_per_second(language)


def subtitle_font(
    config: LanguagesConfig | Mapping[str, Any],
    language: str | None = None,
) -> str:
    """Return the configured subtitle font for a language."""
    return LanguageResolver(config).subtitle_font(language)


def hashtag_style(
    config: LanguagesConfig | Mapping[str, Any],
    language: str | None = None,
) -> str:
    """Return the configured hashtag style for a language."""
    return LanguageResolver(config).hashtag_style(language)


def language_locale(
    config: LanguagesConfig | Mapping[str, Any],
    language: str | None = None,
) -> str:
    """Return the configured locale for a language."""
    return LanguageResolver(config).locale(language)


def language_name(
    config: LanguagesConfig | Mapping[str, Any],
    language: str | None = None,
) -> str:
    """Return the configured display name for a language."""
    return LanguageResolver(config).name(language)

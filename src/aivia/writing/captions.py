"""Generate platform and language aware captions from a written script.

A caption is more than the script title. It has to fit the target platform's
caption, description, title and hashtag limits, follow the language's hashtag
casing, avoid the language's forbidden phrases and still carry the two pieces
of provenance a published item needs: a source line and an AI disclosure.

This module builds that package deterministically from the script, the platform
data and the language configuration. It never calls a provider or the network,
so the same script, platform and language always produce the same captions. The
generator truncates only when a platform limit requires it and records that it
did, so a caller can tell a short caption from a cut one.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from aivia.core.config.sections import LanguagesConfig
from aivia.core.errors import ConfigError, ContentPolicyError
from aivia.core.platforms import (
    CaptionLimits,
    Platform,
    PlatformCatalog,
    get_platform,
)
from aivia.llm.languages import (
    DEFAULT_LANGUAGE,
    LanguageResolver,
    normalize_language_code,
)
from aivia.writing.multilingual import find_forbidden_phrases
from aivia.writing.schema import Script

__all__ = [
    "DEFAULT_AI_DISCLOSURE",
    "DEFAULT_HASHTAG_STYLE",
    "DEFAULT_MAX_DERIVED_TAGS",
    "DEFAULT_SOURCES_LABEL",
    "DEFAULT_SOURCE_LABEL",
    "CaptionBuilder",
    "CaptionFactory",
    "CaptionGenerator",
    "CaptionOutput",
    "CaptionResult",
    "CaptionSet",
    "Captions",
    "apply_hashtag_style",
    "build_caption_set",
    "build_captions",
    "build_source_line",
    "caption_set",
    "captions_for",
    "format_caption",
    "format_source_line",
    "generate_captions",
    "make_captions",
    "normalise_hashtags",
    "normalise_tags",
    "normalize_hashtags",
    "normalize_tags",
    "source_line_for",
]

DEFAULT_AI_DISCLOSURE = "This content was generated with AI."
"""The disclosure used when a platform requires one but supplies no text."""

DEFAULT_SOURCE_LABEL = "Source:"
DEFAULT_SOURCES_LABEL = "Sources:"
DEFAULT_HASHTAG_STYLE = "lower"
DEFAULT_MAX_DERIVED_TAGS = 8

_WHITESPACE = re.compile(r"\s+", re.UNICODE)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+", re.UNICODE)
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_TAG_SPLIT = re.compile(r"[^\w]+", re.UNICODE)
_ELLIPSIS = "…"

_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "after",
        "again",
        "all",
        "also",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
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
        "into",
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
        "out",
        "she",
        "should",
        "so",
        "some",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "too",
        "up",
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
        "you",
        "your",
    }
)


def _clean(value: Any) -> str:
    return _WHITESPACE.sub(" ", value).strip() if isinstance(value, str) else ""


def _sentence(value: str) -> str:
    text = _clean(value)
    if text and text[-1] not in ".!?":
        return f"{text}."
    return text


def _unique(values: Iterable[str]) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean(value)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            items.append(text)
    return items


def _truncate(text: str, limit: int | None) -> str:
    if limit is None or len(text) <= limit:
        return text
    if limit is not None and not limit:
        return ""
    return text[: limit - len(_ELLIPSIS)].rstrip() + _ELLIPSIS


def _join(parts: Iterable[str]) -> str:
    return " ".join(part for part in (_clean(value) for value in parts) if part)


def _fit(parts: Sequence[str], limit: int | None) -> tuple[str, bool]:
    """Join *parts* under *limit*, keeping the trailing provenance first.

    The disclosure is the last part and the source line comes before it, so
    shortening starts at the main text and only reaches the provenance when the
    provenance itself cannot fit.
    """
    full = _join(parts)
    if limit is None or len(full) <= limit:
        return full, False
    kept: list[str] = []
    used = 0
    for part in reversed(parts):
        text = _clean(part)
        if not text:
            continue
        separator = 1 if kept else 0
        if used + separator + len(text) <= limit:
            kept.insert(0, text)
            used += separator + len(text)
            continue
        room = limit - used - separator
        if room > 0:
            kept.insert(0, _truncate(text, room))
        break
    return _join(kept), True


def _values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return []
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return [item for item in value if isinstance(item, str)]
    return []


def _payload_value(payload: Any, *keys: str) -> Any:
    if isinstance(payload, Mapping):
        for key in keys:
            if key in payload:
                return payload[key]
        return None
    for key in keys:
        value = getattr(payload, key, None)
        if value is not None:
            return value
    return None


def _script_sources(script: Script, explicit: Sequence[str] | None) -> list[str]:
    if explicit is not None:
        return _unique(explicit)
    candidates: list[str] = []
    payload = getattr(script, "payload", None)
    for key in ("source_urls", "sources", "urls"):
        candidates.extend(_values(_payload_value(payload, key)))
    facts = _payload_value(payload, "facts", "claims")
    if isinstance(facts, Iterable) and not isinstance(facts, (str, bytes, bytearray)):
        for fact in facts:
            for key in ("source_urls", "sources", "urls", "source_url", "source"):
                candidates.extend(_values(_payload_value(fact, key)))
    for key in ("source_urls", "sources", "urls"):
        candidates.extend(_values(getattr(script, key, None)))
    return _unique(candidates)


def build_source_line(sources: Sequence[str] | None) -> str:
    """Return the source line for *sources*, or an empty string without any."""
    urls = _unique(sources or ())
    if not urls:
        return ""
    label = DEFAULT_SOURCE_LABEL if len(urls) == 1 else DEFAULT_SOURCES_LABEL
    return f"{label} {', '.join(urls)}"


format_source_line = build_source_line
source_line_for = build_source_line


def apply_hashtag_style(value: str, style: str = DEFAULT_HASHTAG_STYLE) -> str:
    """Return *value* as one hashtag word in the requested casing style."""
    words = [word for word in _TAG_SPLIT.split(_clean(value)) if word]
    if not words:
        return ""
    if style == "upper":
        return "".join(word.upper() for word in words)
    if style == "title":
        return "".join(word[:1].upper() + word[1:] for word in words)
    return "".join(words).lower()


def _tag_value(value: str, style: str) -> str:
    return apply_hashtag_style(value, style)


def _normalise_tags(
    values: Sequence[str] | None,
    *,
    style: str,
    limit: int | None,
    forbidden: Sequence[str],
) -> list[str]:
    if limit is not None and not limit:
        return []
    tags: list[str] = []
    seen: set[str] = set()
    for raw in values or ():
        if find_forbidden_phrases(_clean(raw), forbidden):
            continue
        tag = _tag_value(raw, style)
        if not tag:
            continue
        if find_forbidden_phrases(tag, forbidden):
            continue
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        tags.append(tag)
        if limit is not None and len(tags) >= limit:
            break
    return tags


def normalise_tags(
    values: Sequence[str] | None,
    *,
    style: str = DEFAULT_HASHTAG_STYLE,
    limit: int | None = None,
    forbidden_phrases: Sequence[str] = (),
) -> list[str]:
    """Return plain tags with the language style applied and duplicates removed."""
    return _normalise_tags(
        values,
        style=style,
        limit=limit,
        forbidden=forbidden_phrases,
    )


normalize_tags = normalise_tags


def normalise_hashtags(
    values: Sequence[str] | None,
    *,
    style: str = DEFAULT_HASHTAG_STYLE,
    limit: int | None = None,
    forbidden_phrases: Sequence[str] = (),
) -> list[str]:
    """Return ``#`` prefixed hashtags with the language style applied."""
    tags = normalise_tags(
        values,
        style=style,
        limit=limit,
        forbidden_phrases=forbidden_phrases,
    )
    return [f"#{tag}" for tag in tags]


normalize_hashtags = normalise_hashtags


def _caption_text(script: Script, title: str) -> str:
    hook = _clean(getattr(script, "hook", ""))
    if hook:
        if title and title.casefold() not in hook.casefold():
            return _join([title, _sentence(hook)])
        return _sentence(hook)
    if title:
        return _sentence(title)
    for beat in getattr(script, "beats", ()):
        text = _clean(getattr(beat, "text", ""))
        if text:
            return _sentence(text)
    return ""


def _description_text(script: Script, title: str, caption: str) -> str:
    parts: list[str] = [caption]
    for beat in getattr(script, "beats", ()):
        text = _clean(getattr(beat, "text", ""))
        if text:
            parts.append(_sentence(text))
    outro = _clean(getattr(script, "outro", ""))
    if outro:
        parts.append(_sentence(outro))
    return _join(_unique(parts))


def _derived_tags(script: Script, title: str) -> list[str]:
    words: list[str] = []
    for source in (title, _clean(getattr(script, "hook", ""))):
        for word in _WORD.findall(source):
            key = word.casefold()
            if len(word) < 3 or key in _STOPWORDS or word.isdigit():
                continue
            words.append(word)
    return _unique(words)[:DEFAULT_MAX_DERIVED_TAGS]


def _without_forbidden(text: str, phrases: Sequence[str]) -> tuple[str, list[str]]:
    if not phrases or not text:
        return text, []
    kept: list[str] = []
    removed: list[str] = []
    for sentence in _SENTENCE_BREAK.split(text):
        found = find_forbidden_phrases(sentence, phrases)
        if found:
            removed.extend(found)
        elif sentence.strip():
            kept.append(sentence.strip())
    return _join(kept), _unique(removed)


def _resolve_platform(
    value: Platform | str | None,
    *,
    platforms: PlatformCatalog | Mapping[str, Any] | None,
    config_dir: Any,
) -> Platform:
    if isinstance(value, Platform):
        return value
    if value is None:
        raise ConfigError(
            "captions need a platform",
            details={"platform": None},
        )
    catalog: PlatformCatalog | None
    if platforms is None:
        catalog = None
    elif isinstance(platforms, PlatformCatalog):
        catalog = platforms
    elif isinstance(platforms, Mapping):
        catalog = PlatformCatalog.from_mapping(platforms)
    else:
        raise ConfigError(
            "platforms must be a catalog or a mapping",
            details={"platforms": type(platforms).__name__},
        )
    if catalog is None and not isinstance(value, str):
        raise ConfigError(
            "platform must be a name or a Platform",
            details={"platform": type(value).__name__},
        )
    return get_platform(value, catalog=catalog, config_dir=config_dir)


def _resolve_language(
    value: str | None,
    *,
    languages: LanguagesConfig | Mapping[str, Any] | LanguageResolver | None,
    fallback: str,
) -> tuple[str, str, tuple[str, ...]]:
    requested = _clean(value) or _clean(fallback) or DEFAULT_LANGUAGE
    if isinstance(languages, LanguageResolver):
        settings = languages.require(requested)
        return settings.code, settings.hashtag_style, settings.forbidden_phrases
    if languages is not None:
        resolver = LanguageResolver(languages)
        settings = resolver.require(requested)
        return settings.code, settings.hashtag_style, settings.forbidden_phrases
    try:
        code = normalize_language_code(requested)
    except Exception:
        code = requested
    return code, DEFAULT_HASHTAG_STYLE, ()


class CaptionSet(BaseModel):
    """The caption package generated for one platform and language.

    ``caption``, ``description``, ``title`` and the tag lists already obey the
    platform limits. ``source_line`` and ``disclosure`` are kept separately so
    a publisher can send them in metadata as well as in the text. ``truncated``
    says at least one field had to be cut, and ``warnings`` names what was cut
    or removed.
    """

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        validate_assignment=True,
    )

    platform: str = ""
    language: str = ""
    title: str = ""
    caption: str = Field(
        default="",
        validation_alias=AliasChoices("caption", "text", "body", "content"),
    )
    description: str = Field(
        default="",
        validation_alias=AliasChoices(
            "description",
            "summary",
            "desc",
            "long_caption",
        ),
    )
    hashtags: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("hashtags", "tag_list"),
    )
    tags: list[str] = Field(default_factory=list)
    sources: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("sources", "source_urls", "urls"),
    )
    source_line: str = Field(
        default="",
        validation_alias=AliasChoices("source_line", "source", "sources_line"),
    )
    disclosure: str = Field(
        default="",
        validation_alias=AliasChoices(
            "disclosure",
            "disclosure_text",
            "ai_disclosure",
        ),
    )
    disclosure_required: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "disclosure_required",
            "requires_disclosure",
        ),
    )
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list)
    limits: dict[str, int | None] = Field(default_factory=dict)

    @field_validator(
        "platform", "language", "title", "caption", "description", mode="before"
    )
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _clean(value)

    @field_validator("hashtags", "tags", "sources", mode="before")
    @classmethod
    def _coerce_list(cls, value: Any) -> Any:
        return _values(value)

    @field_validator("disclosure", "source_line", mode="before")
    @classmethod
    def _strip_provenance(cls, value: Any) -> Any:
        return _clean(value)

    @model_validator(mode="after")
    def _fill_tags(self) -> CaptionSet:
        if not self.tags and self.hashtags:
            self.tags = [tag.lstrip("#") for tag in self.hashtags]
        if not self.hashtags and self.tags:
            self.hashtags = [f"#{tag}" for tag in self.tags]
        return self

    @property
    def text(self) -> str:
        """Alias for :attr:`caption`."""
        return self.caption

    @property
    def body(self) -> str:
        """Alias for :attr:`caption`."""
        return self.caption

    @property
    def tag_list(self) -> list[str]:
        """Alias for :attr:`hashtags`."""
        return self.hashtags

    @property
    def source(self) -> str:
        """Alias for :attr:`source_line`."""
        return self.source_line

    @property
    def ai_disclosure(self) -> str:
        """Alias for :attr:`disclosure`."""
        return self.disclosure

    @property
    def within_limits(self) -> bool:
        """Whether every generated field stayed inside its platform limit."""
        checks = (
            (self.caption, self.limits.get("max_length")),
            (self.description, self.limits.get("max_description_length")),
            (self.title, self.limits.get("max_title_length")),
        )
        for text, limit in checks:
            if limit is not None and len(text) > limit:
                return False
        hashtag_limit = self.limits.get("max_hashtags")
        return not (hashtag_limit is not None and len(self.hashtags) > hashtag_limit)

    def to_dict(self) -> dict[str, Any]:
        """Return the caption package as a JSON-compatible mapping."""
        return self.model_dump(mode="json")


CaptionResult = CaptionSet
CaptionOutput = CaptionSet
Captions = CaptionSet
CaptionPackage = CaptionSet


class CaptionGenerator:
    """Build captions for a script from platform data and language settings.

    A platform may be passed as a :class:`Platform`, as its name when a catalog
    is supplied, or as a name resolved from the packaged platform data. The
    language configuration supplies hashtag casing and forbidden phrases; when
    it is omitted the language code is kept and the defaults are used.
    """

    def __init__(
        self,
        platform: Platform | str | None = None,
        *,
        platforms: PlatformCatalog | Mapping[str, Any] | None = None,
        language: str | None = None,
        languages: LanguagesConfig | Mapping[str, Any] | LanguageResolver | None = None,
        sources: Sequence[str] | None = None,
        hashtags: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
        disclosure: str | None = None,
        include_disclosure: bool = True,
        config_dir: Any = None,
    ) -> None:
        self.platform = _resolve_platform(
            platform,
            platforms=platforms,
            config_dir=config_dir,
        )
        self.language = language
        self.languages = languages
        self.sources = None if sources is None else list(sources)
        self.hashtags = None if hashtags is None else list(hashtags)
        self.tags = None if tags is None else list(tags)
        self.disclosure = disclosure
        self.include_disclosure = include_disclosure

    @property
    def caption_limits(self) -> CaptionLimits:
        """The caption limits of the selected platform."""
        return self.platform.caption_limits

    def _disclosure_text(self, language: str) -> str:
        if self.disclosure is not None:
            return _clean(self.disclosure)
        if not self.include_disclosure:
            return ""
        text = _clean(self.platform.disclosure.text)
        if text:
            return text
        if self.platform.disclosure.required:
            return DEFAULT_AI_DISCLOSURE
        return ""

    def _tags(
        self,
        script: Script,
        title: str,
        *,
        style: str,
        limit: int | None,
        forbidden: Sequence[str],
    ) -> list[str]:
        explicit = self.hashtags if self.hashtags is not None else self.tags
        if explicit is None:
            explicit = _derived_tags(script, title)
        return normalise_tags(
            explicit,
            style=style,
            limit=limit,
            forbidden_phrases=forbidden,
        )

    def build(self, script: Script) -> CaptionSet:
        """Return the caption package for *script* without changing it."""
        limits = self.caption_limits
        language, style, forbidden = _resolve_language(
            self.language,
            languages=self.languages,
            fallback=getattr(script, "language", ""),
        )
        sources = _script_sources(script, self.sources)
        source_line = build_source_line(sources)
        disclosure = self._disclosure_text(language)
        description_limit = limits.max_description_length
        if description_limit is None:
            description_limit = limits.max_length

        warnings: list[str] = []
        raw_title, removed = _without_forbidden(
            _clean(getattr(script, "title", "")),
            forbidden,
        )
        if removed:
            warnings.append("forbidden phrases removed from title")
        title = _truncate(raw_title, limits.max_title_length)
        if title != raw_title:
            warnings.append("title truncated")

        caption_body = _caption_text(script, title)
        description_body = _description_text(script, title, caption_body)
        caption_body, removed = _without_forbidden(caption_body, forbidden)
        if removed:
            warnings.append("forbidden phrases removed from caption")
        description_body, removed = _without_forbidden(description_body, forbidden)
        if removed:
            warnings.append("forbidden phrases removed from description")

        caption, cut = _fit([caption_body, source_line, disclosure], limits.max_length)
        if cut:
            warnings.append("caption truncated")
        description, cut = _fit(
            [description_body, source_line, disclosure],
            description_limit,
        )
        if cut:
            warnings.append("description truncated")

        tags = self._tags(
            script,
            title,
            style=style,
            limit=limits.max_hashtags,
            forbidden=forbidden,
        )
        hashtags = [f"#{tag}" for tag in tags]

        return CaptionSet(
            platform=self.platform.id,
            language=language,
            title=title,
            caption=caption,
            description=description,
            hashtags=hashtags,
            tags=tags,
            sources=sources,
            source_line=source_line,
            disclosure=disclosure,
            disclosure_required=self.platform.disclosure.required,
            truncated=bool(warnings and any("truncated" in item for item in warnings)),
            warnings=warnings,
            limits={
                "max_length": limits.max_length,
                "max_title_length": limits.max_title_length,
                "max_description_length": description_limit,
                "max_hashtags": limits.max_hashtags,
            },
        )

    generate = build
    apply = build
    __call__ = build
    captions_for = build
    build_captions = build

    def build_strict(self, script: Script) -> CaptionSet:
        """Build captions and refuse a forbidden phrase that survived filtering."""
        result = self.build(script)
        _, _, forbidden = _resolve_language(
            self.language,
            languages=self.languages,
            fallback=getattr(script, "language", ""),
        )
        found = find_forbidden_phrases(
            _join([result.caption, result.description, *result.tags]),
            forbidden,
        )
        if found:
            raise ContentPolicyError(
                "caption still contains a forbidden phrase",
                details={
                    "phrases": list(found),
                    "platform": result.platform,
                    "language": result.language,
                },
            )
        return result

    enforce = build_strict
    require = build_strict


CaptionBuilder = CaptionGenerator
CaptionFactory = CaptionGenerator
CaptionWriter = CaptionGenerator


def build_captions(
    script: Script,
    platform: Platform | str | None = None,
    **kwargs: Any,
) -> CaptionSet:
    """Return the caption package for *script* on *platform*."""
    return CaptionGenerator(platform, **kwargs).build(script)


generate_captions = build_captions
build_caption_set = build_captions
caption_set = build_captions
captions_for = build_captions
make_captions = build_captions
format_caption = build_captions

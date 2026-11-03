"""The quote writing mode: a verbatim quote with a trustworthy attribution.

A quote is only usable when it is both present and attributed. The mode never
invents an author: a named person must be backed by at least one source URL,
while genuinely anonymous or traditional sayings are accepted as such. Building
is deterministic and offline, so the writing stage can later rewrite the prose
without changing the quote or its provenance.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from aivia.core.errors import ConfigError
from aivia.core.models.script import Beat, DialogueLine
from aivia.writing.modes import (
    BuiltinWritingMode,
    register_builtin_writing_mode,
)
from aivia.writing.schema import QuotePayload, Script, WritingMode

__all__ = [
    "ANONYMOUS_ATTRIBUTIONS",
    "Quote",
    "QuoteInput",
    "QuoteMode",
    "QuoteWritingMode",
    "build_quote_script",
    "coerce_quote",
    "extract_quote",
    "is_anonymous_attribution",
    "is_real_person_attribution",
    "looks_like_real_person",
    "normalise_quote",
    "normalize_quote",
    "parse_quote",
    "quote_script",
    "validate_attribution",
    "validate_quote",
]

QuoteInput = Any
"""Any value accepted as a quote: text, a mapping, a payload or an object."""

_QUOTE_KEYS = ("quote", "text", "quote_text", "quotation", "content", "line")
_ATTRIBUTION_KEYS = ("attribution", "author", "speaker", "credit", "attributed_to")
_URL_KEYS = ("source_urls", "sources", "urls", "source_url", "source")
_CONTEXT_KEYS = ("context", "background", "occasion")
_TITLE_KEYS = ("title", "name", "subject")
_LANGUAGE_KEYS = ("language", "lang", "locale")
_HOOK_KEYS = ("hook", "opening", "intro")
_OUTRO_KEYS = ("outro", "closing", "ending")
_PAYLOAD_KEYS = ("payload", "mode_payload", "mode_data", "data", "options")
_MISSING = object()

_URL = re.compile(r"^(?:https?://|www\.)", re.IGNORECASE)
_BY_PREFIX = re.compile(
    r"^(?:by|attributed\s+to|credit|source|author)\s*:?\s+",
    re.IGNORECASE,
)
_INLINE_ATTRIBUTION = re.compile(r"\s+(?:--|\u2014|\u2013|~|-)\s+")
_TRAILING_ATTRIBUTION = re.compile(r"^(?:--|\u2014|\u2013|~|-)\s*(.+)$")
_TRAILING_YEAR = re.compile(r"[\s,]+(?:1[0-9]{3}|20[0-9]{2})\s*$")

_QUOTE_MARKS = {
    '"': '"',
    "'": "'",
    "\u201c": "\u201d",
    "\u2018": "\u2019",
    "\u00ab": "\u00bb",
    "\u201e": "\u201c",
}

ANONYMOUS_ATTRIBUTIONS = frozenset(
    {
        "anon",
        "anonymous",
        "anonymous author",
        "anonymous source",
        "author unknown",
        "common saying",
        "folk saying",
        "folklore",
        "internet",
        "meme",
        "no attribution",
        "not attributed",
        "proverb",
        "proverbs",
        "saying",
        "traditional",
        "traditional proverb",
        "unattributed",
        "unknown",
        "unknown author",
        "unknown source",
    }
)
"""Attributions that name no real person and therefore need no source URL."""

_ANONYMOUS_MARKERS = (
    "anonymous",
    "unattributed",
    "unknown",
    "traditional",
    "proverb",
    "folk",
    "anon",
)

_NON_PERSON_WORDS = frozenset(
    {
        "agency",
        "association",
        "bible",
        "blog",
        "book",
        "broadcast",
        "channel",
        "college",
        "committee",
        "company",
        "corporation",
        "council",
        "daily",
        "department",
        "editor",
        "editors",
        "film",
        "foundation",
        "government",
        "group",
        "history",
        "inc",
        "institute",
        "internet",
        "journal",
        "life",
        "literature",
        "ltd",
        "magazine",
        "ministry",
        "movie",
        "nature",
        "newspaper",
        "news",
        "novel",
        "philosophy",
        "play",
        "poem",
        "post",
        "press",
        "quran",
        "radio",
        "review",
        "scripture",
        "society",
        "song",
        "someone",
        "somebody",
        "source",
        "team",
        "television",
        "times",
        "torah",
        "tv",
        "university",
        "weekly",
        "wikipedia",
    }
)


def _value_of(value: Any, keys: Sequence[str]) -> Any:
    if isinstance(value, Mapping):
        for key in keys:
            if key in value:
                return value[key]
        return _MISSING
    for key in keys:
        if hasattr(value, key):
            return getattr(value, key)
    return _MISSING


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return "\n".join(str(item).strip() for item in value if str(item).strip())
    return ""


def _is_url(value: str) -> bool:
    return bool(_URL.match(value.strip()))


def _urls(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        candidates: Iterable[Any] = [value]
    elif isinstance(value, Mapping):
        url = value.get("url") or value.get("source_url")
        candidates = [] if url is None else [url]
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        candidates = value
    else:
        return []
    urls: list[str] = []
    for raw in candidates:
        if not isinstance(raw, str):
            continue
        url = raw.strip()
        if url and url not in urls:
            urls.append(url)
    return urls


def _strip_quote_marks(text: str) -> str:
    cleaned = text.strip()
    if len(cleaned) >= 2:
        opening = cleaned[0]
        closing = _QUOTE_MARKS.get(opening)
        if closing is not None and cleaned[-1] == closing:
            return cleaned[1:-1].strip()
    return cleaned


def _parse_text(text: str) -> QuotePayload:
    cleaned = text.strip()
    if not cleaned:
        return QuotePayload()

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if len(lines) > 1:
        match = _TRAILING_ATTRIBUTION.match(lines[-1])
        if match is not None:
            quote = "\n".join(lines[:-1]).strip()
            if quote:
                return QuotePayload(
                    quote=_strip_quote_marks(quote),
                    attribution=match.group(1).strip(),
                )

    matches = list(_INLINE_ATTRIBUTION.finditer(cleaned))
    if matches:
        last = matches[-1]
        quote = cleaned[: last.start()].strip()
        attribution = cleaned[last.end() :].strip()
        if quote and attribution:
            return QuotePayload(
                quote=_strip_quote_marks(quote),
                attribution=attribution,
            )

    return QuotePayload(quote=_strip_quote_marks(cleaned))


def _quote_from(value: Any) -> QuotePayload:
    if isinstance(value, QuotePayload):
        return value
    if isinstance(value, Script) and isinstance(value.payload, QuotePayload):
        return value.payload
    if isinstance(value, str):
        return _parse_text(value)
    if isinstance(value, (bytes, bytearray)) or value is None:
        return QuotePayload()

    if isinstance(value, Mapping):
        nested = _value_of(value, _PAYLOAD_KEYS)
        if nested is not _MISSING and nested is not None:
            payload = _quote_from(nested)
            return _merge_metadata(payload, value)
        return _quote_from_mapping(value)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) == 2 and all(isinstance(item, str) for item in value):
            return QuotePayload(
                quote=_strip_quote_marks(value[0]),
                attribution=value[1].strip(),
            )
        return _quote_from_mapping({_QUOTE_KEYS[0]: list(value)})

    if _value_of(value, _QUOTE_KEYS) is not _MISSING:
        return _quote_from_mapping(
            {
                key: getattr(value, key)
                for key in (
                    *_QUOTE_KEYS,
                    *_ATTRIBUTION_KEYS,
                    *_URL_KEYS,
                    *_CONTEXT_KEYS,
                )
                if hasattr(value, key)
            }
        )

    raise ConfigError(
        f"cannot build a quote from {type(value).__name__}",
        details={"value": repr(value)},
    )


def _merge_metadata(payload: QuotePayload, value: Mapping[str, Any]) -> QuotePayload:
    attribution = payload.attribution or _text(_value_of(value, _ATTRIBUTION_KEYS))
    urls = list(payload.source_urls)
    for url in _urls(_value_of(value, _URL_KEYS)):
        if url not in urls:
            urls.append(url)
    context = payload.context or _text(_value_of(value, _CONTEXT_KEYS))
    return QuotePayload(
        quote=payload.quote,
        attribution=attribution,
        source_urls=urls,
        context=context,
    )


def _quote_from_mapping(value: Mapping[str, Any]) -> QuotePayload:
    quote = _text(_value_of(value, _QUOTE_KEYS))
    attribution = _text(_value_of(value, _ATTRIBUTION_KEYS))
    urls = _urls(_value_of(value, _URL_KEYS))
    context = _text(_value_of(value, _CONTEXT_KEYS))

    raw_source = _value_of(value, ("source",))
    if raw_source is not _MISSING:
        source = _text(raw_source)
        if source and _is_url(source):
            if source not in urls:
                urls.append(source)
            if attribution.casefold() == source.casefold():
                attribution = ""

    if not attribution and quote:
        parsed = _parse_text(quote)
        if parsed.attribution:
            quote = parsed.quote
            attribution = parsed.attribution

    return QuotePayload(
        quote=quote,
        attribution=attribution,
        source_urls=urls,
        context=context,
    )


def extract_quote(value: QuoteInput = None) -> QuotePayload:
    """Return *value* as a typed :class:`QuotePayload`.

    A plain string is read as the quote itself, with a trailing ``-- Author`` or
    ``— Author`` line or suffix split off as the attribution. Mappings accept
    the schema aliases (``text``, ``author``, ``sources`` and so on).
    """
    return _quote_from(value)


coerce_quote = extract_quote
parse_quote = extract_quote
normalize_quote = extract_quote
normalise_quote = extract_quote


def _normalise_attribution(attribution: str) -> str:
    cleaned = _BY_PREFIX.sub("", attribution.strip())
    cleaned = " ".join(cleaned.split())
    return cleaned.strip(" .,:;!?\"'()[]").casefold()


def is_anonymous_attribution(attribution: str) -> bool:
    """Return whether *attribution* names no real person.

    Anonymous, unknown, traditional and proverb attributions are legitimate
    without a source URL; a named person is not.
    """
    key = _normalise_attribution(attribution)
    if not key:
        return False
    if key in ANONYMOUS_ATTRIBUTIONS:
        return True
    return any(marker in key.split() for marker in _ANONYMOUS_MARKERS)


def _name_words(text: str) -> list[str]:
    return re.findall(r"[^\W\d_][\w'\u2019.-]*", text, re.UNICODE)


def _looks_like_person(attribution: str) -> bool:
    text = _BY_PREFIX.sub("", attribution.strip()).strip(" .,:;!?\"'()[]")
    if not text or _is_url(text) or is_anonymous_attribution(text):
        return False
    text = _TRAILING_YEAR.sub("", text).strip()
    words = _name_words(text)
    if not words or len(words) > 5:
        return False
    lowered = {word.casefold().strip(".") for word in words}
    if lowered & _NON_PERSON_WORDS:
        return False
    return all(
        word[0].isupper() or (len(word) <= 3 and word.isupper()) for word in words
    )


def looks_like_real_person(attribution: str) -> bool:
    """Return whether *attribution* looks like the name of a real person."""
    return _looks_like_person(attribution)


is_real_person_attribution = looks_like_real_person


def validate_attribution(
    attribution: str,
    source_urls: Iterable[str] = (),
) -> str:
    """Validate an attribution and return it unchanged.

    An empty attribution is rejected. A named person needs at least one source
    URL so the writer cannot invent an attribution to a real person; anonymous
    and traditional sayings are accepted without one.
    """
    text = attribution.strip()
    if not text:
        raise ConfigError(
            "quote mode requires an attribution",
            details={"reason": "attribution_missing"},
        )
    urls = [url for url in source_urls if url]
    if urls:
        return text
    if is_anonymous_attribution(text):
        return text
    if _looks_like_person(text):
        raise ConfigError(
            "quote mode forbids invented attribution to a real person without a source",
            details={
                "reason": "attribution_unverified",
                "attribution": text,
            },
        )
    return text


def validate_quote(value: QuoteInput = None) -> QuotePayload:
    """Return *value* as a payload and check its quote and attribution."""
    payload = extract_quote(value)
    if not payload.quote:
        raise ConfigError(
            "quote mode needs a quote",
            details={"mode": WritingMode.QUOTE.value},
        )
    validate_attribution(payload.attribution, payload.source_urls)
    return payload


def _title_of(value: Any) -> str:
    return _text(_value_of(value, _TITLE_KEYS))


def _language_of(value: Any, default: str) -> str:
    language = _text(_value_of(value, _LANGUAGE_KEYS))
    return language or default


def _optional_text(
    value: Any,
    keys: Sequence[str],
    explicit: str | None,
) -> str | None:
    if explicit is not None:
        return explicit.strip() or None
    return _text(_value_of(value, keys)) or None


class QuoteMode(BuiltinWritingMode):
    """The quote writing mode.

    :meth:`build_script` turns a quote and its attribution into a script whose
    hook and beat carry the quote and whose outro credits the author. The mode
    refuses a missing attribution and an unverified attribution to a real
    person, so a quote card can never silently invent who said it.
    """

    name = "quote"
    description = "A verbatim quotation with its source and attribution."
    instructions = (
        "Keep the quotation verbatim, credit the named source and never "
        "attribute words to a real person without a verifiable source."
    )

    def build_payload(self, value: QuoteInput = None) -> QuotePayload:
        """Validate *value* as a :class:`QuotePayload` without checking credit."""
        return extract_quote(value)

    def build_hook(
        self,
        value: QuoteInput = None,
        *,
        title: str = "",
        hook: str | None = None,
    ) -> str:
        """Return the opening line, using an explicit *hook* when given."""
        if hook is not None:
            return hook.strip()
        payload = extract_quote(value)
        if not payload.quote:
            return ""
        subject = title.strip()
        if subject:
            return f"{subject}: {payload.quote}"
        return payload.quote

    def build_beats(
        self,
        value: QuoteInput = None,
        *,
        speaker: str = "narrator",
        max_beats: int | None = None,
    ) -> list[Beat]:
        """Return one beat carrying the quote, narrated by *speaker*."""
        payload = extract_quote(value)
        if not payload.quote:
            return []
        if max_beats is not None and not max_beats:
            return []
        return [
            Beat(
                index=0,
                text=payload.quote,
                dialogue=[DialogueLine(speaker=speaker, text=payload.quote)],
            )
        ]

    def build_outro(
        self,
        value: QuoteInput = None,
        *,
        outro: str | None = None,
    ) -> str:
        """Return the closing credit, using an explicit *outro* when given."""
        if outro is not None:
            return outro.strip()
        payload = extract_quote(value)
        if not payload.attribution:
            return ""
        return f"— {payload.attribution}"

    def build_script(
        self,
        value: QuoteInput = None,
        *,
        title: str | None = None,
        language: str | None = None,
        context: str = "",
        attribution: str | None = None,
        source_urls: Iterable[str] | None = None,
        hook: str | None = None,
        outro: str | None = None,
        speaker: str = "narrator",
        max_beats: int | None = None,
    ) -> Script:
        """Build a quote script from user text, a mapping or a payload."""
        payload = extract_quote(value)
        if not payload.quote:
            raise ConfigError(
                "quote mode needs a quote",
                details={"mode": self.name},
            )
        resolved_attribution = (
            attribution.strip() if attribution is not None else payload.attribution
        )
        resolved_urls = list(payload.source_urls)
        for url in source_urls or ():
            if url and url not in resolved_urls:
                resolved_urls.append(url)
        validate_attribution(resolved_attribution, resolved_urls)
        resolved_title = title if title is not None else _title_of(value)
        resolved_language = language or _language_of(value, "en")
        resolved_context = context or payload.context
        payload = QuotePayload(
            quote=payload.quote,
            attribution=resolved_attribution,
            source_urls=resolved_urls,
            context=resolved_context,
        )
        return Script(
            mode=WritingMode.QUOTE,
            title=resolved_title,
            language=resolved_language,
            hook=self.build_hook(
                payload,
                title=resolved_title,
                hook=_optional_text(value, _HOOK_KEYS, hook),
            ),
            beats=self.build_beats(
                payload,
                speaker=speaker,
                max_beats=max_beats,
            ),
            outro=self.build_outro(
                payload,
                outro=_optional_text(value, _OUTRO_KEYS, outro),
            ),
            payload=payload,
        )

    script = build_script
    make_script = build_script
    build_from_quote = build_script


QuoteWritingMode = QuoteMode
Quote = QuoteMode


def build_quote_script(
    value: QuoteInput = None,
    **kwargs: Any,
) -> Script:
    """Build a quote script with a default :class:`QuoteMode`."""
    return QuoteMode().build_script(value, **kwargs)


quote_script = build_quote_script


register_builtin_writing_mode(WritingMode.QUOTE, QuoteMode)

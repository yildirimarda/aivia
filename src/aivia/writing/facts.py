"""The facts writing mode: turn a fact list into a script.

A fact list may be supplied as user text, as plain mappings or as the research
module's output. The latter is read structurally through ``claims`` and each
claim's ``text`` and ``source_urls``; this module never imports research, so the
writing stage works with the core fact contract alone.

Building is deterministic and offline. The hook opens the script, one beat
carries each fact and the outro closes it. No model is called here: the writing
stage can rewrite the result later, but the structure and provenance are stable.
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
from aivia.writing.schema import (
    FactsPayload,
    Script,
    ScriptFact,
    WritingMode,
)

__all__ = [
    "FactInput",
    "FactMode",
    "FactWritingMode",
    "Facts",
    "FactsMode",
    "FactsWritingMode",
    "build_facts_script",
    "coerce_facts",
    "extract_facts",
    "facts_script",
    "normalise_facts",
    "normalize_facts",
    "parse_facts",
]

FactInput = Any
"""Any value accepted as a fact list: text, claims, mappings or sequences."""

_TEXT_KEYS = ("text", "statement", "claim", "fact", "content")
_URL_KEYS = ("source_urls", "sources", "urls", "source_url", "source")
_TITLE_KEYS = ("title", "name", "topic", "subject")
_CLAIM_KEYS = ("claims", "facts")
_ANGLE_KEYS = ("angle", "perspective", "viewpoint")
_NOTES_KEYS = ("notes", "note", "summary")
_HOOK_KEYS = ("hook", "opening", "intro")
_OUTRO_KEYS = ("outro", "closing", "ending")
_LANGUAGE_KEYS = ("language", "lang", "locale")
_PAYLOAD_KEYS = ("payload", "mode_payload", "mode_data", "data")
_MISSING = object()
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


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
    return value.strip() if isinstance(value, str) else ""


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


def _normalise_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def _sentence(text: str) -> str:
    cleaned = text.strip()
    if cleaned and cleaned[-1] not in ".!?":
        return f"{cleaned}."
    return cleaned


def _split_user_text(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines()]
    lines = [_BULLET.sub("", line).strip() for line in lines if line.strip()]
    if len(lines) > 1:
        return lines
    if not lines:
        return []
    return [part.strip() for part in _SENTENCE_BREAK.split(lines[0]) if part.strip()]


def _fact_from(value: Any) -> ScriptFact | None:
    if isinstance(value, ScriptFact):
        return value
    if isinstance(value, str):
        text = value.strip()
        return ScriptFact(text=text) if text else None
    if value is None or isinstance(value, (bytes, bytearray)):
        return None
    text = _value_of(value, _TEXT_KEYS)
    if text is _MISSING or not isinstance(text, str) or not text.strip():
        return None
    urls = _urls(_value_of(value, _URL_KEYS))
    confidence = _value_of(value, ("confidence",))
    verified = _value_of(value, ("verified",))
    if confidence is _MISSING and verified is _MISSING:
        return ScriptFact(text=text.strip(), source_urls=urls)
    if confidence is _MISSING:
        return ScriptFact(
            text=text.strip(),
            source_urls=urls,
            verified=bool(verified),
        )
    if verified is _MISSING:
        return ScriptFact(
            text=text.strip(),
            source_urls=urls,
            confidence=confidence,
        )
    return ScriptFact(
        text=text.strip(),
        source_urls=urls,
        confidence=confidence,
        verified=bool(verified),
    )


def _iter_facts(value: Any) -> Iterable[ScriptFact]:
    if value is None:
        return
    if isinstance(value, ScriptFact):
        yield value
        return
    if isinstance(value, str):
        for text in _split_user_text(value):
            yield ScriptFact(text=text)
        return
    if isinstance(value, (bytes, bytearray)):
        return
    if isinstance(value, FactsPayload):
        yield from value.facts
        return
    if isinstance(value, Mapping):
        payload = _value_of(value, _PAYLOAD_KEYS)
        if payload is not _MISSING:
            yield from _iter_facts(payload)
            return
        nested = _value_of(value, _CLAIM_KEYS)
        if nested is not _MISSING:
            yield from _iter_facts(nested)
            return
        fact = _fact_from(value)
        if fact is not None:
            yield fact
        return

    nested = _value_of(value, _CLAIM_KEYS)
    if nested is not _MISSING and not callable(nested):
        yield from _iter_facts(nested)
        return
    fact = _fact_from(value)
    if fact is not None:
        yield fact
        return
    if (
        _value_of(value, _TEXT_KEYS) is not _MISSING
        or _value_of(value, _URL_KEYS) is not _MISSING
    ):
        return
    if isinstance(value, Iterable):
        for item in value:
            yield from _iter_facts(item)
        return
    raise ConfigError(
        f"cannot build a fact list from {type(value).__name__}",
        details={"value": repr(value)},
    )


def _merge(facts: Iterable[ScriptFact]) -> list[ScriptFact]:
    merged: list[ScriptFact] = []
    positions: dict[str, int] = {}
    for fact in facts:
        key = _normalise_text(fact.text)
        if not key:
            continue
        position = positions.get(key)
        if position is None:
            positions[key] = len(merged)
            merged.append(fact)
            continue
        existing = merged[position]
        urls = list(existing.source_urls)
        for url in fact.source_urls:
            if url not in urls:
                urls.append(url)
        merged[position] = ScriptFact(
            text=existing.text,
            source_urls=urls,
            confidence=max(existing.confidence, fact.confidence),
            verified=existing.verified or fact.verified,
        )
    return merged


def extract_facts(value: FactInput = None) -> list[ScriptFact]:
    """Return *value* as a de-duplicated list of :class:`ScriptFact` records.

    A string is split into non-empty lines, or into sentences when it has no
    line breaks. Anything exposing ``claims`` (including a research fact list
    or a topic) is read structurally, so no research import is needed.
    """
    return _merge(_iter_facts(value))


coerce_facts = extract_facts
parse_facts = extract_facts
normalize_facts = extract_facts
normalise_facts = extract_facts


def _title_of(value: Any) -> str:
    return _text(_value_of(value, _TITLE_KEYS))


def _angle_of(value: Any) -> str:
    return _text(_value_of(value, _ANGLE_KEYS))


def _notes_of(value: Any) -> str:
    return _text(_value_of(value, _NOTES_KEYS))


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


def _source_urls(facts: Iterable[ScriptFact]) -> list[str]:
    urls: list[str] = []
    for fact in facts:
        for url in fact.source_urls:
            if url not in urls:
                urls.append(url)
    return urls


class FactsMode(BuiltinWritingMode):
    """The facts writing mode.

    :meth:`build_script` turns a fact list into a script with a hook, one beat
    per fact and an outro. :meth:`build_payload` keeps the base contract and
    returns the typed :class:`FactsPayload`, so research output can be handed to
    either method directly.
    """

    name = "facts"
    description = "A sourced fact list turned into a short narrated script."
    instructions = (
        "Rewrite each fact in your own words, keep every claim tied to its "
        "source and close with the strongest fact."
    )

    def build_payload(self, value: FactInput = None) -> FactsPayload:
        """Validate *value* as a :class:`FactsPayload` without importing research."""
        if isinstance(value, FactsPayload):
            return value
        facts = extract_facts(value)
        return FactsPayload(
            facts=facts,
            source_urls=_source_urls(facts),
            angle=_angle_of(value),
            notes=_notes_of(value),
        )

    def build_hook(
        self,
        facts: FactInput = None,
        *,
        title: str = "",
        angle: str = "",
        hook: str | None = None,
    ) -> str:
        """Return the opening line, using an explicit *hook* when given."""
        if hook is not None:
            return hook.strip()
        items = extract_facts(facts)
        if not items:
            return ""
        subject = title.strip() or angle.strip()
        first = _sentence(items[0].text)
        if subject:
            return f"{subject}: {first}"
        return first

    def build_beats(
        self,
        facts: FactInput = None,
        *,
        speaker: str = "narrator",
        max_beats: int | None = None,
    ) -> list[Beat]:
        """Return one beat per fact, each narrated by *speaker*."""
        items = extract_facts(facts)
        if max_beats is not None:
            items = items[:max_beats]
        return [
            Beat(
                index=index,
                text=fact.text,
                dialogue=[DialogueLine(speaker=speaker, text=fact.text)],
            )
            for index, fact in enumerate(items)
        ]

    def build_outro(
        self,
        facts: FactInput = None,
        *,
        title: str = "",
        outro: str | None = None,
    ) -> str:
        """Return the closing line, using an explicit *outro* when given."""
        if outro is not None:
            return outro.strip()
        if not extract_facts(facts):
            return ""
        if title.strip():
            return f"That is the story of {title.strip()}."
        return "That is the full picture."

    def build_script(
        self,
        facts: FactInput = None,
        *,
        title: str | None = None,
        language: str | None = None,
        angle: str = "",
        notes: str = "",
        hook: str | None = None,
        outro: str | None = None,
        speaker: str = "narrator",
        max_facts: int | None = None,
        max_beats: int | None = None,
    ) -> Script:
        """Build a facts script from user text or a structural fact list."""
        items = extract_facts(facts)
        if max_facts is not None:
            items = items[:max_facts]
        if not items:
            raise ConfigError(
                "facts mode needs at least one fact",
                details={"mode": self.name},
            )
        resolved_title = title if title is not None else _title_of(facts)
        resolved_language = language or _language_of(facts, "en")
        resolved_angle = angle or _angle_of(facts)
        resolved_notes = notes or _notes_of(facts)
        payload = FactsPayload(
            facts=items,
            source_urls=_source_urls(items),
            angle=resolved_angle,
            notes=resolved_notes,
        )
        return Script(
            mode=WritingMode.FACTS,
            title=resolved_title,
            language=resolved_language,
            hook=self.build_hook(
                items,
                title=resolved_title,
                angle=resolved_angle,
                hook=_optional_text(facts, _HOOK_KEYS, hook),
            ),
            beats=self.build_beats(
                items,
                speaker=speaker,
                max_beats=max_beats,
            ),
            outro=self.build_outro(
                items,
                title=resolved_title,
                outro=_optional_text(facts, _OUTRO_KEYS, outro),
            ),
            payload=payload,
        )

    script = build_script
    make_script = build_script
    build_from_facts = build_script


FactsWritingMode = FactsMode
FactWritingMode = FactsMode
FactMode = FactsMode
Facts = FactsMode


def build_facts_script(
    facts: FactInput = None,
    **kwargs: Any,
) -> Script:
    """Build a facts script with a default :class:`FactsMode`."""
    return FactsMode().build_script(facts, **kwargs)


facts_script = build_facts_script


register_builtin_writing_mode(WritingMode.FACTS, FactsMode)

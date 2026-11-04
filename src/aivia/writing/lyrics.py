"""The lyrics writing mode: named sections, rhyme, syllables and keywords.

Lyrics may arrive as plain text, a list of lines, a list of section mappings, a
mapping keyed by section name or an existing payload. Text headers such as
``[Verse 1]`` or ``Chorus:`` start a named section; blank lines separate
unlabelled sections. A requested ``section_counts`` map can repeat or trim a
section, and an optional keyword list is inserted into the chorus (or the last
section) when a keyword is missing.

The mode is deterministic and offline. No model is called here and nothing is
imported from research, so a lyric's section structure stays stable and
testable.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from aivia.core.errors import ConfigError
from aivia.core.models.script import Beat, DialogueLine
from aivia.writing.modes import (
    BuiltinWritingMode,
    register_builtin_writing_mode,
)
from aivia.writing.schema import (
    LyricsPayload,
    LyricsSection,
    Script,
    WritingMode,
)

__all__ = [
    "Lyrics",
    "LyricsInput",
    "LyricsMode",
    "LyricsPayload",
    "LyricsSection",
    "LyricsWritingMode",
    "apply_keywords",
    "build_lyrics_script",
    "check_syllable_budget",
    "coerce_lyrics",
    "count_sections",
    "count_syllables",
    "count_syllables_in_line",
    "derive_section_counts",
    "extract_lyrics",
    "inject_keywords",
    "insert_keywords",
    "lyrics_script",
    "normalise_lyrics",
    "normalise_section_counts",
    "normalize_lyrics",
    "normalize_section_counts",
    "parse_lyrics",
    "section_counts",
    "syllable_count",
    "syllables",
    "validate_lyrics",
]

LyricsInput = Any
"""Any value accepted as lyrics: text, lines, sections, a mapping or a payload."""

_SECTION_KEYS = ("sections", "section", "parts")
_LINE_KEYS = ("lines", "lyrics", "verses", "verse", "text", "content", "body")
_TEXT_KEYS = ("lyrics", "song", "text", "content", "body")
_COUNT_KEYS = ("section_counts", "section_count", "counts", "structure")
_RHYME_KEYS = ("rhyme_scheme", "rhyme", "scheme")
_SYLLABLE_KEYS = (
    "syllable_budget",
    "syllables",
    "syllable_count",
    "syllable_limit",
)
_KEYWORD_KEYS = ("keywords", "keyword", "terms")
_GENRE_KEYS = ("genre", "style", "kind")
_TEMPO_KEYS = ("tempo", "speed", "bpm")
_TITLE_KEYS = ("title", "name", "subject")
_LANGUAGE_KEYS = ("language", "lang", "locale")
_HOOK_KEYS = ("hook", "opening", "intro")
_OUTRO_KEYS = ("outro", "closing", "ending")
_PAYLOAD_KEYS = ("payload", "mode_payload", "mode_data", "data", "options")
_MISSING = object()

_SECTION_WORDS = {
    "verse",
    "chorus",
    "bridge",
    "hook",
    "pre-chorus",
    "prechorus",
    "pre chorus",
    "post-chorus",
    "postchorus",
    "post chorus",
    "refrain",
    "intro",
    "outro",
    "interlude",
    "breakdown",
    "coda",
    "tag",
    "vamp",
    "spoken",
    "rap",
}
_HEADER = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _/'-]{0,40}?)\s*:\s*(.*)$")
_BRACKET = re.compile(r"^\s*[\[\(\{]\s*([^\]\)\}]+?)\s*[\]\)\}]\s*$")
_BARE = re.compile(r"^\s*#*\s*([A-Za-z][A-Za-z0-9 _/'-]{0,40}?)\s*$")
_NUMBERED = re.compile(r"^(.*?)[\s_-]*\d+$")
_PLACEHOLDER = re.compile(
    r"\{\{?\s*(?:keyword|keywords|word|words)\s*\}?\}"
    r"|\[\s*(?:keyword|keywords|word|words)\s*\]"
    r"|<\s*(?:keyword|keywords|word|words)\s*>",
    re.IGNORECASE,
)
_VOWEL_GROUP = re.compile(r"[aeiouy]+", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z]+")


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
    return ""


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [
            item.strip() for item in value if isinstance(item, str) and item.strip()
        ]
    return []


def _keyword_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[,;\n]+", value)
        return [part.strip() for part in parts if part.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [
            item.strip() for item in value if isinstance(item, str) and item.strip()
        ]
    return []


def _optional_int(value: Any) -> int | None:
    if value is _MISSING or value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ConfigError(
            "syllable_budget must be an integer",
            details={"syllable_budget": repr(value)},
        )
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ConfigError(
            "syllable_budget must be an integer",
            details={"syllable_budget": repr(value)},
        ) from None
    if number < 1:
        raise ConfigError(
            "syllable_budget must be positive",
            details={"syllable_budget": number},
        )
    return number


def _normalise_name(value: Any, *, default: str = "verse") -> str:
    name = _text(value)
    if not name:
        return default
    name = name.strip().strip("[](){}<>").strip().rstrip(":").strip()
    return re.sub(r"\s+", " ", name).lower() or default


def _header(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped:
        return None
    bracket = _BRACKET.match(stripped)
    if bracket is not None:
        return _normalise_name(bracket.group(1)), ""
    label = _HEADER.match(stripped)
    if label is not None:
        name = _normalise_name(label.group(1))
        if _is_section_word(name):
            return name, label.group(2).strip()
    bare = _BARE.match(stripped)
    if bare is not None:
        name = _normalise_name(bare.group(1))
        if _is_section_word(name):
            return name, ""
    return None


def _is_section_word(name: str) -> bool:
    base = _NUMBERED.sub(r"\1", name).strip()
    return base in _SECTION_WORDS or name in _SECTION_WORDS


def _parse_text(text: str) -> list[LyricsSection]:
    if not text.strip():
        return []
    blocks = [
        block.strip() for block in re.split(r"\n\s*\n", text.strip()) if block.strip()
    ]
    sections: list[LyricsSection] = []
    current: LyricsSection | None = None
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not any(_header(line) is not None for line in lines):
            if current is not None:
                current.lines.extend(lines)
            else:
                sections.append(LyricsSection(name="verse", lines=lines))
            continue
        for line in lines:
            header = _header(line)
            if header is not None:
                if current is not None and current.lines:
                    sections.append(current)
                name, remainder = header
                current = LyricsSection(
                    name=name,
                    lines=[remainder] if remainder else [],
                )
            else:
                if current is None:
                    current = LyricsSection(name="verse", lines=[])
                current.lines.append(line)
    if current is not None and current.lines:
        sections.append(current)
    return sections


def _section_from(value: Any, *, default_name: str = "verse") -> LyricsSection:
    if isinstance(value, LyricsSection):
        return value
    if isinstance(value, str):
        return LyricsSection(
            name=default_name,
            lines=_string_list(value),
        )
    if isinstance(value, Mapping):
        raw_lines = _value_of(value, _LINE_KEYS)
        if raw_lines is _MISSING or raw_lines is None:
            raw_lines = _value_of(value, _SECTION_KEYS)
        lines = _string_list(raw_lines)
        if not lines and isinstance(raw_lines, str):
            lines = _string_list(raw_lines)
        explicit_name = _text(_value_of(value, ("name", "kind", "type", "role")))
        return LyricsSection(
            name=explicit_name or default_name,
            lines=lines,
            rhyme_scheme=_text(_value_of(value, _RHYME_KEYS)),
            syllable_budget=_optional_int(_value_of(value, _SYLLABLE_KEYS)),
        )
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return LyricsSection(name=default_name, lines=_string_list(value))
    raise ConfigError(
        f"cannot build a lyric section from {type(value).__name__}",
        details={"value": repr(value)},
    )


def _section_map(value: Mapping[str, Any]) -> list[LyricsSection] | None:
    keys = [
        key
        for key in value
        if isinstance(key, str) and _is_section_word(_normalise_name(key))
    ]
    if not keys:
        return None
    sections: list[LyricsSection] = []
    for key in keys:
        raw = value[key]
        if isinstance(raw, Mapping):
            merged = {"name": key, **raw}
            sections.append(_section_from(merged))
        else:
            sections.append(_section_from({"name": key, "lines": raw}))
    return sections


def _sections_from(value: Any) -> list[LyricsSection]:
    if value is None:
        return []
    if isinstance(value, LyricsPayload):
        return list(value.sections)
    if isinstance(value, Script) and isinstance(value.payload, LyricsPayload):
        return list(value.payload.sections)
    if isinstance(value, LyricsSection):
        return [value]
    if isinstance(value, str):
        return _parse_text(value)
    if isinstance(value, Mapping):
        container = _value_of(value, _SECTION_KEYS)
        if container is not _MISSING and container is not None:
            return _sections_from(container)
        mapped = _section_map(value)
        if mapped is not None:
            return mapped
        for key in _TEXT_KEYS:
            raw_text = value.get(key)
            if isinstance(raw_text, str):
                return _parse_text(raw_text)
        raw = _value_of(value, _LINE_KEYS)
        if raw is not _MISSING and raw is not None:
            return [_section_from({"name": "verse", "lines": raw})]
        return [_section_from(value)]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = list(value)
        if not items:
            return []
        if all(isinstance(item, str) for item in items):
            if any("\n" in item or _header(item) is not None for item in items):
                sections: list[LyricsSection] = []
                for item in items:
                    sections.extend(_parse_text(item))
                return sections
            return [LyricsSection(name="verse", lines=_string_list(items))]
        sections = []
        for item in items:
            if isinstance(item, str):
                sections.extend(_parse_text(item))
            elif isinstance(item, (Mapping, LyricsSection)) or (
                isinstance(item, Sequence) and not isinstance(item, (bytes, bytearray))
            ):
                sections.append(_section_from(item))
            else:
                raise ConfigError(
                    f"cannot build a lyric section from {type(item).__name__}",
                    details={"value": repr(item)},
                )
        return sections
    raise ConfigError(
        f"cannot build lyrics from {type(value).__name__}",
        details={"value": repr(value)},
    )


def _derive_counts(sections: Any) -> dict[str, int]:
    if isinstance(sections, LyricsPayload):
        sections = sections.sections
    elif isinstance(sections, Script) and isinstance(sections.payload, LyricsPayload):
        sections = sections.payload.sections
    counts: dict[str, int] = {}
    for section in sections or []:
        name = section.name or "verse"
        counts[name] = counts.get(name, 0) + 1
    return counts


def _coerce_counts(value: Any) -> dict[str, int] | None:
    if value is _MISSING or value is None or value == "":
        return None
    if isinstance(value, Mapping):
        counts: dict[str, int] = {}
        for key, raw in value.items():
            name = _text(key) or "verse"
            try:
                count = int(raw)
            except (TypeError, ValueError):
                raise ConfigError(
                    "section counts must be integers",
                    details={"section": key, "count": repr(raw)},
                ) from None
            if count < 0:
                raise ConfigError(
                    "section counts must not be negative",
                    details={"section": name, "count": count},
                )
            counts[name] = count
        return counts
    if isinstance(value, str):
        counts = {}
        for part in re.split(r"[,;]+", value):
            piece = part.strip()
            if not piece:
                continue
            match = re.match(r"^(.*?)\s*(?::|=|x|\*)\s*(\d+)$", piece, re.I)
            if match is None:
                match = re.match(r"^(\d+)\s+(.+)$", piece)
                if match is not None:
                    counts[_text(match.group(2)) or "verse"] = int(match.group(1))
                    continue
                raise ConfigError(
                    f"cannot read a section count from {piece!r}",
                    details={"section_counts": value},
                )
            counts[_text(match.group(1)) or "verse"] = int(match.group(2))
        return counts
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        counts = {}
        for item in value:
            if isinstance(item, Mapping):
                counts.update(_coerce_counts(item) or {})
            elif (
                isinstance(item, Sequence)
                and not isinstance(item, (str, bytes, bytearray))
                and len(item) == 2
            ):
                counts[_text(item[0]) or "verse"] = int(item[1])
            elif isinstance(item, str):
                counts.update(_coerce_counts(item) or {})
            else:
                raise ConfigError(
                    "cannot read section counts",
                    details={"section_counts": repr(value)},
                )
        return counts
    raise ConfigError(
        f"cannot read section counts from {type(value).__name__}",
        details={"section_counts": repr(value)},
    )


normalise_section_counts = _coerce_counts
normalize_section_counts = _coerce_counts
count_sections = _derive_counts
section_counts = _derive_counts
derive_section_counts = _derive_counts


def _count_matches(section_name: str, requested: str) -> bool:
    left = section_name.casefold()
    right = requested.casefold()
    if left == right:
        return True
    return _NUMBERED.sub(r"\1", left).strip() == _NUMBERED.sub(r"\1", right).strip()


def _apply_counts(
    sections: Sequence[LyricsSection],
    counts: Mapping[str, int],
) -> list[LyricsSection]:
    result = [section.model_copy(deep=True) for section in sections]
    for name, wanted in counts.items():
        indices = [
            index
            for index, section in enumerate(result)
            if _count_matches(section.name or "verse", name)
        ]
        actual = len(indices)
        if wanted == actual:
            continue
        if wanted > actual:
            if actual == 0:
                for _ in range(wanted):
                    result.append(LyricsSection(name=name, lines=[]))
                continue
            last = indices[-1]
            template = result[last]
            insert_at = last + 1
            for _ in range(wanted - actual):
                result.insert(insert_at, template.model_copy(deep=True))
                insert_at += 1
        else:
            for index in reversed(indices[wanted:]):
                del result[index]
    return result


def _lyrics_from_mapping(value: Mapping[str, Any]) -> LyricsPayload:
    sections = _sections_from(value)
    if not sections:
        raw = _value_of(value, _LINE_KEYS)
        if isinstance(raw, str):
            sections = _parse_text(raw)
        elif raw is not _MISSING and raw is not None:
            sections = _sections_from(raw)
    counts = _coerce_counts(_value_of(value, _COUNT_KEYS))
    if counts is None:
        counts = _derive_counts(sections)
    return LyricsPayload(
        sections=sections,
        section_counts=counts,
        rhyme_scheme=_text(_value_of(value, _RHYME_KEYS)),
        syllable_budget=_optional_int(_value_of(value, _SYLLABLE_KEYS)),
        keywords=_keyword_list(_value_of(value, _KEYWORD_KEYS)),
        genre=_text(_value_of(value, _GENRE_KEYS)),
        tempo=_text(_value_of(value, _TEMPO_KEYS)),
    )


def _merge_lyrics(
    payload: LyricsPayload,
    value: Mapping[str, Any],
) -> LyricsPayload:
    outer = _lyrics_from_mapping(value)
    sections = outer.sections or list(payload.sections)
    counts = _coerce_counts(_value_of(value, _COUNT_KEYS))
    if counts is None:
        counts = dict(payload.section_counts) or _derive_counts(sections)
    keywords = _keyword_list(_value_of(value, _KEYWORD_KEYS))
    return LyricsPayload(
        sections=sections,
        section_counts=counts,
        rhyme_scheme=outer.rhyme_scheme or payload.rhyme_scheme,
        syllable_budget=(
            outer.syllable_budget
            if outer.syllable_budget is not None
            else payload.syllable_budget
        ),
        keywords=keywords or list(payload.keywords),
        genre=outer.genre or payload.genre,
        tempo=outer.tempo or payload.tempo,
    )


def _lyrics_from(value: Any) -> LyricsPayload:
    if isinstance(value, LyricsPayload):
        return value
    if isinstance(value, Script) and isinstance(value.payload, LyricsPayload):
        return value.payload
    if isinstance(value, str):
        sections = _parse_text(value)
        return LyricsPayload(
            sections=sections,
            section_counts=_derive_counts(sections),
        )
    if isinstance(value, (bytes, bytearray)) or value is None:
        return LyricsPayload()

    if isinstance(value, Mapping):
        nested = _value_of(value, _PAYLOAD_KEYS)
        if nested is not _MISSING and nested is not None:
            return _merge_lyrics(_lyrics_from(nested), value)
        return _lyrics_from_mapping(value)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        sections = _sections_from(value)
        return LyricsPayload(
            sections=sections,
            section_counts=_derive_counts(sections),
        )

    if (
        _value_of(value, _SECTION_KEYS) is not _MISSING
        or _value_of(value, _LINE_KEYS) is not _MISSING
    ):
        fields = {
            key: getattr(value, key)
            for key in (
                *_SECTION_KEYS,
                *_LINE_KEYS,
                *_COUNT_KEYS,
                *_RHYME_KEYS,
                *_SYLLABLE_KEYS,
                *_KEYWORD_KEYS,
                *_GENRE_KEYS,
                *_TEMPO_KEYS,
            )
            if hasattr(value, key)
        }
        return _lyrics_from_mapping(fields)

    raise ConfigError(
        f"cannot build lyrics from {type(value).__name__}",
        details={"value": repr(value)},
    )


def extract_lyrics(value: LyricsInput = None) -> LyricsPayload:
    """Return *value* as a typed :class:`LyricsPayload`.

    A string is split into sections on ``[Verse]``-style headers or blank
    lines. A mapping accepts the schema aliases for sections, section counts,
    rhyme, syllables, keywords, genre and tempo.
    """
    return _lyrics_from(value)


coerce_lyrics = extract_lyrics
parse_lyrics = extract_lyrics
normalize_lyrics = extract_lyrics
normalise_lyrics = extract_lyrics


def count_syllables(text: str | None) -> int:
    """Return a deterministic syllable estimate for *text*.

    The estimate is a vowel group count with a silent trailing ``e`` removed,
    which is stable and good enough for a budget check. Every word counts at
    least once, so punctuation and digits do not disappear.
    """
    if not isinstance(text, str):
        return 0
    total = 0
    for word in _WORD.findall(text.lower()):
        groups = len(_VOWEL_GROUP.findall(word))
        if word.endswith("e") and not word.endswith(("le", "ee", "ye")):
            groups -= 1
        total += max(1, groups)
    return total


syllable_count = count_syllables
syllables = count_syllables
count_syllables_in_line = count_syllables


def check_syllable_budget(
    value: LyricsInput = None,
    *,
    strict: bool = True,
) -> LyricsPayload:
    """Check each line against its section or payload syllable budget.

    A budget is a per-line limit. When *strict* is false the payload is
    returned unchanged and the check only reports through the return value;
    otherwise the first over-budget line raises :class:`ConfigError`.
    """
    payload = extract_lyrics(value)
    for section in payload.sections:
        budget = section.syllable_budget or payload.syllable_budget
        if not budget:
            continue
        for line in section.lines:
            measured = count_syllables(line)
            if measured > budget and strict:
                raise ConfigError(
                    f"lyric line has {measured} syllables, over the budget of {budget}",
                    details={
                        "mode": "lyrics",
                        "section": section.name,
                        "line": line,
                        "syllables": measured,
                        "syllable_budget": budget,
                    },
                )
    return payload


def _replace_placeholders(lines: Sequence[str], keywords: Sequence[str]) -> list[str]:
    if not keywords:
        return list(lines)
    placeholders = sum(len(_PLACEHOLDER.findall(line)) for line in lines)
    if placeholders <= 1 and len(keywords) > 1:
        phrase = ", ".join(keywords)
        return [_PLACEHOLDER.sub(phrase, line) for line in lines]
    replaced: list[str] = []
    cursor = 0
    for line in lines:
        while _PLACEHOLDER.search(line):
            keyword = keywords[cursor % len(keywords)]
            line = _PLACEHOLDER.sub(keyword, line, count=1)
            cursor += 1
        replaced.append(line)
    return replaced


def _insert_keywords_into_sections(
    sections: Sequence[LyricsSection],
    keywords: Sequence[str],
) -> list[LyricsSection]:
    if not keywords:
        return [section.model_copy(deep=True) for section in sections]
    result = [section.model_copy(deep=True) for section in sections]
    for section in result:
        section.lines = _replace_placeholders(section.lines, keywords)
    joined = "\n".join(line for section in result for line in section.lines)
    missing = [
        keyword for keyword in keywords if keyword.casefold() not in joined.casefold()
    ]
    if not missing:
        return result
    target = next(
        (
            section
            for section in reversed(result)
            if section.name.casefold() == "chorus"
        ),
        result[-1] if result else None,
    )
    if target is None:
        target = LyricsSection(name="chorus", lines=[])
        result.append(target)
    target.lines.append(", ".join(missing))
    return result


def insert_keywords(
    value: LyricsInput = None,
    keywords: Any = None,
) -> Any:
    """Insert *keywords* into *value*, replacing placeholders when present.

    A payload, script or mapping returns a :class:`LyricsPayload`; a sequence
    of lines returns the updated lines. Missing keywords are added as one line
    to the chorus, or to the last section when there is no chorus.
    """
    words = _keyword_list(keywords)
    if isinstance(value, (LyricsPayload, Script, Mapping)):
        payload = extract_lyrics(value)
        if not words:
            words = list(payload.keywords)
        return LyricsPayload(
            sections=_insert_keywords_into_sections(payload.sections, words),
            section_counts=dict(payload.section_counts),
            rhyme_scheme=payload.rhyme_scheme,
            syllable_budget=payload.syllable_budget,
            keywords=words or list(payload.keywords),
            genre=payload.genre,
            tempo=payload.tempo,
        )
    if isinstance(value, str):
        lines = _replace_placeholders(_string_list(value), words)
        joined = "\n".join(lines)
        missing = [word for word in words if word.casefold() not in joined.casefold()]
        if missing:
            lines.append(", ".join(missing))
        return "\n".join(lines)
    if isinstance(value, LyricsSection):
        return _insert_keywords_into_sections([value], words)[0]
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (bytes, bytearray))
        and value
        and all(isinstance(item, LyricsSection) for item in value)
    ):
        return _insert_keywords_into_sections(list(value), words)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        lines = _replace_placeholders(_string_list(value), words)
        joined = "\n".join(lines)
        missing = [word for word in words if word.casefold() not in joined.casefold()]
        if missing:
            lines.append(", ".join(missing))
        return lines
    if value is None:
        return words
    raise ConfigError(
        f"cannot insert keywords into {type(value).__name__}",
        details={"value": repr(value)},
    )


apply_keywords = insert_keywords
inject_keywords = insert_keywords


def validate_lyrics(
    value: LyricsInput = None,
    *,
    enforce_syllables: bool = True,
) -> LyricsPayload:
    """Return *value* as a payload and require at least one non-empty line."""
    payload = extract_lyrics(value)
    sections = list(payload.sections)
    if payload.section_counts:
        sections = _apply_counts(sections, payload.section_counts)
    lines = [line for section in sections for line in section.lines]
    if not lines:
        raise ConfigError(
            "lyrics mode needs at least one line",
            details={"mode": "lyrics"},
        )
    payload = LyricsPayload(
        sections=sections,
        section_counts=(dict(payload.section_counts) or _derive_counts(sections)),
        rhyme_scheme=payload.rhyme_scheme,
        syllable_budget=payload.syllable_budget,
        keywords=list(payload.keywords),
        genre=payload.genre,
        tempo=payload.tempo,
    )
    if enforce_syllables:
        check_syllable_budget(payload)
    return payload


class LyricsMode(BuiltinWritingMode):
    """The lyrics writing mode.

    :meth:`build_script` turns sections into a script with one beat per lyric
    line, records section counts, rhyme scheme, syllable budget and keywords,
    and inserts missing keywords into the chorus. :meth:`build_payload` keeps
    the base contract and returns the typed :class:`LyricsPayload`.
    """

    name = "lyrics"
    description = "Named song sections with rhyme, syllables and keywords."
    instructions = (
        "Keep the requested section order and counts, follow the rhyme scheme, "
        "stay inside the syllable budget and use each keyword naturally."
    )

    def build_payload(self, value: LyricsInput = None) -> LyricsPayload:
        """Validate *value* as a :class:`LyricsPayload`."""
        return extract_lyrics(value)

    def build_hook(
        self,
        value: LyricsInput = None,
        *,
        title: str = "",
        hook: str | None = None,
    ) -> str:
        """Return the first lyric line, using an explicit *hook* when given."""
        if hook is not None:
            return hook.strip()
        payload = extract_lyrics(value)
        lines = [line for section in payload.sections for line in section.lines]
        if not lines:
            return ""
        subject = title.strip()
        if subject:
            return f"{subject}: {lines[0]}"
        return lines[0]

    def build_beats(
        self,
        value: LyricsInput = None,
        *,
        speaker: str = "narrator",
        max_beats: int | None = None,
    ) -> list[Beat]:
        """Return one beat per lyric line, each sung by *speaker*."""
        payload = extract_lyrics(value)
        lines = [line for section in payload.sections for line in section.lines]
        if max_beats is not None:
            lines = lines[:max_beats]
        return [
            Beat(
                index=index,
                text=line,
                dialogue=[DialogueLine(speaker=speaker, text=line)],
            )
            for index, line in enumerate(lines)
        ]

    def build_outro(
        self,
        value: LyricsInput = None,
        *,
        outro: str | None = None,
    ) -> str:
        """Return the closing line, using an explicit *outro* when given."""
        if outro is not None:
            return outro.strip()
        return ""

    def insert_keywords(
        self,
        value: LyricsInput = None,
        keywords: Any = None,
    ) -> Any:
        """Insert *keywords* into *value* and return the updated shape."""
        return insert_keywords(value, keywords)

    def build_script(
        self,
        value: LyricsInput = None,
        *,
        title: str | None = None,
        language: str | None = None,
        sections: Any = None,
        section_counts: Any = None,
        counts: Any = None,
        rhyme_scheme: str | None = None,
        rhyme: str | None = None,
        syllable_budget: int | None = None,
        syllables: int | None = None,
        keywords: Any = None,
        keyword: Any = None,
        genre: str | None = None,
        tempo: str | None = None,
        insert_keywords: bool = True,
        hook: str | None = None,
        outro: str | None = None,
        speaker: str = "narrator",
        max_beats: int | None = None,
    ) -> Script:
        """Build a lyrics script from text, sections, a mapping or a payload."""
        if section_counts is None:
            section_counts = counts
        if rhyme_scheme is None:
            rhyme_scheme = rhyme
        if syllable_budget is None:
            syllable_budget = syllables
        if keywords is None:
            keywords = keyword
        payload = extract_lyrics(value)
        resolved_sections = (
            _sections_from(sections) if sections is not None else list(payload.sections)
        )
        if section_counts is not None:
            resolved_counts = _coerce_counts(section_counts)
        elif sections is not None:
            resolved_counts = None
        else:
            resolved_counts = dict(payload.section_counts) or None
        if resolved_counts:
            resolved_sections = _apply_counts(resolved_sections, resolved_counts)
        if not [line for section in resolved_sections for line in section.lines]:
            raise ConfigError(
                "lyrics mode needs at least one line",
                details={"mode": self.name},
            )
        resolved_keywords = (
            _keyword_list(keywords) if keywords is not None else list(payload.keywords)
        )
        if insert_keywords and resolved_keywords:
            resolved_sections = _insert_keywords_into_sections(
                resolved_sections,
                resolved_keywords,
            )
        payload = LyricsPayload(
            sections=resolved_sections,
            section_counts=(
                resolved_counts
                if resolved_counts is not None
                else _derive_counts(resolved_sections)
            ),
            rhyme_scheme=(
                rhyme_scheme.strip()
                if rhyme_scheme is not None
                else payload.rhyme_scheme
            ),
            syllable_budget=(
                syllable_budget
                if syllable_budget is not None
                else payload.syllable_budget
            ),
            keywords=resolved_keywords,
            genre=genre.strip() if genre is not None else payload.genre,
            tempo=tempo.strip() if tempo is not None else payload.tempo,
        )
        resolved_title = (
            title if title is not None else _text(_value_of(value, _TITLE_KEYS))
        )
        return Script(
            mode=WritingMode.LYRICS,
            title=resolved_title,
            language=(language or _text(_value_of(value, _LANGUAGE_KEYS)) or "en"),
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
    build_from_lyrics = build_script


LyricsWritingMode = LyricsMode
Lyrics = LyricsMode


def _optional_text(
    value: Any,
    keys: Sequence[str],
    explicit: str | None,
) -> str | None:
    if explicit is not None:
        return explicit.strip() or None
    found = _text(_value_of(value, keys))
    return found or None


def build_lyrics_script(
    value: LyricsInput = None,
    **kwargs: Any,
) -> Script:
    """Build a lyrics script with a default :class:`LyricsMode`."""
    return LyricsMode().build_script(value, **kwargs)


lyrics_script = build_lyrics_script


register_builtin_writing_mode(WritingMode.LYRICS, LyricsMode)

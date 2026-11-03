"""The poem writing mode: verse with a chosen form, tone and reading time.

A poem may arrive as plain text, a list of lines, a mapping or an existing
payload. Blank lines separate stanzas. The mode keeps the form and tone as data
for a later rewrite, builds one beat per line and estimates a deterministic
reading time. When a length is requested the reading time must fall inside its
inclusive range, otherwise building fails with the measured time and the target
range in the error details.

No model is called here and nothing is imported from research, so the poem
structure and its timing stay stable and offline.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from aivia.core.errors import ConfigError
from aivia.core.models.script import Beat, DialogueLine
from aivia.writing.length import (
    DEFAULT_WORDS_PER_SECOND,
    LENGTH_PRESETS,
    LengthRange,
    coerce_length_range,
    count_words,
    reading_time_seconds,
    resolve_length_range,
)
from aivia.writing.modes import (
    BuiltinWritingMode,
    register_builtin_writing_mode,
)
from aivia.writing.schema import PoemPayload, Script, WritingMode

__all__ = [
    "DEFAULT_WORDS_PER_SECOND",
    "LENGTH_PRESETS",
    "LengthRange",
    "Poem",
    "PoemInput",
    "PoemMode",
    "PoemWritingMode",
    "build_poem_script",
    "coerce_length_range",
    "coerce_poem",
    "count_words",
    "extract_poem",
    "normalise_poem",
    "normalize_poem",
    "parse_poem",
    "poem_script",
    "reading_time_seconds",
]

PoemInput = Any
"""Any value accepted as a poem: text, lines, a mapping or a payload."""

_FORM_KEYS = ("form", "style", "poem_form")
_TONE_KEYS = ("tone", "mood", "voice")
_LINE_KEYS = ("lines", "verses", "verse")
_STANZA_KEYS = ("stanzas", "stanza")
_POEM_KEYS = ("poem", "text", "content", "body")
_RHYME_KEYS = ("rhyme_scheme", "rhyme", "scheme")
_SYLLABLE_KEYS = ("syllable_budget", "syllables", "syllable_count")
_LENGTH_KEYS = ("length", "target_length", "duration")
_TITLE_KEYS = ("title", "name", "subject")
_LANGUAGE_KEYS = ("language", "lang", "locale")
_HOOK_KEYS = ("hook", "opening", "intro")
_OUTRO_KEYS = ("outro", "closing", "ending")
_PAYLOAD_KEYS = ("payload", "mode_payload", "mode_data", "data", "options")
_MISSING = object()


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
        lines: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                lines.append(item.strip())
        return lines
    return []


def _stanza_list(value: Any) -> list[list[str]]:
    if value is None:
        return []
    if isinstance(value, str):
        blocks = re.split(r"\n\s*\n", value.strip())
        return [
            [line.strip() for line in block.splitlines() if line.strip()]
            for block in blocks
            if block.strip()
        ]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        stanzas: list[list[str]] = []
        for item in value:
            lines = _string_list(item)
            if lines:
                stanzas.append(lines)
        return stanzas
    return []


def _parse_text(text: str) -> tuple[list[str], list[list[str]]]:
    stanzas = _stanza_list(text)
    lines = [line for stanza in stanzas for line in stanza]
    return lines, stanzas


def _length_fields(value: Any) -> tuple[str, float | None, float | None]:
    raw = _value_of(value, _LENGTH_KEYS)
    if raw is _MISSING or raw is None or raw == "":
        return "", None, None
    target = coerce_length_range(raw)
    label = raw.strip() if isinstance(raw, str) else "custom"
    maximum = target.max_seconds if math.isfinite(target.max_seconds) else None
    return label, target.min_seconds, maximum


def _poem_from_mapping(value: Mapping[str, Any]) -> PoemPayload:
    lines = _string_list(_value_of(value, _LINE_KEYS))
    stanzas = _stanza_list(_value_of(value, _STANZA_KEYS))
    if not lines and not stanzas:
        raw = _value_of(value, _POEM_KEYS)
        if isinstance(raw, str):
            lines, stanzas = _parse_text(raw)
        elif raw is not _MISSING and raw is not None:
            lines = _string_list(raw)
    if stanzas and not lines:
        lines = [line for stanza in stanzas for line in stanza]

    length, min_seconds, max_seconds = _length_fields(value)
    return PoemPayload(
        form=_text(_value_of(value, _FORM_KEYS)),
        tone=_text(_value_of(value, _TONE_KEYS)),
        lines=lines,
        stanzas=stanzas,
        rhyme_scheme=_text(_value_of(value, _RHYME_KEYS)),
        syllable_budget=_optional_int(_value_of(value, _SYLLABLE_KEYS)),
        length=length,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
    )


def _optional_int(value: Any) -> int | None:
    if value is _MISSING or value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ConfigError(
            "syllable_budget must be an integer",
            details={"syllable_budget": repr(value)},
        ) from None


def _merge_metadata(payload: PoemPayload, value: Mapping[str, Any]) -> PoemPayload:
    outer = _poem_from_mapping(value)
    lines = outer.lines or payload.lines
    stanzas = outer.stanzas or payload.stanzas
    if stanzas and not lines:
        lines = [line for stanza in stanzas for line in stanza]
    return PoemPayload(
        form=outer.form or payload.form,
        tone=outer.tone or payload.tone,
        lines=lines,
        stanzas=stanzas,
        rhyme_scheme=outer.rhyme_scheme or payload.rhyme_scheme,
        syllable_budget=(
            outer.syllable_budget
            if outer.syllable_budget is not None
            else payload.syllable_budget
        ),
        length=outer.length or payload.length,
        min_seconds=(
            outer.min_seconds if outer.min_seconds is not None else payload.min_seconds
        ),
        max_seconds=(
            outer.max_seconds if outer.max_seconds is not None else payload.max_seconds
        ),
        reading_time_seconds=payload.reading_time_seconds,
    )


def _poem_from(value: Any) -> PoemPayload:
    if isinstance(value, PoemPayload):
        return value
    if isinstance(value, Script) and isinstance(value.payload, PoemPayload):
        return value.payload
    if isinstance(value, str):
        lines, stanzas = _parse_text(value)
        return PoemPayload(lines=lines, stanzas=stanzas)
    if isinstance(value, (bytes, bytearray)) or value is None:
        return PoemPayload()

    if isinstance(value, Mapping):
        nested = _value_of(value, _PAYLOAD_KEYS)
        if nested is not _MISSING and nested is not None:
            return _merge_metadata(_poem_from(nested), value)
        return _poem_from_mapping(value)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        if all(isinstance(item, str) for item in items):
            return _poem_from_mapping({"lines": items})
        return _poem_from_mapping({"stanzas": items})

    if (
        _value_of(value, _LINE_KEYS) is not _MISSING
        or _value_of(value, _STANZA_KEYS) is not _MISSING
        or _value_of(value, _POEM_KEYS) is not _MISSING
    ):
        fields = {
            key: getattr(value, key)
            for key in (
                *_FORM_KEYS,
                *_TONE_KEYS,
                *_LINE_KEYS,
                *_STANZA_KEYS,
                *_POEM_KEYS,
                *_RHYME_KEYS,
                *_SYLLABLE_KEYS,
                *_LENGTH_KEYS,
            )
            if hasattr(value, key)
        }
        return _poem_from_mapping(fields)

    raise ConfigError(
        f"cannot build a poem from {type(value).__name__}",
        details={"value": repr(value)},
    )


def extract_poem(value: PoemInput = None) -> PoemPayload:
    """Return *value* as a typed :class:`PoemPayload`.

    A string is split into lines, with blank lines separating stanzas. A
    mapping accepts the schema aliases for form, tone, lines, stanzas, rhyme,
    syllables and length.
    """
    return _poem_from(value)


coerce_poem = extract_poem
parse_poem = extract_poem
normalize_poem = extract_poem
normalise_poem = extract_poem


def _target_range(
    payload: PoemPayload,
    length: Any,
    min_seconds: float | None,
    max_seconds: float | None,
) -> LengthRange | None:
    base = length if length is not None else (payload.length or None)
    if (
        length is None
        and min_seconds is None
        and max_seconds is None
        and payload.min_seconds is not None
    ):
        maximum = payload.max_seconds if payload.max_seconds is not None else math.inf
        return LengthRange(payload.min_seconds, maximum)
    return resolve_length_range(
        base,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
    )


def _check_reading_time(
    mode: str,
    measured: float,
    target: LengthRange | None,
) -> None:
    if target is None or target.contains(measured):
        return
    raise ConfigError(
        f"{mode} reading time {measured:.1f}s is outside the target range "
        f"{target.min_seconds:.1f}-{target.max_seconds:.1f}s",
        details={
            "mode": mode,
            "measured_seconds": round(measured, 3),
            "min_seconds": target.min_seconds,
            "max_seconds": target.max_seconds,
        },
    )


class PoemMode(BuiltinWritingMode):
    """The poem writing mode.

    :meth:`build_script` turns verse into a script with one beat per line, a
    recorded form, tone and reading time, and an optional length range it must
    satisfy. :meth:`build_payload` keeps the base contract and returns the typed
    :class:`PoemPayload`.
    """

    name = "poem"
    description = "Verse with a chosen form, tone and reading time."
    instructions = (
        "Keep the requested form and tone, preserve the line breaks and stay "
        "inside the reading time the profile asks for."
    )

    def build_payload(self, value: PoemInput = None) -> PoemPayload:
        """Validate *value* as a :class:`PoemPayload`."""
        return extract_poem(value)

    def build_hook(
        self,
        value: PoemInput = None,
        *,
        title: str = "",
        hook: str | None = None,
    ) -> str:
        """Return the opening line, using an explicit *hook* when given."""
        if hook is not None:
            return hook.strip()
        payload = extract_poem(value)
        if not payload.lines:
            return ""
        subject = title.strip()
        if subject:
            return f"{subject}: {payload.lines[0]}"
        return payload.lines[0]

    def build_beats(
        self,
        value: PoemInput = None,
        *,
        speaker: str = "narrator",
        max_beats: int | None = None,
    ) -> list[Beat]:
        """Return one beat per line, each narrated by *speaker*."""
        lines = extract_poem(value).lines
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
        value: PoemInput = None,
        *,
        outro: str | None = None,
    ) -> str:
        """Return the closing line, using an explicit *outro* when given."""
        if outro is not None:
            return outro.strip()
        return ""

    def build_script(
        self,
        value: PoemInput = None,
        *,
        title: str | None = None,
        language: str | None = None,
        form: str | None = None,
        tone: str | None = None,
        length: Any = None,
        min_seconds: float | None = None,
        max_seconds: float | None = None,
        words_per_second: float | None = None,
        rhyme_scheme: str | None = None,
        syllable_budget: int | None = None,
        hook: str | None = None,
        outro: str | None = None,
        speaker: str = "narrator",
        max_beats: int | None = None,
    ) -> Script:
        """Build a poem script from text, lines, a mapping or a payload."""
        payload = extract_poem(value)
        lines = payload.lines or [line for stanza in payload.stanzas for line in stanza]
        if not lines:
            raise ConfigError(
                "poem mode needs at least one line",
                details={"mode": self.name},
            )

        resolved_form = form.strip() if form is not None else payload.form
        resolved_tone = tone.strip() if tone is not None else payload.tone
        resolved_rhyme = (
            rhyme_scheme.strip() if rhyme_scheme is not None else payload.rhyme_scheme
        )
        resolved_syllables = (
            syllable_budget if syllable_budget is not None else payload.syllable_budget
        )
        target = _target_range(payload, length, min_seconds, max_seconds)
        measured = reading_time_seconds(
            "\n".join(lines),
            words_per_second=words_per_second,
        )
        if measured <= 0:
            raise ConfigError(
                "poem mode needs at least one word",
                details={"mode": self.name},
            )
        _check_reading_time(self.name, measured, target)
        duration = round(measured, 3)
        payload = PoemPayload(
            form=resolved_form,
            tone=resolved_tone,
            lines=lines,
            stanzas=payload.stanzas,
            rhyme_scheme=resolved_rhyme,
            syllable_budget=resolved_syllables,
            length=(
                length.strip()
                if isinstance(length, str)
                else (payload.length if length is None else "custom")
            ),
            min_seconds=target.min_seconds if target is not None else None,
            max_seconds=(
                target.max_seconds
                if target is not None and math.isfinite(target.max_seconds)
                else None
            ),
            reading_time_seconds=duration,
        )
        resolved_title = (
            title if title is not None else _text(_value_of(value, _TITLE_KEYS))
        )
        return Script(
            mode=WritingMode.POEM,
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
            duration_seconds=duration,
        )

    script = build_script
    make_script = build_script
    build_from_poem = build_script


def _optional_text(
    value: Any,
    keys: Sequence[str],
    explicit: str | None,
) -> str | None:
    if explicit is not None:
        return explicit.strip() or None
    found = _text(_value_of(value, keys))
    return found or None


PoemWritingMode = PoemMode
Poem = PoemMode


def build_poem_script(
    value: PoemInput = None,
    **kwargs: Any,
) -> Script:
    """Build a poem script with a default :class:`PoemMode`."""
    return PoemMode().build_script(value, **kwargs)


poem_script = build_poem_script


register_builtin_writing_mode(WritingMode.POEM, PoemMode)

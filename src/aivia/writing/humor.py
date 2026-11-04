"""The joke and riddle writing modes: short setups and their payoffs.

A joke is a setup followed by a punchline; a riddle is a question followed by
its answer, with an optional hint and difficulty. Either may arrive as plain
text, a two item sequence, a mapping or an existing payload. The modes build a
hook from the opening half, one beat for each half and a payload that keeps the
typed setup/punchline or question/answer fields.

Building is deterministic and offline. No model is called here and nothing is
imported from research, so the humour structure stays stable and testable.
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
    JokePayload,
    RiddlePayload,
    Script,
    WritingMode,
)

__all__ = [
    "Joke",
    "JokeInput",
    "JokeMode",
    "JokeWritingMode",
    "Riddle",
    "RiddleInput",
    "RiddleMode",
    "RiddleWritingMode",
    "build_joke_script",
    "build_riddle_script",
    "coerce_joke",
    "coerce_riddle",
    "extract_joke",
    "extract_riddle",
    "joke_script",
    "normalise_joke",
    "normalise_riddle",
    "normalize_joke",
    "normalize_riddle",
    "parse_joke",
    "parse_riddle",
    "riddle_script",
    "validate_joke",
    "validate_riddle",
]

JokeInput = Any
"""Any value accepted as a joke: text, a pair, a mapping or a payload."""

RiddleInput = Any
"""Any value accepted as a riddle: text, a pair, a mapping or a payload."""

_SETUP_KEYS = ("setup", "question", "premise", "prompt")
_PUNCHLINE_KEYS = ("punchline", "payoff", "answer", "response")
_STYLE_KEYS = ("style", "form", "kind", "genre")
_TAG_KEYS = ("tags", "labels", "keywords")
_JOKE_KEYS = ("joke", "riddle", "text", "content", "body")
_QUESTION_KEYS = ("question", "setup", "prompt")
_ANSWER_KEYS = ("answer", "solution", "punchline", "response")
_HINT_KEYS = ("hint", "clue")
_DIFFICULTY_KEYS = ("difficulty", "level")
_TITLE_KEYS = ("title", "name", "subject")
_LANGUAGE_KEYS = ("language", "lang", "locale")
_HOOK_KEYS = ("hook", "opening", "intro")
_OUTRO_KEYS = ("outro", "closing", "ending")
_PAYLOAD_KEYS = ("payload", "mode_payload", "mode_data", "data", "options")
_MISSING = object()

_SETUP_LABELS = {"setup", "q", "question", "premise"}
_PUNCHLINE_LABELS = {"punchline", "a", "answer", "payoff"}
_QUESTION_LABELS = {"question", "q", "riddle", "setup"}
_ANSWER_LABELS = {"answer", "a", "solution", "punchline"}
_LABEL = re.compile(r"^\s*([A-Za-z]+)\s*[:.\-]\s*", re.UNICODE)


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
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [
            item.strip() for item in value if isinstance(item, str) and item.strip()
        ]
    return []


def _without_label(text: str, labels: set[str]) -> str:
    match = _LABEL.match(text)
    if match is not None and match.group(1).casefold() in labels:
        return text[match.end() :].strip()
    return text.strip()


def _split_halves(text: str) -> tuple[str, str]:
    cleaned = text.strip()
    if not cleaned:
        return "", ""
    blocks = [block.strip() for block in re.split(r"\n\s*\n", cleaned) if block.strip()]
    if len(blocks) >= 2:
        return blocks[0], "\n\n".join(blocks[1:])
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if len(lines) >= 2:
        return lines[0], "\n".join(lines[1:])
    return cleaned, ""


def _joke_text(text: str) -> tuple[str, str]:
    setup, punchline = _split_halves(text)
    return (
        _without_label(setup, _SETUP_LABELS),
        _without_label(punchline, _PUNCHLINE_LABELS),
    )


def _riddle_text(text: str) -> tuple[str, str]:
    question, answer = _split_halves(text)
    return (
        _without_label(question, _QUESTION_LABELS),
        _without_label(answer, _ANSWER_LABELS),
    )


def _pair(value: Any) -> tuple[str, str]:
    if isinstance(value, str):
        return "", ""
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = [item for item in value if isinstance(item, str) and item.strip()]
        if len(items) == 2:
            return items[0].strip(), items[1].strip()
    return "", ""


def _joke_from_mapping(value: Mapping[str, Any]) -> JokePayload:
    setup = _text(_value_of(value, _SETUP_KEYS))
    punchline = _text(_value_of(value, _PUNCHLINE_KEYS))
    if not setup and not punchline:
        raw = _value_of(value, _JOKE_KEYS)
        if isinstance(raw, str):
            setup, punchline = _joke_text(raw)
        elif raw is not _MISSING and raw is not None:
            setup, punchline = _pair(raw)
    return JokePayload(
        setup=setup,
        punchline=punchline,
        style=_text(_value_of(value, _STYLE_KEYS)),
        tags=_string_list(_value_of(value, _TAG_KEYS)),
    )


def _merge_joke(payload: JokePayload, value: Mapping[str, Any]) -> JokePayload:
    outer = _joke_from_mapping(value)
    return JokePayload(
        setup=outer.setup or payload.setup,
        punchline=outer.punchline or payload.punchline,
        style=outer.style or payload.style,
        tags=outer.tags or payload.tags,
    )


def _joke_from(value: Any) -> JokePayload:
    if isinstance(value, JokePayload):
        return value
    if isinstance(value, Script) and isinstance(value.payload, JokePayload):
        return value.payload
    if isinstance(value, str):
        setup, punchline = _joke_text(value)
        return JokePayload(setup=setup, punchline=punchline)
    if isinstance(value, (bytes, bytearray)) or value is None:
        return JokePayload()

    if isinstance(value, Mapping):
        nested = _value_of(value, _PAYLOAD_KEYS)
        if nested is not _MISSING and nested is not None:
            return _merge_joke(_joke_from(nested), value)
        return _joke_from_mapping(value)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        setup, punchline = _pair(value)
        return JokePayload(setup=setup, punchline=punchline)

    if (
        _value_of(value, _SETUP_KEYS) is not _MISSING
        or _value_of(value, _PUNCHLINE_KEYS) is not _MISSING
        or _value_of(value, _JOKE_KEYS) is not _MISSING
    ):
        fields = {
            key: getattr(value, key)
            for key in (
                *_SETUP_KEYS,
                *_PUNCHLINE_KEYS,
                *_STYLE_KEYS,
                *_TAG_KEYS,
                *_JOKE_KEYS,
            )
            if hasattr(value, key)
        }
        return _joke_from_mapping(fields)

    raise ConfigError(
        f"cannot build a joke from {type(value).__name__}",
        details={"value": repr(value)},
    )


def _riddle_from_mapping(value: Mapping[str, Any]) -> RiddlePayload:
    question = _text(_value_of(value, _QUESTION_KEYS))
    answer = _text(_value_of(value, _ANSWER_KEYS))
    if not question and not answer:
        raw = _value_of(value, _JOKE_KEYS)
        if isinstance(raw, str):
            question, answer = _riddle_text(raw)
        elif raw is not _MISSING and raw is not None:
            question, answer = _pair(raw)
    return RiddlePayload(
        question=question,
        answer=answer,
        hint=_text(_value_of(value, _HINT_KEYS)),
        difficulty=_text(_value_of(value, _DIFFICULTY_KEYS)),
    )


def _merge_riddle(
    payload: RiddlePayload,
    value: Mapping[str, Any],
) -> RiddlePayload:
    outer = _riddle_from_mapping(value)
    return RiddlePayload(
        question=outer.question or payload.question,
        answer=outer.answer or payload.answer,
        hint=outer.hint or payload.hint,
        difficulty=outer.difficulty or payload.difficulty,
    )


def _riddle_from(value: Any) -> RiddlePayload:
    if isinstance(value, RiddlePayload):
        return value
    if isinstance(value, Script) and isinstance(value.payload, RiddlePayload):
        return value.payload
    if isinstance(value, str):
        question, answer = _riddle_text(value)
        return RiddlePayload(question=question, answer=answer)
    if isinstance(value, (bytes, bytearray)) or value is None:
        return RiddlePayload()

    if isinstance(value, Mapping):
        nested = _value_of(value, _PAYLOAD_KEYS)
        if nested is not _MISSING and nested is not None:
            return _merge_riddle(_riddle_from(nested), value)
        return _riddle_from_mapping(value)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        question, answer = _pair(value)
        return RiddlePayload(question=question, answer=answer)

    if (
        _value_of(value, _QUESTION_KEYS) is not _MISSING
        or _value_of(value, _ANSWER_KEYS) is not _MISSING
    ):
        fields = {
            key: getattr(value, key)
            for key in (
                *_QUESTION_KEYS,
                *_ANSWER_KEYS,
                *_HINT_KEYS,
                *_DIFFICULTY_KEYS,
            )
            if hasattr(value, key)
        }
        return _riddle_from_mapping(fields)

    raise ConfigError(
        f"cannot build a riddle from {type(value).__name__}",
        details={"value": repr(value)},
    )


def extract_joke(value: JokeInput = None) -> JokePayload:
    """Return *value* as a typed :class:`JokePayload`.

    A string is split into a setup and a punchline on the first blank line or
    newline. A mapping accepts the schema aliases for setup, punchline, style
    and tags.
    """
    return _joke_from(value)


coerce_joke = extract_joke
parse_joke = extract_joke
normalize_joke = extract_joke
normalise_joke = extract_joke


def extract_riddle(value: RiddleInput = None) -> RiddlePayload:
    """Return *value* as a typed :class:`RiddlePayload`.

    A string is split into a question and an answer on the first blank line or
    newline. A mapping accepts the schema aliases for question, answer, hint and
    difficulty.
    """
    return _riddle_from(value)


coerce_riddle = extract_riddle
parse_riddle = extract_riddle
normalize_riddle = extract_riddle
normalise_riddle = extract_riddle


def _article(word: str) -> str:
    return "an" if word[:1].casefold() in "aeiou" else "a"


def _check_halves(mode: str, first: str, second: str, names: tuple[str, str]) -> None:
    first_name, second_name = names
    if not first and not second:
        raise ConfigError(
            f"{mode} mode needs a {first_name} and {_article(second_name)} "
            f"{second_name}",
            details={
                "mode": mode,
                "missing": [first_name, second_name],
            },
        )
    if not first:
        raise ConfigError(
            f"{mode} mode needs a {first_name}",
            details={"mode": mode, "missing": [first_name]},
        )
    if not second:
        raise ConfigError(
            f"{mode} mode needs {_article(second_name)} {second_name}",
            details={"mode": mode, "missing": [second_name]},
        )


def validate_joke(value: JokeInput = None) -> JokePayload:
    """Return *value* as a payload and require a setup and a punchline."""
    payload = extract_joke(value)
    _check_halves("joke", payload.setup, payload.punchline, ("setup", "punchline"))
    return payload


def validate_riddle(value: RiddleInput = None) -> RiddlePayload:
    """Return *value* as a payload and require a question and an answer."""
    payload = extract_riddle(value)
    _check_halves(
        "riddle",
        payload.question,
        payload.answer,
        ("question", "answer"),
    )
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
    found = _text(_value_of(value, keys))
    return found or None


class JokeMode(BuiltinWritingMode):
    """The joke writing mode: a setup and its punchline.

    :meth:`build_script` turns a joke into a script whose hook opens with the
    setup, whose beats carry the setup and punchline and whose payload keeps the
    typed fields. Both halves are required, so a joke card never silently loses
    its payoff.
    """

    name = "joke"
    description = "A setup followed by its punchline."
    instructions = (
        "Keep the setup short and lead into one clear punchline; never drop "
        "the payoff or repeat the setup as the punchline."
    )

    def build_payload(self, value: JokeInput = None) -> JokePayload:
        """Validate *value* as a :class:`JokePayload` without requiring halves."""
        return extract_joke(value)

    def build_hook(
        self,
        value: JokeInput = None,
        *,
        title: str = "",
        hook: str | None = None,
    ) -> str:
        """Return the opening line, using an explicit *hook* when given."""
        if hook is not None:
            return hook.strip()
        payload = extract_joke(value)
        if not payload.setup:
            return ""
        subject = title.strip()
        if subject:
            return f"{subject}: {payload.setup}"
        return payload.setup

    def build_beats(
        self,
        value: JokeInput = None,
        *,
        speaker: str = "narrator",
        max_beats: int | None = None,
    ) -> list[Beat]:
        """Return one beat for the setup and one for the punchline."""
        payload = extract_joke(value)
        parts = [part for part in (payload.setup, payload.punchline) if part]
        if max_beats is not None:
            parts = parts[:max_beats]
        return [
            Beat(
                index=index,
                text=text,
                dialogue=[DialogueLine(speaker=speaker, text=text)],
            )
            for index, text in enumerate(parts)
        ]

    def build_outro(
        self,
        value: JokeInput = None,
        *,
        outro: str | None = None,
    ) -> str:
        """Return the closing line, using an explicit *outro* when given."""
        if outro is not None:
            return outro.strip()
        return ""

    def build_script(
        self,
        value: JokeInput = None,
        *,
        title: str | None = None,
        language: str | None = None,
        setup: str | None = None,
        punchline: str | None = None,
        style: str | None = None,
        tags: Sequence[str] | str | None = None,
        hook: str | None = None,
        outro: str | None = None,
        speaker: str = "narrator",
        max_beats: int | None = None,
    ) -> Script:
        """Build a joke script from text, a pair, a mapping or a payload."""
        payload = extract_joke(value)
        resolved_setup = setup.strip() if setup is not None else payload.setup
        resolved_punchline = (
            punchline.strip() if punchline is not None else payload.punchline
        )
        _check_halves(
            "joke",
            resolved_setup,
            resolved_punchline,
            ("setup", "punchline"),
        )
        payload = JokePayload(
            setup=resolved_setup,
            punchline=resolved_punchline,
            style=style.strip() if style is not None else payload.style,
            tags=_string_list(tags) if tags is not None else list(payload.tags),
        )
        resolved_title = title if title is not None else _title_of(value)
        return Script(
            mode=WritingMode.JOKE,
            title=resolved_title,
            language=language or _language_of(value, "en"),
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
    build_from_joke = build_script


class RiddleMode(BuiltinWritingMode):
    """The riddle writing mode: a question and its answer.

    :meth:`build_script` turns a riddle into a script whose hook opens with the
    question, whose beats carry the question and answer and whose payload keeps
    the typed fields plus an optional hint and difficulty. Both halves are
    required, so a riddle card never silently loses its answer.
    """

    name = "riddle"
    description = "A question followed by its answer and an optional hint."
    instructions = (
        "Ask one fair question, reveal one answer and keep any hint from giving "
        "the answer away."
    )

    def build_payload(self, value: RiddleInput = None) -> RiddlePayload:
        """Validate *value* as a :class:`RiddlePayload` without requiring halves."""
        return extract_riddle(value)

    def build_hook(
        self,
        value: RiddleInput = None,
        *,
        title: str = "",
        hook: str | None = None,
    ) -> str:
        """Return the opening question, using an explicit *hook* when given."""
        if hook is not None:
            return hook.strip()
        payload = extract_riddle(value)
        if not payload.question:
            return ""
        subject = title.strip()
        if subject:
            return f"{subject}: {payload.question}"
        return payload.question

    def build_beats(
        self,
        value: RiddleInput = None,
        *,
        speaker: str = "narrator",
        max_beats: int | None = None,
    ) -> list[Beat]:
        """Return one beat for the question and one for the answer."""
        payload = extract_riddle(value)
        parts = [part for part in (payload.question, payload.answer) if part]
        if max_beats is not None:
            parts = parts[:max_beats]
        return [
            Beat(
                index=index,
                text=text,
                dialogue=[DialogueLine(speaker=speaker, text=text)],
            )
            for index, text in enumerate(parts)
        ]

    def build_outro(
        self,
        value: RiddleInput = None,
        *,
        outro: str | None = None,
    ) -> str:
        """Return the closing line, using an explicit *outro* when given."""
        if outro is not None:
            return outro.strip()
        return ""

    def build_script(
        self,
        value: RiddleInput = None,
        *,
        title: str | None = None,
        language: str | None = None,
        question: str | None = None,
        answer: str | None = None,
        hint: str | None = None,
        difficulty: str | None = None,
        hook: str | None = None,
        outro: str | None = None,
        speaker: str = "narrator",
        max_beats: int | None = None,
    ) -> Script:
        """Build a riddle script from text, a pair, a mapping or a payload."""
        payload = extract_riddle(value)
        resolved_question = (
            question.strip() if question is not None else payload.question
        )
        resolved_answer = answer.strip() if answer is not None else payload.answer
        _check_halves(
            "riddle",
            resolved_question,
            resolved_answer,
            ("question", "answer"),
        )
        payload = RiddlePayload(
            question=resolved_question,
            answer=resolved_answer,
            hint=hint.strip() if hint is not None else payload.hint,
            difficulty=(
                difficulty.strip() if difficulty is not None else payload.difficulty
            ),
        )
        resolved_title = title if title is not None else _title_of(value)
        return Script(
            mode=WritingMode.RIDDLE,
            title=resolved_title,
            language=language or _language_of(value, "en"),
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
    build_from_riddle = build_script


JokeWritingMode = JokeMode
Joke = JokeMode
RiddleWritingMode = RiddleMode
Riddle = RiddleMode


def build_joke_script(
    value: JokeInput = None,
    **kwargs: Any,
) -> Script:
    """Build a joke script with a default :class:`JokeMode`."""
    return JokeMode().build_script(value, **kwargs)


joke_script = build_joke_script


def build_riddle_script(
    value: RiddleInput = None,
    **kwargs: Any,
) -> Script:
    """Build a riddle script with a default :class:`RiddleMode`."""
    return RiddleMode().build_script(value, **kwargs)


riddle_script = build_riddle_script


register_builtin_writing_mode(WritingMode.JOKE, JokeMode)
register_builtin_writing_mode(WritingMode.RIDDLE, RiddleMode)

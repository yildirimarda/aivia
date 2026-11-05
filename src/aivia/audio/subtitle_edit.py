"""Edit subtitle words and timings without re-synthesizing audio.

The subtitle stage derives one ASS document per language from a timeline's
timed words. Fixing a word or nudging a timing should not pay for voice
synthesis or re-render every language. This module exposes the effective words
of each language as a small YAML-friendly document and applies an edited
document back to the timeline, reporting which languages actually changed.

Only data transformation lives here: no file is read, no provider is called and
no binary runs. The CLI command owns file IO and rendering.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import yaml
from pydantic import ValidationError as PydanticValidationError

from aivia.compose.subtitles import SubtitleTrack, SubtitleWord
from aivia.compose.timeline import Timeline, TimelineLanguage
from aivia.core.errors import ConfigError

__all__ = [
    "SubtitleEditError",
    "apply_subtitle_edits",
    "load_subtitle_edit_yaml",
    "parse_subtitle_edit_document",
    "subtitle_edit_document",
    "subtitle_edit_yaml",
    "subtitle_words_by_language",
]

_LANGUAGE_KEYS = ("language", "lang", "locale", "code")


class SubtitleEditError(ConfigError):
    """A subtitle edit document is missing or invalid."""

    code = "subtitle_edit"


def _language_key(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise SubtitleEditError(
            "language must be a string",
            details={"value": type(value).__name__},
        )
    return value.strip().replace("_", "-").lower()


def _effective_words(entry: TimelineLanguage) -> list[SubtitleWord]:
    """Return the words the subtitle stage would burn for *entry*."""
    track = entry.subtitles
    if track is not None:
        if track.words:
            return list(track.words)
        if track.cues:
            return [word for cue in track.cues for word in cue.words]
        return []
    return [word for voice in entry.voices if voice.enabled for word in voice.words]


def _word_document(word: SubtitleWord) -> dict[str, Any]:
    return word.model_dump(mode="python", by_alias=False, exclude_none=True)


def _word_signature(words: Sequence[SubtitleWord]) -> list[tuple[Any, ...]]:
    return [
        (word.word, float(word.start_seconds), float(word.end_seconds or 0.0))
        for word in words
    ]


def _language_map(timeline: Timeline) -> dict[str, TimelineLanguage]:
    mapping: dict[str, TimelineLanguage] = {}
    for entry in timeline.resolved_languages():
        mapping.setdefault(_language_key(entry.language), entry)
    return mapping


def subtitle_words_by_language(
    timeline: Timeline,
    *,
    languages: Sequence[str] | str | None = None,
) -> dict[str, list[SubtitleWord]]:
    """Return the effective timed words for every requested language.

    An unknown language is an error rather than an empty result, so a typo
    cannot silently export or wipe a different locale.
    """
    available = _language_map(timeline)
    requested = _requested(languages)
    if requested:
        missing = [code for code in requested if code not in available]
        if missing:
            raise SubtitleEditError(
                "timeline has no such language",
                details={"missing": missing, "available": list(available)},
            )
        codes = requested
    else:
        codes = list(available)

    result: dict[str, list[SubtitleWord]] = {}
    for code in codes:
        result[code] = list(_effective_words(available[code]))
    return result


def subtitle_edit_document(
    timeline: Timeline,
    *,
    languages: Sequence[str] | str | None = None,
) -> dict[str, Any]:
    """Return a YAML-friendly document of every language's words and timings."""
    words = subtitle_words_by_language(timeline, languages=languages)
    return {
        "languages": [
            {
                "language": code,
                "words": [_word_document(word) for word in items],
            }
            for code, items in words.items()
        ]
    }


def subtitle_edit_yaml(
    timeline: Timeline,
    *,
    languages: Sequence[str] | str | None = None,
) -> str:
    """Return the subtitle edit document as deterministic YAML text."""
    return yaml.safe_dump(
        subtitle_edit_document(timeline, languages=languages),
        sort_keys=False,
        allow_unicode=True,
    )


def load_subtitle_edit_yaml(text: str) -> dict[str, list[SubtitleWord]]:
    """Parse YAML text into validated words per language."""
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SubtitleEditError(
            f"invalid subtitle YAML: {exc}",
            details={"field": "yaml"},
        ) from exc
    return parse_subtitle_edit_document(payload)


def parse_subtitle_edit_document(
    document: Any,
) -> dict[str, list[SubtitleWord]]:
    """Validate an edit document and return its words keyed by language.

    Both the exported ``languages`` list and a shorthand mapping of language
    code to words are accepted. Words must already be ordered by start time,
    because reordering them would silently rewrite the spoken sequence.
    """
    if not isinstance(document, Mapping):
        raise SubtitleEditError(
            "subtitle edit document must be a mapping",
            details={"field": "document"},
        )

    raw = document.get("languages")
    if raw is None:
        raw = {
            key: value
            for key, value in document.items()
            if key not in {"timeline", "run", "id", "metadata"}
        }

    items: list[Any]
    if isinstance(raw, Mapping):
        items = []
        for code, value in raw.items():
            if isinstance(value, Mapping):
                item = dict(value)
                item.setdefault("language", code)
            elif isinstance(value, (list, tuple)):
                item = {"language": code, "words": list(value)}
            else:
                raise SubtitleEditError(
                    "each language entry must be a mapping or a word list",
                    details={"language": code},
                )
            items.append(item)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        items = list(raw)
    else:
        raise SubtitleEditError(
            "subtitle edit document needs a languages list or mapping",
            details={"field": "languages"},
        )

    parsed: dict[str, list[SubtitleWord]] = {}
    for position, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise SubtitleEditError(
                "each language entry must be a mapping",
                details={"index": position},
            )
        code = _entry_language(item)
        if not code:
            raise SubtitleEditError(
                "each language entry needs a language code",
                details={"index": position},
            )
        if code in parsed:
            raise SubtitleEditError(
                "duplicate language in the edit document",
                details={"language": code},
            )
        parsed[code] = _parse_words(item.get("words"), code)
    return parsed


def _entry_language(item: Mapping[str, Any]) -> str:
    for key in _LANGUAGE_KEYS:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return _language_key(value)
    return ""


def _parse_words(value: Any, language: str) -> list[SubtitleWord]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise SubtitleEditError(
            "words must be a list",
            details={"language": language},
        )
    words: list[SubtitleWord] = []
    for index, item in enumerate(value):
        try:
            words.append(SubtitleWord.model_validate(item))
        except (PydanticValidationError, TypeError, ValueError) as exc:
            raise SubtitleEditError(
                "invalid subtitle word",
                details={
                    "language": language,
                    "index": index,
                    "error": str(exc),
                },
            ) from exc
    starts = [word.start_seconds for word in words]
    if starts != sorted(starts):
        raise SubtitleEditError(
            "subtitle words must be ordered by start time",
            details={"language": language},
        )
    return words


def _requested(languages: Sequence[str] | str | None) -> list[str]:
    if languages is None:
        return []
    if isinstance(languages, str):
        values: Sequence[Any] = [languages]
    else:
        values = languages
    codes: list[str] = []
    for value in values:
        for part in str(value).split(","):
            code = _language_key(part)
            if code and code not in codes:
                codes.append(code)
    return codes


def _normalise_timeline(timeline: Timeline) -> Timeline:
    """Fold top-level single-language fields into the ``languages`` list."""
    if not (timeline.language or timeline.voices or timeline.subtitles is not None):
        return timeline
    entries = timeline.resolved_languages()
    data = timeline.model_dump(mode="python")
    data["languages"] = [entry.model_dump(mode="python") for entry in entries]
    data["language"] = ""
    data["voices"] = []
    data["subtitles"] = None
    return Timeline.model_validate(data)


def apply_subtitle_edits(
    timeline: Timeline,
    document: Any,
) -> tuple[Timeline, list[str]]:
    """Apply an edit document and return the timeline plus changed languages.

    Only languages whose word text or timing differs are reported as changed,
    so a caller can regenerate and re-render just those outputs.
    """
    updates = (
        load_subtitle_edit_yaml(document)
        if isinstance(document, str)
        else parse_subtitle_edit_document(document)
    )
    normalised = _normalise_timeline(timeline)
    entries = list(normalised.languages)
    by_code: dict[str, int] = {}
    for index, entry in enumerate(entries):
        by_code.setdefault(_language_key(entry.language), index)

    changed: list[str] = []
    for code, words in updates.items():
        position = by_code.get(code)
        if position is None:
            raise SubtitleEditError(
                "timeline has no such language",
                details={"language": code, "available": list(by_code)},
            )
        entry = entries[position]
        if _word_signature(_effective_words(entry)) == _word_signature(words):
            continue
        existing = entry.subtitles
        track = SubtitleTrack.from_words(
            words,
            language=code,
            style=None if existing is None else existing.style,
            metadata={} if existing is None else dict(existing.metadata),
        )
        entries[position] = entry.model_copy(update={"subtitles": track})
        changed.append(code)

    if not changed:
        return normalised, []
    return normalised.model_copy(update={"languages": entries}), changed

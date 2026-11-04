"""Wrap writer sources and labels as untrusted data blocks.

The writing stage receives text that it did not author: fact statements, source
URLs, language labels, forbidden phrases and the topic a caller supplied. Those
values are data. This module renders them inside the shared
:mod:`aivia.llm.untrusted` delimited blocks, with each source URL and label
escaped in the block metadata, so hostile text cannot close a block and pose as
a new instruction.

The helpers are deterministic and offline. They do not decide what a prompt
means or call a provider; they only give a writer prompt one safe place to put
untrusted values. ``escape_writer_context`` performs the matching escaping for
values that are also serialised as JSON, so the same hostile string is never
left raw in the prompt.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from html import escape
from typing import Any

from aivia.llm.untrusted import (
    DATA_ONLY_INSTRUCTION,
    DataBlock,
    wrap_untrusted,
    wrap_untrusted_many,
)

__all__ = [
    "DEFAULT_FACT_LABEL",
    "DEFAULT_LABEL_LABEL",
    "DEFAULT_SOURCE_LABEL",
    "WRITER_DATA_INSTRUCTION",
    "build_writer_prompt",
    "escape_writer_context",
    "escape_writer_text",
    "render_writer_context",
    "render_writer_data",
    "wrap_label",
    "wrap_labels",
    "wrap_source",
    "wrap_sources",
    "wrap_writer_context",
    "wrap_writer_data",
    "wrap_writer_label",
    "wrap_writer_labels",
    "wrap_writer_source",
    "wrap_writer_sources",
]

WRITER_DATA_INSTRUCTION = DATA_ONLY_INSTRUCTION
"""The data-only instruction placed before writer data blocks."""

DEFAULT_SOURCE_LABEL = "source"
"""The kind used when a source value has no explicit label."""

DEFAULT_FACT_LABEL = "fact"
"""The kind used for a fact statement."""

DEFAULT_LABEL_LABEL = "label"
"""The kind used when only a label is wrapped."""

_TEXT_KEYS = ("text", "statement", "claim", "fact", "content", "value")
_URL_KEYS = ("source_urls", "sources", "urls", "source_url", "source")
_MISSING = object()


def _text(value: Any, field: str) -> str:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if isinstance(value, bytes):
        raise TypeError(f"{field} must be text, not bytes")
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string or path")
    return value


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


def _urls(value: Any) -> list[str]:
    if value is _MISSING or value is None:
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


def _fact_blocks(value: Any, *, label: str) -> list[DataBlock] | None:
    text = _value_of(value, _TEXT_KEYS)
    urls = _urls(_value_of(value, _URL_KEYS))
    if text is _MISSING or not isinstance(text, str) or not text.strip():
        return None
    blocks = [
        DataBlock(
            content=text.strip(),
            kind=label,
            source=urls[0] if urls else None,
        )
    ]
    blocks.extend(DataBlock(content=url, kind=DEFAULT_SOURCE_LABEL) for url in urls)
    return blocks


def _source_blocks(items: Any, *, label: str) -> list[DataBlock]:
    if items is None:
        return []
    if isinstance(items, DataBlock):
        return [items]
    if isinstance(items, (str, os.PathLike)):
        return [DataBlock(content=items, kind=label)]
    if isinstance(items, Mapping):
        blocks: list[DataBlock] = []
        for item_label, item_value in items.items():
            name = str(item_label)
            fact = _fact_blocks(item_value, label=name)
            if fact is not None:
                blocks.extend(fact)
            else:
                blocks.append(DataBlock(content=item_value, kind=name))
        return blocks
    if isinstance(items, Iterable) and not isinstance(items, (bytes, bytearray)):
        blocks = []
        for item in items:
            if isinstance(item, DataBlock):
                blocks.append(item)
                continue
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                blocks.append(DataBlock(content=item[1], kind=item[0]))
                continue
            fact = _fact_blocks(item, label=label)
            if fact is not None:
                blocks.extend(fact)
                continue
            blocks.append(DataBlock(content=item, kind=label))
        return blocks
    raise TypeError("sources must be text, a mapping or a sequence")


def _label_blocks(items: Any) -> list[DataBlock]:
    if items is None:
        return []
    if isinstance(items, DataBlock):
        return [items]
    if isinstance(items, Mapping):
        return [
            DataBlock(content=value, kind=DEFAULT_LABEL_LABEL, source=str(label))
            for label, value in items.items()
        ]
    if isinstance(items, (str, os.PathLike)):
        return [DataBlock(content=items, kind=DEFAULT_LABEL_LABEL)]
    if isinstance(items, Iterable) and not isinstance(items, (bytes, bytearray)):
        blocks: list[DataBlock] = []
        for item in items:
            if isinstance(item, DataBlock):
                blocks.append(item)
            elif (
                isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
            ):
                blocks.append(DataBlock(content=item[1], kind=item[0]))
            else:
                blocks.append(DataBlock(content=item, kind=DEFAULT_LABEL_LABEL))
        return blocks
    raise TypeError("labels must be text, a mapping or a sequence")


def wrap_source(
    content: Any,
    *,
    label: str = DEFAULT_SOURCE_LABEL,
    source: str | os.PathLike[str] | None = None,
    instruction: str | None = None,
) -> str:
    """Wrap one source value as a delimited data block."""
    fact = _fact_blocks(content, label=label)
    if fact is not None:
        return wrap_untrusted_many(fact, instruction=instruction)
    return wrap_untrusted(
        content,
        kind=label,
        source=source,
        instruction=instruction,
    )


def wrap_sources(
    items: Any,
    *,
    label: str = DEFAULT_SOURCE_LABEL,
    instruction: str | None = None,
) -> str:
    """Wrap source values, sharing one data-only instruction."""
    return wrap_untrusted_many(
        _source_blocks(items, label=label),
        instruction=instruction,
    )


def wrap_label(
    label: str,
    value: Any = None,
    *,
    source: str | os.PathLike[str] | None = None,
    instruction: str | None = None,
) -> str:
    """Wrap a label, or a value under that label, as untrusted data."""
    if value is None:
        return wrap_untrusted(
            label,
            kind=DEFAULT_LABEL_LABEL,
            source=source,
            instruction=instruction,
        )
    return wrap_untrusted(
        value,
        kind=label,
        source=source,
        instruction=instruction,
    )


def wrap_labels(
    items: Any,
    *,
    instruction: str | None = None,
) -> str:
    """Wrap label values, sharing one data-only instruction."""
    return wrap_untrusted_many(
        _label_blocks(items),
        instruction=instruction,
    )


def render_writer_data(
    sources: Any = None,
    labels: Any = None,
    *,
    label: str = DEFAULT_SOURCE_LABEL,
    instruction: str | None = None,
) -> str:
    """Render sources and labels together as one untrusted data section."""
    blocks = [
        *_source_blocks(sources, label=label),
        *_label_blocks(labels),
    ]
    return wrap_untrusted_many(blocks, instruction=instruction)


def render_writer_context(
    sources: Any = None,
    labels: Any = None,
    *,
    label: str = DEFAULT_SOURCE_LABEL,
    instruction: str | None = None,
) -> str:
    """Alias for :func:`render_writer_data`."""
    return render_writer_data(
        sources,
        labels,
        label=label,
        instruction=instruction,
    )


wrap_writer_data = render_writer_data
wrap_writer_context = render_writer_context


def wrap_writer_source(
    content: Any,
    *,
    label: str = DEFAULT_SOURCE_LABEL,
    source: str | os.PathLike[str] | None = None,
    instruction: str | None = None,
) -> str:
    """Alias for :func:`wrap_source`."""
    return wrap_source(
        content,
        label=label,
        source=source,
        instruction=instruction,
    )


def wrap_writer_sources(
    items: Any,
    *,
    label: str = DEFAULT_SOURCE_LABEL,
    instruction: str | None = None,
) -> str:
    """Alias for :func:`wrap_sources`."""
    return wrap_sources(items, label=label, instruction=instruction)


def wrap_writer_label(
    label: str,
    value: Any = None,
    *,
    source: str | os.PathLike[str] | None = None,
    instruction: str | None = None,
) -> str:
    """Alias for :func:`wrap_label`."""
    return wrap_label(
        label,
        value,
        source=source,
        instruction=instruction,
    )


def wrap_writer_labels(
    items: Any,
    *,
    instruction: str | None = None,
) -> str:
    """Alias for :func:`wrap_labels`."""
    return wrap_labels(items, instruction=instruction)


def escape_writer_text(value: str) -> str:
    """Escape one writer value for safe inclusion in prompt text."""
    return escape(_text(value, "value"), quote=False)


def escape_writer_context(value: Any) -> Any:
    """Escape every string in a writer context before JSON serialisation.

    The context still has the same shape and ordering, but a hostile value can
    no longer introduce a literal data-block tag or markup into the prompt.
    Non-string scalars are returned unchanged.
    """
    if isinstance(value, str):
        return escape_writer_text(value)
    if isinstance(value, Mapping):
        return {
            escape_writer_text(key) if isinstance(key, str) else key: (
                escape_writer_context(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(escape_writer_context(item) for item in value)
    if isinstance(value, list):
        return [escape_writer_context(item) for item in value]
    return value


def build_writer_prompt(
    instructions: str = "",
    sources: Any = None,
    labels: Any = None,
    *,
    label: str = DEFAULT_SOURCE_LABEL,
    instruction: str | None = None,
) -> str:
    """Return instructions followed by sources and labels as data blocks.

    An empty ``instructions`` string is allowed: the result is then only the
    safe data section. When there is no data, the instructions are returned
    unchanged.
    """
    text = instructions.strip() if isinstance(instructions, str) else ""
    data = render_writer_data(
        sources,
        labels,
        label=label,
        instruction=instruction,
    )
    if text and data:
        return f"{text}\n\n{data}"
    return text or data

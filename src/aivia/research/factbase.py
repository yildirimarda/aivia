"""Build a compact fact list with source URLs from research output.

The fact list is the contract the writing stage consumes. It is structural:
anything that exposes a ``claims`` sequence is a fact list, and every claim
carries ``text`` and ``source_urls``. This module deliberately imports no core
code so research can hand facts to writing without taking a core dependency;
the records below simply satisfy the same shape.

Building is deterministic and offline. Facts are stripped and de-duplicated by
their normalized text, and duplicate facts merge their source URLs. A fact
without text or without at least one absolute http(s) source URL is dropped,
because the accuracy invariant says every fact maps to a source.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

__all__ = [
    "DEFAULT_MAX_FACTS",
    "CompactFact",
    "Fact",
    "FactBase",
    "FactList",
    "FactListContract",
    "build_fact_list",
    "build_factbase",
    "compact_fact_list",
    "compact_facts",
    "fact_list",
    "to_fact_list",
]

DEFAULT_MAX_FACTS: int | None = None
"""No default cap; callers choose a limit when they need one."""

_TEXT_KEYS = ("text", "statement", "claim", "fact", "content")
_URL_KEYS = ("source_urls", "sources", "urls", "source_url", "source")
_TITLE_KEYS = ("title", "name", "topic", "subject")
_MISSING = object()
_MAX_UNIT = 1.0


def _clean_text(value: Any, label: str = "fact text") -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{label} must be non-empty")
    return text


def _clean_urls(values: Iterable[Any]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for raw in values:
        if not isinstance(raw, str):
            raise ValueError("source URL must be a string")
        url = raw.strip()
        if not url:
            raise ValueError("source URL must be non-empty")
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"source URL must be an absolute http(s) URL: {url!r}")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def _coerce_urls(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        url = value.get("url") or value.get("source_url")
        return [url] if url is not None else []
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return list(value)
    raise ValueError("source URLs must be a URL, a mapping or a sequence")


def _coerce_unit(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be between zero and one")
    return number


def _normalise_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def _host(url: str) -> str:
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _mapping_value(value: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in value:
            return value[key]
    return _MISSING


def _attribute_value(value: Any, keys: Sequence[str]) -> Any:
    for key in keys:
        if hasattr(value, key):
            return getattr(value, key)
    return _MISSING


@dataclass(frozen=True, slots=True)
class Fact:
    """One compact factual statement and the URLs that support it."""

    text: str
    source_urls: list[str]
    confidence: float = _MAX_UNIT
    verified: bool = False

    def __post_init__(self) -> None:
        text = _clean_text(self.text)
        urls = _clean_urls(self.source_urls)
        if not urls:
            raise ValueError("fact requires at least one source URL")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "source_urls", urls)
        object.__setattr__(
            self, "confidence", _coerce_unit(self.confidence, "confidence")
        )
        object.__setattr__(self, "verified", bool(self.verified))

    @property
    def statement(self) -> str:
        """Alias for :attr:`text`."""
        return self.text

    @property
    def sources(self) -> list[str]:
        """Alias for :attr:`source_urls`."""
        return list(self.source_urls)

    @property
    def urls(self) -> list[str]:
        """Alias for :attr:`source_urls`."""
        return list(self.source_urls)

    @property
    def source_domains(self) -> list[str]:
        """The distinct hostnames behind the fact's sources."""
        domains: list[str] = []
        for url in self.source_urls:
            host = _host(url)
            if host and host not in domains:
                domains.append(host)
        return domains

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the fact."""
        return {
            "text": self.text,
            "source_urls": list(self.source_urls),
            "confidence": self.confidence,
            "verified": self.verified,
        }


CompactFact = Fact
"""Alias for :class:`Fact` for callers that use the compact wording."""


@dataclass(frozen=True, slots=True)
class FactList:
    """A compact fact list satisfying the core structural contract."""

    claims: list[Fact] = field(default_factory=list)
    title: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "claims", list(self.claims))
        object.__setattr__(self, "title", self.title.strip())

    @property
    def facts(self) -> list[Fact]:
        """Alias for :attr:`claims`."""
        return list(self.claims)

    @property
    def source_urls(self) -> list[str]:
        """Every source URL in first-seen order, without duplicates."""
        urls: list[str] = []
        for claim in self.claims:
            for url in claim.source_urls:
                if url not in urls:
                    urls.append(url)
        return urls

    @property
    def urls(self) -> list[str]:
        """Alias for :attr:`source_urls`."""
        return self.source_urls

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the fact list."""
        return {
            "title": self.title,
            "claims": [claim.to_dict() for claim in self.claims],
            "source_urls": self.source_urls,
        }

    def __iter__(self) -> Iterator[Fact]:
        return iter(self.claims)

    def __len__(self) -> int:
        return len(self.claims)

    def __bool__(self) -> bool:
        return bool(self.claims)


FactListContract = FactList
"""Alias for :class:`FactList` for callers that use the contract wording."""

FactBase = FactList
"""Alias for :class:`FactList` for callers that prefer the fact base wording."""


def _claim_value(value: Any, keys: Sequence[str]) -> Any:
    if isinstance(value, Mapping):
        return _mapping_value(value, keys)
    return _attribute_value(value, keys)


def _fact_from(value: Any) -> Fact | None:
    if isinstance(value, Fact):
        return value
    if isinstance(value, (str, bytes, bytearray)) or value is None:
        return None

    text = _claim_value(value, _TEXT_KEYS)
    urls = _claim_value(value, _URL_KEYS)
    if text is _MISSING or urls is _MISSING:
        return None
    if not isinstance(text, str) or not text.strip():
        return None

    source_urls = _coerce_urls(urls)
    if not source_urls:
        return None

    confidence = _claim_value(value, ("confidence",))
    verified = _claim_value(value, ("verified",))
    return Fact(
        text=text,
        source_urls=source_urls,
        confidence=(
            _MAX_UNIT if confidence is _MISSING or confidence is None else confidence
        ),
        verified=False if verified is _MISSING else bool(verified),
    )


def _iter_facts(value: Any) -> Iterator[Fact]:
    if value is None:
        return
    if isinstance(value, Fact):
        yield value
        return
    if isinstance(value, Mapping):
        nested = _mapping_value(value, ("claims", "facts"))
        if nested is not _MISSING:
            yield from _iter_facts(nested)
            return
        topic = value.get("topic")
        if isinstance(topic, Mapping):
            nested = _mapping_value(topic, ("claims", "facts"))
            if nested is not _MISSING:
                yield from _iter_facts(nested)
                return
        fact = _fact_from(value)
        if fact is not None:
            yield fact
        return
    if isinstance(value, (str, bytes, bytearray)):
        return

    claims = getattr(value, "claims", _MISSING)
    if claims is not _MISSING and not callable(claims):
        yield from _iter_facts(claims)
        return

    fact = _fact_from(value)
    if fact is not None:
        yield fact
        return
    if (
        _claim_value(value, _TEXT_KEYS) is not _MISSING
        or _claim_value(value, _URL_KEYS) is not _MISSING
    ):
        return

    if isinstance(value, Iterable):
        for item in value:
            yield from _iter_facts(item)
        return

    raise ValueError(f"cannot build a fact list from {value!r}")


def _title_of(value: Any) -> str:
    if isinstance(value, Mapping):
        raw = _mapping_value(value, _TITLE_KEYS)
        if raw is _MISSING and isinstance(value.get("topic"), Mapping):
            raw = _mapping_value(value["topic"], _TITLE_KEYS)
    else:
        raw = _attribute_value(value, _TITLE_KEYS)
    return raw.strip() if isinstance(raw, str) else ""


def _compact(facts: Iterable[Fact]) -> list[Fact]:
    compacted: list[Fact] = []
    positions: dict[str, int] = {}
    for fact in facts:
        key = _normalise_text(fact.text)
        if not key:
            continue
        position = positions.get(key)
        if position is None:
            positions[key] = len(compacted)
            compacted.append(fact)
            continue
        existing = compacted[position]
        merged = list(existing.source_urls)
        for url in fact.source_urls:
            if url not in merged:
                merged.append(url)
        compacted[position] = Fact(
            text=existing.text,
            source_urls=merged,
            confidence=max(existing.confidence, fact.confidence),
            verified=existing.verified or fact.verified,
        )
    return compacted


def compact_fact_list(
    value: Any = None,
    *,
    title: str | None = None,
    max_facts: int | None = DEFAULT_MAX_FACTS,
) -> FactList:
    """Build a compact :class:`FactList` from research output or claim records.

    ``value`` may be an object exposing ``claims`` (such as a research topic),
    a mapping with ``claims`` or ``facts``, a sequence of claims, or one claim.
    Claims are read structurally through ``text``/``statement`` and
    ``source_urls``/``sources``, so no core model import is needed. Claims with
    no text or no source URL are dropped; duplicate texts merge their sources.
    """
    if max_facts is not None:
        if isinstance(max_facts, bool) or not isinstance(max_facts, int):
            raise ValueError("max facts must be an integer")
        if max_facts < 0:
            raise ValueError("max facts must not be negative")
    facts = _compact(_iter_facts(value))
    if max_facts is not None:
        facts = facts[:max_facts]
    resolved_title = _title_of(value) if title is None else _clean_text(title, "title")
    return FactList(claims=facts, title=resolved_title)


build_fact_list = compact_fact_list
"""Alias for :func:`compact_fact_list`."""

build_factbase = compact_fact_list
"""Alias for :func:`compact_fact_list`."""

fact_list = compact_fact_list
"""Alias for :func:`compact_fact_list`."""

compact_facts = compact_fact_list
"""Alias for :func:`compact_fact_list`."""

to_fact_list = compact_fact_list
"""Alias for :func:`compact_fact_list`."""

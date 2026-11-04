"""Lint a title, caption and cover text as one publication pairing.

The three pieces of text around a video are read together by a viewer but are
often written apart. A title can be cut by a platform limit, the cover can
repeat it word for word, and the caption can promise something vague that the
title never names. This module checks all three in one deterministic pass:

* **truncation** -- a field is over its own character limit, ends in an
  ellipsis that may hide a cut, or repeats itself;
* **duplication** -- two fields say the same thing, ignoring case, accents and
  punctuation;
* **vagueness** -- a field is empty, has no content words, or leans on
  clickbait wording such as ``you won't believe`` or ``amazing``.

The lint is offline and dependency free. It never calls a provider, rewrites a
field or changes the inputs; the same three strings and limits always produce
the same findings. Limits normally come from a platform's caption data, but a
caller can pass explicit title, caption and cover limits instead.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aivia.core.config.writing import DEFAULT_COVER_TEXT_MAX_LENGTH
from aivia.core.similarity import (
    DEFAULT_TITLE_THRESHOLD,
    normalize_title,
    title_similarity,
)

__all__ = [
    "DEFAULT_COVER_MAX_LENGTH",
    "DEFAULT_COVER_TEXT_MAX_LENGTH",
    "DEFAULT_DUPLICATE_THRESHOLD",
    "DEFAULT_VAGUE_TERMS",
    "TitleCoverFinding",
    "TitleCoverFindings",
    "TitleCoverLint",
    "TitleCoverLintResult",
    "TitleCoverLinter",
    "TitleCoverPairing",
    "TitleCoverReport",
    "check_title_cover",
    "lint_caption_set",
    "lint_cover_pairing",
    "lint_pairing",
    "lint_script",
    "lint_title_caption_cover",
    "lint_title_cover",
    "title_cover_findings",
]

DEFAULT_DUPLICATE_THRESHOLD = DEFAULT_TITLE_THRESHOLD
"""Similarity at or above which two fields are treated as duplicates."""

DEFAULT_COVER_MAX_LENGTH = DEFAULT_COVER_TEXT_MAX_LENGTH
"""The fallback cover text limit when no platform or explicit limit is given."""

DEFAULT_VAGUE_TERMS: tuple[str, ...] = (
    "amazing",
    "awesome",
    "changes everything",
    "crazy",
    "epic",
    "game changer",
    "game-changer",
    "here's why",
    "incredible",
    "insane",
    "life changing",
    "life-changing",
    "mind blowing",
    "mind-blowing",
    "must see",
    "must-see",
    "no one talks about",
    "nobody talks about",
    "shocking",
    "shocking truth",
    "something",
    "someone",
    "somewhere",
    "stuff",
    "the real reason",
    "the truth about",
    "things you didn't know",
    "things you dont know",
    "this is why",
    "unbelievable",
    "wait for it",
    "watch until the end",
    "what happens next",
    "will blow your mind",
    "you need to know",
    "you won't believe",
    "you wont believe",
)
"""Wording that promises a story without naming its subject."""

_FIELDS = ("title", "caption", "cover")
_LIMIT_KEYS = {
    "title": "max_title_length",
    "caption": "max_caption_length",
    "cover": "max_cover_length",
}
_ELLIPSIS = re.compile(r"(?:\u2026|\.{3,})\s*$", re.UNICODE)
_REPEATED_WORD = re.compile(
    r"\b([^\W_]+)(?:\s+\1\b)+",
    re.IGNORECASE | re.UNICODE,
)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+", re.UNICODE)

_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "she",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "those",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "you",
        "your",
    }
)

_SEVERITY = {
    "truncated": "error",
    "duplicate": "warning",
    "vague": "warning",
}


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(f"text must be a string, not {type(value).__name__}")
    return " ".join(value.split())


def _validate_limit(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer or None")
    if value < 0:
        raise ValueError(f"{label} must not be negative")
    return value


def _validate_threshold(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("similarity threshold must be a number")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError("similarity threshold must be between 0 and 1")
    return number


def _normalise_terms(values: Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    terms: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError("vague terms must be strings")
        text = _clean(value)
        if text:
            terms.append(text)
    return tuple(dict.fromkeys(terms))


def _vague_terms(text: str, extra: Sequence[str]) -> list[str]:
    folded = normalize_title(text)
    if not folded:
        return []
    found: list[str] = []
    for term in (*DEFAULT_VAGUE_TERMS, *extra):
        key = normalize_title(term)
        if not key:
            continue
        if re.search(rf"(?<!\w){re.escape(key)}(?!\w)", folded):
            found.append(term)
    return list(dict.fromkeys(found))


def _repeated_words(text: str) -> list[str]:
    return list(
        dict.fromkeys(match.group(1) for match in _REPEATED_WORD.finditer(text))
    )


def _repeated_sentences(text: str) -> list[str]:
    seen: set[str] = set()
    repeated: list[str] = []
    for sentence in _SENTENCE_BREAK.split(text):
        key = normalize_title(sentence)
        if len(key.split()) < 4:
            continue
        if key in seen and key not in repeated:
            repeated.append(key)
        seen.add(key)
    return repeated


@dataclass(frozen=True, slots=True)
class TitleCoverFinding:
    """One reason a title, caption or cover text is not ready to publish."""

    field: str
    code: str
    message: str
    severity: str = "warning"
    length: int | None = None
    limit: int | None = None
    related: str = ""

    @property
    def path(self) -> str:
        """Alias for :attr:`field`."""
        return self.field

    @property
    def kind(self) -> str:
        """Alias for :attr:`code`."""
        return self.code

    def to_dict(self) -> dict[str, Any]:
        """Return the finding as a JSON-compatible mapping."""
        return {
            "field": self.field,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "length": self.length,
            "limit": self.limit,
            "related": self.related,
        }

    def __str__(self) -> str:
        return f"{self.field}: {self.message}" if self.field else self.message


class TitleCoverFindings(list[TitleCoverFinding]):
    """Every finding from one lint pass, with the checked pairing attached."""

    def __init__(
        self,
        iterable: Sequence[TitleCoverFinding] = (),
        *,
        title: str = "",
        caption: str = "",
        cover_text: str = "",
        limits: Mapping[str, int | None] | None = None,
    ) -> None:
        super().__init__(iterable)
        self.title = title
        self.caption = caption
        self.cover_text = cover_text
        self.limits = dict(limits or {})

    @property
    def findings(self) -> list[TitleCoverFinding]:
        """The findings as a plain list."""
        return list(self)

    @property
    def issues(self) -> TitleCoverFindings:
        """Alias for the collection itself."""
        return self

    @property
    def ok(self) -> bool:
        """Whether the pairing has no findings."""
        return not self

    @property
    def errors(self) -> list[TitleCoverFinding]:
        """The findings severe enough to block publication."""
        return [finding for finding in self if finding.severity == "error"]

    @property
    def warnings(self) -> list[TitleCoverFinding]:
        """The advisory findings."""
        return [finding for finding in self if finding.severity == "warning"]

    @property
    def truncated(self) -> list[TitleCoverFinding]:
        """The truncation findings."""
        return self.by_code("truncated")

    @property
    def duplicates(self) -> list[TitleCoverFinding]:
        """The duplication findings."""
        return self.by_code("duplicate")

    @property
    def vague(self) -> list[TitleCoverFinding]:
        """The vagueness findings."""
        return self.by_code("vague")

    def by_code(self, code: str) -> list[TitleCoverFinding]:
        """Return the findings with *code*."""
        return [finding for finding in self if finding.code == code]

    def by_field(self, field: str) -> list[TitleCoverFinding]:
        """Return the findings for *field* (``title``, ``caption`` or ``cover``)."""
        return [finding for finding in self if finding.field == field]

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-compatible mapping."""
        return {
            "ok": self.ok,
            "title": self.title,
            "caption": self.caption,
            "cover_text": self.cover_text,
            "limits": dict(self.limits),
            "findings": [finding.to_dict() for finding in self],
        }


TitleCoverReport = TitleCoverFindings
TitleCoverLintResult = TitleCoverFindings


@dataclass(frozen=True, slots=True)
class TitleCoverPairing:
    """The three strings a lint pass reads as one publication pairing."""

    title: str = ""
    caption: str = ""
    cover_text: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "title", _clean(self.title))
        object.__setattr__(self, "caption", _clean(self.caption))
        object.__setattr__(self, "cover_text", _clean(self.cover_text))

    def lint(self, **kwargs: Any) -> TitleCoverFindings:
        """Lint this pairing with any :func:`lint_title_cover` options."""
        return lint_title_cover(
            self.title,
            self.caption,
            self.cover_text,
            **kwargs,
        )


def _platform_limits(platform: Any) -> dict[str, int | None]:
    if platform is None:
        return {}
    limits: dict[str, int | None] = {}
    caption_limits = getattr(platform, "caption_limits", None)
    if caption_limits is not None:
        title = getattr(caption_limits, "max_title_length", None)
        caption = getattr(caption_limits, "max_length", None)
        if title is not None:
            limits["max_title_length"] = _validate_limit(
                title,
                "max_title_length",
            )
        if caption is not None:
            limits["max_caption_length"] = _validate_limit(
                caption,
                "max_caption_length",
            )
    if "max_title_length" not in limits:
        title = getattr(platform, "max_title_length", None)
        if title is not None:
            limits["max_title_length"] = _validate_limit(
                title,
                "max_title_length",
            )
    if "max_caption_length" not in limits:
        caption = getattr(platform, "max_caption_length", None)
        if caption is not None:
            limits["max_caption_length"] = _validate_limit(
                caption,
                "max_caption_length",
            )
    return limits


def _mapping_limits(limits: Mapping[str, Any] | None) -> dict[str, int | None]:
    if limits is None:
        return {}
    if not isinstance(limits, Mapping):
        raise TypeError("limits must be a mapping")
    aliases = {
        "max_title": "max_title_length",
        "max_title_length": "max_title_length",
        "max_length": "max_caption_length",
        "max_caption": "max_caption_length",
        "max_caption_length": "max_caption_length",
        "max_cover": "max_cover_length",
        "max_cover_length": "max_cover_length",
    }
    resolved: dict[str, int | None] = {}
    for key, value in limits.items():
        name = aliases.get(str(key))
        if name is not None:
            resolved[name] = _validate_limit(value, name)
    return resolved


class TitleCoverLinter:
    """Lint a title, caption and cover text as one deterministic pairing.

    ``platform`` supplies the title and caption limits from its caption data;
    ``limits`` and the explicit ``max_*`` arguments override them. Cover text
    has its own limit because a thumbnail has less room than a caption. A
    caller can pass extra ``vague_terms`` to tune the lint for a language or a
    brand without changing this module.
    """

    def __init__(
        self,
        *,
        max_title_length: int | None = None,
        max_caption_length: int | None = None,
        max_cover_length: int | None = DEFAULT_COVER_MAX_LENGTH,
        similarity_threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
        vague_terms: Sequence[str] | None = None,
        platform: Any = None,
        limits: Mapping[str, Any] | None = None,
        severity: str | None = None,
    ) -> None:
        self.max_title_length = _validate_limit(
            max_title_length,
            "max_title_length",
        )
        self.max_caption_length = _validate_limit(
            max_caption_length,
            "max_caption_length",
        )
        self.max_cover_length = _validate_limit(
            max_cover_length,
            "max_cover_length",
        )
        self.similarity_threshold = _validate_threshold(similarity_threshold)
        self.vague_terms = _normalise_terms(vague_terms)
        self.platform = platform
        self.limits = _mapping_limits(limits)
        if severity is not None and severity not in {"error", "warning"}:
            raise ValueError("severity must be 'error' or 'warning'")
        self.severity = severity

    def _resolve_limits(
        self,
        *,
        max_title_length: int | None,
        max_caption_length: int | None,
        max_cover_length: int | None,
        platform: Any,
        limits: Mapping[str, Any] | None,
    ) -> dict[str, int | None]:
        resolved = _platform_limits(platform if platform is not None else self.platform)
        resolved.update(self.limits)
        resolved.update(_mapping_limits(limits))

        overrides = {
            "max_title_length": max_title_length
            if max_title_length is not None
            else self.max_title_length,
            "max_caption_length": max_caption_length
            if max_caption_length is not None
            else self.max_caption_length,
            "max_cover_length": max_cover_length
            if max_cover_length is not None
            else self.max_cover_length,
        }
        for key, value in overrides.items():
            if value is not None:
                resolved[key] = value

        if resolved.get("max_cover_length") is None:
            resolved["max_cover_length"] = DEFAULT_COVER_MAX_LENGTH
        return resolved

    def _finding(
        self,
        field_name: str,
        code: str,
        message: str,
        *,
        length: int | None = None,
        limit: int | None = None,
        related: str = "",
    ) -> TitleCoverFinding:
        return TitleCoverFinding(
            field=field_name,
            code=code,
            message=message,
            severity=self.severity or _SEVERITY.get(code, "warning"),
            length=length,
            limit=limit,
            related=related,
        )

    def _length_findings(
        self,
        field_name: str,
        text: str,
        limit: int | None,
    ) -> list[TitleCoverFinding]:
        findings: list[TitleCoverFinding] = []
        if limit is not None and len(text) > limit:
            findings.append(
                self._finding(
                    field_name,
                    "truncated",
                    f"{field_name} is {len(text)} characters, over the "
                    f"{limit} character limit",
                    length=len(text),
                    limit=limit,
                )
            )
        if _ELLIPSIS.search(text):
            findings.append(
                self._finding(
                    field_name,
                    "truncated",
                    f"{field_name} ends with an ellipsis, which can hide a cut",
                )
            )
        return findings

    def _duplicate_findings(
        self,
        fields: Mapping[str, str],
    ) -> list[TitleCoverFinding]:
        findings: list[TitleCoverFinding] = []
        pairs = (
            ("title", "caption"),
            ("title", "cover"),
            ("caption", "cover"),
        )
        for left, right in pairs:
            left_text = fields[left]
            right_text = fields[right]
            if not left_text or not right_text:
                continue
            left_key = normalize_title(left_text)
            right_key = normalize_title(right_text)
            if not left_key or not right_key:
                continue
            if (
                left_key == right_key
                or title_similarity(left_text, right_text) >= self.similarity_threshold
            ):
                findings.append(
                    self._finding(
                        right,
                        "duplicate",
                        f"{right} duplicates {left}",
                        related=left,
                    )
                )
        return findings

    def _vagueness_findings(
        self,
        field_name: str,
        text: str,
    ) -> list[TitleCoverFinding]:
        findings: list[TitleCoverFinding] = []
        if field_name == "title" and not text:
            findings.append(self._finding(field_name, "vague", "title is empty"))
            return findings
        if not text:
            return findings

        tokens = normalize_title(text).split()
        if (
            field_name == "title"
            and tokens
            and all(token in _STOPWORDS for token in tokens)
        ):
            findings.append(
                self._finding(
                    field_name,
                    "vague",
                    "title has no content words",
                )
            )
            return findings

        terms = _vague_terms(text, self.vague_terms)
        if terms:
            findings.append(
                self._finding(
                    field_name,
                    "vague",
                    f"{field_name} uses vague wording: {', '.join(terms)}",
                )
            )
        return findings

    def lint(
        self,
        title: Any,
        caption: Any = "",
        cover_text: Any = "",
        *,
        cover: Any = None,
        max_title_length: int | None = None,
        max_caption_length: int | None = None,
        max_cover_length: int | None = None,
        platform: Any = None,
        limits: Mapping[str, Any] | None = None,
    ) -> TitleCoverFindings:
        """Return every finding for the title, caption and cover text."""
        if cover is not None:
            if cover_text not in ("", None):
                raise ValueError("pass cover_text or cover, not both")
            cover_text = cover

        title_text = _clean(title)
        caption_text = _clean(caption)
        cover_value = _clean(cover_text)
        resolved = self._resolve_limits(
            max_title_length=max_title_length,
            max_caption_length=max_caption_length,
            max_cover_length=max_cover_length,
            platform=platform,
            limits=limits,
        )
        fields = {
            "title": title_text,
            "caption": caption_text,
            "cover": cover_value,
        }
        findings = TitleCoverFindings(
            title=title_text,
            caption=caption_text,
            cover_text=cover_value,
            limits=resolved,
        )

        for field_name in _FIELDS:
            text = fields[field_name]
            limit = resolved.get(_LIMIT_KEYS[field_name])
            findings.extend(self._length_findings(field_name, text, limit))
            if text:
                for word in _repeated_words(text):
                    findings.append(
                        self._finding(
                            field_name,
                            "duplicate",
                            f"{field_name} repeats the word {word!r}",
                        )
                    )
                for sentence in _repeated_sentences(text):
                    findings.append(
                        self._finding(
                            field_name,
                            "duplicate",
                            f"{field_name} repeats the sentence {sentence!r}",
                        )
                    )
            findings.extend(self._vagueness_findings(field_name, text))

        findings.extend(self._duplicate_findings(fields))
        return findings

    check = lint
    apply = lint
    __call__ = lint
    lint_pairing = lint
    lint_title_cover = lint


TitleCoverLint = TitleCoverLinter


def lint_title_cover(
    title: Any,
    caption: Any = "",
    cover_text: Any = "",
    *,
    cover: Any = None,
    max_title_length: int | None = None,
    max_caption_length: int | None = None,
    max_cover_length: int | None = None,
    similarity_threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
    vague_terms: Sequence[str] | None = None,
    platform: Any = None,
    limits: Mapping[str, Any] | None = None,
    severity: str | None = None,
) -> TitleCoverFindings:
    """Lint a title, caption and cover text as one pairing.

    ``cover`` is accepted as an alias for ``cover_text``. Limits can come from
    *platform*, a *limits* mapping, the explicit ``max_*`` arguments or the
    linter defaults, in that order of increasing precedence.
    """
    linter = TitleCoverLinter(
        max_title_length=max_title_length,
        max_caption_length=max_caption_length,
        max_cover_length=(
            max_cover_length
            if max_cover_length is not None
            else DEFAULT_COVER_MAX_LENGTH
        ),
        similarity_threshold=similarity_threshold,
        vague_terms=vague_terms,
        platform=platform,
        limits=limits,
        severity=severity,
    )
    return linter.lint(title, caption, cover_text, cover=cover)


check_title_cover = lint_title_cover
lint_title_caption_cover = lint_title_cover
lint_cover_pairing = lint_title_cover
title_cover_findings = lint_title_cover


def lint_pairing(
    pairing: Any,
    *,
    cover_text: Any = None,
    **kwargs: Any,
) -> TitleCoverFindings:
    """Lint a mapping or object that carries a title, caption and cover text."""
    if isinstance(pairing, Mapping):
        title = pairing.get("title", "")
        caption = pairing.get("caption", pairing.get("description", ""))
        if cover_text is None:
            cover_text = pairing.get("cover_text", pairing.get("cover", ""))
    else:
        title = getattr(pairing, "title", "")
        caption = getattr(pairing, "caption", "")
        if cover_text is None:
            cover_text = getattr(pairing, "cover_text", "")
    return lint_title_cover(title, caption, cover_text, **kwargs)


def lint_script(
    script: Any,
    caption: Any = None,
    cover_text: Any = "",
    **kwargs: Any,
) -> TitleCoverFindings:
    """Lint a script title against a caption (or its hook) and cover text."""
    title = getattr(script, "title", "")
    if caption is None:
        caption = getattr(script, "hook", "")
    elif hasattr(caption, "caption"):
        caption = caption.caption
    return lint_title_cover(title, caption, cover_text, **kwargs)


def lint_caption_set(
    caption_set: Any,
    cover_text: Any = "",
    **kwargs: Any,
) -> TitleCoverFindings:
    """Lint a generated caption package against its cover text."""
    title = getattr(caption_set, "title", "")
    caption = getattr(caption_set, "caption", "")
    limits = kwargs.pop("limits", None)
    if limits is None:
        limits = getattr(caption_set, "limits", None)
    return lint_title_cover(
        title,
        caption,
        cover_text,
        limits=limits,
        **kwargs,
    )

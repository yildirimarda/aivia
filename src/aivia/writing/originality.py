"""Originality and lyrics safety checks.

Two deterministic gates protect the writing stage before any generated text is
shipped:

* :func:`check_originality` measures word *n*-gram overlap against supplied
  reference text. When the score is at or above the threshold it calls the
  caller supplied rewriter and measures again, up to a bounded number of
  rewrites. A candidate that never drops below the threshold is refused rather
  than shipped as a copy.
* :func:`check_lyrics_safety` adds a profanity blocklist on top. A blocked word
  is refused, never rewritten away, because a rewrite could hide the original
  intent. Terms come from the packaged baseline, an explicit list, or the
  ``safety`` configuration section.

Both checks are pure and offline. They never call a provider, a model or the
network, so they are safe to run on every draft.
"""

from __future__ import annotations

import inspect
import math
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from aivia.core.errors import ConfigError, ContentPolicyError
from aivia.core.similarity import (
    DEFAULT_NGRAM_THRESHOLD,
    NgramUnit,
    ngram_containment,
    ngram_overlap,
)
from aivia.writing.schema import LyricsPayload, LyricsSection, Script

__all__ = [
    "DEFAULT_MAX_REGENERATIONS",
    "DEFAULT_MAX_REWRITES",
    "DEFAULT_ORIGINALITY_THRESHOLD",
    "DEFAULT_OVERLAP_LIMIT",
    "DEFAULT_OVERLAP_THRESHOLD",
    "DEFAULT_PROFANITY_TERMS",
    "ORIGINALITY_THRESHOLD",
    "OVERLAP_LIMIT",
    "OVERLAP_THRESHOLD",
    "PROFANITY_TERMS",
    "LyricsSafetyCheck",
    "LyricsSafetyChecker",
    "LyricsSafetyGate",
    "LyricsSafetyGuard",
    "LyricsSafetyResult",
    "OriginalityCheck",
    "OriginalityChecker",
    "OriginalityGate",
    "OriginalityGuard",
    "OriginalityResult",
    "SafetyResult",
    "apply_lyrics_safety",
    "block_profanity",
    "blocked_profanity",
    "check_lyrics",
    "check_lyrics_safety",
    "check_originality",
    "check_profanity",
    "contains_profanity",
    "enforce_lyrics_safety",
    "enforce_originality",
    "ensure_lyrics_safety",
    "ensure_originality",
    "find_profanity",
    "has_profanity",
    "is_clean",
    "is_original",
    "lyrics_safety",
    "lyrics_safety_check",
    "lyrics_text",
    "max_overlap",
    "overlap",
    "overlap_score",
    "profanity_check",
    "profanity_in",
    "require_clean",
    "require_lyrics_safety",
    "require_original",
    "rewrite_above_threshold",
    "rewrite_if_needed",
    "rewrite_original",
    "screen_lyrics",
    "screen_profanity",
    "validate_lyrics_safety",
    "validate_originality",
]

DEFAULT_ORIGINALITY_THRESHOLD = DEFAULT_NGRAM_THRESHOLD
"""Word-trigram overlap at or above this is treated as reused wording."""

DEFAULT_OVERLAP_THRESHOLD = DEFAULT_ORIGINALITY_THRESHOLD
"""Alias for :data:`DEFAULT_ORIGINALITY_THRESHOLD`."""

OVERLAP_THRESHOLD = DEFAULT_ORIGINALITY_THRESHOLD
"""Alias for :data:`DEFAULT_ORIGINALITY_THRESHOLD`."""

ORIGINALITY_THRESHOLD = DEFAULT_ORIGINALITY_THRESHOLD
"""Alias for :data:`DEFAULT_ORIGINALITY_THRESHOLD`."""

DEFAULT_OVERLAP_LIMIT = DEFAULT_ORIGINALITY_THRESHOLD
"""Alias for :data:`DEFAULT_ORIGINALITY_THRESHOLD`."""

OVERLAP_LIMIT = DEFAULT_ORIGINALITY_THRESHOLD
"""Alias for :data:`DEFAULT_ORIGINALITY_THRESHOLD`."""

DEFAULT_MAX_REWRITES = 2
"""How many times a candidate above the threshold is rewritten."""

DEFAULT_MAX_REGENERATIONS = DEFAULT_MAX_REWRITES
"""Alias for :data:`DEFAULT_MAX_REWRITES`."""

DEFAULT_PROFANITY_TERMS: tuple[str, ...] = (
    "arse",
    "arsehole",
    "ass",
    "asshole",
    "bastard",
    "bitch",
    "bitches",
    "bollocks",
    "bullshit",
    "cock",
    "crap",
    "cunt",
    "damn",
    "damned",
    "dick",
    "dickhead",
    "fuck",
    "fucked",
    "fucker",
    "fucking",
    "motherfucker",
    "piss",
    "pissed",
    "prick",
    "pussy",
    "shit",
    "shitty",
    "slut",
    "twat",
    "wanker",
    "whore",
)
"""The packaged profanity baseline. Callers may replace or extend it."""

PROFANITY_TERMS = DEFAULT_PROFANITY_TERMS
"""Alias for :data:`DEFAULT_PROFANITY_TERMS`."""

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_CONFIG_MISSING = object()


def _text(value: Any, label: str = "text") -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ConfigError(f"{label} must be valid UTF-8") from exc
    raise TypeError(f"{label} must be a string")


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _lines(value: Any) -> list[str]:
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [
            item.strip() for item in value if isinstance(item, str) and item.strip()
        ]
    return []


def _mapping_text(value: Mapping[str, Any]) -> str | None:
    for key in ("sections", "lyrics", "lines", "verses", "text", "content", "body"):
        if key not in value:
            continue
        nested = value[key]
        if isinstance(nested, str):
            return nested
        if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
            parts: list[str] = []
            for item in nested:
                if isinstance(item, Mapping):
                    parts.extend(_lines(item.get("lines", item.get("text", ""))))
                elif isinstance(item, str):
                    parts.extend(_lines(item))
            if parts:
                return "\n".join(parts)
    return None


def lyrics_text(value: Any) -> str:
    """Return the lyric text inside *value*.

    A string is returned unchanged. A :class:`~aivia.writing.schema.Script`,
    :class:`~aivia.writing.schema.LyricsPayload`, section, mapping or sequence
    is flattened into newline separated lines.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return _text(value, "lyrics")
    if isinstance(value, Script):
        beats = [beat.text for beat in value.beats if beat.text]
        if beats:
            return "\n".join(beats)
        dialogue = [line.text for line in value.all_dialogue if line.text]
        if dialogue:
            return "\n".join(dialogue)
        payload_text = lyrics_text(value.payload)
        if payload_text:
            return payload_text
        parts = [item for item in (value.title, value.hook, value.outro) if item]
        return "\n".join(parts)
    if isinstance(value, LyricsPayload):
        return "\n".join(
            line for section in value.sections for line in section.lines if line
        )
    if isinstance(value, LyricsSection):
        return "\n".join(line for line in value.lines if line)
    if isinstance(value, Mapping):
        found = _mapping_text(value)
        if found is not None:
            return found
        parts = [
            lyrics_text(item)
            for item in value.values()
            if isinstance(item, (str, Mapping, Sequence, Script, LyricsPayload))
        ]
        return "\n".join(part for part in parts if part)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts = [lyrics_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    raise TypeError(f"cannot read lyrics from {type(value).__name__}")


def _references(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, bytes):
        text = _text(value, "reference")
        return [text] if text.strip() else []
    if isinstance(value, (Script, LyricsPayload, LyricsSection, Mapping)):
        text = lyrics_text(value)
        return [text] if text.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        references: list[str] = []
        for item in value:
            references.extend(_references(item))
        return references
    raise TypeError(f"cannot read a reference from {type(value).__name__}")


def _threshold(value: Any, label: str = "threshold") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1")
    return number


def _rewrite_limit(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_rewrites must be an integer")
    if value < 0:
        raise ValueError("max_rewrites must not be negative")
    return value


def _metric(value: Any) -> str:
    text = str(value or "jaccard").strip().lower()
    if text not in {"jaccard", "containment"}:
        raise ValueError("metric must be 'jaccard' or 'containment'")
    return text


def overlap_score(
    candidate: Any,
    references: Any = None,
    *,
    n: int = 3,
    unit: NgramUnit | str = NgramUnit.WORD,
    metric: str = "jaccard",
) -> float:
    """Return the highest overlap between *candidate* and any reference.

    The default is word-trigram Jaccard overlap, matching
    :data:`DEFAULT_ORIGINALITY_THRESHOLD`. ``metric="containment"`` answers how
    much of the candidate appears in a longer reference. No references means no
    measurable overlap and therefore ``0.0``.
    """
    text = lyrics_text(candidate)
    reference_texts = _references(references)
    if not reference_texts:
        return 0.0
    kind = _metric(metric)
    compare = ngram_overlap if kind == "jaccard" else ngram_containment
    return max(compare(text, reference, n, unit=unit) for reference in reference_texts)


def max_overlap(
    candidate: Any,
    references: Any = None,
    *,
    n: int = 3,
    unit: NgramUnit | str = NgramUnit.WORD,
    metric: str = "jaccard",
) -> float:
    """Alias for :func:`overlap_score`."""
    return overlap_score(candidate, references, n=n, unit=unit, metric=metric)


overlap = overlap_score


class OriginalityResult(str):
    """The checked text with the measured overlap attached.

    The value is a string so a caller can use the result directly as the final
    text, while ``overlap``, ``passed`` and ``rewritten`` describe the check.
    """

    __slots__ = (
        "_attempts",
        "_overlap",
        "_passed",
        "_reason",
        "_references",
        "_rewritten",
        "_text",
        "_threshold",
    )

    _text: str
    _overlap: float
    _threshold: float
    _rewritten: bool
    _attempts: int
    _references: tuple[str, ...]
    _passed: bool
    _reason: str

    def __new__(
        cls,
        text: str,
        *,
        overlap: float,
        threshold: float,
        rewritten: bool = False,
        attempts: int = 0,
        references: Sequence[str] = (),
        passed: bool = True,
        reason: str = "",
    ) -> OriginalityResult:
        value = _text(text)
        result = super().__new__(cls, value)
        result._text = value
        result._overlap = float(overlap)
        result._threshold = float(threshold)
        result._rewritten = bool(rewritten)
        result._attempts = int(attempts)
        result._references = tuple(references)
        result._passed = bool(passed)
        result._reason = reason
        return result

    @property
    def text(self) -> str:
        """The checked (possibly rewritten) text."""
        return self._text

    @property
    def value(self) -> str:
        """Alias for :attr:`text`."""
        return self._text

    @property
    def score(self) -> float:
        """Alias for :attr:`overlap`."""
        return self._overlap

    @property
    def overlap(self) -> float:
        """The highest measured overlap against the references."""
        return self._overlap

    @property
    def threshold(self) -> float:
        """The threshold the overlap was compared with."""
        return self._threshold

    @property
    def rewritten(self) -> bool:
        """Whether at least one rewrite was applied."""
        return self._rewritten

    @property
    def attempts(self) -> int:
        """How many rewrites were applied."""
        return self._attempts

    @property
    def references(self) -> tuple[str, ...]:
        """The reference texts the candidate was compared with."""
        return self._references

    @property
    def passed(self) -> bool:
        """Whether the final text is below the threshold."""
        return self._passed

    @property
    def is_original(self) -> bool:
        """Alias for :attr:`passed`."""
        return self._passed

    @property
    def reason(self) -> str:
        """A short machine-readable failure reason, empty when it passed."""
        return self._reason

    def to_dict(self) -> dict[str, Any]:
        """Return the check as a JSON-compatible mapping."""
        return {
            "text": self._text,
            "overlap": self._overlap,
            "threshold": self._threshold,
            "rewritten": self._rewritten,
            "attempts": self._attempts,
            "references": list(self._references),
            "passed": self._passed,
            "reason": self._reason,
        }


OriginalityCheck = OriginalityResult


class LyricsSafetyResult(OriginalityResult):
    """An originality result plus the profanity terms that were blocked."""

    __slots__ = ("_blocked_terms",)

    _blocked_terms: tuple[str, ...]

    def __new__(
        cls,
        text: str,
        *,
        overlap: float,
        threshold: float,
        rewritten: bool = False,
        attempts: int = 0,
        references: Sequence[str] = (),
        passed: bool = True,
        reason: str = "",
        blocked_terms: Sequence[str] = (),
    ) -> LyricsSafetyResult:
        result = cast(
            LyricsSafetyResult,
            super().__new__(
                cls,
                text,
                overlap=overlap,
                threshold=threshold,
                rewritten=rewritten,
                attempts=attempts,
                references=references,
                passed=passed,
                reason=reason,
            ),
        )
        result._blocked_terms = tuple(blocked_terms)
        return result

    @property
    def blocked_terms(self) -> tuple[str, ...]:
        """The profanity terms found in the final text."""
        return self._blocked_terms

    @property
    def profanity(self) -> tuple[str, ...]:
        """Alias for :attr:`blocked_terms`."""
        return self._blocked_terms

    def to_dict(self) -> dict[str, Any]:
        """Return the check as a JSON-compatible mapping."""
        payload = super().to_dict()
        payload["blocked_terms"] = list(self._blocked_terms)
        return payload


LyricsSafetyCheck = LyricsSafetyResult
SafetyResult = LyricsSafetyResult


def _call_rewriter(
    rewrite: Callable[..., Any],
    text: str,
    overlap: float,
    attempt: int,
) -> str:
    try:
        signature = inspect.signature(rewrite)
    except (TypeError, ValueError):
        replacement = rewrite(text)
    else:
        parameters = list(signature.parameters.values())
        positional = [
            parameter
            for parameter in parameters
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        keyword = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
        )
        if keyword:
            replacement = rewrite(text, overlap=overlap, attempt=attempt)
        elif len(positional) >= 3:
            replacement = rewrite(text, overlap, attempt)
        elif len(positional) == 2:
            replacement = rewrite(text, overlap)
        elif len(positional) == 1:
            replacement = rewrite(text)
        else:
            replacement = rewrite()
    if not isinstance(replacement, str):
        raise ConfigError(
            "the originality rewriter must return text",
            details={"returned": type(replacement).__name__},
        )
    cleaned = replacement.strip()
    if not cleaned:
        raise ConfigError(
            "the originality rewriter returned empty text",
            details={"attempt": attempt},
        )
    return cleaned


def check_originality(
    candidate: Any,
    references: Any = None,
    *,
    threshold: float | None = None,
    rewrite: Callable[..., Any] | None = None,
    max_rewrites: int | None = None,
    max_regenerations: int | None = None,
    n: int = 3,
    unit: NgramUnit | str = NgramUnit.WORD,
    metric: str = "jaccard",
    strict: bool = False,
) -> OriginalityResult:
    """Check *candidate* for reused wording and rewrite it above the limit.

    The candidate is compared with every reference and the highest score is
    used. While the score is at or above *threshold* and rewrites remain, the
    rewriter is called with the current text. When no rewrite is supplied, or
    the limit is exhausted, the result reports ``passed=False``. With
    ``strict=True`` a refused candidate raises :class:`ContentPolicyError`
    instead.
    """
    text = lyrics_text(candidate)
    reference_texts = _references(references)
    limit = _rewrite_limit(
        max_rewrites
        if max_rewrites is not None
        else max_regenerations
        if max_regenerations is not None
        else DEFAULT_MAX_REWRITES
    )
    limit_value = _threshold(
        threshold if threshold is not None else DEFAULT_ORIGINALITY_THRESHOLD
    )
    score = overlap_score(
        text,
        reference_texts,
        n=n,
        unit=unit,
        metric=metric,
    )
    current = text
    applied = 0
    rewritten = False
    while (
        reference_texts
        and score >= limit_value
        and applied < limit
        and rewrite is not None
    ):
        current = _call_rewriter(rewrite, current, score, applied + 1)
        applied += 1
        rewritten = True
        score = overlap_score(
            current,
            reference_texts,
            n=n,
            unit=unit,
            metric=metric,
        )
    passed = not reference_texts or score < limit_value
    reason = "" if passed else "overlap"
    result = OriginalityResult(
        current,
        overlap=score,
        threshold=limit_value,
        rewritten=rewritten,
        attempts=applied,
        references=reference_texts,
        passed=passed,
        reason=reason,
    )
    if strict and not passed:
        raise ContentPolicyError(
            "lyrics overlap the reference above the allowed threshold",
            details={
                "source": "originality",
                "reason": "overlap",
                "overlap": score,
                "threshold": limit_value,
                "attempts": applied,
            },
        )
    return result


def enforce_originality(
    candidate: Any,
    references: Any = None,
    *,
    threshold: float | None = None,
    rewrite: Callable[..., Any] | None = None,
    max_rewrites: int | None = None,
    max_regenerations: int | None = None,
    n: int = 3,
    unit: NgramUnit | str = NgramUnit.WORD,
    metric: str = "jaccard",
) -> OriginalityResult:
    """Like :func:`check_originality` but refuse a candidate that never passes."""
    return check_originality(
        candidate,
        references,
        threshold=threshold,
        rewrite=rewrite,
        max_rewrites=max_rewrites,
        max_regenerations=max_regenerations,
        n=n,
        unit=unit,
        metric=metric,
        strict=True,
    )


rewrite_above_threshold = enforce_originality
rewrite_if_needed = enforce_originality
rewrite_original = enforce_originality
require_original = enforce_originality
ensure_originality = enforce_originality
validate_originality = enforce_originality


def is_original(
    candidate: Any,
    references: Any = None,
    *,
    threshold: float | None = None,
    n: int = 3,
    unit: NgramUnit | str = NgramUnit.WORD,
    metric: str = "jaccard",
) -> bool:
    """Return whether *candidate* is below the overlap threshold."""
    limit = _threshold(
        threshold if threshold is not None else DEFAULT_ORIGINALITY_THRESHOLD
    )
    return overlap_score(candidate, references, n=n, unit=unit, metric=metric) < limit


def _normalise_term(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _is_word_character(character: str) -> bool:
    return character.isalnum() or character == "_"


def _clean_terms(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        candidates: Sequence[Any] = (values,)
    elif isinstance(values, Sequence):
        candidates = values
    else:
        candidates = (values,)
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        if not isinstance(value, str):
            continue
        term = value.strip()
        if not term:
            continue
        key = _normalise_term(term)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(term)
    return tuple(cleaned)


def _term_pattern(term: str) -> re.Pattern[str]:
    normalised = _normalise_term(term).strip()
    parts = re.split(r"\s+", normalised)
    body = r"\s+".join(re.escape(part) for part in parts)
    prefix = r"(?<!\w)" if normalised and _is_word_character(normalised[0]) else ""
    suffix = r"(?!\w)" if normalised and _is_word_character(normalised[-1]) else ""
    return re.compile(prefix + body + suffix)


def _find_terms(text: str, terms: Sequence[str]) -> tuple[str, ...]:
    normalised = _normalise_term(text)
    matches: list[str] = []
    for term in _clean_terms(terms):
        if _term_pattern(term).search(normalised):
            matches.append(term)
    return tuple(matches)


def _config_value(config: Any, name: str, default: Any = _CONFIG_MISSING) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _effective_terms(
    terms: Any,
    config: Any,
    use_defaults: bool,
) -> tuple[str, ...]:
    if config is not None:
        if not _config_value(config, "enabled", True):
            return ()
        if not _config_value(config, "screen_prompts", True):
            return ()
    values: list[Any] = list(DEFAULT_PROFANITY_TERMS) if use_defaults else []
    if terms is not None:
        values.extend(_clean_terms(terms))
    if config is not None:
        blocked = _config_value(config, "blocked_terms", ())
        categories = _config_value(config, "block_categories", ())
        if isinstance(blocked, (str, Sequence)):
            values.extend(_clean_terms(blocked))
        if isinstance(categories, (str, Sequence)):
            values.extend(_clean_terms(categories))
    return _clean_terms(values)


def find_profanity(
    value: Any,
    terms: Any = None,
    *,
    config: Any = None,
    use_defaults: bool = True,
) -> tuple[str, ...]:
    """Return the profanity terms found in *value*, in term order.

    An explicit *terms* list replaces the packaged baseline; configured terms
    are always added. Matching is literal, case-insensitive and whole-word, so
    ``class`` never triggers ``ass``.
    """
    text = lyrics_text(value)
    effective = _effective_terms(terms, config, use_defaults)
    if not effective:
        return ()
    return _find_terms(text, effective)


profanity_in = find_profanity
profanity_check = find_profanity
blocked_profanity = find_profanity


def has_profanity(
    value: Any,
    terms: Any = None,
    *,
    config: Any = None,
    use_defaults: bool = True,
) -> bool:
    """Return whether *value* contains a blocked profanity term."""
    return bool(find_profanity(value, terms, config=config, use_defaults=use_defaults))


def is_clean(
    value: Any,
    terms: Any = None,
    *,
    config: Any = None,
    use_defaults: bool = True,
) -> bool:
    """Alias for the negation of :func:`has_profanity`."""
    return not has_profanity(
        value,
        terms,
        config=config,
        use_defaults=use_defaults,
    )


contains_profanity = has_profanity


def check_profanity(
    value: Any,
    terms: Any = None,
    *,
    config: Any = None,
    use_defaults: bool = True,
    strict: bool = False,
) -> LyricsSafetyResult:
    """Check *value* for profanity without raising unless *strict* is set."""
    text = lyrics_text(value)
    blocked = find_profanity(
        text,
        terms,
        config=config,
        use_defaults=use_defaults,
    )
    passed = not blocked
    result = LyricsSafetyResult(
        text,
        overlap=0.0,
        threshold=DEFAULT_ORIGINALITY_THRESHOLD,
        passed=passed,
        reason="" if passed else "profanity",
        blocked_terms=blocked,
    )
    if strict and blocked:
        raise ContentPolicyError(
            "lyrics contain blocked profanity",
            details={
                "source": "lyrics_safety",
                "reason": "profanity",
                "blocked_terms": list(blocked),
            },
        )
    return result


def block_profanity(
    value: Any,
    terms: Any = None,
    *,
    config: Any = None,
    use_defaults: bool = True,
) -> LyricsSafetyResult:
    """Return *value* or raise when it contains blocked profanity."""
    return check_profanity(
        value,
        terms,
        config=config,
        use_defaults=use_defaults,
        strict=True,
    )


screen_profanity = block_profanity
require_clean = block_profanity


def check_lyrics_safety(
    value: Any,
    references: Any = None,
    *,
    threshold: float | None = None,
    rewrite: Callable[..., Any] | None = None,
    max_rewrites: int | None = None,
    max_regenerations: int | None = None,
    terms: Any = None,
    profanity_terms: Any = None,
    config: Any = None,
    use_defaults: bool = True,
    n: int = 3,
    unit: NgramUnit | str = NgramUnit.WORD,
    metric: str = "jaccard",
    strict: bool = False,
) -> LyricsSafetyResult:
    """Run the profanity block and the originality gate over *value*.

    Profanity is never rewritten away. Overlap is rewritten through *rewrite*
    up to the limit. With ``strict=True`` either failure raises
    :class:`ContentPolicyError`; otherwise the result reports ``passed=False``
    and the reason.
    """
    text = lyrics_text(value)
    selected_terms = profanity_terms if profanity_terms is not None else terms
    blocked = find_profanity(
        text,
        selected_terms,
        config=config,
        use_defaults=use_defaults,
    )
    limit_value = _threshold(
        threshold if threshold is not None else DEFAULT_ORIGINALITY_THRESHOLD
    )
    score = overlap_score(text, references, n=n, unit=unit, metric=metric)
    if blocked:
        result = LyricsSafetyResult(
            text,
            overlap=score,
            threshold=limit_value,
            passed=False,
            reason="profanity",
            blocked_terms=blocked,
        )
        if strict:
            raise ContentPolicyError(
                "lyrics contain blocked profanity",
                details={
                    "source": "lyrics_safety",
                    "reason": "profanity",
                    "blocked_terms": list(blocked),
                },
            )
        return result
    original = check_originality(
        text,
        references,
        threshold=limit_value,
        rewrite=rewrite,
        max_rewrites=max_rewrites,
        max_regenerations=max_regenerations,
        n=n,
        unit=unit,
        metric=metric,
    )
    blocked_after = find_profanity(
        original.text,
        selected_terms,
        config=config,
        use_defaults=use_defaults,
    )
    passed = original.passed and not blocked_after
    reason = "profanity" if blocked_after else original.reason
    result = LyricsSafetyResult(
        original.text,
        overlap=original.overlap,
        threshold=original.threshold,
        rewritten=original.rewritten,
        attempts=original.attempts,
        references=original.references,
        passed=passed,
        reason=reason,
        blocked_terms=blocked_after,
    )
    if strict and not passed:
        if blocked_after:
            raise ContentPolicyError(
                "lyrics contain blocked profanity",
                details={
                    "source": "lyrics_safety",
                    "reason": "profanity",
                    "blocked_terms": list(blocked_after),
                },
            )
        raise ContentPolicyError(
            "lyrics overlap the reference above the allowed threshold",
            details={
                "source": "originality",
                "reason": "overlap",
                "overlap": original.overlap,
                "threshold": original.threshold,
                "attempts": original.attempts,
            },
        )
    return result


def enforce_lyrics_safety(
    value: Any,
    references: Any = None,
    **kwargs: Any,
) -> LyricsSafetyResult:
    """Like :func:`check_lyrics_safety` but always refuse on failure."""
    kwargs["strict"] = True
    return check_lyrics_safety(value, references, **kwargs)


lyrics_safety = check_lyrics_safety
lyrics_safety_check = check_lyrics_safety
check_lyrics = check_lyrics_safety
require_lyrics_safety = enforce_lyrics_safety
ensure_lyrics_safety = enforce_lyrics_safety
validate_lyrics_safety = enforce_lyrics_safety
screen_lyrics = enforce_lyrics_safety


def apply_lyrics_safety(
    value: Any,
    references: Any = None,
    *,
    threshold: float | None = None,
    rewrite: Callable[..., Any] | None = None,
    max_rewrites: int | None = None,
    max_regenerations: int | None = None,
    terms: Any = None,
    profanity_terms: Any = None,
    config: Any = None,
    use_defaults: bool = True,
    n: int = 3,
    unit: NgramUnit | str = NgramUnit.WORD,
    metric: str = "containment",
    strict: bool = True,
) -> LyricsPayload:
    """Return *value* as a payload with every lyric line made safe.

    Each line is checked separately so a rewrite replaces only the reused line
    and the section structure survives. The whole lyric is checked once more
    for profanity that a rewriter could have introduced.
    """
    if isinstance(value, LyricsPayload):
        payload = value
    else:
        from aivia.writing.lyrics import extract_lyrics

        payload = extract_lyrics(value)
    sections: list[LyricsSection] = []
    for section in payload.sections:
        lines: list[str] = []
        for line in section.lines:
            checked = check_lyrics_safety(
                line,
                references,
                threshold=threshold,
                rewrite=rewrite,
                max_rewrites=max_rewrites,
                max_regenerations=max_regenerations,
                terms=terms,
                profanity_terms=profanity_terms,
                config=config,
                use_defaults=use_defaults,
                n=n,
                unit=unit,
                metric=metric,
                strict=strict,
            )
            lines.extend(checked.text.splitlines() or [checked.text])
        sections.append(section.model_copy(update={"lines": lines}, deep=True))
    check_lyrics_safety(
        LyricsPayload(
            sections=sections,
            section_counts=dict(payload.section_counts),
            rhyme_scheme=payload.rhyme_scheme,
            syllable_budget=payload.syllable_budget,
            keywords=list(payload.keywords),
            genre=payload.genre,
            tempo=payload.tempo,
        ),
        terms=terms,
        profanity_terms=profanity_terms,
        config=config,
        use_defaults=use_defaults,
        strict=strict,
    )
    return LyricsPayload(
        sections=sections,
        section_counts=dict(payload.section_counts),
        rhyme_scheme=payload.rhyme_scheme,
        syllable_budget=payload.syllable_budget,
        keywords=list(payload.keywords),
        genre=payload.genre,
        tempo=payload.tempo,
    )


class OriginalityChecker:
    """A reusable originality check bound to references and a rewriter."""

    def __init__(
        self,
        references: Any = None,
        *,
        threshold: float | None = None,
        rewrite: Callable[..., Any] | None = None,
        max_rewrites: int | None = None,
        n: int = 3,
        unit: NgramUnit | str = NgramUnit.WORD,
        metric: str = "jaccard",
    ) -> None:
        self.references = references
        self.threshold = threshold
        self.rewrite = rewrite
        self.max_rewrites = max_rewrites
        self.n = n
        self.unit = unit
        self.metric = metric

    def check(self, value: Any, *, strict: bool = False) -> OriginalityResult:
        """Check *value* with the bound options."""
        return check_originality(
            value,
            self.references,
            threshold=self.threshold,
            rewrite=self.rewrite,
            max_rewrites=self.max_rewrites,
            n=self.n,
            unit=self.unit,
            metric=self.metric,
            strict=strict,
        )

    def enforce(self, value: Any) -> OriginalityResult:
        """Check *value* and refuse a candidate that never passes."""
        return self.check(value, strict=True)

    __call__ = check


class LyricsSafetyChecker(OriginalityChecker):
    """A reusable lyrics safety check bound to references and profanity terms."""

    def __init__(
        self,
        references: Any = None,
        *,
        threshold: float | None = None,
        rewrite: Callable[..., Any] | None = None,
        max_rewrites: int | None = None,
        terms: Any = None,
        profanity_terms: Any = None,
        config: Any = None,
        use_defaults: bool = True,
        n: int = 3,
        unit: NgramUnit | str = NgramUnit.WORD,
        metric: str = "jaccard",
    ) -> None:
        super().__init__(
            references,
            threshold=threshold,
            rewrite=rewrite,
            max_rewrites=max_rewrites,
            n=n,
            unit=unit,
            metric=metric,
        )
        self.terms = terms
        self.profanity_terms = profanity_terms
        self.config = config
        self.use_defaults = use_defaults

    def check(self, value: Any, *, strict: bool = False) -> LyricsSafetyResult:
        """Check *value* with the bound options."""
        return check_lyrics_safety(
            value,
            self.references,
            threshold=self.threshold,
            rewrite=self.rewrite,
            max_rewrites=self.max_rewrites,
            terms=self.terms,
            profanity_terms=self.profanity_terms,
            config=self.config,
            use_defaults=self.use_defaults,
            n=self.n,
            unit=self.unit,
            metric=self.metric,
            strict=strict,
        )

    def enforce(self, value: Any) -> LyricsSafetyResult:
        """Check *value* and refuse any failure."""
        return self.check(value, strict=True)

    __call__ = check


OriginalityGate = OriginalityChecker
OriginalityGuard = OriginalityChecker
LyricsSafetyGate = LyricsSafetyChecker
LyricsSafetyGuard = LyricsSafetyChecker

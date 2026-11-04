"""Reading time estimates and target length ranges for prose and verse.

A writing mode accepts a length as a preset name, a target number of seconds or
an explicit range. The helpers here turn any of those shapes into a
:class:`LengthRange`, count the words in a script and estimate how long it takes
to read. The estimate is deterministic and offline so the same text always
yields the same duration.
"""

from __future__ import annotations

import inspect
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from aivia.core.errors import ConfigError

__all__ = [
    "DEFAULT_MAX_LENGTH_REWRITES",
    "DEFAULT_MAX_REWRITES",
    "DEFAULT_WORDS_PER_SECOND",
    "LENGTH_PRESETS",
    "DurationCheck",
    "DurationEstimate",
    "DurationResult",
    "LengthCheck",
    "LengthEstimate",
    "LengthRange",
    "LengthResult",
    "check_duration",
    "check_length",
    "check_reading_time",
    "coerce_length_range",
    "count_words",
    "enforce_duration",
    "enforce_length",
    "enforce_reading_time",
    "ensure_length",
    "estimate_duration",
    "estimate_duration_range",
    "estimate_length",
    "estimate_reading_time",
    "fit_length",
    "fit_reading_time",
    "fit_to_length",
    "measure_duration",
    "measure_length",
    "reading_time_range",
    "reading_time_seconds",
    "require_length",
    "resolve_length_range",
    "resolve_words_per_second",
    "resolve_words_per_second_range",
    "rewrite_duration",
    "rewrite_length",
    "rewrite_reading_time",
    "rewrite_to_duration",
    "rewrite_to_length",
    "validate_length",
]

DEFAULT_WORDS_PER_SECOND = 2.5
"""Fallback reading speed in words per second when no rate is supplied."""

DEFAULT_MAX_LENGTH_REWRITES = 2
"""How many times a script outside its target range is rewritten."""

DEFAULT_MAX_REWRITES = DEFAULT_MAX_LENGTH_REWRITES
"""Alias for :data:`DEFAULT_MAX_LENGTH_REWRITES`."""

LENGTH_PRESETS: Mapping[str, tuple[float, float]] = MappingProxyType(
    {
        "micro": (0.0, 10.0),
        "short": (0.0, 30.0),
        "medium": (15.0, 60.0),
        "long": (45.0, 180.0),
        "epic": (90.0, 600.0),
    }
)
"""Named target ranges in seconds, from shortest to longest."""

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_ZERO = 0.0
_MIN_KEYS = ("min_seconds", "min", "minimum")
_MAX_KEYS = ("max_seconds", "max", "maximum")
_TARGET_KEYS = ("target_seconds", "target", "seconds", "duration", "length")
_TOLERANCE_KEYS = ("tolerance", "tolerance_seconds")
_RATE_KEYS = ("words_per_second", "rate", "wps")
_MIN_RATE_KEYS = ("min_words_per_second", "min_rate", "minimum", "min")
_MAX_RATE_KEYS = ("max_words_per_second", "max_rate", "maximum", "max")


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(
            f"{field} must be a number",
            details={"field": field, "value": repr(value)},
        )
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            raise ConfigError(
                f"{field} must be a number",
                details={"field": field, "value": value},
            ) from None
    else:
        raise ConfigError(
            f"{field} must be a number",
            details={"field": field, "value": repr(value)},
        )
    if not math.isfinite(number) or number < 0:
        raise ConfigError(
            f"{field} must be a non-negative number",
            details={"field": field, "value": number},
        )
    return number


def _first(value: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in value:
            return value[key]
    return None


@dataclass(frozen=True, slots=True)
class LengthRange:
    """An inclusive range of reading times in seconds."""

    min_seconds: float
    max_seconds: float

    def __post_init__(self) -> None:
        if self.min_seconds < _ZERO:
            raise ValueError("minimum reading time must not be negative")
        if self.max_seconds < self.min_seconds:
            raise ValueError("maximum reading time must not be below the minimum")

    @property
    def minimum(self) -> float:
        """Alias for :attr:`min_seconds`."""
        return self.min_seconds

    @property
    def maximum(self) -> float:
        """Alias for :attr:`max_seconds`."""
        return self.max_seconds

    def contains(self, seconds: float) -> bool:
        """Return whether *seconds* falls inside the inclusive range."""
        return self.min_seconds <= seconds <= self.max_seconds

    def to_dict(self) -> dict[str, float]:
        """Return the range as a JSON-compatible mapping."""
        return {"min_seconds": self.min_seconds, "max_seconds": self.max_seconds}


class LengthResult(str):
    """The checked text with its measured duration and target attached.

    The value is a string so a caller can use the result directly as the final
    text, while ``seconds``, ``passed`` and ``rewritten`` describe the check.
    """

    __slots__ = (
        "_attempts",
        "_passed",
        "_reason",
        "_rewritten",
        "_seconds",
        "_target",
        "_text",
        "_words",
        "_words_per_second",
    )

    _text: str
    _seconds: float
    _target: LengthRange | None
    _words: int
    _words_per_second: float
    _rewritten: bool
    _attempts: int
    _passed: bool
    _reason: str

    def __new__(
        cls,
        text: str,
        *,
        seconds: float,
        target: LengthRange | None = None,
        words: int = 0,
        words_per_second: float = DEFAULT_WORDS_PER_SECOND,
        rewritten: bool = False,
        attempts: int = 0,
        passed: bool = True,
        reason: str = "",
    ) -> LengthResult:
        value = text.strip() if isinstance(text, str) else str(text)
        result = super().__new__(cls, value)
        result._text = value
        result._seconds = float(seconds)
        result._target = target
        result._words = int(words)
        result._words_per_second = float(words_per_second)
        result._rewritten = bool(rewritten)
        result._attempts = int(attempts)
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
    def seconds(self) -> float:
        """The measured reading time of the final text."""
        return self._seconds

    @property
    def measured(self) -> float:
        """Alias for :attr:`seconds`."""
        return self._seconds

    @property
    def measured_seconds(self) -> float:
        """Alias for :attr:`seconds`."""
        return self._seconds

    @property
    def duration(self) -> float:
        """Alias for :attr:`seconds`."""
        return self._seconds

    @property
    def duration_seconds(self) -> float:
        """Alias for :attr:`seconds`."""
        return self._seconds

    @property
    def reading_time_seconds(self) -> float:
        """Alias for :attr:`seconds`."""
        return self._seconds

    @property
    def target(self) -> LengthRange | None:
        """The target range, or ``None`` when no length was requested."""
        return self._target

    @property
    def target_range(self) -> LengthRange | None:
        """Alias for :attr:`target`."""
        return self._target

    @property
    def min_seconds(self) -> float | None:
        """The lower bound of the target range, if any."""
        return None if self._target is None else self._target.min_seconds

    @property
    def max_seconds(self) -> float | None:
        """The upper bound of the target range, if any."""
        return None if self._target is None else self._target.max_seconds

    @property
    def words(self) -> int:
        """The number of words in the final text."""
        return self._words

    @property
    def word_count(self) -> int:
        """Alias for :attr:`words`."""
        return self._words

    @property
    def words_per_second(self) -> float:
        """The reading speed used for the estimate."""
        return self._words_per_second

    @property
    def rate(self) -> float:
        """Alias for :attr:`words_per_second`."""
        return self._words_per_second

    @property
    def rewritten(self) -> bool:
        """Whether at least one rewrite was applied."""
        return self._rewritten

    @property
    def attempts(self) -> int:
        """How many rewrites were applied."""
        return self._attempts

    @property
    def passed(self) -> bool:
        """Whether the final text falls inside the target range."""
        return self._passed

    @property
    def reason(self) -> str:
        """A short machine-readable failure reason, empty when it passed."""
        return self._reason

    def to_dict(self) -> dict[str, Any]:
        """Return the check as a JSON-compatible mapping."""
        return {
            "text": self._text,
            "words": self._words,
            "word_count": self._words,
            "words_per_second": self._words_per_second,
            "seconds": self._seconds,
            "measured_seconds": self._seconds,
            "duration_seconds": self._seconds,
            "target": None if self._target is None else self._target.to_dict(),
            "min_seconds": self.min_seconds,
            "max_seconds": self.max_seconds,
            "rewritten": self._rewritten,
            "attempts": self._attempts,
            "passed": self._passed,
            "reason": self._reason,
        }


LengthCheck = LengthResult
LengthEstimate = LengthResult
DurationCheck = LengthResult
DurationEstimate = LengthResult
DurationResult = LengthResult


def _from_target(center: float, tolerance: float) -> LengthRange:
    span = center * tolerance
    return LengthRange(max(0.0, center - span), center + span)


def _rate_number(value: Any, field: str) -> float:
    number = _number(value, field)
    if number <= 0:
        raise ConfigError(
            f"{field} must be a positive number",
            details={"field": field, "value": value},
        )
    return number


def _rate_bounds(value: Any, field: str = "words_per_second") -> tuple[float, float]:
    """Return the inclusive speaking-rate bounds in *value*.

    A scalar is both its own minimum and maximum. A mapping, pair, config model
    or any object with ``min`` and ``max`` attributes is read as a bound pair.
    """
    if isinstance(value, bool):
        raise ConfigError(
            f"{field} must be a number",
            details={"field": field, "value": repr(value)},
        )
    if isinstance(value, (int, float)):
        rate = _rate_number(value, field)
        return rate, rate
    if isinstance(value, Mapping):
        nested = _first(value, _RATE_KEYS)
        if nested is not None and nested is not value:
            return _rate_bounds(nested, field)
        minimum = _first(value, _MIN_RATE_KEYS)
        maximum = _first(value, _MAX_RATE_KEYS)
        if minimum is None and maximum is None:
            raise ConfigError(
                f"{field} must define min and max",
                details={"field": field, "value": dict(value)},
            )
        low = _rate_number(minimum, f"{field}.min") if minimum is not None else None
        high = _rate_number(maximum, f"{field}.max") if maximum is not None else None
        if low is None:
            assert high is not None
            low = high
        if high is None:
            high = low
        if low > high:
            raise ConfigError(
                f"{field}.min must not exceed {field}.max",
                details={"field": field, "min": low, "max": high},
            )
        return low, high
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        if len(items) != 2:
            raise ConfigError(
                f"{field} must have exactly two values",
                details={"field": field, "value": value},
            )
        low = _rate_number(items[0], f"{field}.min")
        high = _rate_number(items[1], f"{field}.max")
        if low > high:
            raise ConfigError(
                f"{field}.min must not exceed {field}.max",
                details={"field": field, "min": low, "max": high},
            )
        return low, high
    minimum = getattr(value, "min", None)
    maximum = getattr(value, "max", None)
    if minimum is not None and maximum is not None:
        low = _rate_number(minimum, f"{field}.min")
        high = _rate_number(maximum, f"{field}.max")
        if low > high:
            raise ConfigError(
                f"{field}.min must not exceed {field}.max",
                details={"field": field, "min": low, "max": high},
            )
        return low, high
    nested = getattr(value, "words_per_second", None)
    if nested is not None and not callable(nested):
        return _rate_bounds(nested, field)
    raise ConfigError(
        f"cannot read {field} from {type(value).__name__}",
        details={"field": field, "value": repr(value)},
    )


def _normalise_language(value: str) -> str:
    return value.strip().casefold().replace("-", "_")


def _language_key(
    language: str | None,
    default: Any,
    mapping: Mapping[Any, Any],
) -> Any | None:
    wanted = language if language else default
    if wanted is None:
        return None
    text = str(wanted)
    candidates = [text, _normalise_language(text)]
    base = _normalise_language(text).split("_", 1)[0]
    candidates.append(base)
    for key in mapping:
        key_text = str(key)
        if key_text in candidates or _normalise_language(key_text) in candidates:
            return key
    return None


def _bounds_from_source(
    source: Any,
    language: str | None,
) -> tuple[float, float] | None:
    if source is None:
        return None
    method = getattr(source, "words_per_second", None)
    if callable(method):
        try:
            value = method(language) if language is not None else method()
        except TypeError:
            value = method(language)
        return _rate_bounds(value)
    languages = getattr(source, "languages", None)
    if isinstance(languages, Mapping):
        key = _language_key(language, getattr(source, "default", None), languages)
        if key is None:
            return None
        return _rate_bounds(languages[key])
    if languages is not None and not callable(languages):
        return _bounds_from_source(languages, language)
    if isinstance(source, Mapping):
        if any(
            key in source for key in (*_RATE_KEYS, *_MIN_RATE_KEYS, *_MAX_RATE_KEYS)
        ):
            return _rate_bounds(source)
        nested_languages = source.get("languages")
        if nested_languages is not None:
            return _bounds_from_source(nested_languages, language)
        key = _language_key(language, source.get("default"), source)
        if key is None:
            return None
        return _rate_bounds(source[key])
    nested = getattr(source, "words_per_second", None)
    if nested is not None and not callable(nested):
        return _rate_bounds(nested)
    if hasattr(source, "min") or hasattr(source, "max"):
        return _rate_bounds(source)
    return _rate_bounds(source)


def resolve_words_per_second_range(
    language: str | None = None,
    *,
    words_per_second: Any = None,
    min_words_per_second: Any = None,
    max_words_per_second: Any = None,
    languages: Any = None,
    resolver: Any = None,
    config: Any = None,
    language_config: Any = None,
) -> tuple[float, float]:
    """Resolve per-language speaking-rate bounds, falling back to the default.

    ``words_per_second`` may be a scalar, a pair, a mapping with ``min``/``max``
    or a config model. ``language`` is looked up through ``resolver``, a
    languages config, a mapping of language codes, or a single language config.
    """
    if words_per_second is not None:
        return _rate_bounds(words_per_second)
    if min_words_per_second is not None or max_words_per_second is not None:
        low = (
            _rate_number(min_words_per_second, "min_words_per_second")
            if min_words_per_second is not None
            else None
        )
        high = (
            _rate_number(max_words_per_second, "max_words_per_second")
            if max_words_per_second is not None
            else None
        )
        if low is None:
            assert high is not None
            low = high
        if high is None:
            high = low
        if low > high:
            raise ConfigError(
                "min_words_per_second must not exceed max_words_per_second",
                details={"min": low, "max": high},
            )
        return low, high
    sources: list[Any] = [language_config, resolver, languages, config]
    if language is not None and not isinstance(language, str):
        sources.insert(0, language)
        language = None
    for source in sources:
        if source is None:
            continue
        bounds = _bounds_from_source(source, language)
        if bounds is not None:
            return bounds
    return DEFAULT_WORDS_PER_SECOND, DEFAULT_WORDS_PER_SECOND


def resolve_words_per_second(
    language: str | None = None,
    *,
    words_per_second: Any = None,
    min_words_per_second: Any = None,
    max_words_per_second: Any = None,
    languages: Any = None,
    resolver: Any = None,
    config: Any = None,
    language_config: Any = None,
) -> float:
    """Resolve one speaking rate from per-language bounds.

    When the source supplies a min/max pair the midpoint is used, so the
    estimate stays deterministic while a range is still available through
    :func:`resolve_words_per_second_range`.
    """
    low, high = resolve_words_per_second_range(
        language,
        words_per_second=words_per_second,
        min_words_per_second=min_words_per_second,
        max_words_per_second=max_words_per_second,
        languages=languages,
        resolver=resolver,
        config=config,
        language_config=language_config,
    )
    return (low + high) / 2.0


def _from_mapping(value: Mapping[str, Any]) -> LengthRange:
    minimum = _first(value, _MIN_KEYS)
    maximum = _first(value, _MAX_KEYS)
    target = _first(value, _TARGET_KEYS)
    tolerance = _first(value, _TOLERANCE_KEYS)
    if target is not None:
        center = _number(target, "length")
        span = (
            _number(tolerance, "length tolerance")
            if tolerance is not None
            else center * 0.2
        )
        low = _number(minimum, "min_seconds") if minimum is not None else center - span
        high = _number(maximum, "max_seconds") if maximum is not None else center + span
        return LengthRange(max(0.0, low), high)
    if minimum is None and maximum is None:
        raise ConfigError(
            "length must define min_seconds and max_seconds or a target",
            details={"length": dict(value)},
        )
    low = _number(minimum, "min_seconds") if minimum is not None else 0.0
    high = _number(maximum, "max_seconds") if maximum is not None else math.inf
    return LengthRange(low, high)


def coerce_length_range(value: Any) -> LengthRange:
    """Turn a preset name, target, pair or mapping into a :class:`LengthRange`.

    A bare number is a target in seconds with a twenty percent tolerance, while
    a two-item sequence is an explicit ``(min, max)`` pair.
    """
    if isinstance(value, LengthRange):
        return value
    if isinstance(value, Mapping):
        return _from_mapping(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ConfigError("length must not be empty")
        preset = LENGTH_PRESETS.get(text.casefold())
        if preset is not None:
            return LengthRange(*preset)
        try:
            return _from_target(float(text), 0.2)
        except ValueError:
            raise ConfigError(
                f"unknown length preset: {value!r} "
                f"(known: {', '.join(sorted(LENGTH_PRESETS))})",
                details={"length": value},
            ) from None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _from_target(float(value), 0.2)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        if len(items) != 2:
            raise ConfigError(
                "length range must have exactly two values",
                details={"length": value},
            )
        return LengthRange(
            _number(items[0], "min_seconds"),
            _number(items[1], "max_seconds"),
        )
    raise ConfigError(
        f"cannot read a length range from {type(value).__name__}",
        details={"length": repr(value)},
    )


def resolve_length_range(
    value: Any = None,
    *,
    min_seconds: float | None = None,
    max_seconds: float | None = None,
) -> LengthRange | None:
    """Resolve an optional length with explicit bound overrides.

    ``None`` means no target was requested, so the caller should estimate the
    reading time without rejecting the script. An explicit bound overrides the
    matching side of the base range while keeping the other side.
    """
    if value is None and min_seconds is None and max_seconds is None:
        return None
    base = coerce_length_range(value) if value is not None else None
    if min_seconds is None and max_seconds is None:
        return base
    low = (
        _number(min_seconds, "min_seconds")
        if min_seconds is not None
        else (base.min_seconds if base is not None else 0.0)
    )
    high = (
        _number(max_seconds, "max_seconds")
        if max_seconds is not None
        else (base.max_seconds if base is not None else math.inf)
    )
    return LengthRange(low, high)


def count_words(text: str | None) -> int:
    """Return the number of words in *text*, ignoring punctuation."""
    if not isinstance(text, str):
        return 0
    return len(_WORD.findall(text))


def reading_time_seconds(
    text: str | None,
    *,
    words_per_second: Any = None,
    language: str | None = None,
    languages: Any = None,
    resolver: Any = None,
    config: Any = None,
    language_config: Any = None,
    min_words_per_second: Any = None,
    max_words_per_second: Any = None,
) -> float:
    """Estimate how long *text* takes to read, in seconds.

    A language can be supplied instead of a rate, in which case the speaking
    rate is read from the matching language config. When only bounds are known
    their midpoint is used.
    """
    rate = resolve_words_per_second(
        language,
        words_per_second=words_per_second,
        min_words_per_second=min_words_per_second,
        max_words_per_second=max_words_per_second,
        languages=languages,
        resolver=resolver,
        config=config,
        language_config=language_config,
    )
    if not math.isfinite(rate) or rate <= 0:
        raise ConfigError(
            "words_per_second must be a positive number",
            details={"words_per_second": words_per_second},
        )
    return count_words(text) / rate


def estimate_duration(
    text: str | None,
    *,
    words_per_second: Any = None,
    language: str | None = None,
    languages: Any = None,
    resolver: Any = None,
    config: Any = None,
    language_config: Any = None,
    min_words_per_second: Any = None,
    max_words_per_second: Any = None,
) -> float:
    """Estimate the reading time of *text* using a per-language speaking rate."""
    return reading_time_seconds(
        text,
        words_per_second=words_per_second,
        language=language,
        languages=languages,
        resolver=resolver,
        config=config,
        language_config=language_config,
        min_words_per_second=min_words_per_second,
        max_words_per_second=max_words_per_second,
    )


def estimate_duration_range(
    text: str | None,
    *,
    words_per_second: Any = None,
    language: str | None = None,
    languages: Any = None,
    resolver: Any = None,
    config: Any = None,
    language_config: Any = None,
    min_words_per_second: Any = None,
    max_words_per_second: Any = None,
) -> LengthRange:
    """Estimate the fastest and slowest reading times of *text* in seconds."""
    low_rate, high_rate = resolve_words_per_second_range(
        language,
        words_per_second=words_per_second,
        min_words_per_second=min_words_per_second,
        max_words_per_second=max_words_per_second,
        languages=languages,
        resolver=resolver,
        config=config,
        language_config=language_config,
    )
    words = count_words(text)
    return LengthRange(words / high_rate, words / low_rate)


reading_time_range = estimate_duration_range
measure_length = estimate_duration
measure_duration = estimate_duration
estimate_reading_time = estimate_duration
estimate_length = estimate_duration


def _rewrite_limit(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_rewrites must be an integer")
    if value < 0:
        raise ValueError("max_rewrites must not be negative")
    return value


def _call_length_rewriter(
    rewrite: Callable[..., Any],
    text: str,
    measured: float,
    target: LengthRange,
    attempt: int,
    words_per_second: float,
) -> str:
    known: dict[str, Any] = {
        "text": text,
        "candidate": text,
        "measured": measured,
        "measured_seconds": measured,
        "seconds": measured,
        "duration": measured,
        "duration_seconds": measured,
        "reading_time": measured,
        "target": target,
        "target_range": target,
        "min_seconds": target.min_seconds,
        "max_seconds": target.max_seconds,
        "attempt": attempt,
        "rewrite": attempt,
        "words_per_second": words_per_second,
        "rate": words_per_second,
    }
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
        args: list[Any] = []
        for index, parameter in enumerate(positional):
            if index == 0 and parameter.name in {"text", "candidate", "value"}:
                args.append(text)
            elif parameter.name in known:
                args.append(known[parameter.name])
            elif index == 0:
                args.append(text)
            elif index == 1:
                args.append(measured)
            elif index == 2:
                args.append(target)
            else:
                args.append(attempt)
        kwargs: dict[str, Any] = {
            parameter.name: known[parameter.name]
            for parameter in parameters
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY
            and parameter.name in known
        }
        if keyword:
            positional_names = {parameter.name for parameter in positional}
            kwargs.update(
                {
                    name: value
                    for name, value in known.items()
                    if name not in positional_names
                }
            )
        if not positional and not keyword:
            replacement = rewrite()
        else:
            replacement = rewrite(*args, **kwargs)
    if not isinstance(replacement, str):
        raise ConfigError(
            "the length rewriter must return text",
            details={"returned": type(replacement).__name__},
        )
    cleaned = replacement.strip()
    if not cleaned:
        raise ConfigError(
            "the length rewriter returned empty text",
            details={"attempt": attempt},
        )
    return cleaned


def check_length(
    candidate: Any,
    target: Any = None,
    *,
    rewrite: Callable[..., Any] | None = None,
    rewriter: Callable[..., Any] | None = None,
    max_rewrites: int | None = None,
    max_attempts: int | None = None,
    max_regenerations: int | None = None,
    words_per_second: Any = None,
    language: str | None = None,
    languages: Any = None,
    resolver: Any = None,
    config: Any = None,
    language_config: Any = None,
    min_words_per_second: Any = None,
    max_words_per_second: Any = None,
    min_seconds: float | None = None,
    max_seconds: float | None = None,
    target_seconds: Any = None,
    duration: Any = None,
    length: Any = None,
    label: str = "length",
    strict: bool = False,
) -> LengthResult:
    """Measure *candidate* against *target* and rewrite it until it fits.

    While the estimate falls outside the target and rewrites remain, the
    rewriter is called with the current text. When no rewriter is supplied, or
    the limit is exhausted, the result reports ``passed=False``. With
    ``strict=True`` a candidate that never fits raises :class:`ConfigError`
    with the measured time and the target range in the details.
    """
    text = candidate.strip() if isinstance(candidate, str) else str(candidate)
    requested = target
    for alias in (target_seconds, duration, length):
        if requested is None:
            requested = alias
    resolved_target = resolve_length_range(
        requested,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
    )
    if rewrite is None:
        rewrite = rewriter
    limit = _rewrite_limit(
        max_rewrites
        if max_rewrites is not None
        else max_attempts
        if max_attempts is not None
        else max_regenerations
        if max_regenerations is not None
        else DEFAULT_MAX_LENGTH_REWRITES
    )
    rate = resolve_words_per_second(
        language,
        words_per_second=words_per_second,
        min_words_per_second=min_words_per_second,
        max_words_per_second=max_words_per_second,
        languages=languages,
        resolver=resolver,
        config=config,
        language_config=language_config,
    )
    current = text
    measured = reading_time_seconds(current, words_per_second=rate)
    applied = 0
    rewritten = False
    while (
        resolved_target is not None
        and not resolved_target.contains(measured)
        and applied < limit
        and rewrite is not None
    ):
        current = _call_length_rewriter(
            rewrite,
            current,
            measured,
            resolved_target,
            applied + 1,
            rate,
        )
        applied += 1
        rewritten = True
        measured = reading_time_seconds(current, words_per_second=rate)
    passed = resolved_target is None or resolved_target.contains(measured)
    reason = "" if passed else "length"
    result = LengthResult(
        current,
        seconds=measured,
        target=resolved_target,
        words=count_words(current),
        words_per_second=rate,
        rewritten=rewritten,
        attempts=applied,
        passed=passed,
        reason=reason,
    )
    if strict and not passed:
        assert resolved_target is not None
        raise ConfigError(
            f"{label} reading time {measured:.1f}s is outside the target range "
            f"{resolved_target.min_seconds:.1f}-"
            f"{resolved_target.max_seconds:.1f}s",
            details={
                "reason": "length",
                "measured": round(measured, 3),
                "measured_seconds": round(measured, 3),
                "duration_seconds": round(measured, 3),
                "min_seconds": resolved_target.min_seconds,
                "max_seconds": resolved_target.max_seconds,
                "target": resolved_target.to_dict(),
                "attempts": applied,
                "rewritten": rewritten,
                "words": count_words(current),
                "words_per_second": rate,
            },
        )
    return result


def enforce_length(
    candidate: Any,
    target: Any = None,
    **kwargs: Any,
) -> LengthResult:
    """Like :func:`check_length` but refuse a script that never fits."""
    kwargs["strict"] = True
    return check_length(candidate, target, **kwargs)


check_duration = check_length
check_reading_time = check_length
enforce_duration = enforce_length
enforce_reading_time = enforce_length
rewrite_to_length = enforce_length
rewrite_to_duration = enforce_length
rewrite_length = enforce_length
rewrite_duration = enforce_length
rewrite_reading_time = enforce_length
require_length = enforce_length
ensure_length = enforce_length
validate_length = enforce_length
fit_length = enforce_length
fit_to_length = enforce_length
fit_reading_time = enforce_length

"""Adapt canonical media requests to the limits one provider declares.

A provider rarely accepts every value a caller may ask for. An image API often
allows only a handful of pixel sizes, a video model caps the clip length and a
speech model may not understand pronunciation hints. Silently dropping those
values makes the result differ from the request without anyone noticing, so
this module makes every change explicit:

* a requested size is snapped to the nearest supported size and a warning
  records both values;
* a duration outside the provider range is clamped and recorded;
* a prompt longer than the provider limit is shortened at a word boundary and
  recorded;
* an option the provider cannot consume is rejected with
  :class:`UnsupportedOptionError` instead of being ignored.

Constraints can be declared directly, or derived from the
``ProviderMeta.supported_options`` tokens an adapter already publishes. The
adaptation is pure: the request is immutable, the original is never changed and
the result carries the adapted request plus the ordered warnings.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator, model_validator

from aivia.core.errors import InvalidRequestError
from aivia.core.models.provider import ProviderMeta
from aivia.providers.media.dto import (
    ImageRequest,
    MediaKind,
    MediaRequest,
    MusicRequest,
    SpeechRequest,
    VideoRequest,
)
from aivia.providers.protocols.base import ProtocolModel
from aivia.providers.protocols.image import normalize_aspect_ratio
from aivia.providers.protocols.media import MediaDimensions

__all__ = [
    "AdaptationAction",
    "AdaptationError",
    "AdaptationResult",
    "AdaptationWarning",
    "AdaptedRequest",
    "MediaConstraints",
    "ProviderConstraints",
    "UnadaptableRequestError",
    "UnsupportedOptionError",
    "adapt_image_request",
    "adapt_media_request",
    "adapt_music_request",
    "adapt_request",
    "adapt_speech_request",
    "adapt_video_request",
    "clamp_duration",
    "snap_size",
]

_SIZE_PATTERN = re.compile(r"^(?P<width>\d+)\s*x\s*(?P<height>\d+)$", re.IGNORECASE)
_RATIO_PATTERN = re.compile(r"^(?P<width>\d+)\s*:\s*(?P<height>\d+)$")
_TOKEN_SEPARATOR = re.compile(r"[,|]")


class AdaptationAction(StrEnum):
    """The kind of change an adaptation made."""

    SNAPPED_SIZE = "snapped_size"
    SNAPPED_ASPECT_RATIO = "snapped_aspect_ratio"
    CLAMPED_DURATION = "clamped_duration"
    CLAMPED_COUNT = "clamped_count"
    TRUNCATED_PROMPT = "truncated_prompt"


class AdaptationError(InvalidRequestError):
    """A request could not be adapted to the provider's limits."""

    code = "adaptation"


class UnsupportedOptionError(AdaptationError):
    """The provider cannot consume an option the request explicitly set."""

    code = "unsupported_option"

    def __init__(
        self,
        option: str,
        *,
        value: Any = None,
        supported: Iterable[str] = (),
        message: str | None = None,
    ) -> None:
        self.option = option
        self.value = value
        self.supported = tuple(supported)
        details: dict[str, Any] = {"option": option}
        if value is not None:
            details["value"] = _option_text(value)
        if self.supported:
            details["supported"] = list(self.supported)
        super().__init__(
            message or f"provider does not support the {option!r} option",
            details=details,
        )


class UnadaptableRequestError(AdaptationError):
    """No supported value is close enough to adapt the request to."""

    code = "unadaptable_request"


@dataclass(frozen=True, slots=True)
class AdaptationWarning:
    """One explicit change made while adapting a request.

    ``requested`` and ``applied`` are JSON-friendly values so the warning can be
    written to a sidecar or a run report without another conversion.
    """

    action: AdaptationAction
    option: str
    message: str
    requested: Any = None
    applied: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for logs and sidecars."""
        payload: dict[str, Any] = {
            "action": self.action.value,
            "option": self.option,
            "message": self.message,
        }
        if self.requested is not None:
            payload["requested"] = self.requested
        if self.applied is not None:
            payload["applied"] = self.applied
        return payload


@dataclass(frozen=True, slots=True)
class AdaptationResult[T: ProtocolModel]:
    """An adapted request and every change that was made to it."""

    request: T
    warnings: tuple[AdaptationWarning, ...] = ()

    @property
    def adapted(self) -> T:
        """Alias for :attr:`request`."""
        return self.request

    @property
    def changed(self) -> bool:
        """Whether adaptation changed at least one value."""
        return bool(self.warnings)

    @property
    def messages(self) -> tuple[str, ...]:
        """The human-readable warning messages in adaptation order."""
        return tuple(warning.message for warning in self.warnings)


AdaptedRequest = AdaptationResult
"""Alias for :class:`AdaptationResult` for callers that prefer the longer name."""


def _option_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, StrEnum):
        return value.value
    return str(value)


def _coerce_size(value: Any) -> MediaDimensions:
    if isinstance(value, MediaDimensions):
        return value
    if isinstance(value, str):
        text = value.strip()
        parsed = _SIZE_PATTERN.fullmatch(text)
        if parsed is None:
            raise ValueError(
                f"size must be WIDTHxHEIGHT, for example 1024x1024: {value!r}"
            )
        return MediaDimensions(
            width=int(parsed.group("width")),
            height=int(parsed.group("height")),
        )
    if isinstance(value, Mapping):
        return MediaDimensions.model_validate(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != 2:
            raise ValueError("a size pair must contain exactly width and height")
        return MediaDimensions(width=int(value[0]), height=int(value[1]))
    raise ValueError(f"unsupported size value: {value!r}")


def _normalize_sizes(values: Iterable[Any]) -> tuple[MediaDimensions, ...]:
    sizes: list[MediaDimensions] = []
    seen: set[tuple[int, int]] = set()
    for value in values:
        size = _coerce_size(value)
        key = (size.width, size.height)
        if key not in seen:
            seen.add(key)
            sizes.append(size)
    return tuple(sizes)


def _normalize_ratio(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"aspect ratio must be a string: {value!r}")
    return normalize_aspect_ratio(value)


def _size_text(size: MediaDimensions) -> str:
    return f"{size.width}x{size.height}"


def _ratio_text(width: int, height: int) -> str:
    divisor = math.gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def _parse_ratio(value: str) -> float:
    parsed = _RATIO_PATTERN.fullmatch(value.strip())
    if parsed is None:
        raise ValueError(f"aspect ratio must be WIDTH:HEIGHT: {value!r}")
    width = int(parsed.group("width"))
    height = int(parsed.group("height"))
    if width <= 0 or height <= 0:
        raise ValueError("aspect ratio terms must be positive")
    return width / height


def _token_values(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in _TOKEN_SEPARATOR.split(value) if part.strip())


class MediaConstraints(ProtocolModel):
    """The limits one provider declares for one media capability.

    ``sizes`` is the set of exact pixel sizes the provider accepts and
    ``aspect_ratios`` the shapes it accepts when no exact size is required.
    ``min_duration_seconds`` and ``max_duration_seconds`` bound a clip or track.
    ``formats``, ``fps``, ``models`` and ``supported_options`` restrict values.
    ``max_count`` and ``max_prompt_chars`` cap variations and prompt length.

    ``supports`` is the allowlist of optional option names the provider can
    consume. ``None`` means the provider did not publish an allowlist, so no
    option is rejected on support grounds; an empty set means the provider
    explicitly supports no optional options.
    """

    sizes: tuple[MediaDimensions, ...] = ()
    aspect_ratios: tuple[str, ...] = ()
    min_duration_seconds: float | None = Field(default=None, gt=0.0)
    max_duration_seconds: float | None = Field(default=None, gt=0.0)
    fps: tuple[int, ...] = ()
    formats: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    max_count: int | None = Field(default=None, ge=1)
    max_prompt_chars: int | None = Field(default=None, ge=1)
    supports: frozenset[str] | None = None
    supported_options: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    @field_validator("sizes", mode="before")
    @classmethod
    def _coerce_sizes(cls, value: Any) -> Any:
        if value is None:
            return ()
        if isinstance(value, (str, Mapping, MediaDimensions)):
            value = (value,)
        return tuple(_coerce_size(item) for item in value)

    @field_validator("sizes")
    @classmethod
    def _validate_sizes(
        cls, value: tuple[MediaDimensions, ...]
    ) -> tuple[MediaDimensions, ...]:
        return _normalize_sizes(value)

    @field_validator("aspect_ratios", mode="before")
    @classmethod
    def _coerce_aspect_ratios(cls, value: Any) -> Any:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        return value

    @field_validator("aspect_ratios")
    @classmethod
    def _validate_aspect_ratios(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        ratios: list[str] = []
        for raw in value:
            ratio = _normalize_ratio(raw)
            if ratio not in ratios:
                ratios.append(ratio)
        return tuple(ratios)

    @field_validator("fps", "formats", "models", mode="before")
    @classmethod
    def _coerce_string_sequences(cls, value: Any) -> Any:
        if value is None:
            return ()
        if isinstance(value, (str, int)):
            return (value,)
        return value

    @field_validator("fps")
    @classmethod
    def _validate_fps(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        rates: list[int] = []
        for raw in value:
            rate = int(raw)
            if rate <= 0:
                raise ValueError("fps values must be positive")
            if rate not in rates:
                rates.append(rate)
        return tuple(rates)

    @field_validator("formats", "models")
    @classmethod
    def _validate_string_sequences(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        entries: list[str] = []
        for raw in value:
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError("entries must be non-empty strings")
            entry = raw.strip()
            if entry not in entries:
                entries.append(entry)
        return tuple(entries)

    @field_validator("supports", mode="before")
    @classmethod
    def _coerce_supports(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            return frozenset({value})
        return frozenset(str(item).strip() for item in value if str(item).strip())

    @field_validator("supported_options", mode="before")
    @classmethod
    def _coerce_supported_options(cls, value: Any) -> Any:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            return value
        options: dict[str, Any] = {}
        for key, entries in value.items():
            if isinstance(entries, str):
                entries = _token_values(entries)
            options[str(key).strip()] = entries
        return options

    @field_validator("supported_options")
    @classmethod
    def _validate_supported_options(
        cls, value: dict[str, tuple[str, ...]]
    ) -> dict[str, tuple[str, ...]]:
        options: dict[str, tuple[str, ...]] = {}
        for key, entries in value.items():
            name = key.strip()
            if not name:
                raise ValueError("supported option keys must be non-empty")
            cleaned: list[str] = []
            for raw in entries:
                if not isinstance(raw, str) or not raw.strip():
                    raise ValueError(
                        "supported option values must be non-empty strings"
                    )
                entry = raw.strip()
                if entry not in cleaned:
                    cleaned.append(entry)
            options[name] = tuple(cleaned)
        return options

    @model_validator(mode="after")
    def _validate_duration_range(self) -> MediaConstraints:
        if (
            self.min_duration_seconds is not None
            and self.max_duration_seconds is not None
            and self.min_duration_seconds > self.max_duration_seconds
        ):
            raise ValueError(
                "min_duration_seconds must not exceed max_duration_seconds"
            )
        return self

    @classmethod
    def from_metadata(
        cls,
        metadata: ProviderMeta | Mapping[str, Any],
        capability: str,
    ) -> MediaConstraints:
        """Derive constraints from a provider's declared option tokens.

        ``ProviderMeta.supported_options`` maps a capability to human-readable
        tokens. The tokens this layer understands are ``1024x1024`` or
        ``size:...`` for sizes, ``16:9`` or ``aspect:...`` for aspect ratios,
        ``duration:MIN-MAX`` (or ``duration:MAX``), ``fps:24,30``,
        ``format:png,webp``, ``count:4``, ``prompt:1000``, ``models:a,b`` and
        ``name:value`` or ``name=value`` for a valued option. A bare token such
        as ``negative_prompt`` or ``seed`` means the option is supported.

        A capability without any declared tokens is treated as unconstrained
        rather than as supporting nothing, because an empty list usually means
        the adapter did not publish options at all.
        """
        source = (
            metadata
            if isinstance(metadata, ProviderMeta)
            else ProviderMeta.model_validate(metadata)
        )
        tokens = source.supported_options.get(capability.strip(), ())
        return cls.from_options(tokens)

    @classmethod
    def from_options(cls, tokens: Iterable[str]) -> MediaConstraints:
        """Build constraints from the option token grammar in :meth:`from_metadata`."""
        sizes: list[str] = []
        ratios: list[str] = []
        fps: list[int] = []
        formats: list[str] = []
        models: list[str] = []
        supports: set[str] = set()
        supported_options: dict[str, list[str]] = {}
        min_duration: float | None = None
        max_duration: float | None = None
        max_count: int | None = None
        max_prompt_chars: int | None = None
        found = False

        def values(rest: str) -> tuple[str, ...]:
            return _token_values(rest)

        for raw in tokens:
            if not isinstance(raw, str):
                raise ValueError("option tokens must be strings")
            text = raw.strip()
            if not text:
                continue
            found = True
            lower = text.lower()

            if _SIZE_PATTERN.fullmatch(lower):
                sizes.append(lower)
                supports.add("size")
                continue
            if _RATIO_PATTERN.fullmatch(lower):
                ratios.append(lower)
                supports.add("aspect_ratio")
                continue
            if lower.startswith("size:"):
                sizes.extend(values(text.split(":", 1)[1]))
                supports.add("size")
                continue
            if lower.startswith("aspect:"):
                ratios.extend(values(text.split(":", 1)[1]))
                supports.add("aspect_ratio")
                continue
            if lower.startswith("duration:"):
                rest = lower.split(":", 1)[1].strip()
                if "-" in rest:
                    low, high = rest.split("-", 1)
                    min_duration = float(low)
                    max_duration = float(high)
                else:
                    max_duration = float(rest)
                supports.add("duration")
                continue
            if lower.startswith("fps:"):
                fps.extend(int(item) for item in values(text.split(":", 1)[1]))
                supports.add("fps")
                continue
            if lower.startswith("format:"):
                formats.extend(values(text.split(":", 1)[1]))
                supports.add("format")
                continue
            if lower.startswith("models:"):
                models.extend(values(text.split(":", 1)[1]))
                supports.add("model")
                continue
            if lower.startswith("count:"):
                max_count = int(lower.split(":", 1)[1])
                supports.add("count")
                continue
            if lower.startswith("prompt:"):
                max_prompt_chars = int(lower.split(":", 1)[1])
                supports.add("prompt")
                continue
            if lower.startswith("max_prompt_chars:"):
                max_prompt_chars = int(lower.split(":", 1)[1])
                supports.add("prompt")
                continue

            separator = "=" if "=" in text else ":" if ":" in text else ""
            if separator:
                name, rest = text.split(separator, 1)
                key = name.strip().lower()
                if not key:
                    raise ValueError(f"option token has no name: {raw!r}")
                supported_options.setdefault(key, []).extend(values(rest))
                supports.add(key)
                continue

            supports.add(lower)

        if not found:
            return cls()

        return cls(
            sizes=tuple(_coerce_size(size) for size in sizes),
            aspect_ratios=tuple(ratios),
            min_duration_seconds=min_duration,
            max_duration_seconds=max_duration,
            fps=tuple(fps),
            formats=tuple(formats),
            models=tuple(models),
            max_count=max_count,
            max_prompt_chars=max_prompt_chars,
            supports=frozenset(supports),
            supported_options={
                key: tuple(entries) for key, entries in supported_options.items()
            },
        )


ProviderConstraints = MediaConstraints
"""Alias for :class:`MediaConstraints` for callers that name the owner."""


def snap_size(
    width: int | None,
    height: int | None,
    sizes: Iterable[Any],
    *,
    aspect_ratio: str | None = None,
) -> MediaDimensions:
    """Return the supported size closest to the requested shape and area.

    With both *width* and *height* the closest aspect ratio wins, then the
    closest pixel area. With only *aspect_ratio* the closest shape wins and the
    smallest matching size is chosen, so a caller is not charged for a larger
    output than it asked for. An empty *sizes* collection is a programming
    error and raises :class:`ValueError`.
    """
    allowed = _normalize_sizes(sizes)
    if not allowed:
        raise ValueError("snap_size requires at least one supported size")

    if width is not None and height is not None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        target_ratio = width / height
        target_pixels: int | None = width * height
    elif aspect_ratio is not None:
        target_ratio = _parse_ratio(aspect_ratio)
        target_pixels = None
    else:
        raise ValueError("snap_size needs width and height or an aspect ratio")

    def distance(size: MediaDimensions) -> tuple[float, float, int, int, int]:
        ratio_penalty = abs(math.log(size.aspect_ratio / target_ratio))
        scale_penalty = (
            0.0 if target_pixels is None else abs(math.log(size.pixels / target_pixels))
        )
        return (
            ratio_penalty,
            scale_penalty,
            size.pixels,
            size.width,
            size.height,
        )

    return min(allowed, key=distance)


def clamp_duration(
    seconds: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Clamp *seconds* into the inclusive provider range."""
    if minimum is not None and seconds < minimum:
        return minimum
    if maximum is not None and seconds > maximum:
        return maximum
    return seconds


def _snap_aspect_ratio(value: str, allowed: Iterable[str]) -> str:
    ratios = tuple(allowed)
    if not ratios:
        raise ValueError("snap_aspect_ratio requires at least one ratio")
    target = _parse_ratio(value)
    return min(
        ratios,
        key=lambda ratio: abs(math.log(_parse_ratio(ratio) / target)),
    )


def _require_supported(
    constraints: MediaConstraints,
    option: str,
    *,
    value: Any = None,
    active: bool = True,
    implied: bool = False,
) -> None:
    if not active:
        return
    allowed = constraints.supported_options.get(option)
    if allowed is not None and _option_text(value) not in allowed:
        raise UnsupportedOptionError(
            option,
            value=value,
            supported=allowed,
            message=(
                f"provider does not support {option}={_option_text(value)!r}; "
                f"supported values: {', '.join(allowed)}"
            ),
        )
    if implied:
        return
    if constraints.supports is not None and option not in constraints.supports:
        raise UnsupportedOptionError(option, value=value)


def _check_format(
    value: Any,
    constraints: MediaConstraints,
) -> None:
    text = _option_text(value)
    if constraints.formats and text not in constraints.formats:
        raise UnsupportedOptionError(
            "format",
            value=text,
            supported=constraints.formats,
            message=(
                f"provider does not support format={text!r}; "
                f"supported formats: {', '.join(constraints.formats)}"
            ),
        )
    _require_supported(
        constraints,
        "format",
        value=text,
        active=bool(constraints.supported_options.get("format")),
        implied=bool(constraints.formats),
    )


def _check_model(model: str | None, constraints: MediaConstraints) -> None:
    if model is None:
        return
    if constraints.models and model not in constraints.models:
        raise UnsupportedOptionError(
            "model",
            value=model,
            supported=constraints.models,
            message=(
                f"provider does not support model={model!r}; "
                f"supported models: {', '.join(constraints.models)}"
            ),
        )
    _require_supported(
        constraints,
        "model",
        value=model,
        implied=bool(constraints.models),
    )


def _adapt_prompt(
    value: str,
    constraints: MediaConstraints,
    warnings: list[AdaptationWarning],
    *,
    option: str,
) -> str:
    limit = constraints.max_prompt_chars
    if limit is None or len(value) <= limit:
        return value
    truncated = value[:limit]
    if " " in truncated:
        at_word = truncated.rsplit(" ", 1)[0].rstrip()
        if at_word:
            truncated = at_word
    truncated = truncated.rstrip() or value[:limit]
    warnings.append(
        AdaptationWarning(
            action=AdaptationAction.TRUNCATED_PROMPT,
            option=option,
            message=(
                f"{option} was shortened from {len(value)} to "
                f"{len(truncated)} characters to fit the provider limit"
            ),
            requested=len(value),
            applied=len(truncated),
        )
    )
    return truncated


def _adapt_count(
    count: int,
    constraints: MediaConstraints,
    warnings: list[AdaptationWarning],
) -> int:
    if constraints.max_count is None:
        _require_supported(
            constraints,
            "count",
            value=count,
            active=count > 1,
        )
        return count
    if count <= constraints.max_count:
        return count
    warnings.append(
        AdaptationWarning(
            action=AdaptationAction.CLAMPED_COUNT,
            option="count",
            message=(
                f"count was reduced from {count} to {constraints.max_count} "
                "to fit the provider limit"
            ),
            requested=count,
            applied=constraints.max_count,
        )
    )
    return constraints.max_count


def _adapt_duration(
    seconds: float | None,
    constraints: MediaConstraints,
    warnings: list[AdaptationWarning],
) -> float | None:
    if seconds is None:
        return None
    bounded = (
        constraints.min_duration_seconds is not None
        or constraints.max_duration_seconds is not None
    )
    _require_supported(
        constraints,
        "duration",
        value=seconds,
        implied=bounded,
    )
    clamped = clamp_duration(
        seconds,
        minimum=constraints.min_duration_seconds,
        maximum=constraints.max_duration_seconds,
    )
    if clamped != seconds:
        warnings.append(
            AdaptationWarning(
                action=AdaptationAction.CLAMPED_DURATION,
                option="duration_seconds",
                message=(
                    f"duration_seconds was clamped from {seconds:g} to "
                    f"{clamped:g} to fit the provider limit"
                ),
                requested=seconds,
                applied=clamped,
            )
        )
    return clamped


def _adapt_size(
    width: int | None,
    height: int | None,
    aspect_ratio: str | None,
    constraints: MediaConstraints,
    warnings: list[AdaptationWarning],
) -> tuple[int | None, int | None, str | None]:
    has_size = width is not None and height is not None
    if constraints.sizes:
        if has_size:
            assert width is not None and height is not None
            snapped = snap_size(width, height, constraints.sizes)
            if (snapped.width, snapped.height) != (width, height):
                warnings.append(
                    AdaptationWarning(
                        action=AdaptationAction.SNAPPED_SIZE,
                        option="size",
                        message=(
                            f"size was snapped from {width}x{height} to "
                            f"{_size_text(snapped)} to fit the provider sizes"
                        ),
                        requested=f"{width}x{height}",
                        applied=_size_text(snapped),
                    )
                )
            next_ratio = (
                _ratio_text(snapped.width, snapped.height)
                if aspect_ratio is not None
                else None
            )
            return snapped.width, snapped.height, next_ratio
        if aspect_ratio is not None:
            snapped = snap_size(
                None,
                None,
                constraints.sizes,
                aspect_ratio=aspect_ratio,
            )
            warnings.append(
                AdaptationWarning(
                    action=AdaptationAction.SNAPPED_SIZE,
                    option="size",
                    message=(
                        f"size was snapped from aspect ratio {aspect_ratio} to "
                        f"{_size_text(snapped)} to fit the provider sizes"
                    ),
                    requested=aspect_ratio,
                    applied=_size_text(snapped),
                )
            )
            return (
                snapped.width,
                snapped.height,
                _ratio_text(snapped.width, snapped.height),
            )
        return width, height, aspect_ratio

    if aspect_ratio is not None and constraints.aspect_ratios:
        snapped_ratio = _snap_aspect_ratio(aspect_ratio, constraints.aspect_ratios)
        if normalize_aspect_ratio(snapped_ratio) != normalize_aspect_ratio(
            aspect_ratio
        ):
            warnings.append(
                AdaptationWarning(
                    action=AdaptationAction.SNAPPED_ASPECT_RATIO,
                    option="aspect_ratio",
                    message=(
                        f"aspect_ratio was snapped from {aspect_ratio} to "
                        f"{snapped_ratio} to fit the provider ratios"
                    ),
                    requested=aspect_ratio,
                    applied=snapped_ratio,
                )
            )
        return width, height, snapped_ratio

    return width, height, aspect_ratio


def _check_size_support(
    width: int | None,
    height: int | None,
    aspect_ratio: str | None,
    constraints: MediaConstraints,
) -> None:
    has_size = width is not None and height is not None
    _require_supported(
        constraints,
        "size",
        value=f"{width}x{height}" if has_size else None,
        active=has_size,
        implied=bool(constraints.sizes),
    )
    _require_supported(
        constraints,
        "aspect_ratio",
        value=aspect_ratio,
        active=aspect_ratio is not None,
        implied=bool(constraints.sizes or constraints.aspect_ratios),
    )


def adapt_image_request(
    request: ImageRequest,
    constraints: MediaConstraints,
) -> AdaptationResult[ImageRequest]:
    """Adapt an image request to *constraints*, rejecting unsupported options."""
    warnings: list[AdaptationWarning] = []
    _check_model(request.model, constraints)
    _require_supported(
        constraints,
        "negative_prompt",
        value=request.negative_prompt,
        active=request.negative_prompt is not None,
    )
    _require_supported(
        constraints,
        "quality",
        value=request.quality,
        active=request.quality is not None,
    )
    _require_supported(
        constraints,
        "style",
        value=request.style,
        active=request.style is not None,
    )
    _require_supported(
        constraints,
        "seed",
        value=request.seed,
        active=request.seed is not None,
    )
    _require_supported(
        constraints,
        "reference_images",
        value=len(request.reference_images),
        active=bool(request.reference_images),
    )
    _check_size_support(
        request.width,
        request.height,
        request.aspect_ratio,
        constraints,
    )
    _check_format(request.format, constraints)

    width, height, aspect_ratio = _adapt_size(
        request.width,
        request.height,
        request.aspect_ratio,
        constraints,
        warnings,
    )
    updates: dict[str, Any] = {
        "prompt": _adapt_prompt(
            request.prompt,
            constraints,
            warnings,
            option="prompt",
        ),
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "count": _adapt_count(request.count, constraints, warnings),
    }
    return AdaptationResult(
        request=request.model_copy(update=updates),
        warnings=tuple(warnings),
    )


def adapt_video_request(
    request: VideoRequest,
    constraints: MediaConstraints,
) -> AdaptationResult[VideoRequest]:
    """Adapt a video request to *constraints*, rejecting unsupported options."""
    warnings: list[AdaptationWarning] = []
    _check_model(request.model, constraints)
    _require_supported(
        constraints,
        "negative_prompt",
        value=request.negative_prompt,
        active=request.negative_prompt is not None,
    )
    _require_supported(
        constraints,
        "first_frame",
        value="set",
        active=request.first_frame is not None,
    )
    _require_supported(
        constraints,
        "last_frame",
        value="set",
        active=request.last_frame is not None,
    )
    _require_supported(
        constraints,
        "seed",
        value=request.seed,
        active=request.seed is not None,
    )
    _require_supported(
        constraints,
        "generate_audio",
        value=request.generate_audio,
        active=request.generate_audio,
    )
    _require_supported(
        constraints,
        "audio_prompt",
        value=request.audio_prompt,
        active=request.audio_prompt is not None,
    )
    _require_supported(
        constraints,
        "fps",
        value=request.fps,
        active=request.fps is not None,
        implied=bool(constraints.fps),
    )
    if (
        constraints.fps
        and request.fps is not None
        and request.fps not in constraints.fps
    ):
        raise UnsupportedOptionError(
            "fps",
            value=request.fps,
            supported=(str(rate) for rate in constraints.fps),
            message=(
                f"provider does not support fps={request.fps}; "
                f"supported values: {', '.join(str(rate) for rate in constraints.fps)}"
            ),
        )
    _check_size_support(
        request.width,
        request.height,
        request.aspect_ratio,
        constraints,
    )
    _check_format(request.format, constraints)

    width, height, aspect_ratio = _adapt_size(
        request.width,
        request.height,
        request.aspect_ratio,
        constraints,
        warnings,
    )
    updates: dict[str, Any] = {
        "prompt": _adapt_prompt(
            request.prompt,
            constraints,
            warnings,
            option="prompt",
        ),
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "duration_seconds": _adapt_duration(
            request.duration_seconds,
            constraints,
            warnings,
        ),
        "count": _adapt_count(request.count, constraints, warnings),
    }
    return AdaptationResult(
        request=request.model_copy(update=updates),
        warnings=tuple(warnings),
    )


def adapt_music_request(
    request: MusicRequest,
    constraints: MediaConstraints,
) -> AdaptationResult[MusicRequest]:
    """Adapt a music request to *constraints*, rejecting unsupported options."""
    warnings: list[AdaptationWarning] = []
    _check_model(request.model, constraints)
    _require_supported(
        constraints,
        "lyrics",
        value="set",
        active=request.lyrics is not None,
    )
    _require_supported(
        constraints,
        "instrumental",
        value=request.instrumental,
        active=request.instrumental,
    )
    _require_supported(
        constraints,
        "sample_rate",
        value=request.sample_rate,
        active=request.sample_rate is not None,
    )
    _require_supported(
        constraints,
        "tempo_bpm",
        value=request.tempo_bpm,
        active=request.tempo_bpm is not None,
    )
    _require_supported(
        constraints,
        "style",
        value=request.style,
        active=request.style is not None,
    )
    _require_supported(
        constraints,
        "seed",
        value=request.seed,
        active=request.seed is not None,
    )
    _check_format(request.format, constraints)

    updates: dict[str, Any] = {
        "prompt": _adapt_prompt(
            request.prompt,
            constraints,
            warnings,
            option="prompt",
        ),
        "duration_seconds": _adapt_duration(
            request.duration_seconds,
            constraints,
            warnings,
        ),
        "count": _adapt_count(request.count, constraints, warnings),
    }
    return AdaptationResult(
        request=request.model_copy(update=updates),
        warnings=tuple(warnings),
    )


def adapt_speech_request(
    request: SpeechRequest,
    constraints: MediaConstraints,
) -> AdaptationResult[SpeechRequest]:
    """Adapt a speech request to *constraints*, rejecting unsupported options."""
    warnings: list[AdaptationWarning] = []
    _check_model(request.model, constraints)
    _require_supported(
        constraints,
        "language",
        value=request.language,
        active=request.language is not None,
    )
    _require_supported(
        constraints,
        "sample_rate",
        value=request.sample_rate,
        active=request.sample_rate is not None,
    )
    _require_supported(
        constraints,
        "speed",
        value=request.speed,
        active=request.speed != 1.0,
    )
    _require_supported(
        constraints,
        "instructions",
        value=request.instructions,
        active=request.instructions is not None,
    )
    _require_supported(
        constraints,
        "pronunciation",
        value=len(request.pronunciation),
        active=bool(request.pronunciation),
    )
    _require_supported(
        constraints,
        "seed",
        value=request.seed,
        active=request.seed is not None,
    )
    _require_supported(
        constraints,
        "word_timestamps",
        value=request.word_timestamps,
        active=request.word_timestamps,
    )
    _check_format(request.format, constraints)

    updates: dict[str, Any] = {
        "text": _adapt_prompt(
            request.text,
            constraints,
            warnings,
            option="text",
        ),
    }
    return AdaptationResult(
        request=request.model_copy(update=updates),
        warnings=tuple(warnings),
    )


def adapt_media_request(
    request: MediaRequest,
    constraints: MediaConstraints,
) -> AdaptationResult[MediaRequest]:
    """Adapt the payload of a service-level media request."""
    kind = request.kind
    result: AdaptationResult[Any]
    adapted: MediaRequest
    if kind is MediaKind.IMAGE:
        assert request.image is not None
        result = adapt_image_request(request.image, constraints)
        adapted = MediaRequest(kind=kind, image=result.request)
    elif kind is MediaKind.VIDEO:
        assert request.video is not None
        result = adapt_video_request(request.video, constraints)
        adapted = MediaRequest(kind=kind, video=result.request)
    elif kind is MediaKind.MUSIC:
        assert request.music is not None
        result = adapt_music_request(request.music, constraints)
        adapted = MediaRequest(kind=kind, music=result.request)
    elif kind is MediaKind.SPEECH:
        assert request.speech is not None
        result = adapt_speech_request(request.speech, constraints)
        adapted = MediaRequest(kind=kind, speech=result.request)
    else:  # pragma: no cover - MediaKind is exhaustive
        raise TypeError(f"unsupported media kind: {kind!r}")

    return AdaptationResult(request=adapted, warnings=result.warnings)


def adapt_request(
    request: MediaRequest | ImageRequest | VideoRequest | MusicRequest | SpeechRequest,
    constraints: MediaConstraints,
) -> AdaptationResult[Any]:
    """Adapt any canonical media request, dispatching on its concrete type."""
    if isinstance(request, MediaRequest):
        return adapt_media_request(request, constraints)
    if isinstance(request, ImageRequest):
        return adapt_image_request(request, constraints)
    if isinstance(request, VideoRequest):
        return adapt_video_request(request, constraints)
    if isinstance(request, MusicRequest):
        return adapt_music_request(request, constraints)
    if isinstance(request, SpeechRequest):
        return adapt_speech_request(request, constraints)
    raise TypeError(f"unsupported media request: {type(request).__name__}")

"""Accessibility sidecars written next to audio and video outputs.

Every generated audio or video file can carry three small companion files:

* ``clip.mp4.alt.txt`` is a short plain-text description of the output;
* ``clip.mp4.srt`` is a SubRip subtitle track;
* ``clip.mp4.vtt`` is the same track as WebVTT.

The sidecars keep the full output name and append their suffix, exactly like
the cost sidecar, so ``voice.mp3`` and ``voice.wav`` never share a subtitle
file. Writing is pure file I/O: no provider call, no ffmpeg invocation and no
cost event. Whether a file is written is decided only by the
:class:`~aivia.core.config.sections.AccessibilityConfig` flags and by whether
there is content to write.

A remote output cannot receive local files, so
:func:`store_accessibility_sidecars` uploads the same three payloads through a
storage provider under the output key plus each suffix. The content rules are
identical; only the destination changes.

Timed words are the preferred subtitle source because they keep the provider's
own timing. A plain text fallback distributes the duration by word length when
the caller knows the media duration, so a service can emit subtitles even when
the provider returned no word timestamps.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from aivia.core.config.sections import AccessibilityConfig
from aivia.media.subtitles import (
    Subtitle,
    SubtitleCue,
    SubtitleWord,
)
from aivia.providers.protocols.storage import (
    StorageObject,
    StorageProvider,
    StoragePutRequest,
)

__all__ = [
    "ACCESSIBILITY_ALT_TEXT_SUFFIX",
    "ACCESSIBILITY_SRT_SUFFIX",
    "ACCESSIBILITY_VTT_SUFFIX",
    "ALT_TEXT_SUFFIX",
    "SRT_SUFFIX",
    "VTT_SUFFIX",
    "AccessibilityArtifacts",
    "AccessibilityConfig",
    "AccessibilityEmitter",
    "AccessibilityFiles",
    "AccessibilityOutput",
    "AccessibilityOutputs",
    "AccessibilityResult",
    "AccessibilitySidecarPaths",
    "AccessibilitySidecarWriter",
    "AccessibilitySidecars",
    "AccessibilityStoredSidecars",
    "AccessibilityWriter",
    "StoredAccessibilitySidecars",
    "accessibility_paths",
    "accessibility_sidecar_path",
    "accessibility_sidecar_paths",
    "accessibility_sidecars",
    "accessibility_storage_key",
    "alt_path",
    "alt_text_path",
    "alt_text_sidecar_path",
    "build_accessibility_sidecars",
    "create_accessibility_sidecars",
    "emit_accessibility",
    "emit_accessibility_sidecars",
    "srt_path",
    "srt_sidecar_path",
    "store_accessibility",
    "store_accessibility_sidecars",
    "stored_accessibility_sidecars",
    "subtitle_paths",
    "subtitle_sidecar_paths",
    "subtitle_srt_path",
    "subtitle_vtt_path",
    "vtt_path",
    "vtt_sidecar_path",
    "write_accessibility",
    "write_accessibility_sidecar_files",
    "write_accessibility_sidecars",
    "write_alt_text",
    "write_alt_text_sidecar",
    "write_srt",
    "write_subtitle_sidecars",
    "write_subtitles",
    "write_vtt",
]

ALT_TEXT_SUFFIX = ".alt.txt"
"""The suffix appended to an output path for its alt text."""

SRT_SUFFIX = ".srt"
"""The suffix appended to an output path for its SubRip track."""

VTT_SUFFIX = ".vtt"
"""The suffix appended to an output path for its WebVTT track."""

ACCESSIBILITY_ALT_TEXT_SUFFIX = ALT_TEXT_SUFFIX
ACCESSIBILITY_SRT_SUFFIX = SRT_SUFFIX
ACCESSIBILITY_VTT_SUFFIX = VTT_SUFFIX


@dataclass(frozen=True, slots=True, eq=False)
class AccessibilitySidecars:
    """The accessibility files written next to one output.

    A field is ``None`` when that kind of output is disabled or when there was
    no content to write. :attr:`paths` contains only the files actually
    written, in a stable alt-text, SRT, VTT order.
    """

    output: Path
    alt_text: Path | None = None
    srt: Path | None = None
    vtt: Path | None = None

    @property
    def paths(self) -> tuple[Path, ...]:
        """Every written sidecar path in alt-text, SRT, VTT order."""
        return tuple(
            path for path in (self.alt_text, self.srt, self.vtt) if path is not None
        )

    @property
    def written(self) -> tuple[Path, ...]:
        """Alias for :attr:`paths`."""
        return self.paths

    @property
    def alt(self) -> Path | None:
        """Alias for :attr:`alt_text`."""
        return self.alt_text

    @property
    def alt_text_path(self) -> Path | None:
        """Alias for :attr:`alt_text`."""
        return self.alt_text

    @property
    def srt_path(self) -> Path | None:
        """Alias for :attr:`srt`."""
        return self.srt

    @property
    def vtt_path(self) -> Path | None:
        """Alias for :attr:`vtt`."""
        return self.vtt

    @property
    def subtitle_paths(self) -> tuple[Path, ...]:
        """The written subtitle paths in SRT, VTT order."""
        return tuple(path for path in (self.srt, self.vtt) if path is not None)

    @property
    def subtitles(self) -> tuple[Path, ...]:
        """Alias for :attr:`subtitle_paths`."""
        return self.subtitle_paths

    @property
    def sidecar_paths(self) -> tuple[Path, ...]:
        """Alias for :attr:`paths`."""
        return self.paths

    @property
    def all_paths(self) -> tuple[Path, ...]:
        """Alias for :attr:`paths`."""
        return self.paths

    @property
    def alt_text_sidecar(self) -> Path | None:
        """Alias for :attr:`alt_text`."""
        return self.alt_text

    @property
    def subtitle_srt(self) -> Path | None:
        """Alias for :attr:`srt`."""
        return self.srt

    @property
    def subtitle_vtt(self) -> Path | None:
        """Alias for :attr:`vtt`."""
        return self.vtt

    def __getitem__(self, index: int) -> Path:
        return self.paths[index]

    def __contains__(self, value: object) -> bool:
        return value in self.paths

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AccessibilitySidecars):
            return (
                self.output,
                self.alt_text,
                self.srt,
                self.vtt,
            ) == (
                other.output,
                other.alt_text,
                other.srt,
                other.vtt,
            )
        if isinstance(other, (tuple, list)):
            return self.paths == tuple(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.output, self.alt_text, self.srt, self.vtt))

    @property
    def enabled(self) -> bool:
        """Whether any accessibility sidecar was written."""
        return bool(self.paths)

    def __bool__(self) -> bool:
        return self.enabled

    def __iter__(self) -> Iterator[Path]:
        return iter(self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the written files."""
        return {
            "output": str(self.output),
            "alt_text": str(self.alt_text) if self.alt_text is not None else None,
            "srt": str(self.srt) if self.srt is not None else None,
            "vtt": str(self.vtt) if self.vtt is not None else None,
            "paths": [str(path) for path in self.paths],
        }


AccessibilityOutput = AccessibilitySidecars
"""Alias for :class:`AccessibilitySidecars`."""

AccessibilityResult = AccessibilitySidecars
"""Alias for :class:`AccessibilitySidecars`."""

AccessibilityOutputs = AccessibilitySidecars
"""Alias for :class:`AccessibilitySidecars`."""

AccessibilitySidecarPaths = AccessibilitySidecars
"""Alias for :class:`AccessibilitySidecars`."""

AccessibilityFiles = AccessibilitySidecars
"""Alias for :class:`AccessibilitySidecars`."""

AccessibilityArtifacts = AccessibilitySidecars
"""Alias for :class:`AccessibilitySidecars`."""


@dataclass(frozen=True, slots=True, eq=False)
class AccessibilityStoredSidecars:
    """The accessibility files stored for one remote output.

    This is the storage-provider counterpart of :class:`AccessibilitySidecars`.
    Each field holds the :class:`~aivia.providers.protocols.storage.StorageObject`
    returned by the provider, or ``None`` when that kind of sidecar was disabled
    or had no content. :attr:`keys` and :attr:`paths` contain the stored keys in
    a stable alt-text, SRT, VTT order.
    """

    output: str
    alt_text: StorageObject | None = None
    srt: StorageObject | None = None
    vtt: StorageObject | None = None

    @property
    def objects(self) -> tuple[StorageObject, ...]:
        """Every stored sidecar object in alt-text, SRT, VTT order."""
        return tuple(
            obj for obj in (self.alt_text, self.srt, self.vtt) if obj is not None
        )

    @property
    def storage_objects(self) -> tuple[StorageObject, ...]:
        """Alias for :attr:`objects`."""
        return self.objects

    @property
    def keys(self) -> tuple[str, ...]:
        """Every stored sidecar key in alt-text, SRT, VTT order."""
        return tuple(obj.key for obj in self.objects)

    @property
    def paths(self) -> tuple[str, ...]:
        """Alias for :attr:`keys`."""
        return self.keys

    @property
    def written(self) -> tuple[str, ...]:
        """Alias for :attr:`keys`."""
        return self.keys

    @property
    def alt(self) -> StorageObject | None:
        """Alias for :attr:`alt_text`."""
        return self.alt_text

    @property
    def alt_text_path(self) -> str | None:
        """The stored alt-text key, when written."""
        return self.alt_text.key if self.alt_text is not None else None

    @property
    def srt_path(self) -> str | None:
        """The stored SRT key, when written."""
        return self.srt.key if self.srt is not None else None

    @property
    def vtt_path(self) -> str | None:
        """The stored VTT key, when written."""
        return self.vtt.key if self.vtt is not None else None

    @property
    def subtitle_paths(self) -> tuple[str, ...]:
        """The stored subtitle keys in SRT, VTT order."""
        return tuple(obj.key for obj in (self.srt, self.vtt) if obj is not None)

    @property
    def subtitles(self) -> tuple[str, ...]:
        """Alias for :attr:`subtitle_paths`."""
        return self.subtitle_paths

    @property
    def sidecar_paths(self) -> tuple[str, ...]:
        """Alias for :attr:`keys`."""
        return self.keys

    @property
    def all_paths(self) -> tuple[str, ...]:
        """Alias for :attr:`keys`."""
        return self.keys

    @property
    def enabled(self) -> bool:
        """Whether any accessibility sidecar was stored."""
        return bool(self.objects)

    def __bool__(self) -> bool:
        return self.enabled

    def __iter__(self) -> Iterator[StorageObject]:
        return iter(self.objects)

    def __len__(self) -> int:
        return len(self.objects)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AccessibilityStoredSidecars):
            return (
                self.output,
                self.alt_text,
                self.srt,
                self.vtt,
            ) == (
                other.output,
                other.alt_text,
                other.srt,
                other.vtt,
            )
        if isinstance(other, (tuple, list)):
            return self.keys == tuple(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.output, self.alt_text, self.srt, self.vtt))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the stored files."""
        return {
            "output": self.output,
            "alt_text": self.alt_text.key if self.alt_text is not None else None,
            "srt": self.srt.key if self.srt is not None else None,
            "vtt": self.vtt.key if self.vtt is not None else None,
            "keys": list(self.keys),
        }


StoredAccessibilitySidecars = AccessibilityStoredSidecars
"""Alias for :class:`AccessibilityStoredSidecars`."""

stored_accessibility_sidecars = AccessibilityStoredSidecars
"""Alias for :class:`AccessibilityStoredSidecars`."""


def _output_path(output: str | Path) -> Path:
    path = Path(output).expanduser()
    if not path.name:
        raise ValueError("output path must name a file")
    return path


def alt_text_path(output: str | Path) -> Path:
    """Return the alt-text sidecar path for *output*."""
    target = _output_path(output)
    return target.with_name(target.name + ALT_TEXT_SUFFIX)


def srt_path(output: str | Path) -> Path:
    """Return the SRT sidecar path for *output*."""
    target = _output_path(output)
    return target.with_name(target.name + SRT_SUFFIX)


def vtt_path(output: str | Path) -> Path:
    """Return the VTT sidecar path for *output*."""
    target = _output_path(output)
    return target.with_name(target.name + VTT_SUFFIX)


def subtitle_paths(output: str | Path) -> tuple[Path, Path]:
    """Return the SRT and VTT sidecar paths for *output*."""
    return srt_path(output), vtt_path(output)


def accessibility_paths(output: str | Path) -> tuple[Path, Path, Path]:
    """Return the alt-text, SRT and VTT sidecar paths for *output*."""
    return alt_text_path(output), srt_path(output), vtt_path(output)


def accessibility_sidecar_paths(output: str | Path) -> tuple[Path, Path, Path]:
    """Alias for :func:`accessibility_paths`."""
    return accessibility_paths(output)


def accessibility_sidecar_path(output: str | Path) -> tuple[Path, Path, Path]:
    """Alias for :func:`accessibility_paths`."""
    return accessibility_paths(output)


def alt_text_sidecar_path(output: str | Path) -> Path:
    """Alias for :func:`alt_text_path`."""
    return alt_text_path(output)


def srt_sidecar_path(output: str | Path) -> Path:
    """Alias for :func:`srt_path`."""
    return srt_path(output)


def vtt_sidecar_path(output: str | Path) -> Path:
    """Alias for :func:`vtt_path`."""
    return vtt_path(output)


def subtitle_sidecar_paths(output: str | Path) -> tuple[Path, Path]:
    """Alias for :func:`subtitle_paths`."""
    return subtitle_paths(output)


alt_path = alt_text_path
"""Alias for :func:`alt_text_path`."""

subtitle_srt_path = srt_path
"""Alias for :func:`srt_path`."""

subtitle_vtt_path = vtt_path
"""Alias for :func:`vtt_path`."""


def _config(value: Any) -> AccessibilityConfig:
    if value is None:
        return AccessibilityConfig()
    if isinstance(value, AccessibilityConfig):
        return value
    if isinstance(value, bool):
        return AccessibilityConfig(enabled=value)
    if isinstance(value, Mapping):
        nested = value.get("accessibility", value)
        if isinstance(nested, AccessibilityConfig):
            return nested
        if not isinstance(nested, Mapping):
            raise TypeError("accessibility config must be a mapping")
        return AccessibilityConfig.from_mapping(dict(nested))
    raise TypeError(
        "accessibility config must be an AccessibilityConfig, mapping, bool or None"
    )


def _positive_duration(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        raise TypeError("duration_seconds must be a number or None")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _words_from_text(text: str, duration: float | None) -> list[SubtitleWord]:
    words = text.split()
    if not words:
        return []
    total = duration if duration is not None else float(len(words))
    weights = [max(1, len(word)) for word in words]
    weight_sum = sum(weights)
    cursor = 0.0
    timed: list[SubtitleWord] = []
    for word, weight in zip(words, weights, strict=True):
        share = total * weight / weight_sum
        timed.append(
            SubtitleWord(
                word=word,
                start_seconds=cursor,
                end_seconds=cursor + share,
            )
        )
        cursor += share
    return timed


def _coerce_subtitle(
    value: Any,
    *,
    language: str = "",
    duration: float | None = None,
    style: Any = None,
) -> Subtitle | None:
    if value is None:
        return None
    if isinstance(value, Subtitle):
        return value
    if isinstance(value, str):
        words = _words_from_text(value, duration)
        if not words:
            return None
        return Subtitle.from_words(words, language=language, style=style)
    if isinstance(value, Mapping):
        return Subtitle.from_transcript(value, language=language, style=style)
    if hasattr(value, "all_words") or hasattr(value, "segments"):
        return Subtitle.from_transcript(value, language=language, style=style)
    if isinstance(value, Iterable):
        items = list(value)
        if not items:
            return None
        if all(isinstance(item, SubtitleCue) for item in items):
            return Subtitle(cues=items, language=language, style=style)
        return Subtitle.from_words(items, language=language, style=style)
    raise TypeError("subtitle source must be a Subtitle, words, mapping or text")


def _subtitle_from_sources(
    *,
    subtitle: Any,
    subtitles: Any,
    words: Any,
    transcript: Any,
    text: str | None,
    language: str,
    duration: float | None,
    style: Any,
) -> Subtitle | None:
    for source in (subtitle, subtitles, words, transcript, text):
        if source is None:
            continue
        if isinstance(source, str) and not source.strip():
            continue
        track = _coerce_subtitle(
            source,
            language=language,
            duration=duration,
            style=style,
        )
        if track is not None and track.cues:
            return track
    return None


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return path


def _write_alt_text_file(output: str | Path, alt_text: str) -> Path:
    if not isinstance(alt_text, str):
        raise TypeError("alt_text must be a string")
    text = alt_text.strip()
    if not text:
        raise ValueError("alt_text must not be empty")
    return _write_text(alt_text_path(output), text + "\n")


def write_alt_text(output: str | Path, alt_text: str) -> Path:
    """Write *alt_text* next to *output* and return the sidecar path."""
    return _write_alt_text_file(output, alt_text)


write_alt_text_sidecar = write_alt_text
"""Alias for :func:`write_alt_text`."""


def write_subtitle_sidecars(
    output: str | Path,
    subtitle: Any,
    *,
    language: str = "",
    duration_seconds: float | None = None,
    style: Any = None,
) -> tuple[Path, Path]:
    """Write SRT and VTT sidecars for *subtitle* and return both paths."""
    track = _coerce_subtitle(
        subtitle,
        language=language,
        duration=_positive_duration(duration_seconds),
        style=style,
    )
    if track is None or not track.cues:
        raise ValueError("subtitle must contain at least one cue")
    return (
        _write_text(srt_path(output), track.to_srt()),
        _write_text(vtt_path(output), track.to_vtt()),
    )


write_subtitles = write_subtitle_sidecars
"""Alias for :func:`write_subtitle_sidecars`."""


def write_srt(
    output: str | Path,
    subtitle: Any,
    *,
    language: str = "",
    duration_seconds: float | None = None,
    style: Any = None,
) -> Path:
    """Write only the SRT sidecar for *subtitle*."""
    track = _coerce_subtitle(
        subtitle,
        language=language,
        duration=_positive_duration(duration_seconds),
        style=style,
    )
    if track is None or not track.cues:
        raise ValueError("subtitle must contain at least one cue")
    return _write_text(srt_path(output), track.to_srt())


def write_vtt(
    output: str | Path,
    subtitle: Any,
    *,
    language: str = "",
    duration_seconds: float | None = None,
    style: Any = None,
) -> Path:
    """Write only the VTT sidecar for *subtitle*."""
    track = _coerce_subtitle(
        subtitle,
        language=language,
        duration=_positive_duration(duration_seconds),
        style=style,
    )
    if track is None or not track.cues:
        raise ValueError("subtitle must contain at least one cue")
    return _write_text(vtt_path(output), track.to_vtt())


@dataclass(frozen=True, slots=True)
class _RenderedAccessibility:
    output: Path
    alt_text: str | None = None
    srt: str | None = None
    vtt: str | None = None


def _render_accessibility(
    output: str | Path,
    config: Any,
    values: Mapping[str, Any],
) -> _RenderedAccessibility:
    selected_config = next(
        (
            value
            for value in (
                values.get("accessibility"),
                values.get("accessibility_config"),
                config,
            )
            if value is not None
        ),
        None,
    )
    selected = _config(selected_config)
    enabled = values.get("enabled")
    active = selected.enabled if enabled is None else bool(enabled)
    result = _RenderedAccessibility(output=_output_path(output))
    if not active:
        return result

    duration_seconds = values.get("duration_seconds")
    total = _positive_duration(
        duration_seconds if duration_seconds is not None else values.get("duration")
    )
    write_alt = values.get("write_alt_text")
    alt_enabled = selected.alt_text if write_alt is None else bool(write_alt)
    write_subtitle_flag = values.get("write_subtitles")
    subtitle_enabled = (
        selected.subtitles if write_subtitle_flag is None else bool(write_subtitle_flag)
    )
    subtitle = values.get("subtitle")
    subtitle_source = (
        subtitle
        if subtitle is not None
        else (values.get("track") or values.get("cues"))
    )
    track = _subtitle_from_sources(
        subtitle=subtitle_source,
        subtitles=values.get("subtitles"),
        words=values.get("words"),
        transcript=values.get("transcript"),
        text=values.get("text"),
        language=values.get("language") or "",
        duration=total,
        style=values.get("style"),
    )

    alt_value = next(
        (
            value
            for value in (
                values.get("alt_text"),
                values.get("alt"),
                values.get("description"),
            )
            if value is not None
        ),
        None,
    )
    if alt_value is None and track is not None:
        alt_value = track.text

    alt_content: str | None = None
    if alt_enabled and isinstance(alt_value, str) and alt_value.strip():
        alt_content = alt_value.strip() + "\n"

    srt_content: str | None = None
    vtt_content: str | None = None
    if subtitle_enabled and track is not None and track.cues:
        srt_content = track.to_srt()
        vtt_content = track.to_vtt()

    return _RenderedAccessibility(
        output=result.output,
        alt_text=alt_content,
        srt=srt_content,
        vtt=vtt_content,
    )


def write_accessibility_sidecars(
    output: str | Path,
    config: Any = None,
    *,
    accessibility: Any = None,
    accessibility_config: Any = None,
    alt_text: str | None = None,
    alt: str | None = None,
    description: str | None = None,
    subtitle: Any = None,
    subtitles: Any = None,
    track: Any = None,
    cues: Any = None,
    words: Any = None,
    transcript: Any = None,
    text: str | None = None,
    language: str = "",
    duration_seconds: float | None = None,
    duration: float | None = None,
    style: Any = None,
    enabled: bool | None = None,
    write_alt_text: bool | None = None,
    write_subtitles: bool | None = None,
    media_type: str | None = None,
    kind: str | None = None,
    **extra: Any,
) -> AccessibilitySidecars:
    """Write alt text, SRT and VTT sidecars next to *output*.

    *config* may be an :class:`AccessibilityConfig`, a root or section mapping,
    a bool, or ``None`` for the defaults. ``accessibility`` is an alias for it.
    When ``enabled`` is false nothing is written. The individual
    ``write_alt_text`` and ``write_subtitles`` flags override the config flags.

    Subtitles come from the first available source in this order: *subtitle*,
    *subtitles*, *words*, *transcript*, then *text*. A plain text source is
    timed over *duration_seconds* (or one second per word when no duration is
    known). Alt text falls back to *description*, then to the subtitle text, so
    an audio-only output still gets a useful description.
    """
    rendered = _render_accessibility(
        output,
        config,
        {
            "accessibility": accessibility,
            "accessibility_config": accessibility_config,
            "alt_text": alt_text,
            "alt": alt,
            "description": description,
            "subtitle": subtitle,
            "subtitles": subtitles,
            "track": track,
            "cues": cues,
            "words": words,
            "transcript": transcript,
            "text": text,
            "language": language,
            "duration_seconds": duration_seconds,
            "duration": duration,
            "style": style,
            "enabled": enabled,
            "write_alt_text": write_alt_text,
            "write_subtitles": write_subtitles,
        },
    )
    written_alt: Path | None = None
    if rendered.alt_text is not None:
        written_alt = _write_text(alt_text_path(rendered.output), rendered.alt_text)

    written_srt: Path | None = None
    written_vtt: Path | None = None
    if rendered.srt is not None and rendered.vtt is not None:
        written_srt = _write_text(srt_path(rendered.output), rendered.srt)
        written_vtt = _write_text(vtt_path(rendered.output), rendered.vtt)

    return AccessibilitySidecars(
        output=rendered.output,
        alt_text=written_alt,
        srt=written_srt,
        vtt=written_vtt,
    )


def accessibility_storage_key(
    output: str | Path,
    *,
    key: str | None = None,
) -> str:
    """Return the storage key an output's sidecars are stored under.

    An explicit *key* wins. A plain output name is used unchanged. A ``file``
    URL becomes its file name and an HTTP URL becomes its decoded path, so a
    sidecar key never contains a scheme, host or query string.
    """
    if key is not None:
        text = key.strip()
        if not text:
            raise ValueError("storage key must be non-empty")
        return text
    if isinstance(output, Path):
        if not output.name:
            raise ValueError("output must name a file")
        return output.name
    text = str(output).strip()
    if not text:
        raise ValueError("output must not be empty")
    parsed = urlsplit(text)
    if parsed.scheme.lower() == "file":
        return Path(unquote(parsed.path)).name
    if parsed.scheme.lower() in {"http", "https"}:
        path = unquote(parsed.path).lstrip("/")
        return path or text
    return text


async def store_accessibility_sidecars(
    output: str | Path,
    storage: StorageProvider,
    config: Any = None,
    *,
    key: str | None = None,
    **values: Any,
) -> AccessibilityStoredSidecars:
    """Store alt text, SRT and VTT sidecars through *storage*.

    This is the remote counterpart of :func:`write_accessibility_sidecars`. The
    content rules are identical; only the destination changes. Each sidecar is
    stored under the output key plus its usual suffix, so a remote
    ``clip.mp4`` gets ``clip.mp4.alt.txt``, ``clip.mp4.srt`` and
    ``clip.mp4.vtt`` as sibling objects. Re-running the same output overwrites
    its sidecars instead of failing on an existing key.
    """
    if not isinstance(storage, StorageProvider):
        raise TypeError("storage must be a StorageProvider")
    rendered = _render_accessibility(output, config, values)
    base_key = accessibility_storage_key(output, key=key)
    stored_alt: StorageObject | None = None
    stored_srt: StorageObject | None = None
    stored_vtt: StorageObject | None = None
    if rendered.alt_text is not None:
        stored_alt = await storage.put(
            StoragePutRequest(
                data=rendered.alt_text.encode("utf-8"),
                key=f"{base_key}{ALT_TEXT_SUFFIX}",
                content_type="text/plain; charset=utf-8",
                overwrite=True,
                metadata={"kind": "accessibility", "sidecar": "alt_text"},
            )
        )
    if rendered.srt is not None:
        stored_srt = await storage.put(
            StoragePutRequest(
                data=rendered.srt.encode("utf-8"),
                key=f"{base_key}{SRT_SUFFIX}",
                content_type="application/x-subrip; charset=utf-8",
                overwrite=True,
                metadata={"kind": "accessibility", "sidecar": "srt"},
            )
        )
    if rendered.vtt is not None:
        stored_vtt = await storage.put(
            StoragePutRequest(
                data=rendered.vtt.encode("utf-8"),
                key=f"{base_key}{VTT_SUFFIX}",
                content_type="text/vtt; charset=utf-8",
                overwrite=True,
                metadata={"kind": "accessibility", "sidecar": "vtt"},
            )
        )
    return AccessibilityStoredSidecars(
        output=str(output),
        alt_text=stored_alt,
        srt=stored_srt,
        vtt=stored_vtt,
    )


store_accessibility = store_accessibility_sidecars
"""Alias for :func:`store_accessibility_sidecars`."""


emit_accessibility_sidecars = write_accessibility_sidecars
"""Alias for :func:`write_accessibility_sidecars`."""

write_accessibility_sidecar_files = write_accessibility_sidecars
"""Alias for :func:`write_accessibility_sidecars`."""

write_accessibility = write_accessibility_sidecars
"""Alias for :func:`write_accessibility_sidecars`."""

emit_accessibility = write_accessibility_sidecars
"""Alias for :func:`write_accessibility_sidecars`."""

build_accessibility_sidecars = write_accessibility_sidecars
"""Alias for :func:`write_accessibility_sidecars`."""

create_accessibility_sidecars = write_accessibility_sidecars
"""Alias for :func:`write_accessibility_sidecars`."""

accessibility_sidecars = write_accessibility_sidecars
"""Alias for :func:`write_accessibility_sidecars`."""


class AccessibilityEmitter:
    """Write accessibility sidecars with one reusable configuration."""

    def __init__(self, config: Any = None) -> None:
        self._config = _config(config)

    @property
    def config(self) -> AccessibilityConfig:
        """The accessibility configuration this emitter applies."""
        return self._config

    @property
    def enabled(self) -> bool:
        """Whether accessibility output is enabled."""
        return self._config.enabled

    def write(self, output: str | Path, **values: Any) -> AccessibilitySidecars:
        """Write sidecars for *output*, using this emitter's config."""
        values.setdefault("config", self._config)
        return write_accessibility_sidecars(output, **values)

    def emit(self, output: str | Path, **values: Any) -> AccessibilitySidecars:
        """Alias for :meth:`write`."""
        return self.write(output, **values)

    def __call__(self, output: str | Path, **values: Any) -> AccessibilitySidecars:
        return self.write(output, **values)


AccessibilityWriter = AccessibilityEmitter
"""Alias for :class:`AccessibilityEmitter`."""

AccessibilitySidecarWriter = AccessibilityEmitter
"""Alias for :class:`AccessibilityEmitter`."""

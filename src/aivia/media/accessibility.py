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

from aivia.core.config.sections import AccessibilityConfig
from aivia.media.subtitles import (
    Subtitle,
    SubtitleCue,
    SubtitleWord,
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
    "AccessibilityWriter",
    "accessibility_paths",
    "accessibility_sidecar_path",
    "accessibility_sidecar_paths",
    "accessibility_sidecars",
    "alt_path",
    "alt_text_path",
    "alt_text_sidecar_path",
    "build_accessibility_sidecars",
    "create_accessibility_sidecars",
    "emit_accessibility",
    "emit_accessibility_sidecars",
    "srt_path",
    "srt_sidecar_path",
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
    selected_config = next(
        (
            value
            for value in (accessibility, accessibility_config, config)
            if value is not None
        ),
        None,
    )
    selected = _config(selected_config)
    active = selected.enabled if enabled is None else bool(enabled)
    result = AccessibilitySidecars(output=_output_path(output))
    if not active:
        return result

    total = _positive_duration(
        duration_seconds if duration_seconds is not None else duration
    )
    alt_enabled = selected.alt_text if write_alt_text is None else bool(write_alt_text)
    subtitle_enabled = (
        selected.subtitles if write_subtitles is None else bool(write_subtitles)
    )
    subtitle_source = subtitle if subtitle is not None else (track or cues)
    track = _subtitle_from_sources(
        subtitle=subtitle_source,
        subtitles=subtitles,
        words=words,
        transcript=transcript,
        text=text,
        language=language,
        duration=total,
        style=style,
    )

    alt_value = next(
        (value for value in (alt_text, alt, description) if value is not None),
        None,
    )
    if alt_value is None and track is not None:
        alt_value = track.text
    written_alt: Path | None = None
    if alt_enabled and isinstance(alt_value, str) and alt_value.strip():
        written_alt = _write_alt_text_file(output, alt_value)

    written_srt: Path | None = None
    written_vtt: Path | None = None
    if subtitle_enabled and track is not None and track.cues:
        written_srt, written_vtt = write_subtitle_sidecars(output, track)

    return AccessibilitySidecars(
        output=result.output,
        alt_text=written_alt,
        srt=written_srt,
        vtt=written_vtt,
    )


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

"""Translate a style spec into style bible entries and prompt suffixes.

A style spec is the structured, editable description learned from references.
The writer and the media prompt builder do not consume that structure directly;
they read a style bible and short prompt suffixes. This module is the
deterministic bridge between the two: the same spec always produces the same
entries and the same suffixes, with no model or provider involved.

The translation deliberately ignores the parts of a reference that must never
be copied. A :class:`~aivia.reference.schema.ReferenceAnalysis` is accepted as
a :class:`~aivia.reference.schema.StyleSpec`, but its subject, detected text,
per-scene text and reproducing prompt never reach the output. Only descriptive
style fields are used.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aivia.core.errors import ConfigError
from aivia.reference.schema import (
    AspectSpec,
    CaptionStyle,
    MusicMoodSpec,
    PacingSpec,
    PaletteSpec,
    StructureSpec,
    StyleSpec,
    TextCase,
    TypographySpec,
)
from aivia.writing.style_bible import StyleBible, StyleBibleEntries, StyleBibleEntry

__all__ = [
    "PromptSuffixes",
    "StylePromptError",
    "StylePromptTranslation",
    "StylePrompts",
    "build_style_prompts",
    "prompt_suffix_from_spec",
    "prompt_suffixes_from_spec",
    "spec_prompt_suffixes",
    "spec_to_style_bible",
    "style_bible_entries_from_spec",
    "style_bible_from_spec",
    "style_prompts",
    "style_prompts_from_spec",
    "to_style_bible",
    "translate_style_spec",
]

_WHITESPACE = re.compile(r"\s+", re.UNICODE)
_AVOID_KEYS = ("avoid", "negative", "negatives", "negative_prompt", "exclude")


class StylePromptError(ConfigError):
    """A style spec cannot be translated into usable prompt guidance."""

    code = "style_prompt"


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return _WHITESPACE.sub(" ", str(value)).strip()


def _sentence(value: str) -> str:
    text = _clean(value)
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?;:":
        text = f"{text}."
    return text


def _lower_first(value: str) -> str:
    text = _clean(value)
    if not text:
        return ""
    return text[0].lower() + text[1:]


def _with_article(value: str) -> str:
    text = _clean(value)
    if not text:
        return ""
    article = "an" if text[0].lower() in "aeiou" else "a"
    return f"{article} {text}"


def _number(value: float | int | None) -> str:
    if value is None:
        return ""
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def _percent(value: float | None) -> str:
    if value is None:
        return ""
    return f"{round(value * 100):d}%"


def _join(values: Iterable[Any], *, sep: str = ", ") -> str:
    seen: set[str] = set()
    parts: list[str] = []
    for value in values:
        text = _clean(value)
        if text and text not in seen:
            seen.add(text)
            parts.append(text)
    return sep.join(parts)


def _labels(values: Iterable[Any]) -> list[str]:
    return [str(value).replace("_", "-") for value in values]


def _energy_label(value: float | None) -> str:
    if value is None:
        return ""
    if value < 0.34:
        return "low"
    if value < 0.67:
        return "medium"
    return "high"


def _caption_label(value: CaptionStyle) -> str:
    return value.value.replace("_", "-")


def _case_label(value: TextCase) -> str:
    return {
        TextCase.UPPER: "uppercase",
        TextCase.LOWER: "lowercase",
        TextCase.TITLE: "title case",
        TextCase.SENTENCE: "sentence case",
    }.get(value, value.value)


def _coerce_spec(value: Any) -> StyleSpec:
    if isinstance(value, StyleSpec):
        return value
    if isinstance(value, Mapping):
        return StyleSpec.model_validate(dict(value))
    if hasattr(value, "analysis"):
        return _coerce_spec(value.analysis)
    return StyleSpec.model_validate(value)


def _subject(value: StyleSpec) -> str:
    subject = getattr(value, "subject", "")
    return _clean(subject)


def _without_subject(text: str, subject: str) -> str:
    """Remove any mention of the original subject from generated text."""
    cleaned = _clean(text)
    if not cleaned or not subject:
        return cleaned
    cleaned = re.sub(
        rf"\b{re.escape(subject)}\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"^[\s,.;:!?-]+", "", cleaned)
    cleaned = re.sub(r"[\s,;:]+$", "", cleaned)
    return _clean(cleaned)


def _scenes_label(count: int | None) -> str:
    if not count:
        return ""
    return f"{count} scene" if count == 1 else f"{count} scenes"


def _structure_text(structure: StructureSpec) -> str:
    parts: list[str] = []
    if structure.narrative_arc:
        arc = _lower_first(structure.narrative_arc)
        parts.append(f"follow a {arc} narrative arc")
    if structure.hook:
        parts.append(f'open with the hook "{structure.hook}"')
    if structure.beats:
        parts.append("build the piece as " + " -> ".join(structure.beats))
    elif structure.scene_count:
        parts.append(f"use {_scenes_label(structure.scene_count)}")
    if structure.shot_types:
        shots = _join(_labels(structure.shot_types))
        parts.append(f"use {shots} shots")
    if structure.transitions:
        parts.append("transition with " + _join(structure.transitions))
    if structure.total_duration_seconds is not None:
        seconds = _number(structure.total_duration_seconds)
        parts.append(f"keep the total length near {seconds} seconds")
    return _sentence("; ".join(parts))


def _structure_suffix(structure: StructureSpec) -> str:
    parts: list[str] = []
    if structure.beats:
        parts.append("one idea per beat")
    if structure.scene_count:
        parts.append(_scenes_label(structure.scene_count))
    if structure.shot_types:
        parts.append(_join(_labels(structure.shot_types)) + " framing")
    if structure.transitions:
        parts.append(_join(structure.transitions) + " transitions")
    if structure.total_duration_seconds is not None:
        parts.append(f"{_number(structure.total_duration_seconds)}s total")
    return _sentence(_join(parts))


def _pacing_text(pacing: PacingSpec) -> str:
    parts: list[str] = []
    if pacing.style:
        parts.append(f"keep the pace {_lower_first(pacing.style)}")
    if pacing.cuts_per_minute is not None:
        parts.append(f"aim for about {_number(pacing.cuts_per_minute)} cuts per minute")
    if pacing.average_shot_seconds is not None:
        seconds = _number(pacing.average_shot_seconds)
        parts.append(f"hold each shot for about {seconds} seconds")
    if pacing.min_shot_seconds is not None and pacing.max_shot_seconds is not None:
        shortest = _number(pacing.min_shot_seconds)
        longest = _number(pacing.max_shot_seconds)
        parts.append(f"keep shots between {shortest} and {longest} seconds")
    elif pacing.min_shot_seconds is not None:
        parts.append(f"keep shots at least {_number(pacing.min_shot_seconds)} seconds")
    elif pacing.max_shot_seconds is not None:
        parts.append(f"keep shots at most {_number(pacing.max_shot_seconds)} seconds")
    if pacing.words_per_second is not None:
        rate = _number(pacing.words_per_second)
        parts.append(f"speak at about {rate} words per second")
    if pacing.energy is not None:
        parts.append(f"keep the energy {_energy_label(pacing.energy)}")
    return _sentence("; ".join(parts))


def _pacing_suffix(pacing: PacingSpec) -> str:
    parts: list[str] = []
    if pacing.style:
        parts.append(f"{_lower_first(pacing.style)} pacing")
    if pacing.cuts_per_minute is not None:
        parts.append(f"{_number(pacing.cuts_per_minute)} cuts per minute")
    if pacing.average_shot_seconds is not None:
        seconds = _number(pacing.average_shot_seconds)
        parts.append(f"about {seconds} second shots")
    if pacing.energy is not None:
        parts.append(f"{_energy_label(pacing.energy)} energy")
    return _sentence(_join(parts))


def _palette_colors(palette: PaletteSpec) -> list[str]:
    values: list[str] = list(palette.colors)
    for colour in palette.roles.values():
        if colour not in values:
            values.append(colour)
    return values


def _palette_text(palette: PaletteSpec) -> str:
    parts: list[str] = []
    colors = _palette_colors(palette)
    if colors:
        parts.append("use the palette " + _join(colors))
    if palette.scheme:
        parts.append(_with_article(_lower_first(palette.scheme)) + " colour scheme")
    if palette.temperature is not None:
        parts.append(f"keep a {palette.temperature.value} tone")
    if palette.saturation is not None and palette.contrast is not None:
        saturation = _percent(palette.saturation)
        contrast = _percent(palette.contrast)
        parts.append(f"at {saturation} saturation and {contrast} contrast")
    elif palette.saturation is not None:
        parts.append(f"at {_percent(palette.saturation)} saturation")
    elif palette.contrast is not None:
        parts.append(f"at {_percent(palette.contrast)} contrast")
    return _sentence("; ".join(parts))


def _palette_suffix(palette: PaletteSpec) -> str:
    parts: list[str] = []
    colors = _palette_colors(palette)
    if colors:
        parts.append("palette " + _join(colors))
    if palette.scheme:
        parts.append(f"{_lower_first(palette.scheme)} palette")
    if palette.temperature is not None:
        parts.append(f"{palette.temperature.value} tone")
    if palette.saturation is not None:
        parts.append(f"{_percent(palette.saturation)} saturation")
    if palette.contrast is not None:
        parts.append(f"{_percent(palette.contrast)} contrast")
    return _sentence(_join(parts))


def _typography_text(typography: TypographySpec) -> str:
    parts: list[str] = []
    fonts = typography.typefaces
    if fonts:
        parts.append("set text in " + _join(fonts))
    if typography.font_weight is not None:
        parts.append(f"at weight {typography.font_weight}")
    if typography.font_style:
        parts.append(f"in {_lower_first(typography.font_style)} style")
    sizes: list[str] = []
    if typography.font_size is not None:
        sizes.append(f"{_number(typography.font_size)}px text")
    if typography.heading_size is not None:
        sizes.append(f"{_number(typography.heading_size)}px headings")
    if typography.body_size is not None:
        sizes.append(f"{_number(typography.body_size)}px body")
    if sizes:
        parts.append("with " + _join(sizes))
    if typography.align.value != "center":
        parts.append(f"{typography.align.value}-aligned")
    if typography.case is not TextCase.NONE:
        parts.append(f"in {_case_label(typography.case)}")
    if typography.line_height is not None:
        parts.append(f"at {_number(typography.line_height)} line height")
    if typography.letter_spacing is not None:
        spacing = _number(typography.letter_spacing)
        parts.append(f"with {spacing}px letter spacing")
    if typography.color:
        parts.append(f"in {typography.color}")
    if typography.placement:
        parts.append(f"placed in the {_lower_first(typography.placement)}")
    if typography.caption_style is not CaptionStyle.NONE:
        label = _caption_label(typography.caption_style)
        parts.append(f"with {label} captions")
    return _sentence("; ".join(parts))


def _typography_suffix(typography: TypographySpec) -> str:
    parts: list[str] = []
    fonts = typography.typefaces
    if fonts:
        parts.append(f"{fonts[0]} typography")
    if typography.font_weight is not None:
        parts.append(f"weight {typography.font_weight}")
    if typography.case is not TextCase.NONE:
        parts.append(_case_label(typography.case))
    if typography.color:
        parts.append(f"text {typography.color}")
    if typography.placement:
        parts.append(f"{_lower_first(typography.placement)} text")
    if typography.caption_style is not CaptionStyle.NONE:
        label = _caption_label(typography.caption_style)
        parts.append(f"{label} captions")
    return _sentence(_join(parts))


def _music_names(value: str, values: Sequence[str]) -> list[str]:
    names = list(values)
    if value and value not in names:
        names.insert(0, value)
    return names


def _music_text(music: MusicMoodSpec) -> str:
    parts: list[str] = []
    genres = _music_names(music.genre, music.genres)
    moods = _music_names(music.mood, music.moods)
    if genres:
        parts.append("use " + _join(genres) + " music")
    if moods:
        parts.append(f"with {_with_article(_join(moods))} mood")
    if music.bpm is not None:
        parts.append(f"around {_number(music.bpm)} BPM")
    key_scale = _join([music.key, music.scale], sep=" ")
    if key_scale:
        parts.append(f"in {key_scale}")
    if music.instruments:
        parts.append("featuring " + _join(music.instruments))
    if music.vocals:
        if music.is_instrumental:
            parts.append("instrumental with no vocals")
        else:
            parts.append(f"with {_lower_first(music.vocals)} vocals")
    if music.energy is not None:
        parts.append(f"at {_energy_label(music.energy)} energy")
    if music.valence is not None:
        parts.append(f"and {_energy_label(music.valence)} valence")
    if music.loudness_lufs is not None:
        parts.append(f"around {_number(music.loudness_lufs)} LUFS")
    return _sentence("; ".join(parts))


def _music_suffix(music: MusicMoodSpec) -> str:
    parts: list[str] = []
    genres = _music_names(music.genre, music.genres)
    moods = _music_names(music.mood, music.moods)
    if genres:
        parts.append(_join(genres))
    if moods:
        parts.append(_join(moods) + " mood")
    if music.bpm is not None:
        parts.append(f"{_number(music.bpm)} BPM")
    if music.instruments:
        parts.append(_join(music.instruments))
    if music.vocals:
        parts.append(
            "instrumental" if music.is_instrumental else _lower_first(music.vocals)
        )
    return _sentence(_join(parts))


def _aspect_text(aspect: AspectSpec) -> str:
    parts: list[str] = []
    if aspect.ratio:
        parts.append(f"frame in a {aspect.ratio} ratio")
    if aspect.resolution:
        parts.append(f"at {aspect.resolution}")
    if aspect.orientation is not None:
        parts.append(f"with a {aspect.orientation.value} orientation")
    return _sentence(" ".join(parts))


def _aspect_suffix(aspect: AspectSpec) -> str:
    parts: list[str] = []
    if aspect.orientation is not None:
        parts.append(aspect.orientation.value)
    if aspect.ratio:
        parts.append(aspect.ratio)
    if aspect.resolution:
        parts.append(aspect.resolution)
    if not parts:
        return ""
    return _sentence(_join(parts, sep=" ") + " frame")


def _has_structure(structure: StructureSpec) -> bool:
    return bool(
        structure.scenes
        or structure.scene_count
        or structure.hook
        or structure.beats
        or structure.shot_types
        or structure.transitions
        or structure.narrative_arc
        or structure.total_duration_seconds is not None
    )


def _has_pacing(pacing: PacingSpec) -> bool:
    return any(
        value is not None
        for value in (
            pacing.cuts_per_minute,
            pacing.average_shot_seconds,
            pacing.min_shot_seconds,
            pacing.max_shot_seconds,
            pacing.words_per_second,
            pacing.energy,
        )
    ) or bool(pacing.style)


def _has_palette(palette: PaletteSpec) -> bool:
    return bool(
        palette.colors
        or palette.roles
        or palette.temperature is not None
        or palette.saturation is not None
        or palette.contrast is not None
        or palette.scheme
    )


def _has_typography(typography: TypographySpec) -> bool:
    return bool(
        typography.typefaces
        or typography.font_weight is not None
        or typography.font_style
        or typography.font_size is not None
        or typography.heading_size is not None
        or typography.body_size is not None
        or typography.align.value != "center"
        or typography.case is not TextCase.NONE
        or typography.line_height is not None
        or typography.letter_spacing is not None
        or typography.color
        or typography.placement
        or typography.caption_style is not CaptionStyle.NONE
    )


def _has_music(music: MusicMoodSpec) -> bool:
    return bool(
        music.genre
        or music.genres
        or music.mood
        or music.moods
        or music.bpm is not None
        or music.key
        or music.scale
        or music.instruments
        or music.vocals
        or music.energy is not None
        or music.valence is not None
        or music.loudness_lufs is not None
    )


def _has_aspect(aspect: AspectSpec) -> bool:
    return bool(aspect.ratio or aspect.resolution or aspect.orientation is not None)


def _avoid_items(spec: StyleSpec) -> list[str]:
    items: list[str] = []
    for key in _AVOID_KEYS:
        value = spec.metadata.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            items.extend(part.strip() for part in value.split(",") if part.strip())
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            items.extend(_clean(item) for item in value if _clean(item))
    return list(dict.fromkeys(items))


def _entry(
    key: str,
    title: str,
    text: str,
    prompt_suffix: str = "",
) -> StyleBibleEntry:
    return StyleBibleEntry(
        key=key,
        title=title,
        text=text or prompt_suffix,
        prompt_suffix=prompt_suffix,
    )


@dataclass(frozen=True)
class PromptSuffixes:
    """Short phrases a media prompt can append, grouped by prompt kind.

    The groups are independent because an image prompt should not repeat the
    music mood and an audio prompt should not repeat the palette. ``text``
    joins every group in a stable order for callers that need one suffix.
    """

    visual: str = ""
    motion: str = ""
    audio: str = ""
    writing: str = ""
    negative: str = ""

    def for_kind(self, kind: str) -> str:
        """Return the suffix for an image, video, audio or writing prompt."""
        key = _clean(kind).casefold().replace("-", "_").replace(" ", "_")
        aliases = {
            "image": "visual",
            "images": "visual",
            "still": "visual",
            "visuals": "visual",
            "video": "motion",
            "videos": "motion",
            "motion": "motion",
            "camera": "motion",
            "audio": "audio",
            "music": "audio",
            "sound": "audio",
            "text": "writing",
            "writing": "writing",
            "script": "writing",
            "voice": "writing",
            "avoid": "negative",
            "negative": "negative",
        }
        field = aliases.get(key, key)
        if field not in {"visual", "motion", "audio", "writing", "negative"}:
            raise KeyError(f"unknown prompt suffix kind: {kind!r}")
        return str(getattr(self, field))

    @property
    def values(self) -> tuple[str, ...]:
        """Every non-empty suffix, visual first."""
        return tuple(
            value
            for value in (
                self.visual,
                self.motion,
                self.audio,
                self.writing,
                self.negative,
            )
            if value
        )

    @property
    def all(self) -> tuple[str, ...]:
        """Alias for :attr:`values`."""
        return self.values

    @property
    def text(self) -> str:
        """Every suffix joined into one prompt fragment."""
        return _join(self.values, sep=" ")

    def as_dict(self) -> dict[str, str]:
        """Return the non-empty groups as a plain mapping."""
        return {
            name: value
            for name, value in (
                ("visual", self.visual),
                ("motion", self.motion),
                ("audio", self.audio),
                ("writing", self.writing),
                ("negative", self.negative),
            )
            if value
        }

    def __bool__(self) -> bool:
        return bool(self.values)

    def __getitem__(self, kind: str) -> str:
        return self.for_kind(kind)


@dataclass(frozen=True)
class StylePrompts:
    """The style bible entries and prompt suffixes translated from one spec."""

    spec: StyleSpec
    entries: tuple[StyleBibleEntry, ...] = ()
    suffixes: PromptSuffixes = PromptSuffixes()
    name: str = ""
    description: str = ""

    @property
    def is_empty(self) -> bool:
        """Whether the spec produced no entries and no prompt suffix."""
        return not self.entries and not self.suffixes

    @property
    def prompt_suffixes(self) -> dict[str, str]:
        """The non-empty prompt suffix groups keyed by kind."""
        return self.suffixes.as_dict()

    @property
    def prompt_suffix(self) -> str:
        """Every prompt suffix joined into one string."""
        return self.suffixes.text

    @property
    def suffix(self) -> str:
        """Alias for :attr:`prompt_suffix`."""
        return self.prompt_suffix

    def suffix_for(self, kind: str) -> str:
        """Return the prompt suffix for one kind, such as ``image``."""
        return self.suffixes.for_kind(kind)

    def to_bible(self, *, name: str | None = None) -> StyleBible:
        """Build a validated style bible from the translated entries."""
        if not self.entries:
            raise StylePromptError(
                "style spec has no style to translate",
                details={"name": self.name or self.spec.name},
            )
        return StyleBible(
            name=_clean(name) or self.name or self.spec.name or "style-spec",
            description=self.description,
            entries=StyleBibleEntries(self.entries),
        )

    @property
    def bible(self) -> StyleBible:
        """The translated style bible, built on demand."""
        return self.to_bible()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this translation."""
        return {
            "name": self.name or self.spec.name,
            "description": self.description,
            "entries": [entry.to_dict() for entry in self.entries],
            "prompt_suffixes": self.suffixes.as_dict(),
        }

    def __iter__(self) -> Iterator[StyleBibleEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


StylePromptTranslation = StylePrompts
"""Alias for :class:`StylePrompts`."""


def translate_style_spec(
    spec: StyleSpec | Mapping[str, Any] | Any,
    *,
    name: str | None = None,
    description: str | None = None,
) -> StylePrompts:
    """Translate *spec* into style bible entries and prompt suffixes.

    A mapping or an existing :class:`StyleSpec` is accepted. Every output is
    derived only from descriptive style fields; a reference subject, detected
    text, scene text and reproducing prompt are never copied.
    """
    value = _coerce_spec(spec)
    subject = _subject(value)
    entries: list[StyleBibleEntry] = []
    visual: list[str] = []
    writing: list[str] = []

    style_text = _clean(value.description) or _clean(value.prompt_suffix)
    style_suffix = _without_subject(value.prompt_suffix, subject)
    if style_text or style_suffix:
        text = _without_subject(style_text, subject) or style_suffix
        entries.append(_entry("style", "Style", text, style_suffix))
        if style_suffix:
            visual.append(style_suffix)

    if _has_structure(value.structure):
        text = _without_subject(_structure_text(value.structure), subject)
        suffix = _without_subject(_structure_suffix(value.structure), subject)
        entries.append(_entry("structure", "Structure", text, suffix))
        if suffix:
            writing.append(suffix)

    if _has_pacing(value.pacing):
        text = _without_subject(_pacing_text(value.pacing), subject)
        suffix = _without_subject(_pacing_suffix(value.pacing), subject)
        entries.append(_entry("pacing", "Pacing", text, suffix))
        if suffix:
            visual.append(suffix)
            writing.append(suffix)

    if _has_palette(value.palette):
        text = _without_subject(_palette_text(value.palette), subject)
        suffix = _without_subject(_palette_suffix(value.palette), subject)
        entries.append(_entry("palette", "Palette", text, suffix))
        if suffix:
            visual.append(suffix)

    if _has_typography(value.typography):
        text = _without_subject(_typography_text(value.typography), subject)
        suffix = _without_subject(_typography_suffix(value.typography), subject)
        entries.append(_entry("typography", "Typography", text, suffix))
        if suffix:
            visual.append(suffix)

    if _has_music(value.music):
        text = _without_subject(_music_text(value.music), subject)
        suffix = _without_subject(_music_suffix(value.music), subject)
        entries.append(_entry("music", "Music", text, suffix))

    if _has_aspect(value.aspect):
        text = _without_subject(_aspect_text(value.aspect), subject)
        suffix = _without_subject(_aspect_suffix(value.aspect), subject)
        entries.append(_entry("aspect", "Aspect", text, suffix))
        if suffix:
            visual.append(suffix)

    avoid = _avoid_items(value)
    negative = ""
    if avoid:
        text = _sentence("avoid " + _join(avoid))
        negative = _sentence("no " + _join(avoid))
        entries.append(_entry("avoid", "Avoid", text, negative))

    suffixes = PromptSuffixes(
        visual=_sentence(_join(visual, sep=" ")),
        motion=_without_subject(_pacing_suffix(value.pacing), subject),
        audio=_without_subject(_music_suffix(value.music), subject),
        writing=_sentence(_join(writing, sep=" ")),
        negative=negative,
    )
    resolved_description = _without_subject(
        _clean(description) or value.description,
        subject,
    )
    return StylePrompts(
        spec=value,
        entries=tuple(entries),
        suffixes=suffixes,
        name=_clean(name) or value.name,
        description=resolved_description,
    )


style_prompts_from_spec = translate_style_spec
"""Alias for :func:`translate_style_spec`."""

build_style_prompts = translate_style_spec
"""Alias for :func:`translate_style_spec`."""

style_prompts = translate_style_spec
"""Alias for :func:`translate_style_spec`."""


def style_bible_entries_from_spec(
    spec: StyleSpec | Mapping[str, Any] | Any,
) -> tuple[StyleBibleEntry, ...]:
    """Return only the style bible entries translated from *spec*."""
    return translate_style_spec(spec).entries


def prompt_suffixes_from_spec(
    spec: StyleSpec | Mapping[str, Any] | Any,
) -> PromptSuffixes:
    """Return only the prompt suffix groups translated from *spec*."""
    return translate_style_spec(spec).suffixes


spec_prompt_suffixes = prompt_suffixes_from_spec
"""Alias for :func:`prompt_suffixes_from_spec`."""


def prompt_suffix_from_spec(spec: StyleSpec | Mapping[str, Any] | Any) -> str:
    """Return every prompt suffix translated from *spec* joined together."""
    return translate_style_spec(spec).prompt_suffix


def style_bible_from_spec(
    spec: StyleSpec | Mapping[str, Any] | Any,
    *,
    name: str | None = None,
) -> StyleBible:
    """Translate *spec* into a validated style bible."""
    return translate_style_spec(spec, name=name).to_bible(name=name)


spec_to_style_bible = style_bible_from_spec
"""Alias for :func:`style_bible_from_spec`."""

to_style_bible = style_bible_from_spec
"""Alias for :func:`style_bible_from_spec`."""

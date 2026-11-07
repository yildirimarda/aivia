"""Convert a style spec into a format recipe, style bible and preset.

A style spec is the structured style learned from references. A format recipe
selects the pipeline stages and how the output is handled, a style bible is the
reusable guidance shared by the writer and media prompts, and a preset is a
named bundle of configuration overrides. All three are derived deterministically
from the spec, with no model or provider involved.

Only the sections a spec actually learned are turned into stages or overrides,
so a partial spec still produces a usable, validated result.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from aivia.core.config.formats import FormatRecipeConfig, OutputType, StageConfig
from aivia.core.config.sections import PresetConfig
from aivia.reference.schema import CaptionStyle, StyleSpec
from aivia.reference.style_prompts import (
    StylePromptError,
    style_bible_from_spec,
    translate_style_spec,
)
from aivia.writing.style_bible import StyleBible

__all__ = [
    "FormatConversion",
    "convert_style_spec",
    "style_spec_to_format_recipe",
    "style_spec_to_preset",
    "style_spec_to_style_bible",
]

_STAGE_WRITE = "write"
_STAGE_VISUALS = "visuals"
_STAGE_VOICE = "voice"
_STAGE_MUSIC = "music"
_STAGE_SUBTITLES = "subtitles"

# Map each learned spec section to its recipe stage, in execution order.
_SECTION_STAGES = (
    ("structure", _STAGE_WRITE),
    ("palette", _STAGE_VISUALS),
    ("typography", _STAGE_SUBTITLES),
    ("music", _STAGE_MUSIC),
    ("pacing", _STAGE_VOICE),
)

_STAGE_CAPABILITIES = {
    _STAGE_WRITE: "writing",
    _STAGE_VISUALS: "image",
    _STAGE_VOICE: "tts",
    _STAGE_MUSIC: "music",
    _STAGE_SUBTITLES: "captions",
}


@dataclass(frozen=True)
class FormatConversion:
    """The validated outputs produced from one style spec.

    ``recipe`` selects the pipeline stages and output handling, ``bible`` is the
    reusable style guidance and ``preset`` is a named bundle of configuration
    overrides derived from the spec.
    """

    recipe: FormatRecipeConfig
    bible: StyleBible
    preset: PresetConfig


def _structure_learned(structure: Any) -> bool:
    return bool(
        structure.beats
        or structure.hook
        or structure.shot_types
        or structure.transitions
        or structure.narrative_arc
        or structure.total_duration_seconds is not None
        or structure.scenes
    )


def _palette_learned(palette: Any) -> bool:
    return bool(
        palette.colors
        or palette.roles
        or palette.temperature is not None
        or palette.saturation is not None
        or palette.contrast is not None
        or palette.scheme
    )


def _typography_learned(typography: Any) -> bool:
    return bool(
        typography.typefaces
        or typography.font_weight is not None
        or typography.font_style
        or typography.font_size is not None
        or typography.heading_size is not None
        or typography.body_size is not None
        or typography.line_height is not None
        or typography.letter_spacing is not None
        or typography.color
        or typography.placement
        or typography.caption_style is not CaptionStyle.NONE
    )


def _music_learned(music: Any) -> bool:
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


def _pacing_learned(pacing: Any) -> bool:
    return bool(
        pacing.cuts_per_minute is not None
        or pacing.average_shot_seconds is not None
        or pacing.min_shot_seconds is not None
        or pacing.max_shot_seconds is not None
        or pacing.words_per_second is not None
        or pacing.energy is not None
        or pacing.style
    )


def _aspect_learned(aspect: Any) -> bool:
    return bool(aspect.ratio or aspect.resolution or aspect.orientation is not None)


def _non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set, frozenset)):
        return len(value) > 0
    return True


def _options(**items: Any) -> dict[str, Any]:
    return {key: value for key, value in items.items() if _non_empty(value)}


def _format_recipe_stages(spec: StyleSpec) -> list[StageConfig]:
    stages: list[StageConfig] = []
    for section, stage in _SECTION_STAGES:
        learned = {
            "structure": _structure_learned(spec.structure),
            "palette": _palette_learned(spec.palette),
            "typography": _typography_learned(spec.typography),
            "music": _music_learned(spec.music),
            "pacing": _pacing_learned(spec.pacing),
        }[section]
        if not learned:
            continue
        options: dict[str, Any] = {}
        if section == "structure":
            options = _options(
                beats=list(spec.structure.beats) or None,
                hook=spec.structure.hook or None,
                narrative_arc=spec.structure.narrative_arc or None,
                total_duration_seconds=spec.structure.total_duration_seconds,
            )
        elif section == "palette":
            options = _options(
                palette=list(spec.palette.colors) or None,
                primary=spec.palette.primary or None,
                background=spec.palette.background or None,
                temperature=spec.palette.temperature.value
                if spec.palette.temperature is not None
                else None,
                saturation=spec.palette.saturation,
                contrast=spec.palette.contrast,
                scheme=spec.palette.scheme or None,
            )
        elif section == "typography":
            options = _options(
                font=list(spec.typography.typefaces) or None,
                font_weight=spec.typography.font_weight,
                font_style=spec.typography.font_style or None,
                font_size=spec.typography.font_size,
                heading_size=spec.typography.heading_size,
                body_size=spec.typography.body_size,
                align=spec.typography.align.value,
                case=spec.typography.case.value,
                line_height=spec.typography.line_height,
                letter_spacing=spec.typography.letter_spacing,
                color=spec.typography.color or None,
                placement=spec.typography.placement or None,
                caption_style=spec.typography.caption_style.value,
            )
        elif section == "music":
            options = _options(
                genre=spec.music.genre or None,
                moods=list(spec.music.moods) or None,
                bpm=spec.music.bpm,
                key=spec.music.key or None,
                scale=spec.music.scale or None,
                instruments=list(spec.music.instruments) or None,
                energy=spec.music.energy,
                valence=spec.music.valence,
                loudness_lufs=spec.music.loudness_lufs,
            )
        elif section == "pacing":
            options = _options(
                cuts_per_minute=spec.pacing.cuts_per_minute,
                average_shot_seconds=spec.pacing.average_shot_seconds,
                min_shot_seconds=spec.pacing.min_shot_seconds,
                max_shot_seconds=spec.pacing.max_shot_seconds,
                words_per_second=spec.pacing.words_per_second,
                energy=spec.pacing.energy,
                style=spec.pacing.style or None,
            )
        stages.append(StageConfig(stage=stage, options=options))
    return stages


def _format_recipe_output_type(spec: StyleSpec) -> OutputType:
    temporal = (
        _structure_learned(spec.structure)
        or spec.pacing.words_per_second is not None
        or _music_learned(spec.music)
    )
    return OutputType.VIDEO if temporal else OutputType.IMAGE


def style_spec_to_format_recipe(
    spec: StyleSpec | Mapping[str, Any],
) -> FormatRecipeConfig:
    """Build a validated :class:`FormatRecipeConfig` from *spec*.

    Stages are added for each section the spec learned, in a fixed execution
    order, each carrying the section's fields as recipe-specific options. The
    output type is video when the spec describes temporal content and image
    otherwise; the output is always saved.
    """
    value = (
        spec if isinstance(spec, StyleSpec) else StyleSpec.model_validate(dict(spec))
    )
    stages = _format_recipe_stages(value)
    capabilities: list[str] = []
    seen: set[str] = set()
    for stage in stages:
        capability = _STAGE_CAPABILITIES[stage.stage]
        if capability not in seen:
            seen.add(capability)
            capabilities.append(capability)
    return FormatRecipeConfig(
        description=value.description or "",
        stages=stages,
        capabilities=capabilities,
        output_type=_format_recipe_output_type(value),
    )


def style_spec_to_style_bible(
    spec: StyleSpec | Mapping[str, Any],
    *,
    name: str | None = None,
    description: str | None = None,
) -> StyleBible:
    """Build a validated :class:`StyleBible` from *spec*.

    Delegates to the existing style spec translation, so the bible entries and
    prompt suffixes match those produced by :func:`translate_style_spec`.
    Raises when the spec has no style to translate.
    """
    value = (
        spec if isinstance(spec, StyleSpec) else StyleSpec.model_validate(dict(spec))
    )
    return style_bible_from_spec(value, name=name)


def style_spec_to_preset(
    spec: StyleSpec | Mapping[str, Any],
    *,
    name: str | None = None,
    description: str | None = None,
) -> PresetConfig:
    """Build a validated :class:`PresetConfig` from *spec*.

    The preset overrides carry the spec's aspect, typography, music and pacing
    values under configuration-section keys, so applying the preset to a run
    reproduces the learned style without re-reading the spec.
    """
    value = (
        spec if isinstance(spec, StyleSpec) else StyleSpec.model_validate(dict(spec))
    )
    preset_name = _clean(name) or value.name or "style-spec"
    overrides: dict[str, Any] = {}

    if value.description:
        overrides["description"] = value.description

    video: dict[str, Any] = {}
    if value.aspect.ratio:
        video["aspect_ratio"] = value.aspect.ratio
    if value.aspect.resolution:
        video["resolution"] = value.aspect.resolution
    if video:
        overrides["video"] = video

    subtitle: dict[str, Any] = {}
    if value.typography.typefaces:
        subtitle["font"] = value.typography.typefaces[0]
    if value.typography.font_weight is not None:
        subtitle["font_weight"] = value.typography.font_weight
    if value.typography.font_size is not None:
        subtitle["font_size"] = value.typography.font_size
    if value.typography.align.value != "center":
        subtitle["align"] = value.typography.align.value
    if value.typography.case.value != "none":
        subtitle["case"] = value.typography.case.value
    if value.typography.placement:
        subtitle["placement"] = value.typography.placement
    if value.typography.caption_style.value != "none":
        subtitle["caption_style"] = value.typography.caption_style.value
    if subtitle:
        overrides["subtitle"] = subtitle

    music: dict[str, Any] = {}
    if value.music.genre:
        music["genre"] = value.music.genre
    if value.music.moods:
        music["moods"] = list(value.music.moods)
    if value.music.bpm is not None:
        music["bpm"] = value.music.bpm
    if value.music.instruments:
        music["instruments"] = list(value.music.instruments)
    if value.music.energy is not None:
        music["energy"] = value.music.energy
    if music:
        overrides["music"] = music

    voice: dict[str, Any] = {}
    if value.pacing.words_per_second is not None:
        voice["words_per_second"] = value.pacing.words_per_second
    if value.pacing.energy is not None:
        voice["energy"] = value.pacing.energy
    if value.pacing.style:
        voice["style"] = value.pacing.style
    if voice:
        overrides["voice"] = voice

    if overrides:
        overrides["format"] = {"name": preset_name}

    return PresetConfig(
        name=preset_name,
        description=_clean(description) or value.description or "",
        overrides=overrides,
    )


def convert_style_spec(
    spec: StyleSpec | Mapping[str, Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> FormatConversion:
    """Convert *spec* into a format recipe, style bible and preset.

    *spec* may be a :class:`StyleSpec`, a mapping or a reference analysis. An
    empty spec raises :class:`StylePromptError` because there is no style to
    turn into a recipe, bible or preset.
    """
    if spec is None:
        spec = StyleSpec()
    value = (
        spec if isinstance(spec, StyleSpec) else StyleSpec.model_validate(dict(spec))
    )
    if value.is_empty:
        raise StylePromptError(
            "style spec has no style to convert",
            details={"name": name or value.name or "style-spec"},
        )
    translation = translate_style_spec(value, name=name, description=description)
    bible = translation.to_bible(name=name)
    recipe = style_spec_to_format_recipe(value)
    preset = style_spec_to_preset(value, name=name, description=description)
    return FormatConversion(recipe=recipe, bible=bible, preset=preset)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())

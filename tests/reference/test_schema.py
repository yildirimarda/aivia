"""Reference and style spec schemas: sections, normalization and round trips."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from pydantic import ValidationError

from aivia.core.models.base import canonical_json
from aivia.reference import (
    CURRENT_REFERENCE_SCHEMA_VERSION,
    CURRENT_STYLE_SPEC_SCHEMA_VERSION,
    Analysis,
    Aspect,
    AspectOrientation,
    AspectSpec,
    CaptionStyle,
    ColorTemperature,
    Music,
    MusicMood,
    MusicMoodSpec,
    Pacing,
    PacingSpec,
    Palette,
    PaletteSpec,
    Reference,
    ReferenceAnalysis,
    ReferenceKind,
    ReferenceSpec,
    SceneSpec,
    ShotType,
    Structure,
    StructureSpec,
    Style,
    StyleSpec,
    TextAlignment,
    Typography,
    TypographySpec,
    build_reference_schema,
    build_style_spec_schema,
)


def _full_style() -> StyleSpec:
    return StyleSpec.model_validate(
        {
            "name": "Cinematic short",
            "description": "Fast opening, calm middle and a single payoff.",
            "prompt_suffix": "Natural light, shallow depth of field.",
            "structure": {
                "scenes": [
                    {
                        "start": 0.0,
                        "end": 2.5,
                        "shot_type": "wide",
                        "description": "Establishing shot",
                        "transition": "cut",
                    },
                    {
                        "start": 2.5,
                        "duration": 3.0,
                        "shot_type": "close-up",
                        "text": "One idea",
                        "has_caption": True,
                    },
                ],
                "hook": "Open on a question",
                "hook_seconds": 1.5,
                "beats": ["hook", "explain", "payoff"],
                "transitions": ["cut", "fade"],
                "narrative_arc": "question to answer",
                "acts": 2,
            },
            "pacing": {
                "cuts_per_minute": 24,
                "average_shot_seconds": 2.75,
                "min_shot_seconds": 1.0,
                "max_shot_seconds": 4.0,
                "words_per_second": 2.8,
                "energy": 0.6,
                "style": "measured",
            },
            "palette": {
                "colors": ["#FFF", "#112233", "#112233"],
                "primary": "#ABCDEF",
                "background": "#000",
                "temperature": "warm",
                "saturation": 0.4,
                "contrast": 0.7,
                "scheme": "analogous",
            },
            "typography": {
                "font": "Lato",
                "heading_font": "Lato Black",
                "body_font": "Lato",
                "weight": "bold",
                "size": 48,
                "line_height": 1.2,
                "letter_spacing": 0.5,
                "alignment": "centre",
                "case": "uppercase",
                "color": "#FFF",
                "placement": "lower third",
                "caption_style": "burned-in",
            },
            "music_mood": {
                "genre": "ambient",
                "tempo": 92,
                "moods": ["calm", "hopeful"],
                "key": "C",
                "scale": "minor",
                "instruments": ["piano", "strings"],
                "vocals": False,
                "energy": 0.3,
                "valence": 0.6,
                "loudness_lufs": -16.0,
            },
            "aspect": "9:16",
            "duration": 5.5,
            "caption_style": "subtitles",
            "reference_ids": ["ref-1"],
        }
    )


def test_style_spec_carries_structure_pacing_palette_typography_music_and_aspect() -> (
    None
):
    spec = _full_style()

    assert spec.name == "Cinematic short"
    assert spec.prompt_suffix == "Natural light, shallow depth of field."
    assert spec.structure.hook == "Open on a question"
    assert spec.structure.scene_count == 2
    assert spec.structure.total_duration_seconds == 5.5
    assert spec.structure.duration == 5.5
    assert spec.structure.cuts == 1
    assert spec.structure.scene_durations == [2.5, 3.0]
    assert spec.structure.shot_types == [ShotType.WIDE, ShotType.CLOSE_UP]
    assert spec.structure.beats == ["hook", "explain", "payoff"]
    assert spec.structure.narrative_arc == "question to answer"

    assert spec.pacing.cuts_per_minute == 24.0
    assert spec.pacing.average_shot_seconds == 2.75
    assert spec.pacing.words_per_second == 2.8
    assert spec.pacing.energy == 0.6
    assert spec.pacing.style == "measured"

    assert spec.palette.colors == ["#ffffff", "#112233"]
    assert spec.palette.primary == "#abcdef"
    assert spec.palette.background == "#000000"
    assert spec.palette.temperature is ColorTemperature.WARM
    assert spec.palette.saturation == 0.4
    assert spec.palette.contrast == 0.7
    assert spec.palette.scheme == "analogous"

    assert spec.typography.font_family == "Lato"
    assert spec.typography.primary_font == "Lato"
    assert spec.typography.typefaces == [
        "Lato",
        "Lato Black",
    ]
    assert spec.typography.font_weight == 700
    assert spec.typography.font_size == 48.0
    assert spec.typography.align is TextAlignment.CENTER
    assert spec.typography.case.value == "upper"
    assert spec.typography.color == "#ffffff"
    assert spec.typography.caption_style is CaptionStyle.BURNED_IN

    assert spec.music.genre == "ambient"
    assert spec.music.genres == ["ambient"]
    assert spec.music.bpm == 92.0
    assert spec.music.tempo == 92.0
    assert spec.music.moods == ["calm", "hopeful"]
    assert spec.music.instruments == ["piano", "strings"]
    assert spec.music.vocals == "instrumental"
    assert spec.music.is_instrumental is True
    assert spec.music.energy == 0.3
    assert spec.music.valence == 0.6
    assert spec.music.loudness_lufs == -16.0

    assert spec.aspect.ratio == "9:16"
    assert spec.aspect.orientation is AspectOrientation.PORTRAIT
    assert spec.aspect.is_portrait is True
    assert spec.aspect.aspect_ratio == pytest.approx(9 / 16)

    assert spec.duration_seconds == 5.5
    assert spec.caption_style is CaptionStyle.SUBTITLES
    assert spec.typography.caption_style is CaptionStyle.BURNED_IN
    assert spec.references == ["ref-1"]


def test_style_spec_defaults_are_versioned_and_empty() -> None:
    spec = StyleSpec()

    assert spec.schema_version == CURRENT_STYLE_SPEC_SCHEMA_VERSION
    assert spec.is_empty is True
    assert spec.duration_seconds is None
    assert spec.aspect.ratio == ""
    assert spec.palette.colors == []
    assert spec.music.genres == []
    assert spec.typography.align is TextAlignment.CENTER
    assert spec.typography.caption_style is CaptionStyle.NONE


def test_style_spec_becomes_non_empty_when_one_section_is_learned() -> None:
    spec = StyleSpec.model_validate({"palette": {"colors": ["#123456"]}})

    assert spec.is_empty is False
    assert spec.palette.colors == ["#123456"]


def test_style_spec_round_trips_canonically() -> None:
    spec = _full_style()

    data = spec.to_dict()
    restored = StyleSpec.model_validate(data)

    assert restored == spec
    assert json.loads(spec.to_json()) == data
    assert spec.to_json() == canonical_json(data)
    assert spec.model_dump_json() == spec.to_json()
    assert StyleSpec.from_dict(data) == spec
    assert StyleSpec.from_json(spec.to_json()) == spec
    assert StyleSpec.model_validate_json(spec.to_json()) == spec


def test_style_spec_json_schema_exposes_every_section() -> None:
    schema = build_style_spec_schema()

    properties = schema["properties"]
    for section in (
        "structure",
        "pacing",
        "palette",
        "typography",
        "music",
        "aspect",
    ):
        assert section in properties
    assert "duration_seconds" in properties
    assert "caption_style" in properties


def test_aspect_parses_ratio_and_resolution_shorthands() -> None:
    ratio = AspectSpec.model_validate("9:16")
    assert ratio.ratio == "9:16"
    assert ratio.orientation is AspectOrientation.PORTRAIT
    assert ratio.dimensions is None

    resolution = AspectSpec.model_validate("1080x1920")
    assert resolution.ratio == "9:16"
    assert resolution.resolution == "1080x1920"
    assert resolution.width == 1080
    assert resolution.height == 1920
    assert resolution.orientation is AspectOrientation.PORTRAIT
    assert resolution.dimensions == (1080, 1920)

    wide = AspectSpec(width=1920, height=1080)
    assert wide.ratio == "16:9"
    assert wide.resolution == "1920x1080"
    assert wide.orientation is AspectOrientation.LANDSCAPE
    assert wide.is_landscape is True

    square = AspectSpec.model_validate({"aspect_ratio": "1:1"})
    assert square.orientation is AspectOrientation.SQUARE
    assert square.is_square is True


def test_aspect_rejects_unknown_shorthand_and_zero_side() -> None:
    with pytest.raises(ValidationError):
        AspectSpec.model_validate("banana")
    with pytest.raises(ValidationError):
        AspectSpec.model_validate("0:16")
    with pytest.raises(ValidationError):
        AspectSpec.model_validate("1080x0")


def test_structure_derives_scene_count_and_duration_from_scenes() -> None:
    structure = StructureSpec.model_validate(
        {
            "scenes": [
                {"start": 0, "end": 1.5, "shot": "establishing"},
                {"start": 1.5, "duration": 2.5, "shot": "close_up"},
            ]
        }
    )

    assert structure.scene_count == 2
    assert structure.total_duration_seconds == 4.0
    assert structure.duration_seconds == 4.0
    assert structure.cuts == 1
    assert structure.shot_types == [ShotType.WIDE, ShotType.CLOSE_UP]


def test_scene_derives_missing_endpoint_from_duration() -> None:
    from_duration = SceneSpec(start_seconds=2.0, duration_seconds=3.0)
    assert from_duration.end_seconds == 5.0
    assert from_duration.duration_seconds == 3.0

    from_end = SceneSpec(end_seconds=5.0, duration_seconds=2.0)
    assert from_end.start_seconds == 3.0
    assert from_end.duration_seconds == 2.0


def test_scene_rejects_backwards_and_empty_durations() -> None:
    with pytest.raises(ValidationError):
        SceneSpec(start_seconds=5.0, end_seconds=2.0)
    with pytest.raises(ValidationError):
        SceneSpec(start_seconds=2.0, end_seconds=2.0)


def test_pacing_validates_bounds_and_reports_fast_or_slow() -> None:
    with pytest.raises(ValidationError):
        PacingSpec(min_shot_seconds=4.0, max_shot_seconds=1.0)
    with pytest.raises(ValidationError):
        PacingSpec(cuts_per_minute=-1)
    with pytest.raises(ValidationError):
        PacingSpec(energy=1.5)

    fast = PacingSpec(cuts_per_minute=90)
    slow = PacingSpec(cuts_per_minute=10)
    assert fast.is_fast is True
    assert fast.is_slow is False
    assert slow.is_slow is True
    assert slow.is_fast is False


def test_palette_normalizes_colours_deduplicates_and_keeps_roles() -> None:
    palette = PaletteSpec.model_validate(
        {
            "palette": ["#FFF", "red", "#112233", "#fff"],
            "accent": "#ABC",
            "background": "#000",
        }
    )

    assert palette.colors == ["#ffffff", "red", "#112233"]
    assert palette.dominant_colors == palette.colors
    assert palette.accent == "#aabbcc"
    assert palette.background == "#000000"
    assert palette.roles == {
        "accent": "#aabbcc",
        "background": "#000000",
    }


def test_palette_rejects_malformed_hex_colour() -> None:
    with pytest.raises(ValidationError):
        PaletteSpec(colors=["#12"])
    with pytest.raises(ValidationError):
        PaletteSpec(primary="#gggggg")


def test_typography_normalizes_weight_alignment_case_and_caption_style() -> None:
    typography = TypographySpec.model_validate(
        {
            "typefaces": ["Inter", "Georgia"],
            "weight": "semi-bold",
            "text_align": "start",
            "capitalization": "all caps",
            "captions": "word by word",
        }
    )

    assert typography.fonts == ["Inter", "Georgia"]
    assert typography.primary_font == "Inter"
    assert typography.font_weight == 600
    assert typography.weight == 600
    assert typography.align is TextAlignment.LEFT
    assert typography.case.value == "upper"
    assert typography.caption_style is CaptionStyle.KARAOKE


def test_typography_rejects_unknown_weight() -> None:
    with pytest.raises(ValidationError):
        TypographySpec.model_validate({"font_weight": "ultra-chunky"})


def test_music_mood_validates_tempo_energy_and_vocal_flag() -> None:
    music = MusicMoodSpec.model_validate(
        {
            "genre": "lo-fi",
            "mood": "calm",
        }
    )
    assert music.genres == ["lo-fi"]
    assert music.moods == ["calm"]

    with pytest.raises(ValidationError):
        MusicMoodSpec.model_validate({"tempo": "not a number"})
    with pytest.raises(ValidationError):
        MusicMoodSpec(bpm=500)
    with pytest.raises(ValidationError):
        MusicMoodSpec(energy=1.2)
    with pytest.raises(ValidationError):
        MusicMoodSpec(valence=-0.1)


def test_music_mood_accepts_a_boolean_vocal_flag() -> None:
    assert MusicMoodSpec.model_validate({"vocals": True}).vocals == "vocal"
    assert MusicMoodSpec.model_validate({"vocals": False}).vocals == "instrumental"
    assert MusicMoodSpec.model_validate({"vocals": False}).is_instrumental is True


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({"path": "clip.MP4"}, ReferenceKind.VIDEO),
        ({"url": "https://example.test/song.mp3?download=1"}, ReferenceKind.AUDIO),
        ({"mime_type": "text/plain"}, ReferenceKind.TEXT),
        ({"path": "photo.jpeg"}, ReferenceKind.IMAGE),
        ({"kind": "document", "path": "notes.bin"}, ReferenceKind.TEXT),
        ({"path": "unknown.bin"}, ReferenceKind.IMAGE),
    ],
)
def test_reference_infers_its_kind_from_mime_or_extension(
    values: dict[str, Any], expected: ReferenceKind
) -> None:
    reference = ReferenceSpec.model_validate(values)

    assert reference.kind is expected


def test_reference_rejects_an_unknown_explicit_kind() -> None:
    with pytest.raises(ValidationError):
        ReferenceSpec.model_validate({"kind": "hologram"})


def test_reference_carries_analysis_rights_and_location() -> None:
    reference = ReferenceSpec.model_validate(
        {
            "path": "references/ref.mp4",
            "mime_type": "video/mp4",
            "owned": True,
            "label": "  Example  ",
            "content_hash": "A" * 64,
            "analysis": {
                "palette": {"colors": ["#ABC"]},
                "confidence": 0.8,
                "detected_text": "  on screen  ",
                "reproducing_prompt": "  A calm, wide shot  ",
            },
        }
    )

    assert reference.kind is ReferenceKind.VIDEO
    assert reference.label == "Example"
    assert reference.content_hash == "a" * 64
    assert reference.owned is True
    assert reference.is_local is True
    assert reference.is_remote is False
    assert reference.extension == ".mp4"
    assert reference.analysis.palette.colors == ["#aabbcc"]
    assert reference.analysis.confidence == 0.8
    assert reference.analysis.detected_text == "on screen"
    assert reference.analysis.reproducing_prompt == "A calm, wide shot"
    assert reference.style_spec is reference.analysis
    assert reference.has_analysis is True


def test_reference_marks_a_url_as_remote() -> None:
    reference = ReferenceSpec(url="https://example.test/ref.png")

    assert reference.is_remote is True
    assert reference.is_local is False
    assert reference.extension == ".png"


def test_reference_accepts_a_style_spec_as_its_analysis() -> None:
    style = StyleSpec.model_validate({"name": "Clean", "palette": {"colors": ["#fff"]}})
    reference = ReferenceSpec(path="ref.png", analysis=cast(ReferenceAnalysis, style))

    assert isinstance(reference.analysis, ReferenceAnalysis)
    assert reference.analysis.name == "Clean"
    assert reference.analysis.palette.colors == ["#ffffff"]


def test_reference_round_trips_canonically() -> None:
    reference = ReferenceSpec.model_validate(
        {
            "path": "ref.png",
            "analysis": {"structure": {"hook": "Start with a question"}},
        }
    )

    restored = ReferenceSpec.model_validate_json(reference.to_json())

    assert restored == reference
    assert reference.to_json() == canonical_json(reference.to_dict())
    assert build_reference_schema()["properties"]["kind"]["$ref"].endswith(
        "ReferenceKind"
    )


def test_public_aliases_point_at_the_canonical_models() -> None:
    assert Reference is ReferenceSpec
    assert Style is StyleSpec
    assert Analysis is ReferenceAnalysis
    assert Structure is StructureSpec
    assert Pacing is PacingSpec
    assert Palette is PaletteSpec
    assert Typography is TypographySpec
    assert MusicMood is MusicMoodSpec
    assert Music is MusicMoodSpec
    assert Aspect is AspectSpec
    assert CURRENT_REFERENCE_SCHEMA_VERSION == 1

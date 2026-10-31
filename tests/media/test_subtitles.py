"""Subtitle cues export to SRT, VTT and styled ASS with safe area margins."""

from __future__ import annotations

from pathlib import Path

import pytest

from aivia.core.config.sections import AccessibilityConfig
from aivia.core.errors import ValidationError
from aivia.core.platforms import Platform, SafeArea
from aivia.media.align_verify import align_text
from aivia.media.subtitles import (
    STYLE_PRESETS,
    Subtitle,
    SubtitleAlignment,
    SubtitleCue,
    SubtitleFormat,
    SubtitleHighlight,
    SubtitleStyle,
    SubtitleStylePreset,
    SubtitleWord,
    ass_color,
    build_cues,
    format_ass_timestamp,
    format_srt_timestamp,
    format_vtt_timestamp,
    from_words,
    get_style_preset,
    list_style_presets,
    to_ass,
    to_srt,
    to_vtt,
)
from aivia.providers.protocols import STTResponse, WordTimestamp


def _word(text: str, start: float, end: float) -> SubtitleWord:
    return SubtitleWord(word=text, start_seconds=start, end_seconds=end)


def _words() -> list[SubtitleWord]:
    return [
        _word("Hello", 0.0, 0.5),
        _word("brave", 0.5, 1.0),
        _word("world", 1.25, 1.9),
    ]


def test_srt_and_vtt_export_use_their_own_timestamp_shapes() -> None:
    subtitle = Subtitle.from_words(_words())

    srt = subtitle.to_srt()
    vtt = subtitle.to_vtt()

    assert srt == ("1\n00:00:00,000 --> 00:00:01,900\nHello brave world\n")
    assert vtt == ("WEBVTT\n\n00:00:00.000 --> 00:00:01.900\nHello brave world\n")
    assert format_srt_timestamp(3661.234) == "01:01:01,234"
    assert format_vtt_timestamp(3661.234) == "01:01:01.234"
    assert format_ass_timestamp(3661.234) == "1:01:01.23"


def test_module_level_exports_accept_words_and_cues() -> None:
    words = _words()
    cue = SubtitleCue.from_words(words)

    assert "Hello brave world" in to_srt(words)
    assert "WEBVTT" in to_vtt([cue])
    assert "[Script Info]" in to_ass(words)


def test_words_group_into_cues_at_gaps_character_limits_and_duration() -> None:
    words = [
        _word("one", 0.0, 0.3),
        _word("two", 0.4, 0.7),
        _word("three", 2.0, 2.3),
    ]

    by_gap = build_cues(words)
    by_chars = build_cues(words[:2], max_chars=5)
    by_duration = build_cues(words[:2], max_duration_seconds=0.5)

    assert [cue.text for cue in by_gap] == ["one two", "three"]
    assert [cue.text for cue in by_chars] == ["one", "two"]
    assert [cue.text for cue in by_duration] == ["one", "two"]
    assert by_gap[1].start_seconds == pytest.approx(2.0)
    assert by_gap[1].end_seconds == pytest.approx(2.3)


def test_ass_uses_karaoke_word_tags_a_preset_and_safe_area_margins() -> None:
    subtitle = Subtitle.from_words(_words(), preset=SubtitleStylePreset.BOLD)

    ass = subtitle.to_ass(
        width=1080,
        height=1920,
        safe_area=SafeArea(top=0.10, bottom=0.20, left=0.06, right=0.06),
    )

    assert "PlayResX: 1080" in ass
    assert "PlayResY: 1920" in ass
    assert "Style: SubtitleBold42,Lato,42," in ass
    assert "{\\k50}Hello" in ass
    assert "{\\k50}brave" in ass
    assert "Dialogue: 0,0:00:00.00,0:00:01.90,SubtitleBold42,,65,65,384,," in ass
    assert ass_color("#FFD400") in ass


def test_word_highlight_mode_emits_one_event_per_word_with_current_color() -> None:
    subtitle = Subtitle.from_words(_words())

    ass = subtitle.to_ass(highlight=SubtitleHighlight.WORD)

    dialogue_lines = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    assert len(dialogue_lines) == len(_words())
    assert "{\\1c&H0000D4FF&}Hello{\\1c&H00FFFFFF&}" in dialogue_lines[0]
    assert dialogue_lines[1].startswith("Dialogue: 0,0:00:00.50,")
    assert "{\\1c&H0000D4FF&}brave{\\1c&H00FFFFFF&}" in dialogue_lines[1]


def test_top_alignment_uses_the_top_safe_area_margin() -> None:
    subtitle = Subtitle.from_words(
        _words(), style=SubtitleStyle(alignment=SubtitleAlignment.TOP_CENTER)
    )

    ass = subtitle.to_ass(
        width=1000,
        height=2000,
        safe_area=SafeArea(top=0.15, bottom=0.30),
        highlight=False,
    )

    assert "Dialogue: 0,0:00:00.00,0:00:01.90,SubtitleRegular32,,0,0,300,," in ass


def test_platform_safe_area_is_used_when_no_explicit_area_is_given() -> None:
    platform = Platform(
        id="demo",
        safe_area=SafeArea(top=0.05, bottom=0.25, left=0.10, right=0.10),
    )

    ass = Subtitle.from_words(_words()).to_ass(
        width=1000,
        height=1000,
        platform=platform,
        highlight=False,
    )

    assert "Dialogue: 0,0:00:00.00,0:00:01.90,SubtitleRegular32,,100,100,250,," in ass


def test_style_presets_colors_and_accessibility_config() -> None:
    assert list_style_presets() == (
        "bold",
        "box",
        "default",
        "karaoke",
        "minimal",
        "outline",
    )
    assert get_style_preset("bold").bold is True
    assert get_style_preset("box").background is True
    assert get_style_preset("minimal").highlight_mode is SubtitleHighlight.NONE
    assert SubtitleStyle.resolve("karaoke").font_size == 40
    assert SubtitleStyle.resolve({"font": "Inter", "font_size": 48}).font == "Inter"
    assert STYLE_PRESETS["default"].alignment == SubtitleAlignment.BOTTOM_CENTER
    assert ass_color("#FFFFFF") == "&H00FFFFFF&"
    assert ass_color("#FFFFFF80") == "&H7FFFFFFF&"
    assert ass_color((255, 212, 0)) == "&H0000D4FF&"

    configured = SubtitleStyle.from_config(
        AccessibilityConfig(
            subtitle_font="Inter",
            subtitle_font_size=40,
            subtitle_outline=False,
            subtitle_background=True,
        )
    )
    assert configured.font == "Inter"
    assert configured.font_size == 40
    assert configured.outline == 0.0
    assert configured.background is True
    assert configured.border_style == 3


def test_vtt_and_ass_escape_markup_and_newlines() -> None:
    cue = SubtitleCue(
        start_seconds=0.0,
        end_seconds=1.0,
        text="a < b & c\n{\\k}",
    )
    subtitle = Subtitle(cues=[cue])

    assert "a < b & c" in subtitle.to_srt()
    assert "a &lt; b &amp; c" in subtitle.to_vtt()
    ass = subtitle.to_ass(highlight=False)
    assert "a < b & c\\N\\{\\\\k\\}" in ass


def test_from_transcript_and_alignment_keep_timed_words_and_metadata() -> None:
    response = STTResponse(
        text="Hello world",
        language="en",
        words=[
            WordTimestamp(word="Hello", start_seconds=0.0, end_seconds=0.4),
            WordTimestamp(word="world", start_seconds=0.5, end_seconds=1.0),
        ],
        provider="local-stt",
        model="tiny",
    )

    subtitle = Subtitle.from_transcript(response, preset="minimal")
    alignment = align_text("Hello brave world", response)
    aligned = Subtitle.from_alignment(alignment, preset="outline")

    assert [word.word for word in subtitle.words] == ["Hello", "world"]
    assert subtitle.language == "en"
    assert subtitle.metadata == {"provider": "local-stt", "model": "tiny"}
    assert subtitle.style.highlight_mode is SubtitleHighlight.NONE
    assert [word.word for word in aligned.words] == ["Hello", "world"]
    assert aligned.language == "en"
    assert aligned.style.bold is True


def test_tracks_write_utf8_files_and_infer_the_format(tmp_path: Path) -> None:
    subtitle = Subtitle.from_words(_words())

    srt = subtitle.write_srt(tmp_path / "track.srt")
    vtt = subtitle.write_vtt(tmp_path / "track.vtt")
    ass = subtitle.write_ass(tmp_path / "track.ass", width=1080, height=1920)
    inferred = subtitle.write(tmp_path / "inferred.vtt")

    assert srt.read_text(encoding="utf-8").startswith("1\n")
    assert vtt.read_text(encoding="utf-8").startswith("WEBVTT\n\n")
    assert ass.read_text(encoding="utf-8").startswith("[Script Info]")
    assert inferred.read_text(encoding="utf-8") == vtt.read_text(encoding="utf-8")
    assert SubtitleFormat.from_path(tmp_path / "x.ass") is SubtitleFormat.ASS


def test_invalid_words_cues_and_render_options_are_rejected() -> None:
    with pytest.raises(ValueError):
        SubtitleWord(word="bad", start_seconds=2.0, end_seconds=1.0)
    with pytest.raises(ValidationError):
        build_cues([_word("a", 1.0, 1.2), _word("b", 0.0, 0.1)])
    with pytest.raises(ValidationError):
        build_cues(_words(), max_chars=0)
    with pytest.raises(ValidationError):
        Subtitle.from_words(_words()).to_ass(width=0)
    with pytest.raises(ValidationError):
        Subtitle.from_words(_words()).to_ass(safe_area=SafeArea(top=0.6, bottom=0.6))
    with pytest.raises(ValidationError):
        Subtitle.from_words(_words()).to_ass(safe_area={"left": 2.0})
    with pytest.raises(ValidationError):
        SubtitleFormat.parse("txt")
    with pytest.raises(ValidationError):
        get_style_preset("missing")


def test_empty_subtitle_exports_are_valid_and_empty() -> None:
    subtitle = Subtitle()

    assert subtitle.to_srt() == ""
    assert subtitle.to_vtt() == "WEBVTT\n\n"
    assert "PlayResX: 1920" in subtitle.to_ass()
    assert from_words([]).cues == []

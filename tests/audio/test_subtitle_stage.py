"""The subtitle stage renders one styled ASS file per language from a timeline."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, cast

import pytest

from aivia.audio import SubtitleStage, SubtitleStageResult
from aivia.compose import Timeline
from aivia.core.errors import ConfigError
from aivia.core.platforms import Platform, SafeArea
from aivia.core.stages import StageContext


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _context(run_dir: Path, *, config: Any = None) -> StageContext:
    return StageContext(run_dir=run_dir, config=config or {})


def _words(*pairs: tuple[str, float, float]) -> list[dict[str, Any]]:
    return [
        {"word": word, "start_seconds": start, "end_seconds": end}
        for word, start, end in pairs
    ]


def _timeline(
    *languages: dict[str, Any], platforms: list[str] | None = None
) -> Timeline:
    return Timeline.model_validate(
        {
            "id": "run",
            "languages": list(languages),
            "platforms": platforms or [],
        }
    )


def _subtitle_language(
    language: str,
    *pairs: tuple[str, float, float],
) -> dict[str, Any]:
    return {
        "language": language,
        "subtitles": {"words": _words(*pairs)},
    }


def test_stage_writes_one_ass_per_language_with_preset_and_safe_area(
    tmp_path: Path,
) -> None:
    timeline = _timeline(
        _subtitle_language("en", ("Hello", 0.0, 0.5), ("world", 0.6, 1.2)),
        _subtitle_language("de", ("Hallo", 0.0, 0.5), ("Welt", 0.6, 1.2)),
    )
    stage = SubtitleStage(
        options={
            "preset": "bold",
            "safe_area": {"top": 0.10, "bottom": 0.20, "left": 0.06, "right": 0.06},
            "width": 1080,
            "height": 1920,
        },
        timeline=timeline,
    )

    result = cast(SubtitleStageResult, _run(stage.run(_context(tmp_path))))

    assert result.stage == "subtitle"
    assert result.artifacts == ["subtitle.en.ass", "subtitle.de.ass"]
    assert result.files == {
        "en": "subtitle.en.ass",
        "de": "subtitle.de.ass",
    }
    assert result.metadata["languages"] == ["en", "de"]
    assert result.metadata["width"] == 1080
    assert result.metadata["height"] == 1920

    english = (tmp_path / "subtitle.en.ass").read_text(encoding="utf-8")
    german = (tmp_path / "subtitle.de.ass").read_text(encoding="utf-8")
    assert "PlayResX: 1080" in english
    assert "PlayResY: 1920" in english
    assert "Style: SubtitleBold42,Lato,42," in english
    assert "Dialogue: 0,0:00:00.00,0:00:01.20,SubtitleBold42,,65,65,384,," in english
    assert "Hello" in english
    assert "Hallo" in german
    assert "Welt" in german


def test_stage_writes_one_ass_per_language_and_platform(
    tmp_path: Path,
) -> None:
    timeline = _timeline(
        _subtitle_language("en", ("Hello", 0.0, 0.5)),
        _subtitle_language("de", ("Hallo", 0.0, 0.5)),
        platforms=["tiktok", "youtube"],
    )
    stage = SubtitleStage(options={"highlight": False}, timeline=timeline)

    result = cast(SubtitleStageResult, _run(stage.run(_context(tmp_path))))

    assert result.artifacts == [
        "subtitle.en.tiktok.ass",
        "subtitle.en.youtube.ass",
        "subtitle.de.tiktok.ass",
        "subtitle.de.youtube.ass",
    ]
    assert result.files == {
        "en/tiktok": "subtitle.en.tiktok.ass",
        "en/youtube": "subtitle.en.youtube.ass",
        "de/tiktok": "subtitle.de.tiktok.ass",
        "de/youtube": "subtitle.de.youtube.ass",
    }
    assert result.metadata["languages"] == ["en", "de"]
    assert result.metadata["platforms"] == ["tiktok", "youtube"]
    assert result.metadata["platform"] == ""
    assert result.metadata["platform_files"]["tiktok"] == {
        "en": "subtitle.en.tiktok.ass",
        "de": "subtitle.de.tiktok.ass",
    }
    assert result.metadata["platform_safe_areas"]["tiktok"]["en"] == {
        "top": 0.10,
        "bottom": 0.25,
        "left": 0.06,
        "right": 0.06,
    }
    assert result.metadata["platform_safe_areas"]["youtube"]["en"] == {
        "top": 0.10,
        "bottom": 0.20,
        "left": 0.06,
        "right": 0.06,
    }

    tiktok = (tmp_path / "subtitle.en.tiktok.ass").read_text(encoding="utf-8")
    youtube = (tmp_path / "subtitle.en.youtube.ass").read_text(encoding="utf-8")
    assert "Dialogue: 0,0:00:00.00,0:00:00.50,SubtitleRegular32,,65,65,480,," in tiktok
    assert "Dialogue: 0,0:00:00.00,0:00:00.50,SubtitleRegular32,,65,65,384,," in youtube


def test_stage_writes_one_ass_per_platform_with_its_own_canvas(
    tmp_path: Path,
) -> None:
    timeline = _timeline(
        _subtitle_language("en", ("Hello", 0.0, 0.5)),
        platforms=["tiktok", "x"],
    )
    stage = SubtitleStage(
        options={
            "filename_pattern": "captions.{code}.{platform}.ass",
            "highlight": False,
        },
        timeline=timeline,
    )

    result = cast(SubtitleStageResult, _run(stage.run(_context(tmp_path))))

    assert result.artifacts == ["captions.en.tiktok.ass", "captions.en.x.ass"]
    assert result.metadata["canvases"] == {
        "tiktok": {"width": 1080, "height": 1920},
        "x": {"width": 1280, "height": 720},
    }
    tiktok = (tmp_path / "captions.en.tiktok.ass").read_text(encoding="utf-8")
    x = (tmp_path / "captions.en.x.ass").read_text(encoding="utf-8")
    assert "PlayResX: 1080" in tiktok
    assert "PlayResY: 1920" in tiktok
    assert "PlayResX: 1280" in x
    assert "PlayResY: 720" in x
    assert "Dialogue: 0,0:00:00.00,0:00:00.50,SubtitleRegular32,,65,65,480,," in tiktok
    assert "Dialogue: 0,0:00:00.00,0:00:00.50,SubtitleRegular32,,0,0,0,," in x


def test_stage_adds_the_platform_to_a_custom_pattern_without_one(
    tmp_path: Path,
) -> None:
    timeline = _timeline(
        _subtitle_language("en", ("Hello", 0.0, 0.5)),
        platforms=["tiktok", "youtube"],
    )
    stage = SubtitleStage(
        options={"filename_pattern": "captions.{code}.ass", "highlight": False},
        timeline=timeline,
    )

    result = cast(SubtitleStageResult, _run(stage.run(_context(tmp_path))))

    assert result.artifacts == [
        "captions.en.tiktok.ass",
        "captions.en.youtube.ass",
    ]
    assert (tmp_path / "captions.en.tiktok.ass").is_file()
    assert (tmp_path / "captions.en.youtube.ass").is_file()


def test_stage_uses_platform_safe_area_and_configured_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    timeline = _timeline(
        _subtitle_language("en", ("Hello", 0.0, 0.5)),
        platforms=["tiktok"],
    )
    stage = SubtitleStage(
        options={
            "platform": "tiktok",
            "highlight": False,
        },
        timeline=timeline,
    )
    config = {
        "video": {"resolution": "1080x1920"},
        "languages": {"default": "en", "en": {}},
    }

    result = cast(
        SubtitleStageResult, _run(stage.run(_context(tmp_path, config=config)))
    )

    assert result.metadata["platform"] == "tiktok"
    assert result.metadata["safe_areas"]["en"] == {
        "top": 0.10,
        "bottom": 0.25,
        "left": 0.06,
        "right": 0.06,
    }
    ass = (tmp_path / "subtitle.en.ass").read_text(encoding="utf-8")
    assert "PlayResX: 1080" in ass
    assert "PlayResY: 1920" in ass
    assert "Dialogue: 0,0:00:00.00,0:00:00.50,SubtitleRegular32,,65,65,480,," in ass


def test_stage_derives_subtitles_from_voice_words_and_applies_language_style(
    tmp_path: Path,
) -> None:
    timeline = _timeline(
        {
            "language": "de",
            "voices": [
                {
                    "id": "de_beat_0",
                    "source": "audio/de.mp3",
                    "start_seconds": 0.0,
                    "end_seconds": 1.2,
                    "words": _words(("Hallo", 0.0, 0.5), ("Welt", 0.6, 1.2)),
                }
            ],
        }
    )
    config = {
        "languages": {
            "default": "de",
            "de": {"subtitle_font": "Inter"},
        }
    }
    stage = SubtitleStage(
        options={
            "styles": {"de": "box"},
            "safe_area": SafeArea(top=0.05, bottom=0.15, left=0.0, right=0.0),
            "width": 1000,
            "height": 1000,
            "highlight": False,
        },
        timeline=timeline,
    )

    _run(stage.run(_context(tmp_path, config=config)))

    ass = (tmp_path / "subtitle.de.ass").read_text(encoding="utf-8")
    assert "Style: SubtitleBold32,Inter,32," in ass
    assert "Hallo Welt" in ass
    assert "Dialogue: 0,0:00:00.00,0:00:01.20,SubtitleBold32,,0,0,150,," in ass


def test_stage_accepts_a_per_language_safe_area_and_filename_pattern(
    tmp_path: Path,
) -> None:
    timeline = _timeline(
        _subtitle_language("en", ("Hello", 0.0, 0.5)),
        _subtitle_language("de", ("Hallo", 0.0, 0.5)),
    )
    stage = SubtitleStage(
        options={
            "filename_pattern": "captions.{code}.ass",
            "safe_areas": {
                "en": {"bottom": 0.10},
                "de": {"bottom": 0.30},
            },
            "width": 1000,
            "height": 1000,
            "highlight": False,
        },
        timeline=timeline,
    )

    result = cast(SubtitleStageResult, _run(stage.run(_context(tmp_path))))

    assert result.artifacts == ["captions.en.ass", "captions.de.ass"]
    english = (tmp_path / "captions.en.ass").read_text(encoding="utf-8")
    german = (tmp_path / "captions.de.ass").read_text(encoding="utf-8")
    assert "Dialogue: 0,0:00:00.00,0:00:00.50,SubtitleRegular32,,0,0,100,," in english
    assert "Dialogue: 0,0:00:00.00,0:00:00.50,SubtitleRegular32,,0,0,300,," in german


def test_stage_reads_the_timeline_file_written_by_the_voice_stage(
    tmp_path: Path,
) -> None:
    timeline = _timeline(_subtitle_language("en", ("Hello", 0.0, 0.5)))
    (tmp_path / "timeline.json").write_text(
        json.dumps(timeline.model_dump(mode="json")),
        encoding="utf-8",
    )
    stage = SubtitleStage(options={"width": 640, "height": 360, "highlight": False})

    result = cast(SubtitleStageResult, _run(stage.run(_context(tmp_path))))

    assert result.artifacts == ["subtitle.en.ass"]
    assert result.timeline == "timeline.json"
    ass = (tmp_path / "subtitle.en.ass").read_text(encoding="utf-8")
    assert "PlayResX: 640" in ass
    assert "PlayResY: 360" in ass


def test_stage_rejects_a_missing_timeline_and_languages_without_subtitles(
    tmp_path: Path,
) -> None:
    stage = SubtitleStage()

    with pytest.raises(ConfigError, match="needs a timeline"):
        _run(stage.run(_context(tmp_path)))

    timeline = _timeline({"language": "en", "voices": []})
    empty = SubtitleStage(timeline=timeline)
    with pytest.raises(ConfigError, match="no subtitles or timed words"):
        _run(empty.run(_context(tmp_path)))

    missing = SubtitleStage(
        options={"languages": ["fr"]},
        timeline=_timeline(_subtitle_language("en", ("Hello", 0.0, 0.5))),
    )
    with pytest.raises(ConfigError, match="missing some languages"):
        _run(missing.run(_context(tmp_path)))


def test_stage_rejects_an_unknown_style_preset(tmp_path: Path) -> None:
    stage = SubtitleStage(
        options={"preset": "neon"},
        timeline=_timeline(_subtitle_language("en", ("Hello", 0.0, 0.5))),
    )

    with pytest.raises(ConfigError, match="unknown subtitle style"):
        _run(stage.run(_context(tmp_path)))

    assert not (tmp_path / "subtitle.en.ass").exists()


def test_stage_uses_an_explicit_platform_object(tmp_path: Path) -> None:
    platform = Platform(
        id="demo",
        safe_area=SafeArea(top=0.05, bottom=0.25, left=0.10, right=0.10),
    )
    stage = SubtitleStage(
        options={
            "platform": platform,
            "width": 1000,
            "height": 1000,
            "highlight": False,
        },
        timeline=_timeline(_subtitle_language("en", ("Hello", 0.0, 0.5))),
    )

    result = cast(SubtitleStageResult, _run(stage.run(_context(tmp_path))))

    assert result.metadata["platform"] == "demo"
    ass = (tmp_path / "subtitle.en.ass").read_text(encoding="utf-8")
    assert "Dialogue: 0,0:00:00.00,0:00:00.50,SubtitleRegular32,,100,100,250,," in ass

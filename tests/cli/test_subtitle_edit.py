"""``aivia subtitle edit``: YAML words and timings, then affected re-burns."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from aivia.cli import main
from aivia.cli import subtitle as subtitle_cli
from aivia.core.db import (
    RunRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.models import Run, RunState
from aivia.core.run_dirs import RunDirectories


def _words(*pairs: tuple[str, float, float]) -> list[dict[str, Any]]:
    return [
        {"word": word, "start_seconds": start, "end_seconds": end}
        for word, start, end in pairs
    ]


def _timeline_document(
    *,
    platforms: list[str] | None = None,
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "id": "run_subtitle",
        "languages": [
            {
                "language": "en",
                "voices": [
                    {
                        "id": "en_voice",
                        "source": "audio/en.mp3",
                        "start_seconds": 0.0,
                        "end_seconds": 2.0,
                        "words": _words(("Hello", 0.0, 0.5), ("world", 0.6, 1.2)),
                    }
                ],
            },
            {
                "language": "de",
                "voices": [
                    {
                        "id": "de_voice",
                        "source": "audio/de.mp3",
                        "start_seconds": 0.0,
                        "end_seconds": 2.0,
                        "words": _words(("Hallo", 0.0, 0.5), ("Welt", 0.6, 1.2)),
                    }
                ],
            },
        ],
    }
    if platforms is not None:
        document["platforms"] = platforms
    return document


def _write_timeline(
    tmp_path: Path,
    document: dict[str, Any] | None = None,
) -> Path:
    path = tmp_path / "timeline.json"
    path.write_text(
        json.dumps(document if document is not None else _timeline_document()),
        encoding="utf-8",
    )
    return path


def _run(
    capsys: pytest.CaptureFixture[str],
    *args: str,
) -> tuple[int, str, str]:
    code = main(["subtitle", *args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _export(
    tmp_path: Path,
    timeline: Path,
    capsys: pytest.CaptureFixture[str] | None = None,
) -> dict[str, Any]:
    target = tmp_path / "subtitles.yaml"
    code = main(
        [
            "subtitle",
            "edit",
            "--timeline",
            str(timeline),
            "--output",
            str(target),
        ]
    )
    assert code == 0
    if capsys is not None:
        capsys.readouterr()
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_edit(
    tmp_path: Path,
    document: dict[str, Any],
    name: str = "edit.yaml",
) -> Path:
    path = tmp_path / name
    path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def _set_word(
    document: dict[str, Any],
    language: str,
    index: int,
    **values: Any,
) -> dict[str, Any]:
    for entry in document["languages"]:
        if entry["language"] == language:
            entry["words"][index].update(values)
            return document
    raise AssertionError(f"language {language!r} is missing")


def test_edit_exports_words_and_timings_as_yaml(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    timeline = _write_timeline(tmp_path)

    code, out, err = _run(
        capsys,
        "edit",
        "--timeline",
        str(timeline),
        "--language",
        "en",
    )

    assert code == 0
    assert err == ""
    document = yaml.safe_load(out)
    assert [entry["language"] for entry in document["languages"]] == ["en"]
    assert document["languages"][0]["words"] == _words(
        ("Hello", 0.0, 0.5), ("world", 0.6, 1.2)
    )


def test_edit_writes_the_export_to_a_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    timeline = _write_timeline(tmp_path)
    target = tmp_path / "subtitles.yaml"

    code, out, err = _run(
        capsys,
        "edit",
        "--timeline",
        str(timeline),
        "--output",
        str(target),
    )

    assert code == 0
    assert err == ""
    assert f"Wrote: {target}" in out
    document = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert [entry["language"] for entry in document["languages"]] == ["en", "de"]


def test_apply_reburns_only_the_affected_language(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeline = _write_timeline(tmp_path)
    document = _export(tmp_path, timeline, capsys)
    _set_word(document, "en", 0, word="Hi")
    edit = _write_edit(tmp_path, document)
    calls: list[tuple[str, Path]] = []

    def fake_video(composition: Any, path: Any, **options: Any) -> None:
        calls.append((composition.language, Path(path)))
        Path(path).write_bytes(f"video-{composition.language}".encode())

    monkeypatch.setattr(subtitle_cli, "render_video", fake_video)

    code, out, err = _run(
        capsys,
        "edit",
        "--timeline",
        str(timeline),
        "--input",
        str(edit),
    )

    assert code == 0, err
    assert err == ""
    assert [language for language, _path in calls] == ["en"]
    assert "Affected: en" in out
    assert "Subtitle: subtitle.en.ass" in out
    assert "Rendered:" in out

    payload = json.loads(timeline.read_text(encoding="utf-8"))
    english = next(entry for entry in payload["languages"] if entry["language"] == "en")
    german = next(entry for entry in payload["languages"] if entry["language"] == "de")
    assert english["subtitles"]["words"][0]["word"] == "Hi"
    assert german["subtitles"] is None

    ass = (tmp_path / "subtitle.en.ass").read_text(encoding="utf-8")
    assert "Hi" in ass
    assert "world" in ass
    assert "Hello" not in ass
    assert not (tmp_path / "subtitle.de.ass").exists()
    assert (tmp_path / "renders" / "run_subtitle_en.mp4").read_bytes() == b"video-en"
    assert not (tmp_path / "renders" / "run_subtitle_de.mp4").exists()


def test_apply_regenerates_every_platform_subtitle_for_a_changed_language(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeline = _write_timeline(
        tmp_path,
        _timeline_document(platforms=["tiktok", "youtube"]),
    )
    for language in ("en", "de"):
        for platform in ("tiktok", "youtube"):
            (tmp_path / f"subtitle.{language}.{platform}.ass").write_text(
                f"stale {language} {platform} Hello",
                encoding="utf-8",
            )
    document = _export(tmp_path, timeline, capsys)
    _set_word(document, "en", 0, word="Hi")
    edit = _write_edit(tmp_path, document)
    calls: list[tuple[str, str | None]] = []

    def fake_video(composition: Any, path: Any, **options: Any) -> None:
        calls.append((composition.language, composition.platform))
        Path(path).write_bytes(f"video-{composition.id}".encode())

    monkeypatch.setattr(subtitle_cli, "render_video", fake_video)

    code, out, err = _run(
        capsys,
        "edit",
        "--timeline",
        str(timeline),
        "--input",
        str(edit),
    )

    assert code == 0, err
    assert err == ""
    assert "Affected: en" in out
    assert sorted(calls) == [("en", "tiktok"), ("en", "youtube")]
    for platform in ("tiktok", "youtube"):
        ass = (tmp_path / f"subtitle.en.{platform}.ass").read_text(encoding="utf-8")
        assert "Hi" in ass
        assert "Hello" not in ass
    for platform in ("tiktok", "youtube"):
        stale = (tmp_path / f"subtitle.de.{platform}.ass").read_text(encoding="utf-8")
        assert stale == f"stale de {platform} Hello"
        assert "Hi" not in stale
    assert not (tmp_path / "subtitle.en.ass").exists()


def test_applying_an_edit_registers_edited_subtitles_and_renders_in_the_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeline = _write_timeline(tmp_path)
    document = _export(tmp_path, timeline, capsys)
    _set_word(document, "en", 0, word="Hi")
    edit = _write_edit(tmp_path, document)

    def fake_video(composition: Any, path: Any, **options: Any) -> None:
        Path(path).write_bytes(f"video-{composition.language}".encode())

    monkeypatch.setattr(subtitle_cli, "render_video", fake_video)

    code, _out, err = _run(
        capsys,
        "edit",
        "--timeline",
        str(timeline),
        "--input",
        str(edit),
    )

    assert code == 0, err
    assert err == ""
    manifest = json.loads((tmp_path / "artifacts.json").read_text(encoding="utf-8"))
    subtitles = manifest["stages"]["subtitle"]
    renders = manifest["stages"]["render"]
    assert [entry["path"] for entry in manifest["stages"]["voice"]] == ["timeline.json"]
    assert [entry["path"] for entry in subtitles] == ["subtitle.en.ass"]
    assert [entry["path"] for entry in renders] == ["renders/run_subtitle_en.mp4"]
    for entry in [*subtitles, *renders]:
        artifact = tmp_path / entry["path"]
        assert entry["size_bytes"] == artifact.stat().st_size
        assert entry["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()


def test_applying_an_edit_registers_the_timeline_under_the_voice_stage(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
database:
  filename: aivia.db
""".lstrip(),
        encoding="utf-8",
    )
    database = tmp_path / "data" / "aivia.db"
    upgrade_database(database)
    engine = create_database_engine(database)
    try:
        run = Run(id="run_subtitle", format="sample", state=RunState.QUEUED)
        RunDirectories(tmp_path / "data").create(run)
        RunRepository(engine).create(run)
    finally:
        engine.dispose()
    timeline = Path(run.run_dir) / "timeline.json"
    timeline.write_text(json.dumps(_timeline_document()), encoding="utf-8")
    document = _export(tmp_path, timeline, capsys)
    _set_word(document, "en", 0, word="Hi")
    edit = _write_edit(tmp_path, document)
    monkeypatch.setattr(
        subtitle_cli,
        "render_video",
        lambda composition, path, **options: Path(path).write_bytes(b"video"),
    )

    code, _out, err = _run(
        capsys,
        "edit",
        run.id,
        "--input",
        str(edit),
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0, err
    assert err == ""
    manifest = json.loads(
        (Path(run.run_dir) / "artifacts.json").read_text(encoding="utf-8")
    )
    assert [entry["path"] for entry in manifest["stages"]["voice"]] == ["timeline.json"]
    assert [entry["path"] for entry in manifest["stages"]["subtitle"]] == [
        "subtitle.en.ass"
    ]
    assert [entry["path"] for entry in manifest["stages"]["render"]] == [
        "renders/run_subtitle_en.mp4"
    ]


def test_reapplying_a_previous_edit_is_served_from_the_render_cache(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeline = _write_timeline(tmp_path)
    original = _export(tmp_path, timeline, capsys)
    changed = _export(tmp_path, timeline, capsys)
    _set_word(changed, "en", 0, word="Hi")
    changed_path = _write_edit(tmp_path, changed, "changed.yaml")
    original_path = _write_edit(tmp_path, original, "original.yaml")
    calls: list[str] = []

    def fake_video(composition: Any, path: Any, **options: Any) -> None:
        calls.append(composition.language)
        Path(path).write_bytes(f"render-{len(calls)}".encode())

    monkeypatch.setattr(subtitle_cli, "render_video", fake_video)

    assert (
        _run(
            capsys,
            "edit",
            "--timeline",
            str(timeline),
            "--input",
            str(changed_path),
        )[0]
        == 0
    )
    assert (
        _run(
            capsys,
            "edit",
            "--timeline",
            str(timeline),
            "--input",
            str(original_path),
        )[0]
        == 0
    )
    code, out, err = _run(
        capsys,
        "edit",
        "--timeline",
        str(timeline),
        "--input",
        str(changed_path),
    )

    assert code == 0
    assert err == ""
    assert calls == ["en", "en"]
    assert "Cached:" in out
    output = tmp_path / "renders" / "run_subtitle_en.mp4"
    assert output.read_bytes() == b"render-1"


def test_an_invalid_timing_is_rejected_without_touching_the_timeline(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeline = _write_timeline(tmp_path)
    before = timeline.read_text(encoding="utf-8")
    document = _export(tmp_path, timeline, capsys)
    _set_word(document, "en", 0, end_seconds=-1.0)
    edit = _write_edit(tmp_path, document)
    calls: list[Any] = []
    monkeypatch.setattr(
        subtitle_cli,
        "render_video",
        lambda composition, path, **options: calls.append(composition),
    )

    code, out, err = _run(
        capsys,
        "edit",
        "--timeline",
        str(timeline),
        "--input",
        str(edit),
    )

    assert code == 1
    assert out == ""
    assert "invalid subtitle word" in err
    assert calls == []
    assert timeline.read_text(encoding="utf-8") == before


def test_an_unknown_language_is_rejected(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    timeline = _write_timeline(tmp_path)
    edit = _write_edit(
        tmp_path,
        {"languages": [{"language": "fr", "words": _words(("Bonjour", 0.0, 0.5))}]},
    )

    code, out, err = _run(
        capsys,
        "edit",
        "--timeline",
        str(timeline),
        "--input",
        str(edit),
    )

    assert code == 1
    assert out == ""
    assert "timeline has no such language" in err


def test_edit_reports_a_missing_timeline_and_a_missing_selector(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.json"
    code, out, err = _run(capsys, "edit", "--timeline", str(missing))
    assert code == 1
    assert out == ""
    assert "timeline not found" in err

    code, out, err = _run(capsys, "edit")
    assert code == 1
    assert out == ""
    assert "a run id or --timeline is required" in err


def test_edit_resolves_the_timeline_from_a_run_id(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
database:
  filename: aivia.db
""".lstrip(),
        encoding="utf-8",
    )
    database = tmp_path / "data" / "aivia.db"
    upgrade_database(database)
    engine = create_database_engine(database)
    try:
        run = Run(id="run_subtitle", format="sample", state=RunState.QUEUED)
        RunDirectories(tmp_path / "data").create(run)
        RunRepository(engine).create(run)
    finally:
        engine.dispose()
    (Path(run.run_dir) / "timeline.json").write_text(
        json.dumps(_timeline_document()),
        encoding="utf-8",
    )

    code, out, err = _run(
        capsys,
        "edit",
        "run_subtitle",
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    document = yaml.safe_load(out)
    assert [entry["language"] for entry in document["languages"]] == ["en", "de"]

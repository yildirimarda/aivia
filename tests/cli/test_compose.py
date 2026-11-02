"""``aivia compose``: validate, schema and render commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from aivia.cli import compose as compose_cli
from aivia.cli import main
from aivia.compose import dump_composition_schema


def _composition(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "canvas": {"width": 64, "height": 48, "background": "#101010"},
        "duration_seconds": 2.0,
        "fps": 10,
        "layers": [],
    }
    values.update(overrides)
    return values


def _write(path: Path, document: dict[str, Any]) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _run(
    capsys: pytest.CaptureFixture[str],
    *args: str,
) -> tuple[int, str, str]:
    code = main(["compose", *args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_validate_accepts_a_valid_composition(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write(tmp_path / "composition.json", _composition())

    code, output, err = _run(capsys, "validate", str(source))

    assert code == 0
    assert output.strip() == "composition is valid"
    assert err == ""


def test_validate_prints_every_cross_object_issue(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write(
        tmp_path / "composition.json",
        _composition(
            layers=[
                {"id": "same", "type": "text", "text": "one", "start": 0},
                {"id": "same", "type": "text", "text": "two", "start": 5},
            ]
        ),
    )

    code, output, err = _run(capsys, "validate", str(source))

    assert code == 1
    assert output == ""
    assert "layers[1].id" in err
    assert "duplicate layer id" in err
    assert "layers[1]" in err
    assert "duration" in err


def test_validate_reports_schema_errors_with_their_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write(
        tmp_path / "composition.json",
        {**_composition(), "fps": "fast"},
    )

    code, output, err = _run(capsys, "validate", str(source))

    assert code == 1
    assert output == ""
    assert "fps" in err
    assert "error:" in err


def test_validate_json_prints_every_issue_as_data(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write(
        tmp_path / "composition.json",
        _composition(
            layers=[
                {"id": "same", "type": "text", "text": "one"},
                {"id": "same", "type": "text", "text": "two"},
            ]
        ),
    )

    code, output, err = _run(capsys, "validate", str(source), "--json")

    assert code == 1
    assert err == ""
    issues = json.loads(output)
    assert issues[0]["path"] == "layers[1].id"
    assert issues[0]["code"] == "duplicate_id"


def test_validate_checks_assets_when_a_directory_is_given(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "present.png").write_bytes(b"image")
    source = _write(
        tmp_path / "composition.json",
        _composition(
            layers=[
                {"id": "present", "type": "image", "source": "present.png"},
                {"id": "missing", "type": "image", "source": "missing.png"},
            ]
        ),
    )

    code, output, err = _run(capsys, "validate", str(source), "--assets", str(assets))

    assert code == 1
    assert output == ""
    assert "missing.png" in err
    assert "present.png" not in err


def test_validate_reports_a_missing_file_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, output, err = _run(capsys, "validate", str(tmp_path / "missing.yaml"))

    assert code == 1
    assert output == ""
    assert "not found" in err
    assert "Traceback" not in err


def test_validate_resolves_relative_assets_against_the_document_directory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = tmp_path / "documents"
    assets = documents / "assets"
    assets.mkdir(parents=True)
    (assets / "present.png").write_bytes(b"image")
    source = _write(
        documents / "composition.json",
        _composition(
            layers=[{"id": "present", "type": "image", "source": "assets/present.png"}]
        ),
    )
    monkeypatch.chdir(tmp_path)

    code, output, err = _run(capsys, "validate", str(source))

    assert code == 0
    assert output.strip() == "composition is valid"
    assert err == ""


def test_validate_reports_missing_assets_relative_to_the_document_directory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = tmp_path / "documents"
    assets = documents / "assets"
    assets.mkdir(parents=True)
    (assets / "present.png").write_bytes(b"image")
    (tmp_path / "missing.png").write_bytes(b"decoy in the working directory")
    source = _write(
        documents / "composition.json",
        _composition(
            layers=[
                {"id": "present", "type": "image", "source": "assets/present.png"},
                {"id": "missing", "type": "image", "source": "missing.png"},
            ]
        ),
    )
    monkeypatch.chdir(tmp_path)

    code, output, err = _run(capsys, "validate", str(source))

    assert code == 1
    assert output == ""
    assert "missing.png" in err
    assert "present.png" not in err


def test_schema_prints_the_deterministic_composition_schema(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, output, err = _run(capsys, "schema")

    assert code == 0
    assert err == ""
    assert output == dump_composition_schema()
    assert json.loads(output)["title"] == "AIVIA composition"


def test_schema_writes_to_the_selected_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "schema" / "composition.schema.json"

    code, output, err = _run(capsys, "schema", "--output", str(target))

    assert code == 0
    assert err == ""
    assert f"wrote {target}" in output
    assert target.read_text(encoding="utf-8") == dump_composition_schema()


def test_render_refuses_an_invalid_composition(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write(
        tmp_path / "composition.json",
        _composition(
            layers=[
                {"id": "same", "type": "text", "text": "one"},
                {"id": "same", "type": "text", "text": "two"},
            ]
        ),
    )
    called = False

    def fake_video(composition: Any, path: Any, **options: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(compose_cli, "render_video", fake_video)

    code, output, err = _run(capsys, "render", str(source), str(tmp_path / "out.mp4"))

    assert code == 1
    assert output == ""
    assert "duplicate layer id" in err
    assert called is False


def test_render_selects_the_video_renderer_from_the_output_suffix(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write(tmp_path / "composition.json", _composition())
    output = tmp_path / "out.mp4"
    calls: list[tuple[Any, Path, dict[str, Any]]] = []

    def fake_video(composition: Any, path: Any, **options: Any) -> None:
        calls.append((composition, Path(path), options))
        Path(path).write_bytes(b"video")

    monkeypatch.setattr(compose_cli, "render_video", fake_video)

    code, stdout, err = _run(capsys, "render", str(source), "--output", str(output))

    assert code == 0
    assert err == ""
    assert calls and calls[0][1] == output
    assert calls[0][2] == {}
    assert output.read_bytes() == b"video"
    assert f"Output: {output}" in stdout


def test_render_selects_gif_and_audio_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write(tmp_path / "composition.json", _composition())
    calls: list[Path] = []

    def fake_video(composition: Any, path: Any, **options: Any) -> None:
        calls.append(Path(path))
        Path(path).write_bytes(b"render")

    monkeypatch.setattr(compose_cli, "render_video", fake_video)

    gif = tmp_path / "out.gif"
    audio = tmp_path / "out"
    assert _run(capsys, "render", str(source), str(gif))[0] == 0
    assert (
        _run(
            capsys,
            "render",
            str(source),
            str(audio),
            "--format",
            "audio",
        )[0]
        == 0
    )

    assert calls == [gif, audio.with_suffix(".mp3")]
    assert audio.with_suffix(".mp3").is_file()


def test_render_preview_passes_reduction_options(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write(tmp_path / "composition.json", _composition())
    output = tmp_path / "preview.mp4"
    calls: list[tuple[Any, Path, dict[str, Any]]] = []

    def fake_preview(composition: Any, path: Any, **options: Any) -> None:
        calls.append((composition, Path(path), options))
        Path(path).write_bytes(b"preview")

    monkeypatch.setattr(compose_cli, "render_preview", fake_preview)

    code, stdout, err = _run(
        capsys,
        "render",
        str(source),
        "--output",
        str(output),
        "--preview",
        "--scale",
        "0.5",
        "--max-fps",
        "6",
    )

    assert code == 0
    assert err == ""
    assert len(calls) == 1
    assert calls[0][1] == output
    assert calls[0][2] == {"scale": 0.5, "max_fps": 6}
    assert output.read_bytes() == b"preview"
    assert f"Output: {output}" in stdout


def test_render_still_writes_the_selected_image(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write(tmp_path / "composition.json", _composition())
    output = tmp_path / "frame.png"

    code, stdout, err = _run(capsys, "render", str(source), str(output))

    assert code == 0
    assert err == ""
    with Image.open(output) as image:
        assert image.size == (64, 48)
    assert f"Output: {output}" in stdout


def test_render_resolves_relative_assets_against_the_document_directory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = tmp_path / "documents"
    assets = documents / "assets"
    assets.mkdir(parents=True)
    Image.new("RGBA", (8, 8), (0, 128, 0, 255)).save(assets / "background.png")
    source = _write(
        documents / "composition.json",
        _composition(
            canvas={"width": 8, "height": 8, "background": "#000000"},
            layers=[
                {
                    "id": "background",
                    "type": "image",
                    "source": "assets/background.png",
                    "fit": "stretch",
                }
            ],
        ),
    )
    output = tmp_path / "frame.png"
    monkeypatch.chdir(tmp_path)

    code, _, err = _run(capsys, "render", str(source), str(output))

    assert code == 0
    assert err == ""
    with Image.open(output) as image:
        assert image.convert("RGB").getpixel((0, 0)) == (0, 128, 0)


def test_render_resolves_relative_assets_before_the_video_renderer(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = tmp_path / "documents"
    media = documents / "media"
    media.mkdir(parents=True)
    (media / "clip.mp4").write_bytes(b"clip")
    (media / "voice.mp3").write_bytes(b"voice")
    source = _write(
        documents / "composition.json",
        _composition(
            layers=[{"id": "clip", "type": "video", "source": "media/clip.mp4"}],
            audio_tracks=[{"id": "voice", "source": "media/voice.mp3"}],
        ),
    )
    output = tmp_path / "out.mp4"
    seen: dict[str, Any] = {}

    def fake_video(composition: Any, path: Any, **options: Any) -> None:
        seen["composition"] = composition
        Path(path).write_bytes(b"video")

    monkeypatch.setattr(compose_cli, "render_video", fake_video)
    monkeypatch.chdir(tmp_path)

    code, _, err = _run(capsys, "render", str(source), str(output))

    assert code == 0
    assert err == ""
    resolved = seen["composition"]
    assert resolved.layers[0].source == str((media / "clip.mp4").resolve())
    assert resolved.audio_tracks[0].source == str((media / "voice.mp3").resolve())


def test_render_still_preview_reduces_the_image(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write(tmp_path / "composition.json", _composition())
    output = tmp_path / "preview.png"

    code, _, err = _run(
        capsys,
        "render",
        str(source),
        str(output),
        "--preview",
        "--scale",
        "0.5",
    )

    assert code == 0
    assert err == ""
    with Image.open(output) as image:
        assert image.size == (32, 24)


def test_render_writes_a_web_card(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write(tmp_path / "composition.json", _composition())
    output = tmp_path / "card.html"

    code, _, err = _run(capsys, "render", str(source), str(output))

    assert code == 0
    assert err == ""
    assert "<html" in output.read_text(encoding="utf-8")


def test_render_passes_platform_and_profile_options(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write(tmp_path / "composition.json", _composition())
    output = tmp_path / "out.mp4"
    seen: dict[str, Any] = {}

    def fake_video(composition: Any, path: Any, **options: Any) -> None:
        seen.update(options)
        Path(path).write_bytes(b"video")

    monkeypatch.setattr(compose_cli, "render_video", fake_video)

    code, _, err = _run(
        capsys,
        "render",
        str(source),
        str(output),
        "--platform",
        "tiktok",
        "--profile",
        "short",
    )

    assert code == 0
    assert err == ""
    assert seen == {"platform": "tiktok", "profile": "short"}


def test_render_rejects_an_unknown_output_extension(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write(tmp_path / "composition.json", _composition())

    code, output, err = _run(
        capsys, "render", str(source), str(tmp_path / "out.unknown")
    )

    assert code == 1
    assert output == ""
    assert "unsupported output extension" in err


def test_render_rejects_a_preview_for_an_animated_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write(tmp_path / "composition.json", _composition())

    code, output, err = _run(
        capsys,
        "render",
        str(source),
        str(tmp_path / "out.gif"),
        "--preview",
    )

    assert code == 1
    assert output == ""
    assert "preview" in err


def test_render_reuses_the_cached_render_for_an_unchanged_composition(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write(tmp_path / "composition.json", _composition())
    output = tmp_path / "out.mp4"
    calls: list[Path] = []

    def fake_video(composition: Any, path: Any, **options: Any) -> None:
        calls.append(Path(path))
        Path(path).write_bytes(f"video-{len(calls)}".encode())

    monkeypatch.setattr(compose_cli, "render_video", fake_video)

    assert _run(capsys, "render", str(source), str(output))[0] == 0
    output.unlink()
    code, stdout, err = _run(capsys, "render", str(source), str(output))

    assert code == 0
    assert err == ""
    assert len(calls) == 1
    assert output.read_bytes() == b"video-1"
    assert f"Output: {output}" in stdout


def test_render_no_cache_flag_forces_a_fresh_render(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write(tmp_path / "composition.json", _composition())
    output = tmp_path / "out.mp4"
    calls: list[Path] = []

    def fake_video(composition: Any, path: Any, **options: Any) -> None:
        calls.append(Path(path))
        Path(path).write_bytes(f"video-{len(calls)}".encode())

    monkeypatch.setattr(compose_cli, "render_video", fake_video)

    assert _run(capsys, "render", str(source), str(output))[0] == 0
    assert _run(capsys, "render", str(source), str(output), "--no-cache")[0] == 0
    assert len(calls) == 2
    assert output.read_bytes() == b"video-2"

    assert _run(capsys, "render", str(source), str(output))[0] == 0
    assert len(calls) == 2
    assert output.read_bytes() == b"video-2"


def test_render_cache_misses_when_the_composition_changes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write(tmp_path / "composition.json", _composition())
    output = tmp_path / "out.mp4"
    calls: list[Path] = []

    def fake_video(composition: Any, path: Any, **options: Any) -> None:
        calls.append(Path(path))
        Path(path).write_bytes(f"video-{len(calls)}".encode())

    monkeypatch.setattr(compose_cli, "render_video", fake_video)

    assert _run(capsys, "render", str(source), str(output))[0] == 0
    _write(
        source,
        _composition(canvas={"width": 32, "height": 24, "background": "#202020"}),
    )
    assert _run(capsys, "render", str(source), str(output))[0] == 0

    assert len(calls) == 2
    assert output.read_bytes() == b"video-2"


def test_compose_without_a_subcommand_prints_usage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, output, err = _run(capsys)

    assert code == 2
    assert output == ""
    assert "usage: aivia compose" in err


def test_render_auto_validates_a_still_against_the_selected_platform(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write(tmp_path / "composition.json", _composition())
    output = tmp_path / "frame.png"

    code, stdout, err = _run(
        capsys,
        "render",
        str(source),
        str(output),
        "--platform",
        "instagram",
    )

    assert code == 1
    assert stdout == ""
    assert "Instagram Reels image resolution 64x48" in err


def test_render_prints_every_platform_issue_returned_for_the_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aivia.compose import OutputIssue, OutputIssues

    source = _write(tmp_path / "composition.json", _composition())
    output = tmp_path / "out.mp4"

    def fake_video(composition: Any, path: Any, **options: Any) -> None:
        Path(path).write_bytes(b"video")

    def fake_validate(rendered: Any, platform: str, **options: Any) -> OutputIssues:
        assert Path(rendered) == output
        assert platform == "youtube"
        return OutputIssues(
            [
                OutputIssue(
                    path="duration",
                    message="YouTube Shorts video duration 200s is too long",
                    code="duration",
                    media_type="video",
                ),
                OutputIssue(
                    path="resolution",
                    message="YouTube Shorts video resolution 1920x1080 is not allowed",
                    code="resolution",
                    media_type="video",
                ),
            ]
        )

    monkeypatch.setattr(compose_cli, "render_video", fake_video)
    monkeypatch.setattr(compose_cli, "validate_output", fake_validate)

    code, stdout, err = _run(
        capsys,
        "render",
        str(source),
        str(output),
        "--platform",
        "youtube",
        "--validate",
    )

    assert code == 1
    assert stdout == ""
    assert "duration" in err
    assert "YouTube Shorts video duration 200s is too long" in err
    assert "resolution" in err
    assert "YouTube Shorts video resolution 1920x1080 is not allowed" in err


def test_render_passes_platform_validation_when_the_output_fits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write(tmp_path / "composition.json", _composition())
    output = tmp_path / "card.html"

    code, stdout, err = _run(
        capsys,
        "render",
        str(source),
        str(output),
        "--platform",
        "telegram",
    )

    assert code == 0
    assert err == ""
    assert f"Output: {output}" in stdout


def test_render_validate_requires_a_platform(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write(tmp_path / "composition.json", _composition())

    code, stdout, err = _run(
        capsys,
        "render",
        str(source),
        str(tmp_path / "out.png"),
        "--validate",
    )

    assert code == 1
    assert stdout == ""
    assert "output validation requires --platform" in err


def test_validate_output_command_prints_every_issue_as_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "frame.png"
    Image.new("RGBA", (64, 48), (0, 0, 0, 255)).save(output)

    code, stdout, err = _run(
        capsys,
        "validate-output",
        str(output),
        "--platform",
        "instagram",
        "--json",
    )

    assert code == 1
    assert err == ""
    issues = json.loads(stdout)
    assert issues[0]["path"] == "resolution"
    assert issues[0]["media_type"] == "image"


def test_render_preview_skips_automatic_platform_validation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write(tmp_path / "composition.json", _composition())
    output = tmp_path / "preview.png"

    code, stdout, err = _run(
        capsys,
        "render",
        str(source),
        str(output),
        "--preview",
        "--scale",
        "0.5",
        "--platform",
        "instagram",
    )

    assert code == 0
    assert err == ""
    assert f"Output: {output}" in stdout

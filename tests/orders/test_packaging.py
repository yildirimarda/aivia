"""Packaging order outputs into pinned finals at size presets.

The packager turns rendered masters (video, audio, image, composition) into the
order's finals: MP4, animated GIF, animated WebP, MP3, poster and web card. Every
final is pinned so retention never removes it. ffmpeg is injected so the suite
stays offline and fast.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from aivia.orders.packaging import (
    DEFAULT_SIZE_PRESETS,
    Packager,
    PackagingError,
    SizePreset,
    package_outputs,
)

MP4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 12

_BUFFER = BytesIO()
Image.new("RGB", (8, 8), (255, 0, 0)).save(_BUFFER, format="PNG")
REAL_PNG = _BUFFER.getvalue()

COMPOSITION = {
    "canvas": {"width": 1080, "height": 1920, "background": "#111111"},
    "duration_seconds": 5.0,
    "layers": [],
}

EXPECTED_KINDS = {"mp4", "gif", "webp", "mp3", "poster", "web_card"}


def _ffmpeg_writer(
    *,
    calls: list[list[str]] | None = None,
    payload: bytes = REAL_PNG,
) -> Any:
    def run(argv: Sequence[str]) -> subprocess.CompletedProcess[Any]:
        if calls is not None:
            calls.append(list(argv))
        target = Path(argv[-1])
        target.write_bytes(payload)
        return subprocess.CompletedProcess(list(argv), 0, stdout=b"", stderr=b"")

    return run


def _write_video(destination: Path) -> Path:
    destination.write_bytes(MP4)
    return destination


def _write_image(destination: Path) -> Path:
    image = Image.new("RGB", (200, 100), (255, 0, 0))
    image.save(destination, format="PNG")
    return destination


def _packager(tmp_path: Path, **kwargs: Any) -> Packager:
    return Packager(tmp_path / "packaged", **kwargs)


def _all_kinds() -> list[str]:
    return sorted(EXPECTED_KINDS)


def _extension(kind: str) -> str:
    return {
        "mp4": "mp4",
        "gif": "gif",
        "webp": "webp",
        "mp3": "mp3",
        "poster": "png",
        "web_card": "html",
    }[kind]


def test_package_all_produces_every_packed_format(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    packager = _packager(tmp_path, ffmpeg_runner=_ffmpeg_writer(calls=calls))
    _write_video(tmp_path / "master.mp4")
    _write_image(tmp_path / "background.png")

    outputs = packager.package_all(
        video=tmp_path / "master.mp4",
        image=tmp_path / "background.png",
        composition=COMPOSITION,
        formats=_all_kinds(),
    )

    assert set(outputs) == EXPECTED_KINDS
    assert len(calls) >= 3  # mp4, gif and webp each invoke ffmpeg
    for kind, output in outputs.items():
        assert output.path.is_file()
        assert output.path.stat().st_size > 0
        assert output.extension == _extension(kind)
        assert output.format == _extension(kind)
        assert output.pinned is True


def test_packaged_outputs_are_marked_pinned(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    packager = _packager(tmp_path, ffmpeg_runner=_ffmpeg_writer(calls=calls))
    _write_video(tmp_path / "master.mp4")

    outputs = packager.package_all(
        video=tmp_path / "master.mp4",
        composition=COMPOSITION,
    )

    assert outputs
    for output in outputs.values():
        assert output.pinned is True
        assert output.is_pinned is True


def test_packaged_output_records_kind_preset_and_dimensions(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    packager = _packager(tmp_path, ffmpeg_runner=_ffmpeg_writer(calls=calls))
    _write_video(tmp_path / "master.mp4")

    output = packager.package_mp4(tmp_path / "master.mp4", preset="portrait")

    assert output.kind == "mp4"
    assert output.preset == "portrait"
    assert (output.width, output.height) == (1080, 1350)
    assert output.pinned is True


def _scale_expression(width: int, height: int) -> str:
    return (
        f"scale=trunc({width}*1.000000/2)*2:trunc({height}*1.000000/2)*2:flags=lanczos"
    )


def test_size_preset_applies_scale_to_video_outputs(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    packager = _packager(tmp_path, ffmpeg_runner=_ffmpeg_writer(calls=calls))
    _write_video(tmp_path / "master.mp4")

    packager.package_mp4(tmp_path / "master.mp4", preset="story")
    assert _scale_expression(1080, 1920) in " ".join(calls[0])

    calls.clear()
    packager.package_gif(tmp_path / "master.mp4", preset="landscape")
    assert _scale_expression(1920, 1080) in " ".join(calls[0])

    calls.clear()
    packager.package_webp(tmp_path / "master.mp4", preset="square")
    assert _scale_expression(1080, 1080) in " ".join(calls[0])


def test_size_preset_resizes_poster_from_image(tmp_path: Path) -> None:
    packager = _packager(tmp_path)
    _write_image(tmp_path / "photo.png")

    output = packager.package_poster(tmp_path / "photo.png", preset="landscape")

    assert output.path.is_file()
    assert (output.width, output.height) == (1920, 1080)
    with Image.open(output.path) as saved:
        assert saved.size == (1920, 1080)


def test_poster_from_image_keeps_source_size_without_preset(tmp_path: Path) -> None:
    packager = _packager(tmp_path)
    _write_image(tmp_path / "photo.png")

    output = packager.package_poster(tmp_path / "photo.png")

    assert (output.width, output.height) == (200, 100)
    with Image.open(output.path) as saved:
        assert saved.size == (200, 100)


def test_poster_from_video_extracts_a_frame(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    packager = _packager(tmp_path, ffmpeg_runner=_ffmpeg_writer(calls=calls))
    _write_video(tmp_path / "master.mp4")

    output = packager.package_poster(tmp_path / "master.mp4")

    assert output.path.is_file()
    assert "-frames:v" in calls[0]
    assert "1" in calls[0]


def test_web_card_renders_self_contained_html(tmp_path: Path) -> None:
    packager = _packager(tmp_path)
    _write_image(tmp_path / "background.png")

    output = packager.package_web_card(
        COMPOSITION,
        output=tmp_path / "packaged" / "card.html",
        background_image=tmp_path / "background.png",
        title="Hello",
        summary="A packaged card.",
    )

    assert output.path.is_file()
    document = output.path.read_text(encoding="utf-8")
    assert document.startswith("<!doctype html>")
    assert "data:image/png;base64," in document
    assert "Hello" in document
    assert "A packaged card." in document
    assert output.pinned is True


def test_web_card_with_preset_resizes_background(tmp_path: Path) -> None:
    packager = _packager(tmp_path)
    _write_image(tmp_path / "background.png")

    output = packager.package_web_card(
        COMPOSITION,
        preset="story",
        background_image=tmp_path / "background.png",
    )

    assert output.path.is_file()
    document = output.path.read_text(encoding="utf-8")
    assert document.startswith("<!doctype html>")
    assert output.preset == "story"
    assert (output.width, output.height) == (1080, 1920)


def test_unknown_output_kind_is_rejected(tmp_path: Path) -> None:
    packager = _packager(tmp_path, ffmpeg_runner=_ffmpeg_writer())
    _write_video(tmp_path / "master.mp4")

    with pytest.raises(PackagingError, match="unknown output kinds"):
        packager.package_all(
            video=tmp_path / "master.mp4",
            composition=COMPOSITION,
            formats=["mp4", "mov"],
        )


def test_unknown_size_preset_is_rejected(tmp_path: Path) -> None:
    packager = _packager(tmp_path, ffmpeg_runner=_ffmpeg_writer())
    _write_video(tmp_path / "master.mp4")

    with pytest.raises(PackagingError, match="unknown size preset"):
        packager.package_mp4(tmp_path / "master.mp4", preset="not-a-preset")


def test_missing_source_is_rejected(tmp_path: Path) -> None:
    packager = _packager(tmp_path)

    with pytest.raises(PackagingError, match="does not exist"):
        packager.package_mp4(tmp_path / "missing.mp4")


def test_destination_must_not_overwrite_source(tmp_path: Path) -> None:
    packager = _packager(tmp_path, ffmpeg_runner=_ffmpeg_writer())
    source = tmp_path / "master.mp4"
    _write_video(source)

    with pytest.raises(PackagingError, match="overwrite the source"):
        packager.package_mp4(source, output=source)


def test_mp3_from_audio_source(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    packager = _packager(tmp_path, ffmpeg_runner=_ffmpeg_writer(calls=calls))
    audio = tmp_path / "speech.mp3"
    audio.write_bytes(MP4)

    output = packager.package_mp3(audio)

    assert output.path.is_file()
    assert output.kind == "mp3"
    assert output.pinned is True
    assert any("-vn" in call for call in calls)


def test_mp3_uses_video_when_no_audio_source(tmp_path: Path) -> None:
    packager = _packager(tmp_path, ffmpeg_runner=_ffmpeg_writer())
    _write_video(tmp_path / "master.mp4")

    output = packager.package_mp3(tmp_path / "master.mp4")

    assert output.path.is_file()
    assert output.pinned is True


def test_convenience_function_packages_all_kinds(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    _write_video(tmp_path / "master.mp4")
    _write_image(tmp_path / "background.png")

    outputs = package_outputs(
        tmp_path / "out",
        video=tmp_path / "master.mp4",
        image=tmp_path / "background.png",
        composition=COMPOSITION,
        formats=_all_kinds(),
        ffmpeg_runner=_ffmpeg_writer(calls=calls),
    )

    assert set(outputs) == EXPECTED_KINDS
    assert all(output.pinned for output in outputs.values())
    assert calls


def test_default_presets_include_the_documented_sizes() -> None:
    assert DEFAULT_SIZE_PRESETS["square"].size == (1080, 1080)
    assert DEFAULT_SIZE_PRESETS["portrait"].size == (1080, 1350)
    assert DEFAULT_SIZE_PRESETS["story"].size == (1080, 1920)
    assert DEFAULT_SIZE_PRESETS["landscape"].size == (1920, 1080)


def test_preset_can_be_passed_as_a_sizepreset_instance(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    packager = _packager(tmp_path, ffmpeg_runner=_ffmpeg_writer(calls=calls))
    _write_video(tmp_path / "master.mp4")

    custom = SizePreset(name="custom", width=640, height=480, label="640x480")
    output = packager.package_mp4(tmp_path / "master.mp4", preset=custom)

    assert output.preset == "custom"
    assert (output.width, output.height) == (640, 480)


def test_bytes_source_is_packaged(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    packager = _packager(tmp_path, ffmpeg_runner=_ffmpeg_writer(calls=calls))

    output = packager.package_mp4(MP4)

    assert output.path.is_file()
    assert output.source.endswith(" bytes")
    assert calls

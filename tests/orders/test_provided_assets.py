"""Provided background, loop, audio and logo assets are accepted, validated and
fitted into new files while the originals are preserved."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from PIL import Image

from aivia.core.errors import ValidationError
from aivia.core.media_probe import MediaFormat, MediaLimits
from aivia.media.image_utils import ImageFit
from aivia.orders.provided_assets import (
    AcceptProvidedAssetsError,
    AssetKind,
    accept_provided_assets,
)

PNG = b"\x89PNG\r\n\x1a\n"
MP3 = b"ID3" + b"\x00" * 32
MPEG = b"\x00\x00\x01\xba" + b"\x00" * 32


def probe_json(
    duration: float,
    *,
    width: int | None = None,
    height: int | None = None,
    codec_type: str = "audio",
    codec_name: str = "mp3",
) -> str:
    streams: list[dict[str, object]] = [
        {
            "index": 0,
            "codec_type": codec_type,
            "codec_name": codec_name,
            "duration": f"{duration:.3f}",
        }
    ]
    if width is not None and height is not None:
        streams[0]["width"] = width
        streams[0]["height"] = height
    return json.dumps(
        {
            "streams": streams,
            "format": {
                "format_name": codec_name,
                "format_long_name": "synthetic",
                "duration": f"{duration:.3f}",
                "size": "12345",
            },
        }
    )


def ffprobe_runner(
    duration: float,
    *,
    width: int | None = None,
    height: int | None = None,
    codec_type: str = "audio",
    calls: list[list[str]] | None = None,
) -> Callable[[Sequence[str]], subprocess.CompletedProcess[str]]:
    def run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if calls is not None:
            calls.append(list(argv))
        return subprocess.CompletedProcess(
            list(argv),
            0,
            stdout=probe_json(
                duration, width=width, height=height, codec_type=codec_type
            ),
            stderr="",
        )

    return run


def ffmpeg_runner(
    payload: bytes = MP3,
    *,
    calls: list[list[str]] | None = None,
) -> Callable[[Sequence[str]], subprocess.CompletedProcess[bytes]]:
    def run(argv: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
        if calls is not None:
            calls.append(list(argv))
        Path(argv[-1]).write_bytes(payload)
        return subprocess.CompletedProcess(list(argv), 0, stdout=b"", stderr=b"")

    return run


def write_image(
    path: Path,
    size: tuple[int, int] = (100, 100),
    color: tuple[int, int, int] = (10, 150, 200),
    format: str = "PNG",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format=format)
    return path


def test_accepts_a_background_image_and_writes_a_fitted_copy(
    tmp_path: Path,
) -> None:
    source = write_image(tmp_path / "background.png", size=(200, 200))
    before = source.read_bytes()
    destination = tmp_path / "fitted"
    assets = accept_provided_assets(
        destination=destination,
        background=source,
        size=(800, 600),
    )

    assert assets.supplied is True
    assert assets.background is not None
    assert assets.background.source == source
    assert assets.background.fitted != source
    assert assets.background.fitted.is_file()
    assert assets.background.probe.format.value == "png"
    assert assets.background.probe.width == 200
    assert assets.background.probe.height == 200
    assert source.read_bytes() == before


def test_accepts_a_logo_image_and_preserves_the_original(
    tmp_path: Path,
) -> None:
    source = write_image(tmp_path / "logo.png", size=(50, 50))
    before = source.read_bytes()
    destination = tmp_path / "fitted"
    assets = accept_provided_assets(
        destination=destination,
        logo=source,
        size=(200, 200),
        fit=ImageFit.CONTAIN,
    )

    assert assets.logo is not None
    assert assets.logo.fitted.is_file()
    assert assets.logo.fitted != source
    assert source.read_bytes() == before


def test_accepts_audio_and_trims_it_to_the_requested_duration(
    tmp_path: Path,
) -> None:
    source = tmp_path / "track.mp3"
    source.write_bytes(MP3)
    before = source.read_bytes()
    destination = tmp_path / "fitted"
    calls: list[list[str]] = []
    assets = accept_provided_assets(
        destination=destination,
        audio=source,
        duration_seconds=3.0,
        ffprobe_runner=ffprobe_runner(10.0),
        ffmpeg_runner=ffmpeg_runner(calls=calls),
    )

    assert assets.audio is not None
    assert assets.audio.fitted.is_file()
    assert assets.audio.fitted != source
    assert assets.audio.probe.format.value == "mp3"
    assert assets.audio.probe.duration_seconds == pytest.approx(10.0)
    assert source.read_bytes() == before
    assert len(calls) == 1


def test_accepts_a_video_loop_and_scales_it_to_the_canvas(
    tmp_path: Path,
) -> None:
    source = tmp_path / "intro.mpeg"
    source.write_bytes(MPEG)
    before = source.read_bytes()
    destination = tmp_path / "fitted"
    calls: list[list[str]] = []
    assets = accept_provided_assets(
        destination=destination,
        loop=source,
        size=(1280, 720),
        ffprobe_runner=ffprobe_runner(5.0, width=1920, height=1080, codec_type="video"),
        ffmpeg_runner=ffmpeg_runner(calls=calls),
    )

    assert assets.loop is not None
    assert assets.loop.fitted.is_file()
    assert assets.loop.fitted != source
    assert assets.loop.probe.format.value == "mpeg"
    assert assets.loop.probe.width == 1920
    assert assets.loop.probe.height == 1080
    assert source.read_bytes() == before
    assert len(calls) == 1


def test_accepts_no_assets_and_reports_nothing_supplied(
    tmp_path: Path,
) -> None:
    assets = accept_provided_assets(destination=tmp_path / "fitted")

    assert assets.supplied is False
    assert assets.background is None
    assert assets.loop is None
    assert assets.audio is None
    assert assets.logo is None


def test_rejects_an_audio_file_as_a_background(
    tmp_path: Path,
) -> None:
    source = tmp_path / "track.mp3"
    source.write_bytes(MP3)
    destination = tmp_path / "fitted"

    with pytest.raises(ValidationError):
        accept_provided_assets(
            destination=destination,
            background=source,
            ffprobe_runner=ffprobe_runner(10.0),
        )


def test_rejects_a_missing_file(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "fitted"

    with pytest.raises(ValidationError):
        accept_provided_assets(
            destination=destination, background=tmp_path / "missing.png"
        )


def test_rejects_a_resolution_above_the_maximum(
    tmp_path: Path,
) -> None:
    source = write_image(tmp_path / "big.png", size=(2000, 2000))
    destination = tmp_path / "fitted"

    with pytest.raises(ValidationError):
        accept_provided_assets(
            destination=destination,
            background=source,
            size=(800, 600),
            limits=MediaLimits(max_width=1000, max_height=1000),
        )


def test_rejects_a_resolution_below_the_minimum(
    tmp_path: Path,
) -> None:
    source = write_image(tmp_path / "small.png", size=(50, 50))
    destination = tmp_path / "fitted"

    with pytest.raises(ValidationError):
        accept_provided_assets(
            destination=destination,
            background=source,
            limits=MediaLimits(min_width=100, min_height=100),
        )


def test_rejects_a_disallowed_format(
    tmp_path: Path,
) -> None:
    source = write_image(tmp_path / "background.bmp", format="BMP")
    destination = tmp_path / "fitted"

    with pytest.raises(ValidationError):
        accept_provided_assets(
            destination=destination,
            background=source,
            background_formats=frozenset({MediaFormat.PNG}),
        )


def test_accept_provided_assets_error_is_a_validation_error() -> None:
    assert issubclass(AcceptProvidedAssetsError, ValidationError)


def test_by_kind_returns_the_fitted_asset(tmp_path: Path) -> None:
    source = write_image(tmp_path / "logo.png")
    assets = accept_provided_assets(
        destination=tmp_path / "fitted",
        logo=source,
        size=(200, 200),
    )

    fetched = assets.by_kind(AssetKind.LOGO)
    assert fetched is not None
    assert fetched.kind is AssetKind.LOGO
    assert assets.logo is not None
    assert fetched.fitted == assets.logo.fitted

"""Media downloads are sniffed, capped and validated before they are trusted."""

from __future__ import annotations

import asyncio
import hashlib
import json
import struct
import subprocess
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image

from aivia.core.errors import ValidationError
from aivia.core.http import HttpClient
from aivia.core.media_probe import (
    DEFAULT_MAX_FRAMES,
    DEFAULT_MAX_TOTAL_PIXELS,
    FFprobeInfo,
    MediaCategory,
    MediaDownloader,
    MediaFormat,
    MediaLimits,
    image_dimensions,
    image_frame_count,
    run_ffprobe,
    sniff_media,
    validate_media,
)

PNG = MediaFormat.PNG


def png(width: int, height: int) -> bytes:
    header = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR"
    body = struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00"
    return header + body + b"\x00\x00\x00\x00"


def jpeg(width: int, height: int) -> bytes:
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x00" * 9
    sof = (
        b"\xff\xc0"
        + struct.pack(">H", 17)
        + b"\x08"
        + struct.pack(">HH", height, width)
        + b"\x03"
        + b"\x01\x11\x00" * 3
    )
    return b"\xff\xd8\xff" + app0 + sof + b"\xff\xd9"


def gif(width: int, height: int) -> bytes:
    return b"GIF89a" + struct.pack("<HH", width, height) + b"\x00\x00\x00"


def animated_image(
    image_format: str,
    frames: int,
    size: tuple[int, int] = (10, 10),
) -> bytes:
    images = [
        Image.new("RGB", size, (index * 10 % 256, index * 3 % 256, 0))
        for index in range(frames)
    ]
    buffer = BytesIO()
    images[0].save(
        buffer,
        format=image_format,
        save_all=True,
        append_images=images[1:],
        duration=20,
        loop=0,
    )
    return buffer.getvalue()


def gif_with_frames(frames: int, size: tuple[int, int] = (10, 10)) -> bytes:
    width, height = size
    payload = b"GIF89a" + struct.pack("<HH", width, height) + b"\x00\x00\x00"
    descriptor = b"\x2c" + struct.pack("<HHHH", 0, 0, width, height) + b"\x00"
    for _ in range(frames):
        payload += descriptor + b"\x02\x02\x00\x00\x00"
    return payload + b"\x3b"


def webp(width: int, height: int) -> bytes:
    chunk = (
        b"VP8X"
        + struct.pack("<I", 10)
        + b"\x00\x00\x00\x00"
        + (width - 1).to_bytes(3, "little")
        + (height - 1).to_bytes(3, "little")
    )
    return b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk


def bmp(width: int, height: int) -> bytes:
    payload = bytearray(54)
    payload[0:2] = b"BM"
    payload[18:22] = struct.pack("<i", width)
    payload[22:26] = struct.pack("<i", height)
    return bytes(payload)


def tiff(width: int, height: int) -> bytes:
    entries = b""
    for tag, value in ((0x0100, width), (0x0101, height)):
        entries += struct.pack("<HHI", tag, 3, 0) + struct.pack("<HH", value, 0)
    return (
        b"II*\x00"
        + struct.pack("<I", 8)
        + struct.pack("<H", 2)
        + entries
        + b"\x00\x00\x00\x00"
    )


def avif(width: int, height: int) -> bytes:
    ftyp = struct.pack(">I", 24) + b"ftypavif" + b"\x00" * 12
    ispe = b"ispe" + b"\x00\x00\x00\x00" + struct.pack(">II", width, height)
    return ftyp + ispe


def mp4() -> bytes:
    return struct.pack(">I", 24) + b"ftypisom" + b"\x00" * 12


def wav() -> bytes:
    return b"RIFF" + struct.pack("<I", 36) + b"WAVEfmt " + b"\x00" * 28


def probe_json(
    *,
    width: int | None = 1920,
    height: int | None = 1080,
    duration: str = "5.25",
    audio: bool = True,
) -> str:
    streams: list[dict[str, Any]] = [
        {
            "index": 0,
            "codec_type": "video",
            "codec_name": "h264",
            "width": width,
            "height": height,
            "duration": duration,
            "avg_frame_rate": "30/1",
        }
    ]
    if audio:
        streams.append(
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "duration": duration,
                "sample_rate": "48000",
                "channels": 2,
            }
        )
    return json.dumps(
        {
            "streams": streams,
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "format_long_name": "QuickTime / MOV",
                "duration": duration,
                "size": "12345",
            },
        }
    )


def runner_for(
    stdout: str,
    *,
    returncode: int = 0,
    stderr: str = "",
    calls: list[list[str]] | None = None,
) -> Any:
    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if calls is not None:
            calls.append(list(argv))
        return subprocess.CompletedProcess(
            list(argv),
            returncode,
            stdout=stdout,
            stderr=stderr,
        )

    return runner


def test_sniffing_uses_magic_bytes_not_a_filename() -> None:
    cases = {
        png(10, 20): MediaFormat.PNG,
        jpeg(10, 20): MediaFormat.JPEG,
        gif(10, 20): MediaFormat.GIF,
        webp(10, 20): MediaFormat.WEBP,
        bmp(10, 20): MediaFormat.BMP,
        tiff(10, 20): MediaFormat.TIFF,
        avif(10, 20): MediaFormat.AVIF,
        mp4(): MediaFormat.MP4,
        wav(): MediaFormat.WAV,
        b"ID3\x04\x00\x00\x00\x00\x00\x00": MediaFormat.MP3,
        b"fLaC\x00\x00\x00\x22": MediaFormat.FLAC,
        b"OggS\x00\x02\x00\x00": MediaFormat.OGG,
    }
    for payload, expected in cases.items():
        signature = sniff_media(payload)
        assert signature is not None
        assert signature.format is expected

    assert sniff_media(b"<html>not media</html>") is None
    assert sniff_media(b"") is None


def test_sniffing_reports_category_mime_type_and_extension() -> None:
    signature = sniff_media(png(10, 20))
    assert signature is not None
    assert signature.category is MediaCategory.IMAGE
    assert signature.mime_type == "image/png"
    assert signature.extension == "png"

    signature = sniff_media(mp4())
    assert signature is not None
    assert signature.category is MediaCategory.VIDEO
    assert signature.mime_type == "video/mp4"

    signature = sniff_media(wav())
    assert signature is not None
    assert signature.category is MediaCategory.AUDIO
    assert signature.mime_type == "audio/wav"


def test_image_dimensions_are_read_without_decoding() -> None:
    assert image_dimensions(png(640, 480)) == (640, 480)
    assert image_dimensions(jpeg(641, 481)) == (641, 481)
    assert image_dimensions(gif(642, 482)) == (642, 482)
    assert image_dimensions(webp(643, 483)) == (643, 483)
    assert image_dimensions(bmp(644, 484)) == (644, 484)
    assert image_dimensions(tiff(645, 485)) == (645, 485)
    assert image_dimensions(avif(646, 486)) == (646, 486)
    assert image_dimensions(mp4()) is None
    assert image_dimensions(b"not an image") is None


def test_image_frame_count_is_read_from_the_header_without_decoding() -> None:
    assert image_frame_count(png(10, 10)) == 1
    assert image_frame_count(jpeg(10, 10)) == 1
    assert image_frame_count(bmp(10, 10)) == 1
    assert image_frame_count(tiff(10, 10)) == 1
    assert image_frame_count(avif(10, 10)) == 1
    assert image_frame_count(animated_image("GIF", 4)) == 4
    assert image_frame_count(animated_image("WEBP", 4)) == 4
    assert image_frame_count(animated_image("PNG", 4)) == 4


def test_validate_rejects_an_animation_above_the_frame_cap() -> None:
    payload = animated_image("GIF", 4, (10, 10))

    with pytest.raises(ValidationError, match="frame cap") as caught:
        validate_media(payload, limits=MediaLimits(max_frames=3))

    assert caught.value.details == {"frames": 4, "limit": 3}
    probe = validate_media(payload, limits=MediaLimits(max_frames=4))
    assert probe.frames == 4
    assert probe.total_pixels == 400


@pytest.mark.parametrize("image_format", ["GIF", "WEBP", "PNG"])
def test_validate_applies_the_frame_cap_to_every_animated_format(
    image_format: str,
) -> None:
    payload = animated_image(image_format, 4, (10, 10))

    with pytest.raises(ValidationError, match="frame cap") as caught:
        validate_media(payload, limits=MediaLimits(max_frames=3))

    assert caught.value.details == {"frames": 4, "limit": 3}


def test_validate_rejects_an_animation_above_the_default_frame_cap() -> None:
    payload = gif_with_frames(DEFAULT_MAX_FRAMES + 1, (2, 2))

    with pytest.raises(ValidationError, match="frame cap") as caught:
        validate_media(payload)

    assert caught.value.details["frames"] == DEFAULT_MAX_FRAMES + 1
    assert caught.value.details["limit"] == DEFAULT_MAX_FRAMES


def test_validate_counts_total_decoded_pixels_across_every_frame() -> None:
    payload = animated_image("GIF", 5, (20, 10))

    with pytest.raises(ValidationError, match="total decoded pixel cap") as caught:
        validate_media(
            payload,
            limits=MediaLimits(max_frames=10, max_total_pixels=999),
        )

    assert caught.value.details == {
        "width": 20,
        "height": 10,
        "frames": 5,
        "pixels": 1_000,
        "limit": 999,
    }
    probe = validate_media(
        payload,
        limits=MediaLimits(max_frames=10, max_total_pixels=1_000),
    )
    assert probe.frames == 5
    assert probe.total_pixels == 1_000


def test_validate_allows_a_static_image_at_the_total_pixel_cap() -> None:
    probe = validate_media(
        png(100, 50),
        limits=MediaLimits(max_total_pixels=5_000),
    )

    assert probe.frames == 1
    assert probe.total_pixels == 5_000


def test_validate_uses_the_default_total_decoded_pixel_cap() -> None:
    payload = gif_with_frames(4, (8_000, 8_000))

    with pytest.raises(ValidationError, match="total decoded pixel cap") as caught:
        validate_media(payload)

    assert caught.value.details["frames"] == 4
    assert caught.value.details["limit"] == DEFAULT_MAX_TOTAL_PIXELS
    assert caught.value.details["pixels"] == 256_000_000


def test_validate_image_returns_dimensions_digest_and_size() -> None:
    payload = png(320, 200)

    probe = validate_media(payload, expected_format="png")

    assert probe.format is MediaFormat.PNG
    assert probe.category is MediaCategory.IMAGE
    assert probe.width == 320
    assert probe.height == 200
    assert probe.pixels == 64_000
    assert probe.size_bytes == len(payload)
    assert probe.sha256 == hashlib.sha256(payload).hexdigest()
    assert probe.data == payload
    assert probe.ffprobe is None


def test_validate_rejects_empty_unknown_and_oversized_payloads() -> None:
    with pytest.raises(ValidationError, match="empty"):
        validate_media(b"")

    with pytest.raises(ValidationError, match="magic bytes"):
        validate_media(b"<html>error page</html>")

    with pytest.raises(ValidationError, match="byte cap"):
        validate_media(png(10, 10), limits=MediaLimits(max_bytes=8))


def test_decompression_bomb_guard_rejects_a_huge_declared_canvas() -> None:
    payload = png(20_000, 20_000)

    with pytest.raises(ValidationError, match="decompression bomb guard") as caught:
        validate_media(payload)

    assert caught.value.details["pixels"] == 400_000_000
    assert caught.value.details["limit"] == 100_000_000
    assert caught.value.details["width"] == 20_000


def test_dimension_bounds_and_requested_size_are_enforced() -> None:
    payload = png(320, 200)

    with pytest.raises(ValidationError, match="below the minimum"):
        validate_media(payload, limits=MediaLimits(min_width=400))
    with pytest.raises(ValidationError, match="above the maximum"):
        validate_media(payload, limits=MediaLimits(max_height=100))
    with pytest.raises(ValidationError, match="does not match the requested"):
        validate_media(payload, expected_width=321)
    with pytest.raises(ValidationError, match="does not match the requested"):
        validate_media(payload, expected_height=201)


def test_allowed_and_expected_formats_reject_a_mismatch() -> None:
    payload = png(10, 10)

    with pytest.raises(ValidationError, match="not allowed"):
        validate_media(
            payload,
            limits=MediaLimits(allowed_formats=frozenset({MediaFormat.JPEG})),
        )
    with pytest.raises(ValidationError, match="does not match the requested"):
        validate_media(payload, expected_format=MediaFormat.WEBP)


def test_video_is_probed_for_streams_dimensions_and_duration() -> None:
    calls: list[list[str]] = []
    runner = runner_for(probe_json(), calls=calls)

    probe = validate_media(
        mp4(),
        expected_width=1920,
        expected_height=1080,
        limits=MediaLimits(
            min_duration_seconds=5.0,
            max_duration_seconds=6.0,
        ),
        ffprobe_runner=runner,
    )

    assert probe.format is MediaFormat.MP4
    assert probe.width == 1920
    assert probe.height == 1080
    assert probe.duration_seconds == 5.25
    assert probe.ffprobe is not None
    assert probe.ffprobe.primary_video is not None
    assert probe.ffprobe.primary_video.codec_name == "h264"
    assert len(probe.ffprobe.audio_streams) == 1
    assert calls
    assert calls[0][0] == "ffprobe"
    assert "-show_streams" in calls[0]
    assert calls[0][-1].endswith(".mp4")


def test_video_without_a_video_stream_is_rejected() -> None:
    payload = json.dumps(
        {
            "streams": [{"index": 0, "codec_type": "audio", "codec_name": "aac"}],
            "format": {"duration": "1.0"},
        }
    )
    with pytest.raises(ValidationError, match="no video stream"):
        validate_media(mp4(), ffprobe_runner=runner_for(payload))


def test_audio_is_probed_and_requires_an_audio_stream() -> None:
    audio_only = json.dumps(
        {
            "streams": [
                {
                    "index": 0,
                    "codec_type": "audio",
                    "codec_name": "pcm_s16le",
                    "duration": "2.5",
                }
            ],
            "format": {"format_name": "wav", "duration": "2.5"},
        }
    )
    probe = validate_media(
        wav(),
        limits=MediaLimits(min_duration_seconds=2.0),
        ffprobe_runner=runner_for(audio_only),
    )
    assert probe.category is MediaCategory.AUDIO
    assert probe.duration_seconds == 2.5
    assert probe.width is None

    no_audio = json.dumps(
        {
            "streams": [{"index": 0, "codec_type": "video", "codec_name": "h264"}],
            "format": {"duration": "2.5"},
        }
    )
    with pytest.raises(ValidationError, match="no audio stream"):
        validate_media(wav(), ffprobe_runner=runner_for(no_audio))


def test_ffprobe_failures_become_validation_errors(tmp_path: Path) -> None:
    target = tmp_path / "clip.mp4"
    target.write_bytes(mp4())

    with pytest.raises(ValidationError, match="rejected the media"):
        run_ffprobe(target, runner=runner_for("", returncode=1, stderr="bad data"))
    with pytest.raises(ValidationError, match="invalid JSON"):
        run_ffprobe(target, runner=runner_for("not json"))
    with pytest.raises(ValidationError, match="no media streams"):
        run_ffprobe(target, runner=runner_for('{"format": {}, "streams": []}'))

    def missing(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(argv[0])

    with pytest.raises(ValidationError, match="not available"):
        run_ffprobe(target, runner=missing)


def test_ffprobe_reports_parsed_streams_and_format(tmp_path: Path) -> None:
    target = tmp_path / "clip.mp4"
    target.write_bytes(mp4())

    info = run_ffprobe(target, runner=runner_for(probe_json()))

    assert isinstance(info, FFprobeInfo)
    assert info.format_name.startswith("mov")
    assert info.duration_seconds == 5.25
    assert info.size_bytes == 12345
    assert info.video_streams[0].width == 1920
    assert info.audio_streams[0].channels == 2


def test_media_limits_validate_their_bounds() -> None:
    with pytest.raises(ValueError, match="max_bytes"):
        MediaLimits(max_bytes=0)
    with pytest.raises(ValueError, match="max_pixels"):
        MediaLimits(max_pixels=0)
    with pytest.raises(ValueError, match="max_frames"):
        MediaLimits(max_frames=0)
    with pytest.raises(ValueError, match="max_total_pixels"):
        MediaLimits(max_total_pixels=0)
    with pytest.raises(ValueError, match="min_width"):
        MediaLimits(min_width=20, max_width=10)
    with pytest.raises(ValueError, match="duration"):
        MediaLimits(min_duration_seconds=10, max_duration_seconds=5)


def test_download_validates_then_writes_the_destination(tmp_path: Path) -> None:
    payload = png(320, 200)

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://cdn.example.com/result.png"
        return httpx.Response(
            200,
            content=payload,
            headers={"content-length": str(len(payload))},
        )

    downloader = MediaDownloader(
        http=HttpClient(transport=httpx.MockTransport(handler))
    )
    destination = tmp_path / "nested" / "result.png"

    probe = asyncio.run(
        downloader.download(
            "https://cdn.example.com/result.png",
            destination=destination,
        )
    )

    assert probe.format is MediaFormat.PNG
    assert probe.width == 320
    assert probe.height == 200
    assert probe.path == destination
    assert destination.read_bytes() == payload


def test_download_rejects_a_declared_or_actual_oversize_body(tmp_path: Path) -> None:
    payload = png(10, 10)

    def declared(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=payload,
            headers={"content-length": str(len(payload) + 100)},
        )

    def actual(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    for handler in (declared, actual):
        downloader = MediaDownloader(
            http=HttpClient(transport=httpx.MockTransport(handler))
        )
        destination = tmp_path / "too-big.png"
        with pytest.raises(ValidationError, match="byte cap"):
            asyncio.run(
                downloader.download(
                    "https://cdn.example.com/result.png",
                    destination=destination,
                    limits=MediaLimits(max_bytes=16),
                )
            )
        assert not destination.exists()


def test_download_rejects_unknown_media_without_writing(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>error</html>")

    downloader = MediaDownloader(
        http=HttpClient(transport=httpx.MockTransport(handler))
    )
    destination = tmp_path / "result.png"

    with pytest.raises(ValidationError, match="magic bytes"):
        asyncio.run(
            downloader.download(
                "https://cdn.example.com/result.png",
                destination=destination,
            )
        )
    assert not destination.exists()


def test_download_rejects_a_non_http_url() -> None:
    downloader = MediaDownloader()

    with pytest.raises(ValueError, match="http"):
        asyncio.run(downloader.download("ftp://example.com/result.png"))


def test_download_runs_ffprobe_for_video(tmp_path: Path) -> None:
    payload = mp4()
    calls: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    downloader = MediaDownloader(
        http=HttpClient(transport=httpx.MockTransport(handler)),
        ffprobe_runner=runner_for(probe_json(), calls=calls),
    )
    destination = tmp_path / "clip.mp4"

    probe = asyncio.run(
        downloader.download(
            "https://cdn.example.com/result.mp4",
            destination=destination,
        )
    )

    assert probe.format is MediaFormat.MP4
    assert probe.width == 1920
    assert probe.duration_seconds == 5.25
    assert destination.read_bytes() == payload
    assert calls and calls[0][-1].endswith(".mp4")

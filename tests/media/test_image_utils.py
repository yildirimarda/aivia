"""Image transforms write new files, strip metadata and never edit originals."""

from __future__ import annotations

import struct
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

from aivia.core.errors import ValidationError
from aivia.media.image_utils import (
    DEFAULT_MAX_IMAGE_PIXELS,
    ImageFit,
    ImageFormat,
    aspect_crop,
    blurred_fill_pad,
    convert_image,
    open_image,
    resize_image,
    save_image,
    strip_metadata,
    transform_image,
)


def declared_png(width: int, height: int) -> bytes:
    header = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR"
    body = struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00"
    return header + body + b"\x00\x00\x00\x00"


def encoded_png(size: tuple[int, int] = (100, 50)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, (1, 2, 3)).save(buffer, format="PNG")
    return buffer.getvalue()


def two_tone(size: tuple[int, int] = (100, 50)) -> Image.Image:
    image = Image.new("RGB", size, (255, 0, 0))
    image.paste((0, 0, 255), (size[0] // 2, 0, size[0], size[1]))
    return image


def rgb_pixel(image: Image.Image, xy: tuple[int, int]) -> tuple[int, int, int]:
    value = image.getpixel(xy)
    assert isinstance(value, tuple)
    return value[0], value[1], value[2]


def jpeg_with_metadata(path: Path, *, orientation: int | None = None) -> None:
    image = Image.new("RGB", (40, 20), (10, 20, 30))
    exif = Image.Exif()
    exif[0x013B] = "Example Artist"
    exif[0x8298] = "Example Copyright"
    if orientation is not None:
        exif[0x0112] = orientation
    image.save(
        path,
        format="JPEG",
        exif=exif.tobytes(),
        icc_profile=b"not-a-real-profile",
    )


def test_resize_returns_a_new_exact_size_image_and_keeps_the_input() -> None:
    source = Image.new("RGB", (100, 50), (10, 20, 30))

    resized = resize_image(source, (20, 30))

    assert resized.size == (20, 30)
    assert resized is not source
    assert source.size == (100, 50)
    assert source.getpixel((0, 0)) == (10, 20, 30)


def test_resize_rejects_a_non_positive_dimension() -> None:
    source = Image.new("RGB", (10, 10))

    with pytest.raises(ValidationError, match="positive integer"):
        resize_image(source, (0, 10))
    with pytest.raises(ValidationError, match="positive integer"):
        resize_image(source, (10, -1))


def test_aspect_crop_centers_by_default_and_honors_centering() -> None:
    source = two_tone((100, 50))

    centered = aspect_crop(source, (50, 50))
    assert centered.size == (50, 50)
    assert centered.getpixel((10, 10)) == (255, 0, 0)
    assert centered.getpixel((40, 10)) == (0, 0, 255)

    top_left = aspect_crop(source, (50, 50), centering=(0.0, 0.0))
    assert top_left.getpixel((25, 25)) == (255, 0, 0)

    bottom_right = aspect_crop(source, (50, 50), centering=(1.0, 1.0))
    assert bottom_right.getpixel((25, 25)) == (0, 0, 255)


def test_aspect_crop_rejects_invalid_centering() -> None:
    source = Image.new("RGB", (100, 50))

    with pytest.raises(ValidationError, match="between 0 and 1"):
        aspect_crop(source, (50, 50), centering=(1.5, 0.5))


def test_blurred_fill_pad_centers_the_foreground_on_a_blurred_fill() -> None:
    source = Image.new("RGB", (100, 100), (255, 255, 255))
    source.paste((255, 0, 0), (0, 0, 20, 20))

    padded = blurred_fill_pad(source, (100, 300), blur_radius=8.0)

    assert padded.size == (100, 300)
    assert padded.mode == "RGB"
    assert padded.getpixel((5, 5)) == (255, 255, 255)
    assert padded.getpixel((5, 105)) == (255, 0, 0)
    assert padded.getpixel((95, 105)) == (255, 255, 255)


def test_blurred_fill_pad_blurs_a_hard_edge_in_the_padding() -> None:
    source = two_tone((100, 100))

    padded = blurred_fill_pad(source, (100, 300), blur_radius=10.0)

    row = [rgb_pixel(padded, (x, 0)) for x in range(100)]
    assert any(0 < red < 255 and 0 < blue < 255 for red, _, blue in row)


def test_blurred_fill_pad_rejects_a_negative_radius() -> None:
    source = Image.new("RGB", (10, 10))

    with pytest.raises(ValidationError, match="not be negative"):
        blurred_fill_pad(source, (20, 20), blur_radius=-1.0)


def test_convert_image_flattens_transparency_for_jpeg() -> None:
    source = Image.new("RGBA", (10, 10), (255, 0, 0, 0))

    converted = convert_image(source, "jpg")

    assert converted.mode == "RGB"
    assert converted.getpixel((0, 0)) == (255, 255, 255)
    assert source.mode == "RGBA"
    assert source.getpixel((0, 0)) == (255, 0, 0, 0)


def test_convert_image_accepts_a_format_enum_and_suffix() -> None:
    source = Image.new("RGB", (10, 10), (1, 2, 3))

    assert convert_image(source, ImageFormat.PNG).mode == "RGB"
    assert convert_image(source, ".webp").mode == "RGB"


def test_strip_metadata_removes_exif_icc_and_text_without_mutating_input(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.jpg"
    jpeg_with_metadata(source_path)
    source = open_image(source_path)

    clean = strip_metadata(source)

    assert dict(clean.getexif()) == {}
    assert "icc_profile" not in clean.info
    assert dict(source.getexif()) == {
        0x013B: "Example Artist",
        0x8298: "Example Copyright",
    }
    assert "icc_profile" in source.info


def test_save_image_strips_png_text_and_writes_the_requested_format(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.png"
    info = PngImagePlugin.PngInfo()
    info.add_text("Comment", "private note")
    Image.new("RGB", (20, 10), (5, 6, 7)).save(source_path, format="PNG", pnginfo=info)
    destination = tmp_path / "clean.png"

    save_image(open_image(source_path), destination)

    with Image.open(destination) as saved:
        assert saved.format == "PNG"
        assert saved.size == (20, 10)
        assert "Comment" not in saved.info
        assert dict(saved.getexif()) == {}


def test_save_image_rejects_invalid_quality(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="between 1 and 100"):
        save_image(
            Image.new("RGB", (10, 10)),
            tmp_path / "bad.png",
            quality=0,
        )


def test_transform_image_writes_a_new_file_and_leaves_the_source_bytes(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.jpg"
    jpeg_with_metadata(source_path)
    before = source_path.read_bytes()
    destination = tmp_path / "out" / "cover.webp"

    result = transform_image(
        source_path,
        destination,
        size=(30, 30),
        fit=ImageFit.COVER,
        image_format=ImageFormat.WEBP,
    )

    assert result == destination
    assert destination.exists()
    assert source_path.read_bytes() == before
    with Image.open(destination) as saved:
        assert saved.format == "WEBP"
        assert saved.size == (30, 30)
        assert dict(saved.getexif()) == {}
        assert "icc_profile" not in saved.info


@pytest.mark.parametrize(
    ("fit", "expected"),
    [
        (ImageFit.STRETCH, (40, 40)),
        (ImageFit.CONTAIN, (40, 20)),
        (ImageFit.COVER, (40, 40)),
        (ImageFit.BLURRED_PADDING, (40, 40)),
    ],
)
def test_transform_image_applies_each_fit(
    tmp_path: Path,
    fit: ImageFit,
    expected: tuple[int, int],
) -> None:
    source_path = tmp_path / "source.png"
    Image.new("RGB", (100, 50), (1, 2, 3)).save(source_path)
    destination = tmp_path / f"{fit.value}.png"

    transform_image(source_path, destination, size=(40, 40), fit=fit)

    with Image.open(destination) as saved:
        assert saved.size == expected


def test_transform_image_infers_the_format_from_the_suffix(tmp_path: Path) -> None:
    source_path = tmp_path / "source.png"
    Image.new("RGB", (20, 20), (1, 2, 3)).save(source_path)
    destination = tmp_path / "out.jpg"

    transform_image(source_path, destination)

    with Image.open(destination) as saved:
        assert saved.format == "JPEG"


def test_transform_image_rejects_an_unknown_suffix(tmp_path: Path) -> None:
    source_path = tmp_path / "source.png"
    Image.new("RGB", (20, 20), (1, 2, 3)).save(source_path)

    with pytest.raises(ValidationError, match="unsupported image format"):
        transform_image(source_path, tmp_path / "out.unknown")


def test_transform_image_refuses_to_overwrite_the_source(tmp_path: Path) -> None:
    source_path = tmp_path / "source.png"
    Image.new("RGB", (20, 20), (1, 2, 3)).save(source_path)
    before = source_path.read_bytes()

    with pytest.raises(ValidationError, match="must not overwrite"):
        transform_image(source_path, source_path, size=(10, 10))

    assert source_path.read_bytes() == before


def test_transform_image_applies_exif_orientation_before_stripping(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "rotated.jpg"
    jpeg_with_metadata(source_path, orientation=6)
    destination = tmp_path / "upright.png"

    transform_image(source_path, destination)

    with Image.open(destination) as saved:
        assert saved.size == (20, 40)
        assert dict(saved.getexif()) == {}


def test_open_image_accepts_bytes_and_rejects_invalid_data() -> None:
    image = Image.new("RGB", (10, 10), (1, 2, 3))
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    opened = open_image(buffer.getvalue())

    assert opened.size == (10, 10)
    with pytest.raises(ValidationError, match="could not open"):
        open_image(b"not an image")


def test_open_image_rejects_a_decompression_bomb_before_decoding() -> None:
    payload = declared_png(20_000, 20_000)

    with pytest.raises(ValidationError, match="decompression bomb guard") as caught:
        open_image(payload)

    assert caught.value.details["pixels"] == 400_000_000
    assert caught.value.details["limit"] == DEFAULT_MAX_IMAGE_PIXELS
    assert caught.value.details["width"] == 20_000


def test_open_image_rejects_a_decompression_bomb_file(tmp_path: Path) -> None:
    source_path = tmp_path / "bomb.png"
    source_path.write_bytes(declared_png(20_000, 20_000))

    with pytest.raises(ValidationError, match="decompression bomb guard") as caught:
        open_image(source_path)

    assert caught.value.details["pixels"] == 400_000_000


def test_open_image_honors_a_lower_pixel_limit() -> None:
    payload = encoded_png((100, 50))

    with pytest.raises(ValidationError, match="decompression bomb guard") as caught:
        open_image(payload, max_pixels=4_999)

    assert caught.value.details["pixels"] == 5_000
    assert caught.value.details["limit"] == 4_999

    opened = open_image(payload, max_pixels=5_000)
    assert opened.size == (100, 50)


def test_open_image_rejects_oversized_bytes_before_decoding() -> None:
    payload = encoded_png()

    with pytest.raises(ValidationError, match="byte cap") as caught:
        open_image(payload, max_bytes=len(payload) - 1)

    assert caught.value.details["size_bytes"] == len(payload)
    assert caught.value.details["limit"] == len(payload) - 1


def test_open_image_rejects_an_oversized_file(tmp_path: Path) -> None:
    source_path = tmp_path / "large.png"
    source_path.write_bytes(encoded_png())

    with pytest.raises(ValidationError, match="byte cap"):
        open_image(source_path, max_bytes=8)


def test_transform_image_rejects_a_decompression_bomb_source(tmp_path: Path) -> None:
    source_path = tmp_path / "bomb.png"
    source_path.write_bytes(declared_png(20_000, 20_000))
    destination = tmp_path / "out.png"

    with pytest.raises(ValidationError, match="decompression bomb guard"):
        transform_image(source_path, destination)

    assert not destination.exists()


def test_format_parse_rejects_an_unsupported_format() -> None:
    with pytest.raises(ValidationError, match="unsupported image format"):
        ImageFormat.parse("avif")

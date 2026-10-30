"""Local background removal: alpha output, zero cost and no network."""

from __future__ import annotations

import asyncio
from io import BytesIO
from typing import Any, cast

import pytest
from PIL import Image, ImageDraw

from aivia.core.errors import InvalidRequestError, ValidationError
from aivia.providers import (
    DEFAULT_TOLERANCE,
    REMBG_DEFAULT_MODEL,
    LocalBackgroundRemoval,
    LocalBackgroundRemovalProvider,
    LocalBackgroundRemover,
    Rembg,
    RembgImageEdit,
    RembgImageEditProvider,
    RembgProvider,
    provider_registry,
    remove_background,
)
from aivia.providers.image_edit import (
    DEFAULT_MODEL,
)
from aivia.providers.image_edit import (
    LocalBackgroundRemoval as PackageLocalBackgroundRemoval,
)
from aivia.providers.image_edit import (
    RembgImageEditProvider as PackageRembgImageEditProvider,
)
from aivia.providers.image_edit.rembg import SUPPORTED_OPERATIONS
from aivia.providers.protocols import (
    CAPABILITY_IMAGE_EDIT,
    ImageEditOperation,
    ImageEditRequest,
    ImageFormat,
    MediaInput,
)

WIDTH = 64
HEIGHT = 48
BACKGROUND = (255, 255, 255)
FOREGROUND = (220, 20, 20)


def _source_image(
    *,
    size: tuple[int, int] = (WIDTH, HEIGHT),
    background: tuple[int, int, int] = BACKGROUND,
    foreground: tuple[int, int, int] = FOREGROUND,
) -> Image.Image:
    image = Image.new("RGB", size, background)
    draw = ImageDraw.Draw(image)
    width, height = size
    draw.rectangle(
        (width // 3, height // 4, (width * 2) // 3, (height * 3) // 4),
        fill=foreground,
    )
    return image


def _encode(image: Image.Image, image_format: str = "PNG") -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def _decode(data: bytes) -> Image.Image:
    with Image.open(BytesIO(data)) as opened:
        opened.load()
        return opened.convert("RGBA")


def _pixel(image: Image.Image, xy: tuple[int, int]) -> tuple[int, int, int, int]:
    return cast("tuple[int, int, int, int]", image.getpixel(xy))


def _request(
    data: bytes,
    *,
    operation: ImageEditOperation = ImageEditOperation.REMOVE_BACKGROUND,
    image_format: ImageFormat | None = ImageFormat.PNG,
    **kwargs: Any,
) -> ImageEditRequest:
    return ImageEditRequest(
        image=MediaInput(data=data, media_type="image/png"),
        operation=operation,
        format=image_format,
        **kwargs,
    )


def test_metadata_registration_and_aliases_describe_the_provider() -> None:
    provider = RembgImageEditProvider()

    assert provider.name == "rembg"
    assert provider.capabilities == (CAPABILITY_IMAGE_EDIT,)
    assert provider.supports(CAPABILITY_IMAGE_EDIT)
    assert provider.default_model == "rembg"
    assert provider.model == "rembg"
    assert provider.tolerance == DEFAULT_TOLERANCE
    assert provider.remover is None
    assert provider.supported_operations == (ImageEditOperation.REMOVE_BACKGROUND,)
    assert provider.supports_operation("remove_background")
    assert provider.supports_operation(ImageEditOperation.REMOVE_BACKGROUND)
    assert not provider.supports_operation("edit")
    assert not provider.supports_operation("not-an-operation")
    assert DEFAULT_MODEL == "rembg"
    assert REMBG_DEFAULT_MODEL == "rembg"
    assert SUPPORTED_OPERATIONS == (ImageEditOperation.REMOVE_BACKGROUND,)
    assert RembgProvider is RembgImageEditProvider
    assert Rembg is RembgImageEditProvider
    assert RembgImageEdit is RembgImageEditProvider
    assert LocalBackgroundRemoval is RembgImageEditProvider
    assert LocalBackgroundRemovalProvider is RembgImageEditProvider
    assert LocalBackgroundRemover is RembgImageEditProvider
    assert PackageRembgImageEditProvider is RembgImageEditProvider
    assert PackageLocalBackgroundRemoval is RembgImageEditProvider
    assert provider_registry.resolve("rembg") is RembgImageEditProvider
    assert (
        provider_registry.resolve("local-background-removal") is RembgImageEditProvider
    )
    assert provider_registry.resolve("local-rembg") is RembgImageEditProvider


def test_the_provider_declares_zero_cost() -> None:
    provider = RembgImageEditProvider()
    price = provider.metadata.pricing_for(CAPABILITY_IMAGE_EDIT)

    assert price is not None
    assert price.amount_usd == 0.0
    assert price.unit == "image"


def test_the_builtin_remover_returns_an_alpha_png() -> None:
    source = _encode(_source_image())

    response = asyncio.run(RembgImageEditProvider().edit(_request(source)))

    assert response.operation is ImageEditOperation.REMOVE_BACKGROUND
    assert response.model == "rembg"
    assert response.provider == "rembg"
    assert response.count == 1
    assert response.metadata == {"local": True, "cost_usd": 0.0}
    assert response.first.format is ImageFormat.PNG
    assert response.first.data is not None
    assert response.first.data.startswith(b"\x89PNG\r\n\x1a\n")
    assert response.first.width == WIDTH
    assert response.first.height == HEIGHT

    result = _decode(response.first.data)
    assert result.mode == "RGBA"
    assert result.size == (WIDTH, HEIGHT)
    assert _pixel(result, (0, 0))[3] == 0
    assert _pixel(result, (WIDTH - 1, HEIGHT - 1))[3] == 0
    assert _pixel(result, (WIDTH // 2, HEIGHT // 2))[3] == 255
    assert _pixel(result, (WIDTH // 2, HEIGHT // 2))[:3] == FOREGROUND


def test_a_jpeg_source_becomes_an_alpha_png() -> None:
    source = _encode(_source_image(), "JPEG")

    response = asyncio.run(RembgImageEditProvider().edit(_request(source)))

    result = _decode(response.first.data or b"")
    assert response.first.format is ImageFormat.PNG
    assert result.mode == "RGBA"
    assert _pixel(result, (0, 0))[3] == 0
    assert _pixel(result, (WIDTH // 2, HEIGHT // 2))[3] == 255


def test_an_injected_remover_runs_once_and_keeps_its_alpha() -> None:
    calls: list[bytes] = []
    transparent = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    transparent.putpixel((WIDTH // 2, HEIGHT // 2), (10, 20, 30, 128))

    def remover(data: bytes) -> bytes:
        calls.append(data)
        return _encode(transparent)

    source = _encode(_source_image())
    provider = RembgImageEditProvider(remover=remover)

    response = asyncio.run(provider.edit(_request(source, count=2)))

    assert calls == [source]
    assert response.count == 2
    assert response.first.data == response.images[1].data
    result = _decode(response.first.data or b"")
    assert _pixel(result, (0, 0))[3] == 0
    assert _pixel(result, (WIDTH // 2, HEIGHT // 2))[3] == 128


def test_the_request_size_resizes_the_alpha_output() -> None:
    source = _encode(_source_image())

    response = asyncio.run(
        RembgImageEditProvider().edit(_request(source, width=32, height=24))
    )

    assert response.first.width == 32
    assert response.first.height == 24
    result = _decode(response.first.data or b"")
    assert result.size == (32, 24)
    assert _pixel(result, (0, 0))[3] == 0


def test_existing_transparency_is_preserved() -> None:
    image = _source_image().convert("RGBA")
    image.putpixel((4, 4), (0, 0, 0, 0))
    source = _encode(image)

    response = asyncio.run(RembgImageEditProvider().edit(_request(source)))

    result = _decode(response.first.data or b"")
    assert _pixel(result, (4, 4))[3] == 0
    assert _pixel(result, (WIDTH // 2, HEIGHT // 2))[3] == 255


def test_remove_background_function_is_reusable_directly() -> None:
    source = _encode(_source_image())

    output = remove_background(source, tolerance=0)

    result = _decode(output)
    assert _pixel(result, (0, 0))[3] == 0
    assert _pixel(result, (WIDTH // 2, HEIGHT // 2))[3] == 255


def test_a_url_source_is_rejected_without_fetching() -> None:
    provider = RembgImageEditProvider()
    request = ImageEditRequest(
        image=MediaInput(url="https://media.example.com/source.png"),
        operation=ImageEditOperation.REMOVE_BACKGROUND,
    )

    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(provider.edit(request))

    assert caught.value.details["provider"] == "rembg"
    assert "inline" in str(caught.value)


def test_other_operations_are_rejected_before_any_work() -> None:
    provider = RembgImageEditProvider()
    request = ImageEditRequest(
        image=MediaInput(data=_encode(_source_image()), media_type="image/png"),
        operation=ImageEditOperation.EDIT,
        prompt="add a hat",
    )

    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(provider.edit(request))

    assert caught.value.details["operation"] == "edit"


def test_a_non_png_output_format_is_rejected() -> None:
    provider = RembgImageEditProvider()

    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(
            provider.edit(
                _request(_encode(_source_image()), image_format=ImageFormat.JPEG)
            )
        )

    assert caught.value.details["format"] == "jpeg"


def test_invalid_source_or_remover_output_is_a_validation_error() -> None:
    provider = RembgImageEditProvider()

    with pytest.raises(ValidationError):
        asyncio.run(provider.edit(_request(b"not an image")))

    with pytest.raises(ValidationError):
        asyncio.run(
            RembgImageEditProvider(remover=lambda data: b"not an image").edit(
                _request(_encode(_source_image()))
            )
        )

    with pytest.raises(ValidationError):
        asyncio.run(
            RembgImageEditProvider(
                remover=lambda data: "not bytes"  # type: ignore[arg-type,return-value]
            ).edit(_request(_encode(_source_image())))
        )


def test_constructor_arguments_are_validated() -> None:
    with pytest.raises(TypeError):
        RembgImageEditProvider(remover=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        RembgImageEditProvider(tolerance=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        RembgImageEditProvider(tolerance=766)
    with pytest.raises(TypeError):
        RembgImageEditProvider(default_model=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        RembgImageEditProvider(default_model="   ")

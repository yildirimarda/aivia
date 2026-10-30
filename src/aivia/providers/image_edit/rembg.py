"""Dependency-free local background removal producing transparent PNGs.

The adapter removes the background of an inline image without a network call,
an API key or a model download. It finds the background by flood filling the
regions connected to the image border that are within a colour tolerance of
their border seed, then writes the result as an RGBA PNG. That makes it a
deterministic, no-cost fallback for cutouts with a flat or near-flat backdrop.

A caller that has a stronger local model can pass a ``remover`` callable with
the same ``bytes -> bytes`` shape as ``rembg.remove``; the adapter then uses
that instead of the built-in flood fill. The callable still runs locally, so
the provider never downloads models, fetches URLs or spends money on its own.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Iterator
from io import BytesIO
from typing import Any, ClassVar, cast

from PIL import Image, ImageChops, ImageOps, UnidentifiedImageError

from aivia.core.errors import InvalidRequestError, ValidationError
from aivia.core.models.provider import ProviderMeta, ProviderPrice
from aivia.providers.base import Provider
from aivia.providers.protocols.image import ImageResult
from aivia.providers.protocols.image_edit import (
    CAPABILITY_IMAGE_EDIT,
    ImageEditOperation,
    ImageEditRequest,
    ImageEditResponse,
)
from aivia.providers.protocols.media import ImageFormat
from aivia.providers.registry import register_provider

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_TOLERANCE",
    "SUPPORTED_OPERATIONS",
    "LocalBackgroundRemoval",
    "LocalBackgroundRemovalProvider",
    "LocalBackgroundRemover",
    "Rembg",
    "RembgImageEdit",
    "RembgImageEditProvider",
    "RembgProvider",
    "Remover",
    "remove_background",
]

DEFAULT_MODEL = "rembg"
"""The model id reported when a request does not name one."""

DEFAULT_TOLERANCE = 24
"""The default per-pixel colour distance treated as background."""

SUPPORTED_OPERATIONS: tuple[ImageEditOperation, ...] = (
    ImageEditOperation.REMOVE_BACKGROUND,
)
"""The only edit operation this local adapter implements."""

Remover = Callable[[bytes], bytes]
"""A local function turning encoded image bytes into encoded RGBA image bytes."""

_DEFAULT_METADATA = ProviderMeta(
    name="rembg",
    capabilities=[CAPABILITY_IMAGE_EDIT],
    pricing={
        CAPABILITY_IMAGE_EDIT: ProviderPrice(amount_usd=0.0, unit="image"),
    },
)
"""Static metadata: local removal is always free."""


def _validate_tolerance(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("tolerance must be an integer")
    if not 0 <= value <= 765:
        raise ValueError("tolerance must be between 0 and 765")
    return value


def _color_distance(first: tuple[int, ...], second: tuple[int, ...]) -> int:
    return sum(abs(left - right) for left, right in zip(first, second, strict=True))


def _pixel(pixels: Any, x: int, y: int) -> tuple[int, ...]:
    return cast("tuple[int, ...]", pixels[x, y])


def _border_seeds(width: int, height: int) -> Iterator[tuple[int, int]]:
    for x in range(width):
        yield x, 0
        if height > 1:
            yield x, height - 1
    for y in range(1, height - 1):
        yield 0, y
        if width > 1:
            yield width - 1, y


def _flood(
    pixels: Any,
    filled: bytearray,
    width: int,
    height: int,
    seed: tuple[int, int],
    reference: tuple[int, ...],
    tolerance: int,
) -> None:
    x, y = seed
    filled[y * width + x] = 1
    queue: deque[tuple[int, int]] = deque([seed])
    while queue:
        current_x, current_y = queue.popleft()
        for next_x, next_y in (
            (current_x + 1, current_y),
            (current_x - 1, current_y),
            (current_x, current_y + 1),
            (current_x, current_y - 1),
        ):
            if not (0 <= next_x < width and 0 <= next_y < height):
                continue
            index = next_y * width + next_x
            if filled[index]:
                continue
            if _color_distance(_pixel(pixels, next_x, next_y), reference) <= tolerance:
                filled[index] = 1
                queue.append((next_x, next_y))


def _background_mask(image: Image.Image, tolerance: int) -> Image.Image:
    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    assert pixels is not None
    filled = bytearray(width * height)
    for seed in _border_seeds(width, height):
        index = seed[1] * width + seed[0]
        if filled[index]:
            continue
        _flood(
            pixels,
            filled,
            width,
            height,
            seed,
            _pixel(pixels, seed[0], seed[1]),
            tolerance,
        )
    alpha = bytes(0 if value else 255 for value in filled)
    return Image.frombytes("L", (width, height), alpha)


def _open_image(data: bytes) -> Image.Image:
    if not isinstance(data, bytes):
        raise TypeError("image data must be bytes")
    if not data:
        raise ValidationError("image data must be non-empty")
    try:
        with Image.open(BytesIO(data)) as opened:
            opened.load()
            return ImageOps.exif_transpose(opened).copy()
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ValidationError("could not open image data") from exc


def _encode_png(image: Image.Image) -> bytes:
    clean = image.copy()
    clean.info.clear()
    buffer = BytesIO()
    clean.save(buffer, format="PNG")
    return buffer.getvalue()


def remove_background(data: bytes, *, tolerance: int = DEFAULT_TOLERANCE) -> bytes:
    """Return *data* as an RGBA PNG with the border-connected background clear.

    The background is every region reachable from the image border whose
    pixels stay within ``tolerance`` of the border pixel that seeded that
    region. Foreground pixels keep their colour and alpha, and existing
    transparency is preserved.
    """
    checked_tolerance = _validate_tolerance(tolerance)
    image = _open_image(data)
    rgba = image.convert("RGBA")
    mask = _background_mask(rgba, checked_tolerance)
    alpha = ImageChops.multiply(rgba.getchannel("A"), mask)
    red, green, blue, _ = rgba.split()
    return _encode_png(Image.merge("RGBA", (red, green, blue, alpha)))


class RembgImageEditProvider(Provider):
    """A local image edit provider that removes backgrounds for free.

    ``remover`` may replace the built-in flood fill with another local
    ``bytes -> bytes`` function. ``tolerance`` tunes the built-in remover's
    colour distance. The provider accepts only the ``remove_background``
    operation and only inline image bytes, because a local adapter must not
    fetch a remote image or contact a provider.
    """

    default_metadata: ClassVar[ProviderMeta] = _DEFAULT_METADATA

    def __init__(
        self,
        *,
        remover: Remover | None = None,
        tolerance: int = DEFAULT_TOLERANCE,
        default_model: str = DEFAULT_MODEL,
        name: str | None = None,
        **values: Any,
    ) -> None:
        super().__init__(name=name, **values)
        if remover is not None and not callable(remover):
            raise TypeError("remover must be callable or None")
        if not isinstance(default_model, str):
            raise TypeError("default_model must be a string")
        model = default_model.strip()
        if not model:
            raise ValueError("default_model must be non-empty")
        self._remover = remover
        self._tolerance = _validate_tolerance(tolerance)
        self._default_model = model

    @property
    def remover(self) -> Remover | None:
        """The injected local remover, or ``None`` for the built-in flood fill."""
        return self._remover

    @property
    def tolerance(self) -> int:
        """The colour distance the built-in remover treats as background."""
        return self._tolerance

    @property
    def default_model(self) -> str:
        """The model id reported when a request does not name one."""
        return self._default_model

    @property
    def model(self) -> str:
        """Alias for :attr:`default_model`."""
        return self._default_model

    @property
    def supported_operations(self) -> tuple[ImageEditOperation, ...]:
        """The edit operations this adapter implements."""
        return SUPPORTED_OPERATIONS

    def supports_operation(self, operation: ImageEditOperation | str) -> bool:
        """Return whether *operation* is a local background removal."""
        try:
            selected = ImageEditOperation(operation)
        except ValueError:
            return False
        return selected in SUPPORTED_OPERATIONS

    def _remove(self, data: bytes) -> bytes:
        remover = self._remover
        if remover is None:
            return remove_background(data, tolerance=self._tolerance)
        result = remover(data)
        if not isinstance(result, bytes):
            raise ValidationError(
                "remover must return bytes",
                details={"provider": self.name},
            )
        return result

    @staticmethod
    def _prepare_result(data: bytes) -> tuple[bytes, Image.Image]:
        image = _open_image(data).convert("RGBA")
        clean = image.copy()
        clean.info.clear()
        return _encode_png(clean), clean

    async def edit(self, request: ImageEditRequest) -> ImageEditResponse:
        """Remove the background of *request*'s inline image."""
        if not isinstance(request, ImageEditRequest):
            raise TypeError("request must be an ImageEditRequest")
        if request.operation is not ImageEditOperation.REMOVE_BACKGROUND:
            raise InvalidRequestError(
                f"rembg only supports {ImageEditOperation.REMOVE_BACKGROUND.value}",
                details={
                    "provider": self.name,
                    "operation": request.operation.value,
                },
            )
        if request.image.data is None:
            raise InvalidRequestError(
                "local background removal needs inline image data",
                details={"provider": self.name},
            )
        if request.format is not None and request.format is not ImageFormat.PNG:
            raise InvalidRequestError(
                "local background removal outputs alpha png",
                details={"provider": self.name, "format": request.format.value},
            )

        data = await asyncio.to_thread(self._remove, request.image.data)
        encoded, image = self._prepare_result(data)
        if request.width is not None and request.height is not None:
            target = (request.width, request.height)
            if image.size != target:
                image = image.resize(target, Image.Resampling.LANCZOS)
                encoded = _encode_png(image)
        images = [
            ImageResult(
                data=encoded,
                format=ImageFormat.PNG,
                width=image.width,
                height=image.height,
                seed=request.seed,
            )
            for _ in range(request.count)
        ]
        return ImageEditResponse(
            images=images,
            operation=request.operation,
            model=request.model or self._default_model,
            provider=self.name,
            metadata={"local": True, "cost_usd": 0.0},
        )


RembgProvider = RembgImageEditProvider
"""Alias for :class:`RembgImageEditProvider`."""

Rembg = RembgImageEditProvider
"""Short alias for :class:`RembgImageEditProvider`."""

RembgImageEdit = RembgImageEditProvider
"""Alias that names the capability explicitly."""

LocalBackgroundRemovalProvider = RembgImageEditProvider
"""Alias describing the local background removal capability."""

LocalBackgroundRemoval = RembgImageEditProvider
"""Short alias for the local background removal provider."""

LocalBackgroundRemover = RembgImageEditProvider
"""Alias for the local background removal provider."""

register_provider("rembg", RembgImageEditProvider, module=__name__)
register_provider("local-background-removal", RembgImageEditProvider, module=__name__)
register_provider("local-rembg", RembgImageEditProvider, module=__name__)

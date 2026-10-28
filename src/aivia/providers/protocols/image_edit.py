"""Typed image edit protocol, request and response models.

An image edit provider changes an existing image: an instruction edit, an
inpaint, an outpaint, an upscale or background removal. The operation is an
explicit enum because each one has different required inputs, and a provider
must reject an unsupported operation rather than guess what the caller meant.

The source and optional mask are :class:`MediaInput` values, so they may be
inline bytes or a URL. Output images use the same result shape as generation.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, Self, runtime_checkable

from pydantic import Field, field_validator, model_validator

from aivia.providers.protocols.base import CapabilityProvider, ProtocolModel
from aivia.providers.protocols.image import ImageResult
from aivia.providers.protocols.media import ImageFormat, MediaInput

__all__ = [
    "CAPABILITY_IMAGE_EDIT",
    "ImageEditOperation",
    "ImageEditProvider",
    "ImageEditRequest",
    "ImageEditResponse",
]

CAPABILITY_IMAGE_EDIT = "image_edit"
"""The capability name an image edit provider declares."""


class ImageEditOperation(StrEnum):
    """The kind of change an image edit provider should make."""

    EDIT = "edit"
    INPAINT = "inpaint"
    OUTPAINT = "outpaint"
    UPSCALE = "upscale"
    REMOVE_BACKGROUND = "remove_background"
    REPLACE_BACKGROUND = "replace_background"


def _strip_optional(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return value


class ImageEditRequest(ProtocolModel):
    """A validated request to edit one image.

    ``prompt`` is required for instruction edits, inpainting and background
    replacement. ``mask`` is required for inpainting and ``scale`` for an
    upscale, so a provider never receives a request missing the input its
    operation needs. ``width`` and ``height`` describe a target size and must
    be supplied together.
    """

    image: MediaInput
    operation: ImageEditOperation = ImageEditOperation.EDIT
    prompt: str | None = None
    mask: MediaInput | None = None
    model: str | None = None
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    scale: float | None = Field(default=None, gt=1.0, le=8.0)
    format: ImageFormat | None = None
    seed: int | None = None
    count: int = Field(default=1, ge=1, le=8)
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt", "model", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: Any) -> Any:
        return _strip_optional(value)

    @model_validator(mode="after")
    def _validate_operation(self) -> Self:
        if (self.width is None) != (self.height is None):
            raise ValueError("image width and height must be set together")
        if (
            self.operation
            in {
                ImageEditOperation.EDIT,
                ImageEditOperation.INPAINT,
                ImageEditOperation.REPLACE_BACKGROUND,
            }
            and not self.prompt
        ):
            raise ValueError(f"{self.operation.value} requires a prompt")
        if self.operation is ImageEditOperation.INPAINT and self.mask is None:
            raise ValueError("inpaint requires a mask")
        if self.operation is ImageEditOperation.UPSCALE and self.scale is None:
            raise ValueError("upscale requires a scale greater than 1")
        return self


class ImageEditResponse(ProtocolModel):
    """A validated image edit response from one provider."""

    images: list[ImageResult] = Field(min_length=1)
    operation: ImageEditOperation = ImageEditOperation.EDIT
    model: str = ""
    provider: str = ""
    request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model", "provider", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("request_id", mode="before")
    @classmethod
    def _strip_request_id(cls, value: Any) -> Any:
        return _strip_optional(value)

    @property
    def first(self) -> ImageResult:
        """The first edited image."""
        return self.images[0]

    @property
    def count(self) -> int:
        """The number of edited images."""
        return len(self.images)


@runtime_checkable
class ImageEditProvider(CapabilityProvider, Protocol):
    """A provider that serves the ``image_edit`` capability."""

    async def edit(self, request: ImageEditRequest) -> ImageEditResponse:
        """Apply *request* to its source image and return validated results."""

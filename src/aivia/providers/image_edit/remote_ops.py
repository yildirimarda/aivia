"""Remote background removal and upscaling through an aggregator queue.

Local background removal is free but only understands simple backdrops, and a
local resize cannot add detail. These operations need a remote model instead.
Both are image edits, so they use the same request and response models as the
other edit adapters, but they do not take a prompt: background removal erases
the backdrop and upscaling enlarges the source by a fixed factor.

The adapter maps one model id per operation, like the aggregator edit adapter,
and validates the upscale factor before a paid call. After the queue finishes
it checks the returned pixel dimensions against the source size, so a provider
that silently returns the input or the wrong factor is rejected instead of
being stored as a successful edit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from io import BytesIO
from typing import Any, ClassVar

from PIL import Image, UnidentifiedImageError

from aivia.core.errors import (
    ConfigError,
    ContentPolicyError,
    InvalidRequestError,
    ValidationError,
)
from aivia.core.models.provider import ProviderMeta
from aivia.providers.image.fal import (
    FalImageProvider,
    _is_content_policy,
    _validate_model,
)
from aivia.providers.image_edit.aggregator import _media_url
from aivia.providers.media import AsyncJobState
from aivia.providers.protocols.image import ImageRequest, ImageResult
from aivia.providers.protocols.image_edit import (
    CAPABILITY_IMAGE_EDIT,
    ImageEditOperation,
    ImageEditRequest,
    ImageEditResponse,
)
from aivia.providers.protocols.media import ImageFormat
from aivia.providers.registry import register_provider

__all__ = [
    "DEFAULT_SCALES",
    "DEFAULT_UPSCALE_FACTORS",
    "REMOTE_OPS_DEFAULT_BASE_URL",
    "SUPPORTED_OPERATIONS",
    "SUPPORTED_UPSCALE_FACTORS",
    "RemoteBackgroundRemovalAndUpscale",
    "RemoteBackgroundRemovalAndUpscaleProvider",
    "RemoteImageEdit",
    "RemoteImageEditProvider",
    "RemoteImageOps",
    "RemoteImageOpsProvider",
    "RemoteOps",
    "RemoteOpsImageEditProvider",
    "RemoteOpsProvider",
]

REMOTE_OPS_DEFAULT_BASE_URL = "https://queue.fal.run"
"""The default queue root, without a trailing slash."""

SUPPORTED_OPERATIONS: tuple[ImageEditOperation, ...] = (
    ImageEditOperation.REMOVE_BACKGROUND,
    ImageEditOperation.UPSCALE,
)
"""The remote edit operations this adapter implements."""

DEFAULT_UPSCALE_FACTORS: tuple[float, ...] = (2.0, 4.0)
"""The upscale factors the adapter accepts unless configured otherwise."""

SUPPORTED_UPSCALE_FACTORS = DEFAULT_UPSCALE_FACTORS
"""Alias for :data:`DEFAULT_UPSCALE_FACTORS`."""

DEFAULT_SCALES = DEFAULT_UPSCALE_FACTORS
"""Alias for :data:`DEFAULT_UPSCALE_FACTORS` that names scales."""

_DEFAULT_METADATA = ProviderMeta(
    name="remote-ops",
    capabilities=[CAPABILITY_IMAGE_EDIT],
)

_OPERATION_ALIASES = {
    "remove_background": ImageEditOperation.REMOVE_BACKGROUND,
    "remove-background": ImageEditOperation.REMOVE_BACKGROUND,
    "background_removal": ImageEditOperation.REMOVE_BACKGROUND,
    "background-removal": ImageEditOperation.REMOVE_BACKGROUND,
    "bg_removal": ImageEditOperation.REMOVE_BACKGROUND,
    "bg-removal": ImageEditOperation.REMOVE_BACKGROUND,
    "cutout": ImageEditOperation.REMOVE_BACKGROUND,
    "upscale": ImageEditOperation.UPSCALE,
    "upscaling": ImageEditOperation.UPSCALE,
    "upscale_image": ImageEditOperation.UPSCALE,
    "super_resolution": ImageEditOperation.UPSCALE,
    "super-resolution": ImageEditOperation.UPSCALE,
}


def _operation(value: ImageEditOperation | str) -> ImageEditOperation:
    if isinstance(value, ImageEditOperation):
        return value
    if isinstance(value, str):
        text = value.strip().lower().replace(" ", "_")
        operation = _OPERATION_ALIASES.get(text)
        if operation is not None:
            return operation
    raise ConfigError(
        f"unknown remote image operation: {value!r}",
        details={"operation": str(value)},
    )


def _normalize_models(
    models: Mapping[Any, str] | None,
) -> dict[ImageEditOperation, str]:
    if models is None:
        return {}
    if not isinstance(models, Mapping):
        raise TypeError("models must be a mapping of operation to model id")
    normalized: dict[ImageEditOperation, str] = {}
    for key, value in models.items():
        operation = _operation(key)
        if operation not in SUPPORTED_OPERATIONS:
            raise ConfigError(
                f"unsupported remote image operation: {operation.value}",
                details={"operation": operation.value},
            )
        if operation in normalized:
            raise ConfigError(
                f"duplicate remote image model for {operation.value}",
                details={"operation": operation.value},
            )
        if not isinstance(value, str):
            raise TypeError("remote image model ids must be strings")
        normalized[operation] = _validate_model(value)
    return normalized


def _validate_factors(
    factors: Sequence[float] | None,
) -> tuple[float, ...]:
    if factors is None:
        return DEFAULT_UPSCALE_FACTORS
    if isinstance(factors, (str, bytes)) or not isinstance(factors, Sequence):
        raise TypeError("factors must be a sequence of numbers or None")
    validated: list[float] = []
    for value in factors:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("upscale factors must be numbers")
        factor = float(value)
        if not math.isfinite(factor) or factor <= 1.0 or factor > 8.0:
            raise ConfigError(
                f"upscale factor must be greater than 1 and at most 8: {value!r}",
                details={"factor": value},
            )
        if factor in validated:
            raise ConfigError(
                f"duplicate upscale factor: {value!r}",
                details={"factor": value},
            )
        validated.append(factor)
    if not validated:
        raise ConfigError("at least one upscale factor is required")
    return tuple(validated)


def _validate_scale(
    scale: float | None,
    factors: tuple[float, ...],
) -> float:
    if scale is None:
        raise InvalidRequestError("upscale requires a scale greater than 1")
    factor = float(scale)
    if not math.isfinite(factor) or factor <= 1.0 or factor > 8.0:
        raise InvalidRequestError(
            f"upscale scale must be greater than 1 and at most 8: {scale!r}",
            details={"scale": scale},
        )
    if factor not in factors:
        raise InvalidRequestError(
            f"unsupported upscale factor: {scale!r}",
            details={
                "scale": scale,
                "supported": list(factors),
            },
        )
    return factor


def _source_size(data: bytes) -> tuple[int, int]:
    if not isinstance(data, bytes):
        raise TypeError("image data must be bytes")
    if not data:
        raise ValidationError("image data must be non-empty")
    try:
        with Image.open(BytesIO(data)) as opened:
            opened.load()
            return opened.size
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ValidationError("could not read image dimensions") from exc


def _expected_dimensions(
    request: ImageEditRequest,
    factor: float | None,
) -> tuple[int, int] | None:
    if (
        request.operation is ImageEditOperation.UPSCALE
        and factor is not None
        and request.image.data is not None
    ):
        width, height = _source_size(request.image.data)
        return (round(width * factor), round(height * factor))
    if request.width is not None and request.height is not None:
        return (request.width, request.height)
    if request.image.data is not None:
        return _source_size(request.image.data)
    return None


def _build_payload(
    request: ImageEditRequest,
    factor: float | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "image_url": _media_url(request.image),
        "num_images": request.count,
        "output_format": (request.format or ImageFormat.PNG).value,
    }
    if request.operation is ImageEditOperation.UPSCALE:
        if factor is None:
            raise InvalidRequestError("upscale requires a scale greater than 1")
        payload["scale"] = factor
    if request.seed is not None:
        payload["seed"] = request.seed
    return payload


def _result_request(request: ImageEditRequest) -> ImageRequest:
    return ImageRequest(
        prompt=request.operation.value,
        model=request.model,
        width=request.width,
        height=request.height,
        format=request.format or ImageFormat.PNG,
        seed=request.seed,
        count=request.count,
    )


class RemoteOpsImageEditProvider(FalImageProvider):
    """Remote background removal and upscaling through an aggregator queue.

    ``models`` maps ``remove_background`` and ``upscale`` to model ids. A
    request may override the model for one call, and ``model`` (or
    ``default_model``) supplies a fallback used only when the operation has no
    configured model. ``factors`` limits the upscale factors this adapter will
    send, so an unsupported factor is rejected before a paid call.
    """

    default_metadata: ClassVar[ProviderMeta] = _DEFAULT_METADATA

    def __init__(
        self,
        *,
        models: Mapping[Any, str] | None = None,
        model: str | None = None,
        default_model: str | None = None,
        factors: Sequence[float] | None = None,
        **kwargs: Any,
    ) -> None:
        self._models = _normalize_models(models)
        self._factors = _validate_factors(factors)
        selected = model if model is not None else default_model
        super().__init__(default_model=selected, **kwargs)

    @property
    def models(self) -> dict[str, str]:
        """The configured model id for each supported operation."""
        return {operation.value: model for operation, model in self._models.items()}

    @property
    def factors(self) -> tuple[float, ...]:
        """The upscale factors this adapter accepts."""
        return self._factors

    @property
    def supported_operations(self) -> tuple[ImageEditOperation, ...]:
        """The remote edit operations this adapter implements."""
        return SUPPORTED_OPERATIONS

    def supports_operation(self, operation: ImageEditOperation | str) -> bool:
        """Return whether *operation* is handled by this adapter."""
        try:
            selected = ImageEditOperation(operation)
        except ValueError:
            return False
        return selected in SUPPORTED_OPERATIONS

    def model_for(self, operation: ImageEditOperation | str) -> str | None:
        """Return the configured model for *operation*, if any."""
        selected = _operation(operation)
        return self._models.get(selected) or self._default_model

    def _model_for_request(self, request: ImageEditRequest) -> str:
        if request.model is not None:
            return _validate_model(request.model)
        selected = self._models.get(request.operation) or self._default_model
        if selected is None:
            raise InvalidRequestError(
                f"remote image {request.operation.value} needs a model id",
                details={"operation": request.operation.value},
            )
        return _validate_model(selected)

    def _check_dimensions(
        self,
        images: list[ImageResult],
        expected: tuple[int, int] | None,
    ) -> list[ImageResult]:
        checked: list[ImageResult] = []
        for image in images:
            measured = _source_size(image.data) if image.data is not None else None
            if (
                measured is not None
                and image.width is not None
                and image.height is not None
                and measured != (image.width, image.height)
            ):
                raise ValidationError(
                    "remote image dimensions do not match its data",
                    details={
                        "provider": self.name,
                        "reported": [image.width, image.height],
                        "measured": list(measured),
                    },
                )
            actual: tuple[int, int] | None
            if measured is not None:
                actual = measured
            elif image.dimensions is not None:
                actual = (image.dimensions.width, image.dimensions.height)
            else:
                actual = None
            if expected is not None:
                if actual is None:
                    raise ValidationError(
                        "remote image returned no dimensions",
                        details={
                            "provider": self.name,
                            "expected": list(expected),
                        },
                    )
                if actual != expected:
                    raise ValidationError(
                        "remote image returned unexpected dimensions",
                        details={
                            "provider": self.name,
                            "expected": list(expected),
                            "actual": list(actual),
                        },
                    )
            update: dict[str, Any] = {}
            if image.width is None and measured is not None:
                update["width"] = measured[0]
            if image.height is None and measured is not None:
                update["height"] = measured[1]
            checked.append(image.model_copy(update=update) if update else image)
        return checked

    async def edit(self, request: ImageEditRequest) -> ImageEditResponse:
        """Remove the background or upscale *request*'s image remotely."""
        if not isinstance(request, ImageEditRequest):
            raise TypeError("request must be an ImageEditRequest")
        if request.operation not in SUPPORTED_OPERATIONS:
            supported = ", ".join(operation.value for operation in SUPPORTED_OPERATIONS)
            raise InvalidRequestError(
                f"remote ops does not support {request.operation.value} "
                f"(supported: {supported})",
                details={"operation": request.operation.value},
            )

        factor: float | None = None
        if request.operation is ImageEditOperation.UPSCALE:
            factor = _validate_scale(request.scale, self._factors)
        elif request.scale is not None:
            raise InvalidRequestError(
                f"{request.operation.value} does not take an upscale scale",
                details={"operation": request.operation.value},
            )

        expected = _expected_dimensions(request, factor)
        model = self._model_for_request(request)
        payload = _build_payload(request, factor)
        idempotency_key = ""
        raw_key = request.metadata.get("idempotency_key")
        if isinstance(raw_key, str):
            idempotency_key = raw_key.strip()
        job = await self._helper.submit(
            {"model": model, "payload": payload},
            idempotency_key=idempotency_key,
            metadata={
                "model": model,
                "operation": request.operation.value,
            },
        )
        job = await self._helper.poll(
            job,
            timeout_seconds=request.timeout_seconds,
        )
        if job.state is AsyncJobState.FAILED:
            message = job.error_message or "remote image edit job failed"
            if _is_content_policy(message, job.error_code):
                raise ContentPolicyError(
                    message,
                    details={"provider": self.name, "job_id": job.id},
                )
            raise InvalidRequestError(
                message,
                details={"provider": self.name, "job_id": job.id},
            )
        if not job.succeeded:
            raise InvalidRequestError(
                f"remote image edit job ended in state {job.state.value}",
                details={"provider": self.name, "job_id": job.id},
            )

        images = self._result_from_job(job, _result_request(request))
        if self._download_results:
            images = await self._download(images)
        images = self._check_dimensions(images, expected)
        return ImageEditResponse(
            images=images,
            operation=request.operation,
            model=model,
            provider=self.name,
            request_id=job.remote_id or None,
            metadata={"job_id": job.id},
        )


RemoteOpsProvider = RemoteOpsImageEditProvider
"""Alias for :class:`RemoteOpsImageEditProvider`."""

RemoteOps = RemoteOpsImageEditProvider
"""Short alias for :class:`RemoteOpsImageEditProvider`."""

RemoteImageOpsProvider = RemoteOpsImageEditProvider
"""Alias that names the image operations explicitly."""

RemoteImageOps = RemoteOpsImageEditProvider
"""Short alias for :class:`RemoteImageOpsProvider`."""

RemoteImageEditProvider = RemoteOpsImageEditProvider
"""Alias for :class:`RemoteOpsImageEditProvider`."""

RemoteImageEdit = RemoteOpsImageEditProvider
"""Short alias for :class:`RemoteImageEditProvider`."""

RemoteBackgroundRemovalAndUpscaleProvider = RemoteOpsImageEditProvider
"""Alias describing both operations."""

RemoteBackgroundRemovalAndUpscale = RemoteOpsImageEditProvider
"""Short alias for :class:`RemoteBackgroundRemovalAndUpscaleProvider`."""

register_provider("remote-ops", RemoteOpsImageEditProvider, module=__name__)
register_provider("remote_ops", RemoteOpsImageEditProvider, module=__name__)
register_provider("remote-image-ops", RemoteOpsImageEditProvider, module=__name__)
register_provider("remote_image_ops", RemoteOpsImageEditProvider, module=__name__)
register_provider("remote-image-edit", RemoteOpsImageEditProvider, module=__name__)
register_provider(
    "remote-background-removal",
    RemoteOpsImageEditProvider,
    module=__name__,
)
register_provider("remote-upscale", RemoteOpsImageEditProvider, module=__name__)

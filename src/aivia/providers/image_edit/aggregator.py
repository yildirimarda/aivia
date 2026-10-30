"""Image editing through an aggregator's asynchronous model queue.

An aggregator exposes many edit models behind one queue API. Each operation
therefore needs its own model id and payload: an instruction edit sends the
source image and a prompt, an inpaint also sends a mask, and an outpaint sends
a target size. This adapter keeps that mapping explicit so a caller configures
one model per operation instead of relying on a model that happens to accept
every field.

The adapter is built on the same queue lifecycle as the image generation
adapters: submit, poll, download, timeout and cancel are owned by
:class:`~aivia.providers.media.async_job.AsyncMediaJobHelper`. It never decides
routing, fallback or budget policy, and a safety filter becomes
:class:`~aivia.core.errors.ContentPolicyError` so the router does not retry the
same prompt against the same provider.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any, ClassVar

from aivia.core.errors import (
    ConfigError,
    ContentPolicyError,
    InvalidRequestError,
)
from aivia.core.models.provider import ProviderMeta
from aivia.providers.image.fal import (
    FalImageProvider,
    _is_content_policy,
    _validate_model,
)
from aivia.providers.media import AsyncJobState
from aivia.providers.protocols.image import ImageRequest
from aivia.providers.protocols.image_edit import (
    CAPABILITY_IMAGE_EDIT,
    ImageEditOperation,
    ImageEditRequest,
    ImageEditResponse,
)
from aivia.providers.protocols.media import ImageFormat, MediaInput
from aivia.providers.registry import register_provider

__all__ = [
    "AGGREGATOR_DEFAULT_BASE_URL",
    "SUPPORTED_OPERATIONS",
    "Aggregator",
    "AggregatorImageEdit",
    "AggregatorImageEditProvider",
    "AggregatorProvider",
]

AGGREGATOR_DEFAULT_BASE_URL = "https://queue.fal.run"
"""The default queue root, without a trailing slash."""

SUPPORTED_OPERATIONS: tuple[ImageEditOperation, ...] = (
    ImageEditOperation.EDIT,
    ImageEditOperation.INPAINT,
    ImageEditOperation.OUTPAINT,
)
"""The edit operations this aggregator can map to a model."""

_DEFAULT_METADATA = ProviderMeta(
    name="aggregator",
    capabilities=[CAPABILITY_IMAGE_EDIT],
)

_OPERATION_ALIASES = {
    "edit": ImageEditOperation.EDIT,
    "instruction_edit": ImageEditOperation.EDIT,
    "inpaint": ImageEditOperation.INPAINT,
    "outpaint": ImageEditOperation.OUTPAINT,
}


def _operation(value: ImageEditOperation | str) -> ImageEditOperation:
    if isinstance(value, ImageEditOperation):
        return value
    if isinstance(value, str):
        text = value.strip().lower().replace("-", "_")
        operation = _OPERATION_ALIASES.get(text)
        if operation is not None:
            return operation
    raise ConfigError(
        f"unknown image edit operation: {value!r}",
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
                f"unsupported image edit operation: {operation.value}",
                details={"operation": operation.value},
            )
        if operation in normalized:
            raise ConfigError(
                f"duplicate image edit model for {operation.value}",
                details={"operation": operation.value},
            )
        if not isinstance(value, str):
            raise TypeError("image edit model ids must be strings")
        model = _validate_model(value)
        normalized[operation] = model
    return normalized


def _media_url(media: MediaInput) -> str:
    if media.url is not None:
        return media.url
    if media.data is None:
        raise InvalidRequestError("image edit input needs inline data or a URL")
    media_type = media.media_type or "image/png"
    encoded = base64.b64encode(media.data).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _build_payload(request: ImageEditRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "prompt": request.prompt,
        "num_images": request.count,
        "output_format": (request.format or ImageFormat.PNG).value,
        "image_url": _media_url(request.image),
    }
    if request.operation is ImageEditOperation.INPAINT:
        if request.mask is None:
            raise InvalidRequestError("inpaint requires a mask")
        payload["mask_url"] = _media_url(request.mask)
    if request.seed is not None:
        payload["seed"] = request.seed
    if request.width is not None and request.height is not None:
        payload["image_size"] = {
            "width": request.width,
            "height": request.height,
        }
    return payload


def _result_request(request: ImageEditRequest) -> ImageRequest:
    return ImageRequest(
        prompt=request.prompt or request.operation.value,
        model=request.model,
        width=request.width,
        height=request.height,
        format=request.format or ImageFormat.PNG,
        seed=request.seed,
        count=request.count,
    )


class AggregatorImageEditProvider(FalImageProvider):
    """An aggregator image edit provider with one model per operation.

    ``models`` maps ``edit``, ``inpaint`` and ``outpaint`` to model ids. A
    request may override the model for one call, and ``model`` (or
    ``default_model``) supplies a fallback used only when the operation has no
    configured model. Operations without a model, and operations the adapter
    does not implement, are rejected instead of being sent to the wrong model.
    """

    default_metadata: ClassVar[ProviderMeta] = _DEFAULT_METADATA

    def __init__(
        self,
        *,
        models: Mapping[Any, str] | None = None,
        model: str | None = None,
        default_model: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._models = _normalize_models(models)
        selected = model if model is not None else default_model
        super().__init__(default_model=selected, **kwargs)

    @property
    def models(self) -> dict[str, str]:
        """The configured model id for each supported operation."""
        return {operation.value: model for operation, model in self._models.items()}

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
                f"aggregator image {request.operation.value} needs a model id",
                details={"operation": request.operation.value},
            )
        return _validate_model(selected)

    async def edit(self, request: ImageEditRequest) -> ImageEditResponse:
        """Apply *request* through its operation's configured model."""
        if not isinstance(request, ImageEditRequest):
            raise TypeError("request must be an ImageEditRequest")
        if request.operation not in SUPPORTED_OPERATIONS:
            supported = ", ".join(operation.value for operation in SUPPORTED_OPERATIONS)
            raise InvalidRequestError(
                f"aggregator does not support {request.operation.value} "
                f"(supported: {supported})",
                details={"operation": request.operation.value},
            )

        model = self._model_for_request(request)
        payload = _build_payload(request)
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
            message = job.error_message or "aggregator image edit job failed"
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
                f"aggregator image edit job ended in state {job.state.value}",
                details={"provider": self.name, "job_id": job.id},
            )

        images = self._result_from_job(job, _result_request(request))
        if self._download_results:
            images = await self._download(images)
        return ImageEditResponse(
            images=images,
            operation=request.operation,
            model=model,
            provider=self.name,
            request_id=job.remote_id or None,
            metadata={"job_id": job.id},
        )


AggregatorProvider = AggregatorImageEditProvider
"""Alias for :class:`AggregatorImageEditProvider`."""

Aggregator = AggregatorImageEditProvider
"""Short alias for :class:`AggregatorImageEditProvider`."""

AggregatorImageEdit = AggregatorImageEditProvider
"""Alias that names the capability explicitly."""

register_provider("aggregator", AggregatorImageEditProvider, module=__name__)
register_provider("aggregator-image-edit", AggregatorImageEditProvider, module=__name__)

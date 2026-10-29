"""Image input support and vision model filtering for LLM calls.

The wire protocol already carries typed :class:`ImagePart` content. This module
answers the policy question that sits in front of a call: does a request need a
vision capable model, and which configured models declare that capability in
``models.yaml``?

A model declares image input with the ``vision`` capability. Filtering is
deliberately conservative: an unknown model is not assumed to see images.
Callers may filter a candidate list themselves or ask the selector to pick the
first capable model and fail with a clear error when none is available.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from aivia.core.errors import ConfigError, InvalidRequestError
from aivia.core.model_ids import (
    VISION_CAPABILITY,
    ModelCatalog,
    ModelInfo,
    load_model_catalog,
)
from aivia.providers.protocols.llm import ChatMessage, LLMRequest

__all__ = [
    "CAPABILITY_VISION",
    "VISION_CAPABILITY",
    "VisionModelFilter",
    "VisionModelSelector",
    "VisionNotSupportedError",
    "check_vision_model",
    "ensure_vision_model",
    "filter_models_for_request",
    "filter_models_for_vision",
    "filter_vision_models",
    "has_vision",
    "message_has_images",
    "messages_need_vision",
    "model_supports_vision",
    "models_for_request",
    "models_with_vision",
    "request_needs_vision",
    "request_requires_vision",
    "require_vision",
    "require_vision_model",
    "select_vision_model",
    "supports_vision",
    "vision_models",
]

CAPABILITY_VISION = VISION_CAPABILITY
"""Alias for the catalog capability that marks image input support."""


class VisionNotSupportedError(ConfigError, InvalidRequestError):
    """A request with images was routed to a model that cannot see them."""

    code = "vision_not_supported"

    def __init__(
        self,
        model: str,
        *,
        available: Iterable[str] = (),
        details: dict[str, object] | None = None,
    ) -> None:
        candidates = tuple(available)
        merged: dict[str, object] = {
            "model": model,
            "required_capability": VISION_CAPABILITY,
        }
        if candidates:
            merged["vision_models"] = list(candidates)
        if details:
            merged.update(details)
        message = f"model {model!r} does not declare {VISION_CAPABILITY} support"
        if candidates:
            message += f" (vision models: {', '.join(candidates)})"
        super().__init__(message, details=merged)
        self.model = model
        self.available = candidates


def _catalog(catalog: ModelCatalog | None) -> ModelCatalog:
    return load_model_catalog() if catalog is None else catalog


def _models(models: Iterable[str] | str) -> list[str]:
    values = [models] if isinstance(models, str) else list(models)
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError("model ids must be strings")
        model = value.strip()
        if not model or model in seen:
            continue
        seen.add(model)
        unique.append(model)
    return unique


def message_has_images(message: ChatMessage) -> bool:
    """Whether *message* carries at least one image part."""
    if not isinstance(message, ChatMessage):
        raise TypeError("message must be a ChatMessage")
    return message.has_images


def messages_need_vision(messages: Sequence[ChatMessage]) -> bool:
    """Whether any message in *messages* carries an image part."""
    return any(message_has_images(message) for message in messages)


def request_needs_vision(request: LLMRequest) -> bool:
    """Whether *request* can only be served by a vision capable model."""
    if not isinstance(request, LLMRequest):
        raise TypeError("request must be an LLMRequest")
    return request.requires_vision


request_requires_vision = request_needs_vision


def model_supports_vision(
    model: str | ModelInfo | None,
    catalog: ModelCatalog | None = None,
) -> bool:
    """Whether *model* declares image input support.

    A :class:`ModelInfo` is answered directly. A model id is looked up in
    *catalog*, or the active catalog when none is supplied. Unknown ids are
    not assumed to support vision.
    """
    if model is None:
        return False
    if isinstance(model, ModelInfo):
        return model.supports_vision
    if not isinstance(model, str):
        raise TypeError("model must be a model id or ModelInfo")
    key = model.strip()
    if not key:
        return False
    return _catalog(catalog).supports_vision(key)


supports_vision = model_supports_vision
has_vision = model_supports_vision


def filter_vision_models(
    models: Iterable[str] | str,
    catalog: ModelCatalog | None = None,
) -> list[str]:
    """Return the ids in *models* that declare vision support, order preserved."""
    resolved = _catalog(catalog)
    return [model for model in _models(models) if resolved.supports_vision(model)]


filter_models_for_vision = filter_vision_models
models_with_vision = filter_vision_models
vision_models = filter_vision_models


def filter_models_for_request(
    models: Iterable[str] | str,
    request: LLMRequest,
    catalog: ModelCatalog | None = None,
) -> list[str]:
    """Filter *models* only when *request* needs vision.

    A text-only request leaves the candidate list untouched; an image request
    drops every model that does not declare the ``vision`` capability.
    """
    if not isinstance(request, LLMRequest):
        raise TypeError("request must be an LLMRequest")
    if request.requires_vision:
        return filter_vision_models(models, catalog)
    return _models(models)


models_for_request = filter_models_for_request


def require_vision_model(
    model: str,
    catalog: ModelCatalog | None = None,
    *,
    request: LLMRequest | None = None,
) -> str:
    """Return *model* when it can serve *request*, else raise.

    Without a request the model must declare vision. With a text-only request
    no capability check is needed; with an image request the model must declare
    vision.
    """
    if not isinstance(model, str) or not model.strip():
        raise ConfigError("vision model must be a non-empty model id")
    key = model.strip()
    if request is not None and not request.requires_vision:
        return key
    resolved = _catalog(catalog)
    if resolved.supports_vision(key):
        return key
    raise VisionNotSupportedError(key, available=resolved.vision_models())


require_vision = require_vision_model
ensure_vision_model = require_vision_model
check_vision_model = require_vision_model


def select_vision_model(
    models: Iterable[str] | str,
    catalog: ModelCatalog | None = None,
    *,
    request: LLMRequest | None = None,
) -> str:
    """Return the first model in *models* that can serve *request*.

    When *request* is text-only the first candidate is enough. An image request
    or no request at all requires the first model that declares vision.
    """
    resolved = _catalog(catalog)
    candidates = _models(models)
    if request is not None and not request.requires_vision:
        if not candidates:
            raise ConfigError("no model candidates were supplied")
        return candidates[0]
    for model in candidates:
        if resolved.supports_vision(model):
            return model
    raise VisionNotSupportedError(
        candidates[0] if candidates else "<none>",
        available=resolved.vision_models(),
        details={"candidates": candidates},
    )


class VisionModelSelector:
    """Filter and pick candidate models for image aware LLM calls."""

    def __init__(
        self,
        catalog: ModelCatalog | None = None,
        models: Iterable[str] | str | None = None,
    ) -> None:
        self._catalog = _catalog(catalog)
        self._models = None if models is None else _models(models)

    @property
    def catalog(self) -> ModelCatalog:
        """The model catalog used for capability lookups."""
        return self._catalog

    @property
    def models(self) -> list[str]:
        """The default candidate ids, or an empty list when none were set."""
        return list(self._models or ())

    def supports(self, model: str | ModelInfo | None) -> bool:
        """Whether *model* declares image input support."""
        return model_supports_vision(model, self._catalog)

    def filter(
        self,
        models: Iterable[str] | str | None = None,
        *,
        request: LLMRequest | None = None,
    ) -> list[str]:
        """Return the vision capable candidates from *models* or the defaults."""
        candidates = self._models if models is None else _models(models)
        if candidates is None:
            candidates = []
        if request is not None and not request.requires_vision:
            return list(candidates)
        return filter_vision_models(candidates, self._catalog)

    def for_request(
        self,
        request: LLMRequest,
        models: Iterable[str] | str | None = None,
    ) -> list[str]:
        """Alias for :meth:`filter` with the request in a natural position."""
        return self.filter(models, request=request)

    def ensure(
        self,
        model: str,
        *,
        request: LLMRequest | None = None,
    ) -> str:
        """Validate one model against the request and return it."""
        return require_vision_model(model, self._catalog, request=request)

    def select(
        self,
        request: LLMRequest | None = None,
        models: Iterable[str] | str | None = None,
    ) -> str:
        """Pick the first capable model from *models* or the defaults."""
        candidates = self._models if models is None else _models(models)
        return select_vision_model(
            candidates or [],
            self._catalog,
            request=request,
        )


VisionModelFilter = VisionModelSelector
"""Alias for :class:`VisionModelSelector`."""

"""Render and deliver tools for the assistant.

The render tool turns a composition into a still image and writes it to an
output file, with output selection choosing which frame, format and path to
write. The deliver tool hands a rendered output to a delivery provider and is
only registered when a delivery provider is configured.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar, TypeGuard, cast

from pydantic import AliasChoices, Field

from aivia.assistant.tool_framework import (
    Tool,
    ToolArguments,
    ToolError,
    ToolRegistry,
    ToolResult,
)
from aivia.compose import render_still
from aivia.compose.schema import Composition
from aivia.core.router import Router
from aivia.providers.protocols.delivery import (
    CAPABILITY_DELIVERY,
    DeliveryArtifact,
    DeliveryProvider,
    DeliveryRequest,
)

__all__ = [
    "ALL_RENDER_TOOLS",
    "DeliverTool",
    "RenderTool",
    "render_tool_registry",
]


def _run_async(call: Any) -> Any:
    """Run an async delivery call from a synchronous tool execution."""
    return asyncio.run(call)


def _has_delivery_provider(router: Router | None) -> TypeGuard[Router]:
    """Return whether *router* has a provider serving the delivery capability."""
    if router is None:
        return False
    return any(
        provider.supports(CAPABILITY_DELIVERY) for provider in router.providers.values()
    )


class _RenderArgs(ToolArguments):
    composition: dict[str, Any]
    output: str | None = None
    time_seconds: float | None = Field(default=None, ge=0.0)
    format: str | None = Field(default=None, min_length=1)


class RenderTool(Tool):
    """Render a composition still to an output image file."""

    name = "render.render"
    description = "Render a composition still to an output image file."
    arguments_model = _RenderArgs

    @classmethod
    def execute(cls, arguments: dict[str, Any]) -> ToolResult:
        composition = arguments.pop("composition")
        output = arguments.pop("output", None)
        time_seconds = arguments.pop("time_seconds", None)
        fmt = arguments.pop("format", None)

        validated = Composition.model_validate(composition)
        image = render_still(validated, time_seconds=time_seconds or 0.0)

        if output is None:
            output = "render.png"
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format=fmt)

        return ToolResult(
            output={
                "output": str(path),
                "path": str(path),
                "width": image.width,
                "height": image.height,
            },
            cost_usd=0.0,
            estimated=False,
        )


class _DeliverArgs(ToolArguments):
    output: str
    method: str = Field(
        validation_alias=AliasChoices("method", "transport", "channel", "type")
    )
    target: str = Field(
        validation_alias=AliasChoices("target", "destination", "to", "address")
    )
    order_id: str = ""
    run_id: str = ""
    subject: str = ""
    message: str = ""
    expires_seconds: int | None = Field(default=None, gt=0)
    idempotency_key: str = ""


class DeliverTool(Tool):
    """Deliver a rendered output through a delivery provider."""

    name = "render.deliver"
    description = (
        "Deliver a rendered output to a recipient through a delivery provider."
    )
    arguments_model = _DeliverArgs

    _router: ClassVar[Router | None] = None

    @classmethod
    def bind_router(cls, router: Router) -> None:
        """Bind *router* to this tool class so execute can route deliveries."""
        cls._router = router

    @classmethod
    def execute(cls, arguments: dict[str, Any]) -> ToolResult:
        router = cls._router
        if router is None:
            raise ToolError(
                f"tool {cls.name!r} has no router bound",
                details={"tool": cls.name},
            )
        output = arguments.pop("output")
        method = arguments.pop("method")
        target = arguments.pop("target")
        order_id = arguments.pop("order_id", "")
        run_id = arguments.pop("run_id", "")
        subject = arguments.pop("subject", "")
        message = arguments.pop("message", "")
        expires_seconds = arguments.pop("expires_seconds", None)
        idempotency_key = arguments.pop("idempotency_key", "")

        artifact = DeliveryArtifact(path=output)
        request = DeliveryRequest(
            method=method,
            target=target,
            order_id=order_id,
            run_id=run_id,
            subject=subject,
            message=message,
            artifacts=[artifact],
            expires_seconds=expires_seconds,
            idempotency_key=idempotency_key,
        )
        receipt = _run_async(
            router.execute(
                CAPABILITY_DELIVERY,
                lambda provider: cast(DeliveryProvider, provider).deliver(request),
            )
        )
        return ToolResult(
            output={
                "receipt": receipt.model_dump(),
                "method": receipt.method,
                "reference": receipt.reference,
                "target": receipt.target,
                "provider": receipt.provider,
                "delivered": receipt.delivered,
                "url": receipt.url,
            },
            cost_usd=0.0,
            estimated=False,
        )


ALL_RENDER_TOOLS: tuple[type[Tool], ...] = (RenderTool, DeliverTool)


def render_tool_registry(router: Router | None = None) -> ToolRegistry:
    """Return a registry with the render tool and, when a delivery provider is
    configured, the deliver tool."""
    registry = ToolRegistry()
    registry.register(RenderTool)
    if _has_delivery_provider(router):
        DeliverTool.bind_router(router)
        registry.register(DeliverTool)
    return registry

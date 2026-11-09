"""Render and deliver tools.

The render tool is exercised through the
:class:`~aivia.assistant.tool_framework.ToolFramework`, so arguments are
validated against the JSON schema and every render writes a real image file.
The deliver tool is registered only when a router has a delivery provider and
is exercised with the deterministic mock delivery provider.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from aivia.assistant import (
    DeliverTool,
    ToolFramework,
    ToolNotAllowedError,
    ToolRegistry,
    ToolUnknownError,
    ToolValidationError,
    render_tool_registry,
)
from aivia.core.config.capability import CapabilityChainConfig
from aivia.core.router import Router
from aivia.providers.mock import MockDeliveryProvider


def _composition() -> dict[str, Any]:
    return {
        "canvas": {"width": 1280, "height": 720, "background": "#ff0000"},
        "duration_seconds": 5.0,
        "layers": [],
        "audio_tracks": [],
        "effects": [],
    }


def _framework(registry: ToolRegistry) -> ToolFramework:
    return ToolFramework(registry)


def _delivery_router() -> Router:
    provider = MockDeliveryProvider(name="mock-delivery")
    return Router(
        {provider.name: provider},
        config=CapabilityChainConfig(chain=[provider.name]),
    )


def test_render_tool_is_always_registered() -> None:
    registry = render_tool_registry()
    assert "render.render" in registry
    assert "render.deliver" not in registry
    schema = registry.schemas()["render.render"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert "composition" in schema["required"]


def test_deliver_tool_not_registered_without_delivery_provider() -> None:
    registry = render_tool_registry()
    assert "render.deliver" not in registry
    assert registry.names() == ("render.render",)


def test_deliver_tool_registered_with_delivery_provider() -> None:
    registry = render_tool_registry(_delivery_router())
    assert "render.render" in registry
    assert "render.deliver" in registry
    assert registry.names() == ("render.deliver", "render.render")


def test_render_tool_renders_to_output_file(tmp_path: Path) -> None:
    output = tmp_path / "still.png"
    result = _framework(render_tool_registry()).execute(
        "render.render",
        {"composition": _composition(), "output": str(output)},
    )

    assert output.is_file()
    with Image.open(output) as image:
        assert image.size == (1280, 720)

    output_payload = result.output
    assert output_payload["output"] == str(output)
    assert output_payload["path"] == str(output)
    assert output_payload["width"] == 1280
    assert output_payload["height"] == 720


def test_render_tool_output_selection(tmp_path: Path) -> None:
    output = tmp_path / "frame.png"
    result = _framework(render_tool_registry()).execute(
        "render.render",
        {
            "composition": _composition(),
            "output": str(output),
            "time_seconds": 2.5,
            "format": "PNG",
        },
    )

    assert output.is_file()
    with Image.open(output) as image:
        assert image.size == (1280, 720)
    assert result.output["path"] == str(output)


def test_render_tool_default_output_path(tmp_path: Path) -> None:
    result = _framework(render_tool_registry()).execute(
        "render.render", {"composition": _composition()}
    )

    assert Path(result.output["output"]).is_file()
    assert result.output["path"] == "render.png"


def test_render_tool_rejects_missing_composition() -> None:
    framework = _framework(render_tool_registry())
    with pytest.raises(ToolValidationError):
        framework.execute("render.render", {"output": "x.png"})
    assert framework.tool_calls == 0


def test_allowlist_restricts_render_tool(tmp_path: Path) -> None:
    registry = render_tool_registry(_delivery_router())
    from aivia.core.config.planner import PlannerConfig

    config = PlannerConfig(allowed_tools=["render.render"])
    framework = ToolFramework(registry, planner_config=config)

    assert framework.permits("render.render") is True
    assert framework.permits("render.deliver") is False

    output = tmp_path / "still.png"
    framework.execute(
        "render.render", {"composition": _composition(), "output": str(output)}
    )

    with pytest.raises(ToolNotAllowedError):
        framework.execute(
            "render.deliver",
            {"output": str(output), "method": "folder", "target": str(tmp_path)},
        )
    assert framework.tool_calls == 1


def test_deliver_tool_delivers_output(tmp_path: Path) -> None:
    router = _delivery_router()
    DeliverTool.bind_router(router)
    registry = render_tool_registry(router)
    framework = _framework(registry)

    output = tmp_path / "still.png"
    framework.execute(
        "render.render", {"composition": _composition(), "output": str(output)}
    )

    result = framework.execute(
        "render.deliver",
        {
            "output": str(output),
            "method": "folder",
            "target": str(tmp_path / "delivered"),
            "subject": "rendered output",
        },
    )

    assert result.output["delivered"] is True
    assert result.output["method"] == "folder"
    assert result.output["target"] == str(tmp_path / "delivered")
    assert result.output["provider"] == "mock-delivery"
    assert result.output["reference"].startswith("mock-delivery-")


def test_deliver_tool_unknown_without_delivery_provider() -> None:
    framework = _framework(render_tool_registry())
    with pytest.raises(ToolUnknownError):
        framework.execute(
            "render.deliver",
            {"output": "x.png", "method": "folder", "target": "/tmp"},
        )
    assert framework.tool_calls == 0

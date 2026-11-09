"""Generation tools for image, image edit, background removal, upscale,
video, voice and music.

Each tool wraps a :class:`~aivia.media.MediaToolkit` and exposes one media
capability through the :class:`~aivia.assistant.tool_framework.Tool`
interface: a JSON schema, an estimate and an execution that runs the screened,
cached, budgeted media path.

The media toolkit is bound to each tool class with :meth:`bind_toolkit` before
the tools are executed, so the same registry can be reused with different
toolkit configurations.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any, ClassVar

from pydantic import Field

from aivia.assistant.tool_framework import (
    Tool,
    ToolArguments,
    ToolError,
    ToolRegistry,
    ToolResult,
)
from aivia.media import MediaResult, MediaToolkit
from aivia.providers.protocols.image_edit import ImageEditOperation

__all__ = [
    "ALL_GENERATION_TOOLS",
    "BackgroundRemovalTool",
    "ImageEditTool",
    "ImageGenerationTool",
    "MusicGenerationTool",
    "UpscaleTool",
    "VideoGenerationTool",
    "VoiceGenerationTool",
    "generation_tool_registry",
]


def _run_async(call: Any) -> Any:
    """Run an async media call from a synchronous tool execution."""
    return asyncio.run(call)


def _present(arguments: dict[str, Any]) -> dict[str, Any]:
    """Return only the explicitly set (non-None) arguments.

    Pydantic then applies defaults for the omitted fields.
    """
    return {key: value for key, value in arguments.items() if value is not None}


class GenerationTool(Tool):
    """Base class for generation tools bound to one media toolkit."""

    _toolkit: ClassVar[MediaToolkit | None] = None

    @classmethod
    def bind_toolkit(cls, toolkit: MediaToolkit) -> None:
        """Bind *toolkit* to this tool class so execute can run media calls."""
        cls._toolkit = toolkit

    @classmethod
    def _media_call(cls, arguments: dict[str, Any]) -> Awaitable[MediaResult]:
        """Build and run the media call for *arguments*."""
        raise NotImplementedError

    @classmethod
    def execute(cls, arguments: dict[str, Any]) -> ToolResult:
        """Run the bound media call and return a uniform cost result."""
        toolkit = cls._toolkit
        if toolkit is None:
            raise ToolError(
                f"tool {cls.name!r} has no media toolkit bound",
                details={"tool": cls.name},
            )
        result = _run_async(cls._media_call(arguments))
        return ToolResult(
            output=result.to_dict(),
            cost_usd=result.cost_usd,
            estimated=False,
        )


class _ImageArgs(ToolArguments):
    prompt: str
    model: str | None = None
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    aspect_ratio: str | None = None
    format: str | None = None
    quality: str | None = None
    style: str | None = None
    seed: int | None = None
    count: int = Field(default=1, ge=1, le=8)


class ImageGenerationTool(GenerationTool):
    """Generate one or more images from a text prompt."""

    name = "image.generate"
    description = "Generate one or more images from a text prompt."
    arguments_model = _ImageArgs

    @classmethod
    def _media_call(cls, arguments: dict[str, Any]) -> Awaitable[MediaResult]:
        toolkit = cls._toolkit
        assert toolkit is not None
        prompt = arguments.pop("prompt")
        return toolkit.image(prompt=prompt, **_present(arguments))


class _ImageEditArgs(ToolArguments):
    image: Any
    operation: str = "edit"
    prompt: str | None = None
    mask: Any = None
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    scale: float | None = Field(default=None, gt=1.0, le=8.0)
    format: str | None = None
    seed: int | None = None
    count: int = Field(default=1, ge=1, le=8)


class ImageEditTool(GenerationTool):
    """Edit an image: instruct, inpaint, outpaint, upscale or remove its background."""

    name = "image.edit"
    description = (
        "Edit an image: instruct, inpaint, outpaint, upscale or remove its background."
    )
    arguments_model = _ImageEditArgs

    @classmethod
    def _media_call(cls, arguments: dict[str, Any]) -> Awaitable[MediaResult]:
        toolkit = cls._toolkit
        assert toolkit is not None
        image = arguments.pop("image")
        operation = arguments.pop("operation", "edit")
        prompt = arguments.pop("prompt", None)
        mask = arguments.pop("mask", None)
        return toolkit.edit(
            image=image,
            operation=operation,
            prompt=prompt,
            mask=mask,
            **_present(arguments),
        )


class _BackgroundRemovalArgs(ToolArguments):
    image: Any
    count: int = Field(default=1, ge=1, le=8)


class BackgroundRemovalTool(GenerationTool):
    """Remove the background from an image."""

    name = "image.background_removal"
    description = "Remove the background from an image."
    arguments_model = _BackgroundRemovalArgs

    @classmethod
    def _media_call(cls, arguments: dict[str, Any]) -> Awaitable[MediaResult]:
        toolkit = cls._toolkit
        assert toolkit is not None
        image = arguments.pop("image")
        count = arguments.pop("count", 1)
        return toolkit.edit(
            image=image,
            operation=ImageEditOperation.REMOVE_BACKGROUND,
            count=count,
        )


class _UpscaleArgs(ToolArguments):
    image: Any
    scale: float = Field(gt=1.0, le=8.0)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    count: int = Field(default=1, ge=1, le=8)


class UpscaleTool(GenerationTool):
    """Upscale an image by a scale factor."""

    name = "image.upscale"
    description = "Upscale an image by a scale factor."
    arguments_model = _UpscaleArgs

    @classmethod
    def _media_call(cls, arguments: dict[str, Any]) -> Awaitable[MediaResult]:
        toolkit = cls._toolkit
        assert toolkit is not None
        image = arguments.pop("image")
        scale = arguments.pop("scale")
        return toolkit.edit(
            image=image,
            operation=ImageEditOperation.UPSCALE,
            scale=scale,
            **arguments,
        )


class _VideoArgs(ToolArguments):
    prompt: str
    model: str | None = None
    duration_seconds: float | None = Field(default=None, gt=0.0)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    aspect_ratio: str | None = None
    fps: int | None = Field(default=None, gt=0, le=120)
    format: str | None = None
    seed: int | None = None
    count: int = Field(default=1, ge=1, le=4)
    generate_audio: bool = False
    audio_prompt: str | None = None


class VideoGenerationTool(GenerationTool):
    """Generate one or more video clips from a text prompt."""

    name = "video.generate"
    description = "Generate one or more video clips from a text prompt."
    arguments_model = _VideoArgs

    @classmethod
    def _media_call(cls, arguments: dict[str, Any]) -> Awaitable[MediaResult]:
        toolkit = cls._toolkit
        assert toolkit is not None
        prompt = arguments.pop("prompt")
        return toolkit.video(prompt=prompt, **_present(arguments))


class _VoiceArgs(ToolArguments):
    text: str
    voice: str = "default"
    model: str | None = None
    language: str | None = None
    format: str | None = None
    sample_rate: int | None = Field(default=None, gt=0)
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    instructions: str | None = None
    pronunciation: dict[str, str] = Field(default_factory=dict)
    seed: int | None = None
    word_timestamps: bool = False


class VoiceGenerationTool(GenerationTool):
    """Synthesize speech from text."""

    name = "voice.generate"
    description = "Synthesize speech from text."
    arguments_model = _VoiceArgs

    @classmethod
    def _media_call(cls, arguments: dict[str, Any]) -> Awaitable[MediaResult]:
        toolkit = cls._toolkit
        assert toolkit is not None
        text = arguments.pop("text")
        return toolkit.speech(text=text, **_present(arguments))


class _MusicArgs(ToolArguments):
    prompt: str
    lyrics: str | None = None
    instrumental: bool = False
    model: str | None = None
    duration_seconds: float | None = Field(default=None, gt=0.0)
    format: str | None = None
    sample_rate: int | None = Field(default=None, gt=0)
    tempo_bpm: float | None = Field(default=None, gt=0.0)
    style: str | None = None
    seed: int | None = None
    count: int = Field(default=1, ge=1, le=8)


class MusicGenerationTool(GenerationTool):
    """Generate one or more music tracks from a text prompt."""

    name = "music.generate"
    description = "Generate one or more music tracks from a text prompt."
    arguments_model = _MusicArgs

    @classmethod
    def _media_call(cls, arguments: dict[str, Any]) -> Awaitable[MediaResult]:
        toolkit = cls._toolkit
        assert toolkit is not None
        prompt = arguments.pop("prompt")
        return toolkit.music(prompt=prompt, **_present(arguments))


ALL_GENERATION_TOOLS: tuple[type[GenerationTool], ...] = (
    ImageGenerationTool,
    ImageEditTool,
    BackgroundRemovalTool,
    UpscaleTool,
    VideoGenerationTool,
    VoiceGenerationTool,
    MusicGenerationTool,
)


def generation_tool_registry() -> ToolRegistry:
    """Return a registry with every generation tool registered."""
    registry = ToolRegistry()
    for tool in ALL_GENERATION_TOOLS:
        registry.register(tool)
    return registry

"""Asset registry, tool framework and session helpers for one session or run.

The registry holds files with labels, content hashes and analysis. Originals
are copied into the registry directory and marked read-only, so they can never
be modified in place. Duplicate labels are rejected.

The tool framework provides an allowlisted registry of tools with JSON schemas,
estimates, execution and safety limits.
"""

from aivia.assistant.composition_tools import (
    ALL_COMPOSITION_TOOLS,
    AnimationTool,
    AudioMixTool,
    CompositionTool,
    OverlayTool,
    PlacementTool,
    TextTool,
    composition_tool_registry,
)
from aivia.assistant.generation_tools import (
    ALL_GENERATION_TOOLS,
    BackgroundRemovalTool,
    ImageEditTool,
    ImageGenerationTool,
    MusicGenerationTool,
    UpscaleTool,
    VideoGenerationTool,
    VoiceGenerationTool,
    generation_tool_registry,
)
from aivia.assistant.registry import (
    AssetRegistry,
    DuplicateLabelError,
    UnknownLabelError,
    resolve_labels,
)
from aivia.assistant.tool_framework import (
    SafetyLimits,
    Tool,
    ToolArguments,
    ToolBudgetError,
    ToolError,
    ToolEstimate,
    ToolFramework,
    ToolNotAllowedError,
    ToolRegistry,
    ToolResult,
    ToolSafetyLimitError,
    ToolUnknownError,
    ToolValidationError,
)

__all__ = [
    "ALL_COMPOSITION_TOOLS",
    "ALL_GENERATION_TOOLS",
    "AnimationTool",
    "AssetRegistry",
    "AudioMixTool",
    "BackgroundRemovalTool",
    "CompositionTool",
    "DuplicateLabelError",
    "ImageEditTool",
    "ImageGenerationTool",
    "MusicGenerationTool",
    "OverlayTool",
    "PlacementTool",
    "SafetyLimits",
    "TextTool",
    "Tool",
    "ToolArguments",
    "ToolBudgetError",
    "ToolError",
    "ToolEstimate",
    "ToolFramework",
    "ToolNotAllowedError",
    "ToolRegistry",
    "ToolResult",
    "ToolSafetyLimitError",
    "ToolUnknownError",
    "ToolValidationError",
    "UnknownLabelError",
    "UpscaleTool",
    "VideoGenerationTool",
    "VoiceGenerationTool",
    "composition_tool_registry",
    "generation_tool_registry",
    "resolve_labels",
]

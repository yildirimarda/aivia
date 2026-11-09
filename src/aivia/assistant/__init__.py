"""Asset registry, tool framework and session helpers for one session or run.

The registry holds files with labels, content hashes and analysis. Originals
are copied into the registry directory and marked read-only, so they can never
be modified in place. Duplicate labels are rejected.

The tool framework provides an allowlisted registry of tools with JSON schemas,
estimates, execution and safety limits.
"""

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
    "ALL_GENERATION_TOOLS",
    "AssetRegistry",
    "BackgroundRemovalTool",
    "DuplicateLabelError",
    "ImageEditTool",
    "ImageGenerationTool",
    "MusicGenerationTool",
    "SafetyLimits",
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
    "generation_tool_registry",
    "resolve_labels",
]

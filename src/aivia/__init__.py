"""AIVIA: configurable AI-backed media generation assistant."""

from typing import Any

__version__ = "0.1.0"
__all__ = [
    "MediaResult",
    "MediaToolkit",
    "MediaToolkitResult",
    "ToolkitResult",
    "__version__",
]


def __getattr__(name: str) -> Any:
    if name in {"MediaResult", "MediaToolkit", "MediaToolkitResult", "ToolkitResult"}:
        from aivia.media import toolkit

        return getattr(toolkit, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

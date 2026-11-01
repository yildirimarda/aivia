"""Errors raised while composing or rendering a composition."""

from __future__ import annotations

from aivia.core.errors import ValidationError

__all__ = ["FontError", "RenderError"]


class RenderError(ValidationError):
    """A composition could not be rendered deterministically."""

    code = "render"


class FontError(RenderError):
    """A font family or face could not be resolved for rendering."""

    code = "font"

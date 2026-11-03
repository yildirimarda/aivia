"""Compatibility import for the writing script schema."""

from aivia.writing.schema import *  # noqa: F403
from aivia.writing.schema import __all__ as _schema_all

__all__ = list(_schema_all)

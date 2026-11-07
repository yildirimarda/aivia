"""Reference inputs and the style specs extracted from them.

The module is optional: the core never imports it. It consumes the core model
serialization helpers and returns plain validated data that later analysis,
merge, store and generation stages can share.
"""

from aivia.reference.schema import *  # noqa: F403
from aivia.reference.schema import __all__ as _schema_all

__all__ = [*_schema_all]

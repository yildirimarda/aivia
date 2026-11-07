"""Reference inputs and the style specs extracted from them.

The module is optional: the core never imports it. It consumes the core model
serialization helpers and returns plain validated data that later analysis,
merge, store and generation stages can share.
"""

from aivia.reference.analyze import *  # noqa: F403
from aivia.reference.analyze import __all__ as _analyze_all
from aivia.reference.analyze_audio import *  # noqa: F403
from aivia.reference.analyze_audio import __all__ as _analyze_audio_all
from aivia.reference.analyze_video import *  # noqa: F403
from aivia.reference.analyze_video import __all__ as _analyze_video_all
from aivia.reference.inputs import *  # noqa: F403
from aivia.reference.inputs import __all__ as _inputs_all
from aivia.reference.schema import *  # noqa: F403
from aivia.reference.schema import __all__ as _schema_all

__all__ = [
    *_schema_all,
    *_inputs_all,
    *_analyze_all,
    *_analyze_video_all,
    *_analyze_audio_all,
]

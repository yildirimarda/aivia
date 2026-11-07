"""Reference inputs and the style specs extracted from them.

The module is optional: the core never imports it. It consumes the core model
serialization helpers and returns plain validated data that later analysis,
merge, store and generation stages can share.
"""

from aivia.reference.analyze import *  # noqa: F403
from aivia.reference.analyze import __all__ as _analyze_all
from aivia.reference.analyze_audio import *  # noqa: F403
from aivia.reference.analyze_audio import __all__ as _analyze_audio_all
from aivia.reference.analyze_text import *  # noqa: F403
from aivia.reference.analyze_text import __all__ as _analyze_text_all
from aivia.reference.analyze_video import *  # noqa: F403
from aivia.reference.analyze_video import __all__ as _analyze_video_all
from aivia.reference.capabilities import *  # noqa: F403
from aivia.reference.capabilities import __all__ as _capabilities_all
from aivia.reference.conditioned import *  # noqa: F403
from aivia.reference.conditioned import __all__ as _conditioned_all
from aivia.reference.inputs import *  # noqa: F403
from aivia.reference.inputs import __all__ as _inputs_all
from aivia.reference.merge import *  # noqa: F403
from aivia.reference.merge import __all__ as _merge_all
from aivia.reference.schema import *  # noqa: F403
from aivia.reference.schema import __all__ as _schema_all
from aivia.reference.store import *  # noqa: F403
from aivia.reference.store import __all__ as _store_all
from aivia.reference.structure import *  # noqa: F403
from aivia.reference.structure import __all__ as _structure_all
from aivia.reference.style_prompts import *  # noqa: F403
from aivia.reference.style_prompts import __all__ as _style_prompts_all
from aivia.reference.to_format import *  # noqa: F403
from aivia.reference.to_format import __all__ as _to_format_all

__all__ = [
    *_schema_all,
    *_inputs_all,
    *_analyze_all,
    *_analyze_video_all,
    *_analyze_audio_all,
    *_analyze_text_all,
    *_merge_all,
    *_store_all,
    *_structure_all,
    *_style_prompts_all,
    *_to_format_all,
    *_capabilities_all,
    *_conditioned_all,
]

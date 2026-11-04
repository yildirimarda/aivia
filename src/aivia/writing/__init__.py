"""Writing stage: mode aware scripts, payloads and the mode registry."""

from aivia.writing.facts import *  # noqa: F403
from aivia.writing.facts import __all__ as _facts_all
from aivia.writing.hooks import *  # noqa: F403
from aivia.writing.hooks import __all__ as _hooks_all
from aivia.writing.humor import *  # noqa: F403
from aivia.writing.humor import __all__ as _humor_all
from aivia.writing.length import *  # noqa: F403
from aivia.writing.length import __all__ as _length_all
from aivia.writing.lyrics import *  # noqa: F403
from aivia.writing.lyrics import __all__ as _lyrics_all
from aivia.writing.modes import *  # noqa: F403
from aivia.writing.modes import __all__ as _modes_all
from aivia.writing.originality import *  # noqa: F403
from aivia.writing.originality import __all__ as _originality_all
from aivia.writing.poem import *  # noqa: F403
from aivia.writing.poem import __all__ as _poem_all
from aivia.writing.quote import *  # noqa: F403
from aivia.writing.quote import __all__ as _quote_all
from aivia.writing.schema import *  # noqa: F403
from aivia.writing.schema import __all__ as _schema_all
from aivia.writing.story import *  # noqa: F403
from aivia.writing.story import __all__ as _story_all

__all__ = list(
    dict.fromkeys(
        [
            *_schema_all,
            *_modes_all,
            *_length_all,
            *_facts_all,
            *_hooks_all,
            *_humor_all,
            *_quote_all,
            *_poem_all,
            *_story_all,
            *_lyrics_all,
            *_originality_all,
        ]
    )
)

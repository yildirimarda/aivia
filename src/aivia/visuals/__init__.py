"""Visual stages: resolve beats into deterministic asset requests."""

from aivia.visuals.downgrade import *  # noqa: F403
from aivia.visuals.downgrade import __all__ as _downgrade_all
from aivia.visuals.image_stage import *  # noqa: F403
from aivia.visuals.image_stage import __all__ as _image_all
from aivia.visuals.qa import *  # noqa: F403
from aivia.visuals.qa import __all__ as _qa_all
from aivia.visuals.stock_stage import *  # noqa: F403
from aivia.visuals.stock_stage import __all__ as _stock_all
from aivia.visuals.strategy import *  # noqa: F403
from aivia.visuals.strategy import __all__ as _strategy_all
from aivia.visuals.video_stage import *  # noqa: F403
from aivia.visuals.video_stage import __all__ as _video_all

__all__ = [
    *_strategy_all,
    *_stock_all,
    *_image_all,
    *_video_all,
    *_qa_all,
    *_downgrade_all,
]

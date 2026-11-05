"""Visual stages: resolve beats into deterministic asset requests."""

from aivia.visuals.image_stage import *  # noqa: F403
from aivia.visuals.image_stage import __all__ as _image_all
from aivia.visuals.stock_stage import *  # noqa: F403
from aivia.visuals.stock_stage import __all__ as _stock_all
from aivia.visuals.strategy import *  # noqa: F403
from aivia.visuals.strategy import __all__ as _strategy_all

__all__ = [*_strategy_all, *_stock_all, *_image_all]

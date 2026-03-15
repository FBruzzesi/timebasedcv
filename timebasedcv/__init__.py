from __future__ import annotations

from importlib import metadata

from timebasedcv.core import ExpandingTimeSplit, RollingTimeSplit, TimeBasedSplit
from timebasedcv.splitstate import SplitState

__title__ = __name__
__version__ = metadata.version(__title__)

__all__ = (
    "ExpandingTimeSplit",
    "RollingTimeSplit",
    "SplitState",
    "TimeBasedSplit",
)

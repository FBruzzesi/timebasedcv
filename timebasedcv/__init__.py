from __future__ import annotations

from importlib import metadata

from timebasedcv.core import ExpandingTimeSplit
from timebasedcv.core import RollingTimeSplit
from timebasedcv.core import TimeBasedSplit
from timebasedcv.splitstate import SplitState

__title__ = __name__
__version__ = metadata.version(__title__)

__all__ = (
    "SplitState",
    "TimeBasedSplit",
    "ExpandingTimeSplit",
    "RollingTimeSplit",
    "_CoreTimeBasedSplit",
)

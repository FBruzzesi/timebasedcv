from importlib import metadata

from timebasedcv.splitstate import SplitState
from timebasedcv.timebasedsplit import (
    ExpandingTimeSplit,
    RollingTimeSplit,
    TimeBasedCVSplitter,
    TimeBasedSplit,
    _CoreTimeBasedSplit,
)

__title__ = __name__
__version__ = metadata.version(__title__)

__all__ = (
    "SplitState",
    "TimeBasedSplit",
    "ExpandingTimeSplit",
    "RollingTimeSplit",
    "TimeBasedCVSplitter",
    "_CoreTimeBasedSplit",
)

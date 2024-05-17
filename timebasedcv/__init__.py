from importlib import metadata

from timebasedcv.core import ExpandingTimeSplit, RollingTimeSplit, TimeBasedSplit
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

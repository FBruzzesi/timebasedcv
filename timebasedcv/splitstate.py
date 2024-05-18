from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from operator import le as less_or_equal
from typing import TYPE_CHECKING, Generic, Union

from timebasedcv.utils._funcs import pairwise, pairwise_comparison
from timebasedcv.utils._types import DateTimeLike

if sys.version_info >= (3, 11):  # pragma: no cover
    from typing import Self
else:  # pragma: no cover
    from typing_extensions import Self

from narwhals.dependencies import get_pandas

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


@dataclass(frozen=True)
class SplitState(Generic[DateTimeLike]):
    """A `SplitState` represents the state of a split in terms of its 4 cut/split points.

    Namely these are start and end of training set, start and end of forecasting/test set.

    The class ensures that the split is valid by checking that the attributes are of the correct type and are ordered
    chronologically.

    The class provides properties to calculate the length of the training set, forecast set, gap between them, and the
    total length of the split.

    Arguments:
        train_start: The start of the training set.
        train_end: The end of the training set.
        forecast_start: The start of the forecast set.
        forecast_end: The end of the forecast set.

    Raises:
        TypeError: If any of the attributes is not of type `datetime`, `date` or `pd.Timestamp`.
        ValueError: If the attributes are not ordered chronologically.
    """

    __slots__ = (
        "train_start",
        "train_end",
        "forecast_start",
        "forecast_end",
    )

    train_start: DateTimeLike
    train_end: DateTimeLike
    forecast_start: DateTimeLike
    forecast_end: DateTimeLike

    def __post_init__(self: Self) -> None:
        """Post init used to validate the `SplitState` instance attributes."""
        # Validate types
        _slots = self.__slots__
        _values = tuple(getattr(self, _attr) for _attr in _slots)
        _types = tuple(type(_value) for _value in _values)

        pd = get_pandas()

        if not (
            all(_type is datetime for _type in _types)
            or all(_type is date for _type in _types)
            or (pd is not None and all(_type is pd.Timestamp for _type in _types))
        ):
            # cfr: https://stackoverflow.com/questions/16991948/detect-if-a-variable-is-a-datetime-object
            msg = "All attributes must be of type `datetime`, `date` or `pd.Timestamp`."
            raise TypeError(msg)

        # Validate order
        _ordered = tuple(pairwise_comparison(_values, less_or_equal))

        if not all(_ordered):
            _error_msg = "\n".join(
                f"{s1}({v1}) is greater or equal to {s2}({v2})"
                for (s1, s2), (v1, v2), is_ordered in zip(pairwise(_slots), pairwise(_values), _ordered)
                if not is_ordered
            )
            msg = f"`{'`, `'.join(_slots)}` must be ordered. Found:\n{_error_msg}"
            raise ValueError(msg)

    @property
    def train_length(self: Self) -> timedelta:
        """Returns the time between `train_start` and `train_end`.

        Returns:
            A `timedelta` object representing the time between `train_start` and `train_end`.
        """
        return self.train_end - self.train_start

    @property
    def forecast_length(self: Self) -> timedelta:
        """Returns the time between `forecast_start` and `forecast_end`.

        Returns:
            A `timedelta` object representing the time between `forecast_start` and `forecast_end`.
        """
        return self.forecast_end - self.forecast_start

    @property
    def gap_length(self: Self) -> timedelta:
        """Returns the time between `train_end` and `forecast_start`.

        Returns:
            A `timedelta` object representing the time between `train_end` and `forecast_start`.
        """
        return self.forecast_start - self.train_end

    @property
    def total_length(self: Self) -> timedelta:
        """Returns the time between `train_start` and `forecast_end`.

        Returns:
            A `timedelta` object representing the time between `train_start` and `forecast_end`.
        """
        return self.forecast_end - self.train_start

    def __add__(self: Self, other: Union[timedelta, pd.Timedelta]) -> SplitState:
        """Adds `other` to each value of the state."""
        return SplitState(
            train_start=self.train_start + other,
            train_end=self.train_end + other,
            forecast_start=self.forecast_start + other,
            forecast_end=self.forecast_end + other,
        )

    def __sub__(self: Self, other: Union[timedelta, pd.Timedelta]) -> SplitState:
        """Subtracts other to each value of the state."""
        return SplitState(
            train_start=self.train_start - other,
            train_end=self.train_end - other,
            forecast_start=self.forecast_start - other,
            forecast_end=self.forecast_end - other,
        )

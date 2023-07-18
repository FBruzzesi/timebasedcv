from dataclasses import dataclass
from datetime import date, datetime, timedelta
from operator import lt as less_than

import pandas as pd

from timebasedcv._utils import pairwise, pairwise_comparison


@dataclass(frozen=True, slots=True)
class SplitState:
    """Class that represents the state of a split."""

    train_start: datetime | date | pd.Timestamp
    train_end: datetime | date | pd.Timestamp
    forecast_start: datetime | date | pd.Timestamp
    forecast_end: datetime | date | pd.Timestamp

    def __post_init__(self):
        """
        Post init used to validate the `SplitState` instance attributes.

        Raises:
            TypeError: If any of the attributes is not of type `datetime`, `date` or
                `pd.Timestamp`.
            ValueError: If the attributes are not ordered chronologically.
        """

        # Validate types
        _slots = self.__slots__
        _values = tuple(getattr(self, _attr) for _attr in _slots)
        _types = tuple(type(_value) for _value in _values)

        if not (
            all(_type is datetime for _type in _types)
            or all(_type is date for _type in _types)
            or all(_type is pd.Timestamp for _type in _types)
        ):
            # cfr: https://stackoverflow.com/questions/16991948/detect-if-a-variable-is-a-datetime-object
            raise TypeError(
                "All attributes must be of type `datetime`, `date` or `pd.Timestamp`."
            )

        # Validate order
        _ordered = tuple(pairwise_comparison(_values, less_than))

        if not all(_ordered):
            _error_msg = "\n".join(
                f"{s1}({v1}) is greater or equal to {s2}({v2})"
                for (s1, s2), (v1, v2), is_ordered in zip(
                    pairwise(_slots), pairwise(_values), _ordered
                )
                if not is_ordered
            )

            raise ValueError(
                f"`{'`, `'.join(_slots)}` must be ordered. Found:\n" f"{_error_msg}"
            )

    @property
    def train_time(self) -> timedelta:
        """Returns the time between `train_start` and `train_end`"""
        return self.train_end - self.train_start  # type: ignore

    @property
    def forecast_time(self) -> timedelta:
        """Returns the time between `forecast_start` and `forecast_end`"""
        return self.forecast_end - self.forecast_start  # type: ignore

    @property
    def gap(self) -> timedelta:
        """Returns the time between `train_end` and `forecast_start`"""
        return self.forecast_start - self.train_end  # type: ignore

    @property
    def total_time(self) -> timedelta:
        """Returns the time between `train_start` and `forecast_end`"""
        return self.forecast_end - self.train_start  # type: ignore

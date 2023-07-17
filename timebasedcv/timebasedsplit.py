from datetime import date, datetime, timedelta
from itertools import chain
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from timebasedcv._backends import BACKEND_TO_INDEXING_METHOD, DEFAULT_INDEXING_METHOD
from timebasedcv.splitstate import SplitState

ArrayLike = pd.DataFrame | np.ndarray  # |  pl.DataFrame
SeriesLike = pd.Series | np.ndarray  # |  pl.Series

FrequencyUnit = Literal[
    "days", "seconds", "microseconds", "milliseconds", "minutes", "hours", "weeks"
]


class TimeBasedSplit:
    """
    Class that generates splits which as time based splits, indepedently from the number
    of samples in each split.

    Arguments:
        frequency: The frequency of the time series. Must be one of "days", "seconds",
            "microseconds", "milliseconds", "minutes", "hours", "weeks".
            These are the only valid values for the `unit` argument of the `timedelta`.
        train_size: The size of the training set.
        forecast_horizon: The size of the forecast horizon.
        gap: The size of the gap between the training set and the forecast horizon.
        stride: The size of the stride between consecutive splits.
        window: The type of window to use. Must be one of "rolling" or "expanding".
        indexing_methods: A dictionary mapping types to indexing methods.
    """

    def __init__(
        self,
        frequency: FrequencyUnit,
        train_size: int,
        forecast_horizon: int,
        gap: int = 1,
        stride: int = 1,
        window: Literal["rolling", "expanding"] = "rolling",
    ):
        self.frequency_ = frequency
        self.train_size_ = train_size
        self.forecast_horizon_ = forecast_horizon
        self.gap_ = gap
        self.stride_ = stride
        self.window_ = window

        self.__post_init__()

    def __post_init__(self):
        """
        Post init used to validate the TimeSpacedSplit attributes

        Raises:
            ValueError: If `frequency` is not one of "days", "seconds", "microseconds",
                "milliseconds", "minutes", "hours", "weeks".
            ValueError: If `window` is not one of "rolling" or "expanding".
            TypeError: If `train_size`, `forecast_horizon`, `gap` or `stride` are not of
                type `int`.
            ValueError: If `train_size`, `forecast_horizon`, `gap` or `stride` are not
                strictly positive.
        """

        # Validate frequency
        _frequency_values = (
            "days",
            "seconds",
            "microseconds",
            "milliseconds",
            "minutes",
            "hours",
            "weeks",
        )

        if self.frequency_ not in _frequency_values:
            raise ValueError(
                f"`frequency` must be one of {_frequency_values}. Found {self.frequency_}"
            )

        # Validate window
        _window_values = ("rolling", "expanding")

        if self.window_ not in _window_values:
            raise ValueError(
                f"`window` must be one of {_window_values}. Found {self.window_}"
            )

        # Validate positive integer arguments
        _slot_names = ("train_size", "forecast_horizon_", "gap_", "stride_")
        _values = tuple(getattr(self, _attr) for _attr in _slot_names)
        _types = tuple(type(v) for v in _values)

        if not all(t is int for t in _types):
            print(_types)
            raise TypeError(
                f"(`{'`, `'.join(_slot_names)}`) arguments must be of type `int`. "
                f"Found (`{'`, `'.join(t._name_ for t in _types)}`)"
            )

        if not all(v > 0 for v in _values):
            raise ValueError(
                f"(`{'`, `'.join(_slot_names)}`) must be greater than 0. "
                f"Found ({', '.join(str(v) for v in _values)})"
            )

    @property
    def train_delta(self) -> timedelta:
        """Returns the `timedelta` object corresponding to the `train_size`"""
        return timedelta(**{self.frequency_: self.train_size_})

    @property
    def forecast_delta(self) -> timedelta:
        """Returns the `timedelta` object corresponding to the `forecast_horizon`"""
        return timedelta(**{self.frequency_: self.forecast_horizon_})

    @property
    def gap_delta(self) -> timedelta:
        """Returns the `timedelta` object corresponding to the `gap` and `frequency`."""
        return timedelta(**{self.frequency_: self.gap_})

    @property
    def stride_delta(self) -> timedelta:
        """Returns the `timedelta` object corresponding to `stride`."""
        return timedelta(**{self.frequency_: self.stride_})

    def _splits_from_period(
        self, time_start: datetime | date, time_end: datetime | date
    ) -> Iterable[SplitState]:
        """
        Generate splits from `time_start` to `time_end` based on the parameters passed
        to the class instance.

        Arguments:
            time_start: The start of the time period.
            time_end: The end of the time period.

        Returns:
            An iterable of `SplitState` instances.
        """

        if time_start >= time_end:
            raise ValueError("`time_start` must be before `time_end`.")

        start_training = current_date = time_start

        while current_date + self.train_delta + self.gap_delta < time_end:
            end_training: date = current_date + self.train_delta  # type: ignore

            start_forecast = end_training + self.gap_delta
            end_forecast = end_training + self.gap_delta + self.forecast_delta

            current_date = current_date + self.stride_delta

            yield SplitState(start_training, end_training, start_forecast, end_forecast)

            if self.window_ == "rolling":
                start_training = current_date

    def split(
        self,
        *arrays: tuple[ArrayLike, ...],
        time_series: SeriesLike,
    ) -> Iterable[tuple[ArrayLike, ...]]:
        """
        Returns a generator of splits.
        """
        n_arrays = len(arrays)
        if n_arrays == 0:
            raise ValueError("At least one array required as input")

        a0 = arrays[0]

        if a0.shape[0] != time_series.shape[0]:
            raise ValueError(
                "Time series and arrays must have the same length."
                f"Got {a0.shape[0]} and {time_series.shape[0]}"
            )

        index_method = BACKEND_TO_INDEXING_METHOD.get(type(a0), DEFAULT_INDEXING_METHOD)

        time_start, time_end = time_series.min(), time_series.max()

        for split in self._splits_from_period(time_start, time_end):
            train_mask = (time_series >= split.train_start) & (
                time_series <= split.train_end
            )
            forecast_mask = (time_series >= split.forecast_start) & (
                time_series <= split.forecast_end
            )

            yield tuple(
                chain.from_iterable(
                    (index_method(a, train_mask), index_method(a, forecast_mask))
                    for a in arrays
                )
            )

    def n_splits_of(self, time_series) -> int:
        """Returns the number of splits that can be generated from `time_series`"""

        time_start, time_end = time_series.min(), time_series.max()

        return len(tuple(self._splits_from_period(time_start, time_end)))


class ExpandingTimeSplit(TimeBasedSplit):
    """
    Alias for `TimeBasedSplit` with `window="expanding"`.
    """

    def __init__(
        self,
        frequency: FrequencyUnit,
        train_size: int,
        forecast_horizon: int,
        gap: int = 1,
        stride: int = 1,
    ):
        super().__init__(
            frequency,
            train_size,
            forecast_horizon,
            gap,
            stride,
            window="expanding",
        )


class RollingTimeSplit(TimeBasedSplit):
    """
    Alias for `TimeBasedSplit` with `window="rolling"`.
    """

    def __init__(
        self,
        frequency: FrequencyUnit,
        train_size: int,
        forecast_horizon: int,
        gap: int = 1,
        stride: int = 1,
    ):
        super().__init__(
            frequency,
            train_size,
            forecast_horizon,
            gap,
            stride,
            window="rolling",
        )

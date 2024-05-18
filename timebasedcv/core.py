from __future__ import annotations

import sys
from datetime import timedelta
from itertools import chain
from typing import TYPE_CHECKING, Generator, Literal, Tuple, TypeVar, Union, get_args, overload

import narwhals as nw

from timebasedcv.splitstate import SplitState
from timebasedcv.utils._backends import (
    BACKEND_TO_INDEXING_METHOD,
    default_indexing_method,
)
from timebasedcv.utils._types import (
    DateTimeLike,
    FrequencyUnit,
    ModeType,
    NullableDatetime,
    SeriesLike,
    TensorLike,
    WindowType,
)

if sys.version_info >= (3, 11):  # pragma: no cover
    from typing import Self
else:  # pragma: no cover
    from typing_extensions import Self

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

_frequency_values = get_args(FrequencyUnit)
_window_values = get_args(WindowType)
_mode_values = get_args(ModeType)

TL = TypeVar("TL", bound=TensorLike)


class _CoreTimeBasedSplit:
    """Base class for time based splits. This class is not meant to be used directly.

    `_CoreTimeBasedSplit` implements all the logics to set up a time based splits class.

    In particular it implements `_splits_from_period` which is used to generate splits from a given time period (from
    start to end dates) from the given arguments of the class (frequency, train_size, forecast_horizon, gap, stride and
    window type).

    Arguments:
        frequency: The frequency (or time unit) of the time series. Must be one of "days", "seconds", "microseconds",
            "milliseconds", "minutes", "hours", "weeks". These are the only valid values for the `unit` argument of
            `timedelta` from python `datetime` standard library.
        train_size: Defines the minimum number of time units required to be in the train set.
        forecast_horizon: Specifies the number of time units to forecast.
        gap: Sets the number of time units to skip between the end of the train set and the start of the forecast set.
        stride: How many time unit to move forward after each split. If `None` (or set to 0), the stride is equal to the
            `forecast_horizon` quantity.
        window: The type of window to use, either "rolling" or "expanding".
        mode: Determines in which orders the splits are generated, either "forward" (start to end) or "backward"
            (end to start).

    Raises:
        ValueError:
            - If `frequency` is not one of "days", "seconds", "microseconds", "milliseconds", "minutes", "hours",
            "weeks".
            - If `window` is not one of "rolling" or "expanding".
            - If `mode` is not one of "forward" or "backward"
            - If `train_size`, `forecast_horizon`, `gap` or `stride` are not strictly positive.
        TypeError: If `train_size`, `forecast_horizon`, `gap` or `stride` are not of type `int`.

    Although `_CoreTimeBasedSplit` is not meant to be used directly, it can be used as a template to create new time
    based splits classes.

    Examples:
        ```python
        from timebasedcv.core import _CoreTimeBasedSplit


        class MyTimeBasedSplit(_CoreTimeBasedSplit):
            ...

            def split(self, X, timeseries):
                '''Implement the split method to return a generator'''

                for split in self._splits_from_period(timeseries.min(), timeseries.max()):
                    # Do something with the split to compute the train and forecast sets
                    ...
                    yield X_train, y_test
        ```
    """

    def __init__(  # noqa: PLR0913
        self: Self,
        *,
        frequency: FrequencyUnit,
        train_size: int,
        forecast_horizon: int,
        gap: int = 0,
        stride: Union[int, None] = None,
        window: WindowType = "rolling",
        mode: ModeType = "forward",
    ) -> None:
        self.frequency_ = frequency
        self.train_size_ = train_size
        self.forecast_horizon_ = forecast_horizon
        self.gap_ = gap
        self.stride_ = stride or forecast_horizon
        self.window_ = window
        self.mode_ = mode

        self._validate_arguments()

    def _validate_arguments(self: Self) -> None:
        """Post init used to validate the TimeSpacedSplit attributes."""
        # Validate frequency
        if self.frequency_ not in _frequency_values:
            msg = f"`frequency` must be one of {_frequency_values}. Found {self.frequency_}"
            raise ValueError(msg)

        # Validate window
        if self.window_ not in _window_values:
            msg = f"`window` must be one of {_window_values}. Found {self.window_}"
            raise ValueError(msg)

        # Validate mode
        if self.mode_ not in _mode_values:
            msg = f"`mode` must be one of {_mode_values}. Found {self.mode_}"
            raise ValueError(msg)

        # Validate positive integer arguments
        _slot_names = ("train_size_", "forecast_horizon_", "gap_", "stride_")
        _values = tuple(getattr(self, _attr) for _attr in _slot_names)
        _lower_bounds = (1, 1, 0, 1)

        _types = tuple(type(v) for v in _values)

        if not all(t is int for t in _types):
            msg = (
                f"(`{'`, `'.join(_slot_names)}`) arguments must be of type `int`. "
                f"Found (`{'`, `'.join(str(t) for t in _types)}`)"
            )
            raise TypeError(msg)

        if not all(v >= lb for v, lb in zip(_values, _lower_bounds)):
            msg = (
                f"(`{'`, `'.join(_slot_names)}`) must be greater or equal than "
                f"({', '.join(map(str, _lower_bounds))}).\n"
                f"Found ({', '.join(str(v) for v in _values)})"
            )
            raise ValueError(msg)

    @property
    def name_(self: Self) -> str:
        return self.__class__.__name__

    def __repr__(self: Self) -> str:
        """Custom repr method."""
        _attrs = (
            "frequency_",
            "train_size_",
            "forecast_horizon_",
            "gap_",
            "stride_",
            "window_",
        )
        _values = tuple(getattr(self, _attr) for _attr in _attrs)
        _new_line_tab = "\n    "

        return f"{self.name_}" "(\n    " f"{_new_line_tab.join(f'{s} = {v}' for s, v in zip(_attrs, _values))}" "\n)"

    @property
    def train_delta(self: Self) -> timedelta:
        """Returns the `timedelta` object corresponding to the `train_size`."""
        return timedelta(**{str(self.frequency_): self.train_size_})

    @property
    def forecast_delta(self: Self) -> timedelta:
        """Returns the `timedelta` object corresponding to the `forecast_horizon`."""
        return timedelta(**{str(self.frequency_): self.forecast_horizon_})

    @property
    def gap_delta(self: Self) -> timedelta:
        """Returns the `timedelta` object corresponding to the `gap` and `frequency`."""
        return timedelta(**{str(self.frequency_): self.gap_})

    @property
    def stride_delta(self: Self) -> timedelta:
        """Returns the `timedelta` object corresponding to `stride`."""
        return timedelta(**{str(self.frequency_): self.stride_})

    def _splits_from_period(
        self: Self,
        time_start: DateTimeLike,
        time_end: DateTimeLike,
    ) -> Generator[SplitState, None, None]:
        """Generate splits from `time_start` to `time_end` based on the parameters passed to the class instance.

        This is the core iteration that generates splits. It is used by the `split` method to generate splits from the
        time series.

        Arguments:
            time_start: The start of the time period.
            time_end: The end of the time period.

        Returns:
            A generator of `SplitState` instances.
        """
        if time_start >= time_end:
            msg = "`time_start` must be before `time_end`."
            raise ValueError(msg)

        if self.mode_ == "forward":
            train_delta = self.train_delta
            forecast_delta = self.forecast_delta
            gap_delta = self.gap_delta
            stride_delta = self.stride_delta

            train_start = time_start
            train_end = time_start + train_delta
            forecast_start = train_end + gap_delta
            forecast_end = forecast_start + forecast_delta

        else:
            train_delta = -self.train_delta
            forecast_delta = -self.forecast_delta
            gap_delta = -self.gap_delta
            stride_delta = -self.stride_delta

            forecast_end = time_end
            forecast_start = forecast_end + forecast_delta
            train_end = forecast_start + gap_delta
            train_start = train_end + train_delta if self.window_ == "rolling" else time_start

        while (forecast_start <= time_end) and (train_start >= time_start) and (train_start <= train_end + train_delta):
            yield SplitState(train_start, train_end, forecast_start, forecast_end)

            # Update state values
            train_start = train_start + stride_delta if self.window_ == "rolling" else train_start
            train_end = train_end + stride_delta
            forecast_start = forecast_start + stride_delta
            forecast_end = forecast_end + stride_delta

    def n_splits_of(
        self: Self,
        *,
        time_series: Union[SeriesLike[DateTimeLike], None] = None,
        start_dt: NullableDatetime = None,
        end_dt: NullableDatetime = None,
    ) -> int:
        """Returns the number of splits that can be generated from `time_series`.

        Arguments:
            time_series: A time series data. If provided it should support `.min()` and `.max().
            start_dt: The start date and time of the time series. If not provided, it will be inferred from
                `time_series`.
            end_dt: The end date and time of the time series. If not provided, it will be inferred from
                `time_series`.

        Returns:
            The number of splits that can be generated from the given time series.

        Raises:
            ValueError:
                - If both `start_dt` and `end_dt` are provided and `start_dt` is greater than or equal to `end_dt`.
                - If neither `time_series` nor (`start_dt`, `end_dt`) pair is provided.
        """
        if (start_dt is not None) and (end_dt is not None):
            if start_dt >= end_dt:
                msg = "`start_dt` must be before `end_dt`."
                raise ValueError(msg)
            else:
                time_start, time_end = start_dt, end_dt
        elif time_series is not None:
            time_start, time_end = time_series.min(), time_series.max()
        else:
            msg = "Either `time_series` or (`start_dt`, `end_dt`) pair must be provided."
            raise ValueError(msg)

        return len(tuple(self._splits_from_period(time_start, time_end)))


class TimeBasedSplit(_CoreTimeBasedSplit):
    """`TimeBasedSplit` generates splits based on time periods, independently from the number of samples in each split.

    It inherits from [`_CoreTimeBasedSplit`][timebasedcv.core._CoreTimeBasedSplit] and it only implements the `.split()`
    method and logic.

    !!! warning "Differences with scikit-learn"

        `TimeBasedSplit` is **not** compatible with
        [scikit-learn CV Splitters](https://scikit-learn.org/stable/common_pitfalls.html#id3){:target="_blank"}.

        In fact, we have made the (opinioned) choice to:

        - Return the sliced arrays from `.split(...)`, while scikit-learn CV Splitters return train and test indices of
            the split.
        - Require to pass the time series as input to `.split(...)` method, while scikit-learn CV Splitters require to
            provide only `X, y, groups` to `.split(...)`.
        - Such time series is used to generate the boolean masks with which we slice the original arrays into train and
            test for each split.

        If you are looking for a class compatible with scikit-learn, check out our
        [`TimeBasedCVSplitter`][timebasedcv.sklearn.TimeBasedCVSplitter] in the `timebasedcv.sklearn` module.

    A few examples on how splits are generated given the parameters. Let:

    - `=` : train period unit
    - `*` : forecast period unit
    - `/` : gap period unit
    - `>` : stride period unit (absorbed in `=` if `window="expanding"`)

    Recall also that if `stride` is not provided, it is set to `forecast_horizon`:
    ```
    train_size, forecast_horizon, gap, stride, window = (4, 3, 0, None, "rolling")
    | ======= *****               |
    | >>>>> ======= *****         |
    |       >>>>> ======= *****   |
    |             >>>>> ======= * |

    train_size, forecast_horizon, gap, stride, window = (4, 3, 2, 2, "rolling")

    | ======= /// *****           |
    | >>> ======= /// *****       |
    |     >>> ======= /// *****   |
    |         >>> ======= /// *** |

    train_size, forecast_horizon, gap, stride, window = (4, 3, 2, 2, "expanding")
    | ======= /// *****           |
    | =========== /// *****       |
    | =============== /// *****   |
    | =================== /// *** |
    ```

    Arguments:
        frequency: The frequency (or time unit) of the time series. Must be one of "days", "seconds", "microseconds",
            "milliseconds", "minutes", "hours", "weeks". These are the only valid values for the `unit` argument of
            `timedelta` from python `datetime` standard library.
        train_size: Defines the minimum number of time units required to be in the train set.
        forecast_horizon: Specifies the number of time units to forecast.
        gap: Sets the number of time units to skip between the end of the train set and the start of the forecast set.
        stride: How many time unit to move forward after each split. If `None` (or set to 0), the stride is equal to the
            `forecast_horizon` quantity.
        window: The type of window to use, either "rolling" or "expanding".
        mode: Determines in which orders the splits are generated, either "forward" (start to end) or "backward"
            (end to start).

    Raises:
        ValueError:
            - If `frequency` is not one of "days", "seconds", "microseconds", "milliseconds", "minutes", "hours",
            "weeks".
            - If `window` is not one of "rolling" or "expanding".
            - If `mode` is not one of "forward" or "backward"
            - If `train_size`, `forecast_horizon`, `gap` or `stride` are not strictly positive.
        TypeError: If `train_size`, `forecast_horizon`, `gap` or `stride` are not of type `int`.


    Examples:
        ```python
        # Let's first generate some data
        import pandas as pd
        import numpy as np

        RNG = np.random.default_rng(seed=42)

        dates = pd.Series(pd.date_range("2023-01-01", "2023-01-31", freq="D"))
        size = len(dates)

        df = (
            pd.concat(
                [
                    pd.DataFrame(
                        {
                            "time": pd.date_range(start, end, periods=_size, inclusive="left"),
                            "a": RNG.normal(size=_size - 1),
                            "b": RNG.normal(size=_size - 1),
                        }
                    )
                    for start, end, _size in zip(dates[:-1], dates[1:], RNG.integers(2, 24, size - 1))
                ]
            )
            .reset_index(drop=True)
            .assign(y=lambda t: t[["a", "b"]].sum(axis=1) + RNG.normal(size=t.shape[0]) / 25)
        )

        df.set_index("time").resample("D").agg(count=("y", np.size)).head(5)
        ```

        ```terminal
                    count
        time
        2023-01-01      2
        2023-01-02     18
        2023-01-03     15
        2023-01-04     10
        2023-01-05     10
        ```

        Now let's run split the data with the provided `TimeBasedSplit` instance:

        ```py
        from timebasedcv import TimeBasedSplit


        tbs = TimeBasedSplit(
            frequency="days",
            train_size=10,
            forecast_horizon=5,
            gap=1,
            stride=3
        )
        X, y, time_series = df.loc[:, ["a", "b"]], df["y"], df["time"]

        for X_train, X_forecast, y_train, y_forecast in tbs.split(X, y, time_series=time_series):
            print(f"Train: {X_train.shape}, Forecast: {X_forecast.shape}")
        ```

        ```terminal
        Train: (100, 2), Forecast: (51, 2)
        Train: (114, 2), Forecast: (50, 2)
        ...
        Train: (124, 2), Forecast: (40, 2)
        Train: (137, 2), Forecast: (22, 2)
        ```
    """

    @overload
    def split(
        self: Self,
        *arrays: TL,
        time_series: SeriesLike[DateTimeLike],
        start_dt: NullableDatetime = None,
        end_dt: NullableDatetime = None,
        return_splitstate: Literal[False],
    ) -> Generator[Tuple[TL, ...], None, None]: ...  # pragma: no cover

    @overload
    def split(
        self: Self,
        *arrays: TL,
        time_series: SeriesLike[DateTimeLike],
        start_dt: NullableDatetime = None,
        end_dt: NullableDatetime = None,
        return_splitstate: Literal[True],
    ) -> Generator[Tuple[Tuple[TL, ...], SplitState], None, None]: ...  # pragma: no cover

    @overload
    def split(
        self: Self,
        *arrays: TL,
        time_series: SeriesLike[DateTimeLike],
        start_dt: NullableDatetime = None,
        end_dt: NullableDatetime = None,
        return_splitstate: bool = False,
    ) -> Generator[
        Union[Tuple[TL, ...], Tuple[Tuple[TL, ...], SplitState]],
        None,
        None,
    ]: ...  # pragma: no cover

    def split(
        self: Self,
        *arrays: TL,
        time_series: SeriesLike[DateTimeLike],
        start_dt: NullableDatetime = None,
        end_dt: NullableDatetime = None,
        return_splitstate: bool = False,
    ) -> Generator[Union[Tuple[TL, ...], Tuple[Tuple[TL, ...], SplitState]], None, None]:
        """Returns a generator of split arrays based on the `time_series`.

        The `time_series` argument is split on split state values to create boolean masks for training - from train_
        start (included) to train_end (excluded) - and forecast - from forecast_start (included) to forecast_end
        (excluded). These masks are then used to index the arrays passed as arguments.

        The `start_dt` and `end_dt` arguments can be used to specify the start and end of the time period. If provided,
        they are used in place of the `time_series.min()` and `time_series.max()` respectively.

        This is useful because the series does not necessarely starts from the first date and/or terminates in the last
        date of the time period of interest.

        The `return_splitstate` argument can be used to return the `SplitState` instance for each split. This can be
        useful if a particular logic has to be applied only on specific cases (e.g. if first day of the week, then
        retrain a model).

        By returning the split state, the user has the freedom and flexibility to apply any logic.

        Arguments:
            *arrays: The arrays to split. Must have the same length as `time_series`.
            time_series: The time series used to create boolean mask for splits. It is not required to be sorted, but it
                must support:

                - comparison operators (with other date-like objects).
                - bitwise operators (with other boolean arrays).
                - `.min()` and `.max()` methods.
                - `.shape` attribute.
            start_dt: The start of the time period. If provided, it is used in place of the `time_series.min()`.
            end_dt: The end of the time period. If provided,it is used in place of the `time_series.max()`.
            return_splitstate: Whether to return the `SplitState` instance for each split.

                - If True, the generator yields tuples of the form `(train_forecast_arrays, split_state)`, where
                `train_forecast_arrays` is a tuple of arrays containing the training and forecast data, and
                `split_state` is a `SplitState` instance representing the current split.
                - If False, the generator yields tuples of the form `train_forecast_arrays`.

        Returns:
            A generator of tuples of arrays containing the training and forecast data.
                Each tuple corresponds to a split generated by the `TimeBasedSplit` instance. If `return_splitstate` is
                True, each tuple is of the form `(train_forecast_arrays, split_state)`, othersiwe it is of the form
                `train_forecast_arrays`.

        Raises:
            ValueError:
                - If no arrays are provided as input.
                - If the arrays provided have different lengths.
                - If the length of the time series does not match the length of the arrays.
        """
        n_arrays = len(arrays)
        if n_arrays == 0:
            msg = "At least one array required as input"
            raise ValueError(msg)

        ts_shape = time_series.shape
        if len(ts_shape) != 1:
            msg = f"Time series must be 1-dimensional. Got {len(ts_shape)} dimensions."
            raise ValueError(msg)

        arrays_: Tuple[Union[nw.DataFrame, nw.Series, np.ndarray], ...] = tuple(
            nw.from_native(array, eager_only=True, allow_series=True, strict=False) for array in arrays
        )
        time_series_: Union[nw.Series, np.ndarray] = nw.from_native(time_series, series_only=True, strict=False)

        a0 = arrays[0]
        arr_len = a0.shape[0]

        if n_arrays > 1 and not all(a.shape[0] == arr_len for a in arrays_[1:]):
            msg = f"All arrays must have the same length. Got {[a.shape[0] for a in arrays_]}"
            raise ValueError(msg)

        if arr_len != ts_shape[0]:
            msg = f"Time series and arrays must have the same length. Got {arr_len} and {ts_shape[0]}"
            raise ValueError(msg)

        time_start, time_end = start_dt or time_series_.min(), end_dt or time_series_.max()

        if time_start >= time_end:
            msg = "`time_start` must be before `time_end`."
            raise ValueError(msg)

        _index_methods = tuple(BACKEND_TO_INDEXING_METHOD.get(str(type(a)), default_indexing_method) for a in arrays_)
        for split in self._splits_from_period(time_start, time_end):
            train_mask = (time_series_ >= split.train_start) & (time_series_ < split.train_end)
            forecast_mask = (time_series_ >= split.forecast_start) & (time_series_ < split.forecast_end)

            train_forecast_arrays = tuple(
                chain.from_iterable(
                    (
                        nw.to_native(_idx_method(_arr, train_mask), strict=False),
                        nw.to_native(_idx_method(_arr, forecast_mask), strict=False),
                    )
                    for _arr, _idx_method in zip(arrays_, _index_methods)
                ),
            )

            if return_splitstate:
                yield train_forecast_arrays, split
            else:
                yield train_forecast_arrays


class ExpandingTimeSplit(TimeBasedSplit):  # pragma: no cover
    """Alias for `TimeBasedSplit(..., window="expanding")`."""

    def __init__(  # noqa: PLR0913
        self: Self,
        *,
        frequency: FrequencyUnit,
        train_size: int,
        forecast_horizon: int,
        gap: int = 0,
        stride: Union[int, None] = None,
        mode: ModeType,
    ) -> None:
        super().__init__(
            frequency=frequency,
            train_size=train_size,
            forecast_horizon=forecast_horizon,
            gap=gap,
            stride=stride,
            window="expanding",
            mode=mode,
        )


class RollingTimeSplit(TimeBasedSplit):  # pragma: no cover
    """Alias for `TimeBasedSplit(..., window="rolling")`."""

    def __init__(  # noqa: PLR0913
        self: Self,
        *,
        frequency: FrequencyUnit,
        train_size: int,
        forecast_horizon: int,
        gap: int = 0,
        stride: Union[int, None] = None,
        mode: ModeType,
    ) -> None:
        super().__init__(
            frequency=frequency,
            train_size=train_size,
            forecast_horizon=forecast_horizon,
            gap=gap,
            stride=stride,
            window="rolling",
            mode=mode,
        )

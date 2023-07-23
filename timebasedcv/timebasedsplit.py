from datetime import timedelta
from itertools import chain
from typing import Iterable, Tuple, Union, get_args

import numpy as np

from timebasedcv.splitstate import SplitState
from timebasedcv.utils._backends import (
    BACKEND_TO_INDEXING_METHOD,
    default_indexing_method,
)
from timebasedcv.utils._types import (
    DateTimeLike,
    FrequencyUnit,
    SeriesLike,
    TensorLike,
    WindowType,
)

_frequency_values = get_args(FrequencyUnit)
_window_values = get_args(WindowType)


class _CoreTimeBasedSplit:
    """
    Base class for time based splits. This class is not meant to be used directly.

    `_CoreTimeBasedSplit` implements all the logics to set up a time based splits class.

    In particular it implements `_splits_from_period` which is used to generate splits
    from a given time period (from start to end dates) from the given arguments of the
    class (frequency, train_size, forecast_horizon, gap, stride and window type).

    Arguments:
        frequency: The frequency of the time series. Must be one of "days", "seconds",
            "microseconds", "milliseconds", "minutes", "hours", "weeks".
            These are the only valid values for the `unit` argument of the `timedelta`.
        train_size: The size of the training set.
        forecast_horizon: The size of the forecast horizon.
        gap: The size of the gap between the training set and the forecast horizon.
        stride: The size of the stride between consecutive splits. Notice that if stride
            is not provided (or set to 0), it is set to `forecast_horizon`.
        window: The type of window to use. Must be one of "rolling" or "expanding".

    Raises:
        ValueError: If `frequency` is not one of "days", "seconds", "microseconds",
            "milliseconds", "minutes", "hours", "weeks".
        ValueError: If `window` is not one of "rolling" or "expanding".
        TypeError: If `train_size`, `forecast_horizon`, `gap` or `stride` are not of
            type `int`.
        ValueError: If `train_size`, `forecast_horizon`, `gap` or `stride` are not
            strictly positive.

    Although `_CoreTimeBasedSplit` is not meant to be used directly, it can be used as
    a template to create new time based splits classes.

    Usage:
    ```python
    from timebasedcv import _CoreTimeBasedSplit

    class MyTimeBasedSplit(_CoreTimeBasedSplit):

        def split(self, X, timeseries):
            # Implement the split method to return a generator

            for split in self._splits_from_period(timeseries.min(), timeseries.max()):

                # Do something with the split to compute the train and forecast sets
                ...
                yield X_train, y_test
    ```
    """

    name_ = "_CoreTimeBasedSplit"

    def __init__(
        self,
        frequency: FrequencyUnit,
        train_size: int,
        forecast_horizon: int,
        gap: int = 0,
        stride: Union[int, None] = None,
        window: WindowType = "rolling",
    ):
        self.frequency_ = frequency
        self.train_size_ = train_size
        self.forecast_horizon_ = forecast_horizon
        self.gap_ = gap
        self.stride_ = stride or forecast_horizon
        self.window_ = window

        self._validate_arguments()

    def _validate_arguments(self):
        """
        Post init used to validate the TimeSpacedSplit attributes
        """

        # Validate frequency
        if self.frequency_ not in _frequency_values:
            raise ValueError(
                f"`frequency` must be one of {_frequency_values}. Found {self.frequency_}"
            )

        # Validate window
        if self.window_ not in _window_values:
            raise ValueError(
                f"`window` must be one of {_window_values}. Found {self.window_}"
            )

        # Validate positive integer arguments
        _slot_names = ("train_size_", "forecast_horizon_", "gap_", "stride_")
        _values = tuple(getattr(self, _attr) for _attr in _slot_names)
        _lower_bounds = (1, 1, 0, 1)

        _types = tuple(type(v) for v in _values)

        if not all(t is int for t in _types):
            raise TypeError(
                f"(`{'`, `'.join(_slot_names)}`) arguments must be of type `int`. "
                f"Found (`{'`, `'.join(str(t) for t in _types)}`)"
            )

        if not all(v >= lb for v, lb in zip(_values, _lower_bounds)):
            raise ValueError(
                f"(`{'`, `'.join(_slot_names)}`) must be greater or equal than"
                f"({', '.join(map(str, _lower_bounds))}).\n"
                f"Found ({', '.join(str(v) for v in _values)})"
            )

    def __repr__(self) -> str:
        """Custom repr method"""

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
        return (
            f"{self.name_}"
            "(\n    "
            f"{_new_line_tab.join(f'{s} = {v}' for s, v in zip(_attrs, _values))}"
            "\n)"
        )

    @property
    def train_delta(self) -> timedelta:
        """Returns the `timedelta` object corresponding to the `train_size`"""
        return timedelta(**{str(self.frequency_): self.train_size_})

    @property
    def forecast_delta(self) -> timedelta:
        """Returns the `timedelta` object corresponding to the `forecast_horizon`"""
        return timedelta(**{str(self.frequency_): self.forecast_horizon_})

    @property
    def gap_delta(self) -> timedelta:
        """Returns the `timedelta` object corresponding to the `gap` and `frequency`."""
        return timedelta(**{str(self.frequency_): self.gap_})

    @property
    def stride_delta(self) -> timedelta:
        """Returns the `timedelta` object corresponding to `stride`."""
        return timedelta(**{str(self.frequency_): self.stride_})

    def _splits_from_period(
        self, time_start: DateTimeLike, time_end: DateTimeLike
    ) -> Iterable[SplitState]:
        """
        Generate splits from `time_start` to `time_end` based on the parameters passed
        to the class instance.

        This is the core iteration that generates splits. It is used by the `split`
        method to generate splits from the time series.

        Arguments:
            time_start: The start of the time period.
            time_end: The end of the time period.

        Returns:
            An iterable of `SplitState` instances.
        """

        if time_start >= time_end:
            raise ValueError("`time_start` must be before `time_end`.")

        train_start = current_time = time_start

        train_delta = self.train_delta
        forecast_delta = self.forecast_delta
        gap_delta = self.gap_delta
        stride_delta = self.stride_delta

        while current_time + train_delta + gap_delta < time_end:
            train_end = current_time + train_delta
            forecast_start = train_end + gap_delta
            forecast_end = forecast_start + forecast_delta

            current_time = current_time + stride_delta

            yield SplitState(train_start, train_end, forecast_start, forecast_end)

            if self.window_ == "rolling":
                train_start = current_time

    def n_splits_of(
        self,
        *,
        time_series: Union[SeriesLike[DateTimeLike], None] = None,
        start_dt: Union[DateTimeLike, None] = None,
        end_dt: Union[DateTimeLike, None] = None,
    ) -> int:
        """Returns the number of splits that can be generated from `time_series`"""

        if time_series is None and (start_dt is None or end_dt is None):
            raise ValueError(
                "Either `time_series` or `start_dt` and `end_dt` must be provided."
            )

        if (start_dt and end_dt) and (start_dt >= end_dt):
            raise ValueError("`start_dt` must be before `end_dt`.")

        time_start, time_end = start_dt or time_series.min(), end_dt or time_series.max()

        return len(tuple(self._splits_from_period(time_start, time_end)))

    def split(self, *args, **kwargs):
        """
        Template method that returns a generator of splits.

        Raises:
            NotImplementedError: The method is not implemented directly
        """
        raise NotImplementedError


class TimeBasedSplit(_CoreTimeBasedSplit):
    """
    Class that generates splits based on time values, indepedently from the number of
    samples in each split.

    Arguments:
        frequency: The frequency of the time series. Must be one of "days", "seconds",
            "microseconds", "milliseconds", "minutes", "hours", "weeks".
            These are the only valid values for the `unit` argument of the `timedelta`.
        train_size: The size of the training set.
        forecast_horizon: The size of the forecast horizon.
        gap: The size of the gap between the training set and the forecast horizon.
        stride: The size of the stride between consecutive splits. Notice that if stride
            is not provided (or set to 0), it is set to `forecast_horizon`.
        window: The type of window to use. Must be one of "rolling" or "expanding".

    Raises:
        ValueError: If `frequency` is not one of "days", "seconds", "microseconds",
            "milliseconds", "minutes", "hours", "weeks".
        ValueError: If `window` is not one of "rolling" or "expanding".
        TypeError: If `train_size`, `forecast_horizon`, `gap` or `stride` are not of
            type `int`.
        ValueError: If `train_size`, `forecast_horizon`, `gap` or `stride` are not
            strictly positive.

    Usage:
    ```python
    import pandas as pd
    import numpy as np

    from timebasedcv import TimeBasedSplit

    tbs = TimeBasedSplit(
        frequency="days",
        train_size=30,
        forecast_horizon=7,
        gap=0,
        stride=3,
        window="rolling",
    )

    dates = pd.Series(pd.date_range("2023-01-01", "2023-12-31", freq="D"))
    size = len(dates)

    df = (pd.DataFrame(data=np.random.randn(size, 2), columns=["a", "b"])
        .assign(y=lambda t : t[["a", "b"]].sum(axis=1))
        )

    X, y = df[["a", "b"]], df["y"]

    print(f"Number of splits: {tbs.n_splits_of(time_series=dates)}")

    for X_train, X_forecast, y_train, y_forecast in tbs.split(X, y, time_series=dates):
        print(f"Train: {X_train.shape}, Forecast: {X_forecast.shape}")
    ```

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
    """

    name_ = "TimeBasedSplit"

    def split(
        self,
        *arrays: TensorLike,
        time_series: SeriesLike[DateTimeLike],
        start_dt: Union[DateTimeLike, None] = None,
        end_dt: Union[DateTimeLike, None] = None,
        return_splitstate: bool = False,
    ) -> Iterable[
        Union[Tuple[TensorLike, ...], Tuple[Tuple[TensorLike, ...], SplitState]]
    ]:
        """
        Returns a generator of splitted arrays based on the `time_series`.

        The `time_series` argument is split on split state values to create boolean
        masks for training [from train_start (included) to train_end (excluded)] and
        forecast [from forecast_start (included) to forecast_end (excluded)]. These
        masks are then used to index the arrays passed as arguments.

        The `start_dt` and `end_dt` arguments can be used to specify the start and end
        of the time period. If provided, they are used in place of the `time_series.min()`
        and `time_series.max()` respectively.

        This is useful because the series does not necessarely starts from the first date
        and/or terminates in the last date of the time period of interest.

        The `return_splitstate` argument can be used to return the `SplitState` instance
        for each split. This can be useful if a particular logic has to be applied only
        on specific cases (e.g. if first day of the week, then retrain a model).

        By returning the split state, the user has the freedom and flexibility to apply
        any logic.

        Arguments:
            *arrays: The arrays to split. Must have the same length as `time_series`.
            time_series: The time series used to create boolean mask for splits.
                It is not required to be sorted, but it must support:

                - comparison operators (with other date-like objects).
                - bitwise operators (with other boolean arrays).
                - `.min()` and `.max()` methods.
                - `.shape` attribute.
            start_dt: The start of the time period. If provided, it is used in place of
                 the `time_series.min()`.
            end_dt: The end of the time period. If provided,it is used in place of
                 the `time_series.max()`.
            return_splitstate: Whether to return the `SplitState` instance for each split.
                If True, the generator yields tuples of the form
                `(train_forecast_arrays, split_state)`, where `train_forecast_arrays` is a
                tuple of arrays containing the training and forecast data,
                and `split_state` is a `SplitState` instance representing the current
                split. If False, the generator yields tuples of the form
                `train_forecast_arrays`.

        Returns:
            A generator of tuples of arrays containing the training and forecast data.
                Each tuple corresponds to a split generated by the `TimeBasedSplit`
                instance. If `return_splitstate` is True, each tuple is of the form
                `(train_forecast_arrays, split_state)`, othersiwe it is of the form
                `train_forecast_arrays`.

        Raises:
            ValueError: If no arrays are provided as input.
            ValueError: If the arrays provided have different lengths.
            ValueError: If the length of the time series does not match the length of the
                arrays.
        """
        n_arrays = len(arrays)
        if n_arrays == 0:
            raise ValueError("At least one array required as input")

        ts_shape = time_series.shape
        if len(ts_shape) != 1:
            raise ValueError(
                f"Time series must be 1-dimensional. Got {len(ts_shape)} dimensions."
            )

        a0 = arrays[0]
        arr_len = a0.shape[0]

        if n_arrays > 1 and not all(a.shape[0] == arr_len for a in arrays[1:]):
            raise ValueError(
                "All arrays must have the same length. "
                f"Got {[a.shape[0] for a in arrays]}"
            )

        if arr_len != ts_shape[0]:
            raise ValueError(
                "Time series and arrays must have the same length."
                f"Got {arr_len} and {ts_shape[0]}"
            )

        time_start, time_end = start_dt or time_series.min(), end_dt or time_series.max()

        if time_start >= time_end:
            raise ValueError("`time_start` must be before `time_end`.")

        _arr_types = tuple(str(type(a)) for a in arrays)
        _index_methods = tuple(
            BACKEND_TO_INDEXING_METHOD.get(_type, default_indexing_method)
            for _type in _arr_types
        )
        for split in self._splits_from_period(time_start, time_end):
            train_mask = (time_series >= split.train_start) & (
                time_series < split.train_end
            )
            forecast_mask = (time_series >= split.forecast_start) & (
                time_series < split.forecast_end
            )

            train_forecast_arrays = tuple(
                chain.from_iterable(
                    (_idx_method(_arr, train_mask), _idx_method(_arr, forecast_mask))
                    for _arr, _idx_method in zip(arrays, _index_methods)
                )
            )

            if return_splitstate:
                yield train_forecast_arrays, split
            else:
                yield train_forecast_arrays


class ExpandingTimeSplit(TimeBasedSplit):  # pragma: no cover
    """
    Alias for `TimeBasedSplit` with `window="expanding"`.
    """

    name_ = "ExpandingTimeSplit"

    def __init__(
        self,
        frequency: FrequencyUnit,
        train_size: int,
        forecast_horizon: int,
        gap: int = 0,
        stride: Union[int, None] = None,
    ):
        super().__init__(
            frequency,
            train_size,
            forecast_horizon,
            gap,
            stride,
            window="expanding",
        )


class RollingTimeSplit(TimeBasedSplit):  # pragma: no cover
    """
    Alias for `TimeBasedSplit` with `window="rolling"`.
    """

    name_ = "RollingTimeSplit"

    def __init__(
        self,
        frequency: FrequencyUnit,
        train_size: int,
        forecast_horizon: int,
        gap: int = 0,
        stride: Union[int, None] = None,
    ):
        super().__init__(
            frequency,
            train_size,
            forecast_horizon,
            gap,
            stride,
            window="rolling",
        )


class TimeBasedCVSplitter(TimeBasedSplit):
    """
    The TimeBasedCVSplitter class conforms with scikit-learn CV Splitters API
    and generates splits based on time values, indepedently from the number
    of samples in each split.

    In order to achive such behaviour we include the arguments of
    `TimeBasedSplit.split()` method (namely `time_series`, `start_dt` and
    `end_dt`) in the constructor (a.k.a. `__init__` method) and store them
    for future use in its `split` and `get_n_splits` methods.

    In this way we can restrict the arguments of `split` and `get_n_splits` to
    the arrays to split (i.e. `X`, `y` and `groups`), which are the only
    arguments required by scikit-learn CV Splitters.

    Arguments:
        frequency: The frequency of the time series. Must be one of "days", "seconds",
            "microseconds", "milliseconds", "minutes", "hours", "weeks".
            These are the only valid values for the `unit` argument of the `timedelta`.
        train_size: The size of the training set.
        forecast_horizon: The size of the forecast horizon.
        time_series: The time series used to create boolean mask for splits.
            It is not required to be sorted, but it must support:

            - comparison operators (with other date-like objects).
            - bitwise operators (with other boolean arrays).
            - `.min()` and `.max()` methods.
            - `.shape` attribute.
        gap: The size of the gap between the training set and the forecast horizon.
        stride: The size of the stride between consecutive splits. Notice that if stride
            is not provided (or set to 0), it is set to `forecast_horizon`.
        window: The type of window to use. Must be one of "rolling" or "expanding".
        start_dt: The start of the time period. If provided, it is used in place of
            the `time_series.min()`.
        end_dt: The end of the time period. If provided,it is used in place of
            the `time_series.max()`.

    Raises:
        ValueError: If `frequency` is not one of "days", "seconds", "microseconds",
            "milliseconds", "minutes", "hours", "weeks".
        ValueError: If `window` is not one of "rolling" or "expanding".
        TypeError: If `train_size`, `forecast_horizon`, `gap` or `stride` are not of
            type `int`.
        ValueError: If `train_size`, `forecast_horizon`, `gap` or `stride` are not
            strictly positive.

    Usage:
    ```python
    import pandas as pd
    import numpy as np

    from sklearn.linear_model import Ridge
    from sklearn.model_selection import RandomizedSearchCV

    from timebasedcv import TimeBasedCVSplitter

    start_dt = pd.Timestamp(2023, 1, 1)
    end_dt = pd.Timestamp(2023, 1, 31)

    time_series = pd.Series(pd.date_range(start_dt, end_dt, freq="D"))
    size = len(time_series)

    df = (pd.DataFrame(data=np.random.randn(size, 2), columns=["a", "b"])
        .assign(y=lambda t: t[["a", "b"]].sum(axis=1)),
    )

    X, y = df[["a", "b"]], df["y"]

    cv = TimeBasedCVSplitter(
        frequency="days",
        train_size=7,
        forecast_horizon=11,
        gap=0,
        stride=1,
        window="rolling",
        time_series=time_series,
        start_dt=start_dt,
        end_dt=end_dt,
    )

    param_grid = {
        "alpha": np.linspace(0.1, 2, 10),
        "fit_intercept": [True, False],
        "positive": [True, False],
    }

    random_search_cv = RandomizedSearchCV(
        estimator=Ridge(),
        param_distributions=param_grid,
        cv=cv,
        n_jobs=-1,
    ).fit(X, y)
    ```
    """

    name_ = "TimeBasedCVSplitter"

    def __init__(
        self,
        frequency: FrequencyUnit,
        train_size: int,
        forecast_horizon: int,
        time_series: SeriesLike[DateTimeLike],
        gap: int = 0,
        stride: Union[int, None] = None,
        window: WindowType = "rolling",
        start_dt: Union[DateTimeLike, None] = None,
        end_dt: Union[DateTimeLike, None] = None,
    ):
        super().__init__(
            frequency=frequency,
            train_size=train_size,
            forecast_horizon=forecast_horizon,
            gap=gap,
            stride=stride,
            window=window,
        )

        self.time_series_ = time_series  # type: ignore
        self.start_dt_ = start_dt  # type: ignore
        self.end_dt_ = end_dt  # type: ignore

        self.n_splits = self._compute_n_splits()
        self.size_ = time_series.shape[0]

    def split(  # type: ignore
        self,
        X: Union[TensorLike, SeriesLike, None] = None,
        y: Union[TensorLike, SeriesLike, None] = None,
        groups: Union[TensorLike, SeriesLike, None] = None,
    ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """
        Split method compatible with scikit-learn CV splitters.

        Arguments:
            X: Unused is the split, exists for compatibility, checked if not None.
            y: Unused is the split, exists for compatibility, checked if not None.
            groups: Unused is the split, exists for compatibility, checked if not None.

        Returns:
            A generator of tuples of arrays containing the training and forecast data.
        """
        self._validate_split_args(self.size_, X, y, groups)

        _indexes = np.arange(self.size_)

        return super().split(
            _indexes,
            time_series=self.time_series_,
            start_dt=self.start_dt_,
            end_dt=self.end_dt_,
            return_splitstate=False,
        )  # type: ignore

    def get_n_splits(
        self,
        X: Union[TensorLike, SeriesLike, None] = None,
        y: Union[TensorLike, SeriesLike, None] = None,
        groups: Union[TensorLike, SeriesLike, None] = None,
    ) -> int:
        """
        Returns the number of splits that can be generated from the instance.

        Arguments:
            X: Unused, exists for compatibility, checked if not None.
            y: Unused, exists for compatibility, checked if not None.
            groups: Unused, exists for compatibility, checked if not None.

        Returns:
            The number of splits that can be generated from `time_series`.
        """
        self._validate_split_args(self.size_, X, y, groups)
        return self.n_splits

    def _compute_n_splits(self) -> int:
        """Computes number of splits just once in the init"""

        time_start = self.start_dt_ or self.time_series_.min()
        time_end = self.end_dt_ or self.time_series_.max()

        return len(tuple(self._splits_from_period(time_start, time_end)))

    @staticmethod
    def _validate_split_args(
        size: int,
        X: Union[TensorLike, SeriesLike, None] = None,
        y: Union[TensorLike, SeriesLike, None] = None,
        groups: Union[TensorLike, SeriesLike, None] = None,
    ) -> None:
        """
        Validates the arguments passed to the `split` and `get_n_splits` methods.
        """
        if X is not None and X.shape[0] != size:
            raise ValueError(
                f"X.shape[0] ({X.shape[0]}) != time_series.shape[0] ({size})"
            )

        if y is not None and y.shape[0] != size:
            raise ValueError(
                f"y.shape[0] ({y.shape[0]}) != time_series.shape[0] ({size})"
            )

        if groups is not None and groups.shape[0] != size:
            raise ValueError(
                f"groups.shape[0] ({groups.shape[0]}) != "
                f"time_series.shape[0] ({size})"
            )

from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING, Generator, Tuple, Union

import numpy as np

from timebasedcv.core import TimeBasedSplit

if sys.version_info >= (3, 11):  # pragma: no cover
    from typing import Self
else:  # pragma: no cover
    from typing_extensions import Self

from importlib.metadata import version

if (sklearn_version := version("scikit-learn")) and tuple(
    int(re.sub(r"\D", "", str(v))) for v in sklearn_version.split(".")
) < (0, 19, 0):  # pragma: no cover
    msg = (
        f"scikit-learn>=0.19.0 is required for this module. Found version {sklearn_version}.\nInstall it with "
        "`python -m pip install scikit-learn>=0.19.0` or `python -m pip install timebasedcv[scikit-learn]`",
    )
    raise ImportError(msg)
else:  # pragma: no cover
    from sklearn.model_selection._split import _BaseKFold


if TYPE_CHECKING:  # pragma: no cover
    from datetime import date, datetime

    import pandas as pd
    from numpy.typing import NDArray

    from timebasedcv.utils._types import (
        FrequencyUnit,
        ModeType,
        NullableDatetime,
        SeriesLike,
        WindowType,
    )


class TimeBasedCVSplitter(_BaseKFold):
    """The `TimeBasedCVSplitter` is a scikit-learn compatible CV Splitter that generates splits based on time values.

    The number of sample in each split is independent of the number of splits but based purely on the timestamp of the
    sample.

    In order to achieve such behaviour we include the arguments of
    [`TimeBasedSplit.split()`][timebasedcv.core.TimeBasedSplit.split] method (namely `time_series`, `start_dt` and
    `end_dt`) in the constructor (a.k.a. `__init__` method) and store the for future use in its `split` and
    `get_n_splits` methods.

    In this way we can restrict the arguments of `split` and `get_n_splits` to the arrays to split (i.e. `X`, `y` and
    `groups`), which are the only arguments required by scikit-learn CV Splitters.

    Arguments:
        frequency: The frequency (or time unit) of the time series. Must be one of "days", "seconds", "microseconds",
            "milliseconds", "minutes", "hours", "weeks". These are the only valid values for the `unit` argument of
            `timedelta` from python `datetime` standard library.
        train_size: Defines the minimum number of time units required to be in the train set.
        forecast_horizon: Specifies the number of time units to forecast.
        time_series: The time series used to create boolean mask for splits. It is not required to be sorted, but it
            must support:

            - comparison operators (with other date-like objects).
            - bitwise operators (with other boolean arrays).
            - `.min()` and `.max()` methods.
            - `.shape` attribute.
        gap: Sets the number of time units to skip between the end of the train set and the start of the forecast set.
        stride: How many time unit to move forward after each split. If `None` (or set to 0), the stride is equal to the
            `forecast_horizon` quantity.
        window: The type of window to use, either "rolling" or "expanding".
        mode: Determines in which orders the splits are generated, either "forward" (start to end) or "backward"
            (end to start).
        start_dt: The start of the time period. If provided, it is used in place of the `time_series.min()`.
        end_dt: The end of the time period. If provided,it is used in place of the `time_series.max()`.

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
        import pandas as pd
        import numpy as np

        from sklearn.linear_model import Ridge
        from sklearn.model_selection import RandomizedSearchCV

        from timebasedcv.sklearn import TimeBasedCVSplitter

        start_dt = pd.Timestamp(2023, 1, 1)
        end_dt = pd.Timestamp(2023, 1, 31)

        time_series = pd.Series(pd.date_range(start_dt, end_dt, freq="D"))
        size = len(time_series)

        df = pd.DataFrame(data=np.random.randn(size, 2), columns=["a", "b"])

        X, y = df[["a", "b"]], df[["a", "b"]].sum(axis=1)

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

    def __init__(  # noqa: PLR0913
        self: Self,
        *,
        frequency: FrequencyUnit,
        train_size: int,
        forecast_horizon: int,
        time_series: Union[SeriesLike[date], SeriesLike[datetime], SeriesLike[pd.Datetime]],
        gap: int = 0,
        stride: Union[int, None] = None,
        window: WindowType = "rolling",
        mode: ModeType = "forward",
        start_dt: NullableDatetime = None,
        end_dt: NullableDatetime = None,
    ) -> None:
        self.splitter = TimeBasedSplit(
            frequency=frequency,
            train_size=train_size,
            forecast_horizon=forecast_horizon,
            gap=gap,
            stride=stride,
            window=window,
            mode=mode,
        )

        self.time_series_ = time_series
        self.start_dt_ = start_dt
        self.end_dt_ = end_dt

        self.n_splits = self._compute_n_splits()
        self.size_ = time_series.shape[0]

    def split(
        self: Self,
        X: Union[NDArray, None] = None,
        y: Union[NDArray, None] = None,
        groups: Union[NDArray, None] = None,
    ) -> Generator[Tuple[NDArray[np.int_], NDArray[np.int_]], None, None]:
        """Generates integer indices corresponding to train and test sets.

        Arguments:
            X: Optional input features array.
            y: Optional target variable array.
            groups: Optional array containing group labels for the samples.

        Returns:
            A generator that yields tuples of train and test indices.

        Raises:
            ValueError: If the input arrays have incompatible lengths with reference `time_series`.

        """
        self._validate_split_args(self.size_, X, y, groups)

        _indexes = np.arange(self.size_)

        yield from self.splitter.split(  # type: ignore[call-overload]
            _indexes,
            time_series=self.time_series_,
            start_dt=self.start_dt_,
            end_dt=self.end_dt_,
            return_splitstate=False,
        )

    def get_n_splits(
        self: Self,
        X: Union[NDArray, None] = None,
        y: Union[NDArray, None] = None,
        groups: Union[NDArray, None] = None,
    ) -> int:
        """Returns the number of splits that can be generated from the instance.

        Arguments:
            X: Unused, exists for compatibility, checked if not None.
            y: Unused, exists for compatibility, checked if not None.
            groups: Unused, exists for compatibility, checked if not None.

        Returns:
            The number of splits that can be generated from `time_series`.
        """
        self._validate_split_args(self.size_, X, y, groups)
        return self.n_splits

    def _compute_n_splits(self: Self) -> int:
        """Computes number of splits just once in the init."""
        time_start = self.start_dt_ or self.time_series_.min()
        time_end = self.end_dt_ or self.time_series_.max()

        return len(tuple(self.splitter._splits_from_period(time_start, time_end)))  # noqa: SLF001

    @staticmethod
    def _validate_split_args(
        size: int,
        X: Union[NDArray, None] = None,
        y: Union[NDArray, None] = None,
        groups: Union[NDArray, None] = None,
    ) -> None:
        """Validates the arguments passed to the `split` and `get_n_splits` methods."""
        if X is not None and X.shape[0] != size:
            msg = f"Invalid shape: {X.shape[0]=} doesn't match time_series.shape[0]={size}"
            raise ValueError(msg)

        if y is not None and y.shape[0] != size:
            msg = f"Invalid shape: {y.shape[0]=} doesn't match time_series.shape[0]={size}"
            raise ValueError(msg)

        if groups is not None and groups.shape[0] != size:
            msg = f"Invalid shape: {groups.shape[0]=} doesn't match time_series.shape[0]={size}"
            raise ValueError(msg)

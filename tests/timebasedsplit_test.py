from contextlib import nullcontext as does_not_raise
from datetime import date, datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV

from timebasedcv import TimeBasedCVSplitter, TimeBasedSplit, _CoreTimeBasedSplit

# Define a fix set of valid arguments
valid_kwargs = {
    "frequency": "days",
    "train_size": 7,
    "forecast_horizon": 1,
    "gap": 0,
    "stride": 1,
    "window": "rolling",
}


start_dt = pd.Timestamp(2023, 1, 1)
end_dt = pd.Timestamp(2023, 1, 31)

time_series = pd.Series(pd.date_range(start_dt, end_dt, freq="D"))
size = len(time_series)

df = pd.DataFrame(data=np.random.randn(size, 2), columns=["a", "b"]).assign(
    date=time_series,
    y=lambda t: t[["a", "b"]].sum(axis=1),
)

X, y = df[["a", "b"]], df["y"]

# Tests for _CoreTimeBasedSplit


@pytest.mark.parametrize(
    "arg_name, arg_value, context",
    [
        ("frequency", "days", does_not_raise()),
        ("frequency", "hours", does_not_raise()),
        ("frequency", "test", pytest.raises(ValueError)),
        ("frequency", 123, pytest.raises(ValueError)),
        ("train_size", 7, does_not_raise()),
        ("train_size", 1.0, pytest.raises(TypeError)),
        ("train_size", "test", pytest.raises(TypeError)),
        ("train_size", -123, pytest.raises(ValueError)),
        ("forecast_horizon", 7, does_not_raise()),
        ("forecast_horizon", 1.0, pytest.raises(TypeError)),
        ("forecast_horizon", "test", pytest.raises(TypeError)),
        ("forecast_horizon", -123, pytest.raises(ValueError)),
        ("gap", 0, does_not_raise()),
        ("gap", 7, does_not_raise()),
        ("gap", 1.0, pytest.raises(TypeError)),
        ("gap", "test", pytest.raises(TypeError)),
        ("gap", -123, pytest.raises(ValueError)),
        ("stride", None, does_not_raise()),
        ("stride", 7, does_not_raise()),
        ("stride", 1.0, pytest.raises(TypeError)),
        ("stride", "test", pytest.raises(TypeError)),
        ("stride", -1, pytest.raises(ValueError)),
        ("window", "rolling", does_not_raise()),
        ("window", "expanding", does_not_raise()),
        ("window", "test", pytest.raises(ValueError)),
        ("window", 123, pytest.raises(ValueError)),
    ],
)
def test_core_init(arg_name, arg_value, context):
    """
    Tests initialization of _CoreTimeBasedSplit with different input values.
    """
    with context:
        _CoreTimeBasedSplit(
            **{
                **valid_kwargs,
                arg_name: arg_value,
            },
        )


def test_core_repr():
    """
    Tests the __repr__ method of _CoreTimeBasedSplit.
    """
    obj = _CoreTimeBasedSplit(**valid_kwargs)

    assert repr(obj) == (
        "_CoreTimeBasedSplit("
        f"\n    frequency_ = {valid_kwargs['frequency']}"
        f"\n    train_size_ = {valid_kwargs['train_size']}"
        f"\n    forecast_horizon_ = {valid_kwargs['forecast_horizon']}"
        f"\n    gap_ = {valid_kwargs['gap']}"
        f"\n    stride_ = {valid_kwargs['stride']}"
        f"\n    window_ = {valid_kwargs['window']}"
        "\n)"
    )


@pytest.mark.parametrize(
    "frequency, train_size, forecast_horizon, gap, stride",
    [("days", 7, 7, 1, None), ("hours", 48, 24, 0, 2), ("weeks", 4, 1, 1, 1)],
)
def test_core_properties(frequency, train_size, forecast_horizon, gap, stride):
    """
    Tests the properties of _CoreTimeBasedSplit.
    """
    cv = _CoreTimeBasedSplit(frequency, train_size, forecast_horizon, gap, stride)

    assert cv.train_delta == timedelta(**{frequency: train_size})
    assert cv.forecast_delta == timedelta(**{frequency: forecast_horizon})
    assert cv.gap_delta == timedelta(**{frequency: gap})
    assert cv.stride_delta == timedelta(**{frequency: stride or forecast_horizon})


@pytest.mark.parametrize(
    "window",
    ["rolling", "expanding"],
)
@pytest.mark.parametrize(
    "time_start, time_end, context",
    [
        (datetime(2023, 1, 1), datetime(2023, 1, 31), does_not_raise()),
        (datetime(2023, 1, 1), datetime(2023, 1, 1), pytest.raises(ValueError)),
        (date(2023, 1, 1), date(2023, 1, 31), does_not_raise()),
        (date(2023, 1, 1), date(2023, 1, 1), pytest.raises(ValueError)),
        (pd.Timestamp(2023, 1, 1), pd.Timestamp(2023, 1, 31), does_not_raise()),
        (pd.Timestamp(2023, 1, 1), pd.Timestamp(2023, 1, 1), pytest.raises(ValueError)),
    ],
)
def test_core_splits_from_period(window, time_start, time_end, context):
    """Tests the _CoreTimeBasedSplit._splits_from_period method."""

    cv = _CoreTimeBasedSplit(
        **{
            **valid_kwargs,
            "window": window,
        },
    )

    with context:
        n_splits = 0
        current_time = time_start
        for split_state in cv._splits_from_period(time_start, time_end):
            train_start = current_time if cv.window_ == "rolling" else time_start
            train_end = current_time + cv.train_delta
            forecast_start = train_end + cv.gap_delta
            forecast_end = forecast_start + cv.forecast_delta

            assert split_state.train_start == train_start
            assert split_state.train_end == train_end
            assert split_state.forecast_start == forecast_start
            assert split_state.forecast_end == forecast_end

            current_time = current_time + cv.stride_delta
            n_splits += 1

        assert n_splits == cv.n_splits_of(start_dt=time_start, end_dt=time_end)


@pytest.mark.parametrize(
    "kwargs, context",
    [
        ({"time_series": time_series}, does_not_raise()),
        ({"start_dt": date(2023, 1, 1), "end_dt": date(2023, 1, 31)}, does_not_raise()),
        ({"time_series": None, "start_dt": date(2023, 1, 1)}, pytest.raises(ValueError)),
        ({"time_series": None, "end_dt": date(2023, 1, 31)}, pytest.raises(ValueError)),
        (
            {"start_dt": date(2023, 1, 31), "end_dt": date(2023, 1, 1)},
            pytest.raises(ValueError),
        ),
    ],
)
def test_core_n_splits_of(kwargs, context):
    """
    Tests the _CoreTimeBasedSplit.n_splits_of method.
    """
    cv = _CoreTimeBasedSplit(**valid_kwargs)

    with context:
        cv.n_splits_of(**kwargs)


def test_core_split():
    """
    Test the _CoreTimeBasedSplit.split method.
    """

    with pytest.raises(NotImplementedError):
        _CoreTimeBasedSplit(**valid_kwargs).split()


# Tests for TimeBasedSplit


@pytest.mark.parametrize(
    "kwargs",
    [
        {"arrays": ()},  # empty arrays
        {"arrays": (X, y[:-1])},  # arrays different shape
        {"time_series": time_series[:-1]},
        # arrays shape different from time_series shape
        {"start_dt": pd.Timestamp(2023, 1, 1), "end_dt": pd.Timestamp(2023, 1, 1)},
        # start_dt >= end_dt
    ],
)
def test_timebasedcv_split_invalid(kwargs):
    """
    Test the TimeBasedSplit.split method with invalid arguments.
    """
    cv = TimeBasedSplit(**valid_kwargs)
    arrays_ = kwargs.get("arrays", (X, y))
    time_series_ = kwargs.get("time_series", time_series)
    start_dt_ = kwargs.get("start_dt")
    end_dt_ = kwargs.get("end_dt")

    with pytest.raises(ValueError):
        next(
            cv.split(*arrays_, time_series=time_series_, start_dt=start_dt_, end_dt=end_dt_),  # type: ignore
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"arrays": (X,)},
        {"arrays": (X, y, X.to_numpy())},  # multi-type arrays
        # arrays shape different from time_series shape
        {"start_dt": pd.Timestamp(2023, 1, 1), "end_dt": pd.Timestamp(2023, 1, 31)},
        {"return_splitstate": True},
    ],
)
def test_timebasedcv_split(kwargs):
    """Tests the TimeBasedSplit.split method."""
    cv = TimeBasedSplit(**valid_kwargs)

    arrays_ = kwargs.get("arrays", (X, y))
    time_series_ = kwargs.get("time_series", time_series)
    start_dt_ = kwargs.get("start_dt")
    end_dt_ = kwargs.get("end_dt")
    return_splitstate_ = kwargs.get("return_splitstate", False)

    n_arrays = len(arrays_)
    split_results = next(
        cv.split(
            *arrays_,
            time_series=time_series_,
            start_dt=start_dt_,
            end_dt=end_dt_,
            return_splitstate=return_splitstate_,
        ),  # type: ignore
    )

    if return_splitstate_:
        train_forecast, _ = split_results
    else:
        train_forecast = split_results

    assert len(train_forecast) == n_arrays * 2


# Tests for TimeBasedCVSplitter


def test_cv_splitter():
    """
    Tests the TimeBasedCVSplitter `__init__` and `split` methods as well as its
    compatibility with sklearn's _CV Splitter_s.
    """
    cv = TimeBasedCVSplitter(
        time_series=time_series,  # type: ignore
        start_dt=start_dt,
        end_dt=end_dt,
        **valid_kwargs,
    )

    assert cv.time_series_ is not None
    assert cv.start_dt_ is not None
    assert cv.end_dt_ is not None
    assert isinstance(cv.n_splits, int)
    assert isinstance(cv.size_, int)

    param_grid = {
        "alpha": np.linspace(0.1, 2, 10),
        "fit_intercept": [True, False],
        "positive": [True, False],
    }

    random_search_cv = RandomizedSearchCV(
        estimator=Ridge(),
        param_distributions=param_grid,
        cv=cv,  # type: ignore
        n_jobs=-1,
    ).fit(X, y)

    assert random_search_cv.best_estimator_ is not None


@pytest.mark.parametrize("size", [size])
@pytest.mark.parametrize(
    "X_shape, y_shape, g_shape, context",
    [
        ((size, 2), (size,), (size, 2), does_not_raise()),
        ((size + 1, 2), (size,), (size, 2), pytest.raises(ValueError)),
        ((size, 2), (size + 1,), (size, 2), pytest.raises(ValueError)),
        ((size, 2), (size,), (size + 1, 2), pytest.raises(ValueError)),
    ],
)
def test_cv_splitter_validate_split(
    size: int,
    X_shape: Tuple[int, int],
    y_shape: Tuple[int],
    g_shape: Tuple[int, int],
    context,
):
    """Test the TimeBasedCVSplitter._validate_split_args static method."""

    with context:
        TimeBasedCVSplitter._validate_split_args(
            size=size,
            X=np.random.randn(*X_shape),
            y=np.random.randn(*y_shape),
            groups=np.random.randn(*g_shape),
        )

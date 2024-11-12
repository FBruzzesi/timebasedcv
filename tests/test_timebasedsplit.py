from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from datetime import date
from datetime import datetime

import narwhals as nw
import numpy as np
import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta

from timebasedcv import TimeBasedSplit
from timebasedcv.core import _CoreTimeBasedSplit

RNG = np.random.default_rng()

# Define a fix set of valid arguments
start_dt = pd.Timestamp(2023, 1, 1)
end_dt = pd.Timestamp(2023, 1, 31)

time_series = pd.Series(pd.date_range(start_dt, end_dt, freq="D"))
size = len(time_series)

df = pd.DataFrame(data=RNG.normal(size=(size, 2)), columns=["a", "b"]).assign(
    date=time_series,
    y=lambda t: t[["a", "b"]].sum(axis=1),
)

X, y = df[["a", "b"]], df["y"]

err_msg_freq = (
    "`frequency` must be one of "
    r"\('days', 'seconds', 'microseconds', 'milliseconds', 'minutes', 'hours', 'weeks', 'months', 'years'\)"
)
err_msg_int = r"\(`train_size_`, `forecast_horizon_`, `gap_`, `stride_`\) arguments must be of type `int`."
err_msg_lower_bound = r"must be greater or equal than \(1, 1, 0, 1\)"
err_msg_window = r"`window` must be one of \('rolling', 'expanding'\)"
err_msg_mode = r"`mode` must be one of \('forward', 'backward'\)"
err_msg_time_order = r"(`time_start` must be before `time_end`)|(`start_dt` must be before `end_dt`)."
err_msg_no_time = r"Either `time_series` or \(`start_dt`, `end_dt`\) pair must be provided."
err_msg_shape = "Invalid shape: "


# Tests for _CoreTimeBasedSplit


@pytest.mark.parametrize(
    "arg_name, arg_value, context",
    [
        ("frequency", "days", does_not_raise()),
        ("frequency", "hours", does_not_raise()),
        ("frequency", "test", pytest.raises(ValueError, match=err_msg_freq)),
        ("frequency", 123, pytest.raises(ValueError, match=err_msg_freq)),
        ("train_size", 7, does_not_raise()),
        ("train_size", 1.0, pytest.raises(TypeError, match=err_msg_int)),
        ("train_size", "test", pytest.raises(TypeError, match=err_msg_int)),
        ("train_size", -123, pytest.raises(ValueError, match=err_msg_lower_bound)),
        ("forecast_horizon", 7, does_not_raise()),
        ("forecast_horizon", 1.0, pytest.raises(TypeError, match=err_msg_int)),
        ("forecast_horizon", "test", pytest.raises(TypeError, match=err_msg_int)),
        ("forecast_horizon", -123, pytest.raises(ValueError, match=err_msg_lower_bound)),
        ("gap", 0, does_not_raise()),
        ("gap", 7, does_not_raise()),
        ("gap", 1.0, pytest.raises(TypeError, match=err_msg_int)),
        ("gap", "test", pytest.raises(TypeError, match=err_msg_int)),
        ("gap", -123, pytest.raises(ValueError, match=err_msg_lower_bound)),
        ("stride", None, does_not_raise()),
        ("stride", 7, does_not_raise()),
        ("stride", 1.0, pytest.raises(TypeError, match=err_msg_int)),
        ("stride", "test", pytest.raises(TypeError, match=err_msg_int)),
        ("stride", -1, pytest.raises(ValueError, match=err_msg_lower_bound)),
        ("window", "rolling", does_not_raise()),
        ("window", "expanding", does_not_raise()),
        ("window", "test", pytest.raises(ValueError, match=err_msg_window)),
        ("window", 123, pytest.raises(ValueError, match=err_msg_window)),
        ("mode", "forward", does_not_raise()),
        ("mode", "backward", does_not_raise()),
        ("mode", "test", pytest.raises(ValueError, match=err_msg_mode)),
        ("mode", 123, pytest.raises(ValueError, match=err_msg_mode)),
    ],
)
def test_core_init(base_kwargs, arg_name, arg_value, context):
    """Tests initialization of _CoreTimeBasedSplit with different input values."""
    with context:
        kwargs = {**base_kwargs, arg_name: arg_value}
        obj = _CoreTimeBasedSplit(**kwargs)

        assert repr(obj) == (
            "_CoreTimeBasedSplit("
            f"\n    frequency_ = {kwargs['frequency']}"
            f"\n    train_size_ = {kwargs['train_size']}"
            f"\n    forecast_horizon_ = {kwargs['forecast_horizon']}"
            f"\n    gap_ = {kwargs['gap']}"
            f"\n    stride_ = {kwargs['stride'] or kwargs['forecast_horizon']}"
            f"\n    window_ = {kwargs['window']}"
            "\n)"
        )


@pytest.mark.parametrize(
    "frequency, train_size, forecast_horizon, gap, stride",
    [("days", 7, 7, 1, None), ("hours", 48, 24, 0, 2), ("weeks", 4, 1, 1, 1)],
)
def test_core_properties(frequency, train_size, forecast_horizon, gap, stride):
    """Tests the properties of _CoreTimeBasedSplit."""
    cv = _CoreTimeBasedSplit(
        frequency=frequency,
        train_size=train_size,
        forecast_horizon=forecast_horizon,
        gap=gap,
        stride=stride,
    )

    assert cv.train_delta == relativedelta(**{frequency: train_size})
    assert cv.forecast_delta == relativedelta(**{frequency: forecast_horizon})
    assert cv.gap_delta == relativedelta(**{frequency: gap})
    assert cv.stride_delta == relativedelta(**{frequency: stride or forecast_horizon})


@pytest.mark.parametrize(
    "time_start, time_end, context",
    [
        (datetime(2023, 1, 1), datetime(2023, 1, 31), does_not_raise()),
        (datetime(2023, 1, 1), datetime(2022, 1, 1), pytest.raises(ValueError, match=err_msg_time_order)),
        (date(2023, 1, 1), date(2023, 1, 31), does_not_raise()),
        (date(2023, 1, 1), date(2022, 1, 1), pytest.raises(ValueError, match=err_msg_time_order)),
        (pd.Timestamp(2023, 1, 1), pd.Timestamp(2023, 1, 31), does_not_raise()),
        (pd.Timestamp(2023, 1, 1), pd.Timestamp(2022, 1, 1), pytest.raises(ValueError, match=err_msg_time_order)),
    ],
)
def test_core_splits_from_period(valid_kwargs, time_start, time_end, context):
    """Tests the _CoreTimeBasedSplit._splits_from_period method."""

    cv = _CoreTimeBasedSplit(**valid_kwargs)

    with context:
        n_splits = sum(1 for _ in cv._splits_from_period(time_start, time_end))  # noqa: SLF001
        assert n_splits == cv.n_splits_of(start_dt=time_start, end_dt=time_end)
        assert n_splits == cv.n_splits_of(time_series=pd.Series(pd.date_range(time_start, time_end, freq="D")))


def test_core_splits_from_period_invalid(base_kwargs):
    """Tests the _CoreTimeBasedSplit._splits_from_period method invalid args."""

    msg = r"Either `time_series` or \(`start_dt`, `end_dt`\) pair must be provided."
    with pytest.raises(ValueError, match=msg):  # noqa: PT012
        cv = _CoreTimeBasedSplit(**base_kwargs)
        cv.n_splits_of(start_dt=start_dt)


# Tests for TimeBasedSplit


@pytest.mark.parametrize(
    "kwargs",
    [
        {"arrays": ()},  # empty arrays
        {"arrays": (X, y[:-1])},  # arrays different shape
        {"time_series": time_series[:-1]},
        # arrays shape different from time_series shape
        {"start_dt": pd.Timestamp(2023, 1, 1), "end_dt": pd.Timestamp(2023, 1, 1)},
    ],
)
def test_timebasedcv_split_invalid(valid_kwargs, kwargs):
    """Test the TimeBasedSplit.split method with invalid arguments."""
    cv = TimeBasedSplit(**valid_kwargs)
    arrays_ = kwargs.get("arrays", (X, y))
    time_series_ = kwargs.get("time_series", time_series)
    start_dt_ = kwargs.get("start_dt")
    end_dt_ = kwargs.get("end_dt")

    with pytest.raises(ValueError):  # noqa: PT011
        next(cv.split(*arrays_, time_series=time_series_, start_dt=start_dt_, end_dt=end_dt_))


@pytest.mark.parametrize("return_splitstate", [True, False])
def test_timebasedcv_split_dataframes(valid_kwargs, frame_constructor, generate_test_data, return_splitstate):
    """Tests the TimeBasedSplit.split method on different dataframe constructors."""
    cv = TimeBasedSplit(**valid_kwargs)

    start_dt, end_dt, time_series, X, y = generate_test_data

    data = {
        "x0": X[:, 0],
        "x1": X[:, 1],
        "y": y,
        "ts": time_series,
    }

    df = nw.from_native(frame_constructor(data), eager_only=True)

    arrays_ = (df.select("x0", "x1").to_native(), df["y"].to_native())
    time_series_ = df["ts"].to_native()

    n_arrays = len(arrays_)
    split_results = next(
        cv.split(
            *arrays_,
            time_series=time_series_,
            start_dt=start_dt,
            end_dt=end_dt,
            return_splitstate=return_splitstate,
        ),
    )

    if return_splitstate:
        train_forecast, _ = split_results
    else:
        train_forecast = split_results

    assert len(train_forecast) == n_arrays * 2


@pytest.mark.parametrize("return_splitstate", [True, False])
def test_timebasedcv_split_arrays(valid_kwargs, array_constructor, generate_test_data, return_splitstate):
    """Tests the TimeBasedSplit.split method on different dataframe constructors."""
    cv = TimeBasedSplit(**valid_kwargs)

    start_dt, end_dt, time_series, X, y = generate_test_data

    arrays_ = (array_constructor(X), array_constructor(y))
    time_series_ = array_constructor(time_series)

    n_arrays = len(arrays_)
    split_results = next(
        cv.split(
            *arrays_,
            time_series=time_series_,
            start_dt=start_dt,
            end_dt=end_dt,
            return_splitstate=return_splitstate,
        ),
    )

    if return_splitstate:
        train_forecast, _ = split_results
    else:
        train_forecast = split_results

    assert len(train_forecast) == n_arrays * 2

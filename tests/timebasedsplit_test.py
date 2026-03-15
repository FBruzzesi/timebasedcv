from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from datetime import date, datetime

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
from dateutil.relativedelta import relativedelta

from timebasedcv import TimeBasedSplit
from timebasedcv.core import _CoreTimeBasedSplit

RNG = np.random.default_rng(seed=42)

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


# Golden / boundary-value tests


def test_split_boundaries_forward_rolling():
    """Asserts exact SplitState boundaries for forward rolling splits with known parameters."""
    cv = TimeBasedSplit(
        frequency="days", train_size=3, forecast_horizon=2, gap=1, stride=2, window="rolling", mode="forward"
    )
    splits = list(cv._splits_from_period(date(2023, 1, 1), date(2023, 1, 15)))  # noqa: SLF001

    assert len(splits) == 6

    expected = [
        (date(2023, 1, 1), date(2023, 1, 4), date(2023, 1, 5), date(2023, 1, 7)),
        (date(2023, 1, 3), date(2023, 1, 6), date(2023, 1, 7), date(2023, 1, 9)),
        (date(2023, 1, 5), date(2023, 1, 8), date(2023, 1, 9), date(2023, 1, 11)),
        (date(2023, 1, 7), date(2023, 1, 10), date(2023, 1, 11), date(2023, 1, 13)),
        (date(2023, 1, 9), date(2023, 1, 12), date(2023, 1, 13), date(2023, 1, 15)),
        (date(2023, 1, 11), date(2023, 1, 14), date(2023, 1, 15), date(2023, 1, 17)),
    ]

    it = zip(splits, expected, strict=True)
    for split, (train_start, train_end, forecast_start, forecast_end) in it:
        assert split.train_start == train_start
        assert split.train_end == train_end
        assert split.forecast_start == forecast_start
        assert split.forecast_end == forecast_end


def test_split_boundaries_backward_rolling():
    """Asserts exact SplitState boundaries for backward rolling splits with known parameters."""
    cv = TimeBasedSplit(
        frequency="days", train_size=3, forecast_horizon=2, gap=1, stride=2, window="rolling", mode="backward"
    )
    splits = list(cv._splits_from_period(date(2023, 1, 1), date(2023, 1, 15)))  # noqa: SLF001

    assert len(splits) == 5

    expected = [
        (date(2023, 1, 9), date(2023, 1, 12), date(2023, 1, 13), date(2023, 1, 15)),
        (date(2023, 1, 7), date(2023, 1, 10), date(2023, 1, 11), date(2023, 1, 13)),
        (date(2023, 1, 5), date(2023, 1, 8), date(2023, 1, 9), date(2023, 1, 11)),
        (date(2023, 1, 3), date(2023, 1, 6), date(2023, 1, 7), date(2023, 1, 9)),
        (date(2023, 1, 1), date(2023, 1, 4), date(2023, 1, 5), date(2023, 1, 7)),
    ]
    it = zip(splits, expected, strict=True)
    for split, (train_start, train_end, forecast_start, forecast_end) in it:
        assert split.train_start == train_start
        assert split.train_end == train_end
        assert split.forecast_start == forecast_start
        assert split.forecast_end == forecast_end


def test_split_boundaries_forward_expanding():
    """Asserts exact SplitState boundaries for forward expanding splits with known parameters."""
    cv = TimeBasedSplit(
        frequency="days", train_size=3, forecast_horizon=2, gap=1, stride=2, window="expanding", mode="forward"
    )
    splits = list(cv._splits_from_period(date(2023, 1, 1), date(2023, 1, 15)))  # noqa: SLF001

    assert len(splits) == 6

    for split in splits:
        assert split.train_start == date(2023, 1, 1), "Expanding window always starts from the beginning"

    expected_train_ends = [
        date(2023, 1, 4),
        date(2023, 1, 6),
        date(2023, 1, 8),
        date(2023, 1, 10),
        date(2023, 1, 12),
        date(2023, 1, 14),
    ]
    for split, expected_te in zip(splits, expected_train_ends, strict=True):
        assert split.train_end == expected_te


def test_split_content_shapes_and_ordering():
    """Verifies that split arrays have correct shapes and temporal ordering."""
    cv = TimeBasedSplit(
        frequency="days", train_size=5, forecast_horizon=3, gap=0, stride=3, window="rolling", mode="forward"
    )

    ts = pd.Series(pd.date_range("2023-01-01", "2023-01-20", freq="D"))
    rng = np.random.default_rng(seed=123)
    _y = pd.Series(rng.normal(size=len(ts)))

    all_splits = list(cv.split(_y, time_series=ts, return_splitstate=True))
    assert len(all_splits) > 0

    for (train, forecast), split_state in all_splits:
        assert len(train) > 0, "Train set must be non-empty"
        assert len(forecast) > 0 or split_state.forecast_end > ts.max(), "Forecast should be non-empty within range"

        assert split_state.train_start <= split_state.train_end
        assert split_state.train_end <= split_state.forecast_start
        assert split_state.forecast_start <= split_state.forecast_end


def test_split_rolling_constant_train_size():
    """Verifies that rolling window produces constant-size train periods."""
    cv = TimeBasedSplit(
        frequency="days", train_size=5, forecast_horizon=3, gap=0, stride=3, window="rolling", mode="forward"
    )

    ts = pd.Series(pd.date_range("2023-01-01", "2023-01-20", freq="D"))
    rng = np.random.default_rng(seed=123)
    _y = pd.Series(rng.normal(size=len(ts)))

    train_periods = []
    for _, split_state in cv.split(_y, time_series=ts, return_splitstate=True):
        train_periods.append(split_state.train_end - split_state.train_start)

    assert len(set(train_periods)) == 1, "Rolling window should have constant train period length"


def test_split_expanding_nondecreasing_train_size():
    """Verifies that expanding window produces non-decreasing train periods."""
    cv = TimeBasedSplit(
        frequency="days", train_size=5, forecast_horizon=3, gap=0, stride=3, window="expanding", mode="forward"
    )

    ts = pd.Series(pd.date_range("2023-01-01", "2023-01-20", freq="D"))
    rng = np.random.default_rng(seed=123)
    _y = pd.Series(rng.normal(size=len(ts)))

    prev_length = None
    for _, split_state in cv.split(_y, time_series=ts, return_splitstate=True):
        current_length = split_state.train_end - split_state.train_start
        if prev_length is not None:
            assert current_length >= prev_length, "Expanding window train period should be non-decreasing"
        prev_length = current_length


# Smoke tests for alias classes


def test_expanding_time_split_smoke():
    """Smoke test for ExpandingTimeSplit convenience class."""
    from timebasedcv import ExpandingTimeSplit

    cv = ExpandingTimeSplit(frequency="days", train_size=5, forecast_horizon=3, gap=0, mode="forward")
    assert cv.window_ == "expanding"

    splits = list(cv._splits_from_period(date(2023, 1, 1), date(2023, 1, 15)))  # noqa: SLF001
    assert len(splits) > 0

    for split in splits:
        assert split.train_start == date(2023, 1, 1)


def test_rolling_time_split_smoke():
    """Smoke test for RollingTimeSplit convenience class."""
    from timebasedcv import RollingTimeSplit

    cv = RollingTimeSplit(frequency="days", train_size=5, forecast_horizon=3, gap=0, mode="forward")
    assert cv.window_ == "rolling"

    splits = list(cv._splits_from_period(date(2023, 1, 1), date(2023, 1, 15)))  # noqa: SLF001
    assert len(splits) > 0

    train_deltas = {split.train_end - split.train_start for split in splits}
    assert len(train_deltas) == 1


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


# Tests for non-pandas backends: Polars and PyArrow


_BASE_CV_KWARGS = {"frequency": "days", "train_size": 5, "forecast_horizon": 3, "gap": 0, "stride": 3}


def _make_test_data():
    """Shared helper for multi-backend tests."""
    rng = np.random.default_rng(seed=99)
    dates = pd.date_range("2023-01-01", "2023-01-20", freq="D")
    n = len(dates)
    return {
        "time": dates,
        "x0": rng.normal(size=n),
        "x1": rng.normal(size=n),
        "y": rng.normal(size=n),
    }


def test_split_polars_preserves_types():
    """Verifies that splitting Polars DataFrames/Series returns Polars types."""
    data = _make_test_data()
    df_pl = pl.DataFrame({"x0": data["x0"], "x1": data["x1"]})
    y_pl = pl.Series("y", data["y"])
    ts_pl = pl.Series("time", data["time"])

    cv = TimeBasedSplit(**_BASE_CV_KWARGS)

    for X_train, X_forecast, y_train, y_forecast in cv.split(df_pl, y_pl, time_series=ts_pl):
        assert isinstance(X_train, pl.DataFrame), f"Expected Polars DataFrame, got {type(X_train)}"
        assert isinstance(X_forecast, pl.DataFrame), f"Expected Polars DataFrame, got {type(X_forecast)}"
        assert isinstance(y_train, pl.Series), f"Expected Polars Series, got {type(y_train)}"
        assert isinstance(y_forecast, pl.Series), f"Expected Polars Series, got {type(y_forecast)}"

        assert X_train.shape[1] == 2
        assert X_forecast.shape[1] == 2
        assert len(y_train) == X_train.shape[0]
        assert len(y_forecast) == X_forecast.shape[0]
        assert X_train.shape[0] > 0
        break


def test_split_pyarrow_preserves_types():
    """Verifies that splitting PyArrow tables returns PyArrow types."""
    data = _make_test_data()
    table_pa = pa.table({"x0": data["x0"], "x1": data["x1"]})
    ts_pa = pa.table({"time": data["time"]})["time"]

    cv = TimeBasedSplit(**_BASE_CV_KWARGS)

    start_dt = data["time"].min()
    end_dt = data["time"].max()
    for X_train, X_forecast in cv.split(table_pa, time_series=ts_pa, start_dt=start_dt, end_dt=end_dt):
        assert isinstance(X_train, pa.Table), f"Expected PyArrow Table, got {type(X_train)}"
        assert isinstance(X_forecast, pa.Table), f"Expected PyArrow Table, got {type(X_forecast)}"

        assert X_train.num_columns == 2
        assert X_forecast.num_columns == 2
        assert X_train.num_rows > 0
        break


def test_split_polars_series_only():
    """Verifies that splitting a single Polars Series works correctly."""
    data = _make_test_data()
    y_pl = pl.Series("y", data["y"])
    ts_pl = pl.Series("time", data["time"])

    cv = TimeBasedSplit(**_BASE_CV_KWARGS)

    for train, forecast in cv.split(y_pl, time_series=ts_pl):
        assert isinstance(train, pl.Series), f"Expected Polars Series, got {type(train)}"
        assert isinstance(forecast, pl.Series), f"Expected Polars Series, got {type(forecast)}"
        assert len(train) > 0
        break


def test_split_cross_backend_consistency():
    """Verifies that splits from pandas, Polars, and PyArrow produce the same number of rows per fold."""
    data = _make_test_data()
    cv = TimeBasedSplit(**_BASE_CV_KWARGS)

    start_dt = data["time"].min()
    end_dt = data["time"].max()

    ts_pd = pd.Series(data["time"])
    df_pd = pd.DataFrame({"x0": data["x0"], "x1": data["x1"]})
    y_pd = pd.Series(data["y"])

    df_pl = pl.DataFrame({"x0": data["x0"], "x1": data["x1"]})
    y_pl = pl.Series("y", data["y"])
    ts_pl = pl.Series("time", data["time"])

    table_pa = pa.table({"x0": data["x0"], "x1": data["x1"]})
    ts_pa = pa.table({"time": data["time"]})["time"]

    pd_shapes = [(X_tr.shape[0], X_fc.shape[0]) for X_tr, X_fc, _, _ in cv.split(df_pd, y_pd, time_series=ts_pd)]

    pl_shapes = [(X_tr.shape[0], X_fc.shape[0]) for X_tr, X_fc, _, _ in cv.split(df_pl, y_pl, time_series=ts_pl)]

    pa_shapes = [
        (X_tr.num_rows, X_fc.num_rows)
        for X_tr, X_fc in cv.split(table_pa, time_series=ts_pa, start_dt=start_dt, end_dt=end_dt)
    ]

    assert pd_shapes == pl_shapes, f"Pandas vs Polars shape mismatch: {pd_shapes} != {pl_shapes}"
    assert pd_shapes == pa_shapes, f"Pandas vs PyArrow shape mismatch: {pd_shapes} != {pa_shapes}"
    assert len(pd_shapes) > 0


def test_split_polars_with_return_splitstate():
    """Verifies that return_splitstate works correctly with Polars."""
    data = _make_test_data()
    y_pl = pl.Series("y", data["y"])
    ts_pl = pl.Series("time", data["time"])

    cv = TimeBasedSplit(**_BASE_CV_KWARGS)
    all_splits = list(cv.split(y_pl, time_series=ts_pl, return_splitstate=True))

    assert len(all_splits) > 0
    for (train, forecast), split_state in all_splits:
        assert isinstance(train, pl.Series)
        assert isinstance(forecast, pl.Series)
        assert split_state.train_start <= split_state.train_end
        assert split_state.forecast_start <= split_state.forecast_end

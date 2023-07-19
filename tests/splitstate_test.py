from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from timebasedcv.splitstate import SplitState


@pytest.mark.parametrize(
    "train_start, train_end, forecast_start, forecast_end",
    [
        (
            datetime(2023, 1, 1, 0),
            datetime(2023, 1, 31, 0),
            datetime(2023, 2, 1, 0),
            datetime(2023, 2, 28, 0),
        ),
        (
            date(2023, 1, 1),
            date(2023, 1, 31),
            date(2023, 2, 1),
            date(2023, 2, 28),
        ),
        (
            pd.Timestamp(2023, 1, 1),
            pd.Timestamp(2023, 1, 31),
            pd.Timestamp(2023, 2, 1),
            pd.Timestamp(2023, 2, 28),
        ),
        (
            pd.Timestamp(2023, 1, 1, 0),
            pd.Timestamp(2023, 1, 31, 0),
            pd.Timestamp(2023, 2, 1, 0),
            pd.Timestamp(2023, 2, 28, 0),
        ),
    ],
)
@pytest.mark.parametrize(
    "expected_train_len, expected_forecast_len, expected_gap_len, expected_total_len",
    [(timedelta(days=30), timedelta(days=27), timedelta(days=1), timedelta(days=58))],
)
def test_splitstate_valid(
    train_start,
    train_end,
    forecast_start,
    forecast_end,
    expected_train_len,
    expected_forecast_len,
    expected_gap_len,
    expected_total_len,
):
    """Test the SplitState class with different input values types."""
    split_state = SplitState(
        train_start=train_start,
        train_end=train_end,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
    )

    assert split_state.train_length == expected_train_len
    assert split_state.forecast_length == expected_forecast_len
    assert split_state.gap_length == expected_gap_len
    assert split_state.total_length == expected_total_len


@pytest.mark.parametrize(
    "train_start, exception_context",
    [
        (date(2023, 1, 1), pytest.raises(TypeError)),
        (pd.Timestamp(2023, 1, 1), pytest.raises(TypeError)),
        ("2023-01-01", pytest.raises(TypeError)),
        (datetime(2023, 2, 1), pytest.raises(ValueError)),
    ],
)
def test_splitstate_invalid(
    train_start,
    exception_context,
):
    """Test the SplitState class with mixed input values types or unordered datetypes."""

    with exception_context:
        SplitState(
            train_start=train_start,
            train_end=datetime(2023, 1, 31),
            forecast_start=datetime(2023, 2, 1),
            forecast_end=datetime(2023, 2, 28),
        )

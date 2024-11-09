from __future__ import annotations

from datetime import date
from datetime import datetime

import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta

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
    [(relativedelta(days=30), relativedelta(days=27), relativedelta(days=1), relativedelta(months=1, days=27))],
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
    "train_start, context",
    [
        (date(2023, 1, 1), pytest.raises(TypeError, match="All attributes must be of type")),
        (pd.Timestamp(2023, 1, 1), pytest.raises(TypeError, match="All attributes must be of type")),
        ("2023-01-01", pytest.raises(TypeError, match="All attributes must be of type")),
        (
            datetime(2023, 2, 1),
            pytest.raises(
                ValueError,
                match="`train_start`, `train_end`, `forecast_start`, `forecast_end` must be ordered",
            ),
        ),
    ],
)
def test_splitstate_invalid(
    train_start,
    context,
):
    """Test the SplitState class with mixed input values types or unordered datetypes."""

    with context:
        SplitState(
            train_start=train_start,
            train_end=datetime(2023, 1, 31),
            forecast_start=datetime(2023, 2, 1),
            forecast_end=datetime(2023, 2, 28),
        )


def test_splitstate_add():
    """Tests SplitState.__add__."""
    split_state = SplitState(
        train_start=datetime(2023, 1, 1, 0),
        train_end=datetime(2023, 1, 31, 0),
        forecast_start=datetime(2023, 2, 1, 0),
        forecast_end=datetime(2023, 2, 28, 0),
    )

    delta = relativedelta(days=1)
    expected_split_state = SplitState(
        train_start=datetime(2023, 1, 2, 0),
        train_end=datetime(2023, 2, 1, 0),
        forecast_start=datetime(2023, 2, 2, 0),
        forecast_end=datetime(2023, 3, 1, 0),
    )

    result = split_state + delta

    assert result.train_start == expected_split_state.train_start
    assert result.train_end == expected_split_state.train_end
    assert result.forecast_start == expected_split_state.forecast_start
    assert result.forecast_end == expected_split_state.forecast_end


def test_splitstate_sub():
    """Tests SplitState.__sub__."""
    split_state = SplitState(
        train_start=datetime(2023, 1, 2, 0),
        train_end=datetime(2023, 2, 1, 0),
        forecast_start=datetime(2023, 2, 2, 0),
        forecast_end=datetime(2023, 3, 1, 0),
    )

    delta = relativedelta(days=1)
    expected_split_state = SplitState(
        train_start=datetime(2023, 1, 1, 0),
        train_end=datetime(2023, 1, 31, 0),
        forecast_start=datetime(2023, 2, 1, 0),
        forecast_end=datetime(2023, 2, 28, 0),
    )

    result = split_state - delta

    assert result.train_start == expected_split_state.train_start
    assert result.train_end == expected_split_state.train_end
    assert result.forecast_start == expected_split_state.forecast_start
    assert result.forecast_end == expected_split_state.forecast_end

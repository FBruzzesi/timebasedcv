from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Literal
from typing import Tuple
from typing import Union

import dask.array as da
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame

    from timebasedcv.utils._types import TensorLike


@pytest.fixture()
def sample_list() -> List[int]:
    """Returns a sample list."""
    return [1, 2, 3, 4, 5]


@pytest.fixture()
def sample_pairs() -> List[Tuple[int, int]]:
    """Returns a sample list of pairs."""
    return [(1, 2), (2, 3), (3, 4), (4, 5)]


@pytest.fixture()
def base_kwargs() -> Dict[str, Any]:
    """Base set of values in the configuration"""
    return {
        "frequency": "days",
        "train_size": 10,
        "forecast_horizon": 5,
        "gap": 2,
        "stride": 1,
        "window": "rolling",
        "mode": "forward",
    }


@pytest.fixture(params=[3, 6])
def train_size(request) -> int:
    """Fixture for setting the train_size parameter."""
    return request.param


@pytest.fixture(params=[3, 4])
def forecast_horizon(request) -> int:
    """Fixture for setting the forecast_horizon parameter."""
    return request.param


@pytest.fixture(params=[1, 2])
def gap(request) -> int:
    """Fixture for setting the gap parameter."""
    return request.param


@pytest.fixture(params=[None, 3])
def stride(request) -> Union[int, None]:
    """Fixture for setting the stride parameter."""
    return request.param


@pytest.fixture(params=["rolling", "expanding"])
def window(request) -> Literal["rolling", "expanding"]:
    """Fixture that provides a parameterized window type for testing."""
    return request.param


@pytest.fixture(params=["forward", "backward"])
def mode(request) -> Literal["forward", "backward"]:
    """Fixture for setting the mode parameter."""
    return request.param


@pytest.fixture()
def valid_kwargs(
    train_size: int,
    forecast_horizon: int,
    gap: int,
    stride: Union[int, None],
    window: Literal["rolling", "expanding"],
    mode: Literal["forward", "backward"],
) -> Dict[str, Any]:
    """Valid combination of settings"""
    return {
        "frequency": "days",
        "train_size": train_size,
        "forecast_horizon": forecast_horizon,
        "gap": gap,
        "stride": stride,
        "window": window,
        "mode": mode,
    }


@pytest.fixture()
def generate_test_data() -> Tuple[datetime, datetime, np.ndarray, np.ndarray, np.ndarray]:
    """Generate start and end time, time series, X, and y for testing purposes.

    Returns:
        tuple: A tuple containing the start datetime, end datetime, time series, X and y.
    """
    RNG = np.random.default_rng()

    start_dt, end_dt = datetime(2023, 1, 1), datetime(2023, 1, 31)
    time_series = np.arange(start_dt, end_dt, timedelta(days=1))
    size = len(time_series)

    X = RNG.normal(size=(size, 2))
    y = X.sum(axis=1) + RNG.normal(size=size) / 100
    return start_dt, end_dt, time_series, X, y


@pytest.fixture(params=[pd.DataFrame, pl.DataFrame, pa.table])
def frame_constructor(request) -> Callable[[Dict[str, Any]], IntoDataFrame]:
    """Fixture to return a eager dataframe constructor."""
    return request.param


@pytest.fixture(params=[np.asarray, da.from_array])
def array_constructor(request) -> Callable[[np.ndarray], TensorLike]:
    """Fixture to return an array constructor."""
    return request.param

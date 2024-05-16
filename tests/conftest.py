from typing import List, Tuple

import pytest


@pytest.fixture()
def sample_list() -> List[int]:
    """Returns a sample list."""
    return [1, 2, 3, 4, 5]


@pytest.fixture()
def sample_pairs() -> List[Tuple[int, int]]:
    """Returns a sample list of pairs."""
    return [(1, 2), (2, 3), (3, 4), (4, 5)]


@pytest.fixture(params=[3, 6])
def train_size(request):
    """Fixture for setting the train_size parameter."""
    return request.param


@pytest.fixture(params=[2, 3])
def forecast_horizon(request):
    """Fixture for setting the forecast_horizon parameter."""
    return request.param


@pytest.fixture(params=[1, 2])
def gap(request):
    """Fixture for setting the gap parameter."""
    return request.param


@pytest.fixture(params=[None, 3])
def stride(request):
    """Fixture for setting the stride parameter."""
    return request.param


@pytest.fixture(params=["rolling", "expanding"])
def window(request):
    """Fixture that provides a parameterized window type for testing."""
    return request.param


@pytest.fixture(params=["forward", "backward"])
def mode(request):
    """Fixture for setting the mode parameter."""
    return request.param


@pytest.fixture()
def valid_kwargs(train_size, forecast_horizon, gap, stride, window, mode):
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
def base_kwargs():
    return {
        "frequency": "days",
        "train_size": 10,
        "forecast_horizon": 5,
        "gap": 2,
        "stride": 1,
        "window": "rolling",
        "mode": "forward",
    }

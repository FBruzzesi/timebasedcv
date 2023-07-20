from contextlib import nullcontext as does_not_raise

import pytest

from timebasedcv import _CoreTimeBasedSplit

# Define a fix set of valid arguments
valid_kwargs = {
    "frequency": "days",
    "train_size": 1,
    "forecast_horizon": 1,
    "gap": 0,
    "stride": 1,
    "window": "rolling",
}


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

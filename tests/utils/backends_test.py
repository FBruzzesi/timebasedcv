from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest

from timebasedcv.utils._backends import (
    BACKEND_TO_INDEXING_METHOD,
    default_indexing_method,
)

size = 10
arr = np.arange(size)
valid_mask = arr % 2 == 0
expected = np.array([v for v in range(size) if v % 2 == 0])

invalid_mask = np.random.randint(0, 1, size=size + 1).astype(bool)


@pytest.mark.parametrize(
    "arr, mask, expected, context",
    [
        (arr, valid_mask, expected, does_not_raise()),
        (arr, invalid_mask, expected, pytest.raises(ValueError)),
    ],
)
def test_default_indexing_method(arr, mask, expected, context):
    """
    Tests the default indexing method with a numpy array.
    """
    with context:
        result = default_indexing_method(arr, mask)
        assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "arr, mask, expected",
    [
        (arr, valid_mask, expected),  # numpy array
        (pd.Series(data=arr), valid_mask, expected),  # pandas series
        (
            pd.DataFrame(data={"a": arr, "b": arr}),
            pd.Series(valid_mask),
            pd.DataFrame(data={"a": expected, "b": expected}),
        ),  # pandas dataframe
    ],
)
def test_backend_to_indexing_method(arr, mask, expected):
    """
    Tests the `BACKEND_TO_INDEXING_METHOD` dictionary with different backends.
    """
    _type = str(type(arr))
    result = BACKEND_TO_INDEXING_METHOD[_type](arr, mask)
    assert np.array_equal(result, expected)

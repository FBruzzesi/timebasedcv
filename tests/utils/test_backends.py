from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
import pytest

from timebasedcv.utils._backends import BACKEND_TO_INDEXING_METHOD
from timebasedcv.utils._backends import default_indexing_method

size = 10
arr = np.arange(size)
valid_mask = arr % 2 == 0
expected = np.array([v for v in range(size) if v % 2 == 0])

invalid_mask = np.zeros(shape=size + 1).astype(bool)


@pytest.mark.parametrize(
    "arr, mask, expected, context",
    [
        (arr, valid_mask, expected, does_not_raise()),
        (arr, invalid_mask, expected, pytest.raises(ValueError, match="Length of arr and mask must be equal.")),
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
    arr = nw.from_native(arr, allow_series=True, eager_only=True, strict=False)
    mask = nw.from_native(mask, series_only=True, strict=False)
    _type = str(type(arr))
    result = BACKEND_TO_INDEXING_METHOD[_type](arr, mask)
    assert np.array_equal(result, expected)

from typing import Callable

import numpy as np
import pandas as pd


def default_indexing_method(arr, mask):
    """
    Default indexing method for arrays.

    Remark that `arr` should support indexing with an array.

    Arguments:
        arr: The array-like to index.
        mask: The boolean mask to use for indexing.
    """
    if len(arr) != len(mask):
        raise ValueError("Length of arr and mask must be equal.")
    return arr[mask]


BACKEND_TO_INDEXING_METHOD: dict[str, Callable] = {
    str(np.ndarray): default_indexing_method,
    str(pd.DataFrame): lambda df, mask: df.loc[mask],
    str(pd.Series): lambda s, mask: s.loc[mask],
}

try:
    import polars as pl

    BACKEND_TO_INDEXING_METHOD[str(pl.DataFrame)] = lambda df, mask: df.filter(mask)
    BACKEND_TO_INDEXING_METHOD[str(pl.Series)] = lambda s, mask: s.filter(mask)

except ImportError:
    pass

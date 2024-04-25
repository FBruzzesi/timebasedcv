from typing import Callable, Dict, TypeVar

import numpy as np
import narwhals as nw


def default_indexing_method(arr, mask):
    """Default indexing method for arrays.

    !!! warning
        Remark that `arr` should support indexing with an array.

    Arguments:
        arr: The array-like to index.
        mask: The boolean mask to use for indexing.
    """
    if len(arr) != len(mask):
        raise ValueError("Length of arr and mask must be equal.")
    return arr[mask]


T_NW = TypeVar("T_NW", nw.DataFrame, nw.Series)


def nw_indexing_method(_dfs: T_NW, mask) -> T_NW:
    """Indexing method for Narwhals dataframes and series.

    Arguments:
        df: The Narwhals dataframe or series to index.
        mask: The boolean mask to use for indexing.
    """
    return _dfs.filter(mask)


BACKEND_TO_INDEXING_METHOD: Dict[str, Callable] = {
    str(np.ndarray): default_indexing_method,
    str(nw.DataFrame): nw_indexing_method,
    str(nw.Series): nw_indexing_method,
}

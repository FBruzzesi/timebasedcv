from typing import Callable, Dict, TypeVar

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


T_PD = TypeVar("T_PD", pd.DataFrame, pd.Series)


def pd_indexing_method(_dfs: T_PD, mask) -> T_PD:
    """
    Indexing method for pandas dataframes and series.

    Arguments:
        df: The pandas dataframe or series to index.
        mask: The boolean mask to use for indexing.
    """
    return _dfs.loc[mask]


BACKEND_TO_INDEXING_METHOD: Dict[str, Callable] = {
    str(np.ndarray): default_indexing_method,
    str(pd.DataFrame): pd_indexing_method,
    str(pd.Series): pd_indexing_method,
}

try:
    import polars as pl

    T_PL = TypeVar("T_PL", pl.DataFrame, pl.Series)

    def pl_indexing_method(_dfs: T_PL, mask) -> T_PL:
        """
        Indexing method for polars dataframes and series.

        Arguments:
            _dfs: The polars dataframe or series to index.
            mask: The boolean mask to use for indexing.
        """
        return _dfs.filter(mask)

    BACKEND_TO_INDEXING_METHOD.update(
        {
            str(pl.DataFrame): pl_indexing_method,
            str(pl.Series): pl_indexing_method,
        }
    )

except ImportError:
    pass

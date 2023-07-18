from typing import Callable, Type


def default_indexing_method(arr, mask):
    return arr[mask]


BACKEND_TO_INDEXING_METHOD: dict[Type, Callable] = {}

try:
    import numpy as np

    BACKEND_TO_INDEXING_METHOD[np.ndarray] = lambda arr, mask: arr[mask]

except ImportError:
    pass

try:
    import pandas as pd

    BACKEND_TO_INDEXING_METHOD[pd.DataFrame] = lambda df, mask: df.loc[mask]

except ImportError:
    pass

try:
    import polars as pl

    BACKEND_TO_INDEXING_METHOD[pl.DataFrame] = lambda df, mask: df.filter(mask)

except ImportError:
    pass

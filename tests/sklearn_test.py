from contextlib import nullcontext as does_not_raise
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV

from timebasedcv.sklearn import TimeBasedCVSplitter

RNG = np.random.default_rng()

# Define a fix set of valid arguments

start_dt = pd.Timestamp(2023, 1, 1)
end_dt = pd.Timestamp(2023, 1, 31)

time_series = pd.Series(pd.date_range(start_dt, end_dt, freq="D"))
size = len(time_series)

df = pd.DataFrame(data=RNG.normal(size=(size, 2)), columns=["a", "b"]).assign(  # noqa: PD901
    date=time_series,
    y=lambda t: t[["a", "b"]].sum(axis=1),
)

X, y = df[["a", "b"]], df["y"]

err_msg_shape = "Invalid shape: "


def test_cv_splitter(valid_kwargs):
    """
    Tests the TimeBasedCVSplitter `__init__` and `split` methods as well as its
    compatibility with sklearn's _CV Splitter_s.
    """
    cv = TimeBasedCVSplitter(
        time_series=time_series,
        start_dt=start_dt,
        end_dt=end_dt,
        **valid_kwargs,
    )

    assert cv.time_series_ is not None
    assert cv.start_dt_ is not None
    assert cv.end_dt_ is not None
    assert isinstance(cv.n_splits, int)
    assert isinstance(cv.size_, int)

    param_grid = {
        "alpha": np.linspace(0.1, 2, 10),
        "fit_intercept": [True, False],
        "positive": [True, False],
    }

    random_search_cv = RandomizedSearchCV(
        estimator=Ridge(),
        param_distributions=param_grid,
        cv=cv,
        n_jobs=-1,
    ).fit(X, y)

    assert random_search_cv.best_estimator_ is not None


@pytest.mark.parametrize("size", [size])
@pytest.mark.parametrize(
    "X_shape, y_shape, g_shape, context",
    [
        ((size, 2), (size,), (size, 2), does_not_raise()),
        ((size + 1, 2), (size,), (size, 2), pytest.raises(ValueError, match=err_msg_shape)),
        ((size, 2), (size + 1,), (size, 2), pytest.raises(ValueError, match=err_msg_shape)),
        ((size, 2), (size,), (size + 1, 2), pytest.raises(ValueError, match=err_msg_shape)),
    ],
)
def test_cv_splitter_validate_split(
    size: int,
    X_shape: Tuple[int, int],
    y_shape: Tuple[int],
    g_shape: Tuple[int, int],
    context,
):
    """Test the TimeBasedCVSplitter._validate_split_args static method."""

    with context:
        TimeBasedCVSplitter._validate_split_args(  # noqa: SLF001
            size=size,
            X=RNG.normal(size=X_shape),
            y=RNG.normal(size=y_shape),
            groups=RNG.normal(size=g_shape),
        )

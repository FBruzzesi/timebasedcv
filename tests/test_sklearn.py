from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV

from timebasedcv.sklearn import TimeBasedCVSplitter

RNG = np.random.default_rng()

# Define a fix set of valid arguments


err_msg_shape = "Invalid shape: "


def test_cv_splitter(valid_kwargs, generate_test_data):
    """
    Tests the TimeBasedCVSplitter `__init__` and `split` methods as well as its
    compatibility with sklearn's _CV Splitter_s.
    """
    start_dt, end_dt, time_series, X, y = generate_test_data
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


@pytest.mark.parametrize(
    "x_extra, y_extra, g_extra, context",
    [
        (0, 0, 0, does_not_raise()),
        (1, 0, 0, pytest.raises(ValueError, match=err_msg_shape)),
        (0, 1, 0, pytest.raises(ValueError, match=err_msg_shape)),
        (0, 0, 1, pytest.raises(ValueError, match=err_msg_shape)),
    ],
)
def test_cv_splitter_validate_split(
    x_extra: int,
    y_extra: int,
    g_extra: int,
    context,
):
    """Test the TimeBasedCVSplitter._validate_split_args static method."""
    SIZE = 10
    with context:
        TimeBasedCVSplitter._validate_split_args(  # noqa: SLF001
            size=SIZE,
            X=RNG.normal(size=(SIZE + x_extra, 2)),
            y=RNG.normal(size=(SIZE + y_extra,)),
            groups=RNG.normal(size=(SIZE + g_extra, 2)),
        )

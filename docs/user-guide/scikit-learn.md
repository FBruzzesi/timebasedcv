
# Scikit-learn component 🚀

[scikit-learn CV Splitters](https://scikit-learn.org/stable/common_pitfalls.html#id3){:target="_blank"} require a splitter to behave in a certain way:

- `.split(...)` method should have the following signature: `.split(self, X, y, groups)`.
- `.split(...)` method should return train and test indices of the split.
- know the total number of splits a priori, independently of `X, y, groups` arrays.

Therefore, our [`TimeBasedSplit`](../api/timebasedcv.md#timebasedcv.core.TimeBasedSplit){:target="_blank"} is **not** compatible with scikit-learn.

## `TimeBasedCVSplitter`

Considering the above requirements, we provide a scikit-learn compatible splitter: [`TimeBasedCVSplitter`](../api/sklearn.md#timebasedcv.sklearn.TimeBasedCVSplitter){:target="_blank"} in the sklearn module.

```py
from timebasedcv.sklearn import TimeBasedCVSplitter
```

To be scikit-learn compatible, `TimeBasedCVSplitter` is _initialized_ with the same parameters of `TimeBasedSplit` **and** the `time_series` containing the time information used to generate the train and test indices of each split.

From a point of view, `TimeBasedCVSplitter` has all the features that `TimeBasedSplit` has, plus the compatibility with scikit-learn.

This comes to the cost of requiring to know `time_series` beforehand, during `.__init__()` step. Therefore it is not possible to instantiate the split class once and re-use it with different time series dynamically.

## Example

In the following example we will see how to use it with a [`Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html){:target="_blank"} model and [`RandomizedSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html){:target="_blank"} to find the best parameters for the model.

```python hl_lines="7" title="Imports"
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV

from timebasedcv.sklearn import TimeBasedCVSplitter
```

```python title="Generate the data"
RNG = np.random.default_rng(seed=42)

dates = pd.Series(pd.date_range("2023-01-01", "2023-01-31", freq="D"))
size = len(dates)

df = (pd.concat([
        pd.DataFrame({
            "time": pd.date_range(start, end, periods=_size, inclusive="left"),
            "a": RNG.normal(size=_size-1),
            "b": RNG.normal(size=_size-1),
        })
        for start, end, _size in zip(dates[:-1], dates[1:], RNG.integers(2, 24, size-1))
    ])
    .reset_index(drop=True)
    .assign(y=lambda t: t[["a", "b"]].sum(axis=1) + RNG.normal(size=t.shape[0])/25)
)

df.set_index("time").resample("D").agg(count=("y", np.size)).head(5)
```

```terminal
            count
time
2023-01-01      1
2023-01-02     10
2023-01-03      9
2023-01-04     14
2023-01-05     20
```

```python title="Run cross validation"
X, y, time_series = df.loc[:, ["a", "b"]], df["y"], df["time"]

cv = TimeBasedCVSplitter(
    frequency="days",
    train_size=10,
    forecast_horizon=3,
    gap=0,
    stride=2,
    window="rolling",
    time_series=time_series,  # (1)
)

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

random_search_cv.best_params_
```

1. Required in `.__init__()` method to generate the train and test indices of each split.

```terminal
{'positive': False, 'fit_intercept': False, 'alpha': 0.522}
```

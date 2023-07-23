# Getting Started

The following sections will guide you through the basic usage of the library.

## TimeBasedSplit

The `TimeBasedSplit` class allows to define a time based split with a given frequency, train size, test size, gap, stride and window type.

```python
from timebasedcv import TimeBasedSplit

tbs = TimeBasedSplit(
    frequency="days",
    train_size=30,
    forecast_horizon=7,
    gap=0,
    stride=3,
    window="rolling",
)
```

The available values for the parameters are:

- `frequency`: "days", "seconds", "microseconds", "milliseconds", "minutes", "hours", "weeks".
- `train_size`: any strictly positive integer.
- `forecast_horizon`: any strictly positive integer.
- `gap`: any non-negative integer.
- `stride`: any strictly positive integer or `None`.
- `window`: "rolling" or "expanding".

Once the instance is created, it is possible to split the data using the `split` method. This method requires to pass a `time_series` as input to create the boolean masks for train and test.

Optionally it is possible to pass a `start_dt` and `end_dt` arguments as well. If provided, they are used in place of the `time_series.min()` and `time_series.max()` respectively to determing the period.

This is useful because the series does not necessarely starts from the first date and/or terminates in the last date of the time period of interest.

```python
import pandas as pd
import numpy as np

dates = pd.Series(pd.date_range("2023-01-01", "2023-12-31", freq="D"))
size = len(dates)

df = (pd.DataFrame(data=np.random.randn(size, 2), columns=["a", "b"])
    .assign(y=lambda t : t[["a", "b"]].sum(axis=1))
)

X, y = df[["a", "b"]], df["y"]

print(f"Number of splits: {tbs.n_splits_of(time_series=dates)}")
#  Number of splits: 112

for X_train, X_forecast, y_train, y_forecast in tbs.split(X, y, time_series=dates):
    print(f"Train: {X_train.shape}, Forecast: {X_forecast.shape}")

# Train: (30, 2), Forecast: (7, 2)
# Train: (30, 2), Forecast: (7, 2)
# ...
# Train: (30, 2), Forecast: (7, 2)
```

Another optional parameter that can be passed to the `split` method is `return_splitstate`. If `True`, the method will return a [`SplitState`](api/splitstate/) dataclass which contains the "split" points for training and test, namely `train_start`, `train_end`, `forecast_start` and `forecast_end`. These can be useful if a particular logic needs to be applied to the data before training and/or forecasting.


## TimeBasedCVSplitter


## Examples of splits

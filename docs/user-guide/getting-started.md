# Getting started 🐍

The following sections will guide you through the basic usage of the library.

## `TimeBasedSplit`

The [`TimeBasedSplit`](../api/timebasedcv.md#timebasedcv.core.TimeBasedSplit) class allows to define a way to split your data based on time. There is a (long) list of parameters that can be set to define how to generate the splits. These allow for a lot of flexibility in how the data is split. Here is an overview of them:

- `frequency`: we do not try to infer the frequency from the data, this information has to be specified beforehand. Available values are "days", "seconds", "microseconds", "milliseconds", "minutes", "hours", "weeks".
- `train_size`: defines the minimum number of time units required to be in the train set, e.g. if `frequency="days"` and `train_size=30`, the train set will have at least 30 days.
- `forecast_horizon`: specifies the number of time units to forecast, e.g. if `frequency="days"` and `forecast_horizon=7`, the forecast set will have 7 days. Notice that at the end of the time series, the forecast set might be smaller than the specified `forecast_horizon`.
- `gap`: the number of time units to skip between the end of the train set and the start of the forecast set.
- `stride`: how many time unit to move forward after each split. If `None`, the stride is equal to the `forecast_horizon`.
- `window`: it can be either "rolling" or "expanding"
- `mode`: it can be either "forward" or "backward" (generating splits either starting from the beginning or the end of the time series).

Well that is a lot of parameters! But in our opinion it is what makes the library so flexible and powerful to be able to cover the large majority of use cases!

!!! info
    As the list of so long, and it could be easy to provide values in the wrong order and/or be very hard to understand what each number means, we require to pass them as keyword only arguments!

```python title="Create a TimeBasedSplit instance"
from timebasedcv import TimeBasedSplit

tbs = TimeBasedSplit(
    frequency="days",
    train_size=10,
    forecast_horizon=5,
    gap=1,
    stride=3
)
```

Once an instance is created, it is possible to split a list of arrays using the `.split(...)` method, such method requires to pass a `time_series` as input to know how to split each array.

Optionally it is possible to pass a `start_dt` and `end_dt` arguments as well. If provided, they are used in place of the `time_series.min()` and `time_series.max()` respectively to determine the period.

This is useful because the series does not necessarely starts from the first date and/or terminates in the last date of the time period of interest, and it could lead to skewed splits.

!!! info
    We made the opinionated choice of returning the sliced arrays from `.split(...)`, while scikit-learn CV Splitters return train and test indices of the split.

```python title="Generate the data"
import numpy as np
import pandas as pd

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
2023-01-01      2
2023-01-02     18
2023-01-03     15
2023-01-04     10
2023-01-05     10
```

Now let's run split the data with the provided `TimeBasedSplit` instance:

```py title="Generate the splits"
X, y, time_series = df.loc[:, ["a", "b"]], df["y"], df["time"]

for X_train, X_forecast, y_train, y_forecast in tbs.split(X, y, time_series=time_series):
    print(f"Train: {X_train.shape}, Forecast: {X_forecast.shape}")
```

```terminal
Train: (100, 2), Forecast: (51, 2)
Train: (114, 2), Forecast: (50, 2)
...
Train: (124, 2), Forecast: (40, 2)
Train: (137, 2), Forecast: (22, 2)
```

As we can see, each split does not necessarely have the same number of points, this is because the time series has a different number of points per day.

Let's visualize the splits (blue dots represent the train points, while the red dots represent the forecastng points).

![basic-cv-split](../img/basic-cv-split.png)

??? example "Code to generate the plot"

    ```python
    import plotly.graph_objects as go

    fig = go.Figure()

    for _fold, (train_forecast, split_state) in enumerate(
        tbs.split(y/25, time_series=time_series, return_splitstate=True),
        start=1,
        ):

        train, forecast = train_forecast

        ts = split_state.train_start
        te = split_state.train_end
        fs = split_state.forecast_start
        fe = split_state.forecast_end

        fig.add_trace(
            go.Scatter(
                x=time_series[time_series.between(ts, te, inclusive="left")],
                y=train + _fold,
                name=f"Train Fold {_fold}",
                mode="markers",
                marker={"color": "rgb(57, 105, 172)"}
            )
        )

        fig.add_trace(
            go.Scatter(
                x=time_series[time_series.between(fs, fe, inclusive="left")],
                y=forecast + _fold,
                name=f"Forecast Fold {_fold}",
                mode="markers",
                marker={"color": "indianred"}
            )
        )

        fig.update_layout(
            title={
                "text": "Time Based Cross Validation",
                "y":0.95, "x":0.5,
                "xanchor": "center",
                "yanchor": "top"
            },
            showlegend=True,
            height=500,
            yaxis = {"autorange": "reversed", "title": "Fold"}
        )

    fig.show()
    ```

Here is an example of a few different configuration values for the splitter:

![multi-cv-split](../img/cv-multi-config.png)

??? example "Code to generate the plot"

    ```python
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    configs = [
        {
            "frequency": "days",
            "train_size": 14,
            "forecast_horizon": 7,
            "gap": 2,
            "stride": 5,
            "window": "expanding"
        },
        {
            "frequency": "days",
            "train_size": 14,
            "forecast_horizon": 7,
            "gap": 2,
            "stride": 5,
            "window": "rolling"
        },
        {
            "frequency": "days",
            "train_size": 14,
            "forecast_horizon": 7,
            "gap": 0,
            "stride": None,
            "window": "rolling"
        }
    ]

    fig = make_subplots(
        rows=len(configs),
        cols=1,
        subplot_titles=[str(config) for config in configs],
        shared_xaxes=True,
        vertical_spacing=0.1,
        x_title="Time",
    )

    for _row, config in enumerate(configs, start=1):

        tbs = TimeBasedSplit(**config)

        for _fold, (train_forecast, split_state) in enumerate(tbs.split(y/25, time_series=time_series, return_splitstate=True), start=1):

            train, forecast = train_forecast

            ts = split_state.train_start
            te = split_state.train_end
            fs = split_state.forecast_start
            fe = split_state.forecast_end

            fig.add_trace(
                go.Scatter(
                    x=time_series[time_series.between(ts, te, inclusive="left")],
                    y=train + _fold,
                    name=f"Train Fold {_fold}",
                    mode="markers",
                    marker={"color": "rgb(57, 105, 172)"}
                ),
                row=_row,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=time_series[time_series.between(fs, fe, inclusive="left")],
                    y=forecast + _fold,
                    name=f"Forecast Fold {_fold}",
                    mode="markers",
                    marker={"color": "indianred"}
                ),
                row=_row,
                col=1,
            )

    fig.update_layout(
        title={
            "text": "Time Based Cross Validation",
            "y":0.95, "x":0.5,
            "xanchor": "center",
            "yanchor": "top"
        },
        showlegend=False,
        height=750,
        **{
            f"yaxis{i}": {"autorange": "reversed", "title": "Fold"}
            for i in range(1, len(configs)+1)
        }
    )

    fig.show()

    ```

### Multiple arrays

It is possible to split multiple any arbitrary number of arrays at the same time, similarly to how [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html){:target="_blank"} behaves.

```python title="Generate the splits"
time_series, a, b, y  = df.to_numpy().T

for a_train, a_test, b_train, b_test, y_train, y_test in tbs.split(a, b, y, time_series=time_series):
    # Do some magic!
```

!!! info
    The only requirement is that the all the arrays must have the same length, as we use the mask on the `time_series` to slice each one of them.

!!! warning
    Ideally each array can be a different type (numpy, pandas, polars, and so on...), in practice there are a few limitations that might arise from the different types, so please be aware of that.

    We are working to make the library more flexible and to support more types of arrays and more interactions between them.

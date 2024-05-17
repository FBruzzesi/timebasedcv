# Advanced features

```
print(f"Number of splits: {tbs.n_splits_of(time_series=time_series)}")
#  Number of splits: 112
```
Another optional parameter that can be passed to the `split` method is `return_splitstate`. If `True`, the method will return a [`SplitState`](../api/splitstate.md) dataclass which contains the "split" points for training and test, namely `train_start`, `train_end`, `forecast_start` and `forecast_end`. These can be useful if a particular logic needs to be applied to the data before training and/or forecasting.


## Examples of Cross Validation

The following examples show how the CV works with different parameters.

First and foremost let's generate some random data. The following code generates a time series with randomly spaced points between 2023-01-01 and 2023-01-31.

```python
import pandas as pd
import numpy as np
np.random.seed(42)

dates = pd.Series(pd.date_range("2023-01-01", "2023-01-31", freq="D"))
size = len(dates)

df = pd.concat([
    pd.DataFrame({
        "time": pd.date_range(start, end, periods=_size, inclusive="left"),
        "value": np.random.randn(_size-1)/25,
    })
    for start, end, _size in zip(dates[:size], dates[1:], np.random.randint(2, 24, size-1))
]).reset_index(drop=True)

time_series, X = df["time"], df["value"]
df.set_index("time").resample("D").count().head(5)
```

```terminal
time	        value
2023-01-01	14
2023-01-02	2
2023-01-03	22
2023-01-04	11
2023-01-05	1
```

As we can see every day has a different number of points.

Now let's plot train and forecasting splits with different split strategies (or configurations).

The blue dots represent the train points, while the red dots represent the forecastng points.

![cross-validation](../img/cross-validation.png)

??? example "Code to generate the plot"

    ```python
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

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from timebasedcv import TimeBasedSplit

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
        fmt = "%Y-%m-%d"

        for _fold, (train_forecast, split_state) in enumerate(tbs.split(X, time_series=time_series, return_splitstate=True), start=1):

            train, forecast = train_forecast

            ts = split_state.train_start.strftime(fmt)
            te = split_state.train_end.strftime(fmt)
            fs = split_state.forecast_start.strftime(fmt)
            fe = split_state.forecast_end.strftime(fmt)

            print(ts, te, fs, fe)
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
        height=1000,
        **{
            f"yaxis{i}": {"autorange": "reversed", "title": "Fold"}
            for i in range(1, len(configs)+1)
        }
    )

    fig.show()
    ```

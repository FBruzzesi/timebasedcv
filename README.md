<img src="docs/img/timebasedcv-logo.svg" width=185 height=185 align="right">

![license-shield](https://img.shields.io/github/license/FBruzzesi/timebasedcv)
![interrogate-badge](docs/img/interrogate-shield.svg)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![coverage-badge](docs/img/coverage.svg)
![versions-shield](https://img.shields.io/pypi/pyversions/timebasedcv)

# Time based cross validation

**timebasedcv** is a Python codebase that provides a cross validation strategy based on time.

---

[Documentation](https://fbruzzesi.github.io/timebasedcv) | [Repository](https://github.com/fbruzzesi/timebasedcv) | [Issue Tracker](https://github.com/fbruzzesi/timebasedcv/issues)

---

## Alpha Notice

This codebase is experimental and is working for my use cases. It is very probable that there are cases not covered and for which it breaks (badly). If you find them, please feel free to open an issue in the [issue page](https://github.com/FBruzzesi/timebasedcv/issues) of the repo.

## Description

The current implementation of [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) lacks the flexibility of having multiple samples within the same time period/unit.

This codebase addresses such problem by providing a cross validation strategy based on a **time period** rather than the number of samples. This is useful when the data is time dependent, and the model should be trained on past data and tested on future data, independently from the number of observations present within a given time period.

Temporal data leakage is an issue and we want to prevent that from happening!

We introduce two main classes:

- [`TimeBasedSplit`](https://fbruzzesi.github.io/timebasedcv/api/timebasedsplit/#timebasedcv.timebasedsplit.TimeBasedSplit) allows to define a time based split with a given frequency, train size, test size, gap, stride and window type. Its core method `split` requires to pass a time series as input to create the boolean masks for train and test from the instance information defined above. Therefore it is not compatible with [scikit-learn CV Splitters](https://scikit-learn.org/stable/common_pitfalls.html#id3).
- [`TimeBasedCVSplitter`](https://fbruzzesi.github.io/timebasedcv/api/timebasedsplit/#timebasedcv.timebasedsplit.TimeBasedCVSplitter) conforms with scikit-learn CV Splitters but requires to pass the time series as input to the instance. That is because a CV Splitter needs to know a priori the number of splits, and the `split` method shouldn't take any extra arguments as input other than the arrays to split.

## Installation

**timebasedcv** is a published Python package on [pypi](https://pypi.org/), therefore it can be installed directly via pip, as well as from source using pip and git, or with a local clone:

<details open>

<summary> <b>pip</b> (suggested)</summary>

```bash
python -m pip install timebasedcv
```

</details>

<details closed>

<summary> <b>pip + source/git</b></summary>

```bash
python -m pip install git+https://github.com/FBruzzesi/timebasedcv.git
```

</details>

<details closed>

<summary> <b>local clone</b></summary>

```bash
git clone https://github.com/FBruzzesi/timebasedcv.git
cd timebasedcv
python -m pip install .
```

</details>

## Dependencies

As of **timebasecv v0.1.0**, the only two dependencies are [`numpy`](https://numpy.org/doc/stable/index.html) and [`narwhals>=0.7.15`](https://marcogorelli.github.io/narwhals/).

The latter allows to have a compatibility layer between polars, pandas and other dataframe libraries. Therefore, as long as narwhals supports such dataframe object, we will as well.

## Quickstart

The following code snippet is all you need to get started, yet consider checking out the [Getting Started](https://fbruzzesi.github.io/timebasedcv/getting-started/) section of the documentation for a detailed guide on how to use the library.

First let's generate some data with different number of points per day:

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

Now let's run the split with a given frequency, train size, test size, gap, stride and window type:

```python
from timebasedcv import TimeBasedSplit

configs = [
    {
        "frequency": "days",
        "train_size": 14,
        "forecast_horizon": 7,
        "gap": 2,
        "stride": 5,
        "window": "expanding"
    },
    ...
]

tbs = TimeBasedSplit(**config)


fmt = "%Y-%m-%d"
for train_set, forecast_set in tbs.split(X, time_series=time_series):

    # Do some magic here
```

Let's see how `train_set` and `forecasting_set` splits would look likes for different split strategies (or configurations).

The blue dots represent the train points, while the red dots represent the forecastng points.

![cross-validation](docs/img/cross-validation.png)

## Contributing

Please read the [Contributing guidelines](https://fbruzzesi.github.io/timebasedcv/contribute/) in the documentation site.

## License

The project has a [MIT Licence](https://github.com/FBruzzesi/timebasedcv/blob/main/LICENSE)

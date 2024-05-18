
![license-shield](https://img.shields.io/github/license/FBruzzesi/timebasedcv)
![interrogate-badge](img/interrogate-shield.svg)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![coverage-badge](img/coverage.svg)
![versions-shield](https://img.shields.io/pypi/pyversions/timebasedcv)

<img src="img/timebasedcv-logo.svg" width=160 height=160 align="right">

# Timebased Cross Validation

**timebasedcv** is a Python codebase that provides a cross validation strategy based on time.

---

[Documentation](https://fbruzzesi.github.io/timebasedcv) | [Repository](https://github.com/fbruzzesi/timebasedcv){:target="_blank"} | [Issue Tracker](https://github.com/fbruzzesi/timebasedcv/issues){:target="_blank"}

---

## Disclaimer ⚠️

This codebase is experimental and is working for my use cases. It is very probable that there are cases not entirely covered and for which it could break (badly). If you find them, please feel free to open an issue in the [issue page](https://github.com/FBruzzesi/timebasedcv/issues/new){:target="_blank"} of the repo.

## Description ✨

The current implementation of [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html){:target="_blank"} lacks the flexibility of having multiple samples within the same time period (or time unit).

**timebasedcv** addresses such problem by providing a cross validation strategy based on a **time period** rather than the number of samples. This is useful when the data is time dependent, and the split should keep together samples within the same time window.

Temporal data leakage is an issue and we want to prevent it from happening by providing splits that make sure the past and the future are well separated, so that data leakage does not spoil in a model cross validation.

Again, these splits points solely depend on the time period and not the number of observations.

### Features 📜

We introduce two main classes:

- [`TimeBasedSplit`](api/timebasedcv.md#timebasedcv.core.TimeBasedSplit){:target="_blank"} allows to define a split based on time unit (frequency), train size, test size, gap, stride, window type and mode.

    !!! warning
        `TimeBasedSplit` is **not** compatible with [scikit-learn CV Splitters](https://scikit-learn.org/stable/common_pitfalls.html#id3){:target="_blank"}.

        In fact, we have made the (opinioned) choice to:

        - Return the sliced arrays from `.split(...)`, while scikit-learn CV Splitters return train and test indices of the split.
        - Require to pass the time series as input to `.split(...)` method, while scikit-learn CV Splitters require to provide only `X, y, groups` to `.split(...)`.
        - Such time series is used to generate the boolean masks with which we slice the original arrays into train and test for each split.

- Considering the above choices, we also provide a scikit-learn compatible splitter: [`TimeBasedCVSplitter`](api/sklearn.md#timebasedcv.sklearn.TimeBasedCVSplitter){:target="_blank"}. Considering the signature that `.split(...)` requires and the fact that CV Splitters need to know a priori the number of splits, `TimeBasedCVSplitter` is initialized with the time series containing the time information used to generate the train and test indices of each split.

## Installation 💻

TL;DR:

```bash
python -m pip install timebasedcv
```

For further information, please refer to the dedicated [installation](installation.md) section.

## Getting Started 🏃

Please refer to the dedicated [getting started](user-guide/getting-started.md) section.

## Contributing ✌️

Please refer to the dedicated [contributing guidelines](contribute.md) section.

## License 👀

The project has a [MIT Licence](https://github.com/FBruzzesi/timebasedcv/blob/main/LICENSE).

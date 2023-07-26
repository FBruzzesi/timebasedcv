<img src="img/timebasedcv-logo.svg" width=185 height=185 align="right">

![](https://img.shields.io/github/license/FBruzzesi/timebasedcv)
<img src ="img/interrogate-shield.svg">
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Timebased Cross Validation

**timebasedcv** is a Python codebase that provides a cross validation strategy based on time.

---

**Documentation**: https://fbruzzesi.github.io/timebasedcv

**Source Code**: https://github.com/fbruzzesi/timebasedcv

---

## Alpha Notice

This codebase is experimental and is working for my use cases. It is very probable that there are cases not covered and for which it breaks (badly). If you find them, please feel free to open an issue in the [issue page](https://github.com/FBruzzesi/timebasedcv/issues) of the repo.

## Description

The current implementation of [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) lacks the flexibility of having multiple samples within the same time period/unit.

This codebase addresses such problem by providing a cross validation strategy based on a time period rather than the number of samples. This is useful when the data is time dependent, and the model should be trained on past data and tested on future data, independently from the number of observations present within a given time period.

We introduce two main classes:

- [`TimeBasedSplit`](api/timebasedsplit/#timebasedcv.timebasedsplit.TimeBasedSplit): a class that allows to define a time based split with a given frequency, train size, test size, gap, stride and window type. It's core method `split` requires to pass a time series as input to create the boolean masks for train and test from the instance information defined above. Therefore it is not compatible with [scikit-learn CV Splitters](https://scikit-learn.org/stable/common_pitfalls.html#id3).
- [`TimeBasedCVSplitter`](api/timebasedsplit/#timebasedcv.timebasedsplit.TimeBasedCVSplitter): a class that conforms with scikit-learn CV Splitters but requires to pass the time series as input to the instance. That is because a CV Splitter needs to know a priori the number of splits and the `split` method shouldn't take any extra arguments as input other than the arrays to split.

## Installation

**timebasedcv** is a published Python package on [pypi](https://pypi.org/), therefore it can be installed directly via pip, as well as from source using pip and git, or with a local clone:

=== "pip"

    ```bash
    python -m pip install timebasedcv
    ```

=== "pip + source/git"

    ```bash
    python -m pip install git+https://github.com/FBruzzesi/timebasedcv.git
    ```

=== "local clone"

    ```bash
    git clone https://github.com/FBruzzesi/timebasedcv.git
    cd timebasedcv
    python -m pip install .
    ```

## Getting Started

Please refer to the dedicated page [Getting Started](getting-started/).

## License

The project has a [MIT Licence](https://github.com/FBruzzesi/timebasedcv/blob/main/LICENSE)

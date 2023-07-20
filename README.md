# Time based cross validation

## Description

**timebasedcv** is a Python codebase that provides a cross validation strategy based on time.

The idea is to split the data in train and test sets based on a time threshold rather than a number of samples.
This is useful when the data is time dependent, and the model should be trained on past data and tested on future data,
independently from the number of observations within a given time period.

The main class is `TimeBasedSplit`, which allows to define:

- a time `frequency` (e.g. "minutes", "hours", "days", and so on)
- a `train_size`: number of time units (in terms of frequency) are used for the train
- a test size (called `forecast_horizon`): the number of time units used for test
- a `gap`: number time units that should be skipped between the train and test sets
- a `stride`: how many time units should we move forward between two consecutive splits
- a `window` type: whether to use a rolling/sliding or expanding window.

The main method to generate the split is `TimeBasedSplit.split()`, which behaves _similarly_ to scikit-learn [CV Splitters](https://scikit-learn.org/stable/common_pitfalls.html#id3), however notice that since the split is based on time and not on a fixed number of samples, we require to pass a `time_series` used to create the boolean masks for train and test from the instance information defined above.

In addition to such class we provide `TimeBasedCVSplitter` which conforms with scikit-learn CV Splitters but requires to pass the time series as input to the instance. That is because a CV Splitter needs to know a priori the number of splits and the `split` method shouldn't take any extra arguments as input other than the arrays to split.

## Installation

**timebasedcv** is _not_ published as a Python package on [pypi](https://pypi.org/), therefore it cannot be installed with pip directly.

However it is possible to install it from source using pip and git, or with a local clone:

### source/git

```bash
python -m pip install git+https://github.com/FBruzzesi/timebased-cv.git
```

### local clone

```bash
git clone https://github.com/FBruzzesi/timebased-cv.git
cd timebased-cv
python -m pip install .
```

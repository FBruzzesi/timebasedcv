# Installation

**timebasedcv** is a published Python package on [pypi](https://pypi.org/project/timebasedcv), therefore it can be installed directly via pip, as well as from source using pip and git, or with a local clone:

=== "pip (suggested)"

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

## Dependencies

!!! info
    The minimum Python version supported is 3.8.

As of **timebasecv v0.1.0**, the only two dependencies are [`numpy`](https://numpy.org/doc/stable/index.html) and [`narwhals>=0.7.15`](https://narwhals-dev.github.io/narwhals/).

**Narwhals** allows to have a compatibility layer between polars, pandas and other dataframe libraries. Therefore, as long as narwhals supports such dataframe object, we will as well.

As of **timebasedcv v0.2.0**, in order to use `TimeBasedCVSplitter`, [`scikit-learn>=0.19`](https://scikit-learn.org/stable/) is required.

# Contributing

## Guidelines

We welcome contributions to the library! If you have a bug fix or new feature that you would like to contribute, please follow the steps below:

1. Fork the repository on GitHub.
2. Clone the repository to your local machine.
3. Create a new branch for your bug fix or feature.
4. Make your changes and test them thoroughly, making sure that it passes all current tests.
5. Commit your changes and push the branch to your fork.
6. Open a pull request on the main repository.

## Code of Conduct

All contributors are expected to follow the project's code of conduct, which is based on the Contributor Covenant.

### Reporting Bugs

If you find a bug in the library, please report it by opening an [issue on GitHub](https://github.com/FBruzzesi/timebasedcv/issues). Be sure to include the version of the library you're using, as well as any error messages or tracebacks and a reproducible example.

### Requesting Features

If you have a suggestion for a new feature, please open an [issue on GitHub](https://github.com/FBruzzesi/timebasedcv/issues). Be sure to explain the problem that you're trying to solve and how you think the feature would solve it.

### Submitting Pull Requests

When submitting a pull request, please make sure that you've followed the steps above and that your code has been thoroughly tested. Also, be sure to include a brief summary of the changes you've made and a reference to any issues that your pull request resolves.

### Code formatting

Compclasses uses [black](https://black.readthedocs.io/en/stable/index.html) and [isort](https://pycqa.github.io/isort/) with the following  parameters for code formatting:

```bash
isort --profile black -l 90 timebasedcv tests
black --target-version py38 --line-length 90 timebasedcv tests
```

As part of the checks on pull requests, it is checked whether the code follows those standards. To ensure that the standard is met, it is recommended to install [pre-commit hooks](https://pre-commit.com/):

```bash
python -m pip install pre-commit
pre-commit install
```

## Developing

Let's suppose that you already did steps 1-3 from the above list, now you should install the library and its developing dependencies  in editable way.

First move into the repo folder: `cd timebasedcv`.

Then:

=== "with make"

    ```bash
    make init-dev
    ```

=== "without make"

    ```bash
    pip install -e ".[all]" --no-cache-dir
    pre-commit install
    ```

Now you are ready to proceed with all the changes you want to!

## Testing

Once you are done with changes, you should:

- add tests for the new features in the `/tests` folder
- make sure that new features do not break existing codebase by running tests:

    === "with make"

        ```bash
        make test
        ```

    === "without make"

        ```bash
        pytest tests -vv
        ```

## Docs

The documentation is generated using [mkdocs-material](https://squidfunk.github.io/mkdocs-material/), the API part uses [mkdocstrings](https://mkdocstrings.github.io/).

If a breaking feature is developed, then we suggest to update documentation in the `/docs` folder as well, in order to describe how this can be used from a user perspective.

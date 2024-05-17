# Contributing

## Guidelines

We welcome contributions to the library! If you have a bug fix or new feature that you would like to contribute, please follow the steps below:

1. Check the [existing issues](https://github.com/FBruzzesi/timebasedcv/issues){:target="_blank"} and/or [open a new one](https://github.com/FBruzzesi/timebasedcv/issues/new){:target="_blank"} to discuss the problem and potential solutions.
2. [Fork the repository](https://github.com/FBruzzesi/timebasedcv/fork){:target="_blank"} on GitHub.
3. [Clone the repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository){:target="_blank"} to your local machine.
4. Create a new branch for your bug fix or feature.
5. Make your changes and test them thoroughly, making sure that it passes all current tests.
6. Commit your changes and push the branch to your fork.
7. [Open a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request){:target="_blank"} on the main repository.

## Submitting Pull Requests

When submitting a pull request, please make sure that you've followed the steps above and that your code has been thoroughly tested. Also, be sure to include a brief summary of the changes you've made and a reference to any issues that your pull request resolves.

## Code formatting

**timebasedcv** uses [ruff](https://docs.astral.sh/ruff/){:target="_blank"} for both formatting and linting. Specific settings are declared in the [pyproject.toml file](https://github.com/FBruzzesi/timebasedcv/blob/3ddead232c2243c8129f6b599b28e486bdd87b3b/pyproject.toml#L75){:target="_blank"}.

To format the code, you can run the following commands:

=== "with Make"

    ```bash
    make lint
    ```

=== "without Make"

    ```bash
    ruff version
    ruff format timebasedcv tests
    ruff check timebasedcv tests --fix
    ruff clean
    ```

As part of the checks on pull requests, it is checked whether the code follows those standards. To ensure that the standard is met, it is recommended to install [pre-commit hooks](https://pre-commit.com/){:target="_blank"}:

```bash
python -m pip install pre-commit
pre-commit install
```

## Developing

Let's suppose that you already did steps 1-4 from the above list, now you should install the library and its developing dependencies  in editable way.

First move into the repo folder: `cd timebasedcv`.

Then:

=== "with Make"

    ```bash
    make init-dev
    ```

=== "without Make"

    ```bash
    pip install -e ".[all-dev]" --no-cache-dir
    pre-commit install
    ```

Now you are ready to proceed with all the changes you want to!

## Testing

Once you are done with changes, you should:

- add tests for the new features in the `/tests` folder
- make sure that new features do not break existing codebase by running tests:

    === "with Make"

        ```bash
        make test
        ```

    === "without Make"

        ```bash
        pytest tests -n auto
        ```

## Docs

The documentation is generated using [mkdocs-material](https://squidfunk.github.io/mkdocs-material/){:target="_blank"}, the API part uses [mkdocstrings](https://mkdocstrings.github.io/){:target="_blank"}.

If a new feature or a breaking change is developed, then we suggest to update documentation in the `/docs` folder as well, in order to describe how this can be used from a user perspective.

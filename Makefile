ARG := $(word 2, $(MAKECMDGOALS))
$(eval $(ARG):;@:)

sources = timebasedcv tests

clean-folders:
	rm -rf __pycache__ */__pycache__ */**/__pycache__ \
		.pytest_cache */.pytest_cache */**/.pytest_cache \
		.ruff_cache */.ruff_cache */**/.ruff_cache \
		.mypy_cache */.mypy_cache */**/.mypy_cache \
		site build dist htmlcov .coverage .tox

lint:
	uvx ruff version
	uvx ruff format $(sources)
	uvx ruff check $(sources) --fix
	uvx ruff clean
	# uv tool run rumdl check .

test:
	uv run --all-extras --group testing pytest tests --cov=timebasedcv --cov=tests --cov-fail-under=95 -n auto

typing:
	mypy timebasedcv
	pyright timebasedcv

check: lint test typing clean-folders

docs-serve:
	mkdocs serve

docs-deploy:
	mkdocs gh-deploy

pypi-push:
	rm -rf dist
	uv build
	uv publish

get-version :
	@echo $(shell grep -m 1 version pyproject.toml | tr -s ' ' | tr -d '"' | tr -d "'" | cut -d' ' -f3)

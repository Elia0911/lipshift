# On Windows without make, run the commands after each target directly.
.PHONY: install install-extract lint format test test-all subset clean

install:
	pip install -e ".[dev]"
	pre-commit install

install-extract:
	pip install -e ".[extract,dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest -m "not integration"

test-all:
	pytest

subset:
	python -m lipshift.data.subset --meta $(META) --config configs/subset.yaml --out data/manifest.csv

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info

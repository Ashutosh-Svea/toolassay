.PHONY: install lint format typecheck test check demo

PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

lint:
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

format:
	$(BIN)/ruff format src tests
	$(BIN)/ruff check --fix src tests

typecheck:
	$(BIN)/mypy

test:
	$(BIN)/pytest -q

check: lint typecheck test

# Runs the bundled bookshop example twice (v1 and v2 contracts) and diffs the two runs.
# Needs ANTHROPIC_API_KEY in the environment.
demo:
	$(BIN)/toolassay run --server examples/bookshop/server-v1.yaml --cases examples/bookshop/cases.yaml --out before.run.json
	$(BIN)/toolassay run --server examples/bookshop/server-v2.yaml --cases examples/bookshop/cases.yaml --out after.run.json
	$(BIN)/toolassay diff before.run.json after.run.json --max-cost-increase 10%

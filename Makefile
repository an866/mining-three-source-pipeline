# Convenience targets. Every one of these is a thin wrapper over a `python -m ...`
# invocation that also works on its own -- see RUN.md. Nothing here is required.

PYTHON ?= python
DAYS   ?= 30
INDEX  ?= .index

.PHONY: help install ingest ingest-online serve eval probe test test-fast lint fmt type check clean docker

help:            ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:         ## Install the package with dev extras
	$(PYTHON) -m pip install -e ".[dev]"

ingest:          ## Build the index from the deterministic offline corpus
	$(PYTHON) -m pipeline.ingest --days $(DAYS) --index-dir $(INDEX)

ingest-online:   ## Build the index from the live sources (needs network)
	$(PYTHON) -m pipeline.ingest --days $(DAYS) --online --index-dir $(INDEX)

serve:           ## Run the API (127.0.0.1:8000)
	$(PYTHON) -m serve.app

eval:            ## Run the evaluation harness and write eval/report.{md,json}
	$(PYTHON) -m eval.harness

probe:           ## Print retrieval ranks for 16 hard questions
	$(PYTHON) -m scripts.probe_retrieval -k 15

test:            ## Run the full test suite
	$(PYTHON) -m pytest

test-fast:       ## Run the test suite without the slow end-to-end tests
	$(PYTHON) -m pytest -m "not slow"

lint:            ## Ruff check
	$(PYTHON) -m ruff check .

fmt:             ## Ruff format
	$(PYTHON) -m ruff format .

type:            ## mypy --strict
	$(PYTHON) -m mypy

check: lint type test   ## Lint, type-check and test

clean:           ## Remove caches (not the index -- rebuild it with `make ingest`)
	rm -rf .pytest_cache .ruff_cache .mypy_cache

docker:          ## Build and run the container
	docker compose up --build

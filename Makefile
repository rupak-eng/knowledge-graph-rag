.PHONY: setup test lint typecheck ingest serve bench demo clean

VENV=.venv
PY=$(VENV)/bin/python

setup:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -q --upgrade pip
	$(VENV)/bin/pip install -q -e ".[dev]"
	$(VENV)/bin/python -m spacy download en_core_web_sm

test:
	$(PY) -m pytest -q

lint:
	$(VENV)/bin/ruff check src tests eval scripts

typecheck:
	$(VENV)/bin/mypy src

ingest:
	$(PY) -m krag.services.ingest --data-dir data

serve:
	$(PY) -m uvicorn krag.api.main:app --host 0.0.0.0 --port 8000

bench:
	$(PY) -m eval.run_benchmark --data-dir data --out bench/results/benchmark.json

demo:
	$(PY) scripts/demo_queries.py --data-dir data --out demo/demo_run.json

clean:
	rm -rf data/processed __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find src tests eval scripts -name "__pycache__" -type d -exec rm -rf {} +

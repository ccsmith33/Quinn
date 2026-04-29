PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: help venv install test lint typecheck format clean verify-schema restore-from-b2 backup-now verify-remote

help:
	@echo "Targets: venv install test lint typecheck format clean"
	@echo "         verify-schema restore-from-b2 backup-now"
	@echo "         verify-remote"

venv:
	python3 -m venv .venv

install:
	$(PIP) install -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests

typecheck:
	$(PY) -m mypy src

format:
	$(PY) -m ruff format src tests

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# --- S8.3 deploy / rehydration helpers ---

# Verify the migration set on disk matches the schema version recorded
# in /var/lib/quinn/journal.db. Used by the rehydration runbook step 6.
verify-schema:
	$(PY) -m journal.migrate --verify

# Download the most-recent B2 backup and write it to the journal path.
# Used by the rehydration runbook step 6.
restore-from-b2:
	$(PY) -m jobs.restore_from_b2 --target /var/lib/quinn/journal.db

# Trigger an out-of-band backup right now (e.g., as a smoke test
# during rehydration step 10).
backup-now:
	$(PY) -m jobs.backup

# --- S9.2 git remote helpers ---

# Verify the configured `origin` remote is reachable. Run after
# `git remote add origin <url>` to confirm push/fetch will succeed.
verify-remote:
	bash scripts/verify_remote.sh

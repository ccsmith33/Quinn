"""AC-6: smoke test that every src/<package>/__init__.py imports cleanly."""

import importlib

import pytest

EXPECTED_PACKAGES = [
    "ingestion",
    "prefilter",
    "analyzer",
    "proposal",
    "execution",
    "broker",
    "reconciler",
    "killswitch",
    "journal",
    "universe",
    "observability",
    "prompts",
    "config",
    "jobs",
]


@pytest.mark.parametrize("package", EXPECTED_PACKAGES)
def test_imports(package: str) -> None:
    importlib.import_module(package)

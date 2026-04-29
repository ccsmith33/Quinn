"""S9.2 — `.gitignore` invariants (AC-3, D-064).

Asserts that every path the rehydration runbook treats as "transient,
operator-local, or contains-secrets" is excluded from the GitHub remote.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GITIGNORE = REPO / ".gitignore"


def _ignored() -> str:
    return GITIGNORE.read_text(encoding="utf-8")


def test_gitignore_exists() -> None:
    assert GITIGNORE.is_file(), f"missing {GITIGNORE}"


def test_secrets_envelope_is_ignored() -> None:
    text = _ignored()
    assert ".env" in text


def test_journal_db_files_ignored() -> None:
    """SQLite journal + WAL + SHM files must not reach the remote."""
    text = _ignored()
    for pattern in ("*.db", "*.db-wal", "*.db-shm"):
        assert pattern in text, f"missing {pattern} in .gitignore"


def test_python_caches_ignored() -> None:
    text = _ignored()
    for pattern in ("__pycache__", ".venv", "*.egg-info", "dist", "build"):
        assert pattern in text, f"missing {pattern} in .gitignore"


def test_tooling_caches_ignored() -> None:
    text = _ignored()
    for pattern in (".pytest_cache", ".mypy_cache", ".ruff_cache"):
        assert pattern in text, f"missing {pattern} in .gitignore"


def test_artifacts_runtime_ignored() -> None:
    """Either `artifacts/` (broad — bmad-swarm convention) or a more
    targeted `artifacts/runtime/` line covers AC-3.
    """
    text = _ignored()
    assert "artifacts/" in text or "artifacts/runtime/" in text


def test_env_example_is_not_ignored() -> None:
    """`.env.example` is the public template; it MUST stay tracked."""
    text = _ignored()
    # `.env` line is glob-narrow (no slash, no `.example`); the file is
    # still tracked. Probe both: the literal `.env.example` is never
    # listed AND the on-disk file exists.
    assert ".env.example" not in text
    assert (REPO / ".env.example").is_file()


def test_quinn_egg_info_specifically_ignored() -> None:
    """The story's AC-3 list calls out `quinn.egg-info` by name; the
    `*.egg-info/` glob covers it.
    """
    text = _ignored()
    assert "*.egg-info" in text or "quinn.egg-info" in text

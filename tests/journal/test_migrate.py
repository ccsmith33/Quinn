"""S1.2 — SQLite migrations runner tests.

Architecture references: §8.1 (tables/indexes), §2.9 (schema-mismatch failure mode),
D-023 (numbered SQL files, no Alembic).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from journal.migrate import SchemaMismatch, apply_migrations, verify_schema

EXPECTED_TABLES = {
    "filings",
    "prefilter_decisions",
    "similarity_cache",
    "prompts",
    "proposals",
    "proposal_reviews",
    "executions",
    "orders",
    "positions",
    "account_snapshots",
    "kill_switch_state",
    "universe_snapshots",
    "universe_members",
    "llm_calls",
    "backups",
    "meta",
}

EXPECTED_INDEXES = {
    "idx_filings_cik_form",
    "idx_filings_filed_at",
    "idx_simcache_cik_form",
    "idx_proposals_filing",
    "idx_proposals_symbol_created",
    "idx_orders_symbol_submitted",
    "idx_positions_symbol_snap",
    "idx_account_snap_at",
    "idx_um_cik",
    "idx_um_ticker_snap",
    "idx_llm_calls_decision",
    "idx_llm_calls_called_at",
}


def _table_names(db_path: str) -> set[str]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    return {r[0] for r in rows}


def _index_names(db_path: str) -> set[str]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    return {r[0] for r in rows}


def test_fresh_db_has_all_tables(tmp_path: Path) -> None:
    db = tmp_path / "journal.db"
    version = apply_migrations(str(db))
    assert version >= 1
    assert EXPECTED_TABLES.issubset(_table_names(str(db)))


def test_indexes_present(tmp_path: Path) -> None:
    db = tmp_path / "journal.db"
    apply_migrations(str(db))
    assert EXPECTED_INDEXES.issubset(_index_names(str(db)))


def test_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "journal.db"
    v1 = apply_migrations(str(db))
    v2 = apply_migrations(str(db))
    assert v1 == v2
    with sqlite3.connect(str(db)) as conn:
        rows = conn.execute("SELECT COUNT(*) FROM meta").fetchone()
    assert rows[0] == v1


def test_wal_mode_enabled(tmp_path: Path) -> None:
    db = tmp_path / "journal.db"
    apply_migrations(str(db))
    with sqlite3.connect(str(db)) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_verify_schema_ok_after_apply(tmp_path: Path) -> None:
    db = tmp_path / "journal.db"
    apply_migrations(str(db))
    verify_schema(str(db))  # no raise


def test_schema_mismatch_when_meta_ahead(tmp_path: Path) -> None:
    """meta records version N+1 but only N files on disk → SchemaMismatch."""
    db = tmp_path / "journal.db"
    apply_migrations(str(db))
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            "INSERT INTO meta(schema_version, applied_at) VALUES (?, CURRENT_TIMESTAMP)",
            (999,),
        )
        conn.commit()
    with pytest.raises(SchemaMismatch):
        verify_schema(str(db))


def test_schema_mismatch_when_meta_behind(tmp_path: Path) -> None:
    """meta missing a version that has a file on disk → SchemaMismatch."""
    db = tmp_path / "journal.db"
    apply_migrations(str(db))
    # delete the recorded version row to simulate missing application
    with sqlite3.connect(str(db)) as conn:
        conn.execute("DELETE FROM meta")
        conn.commit()
    with pytest.raises(SchemaMismatch):
        verify_schema(str(db))

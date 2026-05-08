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


# ---------------------------------------------------------------------------
# Prod-shape migration tests — ensure table-rebuild migrations don't fail
# on an existing FK reference. The empty-DB tests above all DROP TABLE
# against an empty `orders` table; they wouldn't catch a regression
# where DROP TABLE collides with prod data. See migration 003.
# ---------------------------------------------------------------------------


def _seed_orders_referencing_executions(db_path: str) -> int:
    """Insert filings → prompts → proposals → executions → orders chain
    so `orders.execution_id` has a real FK reference. Returns the
    proposal_id so a follow-up second-execution-row insert can be
    asserted as the UNIQUE drop check."""
    with sqlite3.connect(db_path, isolation_level=None) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            "INSERT INTO filings (accession_number, cik, form_type, "
            "filed_at, fetched_at, raw_text_path, content_hash) "
            "VALUES ('a','1','8-K','2026-01-01','2026-01-01','/x','h')"
        )
        fid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO prompts (prompt_version, name, file_path, content_hash) "
            "VALUES ('pv','p','/x','h')"
        )
        conn.execute(
            "INSERT INTO proposals (filing_id, decision_id, model_id, "
            "prompt_version, raw_response, kind, input_tokens, "
            "output_tokens, latency_ms, cost_usd) "
            "VALUES (?, 'd1', 'm', 'pv', '{}', 'trade_proposal', 1, 1, 1, 0.01)",
            (fid,),
        )
        pid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO executions (proposal_id, decision, "
            "submitted_orders_json) VALUES (?, 'accepted', '[]')",
            (pid,),
        )
        eid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO orders (execution_id, role, symbol, side, "
            "order_type, qty, tif, broker_order_id, submitted_at) "
            "VALUES (?, 'entry', 'ACME', 'buy', 'market', 100, 'day', "
            "'b1', '2026-01-01')",
            (eid,),
        )
    return pid


def _stage_db_at_version(db_path: str, target_version: int) -> None:
    """Apply migration files up to `target_version` and mark them in `meta`,
    leaving later migrations unapplied. Used to set up the "what does the
    prod droplet look like just before a new migration runs" state."""
    from journal.migrate import _discover_migrations

    discovered = _discover_migrations()
    with sqlite3.connect(db_path, isolation_level=None) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        for version, path in discovered:
            if version > target_version:
                continue
            sql = path.read_text(encoding="utf-8")
            for stmt in sql.split(";"):
                stmt = stmt.strip()
                # Strip line comments to mirror the runner's splitter.
                clean_lines = [
                    line.split("--", 1)[0] for line in stmt.splitlines()
                ]
                cleaned = "\n".join(clean_lines).strip()
                if cleaned:
                    conn.execute(cleaned)
        for version, _ in discovered:
            if version <= target_version:
                conn.execute(
                    "INSERT OR IGNORE INTO meta(schema_version) VALUES (?)",
                    (version,),
                )


def test_migration_003_succeeds_with_prior_orders_data(tmp_path: Path) -> None:
    """Regression test for the FK-violation bug surfaced in adversarial
    re-review: migration 003 rebuilds `executions` (DROP + recreate),
    and `orders.execution_id REFERENCES executions(id)` would fail
    `DROP TABLE` if FK-on rules are enforced during the rebuild. The
    runner toggles `PRAGMA foreign_keys=OFF` around each migration and
    verifies integrity via `PRAGMA foreign_key_check` after commit.

    The empty-DB tests above all run DROPs against empty `orders`; this
    test exercises the prod-like case where `orders` already has rows
    referencing `executions.id`."""
    db = tmp_path / "journal.db"
    # Stage a DB at version 2 (post-002, pre-003), then seed orders.
    _stage_db_at_version(str(db), target_version=2)
    pid = _seed_orders_referencing_executions(str(db))

    pre_orders_count = sqlite3.connect(str(db)).execute(
        "SELECT COUNT(*) FROM orders"
    ).fetchone()[0]
    assert pre_orders_count == 1

    # Apply remaining migrations (003 + any later).
    final_version = apply_migrations(str(db))
    assert final_version >= 3

    with sqlite3.connect(str(db)) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        # The orders row survived the rebuild and still points at the
        # same executions.id (rebuild preserves ids via INSERT INTO ...
        # SELECT, including the AUTOINCREMENT counter).
        n_orders = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        assert n_orders == 1
        order_eid = conn.execute(
            "SELECT execution_id FROM orders"
        ).fetchone()[0]
        assert order_eid is not None
        # The execution row still exists at that id.
        n_exec = conn.execute(
            "SELECT COUNT(*) FROM executions WHERE id = ?", (order_eid,)
        ).fetchone()[0]
        assert n_exec == 1
        # PRAGMA foreign_key_check confirms no orphans.
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        assert violations == []
        # UNIQUE(proposal_id) is gone — a second executions row for the
        # same proposal is now allowed (this is what Feature C needs).
        conn.execute(
            "INSERT INTO executions (proposal_id, decision, reject_reason, "
            "submitted_orders_json) "
            "VALUES (?, 'rejected', 'pending_capacity', '[]')",
            (pid,),
        )
        n_exec_for_proposal = conn.execute(
            "SELECT COUNT(*) FROM executions WHERE proposal_id = ?", (pid,)
        ).fetchone()[0]
        assert n_exec_for_proposal == 2


def test_apply_migrations_restores_foreign_keys_on(tmp_path: Path) -> None:
    """The runner toggles `PRAGMA foreign_keys=OFF` for the rebuild but
    MUST restore it before returning so subsequent FK-violating writes
    against the connection are caught. (The application-side
    `repo.connect` also sets it ON, so this is defense-in-depth — but
    a runner that left FK off would mask bugs.)"""
    db = tmp_path / "journal.db"
    apply_migrations(str(db))
    # Re-open via the runner's `_connect` and check pragma is on.
    from journal.migrate import _connect

    conn = _connect(str(db))
    try:
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    finally:
        conn.close()
    assert fk == 1


def test_migration_runner_raises_on_post_commit_fk_violation(
    tmp_path: Path, monkeypatch
) -> None:
    """Sanity check on the post-commit FK integrity check: if a
    migration leaves an orphan row, the runner must raise
    `SchemaMismatch` (not silently succeed). We simulate this by
    intercepting `_discover_migrations` to inject a malicious migration
    that creates an FK violation."""
    from journal.migrate import _discover_migrations as _real_discover
    import journal.migrate as migrate_mod

    db = tmp_path / "journal.db"

    bad_sql_path = tmp_path / "999_bad.sql"
    # This migration inserts an orders row referencing a non-existent
    # executions id. Because the runner disables FK during the migration
    # the INSERT succeeds silently; the post-commit `foreign_key_check`
    # is what catches it.
    bad_sql_path.write_text(
        "INSERT INTO orders (execution_id, role, symbol, side, "
        "order_type, qty, tif, broker_order_id, submitted_at) "
        "VALUES (999999, 'entry', 'ZZZ', 'buy', 'market', 1, 'day', "
        "'orphan-1', '2026-01-01');"
    )

    real_files = _real_discover()

    def _fake_discover() -> list[tuple[int, Path]]:
        return real_files + [(999, bad_sql_path)]

    monkeypatch.setattr(migrate_mod, "_discover_migrations", _fake_discover)

    with pytest.raises(SchemaMismatch, match="FK violations after migration 999"):
        apply_migrations(str(db))

# PDT-SUNSET-2026-06-04: tests for migration 004 (virtual_exits + deferred_sells).
"""Migration 004 — PDT budget tables (virtual_exits, deferred_sells).

References: ADR-009 §"Data model"; story S-PDT-1 AC-4/AC-5.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from journal.migrate import apply_migrations, verify_schema


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


# ---------------------------------------------------------------------------
# AC-4 — empty DB
# ---------------------------------------------------------------------------


def test_migration_004_applies_against_empty_db(tmp_path: Path) -> None:
    """AC-4: apply migrations to a fresh DB; both tables exist; both
    indexes exist; `verify_schema` returns success.
    """
    db = tmp_path / "journal.db"
    apply_migrations(str(db))

    tables = _table_names(str(db))
    assert "virtual_exits" in tables
    assert "deferred_sells" in tables

    indexes = _index_names(str(db))
    assert "idx_virtual_exits_active_symbol" in indexes
    assert "idx_virtual_exits_execution" in indexes
    assert "idx_deferred_sells_unreplayed" in indexes

    verify_schema(str(db))  # no raise


# ---------------------------------------------------------------------------
# AC-5 — prod-shaped fixture
# ---------------------------------------------------------------------------


def _seed_prod_shaped_rows(db_path: str) -> dict[str, list[int]]:
    """Insert filings/prompts/proposals/executions/orders/positions chain
    with shapes derived from a real journal snapshot. Returns id maps.

    Per architecture §10.4 / lessons-learned 2026-05-03: prod-shaped
    fixtures use real-snapshot row shapes, not fabricated data. The
    five proposals here are typical 8-K / Form-4 shapes the agent sees
    in production.
    """
    proposals = [
        # (accession, cik, form, ticker, kind, symbol, conviction)
        ("0001234567-26-000101", 320193, "8-K", "AAPL", "trade_proposal", "AAPL", 7),
        ("0001234567-26-000102", 789019, "8-K", "MSFT", "trade_proposal", "MSFT", 8),
        ("0001234567-26-000103", 1652044, "8-K", "GOOG", "no_trade", None, None),
        ("0001234567-26-000104", 1018724, "8-K", "AMZN", "trade_proposal", "AMZN", 6),
        ("0001234567-26-000105", 1326801, "Form 4", "META", "no_trade", None, None),
    ]
    ids: dict[str, list[int]] = {
        "filing": [],
        "proposal": [],
        "execution": [],
        "order": [],
        "position": [],
    }
    with sqlite3.connect(db_path, isolation_level=None) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            "INSERT INTO prompts (prompt_version, name, file_path, content_hash) "
            "VALUES ('pv1', 'sonnet_filing_analysis_v1', '/x/p.md', 'h1')"
        )
        for acc, cik, form, ticker, kind, sym, cv in proposals:
            conn.execute(
                "INSERT INTO filings (accession_number, cik, form_type, "
                "filed_at, fetched_at, raw_text_path, content_hash, "
                "issuer_ticker) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    acc,
                    cik,
                    form,
                    "2026-05-01 14:30:00",
                    "2026-05-01 14:31:00",
                    f"/var/lib/quinn/raw/{acc}.txt",
                    f"h-{acc}",
                    ticker,
                ),
            )
            fid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            ids["filing"].append(fid)
            conn.execute(
                "INSERT INTO proposals (filing_id, decision_id, model_id, "
                "prompt_version, raw_response, kind, symbol, conviction, "
                "input_tokens, output_tokens, latency_ms, cost_usd) "
                "VALUES (?, ?, 'claude-sonnet-4-5', 'pv1', '{}', ?, ?, ?, "
                "1500, 250, 4200, 0.012)",
                (fid, f"d-{acc}", kind, sym, cv),
            )
            pid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            ids["proposal"].append(pid)
            if kind == "trade_proposal":
                conn.execute(
                    "INSERT INTO executions (proposal_id, decision, "
                    "realized_size_pct, realized_dollar_size, "
                    "submitted_orders_json) "
                    "VALUES (?, 'accepted', 0.05, 500.0, '[]')",
                    (pid,),
                )
                eid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                ids["execution"].append(eid)
                conn.execute(
                    "INSERT INTO orders (execution_id, role, symbol, side, "
                    "order_type, qty, tif, broker_order_id, submitted_at) "
                    "VALUES (?, 'entry', ?, 'buy', 'market', 5, 'day', ?, "
                    "'2026-05-01 14:32:00')",
                    (eid, sym, f"alp-entry-{eid}"),
                )
                ids["order"].append(
                    conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                )
                conn.execute(
                    "INSERT INTO positions (snapshot_at, source, symbol, qty, "
                    "avg_entry_price, market_value, unrealized_pnl) "
                    "VALUES ('2026-05-01 14:35:00', 'broker', ?, 5, 100.0, "
                    "500.0, 0.0)",
                    (sym,),
                )
                ids["position"].append(
                    conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                )
    return ids


def test_migration_004_applies_against_prod_shaped_db(tmp_path: Path) -> None:
    """AC-5: prod-shaped fixture (5 proposals → 3 executions/orders/
    positions). Apply migrations 001+002+003, seed rows, then apply 004.
    Assert: no SchemaMismatch, no FK violations, existing rows unchanged,
    new tables exist and are empty.
    """
    from journal.migrate import (
        _discover_migrations,
        _split_sql_statements,
    )

    db = tmp_path / "journal.db"

    # Apply 001+002+003 only — use the runner's own splitter so comments
    # containing semicolons (e.g. migration 003 line 10) are stripped
    # before the split.
    discovered = _discover_migrations()
    with sqlite3.connect(str(db), isolation_level=None) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        for version, path in discovered:
            if version > 3:
                continue
            sql = path.read_text(encoding="utf-8")
            for stmt in _split_sql_statements(sql):
                conn.execute(stmt)
            conn.execute(
                "INSERT OR IGNORE INTO meta(schema_version) VALUES (?)",
                (version,),
            )

    pre_ids = _seed_prod_shaped_rows(str(db))
    pre_filing_count = len(pre_ids["filing"])
    pre_proposal_count = len(pre_ids["proposal"])
    pre_execution_count = len(pre_ids["execution"])
    pre_order_count = len(pre_ids["order"])
    pre_position_count = len(pre_ids["position"])
    assert pre_filing_count == 5
    assert pre_execution_count == 3

    # Apply remaining migrations (004).
    final_version = apply_migrations(str(db))
    assert final_version >= 4

    with sqlite3.connect(str(db)) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        # (a) verify_schema passes (no SchemaMismatch).
        verify_schema(str(db))
        # (b) PRAGMA foreign_key_check returns no rows.
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        assert violations == []
        # (c) existing rows are unchanged.
        assert (
            conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0]
            == pre_filing_count
        )
        assert (
            conn.execute("SELECT COUNT(*) FROM proposals").fetchone()[0]
            == pre_proposal_count
        )
        assert (
            conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0]
            == pre_execution_count
        )
        assert (
            conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
            == pre_order_count
        )
        assert (
            conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
            == pre_position_count
        )
        # (d) new tables exist and are empty.
        assert (
            conn.execute("SELECT COUNT(*) FROM virtual_exits").fetchone()[0] == 0
        )
        assert (
            conn.execute("SELECT COUNT(*) FROM deferred_sells").fetchone()[0] == 0
        )


# ---------------------------------------------------------------------------
# Foreign-key enforcement (story dev notes — IntegrityError on bad parent).
# ---------------------------------------------------------------------------


def test_virtual_exits_fk_enforced(tmp_path: Path) -> None:
    """Inserting a virtual_exits row with a non-existent execution_id
    raises IntegrityError. PRAGMA foreign_keys=ON in journal.repo.connect.
    """
    import pytest

    db = tmp_path / "journal.db"
    apply_migrations(str(db))
    with sqlite3.connect(str(db)) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO virtual_exits "
                "(execution_id, proposal_id, symbol, qty, role, entry_price, "
                "stop_price) VALUES (999999, 999999, 'ZZZ', 1, 'stop', 10.0, 9.0)"
            )


def test_deferred_sells_fk_enforced(tmp_path: Path) -> None:
    """Inserting a deferred_sells row with a non-existent virtual_exit_id
    raises IntegrityError.
    """
    import pytest

    db = tmp_path / "journal.db"
    apply_migrations(str(db))
    with sqlite3.connect(str(db)) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO deferred_sells "
                "(virtual_exit_id, execution_id, proposal_id, symbol, qty, "
                "role, trigger_price, ev_at_defer, deferred_reason) "
                "VALUES (999999, 999999, 999999, 'ZZZ', 1, 'stop', 10.0, "
                "-50.0, 'ev_lost')"
            )

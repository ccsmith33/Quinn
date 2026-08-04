"""Migration 008 (watchlist) + repo API tests.

Pins: table/index creation, oldest-first pending ordering, the
one-pending-row-per-symbol partial unique index, and the resolve
lifecycle (terminal statuses only, pending-only transition, benign
double-resolve).
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

import pytest

from journal.migrate import apply_migrations, verify_schema
from journal.models import FilingRow, PromptRow, ProposalRow, WatchlistRow
from journal.repo import (
    WatchlistDuplicatePending,
    connect,
    get_pending_watchlist,
    has_pending_watchlist_for_symbol,
    insert_filing,
    insert_prompt,
    insert_proposal,
    insert_watchlist_row,
    resolve_watchlist_row,
)


@pytest.fixture
def db(tmp_path: Path) -> str:
    p = tmp_path / "journal.db"
    apply_migrations(str(p))
    return str(p)


def _seed_proposal(db: str, *, symbol: str = "ACME", tag: str = "wl-1") -> int:
    """Minimal filings → prompts → proposals chain so the watchlist FK holds."""
    fid = insert_filing(
        db,
        FilingRow(
            accession_number=f"acc-{tag}",
            cik=320193,
            form_type="8-K",
            filed_at=dt.datetime(2026, 7, 20, 13, 0, tzinfo=dt.UTC),
            fetched_at=dt.datetime(2026, 7, 20, 13, 1, tzinfo=dt.UTC),
            raw_text_path=f"/tmp/{tag}.txt",
            content_hash=f"hash-{tag}",
        ),
    )
    pv = f"sonnet@{tag}"
    insert_prompt(
        db,
        PromptRow(
            prompt_version=pv,
            name="sonnet",
            file_path="src/prompts/sonnet.txt",
            content_hash="x" * 64,
        ),
    )
    return insert_proposal(
        db,
        ProposalRow(
            filing_id=fid,
            decision_id=f"dec-{tag}",
            model_id="claude-sonnet-4-6",
            prompt_version=pv,
            raw_response="{}",
            kind="trade_proposal",
            symbol=symbol,
            direction="long",
            size_pct_requested=0.10,
            conviction=8,
            thesis="t",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
            cost_usd=0.0,
        ),
    )


def _row(pid: int, *, symbol: str = "ACME", ref: float = 100.0) -> WatchlistRow:
    return WatchlistRow(
        proposal_id=pid,
        symbol=symbol,
        conviction=8,
        reference_price=ref,
        expires_at=dt.datetime(2026, 7, 23, 20, 0, tzinfo=dt.UTC),
        notes="reject_reason=ks7_cash_reserve",
    )


# ---------------------------------------------------------------------------
# migration
# ---------------------------------------------------------------------------


def test_migration_008_creates_watchlist_table_and_indexes(db: str) -> None:
    with sqlite3.connect(db) as conn:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        indexes = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
    assert "watchlist" in tables
    assert "idx_watchlist_status_created" in indexes
    assert "idx_watchlist_pending_symbol" in indexes
    verify_schema(db)  # no raise


# ---------------------------------------------------------------------------
# insert / get / dedup
# ---------------------------------------------------------------------------


def test_insert_and_get_pending_roundtrip(db: str) -> None:
    pid = _seed_proposal(db)
    wid = insert_watchlist_row(db, _row(pid))
    assert wid > 0
    rows = get_pending_watchlist(db)
    assert len(rows) == 1
    got = rows[0]
    assert got.id == wid
    assert got.proposal_id == pid
    assert got.symbol == "ACME"
    assert got.conviction == 8
    assert got.reference_price == 100.0
    assert got.status == "pending"
    assert got.resolved_at is None
    assert got.created_at is not None
    assert got.notes == "reject_reason=ks7_cash_reserve"


def test_get_pending_is_oldest_first(db: str) -> None:
    p1 = _seed_proposal(db, symbol="ACME", tag="wl-a")
    p2 = _seed_proposal(db, symbol="WIDG", tag="wl-b")
    insert_watchlist_row(db, _row(p1, symbol="ACME"))
    insert_watchlist_row(db, _row(p2, symbol="WIDG"))
    rows = get_pending_watchlist(db)
    assert [r.symbol for r in rows] == ["ACME", "WIDG"]


def test_one_pending_row_per_symbol_enforced(db: str) -> None:
    p1 = _seed_proposal(db, tag="wl-a")
    p2 = _seed_proposal(db, tag="wl-b")
    insert_watchlist_row(db, _row(p1))
    with pytest.raises(WatchlistDuplicatePending):
        insert_watchlist_row(db, _row(p2))
    assert has_pending_watchlist_for_symbol(db, "ACME") is True
    assert has_pending_watchlist_for_symbol(db, "WIDG") is False


def test_resolved_row_does_not_block_reenrollment(db: str) -> None:
    """The unique index is PARTIAL (status='pending'): once a row is
    resolved, the same symbol may re-enroll on a new filing."""
    p1 = _seed_proposal(db, tag="wl-a")
    p2 = _seed_proposal(db, tag="wl-b")
    wid = insert_watchlist_row(db, _row(p1))
    assert resolve_watchlist_row(db, wid, status="expired") is True
    wid2 = insert_watchlist_row(db, _row(p2))
    assert wid2 != wid
    assert has_pending_watchlist_for_symbol(db, "ACME") is True


# ---------------------------------------------------------------------------
# resolve lifecycle
# ---------------------------------------------------------------------------


def test_resolve_stamps_status_resolved_at_and_notes(db: str) -> None:
    pid = _seed_proposal(db)
    wid = insert_watchlist_row(db, _row(pid))
    assert (
        resolve_watchlist_row(
            db, wid, status="skipped_chase", notes="last=110 ref=100"
        )
        is True
    )
    with connect(db) as conn:
        r = conn.execute("SELECT * FROM watchlist WHERE id = ?", (wid,)).fetchone()
    assert r["status"] == "skipped_chase"
    assert r["resolved_at"] is not None
    assert r["notes"] == "last=110 ref=100"
    assert get_pending_watchlist(db) == []


def test_resolve_without_notes_keeps_existing_notes(db: str) -> None:
    pid = _seed_proposal(db)
    wid = insert_watchlist_row(db, _row(pid))
    resolve_watchlist_row(db, wid, status="skipped_held")
    with connect(db) as conn:
        r = conn.execute("SELECT notes FROM watchlist WHERE id = ?", (wid,)).fetchone()
    assert r["notes"] == "reject_reason=ks7_cash_reserve"


def test_resolve_is_pending_only_and_double_resolve_is_noop(db: str) -> None:
    pid = _seed_proposal(db)
    wid = insert_watchlist_row(db, _row(pid))
    assert resolve_watchlist_row(db, wid, status="entered") is True
    # Second transition loses: status stays 'entered'.
    assert resolve_watchlist_row(db, wid, status="expired") is False
    with connect(db) as conn:
        r = conn.execute("SELECT status FROM watchlist WHERE id = ?", (wid,)).fetchone()
    assert r["status"] == "entered"


def test_resolve_rejects_non_terminal_status(db: str) -> None:
    pid = _seed_proposal(db)
    wid = insert_watchlist_row(db, _row(pid))
    with pytest.raises(ValueError):
        resolve_watchlist_row(db, wid, status="pending")
    with pytest.raises(ValueError):
        resolve_watchlist_row(db, wid, status="bogus")

"""WS1 position-truth contract — journal layer (D-078, delta §2.1/§2.5/§7.4).

Covers migration 005 plus the four new repo functions:
  - get_orders_pending_fill()        — final_status IS NULL
  - record_order_outcome(...)        — one-time NULL→value transition
  - insert_position_tombstone(...)   — qty=0 positions row
  - get_live_protective_order(...)   — pending order lookup per execution
and the lifecycle classifier query get_lifecycle_orders_for_symbol(...).
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

import pytest

from journal.migrate import apply_migrations
from journal.models import OrderRow, PositionRow
from journal.repo import (
    JournalRepo,
    OrderOutcomeConflict,
    get_lifecycle_orders_for_symbol,
    get_live_protective_order,
    get_orders_pending_fill,
    has_open_position,
    insert_order,
    insert_position,
    insert_position_tombstone,
    record_order_outcome,
)


@pytest.fixture
def db(tmp_path: Path) -> str:
    db_path = tmp_path / "journal.db"
    apply_migrations(str(db_path))
    return str(db_path)


def _seed_proposal_and_execution(db_path: str, symbol: str = "AAPL") -> tuple[int, int]:
    """Create the FK chain prompts → filings → proposals → executions.
    Returns (proposal_id, execution_id).
    """
    with sqlite3.connect(db_path, isolation_level=None) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            "INSERT OR IGNORE INTO prompts (prompt_version, name, file_path, "
            "content_hash) VALUES ('pv1','sonnet_filing_analysis_v1','/x/p.md','h1')"
        )
        conn.execute(
            "INSERT INTO filings (accession_number, cik, form_type, filed_at, "
            "fetched_at, raw_text_path, content_hash, issuer_ticker) "
            "VALUES (?, 1, '8-K', '2026-06-01 14:30:00', '2026-06-01 14:31:00', "
            "'/var/lib/quinn/raw/x.txt', 'h', ?)",
            (f"acc-{symbol}-{dt.datetime.now().timestamp()}", symbol),
        )
        fid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO proposals (filing_id, decision_id, model_id, "
            "prompt_version, raw_response, kind, symbol, conviction, "
            "input_tokens, output_tokens, latency_ms, cost_usd) "
            "VALUES (?, ?, 'm', 'pv1', '{}', 'trade_proposal', ?, 7, 1, 1, 1, 0.01)",
            (fid, f"d-{symbol}-{dt.datetime.now().timestamp()}", symbol),
        )
        pid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO executions (proposal_id, decision, "
            "submitted_orders_json) VALUES (?, 'accepted', '[]')",
            (pid,),
        )
        eid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    return pid, eid


def _order(
    eid: int,
    *,
    symbol: str = "AAPL",
    role: str = "entry",
    side: str = "buy",
    qty: int = 10,
    broker_order_id: str = "bid-1",
    submitted_at: dt.datetime | None = None,
    final_status: str | None = None,
    realized_fill_at: dt.datetime | None = None,
    realized_fill_qty: int | None = None,
    realized_fill_price: float | None = None,
) -> OrderRow:
    return OrderRow(
        execution_id=eid,
        role=role,
        symbol=symbol,
        side=side,
        order_type="market",
        qty=qty,
        tif="gtc",
        broker_order_id=broker_order_id,
        submitted_at=submitted_at or dt.datetime(2026, 6, 8, 14, 30, tzinfo=dt.UTC),
        final_status=final_status,
        realized_fill_at=realized_fill_at,
        realized_fill_qty=realized_fill_qty,
        realized_fill_price=realized_fill_price,
    )


# ---------------------------------------------------------------------------
# Migration 005 — indexes
# ---------------------------------------------------------------------------

def test_migration_005_creates_lifecycle_indexes(db: str) -> None:
    with sqlite3.connect(db) as conn:
        names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
    assert "idx_orders_pending_fill" in names
    assert "idx_positions_symbol_latest" in names


# ---------------------------------------------------------------------------
# get_orders_pending_fill
# ---------------------------------------------------------------------------

def test_get_orders_pending_fill_returns_null_status_rows_only(db: str) -> None:
    _, eid = _seed_proposal_and_execution(db)
    pending_id = insert_order(db, _order(eid, broker_order_id="b-pending"))
    insert_order(
        db,
        _order(eid, broker_order_id="b-done", final_status="filled"),
    )

    rows = get_orders_pending_fill(db)

    assert [r.id for r in rows] == [pending_id]


def test_get_orders_pending_fill_empty_when_all_terminal(db: str) -> None:
    _, eid = _seed_proposal_and_execution(db)
    insert_order(db, _order(eid, broker_order_id="b-1", final_status="canceled"))

    assert get_orders_pending_fill(db) == []


# ---------------------------------------------------------------------------
# record_order_outcome — one-time NULL→value transition
# ---------------------------------------------------------------------------

def test_record_order_outcome_populates_fill_columns(db: str) -> None:
    _, eid = _seed_proposal_and_execution(db)
    oid = insert_order(db, _order(eid, broker_order_id="b-1"))
    fill_at = dt.datetime(2026, 6, 9, 13, 31, tzinfo=dt.UTC)

    record_order_outcome(
        db, oid, "filled", fill_price=10.5, fill_qty=10, fill_at=fill_at
    )

    rows = get_orders_pending_fill(db)
    assert rows == []
    repo = JournalRepo(db)
    [row] = repo.get_orders_since(symbol="AAPL", since=dt.datetime(2026, 1, 1))
    assert row.final_status == "filled"
    assert row.realized_fill_price == 10.5
    assert row.realized_fill_qty == 10
    assert row.realized_fill_at is not None


def test_record_order_outcome_idempotent_on_identical_recall(db: str) -> None:
    _, eid = _seed_proposal_and_execution(db)
    oid = insert_order(db, _order(eid, broker_order_id="b-1"))
    fill_at = dt.datetime(2026, 6, 9, 13, 31, tzinfo=dt.UTC)

    record_order_outcome(db, oid, "filled", fill_price=10.5, fill_qty=10, fill_at=fill_at)
    # Identical re-call is a no-op, not an error.
    record_order_outcome(db, oid, "filled", fill_price=10.5, fill_qty=10, fill_at=fill_at)


def test_record_order_outcome_raises_on_conflicting_recall(db: str) -> None:
    _, eid = _seed_proposal_and_execution(db)
    oid = insert_order(db, _order(eid, broker_order_id="b-1"))
    fill_at = dt.datetime(2026, 6, 9, 13, 31, tzinfo=dt.UTC)

    record_order_outcome(db, oid, "filled", fill_price=10.5, fill_qty=10, fill_at=fill_at)
    with pytest.raises(OrderOutcomeConflict):
        record_order_outcome(
            db, oid, "filled", fill_price=11.0, fill_qty=10, fill_at=fill_at
        )


def test_record_order_outcome_raises_on_unknown_order(db: str) -> None:
    with pytest.raises(OrderOutcomeConflict):
        record_order_outcome(db, 999, "filled", fill_price=1.0, fill_qty=1, fill_at=None)


def test_record_order_outcome_terminal_without_fill(db: str) -> None:
    """canceled/expired/rejected/replaced legs carry no fill values."""
    _, eid = _seed_proposal_and_execution(db)
    oid = insert_order(db, _order(eid, broker_order_id="b-1"))

    record_order_outcome(db, oid, "canceled", fill_price=None, fill_qty=None, fill_at=None)

    assert get_orders_pending_fill(db) == []


# ---------------------------------------------------------------------------
# insert_position_tombstone — RC-1 fix
# ---------------------------------------------------------------------------

def test_tombstone_drops_symbol_from_open_positions(db: str) -> None:
    repo = JournalRepo(db)
    insert_position(
        db,
        PositionRow(
            snapshot_at=dt.datetime(2026, 6, 8, 14, 30, tzinfo=dt.UTC),
            source="reconciler",
            symbol="ACET",
            qty=30,
            avg_entry_price=8.20,
            market_value=246.0,
            unrealized_pnl=0.0,
        ),
    )
    assert [p.symbol for p in repo.get_open_positions()] == ["ACET"]

    insert_position_tombstone(
        db,
        symbol="ACET",
        source="fill_ingest",
        notes="closed by order 1",
        snapshot_at=dt.datetime(2026, 6, 9, 13, 35, tzinfo=dt.UTC),
    )

    assert repo.get_open_positions() == []
    # Same stale-snapshot semantics the thesis coordinator's guard uses —
    # the tombstone also stops phantom Opus reviews (delta §2.1 effects).
    assert has_open_position(db, "ACET") is False


def test_tombstone_is_appended_not_an_edit(db: str) -> None:
    insert_position(
        db,
        PositionRow(
            snapshot_at=dt.datetime(2026, 6, 8, 14, 30, tzinfo=dt.UTC),
            source="reconciler",
            symbol="ACET",
            qty=30,
            avg_entry_price=8.20,
            market_value=246.0,
            unrealized_pnl=0.0,
        ),
    )
    insert_position_tombstone(
        db,
        symbol="ACET",
        source="reconciler_external_close",
        notes="external close",
        snapshot_at=dt.datetime(2026, 6, 9, 13, 35, tzinfo=dt.UTC),
    )
    with sqlite3.connect(db) as conn:
        count, = conn.execute(
            "SELECT COUNT(*) FROM positions WHERE symbol='ACET'"
        ).fetchone()
        qty, src = conn.execute(
            "SELECT qty, source FROM positions WHERE symbol='ACET' "
            "ORDER BY snapshot_at DESC LIMIT 1"
        ).fetchone()
    assert count == 2  # NFR-16: append-only — the qty=30 row survives
    assert qty == 0
    assert src == "reconciler_external_close"


# ---------------------------------------------------------------------------
# get_live_protective_order
# ---------------------------------------------------------------------------

def test_get_live_protective_order_returns_pending_stop(db: str) -> None:
    _, eid = _seed_proposal_and_execution(db)
    insert_order(
        db,
        _order(eid, role="entry", side="buy", broker_order_id="b-entry",
               final_status="filled"),
    )
    stop_id = insert_order(
        db, _order(eid, role="stop", side="sell", broker_order_id="b-stop")
    )

    row = get_live_protective_order(db, eid, roles=("stop", "trailing_stop"))

    assert row is not None
    assert row.id == stop_id
    assert row.role == "stop"


def test_get_live_protective_order_ignores_filled_and_foreign_roles(db: str) -> None:
    _, eid = _seed_proposal_and_execution(db)
    insert_order(
        db,
        _order(eid, role="stop", side="sell", broker_order_id="b-stop",
               final_status="filled"),
    )
    insert_order(
        db, _order(eid, role="take_profit", side="sell", broker_order_id="b-tp")
    )

    assert get_live_protective_order(db, eid, roles=("stop", "trailing_stop")) is None


def test_get_live_protective_order_latest_wins(db: str) -> None:
    """A ratchet-replaced stop chain: the newest pending row is the live one."""
    _, eid = _seed_proposal_and_execution(db)
    insert_order(
        db,
        _order(eid, role="stop", side="sell", broker_order_id="b-stop-old",
               final_status="replaced"),
    )
    new_id = insert_order(
        db,
        _order(eid, role="trailing_stop", side="sell", broker_order_id="b-stop-new"),
    )

    row = get_live_protective_order(db, eid, roles=("stop", "trailing_stop"))
    assert row is not None
    assert row.id == new_id


# ---------------------------------------------------------------------------
# get_lifecycle_orders_for_symbol — the §2.2 classifier query
# ---------------------------------------------------------------------------

def test_lifecycle_orders_pending_and_recently_filled(db: str) -> None:
    _, eid = _seed_proposal_and_execution(db)
    now = dt.datetime(2026, 6, 9, 14, 0, tzinfo=dt.UTC)
    filled_since = now - dt.timedelta(seconds=600)

    pending_id = insert_order(
        db, _order(eid, role="thesis_close", side="sell", broker_order_id="b-pend")
    )
    recent_id = insert_order(
        db,
        _order(
            eid, role="stop", side="sell", broker_order_id="b-recent",
            final_status="filled",
            realized_fill_at=now - dt.timedelta(seconds=200),
            realized_fill_qty=10, realized_fill_price=9.0,
        ),
    )
    # Filled long ago — consumed, no longer explains anything (RC-2 keying
    # is lifecycle, but a recorded fill only explains for 2 ticks).
    insert_order(
        db,
        _order(
            eid, role="take_profit", side="sell", broker_order_id="b-old",
            final_status="filled",
            realized_fill_at=now - dt.timedelta(days=10),
            realized_fill_qty=10, realized_fill_price=12.0,
        ),
    )

    rows = get_lifecycle_orders_for_symbol(db, symbol="AAPL", filled_since=filled_since)

    assert {r.id for r in rows} == {pending_id, recent_id}


def test_lifecycle_orders_no_submitted_at_expiry(db: str) -> None:
    """RC-2 regression: a sell submitted 30 days ago but still pending fill
    must keep explaining the diff — explanation is keyed to lifecycle, not
    to a sliding submitted_at window."""
    _, eid = _seed_proposal_and_execution(db)
    now = dt.datetime(2026, 6, 9, 14, 0, tzinfo=dt.UTC)
    old_pending = insert_order(
        db,
        _order(
            eid, role="stop", side="sell", broker_order_id="b-gtc",
            submitted_at=now - dt.timedelta(days=30),
        ),
    )

    rows = get_lifecycle_orders_for_symbol(
        db, symbol="AAPL", filled_since=now - dt.timedelta(seconds=600)
    )

    assert [r.id for r in rows] == [old_pending]

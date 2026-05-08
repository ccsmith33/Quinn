# PDT-SUNSET-2026-06-04: tests for VirtualExitRow + DeferredSellRow repo CRUD.
"""S-PDT-1 — repo functions for virtual_exits and deferred_sells.

References: ADR-009 §"Data model"; story S-PDT-1 AC-3, AC-6.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

import pytest

from journal.migrate import apply_migrations


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
        # Idempotent prompt seed.
        conn.execute(
            "INSERT OR IGNORE INTO prompts (prompt_version, name, file_path, "
            "content_hash) VALUES ('pv1','sonnet_filing_analysis_v1','/x/p.md','h1')"
        )
        conn.execute(
            "INSERT INTO filings (accession_number, cik, form_type, filed_at, "
            "fetched_at, raw_text_path, content_hash, issuer_ticker) "
            "VALUES (?, 1, '8-K', '2026-05-01 14:30:00', '2026-05-01 14:31:00', "
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


# ---------------------------------------------------------------------------
# VirtualExitRow — insert + readback
# ---------------------------------------------------------------------------


def test_insert_and_readback_virtual_exit(db: str) -> None:
    from journal.models import VirtualExitRow
    from journal.repo import (
        insert_virtual_exit,
        list_active_virtual_exits,
    )

    pid, eid = _seed_proposal_and_execution(db, "AAPL")
    new_id = insert_virtual_exit(
        db,
        VirtualExitRow(
            execution_id=eid,
            proposal_id=pid,
            symbol="AAPL",
            qty=5,
            role="stop",
            entry_price=100.0,
            stop_price=95.0,
        ),
    )
    assert isinstance(new_id, int) and new_id > 0
    rows = list_active_virtual_exits(db)
    assert len(rows) == 1
    got = rows[0]
    assert got.id == new_id
    assert got.execution_id == eid
    assert got.proposal_id == pid
    assert got.symbol == "AAPL"
    assert got.qty == 5
    assert got.role == "stop"
    assert got.entry_price == 100.0
    assert got.stop_price == 95.0
    assert got.tp_price is None
    assert got.state == "active"
    assert got.created_at is not None


def test_list_active_virtual_exits_filters_state(db: str) -> None:
    from journal.models import VirtualExitRow
    from journal.repo import (
        insert_virtual_exit,
        list_active_virtual_exits,
        mark_virtual_exit_obsolete,
    )

    pid, eid = _seed_proposal_and_execution(db, "AAPL")
    active_id = insert_virtual_exit(
        db,
        VirtualExitRow(
            execution_id=eid,
            proposal_id=pid,
            symbol="AAPL",
            qty=5,
            role="stop",
            entry_price=100.0,
            stop_price=95.0,
        ),
    )
    obsolete_id = insert_virtual_exit(
        db,
        VirtualExitRow(
            execution_id=eid,
            proposal_id=pid,
            symbol="AAPL",
            qty=5,
            role="tp",
            entry_price=100.0,
            tp_price=110.0,
        ),
    )
    mark_virtual_exit_obsolete(db, obsolete_id, reason="position_closed_externally")

    rows = list_active_virtual_exits(db)
    assert {r.id for r in rows} == {active_id}


def test_list_active_virtual_exits_ordered_by_created_at_asc(db: str) -> None:
    from journal.models import VirtualExitRow
    from journal.repo import (
        insert_virtual_exit,
        list_active_virtual_exits,
    )

    pid, eid = _seed_proposal_and_execution(db, "AAPL")
    first = insert_virtual_exit(
        db,
        VirtualExitRow(
            execution_id=eid, proposal_id=pid, symbol="AAPL", qty=5, role="stop",
            entry_price=100.0, stop_price=95.0,
        ),
    )
    second = insert_virtual_exit(
        db,
        VirtualExitRow(
            execution_id=eid, proposal_id=pid, symbol="AAPL", qty=5, role="tp",
            entry_price=100.0, tp_price=110.0,
        ),
    )
    rows = list_active_virtual_exits(db)
    assert [r.id for r in rows] == [first, second]


def test_list_active_virtual_exits_for_symbol_filters(db: str) -> None:
    from journal.models import VirtualExitRow
    from journal.repo import (
        insert_virtual_exit,
        list_active_virtual_exits_for_symbol,
    )

    p1, e1 = _seed_proposal_and_execution(db, "AAPL")
    p2, e2 = _seed_proposal_and_execution(db, "MSFT")
    aapl_id = insert_virtual_exit(
        db,
        VirtualExitRow(
            execution_id=e1, proposal_id=p1, symbol="AAPL", qty=5, role="stop",
            entry_price=100.0, stop_price=95.0,
        ),
    )
    insert_virtual_exit(
        db,
        VirtualExitRow(
            execution_id=e2, proposal_id=p2, symbol="MSFT", qty=3, role="stop",
            entry_price=200.0, stop_price=190.0,
        ),
    )
    aapl_rows = list_active_virtual_exits_for_symbol(db, "AAPL")
    assert [r.id for r in aapl_rows] == [aapl_id]
    msft_rows = list_active_virtual_exits_for_symbol(db, "MSFT")
    assert len(msft_rows) == 1 and msft_rows[0].symbol == "MSFT"


def test_mark_virtual_exit_submitted(db: str) -> None:
    from journal.models import VirtualExitRow
    from journal.repo import (
        connect,
        insert_virtual_exit,
        list_active_virtual_exits,
        mark_virtual_exit_submitted,
    )

    pid, eid = _seed_proposal_and_execution(db, "AAPL")
    vid = insert_virtual_exit(
        db,
        VirtualExitRow(
            execution_id=eid, proposal_id=pid, symbol="AAPL", qty=5, role="stop",
            entry_price=100.0, stop_price=95.0,
        ),
    )
    mark_virtual_exit_submitted(db, vid, broker_order_id="alp-stop-1")

    # No longer active.
    assert list_active_virtual_exits(db) == []
    # State + broker order id persisted; submitted_at set.
    with connect(db) as conn:
        row = conn.execute(
            "SELECT state, submitted_broker_order_id, submitted_at "
            "FROM virtual_exits WHERE id = ?",
            (vid,),
        ).fetchone()
    assert row["state"] == "submitted"
    assert row["submitted_broker_order_id"] == "alp-stop-1"
    assert row["submitted_at"] is not None


def test_mark_virtual_exit_obsolete_appends_notes(db: str) -> None:
    from journal.models import VirtualExitRow
    from journal.repo import (
        connect,
        insert_virtual_exit,
        mark_virtual_exit_obsolete,
    )

    pid, eid = _seed_proposal_and_execution(db, "AAPL")
    vid = insert_virtual_exit(
        db,
        VirtualExitRow(
            execution_id=eid, proposal_id=pid, symbol="AAPL", qty=5, role="stop",
            entry_price=100.0, stop_price=95.0,
            notes="initial",
        ),
    )
    mark_virtual_exit_obsolete(db, vid, reason="position_closed_externally")
    with connect(db) as conn:
        row = conn.execute(
            "SELECT state, notes FROM virtual_exits WHERE id = ?", (vid,)
        ).fetchone()
    assert row["state"] == "obsolete"
    assert "position_closed_externally" in (row["notes"] or "")
    assert "initial" in (row["notes"] or "")


# ---------------------------------------------------------------------------
# DeferredSellRow — insert + readback
# ---------------------------------------------------------------------------


def test_insert_and_readback_deferred_sell(db: str) -> None:
    from journal.models import DeferredSellRow, VirtualExitRow
    from journal.repo import (
        insert_deferred_sell,
        insert_virtual_exit,
        list_unreplayed_deferred_sells,
    )

    pid, eid = _seed_proposal_and_execution(db, "AAPL")
    vid = insert_virtual_exit(
        db,
        VirtualExitRow(
            execution_id=eid, proposal_id=pid, symbol="AAPL", qty=5, role="stop",
            entry_price=100.0, stop_price=95.0,
        ),
    )
    new_id = insert_deferred_sell(
        db,
        DeferredSellRow(
            virtual_exit_id=vid,
            execution_id=eid,
            proposal_id=pid,
            symbol="AAPL",
            qty=5,
            role="stop",
            trigger_price=94.50,
            ev_at_defer=-22.50,
            deferred_reason="ev_lost",
        ),
    )
    assert isinstance(new_id, int) and new_id > 0
    rows = list_unreplayed_deferred_sells(db)
    assert len(rows) == 1
    got = rows[0]
    assert got.id == new_id
    assert got.virtual_exit_id == vid
    assert got.execution_id == eid
    assert got.proposal_id == pid
    assert got.role == "stop"
    assert got.trigger_price == 94.50
    assert got.ev_at_defer == -22.50
    assert got.deferred_reason == "ev_lost"
    assert got.replayed_at is None
    assert got.replay_broker_order_id is None
    assert got.deferred_at is not None


def test_list_unreplayed_deferred_sells_filters_replayed(db: str) -> None:
    from journal.models import DeferredSellRow, VirtualExitRow
    from journal.repo import (
        insert_deferred_sell,
        insert_virtual_exit,
        list_unreplayed_deferred_sells,
        mark_deferred_replayed,
    )

    pid, eid = _seed_proposal_and_execution(db, "AAPL")
    vid = insert_virtual_exit(
        db,
        VirtualExitRow(
            execution_id=eid, proposal_id=pid, symbol="AAPL", qty=5, role="stop",
            entry_price=100.0, stop_price=95.0,
        ),
    )
    pending = insert_deferred_sell(
        db,
        DeferredSellRow(
            virtual_exit_id=vid, execution_id=eid, proposal_id=pid,
            symbol="AAPL", qty=5, role="stop",
            trigger_price=94.50, ev_at_defer=-22.50, deferred_reason="ev_lost",
        ),
    )
    replayed = insert_deferred_sell(
        db,
        DeferredSellRow(
            virtual_exit_id=vid, execution_id=eid, proposal_id=pid,
            symbol="AAPL", qty=5, role="stop",
            trigger_price=94.50, ev_at_defer=-22.50, deferred_reason="ev_lost",
        ),
    )
    mark_deferred_replayed(db, replayed, broker_order_id="alp-replay-1")

    rows = list_unreplayed_deferred_sells(db)
    assert {r.id for r in rows} == {pending}


def test_mark_deferred_skipped(db: str) -> None:
    """S-PDT-5 AC-3: mark_deferred_skipped sets replayed_at,
    leaves replay_broker_order_id NULL, appends reason to notes.
    """
    from journal.models import DeferredSellRow, VirtualExitRow
    from journal.repo import (
        connect,
        insert_deferred_sell,
        insert_virtual_exit,
        list_unreplayed_deferred_sells,
        mark_deferred_skipped,
    )

    pid, eid = _seed_proposal_and_execution(db, "AAPL")
    vid = insert_virtual_exit(
        db,
        VirtualExitRow(
            execution_id=eid, proposal_id=pid, symbol="AAPL", qty=5, role="stop",
            entry_price=100.0, stop_price=95.0,
        ),
    )
    did = insert_deferred_sell(
        db,
        DeferredSellRow(
            virtual_exit_id=vid, execution_id=eid, proposal_id=pid,
            symbol="AAPL", qty=5, role="stop",
            trigger_price=94.50, ev_at_defer=-22.50, deferred_reason="ev_lost",
            notes="initial",
        ),
    )
    mark_deferred_skipped(db, did, reason="position_closed_externally")
    # Drops out of unreplayed queue.
    assert list_unreplayed_deferred_sells(db) == []
    with connect(db) as conn:
        row = conn.execute(
            "SELECT replayed_at, replay_broker_order_id, notes "
            "FROM deferred_sells WHERE id = ?",
            (did,),
        ).fetchone()
    assert row["replayed_at"] is not None
    assert row["replay_broker_order_id"] is None
    notes = row["notes"] or ""
    assert "initial" in notes
    assert "position_closed_externally" in notes


def test_mark_deferred_replayed(db: str) -> None:
    from journal.models import DeferredSellRow, VirtualExitRow
    from journal.repo import (
        connect,
        insert_deferred_sell,
        insert_virtual_exit,
        mark_deferred_replayed,
    )

    pid, eid = _seed_proposal_and_execution(db, "AAPL")
    vid = insert_virtual_exit(
        db,
        VirtualExitRow(
            execution_id=eid, proposal_id=pid, symbol="AAPL", qty=5, role="stop",
            entry_price=100.0, stop_price=95.0,
        ),
    )
    did = insert_deferred_sell(
        db,
        DeferredSellRow(
            virtual_exit_id=vid, execution_id=eid, proposal_id=pid,
            symbol="AAPL", qty=5, role="stop",
            trigger_price=94.50, ev_at_defer=-22.50, deferred_reason="ev_lost",
        ),
    )
    mark_deferred_replayed(db, did, broker_order_id="alp-replay-1")
    with connect(db) as conn:
        row = conn.execute(
            "SELECT replayed_at, replay_broker_order_id FROM deferred_sells "
            "WHERE id = ?",
            (did,),
        ).fetchone()
    assert row["replayed_at"] is not None
    assert row["replay_broker_order_id"] == "alp-replay-1"


# ---------------------------------------------------------------------------
# CHECK constraints
# ---------------------------------------------------------------------------


def test_virtual_exit_invalid_role_rejected(db: str) -> None:
    pid, eid = _seed_proposal_and_execution(db, "AAPL")
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO virtual_exits (execution_id, proposal_id, symbol, "
                "qty, role, entry_price, stop_price) "
                "VALUES (?, ?, 'AAPL', 1, 'thesis_close', 100.0, 95.0)",
                (eid, pid),
            )


def test_deferred_sell_invalid_reason_rejected(db: str) -> None:
    from journal.models import VirtualExitRow
    from journal.repo import insert_virtual_exit

    pid, eid = _seed_proposal_and_execution(db, "AAPL")
    vid = insert_virtual_exit(
        db,
        VirtualExitRow(
            execution_id=eid, proposal_id=pid, symbol="AAPL", qty=5, role="stop",
            entry_price=100.0, stop_price=95.0,
        ),
    )
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO deferred_sells (virtual_exit_id, execution_id, "
                "proposal_id, symbol, qty, role, trigger_price, ev_at_defer, "
                "deferred_reason) VALUES (?, ?, ?, 'AAPL', 5, 'stop', 94.5, "
                "-22.5, 'unknown_reason')",
                (vid, eid, pid),
            )

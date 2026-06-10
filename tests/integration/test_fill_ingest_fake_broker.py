"""WS1 — integration-harness fake-broker fill support (delta §1,
tests/integration/conftest.py seam: WS1 adds `get_order_by_id` + fill-field
support so FillIngestor paths run against the shared `_FakeBroker`).

Proves the harness fake satisfies the §7.1 contract end-to-end: an order
submitted through the fake, marked filled via `fill_order`, is picked up by
a REAL FillIngestor polling a REAL journal and lands as a recorded outcome
plus a qty-0 tombstone.
"""

from __future__ import annotations

import datetime as dt
import sqlite3

from broker.protocol import OrderRequest
from journal.models import OrderRow, PositionRow
from journal.repo import insert_order
from reconciler.fill_ingest import FillIngestor

_NOW = dt.datetime(2026, 6, 9, 15, 0, tzinfo=dt.UTC)


def _seed_execution(db_path: str, symbol: str = "ACME") -> int:
    """FK chain prompts → filings → proposals → executions; returns execution id."""
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
            (f"acc-{symbol}", symbol),
        )
        fid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO proposals (filing_id, decision_id, model_id, "
            "prompt_version, raw_response, kind, symbol, conviction, "
            "input_tokens, output_tokens, latency_ms, cost_usd) "
            "VALUES (?, ?, 'm', 'pv1', '{}', 'trade_proposal', ?, 7, 1, 1, 1, 0.01)",
            (fid, f"d-{symbol}", symbol),
        )
        pid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO executions (proposal_id, decision, "
            "submitted_orders_json) VALUES (?, 'accepted', '[]')",
            (pid,),
        )
        return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def test_fake_broker_get_order_by_id_round_trip(fake_broker) -> None:
    """Submitted orders are retrievable by broker id with empty fill
    fields (submission-time acknowledgement, §7.1)."""
    sub = fake_broker.submit_order(
        OrderRequest(
            symbol="ACME", side="sell", qty=10, order_type="market",
            tif="gtc", client_order_id="thesis-close-exec-1",
        )
    )

    got = fake_broker.get_order_by_id(sub.broker_order_id)

    assert got is not None
    assert got.broker_order_id == sub.broker_order_id
    assert got.status == "accepted"
    assert got.filled_qty == 0
    assert got.filled_avg_price is None
    assert got.filled_at is None
    # Unknown id → None (mirrors get_order_by_client_id's 404 contract).
    assert fake_broker.get_order_by_id("no-such-order") is None


def test_fake_broker_fill_order_marks_terminal(fake_broker) -> None:
    sub = fake_broker.submit_order(
        OrderRequest(
            symbol="ACME", side="sell", qty=10, order_type="market",
            tif="gtc", client_order_id="thesis-close-exec-2",
        )
    )

    fake_broker.fill_order(
        sub.broker_order_id, price=10.5, filled_at=_NOW,
    )

    got = fake_broker.get_order_by_id(sub.broker_order_id)
    assert got is not None
    assert got.status == "filled"
    assert got.filled_qty == 10
    assert got.filled_avg_price == 10.5
    assert got.filled_at == _NOW


def test_fill_ingestor_runs_against_harness_fake(db_path, journal, fake_broker) -> None:
    """The §7.1 seam end-to-end on the SHARED harness fake: journaled sell
    fills at the (fake) broker → FillIngestor records the outcome and
    tombstones the closed symbol."""
    eid = _seed_execution(db_path)
    sub = fake_broker.submit_order(
        OrderRequest(
            symbol="ACME", side="sell", qty=10, order_type="market",
            tif="gtc", client_order_id="thesis-close-exec-3",
        )
    )
    insert_order(
        db_path,
        OrderRow(
            execution_id=eid,
            role="thesis_close",
            symbol="ACME",
            side="sell",
            order_type="market",
            qty=10,
            tif="gtc",
            broker_order_id=sub.broker_order_id,
            submitted_at=_NOW - dt.timedelta(minutes=10),
        ),
    )
    journal.insert_position(
        PositionRow(
            snapshot_at=_NOW - dt.timedelta(minutes=5),
            source="reconciler",
            symbol="ACME",
            qty=10,
            avg_entry_price=10.0,
            market_value=100.0,
            unrealized_pnl=0.0,
        )
    )
    fake_broker.fill_order(sub.broker_order_id, price=10.5, filled_at=_NOW)

    ingestor = FillIngestor(
        broker=fake_broker, journal=journal, now_fn=lambda: _NOW
    )
    report = ingestor.run_tick()

    assert report.recorded == 1
    assert report.tombstoned == ["ACME"]
    assert journal.get_orders_pending_fill() == []
    assert journal.get_open_positions() == []


def test_fake_broker_bracket_children_retrievable_by_id(fake_broker) -> None:
    """Bracket legs (Alpaca-generated children) must be pollable too —
    the FillIngestor polls every journaled order row, including bracket
    stop/TP legs journaled at submission."""
    from broker.protocol import BracketOrderRequest

    entry, stop, tp = fake_broker.submit_bracket_order(
        BracketOrderRequest(
            entry_symbol="ACME",
            entry_side="buy",
            entry_qty=10,
            entry_order_type="market",
            entry_tif="day",
            entry_client_order_id="prop-1-entry",
            stop_loss_price=9.0,
            take_profit_price=12.0,
        )
    )

    for leg in (entry, stop, tp):
        assert leg is not None
        got = fake_broker.get_order_by_id(leg.broker_order_id)
        assert got is not None
        assert got.broker_order_id == leg.broker_order_id
        assert got.filled_qty == 0

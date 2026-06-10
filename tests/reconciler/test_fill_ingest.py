"""WS1 — FillIngestor (D-078, delta §2.1, ADR-010).

Polls `get_order_by_id` for every journaled order with final_status IS
NULL at the top of each reconcile tick; records terminal outcomes via the
one-time NULL→value `record_order_outcome`; tombstones a symbol when a
recorded sell fill closes the journal's open quantity.

Real JournalRepo on a tmp SQLite db (the SQL is the contract under test);
fake broker per the established reconciler-test pattern.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

import pytest

from broker.alpaca import BrokerUnavailable
from broker.protocol import SubmittedOrder
from journal.migrate import apply_migrations
from journal.models import OrderRow, PositionRow
from journal.repo import JournalRepo
from reconciler.fill_ingest import FillIngestor

_NOW = dt.datetime(2026, 6, 9, 14, 0, tzinfo=dt.UTC)


@pytest.fixture
def repo(tmp_path: Path) -> JournalRepo:
    db_path = tmp_path / "journal.db"
    apply_migrations(str(db_path))
    return JournalRepo(str(db_path))


def _seed_execution(db_path: str, symbol: str = "AAPL") -> int:
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
        return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def _insert_order(
    repo: JournalRepo,
    eid: int,
    *,
    symbol: str = "AAPL",
    role: str = "stop",
    side: str = "sell",
    qty: int = 10,
    broker_order_id: str = "b-1",
) -> int:
    from journal.repo import insert_order

    return insert_order(
        repo.db_path,
        OrderRow(
            execution_id=eid,
            role=role,
            symbol=symbol,
            side=side,
            order_type="market",
            qty=qty,
            tif="gtc",
            broker_order_id=broker_order_id,
            submitted_at=_NOW - dt.timedelta(days=1),
        ),
    )


def _snapshot_position(repo: JournalRepo, symbol: str, qty: int) -> None:
    repo.insert_position(
        PositionRow(
            snapshot_at=_NOW - dt.timedelta(minutes=10),
            source="reconciler",
            symbol=symbol,
            qty=qty,
            avg_entry_price=10.0,
            market_value=qty * 10.0,
            unrealized_pnl=0.0,
        )
    )


def _broker_order(
    broker_order_id: str,
    *,
    symbol: str = "AAPL",
    side: str = "sell",
    status: str = "filled",
    qty: int = 10,
    filled_qty: int = 10,
    filled_avg_price: float | None = 10.5,
    filled_at: dt.datetime | None = _NOW - dt.timedelta(minutes=2),
) -> SubmittedOrder:
    return SubmittedOrder(
        broker_order_id=broker_order_id,
        client_order_id=f"cid-{broker_order_id}",
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        qty=qty,
        order_type="market",
        status=status,  # type: ignore[arg-type]
        submitted_at=_NOW - dt.timedelta(days=1),
        filled_avg_price=filled_avg_price,
        filled_qty=filled_qty,
        filled_at=filled_at,
    )


class _FakeBroker:
    def __init__(self) -> None:
        self.orders: dict[str, SubmittedOrder] = {}
        self.raise_seq: list[Exception] = []
        self.calls: list[str] = []

    def get_order_by_id(self, broker_order_id: str) -> SubmittedOrder | None:
        self.calls.append(broker_order_id)
        if self.raise_seq:
            raise self.raise_seq.pop(0)
        return self.orders.get(broker_order_id)


def _ingestor(broker: _FakeBroker, repo: JournalRepo) -> FillIngestor:
    return FillIngestor(broker=broker, journal=repo, now_fn=lambda: _NOW)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_filled_sell_recorded_and_position_tombstoned(repo: JournalRepo) -> None:
    """The ghost-position killer: a sell that closes the full open qty
    records its fill AND appends the qty=0 tombstone (RC-1 fix)."""
    eid = _seed_execution(repo.db_path)
    oid = _insert_order(repo, eid, broker_order_id="b-stop")
    _snapshot_position(repo, "AAPL", 10)
    broker = _FakeBroker()
    broker.orders["b-stop"] = _broker_order("b-stop")

    report = _ingestor(broker, repo).run_tick()

    assert report.recorded == 1
    assert report.tombstoned == ["AAPL"]
    assert repo.get_orders_pending_fill() == []
    [row] = repo.get_orders_since(symbol="AAPL", since=dt.datetime(2026, 1, 1))
    assert row.id == oid
    assert row.final_status == "filled"
    assert row.realized_fill_price == 10.5
    assert row.realized_fill_qty == 10
    # Journal no longer believes the position is open.
    assert repo.get_open_positions() == []


def test_filled_buy_recorded_without_tombstone(repo: JournalRepo) -> None:
    eid = _seed_execution(repo.db_path)
    _insert_order(repo, eid, role="entry", side="buy", broker_order_id="b-entry")
    broker = _FakeBroker()
    broker.orders["b-entry"] = _broker_order("b-entry", side="buy")

    report = _ingestor(broker, repo).run_tick()

    assert report.recorded == 1
    assert report.tombstoned == []


def test_partial_sell_fill_no_tombstone(repo: JournalRepo) -> None:
    """A sell covering only part of the open qty must NOT tombstone."""
    eid = _seed_execution(repo.db_path)
    _insert_order(repo, eid, qty=4, broker_order_id="b-stop")
    _snapshot_position(repo, "AAPL", 10)
    broker = _FakeBroker()
    broker.orders["b-stop"] = _broker_order("b-stop", qty=4, filled_qty=4)

    report = _ingestor(broker, repo).run_tick()

    assert report.recorded == 1
    assert report.tombstoned == []
    assert [p.symbol for p in repo.get_open_positions()] == ["AAPL"]


def test_nonterminal_status_left_pending(repo: JournalRepo) -> None:
    eid = _seed_execution(repo.db_path)
    oid = _insert_order(repo, eid, broker_order_id="b-stop")
    broker = _FakeBroker()
    broker.orders["b-stop"] = _broker_order(
        "b-stop", status="new", filled_qty=0, filled_avg_price=None, filled_at=None
    )

    report = _ingestor(broker, repo).run_tick()

    assert report.recorded == 0
    assert [r.id for r in repo.get_orders_pending_fill()] == [oid]


def test_canceled_after_partial_fill_maps_to_partially_filled_closed(
    repo: JournalRepo,
) -> None:
    eid = _seed_execution(repo.db_path)
    _insert_order(repo, eid, broker_order_id="b-stop")
    _snapshot_position(repo, "AAPL", 10)
    broker = _FakeBroker()
    broker.orders["b-stop"] = _broker_order(
        "b-stop", status="canceled", filled_qty=3, filled_avg_price=9.8
    )

    report = _ingestor(broker, repo).run_tick()

    assert report.recorded == 1
    assert report.tombstoned == []
    [row] = repo.get_orders_since(symbol="AAPL", since=dt.datetime(2026, 1, 1))
    assert row.final_status == "partially_filled_closed"
    assert row.realized_fill_qty == 3


def test_canceled_without_fill_maps_to_canceled(repo: JournalRepo) -> None:
    eid = _seed_execution(repo.db_path)
    _insert_order(repo, eid, broker_order_id="b-stop")
    broker = _FakeBroker()
    broker.orders["b-stop"] = _broker_order(
        "b-stop", status="canceled", filled_qty=0, filled_avg_price=None,
        filled_at=None,
    )

    _ingestor(broker, repo).run_tick()

    [row] = repo.get_orders_since(symbol="AAPL", since=dt.datetime(2026, 1, 1))
    assert row.final_status == "canceled"


# ---------------------------------------------------------------------------
# Broker-error paths
# ---------------------------------------------------------------------------

def test_broker_unavailable_defers_without_recording(repo: JournalRepo) -> None:
    eid = _seed_execution(repo.db_path)
    oid = _insert_order(repo, eid, broker_order_id="b-stop")
    broker = _FakeBroker()
    broker.raise_seq = [BrokerUnavailable("down")]

    report = _ingestor(broker, repo).run_tick()

    assert report.deferred is True
    assert report.recorded == 0
    assert [r.id for r in repo.get_orders_pending_fill()] == [oid]


def test_broker_404_leaves_row_pending(repo: JournalRepo) -> None:
    """An order the broker doesn't know stays pending (visible in logs) —
    we never invent an outcome."""
    eid = _seed_execution(repo.db_path)
    oid = _insert_order(repo, eid, broker_order_id="b-ghost")
    broker = _FakeBroker()  # empty orders dict → returns None

    report = _ingestor(broker, repo).run_tick()

    assert report.deferred is False
    assert report.recorded == 0
    assert [r.id for r in repo.get_orders_pending_fill()] == [oid]


def test_per_row_error_does_not_block_other_rows(repo: JournalRepo) -> None:
    """A poisoned row must not stop the rest of the poll loop."""
    eid = _seed_execution(repo.db_path)
    _insert_order(repo, eid, broker_order_id="b-bad")
    _insert_order(repo, eid, broker_order_id="b-good", qty=10)
    _snapshot_position(repo, "AAPL", 10)
    broker = _FakeBroker()
    broker.raise_seq = [ValueError("malformed broker payload")]
    broker.orders["b-good"] = _broker_order("b-good")

    report = _ingestor(broker, repo).run_tick()

    assert report.recorded == 1  # b-good still processed


# ---------------------------------------------------------------------------
# Idempotency across ticks
# ---------------------------------------------------------------------------

def test_second_tick_is_noop_after_recording(repo: JournalRepo) -> None:
    eid = _seed_execution(repo.db_path)
    _insert_order(repo, eid, broker_order_id="b-stop")
    _snapshot_position(repo, "AAPL", 10)
    broker = _FakeBroker()
    broker.orders["b-stop"] = _broker_order("b-stop")
    ingestor = _ingestor(broker, repo)

    first = ingestor.run_tick()
    second = ingestor.run_tick()

    assert first.recorded == 1
    assert second.polled == 0
    assert second.recorded == 0
    assert second.tombstoned == []
    # Exactly one tombstone row exists.
    with sqlite3.connect(repo.db_path) as conn:
        count, = conn.execute(
            "SELECT COUNT(*) FROM positions WHERE symbol='AAPL' AND qty=0"
        ).fetchone()
    assert count == 1


def test_same_tick_multi_order_full_close_tombstones_once(repo: JournalRepo) -> None:
    """W1-L3: two sells (5 + 5) against an open 10 both fill in the SAME
    tick. Each fill alone is partial against the (stale) latest snapshot,
    but together they close the position — the tick must sum its own
    recorded sell fills and tombstone exactly once, with fill_ingest
    provenance (not a later external-close absorption + spurious alert)."""
    eid = _seed_execution(repo.db_path)
    _insert_order(repo, eid, qty=5, broker_order_id="b-a")
    _insert_order(
        repo, eid, role="thesis_close", qty=5, broker_order_id="b-b"
    )
    _snapshot_position(repo, "AAPL", 10)
    broker = _FakeBroker()
    broker.orders["b-a"] = _broker_order("b-a", qty=5, filled_qty=5)
    broker.orders["b-b"] = _broker_order("b-b", qty=5, filled_qty=5)

    report = _ingestor(broker, repo).run_tick()

    assert report.recorded == 2
    assert report.tombstoned == ["AAPL"]
    assert repo.get_open_positions() == []
    with sqlite3.connect(repo.db_path) as conn:
        count, = conn.execute(
            "SELECT COUNT(*) FROM positions WHERE symbol='AAPL' AND qty=0"
        ).fetchone()
        src, = conn.execute(
            "SELECT source FROM positions WHERE symbol='AAPL' AND qty=0"
        ).fetchone()
    assert count == 1
    assert src == "fill_ingest"

"""Integration (task #4, W2-L1) — the reconciler↔exit-policy seam.

WS1's reconciler invokes `exit_policy_ticker.run_tick()` WITHOUT await
(delta §3.5: sync, like the scanner hook). WS2 made `run_tick` sync to
match. Nothing structural prevents a future edit from making it async
again — the un-awaited call would then build a coroutine, run nothing,
and raise nothing: the trail would silently die while every unit test
of the ticker still passed. Two guards:

  1. a sync pin on the hook entrypoint;
  2. one composed test: the REAL Reconciler drives the REAL
     ExitPolicyTicker through the real ctor seam over the shared fake
     broker, and the ratchet's effects (broker PATCH + journal chain +
     persisted state) are asserted end-to-end.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import inspect
import sqlite3

import pytest

from broker.protocol import Position
from config.loader import ReconcilerConfig
from execution.exit_policy import ExitPolicyTicker
from journal.exit_policy import get_exit_policy_state
from journal.models import OrderRow, PositionRow
from journal.repo import JournalRepo, insert_order
from killswitch.api import KillSwitch
from reconciler.fill_ingest import FillIngestor
from reconciler.reconciler import Reconciler

# Tuesday 2026-06-09 15:00 UTC = 11:00 ET — market hours.
_T0 = dt.datetime(2026, 6, 9, 15, 0, tzinfo=dt.UTC)


class _Clock:
    def __init__(self, start: dt.datetime) -> None:
        self.now = start

    def __call__(self) -> dt.datetime:
        return self.now


class _FakeAlerter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def notify(self, message: str) -> None:
        self.calls.append(message)


def _seed_execution(db_path: str, symbol: str) -> int:
    """FK chain prompts → filings → proposals → executions; returns
    execution id. raw_response '{}' fails trade-proposal validation, so
    the trail distance falls back to the ADR-011 default — the
    initial-risk percent — which is what this test exercises."""
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


def test_run_tick_is_sync_pin() -> None:
    """W2-L1 pin: the reconciler calls `run_tick()` un-awaited inside
    its protective try/except. An async `run_tick` would silently never
    run. This assertion is the tripwire."""
    assert not inspect.iscoroutinefunction(ExitPolicyTicker.run_tick)
    assert not inspect.isasyncgenfunction(ExitPolicyTicker.run_tick)


@pytest.mark.asyncio
async def test_reconcile_tick_drives_real_ticker_ratchet(
    journal: JournalRepo, fake_broker, db_path: str
) -> None:
    """Composed W2-L1 proof: one REAL reconciler loop tick on a winner
    engages the trail and ratchets the broker-side stop — asserting the
    broker PATCH, the §3.3-shaped journal chain (old row replaced → new
    trailing_stop row live), and the persisted high-water state. No
    stubs at the seam: the hook lives in `_run_loop` (not
    `reconcile_now`), so the test drives `start()`/`stop()`."""
    clock = _Clock(_T0)
    symbol = "ACME"
    eid = _seed_execution(db_path, symbol)

    # Entry: 10 sh filled at 100.00 (recorded via the §7.4 single-writer).
    entry_id = insert_order(
        db_path,
        OrderRow(
            execution_id=eid, role="entry", symbol=symbol, side="buy",
            order_type="market", qty=10, tif="day",
            broker_order_id="b-entry-1",
            submitted_at=_T0 - dt.timedelta(days=1),
        ),
    )
    journal.record_order_outcome(
        entry_id, "filled", fill_price=100.0, fill_qty=10,
        fill_at=_T0 - dt.timedelta(days=1),
    )
    # Initial protective stop at 95.00 → initial risk 5.00 (1R), default
    # trail distance 5% of entry (ADR-011 self-calibrating default).
    insert_order(
        db_path,
        OrderRow(
            execution_id=eid, role="stop", symbol=symbol, side="sell",
            order_type="stop", qty=10, tif="gtc", stop_price=95.0,
            broker_order_id="b-stop-1",
            submitted_at=_T0 - dt.timedelta(days=1),
        ),
    )
    # Position open at broker AND in the journal (prior tick snapshot).
    fake_broker.positions = [
        Position(symbol=symbol, qty=10, avg_entry_price=100.0,
                 market_value=1100.0, unrealized_pnl=100.0)
    ]
    journal.insert_position(
        PositionRow(
            snapshot_at=_T0 - dt.timedelta(minutes=5), source="reconciler",
            symbol=symbol, qty=10, avg_entry_price=100.0,
            market_value=1000.0, unrealized_pnl=0.0,
        )
    )
    # Winner: last=110 ≥ entry + 1.0R (105) → engage; high-water 110;
    # target = 110 × (1 − 5%) = 104.50 > 95 × 1.0025 → ratchet fires.
    fake_broker.quote_last = 110.0

    ks = KillSwitch(journal, now_fn=clock)
    ticker = ExitPolicyTicker(journal=journal, broker=fake_broker, now_fn=clock)
    rec = Reconciler(
        fake_broker,
        journal,
        ks,
        ReconcilerConfig(interval_seconds_market=300),
        alerter=_FakeAlerter(),
        now_fn=clock,
        fill_ingestor=FillIngestor(
            broker=fake_broker, journal=journal, now_fn=clock
        ),
        exit_policy_ticker=ticker,
    )

    # One loop iteration: the first tick fires immediately on start.
    await rec.start()
    await asyncio.sleep(0.05)
    await rec.stop()

    # The matched position raised no halt.
    assert not ks.is_halted()
    # Broker PATCH happened through the seam, at the computed target.
    assert fake_broker.replaced_stops == [
        {
            "old_id": "b-stop-1",
            "new_stop_price": 104.5,
            "client_order_id": f"trail-exec-{eid}-{_T0.timestamp():.0f}",
        }
    ]
    # §3.3-shaped journal chain: old stop completed 'replaced', exactly
    # one live trailing_stop row at the new price.
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        old = conn.execute(
            "SELECT final_status FROM orders WHERE broker_order_id = 'b-stop-1'"
        ).fetchone()
        trail = conn.execute(
            "SELECT stop_price, final_status, tif FROM orders "
            "WHERE role = 'trailing_stop' AND execution_id = ?", (eid,)
        ).fetchall()
    assert old["final_status"] == "replaced"
    assert len(trail) == 1
    assert trail[0]["stop_price"] == 104.5
    assert trail[0]["final_status"] is None
    assert trail[0]["tif"] == "gtc"
    # Persisted trail state: engaged at the observed high-water.
    state = get_exit_policy_state(db_path, execution_id=eid)
    assert state is not None
    assert state.trail_engaged
    assert state.high_water_mark == 110.0
    assert state.trail_distance_pct == 5.0

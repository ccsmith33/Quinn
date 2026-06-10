"""WS1 — end-to-end position-truth cycle (D-078, delta §2; task #2 ACs).

Integration through REAL JournalRepo + REAL KillSwitch + FillIngestor
wired into the Reconciler, fake broker only. Proves the four halt classes
die end-to-end:

  - ghost-position tombstone cycle (RC-1): fill recorded → tombstone →
    next tick clean match, kill-switch stays active;
  - thesis_close reconciliation (RC-3): an LLM-driven close explains its
    diff while pending, any role;
  - halt dedupe (O-5 amplifier): a persisting genuine diff inserts ONE
    halt row and pages once across many ticks; resume → recurrence pages
    again;
  - KS-3 revival (O-7): recorded fills make realized P&L computable.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

import pytest

from broker.protocol import AccountSnapshot, Position, SubmittedOrder
from config.loader import ReconcilerConfig
from journal.migrate import apply_migrations
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

    def advance(self, **kwargs: float) -> None:
        self.now += dt.timedelta(**kwargs)


class _FakeBroker:
    def __init__(self) -> None:
        self.positions: list[Position] = []
        self.orders_by_id: dict[str, SubmittedOrder] = {}

    def get_positions(self) -> list[Position]:
        return list(self.positions)

    def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(
            equity=10_000.0,
            cash=10_000.0,
            buying_power=10_000.0,
            long_market_value=0.0,
            daypl=0.0,
            snapshot_at=_T0,
        )

    def get_order_by_id(self, broker_order_id: str) -> SubmittedOrder | None:
        return self.orders_by_id.get(broker_order_id)


class _FakeAlerter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def notify(self, message: str) -> None:
        self.calls.append(message)


def _seed_execution(db_path: str, symbol: str) -> int:
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


def _journal_order(
    repo: JournalRepo,
    eid: int,
    *,
    symbol: str,
    role: str,
    side: str,
    qty: int,
    broker_order_id: str,
    submitted_at: dt.datetime,
) -> int:
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
            submitted_at=submitted_at,
        ),
    )


def _broker_fill(
    broker_order_id: str,
    *,
    symbol: str,
    side: str,
    qty: int,
    price: float,
    filled_at: dt.datetime,
    status: str = "filled",
) -> SubmittedOrder:
    return SubmittedOrder(
        broker_order_id=broker_order_id,
        client_order_id=f"cid-{broker_order_id}",
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        qty=qty,
        order_type="market",
        status=status,  # type: ignore[arg-type]
        submitted_at=filled_at - dt.timedelta(days=1),
        filled_avg_price=price,
        filled_qty=qty if status == "filled" else 0,
        filled_at=filled_at if status == "filled" else None,
    )


def _position(symbol: str, qty: int, avg: float = 10.0) -> Position:
    return Position(
        symbol=symbol,
        qty=qty,
        avg_entry_price=avg,
        market_value=qty * avg,
        unrealized_pnl=0.0,
    )


@pytest.fixture
def rig(
    tmp_path: Path,
) -> tuple[JournalRepo, _FakeBroker, KillSwitch, _FakeAlerter, Reconciler, _Clock]:
    db_path = tmp_path / "journal.db"
    apply_migrations(str(db_path))
    repo = JournalRepo(str(db_path))
    broker = _FakeBroker()
    clock = _Clock(_T0)
    ks = KillSwitch(repo, now_fn=clock)
    alerter = _FakeAlerter()
    ingestor = FillIngestor(broker=broker, journal=repo, now_fn=clock)
    rec = Reconciler(
        broker,
        repo,
        ks,
        ReconcilerConfig(interval_seconds_market=300),
        alerter=alerter,
        now_fn=clock,
        fill_ingestor=ingestor,
    )
    return repo, broker, ks, alerter, rec, clock


def test_ghost_position_tombstone_cycle(rig) -> None:
    """RC-1 end-to-end: open position → broker stop fill → same tick
    records the fill, tombstones the symbol, reports a clean match — and
    the kill-switch NEVER halts. Under pre-WS1 code this was a permanent
    ghost and a 5-minute halt loop after window expiry."""
    repo, broker, ks, alerter, rec, clock = rig
    eid = _seed_execution(repo.db_path, "ACET")
    _journal_order(
        repo, eid, symbol="ACET", role="stop", side="sell", qty=30,
        broker_order_id="b-stop", submitted_at=_T0 - dt.timedelta(days=9),
    )
    # Open at broker + journal snapshot (an earlier reconcile tick).
    broker.positions = [_position("ACET", 30, avg=8.20)]
    repo.insert_position(
        PositionRow(
            snapshot_at=_T0 - dt.timedelta(minutes=5),
            source="reconciler",
            symbol="ACET",
            qty=30,
            avg_entry_price=8.20,
            market_value=246.0,
            unrealized_pnl=0.0,
        )
    )

    # Stop fills at the broker: position gone, order terminal.
    broker.positions = []
    broker.orders_by_id["b-stop"] = _broker_fill(
        "b-stop", symbol="ACET", side="sell", qty=30, price=8.0,
        filled_at=clock.now - dt.timedelta(seconds=60),
    )
    clock.advance(minutes=5)

    report = rec.reconcile_now()

    assert report.matched is True
    assert report.diffs == []
    assert ks.is_halted() is False
    # Fill recorded.
    [order] = repo.get_orders_since(symbol="ACET", since=dt.datetime(2026, 1, 1))
    assert order.final_status == "filled"
    assert order.realized_fill_price == 8.0
    # Symbol tombstoned out of the journal's open view.
    assert repo.get_open_positions() == []
    # Many further ticks: still clean, still no halt (loop is dead).
    for _ in range(5):
        clock.advance(minutes=5)
        report = rec.reconcile_now()
        assert report.matched is True
    assert ks.is_halted() is False
    assert alerter.calls == []


def test_thesis_close_pending_explains_decrease(rig) -> None:
    """RC-3 end-to-end: position vanishes from the broker book while the
    journaled thesis_close sell is still non-terminal at the order
    endpoint (the INSG-style fill-visibility race). The lifecycle
    classifier explains the decrease — role thesis_close included — and
    nothing halts."""
    repo, broker, ks, alerter, rec, clock = rig
    eid = _seed_execution(repo.db_path, "QNST")
    _journal_order(
        repo, eid, symbol="QNST", role="thesis_close", side="sell", qty=19,
        broker_order_id="b-close", submitted_at=_T0 - dt.timedelta(minutes=30),
    )
    repo.insert_position(
        PositionRow(
            snapshot_at=_T0 - dt.timedelta(minutes=5),
            source="reconciler",
            symbol="QNST",
            qty=19,
            avg_entry_price=12.60,
            market_value=239.4,
            unrealized_pnl=0.0,
        )
    )
    broker.positions = []  # already gone from the positions endpoint
    broker.orders_by_id["b-close"] = _broker_fill(
        "b-close", symbol="QNST", side="sell", qty=19, price=12.0,
        filled_at=clock.now, status="partially_filled",  # non-terminal
    )
    clock.advance(minutes=5)

    report = rec.reconcile_now()

    assert report.matched is True
    assert report.diffs == []
    assert len(report.explained_diffs) == 1
    assert report.explained_diffs[0].diff.symbol == "QNST"
    assert ks.is_halted() is False
    assert alerter.calls == []


def test_genuine_diff_halts_once_not_every_tick(rig) -> None:
    """Dedupe end-to-end (O-5): a genuinely unexplained non-zero mismatch
    halts and pages exactly ONCE while it persists, instead of a row +
    page every 300 s. After an operator resume, recurrence pages again
    (resumes are not free passes). Verification-only ≠ rubber-stamp: the
    halt itself must still fire."""
    repo, broker, ks, alerter, rec, clock = rig
    # Broker says 100, journal says 50, no orders at all → unexpected.
    broker.positions = [_position("XYZ", 100)]
    repo.insert_position(
        PositionRow(
            snapshot_at=_T0 - dt.timedelta(minutes=5),
            source="reconciler",
            symbol="XYZ",
            qty=50,
            avg_entry_price=10.0,
            market_value=500.0,
            unrealized_pnl=0.0,
        )
    )

    first = rec.reconcile_now()
    assert first.matched is False
    assert ks.is_halted() is True
    assert len(alerter.calls) == 1

    # Ten more ticks with the identical diff: no new rows, no new pages.
    # (The reconciler's own snapshot write self-heals the qty mismatch on
    # the next tick, so freeze the journal view by re-inserting the stale
    # snapshot each tick — modelling a diff that genuinely persists.)
    for _ in range(10):
        clock.advance(minutes=5)
        repo.insert_position(
            PositionRow(
                snapshot_at=clock.now - dt.timedelta(seconds=1),
                source="operator_repair",
                symbol="XYZ",
                qty=50,
                avg_entry_price=10.0,
                market_value=500.0,
                unrealized_pnl=0.0,
            )
        )
        rec.reconcile_now()
    with sqlite3.connect(repo.db_path) as conn:
        halt_rows, = conn.execute(
            "SELECT COUNT(*) FROM kill_switch_state WHERE state='halted'"
        ).fetchone()
    assert halt_rows == 1
    assert len(alerter.calls) == 1

    # Operator resumes; identical diff recurs → fresh halt + page.
    clock.advance(minutes=5)
    ks.resume(set_by="operator", notes="investigating")
    clock.advance(minutes=5)
    repo.insert_position(
        PositionRow(
            snapshot_at=clock.now - dt.timedelta(seconds=1),
            source="operator_repair",
            symbol="XYZ",
            qty=50,
            avg_entry_price=10.0,
            market_value=500.0,
            unrealized_pnl=0.0,
        )
    )
    rec.reconcile_now()
    assert ks.is_halted() is True
    assert len(alerter.calls) == 2


def test_recorded_fills_make_realized_pnl_computable(rig) -> None:
    """O-7 / KS-3 revival: once entry and exit fills are recorded, the
    journal alone yields the closed trade's realized P&L — the input
    KS-3 (consecutive-loss auto-halt) has been starving on since launch."""
    repo, broker, ks, alerter, rec, clock = rig
    eid = _seed_execution(repo.db_path, "AORT")
    _journal_order(
        repo, eid, symbol="AORT", role="entry", side="buy", qty=10,
        broker_order_id="b-entry", submitted_at=_T0 - dt.timedelta(days=1),
    )
    _journal_order(
        repo, eid, symbol="AORT", role="stop", side="sell", qty=10,
        broker_order_id="b-stop", submitted_at=_T0 - dt.timedelta(days=1),
    )
    broker.orders_by_id["b-entry"] = _broker_fill(
        "b-entry", symbol="AORT", side="buy", qty=10, price=25.0,
        filled_at=_T0 - dt.timedelta(days=1),
    )
    broker.orders_by_id["b-stop"] = _broker_fill(
        "b-stop", symbol="AORT", side="sell", qty=10, price=24.0,
        filled_at=clock.now - dt.timedelta(seconds=30),
    )

    rec.reconcile_now()

    closes = repo.get_closed_trade_pnls_chronological()
    assert len(closes) == 1
    closed_eid, pnl, _closed_at = closes[0]
    assert closed_eid == eid
    assert pnl == pytest.approx(-10.0)  # 10 × (24 − 25)


def test_manual_flatten_with_live_gtc_stop_absorbed(rig) -> None:
    """W1-M1 end-to-end (the soak's own RC-4 scenario): operator flattens
    RGR in the Alpaca UI while its GTC stop is live — DELETE /v2/positions
    does not cancel open orders. The FillIngestor polls the stop every
    tick and finds it still pending, so 3 consecutive absent ticks cannot
    be fill latency: tombstone + ONE alert naming the live stop for
    operator cancellation; never a halt; phantom thesis reviews stop."""
    repo, broker, ks, alerter, rec, clock = rig
    eid = _seed_execution(repo.db_path, "RGR")
    oid = _journal_order(
        repo, eid, symbol="RGR", role="stop", side="sell", qty=6,
        broker_order_id="b-stop-rgr", submitted_at=_T0 - dt.timedelta(days=5),
    )
    repo.insert_position(
        PositionRow(
            snapshot_at=_T0 - dt.timedelta(minutes=5),
            source="reconciler",
            symbol="RGR",
            qty=6,
            avg_entry_price=39.29,
            market_value=235.74,
            unrealized_pnl=0.0,
        )
    )
    # Manual flatten: position gone; the stop stays live (non-terminal)
    # at the order endpoint, tick after tick.
    broker.positions = []
    broker.orders_by_id["b-stop-rgr"] = _broker_fill(
        "b-stop-rgr", symbol="RGR", side="sell", qty=6, price=39.0,
        filled_at=clock.now, status="accepted",
    )

    # Ticks 1-2: grace — explained, quiet.
    for _ in range(2):
        clock.advance(minutes=5)
        report = rec.reconcile_now()
        assert report.matched is True
        assert report.external_closes == []
        assert alerter.calls == []
        assert ks.is_halted() is False

    # Tick 3: masked absence confirmed — tombstone + single alert naming
    # the live order, no halt.
    clock.advance(minutes=5)
    report = rec.reconcile_now()
    assert report.matched is True
    assert report.external_closes == ["RGR"]
    assert ks.is_halted() is False
    assert len(alerter.calls) == 1
    assert "b-stop-rgr" in alerter.calls[0]
    assert "cancel" in alerter.calls[0].lower()
    assert str(oid) in alerter.calls[0]
    # The journal self-healed: ghost gone, thesis guard sees closed.
    assert repo.get_open_positions() == []
    from journal.repo import has_open_position
    assert has_open_position(repo.db_path, "RGR") is False
    # The stop row stays PENDING (we never invent an outcome) — the
    # operator cancels it; fill ingestion will record 'canceled' then.
    assert [r.id for r in repo.get_orders_pending_fill()] == [oid]

    # Tick 4: clean match, no repeat alert, still no halt.
    clock.advance(minutes=5)
    report = rec.reconcile_now()
    assert report.matched is True
    assert len(alerter.calls) == 1
    assert ks.is_halted() is False

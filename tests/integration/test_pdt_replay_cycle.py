# PDT-SUNSET-2026-06-04: ADR-009 §"Pre-market deferred replay" — end-to-end.
"""S-PDT-5 AC-8 — full PDT replay cycle integration test.

Flow:
  1. Entry under PDT mode → 1 entry order + 2 active virtual exits.
  2. Set price below stop → scanner ticks; budget=0 forces all sells
     into deferred_sells with deferred_reason='ev_lost'.
  3. Simulate next-day boot: replayer.run() drains the prior-day
     deferred row. Position-closed test path: leave position open.
  4. Assert: broker received the sell submit; deferred_sells.replayed_at
     set; replay_broker_order_id populated.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from typing import Any

from broker.protocol import (
    AccountSnapshot,
    OrderRequest,
    Position,
    Quote,
    SubmittedOrder,
)
from execution.orders import AcceptedProposal, OrderSubmitter
from execution.pdt_budget import PDTState
from execution.virtual_exits import DeferredSellReplayer, VirtualExitScanner
from journal.migrate import apply_migrations
from journal.models import FilingRow, PromptRow, ProposalRow
from journal.repo import JournalRepo, insert_filing, insert_prompt, insert_proposal
from proposal.schemas import TradeProposal


class _StubBroker:
    """Mutable broker fake. Tracks all submits, exposes mutable price
    and daytrade_count for cross-tick simulations."""

    def __init__(self) -> None:
        self.submitted: list[OrderRequest] = []
        self._seq = 0
        self.last_price: float = 100.0
        self.daytrade_count: int = 0
        self.position_qty: int = 0

    def submit_order(self, req: OrderRequest) -> SubmittedOrder:
        self._seq += 1
        self.submitted.append(req)
        if req.side == "buy":
            self.position_qty += req.qty
        else:
            self.position_qty -= req.qty
        return SubmittedOrder(
            broker_order_id=f"alp-{self._seq}",
            client_order_id=req.client_order_id,
            symbol=req.symbol, side=req.side, qty=req.qty,
            order_type=req.order_type, status="accepted",
            submitted_at=dt.datetime(2026, 5, 8, 9, 0, tzinfo=dt.UTC),
            limit_price=req.limit_price, stop_price=req.stop_price,
        )

    def get_quote(self, symbol: str) -> Quote:
        return Quote(
            symbol=symbol, bid=self.last_price - 0.05,
            ask=self.last_price + 0.05, last=self.last_price,
            ts=dt.datetime(2026, 5, 7, 14, 30, tzinfo=dt.UTC),
        )

    def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(
            equity=22_300.0, cash=5_000.0, buying_power=44_600.0,
            long_market_value=17_300.0, daypl=0.0,
            snapshot_at=dt.datetime(2026, 5, 7, 14, 30, tzinfo=dt.UTC),
            last_equity=22_000.0, daytrade_count=self.daytrade_count,
        )

    def get_positions(self) -> list[Position]:
        if self.position_qty == 0:
            return []
        return [Position(
            symbol="AAPL", qty=self.position_qty, avg_entry_price=100.0,
            market_value=self.position_qty * self.last_price,
            unrealized_pnl=0.0,
        )]

    def get_order_by_client_id(self, *_: Any, **__: Any) -> Any:
        return None

    def cancel_order(self, *_: Any) -> None: ...


class _StubKS:
    def halt(self, *_: Any, **__: Any) -> None: ...


def _seed_proposal(db_path: str) -> int:
    insert_prompt(db_path, PromptRow(
        prompt_version="sonnet_v1@deadbeef", name="sonnet_v1",
        file_path="src/prompts/sonnet_v1.txt", content_hash="deadbeef",
    ))
    fid = insert_filing(db_path, FilingRow(
        accession_number="0001234567-26-000777", cik=320193, form_type="8-K",
        filed_at=dt.datetime(2026, 5, 7, 14, 0),
        fetched_at=dt.datetime(2026, 5, 7, 14, 1),
        raw_text_path="/tmp/raw.txt", content_hash="abc",
        item_codes='["2.02"]', issuer_ticker="AAPL",
    ))
    return insert_proposal(db_path, ProposalRow(
        filing_id=fid, decision_id="dec-pdt-replay-1",
        model_id="claude-sonnet-4-6",
        prompt_version="sonnet_v1@deadbeef",
        raw_response="{}", kind="trade_proposal",
        symbol="AAPL", direction="long", size_pct_requested=0.05,
        conviction=8, thesis="thesis text",
        input_tokens=10, output_tokens=20, latency_ms=100, cost_usd=0.001,
    ))


def _proposal() -> TradeProposal:
    return TradeProposal.model_validate({
        "symbol": "AAPL", "direction": "long",
        "size_pct_of_capital": 0.05, "entry_style": "market_open",
        "stop_loss_price": 95.0, "take_profit_price": 110.0,
        "time_horizon_days": 10, "conviction": 8,
        "thesis": "Strong material 8-K disclosure indicating near-term catalyst.",
        "signals": ["8-K item 2.02 strong earnings beat"],
        "exit_conditions": ["thesis breaks; stop hit; 30 days elapsed"],
        "risk_factors": ["macro risk; sector rotation; news risk"],
    })


def test_pdt_full_replay_cycle(tmp_path: Path) -> None:
    """AC-8: entry → scanner defers (budget=0) → next-day replayer
    drains the deferred row → broker submitted; replayed_at set."""
    db_path = str(tmp_path / "journal.db")
    apply_migrations(db_path)

    pid = _seed_proposal(db_path)
    journal = JournalRepo(db_path)
    broker = _StubBroker()
    pdt_state = PDTState(_active=True, _last_equity=22_000.0)

    # --- Day 1: entry under PDT mode ---
    accepted = AcceptedProposal(
        proposal=_proposal(), proposal_id=pid, qty=5,
        realized_dollar_size=500.0, realized_pct=0.05,
        realized_dollar_size_request=500.0,
    )
    OrderSubmitter().submit(
        accepted, broker, journal, _StubKS(), pdt_state=pdt_state,
    )

    # --- Day 1: stop crosses; budget=0 forces defer ---
    broker.last_price = 94.50  # stop=95 crossed
    broker.daytrade_count = 3  # exhausted budget → all deferred
    scanner = VirtualExitScanner(
        broker=broker, journal=journal, pdt_state=pdt_state,
    )
    scan_report = scanner.run_tick()
    assert scan_report.ready == 1
    assert scan_report.submitted == 0
    assert scan_report.deferred_ev == 1

    # The scanner deferred one row.
    with sqlite3.connect(db_path) as conn:
        unreplayed = conn.execute(
            "SELECT COUNT(*) FROM deferred_sells WHERE replayed_at IS NULL"
        ).fetchone()[0]
        assert unreplayed == 1

    # Capture broker submission count after scan (entry only — no sells).
    submits_before_replay = len(broker.submitted)
    assert submits_before_replay == 1  # entry buy only

    # --- Day 2 boot: deferred_at < today ET ---
    # The scanner ran on 2026-05-07 (today_fn default is now_utc;
    # deferred_at was inserted with CURRENT_TIMESTAMP which is UTC).
    # Set the replayer's now_fn to a date AFTER the deferral so the
    # row is "yesterday's".
    def _next_day_premarket() -> dt.datetime:
        # 2026-05-09 09:00 UTC = 05:00 ET (well after the deferral).
        return dt.datetime(2026, 5, 9, 9, 0, tzinfo=dt.UTC)

    broker.daytrade_count = 0  # new session, new budget
    replayer = DeferredSellReplayer(
        broker=broker, journal=journal, now_fn=_next_day_premarket,
    )
    report = replayer.run()
    assert report.replayed == 1
    assert report.skipped_today == 0
    assert report.skipped_no_position == 0
    assert report.skipped_error == 0

    # Broker received the replay submit.
    assert len(broker.submitted) == submits_before_replay + 1
    replay_req = broker.submitted[-1]
    assert replay_req.side == "sell"
    assert replay_req.client_order_id.startswith("pdt-replay-")

    # --- Verify journal state ---
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT replayed_at, replay_broker_order_id, virtual_exit_id "
            "FROM deferred_sells"
        ).fetchone()
        assert row["replayed_at"] is not None
        assert row["replay_broker_order_id"] is not None
        assert row["replay_broker_order_id"].startswith("alp-")
        # The unreplayed queue is now empty.
        unreplayed = conn.execute(
            "SELECT COUNT(*) FROM deferred_sells WHERE replayed_at IS NULL"
        ).fetchone()[0]
        assert unreplayed == 0
        # PDT-SUNSET-2026-06-04: FINDING-3 closure — the source
        # virtual_exit must transition from 'active' to 'submitted' so
        # the next scanner tick won't re-pick it via
        # list_active_virtual_exits().
        v_state = journal.get_virtual_exit_state(row["virtual_exit_id"])
        assert v_state == "submitted"

        # HOTFIX-2026-05-08: replayer must journal the sell to the orders
        # table so the reconciler tolerance can explain the broker-position
        # decrease. The orders row's broker_order_id matches
        # deferred_sells.replay_broker_order_id.
        sell_orders = conn.execute(
            "SELECT role, symbol, side, order_type, qty, broker_order_id "
            "FROM orders WHERE side='sell' ORDER BY id ASC"
        ).fetchall()
        assert len(sell_orders) == 1
        sell = sell_orders[0]
        assert sell["role"] == "stop"
        assert sell["symbol"] == "AAPL"
        assert sell["side"] == "sell"
        assert sell["order_type"] == "market"
        assert sell["qty"] == 5
        assert sell["broker_order_id"] == row["replay_broker_order_id"]


def test_pdt_replayer_skips_row_superseded_by_same_session_submit(
    tmp_path: Path,
) -> None:
    """Race: scanner defers virtual_exit V at tick T1 (writes deferred_sells D);
    at T2 budget frees up, scanner picks V again, submits → V.state='submitted'.
    Next-session replayer sees D unreplayed but V already submitted →
    skips D with SUPERSEDED:submitted; broker receives no extra sell.
    """
    db_path = str(tmp_path / "journal.db")
    apply_migrations(db_path)

    pid = _seed_proposal(db_path)
    journal = JournalRepo(db_path)
    broker = _StubBroker()
    pdt_state = PDTState(_active=True, _last_equity=22_000.0)

    # Entry under PDT.
    accepted = AcceptedProposal(
        proposal=_proposal(), proposal_id=pid, qty=5,
        realized_dollar_size=500.0, realized_pct=0.05,
        realized_dollar_size_request=500.0,
    )
    OrderSubmitter().submit(
        accepted, broker, journal, _StubKS(), pdt_state=pdt_state,
    )

    # T1: stop crosses, budget=0 → defer.
    broker.last_price = 94.50
    broker.daytrade_count = 3
    scanner = VirtualExitScanner(
        broker=broker, journal=journal, pdt_state=pdt_state,
    )
    r1 = scanner.run_tick()
    assert r1.deferred_ev == 1

    # T2 (same session): budget frees up; scanner re-evaluates and
    # submits the same virtual_exit. virtual_exits.state → 'submitted'.
    broker.daytrade_count = 0
    r2 = scanner.run_tick()
    assert r2.submitted == 1

    # virtual_exit row is now 'submitted'; deferred_sells still unreplayed.
    with sqlite3.connect(db_path) as conn:
        ve_state = conn.execute(
            "SELECT state FROM virtual_exits WHERE role='stop'"
        ).fetchone()[0]
        assert ve_state == "submitted"
        unreplayed_before = conn.execute(
            "SELECT COUNT(*) FROM deferred_sells WHERE replayed_at IS NULL"
        ).fetchone()[0]
        assert unreplayed_before == 1

    submits_before_replay = len(broker.submitted)

    # Next-day replayer: must skip the deferred row (would short an
    # already-closed-at-broker position otherwise).
    def _next_day_premarket() -> dt.datetime:
        return dt.datetime(2026, 5, 9, 9, 0, tzinfo=dt.UTC)

    replayer = DeferredSellReplayer(
        broker=broker, journal=journal, now_fn=_next_day_premarket,
    )
    report = replayer.run()
    assert report.replayed == 0
    assert report.skipped_superseded == 1
    assert report.skipped_no_position == 0
    # No extra broker submission — the supersede check fired.
    assert len(broker.submitted) == submits_before_replay

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT replayed_at, replay_broker_order_id, notes "
            "FROM deferred_sells"
        ).fetchone()
        # Row drained from the unreplayed queue with a SUPERSEDED note.
        assert row[0] is not None  # replayed_at
        assert row[1] is None      # replay_broker_order_id stays NULL
        assert "SUPERSEDED:submitted" in (row[2] or "")
        unreplayed_after = conn.execute(
            "SELECT COUNT(*) FROM deferred_sells WHERE replayed_at IS NULL"
        ).fetchone()[0]
        assert unreplayed_after == 0

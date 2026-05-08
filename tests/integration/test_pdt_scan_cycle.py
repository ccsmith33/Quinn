# PDT-SUNSET-2026-06-04: ADR-009 §"Scanner integration" — full scan cycle.
"""S-PDT-4 AC-10 — integration test: entry under PDT mode → scanner
tick crosses stop → broker sell submitted; virtual_exit transitions
to state='submitted' with submitted_broker_order_id set.
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
from execution.virtual_exits import VirtualExitScanner
from journal.migrate import apply_migrations
from journal.models import FilingRow, PromptRow, ProposalRow
from journal.repo import JournalRepo, insert_filing, insert_prompt, insert_proposal
from proposal.schemas import TradeProposal


class _StubBroker:
    """Tracks all submitted orders. Quote and account are mutable so the
    test can drive a stop-cross between entry and scanner tick."""

    def __init__(self) -> None:
        self.submitted: list[OrderRequest] = []
        self._seq = 0
        self.last_price: float = 100.0
        self.daytrade_count: int = 0
        self.position_qty: int = 0  # set after entry "fills"

    def submit_order(self, req: OrderRequest) -> SubmittedOrder:
        self._seq += 1
        self.submitted.append(req)
        # Simulate fill: any buy fills the position; sell zeroes it.
        if req.side == "buy":
            self.position_qty += req.qty
        else:
            self.position_qty -= req.qty
        return SubmittedOrder(
            broker_order_id=f"alp-{self._seq}",
            client_order_id=req.client_order_id,
            symbol=req.symbol, side=req.side, qty=req.qty,
            order_type=req.order_type, status="accepted",
            submitted_at=dt.datetime(2026, 5, 7, 14, 30, tzinfo=dt.UTC),
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

    def cancel_order(self, *_: Any) -> None: ...


class _StubKS:
    def halt(self, *_: Any, **__: Any) -> None: ...


def _seed_proposal(db_path: str) -> int:
    insert_prompt(db_path, PromptRow(
        prompt_version="sonnet_v1@deadbeef", name="sonnet_v1",
        file_path="src/prompts/sonnet_v1.txt", content_hash="deadbeef",
    ))
    fid = insert_filing(db_path, FilingRow(
        accession_number="0001234567-26-000888", cik=320193, form_type="8-K",
        filed_at=dt.datetime(2026, 5, 7, 14, 0),
        fetched_at=dt.datetime(2026, 5, 7, 14, 1),
        raw_text_path="/tmp/raw.txt", content_hash="abc",
        item_codes='["2.02"]', issuer_ticker="AAPL",
    ))
    return insert_proposal(db_path, ProposalRow(
        filing_id=fid, decision_id="dec-pdt-scan-1",
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


def test_pdt_full_cycle_entry_then_scanner_submits_stop(tmp_path: Path) -> None:
    """AC-10: entry under PDT mode writes 1 entry order + 2 virtual
    exits; price drops below stop; scanner submits ONE broker sell;
    virtual_exit.state transitions to 'submitted' with broker_order_id set.
    """
    db_path = str(tmp_path / "journal.db")
    apply_migrations(db_path)

    pid = _seed_proposal(db_path)
    journal = JournalRepo(db_path)
    broker = _StubBroker()
    pdt_state = PDTState(_active=True, _last_equity=22_000.0)

    # --- Entry under PDT mode ---
    accepted = AcceptedProposal(
        proposal=_proposal(), proposal_id=pid, qty=5,
        realized_dollar_size=500.0, realized_pct=0.05,
        realized_dollar_size_request=500.0,
    )
    submitter = OrderSubmitter()
    submitter.submit(accepted, broker, journal, _StubKS(), pdt_state=pdt_state)

    # Sanity: 1 broker submit (entry only), 2 virtual exits active.
    assert len(broker.submitted) == 1
    with sqlite3.connect(db_path) as conn:
        active = conn.execute(
            "SELECT COUNT(*) FROM virtual_exits WHERE state='active'"
        ).fetchone()[0]
        assert active == 2

    # --- Price drops below stop_loss_price=95 ---
    broker.last_price = 94.50  # crosses stop=95 (current <= stop)

    # --- Scanner tick ---
    scanner = VirtualExitScanner(
        broker=broker, journal=journal, pdt_state=pdt_state,
    )
    report = scanner.run_tick()

    # Stop crossed; tp NOT crossed (tp_price=110, current=94.50). Budget=3.
    assert report.ready == 1
    assert report.submitted == 1
    assert report.deferred_ev == 0
    assert report.deferred_403 == 0

    # Broker received a sell order for the stop.
    sells = [r for r in broker.submitted if r.side == "sell"]
    assert len(sells) == 1
    sell = sells[0]
    assert sell.order_type == "market"  # market sell post-cross
    assert sell.qty == 5
    assert sell.tif == "day"
    assert sell.client_order_id.startswith("pdt-vexit-")

    # virtual_exits transitions: stop row → submitted; tp stays active.
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT role, state, submitted_broker_order_id, submitted_at "
            "FROM virtual_exits ORDER BY id ASC"
        ).fetchall()
        assert len(rows) == 2
        stop_row = next(r for r in rows if r["role"] == "stop")
        tp_row = next(r for r in rows if r["role"] == "tp")
        assert stop_row["state"] == "submitted"
        assert stop_row["submitted_broker_order_id"] is not None
        assert stop_row["submitted_broker_order_id"].startswith("alp-")
        assert stop_row["submitted_at"] is not None
        # TP not crossed → still active.
        assert tp_row["state"] == "active"

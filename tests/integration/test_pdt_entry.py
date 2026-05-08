# PDT-SUNSET-2026-06-04: ADR-009 §6 — entry path under pdt_mode=True.
"""S-PDT-3 AC-8 — integration test for the PDT entry branch.

Real journal DB (sqlite tmpfile) with migrations 001-004 applied; stub
broker that accepts the entry. With `pdt_state.is_active()==True`,
verify that:
  - one `orders` row with role='entry' is written;
  - zero `orders` rows with role='stop' or 'take_profit';
  - two `virtual_exits` rows in state='active' (one stop + one TP);
  - one `executions` row with decision='accepted'.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from typing import Any

from broker.protocol import OrderRequest, Quote, SubmittedOrder
from execution.orders import AcceptedProposal, OrderSubmitter, SubmissionAccepted
from journal.migrate import apply_migrations
from journal.models import FilingRow, PromptRow, ProposalRow
from journal.repo import JournalRepo, insert_filing, insert_prompt, insert_proposal
from proposal.schemas import TradeProposal


class _StubBroker:
    """Accepts entries; would accept stop/TP if asked. The PDT branch
    must NOT call submit_order for the protective legs.

    Hotfix 2026-05-07: also implements `submit_bracket_order` so the
    non-PDT (bracket) regression path can run end-to-end through this
    integration's stub.
    """

    def __init__(self) -> None:
        self.submitted: list[OrderRequest] = []
        self.submitted_brackets: list[Any] = []

    def submit_order(self, req: OrderRequest) -> SubmittedOrder:
        self.submitted.append(req)
        return SubmittedOrder(
            broker_order_id=f"alp-{len(self.submitted)}",
            client_order_id=req.client_order_id,
            symbol=req.symbol,
            side=req.side,
            qty=req.qty,
            order_type=req.order_type,
            status="accepted",
            submitted_at=dt.datetime(2026, 5, 7, 14, 30, tzinfo=dt.UTC),
            limit_price=req.limit_price,
            stop_price=req.stop_price,
        )

    def submit_bracket_order(
        self, req: Any
    ) -> tuple[SubmittedOrder, SubmittedOrder, SubmittedOrder | None]:
        self.submitted_brackets.append(req)
        n = len(self.submitted_brackets)
        entry = SubmittedOrder(
            broker_order_id=f"brk-{n}-entry",
            client_order_id=req.entry_client_order_id,
            symbol=req.entry_symbol,
            side=req.entry_side,
            qty=req.entry_qty,
            order_type=req.entry_order_type,
            status="accepted",
            submitted_at=dt.datetime(2026, 5, 7, 14, 30, tzinfo=dt.UTC),
            limit_price=req.entry_limit_price,
            stop_price=None,
        )
        stop = SubmittedOrder(
            broker_order_id=f"brk-{n}-stop",
            client_order_id=f"{req.entry_client_order_id}:bracket-stop",
            symbol=req.entry_symbol,
            side="sell",
            qty=req.entry_qty,
            order_type="stop",
            status="accepted",
            submitted_at=dt.datetime(2026, 5, 7, 14, 30, tzinfo=dt.UTC),
            stop_price=req.stop_loss_price,
        )
        tp: SubmittedOrder | None = None
        if req.take_profit_price is not None:
            tp = SubmittedOrder(
                broker_order_id=f"brk-{n}-tp",
                client_order_id=f"{req.entry_client_order_id}:bracket-tp",
                symbol=req.entry_symbol,
                side="sell",
                qty=req.entry_qty,
                order_type="limit",
                status="accepted",
                submitted_at=dt.datetime(2026, 5, 7, 14, 30, tzinfo=dt.UTC),
                limit_price=req.take_profit_price,
            )
        return entry, stop, tp

    def get_quote(self, symbol: str) -> Quote:
        return Quote(
            symbol=symbol, bid=99.95, ask=100.05, last=100.00,
            ts=dt.datetime(2026, 5, 7, 14, 30, tzinfo=dt.UTC),
        )

    # Unused in this test path.
    def cancel_order(self, *_: Any) -> None: ...
    def get_account(self) -> Any: raise AssertionError
    def get_positions(self) -> list: return []


class _StubKS:
    def halt(self, *_: Any, **__: Any) -> None: ...


class _ActivePDTState:
    def is_active(self) -> bool:
        return True


def _seed_proposal(db_path: str) -> int:
    """Build the FK chain prompts → filings → proposals."""
    insert_prompt(db_path, PromptRow(
        prompt_version="sonnet_v1@deadbeef",
        name="sonnet_v1",
        file_path="src/prompts/sonnet_v1.txt",
        content_hash="deadbeef",
    ))
    fid = insert_filing(db_path, FilingRow(
        accession_number="0001234567-26-000999",
        cik=320193,
        form_type="8-K",
        filed_at=dt.datetime(2026, 5, 7, 14, 0),
        fetched_at=dt.datetime(2026, 5, 7, 14, 1),
        raw_text_path="/tmp/raw.txt",
        content_hash="abc",
        item_codes='["2.02"]',
        issuer_ticker="AAPL",
    ))
    pid = insert_proposal(db_path, ProposalRow(
        filing_id=fid,
        decision_id="dec-pdt-1",
        model_id="claude-sonnet-4-6",
        prompt_version="sonnet_v1@deadbeef",
        raw_response="{}",
        kind="trade_proposal",
        symbol="AAPL",
        direction="long",
        size_pct_requested=0.05,
        conviction=8,
        thesis="thesis text",
        input_tokens=10,
        output_tokens=20,
        latency_ms=100,
        cost_usd=0.001,
    ))
    return pid


def _build_proposal() -> TradeProposal:
    return TradeProposal.model_validate({
        "symbol": "AAPL",
        "direction": "long",
        "size_pct_of_capital": 0.05,
        "entry_style": "market_open",
        "stop_loss_price": 95.0,
        "take_profit_price": 110.0,
        "time_horizon_days": 10,
        "conviction": 8,
        "thesis": (
            "Strong material 8-K disclosure indicating near-term "
            "earnings catalyst per filing analysis."
        ),
        "signals": ["8-K item 2.02 strong earnings beat"],
        "exit_conditions": ["thesis breaks; stop hit; 30 days elapsed"],
        "risk_factors": ["macro risk; sector rotation; news risk"],
    })


def test_pdt_entry_writes_virtual_exits_and_no_protective_orders(
    tmp_path: Path,
) -> None:
    """AC-8: under PDT mode, the entry path writes 1 entry order, 0
    protective orders, 2 virtual exits, 1 execution(decision=accepted)."""
    db_path = str(tmp_path / "journal.db")
    apply_migrations(db_path)

    pid = _seed_proposal(db_path)
    accepted = AcceptedProposal(
        proposal=_build_proposal(),
        proposal_id=pid,
        qty=5,
        realized_dollar_size=500.0,
        realized_pct=0.05,
        realized_dollar_size_request=500.0,
    )

    journal = JournalRepo(db_path)
    broker = _StubBroker()
    submitter = OrderSubmitter()
    pdt_state = _ActivePDTState()

    result = submitter.submit(accepted, broker, journal, _StubKS(), pdt_state=pdt_state)

    assert isinstance(result, SubmissionAccepted)
    assert result.stop_broker_order_id is None
    assert result.take_profit_broker_order_id is None

    # Broker received exactly one submission — the entry buy.
    assert len(broker.submitted) == 1
    assert broker.submitted[0].side == "buy"

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        # AC-8 (a): one entry row.
        entry_count = conn.execute(
            "SELECT COUNT(*) FROM orders WHERE role='entry'"
        ).fetchone()[0]
        assert entry_count == 1
        # AC-8 (b): zero stop / take_profit rows.
        protective_count = conn.execute(
            "SELECT COUNT(*) FROM orders WHERE role IN ('stop','take_profit')"
        ).fetchone()[0]
        assert protective_count == 0
        # AC-8 (c): two active virtual exits.
        ve_count = conn.execute(
            "SELECT COUNT(*) FROM virtual_exits WHERE state='active'"
        ).fetchone()[0]
        assert ve_count == 2
        ve_rows = conn.execute(
            "SELECT role, stop_price, tp_price, entry_price, qty, "
            "execution_id FROM virtual_exits ORDER BY id ASC"
        ).fetchall()
        roles = sorted(r["role"] for r in ve_rows)
        assert roles == ["stop", "tp"]
        stop_row = next(r for r in ve_rows if r["role"] == "stop")
        tp_row = next(r for r in ve_rows if r["role"] == "tp")
        assert stop_row["stop_price"] == 95.0
        assert stop_row["tp_price"] is None
        assert tp_row["tp_price"] == 110.0
        assert tp_row["stop_price"] is None
        assert stop_row["entry_price"] == 100.0  # nbbo.last
        assert tp_row["entry_price"] == 100.0
        assert stop_row["qty"] == 5
        assert tp_row["qty"] == 5
        # AC-8 (d): one execution row, decision='accepted'.
        exec_rows = conn.execute(
            "SELECT decision FROM executions"
        ).fetchall()
        assert len(exec_rows) == 1
        assert exec_rows[0]["decision"] == "accepted"

    # The VirtualExitRow execution_id FK must point at the just-written
    # execution row (not a stale id).
    with sqlite3.connect(db_path) as conn:
        exec_id = conn.execute("SELECT id FROM executions").fetchone()[0]
        ve_exec_ids = conn.execute(
            "SELECT execution_id FROM virtual_exits"
        ).fetchall()
    assert all(r[0] == exec_id for r in ve_exec_ids)


def test_pdt_inactive_falls_through_to_existing_path(tmp_path: Path) -> None:
    """Regression: with `pdt_state.is_active()==False`, the existing
    broker pre-placement path runs (3 orders, 0 virtual exits)."""
    db_path = str(tmp_path / "journal.db")
    apply_migrations(db_path)

    pid = _seed_proposal(db_path)
    accepted = AcceptedProposal(
        proposal=_build_proposal(), proposal_id=pid, qty=5,
        realized_dollar_size=500.0, realized_pct=0.05,
        realized_dollar_size_request=500.0,
    )

    journal = JournalRepo(db_path)
    broker = _StubBroker()

    class _InactivePDTState:
        def is_active(self) -> bool:
            return False

    submitter = OrderSubmitter()
    submitter.submit(
        accepted, broker, journal, _StubKS(), pdt_state=_InactivePDTState()
    )

    with sqlite3.connect(db_path) as conn:
        order_count = conn.execute(
            "SELECT COUNT(*) FROM orders"
        ).fetchone()[0]
        assert order_count == 3
        ve_count = conn.execute(
            "SELECT COUNT(*) FROM virtual_exits"
        ).fetchone()[0]
        assert ve_count == 0

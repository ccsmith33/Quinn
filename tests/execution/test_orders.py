"""S6.4 — Order construction + Alpaca submission + journal write tests.

Test scope: pure unit tests with broker / journal / kill-switch fakes. The
real `JournalRepo` is exercised in one integration test to catch SQL drift
in the new `insert_execution` / `insert_order` JournalRepo bindings.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest

from broker.alpaca import BrokerUnavailable
from broker.protocol import BrokerRejected, OpenOrder, OrderRequest, Position, Quote, SubmittedOrder
from execution.orders import (
    AcceptedProposal,
    OrderSubmitter,
    SubmissionAccepted,
    SubmissionFailed,
)
from journal.migrate import apply_migrations
from journal.models import FilingRow, ProposalRow
from journal.repo import JournalRepo, get_orders_for_execution, insert_filing, insert_proposal
from proposal.schemas import TradeProposal

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeBroker:
    """Records every submit/get_quote call. By default returns success."""

    def __init__(
        self,
        *,
        quote: Quote | None = None,
        entry_raises: Exception | None = None,
        stop_raises: Exception | None = None,
        tp_raises: Exception | None = None,
    ) -> None:
        self._quote = quote or Quote(
            symbol="ACME",
            bid=10.00,
            ask=10.05,
            last=10.02,
            ts=dt.datetime(2026, 4, 28, 14, 30, tzinfo=dt.UTC),
        )
        self._entry_raises = entry_raises
        self._stop_raises = stop_raises
        self._tp_raises = tp_raises
        self.submitted: list[OrderRequest] = []
        self.quote_calls: int = 0

    def submit_order(self, req: OrderRequest) -> SubmittedOrder:
        self.submitted.append(req)
        # Order role is encoded in the request via order_type + side combo:
        # entry = buy/market or buy/limit; stop = sell/stop; tp = sell/limit.
        is_entry = req.side == "buy"
        is_stop = req.side == "sell" and req.order_type == "stop"
        is_tp = req.side == "sell" and req.order_type == "limit"
        if is_entry and self._entry_raises is not None:
            raise self._entry_raises
        if is_stop and self._stop_raises is not None:
            raise self._stop_raises
        if is_tp and self._tp_raises is not None:
            raise self._tp_raises
        return SubmittedOrder(
            broker_order_id=f"ord-{len(self.submitted)}",
            client_order_id=req.client_order_id,
            symbol=req.symbol,
            side=req.side,
            qty=req.qty,
            order_type=req.order_type,
            status="accepted",
            submitted_at=dt.datetime(2026, 4, 28, 14, 30, tzinfo=dt.UTC),
            limit_price=req.limit_price,
            stop_price=req.stop_price,
        )

    def cancel_order(self, broker_order_id: str) -> None:  # pragma: no cover
        pass

    def get_account(self):  # pragma: no cover
        raise AssertionError("submitter does not call get_account")

    def get_positions(self) -> list[Position]:  # pragma: no cover
        return []

    def get_open_orders(self) -> list[OpenOrder]:  # pragma: no cover
        return []

    def get_quote(self, symbol: str) -> Quote:
        self.quote_calls += 1
        return self._quote


class _FakeJournal:
    """In-memory journal substitute capturing inserts."""

    def __init__(self) -> None:
        self.executions: list[dict[str, Any]] = []
        self.orders: list[dict[str, Any]] = []
        self._exec_seq = 0
        self._order_seq = 0

    def insert_execution(self, row: Any) -> int:
        self._exec_seq += 1
        d = row.model_dump()
        d["id"] = self._exec_seq
        self.executions.append(d)
        return self._exec_seq

    def insert_order(self, row: Any) -> int:
        self._order_seq += 1
        d = row.model_dump()
        d["id"] = self._order_seq
        self.orders.append(d)
        return self._order_seq


class _FakeKillSwitch:
    def __init__(self) -> None:
        self.halts: list[tuple[str, str, str]] = []

    def halt(self, reason: str, set_by: str, notes: str = "") -> None:
        self.halts.append((reason, set_by, notes))


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _proposal(
    *,
    symbol: str = "ACME",
    entry_style: str = "market_open",
    entry_limit_price: float | None = None,
    take_profit_price: float | None = None,
) -> TradeProposal:
    payload: dict[str, Any] = {
        "symbol": symbol,
        "direction": "long",
        "size_pct_of_capital": 0.05,
        "entry_style": entry_style,
        "stop_loss_price": 8.50,
        "time_horizon_days": 10,
        "conviction": 8,
        "thesis": (
            "Strong fundamentals with material 8-K disclosure indicating "
            "near-term catalyst per filing analysis."
        ),
        "signals": ["8-K item 2.02 strong earnings beat"],
        "exit_conditions": ["thesis breaks; stop hit; 30 days elapsed"],
        "risk_factors": ["macro risk; sector rotation; news risk"],
    }
    if entry_limit_price is not None:
        payload["entry_limit_price"] = entry_limit_price
    if take_profit_price is not None:
        payload["take_profit_price"] = take_profit_price
    return TradeProposal.model_validate(payload)


def _accepted(
    *,
    proposal: TradeProposal | None = None,
    qty: int = 25,
    realized_dollar_size: float = 250.0,
    realized_pct: float = 0.05,
    realized_dollar_size_request: float = 250.0,
    proposal_id: int = 42,
) -> AcceptedProposal:
    return AcceptedProposal(
        proposal=proposal if proposal is not None else _proposal(),
        proposal_id=proposal_id,
        qty=qty,
        realized_dollar_size=realized_dollar_size,
        realized_pct=realized_pct,
        realized_dollar_size_request=realized_dollar_size_request,
    )


# ---------------------------------------------------------------------------
# Tests (story §test plan)
# ---------------------------------------------------------------------------

def test_market_entry_with_stop_and_tp_submitted() -> None:
    """Test #1 (first failing): market entry + stop + TP all submitted; rows persisted."""
    p = _proposal(take_profit_price=12.00)
    accepted = _accepted(proposal=p)
    broker = _FakeBroker()
    journal = _FakeJournal()
    ks = _FakeKillSwitch()
    submitter = OrderSubmitter()

    result = submitter.submit(accepted, broker, journal, ks)

    assert isinstance(result, SubmissionAccepted)
    # Three orders submitted: entry, stop, tp.
    assert len(broker.submitted) == 3
    sides = [(r.side, r.order_type) for r in broker.submitted]
    assert sides[0] == ("buy", "market")
    assert sides[1] == ("sell", "stop")
    assert sides[2] == ("sell", "limit")
    # One execution row, three order rows.
    assert len(journal.executions) == 1
    assert len(journal.orders) == 3
    exec_row = journal.executions[0]
    assert exec_row["decision"] == "accepted"
    assert exec_row["realized_dollar_size"] == 250.0
    assert exec_row["realized_size_pct"] == 0.05
    submitted_orders = json.loads(exec_row["submitted_orders_json"])
    assert {o["role"] for o in submitted_orders} == {"entry", "stop", "take_profit"}


def test_limit_entry_uses_entry_limit_price() -> None:
    """Test #2: entry_style=limit → limit BUY at entry_limit_price, TIF=day."""
    p = _proposal(entry_style="limit", entry_limit_price=9.95)
    accepted = _accepted(proposal=p)
    broker = _FakeBroker()
    journal = _FakeJournal()
    ks = _FakeKillSwitch()
    submitter = OrderSubmitter()

    result = submitter.submit(accepted, broker, journal, ks)

    assert isinstance(result, SubmissionAccepted)
    entry = broker.submitted[0]
    assert entry.order_type == "limit"
    assert entry.limit_price == 9.95
    assert entry.tif == "day"


def test_no_tp_when_take_profit_price_omitted() -> None:
    """Test #3: TP omitted in proposal → only entry + stop submitted."""
    p = _proposal(take_profit_price=None)
    accepted = _accepted(proposal=p)
    broker = _FakeBroker()
    journal = _FakeJournal()
    ks = _FakeKillSwitch()
    submitter = OrderSubmitter()

    result = submitter.submit(accepted, broker, journal, ks)

    assert isinstance(result, SubmissionAccepted)
    assert len(broker.submitted) == 2
    assert len(journal.orders) == 2


def test_pre_submission_nbbo_snapshot_recorded() -> None:
    """Test #4 / ADR-001: pre-submission NBBO captured on entry order row."""
    quote = Quote(
        symbol="ACME",
        bid=10.10,
        ask=10.15,
        last=10.12,
        ts=dt.datetime(2026, 4, 28, 14, 30, 5, tzinfo=dt.UTC),
    )
    accepted = _accepted()
    broker = _FakeBroker(quote=quote)
    journal = _FakeJournal()
    ks = _FakeKillSwitch()
    submitter = OrderSubmitter()

    submitter.submit(accepted, broker, journal, ks)

    # Quote called exactly once just before entry submission.
    assert broker.quote_calls == 1
    entry_row = next(o for o in journal.orders if o["role"] == "entry")
    assert entry_row["pre_submission_bid"] == 10.10
    assert entry_row["pre_submission_ask"] == 10.15
    assert entry_row["pre_submission_last"] == 10.12
    assert entry_row["pre_submission_quote_at"] == quote.ts


def test_entry_failure_no_stop_submitted() -> None:
    """Test #5 / AC-8: entry fails → no stop/TP; execution decision='submission_failed'."""
    accepted = _accepted(proposal=_proposal(take_profit_price=12.00))
    broker = _FakeBroker(entry_raises=BrokerUnavailable("retries exhausted"))
    journal = _FakeJournal()
    ks = _FakeKillSwitch()
    submitter = OrderSubmitter()

    result = submitter.submit(accepted, broker, journal, ks)

    assert isinstance(result, SubmissionFailed)
    # Only the entry attempt — no stop / no TP.
    assert len(broker.submitted) == 1
    assert broker.submitted[0].side == "buy"
    # Execution row recorded; zero order rows.
    assert len(journal.executions) == 1
    assert journal.executions[0]["decision"] == "submission_failed"
    assert journal.orders == []
    # KS not flipped — entry failure pre-position is not the partial-no-stop case.
    assert ks.halts == []


def test_entry_succeeds_stop_fails_kill_switch_halted() -> None:
    """Test #6 / AC-8: entry OK, stop fails → KS flipped 'submission_partial_no_stop'."""
    accepted = _accepted(proposal=_proposal(take_profit_price=12.00))
    broker = _FakeBroker(stop_raises=BrokerUnavailable("retries exhausted"))
    journal = _FakeJournal()
    ks = _FakeKillSwitch()
    submitter = OrderSubmitter()

    result = submitter.submit(accepted, broker, journal, ks)

    # Result reports the partial state.
    assert isinstance(result, SubmissionFailed)
    assert result.reason == "submission_partial_no_stop"
    # Entry submitted, stop attempted (and failed). TP not attempted because
    # stop is the first protective leg and its failure short-circuits.
    assert len(broker.submitted) == 2
    # KS was flipped to halted.
    assert len(ks.halts) == 1
    reason, set_by, _notes = ks.halts[0]
    assert reason == "submission_partial_no_stop"
    assert set_by == "system"
    # Execution + entry-order rows present so the open position is auditable.
    assert len(journal.executions) == 1
    assert journal.executions[0]["decision"] == "submission_partial_no_stop"
    entry_rows = [o for o in journal.orders if o["role"] == "entry"]
    assert len(entry_rows) == 1


def test_entry_succeeds_stop_fails_with_broker_rejected_kill_switch_halted() -> None:
    """Hotfix 2026-05-07 (incident: 27 unprotected positions): the exact
    production failure mode. Entry buy submits; back-to-back stop is
    rejected with `BrokerRejected` (the wrapped wash-trade APIError 40310000).
    Must follow the SAME path as the existing `BrokerUnavailable` branch:
    KS halts on `submission_partial_no_stop`, execution row + entry order
    row are journaled, TP is not attempted.
    """
    accepted = _accepted(proposal=_proposal(take_profit_price=12.00))
    broker = _FakeBroker(
        stop_raises=BrokerRejected(
            "potential wash trade detected",
            status_code=422,
            broker_code=40310000,
        )
    )
    journal = _FakeJournal()
    ks = _FakeKillSwitch()
    submitter = OrderSubmitter()

    result = submitter.submit(accepted, broker, journal, ks)

    assert isinstance(result, SubmissionFailed)
    assert result.reason == "submission_partial_no_stop"
    # Entry submitted, stop attempted; TP not reached.
    assert len(broker.submitted) == 2
    # KS halted with the partial-no-stop reason — same as the
    # BrokerUnavailable branch.
    assert len(ks.halts) == 1
    reason, set_by, _notes = ks.halts[0]
    assert reason == "submission_partial_no_stop"
    assert set_by == "system"
    # Execution row + entry order row written so the live position is
    # auditable at the journal.
    assert len(journal.executions) == 1
    assert journal.executions[0]["decision"] == "submission_partial_no_stop"
    entry_rows = [o for o in journal.orders if o["role"] == "entry"]
    assert len(entry_rows) == 1


def test_entry_failure_with_broker_rejected_journals_submission_failed() -> None:
    """Hotfix 2026-05-07: a non-retryable rejection on the ENTRY leg
    (e.g. 422 insufficient buying power) used to escape the
    `BrokerUnavailable`-only handler unjournaled. Now must write a
    `submission_failed` execution row and not flip the killswitch (no
    exposure, since no fill).
    """
    accepted = _accepted(proposal=_proposal(take_profit_price=12.00))
    broker = _FakeBroker(
        entry_raises=BrokerRejected(
            "insufficient buying power", status_code=422
        )
    )
    journal = _FakeJournal()
    ks = _FakeKillSwitch()
    submitter = OrderSubmitter()

    result = submitter.submit(accepted, broker, journal, ks)

    assert isinstance(result, SubmissionFailed)
    assert result.reason == "submission_failed"
    # Only the entry attempt; no stop/TP because there's no fill to protect.
    assert len(broker.submitted) == 1
    assert broker.submitted[0].side == "buy"
    # Execution row recorded; no order rows.
    assert len(journal.executions) == 1
    assert journal.executions[0]["decision"] == "submission_failed"
    assert journal.orders == []
    # No KS halt — entry failure is pre-position, not unprotected exposure.
    assert ks.halts == []


def test_p95_under_10s_with_fake_broker() -> None:
    """Test #7 / NFR-2: 100 sequential submits under fake broker; p95 < 10s."""
    import time

    submitter = OrderSubmitter()
    durations: list[float] = []
    for i in range(100):
        accepted = _accepted(proposal_id=1000 + i)
        broker = _FakeBroker()
        journal = _FakeJournal()
        ks = _FakeKillSwitch()
        t0 = time.perf_counter()
        submitter.submit(accepted, broker, journal, ks)
        durations.append(time.perf_counter() - t0)
    durations.sort()
    p95 = durations[int(0.95 * len(durations)) - 1]
    assert p95 < 10.0, f"p95 = {p95:.3f}s exceeded 10s budget"


# ---------------------------------------------------------------------------
# Integration: real JournalRepo write path
# ---------------------------------------------------------------------------

def test_real_journal_repo_persists_execution_and_orders(tmp_path: Path) -> None:
    """Integration: real SQLite + JournalRepo bindings round-trip end-to-end."""
    db_path = str(tmp_path / "journal.db")
    apply_migrations(db_path)

    # Need a proposal row first (FK).
    filing = FilingRow(
        accession_number="0001234567-26-000001",
        cik=1234567,
        form_type="8-K",
        filed_at=dt.datetime(2026, 4, 28, 14, 30, 0),
        fetched_at=dt.datetime(2026, 4, 28, 14, 31, 0),
        raw_text_path="/tmp/raw.txt",
        content_hash="abc123",
        item_codes='["2.02"]',
        issuer_ticker="ACME",
    )
    filing_id = insert_filing(db_path, filing)
    # Register a prompt row (FK on proposals.prompt_version).
    from journal.models import PromptRow
    from journal.repo import insert_prompt

    insert_prompt(db_path, PromptRow(
        prompt_version="sonnet_v1@deadbeef",
        name="sonnet_v1",
        file_path="src/prompts/sonnet_v1.txt",
        content_hash="deadbeef",
    ))
    proposal_id = insert_proposal(db_path, ProposalRow(
        filing_id=filing_id,
        decision_id="dec-001",
        model_id="claude-sonnet-4-6",
        prompt_version="sonnet_v1@deadbeef",
        raw_response="{}",
        kind="trade_proposal",
        symbol="ACME",
        direction="long",
        size_pct_requested=0.05,
        conviction=8,
        thesis="thesis text",
        input_tokens=10,
        output_tokens=20,
        latency_ms=100,
        cost_usd=0.001,
    ))

    journal = JournalRepo(db_path)
    accepted = _accepted(proposal=_proposal(take_profit_price=12.0), proposal_id=proposal_id)
    broker = _FakeBroker()
    ks = _FakeKillSwitch()
    submitter = OrderSubmitter()

    result = submitter.submit(accepted, broker, journal, ks)

    assert isinstance(result, SubmissionAccepted)
    # 3 order rows persisted via real SQL path.
    rows = get_orders_for_execution(db_path, result.execution_id)
    assert len(rows) == 3
    roles = {r.role for r in rows}
    assert roles == {"entry", "stop", "take_profit"}
    # Pre-submission NBBO landed on entry row.
    entry = next(r for r in rows if r.role == "entry")
    assert entry.pre_submission_bid == pytest.approx(10.00)

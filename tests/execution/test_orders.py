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
from broker.protocol import (
    BracketOrderRequest,
    BrokerRejected,
    OpenOrder,
    OrderRequest,
    Position,
    Quote,
    SubmittedOrder,
)
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
    """Records every submit/get_quote call. By default returns success.

    Hotfix 2026-05-07: now exposes `submit_bracket_order` since the
    submitter's normal flow is a single atomic bracket / OTO call. The
    legacy `submit_order` surface is kept for the wash-trade defense-in-
    depth test and for any future non-bracket call site.
    """

    def __init__(
        self,
        *,
        quote: Quote | None = None,
        bracket_raises: Exception | None = None,
    ) -> None:
        self._quote = quote or Quote(
            symbol="ACME",
            bid=10.00,
            ask=10.05,
            last=10.02,
            ts=dt.datetime(2026, 4, 28, 14, 30, tzinfo=dt.UTC),
        )
        self._bracket_raises = bracket_raises
        self.submitted: list[OrderRequest] = []
        self.submitted_brackets: list[BracketOrderRequest] = []
        self.quote_calls: int = 0
        self._next_id = 0

    def submit_order(self, req: OrderRequest) -> SubmittedOrder:
        # Retained for orphan-adoption / defense-in-depth tests; the
        # production submitter no longer routes here for the normal flow.
        self.submitted.append(req)
        self._next_id += 1
        return SubmittedOrder(
            broker_order_id=f"ord-{self._next_id}",
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

    def submit_bracket_order(
        self, req: BracketOrderRequest
    ) -> tuple[SubmittedOrder, SubmittedOrder, SubmittedOrder | None]:
        self.submitted_brackets.append(req)
        if self._bracket_raises is not None:
            raise self._bracket_raises
        self._next_id += 1
        entry = SubmittedOrder(
            broker_order_id=f"ord-{self._next_id}",
            client_order_id=req.entry_client_order_id,
            symbol=req.entry_symbol,
            side=req.entry_side,
            qty=req.entry_qty,
            order_type=req.entry_order_type,
            status="accepted",
            submitted_at=dt.datetime(2026, 4, 28, 14, 30, tzinfo=dt.UTC),
            limit_price=req.entry_limit_price,
            stop_price=None,
        )
        self._next_id += 1
        stop = SubmittedOrder(
            broker_order_id=f"ord-{self._next_id}",
            client_order_id=f"{req.entry_client_order_id}:bracket-stop",
            symbol=req.entry_symbol,
            side="sell",
            qty=req.entry_qty,
            order_type="stop",
            status="accepted",
            submitted_at=dt.datetime(2026, 4, 28, 14, 30, tzinfo=dt.UTC),
            stop_price=req.stop_loss_price,
        )
        tp: SubmittedOrder | None = None
        if req.take_profit_price is not None:
            self._next_id += 1
            tp = SubmittedOrder(
                broker_order_id=f"ord-{self._next_id}",
                client_order_id=f"{req.entry_client_order_id}:bracket-tp",
                symbol=req.entry_symbol,
                side="sell",
                qty=req.entry_qty,
                order_type="limit",
                status="accepted",
                submitted_at=dt.datetime(2026, 4, 28, 14, 30, tzinfo=dt.UTC),
                limit_price=req.take_profit_price,
            )
        return entry, stop, tp

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

def test_bracket_submission_writes_three_journal_rows() -> None:
    """Hotfix 2026-05-07: market entry + stop + TP submitted as a single
    atomic BRACKET; three order rows + one execution row persisted.
    """
    p = _proposal(take_profit_price=12.00)
    accepted = _accepted(proposal=p)
    broker = _FakeBroker()
    journal = _FakeJournal()
    ks = _FakeKillSwitch()
    submitter = OrderSubmitter()

    result = submitter.submit(accepted, broker, journal, ks)

    assert isinstance(result, SubmissionAccepted)
    # ONE bracket call (atomic) — not three separate submit_order calls.
    assert len(broker.submitted_brackets) == 1
    assert broker.submitted == []  # no calls to legacy submit_order
    bracket = broker.submitted_brackets[0]
    assert bracket.entry_side == "buy"
    assert bracket.entry_order_type == "market"
    assert bracket.stop_loss_price == 8.50
    assert bracket.take_profit_price == 12.00
    # One execution row, three order rows (entry + stop + tp).
    assert len(journal.executions) == 1
    assert len(journal.orders) == 3
    exec_row = journal.executions[0]
    assert exec_row["decision"] == "accepted"
    assert exec_row["realized_dollar_size"] == 250.0
    assert exec_row["realized_size_pct"] == 0.05
    submitted_orders = json.loads(exec_row["submitted_orders_json"])
    assert {o["role"] for o in submitted_orders} == {"entry", "stop", "take_profit"}
    roles = {o["role"] for o in journal.orders}
    assert roles == {"entry", "stop", "take_profit"}


def test_limit_entry_uses_entry_limit_price() -> None:
    """Test #2: entry_style=limit → limit BUY at entry_limit_price, TIF=day,
    routed through the bracket flow.
    """
    p = _proposal(entry_style="limit", entry_limit_price=9.95)
    accepted = _accepted(proposal=p)
    broker = _FakeBroker()
    journal = _FakeJournal()
    ks = _FakeKillSwitch()
    submitter = OrderSubmitter()

    result = submitter.submit(accepted, broker, journal, ks)

    assert isinstance(result, SubmissionAccepted)
    bracket = broker.submitted_brackets[0]
    assert bracket.entry_order_type == "limit"
    assert bracket.entry_limit_price == 9.95
    assert bracket.entry_tif == "day"


def test_no_tp_when_take_profit_price_omitted() -> None:
    """Hotfix 2026-05-07: TP omitted in proposal → OTO submission (no
    take-profit in the bracket request); only entry + stop journal rows.
    """
    p = _proposal(take_profit_price=None)
    accepted = _accepted(proposal=p)
    broker = _FakeBroker()
    journal = _FakeJournal()
    ks = _FakeKillSwitch()
    submitter = OrderSubmitter()

    result = submitter.submit(accepted, broker, journal, ks)

    assert isinstance(result, SubmissionAccepted)
    assert len(broker.submitted_brackets) == 1
    bracket = broker.submitted_brackets[0]
    # OTO branch: no take-profit price on the request.
    assert bracket.take_profit_price is None
    # Two journal rows (entry + stop) — TP not present.
    assert len(journal.orders) == 2
    roles = {o["role"] for o in journal.orders}
    assert roles == {"entry", "stop"}


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


def test_bracket_submission_rejected_writes_submission_failed_no_order_rows() -> None:
    """Hotfix 2026-05-07: bracket submission is atomic — when Alpaca
    rejects the whole complex order (e.g. 422 insufficient buying power,
    or any other non-retryable rejection), no legs reach the broker. The
    execution row records `submission_failed`; zero order rows are
    written; the killswitch is NOT flipped (no exposure).
    """
    accepted = _accepted(proposal=_proposal(take_profit_price=12.00))
    broker = _FakeBroker(
        bracket_raises=BrokerRejected(
            "insufficient buying power", status_code=422
        )
    )
    journal = _FakeJournal()
    ks = _FakeKillSwitch()
    submitter = OrderSubmitter()

    result = submitter.submit(accepted, broker, journal, ks)

    assert isinstance(result, SubmissionFailed)
    assert result.reason == "submission_failed"
    # The single bracket call was attempted; nothing else.
    assert len(broker.submitted_brackets) == 1
    assert broker.submitted == []
    # Execution row recorded; no order rows.
    assert len(journal.executions) == 1
    assert journal.executions[0]["decision"] == "submission_failed"
    assert journal.orders == []
    # No KS halt — bracket atomicity guarantees no exposure on rejection.
    assert ks.halts == []


def test_bracket_submission_unavailable_writes_submission_failed() -> None:
    """Hotfix 2026-05-07: bracket submission exhausts retries
    (`BrokerUnavailable`) → same clean failure path as `BrokerRejected`:
    `submission_failed` execution row, zero order rows, killswitch
    untouched. With bracket atomicity there is no `submission_partial_no_stop`
    case on normal flow.
    """
    accepted = _accepted(proposal=_proposal(take_profit_price=12.00))
    broker = _FakeBroker(bracket_raises=BrokerUnavailable("retries exhausted"))
    journal = _FakeJournal()
    ks = _FakeKillSwitch()
    submitter = OrderSubmitter()

    result = submitter.submit(accepted, broker, journal, ks)

    assert isinstance(result, SubmissionFailed)
    assert result.reason == "submission_failed"
    assert len(broker.submitted_brackets) == 1
    assert len(journal.executions) == 1
    assert journal.executions[0]["decision"] == "submission_failed"
    assert journal.orders == []
    assert ks.halts == []


def test_bracket_submission_does_not_emit_submission_partial_no_stop() -> None:
    """Regression guard for incident 2026-05-07. The historic
    `submission_partial_no_stop` failure mode was caused by the entry
    succeeding while the back-to-back stop was rejected as a wash trade.
    With bracket / OTO orders that state cannot occur on the normal flow:
    every rejection fails atomically. This test asserts the submitter
    never emits that decision string given a bracket rejection.
    """
    accepted = _accepted(proposal=_proposal(take_profit_price=12.00))
    broker = _FakeBroker(
        bracket_raises=BrokerRejected(
            "potential wash trade detected — opposite side market/stop order exists",
            status_code=422,
            broker_code=40310000,
        )
    )
    journal = _FakeJournal()
    ks = _FakeKillSwitch()
    submitter = OrderSubmitter()

    result = submitter.submit(accepted, broker, journal, ks)

    assert isinstance(result, SubmissionFailed)
    # Critical: the decision must NOT be the historic partial-no-stop value.
    assert result.reason == "submission_failed"
    assert journal.executions[0]["decision"] != "submission_partial_no_stop"
    assert journal.executions[0]["decision"] == "submission_failed"
    # Zero order rows — bracket atomicity precludes a live entry.
    assert journal.orders == []
    # Killswitch stays clean — no unprotected exposure.
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

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
        entry_raises: Exception | None = None,
    ) -> None:
        self._quote = quote or Quote(
            symbol="ACME",
            bid=10.00,
            ask=10.05,
            last=10.02,
            ts=dt.datetime(2026, 4, 28, 14, 30, tzinfo=dt.UTC),
        )
        self._bracket_raises = bracket_raises
        # PDT-SUNSET-2026-06-04: under PDT mode the entry is submitted
        # via `submit_order` (not bracket). `entry_raises` lets tests
        # simulate entry-leg failure on that path.
        self._entry_raises = entry_raises
        self.submitted: list[OrderRequest] = []
        self.submitted_brackets: list[BracketOrderRequest] = []
        self.quote_calls: int = 0
        self._next_id = 0

    def submit_order(self, req: OrderRequest) -> SubmittedOrder:
        # Retained for orphan-adoption / defense-in-depth tests AND for
        # the PDT-active branch (entry-only via submit_order).
        if self._entry_raises is not None:
            raise self._entry_raises
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
        # PDT-SUNSET-2026-06-04: virtual_exit inserts captured per ADR-009.
        self.virtual_exits: list[dict[str, Any]] = []
        self.call_log: list[str] = []  # ordered log of method names invoked
        self._exec_seq = 0
        self._order_seq = 0
        self._ve_seq = 0

    def insert_execution(self, row: Any) -> int:
        self._exec_seq += 1
        d = row.model_dump()
        d["id"] = self._exec_seq
        self.executions.append(d)
        self.call_log.append("insert_execution")
        return self._exec_seq

    def insert_order(self, row: Any) -> int:
        self._order_seq += 1
        d = row.model_dump()
        d["id"] = self._order_seq
        self.orders.append(d)
        self.call_log.append(f"insert_order:{d.get('role')}")
        return self._order_seq

    def insert_virtual_exit(self, row: Any) -> int:
        self._ve_seq += 1
        d = row.model_dump()
        d["id"] = self._ve_seq
        self.virtual_exits.append(d)
        self.call_log.append(f"insert_virtual_exit:{d.get('role')}")
        return self._ve_seq


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
    """Test #2: entry_style=limit → limit BUY at entry_limit_price,
    routed through the bracket flow with the group-wide GTC TIF (D-079 §3.1).
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
    assert bracket.entry_tif == "gtc"


# ---------------------------------------------------------------------------
# D-079 §3.1 — GTC protective legs, honestly journaled (kills O-6).
# Alpaca applies ONE time_in_force to the whole bracket group; children
# inherit it. The pre-fix code sent entry_tif="day" (so stop and TP silently
# expired at end of entry day) while journaling tif="gtc" for both legs —
# the audit record claimed protection that did not exist.
# ---------------------------------------------------------------------------

def test_bracket_group_tif_is_gtc() -> None:
    """D-079 §3.1: the bracket request carries TIF=gtc so the protective
    stop and TP survive past entry day (children inherit the group TIF)."""
    accepted = _accepted(proposal=_proposal(take_profit_price=12.00))
    broker = _FakeBroker()
    journal = _FakeJournal()
    submitter = OrderSubmitter()

    submitter.submit(accepted, broker, journal, _FakeKillSwitch())

    assert broker.submitted_brackets[0].entry_tif == "gtc"


def test_bracket_journal_rows_record_tif_actually_sent() -> None:
    """D-079 §3.1 honesty: every journal row's `tif` equals the TIF that
    was actually sent on the bracket request — derived from the request,
    not hardcoded, so journal and broker can never diverge again."""
    accepted = _accepted(proposal=_proposal(take_profit_price=12.00))
    broker = _FakeBroker()
    journal = _FakeJournal()
    submitter = OrderSubmitter()

    submitter.submit(accepted, broker, journal, _FakeKillSwitch())

    sent_tif = broker.submitted_brackets[0].entry_tif
    assert sent_tif == "gtc"
    assert len(journal.orders) == 3
    for row in journal.orders:
        assert row["tif"] == sent_tif, (
            f"journal row role={row['role']!r} records tif={row['tif']!r} "
            f"but the broker was sent {sent_tif!r}"
        )


def test_oto_journal_rows_record_tif_actually_sent() -> None:
    """Same honesty contract on the OTO (no-TP) path."""
    accepted = _accepted(proposal=_proposal(take_profit_price=None))
    broker = _FakeBroker()
    journal = _FakeJournal()
    submitter = OrderSubmitter()

    submitter.submit(accepted, broker, journal, _FakeKillSwitch())

    sent_tif = broker.submitted_brackets[0].entry_tif
    assert sent_tif == "gtc"
    assert {row["role"] for row in journal.orders} == {"entry", "stop"}
    for row in journal.orders:
        assert row["tif"] == sent_tif


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


# ---------------------------------------------------------------------------
# PDT-SUNSET-2026-06-04: ADR-009 §"Order construction branch" — pdt_mode=True.
# AC-7 (story S-PDT-3).
# ---------------------------------------------------------------------------


class _FakePDTState:
    def __init__(self, *, active: bool) -> None:
        self._active = active

    def is_active(self) -> bool:
        return self._active


def test_submit_pdt_mode_skips_stop_broker_call() -> None:
    """AC-7 #1: under PDT mode, broker.submit_order is called exactly
    once — for the entry only. No stop / no TP at the broker."""
    p = _proposal(take_profit_price=12.00)
    accepted = _accepted(proposal=p)
    broker = _FakeBroker()
    journal = _FakeJournal()
    ks = _FakeKillSwitch()
    submitter = OrderSubmitter()

    result = submitter.submit(
        accepted, broker, journal, ks, pdt_state=_FakePDTState(active=True)
    )

    assert isinstance(result, SubmissionAccepted)
    assert len(broker.submitted) == 1
    assert broker.submitted[0].side == "buy"
    # Zero broker stop / TP submissions.
    sides = [(r.side, r.order_type) for r in broker.submitted]
    assert ("sell", "stop") not in sides
    assert ("sell", "limit") not in sides


def test_submit_pdt_mode_writes_virtual_exit_for_stop() -> None:
    """AC-7 #2: a virtual_exits row is inserted for the stop with the
    correct fields (role, stop_price, entry_price=nbbo.last, qty,
    execution_id)."""
    p = _proposal(take_profit_price=12.00)
    accepted = _accepted(proposal=p, qty=25)
    broker = _FakeBroker(quote=Quote(
        symbol="ACME", bid=10.00, ask=10.05, last=10.02,
        ts=dt.datetime(2026, 4, 28, 14, 30, tzinfo=dt.UTC),
    ))
    journal = _FakeJournal()
    submitter = OrderSubmitter()

    submitter.submit(
        accepted, broker, journal, _FakeKillSwitch(),
        pdt_state=_FakePDTState(active=True),
    )

    stop_ves = [v for v in journal.virtual_exits if v["role"] == "stop"]
    assert len(stop_ves) == 1
    sve = stop_ves[0]
    assert sve["stop_price"] == 8.50  # from _proposal
    assert sve["entry_price"] == 10.02  # nbbo.last
    assert sve["qty"] == 25
    assert sve["state"] == "active"
    assert sve["execution_id"] == journal.executions[0]["id"]
    assert sve["proposal_id"] == accepted.proposal_id
    assert sve["symbol"] == "ACME"


def test_submit_pdt_mode_writes_virtual_exit_for_tp_when_present() -> None:
    """AC-7 #3: with TP set, two virtual_exits rows (stop + tp)."""
    p = _proposal(take_profit_price=12.00)
    accepted = _accepted(proposal=p)
    broker = _FakeBroker()
    journal = _FakeJournal()
    submitter = OrderSubmitter()

    submitter.submit(
        accepted, broker, journal, _FakeKillSwitch(),
        pdt_state=_FakePDTState(active=True),
    )

    roles = [v["role"] for v in journal.virtual_exits]
    assert sorted(roles) == ["stop", "tp"]
    tp = next(v for v in journal.virtual_exits if v["role"] == "tp")
    assert tp["tp_price"] == 12.00
    assert tp["stop_price"] is None
    stop = next(v for v in journal.virtual_exits if v["role"] == "stop")
    assert stop["tp_price"] is None
    assert stop["stop_price"] == 8.50


def test_submit_pdt_mode_skips_tp_virtual_exit_when_tp_none() -> None:
    """AC-7 #4: with no TP, a single virtual_exits row (stop only)."""
    p = _proposal(take_profit_price=None)
    accepted = _accepted(proposal=p)
    broker = _FakeBroker()
    journal = _FakeJournal()
    submitter = OrderSubmitter()

    submitter.submit(
        accepted, broker, journal, _FakeKillSwitch(),
        pdt_state=_FakePDTState(active=True),
    )

    assert len(journal.virtual_exits) == 1
    assert journal.virtual_exits[0]["role"] == "stop"


def test_submit_pdt_mode_returns_accepted_with_null_stop_id() -> None:
    """AC-7 #5: SubmissionAccepted reports None for stop and TP broker
    order ids under PDT mode (no broker order exists)."""
    p = _proposal(take_profit_price=12.00)
    accepted = _accepted(proposal=p)
    broker = _FakeBroker()
    journal = _FakeJournal()
    submitter = OrderSubmitter()

    result = submitter.submit(
        accepted, broker, journal, _FakeKillSwitch(),
        pdt_state=_FakePDTState(active=True),
    )

    assert isinstance(result, SubmissionAccepted)
    assert result.entry_broker_order_id is not None
    assert result.stop_broker_order_id is None
    assert result.take_profit_broker_order_id is None


def test_submit_pdt_mode_journals_execution_decision_accepted() -> None:
    """AC-7 #6: executions.decision='accepted' under PDT mode (the
    protective leg LIVES — just in `virtual_exits`, not at broker)."""
    accepted = _accepted(proposal=_proposal(take_profit_price=12.00))
    broker = _FakeBroker()
    journal = _FakeJournal()
    submitter = OrderSubmitter()

    submitter.submit(
        accepted, broker, journal, _FakeKillSwitch(),
        pdt_state=_FakePDTState(active=True),
    )

    assert len(journal.executions) == 1
    assert journal.executions[0]["decision"] == "accepted"


def test_submit_pdt_mode_zero_orders_rows_for_protective_legs() -> None:
    """AC-3: under PDT mode, ZERO `orders` rows for stop or TP roles."""
    accepted = _accepted(proposal=_proposal(take_profit_price=12.00))
    broker = _FakeBroker()
    journal = _FakeJournal()
    submitter = OrderSubmitter()

    submitter.submit(
        accepted, broker, journal, _FakeKillSwitch(),
        pdt_state=_FakePDTState(active=True),
    )

    roles = [o["role"] for o in journal.orders]
    assert roles == ["entry"]


def test_submit_pdt_mode_journal_write_order() -> None:
    """AC-7 #8: write order is execution → entry order → stop virtual
    exit → tp virtual exit."""
    accepted = _accepted(proposal=_proposal(take_profit_price=12.00))
    broker = _FakeBroker()
    journal = _FakeJournal()
    submitter = OrderSubmitter()

    submitter.submit(
        accepted, broker, journal, _FakeKillSwitch(),
        pdt_state=_FakePDTState(active=True),
    )

    assert journal.call_log == [
        "insert_execution",
        "insert_order:entry",
        "insert_virtual_exit:stop",
        "insert_virtual_exit:tp",
    ]


def test_submit_pdt_mode_submitted_orders_json_includes_virtual_legs() -> None:
    """AC-6: submitted_orders_json carries the virtual legs in audit
    shape with `broker_order_id=null`, `virtual=true`, and the
    virtual_exit_id."""
    accepted = _accepted(proposal=_proposal(take_profit_price=12.00))
    broker = _FakeBroker()
    journal = _FakeJournal()
    submitter = OrderSubmitter()

    submitter.submit(
        accepted, broker, journal, _FakeKillSwitch(),
        pdt_state=_FakePDTState(active=True),
    )

    # Note: we always insert the execution row FIRST with the entry-only
    # placeholder json (so the FK chain is valid). The virtual legs are
    # the in-memory `submitted` list — the persisted json may be the
    # entry-only placeholder. We assert on that contract for now;
    # downstream consumers (dashboard rendering update) is out of scope
    # per AC-6.
    persisted = json.loads(journal.executions[0]["submitted_orders_json"])
    assert any(o["role"] == "entry" for o in persisted)


def test_submit_pdt_mode_inactive_uses_existing_path() -> None:
    """Regression: pdt_state.is_active()==False routes through the
    existing path. Post-bracket-hotfix the existing path is a single
    atomic `submit_bracket_order` call producing 3 order rows (entry +
    stop + tp); zero virtual exits."""
    p = _proposal(take_profit_price=12.00)
    accepted = _accepted(proposal=p)
    broker = _FakeBroker()
    journal = _FakeJournal()
    submitter = OrderSubmitter()

    submitter.submit(
        accepted, broker, journal, _FakeKillSwitch(),
        pdt_state=_FakePDTState(active=False),
    )

    assert len(broker.submitted_brackets) == 1
    assert broker.submitted == []  # no legacy submit_order calls
    assert len(journal.orders) == 3
    assert journal.virtual_exits == []


def test_submit_pdt_state_none_uses_existing_path() -> None:
    """Regression: pdt_state=None (not passed) routes through the
    existing (post-hotfix bracket) path. Backward-compat for legacy
    callers."""
    accepted = _accepted(proposal=_proposal(take_profit_price=12.00))
    broker = _FakeBroker()
    journal = _FakeJournal()
    submitter = OrderSubmitter()

    submitter.submit(accepted, broker, journal, _FakeKillSwitch())

    assert len(broker.submitted_brackets) == 1
    assert broker.submitted == []
    assert journal.virtual_exits == []


def test_submit_pdt_mode_entry_failure_no_virtual_exit_written() -> None:
    """If entry fails, the PDT branch is not taken — no virtual_exits
    rows; SubmissionFailed returned. Same as non-PDT entry-failure
    semantics."""
    accepted = _accepted(proposal=_proposal(take_profit_price=12.00))
    broker = _FakeBroker(entry_raises=BrokerUnavailable("offline"))
    journal = _FakeJournal()
    submitter = OrderSubmitter()

    result = submitter.submit(
        accepted, broker, journal, _FakeKillSwitch(),
        pdt_state=_FakePDTState(active=True),
    )

    assert isinstance(result, SubmissionFailed)
    assert journal.virtual_exits == []
    assert journal.executions[0]["decision"] == "submission_failed"


# ---------------------------------------------------------------------------
# HOTFIX-2026-05-11: inverted-stop guardrail (executor rejects long entries
# whose stop_loss_price >= entry_price). Symptom: 2026-05-08 AORT
# (proposal 1035) where analyzer Haiku produced stop_loss_price=30.5 for a
# long entry that filled at ~$25.31. Scanner immediately fired the crossed
# stop on the next tick after entry; instant-self-sell wasted a PDT slot.
# ---------------------------------------------------------------------------

def test_pdt_long_entry_rejects_inverted_stop() -> None:
    """HOTFIX-2026-05-11: long entry whose stop is at or above the
    pre-submission last price must NOT be submitted. Execution row
    recorded as submission_failed; zero broker entry calls; zero
    virtual_exits.
    """
    # Quote last = 10.02; stop_loss_price = 15.00 → inverted for a long.
    accepted = _accepted(
        proposal=_proposal(),  # default stop_loss_price=8.50; override below
    )
    # Build a proposal with an explicitly inverted stop. _proposal's
    # `stop_loss_price` default is 8.50; we need to override to 15.00
    # while still passing schema validation (gt=0 — fine).
    p = TradeProposal.model_validate({
        "symbol": "AORT",
        "direction": "long",
        "size_pct_of_capital": 0.05,
        "entry_style": "market_open",
        "stop_loss_price": 15.00,  # > nbbo.last=10.02 → inverted
        "time_horizon_days": 10,
        "conviction": 8,
        "thesis": (
            "Strong fundamentals with material 8-K disclosure indicating "
            "near-term catalyst per filing analysis."
        ),
        "signals": ["8-K item 2.02 strong earnings beat"],
        "exit_conditions": ["thesis breaks; stop hit; 30 days elapsed"],
        "risk_factors": ["macro risk; sector rotation; news risk"],
    })
    accepted = _accepted(proposal=p)
    broker = _FakeBroker()
    journal = _FakeJournal()
    ks = _FakeKillSwitch()
    submitter = OrderSubmitter()

    result = submitter.submit(
        accepted, broker, journal, ks,
        pdt_state=_FakePDTState(active=True),
    )

    assert isinstance(result, SubmissionFailed)
    assert result.reason == "submission_failed"
    # NO entry submission attempted.
    assert broker.submitted == []
    # No virtual_exits — entry never happened.
    assert journal.virtual_exits == []
    # Execution row records the rejection so the audit trail is intact.
    assert len(journal.executions) == 1
    assert journal.executions[0]["decision"] == "submission_failed"
    # No KS halt — the rejection is purely defensive; no exposure created.
    assert ks.halts == []


def test_pdt_long_entry_rejects_stop_equal_to_entry_price() -> None:
    """HOTFIX-2026-05-11: stop_loss_price exactly equal to nbbo.last is
    also rejected — a stop at-or-above entry on a long crosses
    immediately (stops are inclusive on the boundary).
    """
    p = TradeProposal.model_validate({
        "symbol": "AORT",
        "direction": "long",
        "size_pct_of_capital": 0.05,
        "entry_style": "market_open",
        "stop_loss_price": 10.02,  # == nbbo.last
        "time_horizon_days": 10,
        "conviction": 8,
        "thesis": (
            "Strong fundamentals with material 8-K disclosure indicating "
            "near-term catalyst per filing analysis."
        ),
        "signals": ["8-K item 2.02 strong earnings beat"],
        "exit_conditions": ["thesis breaks; stop hit; 30 days elapsed"],
        "risk_factors": ["macro risk; sector rotation; news risk"],
    })
    accepted = _accepted(proposal=p)
    broker = _FakeBroker()
    journal = _FakeJournal()
    submitter = OrderSubmitter()

    result = submitter.submit(
        accepted, broker, journal, _FakeKillSwitch(),
        pdt_state=_FakePDTState(active=True),
    )

    assert isinstance(result, SubmissionFailed)
    assert broker.submitted == []
    assert journal.virtual_exits == []


def test_pdt_long_entry_accepts_valid_stop_below_entry() -> None:
    """HOTFIX-2026-05-11 regression: a legitimate stop below entry price
    is still accepted on the PDT path.
    """
    # Default _proposal: stop_loss_price=8.50, _FakeBroker quote.last=10.02
    accepted = _accepted(proposal=_proposal(take_profit_price=12.00))
    broker = _FakeBroker()
    journal = _FakeJournal()
    submitter = OrderSubmitter()

    result = submitter.submit(
        accepted, broker, journal, _FakeKillSwitch(),
        pdt_state=_FakePDTState(active=True),
    )

    assert isinstance(result, SubmissionAccepted)
    # Entry submitted, virtual_exits written.
    assert len(broker.submitted) == 1
    stop_ves = [v for v in journal.virtual_exits if v["role"] == "stop"]
    assert len(stop_ves) == 1


def test_bracket_long_entry_rejects_inverted_stop() -> None:
    """HOTFIX-2026-05-11: same guardrail applies on the non-PDT bracket
    path. Even though Alpaca server-side bracket validation would
    catch this, fail fast locally so the proposal is journaled as
    submission_failed before any broker round-trip — and so a future
    broker that doesn't validate is still safe.
    """
    p = TradeProposal.model_validate({
        "symbol": "AORT",
        "direction": "long",
        "size_pct_of_capital": 0.05,
        "entry_style": "market_open",
        "stop_loss_price": 15.00,  # > nbbo.last=10.02
        "time_horizon_days": 10,
        "conviction": 8,
        "thesis": (
            "Strong fundamentals with material 8-K disclosure indicating "
            "near-term catalyst per filing analysis."
        ),
        "signals": ["8-K item 2.02 strong earnings beat"],
        "exit_conditions": ["thesis breaks; stop hit; 30 days elapsed"],
        "risk_factors": ["macro risk; sector rotation; news risk"],
    })
    accepted = _accepted(proposal=p)
    broker = _FakeBroker()
    journal = _FakeJournal()
    ks = _FakeKillSwitch()
    submitter = OrderSubmitter()

    # No pdt_state → bracket flow.
    result = submitter.submit(accepted, broker, journal, ks)

    assert isinstance(result, SubmissionFailed)
    assert result.reason == "submission_failed"
    # NO bracket submission attempted.
    assert broker.submitted_brackets == []
    assert broker.submitted == []
    assert journal.orders == []
    assert journal.executions[0]["decision"] == "submission_failed"


def test_bracket_long_entry_accepts_valid_stop_below_entry() -> None:
    """HOTFIX-2026-05-11 regression: default proposal (stop_loss_price=8.50,
    nbbo.last=10.02) still routes through bracket submission cleanly.
    """
    accepted = _accepted(proposal=_proposal(take_profit_price=12.00))
    broker = _FakeBroker()
    journal = _FakeJournal()
    submitter = OrderSubmitter()

    result = submitter.submit(accepted, broker, journal, _FakeKillSwitch())

    assert isinstance(result, SubmissionAccepted)
    assert len(broker.submitted_brackets) == 1


def test_bracket_journal_rows_leave_final_status_null_pending_fill() -> None:
    """D-078 §7.4 lifecycle contract: `orders.final_status` is a
    deferred-completion field — NULL until the WS1 FillIngestor records
    a terminal disposition. Submission-time broker acks ('accepted',
    'new') are not terminal and must not be written here, or
    `get_orders_pending_fill` (final_status IS NULL) would never see
    the order and the §2.2 lifecycle classifier could not explain its
    fill."""
    accepted = _accepted(proposal=_proposal(take_profit_price=12.00))
    broker = _FakeBroker()
    journal = _FakeJournal()
    submitter = OrderSubmitter()

    submitter.submit(accepted, broker, journal, _FakeKillSwitch())

    assert len(journal.orders) == 3
    for row in journal.orders:
        assert row["final_status"] is None, (
            f"role={row['role']!r} wrote final_status={row['final_status']!r} "
            "at submission; must stay NULL until a terminal fill outcome"
        )

# PDT-SUNSET-2026-06-04: ADR-009 §"Pre-market deferred replay" — S-PDT-5 tests.
"""S-PDT-5 — `DeferredSellReplayer` unit tests.

References: story S-PDT-5 AC-7; pdt-budget-architecture.md §3.2.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import pytest
from alpaca.common.exceptions import APIError

from broker.alpaca import BrokerUnavailable
from broker.protocol import OrderRequest, Position, Quote, SubmittedOrder
from execution.pdt_budget import PDTBudgetExceeded
from execution.virtual_exits import DeferredSellReplayer, ReplayReport
from journal.models import DeferredSellRow, OrderRow


def _drow(
    *,
    id: int = 1,
    symbol: str = "AAPL",
    role: str = "stop",
    qty: int = 5,
    deferred_reason: str = "ev_lost",
    deferred_at: dt.datetime | None = None,
    trigger_price: float = 95.0,
    ev_at_defer: float = -25.0,
    virtual_exit_id: int = 100,
    execution_id: int = 200,
    proposal_id: int = 300,
    notes: str | None = None,
) -> DeferredSellRow:
    if deferred_at is None:
        deferred_at = dt.datetime(2026, 5, 6, 18, 0, tzinfo=dt.UTC)  # yesterday ET
    return DeferredSellRow(
        id=id,
        virtual_exit_id=virtual_exit_id,
        execution_id=execution_id,
        proposal_id=proposal_id,
        symbol=symbol,
        qty=qty,
        role=role,  # type: ignore[arg-type]
        trigger_price=trigger_price,
        ev_at_defer=ev_at_defer,
        deferred_at=deferred_at,
        deferred_reason=deferred_reason,  # type: ignore[arg-type]
        notes=notes,
    )


class _FakeJournal:
    def __init__(
        self,
        *,
        unreplayed: list[DeferredSellRow] | None = None,
        ve_states: dict[int, str] | None = None,
    ) -> None:
        self._unreplayed = unreplayed or []
        # Defaults every virtual_exit_id to 'active' (preserves existing
        # test behavior). Tests covering the supersede race override.
        self._ve_states = ve_states or {}
        self.replayed_marks: list[tuple[int, str]] = []
        self.skipped_marks: list[tuple[int, str]] = []
        # PDT-SUNSET-2026-06-04: FINDING-3 — capture
        # mark_virtual_exit_submitted calls so the replayer's source-V
        # transition to 'submitted' is testable.
        self.virtual_exits_submitted: list[tuple[int, str]] = []
        # HOTFIX-2026-05-08: capture journaled sell orders so replayer
        # paths can be verified to insert OrderRows for reconciler tolerance.
        self.orders: list[OrderRow] = []
        self._oseq = 0

    def list_active_virtual_exits(self) -> list: return []  # unused here

    def mark_virtual_exit_submitted(
        self, vid: int, broker_order_id: str
    ) -> None:
        self.virtual_exits_submitted.append((vid, broker_order_id))
        self._ve_states[vid] = "submitted"
    def mark_virtual_exit_obsolete(self, *_: Any, **__: Any) -> None: ...
    def insert_deferred_sell(self, *_: Any, **__: Any) -> int: return 0

    def list_unreplayed_deferred_sells(self) -> list[DeferredSellRow]:
        return list(self._unreplayed)

    def mark_deferred_replayed(self, did: int, broker_order_id: str) -> None:
        self.replayed_marks.append((did, broker_order_id))

    def mark_deferred_skipped(self, did: int, reason: str) -> None:
        self.skipped_marks.append((did, reason))

    def get_virtual_exit_state(self, vid: int) -> str | None:
        return self._ve_states.get(vid, "active")

    # HOTFIX-2026-05-08: PDT sells must be journaled for reconciler tolerance.
    def insert_order(self, row: OrderRow) -> int:
        self._oseq += 1
        self.orders.append(row)
        return self._oseq


class _FakeBroker:
    def __init__(
        self,
        *,
        positions: list[Position] | None = None,
        submit_responses: dict[str, SubmittedOrder | Exception] | None = None,
        positions_raises: Exception | None = None,
    ) -> None:
        self._positions = positions if positions is not None else [
            Position(symbol="AAPL", qty=5, avg_entry_price=100.0,
                     market_value=500.0, unrealized_pnl=0.0),
        ]
        self._submit_responses = submit_responses or {}
        self._positions_raises = positions_raises
        self.submitted: list[OrderRequest] = []
        self._seq = 0
        self.lookups: list[str] = []
        self._lookup_responses: dict[str, SubmittedOrder | None] = {}

    def get_positions(self) -> list[Position]:
        if self._positions_raises is not None:
            raise self._positions_raises
        return list(self._positions)

    def get_quote(self, _: str) -> Quote:  # pragma: no cover
        raise AssertionError("replayer does not call get_quote")

    def submit_order(self, req: OrderRequest) -> SubmittedOrder:
        self.submitted.append(req)
        configured = self._submit_responses.get(req.client_order_id)
        if isinstance(configured, Exception):
            raise configured
        if isinstance(configured, SubmittedOrder):
            return configured
        self._seq += 1
        return SubmittedOrder(
            broker_order_id=f"alp-replay-{self._seq}",
            client_order_id=req.client_order_id,
            symbol=req.symbol, side=req.side, qty=req.qty,
            order_type=req.order_type, status="accepted",
            submitted_at=dt.datetime(2026, 5, 7, 9, 0, tzinfo=dt.UTC),
            limit_price=req.limit_price, stop_price=req.stop_price,
        )

    def get_order_by_client_id(self, client_order_id: str) -> SubmittedOrder | None:
        self.lookups.append(client_order_id)
        return self._lookup_responses.get(client_order_id)

    def set_lookup_response(self, client_order_id: str, resp: SubmittedOrder) -> None:
        self._lookup_responses[client_order_id] = resp

    # Unused.
    def cancel_order(self, *_: Any, **__: Any) -> None: ...
    def get_account(self) -> Any: raise AssertionError


def _now_premarket() -> dt.datetime:
    """2026-05-07 09:00 UTC = 2026-05-07 05:00 ET (pre-market)."""
    return dt.datetime(2026, 5, 7, 9, 0, tzinfo=dt.UTC)


# ---------------------------------------------------------------------------
# AC-7 #1
# ---------------------------------------------------------------------------


def test_replayer_skips_today_rows() -> None:
    """deferred_at = today ET → skipped; not submitted."""
    today_in_et = dt.datetime(2026, 5, 7, 6, 0, tzinfo=dt.UTC)  # 02:00 ET today
    j = _FakeJournal(unreplayed=[_drow(id=1, deferred_at=today_in_et)])
    b = _FakeBroker()
    r = DeferredSellReplayer(broker=b, journal=j, now_fn=_now_premarket)
    report = r.run()
    assert report.replayed == 0
    assert report.skipped_today == 1
    assert b.submitted == []
    assert j.replayed_marks == []


# ---------------------------------------------------------------------------
# AC-7 #2 — happy path
# ---------------------------------------------------------------------------


def test_replayer_replays_yesterday_row() -> None:
    """deferred_at=yesterday; position open → broker.submit called once;
    row marked replayed; replay_broker_order_id set.

    HOTFIX-2026-05-08: also asserts insert_order was called so the
    reconciler tolerance can explain the broker-position decrease.
    """
    j = _FakeJournal(unreplayed=[_drow(id=42, role="stop", execution_id=200)])
    b = _FakeBroker()
    r = DeferredSellReplayer(broker=b, journal=j, now_fn=_now_premarket)
    report = r.run()
    assert report.replayed == 1
    assert report.skipped_today == 0
    assert len(b.submitted) == 1
    submitted = b.submitted[0]
    assert submitted.client_order_id == "pdt-replay-42"
    assert submitted.side == "sell"
    assert submitted.qty == 5
    assert submitted.order_type == "market"  # stop → market replay
    assert j.replayed_marks == [(42, "alp-replay-1")]
    # HOTFIX-2026-05-08: journal got an orders row for the sell.
    assert len(j.orders) == 1
    o = j.orders[0]
    assert o.execution_id == 200
    assert o.role == "stop"
    assert o.symbol == "AAPL"
    assert o.side == "sell"
    assert o.order_type == "market"
    assert o.qty == 5
    assert o.tif == "day"
    assert o.broker_order_id == "alp-replay-1"
    assert o.submitted_at is not None
    assert o.final_status == "accepted"


# ---------------------------------------------------------------------------
# AC-7 #3 — closed position
# ---------------------------------------------------------------------------


def test_replayer_skips_closed_position() -> None:
    """Position not in get_positions → mark_deferred_skipped; broker
    not called."""
    j = _FakeJournal(unreplayed=[_drow(id=7, symbol="MSFT")])
    b = _FakeBroker(positions=[
        Position(symbol="AAPL", qty=5, avg_entry_price=100.0,
                 market_value=500.0, unrealized_pnl=0.0),
    ])
    r = DeferredSellReplayer(broker=b, journal=j, now_fn=_now_premarket)
    report = r.run()
    assert report.replayed == 0
    assert report.skipped_no_position == 1
    assert b.submitted == []
    assert j.skipped_marks == [(7, "position_closed_externally")]


# ---------------------------------------------------------------------------
# AC-7 #4 — defense-in-depth on PDT 403
# ---------------------------------------------------------------------------


def test_replayer_pdt_budget_exceeded_leaves_row(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pre-market shouldn't 403, but defense-in-depth: leave row
    unreplayed; warning logged."""
    j = _FakeJournal(unreplayed=[_drow(id=99)])
    b = _FakeBroker(submit_responses={
        "pdt-replay-99": PDTBudgetExceeded("403 day-trade"),
    })
    r = DeferredSellReplayer(broker=b, journal=j, now_fn=_now_premarket)
    with caplog.at_level(logging.WARNING, logger="execution.virtual_exits"):
        report = r.run()
    assert report.skipped_error == 1
    assert report.replayed == 0
    assert j.replayed_marks == []
    assert j.skipped_marks == []  # row STAYS unreplayed for next session
    skipped = [
        rec for rec in caplog.records
        if getattr(rec, "event", None) == "pdt.replay.skipped"
    ]
    assert len(skipped) == 1


# ---------------------------------------------------------------------------
# AC-7 #5 — transient broker error
# ---------------------------------------------------------------------------


def test_replayer_broker_unavailable_leaves_row() -> None:
    """Transient BrokerUnavailable → row stays unreplayed (next session retries)."""
    j = _FakeJournal(unreplayed=[_drow(id=12)])
    b = _FakeBroker(submit_responses={
        "pdt-replay-12": BrokerUnavailable("503"),
    })
    r = DeferredSellReplayer(broker=b, journal=j, now_fn=_now_premarket)
    report = r.run()
    assert report.skipped_error == 1
    assert j.replayed_marks == []
    assert j.skipped_marks == []  # NOT marked


# ---------------------------------------------------------------------------
# AC-7 #6 — per-row error isolation
# ---------------------------------------------------------------------------


def test_replayer_per_row_error_isolation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One bad row + two good rows: good rows still replayed."""
    # Distinct virtual_exit_ids — sharing V across rows isn't a real
    # production state (each V spawns at most one deferred_sells row),
    # and after FINDING-3 the first success transitions V to 'submitted'
    # which would supersede later rows pointing at the same V.
    j = _FakeJournal(unreplayed=[
        _drow(id=1, virtual_exit_id=101),
        _drow(id=2, virtual_exit_id=102),  # this one will raise
        _drow(id=3, virtual_exit_id=103),
    ])
    b = _FakeBroker(submit_responses={
        "pdt-replay-2": RuntimeError("unexpected SDK error"),
    })
    r = DeferredSellReplayer(broker=b, journal=j, now_fn=_now_premarket)
    with caplog.at_level(logging.ERROR, logger="execution.virtual_exits"):
        report = r.run()
    assert report.replayed == 2
    assert report.skipped_error == 1
    replayed_ids = {did for did, _ in j.replayed_marks}
    assert replayed_ids == {1, 3}
    errors = [
        rec for rec in caplog.records
        if getattr(rec, "event", None) == "pdt.replay.row_error"
    ]
    assert len(errors) == 1


# ---------------------------------------------------------------------------
# AC-7 #7 — empty queue
# ---------------------------------------------------------------------------


def test_replayer_idempotent_on_empty_queue() -> None:
    """No unreplayed rows → no broker calls; report all zeros."""
    j = _FakeJournal(unreplayed=[])
    b = _FakeBroker()
    r = DeferredSellReplayer(broker=b, journal=j, now_fn=_now_premarket)
    report = r.run()
    assert report == ReplayReport(0, 0, 0, 0)
    assert b.submitted == []


# ---------------------------------------------------------------------------
# AC-7 #8 / AC-6 — boot-time idempotency
# ---------------------------------------------------------------------------


def test_duplicate_client_order_id_reconciles() -> None:
    """Mid-replay crash: a previous boot already submitted; current
    boot's submit fails with duplicate client_order_id; replayer looks
    up the existing order and marks the row replayed."""
    class _DupAPIError(APIError):
        def __init__(self) -> None:
            Exception.__init__(self, "client_order_id must be unique")
            self._status_code = 422

        @property  # type: ignore[override]
        def status_code(self) -> int:  # type: ignore[override]
            return self._status_code

    j = _FakeJournal(unreplayed=[_drow(id=55, execution_id=400)])
    b = _FakeBroker(submit_responses={"pdt-replay-55": _DupAPIError()})
    # The previous boot's order is what the lookup returns.
    b.set_lookup_response("pdt-replay-55", SubmittedOrder(
        broker_order_id="alp-prior-boot-1",
        client_order_id="pdt-replay-55", symbol="AAPL", side="sell",
        qty=5, order_type="market", status="accepted",
        submitted_at=dt.datetime(2026, 5, 7, 9, 0, tzinfo=dt.UTC),
    ))
    r = DeferredSellReplayer(broker=b, journal=j, now_fn=_now_premarket)
    report = r.run()
    assert report.replayed == 1
    assert b.lookups == ["pdt-replay-55"]
    assert j.replayed_marks == [(55, "alp-prior-boot-1")]
    # HOTFIX-2026-05-08: the duplicate-reconcile branch must also journal
    # the sell so resumed-from-crash paths don't leave the reconciler
    # halting. Uses the EXISTING (prior-boot) broker_order_id, not the
    # failed dup-submit attempt.
    assert len(j.orders) == 1
    o = j.orders[0]
    assert o.execution_id == 400
    assert o.role == "stop"
    assert o.broker_order_id == "alp-prior-boot-1"
    assert o.side == "sell"
    assert o.order_type == "market"
    assert o.qty == 5


# ---------------------------------------------------------------------------
# PDT-SUNSET-2026-06-04: FINDING-3 — round-3 review carry-forward closure.
# Replayer must transition the source virtual_exit to 'submitted' on both
# the success path and the duplicate-reconcile path so the next scanner
# tick filters it out via list_active_virtual_exits().
# ---------------------------------------------------------------------------


def test_replayer_marks_source_virtual_exit_submitted_on_success() -> None:
    """FINDING-3: success-path replay transitions V from 'active' to
    'submitted' with the broker order id from the submit response."""
    d = _drow(id=42, virtual_exit_id=900)
    j = _FakeJournal(unreplayed=[d])
    b = _FakeBroker()
    r = DeferredSellReplayer(broker=b, journal=j, now_fn=_now_premarket)
    report = r.run()
    assert report.replayed == 1
    # The broker_order_id passed to mark_virtual_exit_submitted matches
    # the submit response's broker_order_id.
    assert j.virtual_exits_submitted == [(900, "alp-replay-1")]
    # And the source V is now 'submitted' from the next scanner tick's POV.
    assert j.get_virtual_exit_state(900) == "submitted"


def test_replayer_marks_source_virtual_exit_submitted_on_duplicate_reconcile() -> None:
    """FINDING-3: dupe-reconcile path transitions V from 'active' to
    'submitted' with the existing broker order id (the prior-boot order
    that survived the mid-replay crash)."""
    class _DupAPIError(APIError):
        def __init__(self) -> None:
            Exception.__init__(self, "client_order_id must be unique")
            self._status_code = 422

        @property  # type: ignore[override]
        def status_code(self) -> int:  # type: ignore[override]
            return self._status_code

    d = _drow(id=66, virtual_exit_id=901)
    j = _FakeJournal(unreplayed=[d])
    b = _FakeBroker(submit_responses={"pdt-replay-66": _DupAPIError()})
    b.set_lookup_response("pdt-replay-66", SubmittedOrder(
        broker_order_id="alp-prior-boot-77",
        client_order_id="pdt-replay-66", symbol="AAPL", side="sell",
        qty=5, order_type="market", status="accepted",
        submitted_at=dt.datetime(2026, 5, 7, 9, 0, tzinfo=dt.UTC),
    ))
    r = DeferredSellReplayer(broker=b, journal=j, now_fn=_now_premarket)
    report = r.run()
    assert report.replayed == 1
    # The broker_order_id matches the EXISTING (prior-boot) order, not
    # the dup-failed submit attempt.
    assert j.virtual_exits_submitted == [(901, "alp-prior-boot-77")]
    assert j.get_virtual_exit_state(901) == "submitted"


# ---------------------------------------------------------------------------
# AC-7 #9 — thesis_close role uses market sell
# ---------------------------------------------------------------------------


def test_replayer_role_thesis_close_uses_market_order() -> None:
    """Deferred row with role='thesis_close' → submitted as market sell."""
    j = _FakeJournal(unreplayed=[_drow(id=21, role="thesis_close")])
    b = _FakeBroker()
    r = DeferredSellReplayer(broker=b, journal=j, now_fn=_now_premarket)
    r.run()
    assert b.submitted[0].order_type == "market"
    assert b.submitted[0].side == "sell"


# ---------------------------------------------------------------------------
# Supersede-race guard (S-PDT-5 follow-up).
# ---------------------------------------------------------------------------


def test_replayer_skips_superseded_row(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """D-073 / supersede-race guard. A deferred_sells row whose source
    virtual_exit transitioned to 'submitted' (won the EV race on a
    later same-session tick) must NOT be replayed: the position is
    already closed at the broker, so submitting would short an
    unrelated freshly-reopened position. Reviewer ADV-5 + FINDING-1."""
    d = _drow(id=77, virtual_exit_id=500)
    # Source virtual_exit was submitted later same-session.
    j = _FakeJournal(unreplayed=[d], ve_states={500: "submitted"})
    b = _FakeBroker()
    r = DeferredSellReplayer(broker=b, journal=j, now_fn=_now_premarket)
    with caplog.at_level(logging.INFO, logger="execution.virtual_exits"):
        report = r.run()
    assert report.replayed == 0
    assert report.skipped_superseded == 1
    assert b.submitted == []  # NO broker submit
    assert j.replayed_marks == []
    # Row exits the unreplayed queue with a SUPERSEDED:* note.
    assert j.skipped_marks == [(77, "SUPERSEDED:submitted")]
    skipped = [
        rec for rec in caplog.records
        if getattr(rec, "event", None) == "pdt.replay.skipped_superseded"
    ]
    assert len(skipped) == 1
    assert skipped[0].source_state == "submitted"


def test_replayer_skips_superseded_by_obsolete_virtual_exit() -> None:
    """A deferred_sells row whose source virtual_exit was marked
    obsolete (position closed externally before next session) is also
    skipped — same protective rationale."""
    d = _drow(id=88, virtual_exit_id=600)
    j = _FakeJournal(unreplayed=[d], ve_states={600: "obsolete"})
    b = _FakeBroker()
    r = DeferredSellReplayer(broker=b, journal=j, now_fn=_now_premarket)
    report = r.run()
    assert report.replayed == 0
    assert report.skipped_superseded == 1
    assert b.submitted == []
    assert j.skipped_marks == [(88, "SUPERSEDED:obsolete")]


def test_replayer_does_not_skip_active_virtual_exit() -> None:
    """Regression: virtual_exit still 'active' → row IS replayed."""
    d = _drow(id=99, virtual_exit_id=700)
    j = _FakeJournal(unreplayed=[d], ve_states={700: "active"})
    b = _FakeBroker()
    r = DeferredSellReplayer(broker=b, journal=j, now_fn=_now_premarket)
    report = r.run()
    assert report.replayed == 1
    assert report.skipped_superseded == 0
    assert len(b.submitted) == 1


def test_replayer_supersede_check_runs_before_position_check() -> None:
    """Ordering: even if the broker no longer has a position for the
    symbol, a 'submitted' source virtual_exit short-circuits to the
    SUPERSEDED path (more specific reason). Avoids the misleading
    `position_closed_externally` audit on rows that were really
    superseded by a same-session scanner submit."""
    d = _drow(id=44, symbol="MSFT", virtual_exit_id=800)
    j = _FakeJournal(unreplayed=[d], ve_states={800: "submitted"})
    # Broker has AAPL only (so the closed-position check would also fire).
    b = _FakeBroker(positions=[Position(
        symbol="AAPL", qty=5, avg_entry_price=100.0,
        market_value=500.0, unrealized_pnl=0.0,
    )])
    r = DeferredSellReplayer(broker=b, journal=j, now_fn=_now_premarket)
    report = r.run()
    assert report.skipped_superseded == 1
    assert report.skipped_no_position == 0
    assert j.skipped_marks == [(44, "SUPERSEDED:submitted")]


def test_replayer_role_tp_uses_limit_order() -> None:
    """Deferred row with role='tp' → limit sell at trigger_price."""
    j = _FakeJournal(unreplayed=[_drow(id=22, role="tp", trigger_price=110.0)])
    b = _FakeBroker()
    r = DeferredSellReplayer(broker=b, journal=j, now_fn=_now_premarket)
    r.run()
    submitted = b.submitted[0]
    assert submitted.order_type == "limit"
    assert submitted.limit_price == 110.0


def test_replayer_positions_unavailable_aborts_safely(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If broker.get_positions raises, replayer logs and returns
    all-zeros — does NOT touch any row."""
    j = _FakeJournal(unreplayed=[_drow(id=1)])
    b = _FakeBroker(positions_raises=BrokerUnavailable("offline"))
    r = DeferredSellReplayer(broker=b, journal=j, now_fn=_now_premarket)
    with caplog.at_level(logging.ERROR, logger="execution.virtual_exits"):
        report = r.run()
    assert report == ReplayReport(0, 0, 0, 0)
    assert j.replayed_marks == []
    assert j.skipped_marks == []


def test_replayer_emits_run_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """AC: pdt.replay.run log emitted at INFO with the four counts."""
    j = _FakeJournal(unreplayed=[_drow(id=1)])
    b = _FakeBroker()
    r = DeferredSellReplayer(broker=b, journal=j, now_fn=_now_premarket)
    with caplog.at_level(logging.INFO, logger="execution.virtual_exits"):
        r.run()
    runs = [
        rec for rec in caplog.records
        if getattr(rec, "event", None) == "pdt.replay.run"
    ]
    assert len(runs) == 1
    assert runs[0].replayed_count == 1

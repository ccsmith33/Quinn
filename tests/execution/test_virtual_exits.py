# PDT-SUNSET-2026-06-04: ADR-009 §3.2 + §"EV computation contract".
"""S-PDT-4 — `src/execution/virtual_exits.py` unit tests.

References: story S-PDT-4 AC-9; pdt-budget-architecture.md §10.1.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import pytest

from broker.alpaca import BrokerUnavailable
from broker.protocol import (
    AccountSnapshot,
    OrderRequest,
    Position,
    Quote,
    SubmittedOrder,
)
from execution.pdt_budget import PDTBudgetExceeded, PDTState
from execution.virtual_exits import (
    ScannerReport,
    VirtualExitScanner,
    compute_ev,
)
from journal.models import DeferredSellRow, VirtualExitRow


def _ve(
    *,
    id: int = 1,
    symbol: str = "AAPL",
    role: str = "stop",
    qty: int = 10,
    entry_price: float = 100.0,
    stop_price: float | None = 95.0,
    tp_price: float | None = None,
    state: str = "active",
    execution_id: int = 1,
    proposal_id: int = 1,
) -> VirtualExitRow:
    return VirtualExitRow(
        id=id,
        execution_id=execution_id,
        proposal_id=proposal_id,
        symbol=symbol,
        qty=qty,
        role=role,  # type: ignore[arg-type]
        entry_price=entry_price,
        stop_price=stop_price,
        tp_price=tp_price,
        state=state,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# compute_ev (AC-9 #1, #2, #3)
# ---------------------------------------------------------------------------


def test_compute_ev_stop_loss_avoided() -> None:
    """role='stop': (current - stop_price) * qty.
    current=99 below stop=100 with qty=10 → -10.0 (loss when crystallized)."""
    e = _ve(role="stop", qty=10, entry_price=100.0, stop_price=100.0)
    assert compute_ev(e, current_price=99.0) == pytest.approx(-10.0)


def test_compute_ev_tp_gain_locked() -> None:
    """role='tp': (tp_price - entry) * qty.
    tp=110, entry=100, qty=10 → +100.0."""
    e = _ve(
        role="tp", qty=10, entry_price=100.0,
        stop_price=None, tp_price=110.0,
    )
    assert compute_ev(e, current_price=111.0) == pytest.approx(100.0)


def test_compute_ev_unknown_role_raises() -> None:
    """thesis_close is NOT a virtual_exits role — raises."""
    e = VirtualExitRow.model_construct(
        id=1, execution_id=1, proposal_id=1, symbol="X", qty=1,
        role="thesis_close", entry_price=100.0, state="active",
    )
    with pytest.raises(ValueError, match="unknown virtual_exit role"):
        compute_ev(e, current_price=100.0)


def test_compute_ev_stop_missing_price_raises() -> None:
    e = _ve(role="stop", stop_price=None)
    with pytest.raises(ValueError, match="missing stop_price"):
        compute_ev(e, current_price=99.0)


def test_compute_ev_tp_missing_price_raises() -> None:
    e = _ve(role="tp", stop_price=None, tp_price=None)
    with pytest.raises(ValueError, match="missing tp_price"):
        compute_ev(e, current_price=110.0)


# ---------------------------------------------------------------------------
# _is_threshold_crossed (AC-9 #4..#7) — tested through `run_tick` behavior.
# ---------------------------------------------------------------------------


def test_is_threshold_crossed_stop_at_threshold() -> None:
    from execution.virtual_exits import _is_threshold_crossed
    e = _ve(role="stop", stop_price=100.0)
    assert _is_threshold_crossed(e, current_price=100.0) is True  # inclusive


def test_is_threshold_crossed_stop_above_threshold() -> None:
    from execution.virtual_exits import _is_threshold_crossed
    e = _ve(role="stop", stop_price=100.0)
    assert _is_threshold_crossed(e, current_price=100.01) is False


def test_is_threshold_crossed_tp_at_threshold() -> None:
    from execution.virtual_exits import _is_threshold_crossed
    e = _ve(role="tp", stop_price=None, tp_price=110.0)
    assert _is_threshold_crossed(e, current_price=110.0) is True


def test_is_threshold_crossed_tp_below_threshold() -> None:
    from execution.virtual_exits import _is_threshold_crossed
    e = _ve(role="tp", stop_price=None, tp_price=110.0)
    assert _is_threshold_crossed(e, current_price=109.99) is False


# ---------------------------------------------------------------------------
# Fakes for scanner tests
# ---------------------------------------------------------------------------


class _FakeJournal:
    def __init__(self, *, exits: list[VirtualExitRow] | None = None) -> None:
        self._exits = exits or []
        self.submitted_marks: list[tuple[int, str]] = []
        self.obsolete_marks: list[tuple[int, str]] = []
        self.deferred_sells: list[DeferredSellRow] = []
        self._dseq = 0

    def list_active_virtual_exits(self) -> list[VirtualExitRow]:
        return list(self._exits)

    def mark_virtual_exit_submitted(self, vid: int, broker_order_id: str) -> None:
        self.submitted_marks.append((vid, broker_order_id))

    def mark_virtual_exit_obsolete(self, vid: int, reason: str) -> None:
        self.obsolete_marks.append((vid, reason))

    def insert_deferred_sell(self, row: DeferredSellRow) -> int:
        self._dseq += 1
        self.deferred_sells.append(row)
        return self._dseq


class _FakeBroker:
    def __init__(
        self,
        *,
        account: AccountSnapshot | None = None,
        positions: list[Position] | None = None,
        quotes: dict[str, float] | None = None,
        submit_raises: dict[str, Exception] | None = None,
    ) -> None:
        self._account = account or AccountSnapshot(
            equity=22_300.0, cash=5_000.0, buying_power=44_600.0,
            long_market_value=17_300.0, daypl=0.0,
            snapshot_at=dt.datetime(2026, 5, 7, 14, 30, tzinfo=dt.UTC),
            last_equity=22_000.0, daytrade_count=0,
        )
        self._positions = positions if positions is not None else [
            Position(symbol=s, qty=10, avg_entry_price=100.0,
                     market_value=1000.0, unrealized_pnl=0.0)
            for s in (quotes.keys() if quotes else ["AAPL"])
        ]
        self._quotes = quotes or {"AAPL": 100.0}
        self._submit_raises = submit_raises or {}
        self.submitted: list[OrderRequest] = []
        self._submit_seq = 0

    def get_account(self) -> AccountSnapshot:
        return self._account

    def get_positions(self) -> list[Position]:
        return list(self._positions)

    def get_quote(self, symbol: str) -> Quote:
        return Quote(
            symbol=symbol, bid=self._quotes[symbol] - 0.05,
            ask=self._quotes[symbol] + 0.05, last=self._quotes[symbol],
            ts=dt.datetime(2026, 5, 7, 14, 30, tzinfo=dt.UTC),
        )

    def submit_order(self, req: OrderRequest) -> SubmittedOrder:
        self.submitted.append(req)
        if req.client_order_id in self._submit_raises:
            raise self._submit_raises[req.client_order_id]
        self._submit_seq += 1
        return SubmittedOrder(
            broker_order_id=f"alp-pdt-{self._submit_seq}",
            client_order_id=req.client_order_id,
            symbol=req.symbol, side=req.side, qty=req.qty,
            order_type=req.order_type, status="accepted",
            submitted_at=dt.datetime(2026, 5, 7, 14, 30, tzinfo=dt.UTC),
            limit_price=req.limit_price, stop_price=req.stop_price,
        )

    # Unused.
    def cancel_order(self, *_: Any) -> None: ...
    def get_order_by_client_id(self, *_: Any, **__: Any) -> Any: return None
    def replace_stop_order(self, *_: Any, **__: Any) -> Any:
        raise NotImplementedError


def _active(active: bool = True) -> PDTState:
    return PDTState(_active=active, _last_equity=22_000.0)


# ---------------------------------------------------------------------------
# Scanner behavior (AC-9 #8..#16)
# ---------------------------------------------------------------------------


def test_scanner_inactive_short_circuits() -> None:
    """AC-9 #8: pdt_state.is_active()==False → no broker calls; report all zeros."""
    j = _FakeJournal(exits=[_ve(id=1)])
    b = _FakeBroker()
    s = VirtualExitScanner(broker=b, journal=j, pdt_state=_active(False))
    report = s.run_tick()
    assert report == ScannerReport(0, 0, 0, 0)
    assert b.submitted == []


def test_scanner_no_active_exits_idle_tick() -> None:
    """No active rows → quiet tick, no I/O, all zeros."""
    j = _FakeJournal(exits=[])
    b = _FakeBroker()
    s = VirtualExitScanner(broker=b, journal=j, pdt_state=_active())
    report = s.run_tick()
    assert report == ScannerReport(0, 0, 0, 0)
    assert b.submitted == []


def test_scanner_no_ready_exits_idle_tick() -> None:
    """AC-9 #9: exits exist but threshold not crossed → ready=0, submitted=0."""
    j = _FakeJournal(exits=[_ve(id=1, role="stop", stop_price=95.0)])
    b = _FakeBroker(quotes={"AAPL": 100.0})  # 100 > 95 → not crossed
    s = VirtualExitScanner(broker=b, journal=j, pdt_state=_active())
    report = s.run_tick()
    assert report.ready == 0
    assert report.submitted == 0
    assert b.submitted == []


def test_scanner_ranks_by_ev_descending() -> None:
    """AC-9 #10: 3 ready, EVs [+5, -1, -10] → submitted in EV-desc order
    up to budget (here budget=3 = whole list)."""
    # All stops crossed at current=99 (stop >= 99 fires).
    # EV = (99 - stop) * 10.
    # stop=99 → 0; stop=99.1 → -1; stop=100.0 → -10; stop=98.5 → wouldn't cross.
    # Pick stop levels at-or-above 99 to all cross:
    # stop=99 → EV = 0  ; stop=100 → EV = -10 ; stop=99.5 → EV = -5
    # Pivot to current=100 with stop levels 99..101 to get +5 / -1 / -10:
    # current=100; stop=99 → +10 (NOT crossed since 100>99? CROSSED=current<=stop=> 100<=99 False).
    # Per AC-4: stop fires at current <= stop_price. So stop must be >= current to cross.
    # current=99: stop=104 → EV = (99-104)*10 = -50; stop=100→-10; stop=99.5→-5.
    exits = [
        _ve(id=10, role="stop", stop_price=104.0, qty=10),  # EV = -50
        _ve(id=11, role="stop", stop_price=99.5, qty=10),   # EV = -5  (top)
        _ve(id=12, role="stop", stop_price=100.0, qty=10),  # EV = -10
    ]
    j = _FakeJournal(exits=exits)
    b = _FakeBroker(quotes={"AAPL": 99.0})  # all crossed
    s = VirtualExitScanner(broker=b, journal=j, pdt_state=_active())
    report = s.run_tick()
    assert report.ready == 3
    assert report.submitted == 3
    # Submission order: id=11 (-5) first, id=12 (-10), id=10 (-50) last.
    assert [r.client_order_id for r in b.submitted] == [
        "pdt-vexit-11", "pdt-vexit-12", "pdt-vexit-10",
    ]


def test_scanner_tie_breaker_lower_id_wins() -> None:
    """AC-9 #11: identical EV → lower id submitted first (FIFO)."""
    exits = [
        _ve(id=20, role="stop", stop_price=99.0, qty=10),  # EV=0
        _ve(id=10, role="stop", stop_price=99.0, qty=10),  # EV=0
    ]
    j = _FakeJournal(exits=exits)
    b = _FakeBroker(quotes={"AAPL": 99.0})  # both crossed at threshold
    s = VirtualExitScanner(broker=b, journal=j, pdt_state=_active())
    s.run_tick()
    assert [r.client_order_id for r in b.submitted] == [
        "pdt-vexit-10", "pdt-vexit-20",
    ]


def test_scanner_budget_2_of_5_ready() -> None:
    """AC-9 #12: 5 ready, daytrade_count=1 → budget=2; top-2 submitted;
    bottom-3 deferred with deferred_reason='ev_lost'."""
    # Construct with EVs +50,+40,+30,+20,+10
    exits = [
        _ve(id=1, role="tp", stop_price=None, tp_price=110.0, entry_price=100.0, qty=5),  # 50
        _ve(id=2, role="tp", stop_price=None, tp_price=108.0, entry_price=100.0, qty=5),  # 40
        _ve(id=3, role="tp", stop_price=None, tp_price=106.0, entry_price=100.0, qty=5),  # 30
        _ve(id=4, role="tp", stop_price=None, tp_price=104.0, entry_price=100.0, qty=5),  # 20
        _ve(id=5, role="tp", stop_price=None, tp_price=102.0, entry_price=100.0, qty=5),  # 10
    ]
    account = AccountSnapshot(
        equity=22_300.0, cash=5_000.0, buying_power=44_600.0,
        long_market_value=17_300.0, daypl=0.0,
        snapshot_at=dt.datetime(2026, 5, 7, 14, 30, tzinfo=dt.UTC),
        last_equity=22_000.0, daytrade_count=1,  # budget = 3-1-0 = 2
    )
    j = _FakeJournal(exits=exits)
    b = _FakeBroker(account=account, quotes={"AAPL": 115.0})  # all tp crossed
    s = VirtualExitScanner(broker=b, journal=j, pdt_state=_active())
    report = s.run_tick()
    assert report.ready == 5
    assert report.submitted == 2
    assert report.deferred_ev == 3
    assert report.deferred_403 == 0
    assert [r.client_order_id for r in b.submitted] == [
        "pdt-vexit-1", "pdt-vexit-2",
    ]
    deferred_ids = sorted(d.virtual_exit_id for d in j.deferred_sells)
    assert deferred_ids == [3, 4, 5]
    for d in j.deferred_sells:
        assert d.deferred_reason == "ev_lost"


def test_scanner_budget_zero_all_deferred() -> None:
    """AC-9 #13: daytrade_count=3 → budget=0; all ready deferred."""
    exits = [
        _ve(id=1, role="tp", stop_price=None, tp_price=110.0, qty=5),
        _ve(id=2, role="tp", stop_price=None, tp_price=108.0, qty=5),
    ]
    account = AccountSnapshot(
        equity=22_300.0, cash=5_000.0, buying_power=44_600.0,
        long_market_value=17_300.0, daypl=0.0,
        snapshot_at=dt.datetime(2026, 5, 7, 14, 30, tzinfo=dt.UTC),
        last_equity=22_000.0, daytrade_count=3,  # budget=0
    )
    j = _FakeJournal(exits=exits)
    b = _FakeBroker(account=account, quotes={"AAPL": 115.0})
    s = VirtualExitScanner(broker=b, journal=j, pdt_state=_active())
    report = s.run_tick()
    assert report.submitted == 0
    assert report.deferred_ev == 2
    assert b.submitted == []
    assert {d.deferred_reason for d in j.deferred_sells} == {"ev_lost"}


def test_scanner_403_routes_to_deferred(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """AC-9 #14: PDTBudgetExceeded → deferred_sells row with
    deferred_reason='pdt_403'; warning log; no exception escapes."""
    exits = [_ve(id=99, role="stop", stop_price=100.0, qty=5)]
    j = _FakeJournal(exits=exits)
    b = _FakeBroker(
        quotes={"AAPL": 99.0},
        submit_raises={"pdt-vexit-99": PDTBudgetExceeded("client cannot day-trade")},
    )
    s = VirtualExitScanner(broker=b, journal=j, pdt_state=_active())
    with caplog.at_level(logging.WARNING, logger="execution.virtual_exits"):
        report = s.run_tick()
    assert report.deferred_403 == 1
    assert report.submitted == 0
    assert len(j.deferred_sells) == 1
    assert j.deferred_sells[0].deferred_reason == "pdt_403"
    assert j.deferred_sells[0].virtual_exit_id == 99
    diverged = [
        r for r in caplog.records
        if getattr(r, "event", None) == "pdt.local_budget_diverged"
    ]
    assert len(diverged) == 1


def test_scanner_unrelated_broker_unavailable_skips_no_defer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """AC-9 #15 / AC-6: non-PDT BrokerUnavailable → no deferred row;
    warning logged; continues to next ready exit."""
    exits = [
        _ve(id=1, role="stop", stop_price=100.0, qty=5),
        _ve(id=2, role="stop", stop_price=100.0, qty=5),
    ]
    j = _FakeJournal(exits=exits)
    b = _FakeBroker(
        quotes={"AAPL": 99.0},
        submit_raises={"pdt-vexit-1": BrokerUnavailable("503")},
    )
    s = VirtualExitScanner(broker=b, journal=j, pdt_state=_active())
    with caplog.at_level(logging.WARNING, logger="execution.virtual_exits"):
        report = s.run_tick()
    # First failed transiently; no defer. Second succeeded.
    assert report.submitted == 1
    assert report.deferred_403 == 0
    assert report.deferred_ev == 0
    assert j.deferred_sells == []
    transient = [
        r for r in caplog.records
        if getattr(r, "event", None) == "pdt.scanner.broker_unavailable_transient"
    ]
    assert len(transient) == 1


def test_scanner_position_closed_externally_marks_obsolete() -> None:
    """AC-9 #16 / AC-7: virtual exit for symbol not in broker positions
    → row marked obsolete; not submitted; not deferred."""
    exits = [_ve(id=42, symbol="MSFT", role="stop", stop_price=100.0)]
    j = _FakeJournal(exits=exits)
    # broker has AAPL but no MSFT
    b = _FakeBroker(
        positions=[Position(
            symbol="AAPL", qty=10, avg_entry_price=100.0,
            market_value=1000.0, unrealized_pnl=0.0,
        )],
        quotes={"AAPL": 99.0},
    )
    s = VirtualExitScanner(broker=b, journal=j, pdt_state=_active())
    report = s.run_tick()
    assert j.obsolete_marks == [(42, "position_closed_externally")]
    assert b.submitted == []
    assert j.deferred_sells == []
    assert report.submitted == 0


def test_scanner_client_order_id_is_pdt_vexit_id() -> None:
    """AC-9 #17 / AC-3: submitted OrderRequest.client_order_id ==
    'pdt-vexit-{id}'."""
    exits = [_ve(id=777, role="stop", stop_price=100.0)]
    j = _FakeJournal(exits=exits)
    b = _FakeBroker(quotes={"AAPL": 99.0})
    s = VirtualExitScanner(broker=b, journal=j, pdt_state=_active())
    s.run_tick()
    assert b.submitted[0].client_order_id == "pdt-vexit-777"


def test_scanner_stop_uses_market_order() -> None:
    """AC-3: submitted stop sell is a market order (post-cross,
    immediate exit), not a stop order."""
    exits = [_ve(id=1, role="stop", stop_price=100.0)]
    j = _FakeJournal(exits=exits)
    b = _FakeBroker(quotes={"AAPL": 99.0})
    s = VirtualExitScanner(broker=b, journal=j, pdt_state=_active())
    s.run_tick()
    assert b.submitted[0].order_type == "market"
    assert b.submitted[0].side == "sell"
    assert b.submitted[0].qty == 10
    assert b.submitted[0].tif == "day"


def test_scanner_tp_uses_limit_order() -> None:
    """AC-3: submitted TP sell is a limit order at tp_price."""
    exits = [_ve(
        id=1, role="tp", stop_price=None, tp_price=110.0,
        entry_price=100.0, qty=5,
    )]
    j = _FakeJournal(exits=exits)
    b = _FakeBroker(quotes={"AAPL": 111.0})
    s = VirtualExitScanner(broker=b, journal=j, pdt_state=_active())
    s.run_tick()
    assert b.submitted[0].order_type == "limit"
    assert b.submitted[0].limit_price == 110.0
    assert b.submitted[0].side == "sell"


def test_scanner_marks_submitted_with_broker_order_id() -> None:
    """On successful broker submit, mark_virtual_exit_submitted is
    called with the broker's order id."""
    exits = [_ve(id=5, role="stop", stop_price=100.0)]
    j = _FakeJournal(exits=exits)
    b = _FakeBroker(quotes={"AAPL": 99.0})
    s = VirtualExitScanner(broker=b, journal=j, pdt_state=_active())
    s.run_tick()
    assert len(j.submitted_marks) == 1
    vid, boid = j.submitted_marks[0]
    assert vid == 5
    assert boid == "alp-pdt-1"


def test_scanner_emits_tick_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """AC-8: pdt.scanner.tick log emitted at INFO with the report fields."""
    exits = [_ve(id=1, role="stop", stop_price=100.0)]
    j = _FakeJournal(exits=exits)
    b = _FakeBroker(quotes={"AAPL": 99.0})
    s = VirtualExitScanner(broker=b, journal=j, pdt_state=_active())
    with caplog.at_level(logging.INFO, logger="execution.virtual_exits"):
        s.run_tick()
    ticks = [
        r for r in caplog.records
        if getattr(r, "event", None) == "pdt.scanner.tick"
    ]
    assert len(ticks) == 1
    rec = ticks[0]
    assert rec.ready_count == 1
    assert rec.submitted == 1
    assert rec.budget == 3


def test_scanner_skip_only_tick_emits_no_tick_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """AC-8: gate-inactive ticks emit no tick log."""
    j = _FakeJournal(exits=[_ve(id=1)])
    b = _FakeBroker()
    s = VirtualExitScanner(broker=b, journal=j, pdt_state=_active(False))
    with caplog.at_level(logging.INFO, logger="execution.virtual_exits"):
        s.run_tick()
    ticks = [
        r for r in caplog.records
        if getattr(r, "event", None) == "pdt.scanner.tick"
    ]
    assert ticks == []


# ---------------------------------------------------------------------------
# FINDING-2: scanner local pending counter (ADR-009 §"Why pending_dt_orders
# is tracked locally"). The pending counter prevents within-tick
# over-allocation when an external day-trade bumps Alpaca's count between
# our `get_account()` snapshot and a per-row submit.
# ---------------------------------------------------------------------------


def test_scanner_local_pending_counter_increments_on_successful_submit() -> None:
    """3 ready, daytrade_count=0 → budget=3 at start. After two
    successful submits, the local pending counter is 2, so
    compute_budget_remaining(account, pending=2) = 1. The third
    submit attempts and (in this test, because the broker accepts)
    succeeds — pending=3 at end. Submitted=3, deferred=0."""
    exits = [
        _ve(id=1, role="stop", stop_price=100.0, qty=5),
        _ve(id=2, role="stop", stop_price=100.0, qty=5),
        _ve(id=3, role="stop", stop_price=100.0, qty=5),
    ]
    j = _FakeJournal(exits=exits)
    # daytrade_count=0 → initial budget=3.
    b = _FakeBroker(quotes={"AAPL": 99.0})
    s = VirtualExitScanner(broker=b, journal=j, pdt_state=_active())
    report = s.run_tick()
    assert report.submitted == 3
    assert report.deferred_ev == 0
    assert report.deferred_403 == 0


def test_scanner_local_pending_blocks_within_tick_over_allocation() -> None:
    """ADR-009 §pending_dt_orders / FINDING-2: with daytrade_count=1
    at tick start (e.g., an external operator day-trade landed before
    the scanner ran), 3 ready exits should produce 2 submits + 1
    defer — the local pending counter prevents the third submit even
    though without it the static slice would have allowed it.

    Budget evolution:
      pending=0 → compute_budget_remaining(dc=1, p=0) = 2  → submit
      pending=1 → compute_budget_remaining(dc=1, p=1) = 1  → submit
      pending=2 → compute_budget_remaining(dc=1, p=2) = 0  → defer
    """
    from broker.protocol import AccountSnapshot

    exits = [
        _ve(id=1, role="stop", stop_price=100.0, qty=5),
        _ve(id=2, role="stop", stop_price=100.0, qty=5),
        _ve(id=3, role="stop", stop_price=100.0, qty=5),
    ]
    account = AccountSnapshot(
        equity=22_300.0, cash=5_000.0, buying_power=44_600.0,
        long_market_value=17_300.0, daypl=0.0,
        snapshot_at=dt.datetime(2026, 5, 7, 14, 30, tzinfo=dt.UTC),
        last_equity=22_000.0, daytrade_count=1,
    )
    j = _FakeJournal(exits=exits)
    b = _FakeBroker(account=account, quotes={"AAPL": 99.0})
    s = VirtualExitScanner(broker=b, journal=j, pdt_state=_active())
    report = s.run_tick()
    assert report.submitted == 2
    assert report.deferred_ev == 1
    assert report.deferred_403 == 0
    # The third row (id=3) is the one deferred.
    assert len(j.deferred_sells) == 1
    assert j.deferred_sells[0].virtual_exit_id == 3
    assert j.deferred_sells[0].deferred_reason == "ev_lost"


def test_scanner_local_pending_not_incremented_on_403() -> None:
    """When a submit raises PDTBudgetExceeded, the local pending
    counter is NOT incremented (Alpaca rejected, so no broker-side
    slot was consumed beyond what daytrade_count already reports).
    The remaining ready exit can still attempt submit on the next
    iteration."""
    exits = [
        _ve(id=1, role="stop", stop_price=100.0, qty=5),
        _ve(id=2, role="stop", stop_price=100.0, qty=5),
    ]
    j = _FakeJournal(exits=exits)
    # First submit raises 403; second should still attempt.
    b = _FakeBroker(
        quotes={"AAPL": 99.0},
        submit_raises={"pdt-vexit-1": PDTBudgetExceeded("403")},
    )
    s = VirtualExitScanner(broker=b, journal=j, pdt_state=_active())
    report = s.run_tick()
    # First → 403 → deferred_403; pending unchanged.
    # Second → submit attempted (pending was 0, budget=3>0); succeeds.
    assert report.submitted == 1
    assert report.deferred_403 == 1


def test_scanner_local_pending_not_incremented_on_transient_failure() -> None:
    """Transient BrokerUnavailable does NOT burn a slot — no pending
    increment, no defer (next tick retries)."""
    exits = [
        _ve(id=1, role="stop", stop_price=100.0, qty=5),
        _ve(id=2, role="stop", stop_price=100.0, qty=5),
    ]
    j = _FakeJournal(exits=exits)
    b = _FakeBroker(
        quotes={"AAPL": 99.0},
        submit_raises={"pdt-vexit-1": BrokerUnavailable("503")},
    )
    s = VirtualExitScanner(broker=b, journal=j, pdt_state=_active())
    report = s.run_tick()
    # Row 1 transient-skipped (no defer); row 2 submitted successfully.
    assert report.submitted == 1
    assert report.deferred_ev == 0
    assert report.deferred_403 == 0

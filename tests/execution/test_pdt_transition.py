# PDT-TRANSITION-D-077: tests for the one-shot ADR-012 §4.2 converter.
# This file survives P2 with the converter; both are removed in the
# post-soak cleanup (ADR-012 "Consequences").
"""PDTTransitionConverter — active virtual exits become real GTC broker orders.

ADR-012 invariant under test: at every instant, every open position has
either an active virtual exit (scanner still running) or a live GTC
protective order at the broker. The converter must never strand a
position in between — on any per-group failure the virtual exit stays
`active`.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from typing import Any

import pytest

from broker.alpaca import BrokerUnavailable
from broker.protocol import OrderRequest, Position, Quote, SubmittedOrder
from execution.pdt_transition import ConversionReport, PDTTransitionConverter
from journal.models import DeferredSellRow, OrderRow, VirtualExitRow

_NOW = dt.datetime(2026, 6, 9, 12, 0, tzinfo=dt.UTC)


def _position(symbol: str, qty: int = 10) -> Position:
    return Position(
        symbol=symbol,
        qty=qty,
        avg_entry_price=100.0,
        market_value=qty * 100.0,
        unrealized_pnl=0.0,
    )


def _quote(symbol: str, last: float) -> Quote:
    return Quote(symbol=symbol, bid=last - 0.05, ask=last + 0.05, last=last, ts=_NOW)


def _submitted(
    broker_order_id: str,
    client_order_id: str,
    symbol: str,
    qty: int,
    order_type: str = "stop",
    limit_price: float | None = None,
    stop_price: float | None = None,
    status: str = "accepted",
) -> SubmittedOrder:
    return SubmittedOrder(
        broker_order_id=broker_order_id,
        client_order_id=client_order_id,
        symbol=symbol,
        side="sell",
        qty=qty,
        order_type=order_type,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        submitted_at=_NOW,
        limit_price=limit_price,
        stop_price=stop_price,
    )


def _vexit(
    vid: int,
    *,
    execution_id: int,
    symbol: str,
    role: str,
    qty: int = 10,
    entry_price: float = 100.0,
    stop_price: float | None = None,
    tp_price: float | None = None,
    state: str = "active",
) -> VirtualExitRow:
    return VirtualExitRow(
        id=vid,
        execution_id=execution_id,
        proposal_id=execution_id,
        symbol=symbol,
        qty=qty,
        role=role,  # type: ignore[arg-type]
        entry_price=entry_price,
        stop_price=stop_price,
        tp_price=tp_price,
        state=state,  # type: ignore[arg-type]
    )


def _deferred(
    did: int,
    *,
    virtual_exit_id: int,
    symbol: str,
    role: str = "stop",
    qty: int = 10,
) -> DeferredSellRow:
    return DeferredSellRow(
        id=did,
        virtual_exit_id=virtual_exit_id,
        execution_id=virtual_exit_id,
        proposal_id=virtual_exit_id,
        symbol=symbol,
        qty=qty,
        role=role,  # type: ignore[arg-type]
        trigger_price=95.0,
        ev_at_defer=-50.0,
        deferred_reason="ev_lost",
    )


class _FakeBroker:
    """Implements the ConverterBroker surface, including WS2's
    `submit_oco_sell` per the delta §7.3 contract."""

    def __init__(
        self,
        *,
        positions: list[Position] | None = None,
        quotes: dict[str, Quote] | None = None,
        existing: dict[str, SubmittedOrder] | None = None,
    ) -> None:
        self.positions = positions or []
        self.quotes = quotes or {}
        self.existing = existing or {}
        self.submitted: list[OrderRequest] = []
        self.oco_calls: list[dict[str, Any]] = []
        self.fail_submit: Exception | None = None
        self.call_log: list[str] = []
        self._seq = 0

    def _next_id(self) -> str:
        self._seq += 1
        return f"alp-conv-{self._seq}"

    def get_positions(self) -> list[Position]:
        self.call_log.append("get_positions")
        return self.positions

    def get_quote(self, symbol: str) -> Quote:
        self.call_log.append(f"get_quote:{symbol}")
        return self.quotes[symbol]

    def get_order_by_client_id(self, client_order_id: str) -> SubmittedOrder | None:
        self.call_log.append(f"get_order_by_client_id:{client_order_id}")
        return self.existing.get(client_order_id)

    def submit_order(self, req: OrderRequest) -> SubmittedOrder:
        if self.fail_submit is not None:
            raise self.fail_submit
        self.submitted.append(req)
        self.call_log.append(f"submit_order:{req.client_order_id}")
        return _submitted(
            self._next_id(),
            req.client_order_id,
            req.symbol,
            req.qty,
            order_type=req.order_type,
            limit_price=req.limit_price,
            stop_price=req.stop_price,
        )

    def submit_oco_sell(
        self,
        *,
        symbol: str,
        qty: int,
        stop_price: float,
        limit_price: float | None,
        client_order_id: str,
    ) -> tuple[SubmittedOrder, SubmittedOrder | None]:
        if self.fail_submit is not None:
            raise self.fail_submit
        self.oco_calls.append(
            {
                "symbol": symbol,
                "qty": qty,
                "stop_price": stop_price,
                "limit_price": limit_price,
                "client_order_id": client_order_id,
            }
        )
        self.call_log.append(f"submit_oco_sell:{client_order_id}")
        stop_so = _submitted(
            self._next_id(), client_order_id, symbol, qty,
            order_type="stop", stop_price=stop_price,
        )
        tp_so = None
        if limit_price is not None:
            tp_so = _submitted(
                self._next_id(), client_order_id, symbol, qty,
                order_type="limit", limit_price=limit_price,
            )
        return stop_so, tp_so


class _FakeJournal:
    def __init__(
        self,
        exits: list[VirtualExitRow] | None = None,
        deferred: list[DeferredSellRow] | None = None,
    ) -> None:
        self.exits = {e.id: e for e in (exits or [])}
        self.deferred = deferred or []
        self.orders: list[Any] = []
        self.skipped_deferred: list[tuple[int, str]] = []
        self.call_log: list[str] = []

    def list_active_virtual_exits(self) -> list[VirtualExitRow]:
        self.call_log.append("list_active_virtual_exits")
        return [e for e in self.exits.values() if e.state == "active"]

    def mark_virtual_exit_submitted(self, vid: int, broker_order_id: str) -> None:
        self.call_log.append(f"mark_submitted:{vid}:{broker_order_id}")
        self.exits[vid].state = "submitted"
        self.exits[vid].submitted_broker_order_id = broker_order_id

    def mark_virtual_exit_obsolete(self, vid: int, reason: str) -> None:
        self.call_log.append(f"mark_obsolete:{vid}:{reason}")
        self.exits[vid].state = "obsolete"
        self.exits[vid].notes = reason

    def insert_order(self, row: Any) -> int:
        self.call_log.append(f"insert_order:{row.role}:{row.broker_order_id}")
        # Mirror the real schema: orders.broker_order_id is UNIQUE
        # (001_init.sql:150) — the W3-M1 already-journaled signal.
        if any(o.broker_order_id == row.broker_order_id for o in self.orders):
            raise sqlite3.IntegrityError(
                "UNIQUE constraint failed: orders.broker_order_id"
            )
        self.orders.append(row)
        return len(self.orders)

    def list_unreplayed_deferred_sells(self) -> list[DeferredSellRow]:
        self.call_log.append("list_unreplayed_deferred_sells")
        return [
            d
            for d in self.deferred
            if d.replayed_at is None
            and (d.id, ) not in {(s[0], ) for s in self.skipped_deferred}
        ]

    def mark_deferred_skipped(self, did: int, reason: str) -> None:
        self.call_log.append(f"mark_deferred_skipped:{did}")
        self.skipped_deferred.append((did, reason))


def _run(
    broker: _FakeBroker, journal: _FakeJournal
) -> ConversionReport:
    converter = PDTTransitionConverter(broker=broker, journal=journal)
    return converter.run()


# ---------------------------------------------------------------------------
# No-op fast path
# ---------------------------------------------------------------------------


def test_noop_when_no_active_exits_and_no_unreplayed_deferred() -> None:
    """Zero active rows + zero unreplayed rows -> no broker I/O at all."""
    broker = _FakeBroker()
    journal = _FakeJournal()
    report = _run(broker, journal)
    assert report == ConversionReport(0, 0, 0, 0, 0, 0, 0, 0)
    assert broker.call_log == []


# ---------------------------------------------------------------------------
# Conversion shapes (ADR-012 §4.2 steps 2-4)
# ---------------------------------------------------------------------------


def test_stop_tp_pair_converts_to_oco() -> None:
    """Both legs active, stop not breached -> one submit_oco_sell with both
    prices, client id keyed on the STOP virtual exit id."""
    exits = [
        _vexit(1, execution_id=42, symbol="AAPL", role="stop", stop_price=95.0),
        _vexit(2, execution_id=42, symbol="AAPL", role="tp", tp_price=120.0),
    ]
    broker = _FakeBroker(
        positions=[_position("AAPL")], quotes={"AAPL": _quote("AAPL", 100.0)}
    )
    journal = _FakeJournal(exits)
    report = _run(broker, journal)

    assert report.converted_oco == 1
    assert broker.oco_calls == [
        {
            "symbol": "AAPL",
            "qty": 10,
            "stop_price": 95.0,
            "limit_price": 120.0,
            "client_order_id": "pdt-convert-1",
        }
    ]
    assert journal.exits[1].state == "submitted"
    assert journal.exits[2].state == "submitted"
    # Two honest order rows: GTC stop + GTC take_profit, D-077 noted.
    roles = sorted(o.role for o in journal.orders)
    assert roles == ["stop", "take_profit"]
    for o in journal.orders:
        assert o.tif == "gtc"
        assert o.notes == "D-077 conversion"
        # D-078 §7.4 (W3-H1): NULL at submission so the FillIngestor's
        # get_orders_pending_fill (final_status IS NULL) can see it.
        assert o.final_status is None
    # Distinct broker ids per leg, propagated to the right vexit.
    stop_row = next(o for o in journal.orders if o.role == "stop")
    tp_row = next(o for o in journal.orders if o.role == "take_profit")
    assert journal.exits[1].submitted_broker_order_id == stop_row.broker_order_id
    assert journal.exits[2].submitted_broker_order_id == tp_row.broker_order_id


def test_stop_only_converts_to_plain_gtc_stop() -> None:
    """No TP leg -> submit_oco_sell with limit_price=None (plain GTC stop
    per §7.3); one journaled stop row."""
    exits = [_vexit(1, execution_id=42, symbol="AAPL", role="stop", stop_price=95.0)]
    broker = _FakeBroker(
        positions=[_position("AAPL")], quotes={"AAPL": _quote("AAPL", 100.0)}
    )
    journal = _FakeJournal(exits)
    report = _run(broker, journal)

    assert report.converted_stop == 1
    assert broker.oco_calls[0]["limit_price"] is None
    assert [o.role for o in journal.orders] == ["stop"]
    assert journal.orders[0].tif == "gtc"
    assert journal.orders[0].final_status is None  # D-078 §7.4 (W3-H1)
    assert journal.exits[1].state == "submitted"


def test_tp_only_group_converts_to_gtc_limit_sell() -> None:
    """Stop already fired in a prior session (state='submitted'), TP still
    active -> plain GTC limit sell via submit_order, not OCO."""
    exits = [
        _vexit(1, execution_id=42, symbol="AAPL", role="stop",
               stop_price=95.0, state="submitted"),
        _vexit(2, execution_id=42, symbol="AAPL", role="tp", tp_price=120.0),
    ]
    broker = _FakeBroker(
        positions=[_position("AAPL")], quotes={"AAPL": _quote("AAPL", 100.0)}
    )
    journal = _FakeJournal(exits)
    report = _run(broker, journal)

    assert report.converted_tp_limit == 1
    assert broker.oco_calls == []
    assert len(broker.submitted) == 1
    req = broker.submitted[0]
    assert req.order_type == "limit"
    assert req.tif == "gtc"
    assert req.limit_price == 120.0
    assert req.client_order_id == "pdt-convert-2"
    assert journal.exits[2].state == "submitted"
    assert [o.role for o in journal.orders] == ["take_profit"]
    assert journal.orders[0].final_status is None  # D-078 §7.4 (W3-H1)


def test_breached_stop_converts_to_market_sell() -> None:
    """quote.last <= stop_price -> immediate market sell (DAY), not a GTC
    stop that would trigger instantly with worse price control."""
    exits = [_vexit(1, execution_id=42, symbol="AAPL", role="stop", stop_price=95.0)]
    broker = _FakeBroker(
        positions=[_position("AAPL")], quotes={"AAPL": _quote("AAPL", 94.0)}
    )
    journal = _FakeJournal(exits)
    report = _run(broker, journal)

    assert report.converted_market == 1
    assert broker.oco_calls == []
    req = broker.submitted[0]
    assert req.order_type == "market"
    assert req.side == "sell"
    assert req.client_order_id == "pdt-convert-1"
    assert [o.role for o in journal.orders] == ["stop"]
    assert journal.orders[0].notes == "D-077 conversion"
    assert journal.orders[0].final_status is None  # D-078 §7.4 (W3-H1)
    assert journal.exits[1].state == "submitted"


def test_breached_stop_supersedes_sibling_tp() -> None:
    """Market-selling the whole position makes the TP leg moot: marked
    obsolete (no broker order corresponds to it), not submitted."""
    exits = [
        _vexit(1, execution_id=42, symbol="AAPL", role="stop", stop_price=95.0),
        _vexit(2, execution_id=42, symbol="AAPL", role="tp", tp_price=120.0),
    ]
    broker = _FakeBroker(
        positions=[_position("AAPL")], quotes={"AAPL": _quote("AAPL", 94.0)}
    )
    journal = _FakeJournal(exits)
    report = _run(broker, journal)

    assert report.converted_market == 1
    assert journal.exits[1].state == "submitted"
    assert journal.exits[2].state == "obsolete"
    assert "D-077" in (journal.exits[2].notes or "")
    # Only ONE sell at the broker — no double-sell of the same shares.
    assert len(broker.submitted) == 1
    assert broker.oco_calls == []


def test_journal_write_precedes_state_flip() -> None:
    """§4.2 step 4 ordering: orders row insert happens before the virtual
    exit is marked submitted (crash between them is healed by the client
    id pre-check, never by an unjournaled broker order)."""
    exits = [_vexit(1, execution_id=42, symbol="AAPL", role="stop", stop_price=95.0)]
    broker = _FakeBroker(
        positions=[_position("AAPL")], quotes={"AAPL": _quote("AAPL", 100.0)}
    )
    journal = _FakeJournal(exits)
    _run(broker, journal)

    insert_idx = next(
        i for i, c in enumerate(journal.call_log) if c.startswith("insert_order:")
    )
    mark_idx = next(
        i for i, c in enumerate(journal.call_log) if c.startswith("mark_submitted:")
    )
    assert insert_idx < mark_idx


# ---------------------------------------------------------------------------
# Invariant: never strand a position
# ---------------------------------------------------------------------------


def test_closed_position_skipped_exit_left_active() -> None:
    """Symbol not broker-open -> NO conversion and NO obsolete-marking
    (the scanner's consecutive-miss grace owns that call; converter
    must not reintroduce the 2026-05-11 fill-burst race)."""
    exits = [_vexit(1, execution_id=42, symbol="GONE", role="stop", stop_price=95.0)]
    broker = _FakeBroker(positions=[])
    journal = _FakeJournal(exits)
    report = _run(broker, journal)

    assert report.skipped_no_position == 1
    assert journal.exits[1].state == "active"
    assert broker.submitted == []
    assert broker.oco_calls == []


def test_broker_failure_leaves_exit_active_and_continues() -> None:
    """Submit raises BrokerUnavailable -> vexit stays active (scanner
    still covers it: the ADR-012 invariant), error counted, the run
    continues to the deferred-drain phase."""
    exits = [_vexit(1, execution_id=42, symbol="AAPL", role="stop", stop_price=95.0)]
    deferred = [_deferred(7, virtual_exit_id=99, symbol="GONE")]
    broker = _FakeBroker(
        positions=[_position("AAPL")], quotes={"AAPL": _quote("AAPL", 100.0)}
    )
    broker.fail_submit = BrokerUnavailable("503")
    journal = _FakeJournal(exits, deferred)
    report = _run(broker, journal)

    assert report.errors == 1
    assert journal.exits[1].state == "active"
    assert journal.orders == []
    # Drain still ran: the closed-position deferred row was invalidated.
    assert report.invalidated_deferred == 1


def test_groups_isolated_one_failure_does_not_abort_run() -> None:
    """A quote failure for one symbol must not prevent converting the
    other group."""
    exits = [
        _vexit(1, execution_id=42, symbol="BAD", role="stop", stop_price=95.0),
        _vexit(2, execution_id=43, symbol="GOOD", role="stop", stop_price=50.0),
    ]
    broker = _FakeBroker(
        positions=[_position("BAD"), _position("GOOD")],
        quotes={"GOOD": _quote("GOOD", 60.0)},  # BAD missing -> KeyError
    )
    journal = _FakeJournal(exits)
    report = _run(broker, journal)

    assert report.errors == 1
    assert report.converted_stop == 1
    assert journal.exits[1].state == "active"
    assert journal.exits[2].state == "submitted"


# ---------------------------------------------------------------------------
# Idempotency (ADR-012 §4.2 step 1)
# ---------------------------------------------------------------------------


def test_precheck_hit_heals_without_resubmitting() -> None:
    """A prior boot crashed after broker-accept but before the journal
    write: get_order_by_client_id finds the order -> journal it, mark the
    vexits, submit NOTHING new."""
    exits = [
        _vexit(1, execution_id=42, symbol="AAPL", role="stop", stop_price=95.0),
        _vexit(2, execution_id=42, symbol="AAPL", role="tp", tp_price=120.0),
    ]
    prior = _submitted(
        "alp-prior-1", "pdt-convert-1", "AAPL", 10,
        order_type="stop", stop_price=95.0,
    )
    broker = _FakeBroker(
        positions=[_position("AAPL")],
        quotes={"AAPL": _quote("AAPL", 100.0)},
        existing={"pdt-convert-1": prior},
    )
    journal = _FakeJournal(exits)
    report = _run(broker, journal)

    assert report.healed == 1
    assert broker.submitted == []
    assert broker.oco_calls == []
    # The found order is journaled so the reconciler can explain its fill.
    assert [o.broker_order_id for o in journal.orders] == ["alp-prior-1"]
    assert journal.orders[0].final_status is None  # D-078 §7.4 (W3-H1)
    assert journal.exits[1].state == "submitted"
    assert journal.exits[1].submitted_broker_order_id == "alp-prior-1"
    # The OCO partner leg can't be recovered by client id; the row is
    # closed out against the same broker order so it leaves the active set.
    assert journal.exits[2].state == "submitted"


def test_heal_proceeds_when_order_already_journaled() -> None:
    """W3-M1 crash window: a prior boot crashed BETWEEN insert_order and
    mark_virtual_exit_submitted. Next boot the client-id pre-check hits
    AND the orders row already exists — the UNIQUE violation must be
    consumed as the already-journaled signal so the vexits still get
    marked, instead of a per-group error every boot pinning them
    'active'."""
    exits = [
        _vexit(1, execution_id=42, symbol="AAPL", role="stop", stop_price=95.0),
        _vexit(2, execution_id=42, symbol="AAPL", role="tp", tp_price=120.0),
    ]
    prior = _submitted(
        "alp-prior-1", "pdt-convert-1", "AAPL", 10,
        order_type="stop", stop_price=95.0,
    )
    broker = _FakeBroker(
        positions=[_position("AAPL")],
        quotes={"AAPL": _quote("AAPL", 100.0)},
        existing={"pdt-convert-1": prior},
    )
    journal = _FakeJournal(exits)
    # The prior boot's journal write survived the crash.
    journal.orders.append(OrderRow(
        execution_id=42, role="stop", symbol="AAPL", side="sell",
        order_type="stop", qty=10, stop_price=95.0, tif="gtc",
        broker_order_id="alp-prior-1", submitted_at=_NOW,
        notes="D-077 conversion",
    ))
    report = _run(broker, journal)

    assert report.healed == 1
    assert report.errors == 0
    assert broker.submitted == []
    assert broker.oco_calls == []
    # No duplicate row; both vexits leave the active set.
    assert [o.broker_order_id for o in journal.orders] == ["alp-prior-1"]
    assert journal.exits[1].state == "submitted"
    assert journal.exits[1].submitted_broker_order_id == "alp-prior-1"
    assert journal.exits[2].state == "submitted"


@pytest.mark.parametrize(
    "dead_status",
    [
        "rejected",
        "canceled",
        "expired",
        "suspended",
        "replaced",
        # Not in the review's dead set, but unrecognized -> fail-safe
        # into the dead path rather than risk a false heal.
        "held",
    ],
)
def test_heal_dead_order_leaves_group_active_no_resubmit(
    dead_status: str,
) -> None:
    """W3-H2: the client-id pre-check finds the prior boot's order DEAD
    (async rejection, broker sweep, or operator cancel during triage).
    Healing it would flip the group 'submitted' and stop the scanner
    with zero live protection left (the 2026-05-07 class). Required:
    no journal write, no mark, rows stay 'active', counted as error —
    and NO automatic resubmission (recovery is the documented operator
    runbook step; a 'replaced' order in particular means live
    protection already exists under a different id)."""
    exits = [
        _vexit(1, execution_id=42, symbol="AAPL", role="stop", stop_price=95.0),
        _vexit(2, execution_id=42, symbol="AAPL", role="tp", tp_price=120.0),
    ]
    prior = _submitted(
        "alp-prior-1", "pdt-convert-1", "AAPL", 10,
        order_type="stop", stop_price=95.0, status=dead_status,
    )
    broker = _FakeBroker(
        positions=[_position("AAPL")],
        quotes={"AAPL": _quote("AAPL", 100.0)},
        existing={"pdt-convert-1": prior},
    )
    journal = _FakeJournal(exits)
    report = _run(broker, journal)

    assert report.errors == 1
    assert report.healed == 0
    assert journal.orders == []
    assert journal.exits[1].state == "active"
    assert journal.exits[2].state == "active"
    assert broker.submitted == []
    assert broker.oco_calls == []


@pytest.mark.parametrize(
    "live_status",
    ["new", "pending_new", "partially_filled", "filled", "pending_cancel"],
)
def test_heal_proceeds_for_live_and_executed_statuses(
    live_status: str,
) -> None:
    """W3-H2 live branch: every status in the healable set heals exactly
    like 'accepted' (test_precheck_hit_heals_without_resubmitting) —
    'filled'/'partially_filled' included, since the journaled row is
    what lets the §2.2 classifier explain the executed sell."""
    exits = [
        _vexit(1, execution_id=42, symbol="AAPL", role="stop", stop_price=95.0),
    ]
    prior = _submitted(
        "alp-prior-1", "pdt-convert-1", "AAPL", 10,
        order_type="stop", stop_price=95.0, status=live_status,
    )
    broker = _FakeBroker(
        positions=[_position("AAPL")],
        quotes={"AAPL": _quote("AAPL", 100.0)},
        existing={"pdt-convert-1": prior},
    )
    journal = _FakeJournal(exits)
    report = _run(broker, journal)

    assert report.healed == 1
    assert report.errors == 0
    assert broker.submitted == []
    assert broker.oco_calls == []
    assert [o.broker_order_id for o in journal.orders] == ["alp-prior-1"]
    assert journal.orders[0].final_status is None  # D-078 §7.4 (W3-H1)
    assert journal.exits[1].state == "submitted"


def test_heal_prefers_live_sibling_over_dead_order() -> None:
    """Defensive: if one row's client id resolves to a dead order but a
    sibling's resolves to a live one, the live order wins — the group
    heals instead of erroring."""
    exits = [
        _vexit(1, execution_id=42, symbol="AAPL", role="stop", stop_price=95.0),
        _vexit(2, execution_id=42, symbol="AAPL", role="tp", tp_price=120.0),
    ]
    dead = _submitted(
        "alp-dead-1", "pdt-convert-1", "AAPL", 10,
        order_type="stop", stop_price=95.0, status="canceled",
    )
    live = _submitted(
        "alp-live-2", "pdt-convert-2", "AAPL", 10,
        order_type="limit", limit_price=120.0, status="accepted",
    )
    broker = _FakeBroker(
        positions=[_position("AAPL")],
        quotes={"AAPL": _quote("AAPL", 100.0)},
        existing={"pdt-convert-1": dead, "pdt-convert-2": live},
    )
    journal = _FakeJournal(exits)
    report = _run(broker, journal)

    assert report.healed == 1
    assert report.errors == 0
    assert [o.broker_order_id for o in journal.orders] == ["alp-live-2"]
    assert journal.exits[1].state == "submitted"
    assert journal.exits[2].state == "submitted"


def test_failed_boot_retries_cleanly_on_next_boot() -> None:
    """Addendum item 2 rider: a synchronous submit failure on boot 1
    leaves the rows 'active' with nothing at the broker (no journal
    write precedes the submit), so boot 2 converts cleanly under the
    same deterministic client id."""
    exits = [
        _vexit(1, execution_id=42, symbol="AAPL", role="stop", stop_price=95.0),
        _vexit(2, execution_id=42, symbol="AAPL", role="tp", tp_price=120.0),
    ]
    broker = _FakeBroker(
        positions=[_position("AAPL")], quotes={"AAPL": _quote("AAPL", 100.0)}
    )
    broker.fail_submit = BrokerUnavailable("503")
    journal = _FakeJournal(exits)
    first = _run(broker, journal)

    assert first.errors == 1
    assert first.converted_oco == 0
    assert journal.exits[1].state == "active"
    assert journal.orders == []

    broker.fail_submit = None
    second = _run(broker, journal)

    assert second.errors == 0
    assert second.converted_oco == 1
    assert broker.oco_calls[0]["client_order_id"] == "pdt-convert-1"
    assert journal.exits[1].state == "submitted"
    assert journal.exits[2].state == "submitted"
    for o in journal.orders:
        assert o.final_status is None  # D-078 §7.4 (W3-H1)


def test_second_run_is_noop() -> None:
    """Run twice: the first converts, the second sees zero active rows and
    performs zero broker I/O — the boot-loop idempotency contract."""
    exits = [
        _vexit(1, execution_id=42, symbol="AAPL", role="stop", stop_price=95.0),
        _vexit(2, execution_id=42, symbol="AAPL", role="tp", tp_price=120.0),
    ]
    broker = _FakeBroker(
        positions=[_position("AAPL")], quotes={"AAPL": _quote("AAPL", 100.0)}
    )
    journal = _FakeJournal(exits)
    first = _run(broker, journal)
    assert first.converted_oco == 1

    broker.call_log.clear()
    second = _run(broker, journal)
    assert second == ConversionReport(0, 0, 0, 0, 0, 0, 0, 0)
    assert broker.call_log == []


# ---------------------------------------------------------------------------
# Deferred-sell drain (ADR-012 §4.2 step 5)
# ---------------------------------------------------------------------------


def test_drain_invalidates_deferred_rows_for_converted_exits() -> None:
    exits = [_vexit(1, execution_id=42, symbol="AAPL", role="stop", stop_price=95.0)]
    deferred = [_deferred(7, virtual_exit_id=1, symbol="AAPL")]
    broker = _FakeBroker(
        positions=[_position("AAPL")], quotes={"AAPL": _quote("AAPL", 100.0)}
    )
    journal = _FakeJournal(exits, deferred)
    report = _run(broker, journal)

    assert report.invalidated_deferred == 1
    assert journal.skipped_deferred == [
        (7, "invalidated: superseded by D-077 conversion")
    ]


def test_drain_invalidates_deferred_rows_for_closed_positions() -> None:
    """Unreplayed rows whose position is gone are invalidated with the
    distinct §4.2 step-5 closed-position note (W3-L2), even with zero
    active virtual exits (stale-prod-DB self-heal)."""
    deferred = [_deferred(7, virtual_exit_id=99, symbol="GONE")]
    broker = _FakeBroker(positions=[_position("AAPL")])
    journal = _FakeJournal([], deferred)
    report = _run(broker, journal)

    assert report.invalidated_deferred == 1
    assert journal.skipped_deferred == [(7, "invalidated: position closed")]


def test_drain_leaves_unrelated_unreplayed_rows_for_replayer() -> None:
    """A row for a still-open position whose source vexit was NOT
    converted this boot stays in the queue — the replayer's own
    supersede guard owns it."""
    deferred = [_deferred(7, virtual_exit_id=50, symbol="AAPL")]
    broker = _FakeBroker(positions=[_position("AAPL")])
    journal = _FakeJournal([], deferred)
    report = _run(broker, journal)

    assert report.invalidated_deferred == 0
    assert journal.skipped_deferred == []


def test_drain_covers_breached_stop_sibling_tp_deferred_rows() -> None:
    """Deferred rows pointing at BOTH legs of a breached-stop group are
    invalidated (the market sell superseded the whole group)."""
    exits = [
        _vexit(1, execution_id=42, symbol="AAPL", role="stop", stop_price=95.0),
        _vexit(2, execution_id=42, symbol="AAPL", role="tp", tp_price=120.0),
    ]
    deferred = [
        _deferred(7, virtual_exit_id=1, symbol="AAPL"),
        _deferred(8, virtual_exit_id=2, symbol="AAPL", role="tp"),
    ]
    broker = _FakeBroker(
        positions=[_position("AAPL")], quotes={"AAPL": _quote("AAPL", 94.0)}
    )
    journal = _FakeJournal(exits, deferred)
    report = _run(broker, journal)

    assert report.converted_market == 1
    assert report.invalidated_deferred == 2
    assert {d for d, _ in journal.skipped_deferred} == {7, 8}

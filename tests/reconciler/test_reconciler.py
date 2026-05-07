"""S6.5 — Position reconciler tests.

The reconciler compares the broker's truth to the journal's latest
position snapshot per symbol. On divergence: soft halt (KS reason
`reconciler:discrepancy`) + operator alert. On match: snapshot the
positions + account so the journal stays current. Transient broker
unavailability is suppressed (≤ 2 consecutive failures defer; 3+ logs
WARN; never halts on broker outage alone).
"""

from __future__ import annotations

import asyncio
import datetime as dt

import pytest

from broker.alpaca import BrokerUnavailable
from broker.protocol import AccountSnapshot, OrderRequest, Position, Quote, SubmittedOrder
from config.loader import ReconcilerConfig
from journal.models import OrderRow
from reconciler.reconciler import (
    ExplainedDiff,
    PositionDiff,
    Reconciler,
    ReconcileReport,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeBroker:
    def __init__(
        self,
        *,
        positions: list[Position] | None = None,
        account: AccountSnapshot | None = None,
        raise_seq: list[Exception] | None = None,
    ) -> None:
        self._positions = positions or []
        self._account = account or AccountSnapshot(
            equity=10_000.0,
            cash=10_000.0,
            buying_power=10_000.0,
            long_market_value=0.0,
            daypl=0.0,
            snapshot_at=dt.datetime(2026, 4, 28, 14, 30, tzinfo=dt.UTC),
        )
        # Pop one exception per call from front; if list empty, return ok.
        self._raise_seq: list[Exception] = list(raise_seq or [])
        self.calls: int = 0

    def submit_order(self, req: OrderRequest) -> SubmittedOrder:  # pragma: no cover
        raise NotImplementedError

    def cancel_order(self, broker_order_id: str) -> None:  # pragma: no cover
        pass

    def get_account(self) -> AccountSnapshot:
        if self._raise_seq:
            raise self._raise_seq.pop(0)
        return self._account

    def get_positions(self) -> list[Position]:
        self.calls += 1
        if self._raise_seq:
            raise self._raise_seq.pop(0)
        return self._positions

    def get_quote(self, symbol: str) -> Quote:  # pragma: no cover
        raise NotImplementedError


class _FakeJournal:
    def __init__(
        self,
        *,
        expected_positions: list[Position] | None = None,
        recent_orders: list[OrderRow] | None = None,
    ) -> None:
        self._expected = expected_positions or []
        self._orders: list[OrderRow] = list(recent_orders or [])
        self.position_inserts: list[dict] = []
        self.account_inserts: list[dict] = []
        self.get_orders_since_calls: list[tuple[str, dt.datetime]] = []

    def get_open_positions(self) -> list:
        # Return as the same Position-like shape the reconciler expects from
        # the journal (we mirror only the comparison fields used).
        out = []
        for p in self._expected:
            class _Row:
                pass
            r = _Row()
            r.symbol = p.symbol
            r.qty = p.qty
            r.avg_entry_price = p.avg_entry_price
            out.append(r)
        return out

    def insert_position(self, row) -> int:
        self.position_inserts.append(row.model_dump())
        return len(self.position_inserts)

    def insert_account_snapshot(self, row) -> int:
        self.account_inserts.append(row.model_dump())
        return len(self.account_inserts)

    def get_orders_since(self, symbol: str, since: dt.datetime) -> list[OrderRow]:
        self.get_orders_since_calls.append((symbol, since))
        return [
            o for o in self._orders
            if o.symbol == symbol and o.submitted_at >= since
        ]


class _FakeKillSwitch:
    def __init__(self) -> None:
        self.halts: list[tuple[str, str, str]] = []

    def halt(self, reason: str, set_by: str, notes: str = "") -> None:
        self.halts.append((reason, set_by, notes))


class _FakeAlerter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def notify(self, message: str) -> None:
        self.calls.append(message)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

_MARKET_NOON_ET = dt.datetime(2026, 4, 28, 12, 0, tzinfo=dt.UTC).astimezone(
    dt.timezone(dt.timedelta(hours=-4))  # ET = UTC-4 in DST
)
# 2026-04-28 12:00 ET (Tuesday, market hours).
_MARKET_OPEN_TIME = dt.datetime(2026, 4, 28, 16, 0, tzinfo=dt.UTC)  # 12:00 ET
# 2026-04-28 22:00 UTC = 18:00 ET (market closed).
_AFTER_HOURS_TIME = dt.datetime(2026, 4, 28, 22, 0, tzinfo=dt.UTC)


def _position(symbol: str, qty: int, avg_entry: float = 10.0) -> Position:
    return Position(
        symbol=symbol,
        qty=qty,
        avg_entry_price=avg_entry,
        market_value=qty * avg_entry,
        unrealized_pnl=0.0,
    )


def _cfg(
    *,
    interval_seconds_market: int = 300,
    expected_fill_window_minutes: int = 30,
) -> ReconcilerConfig:
    return ReconcilerConfig(
        interval_seconds_market=interval_seconds_market,
        expected_fill_window_minutes=expected_fill_window_minutes,
    )


def _now_market() -> dt.datetime:
    return _MARKET_OPEN_TIME


def _now_off() -> dt.datetime:
    return _AFTER_HOURS_TIME


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_match_writes_snapshot_no_halt() -> None:
    """Test #1 (first failing): broker matches journal → snapshot, no halt."""
    pos = [_position("ACME", 25)]
    broker = _FakeBroker(positions=pos)
    journal = _FakeJournal(expected_positions=pos)
    ks = _FakeKillSwitch()
    alerter = _FakeAlerter()
    rec = Reconciler(broker, journal, ks, _cfg(), alerter=alerter, now_fn=_now_market)

    report = rec.reconcile_now()

    assert isinstance(report, ReconcileReport)
    assert report.matched is True
    assert report.diffs == []
    assert ks.halts == []
    # Snapshots written: one per broker position + one account snapshot.
    assert len(journal.position_inserts) == 1
    assert journal.position_inserts[0]["symbol"] == "ACME"
    assert journal.position_inserts[0]["source"] == "reconciler"
    assert len(journal.account_inserts) == 1
    assert alerter.calls == []


def test_divergence_halts_kill_switch() -> None:
    """Test #2: broker has 100 shares, journal expects 50 → halt."""
    broker = _FakeBroker(positions=[_position("ACME", 100)])
    journal = _FakeJournal(expected_positions=[_position("ACME", 50)])
    ks = _FakeKillSwitch()
    alerter = _FakeAlerter()
    rec = Reconciler(broker, journal, ks, _cfg(), alerter=alerter, now_fn=_now_market)

    report = rec.reconcile_now()

    assert report.matched is False
    assert len(report.diffs) == 1
    assert report.diffs[0].symbol == "ACME"
    assert len(ks.halts) == 1
    reason, set_by, _notes = ks.halts[0]
    assert reason == "reconciler:discrepancy"
    assert set_by == "system"
    # Operator alerted.
    assert len(alerter.calls) == 1
    # Snapshot still written (truthful record of broker state).
    assert len(journal.position_inserts) == 1


def test_divergence_when_journal_has_position_broker_does_not() -> None:
    """Asymmetric divergence: journal believes ACME held; broker says no."""
    broker = _FakeBroker(positions=[])
    journal = _FakeJournal(expected_positions=[_position("ACME", 25)])
    ks = _FakeKillSwitch()
    rec = Reconciler(broker, journal, ks, _cfg(), now_fn=_now_market)

    report = rec.reconcile_now()

    assert report.matched is False
    # Halt fired.
    assert len(ks.halts) == 1


def test_post_submission_trigger() -> None:
    """Test #3: trigger_after_submission() runs reconcile_now()."""
    pos = [_position("ACME", 25)]
    broker = _FakeBroker(positions=pos)
    journal = _FakeJournal(expected_positions=pos)
    ks = _FakeKillSwitch()
    rec = Reconciler(broker, journal, ks, _cfg(), now_fn=_now_market)

    rec.trigger_after_submission()

    assert len(journal.position_inserts) == 1


def test_transient_broker_failure_no_halt() -> None:
    """Test #4: 1 broker failure → defer (no halt, no diff)."""
    broker = _FakeBroker(raise_seq=[BrokerUnavailable("transient")])
    journal = _FakeJournal()
    ks = _FakeKillSwitch()
    rec = Reconciler(broker, journal, ks, _cfg(), now_fn=_now_market)

    report = rec.reconcile_now()

    assert report.deferred is True
    assert ks.halts == []


def test_two_consecutive_broker_failures_no_halt_no_warn() -> None:
    """≤ 2 transient failures → defer; no WARN, no halt."""
    broker = _FakeBroker(raise_seq=[BrokerUnavailable("t1"), BrokerUnavailable("t2")])
    journal = _FakeJournal()
    ks = _FakeKillSwitch()
    rec = Reconciler(broker, journal, ks, _cfg(), now_fn=_now_market)

    rec.reconcile_now()
    report2 = rec.reconcile_now()

    assert report2.deferred is True
    assert ks.halts == []


def test_multiple_failures_logged_no_halt() -> None:
    """Test #5: 3+ consecutive failures → log WARN; no halt.

    We attach a captured handler directly to the named logger so the test
    is robust to test-suite logging reconfiguration (caplog relies on
    propagation which other tests may modify).
    """
    import logging

    captured: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = _Capture(level=logging.WARNING)
    target = logging.getLogger("reconciler.reconciler")
    prior_level = target.level
    target.addHandler(handler)
    target.setLevel(logging.WARNING)
    try:
        broker = _FakeBroker(raise_seq=[
            BrokerUnavailable("t1"),
            BrokerUnavailable("t2"),
            BrokerUnavailable("t3"),
        ])
        journal = _FakeJournal()
        ks = _FakeKillSwitch()
        rec = Reconciler(broker, journal, ks, _cfg(), now_fn=_now_market)
        for _ in range(3):
            rec.reconcile_now()
    finally:
        target.removeHandler(handler)
        target.setLevel(prior_level)

    assert ks.halts == []
    assert any(
        getattr(r, "event", None) == "reconciler.broker_unavailable_persistent"
        for r in captured
    )


def test_failure_counter_resets_on_success() -> None:
    """After a successful reconcile, the failure counter resets."""
    broker = _FakeBroker(
        positions=[],
        raise_seq=[BrokerUnavailable("t1"), BrokerUnavailable("t2")],
    )
    journal = _FakeJournal()
    ks = _FakeKillSwitch()
    rec = Reconciler(broker, journal, ks, _cfg(), now_fn=_now_market)

    rec.reconcile_now()  # fail 1
    rec.reconcile_now()  # fail 2
    rec.reconcile_now()  # success — broker.raise_seq exhausted
    # Counter reset; another fail should not log persistent yet.
    broker._raise_seq = [BrokerUnavailable("t-after-success")]
    report = rec.reconcile_now()
    assert report.deferred is True
    assert ks.halts == []


def test_off_hours_suppression() -> None:
    """Test #6: outside market hours, reconcile_now() is suppressed."""
    pos = [_position("ACME", 25)]
    broker = _FakeBroker(positions=pos)
    journal = _FakeJournal(expected_positions=pos)
    ks = _FakeKillSwitch()
    rec = Reconciler(broker, journal, ks, _cfg(), now_fn=_now_off)

    report = rec.reconcile_now()

    assert report.suppressed is True
    assert broker.calls == 0  # broker not even queried
    assert journal.position_inserts == []


def test_post_submission_trigger_runs_off_hours_too() -> None:
    """trigger_after_submission overrides the off-hours suppression: a
    submission has just occurred, so we must reconcile regardless of
    market clock (otherwise the submission would never be confirmed)."""
    pos = [_position("ACME", 25)]
    broker = _FakeBroker(positions=pos)
    journal = _FakeJournal(expected_positions=pos)
    ks = _FakeKillSwitch()
    rec = Reconciler(broker, journal, ks, _cfg(), now_fn=_now_off)

    rec.trigger_after_submission()

    assert len(journal.position_inserts) == 1


def test_limit_at_non_crossing_no_fill_no_discrepancy() -> None:
    """Test #7 / ADR-001: limit order at non-crossing price → broker has no
    new position; journal also reflects no position for that symbol → match.

    This exercises the "submitted-but-not-filled" path: the orders table
    has the entry order but the positions snapshot has nothing yet, and
    the broker also has nothing yet — they agree.
    """
    broker = _FakeBroker(positions=[])
    journal = _FakeJournal(expected_positions=[])
    ks = _FakeKillSwitch()
    rec = Reconciler(broker, journal, ks, _cfg(), now_fn=_now_market)

    report = rec.reconcile_now()

    assert report.matched is True
    assert ks.halts == []


# ---------------------------------------------------------------------------
# Hotfix 2026-05-07 — pending-fill diff classification
# ---------------------------------------------------------------------------


def _order(
    *,
    symbol: str,
    role: str,
    side: str,
    qty: int,
    submitted_at: dt.datetime,
    order_id: int = 1,
    execution_id: int = 1,
) -> OrderRow:
    return OrderRow(
        id=order_id,
        execution_id=execution_id,
        role=role,
        symbol=symbol,
        side=side,
        order_type="market",
        qty=qty,
        tif="day",
        broker_order_id=f"ord-{symbol}-{role}-{order_id}",
        submitted_at=submitted_at,
    )


def test_diff_classified_expected_when_recent_entry_explains_broker_increase() -> None:
    """CXW-style: broker has 46, journal snapshot has 0, journal `orders`
    has an entry buy of 46 submitted 30 seconds ago → classified as
    expected → no halt, broker snapshot inserted, ReconcileReport.matched
    is True with the diff in `explained_diffs`.
    """
    broker = _FakeBroker(positions=[_position("CXW", 46, avg_entry=20.0)])
    journal = _FakeJournal(
        expected_positions=[],
        recent_orders=[
            _order(
                symbol="CXW",
                role="entry",
                side="buy",
                qty=46,
                submitted_at=_MARKET_OPEN_TIME - dt.timedelta(seconds=30),
                order_id=37,
            ),
        ],
    )
    ks = _FakeKillSwitch()
    alerter = _FakeAlerter()
    rec = Reconciler(broker, journal, ks, _cfg(), alerter=alerter, now_fn=_now_market)

    report = rec.reconcile_now()

    assert report.matched is True
    assert report.diffs == []
    assert len(report.explained_diffs) == 1
    ed = report.explained_diffs[0]
    assert ed.diff.symbol == "CXW"
    assert ed.diff.broker_qty == 46
    assert ed.diff.journal_qty == 0
    assert 37 in ed.explained_by_order_ids
    # No halt, no operator alert, but broker truth IS snapshotted.
    assert ks.halts == []
    assert alerter.calls == []
    assert len(journal.position_inserts) == 1
    assert journal.position_inserts[0]["symbol"] == "CXW"


def test_diff_classified_expected_when_stop_fill_explains_broker_decrease() -> None:
    """Inverse path: broker has 0, journal snapshot has 46, recent stop
    sell of 46 → expected, no halt."""
    broker = _FakeBroker(positions=[])
    journal = _FakeJournal(
        expected_positions=[_position("CXW", 46, avg_entry=20.0)],
        recent_orders=[
            _order(
                symbol="CXW",
                role="stop",
                side="sell",
                qty=46,
                submitted_at=_MARKET_OPEN_TIME - dt.timedelta(minutes=2),
                order_id=42,
            ),
        ],
    )
    ks = _FakeKillSwitch()
    rec = Reconciler(broker, journal, ks, _cfg(), now_fn=_now_market)

    report = rec.reconcile_now()

    assert report.matched is True
    assert report.diffs == []
    assert ks.halts == []
    assert len(report.explained_diffs) == 1


def test_diff_classified_expected_when_take_profit_explains_broker_decrease() -> None:
    """Take-profit fill (the OTO sell that closes a winning trade) also
    counts as an expected explanation for a broker_qty < journal_qty diff."""
    broker = _FakeBroker(positions=[])
    journal = _FakeJournal(
        expected_positions=[_position("CXW", 46, avg_entry=20.0)],
        recent_orders=[
            _order(
                symbol="CXW",
                role="take_profit",
                side="sell",
                qty=46,
                submitted_at=_MARKET_OPEN_TIME - dt.timedelta(minutes=1),
                order_id=43,
            ),
        ],
    )
    ks = _FakeKillSwitch()
    rec = Reconciler(broker, journal, ks, _cfg(), now_fn=_now_market)

    report = rec.reconcile_now()

    assert report.matched is True
    assert ks.halts == []


def test_diff_classified_unexpected_when_no_orders_explain_increase() -> None:
    """Broker has 50 of XYZ, journal has 0, no recent orders → halt as
    today (genuine concern: manual broker action / corruption)."""
    broker = _FakeBroker(positions=[_position("XYZ", 50)])
    journal = _FakeJournal(expected_positions=[], recent_orders=[])
    ks = _FakeKillSwitch()
    alerter = _FakeAlerter()
    rec = Reconciler(broker, journal, ks, _cfg(), alerter=alerter, now_fn=_now_market)

    report = rec.reconcile_now()

    assert report.matched is False
    assert len(report.diffs) == 1
    assert report.diffs[0].symbol == "XYZ"
    assert len(ks.halts) == 1
    assert ks.halts[0][0] == "reconciler:discrepancy"
    assert len(alerter.calls) == 1


def test_diff_classified_unexpected_when_avg_entry_differs() -> None:
    """Broker has 46 of CXW at $20, journal has 46 at $21 → halt regardless
    of recent orders. avg_entry mismatch on equal qty is always unexpected.
    """
    broker = _FakeBroker(positions=[_position("CXW", 46, avg_entry=20.0)])
    journal = _FakeJournal(
        expected_positions=[_position("CXW", 46, avg_entry=21.0)],
        recent_orders=[
            # An entry buy of 46 — would explain a qty diff, but qtys are
            # equal here so the avg_entry mismatch dominates.
            _order(
                symbol="CXW",
                role="entry",
                side="buy",
                qty=46,
                submitted_at=_MARKET_OPEN_TIME - dt.timedelta(seconds=30),
                order_id=37,
            ),
        ],
    )
    ks = _FakeKillSwitch()
    rec = Reconciler(broker, journal, ks, _cfg(), now_fn=_now_market)

    report = rec.reconcile_now()

    assert report.matched is False
    assert len(report.diffs) == 1
    assert report.diffs[0].symbol == "CXW"
    assert len(ks.halts) == 1


def test_window_guard_excludes_stale_orders() -> None:
    """Broker has 50 of XYZ, journal has 0, orders has entry buy 50
    submitted 90 minutes ago, window is 30 → unexpected, halt. The
    window cutoff is enforced in the journal query, not the classifier
    — the classifier never sees the stale row.
    """
    broker = _FakeBroker(positions=[_position("XYZ", 50)])
    journal = _FakeJournal(
        expected_positions=[],
        recent_orders=[
            _order(
                symbol="XYZ",
                role="entry",
                side="buy",
                qty=50,
                submitted_at=_MARKET_OPEN_TIME - dt.timedelta(minutes=90),
                order_id=99,
            ),
        ],
    )
    ks = _FakeKillSwitch()
    rec = Reconciler(
        broker, journal, ks, _cfg(expected_fill_window_minutes=30),
        now_fn=_now_market,
    )

    report = rec.reconcile_now()

    assert report.matched is False
    assert len(ks.halts) == 1
    # Confirm the journal was queried with a window cutoff that excludes
    # the 90-minute-old order.
    assert len(journal.get_orders_since_calls) == 1
    queried_symbol, queried_since = journal.get_orders_since_calls[0]
    assert queried_symbol == "XYZ"
    assert queried_since == _MARKET_OPEN_TIME - dt.timedelta(minutes=30)


def test_mixed_diffs_halt_if_any_unexpected() -> None:
    """Two diffs in one tick — one explained by recent orders (CXW), one
    not (XYZ). Halt fires; payload distinguishes the two classifications.
    """
    broker = _FakeBroker(positions=[
        _position("CXW", 46, avg_entry=20.0),
        _position("XYZ", 50, avg_entry=10.0),
    ])
    journal = _FakeJournal(
        expected_positions=[],
        recent_orders=[
            _order(
                symbol="CXW",
                role="entry",
                side="buy",
                qty=46,
                submitted_at=_MARKET_OPEN_TIME - dt.timedelta(seconds=30),
                order_id=37,
            ),
            # No orders for XYZ.
        ],
    )
    ks = _FakeKillSwitch()
    rec = Reconciler(broker, journal, ks, _cfg(), now_fn=_now_market)

    report = rec.reconcile_now()

    assert report.matched is False
    # Only the unexpected diff appears in `diffs` (the halt-triggering set).
    assert len(report.diffs) == 1
    assert report.diffs[0].symbol == "XYZ"
    # The CXW diff is preserved as explained.
    assert len(report.explained_diffs) == 1
    assert report.explained_diffs[0].diff.symbol == "CXW"
    # Halt fired.
    assert len(ks.halts) == 1
    _reason, _set_by, notes = ks.halts[0]
    # Payload includes both rows with classifications.
    import json as _json
    parsed = _json.loads(notes)
    by_symbol = {row["symbol"]: row for row in parsed}
    assert by_symbol["CXW"]["classification"] == "expected"
    assert by_symbol["XYZ"]["classification"] == "unexpected"
    assert 37 in by_symbol["CXW"]["explained_by_order_ids"]
    assert by_symbol["XYZ"]["explained_by_order_ids"] == []


def test_post_submission_trigger_path_still_silent_on_zero_diff() -> None:
    """Regression: trigger-after-submission flow still works without
    surprises when broker and journal already agree (the common case
    where the entry hasn't filled yet so both are at zero for the new
    symbol)."""
    broker = _FakeBroker(positions=[_position("ACME", 25)])
    journal = _FakeJournal(expected_positions=[_position("ACME", 25)])
    ks = _FakeKillSwitch()
    rec = Reconciler(broker, journal, ks, _cfg(), now_fn=_now_market)

    report = rec.trigger_after_submission()

    assert report.matched is True
    assert report.diffs == []
    assert report.explained_diffs == []
    assert ks.halts == []
    # No journal query for orders when there are no diffs.
    assert journal.get_orders_since_calls == []


def test_classify_diff_partial_explanation_is_unexpected() -> None:
    """Pure unit test for `_classify_diff`: if recent orders only partially
    cover the qty delta, classify as unexpected. Halt; do not silently
    accept a partial explanation as full."""
    diff = PositionDiff(
        symbol="XYZ",
        broker_qty=100,
        journal_qty=0,
        broker_avg_entry=10.0,
        journal_avg_entry=None,
    )
    recent = [
        _order(
            symbol="XYZ",
            role="entry",
            side="buy",
            qty=50,  # only half the delta
            submitted_at=_MARKET_OPEN_TIME,
            order_id=1,
        ),
    ]
    classification, ids = Reconciler._classify_diff(diff, recent)
    assert classification == "unexpected"
    # Even when unexpected, the partially-explaining ids are surfaced for
    # the operator's diff_summary payload.
    assert ids == (1,)


def test_classify_diff_ignores_wrong_side_orders() -> None:
    """If broker_qty > journal_qty (need an entry buy) but the only recent
    order is a sell, classification is unexpected — direction matters."""
    diff = PositionDiff(
        symbol="XYZ",
        broker_qty=50,
        journal_qty=0,
        broker_avg_entry=10.0,
        journal_avg_entry=None,
    )
    recent = [
        _order(
            symbol="XYZ",
            role="stop",
            side="sell",
            qty=50,
            submitted_at=_MARKET_OPEN_TIME,
            order_id=2,
        ),
    ]
    classification, _ids = Reconciler._classify_diff(diff, recent)
    assert classification == "unexpected"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lifecycle_start_calls_reconcile_then_stop_cancels() -> None:
    """start() spawns a task that calls reconcile_now() periodically; stop()
    cancels it. We use a tiny interval and trip on the call counter."""
    pos = [_position("ACME", 25)]
    broker = _FakeBroker(positions=pos)
    journal = _FakeJournal(expected_positions=pos)
    ks = _FakeKillSwitch()
    cfg = ReconcilerConfig(interval_seconds_market=1)  # min positive
    rec = Reconciler(broker, journal, ks, cfg, now_fn=_now_market)

    await rec.start()
    # Wait for at least one tick of the loop.
    await asyncio.sleep(0.05)
    await rec.stop()

    # At least one snapshot written by the periodic task. (May be 0 if the
    # tick interval hasn't elapsed; we test with sleep(0) immediate-tick
    # behavior — see Reconciler.start docstring.)
    assert broker.calls >= 1

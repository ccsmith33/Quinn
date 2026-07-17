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
from typing import Any

import pytest

from broker.alpaca import BrokerUnavailable
from broker.protocol import AccountSnapshot, OrderRequest, Position, Quote, SubmittedOrder
from config.loader import ReconcilerConfig
from journal.models import OrderRow
from reconciler.reconciler import (
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
        self.tombstones: list[dict] = []
        self.lifecycle_calls: list[tuple[str, dt.datetime]] = []

    def get_open_positions(self) -> list:
        # Return as the same Position-like shape the reconciler expects from
        # the journal (we mirror only the comparison fields used). A
        # tombstoned symbol drops out of the open view, mirroring the real
        # repo's latest-row-per-symbol qty!=0 semantics.
        tombstoned = {t["symbol"] for t in self.tombstones}
        out = []
        for p in self._expected:
            if p.symbol in tombstoned:
                continue
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

    # WS1 (D-078): lifecycle classifier query — pending fill (any age) OR
    # filled within the caller's window. Mirrors repo SQL.
    def get_lifecycle_orders_for_symbol(
        self, symbol: str, filled_since: dt.datetime
    ) -> list[OrderRow]:
        self.lifecycle_calls.append((symbol, filled_since))
        out = []
        for o in self._orders:
            if o.symbol != symbol:
                continue
            if o.final_status is None:
                out.append(o)
            elif (
                o.final_status in ("filled", "partially_filled_closed")
                and o.realized_fill_at is not None
                and o.realized_fill_at >= filled_since
            ):
                out.append(o)
        return out

    def insert_position_tombstone(
        self,
        symbol: str,
        source: str,
        notes: str,
        *,
        snapshot_at: dt.datetime | None = None,
    ) -> int:
        self.tombstones.append(
            {
                "symbol": symbol,
                "source": source,
                "notes": notes,
                "snapshot_at": snapshot_at,
            }
        )
        return len(self.tombstones)


class _FakeKillSwitch:
    def __init__(self) -> None:
        self.halts: list[tuple[str, str, str]] = []
        self.fingerprints: list[str | None] = []

    def halt(
        self,
        reason: str,
        set_by: str,
        notes: str = "",
        fingerprint: str | None = None,
    ) -> bool:
        self.halts.append((reason, set_by, notes))
        self.fingerprints.append(fingerprint)
        return True


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


def test_journal_open_broker_absent_absorbed_as_external_close() -> None:
    """WS1 (delta §2.2, kills RC-4): journal believes ACME held; broker says
    no; no journaled sell explains it. NOT a halt — an absent position is
    zero exposure. After 3 consecutive ticks of confirmed absence the
    reconciler tombstones the symbol and alerts ONCE; the journal's open
    view self-heals and tick 4 reports a clean match."""
    broker = _FakeBroker(positions=[])
    journal = _FakeJournal(expected_positions=[_position("ACME", 25)])
    ks = _FakeKillSwitch()
    alerter = _FakeAlerter()
    rec = Reconciler(broker, journal, ks, _cfg(), alerter=alerter, now_fn=_now_market)

    # Ticks 1-2: grace window — pending, no halt, no tombstone, no alert.
    for expected_misses in (1, 2):
        report = rec.reconcile_now()
        assert report.matched is True
        assert report.diffs == []
        assert [d.symbol for d in report.external_close_pending] == ["ACME"]
        assert report.external_closes == []
        assert ks.halts == []
        assert journal.tombstones == []
        del expected_misses

    # Tick 3: absence confirmed — tombstone + single alert, still no halt.
    report = rec.reconcile_now()
    assert report.matched is True
    assert report.external_closes == ["ACME"]
    assert report.external_close_pending == []
    assert ks.halts == []
    assert len(journal.tombstones) == 1
    assert journal.tombstones[0]["symbol"] == "ACME"
    assert journal.tombstones[0]["source"] == "reconciler_external_close"
    assert len(alerter.calls) == 1
    assert "ACME" in alerter.calls[0]

    # Tick 4: journal open view no longer contains ACME — clean match,
    # no second alert (the RC-4 permanent-halt loop is dead).
    report = rec.reconcile_now()
    assert report.matched is True
    assert report.external_closes == []
    assert len(alerter.calls) == 1
    assert ks.halts == []


def test_external_close_counter_resets_when_symbol_reappears() -> None:
    """Absence must be CONSECUTIVE: if the symbol reappears at the broker
    mid-grace (transient API weirdness), the counter resets and a later
    absence starts over."""
    journal = _FakeJournal(expected_positions=[_position("ACME", 25)])
    ks = _FakeKillSwitch()

    # Two absent ticks.
    broker = _FakeBroker(positions=[])
    rec = Reconciler(broker, journal, ks, _cfg(), now_fn=_now_market)
    rec.reconcile_now()
    rec.reconcile_now()
    assert journal.tombstones == []

    # Symbol reappears (match tick) → counter must clear.
    broker._positions = [_position("ACME", 25)]
    report = rec.reconcile_now()
    assert report.matched is True

    # Absent again: needs the FULL 3 ticks before absorption.
    broker._positions = []
    rec.reconcile_now()
    rec.reconcile_now()
    assert journal.tombstones == []
    rec.reconcile_now()
    assert len(journal.tombstones) == 1
    assert ks.halts == []


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


def test_avg_entry_drift_with_matching_qty_is_expected_log_only() -> None:
    """WS1 (delta §2.2, kills RC-5/O-4): broker has 46 of CXW at $20,
    journal has 46 at $21 — qty agrees, price-only drift (add-on buy
    weighted avg, partial-fill drift, splits). Informational: no halt; the
    broker-truth snapshot self-heals the journal on this same tick."""
    broker = _FakeBroker(positions=[_position("CXW", 46, avg_entry=20.0)])
    journal = _FakeJournal(
        expected_positions=[_position("CXW", 46, avg_entry=21.0)],
    )
    ks = _FakeKillSwitch()
    alerter = _FakeAlerter()
    rec = Reconciler(broker, journal, ks, _cfg(), alerter=alerter, now_fn=_now_market)

    report = rec.reconcile_now()

    assert report.matched is True
    assert report.diffs == []
    assert len(report.explained_diffs) == 1
    assert report.explained_diffs[0].diff.symbol == "CXW"
    assert ks.halts == []
    assert alerter.calls == []
    # Broker truth still snapshotted (self-heal).
    assert len(journal.position_inserts) == 1


def test_qty_mismatch_both_nonzero_unexplained_still_halts() -> None:
    """Verification-only is not rubber-stamp: a qty mismatch where both
    sides are non-zero and no lifecycle order explains it remains
    halt-worthy even when avg_entry also drifted (delta §2.2)."""
    broker = _FakeBroker(positions=[_position("CXW", 46, avg_entry=20.0)])
    journal = _FakeJournal(
        expected_positions=[_position("CXW", 30, avg_entry=21.0)],
    )
    ks = _FakeKillSwitch()
    rec = Reconciler(broker, journal, ks, _cfg(), now_fn=_now_market)

    report = rec.reconcile_now()

    assert report.matched is False
    assert len(report.diffs) == 1
    assert report.diffs[0].symbol == "CXW"
    assert len(ks.halts) == 1


def test_old_pending_order_still_explains_diff_no_window_expiry() -> None:
    """WS1 RC-2 regression guard (replaces the retired window-guard test):
    a GTC sell submitted 30 DAYS ago but still pending fill keeps
    explaining the position decrease with NO submitted_at expiry — under
    the old classifier this halted on day 8 (the tier-1 hotfix time
    bomb). Explanation never turns into a halt; for a broker-ABSENT
    symbol the W1-M1 masked counter quietly absorbs the ghost after 3
    consecutive ticks instead (see the masked-external-close tests)."""
    broker = _FakeBroker(positions=[])
    journal = _FakeJournal(
        expected_positions=[_position("XYZ", 50)],
        recent_orders=[
            _order(
                symbol="XYZ",
                role="stop",
                side="sell",
                qty=50,
                submitted_at=_MARKET_OPEN_TIME - dt.timedelta(days=30),
                order_id=99,
            ),
        ],
    )
    ks = _FakeKillSwitch()
    rec = Reconciler(broker, journal, ks, _cfg(), now_fn=_now_market)

    report = rec.reconcile_now()

    assert report.matched is True
    assert report.diffs == []
    assert len(report.explained_diffs) == 1
    assert 99 in report.explained_diffs[0].explained_by_order_ids
    assert ks.halts == []


def test_recorded_fill_explains_only_within_two_ticks() -> None:
    """A FILLED sell explains its diff for 2 reconcile ticks after
    realized_fill_at; beyond that the fill has been consumed (the
    snapshot/tombstone already happened on the tick that recorded it) and
    cannot explain a fresh decrease. Locks the §2.2 lifecycle window."""
    fill_row = _order(
        symbol="XYZ",
        role="thesis_close",
        side="sell",
        qty=50,
        submitted_at=_MARKET_OPEN_TIME - dt.timedelta(days=2),
        order_id=77,
    )
    recent_fill = fill_row.model_copy(
        update={
            "final_status": "filled",
            "realized_fill_qty": 50,
            "realized_fill_price": 9.5,
            # interval 300 s → window is 600 s; 500 s ago is inside.
            "realized_fill_at": _MARKET_OPEN_TIME - dt.timedelta(seconds=500),
        }
    )
    broker = _FakeBroker(positions=[])
    journal = _FakeJournal(
        expected_positions=[_position("XYZ", 50)],
        recent_orders=[recent_fill],
    )
    ks = _FakeKillSwitch()
    rec = Reconciler(broker, journal, ks, _cfg(), now_fn=_now_market)

    report = rec.reconcile_now()
    assert report.matched is True
    assert len(report.explained_diffs) == 1
    assert ks.halts == []

    # Same fill 20 minutes old (beyond 2×300 s) → no longer explains; the
    # diff enters the external-close grace path instead (no halt either —
    # broker absence is zero exposure).
    stale_fill = recent_fill.model_copy(
        update={
            "realized_fill_at": _MARKET_OPEN_TIME - dt.timedelta(minutes=20),
        }
    )
    journal2 = _FakeJournal(
        expected_positions=[_position("XYZ", 50)],
        recent_orders=[stale_fill],
    )
    ks2 = _FakeKillSwitch()
    rec2 = Reconciler(broker, journal2, ks2, _cfg(), now_fn=_now_market)

    report2 = rec2.reconcile_now()
    assert report2.explained_diffs == []
    assert [d.symbol for d in report2.external_close_pending] == ["XYZ"]
    assert ks2.halts == []


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
    assert journal.lifecycle_calls == []


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


# ---------------------------------------------------------------------------
# PDT-SUNSET-2026-06-04: ADR-009 §3.1 — activation gate refresh on tick.
# ---------------------------------------------------------------------------


def test_reconcile_refreshes_pdt_state_on_successful_tick() -> None:
    """ADR-009 §3.1: after a successful reconcile, `pdt_state.refresh`
    is called with the just-fetched broker_account."""
    from execution.pdt_budget import PDTState

    pos = [_position("ACME", 25)]
    account = AccountSnapshot(
        equity=22_300.0, cash=5_000.0, buying_power=44_600.0,
        long_market_value=17_300.0, daypl=0.0,
        snapshot_at=_MARKET_OPEN_TIME,
        last_equity=22_000.0, daytrade_count=1,
    )
    broker = _FakeBroker(positions=pos, account=account)
    journal = _FakeJournal(expected_positions=pos)
    ks = _FakeKillSwitch()
    pdt_state = PDTState(_active=False, _last_equity=0.0)

    rec = Reconciler(
        broker, journal, ks, _cfg(), now_fn=_now_market,
        pdt_state=pdt_state, pdt_enabled=True,
    )
    rec.reconcile_now()

    # Refresh ran and flipped `_active=True` (last_equity=22000 < 25000).
    assert pdt_state.is_active() is True
    assert pdt_state._last_equity == 22_000.0


def test_reconcile_pdt_refresh_skipped_when_pdt_state_none() -> None:
    """Legacy callers that don't pass `pdt_state` keep working unchanged."""
    pos = [_position("ACME", 25)]
    broker = _FakeBroker(positions=pos)
    journal = _FakeJournal(expected_positions=pos)
    ks = _FakeKillSwitch()

    # Construct without pdt_state (default None).
    rec = Reconciler(broker, journal, ks, _cfg(), now_fn=_now_market)
    report = rec.reconcile_now()
    assert report.matched is True
    assert report.suppressed is False


def test_reconcile_pdt_refresh_respects_pdt_enabled_false() -> None:
    """`pdt_enabled=False` short-circuits the gate even when
    last_equity is below threshold."""
    from execution.pdt_budget import PDTState

    pos = [_position("ACME", 25)]
    account = AccountSnapshot(
        equity=22_300.0, cash=5_000.0, buying_power=44_600.0,
        long_market_value=17_300.0, daypl=0.0,
        snapshot_at=_MARKET_OPEN_TIME,
        last_equity=22_000.0, daytrade_count=0,
    )
    broker = _FakeBroker(positions=pos, account=account)
    journal = _FakeJournal(expected_positions=pos)
    ks = _FakeKillSwitch()
    pdt_state = PDTState(_active=True, _last_equity=22_000.0)

    rec = Reconciler(
        broker, journal, ks, _cfg(), now_fn=_now_market,
        pdt_state=pdt_state, pdt_enabled=False,
    )
    rec.reconcile_now()

    # `pdt_enabled=False` forces inactive regardless of last_equity.
    assert pdt_state.is_active() is False


def test_reconcile_pdt_refresh_skipped_on_deferred_tick() -> None:
    """Broker-unavailable path returns deferred=True and never calls
    `pdt_state.refresh` — there's no fresh AccountSnapshot to feed it."""
    from execution.pdt_budget import PDTState

    broker = _FakeBroker(raise_seq=[BrokerUnavailable("offline")])
    journal = _FakeJournal()
    ks = _FakeKillSwitch()
    pdt_state = PDTState(_active=False, _last_equity=27_000.0)

    rec = Reconciler(
        broker, journal, ks, _cfg(), now_fn=_now_market,
        pdt_state=pdt_state, pdt_enabled=True,
    )
    report = rec.reconcile_now()

    assert report.deferred is True
    # Refresh did NOT run — state preserved.
    assert pdt_state._last_equity == 27_000.0
    assert pdt_state.is_active() is False


# ---------------------------------------------------------------------------
# PDT-SUNSET-2026-06-04: scanner hook (AC-11).
# ---------------------------------------------------------------------------


class _RecordingScanner:
    def __init__(self, *, raises: Exception | None = None) -> None:
        self.calls: int = 0
        self._raises = raises

    def run_tick(self) -> Any:
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        from execution.virtual_exits import ScannerReport
        return ScannerReport(0, 0, 0, 0)


@pytest.mark.asyncio
async def test_run_loop_invokes_scanner_after_successful_tick() -> None:
    """Scanner runs once per successful, non-suppressed, non-deferred tick."""
    pos = [_position("ACME", 25)]
    broker = _FakeBroker(positions=pos)
    journal = _FakeJournal(expected_positions=pos)
    ks = _FakeKillSwitch()
    scanner = _RecordingScanner()
    cfg = ReconcilerConfig(interval_seconds_market=1)
    rec = Reconciler(
        broker, journal, ks, cfg, now_fn=_now_market,
        virtual_exits_scanner=scanner,
    )
    await rec.start()
    await asyncio.sleep(0.05)
    await rec.stop()
    assert scanner.calls >= 1


@pytest.mark.asyncio
async def test_run_loop_skips_scanner_on_suppressed_tick() -> None:
    """Off-hours tick → suppressed=True → scanner skipped."""
    pos = [_position("ACME", 25)]
    broker = _FakeBroker(positions=pos)
    journal = _FakeJournal(expected_positions=pos)
    ks = _FakeKillSwitch()
    scanner = _RecordingScanner()
    cfg = ReconcilerConfig(interval_seconds_market=1)
    rec = Reconciler(
        broker, journal, ks, cfg, now_fn=_now_off,  # after-hours
        virtual_exits_scanner=scanner,
    )
    await rec.start()
    await asyncio.sleep(0.05)
    await rec.stop()
    # Suppressed ticks must not invoke the scanner.
    assert scanner.calls == 0


@pytest.mark.asyncio
async def test_run_loop_scanner_exception_does_not_crash_tick(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A scanner exception is logged at ERROR; reconciler continues."""
    import logging
    pos = [_position("ACME", 25)]
    broker = _FakeBroker(positions=pos)
    journal = _FakeJournal(expected_positions=pos)
    ks = _FakeKillSwitch()
    scanner = _RecordingScanner(raises=RuntimeError("scanner boom"))
    cfg = ReconcilerConfig(interval_seconds_market=1)
    rec = Reconciler(
        broker, journal, ks, cfg, now_fn=_now_market,
        virtual_exits_scanner=scanner,
    )
    with caplog.at_level(logging.ERROR, logger="reconciler.reconciler"):
        await rec.start()
        await asyncio.sleep(0.05)
        await rec.stop()
    assert scanner.calls >= 1
    errors = [
        r for r in caplog.records
        if getattr(r, "event", None) == "reconciler.scanner_tick_error"
    ]
    assert len(errors) >= 1


def test_reconcile_pdt_refresh_error_does_not_abort_tick(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A refresh exception is logged and swallowed; the reconcile tick
    completes successfully."""
    import logging

    pos = [_position("ACME", 25)]
    account = AccountSnapshot(
        equity=22_300.0, cash=5_000.0, buying_power=44_600.0,
        long_market_value=17_300.0, daypl=0.0,
        snapshot_at=_MARKET_OPEN_TIME,
        last_equity=22_000.0, daytrade_count=0,
    )
    broker = _FakeBroker(positions=pos, account=account)
    journal = _FakeJournal(expected_positions=pos)
    ks = _FakeKillSwitch()

    class _BrokenPDTState:
        def refresh(self, *_a, **_kw) -> bool:
            raise RuntimeError("boom")

    rec = Reconciler(
        broker, journal, ks, _cfg(), now_fn=_now_market,
        pdt_state=_BrokenPDTState(), pdt_enabled=True,
    )

    with caplog.at_level(logging.ERROR, logger="reconciler.reconciler"):
        report = rec.reconcile_now()

    assert report.matched is True
    errors = [
        r for r in caplog.records
        if getattr(r, "event", None) == "reconciler.pdt_refresh_error"
    ]
    assert len(errors) == 1


# ---------------------------------------------------------------------------
# WS1 (D-078, delta §3.5 hook slot): exit-policy ticker hook — additive
# optional param; WS2's ExitPolicyTicker plugs in here. Invoked AFTER the
# thesis hook on every successful, non-suppressed, non-deferred tick.
# ---------------------------------------------------------------------------


class _RecordingTicker:
    def __init__(
        self,
        *,
        raises: Exception | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.calls: int = 0
        self._raises = raises
        self._events = events

    def run_tick(self) -> None:
        self.calls += 1
        if self._events is not None:
            self._events.append("exit_policy")
        if self._raises is not None:
            raise self._raises


class _RecordingThesis:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def run_tick(self) -> None:
        self._events.append("thesis")


@pytest.mark.asyncio
async def test_run_loop_invokes_exit_policy_ticker_after_thesis_hook() -> None:
    """The ticker runs once per successful, non-suppressed, non-deferred
    tick, in the hook slot after the thesis coordinator (delta §3.5:
    composition order is after the thesis hook so a thesis-driven stop
    replacement lands before the ratchet reads the live stop)."""
    events: list[str] = []
    pos = [_position("ACME", 25)]
    broker = _FakeBroker(positions=pos)
    journal = _FakeJournal(expected_positions=pos)
    ks = _FakeKillSwitch()
    ticker = _RecordingTicker(events=events)
    cfg = ReconcilerConfig(interval_seconds_market=1)
    rec = Reconciler(
        broker, journal, ks, cfg, now_fn=_now_market,
        thesis_coordinator=_RecordingThesis(events),
        exit_policy_ticker=ticker,
    )
    await rec.start()
    await asyncio.sleep(0.05)
    await rec.stop()
    assert ticker.calls >= 1
    # Every exit_policy invocation follows a thesis invocation.
    first_pair = events[:2]
    assert first_pair == ["thesis", "exit_policy"]


@pytest.mark.asyncio
async def test_run_loop_skips_exit_policy_ticker_on_suppressed_tick() -> None:
    """Off-hours tick → suppressed=True → ticker skipped (same gating as
    retro/thesis/scanner hooks)."""
    pos = [_position("ACME", 25)]
    broker = _FakeBroker(positions=pos)
    journal = _FakeJournal(expected_positions=pos)
    ks = _FakeKillSwitch()
    ticker = _RecordingTicker()
    cfg = ReconcilerConfig(interval_seconds_market=1)
    rec = Reconciler(
        broker, journal, ks, cfg, now_fn=_now_off,
        exit_policy_ticker=ticker,
    )
    await rec.start()
    await asyncio.sleep(0.05)
    await rec.stop()
    assert ticker.calls == 0


@pytest.mark.asyncio
async def test_run_loop_exit_policy_ticker_exception_does_not_crash_tick(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A ticker exception is logged at ERROR; the reconcile cadence and
    the broker-side GTC stop (the real protection) are unaffected."""
    import logging
    pos = [_position("ACME", 25)]
    broker = _FakeBroker(positions=pos)
    journal = _FakeJournal(expected_positions=pos)
    ks = _FakeKillSwitch()
    ticker = _RecordingTicker(raises=RuntimeError("ticker boom"))
    cfg = ReconcilerConfig(interval_seconds_market=1)
    rec = Reconciler(
        broker, journal, ks, cfg, now_fn=_now_market,
        exit_policy_ticker=ticker,
    )
    with caplog.at_level(logging.ERROR, logger="reconciler.reconciler"):
        await rec.start()
        await asyncio.sleep(0.05)
        await rec.stop()
    assert ticker.calls >= 1
    errors = [
        r for r in caplog.records
        if getattr(r, "event", None) == "reconciler.exit_policy_tick_error"
    ]
    assert len(errors) >= 1


def test_exit_policy_ticker_default_none_disables_hook() -> None:
    """Additive param: existing constructions (no ticker) are untouched."""
    pos = [_position("ACME", 25)]
    broker = _FakeBroker(positions=pos)
    journal = _FakeJournal(expected_positions=pos)
    ks = _FakeKillSwitch()
    rec = Reconciler(broker, journal, ks, _cfg(), now_fn=_now_market)
    report = rec.reconcile_now()
    assert report.matched is True


# ---------------------------------------------------------------------------
# W1-M1 (review-option-b-ws1-2026-06-10): a never-filling pending sell must
# not mask an external close forever. Broker-absent + pending-sell-covered
# runs the same 3-tick absence counter; only RECORDED fills (within the
# 2-tick window) suppress it.
# ---------------------------------------------------------------------------


def test_pending_sell_masked_external_close_absorbed_after_grace(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """W1-M1: operator manually flattens at the broker while a GTC stop is
    live (Alpaca's DELETE /v2/positions does not cancel open orders).
    Broker 0, journal 25, pending stop covers the qty — but a sell still
    PENDING while the position is absent 3 consecutive ticks cannot be
    fill latency (the FillIngestor polls those same orders every tick).
    Tombstone + ONE alert naming the live order needing cancellation;
    never a halt."""
    import logging
    broker = _FakeBroker(positions=[])
    pending_stop = _order(
        symbol="ACME",
        role="stop",
        side="sell",
        qty=25,
        submitted_at=_MARKET_OPEN_TIME - dt.timedelta(days=2),
        order_id=7,
    )
    journal = _FakeJournal(
        expected_positions=[_position("ACME", 25)],
        recent_orders=[pending_stop],
    )
    ks = _FakeKillSwitch()
    alerter = _FakeAlerter()
    rec = Reconciler(broker, journal, ks, _cfg(), alerter=alerter, now_fn=_now_market)

    # Ticks 1-2: grace — still explained by the pending stop (the
    # INSG-style fill-visibility race must stay halt-free and quiet).
    for _ in range(2):
        report = rec.reconcile_now()
        assert report.matched is True
        assert [ed.diff.symbol for ed in report.explained_diffs] == ["ACME"]
        assert report.external_closes == []
        assert journal.tombstones == []
        assert alerter.calls == []
        assert ks.halts == []

    # Tick 3: absence confirmed while the sell is STILL pending →
    # tombstone + single alert listing the live order, no halt.
    import logging as _logging
    with caplog.at_level(_logging.WARNING, logger="reconciler.reconciler"):
        report = rec.reconcile_now()
    assert report.matched is True
    assert report.external_closes == ["ACME"]
    assert ks.halts == []
    assert len(journal.tombstones) == 1
    assert journal.tombstones[0]["symbol"] == "ACME"
    assert journal.tombstones[0]["source"] == "reconciler_external_close"
    assert len(alerter.calls) == 1
    # The operator must learn WHICH live orders need manual cancellation.
    assert "ord-ACME-stop-7" in alerter.calls[0]
    assert "cancel" in alerter.calls[0].lower()
    events = [
        r for r in caplog.records
        if getattr(r, "event", None) == "position.closed_externally_live_orders"
    ]
    assert len(events) == 1
    del logging

    # Tick 4: journal open view self-healed — clean match, no second alert.
    report = rec.reconcile_now()
    assert report.matched is True
    assert len(alerter.calls) == 1
    assert ks.halts == []


def test_masked_absence_counter_resets_when_symbol_reappears() -> None:
    """Two masked-absent ticks, then the position reappears at the broker
    (transient positions-endpoint weirdness): counter must reset; a later
    absence needs the FULL 3 ticks again."""
    pending_stop = _order(
        symbol="ACME",
        role="stop",
        side="sell",
        qty=25,
        submitted_at=_MARKET_OPEN_TIME - dt.timedelta(days=2),
        order_id=7,
    )
    journal = _FakeJournal(
        expected_positions=[_position("ACME", 25)],
        recent_orders=[pending_stop],
    )
    ks = _FakeKillSwitch()
    broker = _FakeBroker(positions=[])
    rec = Reconciler(broker, journal, ks, _cfg(), now_fn=_now_market)

    rec.reconcile_now()
    rec.reconcile_now()
    assert journal.tombstones == []

    broker._positions = [_position("ACME", 25)]
    report = rec.reconcile_now()
    assert report.matched is True

    broker._positions = []
    rec.reconcile_now()
    rec.reconcile_now()
    assert journal.tombstones == []
    rec.reconcile_now()
    assert len(journal.tombstones) == 1
    assert ks.halts == []


def test_fill_explained_absence_never_trips_masked_counter() -> None:
    """A RECORDED fill (within the 2-tick window) covering the decrease is
    genuine fill latency, not a masked close — the absence counter must
    not run, no matter how many ticks it persists."""
    filled_stop = _order(
        symbol="XYZ",
        role="stop",
        side="sell",
        qty=50,
        submitted_at=_MARKET_OPEN_TIME - dt.timedelta(days=1),
        order_id=11,
    ).model_copy(
        update={
            "final_status": "filled",
            "realized_fill_qty": 50,
            "realized_fill_price": 9.5,
            "realized_fill_at": _MARKET_OPEN_TIME - dt.timedelta(seconds=100),
        }
    )
    broker = _FakeBroker(positions=[])
    journal = _FakeJournal(
        expected_positions=[_position("XYZ", 50)],
        recent_orders=[filled_stop],
    )
    ks = _FakeKillSwitch()
    alerter = _FakeAlerter()
    rec = Reconciler(broker, journal, ks, _cfg(), alerter=alerter, now_fn=_now_market)

    for _ in range(4):
        report = rec.reconcile_now()
        assert report.matched is True
        assert report.external_closes == []
    assert journal.tombstones == []
    assert alerter.calls == []
    assert ks.halts == []


def test_partially_pending_explained_absence_also_counted() -> None:
    """W1-M1 'same logic for partially-pending-explained absence': a
    recorded fill covers part of the decrease, a pending sell the rest —
    still masked, still absorbed after 3 ticks, alert names the pending
    order only."""
    filled_part = _order(
        symbol="ACME",
        role="stop",
        side="sell",
        qty=10,
        submitted_at=_MARKET_OPEN_TIME - dt.timedelta(days=1),
        order_id=21,
    ).model_copy(
        update={
            "final_status": "filled",
            "realized_fill_qty": 10,
            "realized_fill_price": 9.5,
            "realized_fill_at": _MARKET_OPEN_TIME - dt.timedelta(seconds=100),
        }
    )
    pending_rest = _order(
        symbol="ACME",
        role="thesis_close",
        side="sell",
        qty=15,
        submitted_at=_MARKET_OPEN_TIME - dt.timedelta(days=1),
        order_id=22,
    )
    broker = _FakeBroker(positions=[])
    journal = _FakeJournal(
        expected_positions=[_position("ACME", 25)],
        recent_orders=[filled_part, pending_rest],
    )
    ks = _FakeKillSwitch()
    alerter = _FakeAlerter()
    rec = Reconciler(broker, journal, ks, _cfg(), alerter=alerter, now_fn=_now_market)

    rec.reconcile_now()
    rec.reconcile_now()
    assert journal.tombstones == []
    rec.reconcile_now()
    assert len(journal.tombstones) == 1
    assert len(alerter.calls) == 1
    assert "ord-ACME-thesis_close-22" in alerter.calls[0]
    assert ks.halts == []


# ---------------------------------------------------------------------------
# Negative-cash tripwire (hotfix 2026-07-08).
#
# Broker cash < 0 means an entry buy overshot available cash (a KS-7 slip).
# The reconciler — the seam where account snapshots are already taken every
# tick — emits a WARNING journal-log event (`account.cash_negative`) and one
# operator alert per ≥0 → <0 crossing (in-memory edge detector, not one
# alert per 5-minute cycle).
# ---------------------------------------------------------------------------

def _account_with_cash(cash: float, *, equity: float = 2600.0) -> AccountSnapshot:
    return AccountSnapshot(
        equity=equity,
        cash=cash,
        buying_power=max(cash, 0.0),
        long_market_value=equity - cash,
        daypl=0.0,
        snapshot_at=dt.datetime(2026, 7, 8, 14, 30, tzinfo=dt.UTC),
    )


def test_negative_cash_fires_warning_and_alert_once() -> None:
    """(d) cash < 0 → one `account.cash_negative` WARNING + one operator
    alert on the crossing tick; the next tick with cash still negative
    stays silent. No kill-switch halt — the tripwire observes, it does
    not gate."""
    import logging

    captured: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = _Capture(level=logging.WARNING)
    named_logger = logging.getLogger("reconciler.reconciler")
    named_logger.addHandler(handler)
    try:
        broker = _FakeBroker(positions=[], account=_account_with_cash(-17.32))
        journal = _FakeJournal(expected_positions=[])
        ks = _FakeKillSwitch()
        alerter = _FakeAlerter()
        rec = Reconciler(
            broker, journal, ks, _cfg(), alerter=alerter, now_fn=_now_market
        )

        rec.reconcile_now()
        rec.reconcile_now()  # still negative → deduped, no second alert
    finally:
        named_logger.removeHandler(handler)

    assert len(alerter.calls) == 1
    assert "-17.32" in alerter.calls[0]
    warnings = [
        r for r in captured
        if getattr(r, "event", None) == "account.cash_negative"
    ]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING
    assert ks.halts == []


def test_negative_cash_realerts_after_recovery() -> None:
    """(d) The dedupe is per-crossing: negative → recovered → negative
    again alerts twice (once per crossing), while consecutive negative
    ticks inside one episode alert once."""
    broker = _FakeBroker(positions=[], account=_account_with_cash(-5.0))
    journal = _FakeJournal(expected_positions=[])
    ks = _FakeKillSwitch()
    alerter = _FakeAlerter()
    rec = Reconciler(
        broker, journal, ks, _cfg(), alerter=alerter, now_fn=_now_market
    )

    rec.reconcile_now()  # crossing #1 → alert
    assert len(alerter.calls) == 1

    broker._account = _account_with_cash(120.0)  # operator topped up / fill settled
    rec.reconcile_now()  # recovered → resets the edge detector, no alert
    assert len(alerter.calls) == 1

    broker._account = _account_with_cash(-3.0)
    rec.reconcile_now()  # crossing #2 → alert
    rec.reconcile_now()  # still negative → silent
    assert len(alerter.calls) == 2


def test_positive_cash_never_alerts() -> None:
    """(d) Regression: healthy cash produces no tripwire noise."""
    broker = _FakeBroker(positions=[], account=_account_with_cash(1500.0))
    journal = _FakeJournal(expected_positions=[])
    ks = _FakeKillSwitch()
    alerter = _FakeAlerter()
    rec = Reconciler(
        broker, journal, ks, _cfg(), alerter=alerter, now_fn=_now_market
    )

    rec.reconcile_now()
    rec.reconcile_now()

    assert alerter.calls == []
    assert ks.halts == []


# ---------------------------------------------------------------------------
# Naked-position tripwire (hotfix 2026-07-17, incident FEIM).
#
# An open position (qty > 0) with ZERO live protective sell orders at the
# broker means an order-surgery path canceled the protection and re-placed
# nothing (FEIM sat naked at +26% for two days). The reconciler — where
# broker positions are already in hand every tick — WARNs `position.naked`
# and alerts the operator, deduped per position per ET day. No halt.
# ---------------------------------------------------------------------------

from broker.protocol import OpenOrder  # noqa: E402


class _FakeBrokerWithOpenOrders(_FakeBroker):
    def __init__(
        self,
        *,
        positions: list[Position] | None = None,
        open_orders: list[OpenOrder] | None = None,
    ) -> None:
        super().__init__(positions=positions)
        self.open_orders: list[OpenOrder] = list(open_orders or [])

    def get_open_orders(self) -> list[OpenOrder]:
        return list(self.open_orders)


def _protective_sell(symbol: str, *, qty: int = 25) -> OpenOrder:
    return OpenOrder(
        symbol=symbol,
        side="sell",
        qty=qty,
        order_type="stop",
        status="accepted",
        broker_order_id=f"stop-{symbol}",
        client_order_id=f"cid-stop-{symbol}",
    )


def test_naked_position_fires_warning_and_alert_once_per_day() -> None:
    """(c) qty>0 + zero live sell orders at the broker → one
    `position.naked` WARNING + one operator alert; the next tick the
    same day stays silent (dedupe per position per ET day). No halt."""
    import logging

    captured: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = _Capture(level=logging.WARNING)
    named_logger = logging.getLogger("reconciler.reconciler")
    named_logger.addHandler(handler)
    try:
        pos = [_position("FEIM", 19, 52.0)]
        broker = _FakeBrokerWithOpenOrders(positions=pos, open_orders=[])
        journal = _FakeJournal(expected_positions=pos)
        ks = _FakeKillSwitch()
        alerter = _FakeAlerter()
        rec = Reconciler(
            broker, journal, ks, _cfg(), alerter=alerter, now_fn=_now_market
        )

        rec.reconcile_now()
        rec.reconcile_now()  # same ET day → deduped, no second alert
    finally:
        named_logger.removeHandler(handler)

    assert len(alerter.calls) == 1
    assert "FEIM" in alerter.calls[0]
    assert "NAKED" in alerter.calls[0]
    warnings = [
        r for r in captured if getattr(r, "event", None) == "position.naked"
    ]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING
    assert ks.halts == []  # observational — never halts


def test_naked_position_realerts_next_et_day() -> None:
    """(c) The dedupe is per ET day: a position still naked on the next
    trading day pages again."""
    pos = [_position("FEIM", 19, 52.0)]
    broker = _FakeBrokerWithOpenOrders(positions=pos, open_orders=[])
    journal = _FakeJournal(expected_positions=pos)
    ks = _FakeKillSwitch()
    alerter = _FakeAlerter()
    clock = {"now": _MARKET_OPEN_TIME}
    rec = Reconciler(
        broker, journal, ks, _cfg(), alerter=alerter,
        now_fn=lambda: clock["now"],
    )

    rec.reconcile_now()
    assert len(alerter.calls) == 1

    clock["now"] = _MARKET_OPEN_TIME + dt.timedelta(days=1)  # Wed, market hours
    rec.reconcile_now()
    assert len(alerter.calls) == 2


def test_protected_position_never_alerts_and_rearms_dedupe() -> None:
    """(c) A live protective sell order at the broker suppresses the
    tripwire; protection reappearing clears the dedupe so a LATER
    re-nakedness the same day pages again."""
    pos = [_position("FEIM", 19, 52.0)]
    broker = _FakeBrokerWithOpenOrders(
        positions=pos, open_orders=[_protective_sell("FEIM", qty=19)]
    )
    journal = _FakeJournal(expected_positions=pos)
    ks = _FakeKillSwitch()
    alerter = _FakeAlerter()
    rec = Reconciler(
        broker, journal, ks, _cfg(), alerter=alerter, now_fn=_now_market
    )

    rec.reconcile_now()
    assert alerter.calls == []  # protected — no noise

    broker.open_orders = []  # protection vanished mid-day
    rec.reconcile_now()
    assert len(alerter.calls) == 1

    broker.open_orders = [_protective_sell("FEIM", qty=19)]  # operator fixed it
    rec.reconcile_now()
    assert len(alerter.calls) == 1

    broker.open_orders = []  # naked AGAIN the same day → fresh page
    rec.reconcile_now()
    assert len(alerter.calls) == 2


def test_naked_check_inert_without_get_open_orders_surface() -> None:
    """(c) Legacy adapters/fakes without `get_open_orders` leave the
    tripwire inert (no crash, no alert)."""
    pos = [_position("ACME", 25)]
    broker = _FakeBroker(positions=pos)  # no get_open_orders
    journal = _FakeJournal(expected_positions=pos)
    ks = _FakeKillSwitch()
    alerter = _FakeAlerter()
    rec = Reconciler(
        broker, journal, ks, _cfg(), alerter=alerter, now_fn=_now_market
    )

    report = rec.reconcile_now()

    assert report.matched
    assert alerter.calls == []

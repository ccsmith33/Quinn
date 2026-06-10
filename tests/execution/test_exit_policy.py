"""D-079 §3.5/§3.6 — ExitPolicyTicker (ADR-011).

Trailing ratchet: engage at +`trail_activation_r`×R, ratchet the
broker-side GTC stop via atomic replace, never lower, never act without
a live journaled stop. Stale-entry hygiene: cancel unfilled GTC entries
whose ET submission day has passed.

Actuator tests assert broker-call ARGUMENTS, not just journal writes.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest

from broker.protocol import Quote, SubmittedOrder
from execution.exit_policy import ExitPolicyTicker
from journal.exit_policy import (
    ExitPolicyStateRow,
    get_exit_policy_state,
    upsert_exit_policy_state,
)
from journal.migrate import apply_migrations
from journal.models import (
    ExecutionRow,
    FilingRow,
    OrderRow,
    PositionRow,
    PromptRow,
    ProposalRow,
)
from journal.repo import (
    get_orders_for_execution,
    insert_execution,
    insert_filing,
    insert_order,
    insert_position,
    insert_prompt,
    insert_proposal,
)

NOW = dt.datetime(2026, 6, 9, 15, 0, 0, tzinfo=dt.UTC)  # 11:00 ET, market hours


@pytest.fixture
def db(tmp_path: Path) -> str:
    p = str(tmp_path / "journal.db")
    apply_migrations(p)
    return p


class _Journal:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path


class _FakeBroker:
    """Quote + replace/cancel recorder. Asserting against `replaced` /
    `canceled` is the actuator contract — broker-call args, not journal
    rows."""

    def __init__(self, *, last: float = 100.0) -> None:
        self.last_by_symbol: dict[str, float] = {}
        self.default_last = last
        self.replaced: list[dict[str, Any]] = []
        self.canceled: list[str] = []
        self.quote_calls: list[str] = []
        self._replace_should_fail = False
        self._cancel_should_fail = False
        self._fail_quote_symbols: set[str] = set()
        self.next_id = 0

    def fail_next_replace(self) -> None:
        self._replace_should_fail = True

    def fail_next_cancel(self) -> None:
        self._cancel_should_fail = True

    def fail_quotes_for(self, symbol: str) -> None:
        self._fail_quote_symbols.add(symbol)

    def set_last(self, symbol: str, last: float) -> None:
        self.last_by_symbol[symbol] = last

    def get_quote(self, symbol: str) -> Quote:
        self.quote_calls.append(symbol)
        if symbol in self._fail_quote_symbols:
            raise RuntimeError("simulated quote failure")
        last = self.last_by_symbol.get(symbol, self.default_last)
        return Quote(
            symbol=symbol,
            bid=last - 0.01,
            ask=last + 0.01,
            last=last,
            ts=NOW,
        )

    def replace_stop_order(
        self,
        broker_order_id: str,
        *,
        new_stop_price: float,
        client_order_id: str,
    ) -> SubmittedOrder:
        if self._replace_should_fail:
            self._replace_should_fail = False
            raise RuntimeError("simulated replace failure")
        self.replaced.append(
            {
                "old_id": broker_order_id,
                "new_stop_price": new_stop_price,
                "client_order_id": client_order_id,
            }
        )
        self.next_id += 1
        return SubmittedOrder(
            broker_order_id=f"eps-replace-{self.next_id:06d}",
            client_order_id=client_order_id,
            symbol="ACME",
            side="sell",
            qty=100,
            order_type="stop",
            status="accepted",
            submitted_at=NOW,
            stop_price=new_stop_price,
        )

    def cancel_order(self, broker_order_id: str) -> None:
        if self._cancel_should_fail:
            self._cancel_should_fail = False
            raise RuntimeError("simulated cancel failure")
        self.canceled.append(broker_order_id)


class _OutcomeRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def __call__(
        self,
        order_id: int,
        final_status: str,
        *,
        fill_price: float | None,
        fill_qty: int | None,
        fill_at: dt.datetime | None,
    ) -> None:
        self.calls.append((order_id, final_status))


def _proposal_payload(
    *,
    symbol: str = "ACME",
    stop_loss_price: float = 90.0,
    take_profit_price: float | None = 150.0,
    trail_distance_pct: float | None = None,
) -> str:
    payload: dict[str, Any] = {
        "symbol": symbol,
        "direction": "long",
        "size_pct_of_capital": 0.10,
        "entry_style": "market_open",
        "stop_loss_price": stop_loss_price,
        "take_profit_price": take_profit_price,
        "time_horizon_days": 14,
        "conviction": 8,
        "thesis": (
            f"{symbol} announced a material acquisition with concrete "
            "pricing terms; integration timeline plausible."
        ),
        "signals": ["Item 1.01 — Material Definitive Agreement"],
        "exit_conditions": ["Exit on contradicting filing within 5 days"],
        "risk_factors": ["Closing conditions not yet met"],
    }
    if trail_distance_pct is not None:
        payload["trail_distance_pct"] = trail_distance_pct
    return json.dumps(payload)


def _seed_position(
    db: str,
    *,
    symbol: str = "ACME",
    entry_fill: float = 100.0,
    stop_price: float = 90.0,
    qty: int = 100,
    trail_distance_pct: float | None = None,
    raw_response: str | None = None,
    position_open: bool = True,
    stop_final_status: str | None = None,
    accession: str = "eps-acc-1",
) -> tuple[int, int]:
    """filing → prompt → proposal → execution → entry+stop orders (+
    position snapshot). Returns (execution_id, live_stop_order_id)."""
    entry_at = NOW - dt.timedelta(days=3)
    fid = insert_filing(
        db,
        FilingRow(
            accession_number=accession,
            cik=320193,
            form_type="8-K",
            filed_at=entry_at - dt.timedelta(hours=2),
            fetched_at=entry_at - dt.timedelta(hours=1),
            raw_text_path="/tmp/eps.txt",
            content_hash=f"h-{accession}",
            item_codes='["1.01"]',
            issuer_ticker=symbol,
        ),
    )
    try:
        insert_prompt(
            db,
            PromptRow(
                prompt_version="sonnet@epstest",
                name="sonnet",
                file_path="src/prompts/sonnet.txt",
                content_hash="x" * 64,
            ),
        )
    except Exception:  # noqa: BLE001 — already registered by a prior seed
        pass
    pid = insert_proposal(
        db,
        ProposalRow(
            filing_id=fid,
            decision_id=f"eps-d-{accession}",
            model_id="claude-sonnet-4-6",
            prompt_version="sonnet@epstest",
            raw_response=(
                raw_response
                if raw_response is not None
                else _proposal_payload(
                    symbol=symbol,
                    stop_loss_price=stop_price,
                    trail_distance_pct=trail_distance_pct,
                )
            ),
            kind="trade_proposal",
            symbol=symbol,
            direction="long",
            size_pct_requested=0.10,
            conviction=8,
            thesis="material acquisition with concrete pricing terms",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
            cost_usd=0.0,
        ),
    )
    eid = insert_execution(
        db,
        ExecutionRow(
            proposal_id=pid,
            decision="accepted",
            realized_size_pct=0.10,
            realized_dollar_size=qty * entry_fill,
            submitted_orders_json="[]",
        ),
    )
    insert_order(
        db,
        OrderRow(
            execution_id=eid,
            role="entry",
            symbol=symbol,
            side="buy",
            order_type="market",
            qty=qty,
            tif="gtc",
            broker_order_id=f"entry-{accession}",
            submitted_at=entry_at,
            realized_fill_price=entry_fill,
            realized_fill_qty=qty,
            realized_fill_at=entry_at,
            final_status="filled",
        ),
    )
    stop_id = insert_order(
        db,
        OrderRow(
            execution_id=eid,
            role="stop",
            symbol=symbol,
            side="sell",
            order_type="stop",
            qty=qty,
            tif="gtc",
            stop_price=stop_price,
            broker_order_id=f"stop-{accession}",
            submitted_at=entry_at,
            final_status=stop_final_status,
        ),
    )
    if position_open:
        insert_position(
            db,
            PositionRow(
                snapshot_at=entry_at,
                source="reconciler",
                symbol=symbol,
                qty=qty,
                avg_entry_price=entry_fill,
                market_value=qty * entry_fill,
                unrealized_pnl=0.0,
            ),
        )
    return eid, stop_id


def _tick(
    db: str,
    broker: _FakeBroker,
    *,
    recorder: _OutcomeRecorder | None = None,
    now: dt.datetime = NOW,
) -> ExitPolicyTicker:
    kwargs: dict[str, Any] = {}
    if recorder is not None:
        kwargs["record_order_outcome"] = recorder
    ticker = ExitPolicyTicker(
        journal=_Journal(db),
        broker=broker,
        now_fn=lambda: now,
        **kwargs,
    )
    # Sync, like the scanner hook — WS1's reconciler seam invokes the
    # exit-policy ticker without await.
    ticker.run_tick()
    return ticker


# ---------------------------------------------------------------------------
# §3.5 — engagement
# ---------------------------------------------------------------------------


def test_no_engagement_below_activation_threshold(db: str) -> None:
    """entry=100, stop=90 → 1R earned at 110. last=109 → no trail, no
    state, no broker mutation."""
    _seed_position(db)
    broker = _FakeBroker(last=109.0)

    _tick(db, broker)

    assert broker.replaced == []
    eid = 1
    assert get_exit_policy_state(db, execution_id=eid) is None


def test_engage_at_one_r_and_ratchet_same_tick(db: str) -> None:
    """last=110 = entry+1R → engage. Default trail = initial risk as a
    pct of entry = 10%. Target = 110×0.90 = 99 > 90×1.0025 → first
    ratchet fires in the same tick with the right broker args."""
    eid, old_stop_id = _seed_position(db)
    broker = _FakeBroker(last=110.0)
    recorder = _OutcomeRecorder()

    _tick(db, broker, recorder=recorder)

    assert len(broker.replaced) == 1
    rep = broker.replaced[0]
    assert rep["old_id"] == "stop-eps-acc-1"
    assert rep["new_stop_price"] == pytest.approx(99.0)

    state = get_exit_policy_state(db, execution_id=eid)
    assert state is not None
    assert state.trail_engaged is True
    assert state.trail_distance_pct == pytest.approx(10.0)
    assert state.high_water_mark == pytest.approx(110.0)

    # Journal chain: fresh live trailing_stop row; old completed
    # 'replaced'; state pointer rotated to the new row.
    rows = get_orders_for_execution(db, eid)
    trail_rows = [o for o in rows if o.role == "trailing_stop"]
    assert len(trail_rows) == 1
    assert trail_rows[0].stop_price == pytest.approx(99.0)
    assert trail_rows[0].final_status is None
    assert trail_rows[0].tif == "gtc"
    assert recorder.calls == [(old_stop_id, "replaced")]
    assert state.stop_order_journal_id == trail_rows[0].id


def test_ratchet_advances_with_new_high_water(db: str) -> None:
    """Second tick at last=120: high_water 110→120, target 108 >
    99×1.0025 → replace the trailing stop again."""
    eid, _ = _seed_position(db)
    broker = _FakeBroker(last=110.0)
    _tick(db, broker)  # engage + first ratchet to 99

    broker.set_last("ACME", 120.0)
    _tick(db, broker)

    assert len(broker.replaced) == 2
    second = broker.replaced[1]
    # The PATCH target is the broker id of the first replacement — the
    # ratchet tracks the replacement chain, not the original stop.
    assert second["old_id"].startswith("eps-replace-")
    assert second["new_stop_price"] == pytest.approx(108.0)
    state = get_exit_policy_state(db, execution_id=eid)
    assert state is not None
    assert state.high_water_mark == pytest.approx(120.0)


def test_never_lowers_high_water_or_stop_on_price_fall(db: str) -> None:
    """After high_water=120/stop=108, price falls to 112: target stays
    108 (no step), high_water stays 120, no broker call."""
    eid, _ = _seed_position(db)
    broker = _FakeBroker(last=110.0)
    _tick(db, broker)
    broker.set_last("ACME", 120.0)
    _tick(db, broker)
    assert len(broker.replaced) == 2

    broker.set_last("ACME", 112.0)
    _tick(db, broker)

    assert len(broker.replaced) == 2  # no third replace
    state = get_exit_policy_state(db, execution_id=eid)
    assert state is not None
    assert state.high_water_mark == pytest.approx(120.0)


def test_min_ratchet_step_suppresses_patch_spam(db: str) -> None:
    """A high-water advance whose target clears the stop by less than
    min_ratchet_step_pct (0.25%) must NOT replace. stop=99 after first
    ratchet; last=110.2 → target 99.18 < 99×1.0025=99.2475."""
    eid, _ = _seed_position(db)
    broker = _FakeBroker(last=110.0)
    _tick(db, broker)
    assert len(broker.replaced) == 1

    broker.set_last("ACME", 110.2)
    _tick(db, broker)

    assert len(broker.replaced) == 1  # suppressed
    state = get_exit_policy_state(db, execution_id=eid)
    assert state is not None
    assert state.high_water_mark == pytest.approx(110.2)  # advance persisted


def test_no_in_the_money_stop_after_fast_fall(db: str) -> None:
    """high_water=120 (stop ratcheted to 99 only — simulate a missed
    catch-up by seeding state directly), price gaps to 100: target 108
    would sit ABOVE the market and fire instantly. The ticker skips —
    the existing broker stop remains the protection (ADR-011 lag
    consequence, accepted)."""
    eid, stop_id = _seed_position(db)
    upsert_exit_policy_state(
        db,
        ExitPolicyStateRow(
            execution_id=eid,
            symbol="ACME",
            trail_distance_pct=10.0,
            trail_engaged=True,
            high_water_mark=120.0,
            stop_order_journal_id=stop_id,
        ),
    )
    broker = _FakeBroker(last=100.0)

    _tick(db, broker)

    assert broker.replaced == []


def test_proposed_trail_distance_pct_overrides_default(db: str) -> None:
    """Analyzer proposed trail_distance_pct=5 → target = 110×0.95 =
    104.5, not the default-10% 99."""
    _seed_position(db, trail_distance_pct=5.0)
    broker = _FakeBroker(last=110.0)

    _tick(db, broker)

    assert len(broker.replaced) == 1
    assert broker.replaced[0]["new_stop_price"] == pytest.approx(104.5)


def test_default_trail_clamped_to_15_pct(db: str) -> None:
    """entry=100, stop=80 → 20% risk distance clamps to 15. Activation
    at entry+1R=120; last=140 → target 140×0.85 = 119."""
    eid, _ = _seed_position(db, stop_price=80.0)
    broker = _FakeBroker(last=140.0)

    _tick(db, broker)

    state = get_exit_policy_state(db, execution_id=eid)
    assert state is not None
    assert state.trail_distance_pct == pytest.approx(15.0)
    assert len(broker.replaced) == 1
    assert broker.replaced[0]["new_stop_price"] == pytest.approx(119.0)


def test_default_trail_clamped_to_1_pct(db: str) -> None:
    """entry=100, stop=99.6 → 0.4% risk distance clamps up to 1%.
    Activation at 100.4; last=101 → target 101×0.99 = 99.99."""
    eid, _ = _seed_position(db, stop_price=99.6)
    broker = _FakeBroker(last=101.0)

    _tick(db, broker)

    state = get_exit_policy_state(db, execution_id=eid)
    assert state is not None
    assert state.trail_distance_pct == pytest.approx(1.0)
    assert len(broker.replaced) == 1
    assert broker.replaced[0]["new_stop_price"] == pytest.approx(99.99)


def test_restart_resumes_from_persisted_high_water(db: str) -> None:
    """Restart semantics (§3.5): stored high_water=120 survives a
    restart; last=115 keeps it; the stop ratchets from the stored mark,
    not from the live quote."""
    eid, stop_id = _seed_position(db)
    upsert_exit_policy_state(
        db,
        ExitPolicyStateRow(
            execution_id=eid,
            symbol="ACME",
            trail_distance_pct=10.0,
            trail_engaged=True,
            high_water_mark=120.0,
            stop_order_journal_id=stop_id,
        ),
    )
    broker = _FakeBroker(last=115.0)

    _tick(db, broker)

    # target = 120×0.90 = 108 (from stored high-water), below last=115.
    assert len(broker.replaced) == 1
    assert broker.replaced[0]["new_stop_price"] == pytest.approx(108.0)
    state = get_exit_policy_state(db, execution_id=eid)
    assert state is not None
    assert state.high_water_mark == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# §3.5 — invariants
# ---------------------------------------------------------------------------


def test_never_acts_without_live_journaled_stop(db: str) -> None:
    """Stop row already terminal (final_status set) → no live stop → no
    quote, no replace, no state. The invariant is structural."""
    eid, _ = _seed_position(db, stop_final_status="canceled")
    broker = _FakeBroker(last=140.0)

    _tick(db, broker)

    assert broker.replaced == []
    assert broker.quote_calls == []
    assert get_exit_policy_state(db, execution_id=eid) is None


def test_skips_position_closed_in_journal(db: str) -> None:
    """Live stop row but no open journal position (closed/tombstoned)
    → skip."""
    eid, _ = _seed_position(db, position_open=False)
    broker = _FakeBroker(last=140.0)

    _tick(db, broker)

    assert broker.replaced == []
    assert get_exit_policy_state(db, execution_id=eid) is None


def test_replace_failure_keeps_high_water_and_retries_next_tick(db: str) -> None:
    """PATCH failure: original stop still live (broker contract). The
    high-water advance persists; the next tick retries and succeeds."""
    eid, stop_id = _seed_position(db)
    broker = _FakeBroker(last=110.0)
    broker.fail_next_replace()
    recorder = _OutcomeRecorder()

    _tick(db, broker, recorder=recorder)

    assert broker.replaced == []
    assert recorder.calls == []
    state = get_exit_policy_state(db, execution_id=eid)
    assert state is not None
    assert state.high_water_mark == pytest.approx(110.0)
    # No phantom journal row.
    trail_rows = [
        o for o in get_orders_for_execution(db, eid) if o.role == "trailing_stop"
    ]
    assert trail_rows == []

    _tick(db, broker, recorder=recorder)  # retry succeeds

    assert len(broker.replaced) == 1
    assert broker.replaced[0]["new_stop_price"] == pytest.approx(99.0)
    assert recorder.calls == [(stop_id, "replaced")]


def test_quote_failure_on_one_symbol_does_not_stop_others(db: str) -> None:
    """Error isolation: a quote failure on ACME must not prevent the
    WIDG execution from ratcheting."""
    _seed_position(db, accession="eps-acc-1")
    _seed_position(
        db,
        symbol="WIDG",
        accession="eps-acc-2",
    )
    broker = _FakeBroker(last=110.0)
    broker.fail_quotes_for("ACME")

    _tick(db, broker)

    assert len(broker.replaced) == 1
    assert broker.replaced[0]["old_id"] == "stop-eps-acc-2"


# ---------------------------------------------------------------------------
# §3.6 — stale-entry hygiene
# ---------------------------------------------------------------------------


def _seed_entry_order(
    db: str,
    eid: int,
    *,
    broker_order_id: str,
    submitted_at: dt.datetime,
    tif: str = "gtc",
    filled: bool = False,
    symbol: str = "ZZZQ",
) -> None:
    insert_order(
        db,
        OrderRow(
            execution_id=eid,
            role="entry",
            symbol=symbol,
            side="buy",
            order_type="limit",
            qty=10,
            limit_price=10.0,
            tif=tif,
            broker_order_id=broker_order_id,
            submitted_at=submitted_at,
            realized_fill_price=10.0 if filled else None,
            realized_fill_qty=10 if filled else None,
            realized_fill_at=submitted_at if filled else None,
            final_status="filled" if filled else None,
        ),
    )


def test_stale_gtc_entry_canceled_after_entry_day(db: str) -> None:
    """An unfilled GTC entry submitted on a prior ET day is canceled
    (the bracket parent cancel cascades to unfilled children). The
    ticker journals NOTHING — fill ingestion records the outcome."""
    eid, _ = _seed_position(db)  # provides execution to attach to
    yesterday = NOW - dt.timedelta(days=1)
    _seed_entry_order(
        db, eid, broker_order_id="stale-entry-1", submitted_at=yesterday
    )
    broker = _FakeBroker(last=100.0)
    recorder = _OutcomeRecorder()

    _tick(db, broker, recorder=recorder)

    assert "stale-entry-1" in broker.canceled
    # §3.6: no outcome write from the ticker for the canceled entry.
    assert all(status != "canceled" for _, status in recorder.calls)


def test_same_day_gtc_entry_not_canceled(db: str) -> None:
    eid, _ = _seed_position(db)
    _seed_entry_order(
        db,
        eid,
        broker_order_id="fresh-entry-1",
        submitted_at=NOW - dt.timedelta(hours=2),
    )
    broker = _FakeBroker(last=100.0)

    _tick(db, broker)

    assert "fresh-entry-1" not in broker.canceled


def test_filled_entry_not_canceled(db: str) -> None:
    eid, _ = _seed_position(db)
    _seed_entry_order(
        db,
        eid,
        broker_order_id="filled-entry-1",
        submitted_at=NOW - dt.timedelta(days=2),
        filled=True,
    )
    broker = _FakeBroker(last=100.0)

    _tick(db, broker)

    assert "filled-entry-1" not in broker.canceled


def test_day_tif_entry_not_canceled(db: str) -> None:
    """DAY entries expire broker-side; hygiene only owns GTC ones."""
    eid, _ = _seed_position(db)
    _seed_entry_order(
        db,
        eid,
        broker_order_id="day-entry-1",
        submitted_at=NOW - dt.timedelta(days=2),
        tif="day",
    )
    broker = _FakeBroker(last=100.0)

    _tick(db, broker)

    assert "day-entry-1" not in broker.canceled


def test_stale_entry_cancel_failure_does_not_stop_tick(db: str) -> None:
    """A cancel failure is logged and the tick continues — the ratchet
    still runs."""
    eid, _ = _seed_position(db)
    _seed_entry_order(
        db,
        eid,
        broker_order_id="stale-entry-err",
        submitted_at=NOW - dt.timedelta(days=1),
    )
    broker = _FakeBroker(last=110.0)
    broker.fail_next_cancel()

    _tick(db, broker)

    assert broker.canceled == []
    # Ratchet still engaged + fired for the open ACME position.
    assert len(broker.replaced) == 1

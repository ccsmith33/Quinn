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
import logging
from pathlib import Path
from typing import Any

import pytest

from broker.protocol import OpenOrder, Position, Quote, SubmittedOrder
from config.loader import TrailStage
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


class _FakeBrokerWithLookup(_FakeBroker):
    """Adds the WS1 `get_order_by_id` surface + a `submit_order`
    recorder — the self-heal paths interrogate the dead PATCH target and
    place a FRESH stop via plain submit."""

    def __init__(self, *, last: float = 100.0) -> None:
        super().__init__(last=last)
        self.orders_by_id: dict[str, SubmittedOrder] = {}
        self.submitted: list[Any] = []

    def seed_order(self, order: SubmittedOrder) -> None:
        self.orders_by_id[order.broker_order_id] = order

    def get_order_by_id(self, broker_order_id: str) -> SubmittedOrder | None:
        return self.orders_by_id.get(broker_order_id)

    def submit_order(self, req: Any) -> SubmittedOrder:
        self.submitted.append(req)
        self.next_id += 1
        return SubmittedOrder(
            broker_order_id=f"eps-heal-{self.next_id:06d}",
            client_order_id=req.client_order_id,
            symbol=req.symbol,
            side=req.side,
            qty=req.qty,
            order_type=req.order_type,
            status="accepted",
            submitted_at=NOW,
            stop_price=req.stop_price,
        )


class _FakeBrokerFull(_FakeBrokerWithLookup):
    """Adds broker-truth position/open-order surfaces — the stateless
    self-heal arm keys on these (the same condition the naked tripwire
    detects)."""

    def __init__(self, *, last: float = 100.0) -> None:
        super().__init__(last=last)
        self.positions: list[Position] = []
        self.open_orders: list[OpenOrder] = []

    def get_positions(self) -> list[Position]:
        return list(self.positions)

    def get_open_orders(self) -> list[OpenOrder]:
        return list(self.open_orders)


def _broker_position(symbol: str = "ACME", *, qty: int = 100) -> Position:
    return Position(
        symbol=symbol,
        qty=qty,
        avg_entry_price=100.0,
        market_value=qty * 100.0,
        unrealized_pnl=0.0,
    )


def _broker_sell_order(symbol: str = "ACME", *, qty: int = 100) -> OpenOrder:
    return OpenOrder(
        symbol=symbol,
        side="sell",
        qty=qty,
        order_type="stop",
        status="accepted",
        broker_order_id=f"live-sell-{symbol}",
        client_order_id=f"cid-live-sell-{symbol}",
    )


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
    trail_stages: list[TrailStage] | None = None,
    breakeven_floor_gain_pct: float | None = None,
) -> ExitPolicyTicker:
    kwargs: dict[str, Any] = {}
    if recorder is not None:
        kwargs["record_order_outcome"] = recorder
    if trail_stages is not None:
        kwargs["trail_stages"] = trail_stages
    if breakeven_floor_gain_pct is not None:
        kwargs["breakeven_floor_gain_pct"] = breakeven_floor_gain_pct
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
# §3.5 — staged trail tightening
# ---------------------------------------------------------------------------


def test_stage_crossing_tightens_and_ratchets_from_current_hwm(
    db: str, caplog: pytest.LogCaptureFixture
) -> None:
    """entry=100, stop=90 (base 10%), stage (gain 15% → trail 5%).
    Tick 1 last=110: gain 10% < 15 → base width, stop→99. Tick 2
    last=120: gain 20% ≥ 15 → width 5%, stop jumps 99→114 in a single
    PATCH from the CURRENT high-water (intended). The ratcheted payload
    carries the effective width and active stage."""
    stages = [TrailStage(gain_pct=15.0, trail_pct=5.0)]
    _seed_position(db)
    broker = _FakeBroker(last=110.0)

    _tick(db, broker, trail_stages=stages)
    assert len(broker.replaced) == 1
    assert broker.replaced[0]["new_stop_price"] == pytest.approx(99.0)

    broker.set_last("ACME", 120.0)
    with caplog.at_level(logging.INFO, logger="execution.exit_policy"):
        _tick(db, broker, trail_stages=stages)

    assert len(broker.replaced) == 2
    assert broker.replaced[1]["new_stop_price"] == pytest.approx(114.0)
    ratcheted = [
        rec
        for rec in caplog.records
        if getattr(rec, "event", "") == "exit_policy.stop_ratcheted"
    ]
    assert len(ratcheted) == 1
    assert ratcheted[0].effective_trail_pct == pytest.approx(5.0)  # type: ignore[attr-defined]
    assert ratcheted[0].trail_stage_gain_pct == pytest.approx(15.0)  # type: ignore[attr-defined]


def test_hwm_below_milestone_keeps_base_width(db: str) -> None:
    """Gain below the first milestone → base width untouched: last=110
    (gain 10% < 50) trails at the base 10% → target 99."""
    stages = [TrailStage(gain_pct=50.0, trail_pct=5.0)]
    _seed_position(db)
    broker = _FakeBroker(last=110.0)

    _tick(db, broker, trail_stages=stages)

    assert len(broker.replaced) == 1
    assert broker.replaced[0]["new_stop_price"] == pytest.approx(99.0)


def test_stage_never_widens_base_width(db: str) -> None:
    """A crossed stage whose trail_pct is WIDER than the base is ignored
    — the effective width is min(base, stage), tighten-only. gain 10% ≥
    5 but 12% > base 10% → target stays 99."""
    stages = [TrailStage(gain_pct=5.0, trail_pct=12.0)]
    _seed_position(db)
    broker = _FakeBroker(last=110.0)

    _tick(db, broker, trail_stages=stages)

    assert len(broker.replaced) == 1
    assert broker.replaced[0]["new_stop_price"] == pytest.approx(99.0)


def test_empty_stages_matches_legacy_behavior(db: str) -> None:
    """Explicit empty stage list = exactly the pre-feature behavior
    (regression guard for the default config shape)."""
    eid, _ = _seed_position(db)
    broker = _FakeBroker(last=110.0)

    _tick(db, broker, trail_stages=[])

    assert len(broker.replaced) == 1
    assert broker.replaced[0]["new_stop_price"] == pytest.approx(99.0)
    state = get_exit_policy_state(db, execution_id=eid)
    assert state is not None
    assert state.trail_distance_pct == pytest.approx(10.0)


def test_stop_monotonic_across_stage_transition(db: str) -> None:
    """The ratchet's never-lower invariant holds across a stage
    boundary: 99 → 114 (tightened), then a price fall to 116 leaves
    the stop at 114 — every replacement strictly raises."""
    stages = [TrailStage(gain_pct=15.0, trail_pct=5.0)]
    eid, _ = _seed_position(db)
    broker = _FakeBroker(last=110.0)
    _tick(db, broker, trail_stages=stages)
    broker.set_last("ACME", 120.0)
    _tick(db, broker, trail_stages=stages)

    broker.set_last("ACME", 116.0)
    _tick(db, broker, trail_stages=stages)

    prices = [r["new_stop_price"] for r in broker.replaced]
    assert prices == sorted(prices)
    assert len(broker.replaced) == 2  # no replace on the fall
    assert broker.replaced[-1]["new_stop_price"] == pytest.approx(114.0)
    state = get_exit_policy_state(db, execution_id=eid)
    assert state is not None
    assert state.high_water_mark == pytest.approx(120.0)


def test_stages_apply_to_preexisting_state_on_restart(db: str) -> None:
    """Deploy-time semantics: a position whose trail engaged BEFORE
    stages existed (persisted base width 15%) tightens on the first
    tick after restart — the effective width is derived at runtime
    from config + current HWM, never frozen into the state row."""
    stages = [TrailStage(gain_pct=15.0, trail_pct=8.0)]
    eid, stop_id = _seed_position(db, entry_fill=18.31, stop_price=15.20, qty=50)
    upsert_exit_policy_state(
        db,
        ExitPolicyStateRow(
            execution_id=eid,
            symbol="ACME",
            trail_distance_pct=15.0,
            trail_engaged=True,
            high_water_mark=21.42,
            stop_order_journal_id=stop_id,
        ),
    )
    broker = _FakeBroker(last=21.0)

    _tick(db, broker, trail_stages=stages)

    # gain = (21.42 − 18.31)/18.31 ≈ 17% ≥ 15 → width 8% → 21.42×0.92.
    assert len(broker.replaced) == 1
    assert broker.replaced[0]["new_stop_price"] == pytest.approx(19.71)
    state = get_exit_policy_state(db, execution_id=eid)
    assert state is not None
    assert state.trail_distance_pct == pytest.approx(15.0)  # base preserved


def test_live_scenario_base_trail_floor(db: str) -> None:
    """TENX-shaped control: entry 18.31, initial risk 16.99% of entry →
    default width clamps to 15. HWM 21.42 → floor 21.42×0.85 = 18.21."""
    _seed_position(db, entry_fill=18.31, stop_price=15.20, qty=50)
    broker = _FakeBroker(last=21.42)

    _tick(db, broker)

    assert len(broker.replaced) == 1
    assert broker.replaced[0]["new_stop_price"] == pytest.approx(18.21)


def test_live_scenario_staged_floor(db: str) -> None:
    """Same position with stages [(15, 8)]: gain 17% ≥ 15 → width 8 →
    floor 21.42×0.92 = 19.71 (vs 18.21 flat — the TENX giveback fix)."""
    _seed_position(db, entry_fill=18.31, stop_price=15.20, qty=50)
    broker = _FakeBroker(last=21.42)

    _tick(db, broker, trail_stages=[TrailStage(gain_pct=15.0, trail_pct=8.0)])

    assert len(broker.replaced) == 1
    assert broker.replaced[0]["new_stop_price"] == pytest.approx(19.71)


# ---------------------------------------------------------------------------
# §3.5 — breakeven floor (study policy E): once the high-water gain vs
# entry clears the threshold, the stop never sits below entry. HWM-based,
# never narrows the band — a lower bound applied AFTER the width math.
# ---------------------------------------------------------------------------


def test_breakeven_floor_inactive_below_threshold(db: str) -> None:
    """entry=100, stop=90 (base 10%). last=110 → gain 10% < 12% threshold
    → floor OFF: the trail-derived stop (99, BELOW entry) is submitted
    unchanged. Proves the floor never engages before the peak clears the
    threshold."""
    _seed_position(db)
    broker = _FakeBroker(last=110.0)

    _tick(db, broker, breakeven_floor_gain_pct=12.0)

    assert len(broker.replaced) == 1
    assert broker.replaced[0]["new_stop_price"] == pytest.approx(99.0)


def test_breakeven_floor_binds_at_threshold_raises_stop_to_entry(
    db: str, caplog: pytest.LogCaptureFixture
) -> None:
    """entry=100, stop=85 (base 15% trail). last=115 = entry+1R → engage;
    gain 15% >= 12% and the trail-derived stop (115×0.85 = 97.75) is BELOW
    entry → the floor binds and raises the submitted stop to entry (100).
    The binding is logged once."""
    _seed_position(db, stop_price=85.0)
    broker = _FakeBroker(last=115.0)

    with caplog.at_level(logging.INFO, logger="execution.exit_policy"):
        _tick(db, broker, breakeven_floor_gain_pct=12.0)

    assert len(broker.replaced) == 1
    assert broker.replaced[0]["new_stop_price"] == pytest.approx(100.0)
    floored = [
        rec
        for rec in caplog.records
        if getattr(rec, "event", "") == "exit_policy.breakeven_floor_applied"
    ]
    assert len(floored) == 1
    assert floored[0].entry_price == pytest.approx(100.0)  # type: ignore[attr-defined]
    assert floored[0].high_water_mark == pytest.approx(115.0)  # type: ignore[attr-defined]
    assert floored[0].trail_stop == pytest.approx(97.75)  # type: ignore[attr-defined]
    assert floored[0].floored_stop == pytest.approx(100.0)  # type: ignore[attr-defined]


def test_breakeven_floor_does_not_narrow_staged_trail(
    db: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Compose with staged widths: entry=100, stop=85 (base 15%), stage
    (gain 20% → 8%). last=130 → gain 30% ≥ 20 → width 8% → trail stop
    130×0.92 = 119.6, ALREADY above entry. The floor is a lower bound: it
    must NOT drag the staged stop down to entry, and must NOT log a
    binding (it changed nothing)."""
    _seed_position(db, stop_price=85.0)
    broker = _FakeBroker(last=130.0)
    stages = [TrailStage(gain_pct=20.0, trail_pct=8.0)]

    with caplog.at_level(logging.INFO, logger="execution.exit_policy"):
        _tick(db, broker, trail_stages=stages, breakeven_floor_gain_pct=12.0)

    assert len(broker.replaced) == 1
    assert broker.replaced[0]["new_stop_price"] == pytest.approx(119.6)
    assert not [
        rec
        for rec in caplog.records
        if getattr(rec, "event", "") == "exit_policy.breakeven_floor_applied"
    ]


def test_breakeven_floor_preserves_ratchet_only_invariant(db: str) -> None:
    """The floor never lowers a stop. Tick1 last=115 → floor to 100.
    Tick2 price falls to 112 → HWM stays 115, floor still says 100 =
    current stop → no replace. Tick3 last=140 → trail stop 119 (above
    entry) → stop rises to 119. Every submitted stop strictly ascends."""
    _seed_position(db, stop_price=85.0)
    broker = _FakeBroker(last=115.0)
    _tick(db, broker, breakeven_floor_gain_pct=12.0)
    assert broker.replaced[-1]["new_stop_price"] == pytest.approx(100.0)

    broker.set_last("ACME", 112.0)
    _tick(db, broker, breakeven_floor_gain_pct=12.0)
    assert len(broker.replaced) == 1  # floored stop already at entry

    broker.set_last("ACME", 140.0)
    _tick(db, broker, breakeven_floor_gain_pct=12.0)

    prices = [r["new_stop_price"] for r in broker.replaced]
    assert prices == sorted(prices)
    assert prices[-1] == pytest.approx(119.0)  # 140×0.85, floor no longer binds


def test_breakeven_floor_applies_to_preexisting_state_on_restart(db: str) -> None:
    """Runtime derivation: a position whose trail engaged BEFORE the floor
    existed (state row carries no floor) gets floored on the first tick
    after the config change — the floor is derived from config + HWM every
    tick, never frozen into state. Seeded HWM=115 (gain 15% ≥ 12), base
    trail 15% → 97.75 → floored to entry 100."""
    eid, stop_id = _seed_position(db, stop_price=85.0)
    upsert_exit_policy_state(
        db,
        ExitPolicyStateRow(
            execution_id=eid,
            symbol="ACME",
            trail_distance_pct=15.0,
            trail_engaged=True,
            high_water_mark=115.0,
            stop_order_journal_id=stop_id,
        ),
    )
    broker = _FakeBroker(last=113.0)

    _tick(db, broker, breakeven_floor_gain_pct=12.0)

    assert len(broker.replaced) == 1
    assert broker.replaced[0]["new_stop_price"] == pytest.approx(100.0)


def test_breakeven_floor_respected_by_stateless_heal_engagement(db: str) -> None:
    """The stateless-heal engagement path computes a fresh stop — the
    floor must apply there too. Naked position, no state row, gain past
    activation, wide (15%) trail: the healed trailing stop is floored to
    entry (100) instead of 115×0.85 = 97.75."""
    _seed_position(db, stop_price=85.0, stop_final_status="canceled")
    broker = _FakeBrokerFull(last=115.0)
    broker.positions = [_broker_position("ACME")]
    broker.open_orders = []

    _tick(db, broker, breakeven_floor_gain_pct=12.0)

    assert len(broker.submitted) == 1
    assert broker.submitted[0].stop_price == pytest.approx(100.0)


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
# Self-heal (incident FEIM 2026-07-16): trail-armed position whose stop
# was canceled out from under the ratchet must get a FRESH stop.
# ---------------------------------------------------------------------------


def test_selfheal_places_fresh_stop_when_state_references_canceled_order(
    db: str,
) -> None:
    """FEIM reconstruction, silent-blindness arm: fill ingestion has
    recorded the stop's cancel (final_status='canceled'), so the row
    left the live set and the ratchet's live-stop scan goes blind while
    `exit_policy_state` is still trail-engaged — the position sat naked
    for two days in prod. The heal pass must place a fresh GTC stop at
    the computed trail floor on the very next tick and rotate the
    pointer."""
    eid, stop_id = _seed_position(
        db, stop_final_status="canceled", trail_distance_pct=10.0
    )
    upsert_exit_policy_state(
        db,
        ExitPolicyStateRow(
            execution_id=eid,
            symbol="ACME",
            trail_distance_pct=10.0,
            trail_engaged=True,
            high_water_mark=130.0,
            stop_order_journal_id=stop_id,  # points at the CANCELED order
        ),
    )
    broker = _FakeBrokerWithLookup(last=128.0)

    _tick(db, broker)

    # Fresh GTC stop at the trail floor: 130 * (1 - 10%) = 117.0.
    assert len(broker.submitted) == 1
    req = broker.submitted[0]
    assert req.side == "sell"
    assert req.order_type == "stop"
    assert req.tif == "gtc"
    assert req.qty == 100
    assert req.stop_price == pytest.approx(117.0)
    # No doomed PATCH attempts against the dead order.
    assert broker.replaced == []

    # Journal: live trailing_stop row; pointer rotated onto it.
    trail_rows = [
        o for o in get_orders_for_execution(db, eid)
        if o.role == "trailing_stop" and o.final_status is None
    ]
    assert len(trail_rows) == 1
    assert trail_rows[0].stop_price == pytest.approx(117.0)
    assert "trail_selfheal" in (trail_rows[0].notes or "")
    state = get_exit_policy_state(db, execution_id=eid)
    assert state is not None
    assert state.stop_order_journal_id == trail_rows[0].id

    # Next tick: healed position is a normal ratchet citizen again — no
    # second heal row.
    _tick(db, broker)
    assert len(broker.submitted) == 1


def test_selfheal_after_patch_fails_against_dead_order(db: str) -> None:
    """FEIM reconstruction, PATCH-failure arm: the journal row is still
    live (`final_status` NULL) but the broker order is CANCELED — the
    PATCH fails every tick forever. The ratchet must interrogate the
    target, see it is dead, and submit a fresh stop at the computed
    floor instead of retrying a doomed PATCH."""
    eid, stop_id = _seed_position(db, trail_distance_pct=10.0)
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
    broker = _FakeBrokerWithLookup(last=125.0)
    broker.fail_next_replace()
    broker.seed_order(
        SubmittedOrder(
            broker_order_id="stop-eps-acc-1",
            client_order_id="cid-stop",
            symbol="ACME",
            side="sell",
            qty=100,
            order_type="stop",
            status="canceled",  # canceled out from under the ratchet
            submitted_at=NOW,
            stop_price=90.0,
        )
    )
    recorder = _OutcomeRecorder()

    _tick(db, broker, recorder=recorder)

    # Fresh stop at the computed floor: hwm=max(120,125)=125 → 112.5.
    assert len(broker.submitted) == 1
    assert broker.submitted[0].stop_price == pytest.approx(112.5)
    assert broker.replaced == []  # PATCH failed; no successful replace
    # Dead row completed with the observed broker status.
    assert recorder.calls == [(stop_id, "canceled")]
    trail_rows = [
        o for o in get_orders_for_execution(db, eid)
        if o.role == "trailing_stop" and o.final_status is None
    ]
    assert len(trail_rows) == 1
    state = get_exit_policy_state(db, execution_id=eid)
    assert state is not None
    assert state.stop_order_journal_id == trail_rows[0].id


def test_selfheal_not_triggered_when_patch_target_still_live(db: str) -> None:
    """A transient PATCH failure against a still-live order keeps the
    original contract: no fresh submit, no journal row, retry next
    tick."""
    eid, _ = _seed_position(db, trail_distance_pct=10.0)
    broker = _FakeBrokerWithLookup(last=110.0)
    broker.fail_next_replace()
    broker.seed_order(
        SubmittedOrder(
            broker_order_id="stop-eps-acc-1",
            client_order_id="cid-stop",
            symbol="ACME",
            side="sell",
            qty=100,
            order_type="stop",
            status="accepted",  # still live — transient failure
            submitted_at=NOW,
            stop_price=90.0,
        )
    )

    _tick(db, broker)

    assert broker.submitted == []
    trail_rows = [
        o for o in get_orders_for_execution(db, eid) if o.role == "trailing_stop"
    ]
    assert trail_rows == []
    state = get_exit_policy_state(db, execution_id=eid)
    assert state is not None  # engaged this tick; hwm persisted for retry


def test_selfheal_skips_closed_position(db: str) -> None:
    """Engaged state + canceled stop but the position is closed in the
    journal → nothing to protect; no fresh stop."""
    eid, stop_id = _seed_position(
        db, stop_final_status="canceled", position_open=False
    )
    upsert_exit_policy_state(
        db,
        ExitPolicyStateRow(
            execution_id=eid,
            symbol="ACME",
            trail_distance_pct=10.0,
            trail_engaged=True,
            high_water_mark=130.0,
            stop_order_journal_id=stop_id,
        ),
    )
    broker = _FakeBrokerWithLookup(last=128.0)

    _tick(db, broker)

    assert broker.submitted == []


# ---------------------------------------------------------------------------
# Stateless self-heal (FEIM follow-up, execution 37): a position whose
# stop was canceled BEFORE the trail ever engaged has NO exit_policy_state
# row — the engaged-state heal arm can never see it. The stateless arm
# scans open broker positions with zero live sell orders (the naked
# tripwire's condition) and repairs immediately.
# ---------------------------------------------------------------------------


def test_stateless_naked_past_activation_engages_with_fresh_trailing_stop(
    db: str,
) -> None:
    """(a) FEIM-37 reconstruction: no state row, no live stop, gain past
    the +1R activation → the heal ENGAGES properly: state row created
    with HWM initialized from the current price, trailing stop placed at
    the computed floor (mode=engaged_fresh)."""
    eid, _ = _seed_position(
        db, stop_final_status="canceled", trail_distance_pct=10.0
    )
    assert get_exit_policy_state(db, execution_id=eid) is None  # precondition
    broker = _FakeBrokerFull(last=126.0)  # entry 100, risk 10 → 1R at 110
    broker.positions = [_broker_position("ACME")]
    broker.open_orders = []  # ZERO live sell orders — naked at the broker

    _tick(db, broker)

    # Fresh trailing stop at the floor: 126 × (1 − 10%) = 113.4.
    assert len(broker.submitted) == 1
    req = broker.submitted[0]
    assert (req.side, req.order_type, req.tif, req.qty) == (
        "sell", "stop", "gtc", 100
    )
    assert req.stop_price == pytest.approx(113.4)

    # State row created: engaged, HWM = current price, pointer → new row.
    state = get_exit_policy_state(db, execution_id=eid)
    assert state is not None
    assert state.trail_engaged is True
    assert state.high_water_mark == pytest.approx(126.0)
    heal_rows = [
        o for o in get_orders_for_execution(db, eid)
        if o.role == "trailing_stop" and o.final_status is None
    ]
    assert len(heal_rows) == 1
    assert state.stop_order_journal_id == heal_rows[0].id
    assert "mode=engaged_fresh" in (heal_rows[0].notes or "")


def test_stateless_naked_below_activation_restores_original_stop(db: str) -> None:
    """(b) Same nakedness but the gain is below activation → the heal
    restores the position's ORIGINAL entry-time stop price (from the
    journal's entry-time stop row), role='stop', NO state row (the trail
    has not legitimately engaged)."""
    eid, _ = _seed_position(
        db, stop_final_status="canceled", trail_distance_pct=10.0
    )
    broker = _FakeBrokerFull(last=105.0)  # below 1R activation at 110
    broker.positions = [_broker_position("ACME")]
    broker.open_orders = []

    _tick(db, broker)

    assert len(broker.submitted) == 1
    req = broker.submitted[0]
    assert req.stop_price == pytest.approx(90.0)  # the original stop
    assert (req.side, req.order_type, req.tif) == ("sell", "stop", "gtc")
    # No state row — below activation is not an engagement.
    assert get_exit_policy_state(db, execution_id=eid) is None
    restored = [
        o for o in get_orders_for_execution(db, eid)
        if o.role == "stop" and o.final_status is None
    ]
    assert len(restored) == 1
    assert "mode=restored_original" in (restored[0].notes or "")


def test_stateless_engagement_arms_cleanly_and_ratchets_next_tick(db: str) -> None:
    """(c) Engagement without a pre-existing live stop leg arms cleanly:
    the healed stop is a normal ratchet citizen — the next tick PATCHes
    it upward as price advances, and neither heal arm double-places."""
    eid, _ = _seed_position(
        db, stop_final_status="canceled", trail_distance_pct=10.0
    )
    broker = _FakeBrokerFull(last=126.0)
    broker.positions = [_broker_position("ACME")]
    broker.open_orders = []

    _tick(db, broker)  # heal-engage: trailing stop at 113.4, HWM 126
    assert len(broker.submitted) == 1

    broker.default_last = 140.0  # price advances
    _tick(db, broker)

    # Normal ratchet PATCH: 140 × 0.9 = 126.0 — no second fresh submit.
    assert len(broker.submitted) == 1
    assert len(broker.replaced) == 1
    assert broker.replaced[0]["new_stop_price"] == pytest.approx(126.0)
    state = get_exit_policy_state(db, execution_id=eid)
    assert state is not None
    assert state.high_water_mark == pytest.approx(140.0)


def test_stateless_heal_skips_position_covered_by_broker_sell_order(
    db: str,
) -> None:
    """A live sell order at the broker (protective leg or a pending
    close) means the position is not naked — the stateless arm must not
    place a second full-qty stop beside it (double-sell)."""
    _seed_position(db, stop_final_status="canceled", trail_distance_pct=10.0)
    broker = _FakeBrokerFull(last=126.0)
    broker.positions = [_broker_position("ACME")]
    broker.open_orders = [_broker_sell_order("ACME")]

    _tick(db, broker)

    assert broker.submitted == []


def test_stateless_heal_skips_when_live_journal_sell_exists(db: str) -> None:
    """The journal guard covers the submit-vs-open-orders visibility
    race: a live journaled sell (e.g. a thesis_close submitted this
    tick, not yet visible in the broker's open-orders read) suppresses
    the heal."""
    eid, _ = _seed_position(
        db, stop_final_status="canceled", trail_distance_pct=10.0
    )
    insert_order(
        db,
        OrderRow(
            execution_id=eid,
            role="thesis_close",
            symbol="ACME",
            side="sell",
            order_type="market",
            qty=100,
            tif="day",
            broker_order_id="close-in-flight",
            submitted_at=NOW,
            final_status=None,
        ),
    )
    broker = _FakeBrokerFull(last=126.0)
    broker.positions = [_broker_position("ACME")]
    broker.open_orders = []  # stale broker view — sell not visible yet

    _tick(db, broker)

    assert broker.submitted == []


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

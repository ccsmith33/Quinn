"""Capacity-pressure sweep — EventReviewsConfig.capacity_target_slots.

When the portfolio is near its KS-5 effective concurrent-position cap,
every thesis review that fires during a sweep receives an extra context
block ranking the open positions by weakness and biasing the reviewer
toward freeing the weakest NON-EXEMPT slots. These tests cover the slot
math (effective cap + pending entries), the injection gate (block only
under pressure), the ranking order/contents, and that the block reaches
the review context end-to-end.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest

from analyzer.thesis_review import ThesisHold, ThesisReviewContext
from broker.protocol import AccountSnapshot, OpenOrder, Position, Quote
from config.loader import ExecutionConfig, Ks5Tier
from journal.exit_policy import ExitPolicyStateRow, upsert_exit_policy_state
from journal.migrate import apply_migrations
from journal.models import (
    ExecutionRow,
    FilingRow,
    OrderRow,
    PositionRow,
    PromptRow,
    ProposalRow,
    ThesisReviewScheduleRow,
)
from journal.repo import (
    insert_execution,
    insert_filing,
    insert_order,
    insert_position,
    insert_prompt,
    insert_proposal,
    insert_thesis_review_schedule,
)

NOW = dt.datetime(2026, 7, 30, 13, 0, 0, tzinfo=dt.UTC)


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path: Path) -> str:
    p = tmp_path / "journal.db"
    apply_migrations(str(p))
    return str(p)


class _Journal:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path


class _FakeBroker:
    def __init__(
        self,
        *,
        equity: float,
        positions: list[Position],
        open_orders: list[OpenOrder] | None = None,
        last_price: float = 100.0,
    ) -> None:
        self._equity = equity
        self._positions = positions
        self._open_orders = open_orders or []
        self._last_price = last_price

    def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(
            equity=self._equity,
            cash=self._equity * 0.5,
            buying_power=self._equity * 0.5,
            long_market_value=self._equity * 0.5,
            daypl=0.0,
            snapshot_at=NOW,
        )

    def get_positions(self) -> list[Position]:
        return list(self._positions)

    def get_open_orders(self) -> list[OpenOrder]:
        return list(self._open_orders)

    def get_quote(self, symbol: str) -> Quote:
        return Quote(
            symbol=symbol,
            bid=self._last_price - 0.01,
            ask=self._last_price + 0.01,
            last=self._last_price,
            ts=NOW,
        )


class _FakeBrokerNoOpenOrders(_FakeBroker):
    """Pre-WS1 adapter shape — get_open_orders resolves to None so the
    coordinator's `getattr(..., None)` guard treats it as absent."""

    get_open_orders = None  # type: ignore[assignment]


class _CapturingReviewer:
    """Stands in for ThesisReviewer — records each context it is asked to
    review and returns a benign `hold`."""

    def __init__(self) -> None:
        self.contexts: list[ThesisReviewContext] = []

    async def review(self, ctx: ThesisReviewContext) -> ThesisHold:
        self.contexts.append(ctx)
        return ThesisHold(rationale="x" * 60)


def _position(symbol: str, *, qty: int, avg: float, unrealized: float) -> Position:
    return Position(
        symbol=symbol,
        qty=qty,
        avg_entry_price=avg,
        market_value=qty * avg + unrealized,
        unrealized_pnl=unrealized,
    )


def _pending_entry(symbol: str) -> OpenOrder:
    return OpenOrder(
        symbol=symbol,
        side="buy",
        qty=10,
        order_type="market",
        status="new",
        broker_order_id=f"boid-{symbol}",
        client_order_id=f"prop-{symbol}-entry",
    )


def _exec_cfg(**overrides: Any) -> ExecutionConfig:
    base: dict[str, Any] = dict(
        broker_mode="paper",
        ks4_pct_cap=0.15,
        ks4_absolute_cap_usd=1000.0,
        ks5_max_concurrent=5,
        ks7_cash_reserve_pct=0.03,
        sizing_mid_pct=0.07,
        sizing_high_pct=0.10,
    )
    base.update(overrides)
    return ExecutionConfig(**base)


def _seed_entry(
    db: str,
    *,
    symbol: str,
    conviction: int,
    entry_at: dt.datetime,
    trail_engaged: bool = False,
    stop: float = 90.0,
    tp: float | None = 150.0,
    horizon_days: int = 14,
    fill_price: float = 100.0,
    qty: int = 100,
) -> int:
    """Insert filing → proposal → execution(accepted) → entry order →
    position snapshot (+ optional exit_policy_state). Returns execution_id.
    The proposal.raw_response is a valid trade payload so the coordinator
    can build a review context from it.
    """
    pv = f"pv-{symbol}@aabbccddeeff"
    insert_prompt(
        db,
        PromptRow(
            prompt_version=pv,
            name=f"prompt-{symbol}",
            file_path="/tmp/x",
            content_hash=f"hash-{symbol}",
        ),
    )
    fid = insert_filing(
        db,
        FilingRow(
            accession_number=f"acc-{symbol}",
            cik=320193,
            form_type="8-K",
            filed_at=entry_at - dt.timedelta(hours=2),
            fetched_at=entry_at - dt.timedelta(hours=1),
            raw_text_path=f"/tmp/{symbol}.txt",
            content_hash=f"fhash-{symbol}",
            item_codes='["1.01"]',
            issuer_ticker=symbol,
        ),
    )
    payload = {
        "symbol": symbol,
        "direction": "long",
        "size_pct_of_capital": 0.10,
        "entry_style": "market_open",
        "stop_loss_price": stop,
        "take_profit_price": tp,
        "time_horizon_days": horizon_days,
        "conviction": conviction,
        "thesis": f"{symbol} thesis: material acquisition with concrete terms.",
        "signals": ["Item 1.01 — Material Definitive Agreement"],
        "exit_conditions": ["Exit on contradicting filing"],
        "risk_factors": ["Closing conditions not yet met"],
    }
    pid = insert_proposal(
        db,
        ProposalRow(
            filing_id=fid,
            decision_id=f"dec-{symbol}",
            model_id="claude-sonnet-4-6",
            prompt_version=pv,
            raw_response=json.dumps(payload),
            kind="trade_proposal",
            symbol=symbol,
            direction="long",
            size_pct_requested=0.10,
            conviction=conviction,
            thesis=payload["thesis"],
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
            realized_dollar_size=fill_price * qty,
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
            tif="day",
            broker_order_id=f"entry-{symbol}",
            submitted_at=entry_at,
            realized_fill_price=fill_price,
            final_status="filled",
        ),
    )
    insert_position(
        db,
        PositionRow(
            snapshot_at=entry_at,
            source="broker",
            symbol=symbol,
            qty=qty,
            avg_entry_price=fill_price,
            market_value=fill_price * qty,
            unrealized_pnl=0.0,
        ),
    )
    if trail_engaged:
        upsert_exit_policy_state(
            db,
            ExitPolicyStateRow(
                execution_id=eid,
                symbol=symbol,
                trail_distance_pct=0.05,
                trail_engaged=True,
                high_water_mark=fill_price * 1.2,
                stop_order_journal_id=None,
            ),
        )
    return eid


def _coordinator(
    db: str,
    broker: _FakeBroker,
    reviewer: Any,
    *,
    capacity_target_slots: int,
    execution_config: ExecutionConfig | None,
) -> Any:
    from app.thesis_coordinator import ThesisReviewCoordinator

    return ThesisReviewCoordinator(
        journal=_Journal(db),
        broker=broker,
        reviewer=reviewer,
        filings_lookup=lambda *, issuer_ticker, since: "(no filings since entry)",
        now_fn=lambda: NOW,
        execution_config=execution_config,
        capacity_target_slots=capacity_target_slots,
    )


# ---------------------------------------------------------------------------
# Slot math + injection gate
# ---------------------------------------------------------------------------


def test_block_none_when_feature_off(db: str) -> None:
    """capacity_target_slots=0 → feature off → never compute a block even
    if the book is jammed full."""
    positions = [
        _position(f"S{i}", qty=100, avg=100.0, unrealized=-500.0) for i in range(5)
    ]
    broker = _FakeBroker(equity=100_000.0, positions=positions)
    coord = _coordinator(
        db, broker, _CapturingReviewer(),
        capacity_target_slots=0, execution_config=_exec_cfg(ks5_max_concurrent=5),
    )
    assert coord._compute_capacity_pressure_block(NOW) is None


def test_block_none_when_execution_config_missing(db: str) -> None:
    """No ExecutionConfig wired (legacy construction) → no cap curve → no
    block, even with a positive target."""
    broker = _FakeBroker(equity=100_000.0, positions=[])
    coord = _coordinator(
        db, broker, _CapturingReviewer(),
        capacity_target_slots=2, execution_config=None,
    )
    assert coord._compute_capacity_pressure_block(NOW) is None


def test_block_absent_when_slots_available(db: str) -> None:
    """open_slots >= target → context unchanged (no block). Cap 5, two
    held → 3 free >= target 2."""
    for i in range(2):
        _seed_entry(db, symbol=f"S{i}", conviction=8, entry_at=NOW - dt.timedelta(days=20))
    positions = [_position(f"S{i}", qty=100, avg=100.0, unrealized=100.0) for i in range(2)]
    broker = _FakeBroker(equity=100_000.0, positions=positions)
    coord = _coordinator(
        db, broker, _CapturingReviewer(),
        capacity_target_slots=2, execution_config=_exec_cfg(ks5_max_concurrent=5),
    )
    assert coord._compute_capacity_pressure_block(NOW) is None


def test_block_present_under_pressure(db: str) -> None:
    """open_slots < target → block injected. Cap 5, four held → 1 free < 2."""
    for i in range(4):
        _seed_entry(db, symbol=f"S{i}", conviction=8, entry_at=NOW - dt.timedelta(days=20))
    positions = [_position(f"S{i}", qty=100, avg=100.0, unrealized=-100.0) for i in range(4)]
    broker = _FakeBroker(equity=100_000.0, positions=positions)
    coord = _coordinator(
        db, broker, _CapturingReviewer(),
        capacity_target_slots=2, execution_config=_exec_cfg(ks5_max_concurrent=5),
    )
    block = coord._compute_capacity_pressure_block(NOW)
    assert block is not None
    assert "# Capacity pressure (daily sweep)" in block
    assert "open_slots: 1 (target 2)" in block
    # All four open positions appear in the ranking table.
    for i in range(4):
        assert f"| S{i} |" in block


def test_pending_entries_consume_slots(db: str) -> None:
    """A queued (unfilled) entry BUY counts against capacity exactly as
    sizing.py counts it: cap 5, three held + two pending → 0 free < 2."""
    for i in range(3):
        _seed_entry(db, symbol=f"H{i}", conviction=8, entry_at=NOW - dt.timedelta(days=10))
    positions = [_position(f"H{i}", qty=100, avg=100.0, unrealized=0.0) for i in range(3)]
    open_orders = [_pending_entry("P0"), _pending_entry("P1")]
    broker = _FakeBroker(equity=100_000.0, positions=positions, open_orders=open_orders)
    coord = _coordinator(
        db, broker, _CapturingReviewer(),
        capacity_target_slots=2, execution_config=_exec_cfg(ks5_max_concurrent=5),
    )
    block = coord._compute_capacity_pressure_block(NOW)
    assert block is not None
    assert "open_slots: 0 (target 2)" in block


def test_effective_cap_uses_ks5_tiers(db: str) -> None:
    """The cap is the equity-tiered effective cap, not the raw ceiling. A
    small-equity account bands down to 3 positions; 2 held → 1 free < 2,
    so the block fires even though the raw ceiling (5) would leave 3 free.
    """
    for i in range(2):
        _seed_entry(db, symbol=f"T{i}", conviction=7, entry_at=NOW - dt.timedelta(days=8))
    positions = [_position(f"T{i}", qty=100, avg=100.0, unrealized=0.0) for i in range(2)]
    broker = _FakeBroker(equity=20_000.0, positions=positions)
    cfg = _exec_cfg(
        ks5_max_concurrent=5,
        ks5_tiers=[Ks5Tier(equity_max=25_000.0, max_positions=3)],
    )
    coord = _coordinator(
        db, broker, _CapturingReviewer(),
        capacity_target_slots=2, execution_config=cfg,
    )
    block = coord._compute_capacity_pressure_block(NOW)
    assert block is not None
    assert "open_slots: 1 (target 2)" in block
    assert "of an effective 3-position cap" in block


def test_missing_get_open_orders_degrades_gracefully(db: str) -> None:
    """A broker adapter without get_open_orders() must not crash the
    sweep — pending entries are simply treated as empty."""
    for i in range(4):
        _seed_entry(db, symbol=f"S{i}", conviction=8, entry_at=NOW - dt.timedelta(days=20))
    positions = [_position(f"S{i}", qty=100, avg=100.0, unrealized=0.0) for i in range(4)]
    broker = _FakeBrokerNoOpenOrders(equity=100_000.0, positions=positions)
    coord = _coordinator(
        db, broker, _CapturingReviewer(),
        capacity_target_slots=2, execution_config=_exec_cfg(ks5_max_concurrent=5),
    )
    block = coord._compute_capacity_pressure_block(NOW)
    assert block is not None
    assert "open_slots: 1 (target 2)" in block


# ---------------------------------------------------------------------------
# Ranking order + contents + exemptions
# ---------------------------------------------------------------------------


def test_ranking_weakest_first_lowest_gain_then_longest_held(db: str) -> None:
    """Weakest first: lowest unrealized gain %, ties broken by longest
    held. WEAK (-8%) < MIDLONG (+2%, 30d) < MIDSHORT (+2%, 10d) < STRONG
    (+15%)."""
    _seed_entry(db, symbol="WEAK", conviction=6, entry_at=NOW - dt.timedelta(days=15))
    _seed_entry(db, symbol="MIDLONG", conviction=7, entry_at=NOW - dt.timedelta(days=30))
    _seed_entry(db, symbol="MIDSHORT", conviction=7, entry_at=NOW - dt.timedelta(days=10))
    _seed_entry(db, symbol="STRONG", conviction=9, entry_at=NOW - dt.timedelta(days=15))
    positions = [
        _position("STRONG", qty=100, avg=100.0, unrealized=1500.0),  # +15%
        _position("MIDSHORT", qty=100, avg=100.0, unrealized=200.0),  # +2%, 10d
        _position("WEAK", qty=100, avg=100.0, unrealized=-800.0),  # -8%
        _position("MIDLONG", qty=100, avg=100.0, unrealized=200.0),  # +2%, 30d
    ]
    broker = _FakeBroker(equity=100_000.0, positions=positions)
    coord = _coordinator(
        db, broker, _CapturingReviewer(),
        capacity_target_slots=5, execution_config=_exec_cfg(ks5_max_concurrent=4),
    )
    rows = coord._rank_open_positions(positions, NOW)
    assert [r.symbol for r in rows] == ["WEAK", "MIDLONG", "MIDSHORT", "STRONG"]


def test_ranking_table_columns_and_exemptions_text(db: str) -> None:
    """The table carries symbol / days held / gain % / trail armed /
    conviction, and an exempt (trail-armed) position is still listed while
    the binding exemptions text stays intact."""
    _seed_entry(
        db, symbol="ARMED", conviction=8,
        entry_at=NOW - dt.timedelta(days=12), trail_engaged=True,
    )
    _seed_entry(db, symbol="PLAIN", conviction=6, entry_at=NOW - dt.timedelta(days=25))
    positions = [
        _position("ARMED", qty=100, avg=100.0, unrealized=900.0),
        _position("PLAIN", qty=100, avg=100.0, unrealized=-300.0),
    ]
    broker = _FakeBroker(equity=100_000.0, positions=positions)
    coord = _coordinator(
        db, broker, _CapturingReviewer(),
        capacity_target_slots=5, execution_config=_exec_cfg(ks5_max_concurrent=2),
    )
    block = coord._compute_capacity_pressure_block(NOW)
    assert block is not None
    # Header columns.
    assert "| rank | symbol | days held | unrealized gain % | trail armed | conviction |" in block
    # Exempt position listed with trail armed = yes and its calendar days.
    assert "| ARMED | 12 |" in block
    assert "| yes |" in block  # ARMED row shows trail armed
    # Weakest-first: PLAIN (-3%) ranks above ARMED (+9%).
    plain_idx = block.index("| PLAIN |")
    armed_idx = block.index("| ARMED |")
    assert plain_idx < armed_idx
    # Points at the prompt's own doctrine + exemptions (does not restate).
    assert "Dead money and opportunity cost" in block
    assert "Its Exemptions still bind in full" in block
    assert "ARMED trailing stop" in block
    assert "DATED catalyst still ahead" in block
    assert "fewer than ~5 trading days held" in block
    # Conviction surfaced.
    assert "| 8 |" in block  # ARMED conviction
    assert "| 6 |" in block  # PLAIN conviction


# ---------------------------------------------------------------------------
# End-to-end: block reaches the review context via run_tick
# ---------------------------------------------------------------------------


def _schedule(db: str, eid: int, reason: str) -> None:
    insert_thesis_review_schedule(
        db,
        ThesisReviewScheduleRow(
            execution_id=eid,
            due_at=NOW - dt.timedelta(hours=1),
            scheduled_reason=reason,
        ),
    )


@pytest.mark.asyncio
async def test_run_tick_threads_block_into_daily_sweep_reviews(db: str) -> None:
    """Under pressure, every due `event:daily_sweep` review this tick
    carries the block computed once for the tick."""
    eids = [
        _seed_entry(db, symbol=f"S{i}", conviction=8, entry_at=NOW - dt.timedelta(days=20))
        for i in range(4)
    ]
    for eid in eids:
        _schedule(db, eid, "event:daily_sweep")
    positions = [_position(f"S{i}", qty=100, avg=100.0, unrealized=-100.0) for i in range(4)]
    broker = _FakeBroker(equity=100_000.0, positions=positions, last_price=99.0)
    reviewer = _CapturingReviewer()
    coord = _coordinator(
        db, broker, reviewer,
        capacity_target_slots=2, execution_config=_exec_cfg(ks5_max_concurrent=5),
    )
    await coord.run_tick()
    assert len(reviewer.contexts) == 4
    for ctx in reviewer.contexts:
        assert ctx.capacity_pressure_block is not None
        assert "# Capacity pressure (daily sweep)" in ctx.capacity_pressure_block


@pytest.mark.asyncio
async def test_run_tick_block_only_on_daily_sweep_not_other_reviews(db: str) -> None:
    """The block rides ONLY `event:daily_sweep`. A calendar/hold review and
    a non-sweep event review in the SAME under-pressure tick get no block,
    even though the sweep review beside them does."""
    sweep_eid = _seed_entry(db, symbol="SWEEP", conviction=6, entry_at=NOW - dt.timedelta(days=20))
    hold_eid = _seed_entry(db, symbol="HOLD", conviction=7, entry_at=NOW - dt.timedelta(days=20))
    gain_eid = _seed_entry(db, symbol="GAIN", conviction=8, entry_at=NOW - dt.timedelta(days=20))
    _schedule(db, sweep_eid, "event:daily_sweep")
    _schedule(db, hold_eid, "hold")
    _schedule(db, gain_eid, "event:gain_cross:10")
    positions = [
        _position("SWEEP", qty=100, avg=100.0, unrealized=-100.0),
        _position("HOLD", qty=100, avg=100.0, unrealized=-100.0),
        _position("GAIN", qty=100, avg=100.0, unrealized=-100.0),
    ]
    broker = _FakeBroker(equity=100_000.0, positions=positions, last_price=99.0)
    reviewer = _CapturingReviewer()
    coord = _coordinator(
        db, broker, reviewer,
        capacity_target_slots=2, execution_config=_exec_cfg(ks5_max_concurrent=3),
    )
    await coord.run_tick()
    by_symbol = {ctx.proposal.symbol: ctx for ctx in reviewer.contexts}
    assert by_symbol["SWEEP"].capacity_pressure_block is not None
    assert by_symbol["HOLD"].capacity_pressure_block is None
    assert by_symbol["GAIN"].capacity_pressure_block is None


@pytest.mark.asyncio
async def test_run_tick_no_block_computed_without_a_due_sweep(db: str) -> None:
    """No `event:daily_sweep` due this tick → the block is never computed,
    so even under pressure a calendar review carries none."""
    eid = _seed_entry(db, symbol="ONLY", conviction=8, entry_at=NOW - dt.timedelta(days=20))
    _schedule(db, eid, "hold")
    # Under pressure (cap 1, one held → 0 free < target 2) — but no sweep.
    positions = [_position("ONLY", qty=100, avg=100.0, unrealized=-100.0)]
    broker = _FakeBroker(equity=100_000.0, positions=positions, last_price=99.0)
    reviewer = _CapturingReviewer()
    coord = _coordinator(
        db, broker, reviewer,
        capacity_target_slots=2, execution_config=_exec_cfg(ks5_max_concurrent=1),
    )
    await coord.run_tick()
    assert len(reviewer.contexts) == 1
    assert reviewer.contexts[0].capacity_pressure_block is None


@pytest.mark.asyncio
async def test_run_tick_no_block_when_not_under_pressure(db: str) -> None:
    """A daily-sweep review at/above target free slots carries no block."""
    eid = _seed_entry(db, symbol="ONLY", conviction=8, entry_at=NOW - dt.timedelta(days=20))
    _schedule(db, eid, "event:daily_sweep")
    positions = [_position("ONLY", qty=100, avg=100.0, unrealized=50.0)]
    broker = _FakeBroker(equity=100_000.0, positions=positions, last_price=100.5)
    reviewer = _CapturingReviewer()
    coord = _coordinator(
        db, broker, reviewer,
        capacity_target_slots=2, execution_config=_exec_cfg(ks5_max_concurrent=5),
    )
    await coord.run_tick()
    assert len(reviewer.contexts) == 1
    assert reviewer.contexts[0].capacity_pressure_block is None

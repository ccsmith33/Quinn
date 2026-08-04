"""Chopping block — persistence + ranking after the daily pre-market sweep.

Drives `ThesisReviewCoordinator.run_tick()` over a set of `event:daily_sweep`
schedules with a reviewer that returns per-symbol expendability scores, and
asserts the `chopping_block` table is written with the correct rank ordering
(expendability DESC, unrealized gain ASC, days-held DESC), that ARMED
positions are excluded entirely, that positions whose review carried no
expendability are excluded, and that the table is rewritten (not duplicated)
on a re-sweep.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from pathlib import Path

import pytest

from analyzer.thesis_review import ThesisHold, ThesisReviewContext
from broker.protocol import AccountSnapshot, Position, Quote
from journal.exit_policy import ExitPolicyStateRow, upsert_exit_policy_state
from journal.migrate import apply_migrations
from journal.models import (
    ChoppingBlockRow,
    ExecutionRow,
    FilingRow,
    OrderRow,
    PositionRow,
    PromptRow,
    ProposalRow,
    ThesisReviewScheduleRow,
)
from journal.repo import (
    get_latest_chopping_block,
    insert_execution,
    insert_filing,
    insert_order,
    insert_position,
    insert_prompt,
    insert_proposal,
    insert_thesis_review_schedule,
    replace_chopping_block,
)

# 2026-08-03 15:00 UTC = 11:00 ET — inside the pre-market sweep window.
NOW = dt.datetime(2026, 8, 3, 15, 0, 0, tzinfo=dt.UTC)


@pytest.fixture
def db(tmp_path: Path) -> str:
    p = tmp_path / "journal.db"
    apply_migrations(str(p))
    return str(p)


class _Journal:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path


class _FakeBroker:
    def __init__(self, *, prices: dict[str, float], positions: list[Position]) -> None:
        self._prices = prices
        self._positions = positions

    def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(
            equity=100_000.0, cash=50_000.0, buying_power=50_000.0,
            long_market_value=50_000.0, daypl=0.0, snapshot_at=NOW,
        )

    def get_positions(self) -> list[Position]:
        return list(self._positions)

    def get_quote(self, symbol: str) -> Quote:
        last = self._prices[symbol]
        return Quote(symbol=symbol, bid=last - 0.01, ask=last + 0.01, last=last, ts=NOW)


class _Reviewer:
    """Returns a benign hold carrying a per-symbol expendability score
    (None → the model omitted it for that symbol)."""

    def __init__(self, exp_by_symbol: dict[str, int | None]) -> None:
        self._exp = exp_by_symbol

    async def review(self, ctx: ThesisReviewContext) -> ThesisHold:
        sym = ctx.proposal.symbol
        return ThesisHold(
            rationale="x" * 60,
            expendability=self._exp.get(sym),
            expendability_reason=f"reason-{sym}",
        )


def _seed(
    db: str,
    *,
    symbol: str,
    fill_price: float,
    entry_at: dt.datetime,
    armed: bool = False,
) -> int:
    """filing → prompt → proposal(valid payload) → accepted execution →
    filled entry → position snapshot (+ optional armed trail) → due
    daily-sweep schedule. Returns execution_id."""
    pv = f"pv-{symbol}@aabbccddeeff"
    try:
        insert_prompt(
            db,
            PromptRow(prompt_version=pv, name=f"p-{symbol}",
                      file_path="/tmp/x", content_hash="h" * 64),
        )
    except Exception:  # noqa: BLE001 — prompt row already registered
        pass
    fid = insert_filing(
        db,
        FilingRow(
            accession_number=f"acc-{symbol}", cik=hash(symbol) % 999_999,
            form_type="8-K", filed_at=entry_at - dt.timedelta(hours=2),
            fetched_at=entry_at - dt.timedelta(hours=1),
            raw_text_path=f"/tmp/{symbol}.txt", content_hash=f"fh-{symbol}",
            item_codes='["1.01"]', issuer_ticker=symbol,
        ),
    )
    payload = {
        "symbol": symbol, "direction": "long", "size_pct_of_capital": 0.10,
        "entry_style": "market_open", "stop_loss_price": fill_price * 0.9,
        "take_profit_price": fill_price * 1.5, "time_horizon_days": 14,
        "conviction": 8,
        "thesis": (
            f"{symbol} thesis: material acquisition with concrete deal "
            "terms and a near-term close."
        ),
        "signals": ["Item 1.01 — Material Definitive Agreement"],
        "exit_conditions": ["Exit on a contradicting subsequent filing"],
        "risk_factors": ["Closing conditions are not yet fully met"],
    }
    pid = insert_proposal(
        db,
        ProposalRow(
            filing_id=fid, decision_id=f"dec-{symbol}", model_id="claude-sonnet-4-6",
            prompt_version=pv, raw_response=json.dumps(payload), kind="trade_proposal",
            symbol=symbol, direction="long", size_pct_requested=0.10, conviction=8,
            thesis=payload["thesis"], input_tokens=1, output_tokens=1,
            latency_ms=1, cost_usd=0.0,
        ),
    )
    eid = insert_execution(
        db,
        ExecutionRow(proposal_id=pid, decision="accepted", realized_size_pct=0.10,
                     realized_dollar_size=fill_price * 100, submitted_orders_json="[]"),
    )
    insert_order(
        db,
        OrderRow(
            execution_id=eid, role="entry", symbol=symbol, side="buy",
            order_type="market", qty=100, tif="day", broker_order_id=f"entry-{symbol}",
            submitted_at=entry_at, realized_fill_price=fill_price, final_status="filled",
        ),
    )
    insert_position(
        db,
        PositionRow(snapshot_at=entry_at, source="broker", symbol=symbol, qty=100,
                    avg_entry_price=fill_price, market_value=fill_price * 100,
                    unrealized_pnl=0.0),
    )
    if armed:
        upsert_exit_policy_state(
            db,
            ExitPolicyStateRow(execution_id=eid, symbol=symbol, trail_distance_pct=0.05,
                               trail_engaged=True, high_water_mark=fill_price * 1.3),
        )
    insert_thesis_review_schedule(
        db,
        ThesisReviewScheduleRow(execution_id=eid, due_at=NOW,
                                scheduled_reason="event:daily_sweep"),
    )
    return eid


def _coordinator(db: str, broker: _FakeBroker, reviewer: _Reviewer):
    from app.thesis_coordinator import ThesisReviewCoordinator

    return ThesisReviewCoordinator(
        journal=_Journal(db),
        broker=broker,
        reviewer=reviewer,
        filings_lookup=lambda *, issuer_ticker, since: "(no filings since entry)",
        now_fn=lambda: NOW,
        execution_config=None,
        capacity_target_slots=0,
    )


def _pos(symbol: str, price: float) -> Position:
    return Position(symbol=symbol, qty=100, avg_entry_price=100.0,
                    market_value=price * 100, unrealized_pnl=(price - 100.0) * 100)


def test_ranking_order_and_armed_and_missing_exclusion(db: str) -> None:
    # fill=100 for all; gain = price/100 - 1.
    #   BBB exp5 gain -0.10   AAA exp5 gain -0.02   (tie exp → gain ASC)
    #   EEE exp4 gain -0.05 held 20d   FFF exp4 gain -0.05 held 5d (tie → days DESC)
    #   CCC exp3 gain -0.50
    #   ARM exp5 but ARMED → excluded ; NOX no expendability → excluded
    seeds = {
        "AAA": (98.0, NOW - dt.timedelta(days=10), False),
        "BBB": (90.0, NOW - dt.timedelta(days=10), False),
        "CCC": (50.0, NOW - dt.timedelta(days=10), False),
        "EEE": (95.0, NOW - dt.timedelta(days=20), False),
        "FFF": (95.0, NOW - dt.timedelta(days=5), False),
        "ARM": (90.0, NOW - dt.timedelta(days=10), True),
        "NOX": (30.0, NOW - dt.timedelta(days=10), False),
    }
    for sym, (_price, entry, armed) in seeds.items():
        _seed(db, symbol=sym, fill_price=100.0, entry_at=entry, armed=armed)
    prices = {sym: price for sym, (price, _, _) in seeds.items()}
    positions = [_pos(sym, price) for sym, (price, _, _) in seeds.items()]
    exp = {"AAA": 5, "BBB": 5, "CCC": 3, "EEE": 4, "FFF": 4, "ARM": 5, "NOX": None}

    coord = _coordinator(db, _FakeBroker(prices=prices, positions=positions), _Reviewer(exp))
    asyncio.run(coord.run_tick())

    block = get_latest_chopping_block(db)
    assert [r.symbol for r in block] == ["BBB", "AAA", "EEE", "FFF", "CCC"]
    assert [r.rank for r in block] == [1, 2, 3, 4, 5]
    assert [r.expendability for r in block] == [5, 5, 4, 4, 3]
    assert all(r.block_date == "2026-08-03" for r in block)
    assert block[0].reason == "reason-BBB"
    # Armed + no-expendability positions never make the block.
    assert "ARM" not in [r.symbol for r in block]
    assert "NOX" not in [r.symbol for r in block]


def test_all_armed_book_writes_empty_block(db: str) -> None:
    _seed(db, symbol="ARM", fill_price=100.0,
          entry_at=NOW - dt.timedelta(days=10), armed=True)
    coord = _coordinator(
        db,
        _FakeBroker(prices={"ARM": 90.0}, positions=[_pos("ARM", 90.0)]),
        _Reviewer({"ARM": 5}),
    )
    asyncio.run(coord.run_tick())
    assert get_latest_chopping_block(db) == []


def test_replace_chopping_block_is_idempotent_per_date(db: str) -> None:
    eid = _seed(db, symbol="AAA", fill_price=100.0, entry_at=NOW - dt.timedelta(days=10))
    rows1 = [ChoppingBlockRow(block_date="2026-08-03", execution_id=eid,
                              symbol="AAA", rank=1, expendability=5, reason="r1")]
    replace_chopping_block(db, "2026-08-03", rows1)
    rows2 = [ChoppingBlockRow(block_date="2026-08-03", execution_id=eid,
                              symbol="AAA", rank=1, expendability=3, reason="r2")]
    replace_chopping_block(db, "2026-08-03", rows2)
    block = get_latest_chopping_block(db)
    assert len(block) == 1  # replaced, not duplicated
    assert block[0].expendability == 3
    assert block[0].reason == "r2"


def test_non_sweep_reviews_write_no_block(db: str) -> None:
    """A non-sweep review (calendar 'hold') never produces a chopping
    block even when the reviewer returns an expendability score."""
    eid = _seed(db, symbol="AAA", fill_price=100.0, entry_at=NOW - dt.timedelta(days=10))
    # Overwrite the sweep schedule with a plain calendar 'hold' review.
    insert_thesis_review_schedule(
        db, ThesisReviewScheduleRow(execution_id=eid, due_at=NOW, scheduled_reason="hold")
    )
    coord = _coordinator(
        db, _FakeBroker(prices={"AAA": 90.0}, positions=[_pos("AAA", 90.0)]),
        _Reviewer({"AAA": 5}),
    )
    asyncio.run(coord.run_tick())
    assert get_latest_chopping_block(db) == []


def test_write_degrades_when_table_missing_and_warns_once(db: str) -> None:
    """A4 — on a DB migrated before this feature (no chopping_block table),
    the sweep write degrades gracefully: no exception, warned once."""
    from app.thesis_coordinator import _ChoppingBlockCandidate
    from journal.repo import connect

    eid = _seed(db, symbol="AAA", fill_price=100.0, entry_at=NOW - dt.timedelta(days=10))
    with connect(db) as conn:
        conn.execute("DROP TABLE chopping_block")

    coord = _coordinator(
        db, _FakeBroker(prices={"AAA": 90.0}, positions=[_pos("AAA", 90.0)]),
        _Reviewer({"AAA": 5}),
    )
    cands = [
        _ChoppingBlockCandidate(
            execution_id=eid, symbol="AAA", expendability=5,
            reason="stale", unrealized_gain_pct=-0.10, days_held=10,
        )
    ]
    # Must not raise, and the once-latch flips.
    coord._write_chopping_block(cands, now=NOW)
    assert coord._chopping_block_unavailable_warned is True
    # A second write is still a no-op (latched) and never raises.
    coord._write_chopping_block(cands, now=NOW)


def test_run_tick_survives_missing_table(db: str) -> None:
    """End-to-end: a full sweep tick completes even when the table is gone
    (the block write is skipped, the reviews still run)."""
    from journal.repo import connect

    _seed(db, symbol="AAA", fill_price=100.0, entry_at=NOW - dt.timedelta(days=10))
    with connect(db) as conn:
        conn.execute("DROP TABLE chopping_block")
    coord = _coordinator(
        db, _FakeBroker(prices={"AAA": 90.0}, positions=[_pos("AAA", 90.0)]),
        _Reviewer({"AAA": 5}),
    )
    asyncio.run(coord.run_tick())  # must not raise
    assert coord._chopping_block_unavailable_warned is True

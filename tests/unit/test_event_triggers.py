"""Event-triggered thesis reviews — trigger framework (Milestone 1).

The engine's only write is `insert_thesis_review_schedule`; every test
asserts against the `thesis_review_schedule` ledger. Anti-churn: each
trigger fires once, respects the per-position-per-trigger-type
cooldown, and never duplicates a review that is already pending/due.
`enabled=false` writes zero rows (the A/B regression guarantee).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from app.event_triggers import EventTriggerEngine
from broker.protocol import Position, Quote
from config.loader import TrailStage
from journal.exit_policy import ExitPolicyStateRow, upsert_exit_policy_state
from journal.migrate import apply_migrations
from journal.models import (
    ExecutionRow,
    FilingRow,
    OrderRow,
    PromptRow,
    ProposalRow,
    ThesisReviewScheduleRow,
)
from journal.repo import (
    connect,
    find_due_thesis_reviews,
    insert_execution,
    insert_filing,
    insert_order,
    insert_prompt,
    insert_proposal,
    insert_thesis_review_schedule,
)

NOW = dt.datetime(2026, 7, 13, 15, 0, 0, tzinfo=dt.UTC)


@pytest.fixture
def db(tmp_path: Path) -> str:
    p = str(tmp_path / "journal.db")
    apply_migrations(p)
    insert_prompt(
        p,
        PromptRow(
            prompt_version="sonnet@evtest",
            name="sonnet_filing_analysis",
            file_path="src/prompts/sonnet_filing_analysis_v2.txt",
            content_hash="x" * 64,
        ),
    )
    return p


class _Journal:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path


class _FakeBroker:
    def __init__(self) -> None:
        self.positions: list[Position] = []
        self.last_by_symbol: dict[str, float] = {}

    def hold(self, symbol: str, qty: int = 100) -> None:
        self.positions.append(
            Position(
                symbol=symbol,
                qty=qty,
                avg_entry_price=100.0,
                market_value=qty * 100.0,
                unrealized_pnl=0.0,
            )
        )

    def get_positions(self) -> list[Position]:
        return list(self.positions)

    def get_quote(self, symbol: str) -> Quote:
        last = self.last_by_symbol.get(symbol, 100.0)
        return Quote(
            symbol=symbol,
            bid=last - 0.01,
            ask=last + 0.01,
            last=last,
            ts=NOW,
        )


def _seed_position(
    db: str,
    *,
    symbol: str = "ACME",
    fill_price: float = 100.0,
    accession: str | None = None,
) -> int:
    """filing -> proposal -> accepted execution -> filled entry order.
    Returns execution_id."""
    fid = insert_filing(
        db,
        FilingRow(
            accession_number=accession or f"ev-acc-{symbol}",
            cik=320193,
            form_type="8-K",
            filed_at=NOW - dt.timedelta(days=3),
            fetched_at=NOW - dt.timedelta(days=3),
            raw_text_path=f"/tmp/ev-{symbol}.txt",
            content_hash=f"evhash-{symbol}",
            item_codes='["1.01"]',
            issuer_ticker=symbol,
        ),
    )
    pid = insert_proposal(
        db,
        ProposalRow(
            filing_id=fid,
            decision_id=f"ev-d-{symbol}",
            model_id="claude-sonnet-4-6",
            prompt_version="sonnet@evtest",
            raw_response=json.dumps({"symbol": symbol}),
            kind="trade_proposal",
            symbol=symbol,
            direction="long",
            size_pct_requested=0.10,
            conviction=8,
            thesis="material catalyst",
            input_tokens=1,
            output_tokens=1,
            cache_read_tokens=0,
            cache_creation_tokens=0,
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
            realized_dollar_size=100 * fill_price,
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
            qty=100,
            tif="day",
            broker_order_id=f"ev-entry-{symbol}",
            submitted_at=NOW - dt.timedelta(days=3),
            realized_fill_price=fill_price,
            realized_fill_qty=100,
            realized_fill_at=NOW - dt.timedelta(days=3),
            final_status="filled",
        ),
    )
    return eid


def _insert_new_filing(db: str, *, symbol: str, accession: str) -> int:
    return insert_filing(
        db,
        FilingRow(
            accession_number=accession,
            cik=320194,
            form_type="8-K",
            filed_at=NOW,
            fetched_at=NOW,
            raw_text_path=f"/tmp/{accession}.txt",
            content_hash=f"hash-{accession}",
            item_codes='["8.01"]',
            issuer_ticker=symbol,
        ),
    )


def _schedule_rows(db: str) -> list[dict]:
    with connect(db) as conn:
        rows = conn.execute(
            "SELECT * FROM thesis_review_schedule ORDER BY id ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def _mark_latest_reviewed(db: str) -> None:
    """Simulate the thesis coordinator consuming the latest due schedule
    row (in production this happens on the SAME reconciler tick — the
    thesis hook runs right after the trigger hook). Without it, the
    engine's pending-and-due guard correctly refuses to stack more event
    rows on an unconsumed review."""
    from journal.models import ThesisReviewRow
    from journal.repo import insert_thesis_review

    rows = _schedule_rows(db)
    assert rows, "no schedule row to consume"
    latest = rows[-1]
    insert_thesis_review(
        db,
        ThesisReviewRow(
            execution_id=latest["execution_id"],
            schedule_id=latest["id"],
            model_id="claude-opus-4-7",
            prompt_version="sonnet@evtest",
            decision="hold",
            raw_response="{}",
            rationale="test-consumed",
            modifications_json=None,
            input_tokens=1,
            output_tokens=1,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            latency_ms=1,
            cost_usd=0.0,
        ),
    )


def _engine(
    db: str,
    broker: _FakeBroker,
    *,
    enabled: bool = True,
    now: dt.datetime = NOW,
    prev_close: dict[str, float] | None = None,
    trail_stages: tuple[TrailStage, ...] = (),
    gain_thresholds_pct: tuple[float, ...] = (),
    cooldown_hours: float = 24.0,
    anomaly_move_pct: float = 7.0,
) -> EventTriggerEngine:
    clock = {"now": now}
    engine = EventTriggerEngine(
        journal=_Journal(db),
        broker=broker,
        enabled=enabled,
        anomaly_move_pct=anomaly_move_pct,
        cooldown_hours=cooldown_hours,
        gain_thresholds_pct=gain_thresholds_pct,
        trail_stages=trail_stages,
        prev_close_lookup=(prev_close or {}).get,
        now_fn=lambda: clock["now"],
    )
    engine._clock = clock  # type: ignore[attr-defined] — test handle
    return engine


def _advance(engine: EventTriggerEngine, delta: dt.timedelta) -> None:
    engine._clock["now"] = engine._clock["now"] + delta  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Regression guarantee: disabled = zero rows
# ---------------------------------------------------------------------------


def test_disabled_engine_schedules_nothing(db: str) -> None:
    """enabled=false must be byte-identical to no engine at all: even
    with every trigger condition true simultaneously, zero schedule rows
    are written across multiple ticks."""
    broker = _FakeBroker()
    broker.hold("ACME")
    broker.last_by_symbol["ACME"] = 130.0  # +30% gain AND +30% vs prev close
    _seed_position(db, symbol="ACME")
    engine = _engine(
        db, broker, enabled=False, prev_close={"ACME": 100.0}
    )
    engine.run_tick()
    _insert_new_filing(db, symbol="ACME", accession="ev-new-1")
    engine.run_tick()
    assert _schedule_rows(db) == []


# ---------------------------------------------------------------------------
# Gain-cross
# ---------------------------------------------------------------------------


def test_gain_cross_fires_once_at_arming_threshold(db: str) -> None:
    broker = _FakeBroker()
    broker.hold("ACME")
    broker.last_by_symbol["ACME"] = 112.0  # +12% vs 100 entry
    _seed_position(db, symbol="ACME")
    engine = _engine(db, broker, prev_close={"ACME": 111.0})  # no anomaly
    engine.run_tick()
    rows = _schedule_rows(db)
    assert len(rows) == 1
    assert rows[0]["scheduled_reason"] == "event:gain_cross:10"
    assert rows[0]["execution_id"] == 1
    # Same conditions again: threshold already fired -> no duplicate.
    engine.run_tick()
    assert len(_schedule_rows(db)) == 1


def test_gain_cross_next_threshold_respects_cooldown(db: str) -> None:
    stages = (
        TrailStage(gain_pct=20.0, trail_pct=8.0),
        TrailStage(gain_pct=35.0, trail_pct=5.0),
    )
    broker = _FakeBroker()
    broker.hold("ACME")
    broker.last_by_symbol["ACME"] = 112.0
    _seed_position(db, symbol="ACME")
    # prev_close 118 keeps BOTH prices (112 and 125) inside the 7%
    # anomaly band so only gain-cross is in play here.
    engine = _engine(
        db, broker, trail_stages=stages, prev_close={"ACME": 118.0}
    )
    engine.run_tick()  # fires 10
    assert [r["scheduled_reason"] for r in _schedule_rows(db)] == [
        "event:gain_cross:10"
    ]
    _mark_latest_reviewed(db)  # coordinator consumes it same tick
    # Crosses 20 an hour later — suppressed by the 24h gain_cross cooldown.
    broker.last_by_symbol["ACME"] = 125.0
    _advance(engine, dt.timedelta(hours=1))
    engine.run_tick()
    assert len(_schedule_rows(db)) == 1
    # After the cooldown the crossing (still standing, HWM never falls)
    # fires exactly once.
    _advance(engine, dt.timedelta(hours=25))
    engine.run_tick()
    reasons = [r["scheduled_reason"] for r in _schedule_rows(db)]
    assert reasons == ["event:gain_cross:10", "event:gain_cross:20"]
    # Even once consumed and past cooldown, an already-fired threshold
    # never re-fires — the ledger remembers per-position-per-threshold.
    _mark_latest_reviewed(db)
    _advance(engine, dt.timedelta(hours=25))
    engine.run_tick()
    assert len(_schedule_rows(db)) == 2


def test_gain_cross_uses_exit_policy_high_water_mark(db: str) -> None:
    """A pulled-back price must not hide the crossing: the HWM from
    exit_policy_state drives the gain math, and only the HIGHEST crossed
    threshold fires (one review, not a barrage)."""
    stages = (TrailStage(gain_pct=20.0, trail_pct=8.0),)
    broker = _FakeBroker()
    broker.hold("ACME")
    broker.last_by_symbol["ACME"] = 101.0  # pulled back to +1%
    eid = _seed_position(db, symbol="ACME")
    upsert_exit_policy_state(
        db,
        ExitPolicyStateRow(
            execution_id=eid,
            symbol="ACME",
            trail_distance_pct=8.0,
            trail_engaged=True,
            high_water_mark=130.0,  # +30% HWM
        ),
    )
    engine = _engine(
        db, broker, trail_stages=stages, prev_close={"ACME": 100.0}
    )
    engine.run_tick()
    rows = _schedule_rows(db)
    assert [r["scheduled_reason"] for r in rows] == ["event:gain_cross:20"]


# ---------------------------------------------------------------------------
# Held-name filing
# ---------------------------------------------------------------------------


def test_filing_trigger_primes_then_fires_once(db: str) -> None:
    broker = _FakeBroker()
    broker.hold("ACME")
    broker.last_by_symbol["ACME"] = 100.0  # flat: no price triggers
    _seed_position(db, symbol="ACME")
    engine = _engine(db, broker, prev_close={"ACME": 100.0})
    # Tick 1 primes the watermark — the historic entry filing must NOT fire.
    engine.run_tick()
    assert _schedule_rows(db) == []
    # A new held-name filing arrives -> exactly one due-now event row.
    _insert_new_filing(db, symbol="ACME", accession="ev-new-8k")
    engine.run_tick()
    rows = _schedule_rows(db)
    assert len(rows) == 1
    assert rows[0]["scheduled_reason"] == "event:filing:ev-new-8k"
    due = find_due_thesis_reviews(db, now=NOW)
    assert [s.scheduled_reason for s in due] == ["event:filing:ev-new-8k"]
    # Re-tick: watermark advanced, reason recorded -> no duplicate.
    engine.run_tick()
    assert len(_schedule_rows(db)) == 1


def test_filing_trigger_cooldown_suppresses_second_filing_same_day(
    db: str,
) -> None:
    broker = _FakeBroker()
    broker.hold("ACME")
    broker.last_by_symbol["ACME"] = 100.0
    _seed_position(db, symbol="ACME")
    engine = _engine(db, broker, prev_close={"ACME": 100.0})
    engine.run_tick()  # prime
    _insert_new_filing(db, symbol="ACME", accession="ev-f1")
    engine.run_tick()
    _mark_latest_reviewed(db)  # coordinator consumes it same tick
    _insert_new_filing(db, symbol="ACME", accession="ev-f2")
    _advance(engine, dt.timedelta(hours=2))
    engine.run_tick()
    assert [r["scheduled_reason"] for r in _schedule_rows(db)] == [
        "event:filing:ev-f1"
    ]
    # Next day a fresh filing fires again.
    _insert_new_filing(db, symbol="ACME", accession="ev-f3")
    _advance(engine, dt.timedelta(hours=23))
    engine.run_tick()
    assert [r["scheduled_reason"] for r in _schedule_rows(db)] == [
        "event:filing:ev-f1",
        "event:filing:ev-f3",
    ]


def test_filing_for_unheld_symbol_ignored(db: str) -> None:
    broker = _FakeBroker()
    broker.hold("ACME")
    broker.last_by_symbol["ACME"] = 100.0
    _seed_position(db, symbol="ACME")
    engine = _engine(db, broker, prev_close={"ACME": 100.0})
    engine.run_tick()  # prime
    _insert_new_filing(db, symbol="OTHR", accession="ev-othr")
    engine.run_tick()
    assert _schedule_rows(db) == []


# ---------------------------------------------------------------------------
# Tape anomaly
# ---------------------------------------------------------------------------


def test_anomaly_fires_once_and_respects_cooldown(db: str) -> None:
    broker = _FakeBroker()
    broker.hold("ACME")
    broker.last_by_symbol["ACME"] = 92.0  # -8% vs prev close 100
    _seed_position(db, symbol="ACME")
    engine = _engine(db, broker, prev_close={"ACME": 100.0})
    engine.run_tick()
    rows = _schedule_rows(db)
    assert [r["scheduled_reason"] for r in rows] == ["event:anomaly"]
    _mark_latest_reviewed(db)  # coordinator consumes it same tick
    # An hour later, still dislocated: cooldown holds.
    _advance(engine, dt.timedelta(hours=1))
    engine.run_tick()
    assert len(_schedule_rows(db)) == 1
    # Next day, still dislocated vs (stale-fixture) prev close: fires again.
    _advance(engine, dt.timedelta(hours=24))
    engine.run_tick()
    assert [r["scheduled_reason"] for r in _schedule_rows(db)] == [
        "event:anomaly",
        "event:anomaly",
    ]


def test_anomaly_below_threshold_does_not_fire(db: str) -> None:
    broker = _FakeBroker()
    broker.hold("ACME")
    broker.last_by_symbol["ACME"] = 105.0  # +5% < 7% threshold
    _seed_position(db, symbol="ACME")
    engine = _engine(db, broker, prev_close={"ACME": 100.0})
    engine.run_tick()
    assert _schedule_rows(db) == []


def test_anomaly_skipped_when_prev_close_unknown(db: str) -> None:
    """A held name missing from the universe snapshot has no prev close —
    the anomaly trigger silently skips it (no crash, no row)."""
    broker = _FakeBroker()
    broker.hold("ACME")
    broker.last_by_symbol["ACME"] = 92.0
    _seed_position(db, symbol="ACME")
    engine = _engine(db, broker, prev_close={})
    engine.run_tick()
    assert _schedule_rows(db) == []


# ---------------------------------------------------------------------------
# Shared guards
# ---------------------------------------------------------------------------


def test_no_event_row_when_review_already_pending_and_due(db: str) -> None:
    """A due, un-fired schedule row means the coordinator reviews this
    position on the current tick anyway — an event row would double-fire
    the review."""
    broker = _FakeBroker()
    broker.hold("ACME")
    broker.last_by_symbol["ACME"] = 92.0  # anomaly condition true
    eid = _seed_position(db, symbol="ACME")
    insert_thesis_review_schedule(
        db,
        ThesisReviewScheduleRow(
            execution_id=eid,
            due_at=NOW - dt.timedelta(hours=1),
            scheduled_reason="entry",
        ),
    )
    engine = _engine(db, broker, prev_close={"ACME": 100.0})
    engine.run_tick()
    rows = _schedule_rows(db)
    assert len(rows) == 1  # only the seeded calendar row
    assert rows[0]["scheduled_reason"] == "entry"


def test_event_row_supersedes_future_calendar_row(db: str) -> None:
    """The key seam: a FUTURE-dated pending calendar row does not block —
    the event row pulls the review forward and becomes the row
    `find_due_thesis_reviews` surfaces (MAX(id) per execution)."""
    broker = _FakeBroker()
    broker.hold("ACME")
    broker.last_by_symbol["ACME"] = 92.0
    eid = _seed_position(db, symbol="ACME")
    insert_thesis_review_schedule(
        db,
        ThesisReviewScheduleRow(
            execution_id=eid,
            due_at=NOW + dt.timedelta(days=7),
            scheduled_reason="entry",
        ),
    )
    engine = _engine(db, broker, prev_close={"ACME": 100.0})
    engine.run_tick()
    rows = _schedule_rows(db)
    assert [r["scheduled_reason"] for r in rows] == ["entry", "event:anomaly"]
    due = find_due_thesis_reviews(db, now=NOW)
    assert len(due) == 1
    assert due[0].scheduled_reason == "event:anomaly"
    assert due[0].execution_id == eid


def test_one_event_row_per_position_per_tick(db: str) -> None:
    """Anomaly + gain-cross both true on the same tick -> exactly one
    event row (the first trigger wins; the review sees the position
    either way)."""
    broker = _FakeBroker()
    broker.hold("ACME")
    broker.last_by_symbol["ACME"] = 130.0  # +30% gain AND +30% vs prev close
    _seed_position(db, symbol="ACME")
    engine = _engine(db, broker, prev_close={"ACME": 100.0})
    engine.run_tick()
    assert len(_schedule_rows(db)) == 1


def test_position_without_journal_execution_is_skipped(db: str) -> None:
    """An operator-manual broker position with no journal trail can't be
    reviewed (nothing to schedule against) — skip without crashing."""
    broker = _FakeBroker()
    broker.hold("MANL")
    broker.last_by_symbol["MANL"] = 92.0
    engine = _engine(db, broker, prev_close={"MANL": 100.0})
    engine.run_tick()
    assert _schedule_rows(db) == []


def test_quote_failure_skips_symbol_without_crashing(db: str) -> None:
    class _QuoteFailBroker(_FakeBroker):
        def get_quote(self, symbol: str) -> Quote:
            raise RuntimeError("simulated quote outage")

    broker = _QuoteFailBroker()
    broker.hold("ACME")
    _seed_position(db, symbol="ACME")
    engine = _engine(db, broker, prev_close={"ACME": 100.0})
    engine.run_tick()  # must not raise
    assert _schedule_rows(db) == []

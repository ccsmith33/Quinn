"""WATCHLIST / deferred entry — agent-loop integration.

Exercises the REAL `AgentLoop` chain (prefilter → analyzer → Opus →
validator → sizer → submitter) with the fake broker at the boundary:

  - enrollment fires ONLY on capital-class rejects (ks5_concurrent_limit
    / ks7_cash_reserve) of approved, high-conviction proposals, with the
    decision-time quote as reference_price;
  - the retry pass runs through the NORMAL execution path (validator +
    sizing + KS gates), respects expiry (trading days), the held-symbol
    skip, and the chase ceiling (exactly AT the limit is allowed);
  - a still-blocked retry leaves the row pending with a once-per-day log;
  - fresh day-0 pipeline work always preempts the watchlist pass;
  - feature off = no table writes and no tick work.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from types import SimpleNamespace

import pytest

from broker.protocol import AccountSnapshot, Position
from config.calendar import ET, is_market_open_day
from config.loader import ExecutionConfig
from journal.models import FilingRow, KillSwitchStateRow, PrefilterDecisionRow
from journal.repo import (
    connect,
    get_pending_watchlist,
    insert_kill_switch_state,
    insert_prefilter_decision,
)


def _execution_cfg(**overrides: object) -> ExecutionConfig:
    base: dict[str, object] = {
        "broker_mode": "paper",
        "ks4_pct_cap": 0.15,
        "ks4_absolute_cap_usd": 1000.0,
        "ks5_max_concurrent": 10,
        "ks7_cash_reserve_pct": 0.03,
        "sizing_mid_pct": 0.07,
        "sizing_high_pct": 0.10,
        "watchlist_min_conviction": 6,
        "watchlist_expiry_trading_days": 3,
        "watchlist_max_chase_pct": 8.0,
    }
    base.update(overrides)
    return ExecutionConfig(**base)


def _config(**exec_overrides: object) -> SimpleNamespace:
    return SimpleNamespace(
        execution=_execution_cfg(**exec_overrides),
        analyzer=SimpleNamespace(opus_review_conviction_threshold=7),
    )


def _set_low_cash_margin_account(fake_broker) -> None:
    """Margin-account shape: validator's buying-power gate passes but
    KS-7's cash reserve rejects — cash 350 vs the 3% floor (300) on 10k
    equity leaves 50 of headroom, less than one share at quote 100."""
    fake_broker.get_account = lambda: AccountSnapshot(
        equity=10_000.0,
        cash=350.0,
        buying_power=5_000.0,
        long_market_value=9_650.0,
        daypl=0.0,
        snapshot_at=dt.datetime.now(dt.UTC),
    )


def _set_flush_account(fake_broker) -> None:
    """Plenty of settled cash — every capital gate passes."""
    fake_broker.get_account = lambda: AccountSnapshot(
        equity=10_000.0,
        cash=10_000.0,
        buying_power=10_000.0,
        long_market_value=0.0,
        daypl=0.0,
        snapshot_at=dt.datetime.now(dt.UTC),
    )


def _widg_position() -> Position:
    return Position(
        symbol="WIDG",
        qty=20,
        avg_entry_price=105.0,
        market_value=2000.0,
        unrealized_pnl=-100.0,
    )


def _market_now(after: dt.datetime | None = None) -> dt.datetime:
    """A deterministic instant inside regular NYSE hours (11:00 ET) on
    the first market-open day at/after `after` (default: today)."""
    base = after or dt.datetime.now(dt.UTC)
    d = base.astimezone(ET).date()
    while not is_market_open_day(d):
        d += dt.timedelta(days=1)
    return dt.datetime.combine(d, dt.time(11, 0), tzinfo=ET).astimezone(dt.UTC)


async def _run_one_filing(loop_obj, queue) -> None:
    from app.state import AgentState

    runner = asyncio.create_task(loop_obj.run())
    while not (queue.empty() and loop_obj.state == AgentState.IDLE):
        await asyncio.sleep(0.01)
    loop_obj.shutdown_requested = True
    await queue.put(None)
    await asyncio.wait_for(runner, timeout=10.0)


async def _enroll_via_ks7(
    *,
    db_path,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
    config,
    accession="0001234567-26-000900",
    conviction=9,
):
    """Drive one ACME filing through the full loop with low cash so the
    sizer rejects ks7_cash_reserve and (feature on) enrollment fires.
    Returns the constructed AgentLoop for follow-up retry calls."""
    from app.loop import AgentLoop
    from tests.integration.conftest import _opus_ratify_json, _valid_proposal_json

    _set_low_cash_margin_account(fake_broker)
    f = make_filing(accession=accession, cik=320193)
    insert_prefilter_decision(
        db_path,
        PrefilterDecisionRow(
            filing_id=f.id, decision="accept", rule_fired="material_8k_bypass"
        ),
    )
    queue: asyncio.Queue[FilingRow] = asyncio.Queue()
    await queue.put(f)
    fake_anthropic.responses_by_purpose = {
        "analyze": [_valid_proposal_json(conviction=conviction, symbol="ACME")],
        "review": [_opus_ratify_json()],
    }
    components = build_components(queue=queue)
    loop_obj = AgentLoop(
        components=components, config=config, shutdown_grace_seconds=10.0
    )
    await _run_one_filing(loop_obj, queue)
    return loop_obj


def _reject_reasons(db_path) -> list[str | None]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT reject_reason FROM executions ORDER BY id ASC"
        ).fetchall()
    return [r["reject_reason"] for r in rows]


# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ks7_capital_reject_enrolls_watchlist(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
    caplog,
) -> None:
    """ks7_cash_reserve on an approved cv9 proposal → journaled rejection
    UNCHANGED plus one pending watchlist row with the decision-time quote
    as reference_price; `watchlist.added` emitted."""
    caplog.set_level(logging.INFO, logger="app.loop")
    await _enroll_via_ks7(
        db_path=db_path,
        fake_broker=fake_broker,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(),
    )

    assert "ks7_cash_reserve" in _reject_reasons(db_path)
    rows = get_pending_watchlist(db_path)
    assert len(rows) == 1
    row = rows[0]
    assert row.symbol == "ACME"
    assert row.conviction == 9
    assert row.reference_price == 100.0  # fake broker quote_last default
    assert row.status == "pending"
    assert row.notes == "reject_reason=ks7_cash_reserve"
    added = [
        r for r in caplog.records if getattr(r, "event", None) == "watchlist.added"
    ]
    assert len(added) == 1
    assert added[0].symbol == "ACME"
    assert added[0].reject_reason == "ks7_cash_reserve"


@pytest.mark.asyncio
async def test_ks5_book_full_enrolls_watchlist(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    """ks5_concurrent_limit (the book-full case) also enrolls."""
    from app.loop import AgentLoop
    from tests.integration.conftest import _opus_ratify_json, _valid_proposal_json

    _set_flush_account(fake_broker)
    fake_broker.positions = [_widg_position()]

    f = make_filing(accession="0001234567-26-000901", cik=320193)
    insert_prefilter_decision(
        db_path,
        PrefilterDecisionRow(
            filing_id=f.id, decision="accept", rule_fired="material_8k_bypass"
        ),
    )
    queue: asyncio.Queue[FilingRow] = asyncio.Queue()
    await queue.put(f)
    fake_anthropic.responses_by_purpose = {
        "analyze": [_valid_proposal_json(conviction=9, symbol="ACME")],
        "review": [_opus_ratify_json()],
    }
    components = build_components(queue=queue)
    loop_obj = AgentLoop(
        components=components,
        config=_config(ks5_max_concurrent=1),
        shutdown_grace_seconds=10.0,
    )
    await _run_one_filing(loop_obj, queue)

    assert "ks5_concurrent_limit" in _reject_reasons(db_path)
    rows = get_pending_watchlist(db_path)
    assert [r.symbol for r in rows] == ["ACME"]


@pytest.mark.asyncio
async def test_feature_off_no_enrollment_no_table_writes(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    """watchlist_min_conviction = 0 → behavior byte-identical to
    pre-feature: the ks7 rejection is journaled and the watchlist table
    stays empty."""
    await _enroll_via_ks7(
        db_path=db_path,
        fake_broker=fake_broker,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(watchlist_min_conviction=0),
    )
    assert "ks7_cash_reserve" in _reject_reasons(db_path)
    with connect(db_path) as conn:
        n = conn.execute("SELECT COUNT(*) AS n FROM watchlist").fetchone()["n"]
    assert n == 0


@pytest.mark.asyncio
async def test_conviction_below_min_not_enrolled(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    """cv7 rejected on capital with min_conviction 8 → no enrollment."""
    await _enroll_via_ks7(
        db_path=db_path,
        fake_broker=fake_broker,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(watchlist_min_conviction=8),
        conviction=7,
    )
    assert "ks7_cash_reserve" in _reject_reasons(db_path)
    assert get_pending_watchlist(db_path) == []


@pytest.mark.asyncio
async def test_opus_reject_not_enrolled(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    """An Opus-rejected proposal never reaches the capital gates — and
    never the watchlist — even with low cash and high conviction."""
    import json

    from app.loop import AgentLoop
    from tests.integration.conftest import _valid_proposal_json

    _set_low_cash_margin_account(fake_broker)
    f = make_filing(accession="0001234567-26-000902", cik=320193)
    insert_prefilter_decision(
        db_path,
        PrefilterDecisionRow(
            filing_id=f.id, decision="accept", rule_fired="material_8k_bypass"
        ),
    )
    queue: asyncio.Queue[FilingRow] = asyncio.Queue()
    await queue.put(f)
    fake_anthropic.responses_by_purpose = {
        "analyze": [_valid_proposal_json(conviction=9, symbol="ACME")],
        "review": [
            json.dumps(
                {
                    "decision": "reject",
                    "rationale": (
                        "The filing does not support the claimed catalyst; "
                        "the agreement is non-binding and materiality is "
                        "overstated relative to market cap."
                    ),
                }
            )
        ],
    }
    components = build_components(queue=queue)
    loop_obj = AgentLoop(
        components=components, config=_config(), shutdown_grace_seconds=10.0
    )
    await _run_one_filing(loop_obj, queue)

    assert _reject_reasons(db_path) == ["opus_reject"]
    assert get_pending_watchlist(db_path) == []


@pytest.mark.asyncio
async def test_kill_switch_reject_not_enrolled(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    """A kill-switch (validator) reject is not capital-class → no row."""
    insert_kill_switch_state(
        db_path,
        KillSwitchStateRow(
            set_at=dt.datetime.now(dt.UTC),
            state="halted",
            reason="manual:operator",
            set_by="operator",
        ),
    )
    await _enroll_via_ks7(
        db_path=db_path,
        fake_broker=fake_broker,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(),
    )
    assert _reject_reasons(db_path) == ["kill_switch"]
    assert get_pending_watchlist(db_path) == []


@pytest.mark.asyncio
async def test_held_symbol_not_enrolled_even_on_capital_reject(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    """Book full AND the proposal's own symbol already held: KS-5 fires
    (capital class) but the held-symbol guard blocks enrollment — we
    never park a name we already own."""
    from app.loop import AgentLoop
    from tests.integration.conftest import _opus_ratify_json, _valid_proposal_json

    _set_flush_account(fake_broker)
    fake_broker.positions = [
        Position(
            symbol="ACME",
            qty=10,
            avg_entry_price=95.0,
            market_value=1000.0,
            unrealized_pnl=50.0,
        )
    ]
    f = make_filing(accession="0001234567-26-000903", cik=320193)
    insert_prefilter_decision(
        db_path,
        PrefilterDecisionRow(
            filing_id=f.id, decision="accept", rule_fired="material_8k_bypass"
        ),
    )
    queue: asyncio.Queue[FilingRow] = asyncio.Queue()
    await queue.put(f)
    fake_anthropic.responses_by_purpose = {
        "analyze": [_valid_proposal_json(conviction=9, symbol="ACME")],
        "review": [_opus_ratify_json()],
    }
    components = build_components(queue=queue)
    loop_obj = AgentLoop(
        components=components,
        config=_config(ks5_max_concurrent=1),
        shutdown_grace_seconds=10.0,
    )
    await _run_one_filing(loop_obj, queue)

    assert "ks5_concurrent_limit" in _reject_reasons(db_path)
    assert get_pending_watchlist(db_path) == []


@pytest.mark.asyncio
async def test_dedup_one_active_row_per_symbol(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    """Two capital-blocked filings on the same symbol → one pending row
    (the first proposal keeps the slot)."""
    loop_obj = await _enroll_via_ks7(
        db_path=db_path,
        fake_broker=fake_broker,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(),
        accession="0001234567-26-000904",
    )
    first = get_pending_watchlist(db_path)[0]

    from tests.integration.conftest import _opus_ratify_json, _valid_proposal_json

    f2 = make_filing(accession="0001234567-26-000905", cik=320193)
    insert_prefilter_decision(
        db_path,
        PrefilterDecisionRow(
            filing_id=f2.id, decision="accept", rule_fired="material_8k_bypass"
        ),
    )
    # Same components bundle → same ingestion queue the consumer reads.
    queue2 = loop_obj.components.ingestion_queue
    await queue2.put(f2)
    fake_anthropic.responses_by_purpose = {
        "analyze": [_valid_proposal_json(conviction=9, symbol="ACME")],
        "review": [_opus_ratify_json()],
    }
    from app.loop import AgentLoop

    loop2 = AgentLoop(
        components=loop_obj.components, config=_config(), shutdown_grace_seconds=10.0
    )
    await _run_one_filing(loop2, queue2)

    rows = get_pending_watchlist(db_path)
    assert len(rows) == 1
    assert rows[0].id == first.id
    assert rows[0].proposal_id == first.proposal_id


# ---------------------------------------------------------------------------
# Retry pass
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_enters_through_normal_path_when_capital_frees(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
    caplog,
) -> None:
    """Capital frees → the retry submits the REAL bracket through the
    submitter, resolves the row `entered` (with lag + delta in notes),
    journals the accepted execution, and schedules the thesis review."""
    caplog.set_level(logging.INFO, logger="app.loop")
    loop_obj = await _enroll_via_ks7(
        db_path=db_path,
        fake_broker=fake_broker,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(),
    )
    row = get_pending_watchlist(db_path)[0]
    assert fake_broker.submitted_brackets == []

    _set_flush_account(fake_broker)
    await loop_obj._process_watchlist(_market_now())

    # Broker saw the ACME bracket entry.
    assert len(fake_broker.submitted_brackets) == 1
    assert fake_broker.submitted_brackets[0].entry_symbol == "ACME"
    # Row resolved `entered` with the audit notes.
    with connect(db_path) as conn:
        r = conn.execute(
            "SELECT * FROM watchlist WHERE id = ?", (row.id,)
        ).fetchone()
    assert r["status"] == "entered"
    assert r["resolved_at"] is not None
    assert "lag_trading_days=" in r["notes"]
    assert "delta_pct=" in r["notes"]
    # Accepted execution journaled for the SAME proposal.
    with connect(db_path) as conn:
        ex = conn.execute(
            "SELECT decision FROM executions WHERE proposal_id = ? "
            "ORDER BY id ASC",
            (row.proposal_id,),
        ).fetchall()
    assert [e["decision"] for e in ex] == ["rejected", "accepted"]
    # Thesis review armed off the new execution.
    with connect(db_path) as conn:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM thesis_review_schedule"
        ).fetchone()["n"]
    assert n == 1
    entered = [
        r for r in caplog.records if getattr(r, "event", None) == "watchlist.entered"
    ]
    assert len(entered) == 1
    assert entered[0].symbol == "ACME"
    assert entered[0].price_delta_pct == 0.0


@pytest.mark.asyncio
async def test_retry_still_capital_blocked_leaves_pending_logs_once_per_day(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
    caplog,
) -> None:
    """Capital still tight → row stays pending (no status change), no
    broker orders, and `watchlist.retry_blocked` fires at most once per
    ET day per row across repeated ticks."""
    caplog.set_level(logging.INFO, logger="app.loop")
    loop_obj = await _enroll_via_ks7(
        db_path=db_path,
        fake_broker=fake_broker,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(),
    )
    now = _market_now()
    await loop_obj._process_watchlist(now)
    await loop_obj._process_watchlist(now + dt.timedelta(minutes=1))

    rows = get_pending_watchlist(db_path)
    assert len(rows) == 1 and rows[0].status == "pending"
    assert fake_broker.submitted_brackets == []
    blocked = [
        r
        for r in caplog.records
        if getattr(r, "event", None) == "watchlist.retry_blocked"
    ]
    assert len(blocked) == 1
    assert blocked[0].reason == "ks7_cash_reserve"


@pytest.mark.asyncio
async def test_retry_respects_ks_gate_mocked_rejection_leaves_pending(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    """The retry goes through the full gate chain: with the kill-switch
    halted, the (re-run) validator rejects and the row stays pending —
    no broker order, no status change."""
    loop_obj = await _enroll_via_ks7(
        db_path=db_path,
        fake_broker=fake_broker,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(),
    )
    _set_flush_account(fake_broker)
    insert_kill_switch_state(
        db_path,
        KillSwitchStateRow(
            set_at=dt.datetime.now(dt.UTC),
            state="halted",
            reason="manual:operator",
            set_by="operator",
        ),
    )
    await loop_obj._process_watchlist(_market_now())

    rows = get_pending_watchlist(db_path)
    assert len(rows) == 1 and rows[0].status == "pending"
    assert fake_broker.submitted_brackets == []
    # No NEW executions row was journaled by the retry (no reject spam).
    assert _reject_reasons(db_path) == ["ks7_cash_reserve"]


@pytest.mark.asyncio
async def test_retry_chase_at_ceiling_is_allowed(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    """Price exactly AT reference * (1 + max_chase) still enters."""
    loop_obj = await _enroll_via_ks7(
        db_path=db_path,
        fake_broker=fake_broker,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(),
    )
    _set_flush_account(fake_broker)
    fake_broker.quote_last = 108.0  # ceiling = 100 * 1.08
    await loop_obj._process_watchlist(_market_now())

    with connect(db_path) as conn:
        r = conn.execute("SELECT status FROM watchlist").fetchone()
    assert r["status"] == "entered"
    assert len(fake_broker.submitted_brackets) == 1


@pytest.mark.asyncio
async def test_retry_chase_above_ceiling_skips(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
    caplog,
) -> None:
    """Price strictly above the ceiling → skipped_chase with both prices
    journaled and logged; no broker order."""
    caplog.set_level(logging.INFO, logger="app.loop")
    loop_obj = await _enroll_via_ks7(
        db_path=db_path,
        fake_broker=fake_broker,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(),
    )
    _set_flush_account(fake_broker)
    fake_broker.quote_last = 108.5
    await loop_obj._process_watchlist(_market_now())

    with connect(db_path) as conn:
        r = conn.execute("SELECT status, notes FROM watchlist").fetchone()
    assert r["status"] == "skipped_chase"
    assert "last=108.5000" in r["notes"]
    assert "ref=100.0000" in r["notes"]
    assert fake_broker.submitted_brackets == []
    skipped = [
        rec
        for rec in caplog.records
        if getattr(rec, "event", None) == "watchlist.skipped_chase"
    ]
    assert len(skipped) == 1
    assert skipped[0].reference_price == 100.0
    assert skipped[0].current_last == 108.5


@pytest.mark.asyncio
async def test_retry_expires_by_trading_days(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    """A tick past expires_at resolves `expired` without broker calls."""
    loop_obj = await _enroll_via_ks7(
        db_path=db_path,
        fake_broker=fake_broker,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(),
    )
    _set_flush_account(fake_broker)
    late = _market_now(dt.datetime.now(dt.UTC) + dt.timedelta(days=30))
    await loop_obj._process_watchlist(late)

    with connect(db_path) as conn:
        r = conn.execute("SELECT status FROM watchlist").fetchone()
    assert r["status"] == "expired"
    assert fake_broker.submitted_brackets == []


@pytest.mark.asyncio
async def test_retry_skips_symbol_now_held(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    loop_obj = await _enroll_via_ks7(
        db_path=db_path,
        fake_broker=fake_broker,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(),
    )
    _set_flush_account(fake_broker)
    fake_broker.positions = [
        Position(
            symbol="ACME",
            qty=5,
            avg_entry_price=100.0,
            market_value=500.0,
            unrealized_pnl=0.0,
        )
    ]
    await loop_obj._process_watchlist(_market_now())

    with connect(db_path) as conn:
        r = conn.execute("SELECT status FROM watchlist").fetchone()
    assert r["status"] == "skipped_held"
    assert fake_broker.submitted_brackets == []


@pytest.mark.asyncio
async def test_fresh_day0_work_preempts_watchlist_in_same_tick(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    """With day-0 work queued, the whole watchlist pass defers — the row
    is untouched even though capital is available; once the queue drains
    the next tick enters it."""
    loop_obj = await _enroll_via_ks7(
        db_path=db_path,
        fake_broker=fake_broker,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(),
    )
    _set_flush_account(fake_broker)
    # Fresh day-0 work waiting in the ingestion queue.
    loop_obj.components.ingestion_queue.put_nowait(object())
    await loop_obj._process_watchlist(_market_now())
    rows = get_pending_watchlist(db_path)
    assert len(rows) == 1 and rows[0].status == "pending"
    assert fake_broker.submitted_brackets == []

    # Queue drained → the next tick processes the watchlist.
    loop_obj.components.ingestion_queue.get_nowait()
    await loop_obj._process_watchlist(_market_now())
    with connect(db_path) as conn:
        r = conn.execute("SELECT status FROM watchlist").fetchone()
    assert r["status"] == "entered"


@pytest.mark.asyncio
async def test_watchlist_pass_is_noop_outside_market_hours(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    loop_obj = await _enroll_via_ks7(
        db_path=db_path,
        fake_broker=fake_broker,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(),
    )
    _set_flush_account(fake_broker)
    saturday = dt.datetime(2026, 7, 25, 15, 0, tzinfo=dt.UTC)  # Sat
    assert saturday.weekday() == 5
    await loop_obj._process_watchlist(saturday)
    rows = get_pending_watchlist(db_path)
    assert len(rows) == 1 and rows[0].status == "pending"
    assert fake_broker.submitted_brackets == []


@pytest.mark.asyncio
async def test_feature_off_tick_does_no_work(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    """Feature toggled off AFTER a row exists (e.g. operator rollback):
    the tick touches nothing — the row stays pending forever and no
    broker call is made."""
    loop_obj = await _enroll_via_ks7(
        db_path=db_path,
        fake_broker=fake_broker,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(),
    )
    _set_flush_account(fake_broker)
    from app.loop import AgentLoop

    loop_off = AgentLoop(
        components=loop_obj.components,
        config=_config(watchlist_min_conviction=0),
        shutdown_grace_seconds=10.0,
    )
    await loop_off._process_watchlist(_market_now())
    rows = get_pending_watchlist(db_path)
    assert len(rows) == 1 and rows[0].status == "pending"
    assert fake_broker.submitted_brackets == []


# ---------------------------------------------------------------------------
# Review advisories A1/A2 — broker-failure idempotency + displacement interplay
# ---------------------------------------------------------------------------


def _seed_pending_displacement_buy(db_path, make_filing) -> str:
    """Journal lineage for a displacement-funded WIDG entry buy that is
    still PENDING at the broker (submitted, unfilled): accepted execution
    + entry order row priced at pre_submission_last=100. Returns the
    broker_order_id the fake broker's open order must carry."""
    import json as _json

    from journal.models import ExecutionRow, OrderRow, PromptRow, ProposalRow
    from journal.repo import (
        insert_execution,
        insert_order,
        insert_prompt,
        insert_proposal,
    )

    f = make_filing(
        accession="0001234567-26-000970", cik=789019, issuer_ticker="WIDG"
    )
    insert_prefilter_decision(
        db_path,
        PrefilterDecisionRow(
            filing_id=f.id, decision="accept", rule_fired="material_8k_bypass"
        ),
    )
    insert_prompt(
        db_path,
        PromptRow(
            prompt_version="sonnet@wldisp",
            name="sonnet",
            file_path="src/prompts/sonnet.txt",
            content_hash="x" * 64,
        ),
    )
    pid = insert_proposal(
        db_path,
        ProposalRow(
            filing_id=f.id,
            decision_id="wl-disp-widg",
            model_id="claude-sonnet-4-6",
            prompt_version="sonnet@wldisp",
            raw_response=_json.dumps({"symbol": "WIDG"}),
            kind="trade_proposal",
            symbol="WIDG",
            direction="long",
            size_pct_requested=0.10,
            conviction=8,
            thesis="displacement-funded entry",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
            cost_usd=0.0,
        ),
    )
    eid = insert_execution(
        db_path,
        ExecutionRow(
            proposal_id=pid,
            decision="accepted",
            realized_size_pct=0.10,
            realized_dollar_size=1000.0,
            submitted_orders_json='[{"broker_order_id": "disp-buy-1", "role": "entry"}]',
        ),
    )
    insert_order(
        db_path,
        OrderRow(
            execution_id=eid,
            role="entry",
            symbol="WIDG",
            side="buy",
            order_type="market",
            qty=10,
            tif="day",
            broker_order_id="disp-buy-1",
            submitted_at=dt.datetime.now(dt.UTC),
            pre_submission_last=100.0,
            final_status=None,
        ),
    )
    return "disp-buy-1"


@pytest.mark.asyncio
async def test_hard_reject_with_broker_order_gates_retry_forever(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
    caplog,
) -> None:
    """T1 (4cdf3d9 incident class): a `submission_failed` execution WITH
    a broker order row recorded — some leg reached the broker — GATES
    every later retry: the row stays pending (until expiry) and no
    second `prop-{id}-entry` submission ever reaches the broker."""
    from journal.models import ExecutionRow, OrderRow
    from journal.repo import insert_execution, insert_order

    caplog.set_level(logging.INFO, logger="app.loop")
    loop_obj = await _enroll_via_ks7(
        db_path=db_path,
        fake_broker=fake_broker,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(),
    )
    row = get_pending_watchlist(db_path)[0]
    # Journal state of the incident class: the retry's submission failed
    # AFTER the entry leg reached the broker.
    eid = insert_execution(
        db_path,
        ExecutionRow(
            proposal_id=row.proposal_id,
            decision="submission_failed",
            submitted_orders_json='[{"broker_order_id": "stale-entry-1", "role": "entry"}]',
        ),
    )
    insert_order(
        db_path,
        OrderRow(
            execution_id=eid,
            role="entry",
            symbol="ACME",
            side="buy",
            order_type="market",
            qty=10,
            tif="day",
            broker_order_id="stale-entry-1",
            submitted_at=dt.datetime.now(dt.UTC),
            pre_submission_last=100.0,
            final_status=None,
        ),
    )
    _set_flush_account(fake_broker)
    await loop_obj._process_watchlist(_market_now())

    rows = get_pending_watchlist(db_path)
    assert len(rows) == 1 and rows[0].status == "pending"
    # THE assertion: no second prop-{id}-entry submission of any kind.
    assert fake_broker.submitted_brackets == []
    assert fake_broker.submitted_orders == []
    blocked = [
        r
        for r in caplog.records
        if getattr(r, "event", None) == "watchlist.retry_blocked"
    ]
    assert len(blocked) == 1
    assert blocked[0].reason == "already_executed"


@pytest.mark.asyncio
async def test_transient_broker_outage_leaves_row_retryable(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    """T3 (A1 positive case): a transient BrokerUnavailable on the retry
    journals `submission_failed` with NO broker order rows — nothing
    reached the broker — so the row stays pending AND a later retry
    actually submits (one blip must not strand the row until expiry)."""
    from broker.alpaca import BrokerUnavailable

    loop_obj = await _enroll_via_ks7(
        db_path=db_path,
        fake_broker=fake_broker,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(),
    )
    row = get_pending_watchlist(db_path)[0]
    _set_flush_account(fake_broker)

    def _outage(req):
        raise BrokerUnavailable("connection reset")

    fake_broker.submit_bracket_order = _outage  # instance attr shadows method
    now = _market_now()
    await loop_obj._process_watchlist(now)

    rows = get_pending_watchlist(db_path)
    assert len(rows) == 1 and rows[0].status == "pending"
    with connect(db_path) as conn:
        decisions = [
            r["decision"]
            for r in conn.execute(
                "SELECT decision FROM executions WHERE proposal_id = ? "
                "ORDER BY id ASC",
                (row.proposal_id,),
            ).fetchall()
        ]
        n_orders = conn.execute("SELECT COUNT(*) AS n FROM orders").fetchone()["n"]
    assert decisions == ["rejected", "submission_failed"]
    assert n_orders == 0  # nothing reached the broker

    # Broker back up → the NEXT retry submits and resolves `entered`.
    del fake_broker.__dict__["submit_bracket_order"]
    await loop_obj._process_watchlist(now + dt.timedelta(minutes=1))

    with connect(db_path) as conn:
        r = conn.execute("SELECT status FROM watchlist").fetchone()
        decisions = [
            x["decision"]
            for x in conn.execute(
                "SELECT decision FROM executions WHERE proposal_id = ? "
                "ORDER BY id ASC",
                (row.proposal_id,),
            ).fetchall()
        ]
    assert r["status"] == "entered"
    assert decisions == ["rejected", "submission_failed", "accepted"]
    assert len(fake_broker.submitted_brackets) == 1
    assert fake_broker.submitted_brackets[0].entry_symbol == "ACME"


@pytest.mark.asyncio
async def test_retry_sees_pending_displacement_buy_and_does_not_double_spend(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
    caplog,
) -> None:
    """T2: displacement freed cash and its funded entry buy is still
    PENDING at the broker. The next tick's watchlist retry must count
    that committed spend (pending_entry_spend / KS-5 pending view) and
    NOT double-spend the freed cash; once the pending buy is gone the
    same cash funds the watchlist entry."""
    from broker.protocol import OpenOrder

    caplog.set_level(logging.INFO, logger="app.loop")
    loop_obj = await _enroll_via_ks7(
        db_path=db_path,
        fake_broker=fake_broker,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(),
    )
    boid = _seed_pending_displacement_buy(db_path, make_filing)
    fake_broker.open_orders = [
        OpenOrder(
            symbol="WIDG",
            side="buy",
            qty=10,
            order_type="market",
            status="accepted",
            broker_order_id=boid,
            client_order_id="prop-9999-entry",
        )
    ]
    # Victim sale proceeds landed: cash 1200 vs 3% reserve (300) on 10k
    # equity. WITHOUT the pending 10 x 100 = 1000 committed by the
    # displacement buy this would fund ~9 shares; WITH it, available
    # cash is 200 — under the floor → ks7_cash_reserve.
    fake_broker.get_account = lambda: AccountSnapshot(
        equity=10_000.0,
        cash=1_200.0,
        buying_power=5_000.0,
        long_market_value=8_800.0,
        daypl=0.0,
        snapshot_at=dt.datetime.now(dt.UTC),
    )
    now = _market_now()
    await loop_obj._process_watchlist(now)

    rows = get_pending_watchlist(db_path)
    assert len(rows) == 1 and rows[0].status == "pending"
    assert fake_broker.submitted_brackets == []
    blocked = [
        r
        for r in caplog.records
        if getattr(r, "event", None) == "watchlist.retry_blocked"
    ]
    assert len(blocked) == 1
    assert blocked[0].reason == "ks7_cash_reserve"

    # Positive control: the displacement buy fills/clears → the SAME
    # account state now funds the watchlist entry.
    fake_broker.open_orders = []
    await loop_obj._process_watchlist(now + dt.timedelta(minutes=1))
    with connect(db_path) as conn:
        r = conn.execute(
            "SELECT status FROM watchlist WHERE symbol = 'ACME'"
        ).fetchone()
    assert r["status"] == "entered"
    assert len(fake_broker.submitted_brackets) == 1


@pytest.mark.asyncio
async def test_dead_broker_entry_is_not_adopted_as_phantom_position(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    """A2: the broker remembers a crash-window `prop-{id}-entry` that was
    REJECTED — it created no exposure. The retry must NOT adopt it as a
    phantom accepted position; it proceeds to a fresh submission and the
    journal shows a real `accepted` execution with a live bracket."""
    from broker.protocol import SubmittedOrder

    loop_obj = await _enroll_via_ks7(
        db_path=db_path,
        fake_broker=fake_broker,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(),
    )
    row = get_pending_watchlist(db_path)[0]
    cid = f"prop-{row.proposal_id}-entry"
    fake_broker.preseeded_by_client_id[cid] = SubmittedOrder(
        broker_order_id="dead-entry-1",
        client_order_id=cid,
        symbol="ACME",
        side="buy",
        qty=10,
        order_type="market",
        status="rejected",
        submitted_at=dt.datetime.now(dt.UTC) - dt.timedelta(minutes=5),
    )
    _set_flush_account(fake_broker)
    await loop_obj._process_watchlist(_market_now())

    with connect(db_path) as conn:
        r = conn.execute("SELECT status FROM watchlist").fetchone()
        orders = conn.execute(
            "SELECT broker_order_id, notes FROM orders ORDER BY id ASC"
        ).fetchall()
    assert r["status"] == "entered"
    # A REAL bracket was submitted (not a phantom adoption of the dead id).
    assert len(fake_broker.submitted_brackets) == 1
    assert all(o["broker_order_id"] != "dead-entry-1" for o in orders)
    assert all(
        (o["notes"] or "") != "adopted_from_broker_on_recovery" for o in orders
    )

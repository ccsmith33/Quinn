"""API cost levers — agent-loop integration.

Two default-OFF gates that decline to spend LLM tokens on work that
cannot become a trade:

  LEVER 2 (`analyzer.analyzer_unbuyable_gate`) — a prefiltered filing
  whose issuer is outside the universe snapshot, or priced under the
  validator's floor, never reaches the analyzer. Fails OPEN on every
  ambiguity.

  LEVER 1 (`analyzer.opus_review_full_book_gate`) — when the book is at
  its effective KS-5 cap AND KS-7 headroom cannot fund one cheapest-legal
  share, the Opus-review bar rises to displacement-viable conviction.
  Proposals under the raised bar are journaled `review_skipped_full_book`
  and park on the watchlist, which pays for the deferred review once
  capital frees (FR-17 holds on every path).

Both gates OFF must be byte-identical to pre-lever behavior — the last
test in each section pins that.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from broker.protocol import AccountSnapshot, Position
from config.calendar import ET, is_market_open_day
from config.loader import ExecutionConfig
from journal.models import FilingRow, PrefilterDecisionRow
from journal.repo import connect, get_pending_watchlist, insert_prefilter_decision

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _execution_cfg(**overrides: Any) -> ExecutionConfig:
    base: dict[str, Any] = {
        "broker_mode": "paper",
        "ks4_pct_cap": 0.15,
        "ks4_absolute_cap_usd": 1000.0,
        "ks5_max_concurrent": 10,
        "ks7_cash_reserve_pct": 0.03,
        "sizing_mid_pct": 0.07,
        "sizing_high_pct": 0.10,
        "price_floor_usd": 5.0,
        "watchlist_min_conviction": 6,
        "watchlist_expiry_trading_days": 3,
        "watchlist_max_chase_pct": 8.0,
    }
    base.update(overrides)
    return ExecutionConfig(**base)


def _config(
    *,
    unbuyable_gate: bool = False,
    full_book_gate: bool = False,
    threshold: int = 5,
    **exec_overrides: Any,
) -> SimpleNamespace:
    return SimpleNamespace(
        analyzer=SimpleNamespace(
            opus_review_conviction_threshold=threshold,
            analyzer_unbuyable_gate=unbuyable_gate,
            opus_review_full_book_gate=full_book_gate,
        ),
        execution=_execution_cfg(**exec_overrides),
    )


def _account(fake_broker, *, equity: float, cash: float) -> None:
    fake_broker.get_account = lambda: AccountSnapshot(
        equity=equity,
        cash=cash,
        buying_power=cash,
        long_market_value=equity - cash,
        daypl=0.0,
        snapshot_at=dt.datetime.now(dt.UTC),
    )


def _cash_dry(fake_broker) -> None:
    """KS-7 headroom of $1 on 10k equity at a 3% reserve — below the $5
    price floor, so no entry in the universe is fundable."""
    _account(fake_broker, equity=10_000.0, cash=301.0)


def _cash_flush(fake_broker) -> None:
    _account(fake_broker, equity=10_000.0, cash=10_000.0)


def _position(symbol: str) -> Position:
    return Position(
        symbol=symbol,
        qty=10,
        avg_entry_price=50.0,
        market_value=500.0,
        unrealized_pnl=0.0,
    )


def _market_now() -> dt.datetime:
    d = dt.datetime.now(dt.UTC).astimezone(ET).date()
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


async def _drive(
    *,
    db_path,
    fake_anthropic,
    build_components,
    make_filing,
    config,
    conviction: int = 6,
    accession: str = "0001234567-26-00LEV1",
    cik: int = 320193,
    issuer_ticker: str = "ACME",
    preseed_prefilter: bool = True,
):
    """Push one filing through the real loop and hand back the AgentLoop.

    The prefilter decision is pre-seeded so the orchestrator replays it
    rather than re-deciding — that is also the only way to reach the
    gate's universe branch, since the prefilter's own stage-1 CIK gate
    would otherwise reject an out-of-universe filer first.
    """
    from app.loop import AgentLoop
    from tests.integration.conftest import _opus_ratify_json, _valid_proposal_json

    f = make_filing(accession=accession, cik=cik, issuer_ticker=issuer_ticker)
    if preseed_prefilter:
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
    components = build_components(queue=queue, config=config)
    loop_obj = AgentLoop(
        components=components, config=config, shutdown_grace_seconds=10.0
    )
    await _run_one_filing(loop_obj, queue)
    return loop_obj


def _calls(fake_anthropic, purpose: str) -> list[dict[str, Any]]:
    return [c for c in fake_anthropic.calls if c["purpose"] == purpose]


def _reject_reasons(db_path) -> list[str | None]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT reject_reason FROM executions ORDER BY id ASC"
        ).fetchall()
    return [r["reject_reason"] for r in rows]


# ---------------------------------------------------------------------------
# LEVER 2 — pre-analysis buyability gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_in_universe_in_band_filing_is_analyzed(
    db_path, journal, fake_broker, fake_anthropic, build_components, make_filing
) -> None:
    """ACME is a snapshot member at prev_close 100, floor 5 — the gate
    must not touch it."""
    _cash_flush(fake_broker)
    await _drive(
        db_path=db_path,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(unbuyable_gate=True),
        conviction=9,
    )
    assert len(_calls(fake_anthropic, "analyze")) == 1


@pytest.mark.asyncio
async def test_out_of_universe_filing_skips_analysis(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
    caplog,
) -> None:
    """Zero LLM calls, zero proposals, one `analyzer_skipped_unbuyable`."""
    caplog.set_level(logging.INFO, logger="app.loop")
    _cash_flush(fake_broker)
    await _drive(
        db_path=db_path,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(unbuyable_gate=True),
        cik=999999,
        issuer_ticker="ZZZZ",
    )

    assert fake_anthropic.calls == []
    with connect(db_path) as conn:
        n = conn.execute("SELECT COUNT(*) AS n FROM proposals").fetchone()["n"]
    assert n == 0
    skipped = [
        r
        for r in caplog.records
        if getattr(r, "event", None) == "agent.analyzer_skipped_unbuyable"
    ]
    assert len(skipped) == 1
    assert skipped[0].symbol == "ZZZZ"
    assert skipped[0].reason == "universe"
    assert skipped[0].price is None


@pytest.mark.asyncio
async def test_below_price_floor_skips_analysis(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
    caplog,
) -> None:
    """ACME's snapshot prev_close is 100; an operator floor of 150 puts it
    out of band. This is the shape the lever actually catches in prod —
    the snapshot's own screen already enforces the default $5."""
    caplog.set_level(logging.INFO, logger="app.loop")
    _cash_flush(fake_broker)
    await _drive(
        db_path=db_path,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(unbuyable_gate=True, price_floor_usd=150.0),
    )

    assert fake_anthropic.calls == []
    skipped = [
        r
        for r in caplog.records
        if getattr(r, "event", None) == "agent.analyzer_skipped_unbuyable"
    ]
    assert len(skipped) == 1
    assert skipped[0].reason == "price"
    assert skipped[0].price == 100.0


@pytest.mark.asyncio
async def test_unresolvable_ticker_fails_open(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
    caplog,
) -> None:
    """No issuer_ticker → analyze anyway, and say so with a counter."""
    caplog.set_level(logging.INFO, logger="app.loop")
    _cash_flush(fake_broker)
    await _drive(
        db_path=db_path,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(unbuyable_gate=True),
        issuer_ticker=None,
        conviction=9,
    )

    assert len(_calls(fake_anthropic, "analyze")) == 1
    fail_open = [
        r
        for r in caplog.records
        if getattr(r, "event", None) == "agent.analyzer_gate_fail_open"
    ]
    assert len(fail_open) == 1
    assert fail_open[0].detail == "unresolved_ticker"
    assert fail_open[0].fail_open_count == 1


@pytest.mark.asyncio
async def test_missing_universe_snapshot_fails_open(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
    caplog,
) -> None:
    """An empty snapshot answers "not a member" to everything. The gate
    must never read that as a universe-wide rejection — that is exactly
    the wholesale-silence failure the lever is forbidden to cause."""
    caplog.set_level(logging.INFO, logger="app.loop")
    _cash_flush(fake_broker)

    from app.loop import AgentLoop
    from tests.integration.conftest import _opus_ratify_json, _valid_proposal_json

    f = make_filing(accession="0001234567-26-00EMPT", cik=320193)
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
    config = _config(unbuyable_gate=True)
    components = build_components(queue=queue, config=config)
    # Empty the loaded snapshot in place — the shape a failed member load
    # leaves behind (carry-forward D-038).
    components.universe._tickers = set()
    components.universe._ciks = set()
    components.universe._members_by_ticker = {}

    loop_obj = AgentLoop(
        components=components, config=config, shutdown_grace_seconds=10.0
    )
    await _run_one_filing(loop_obj, queue)

    assert len(_calls(fake_anthropic, "analyze")) == 1
    fail_open = [
        r
        for r in caplog.records
        if getattr(r, "event", None) == "agent.analyzer_gate_fail_open"
    ]
    assert len(fail_open) == 1
    assert fail_open[0].detail == "empty_universe_snapshot"


@pytest.mark.asyncio
async def test_unbuyable_gate_off_analyzes_everything(
    db_path, journal, fake_broker, fake_anthropic, build_components, make_filing
) -> None:
    """Gate off — the out-of-universe filing that the gate would have
    dropped is analyzed exactly as before."""
    _cash_flush(fake_broker)
    await _drive(
        db_path=db_path,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(unbuyable_gate=False),
        cik=999999,
        issuer_ticker="ZZZZ",
        conviction=9,
    )
    assert len(_calls(fake_anthropic, "analyze")) == 1


# ---------------------------------------------------------------------------
# LEVER 1 — full-book Opus review gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_book_and_dry_cash_skips_review_and_enrolls_watchlist(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
    caplog,
) -> None:
    """The load-bearing case: cv6 at a configured bar of 5, book at cap,
    KS-7 headroom under one share. Opus is NOT called, the rejection is
    journaled `review_skipped_full_book`, and the proposal parks on the
    watchlist for its second chance."""
    caplog.set_level(logging.INFO, logger="app.loop")
    fake_broker.positions = [_position("WIDG")]
    _cash_dry(fake_broker)

    await _drive(
        db_path=db_path,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(full_book_gate=True, threshold=5, ks5_max_concurrent=1),
        conviction=6,
    )

    assert len(_calls(fake_anthropic, "analyze")) == 1
    assert _calls(fake_anthropic, "review") == [], (
        "Opus must not be paid for a proposal that cannot be bought"
    )
    assert _reject_reasons(db_path) == ["review_skipped_full_book"]
    assert fake_broker.submitted_orders == []

    rows = get_pending_watchlist(db_path)
    assert len(rows) == 1
    assert rows[0].symbol == "ACME"
    assert rows[0].conviction == 6
    assert rows[0].notes == "reject_reason=review_skipped_full_book"

    logged = [
        r
        for r in caplog.records
        if getattr(r, "event", None) == "agent.execution_review_skipped_full_book"
    ]
    assert len(logged) == 1
    assert logged[0].configured_threshold == 5
    assert logged[0].effective_threshold == 8


@pytest.mark.asyncio
async def test_free_slot_keeps_the_configured_review_bar(
    db_path, journal, fake_broker, fake_anthropic, build_components, make_filing
) -> None:
    """One position against a cap of 10 — cash is dry, but a slot is open,
    so the gate does not fire and cv6 is reviewed as usual."""
    fake_broker.positions = [_position("WIDG")]
    _cash_dry(fake_broker)

    await _drive(
        db_path=db_path,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(full_book_gate=True, threshold=5, ks5_max_concurrent=10),
        conviction=6,
    )

    assert len(_calls(fake_anthropic, "review")) == 1
    assert "review_skipped_full_book" not in _reject_reasons(db_path)


@pytest.mark.asyncio
async def test_cash_available_keeps_the_configured_review_bar(
    db_path, journal, fake_broker, fake_anthropic, build_components, make_filing
) -> None:
    """Book full but cash can still fund an entry (a slot may free
    intraday) — both conditions are required, so the bar holds."""
    fake_broker.positions = [_position("WIDG")]
    _cash_flush(fake_broker)

    await _drive(
        db_path=db_path,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(full_book_gate=True, threshold=5, ks5_max_concurrent=1),
        conviction=6,
    )
    assert len(_calls(fake_anthropic, "review")) == 1


@pytest.mark.asyncio
async def test_full_book_gate_off_reviews_exactly_as_before(
    db_path, journal, fake_broker, fake_anthropic, build_components, make_filing
) -> None:
    """Gate off — the identical book state reviews cv6 through Opus, and
    nothing is ever journaled `review_skipped_full_book`."""
    fake_broker.positions = [_position("WIDG")]
    _cash_dry(fake_broker)

    await _drive(
        db_path=db_path,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(full_book_gate=False, threshold=5, ks5_max_concurrent=1),
        conviction=6,
    )

    assert len(_calls(fake_anthropic, "review")) == 1
    assert "review_skipped_full_book" not in _reject_reasons(db_path)


@pytest.mark.asyncio
async def test_conviction_at_displacement_floor_still_gets_reviewed(
    db_path, journal, fake_broker, fake_anthropic, build_components, make_filing
) -> None:
    """Displacement ON at min_conviction 6 makes cv6 displacement-viable —
    it can still fund itself by evicting a position, so the gate must not
    raise the bar past it."""
    fake_broker.positions = [_position("WIDG")]
    _cash_dry(fake_broker)

    await _drive(
        db_path=db_path,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(
            full_book_gate=True,
            threshold=5,
            ks5_max_concurrent=1,
            displacement_enabled=True,
            displacement_min_conviction=6,
        ),
        conviction=6,
    )

    assert len(_calls(fake_anthropic, "review")) == 1
    assert "review_skipped_full_book" not in _reject_reasons(db_path)


@pytest.mark.asyncio
async def test_high_conviction_above_the_raised_bar_still_gets_reviewed(
    db_path, journal, fake_broker, fake_anthropic, build_components, make_filing
) -> None:
    """cv9 clears the raised bar of 8, so the gate leaves it alone and the
    pre-existing pending_capacity path is untouched."""
    fake_broker.positions = [_position("WIDG")]
    _cash_dry(fake_broker)

    await _drive(
        db_path=db_path,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(full_book_gate=True, threshold=5, ks5_max_concurrent=1),
        conviction=9,
    )
    assert len(_calls(fake_anthropic, "review")) == 1


@pytest.mark.asyncio
async def test_watchlist_retry_pays_the_deferred_review_before_entering(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
    caplog,
) -> None:
    """The second chance has to actually work, and FR-17 has to hold: when
    capital frees, the retry buys the Opus review it skipped on day 0 and
    only then submits. Never an unreviewed entry."""
    caplog.set_level(logging.INFO, logger="app.loop")
    fake_broker.positions = [_position("WIDG")]
    _cash_dry(fake_broker)

    loop_obj = await _drive(
        db_path=db_path,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(full_book_gate=True, threshold=5, ks5_max_concurrent=1),
        conviction=6,
    )
    assert _calls(fake_anthropic, "review") == []
    assert len(get_pending_watchlist(db_path)) == 1

    # A slot frees and cash returns.
    fake_broker.positions = []
    _cash_flush(fake_broker)
    await loop_obj._process_watchlist(_market_now())

    # Opus ran exactly once, on the retry — and the entry followed it.
    assert len(_calls(fake_anthropic, "review")) == 1
    assert len(fake_broker.submitted_brackets) == 1
    assert fake_broker.submitted_brackets[0].entry_symbol == "ACME"
    with connect(db_path) as conn:
        decisions = [
            r["decision"]
            for r in conn.execute(
                "SELECT decision FROM executions ORDER BY id ASC"
            ).fetchall()
        ]
        review_rows = conn.execute(
            "SELECT COUNT(*) AS n FROM proposal_reviews"
        ).fetchone()["n"]
    assert decisions == ["rejected", "accepted"]
    assert review_rows == 1
    deferred = [
        r
        for r in caplog.records
        if getattr(r, "event", None) == "agent.watchlist_deferred_review"
    ]
    assert len(deferred) == 1


@pytest.mark.asyncio
async def test_gate_skip_survives_broker_read_failure_after_the_analyzer_decision(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    """BLOCKING-1, primary fix. The analyzer's gate read says full-book
    and skips Opus for a cv6; every LATER account read raises (the
    transient outage that used to make the loop's independent re-read
    fall back to the base threshold and misfile the skip as
    `pending_capacity` — invisible to retro-fill's cv9 floor AND to the
    watchlist). Single-evaluation design: the analyzer's decision is
    persisted at analyze time, so the proposal lands journaled
    `review_skipped_full_book` and RESCUED on the watchlist."""
    fake_broker.positions = [_position("WIDG")]
    _cash_dry(fake_broker)
    account_reads = {"n": 0}
    orig_get_account = fake_broker.get_account

    def _account_fails_after_first_read():
        account_reads["n"] += 1
        if account_reads["n"] > 1:
            raise RuntimeError("transient broker outage")
        return orig_get_account()

    fake_broker.get_account = _account_fails_after_first_read

    await _drive(
        db_path=db_path,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(full_book_gate=True, threshold=5, ks5_max_concurrent=1),
        conviction=6,
    )

    assert _calls(fake_anthropic, "review") == []
    assert _reject_reasons(db_path) == ["review_skipped_full_book"], (
        "the analyzer's skip decision — not a divergent broker re-read — "
        "must classify the missing review"
    )
    rows = get_pending_watchlist(db_path)
    assert len(rows) == 1 and rows[0].symbol == "ACME", (
        "the skipped proposal must be rescued, not stranded"
    )
    # The analyzer's single gate evaluation was the only account read.
    assert account_reads["n"] == 1


@pytest.mark.asyncio
async def test_crash_window_fallback_rescues_pending_capacity_below_retro_floor(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    """BLOCKING-1, belt-and-braces — the reviewer's exact divergence
    scenario, forced through the fallback: the analyzer skips under
    pressure but its recorder never runs (the crash window between store
    and record, simulated by unwiring the seam), and the loop's own gate
    re-read raises (`compute_book_pressure` → None → base threshold).
    The resulting `pending_capacity` row sits below retro-fill's cv9
    floor, so it must ALSO park on the watchlist — and the retry must
    pay for the deferred review before entering (FR-17)."""
    from app.loop import AgentLoop
    from tests.integration.conftest import _opus_ratify_json, _valid_proposal_json

    fake_broker.positions = [_position("WIDG")]
    _cash_dry(fake_broker)
    account_reads = {"n": 0}
    orig_get_account = fake_broker.get_account

    def _account_fails_after_first_read():
        account_reads["n"] += 1
        if account_reads["n"] > 1:
            raise RuntimeError("transient broker outage")
        return orig_get_account()

    fake_broker.get_account = _account_fails_after_first_read

    f = make_filing(accession="0001234567-26-00BLK1")
    insert_prefilter_decision(
        db_path,
        PrefilterDecisionRow(
            filing_id=f.id, decision="accept", rule_fired="material_8k_bypass"
        ),
    )
    queue: asyncio.Queue[FilingRow] = asyncio.Queue()
    await queue.put(f)
    fake_anthropic.responses_by_purpose = {
        "analyze": [_valid_proposal_json(conviction=6, symbol="ACME")],
        "review": [_opus_ratify_json()],
    }
    config = _config(full_book_gate=True, threshold=5, ks5_max_concurrent=1)
    components = build_components(queue=queue, config=config)
    loop_obj = AgentLoop(
        components=components, config=config, shutdown_grace_seconds=10.0
    )
    # Simulate the crash window: the analyzer's skip decision is made but
    # never recorded, so `_execute` must classify it from current truth.
    components.analyzer.set_review_skip_recorder(None)
    await _run_one_filing(loop_obj, queue)

    assert _calls(fake_anthropic, "review") == []
    assert _reject_reasons(db_path) == ["pending_capacity"]
    rows = get_pending_watchlist(db_path)
    assert len(rows) == 1
    assert rows[0].notes == "reject_reason=pending_capacity"

    # Capital frees: the retry pays the deferred review, then enters.
    fake_broker.get_account = orig_get_account
    fake_broker.positions = []
    _cash_flush(fake_broker)
    await loop_obj._process_watchlist(_market_now())
    assert len(_calls(fake_anthropic, "review")) == 1
    assert len(fake_broker.submitted_brackets) == 1
    assert get_pending_watchlist(db_path) == []


@pytest.mark.asyncio
async def test_gate_on_with_watchlist_off_journals_without_crashing(
    db_path, journal, fake_broker, fake_anthropic, build_components, make_filing
) -> None:
    """Runtime belt behind the AppConfig validator (which refuses this
    combo at load time — see tests/config/test_loader.py): if a stub /
    hand-built config reaches the loop with the gate on and the watchlist
    off, nothing crashes, the skip journals `review_skipped_full_book`,
    and no watchlist row is written."""
    fake_broker.positions = [_position("WIDG")]
    _cash_dry(fake_broker)

    await _drive(
        db_path=db_path,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(
            full_book_gate=True,
            threshold=5,
            ks5_max_concurrent=1,
            watchlist_min_conviction=0,
        ),
        conviction=6,
    )

    assert _calls(fake_anthropic, "review") == []
    assert _reject_reasons(db_path) == ["review_skipped_full_book"]
    assert fake_broker.submitted_orders == []
    assert get_pending_watchlist(db_path) == []


@pytest.mark.asyncio
async def test_deferred_review_raising_leaves_row_pending_and_counts_attempts(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    """Advisory #3 — reviewer.review raising mid-retry: the row stays
    pending, nothing is submitted, no extra executions row appears on the
    next tick, and the per-row attempt counter (persisted in notes)
    increments across ticks."""
    fake_broker.positions = [_position("WIDG")]
    _cash_dry(fake_broker)

    loop_obj = await _drive(
        db_path=db_path,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(full_book_gate=True, threshold=5, ks5_max_concurrent=1),
        conviction=6,
    )
    # Capital frees, but Opus is down: the fake raises on an empty queue.
    fake_broker.positions = []
    _cash_flush(fake_broker)
    fake_anthropic.responses_by_purpose["review"] = []

    await loop_obj._process_watchlist(_market_now())
    rows = get_pending_watchlist(db_path)
    assert len(rows) == 1, "row must stay pending after a failed review"
    assert "review_attempts=1" in (rows[0].notes or "")
    assert "reject_reason=review_skipped_full_book" in (rows[0].notes or "")
    assert fake_broker.submitted_brackets == []

    await loop_obj._process_watchlist(_market_now())
    rows = get_pending_watchlist(db_path)
    assert len(rows) == 1
    assert "review_attempts=2" in (rows[0].notes or "")
    assert fake_broker.submitted_brackets == []
    # No double-spend of journal rows: still just the day-0 rejection.
    assert _reject_reasons(db_path) == ["review_skipped_full_book"]
    # Each tick attempted (and failed) exactly one review call.
    assert len(_calls(fake_anthropic, "review")) == 2


@pytest.mark.asyncio
async def test_deferred_review_failures_resolve_terminally_after_five_attempts(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    """Advisory #3 — the attempt budget is 5: the fifth consecutive
    failure resolves the row `review_failed` instead of retrying
    (~390/day) until expiry, and later ticks spend nothing."""
    from app.watchlist import MAX_DEFERRED_REVIEW_ATTEMPTS

    fake_broker.positions = [_position("WIDG")]
    _cash_dry(fake_broker)

    loop_obj = await _drive(
        db_path=db_path,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(full_book_gate=True, threshold=5, ks5_max_concurrent=1),
        conviction=6,
    )
    fake_broker.positions = []
    _cash_flush(fake_broker)
    fake_anthropic.responses_by_purpose["review"] = []

    for _ in range(MAX_DEFERRED_REVIEW_ATTEMPTS):
        await loop_obj._process_watchlist(_market_now())

    assert get_pending_watchlist(db_path) == []
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT status, notes, resolved_at FROM watchlist"
        ).fetchone()
    assert row["status"] == "review_failed"
    assert f"review_attempts={MAX_DEFERRED_REVIEW_ATTEMPTS}" in row["notes"]
    assert row["resolved_at"] is not None
    assert fake_broker.submitted_brackets == []

    # A resolved row is invisible to later ticks — no further spend.
    await loop_obj._process_watchlist(_market_now())
    assert len(_calls(fake_anthropic, "review")) == MAX_DEFERRED_REVIEW_ATTEMPTS


@pytest.mark.asyncio
async def test_deferred_review_reject_resolves_row_terminally(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    """Advisory #4 — a deferred review that comes back reject resolves
    the row `skipped_review_reject` immediately: the verdict won't change
    on a later tick, and a pending row would block the symbol (one
    pending row per symbol) and log daily until expiry."""
    import json as _json

    fake_broker.positions = [_position("WIDG")]
    _cash_dry(fake_broker)

    loop_obj = await _drive(
        db_path=db_path,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(full_book_gate=True, threshold=5, ks5_max_concurrent=1),
        conviction=6,
    )
    fake_broker.positions = []
    _cash_flush(fake_broker)
    fake_anthropic.responses_by_purpose["review"] = [
        _json.dumps(
            {
                "decision": "reject",
                "rationale": (
                    "The filing's catalyst is already reflected in the price "
                    "action and the proposal's stop placement is unsound."
                ),
            }
        )
    ]

    await loop_obj._process_watchlist(_market_now())

    assert fake_broker.submitted_brackets == []
    assert get_pending_watchlist(db_path) == []
    with connect(db_path) as conn:
        row = conn.execute("SELECT status FROM watchlist").fetchone()
        review_rows = conn.execute(
            "SELECT decision FROM proposal_reviews"
        ).fetchall()
    assert row["status"] == "skipped_review_reject"
    assert [r["decision"] for r in review_rows] == ["reject"]
    # Resolved terminally: the next tick spends nothing further.
    await loop_obj._process_watchlist(_market_now())
    assert len(_calls(fake_anthropic, "review")) == 1


@pytest.mark.asyncio
async def test_deferred_review_input_is_capped_like_day_zero(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    """Advisory #5 — the deferred (watchlist) review must see the SAME
    capped filing body a day-0 review would have: cap applied, marker
    appended, over-cap tail dropped."""
    from pathlib import Path

    from analyzer.sonnet import truncation_marker
    from app.loop import AgentLoop
    from tests.integration.conftest import _opus_ratify_json, _valid_proposal_json

    fake_broker.positions = [_position("WIDG")]
    _cash_dry(fake_broker)

    cap = 20_000
    f = make_filing(accession="0001234567-26-00CAP1")
    Path(f.raw_text_path).write_text(
        ("material 8-K item section line\n" * 1_500) + "TAIL_SENTINEL",
        encoding="utf-8",
    )
    insert_prefilter_decision(
        db_path,
        PrefilterDecisionRow(
            filing_id=f.id, decision="accept", rule_fired="material_8k_bypass"
        ),
    )
    queue: asyncio.Queue[FilingRow] = asyncio.Queue()
    await queue.put(f)
    fake_anthropic.responses_by_purpose = {
        "analyze": [_valid_proposal_json(conviction=6, symbol="ACME")],
        "review": [_opus_ratify_json()],
    }
    config = _config(full_book_gate=True, threshold=5, ks5_max_concurrent=1)
    components = build_components(queue=queue, config=config)
    components.analyzer._max_input_chars = cap  # ctor value in prod wiring
    loop_obj = AgentLoop(
        components=components, config=config, shutdown_grace_seconds=10.0
    )
    await _run_one_filing(loop_obj, queue)
    assert len(get_pending_watchlist(db_path)) == 1

    captured: list[str] = []
    orig_review = components.reviewer.review

    async def _spy_review(proposal, filing, raw_text):
        captured.append(raw_text)
        return await orig_review(proposal, filing, raw_text)

    components.reviewer.review = _spy_review  # type: ignore[method-assign]

    fake_broker.positions = []
    _cash_flush(fake_broker)
    await loop_obj._process_watchlist(_market_now())

    assert len(captured) == 1
    marker = truncation_marker(cap)
    assert captured[0].endswith(marker)
    assert "TAIL_SENTINEL" not in captured[0]
    assert len(captured[0]) <= cap + len(marker)
    assert len(fake_broker.submitted_brackets) == 1


@pytest.mark.asyncio
async def test_below_threshold_proposals_never_pay_the_gate_snapshot(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
    monkeypatch,
) -> None:
    """Advisory #7 — the effective bar is always >= the configured one,
    so a cv4 (below the configured 5) can never need review: neither the
    analyzer nor `_execute` may pay the gate's broker snapshot for it."""
    import app.loop as loop_mod
    import execution.review_gate as gate_mod

    fake_broker.positions = [_position("WIDG")]
    _cash_dry(fake_broker)

    resolves = {"n": 0}
    orig_resolve = gate_mod.resolve_review_threshold

    def _counting_resolve(**kwargs):
        resolves["n"] += 1
        return orig_resolve(**kwargs)

    monkeypatch.setattr(gate_mod, "resolve_review_threshold", _counting_resolve)
    monkeypatch.setattr(loop_mod, "resolve_review_threshold", _counting_resolve)

    await _drive(
        db_path=db_path,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(full_book_gate=True, threshold=5, ks5_max_concurrent=1),
        conviction=4,
    )

    assert resolves["n"] == 0, (
        "below-threshold proposals must not trigger the gate's broker reads"
    )
    assert _calls(fake_anthropic, "review") == []
    assert "review_skipped_full_book" not in _reject_reasons(db_path)


@pytest.mark.asyncio
async def test_watchlist_retry_while_still_locked_does_not_spend(
    db_path, journal, fake_broker, fake_anthropic, build_components, make_filing
) -> None:
    """Book still full and cash still dry at retry time — the row stays
    pending and no Opus call is made. The gate is re-evaluated per tick,
    so nothing needs resetting."""
    fake_broker.positions = [_position("WIDG")]
    _cash_dry(fake_broker)

    loop_obj = await _drive(
        db_path=db_path,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        config=_config(full_book_gate=True, threshold=5, ks5_max_concurrent=1),
        conviction=6,
    )
    await loop_obj._process_watchlist(_market_now())

    assert _calls(fake_anthropic, "review") == []
    assert fake_broker.submitted_orders == []
    assert len(get_pending_watchlist(db_path)) == 1

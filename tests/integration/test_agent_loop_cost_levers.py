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

"""Feature B / Feature C HIGH-1 — pending_capacity end-to-end pipeline test.

Pins the contract that when KS-5 is at capacity (broker truth):
- Sonnet runs, emits a high-conviction trade_proposal.
- The analyzer's capacity gate skips Opus (B's cost saving).
- `_execute` writes a `pending_capacity` execution row.
- The broker sees ZERO order submissions (FR-17 protected).
- Feature C's retro-fill query then finds this proposal as a candidate
  on the next slot-open.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.state import AgentState
from broker.protocol import Position
from journal.models import FilingRow
from journal.repo import (
    JournalRepo,
    connect,
)


def _min_config(*, threshold: int = 7) -> SimpleNamespace:
    """A minimal duck-typed config carrying only the fields the loop's
    pending_capacity guard reads. Avoids constructing a full AppConfig
    (which would require dozens of unrelated values)."""
    return SimpleNamespace(
        analyzer=SimpleNamespace(opus_review_conviction_threshold=threshold),
        execution=SimpleNamespace(
            broker_mode="paper",
            ks4_pct_cap=0.20,
            ks4_absolute_cap_usd=100_000.0,
            ks5_max_concurrent=5,
            ks7_cash_reserve_pct=0.05,
            sizing_mid_pct=0.05,
            sizing_high_pct=0.10,
        ),
    )


@pytest.mark.asyncio
async def test_at_capacity_pipeline_writes_pending_capacity_no_broker_submission(
    db_path: str,
    journal: JournalRepo,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    """End-to-end: KS-5 at cap (5/5 broker positions), high-conviction
    8-K arrives. Expected:
      - Sonnet analyze() runs (one llm_calls row).
      - Opus review() does NOT run (B's gate fired). Zero `proposal_reviews`
        rows for this proposal.
      - `_execute` short-circuits at the pending_capacity guard, writes
        `executions.reject_reason='pending_capacity'`.
      - Broker sees ZERO submitted orders (the FR-17 protection)."""
    from app.loop import AgentLoop
    from tests.integration.conftest import _valid_proposal_json

    # Pre-seed the broker with 5 open positions so the KS-5 gate fires.
    fake_broker.positions = [
        Position(
            symbol=f"FILL{i}",
            qty=100,
            avg_entry_price=50.0,
            market_value=5000.0,
            unrealized_pnl=0.0,
        )
        for i in range(5)
    ]

    f = make_filing(accession="0001234567-26-PCAP", cik=320193)
    queue: asyncio.Queue[FilingRow] = asyncio.Queue()

    fake_anthropic.responses_by_purpose = {
        "analyze": [_valid_proposal_json(conviction=9)],
        # No "review" response queued — if Opus is incorrectly invoked,
        # the fake will raise and the test will fail loudly.
        "review": [],
    }

    components = build_components(queue=queue, ks5_max_concurrent=5)
    loop = AgentLoop(
        components=components,
        config=_min_config(threshold=7),
        shutdown_grace_seconds=10.0,
    )

    runner = asyncio.create_task(loop.run())

    async def _wait(predicate, timeout: float = 10.0) -> None:
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if predicate():
                return
            await asyncio.sleep(0.01)
        raise AssertionError("predicate timed out")

    await _wait(lambda: queue.empty() and loop.state == AgentState.IDLE)
    loop.shutdown_requested = True
    await queue.put(None)  # type: ignore[arg-type]
    rc = await asyncio.wait_for(runner, timeout=5.0)
    assert rc == 0

    # Assertions —————————————————————————————————————————————————

    # Exactly one analyze llm_call (Sonnet ran).
    analyze_calls = [c for c in fake_anthropic.calls if c["purpose"] == "analyze"]
    assert len(analyze_calls) == 1

    # ZERO review llm_calls (B's gate skipped Opus).
    review_calls = [c for c in fake_anthropic.calls if c["purpose"] == "review"]
    assert review_calls == [], (
        "Opus must NOT be invoked when KS-5 is at capacity (Feature B)"
    )

    # Exactly one trade_proposal row, no proposal_reviews row.
    with connect(db_path) as conn:
        proposal_rows = conn.execute(
            "SELECT id, kind, conviction FROM proposals "
            "WHERE kind = 'trade_proposal'"
        ).fetchall()
        review_count = conn.execute(
            "SELECT COUNT(*) FROM proposal_reviews"
        ).fetchone()[0]
    assert len(proposal_rows) == 1
    pid = proposal_rows[0]["id"]
    assert proposal_rows[0]["conviction"] == 9
    assert review_count == 0

    # Exactly one execution row with reject_reason='pending_capacity'.
    # CRITICAL: this is the deterministic signal Feature C keys on.
    with connect(db_path) as conn:
        executions = conn.execute(
            "SELECT decision, reject_reason FROM executions "
            "WHERE proposal_id = ?",
            (pid,),
        ).fetchall()
    assert len(executions) == 1
    assert executions[0]["decision"] == "rejected"
    assert executions[0]["reject_reason"] == "pending_capacity"

    # FR-17 protection: broker saw ZERO order submissions.
    assert fake_broker.submitted_orders == [], (
        "Capacity-blocked proposal must NOT reach the broker (FR-17)"
    )


@pytest.mark.asyncio
async def test_at_capacity_pipeline_feeds_feature_c_retro_query(
    db_path: str,
    journal: JournalRepo,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    """The pending_capacity row written by HIGH-1's guard is exactly
    the signal `find_retro_candidate` keys on. Pins the B → C
    composition: a pipeline run at capacity produces a row that the
    retro query immediately finds."""
    from app.loop import AgentLoop
    from journal.repo import find_retro_candidate
    from tests.integration.conftest import _valid_proposal_json

    fake_broker.positions = [
        Position(
            symbol=f"FILL{i}",
            qty=10,
            avg_entry_price=50.0,
            market_value=500.0,
            unrealized_pnl=0.0,
        )
        for i in range(5)
    ]

    f = make_filing(accession="0001234567-26-PCAP2", cik=320193)
    queue: asyncio.Queue[FilingRow] = asyncio.Queue()
    fake_anthropic.responses_by_purpose = {
        "analyze": [_valid_proposal_json(conviction=9)],
        "review": [],
    }

    components = build_components(queue=queue, ks5_max_concurrent=5)
    loop = AgentLoop(
        components=components,
        config=_min_config(threshold=7),
        shutdown_grace_seconds=10.0,
    )

    runner = asyncio.create_task(loop.run())

    async def _wait(predicate, timeout: float = 10.0) -> None:
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if predicate():
                return
            await asyncio.sleep(0.01)
        raise AssertionError("predicate timed out")

    await _wait(lambda: queue.empty() and loop.state == AgentState.IDLE)
    loop.shutdown_requested = True
    await queue.put(None)  # type: ignore[arg-type]
    await asyncio.wait_for(runner, timeout=5.0)

    # Feature C's retro query finds the just-written proposal.
    candidate = find_retro_candidate(
        db_path,
        min_conviction=9,
        not_older_than=dt.datetime.now(dt.UTC) - dt.timedelta(hours=6),
    )
    assert candidate is not None
    assert candidate.kind == "trade_proposal"
    assert candidate.conviction == 9
    assert candidate.symbol == "ACME"

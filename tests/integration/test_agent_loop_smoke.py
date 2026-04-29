"""S5.6 AC-9 — agent-loop smoke integration test (load-bearing).

Runs the real loop against fakes for the external boundaries (broker,
Anthropic), with three pre-loaded fixture filings:

  (a) high-conviction proposal-emitting 8-K → analyzer → Opus → execution.
  (b) no-trade 8-K → analyzer → no execution.
  (c) prefilter-rejected 8-K (out-of-universe CIK) → no analyzer call.

Asserts the journal converges on the expected row counts, the broker
saw exactly one entry-stop pair (no take-profit since the canned
proposal omits it), and the state-transition sequence matches the
story's stated invariant (one PROCESSING per filing).
"""

from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path

import pytest

from app.state import AgentState
from journal.models import FilingRow
from journal.repo import (
    JournalRepo,
    connect,
    get_proposal_by_decision_id,
)


@pytest.mark.asyncio
async def test_agent_loop_smoke(
    db_path: str,
    journal: JournalRepo,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
    monkeypatch,
) -> None:
    from app.loop import AgentLoop
    from tests.integration.conftest import (
        _no_trade_json,
        _opus_ratify_json,
        _valid_proposal_json,
    )

    # Three fixture filings — written to the journal by the (simulated)
    # detail fetcher S3.3. The agent loop's BOOTING crash-recovery scan
    # picks up any filing without a prefilter_decisions row and feeds it
    # into the consumer queue (AC-7) — that's how the smoke fixture
    # delivers work, mirroring production.
    f_proposal = make_filing(accession="0001234567-26-000001", cik=320193)
    f_no_trade = make_filing(
        accession="0001234567-26-000002",
        cik=320193,
        item_codes='["8.01"]',  # 8.01 alone is allow-list per S4.1;
                                  # analyzer returns no_trade per the
                                  # canned response below.
    )
    f_rejected = make_filing(
        accession="0001234567-26-000003",
        cik=999999,  # NOT in universe → prefilter rejects.
    )

    queue: asyncio.Queue[FilingRow] = asyncio.Queue()

    fake_anthropic.responses_by_purpose = {
        "analyze": [_valid_proposal_json(conviction=9), _no_trade_json()],
        "review": [_opus_ratify_json()],
    }

    components = build_components(queue=queue)
    loop = AgentLoop(components=components, shutdown_grace_seconds=10.0)

    state_log: list[AgentState] = []
    monkey_state_observer(loop, state_log)

    # Drive the loop until queue drained, then signal shutdown.
    runner = asyncio.create_task(loop.run())
    # Wait for the queue to drain. The loop blocks on `queue.get()` after
    # consuming all three filings — flip shutdown so it exits cleanly.
    await _wait_for(lambda: queue.empty() and loop.state == AgentState.IDLE, timeout=10.0)
    loop.shutdown_requested = True
    # Push a sentinel so the consumer wakes from `queue.get()`.
    await queue.put(None)  # type: ignore[arg-type]
    rc = await asyncio.wait_for(runner, timeout=10.0)
    assert rc == 0

    # Journal-level assertions (AC-9).
    with connect(db_path) as conn:
        prefilter_count = conn.execute(
            "SELECT COUNT(*) FROM prefilter_decisions"
        ).fetchone()[0]
        proposal_rows = conn.execute(
            "SELECT kind, decision_id FROM proposals ORDER BY id ASC"
        ).fetchall()
        review_count = conn.execute(
            "SELECT COUNT(*) FROM proposal_reviews"
        ).fetchone()[0]
        execution_rows = conn.execute(
            "SELECT decision, reject_reason FROM executions ORDER BY id ASC"
        ).fetchall()

    assert prefilter_count == 3, "every filing produces a prefilter_decisions row"
    kinds = sorted([r["kind"] for r in proposal_rows])
    assert kinds == ["no_trade", "trade_proposal"], (
        "two proposals: high-conviction trade + no-trade (rejected filing "
        "never reaches analyzer)"
    )
    assert review_count == 1, "one Opus review (high-conviction proposal only)"
    assert len(execution_rows) == 1, (
        "exactly one execution row — the high-conviction proposal that "
        "ratified through Opus"
    )
    assert execution_rows[0]["decision"] == "accepted"

    # Broker-level assertions: entry + stop = 2 orders (no take_profit
    # in the canned proposal).
    assert len(fake_broker.submitted_orders) == 2
    sides = [o.side for o in fake_broker.submitted_orders]
    assert sides == ["buy", "sell"]  # entry then stop

    # decision_id uniqueness across filings (AC-9).
    decision_ids = {r["decision_id"] for r in proposal_rows}
    assert len(decision_ids) == len(proposal_rows)

    # State-transition sequence: one PROCESSING per filing.
    processing_count = sum(1 for s in state_log if s == AgentState.PROCESSING)
    assert processing_count == 3, f"expected 3 PROCESSING entries, got {state_log}"


def monkey_state_observer(loop, sink: list) -> None:
    """Attach an observer that records every state change."""
    orig_setter = type(loop)._set_state  # type: ignore[attr-defined]

    def _wrapped(self, new):
        sink.append(new)
        return orig_setter(self, new)

    loop._set_state_observer = sink  # type: ignore[attr-defined]
    # Patch instance-level by monkey-overriding _set_state.
    import types
    loop._set_state = types.MethodType(_wrapped, loop)


async def _wait_for(predicate, *, timeout: float) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise asyncio.TimeoutError(f"predicate did not become true within {timeout}s")

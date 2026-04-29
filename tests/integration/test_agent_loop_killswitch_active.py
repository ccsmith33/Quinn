"""S5.6 AC-11 — kill-switch-active integration test (load-bearing).

Pre-seeds a halted kill-switch row. Verifies that ingestion +
prefilter + analyzer + Opus review still run (FR-32 — journal must
remain complete), but no broker submit_order is called and the
execution row is recorded with `decision="rejected",
reject_reason="kill_switch"`.

Defense-in-depth WARN log: the loop must emit
`event="killswitch_blocked_proposal"` per AC-5 step 5.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging

import pytest

from journal.models import FilingRow, KillSwitchStateRow
from journal.repo import connect, insert_kill_switch_state


@pytest.mark.asyncio
async def test_agent_loop_killswitch_active_blocks_submission(
    db_path: str,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
    caplog,
) -> None:
    from app.loop import AgentLoop
    from app.state import AgentState
    from tests.integration.conftest import (
        _opus_ratify_json,
        _valid_proposal_json,
    )

    # Pre-seed kill-switch as halted (manual:telegram).
    insert_kill_switch_state(
        db_path,
        KillSwitchStateRow(
            set_at=dt.datetime.now(dt.UTC),
            state="halted",
            reason="manual:telegram",
            set_by="operator",
        ),
    )

    f = make_filing(accession="0001234567-26-000020", cik=320193)
    queue: asyncio.Queue[FilingRow] = asyncio.Queue()
    await queue.put(f)

    fake_anthropic.responses_by_purpose = {
        "analyze": [_valid_proposal_json(conviction=9)],
        "review": [_opus_ratify_json()],
    }

    components = build_components(queue=queue)
    loop = AgentLoop(components=components, shutdown_grace_seconds=10.0)

    caplog.set_level(logging.WARNING, logger="app.loop")
    runner = asyncio.create_task(loop.run())
    await _wait_for(lambda: queue.empty() and loop.state == AgentState.IDLE, timeout=10.0)
    loop.shutdown_requested = True
    await queue.put(None)  # type: ignore[arg-type]
    rc = await asyncio.wait_for(runner, timeout=10.0)
    assert rc == 0

    with connect(db_path) as conn:
        prefilter_count = conn.execute(
            "SELECT COUNT(*) FROM prefilter_decisions"
        ).fetchone()[0]
        proposal_row = conn.execute(
            "SELECT id, kind FROM proposals"
        ).fetchone()
        review_row = conn.execute(
            "SELECT id, decision FROM proposal_reviews"
        ).fetchone()
        exec_row = conn.execute(
            "SELECT decision, reject_reason FROM executions"
        ).fetchone()

    # Filing was prefiltered, analyzed, opus-reviewed (journal complete).
    assert prefilter_count == 1
    assert proposal_row is not None and proposal_row["kind"] == "trade_proposal"
    assert review_row is not None and review_row["decision"] == "ratify"
    assert exec_row is not None
    assert exec_row["decision"] == "rejected"
    assert exec_row["reject_reason"] == "kill_switch"

    # Broker NOT called.
    assert fake_broker.submitted_orders == []

    # WARN log emitted.
    matching = [
        r for r in caplog.records
        if getattr(r, "event", None) == "killswitch_blocked_proposal"
    ]
    assert len(matching) >= 1, (
        "expected event='killswitch_blocked_proposal' WARN — got "
        f"{[r.getMessage() for r in caplog.records]}"
    )


async def _wait_for(predicate, *, timeout: float) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise asyncio.TimeoutError(f"predicate did not become true within {timeout}s")

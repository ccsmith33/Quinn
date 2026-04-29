"""S5.6 AC-15 — state transitions emit structured logs.

Each transition emits an event with `from_state` and `to_state`
fields; the per-filing pipeline binds a `correlation_id` for the
duration of one iteration.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from journal.models import FilingRow, PrefilterDecisionRow
from journal.repo import insert_prefilter_decision


@pytest.mark.asyncio
async def test_state_transitions_emit_structured_logs(
    db_path: str,
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

    f = make_filing(accession="0001234567-26-000700", cik=320193)
    insert_prefilter_decision(
        db_path,
        PrefilterDecisionRow(
            filing_id=f.id,
            decision="accept",
            rule_fired="material_8k_bypass",
        ),
    )

    queue: asyncio.Queue[FilingRow] = asyncio.Queue()
    await queue.put(f)

    fake_anthropic.responses_by_purpose = {
        "analyze": [_valid_proposal_json(conviction=9)],
        "review": [_opus_ratify_json()],
    }

    components = build_components(queue=queue)
    loop = AgentLoop(components=components, shutdown_grace_seconds=10.0)

    caplog.set_level(logging.INFO, logger="app.loop")
    runner = asyncio.create_task(loop.run())
    while True:
        if queue.empty() and loop.state == AgentState.IDLE:
            break
        await asyncio.sleep(0.01)
    loop.shutdown_requested = True
    await queue.put(None)  # type: ignore[arg-type]
    await asyncio.wait_for(runner, timeout=10.0)

    transitions = [
        r for r in caplog.records
        if getattr(r, "event", None) == "agent.state_transition"
    ]
    assert len(transitions) >= 4, (
        "expected at least four transition events: "
        "BOOTING→IDLE, IDLE→PROCESSING, PROCESSING→IDLE, IDLE→SHUTTING_DOWN, "
        "SHUTTING_DOWN→STOPPED — got "
        f"{[(r.from_state, r.to_state) for r in transitions]}"
    )
    # Every transition record carries from_state + to_state attrs.
    for r in transitions:
        assert hasattr(r, "from_state")
        assert hasattr(r, "to_state")

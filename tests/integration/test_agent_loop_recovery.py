"""S5.6 AC-12 — crash-recovery boot scan (load-bearing).

Pre-seeds the journal with one filing missing its prefilter row and one
proposal missing its execution row. Boots the loop with an empty
queue. Asserts both backlog items are processed BEFORE the loop reaches
IDLE (they are fed in ahead of the empty RSS-side queue per story
"Crash recovery rule").
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json

import pytest

from journal.models import (
    FilingRow,
    PromptRow,
    ProposalRow,
)
from journal.repo import (
    connect,
    insert_prompt,
    insert_proposal,
)


@pytest.mark.asyncio
async def test_agent_loop_crash_recovery(
    db_path: str,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    from app.loop import AgentLoop
    from app.state import AgentState
    from tests.integration.conftest import (
        _opus_ratify_json,
        _valid_proposal_json,
    )

    # --- Backlog item 1: a filing with no prefilter_decisions row ----
    pending_filing = make_filing(
        accession="0001234567-26-000030", cik=320193
    )
    # No prefilter row — simulates a crash mid-prefilter.

    # --- Backlog item 2: a proposal with no executions row -----------
    # We need a filing + a registered prompt-version + a proposal row.
    pending_filing_2 = make_filing(
        accession="0001234567-26-000031", cik=320193
    )
    pv = "sonnet_filing_analysis_v1@deadbeef0001"
    insert_prompt(
        db_path,
        PromptRow(
            prompt_version=pv,
            name="sonnet_filing_analysis_v1",
            file_path="src/prompts/sonnet_filing_analysis_v1.txt",
            content_hash="deadbeef0001" + "0" * 52,
        ),
    )
    valid_payload = json.loads(_valid_proposal_json(conviction=9))
    proposal_id = insert_proposal(
        db_path,
        ProposalRow(
            filing_id=pending_filing_2.id,
            decision_id="dec-recovery-001",
            model_id="claude-sonnet-4-6",
            prompt_version=pv,
            raw_response=_valid_proposal_json(conviction=9),
            kind="trade_proposal",
            symbol=valid_payload["symbol"],
            direction=valid_payload["direction"],
            size_pct_requested=valid_payload["size_pct_of_capital"],
            conviction=valid_payload["conviction"],
            thesis=valid_payload["thesis"],
            input_tokens=1500,
            output_tokens=800,
            cache_read_tokens=4000,
            cache_creation_tokens=200,
            latency_ms=1234,
            cost_usd=0.0234,
        ),
    )
    # Also need the prefilter row for that filing (so recovery doesn't
    # also re-prefilter it).
    from journal.models import PrefilterDecisionRow
    from journal.repo import insert_prefilter_decision
    insert_prefilter_decision(
        db_path,
        PrefilterDecisionRow(
            filing_id=pending_filing_2.id,
            decision="accept",
            rule_fired="material_8k_bypass",
        ),
    )

    queue: asyncio.Queue[FilingRow] = asyncio.Queue()  # empty RSS side

    # Recovery for pending_filing will re-prefilter (allow-list 1.01 →
    # accept) and then analyze. We need an analyze-canned response.
    fake_anthropic.responses_by_purpose = {
        "analyze": [_valid_proposal_json(conviction=9)],
        "review": [_opus_ratify_json()],
    }

    components = build_components(queue=queue)
    loop = AgentLoop(components=components, shutdown_grace_seconds=10.0)

    runner = asyncio.create_task(loop.run())

    # The loop's BOOTING phase should detect the backlog and feed both
    # pending items into the consumer queue. Wait for the journal to
    # show: 2 prefilter rows total + at least 2 execution rows.
    await _wait_for(
        lambda: _journal_complete(db_path),
        timeout=10.0,
    )
    loop.shutdown_requested = True
    await queue.put(None)  # type: ignore[arg-type]
    rc = await asyncio.wait_for(runner, timeout=10.0)
    assert rc == 0

    with connect(db_path) as conn:
        prefilter_count = conn.execute(
            "SELECT COUNT(*) FROM prefilter_decisions"
        ).fetchone()[0]
        executions_count = conn.execute(
            "SELECT COUNT(*) FROM executions"
        ).fetchone()[0]
        # The pre-existing proposal_id must now have an execution row.
        exec_for_existing = conn.execute(
            "SELECT decision FROM executions WHERE proposal_id = ?",
            (proposal_id,),
        ).fetchone()

    assert prefilter_count == 2  # one new (pending_filing), one pre-seeded
    assert executions_count >= 2
    assert exec_for_existing is not None


def _journal_complete(db_path: str) -> bool:
    with connect(db_path) as conn:
        prefilters = conn.execute(
            "SELECT COUNT(*) FROM prefilter_decisions"
        ).fetchone()[0]
        execs = conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0]
    return prefilters >= 2 and execs >= 2


async def _wait_for(predicate, *, timeout: float) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise asyncio.TimeoutError(f"predicate did not become true within {timeout}s")

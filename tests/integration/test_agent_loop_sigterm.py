"""S5.6 AC-10 — SIGTERM-during-processing test (load-bearing).

The test fires SIGTERM while a filing is mid-analyze. The in-flight
filing must complete; the next filing in the queue must NOT be
consumed; the loop must exit 0 within the grace window.
"""

from __future__ import annotations

import asyncio
import os
import signal

import pytest

from journal.models import FilingRow
from journal.repo import connect


@pytest.mark.asyncio
async def test_agent_loop_sigterm_completes_in_flight(
    db_path: str,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    from app.loop import AgentLoop
    from tests.integration.conftest import (
        _opus_ratify_json,
        _valid_proposal_json,
    )

    f1 = make_filing(accession="0001234567-26-000010", cik=320193)
    f2 = make_filing(accession="0001234567-26-000011", cik=320193)
    f3 = make_filing(accession="0001234567-26-000012", cik=320193)

    # Pre-seed prefilter rows so the BOOTING crash-recovery scan does
    # not also re-queue these filings (the test wants strict control
    # over consumption ordering — see queue.put calls below). The
    # decision "accept" via material_8k_bypass keeps the analyzer path
    # live; that's what we want to delay-instrument with f2.
    from journal.models import PrefilterDecisionRow
    from journal.repo import insert_prefilter_decision
    for f in (f1, f2, f3):
        insert_prefilter_decision(
            db_path,
            PrefilterDecisionRow(
                filing_id=f.id,
                decision="accept",
                rule_fired="material_8k_bypass",
                reason_detail='item_codes=["1.01"]',
            ),
        )

    queue: asyncio.Queue[FilingRow] = asyncio.Queue()
    await queue.put(f1)
    await queue.put(f2)
    await queue.put(f3)

    # Two analyze-purpose responses (we'll only consume two filings) +
    # two opus reviews.
    fake_anthropic.responses_by_purpose = {
        "analyze": [
            _valid_proposal_json(conviction=9),
            _valid_proposal_json(conviction=9),
        ],
        "review": [_opus_ratify_json(), _opus_ratify_json()],
    }
    # Make the SECOND filing's analyze call sleep so SIGTERM lands during
    # the second pipeline. The decision_id is computed deterministically;
    # we precompute it for f2 to set the delay.
    from analyzer.sonnet import compute_decision_id
    from prompts.loader import PromptBuilder
    from pathlib import Path

    from prompts.loader import ACTIVE_SONNET_ANALYSIS_PROMPT

    pb = PromptBuilder(Path("src/prompts"))
    pv = pb.prompt_version(ACTIVE_SONNET_ANALYSIS_PROMPT)
    f2_decision_id = compute_decision_id(
        filing_id=f2.id, model_id="claude-sonnet-4-6", prompt_version=pv
    )
    fake_anthropic.delay_seconds_by_decision_id[f2_decision_id] = 1.5

    components = build_components(queue=queue)
    loop = AgentLoop(components=components, shutdown_grace_seconds=10.0)

    runner = asyncio.create_task(loop.run())

    # Wait until the loop is mid-processing the second filing, then SIGTERM.
    await asyncio.sleep(0.4)  # f1 finishes fast; we want to be inside f2's sleep
    os.kill(os.getpid(), signal.SIGTERM)

    rc = await asyncio.wait_for(runner, timeout=10.0)
    assert rc == 0

    # Journal must reflect: f1 + f2 both went through analyzer +
    # execution; f3 must NOT have been consumed (prefilter rows for all
    # three were pre-seeded to disable the recovery scan).
    with connect(db_path) as conn:
        proposals_count = conn.execute("SELECT COUNT(*) FROM proposals").fetchone()[0]
        executions_count = conn.execute(
            "SELECT COUNT(*) FROM executions"
        ).fetchone()[0]
        executed_filing_ids = {
            r["filing_id"] for r in conn.execute(
                "SELECT pr.filing_id FROM proposals pr "
                "JOIN executions e ON e.proposal_id = pr.id"
            ).fetchall()
        }

    assert proposals_count == 2, (
        f"only f1 and f2 should have proposals (f3 not consumed); got {proposals_count}"
    )
    assert executions_count == 2, (
        "both fully-processed proposals must have execution rows — no orphan proposal"
    )
    assert f3.id not in executed_filing_ids
    # f3 is still in the queue.
    assert queue.qsize() == 1

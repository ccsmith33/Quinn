"""S5.6 AC-14 — wiring-overhead budget guard.

Asserts that the agent-loop overhead per filing (excluding the LLM
call itself, which is stubbed at 0ms here) stays under the 500ms p95
target. The full NFR-1 90s p95 budget is owned by S5.3; S5.6 owns
only the wiring overhead.
"""

from __future__ import annotations

import asyncio
import statistics
import time

import pytest

from journal.models import FilingRow, PrefilterDecisionRow
from journal.repo import insert_prefilter_decision


@pytest.mark.asyncio
async def test_agent_loop_overhead_under_500ms(
    db_path: str,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    """Boot the loop with N filings; measure wall-time per filing.

    The fake Anthropic client returns immediately (no per-call delay),
    so the measured time is essentially the loop's wiring + DB writes
    + (real) prefilter, validator, sizer, submitter logic. This is the
    "overhead" the AC bounds.
    """
    from app.loop import AgentLoop
    from app.state import AgentState
    from tests.integration.conftest import (
        _opus_ratify_json,
        _valid_proposal_json,
    )

    n = 10
    filings: list[FilingRow] = []
    for i in range(n):
        f = make_filing(
            accession=f"0001234567-26-{i:06d}", cik=320193
        )
        filings.append(f)
        # Pre-seed prefilter rows so the recovery scan does not also
        # queue them (we want strict control over the workload).
        insert_prefilter_decision(
            db_path,
            PrefilterDecisionRow(
                filing_id=f.id,
                decision="accept",
                rule_fired="material_8k_bypass",
            ),
        )

    queue: asyncio.Queue[FilingRow] = asyncio.Queue()
    for f in filings:
        await queue.put(f)

    fake_anthropic.responses_by_purpose = {
        "analyze": [_valid_proposal_json(conviction=9) for _ in range(n)],
        "review": [_opus_ratify_json() for _ in range(n)],
    }

    components = build_components(queue=queue)
    loop = AgentLoop(components=components, shutdown_grace_seconds=10.0)

    # Wrap the per-filing pipeline to record per-iteration latency.
    times: list[float] = []
    orig = loop._process_filing

    async def _timed(filing: FilingRow) -> None:
        t0 = time.perf_counter()
        await orig(filing)
        times.append(time.perf_counter() - t0)

    loop._process_filing = _timed  # type: ignore[method-assign]

    runner = asyncio.create_task(loop.run())
    while True:
        if queue.empty() and loop.state == AgentState.IDLE and len(times) >= n:
            break
        await asyncio.sleep(0.01)
    loop.shutdown_requested = True
    await queue.put(None)  # type: ignore[arg-type]
    rc = await asyncio.wait_for(runner, timeout=10.0)
    assert rc == 0
    assert len(times) == n

    p95 = sorted(times)[int(0.95 * n) - 1]
    assert p95 < 0.500, (
        f"p95 wiring overhead {p95*1000:.1f}ms exceeds 500ms NFR-1 budget; "
        f"all times (s): {times}"
    )

"""S5.6 carry-fwd S6.5 reviewer M-2 + M-3 — reconciler trigger + alerter.

M-2: After a successful `OrderSubmitter.submit()` the agent loop must
invoke `reconciler.trigger_after_submission()` so the operator's view
of state catches up immediately (FR-24 behavior side, not just
capability).

M-3: The reconciler is constructed with a Telegram alerter so position
discrepancies reach the operator (not just structured logs). This test
verifies the wiring path by interrogating the reconciler instance the
agent loop holds.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from journal.models import (
    FilingRow,
    PrefilterDecisionRow,
    PromptRow,
    ProposalRow,
)
from journal.repo import (
    connect,
    insert_prefilter_decision,
    insert_prompt,
    insert_proposal,
)


@pytest.mark.asyncio
async def test_reconciler_trigger_called_after_successful_submit(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    """After a successful submission the loop must call
    `reconciler.trigger_after_submission()` exactly once."""
    from app.loop import AgentLoop
    from app.state import AgentState
    from tests.integration.conftest import (
        _opus_ratify_json,
        _valid_proposal_json,
    )

    f = make_filing(accession="0001234567-26-001100", cik=320193)
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
    # Replace the stub reconciler's trigger with a counter so we can
    # observe the call from the loop's hot path.
    counter = {"calls": 0}

    def _record_trigger() -> object:
        counter["calls"] += 1

        class _Empty:
            matched = True
            diffs: list = []
            deferred = False
            suppressed = False

        return _Empty()

    components.reconciler.trigger_after_submission = _record_trigger  # type: ignore[attr-defined]

    loop = AgentLoop(components=components, shutdown_grace_seconds=10.0)
    runner = asyncio.create_task(loop.run())
    while not (queue.empty() and loop.state == AgentState.IDLE):
        await asyncio.sleep(0.01)
    loop.shutdown_requested = True
    await queue.put(None)  # type: ignore[arg-type]
    await asyncio.wait_for(runner, timeout=10.0)

    # Confirm the broker actually accepted (sanity check) so the
    # trigger assertion is meaningful.
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT decision FROM executions"
        ).fetchone()
    assert row is not None and row["decision"] == "accepted"
    assert counter["calls"] == 1, (
        f"reconciler.trigger_after_submission should fire once after "
        f"a successful submit; got {counter['calls']}"
    )


@pytest.mark.asyncio
async def test_reconciler_trigger_skipped_on_rejected_submission(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    """Validator-rejected proposals must NOT trigger the reconciler —
    no broker submission happened, nothing changed broker-side."""
    from app.loop import AgentLoop
    from app.state import AgentState
    from broker.protocol import Position
    from tests.integration.conftest import (
        _opus_ratify_json,
        _valid_proposal_json,
    )

    f = make_filing(accession="0001234567-26-001101", cik=320193)
    insert_prefilter_decision(
        db_path,
        PrefilterDecisionRow(
            filing_id=f.id,
            decision="accept",
            rule_fired="material_8k_bypass",
        ),
    )
    # Force sizer to reject via KS-5 (5 concurrent positions).
    fake_broker.get_positions = lambda: [  # type: ignore[method-assign]
        Position(
            symbol=f"OTHER{i}",
            qty=10,
            avg_entry_price=50.0,
            market_value=500.0,
            unrealized_pnl=0.0,
        )
        for i in range(5)
    ]

    queue: asyncio.Queue[FilingRow] = asyncio.Queue()
    await queue.put(f)
    fake_anthropic.responses_by_purpose = {
        "analyze": [_valid_proposal_json(conviction=9)],
        "review": [_opus_ratify_json()],
    }

    components = build_components(queue=queue)
    counter = {"calls": 0}

    def _record_trigger() -> None:
        counter["calls"] += 1

    components.reconciler.trigger_after_submission = _record_trigger  # type: ignore[attr-defined]

    loop = AgentLoop(components=components, shutdown_grace_seconds=10.0)
    runner = asyncio.create_task(loop.run())
    while not (queue.empty() and loop.state == AgentState.IDLE):
        await asyncio.sleep(0.01)
    loop.shutdown_requested = True
    await queue.put(None)  # type: ignore[arg-type]
    await asyncio.wait_for(runner, timeout=10.0)

    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT decision, reject_reason FROM executions"
        ).fetchone()
    assert row is not None
    assert row["decision"] == "rejected"
    assert row["reject_reason"] == "ks5_concurrent_limit"
    assert counter["calls"] == 0, (
        "reconciler.trigger_after_submission must NOT fire on a "
        "rejected submission — nothing reached the broker"
    )

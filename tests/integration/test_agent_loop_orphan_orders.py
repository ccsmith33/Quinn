"""S5.6 carry-fwd S6.4 reviewer M-4 — cross-process duplicate-order safety.

Simulates the failure mode: a previous agent process submitted entry +
stop to the broker but crashed before writing the `executions` row.
On restart, the broker still has both orders under their deterministic
`prop-{id}-{role}` client_order_ids. The agent loop MUST detect this
and adopt the broker's orders into the journal rather than re-submit
(which would create duplicate orders with new ids).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json

import pytest

from broker.protocol import SubmittedOrder
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
async def test_recovery_adopts_orphan_orders_without_resubmitting(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    from app.loop import AgentLoop
    from app.state import AgentState
    from tests.integration.conftest import _valid_proposal_json

    # --- pre-state: a proposal exists with no execution row -----------
    filing = make_filing(accession="0001234567-26-000900", cik=320193)
    insert_prefilter_decision(
        db_path,
        PrefilterDecisionRow(
            filing_id=filing.id,
            decision="accept",
            rule_fired="material_8k_bypass",
        ),
    )
    pv = "sonnet_filing_analysis_v1@orphantest01"
    insert_prompt(
        db_path,
        PromptRow(
            prompt_version=pv,
            name="sonnet_filing_analysis_v1",
            file_path="src/prompts/sonnet_filing_analysis_v1.txt",
            content_hash="orphantest01" + "0" * 52,
        ),
    )
    payload = json.loads(_valid_proposal_json(conviction=9))
    proposal_id = insert_proposal(
        db_path,
        ProposalRow(
            filing_id=filing.id,
            decision_id="dec-orphan-001",
            model_id="claude-sonnet-4-6",
            prompt_version=pv,
            raw_response=_valid_proposal_json(conviction=9),
            kind="trade_proposal",
            symbol=payload["symbol"],
            direction=payload["direction"],
            size_pct_requested=payload["size_pct_of_capital"],
            conviction=payload["conviction"],
            thesis=payload["thesis"],
            input_tokens=1500,
            output_tokens=800,
            cache_read_tokens=4000,
            cache_creation_tokens=200,
            latency_ms=1234,
            cost_usd=0.0234,
        ),
    )

    # --- simulate "broker remembers" state ---------------------------
    # The previous process submitted entry + stop but crashed before
    # writing executions. The broker still has both orders under their
    # deterministic client_order_ids.
    now = dt.datetime.now(dt.UTC)
    fake_broker.preseeded_by_client_id[f"prop-{proposal_id}-entry"] = SubmittedOrder(
        broker_order_id="orphan-entry-001",
        client_order_id=f"prop-{proposal_id}-entry",
        symbol="ACME",
        side="buy",
        qty=10,
        order_type="market",
        status="accepted",
        submitted_at=now,
    )
    fake_broker.preseeded_by_client_id[f"prop-{proposal_id}-stop"] = SubmittedOrder(
        broker_order_id="orphan-stop-001",
        client_order_id=f"prop-{proposal_id}-stop",
        symbol="ACME",
        side="sell",
        qty=10,
        order_type="stop",
        status="accepted",
        submitted_at=now,
        stop_price=9.50,
    )

    queue: asyncio.Queue[FilingRow] = asyncio.Queue()  # empty RSS side
    components = build_components(queue=queue)
    loop = AgentLoop(components=components, shutdown_grace_seconds=10.0)

    runner = asyncio.create_task(loop.run())
    # Wait until the recovery scan + adoption finishes (executions row
    # appears for the orphan proposal).
    while True:
        with connect(db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM executions WHERE proposal_id = ?", (proposal_id,)
            ).fetchone()
        if row is not None and loop.state == AgentState.IDLE:
            break
        await asyncio.sleep(0.01)
    loop.shutdown_requested = True
    await queue.put(None)  # type: ignore[arg-type]
    await asyncio.wait_for(runner, timeout=10.0)

    # CRITICAL ASSERTION: broker.submit_order MUST NOT have been
    # called. The duplicate-order risk is exactly what M-4 guards.
    assert fake_broker.submitted_orders == [], (
        "agent re-submitted orders on recovery — duplicate-order bug; "
        f"submitted: {fake_broker.submitted_orders}"
    )

    # And the journal must contain the adopted execution + order rows
    # built from broker truth.
    with connect(db_path) as conn:
        exec_row = conn.execute(
            "SELECT decision, submitted_orders_json FROM executions "
            "WHERE proposal_id = ?",
            (proposal_id,),
        ).fetchone()
        order_rows = conn.execute(
            "SELECT role, broker_order_id, notes FROM orders "
            "WHERE execution_id = (SELECT id FROM executions WHERE proposal_id = ?)",
            (proposal_id,),
        ).fetchall()

    assert exec_row is not None
    assert exec_row["decision"] == "accepted"
    submitted = json.loads(exec_row["submitted_orders_json"])
    submitted_broker_ids = {s["broker_order_id"] for s in submitted}
    assert "orphan-entry-001" in submitted_broker_ids
    assert "orphan-stop-001" in submitted_broker_ids

    roles = {r["role"]: r for r in order_rows}
    assert "entry" in roles
    assert "stop" in roles
    assert roles["entry"]["broker_order_id"] == "orphan-entry-001"
    assert roles["stop"]["broker_order_id"] == "orphan-stop-001"
    # Adoption marker on every leg.
    for r in order_rows:
        assert r["notes"] == "adopted_from_broker_on_recovery"


@pytest.mark.asyncio
async def test_orphan_entry_only_halts_killswitch(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    """Adoption path with entry-only (no stop at broker) flips the
    kill-switch to halted with reason 'submission_partial_no_stop' —
    mirrors S6.4's failure-handling vocabulary."""
    from app.loop import AgentLoop
    from app.state import AgentState
    from tests.integration.conftest import _valid_proposal_json
    from journal.repo import get_latest_kill_switch_state

    filing = make_filing(accession="0001234567-26-000901", cik=320193)
    insert_prefilter_decision(
        db_path,
        PrefilterDecisionRow(
            filing_id=filing.id,
            decision="accept",
            rule_fired="material_8k_bypass",
        ),
    )
    pv = "sonnet_filing_analysis_v1@orphantest02"
    insert_prompt(
        db_path,
        PromptRow(
            prompt_version=pv,
            name="sonnet_filing_analysis_v1",
            file_path="src/prompts/sonnet_filing_analysis_v1.txt",
            content_hash="orphantest02" + "0" * 52,
        ),
    )
    payload = json.loads(_valid_proposal_json(conviction=9))
    proposal_id = insert_proposal(
        db_path,
        ProposalRow(
            filing_id=filing.id,
            decision_id="dec-orphan-002",
            model_id="claude-sonnet-4-6",
            prompt_version=pv,
            raw_response=_valid_proposal_json(conviction=9),
            kind="trade_proposal",
            symbol=payload["symbol"],
            direction=payload["direction"],
            size_pct_requested=payload["size_pct_of_capital"],
            conviction=payload["conviction"],
            thesis=payload["thesis"],
            input_tokens=1500,
            output_tokens=800,
            cache_read_tokens=4000,
            cache_creation_tokens=200,
            latency_ms=1234,
            cost_usd=0.0234,
        ),
    )

    # Broker has entry but NO stop — the dangerous partial state.
    now = dt.datetime.now(dt.UTC)
    fake_broker.preseeded_by_client_id[f"prop-{proposal_id}-entry"] = SubmittedOrder(
        broker_order_id="orphan-entry-002",
        client_order_id=f"prop-{proposal_id}-entry",
        symbol="ACME",
        side="buy",
        qty=10,
        order_type="market",
        status="accepted",
        submitted_at=now,
    )

    queue: asyncio.Queue[FilingRow] = asyncio.Queue()
    components = build_components(queue=queue)
    loop = AgentLoop(components=components, shutdown_grace_seconds=10.0)

    runner = asyncio.create_task(loop.run())
    while True:
        with connect(db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM executions WHERE proposal_id = ?", (proposal_id,)
            ).fetchone()
        if row is not None and loop.state == AgentState.IDLE:
            break
        await asyncio.sleep(0.01)
    loop.shutdown_requested = True
    await queue.put(None)  # type: ignore[arg-type]
    await asyncio.wait_for(runner, timeout=10.0)

    # No re-submission attempt.
    assert fake_broker.submitted_orders == []
    # Execution row records the partial state.
    with connect(db_path) as conn:
        exec_row = conn.execute(
            "SELECT decision FROM executions WHERE proposal_id = ?",
            (proposal_id,),
        ).fetchone()
    assert exec_row is not None
    assert exec_row["decision"] == "submission_partial_no_stop"
    # And the kill-switch is halted as a defensive measure.
    ks = get_latest_kill_switch_state(db_path)
    assert ks.state == "halted"
    assert ks.reason == "submission_partial_no_stop"

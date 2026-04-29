"""S5.6 — explicit rejection-journaling tests (carry-fwd S6.4 reviewer M-1).

S6.4's `OrderSubmitter` only writes the ACCEPTED journal branch; the
agent loop owns `executions.reject_reason` for every other path
(validator reject, sizer reject, opus reject, broker_unavailable,
schema). These tests pin each reject path to a concrete row.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json

import pytest

from journal.models import (
    FilingRow,
    KillSwitchStateRow,
    PrefilterDecisionRow,
)
from journal.repo import (
    connect,
    insert_kill_switch_state,
    insert_prefilter_decision,
)


@pytest.mark.asyncio
async def test_validator_kill_switch_reject_writes_journal_row(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    """Validator rejects on kill_switch → exactly one executions row
    with decision='rejected', reject_reason='kill_switch',
    submitted_orders_json='[]'. No broker submit_order call."""
    from app.loop import AgentLoop
    from app.state import AgentState
    from tests.integration.conftest import (
        _opus_ratify_json,
        _valid_proposal_json,
    )

    insert_kill_switch_state(
        db_path,
        KillSwitchStateRow(
            set_at=dt.datetime.now(dt.UTC),
            state="halted",
            reason="manual:operator",
            set_by="operator",
        ),
    )
    f = make_filing(accession="0001234567-26-000800", cik=320193)
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

    runner = asyncio.create_task(loop.run())
    while not (queue.empty() and loop.state == AgentState.IDLE):
        await asyncio.sleep(0.01)
    loop.shutdown_requested = True
    await queue.put(None)  # type: ignore[arg-type]
    await asyncio.wait_for(runner, timeout=10.0)

    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT decision, reject_reason, submitted_orders_json "
            "FROM executions"
        ).fetchall()

    assert len(rows) == 1
    assert rows[0]["decision"] == "rejected"
    assert rows[0]["reject_reason"] == "kill_switch"
    assert rows[0]["submitted_orders_json"] == "[]"
    assert fake_broker.submitted_orders == []


@pytest.mark.asyncio
async def test_sizer_ks5_concurrent_limit_writes_journal_row(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
    monkeypatch,
) -> None:
    """Sizing rejects on KS-5 (concurrent positions) → executions row
    with decision='rejected', reject_reason='ks5_concurrent_limit'."""
    from app.loop import AgentLoop
    from app.state import AgentState
    from broker.protocol import Position
    from tests.integration.conftest import (
        _opus_ratify_json,
        _valid_proposal_json,
    )

    f = make_filing(accession="0001234567-26-000801", cik=320193)
    insert_prefilter_decision(
        db_path,
        PrefilterDecisionRow(
            filing_id=f.id,
            decision="accept",
            rule_fired="material_8k_bypass",
        ),
    )

    # Stuff the broker with 5 open positions to trip KS-5 (config cap=5).
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
    loop = AgentLoop(components=components, shutdown_grace_seconds=10.0)

    runner = asyncio.create_task(loop.run())
    while not (queue.empty() and loop.state == AgentState.IDLE):
        await asyncio.sleep(0.01)
    loop.shutdown_requested = True
    await queue.put(None)  # type: ignore[arg-type]
    await asyncio.wait_for(runner, timeout=10.0)

    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT decision, reject_reason FROM executions"
        ).fetchall()

    assert len(rows) == 1
    assert rows[0]["decision"] == "rejected"
    assert rows[0]["reject_reason"] == "ks5_concurrent_limit"
    assert fake_broker.submitted_orders == []


@pytest.mark.asyncio
async def test_opus_reject_writes_journal_row(
    db_path,
    journal,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
) -> None:
    """Opus reject path → executions row with reason='opus_reject'."""
    from app.loop import AgentLoop
    from app.state import AgentState
    from tests.integration.conftest import _valid_proposal_json

    f = make_filing(accession="0001234567-26-000802", cik=320193)
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
    opus_reject = json.dumps(
        {
            "decision": "reject",
            "rationale": (
                "Filing language does not support the proposed thesis; "
                "catalyst is not material at v1's bar."
            ),
        }
    )
    fake_anthropic.responses_by_purpose = {
        "analyze": [_valid_proposal_json(conviction=9)],
        "review": [opus_reject],
    }

    components = build_components(queue=queue)
    loop = AgentLoop(components=components, shutdown_grace_seconds=10.0)

    runner = asyncio.create_task(loop.run())
    while not (queue.empty() and loop.state == AgentState.IDLE):
        await asyncio.sleep(0.01)
    loop.shutdown_requested = True
    await queue.put(None)  # type: ignore[arg-type]
    await asyncio.wait_for(runner, timeout=10.0)

    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT decision, reject_reason FROM executions"
        ).fetchall()

    assert len(rows) == 1
    assert rows[0]["decision"] == "rejected"
    assert rows[0]["reject_reason"] == "opus_reject"
    assert fake_broker.submitted_orders == []

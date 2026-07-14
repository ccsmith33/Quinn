"""DISPLACEMENT — agent-loop integration (execution/displacement.py seam).

Exercises the REAL `AgentLoop._execute` chain (prefilter → analyzer →
Opus → validator → sizer → displacement → submitter) with the fake
broker at the boundary:

  - enabled + high conviction + qualifying victim: the victim's market
    sell hits the broker BEFORE the new entry's bracket buy, the buy is
    sized against the haircut proceeds credit, and the journal carries
    the `displacement:` audit trail;
  - disabled: byte-identical to pre-feature behavior — the normal
    `ks7_cash_reserve` rejection row is written and NO order of any kind
    reaches the broker.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from types import SimpleNamespace

import pytest

from broker.protocol import AccountSnapshot, Position
from config.loader import ExecutionConfig
from journal.models import (
    ExecutionRow,
    FilingRow,
    OrderRow,
    PrefilterDecisionRow,
    PromptRow,
    ProposalRow,
)
from journal.repo import (
    connect,
    insert_execution,
    insert_order,
    insert_prefilter_decision,
    insert_prompt,
    insert_proposal,
)


def _execution_cfg(*, enabled: bool) -> ExecutionConfig:
    return ExecutionConfig(
        broker_mode="paper",
        ks4_pct_cap=0.15,
        ks4_absolute_cap_usd=1000.0,
        ks5_max_concurrent=10,
        ks7_cash_reserve_pct=0.03,
        sizing_mid_pct=0.07,
        sizing_high_pct=0.10,
        displacement_enabled=enabled,
        displacement_min_conviction=8,
        displacement_proceeds_haircut=0.90,
        displacement_victim_max_gain_pct=5.0,
    )


def _seed_victim_widg(db_path: str, make_filing) -> int:
    """Journal lineage for an open WIDG position entered ~14 days ago:
    accepted execution + filled entry order. Returns the execution id."""
    f = make_filing(
        accession="0001234567-26-000990",
        cik=789019,
        issuer_ticker="WIDG",
    )
    # Prefilter row so the boot recovery scan doesn't re-queue this
    # already-processed filing (which would consume the fake LLM queue).
    insert_prefilter_decision(
        db_path,
        PrefilterDecisionRow(
            filing_id=f.id, decision="accept", rule_fired="material_8k_bypass"
        ),
    )
    insert_prompt(
        db_path,
        PromptRow(
            prompt_version="sonnet@disptest",
            name="sonnet",
            file_path="src/prompts/sonnet.txt",
            content_hash="x" * 64,
        ),
    )
    pid = insert_proposal(
        db_path,
        ProposalRow(
            filing_id=f.id,
            decision_id="disp-victim-widg",
            model_id="claude-sonnet-4-6",
            prompt_version="sonnet@disptest",
            raw_response=json.dumps({"symbol": "WIDG"}),
            kind="trade_proposal",
            symbol="WIDG",
            direction="long",
            size_pct_requested=0.10,
            conviction=7,
            thesis="seeded victim thesis",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
            cost_usd=0.0,
        ),
    )
    eid = insert_execution(
        db_path,
        ExecutionRow(
            proposal_id=pid,
            decision="accepted",
            realized_size_pct=0.10,
            realized_dollar_size=2000.0,
            submitted_orders_json="[]",
        ),
    )
    entry_at = dt.datetime.now(dt.UTC) - dt.timedelta(days=14)
    insert_order(
        db_path,
        OrderRow(
            execution_id=eid,
            role="entry",
            symbol="WIDG",
            side="buy",
            order_type="market",
            qty=20,
            tif="gtc",
            broker_order_id="widg-entry-1",
            submitted_at=entry_at,
            realized_fill_price=105.0,
            realized_fill_qty=20,
            realized_fill_at=entry_at,
            final_status="filled",
        ),
    )
    return eid


def _set_low_cash_margin_account(fake_broker) -> None:
    """Margin-account shape: buying_power well above settled cash — the
    regime where the validator's capital gate passes but KS-7's cash
    reserve rejects (the displacement trigger zone). cash 350 vs the 3%
    reserve floor on 10k equity leaves 50 of headroom, less than one
    share at the fake quote of 100."""
    fake_broker.get_account = lambda: AccountSnapshot(
        equity=10_000.0,
        cash=350.0,
        buying_power=5_000.0,
        long_market_value=9_650.0,
        daypl=0.0,
        snapshot_at=dt.datetime.now(dt.UTC),
    )


async def _run_one_filing(loop_obj, queue) -> None:
    from app.state import AgentState

    runner = asyncio.create_task(loop_obj.run())
    while not (queue.empty() and loop_obj.state == AgentState.IDLE):
        await asyncio.sleep(0.01)
    loop_obj.shutdown_requested = True
    await queue.put(None)
    await asyncio.wait_for(runner, timeout=10.0)


@pytest.mark.asyncio
async def test_displacement_sells_victim_before_buy_and_accepts(
    db_path,
    journal,
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

    victim_eid = _seed_victim_widg(db_path, make_filing)

    # Broker truth: WIDG held at a -4.76% drift; low settled cash on a
    # margin account → ks7_cash_reserve without displacement.
    _set_low_cash_margin_account(fake_broker)
    fake_broker.positions = [
        Position(
            symbol="WIDG",
            qty=20,
            avg_entry_price=105.0,
            market_value=2000.0,
            unrealized_pnl=-100.0,
        )
    ]

    f = make_filing(accession="0001234567-26-000991", cik=320193)
    insert_prefilter_decision(
        db_path,
        PrefilterDecisionRow(
            filing_id=f.id, decision="accept", rule_fired="material_8k_bypass"
        ),
    )
    queue: asyncio.Queue[FilingRow] = asyncio.Queue()
    await queue.put(f)
    fake_anthropic.responses_by_purpose = {
        "analyze": [_valid_proposal_json(conviction=9, symbol="ACME")],
        "review": [_opus_ratify_json()],
    }

    components = build_components(queue=queue)
    config = SimpleNamespace(
        execution=_execution_cfg(enabled=True),
        analyzer=SimpleNamespace(opus_review_conviction_threshold=7),
    )
    loop_obj = AgentLoop(
        components=components, config=config, shutdown_grace_seconds=10.0
    )
    await _run_one_filing(loop_obj, queue)

    # SELL-BEFORE-BUY: the WIDG market sell is the first order the broker
    # saw; the ACME bracket entry follows it.
    sides = [(o.symbol, o.side, o.order_type) for o in fake_broker.submitted_orders]
    assert sides[0] == ("WIDG", "sell", "market")
    entry_idx = sides.index(("ACME", "buy", "market"))
    assert entry_idx > 0
    sell = fake_broker.submitted_orders[0]
    assert sell.qty == 20
    assert sell.client_order_id == f"displace-close-exec-{victim_eid}"

    # Buy sized against the haircut credit: available = 350 + 0.9×2000 −
    # 300 = 1850 ≥ high-tier ask 1000 (10% of 10k, capped by ks4 abs
    # 1000) → qty 10 at the fake quote of 100.
    with connect(db_path) as conn:
        accepted = conn.execute(
            "SELECT decision, realized_dollar_size FROM executions "
            "WHERE decision = 'accepted' AND id != ?",
            (victim_eid,),
        ).fetchall()
        trail = conn.execute(
            "SELECT role, notes FROM orders WHERE notes LIKE 'displacement:%'"
        ).fetchone()
    assert len(accepted) == 1
    assert accepted[0]["realized_dollar_size"] == pytest.approx(1000.0)
    assert trail is not None
    assert trail["role"] == "displacement_close"
    assert "WIDG" in trail["notes"] and "ACME" in trail["notes"]
    assert "conviction 7" in trail["notes"] and "conviction 9" in trail["notes"]


@pytest.mark.asyncio
async def test_displacement_disabled_normal_ks7_rejection_stands(
    db_path,
    journal,
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

    victim_eid = _seed_victim_widg(db_path, make_filing)
    _set_low_cash_margin_account(fake_broker)
    fake_broker.positions = [
        Position(
            symbol="WIDG",
            qty=20,
            avg_entry_price=105.0,
            market_value=2000.0,
            unrealized_pnl=-100.0,
        )
    ]

    f = make_filing(accession="0001234567-26-000992", cik=320193)
    insert_prefilter_decision(
        db_path,
        PrefilterDecisionRow(
            filing_id=f.id, decision="accept", rule_fired="material_8k_bypass"
        ),
    )
    queue: asyncio.Queue[FilingRow] = asyncio.Queue()
    await queue.put(f)
    fake_anthropic.responses_by_purpose = {
        "analyze": [_valid_proposal_json(conviction=9, symbol="ACME")],
        "review": [_opus_ratify_json()],
    }

    components = build_components(queue=queue)
    config = SimpleNamespace(
        execution=_execution_cfg(enabled=False),
        analyzer=SimpleNamespace(opus_review_conviction_threshold=7),
    )
    loop_obj = AgentLoop(
        components=components, config=config, shutdown_grace_seconds=10.0
    )
    await _run_one_filing(loop_obj, queue)

    # Byte-identical to pre-feature: normal ks7 rejection, journal reason
    # unchanged, and the broker saw NO orders (no sell, no buy).
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT decision, reject_reason FROM executions WHERE id != ?",
            (victim_eid,),
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["decision"] == "rejected"
    assert rows[0]["reject_reason"] == "ks7_cash_reserve"
    assert fake_broker.submitted_orders == []

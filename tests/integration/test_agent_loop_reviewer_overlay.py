"""Reviewer-overlay wiring — Opus `modify` stop/TP/exit-condition changes
must reach execution (closes the S6.x gap).

`OpusModified.modifications` were journaled to
`proposal_reviews.modifications_json` but the loop rebuilt the trade
payload from `proposals.raw_response`, so the validator, sizer, and
order submitter all saw the analyzer's ORIGINAL levels. These tests pin
the fixed behavior:

  1. a reviewer stop modification changes the submitted bracket's stop;
  2. a reviewer TP raise flows through to the bracket + journaled TP row;
  3. a modification that breaks exit geometry is rejected through the
     existing `exit_geometry` path (gates are not bypassed);
  4. a modify with no price/exit fields leaves the levels byte-identical
     to the original proposal (regression);
  5. the `proposals` row is never mutated (NFR-16) and the overlay emits
     one `execution.reviewer_overlay_applied` INFO event.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from journal.models import FilingRow, PrefilterDecisionRow
from journal.repo import connect, insert_prefilter_decision


def _proposal_json(
    *,
    stop: float,
    tp: float | None = None,
    conviction: int = 9,
    symbol: str = "ACME",
) -> str:
    payload = {
        "symbol": symbol,
        "direction": "long",
        "size_pct_of_capital": 0.10,
        "entry_style": "market_open",
        "stop_loss_price": stop,
        "time_horizon_days": 14,
        "conviction": conviction,
        "thesis": (
            f"{symbol} announced a material acquisition in 8-K Item 1.01 "
            "with concrete pricing terms; integration timeline is plausible."
        ),
        "signals": ["Item 1.01 — Material Definitive Agreement"],
        "exit_conditions": ["Exit on contradicting filing within 5 trading days"],
        "risk_factors": ["Closing conditions not yet met"],
    }
    if tp is not None:
        payload["take_profit_price"] = tp
    return json.dumps(payload)


def _opus_modify_json(modifications: dict) -> str:
    return json.dumps(
        {
            "decision": "modify",
            "rationale": (
                "Thesis holds but the analyst's protective levels are "
                "mis-calibrated for current volatility; adjusting the "
                "trade plan improves the reward/risk geometry."
            ),
            "modifications": modifications,
        }
    )


async def _run_one_filing(
    *,
    db_path: str,
    fake_broker,
    fake_anthropic,
    build_components,
    make_filing,
    accession: str,
    proposal_json: str,
    review_json: str,
) -> None:
    """Drive the real AgentLoop through a single filing with canned
    analyzer + reviewer responses (same pattern as the reject-journaling
    tests)."""
    from app.loop import AgentLoop
    from app.state import AgentState

    f = make_filing(accession=accession, cik=320193)
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
        "analyze": [proposal_json],
        "review": [review_json],
    }
    components = build_components(queue=queue)
    loop = AgentLoop(components=components, shutdown_grace_seconds=10.0)
    runner = asyncio.create_task(loop.run())
    while not (queue.empty() and loop.state == AgentState.IDLE):
        await asyncio.sleep(0.01)
    loop.shutdown_requested = True
    await queue.put(None)  # type: ignore[arg-type]
    await asyncio.wait_for(runner, timeout=10.0)


def _proposal_payload(db_path: str) -> dict:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT raw_response FROM proposals WHERE kind='trade_proposal'"
        ).fetchone()
    return json.loads(row["raw_response"])


@pytest.mark.asyncio
async def test_reviewer_stop_modification_changes_submitted_bracket_stop(
    db_path, journal, fake_broker, fake_anthropic, build_components, make_filing,
) -> None:
    """(1) Opus modify with stop_loss_price → the bracket goes to the
    broker with the MODIFIED stop and the journaled stop OrderRow matches;
    the `proposals` row keeps the original level (NFR-16)."""
    await _run_one_filing(
        db_path=db_path,
        fake_broker=fake_broker,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        accession="0001234567-26-000900",
        proposal_json=_proposal_json(stop=95.0),
        review_json=_opus_modify_json({"stop_loss_price": 92.0}),
    )

    assert len(fake_broker.submitted_brackets) == 1
    assert fake_broker.submitted_brackets[0].stop_loss_price == 92.0

    with connect(db_path) as conn:
        execution = conn.execute(
            "SELECT decision, reject_reason FROM executions"
        ).fetchone()
        stop_row = conn.execute(
            "SELECT stop_price FROM orders WHERE role='stop'"
        ).fetchone()
    assert execution["decision"] == "accepted"
    assert stop_row["stop_price"] == 92.0

    # NFR-16: original journal row is immutable — the overlay never
    # touches `proposals.raw_response`.
    assert _proposal_payload(db_path)["stop_loss_price"] == 95.0


@pytest.mark.asyncio
async def test_reviewer_tp_raise_flows_through_to_bracket(
    db_path, journal, fake_broker, fake_anthropic, build_components, make_filing,
) -> None:
    """(2) Opus modify raising take_profit_price → the bracket and the
    journaled take-profit OrderRow carry the raised TP; the unmodified
    stop stays the analyzer's original."""
    await _run_one_filing(
        db_path=db_path,
        fake_broker=fake_broker,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        accession="0001234567-26-000901",
        proposal_json=_proposal_json(stop=95.0, tp=110.0),
        review_json=_opus_modify_json({"take_profit_price": 150.0}),
    )

    assert len(fake_broker.submitted_brackets) == 1
    bracket = fake_broker.submitted_brackets[0]
    assert bracket.take_profit_price == 150.0
    assert bracket.stop_loss_price == 95.0  # untouched field stays original

    with connect(db_path) as conn:
        tp_row = conn.execute(
            "SELECT limit_price FROM orders WHERE role='take_profit'"
        ).fetchone()
    assert tp_row["limit_price"] == 150.0
    assert _proposal_payload(db_path)["take_profit_price"] == 110.0


@pytest.mark.asyncio
async def test_reviewer_modification_breaking_geometry_is_rejected(
    db_path, journal, fake_broker, fake_anthropic, build_components, make_filing,
) -> None:
    """(3) A modified TP that collapses reward/risk below the floor must
    reject through the EXISTING `exit_geometry` gate — the overlay never
    bypasses validation. Original geometry (stop 95 / TP 110 @ quote 100,
    RR=2.0) passes; the modified TP 101 (RR=0.2) must not trade."""
    await _run_one_filing(
        db_path=db_path,
        fake_broker=fake_broker,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        accession="0001234567-26-000902",
        proposal_json=_proposal_json(stop=95.0, tp=110.0),
        review_json=_opus_modify_json({"take_profit_price": 101.0}),
    )

    with connect(db_path) as conn:
        execution = conn.execute(
            "SELECT decision, reject_reason FROM executions"
        ).fetchone()
    assert execution["decision"] == "rejected"
    assert execution["reject_reason"] == "exit_geometry"
    assert fake_broker.submitted_brackets == []
    assert fake_broker.submitted_orders == []


@pytest.mark.asyncio
async def test_modify_without_price_fields_leaves_levels_unchanged(
    db_path, journal, fake_broker, fake_anthropic, build_components, make_filing,
    caplog,
) -> None:
    """(4) Regression: a modify carrying only size_pct_of_capital does not
    overlay any trade-plan field — the bracket is built from the
    analyzer's original levels and no overlay event is emitted."""
    caplog.set_level(logging.INFO, logger="app.loop")
    await _run_one_filing(
        db_path=db_path,
        fake_broker=fake_broker,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        accession="0001234567-26-000903",
        proposal_json=_proposal_json(stop=95.0, tp=110.0),
        review_json=_opus_modify_json({"size_pct_of_capital": 0.05}),
    )

    assert len(fake_broker.submitted_brackets) == 1
    bracket = fake_broker.submitted_brackets[0]
    assert bracket.stop_loss_price == 95.0
    assert bracket.take_profit_price == 110.0
    overlay_events = [
        r for r in caplog.records
        if getattr(r, "event", None) == "execution.reviewer_overlay_applied"
    ]
    assert overlay_events == []


@pytest.mark.asyncio
async def test_overlay_emits_info_event_and_applies_exit_conditions(
    db_path, journal, fake_broker, fake_anthropic, build_components, make_filing,
    caplog,
) -> None:
    """(5) A modify with stop + exit_conditions emits exactly one
    `execution.reviewer_overlay_applied` INFO event naming the overlaid
    fields, and the trade still executes with the modified stop."""
    caplog.set_level(logging.INFO, logger="app.loop")
    await _run_one_filing(
        db_path=db_path,
        fake_broker=fake_broker,
        fake_anthropic=fake_anthropic,
        build_components=build_components,
        make_filing=make_filing,
        accession="0001234567-26-000904",
        proposal_json=_proposal_json(stop=95.0),
        review_json=_opus_modify_json(
            {
                "stop_loss_price": 92.0,
                "exit_conditions": [
                    "Exit immediately if a contradicting 8-K amendment is filed"
                ],
            }
        ),
    )

    assert len(fake_broker.submitted_brackets) == 1
    assert fake_broker.submitted_brackets[0].stop_loss_price == 92.0

    overlay_events = [
        r for r in caplog.records
        if getattr(r, "event", None) == "execution.reviewer_overlay_applied"
    ]
    assert len(overlay_events) == 1
    assert overlay_events[0].levelno == logging.INFO
    assert sorted(overlay_events[0].fields) == [
        "exit_conditions",
        "stop_loss_price",
    ]
    assert overlay_events[0].proposal_id is not None
    # NFR-16: journaled exit_conditions stay the analyzer's original.
    assert _proposal_payload(db_path)["exit_conditions"] == [
        "Exit on contradicting filing within 5 trading days"
    ]

"""Hotfix 2026-08-11 (Alpaca 42210000) — `restore_protection` read-side
quantization: the FEIM-class compensation re-places journal prices
verbatim; rows written before the write-side fix may be sub-penny. Stop
DOWN, TP UP; the replacement journal rows record the quantized values
actually sent (§3.1 honesty).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pytest

from broker.protocol import SubmittedOrder
from execution.protection import ClearedLeg, restore_protection
from journal.migrate import apply_migrations
from journal.models import (
    ExecutionRow,
    FilingRow,
    OrderRow,
    PromptRow,
    ProposalRow,
)
from journal.repo import (
    insert_execution,
    insert_filing,
    insert_prompt,
    insert_proposal,
)

NOW = dt.datetime(2026, 8, 11, 14, 0, 0, tzinfo=dt.UTC)


@pytest.fixture
def db(tmp_path: Path) -> str:
    p = str(tmp_path / "journal.db")
    apply_migrations(p)
    return p


def _seed_execution(db: str) -> int:
    fid = insert_filing(
        db,
        FilingRow(
            accession_number="prot-acc-1",
            cik=320193,
            form_type="8-K",
            filed_at=NOW - dt.timedelta(hours=2),
            fetched_at=NOW - dt.timedelta(hours=1),
            raw_text_path="/tmp/prot.txt",
            content_hash="h-prot-1",
            item_codes='["1.01"]',
            issuer_ticker="ACME",
        ),
    )
    insert_prompt(
        db,
        PromptRow(
            prompt_version="sonnet@prottest",
            name="sonnet",
            file_path="src/prompts/sonnet.txt",
            content_hash="x" * 64,
        ),
    )
    pid = insert_proposal(
        db,
        ProposalRow(
            filing_id=fid,
            decision_id="prot-d-1",
            model_id="claude-sonnet-4-6",
            prompt_version="sonnet@prottest",
            raw_response="{}",
            kind="trade_proposal",
            symbol="ACME",
            direction="long",
            size_pct_requested=0.10,
            conviction=8,
            thesis="seed",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
            cost_usd=0.0,
        ),
    )
    return insert_execution(
        db,
        ExecutionRow(
            proposal_id=pid,
            decision="accepted",
            realized_size_pct=0.10,
            realized_dollar_size=1000.0,
            submitted_orders_json="[]",
        ),
    )


def _leg(
    execution_id: int,
    *,
    role: str,
    stop_price: float | None = None,
    limit_price: float | None = None,
) -> ClearedLeg:
    return ClearedLeg(
        row=OrderRow(
            id=None,
            execution_id=execution_id,
            role=role,
            symbol="ACME",
            side="sell",
            order_type="stop" if role != "take_profit" else "limit",
            qty=100,
            tif="gtc",
            stop_price=stop_price,
            limit_price=limit_price,
            broker_order_id=f"dead-{role}",
            submitted_at=NOW,
            final_status=None,
        ),
        disposition="canceled",
    )


class _FakeOcoBroker:
    def __init__(self) -> None:
        self.oco_calls: list[dict[str, Any]] = []

    def submit_oco_sell(
        self,
        *,
        symbol: str,
        qty: int,
        stop_price: float,
        limit_price: float | None,
        client_order_id: str,
    ) -> tuple[SubmittedOrder, SubmittedOrder | None]:
        self.oco_calls.append(
            {
                "symbol": symbol,
                "qty": qty,
                "stop_price": stop_price,
                "limit_price": limit_price,
                "client_order_id": client_order_id,
            }
        )
        stop = SubmittedOrder(
            broker_order_id="oco-stop-1",
            client_order_id=client_order_id,
            symbol=symbol,
            side="sell",
            qty=qty,
            order_type="stop",
            status="accepted",
            submitted_at=NOW,
            stop_price=stop_price,
        )
        tp: SubmittedOrder | None = None
        if limit_price is not None:
            tp = SubmittedOrder(
                broker_order_id="oco-tp-1",
                client_order_id=client_order_id,
                symbol=symbol,
                side="sell",
                qty=qty,
                order_type="limit",
                status="accepted",
                submitted_at=NOW,
                limit_price=limit_price,
            )
        return stop, tp


def test_restore_protection_quantizes_subpenny_journal_prices(db: str) -> None:
    """Sub-penny cleared legs (stop 17.8567, TP 26.4312) restore as a
    grid-clean OCO: stop 17.85 (down), TP 26.44 (up) — at the broker AND
    in the replacement journal rows."""
    eid = _seed_execution(db)
    cleared = [
        _leg(eid, role="take_profit", limit_price=26.4312),
        _leg(eid, role="stop", stop_price=17.8567),
    ]
    broker = _FakeOcoBroker()

    restored = restore_protection(
        db, broker, eid, cleared,
        client_order_id="prot-restore-1", notes="test restore",
    )

    assert len(broker.oco_calls) == 1
    assert broker.oco_calls[0]["stop_price"] == 17.85
    assert broker.oco_calls[0]["limit_price"] == 26.44
    # §3.1 honesty: the journaled replacement stop records the quantized
    # value actually sent, never the raw sub-penny original.
    assert restored is not None
    assert restored.stop_price == 17.85

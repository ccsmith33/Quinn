"""Shared fixtures for `tests/dashboard/` (S9.1)."""

from __future__ import annotations

import base64
import datetime as _dt
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dashboard.app import build_app
from journal.migrate import apply_migrations
from journal.models import (
    AccountSnapshotRow,
    ExecutionRow,
    FilingRow,
    KillSwitchStateRow,
    LlmCallRow,
    OrderRow,
    PromptRow,
    ProposalReviewRow,
    ProposalRow,
)
from journal.repo import JournalRepo, insert_filing, insert_prompt

USER = "operator"
PASSWORD = "correcthorsebatterystaple"


def auth_headers(user: str = USER, password: str = PASSWORD) -> dict[str, str]:
    raw = f"{user}:{password}".encode()
    return {"Authorization": "Basic " + base64.b64encode(raw).decode("ascii")}


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    p = tmp_path / "journal.db"
    apply_migrations(str(p))
    return str(p)


@pytest.fixture
def journal(db_path: str) -> JournalRepo:
    return JournalRepo(db_path)


@pytest.fixture
def app(journal: JournalRepo):
    return build_app(
        journal=journal,
        dashboard_user=USER,
        dashboard_password=PASSWORD,
        auto_refresh_seconds=30,
    )


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# Seeding helpers — exercised by overview/proposals/trades/positions/llm tests
# ---------------------------------------------------------------------------


def _utc() -> _dt.datetime:
    """Yesterday noon UTC. Anchored to the run, not the calendar: a
    fixed date ages out of the dashboard's 30-day windows (D-082's
    documented date-bomb, fixed at integration). Yesterday keeps every
    seeded timestamp in the past regardless of run time-of-day."""
    now = _dt.datetime.now(_dt.UTC)
    noon = now.replace(hour=12, minute=0, second=0, microsecond=0)
    return noon - _dt.timedelta(days=1)


def seed_filings(db_path: str, *, n: int = 3, day: _dt.datetime | None = None) -> list[int]:
    day = day or _utc()
    ids: list[int] = []
    for i in range(n):
        ids.append(
            insert_filing(
                db_path,
                FilingRow(
                    accession_number=f"0000000000-{i:02d}-000001",
                    cik=320193 + i,
                    form_type="8-K",
                    filed_at=day - _dt.timedelta(minutes=10 * i),
                    fetched_at=day - _dt.timedelta(minutes=10 * i),
                    raw_text_path=f"/var/lib/quinn/filings/{i}.txt",
                    content_hash=f"h{i}",
                    item_codes="2.02",
                    issuer_ticker=f"AB{i}",
                ),
            )
        )
    return ids


def seed_prompt(db_path: str, prompt_version: str = "sonnet:v1#abc") -> None:
    insert_prompt(
        db_path,
        PromptRow(
            prompt_version=prompt_version,
            name="sonnet_filing_analysis_v1",
            file_path="src/prompts/files/sonnet.md",
            content_hash="abc",
        ),
    )


def seed_proposals(
    journal: JournalRepo,
    *,
    filing_ids: list[int],
    prompt_version: str = "sonnet:v1#abc",
) -> list[int]:
    """Use the module-level helper to insert proposals via JournalRepo."""
    from journal.repo import insert_proposal

    ids: list[int] = []
    for i, fid in enumerate(filing_ids):
        row = ProposalRow(
            filing_id=fid,
            decision_id=f"d-{i:03d}",
            model_id="claude-sonnet-4-6",
            prompt_version=prompt_version,
            raw_response='{"thesis": "<script>x</script>", "conviction": ' + str(7 + i) + "}",
            kind="trade",
            symbol=f"AB{i}",
            direction="long",
            size_pct_requested=0.05,
            conviction=7 + i,
            thesis="alpha thesis",
            input_tokens=1000,
            output_tokens=200 + i,
            cache_read_tokens=400,
            cache_creation_tokens=600,
            latency_ms=2500 + i,
            cost_usd=0.012 + 0.001 * i,
        )
        ids.append(insert_proposal(journal.db_path, row))
    return ids


def seed_review(journal: JournalRepo, proposal_id: int, prompt_version: str) -> int:
    from journal.repo import insert_proposal_review

    return insert_proposal_review(
        journal.db_path,
        ProposalReviewRow(
            proposal_id=proposal_id,
            model_id="claude-opus-4-7",
            prompt_version=prompt_version,
            decision="approved",
            raw_response='{"decision": "approved", "rationale": "ok"}',
            rationale="ok",
            modifications_json=None,
            input_tokens=2000,
            output_tokens=400,
            cache_read_tokens=800,
            cache_creation_tokens=1200,
            latency_ms=4500,
            cost_usd=0.05,
        ),
    )


def seed_execution_with_orders(
    journal: JournalRepo,
    proposal_id: int,
    *,
    symbol: str = "AB0",
    qty: int = 100,
    fill_price: float = 12.34,
    filled: bool = True,
) -> int:
    eid = journal.insert_execution(
        ExecutionRow(
            proposal_id=proposal_id,
            decision="accepted",
            reject_reason=None,
            realized_size_pct=0.05,
            realized_dollar_size=qty * fill_price,
            submitted_orders_json=None,
        )
    )
    journal.insert_order(
        OrderRow(
            execution_id=eid,
            role="entry",
            symbol=symbol,
            side="buy",
            order_type="limit",
            qty=qty,
            limit_price=fill_price,
            stop_price=None,
            tif="day",
            broker_order_id=f"alpaca-{eid}",
            submitted_at=_utc(),
            pre_submission_bid=fill_price - 0.02,
            pre_submission_ask=fill_price + 0.02,
            pre_submission_last=fill_price,
            pre_submission_quote_at=_utc(),
            final_status="filled" if filled else "submitted",
            realized_fill_price=fill_price if filled else None,
            realized_fill_qty=qty if filled else None,
            realized_fill_at=_utc() if filled else None,
            realized_fee=0.0,
            notes=None,
        )
    )
    return eid


def seed_account_snapshots(journal: JournalRepo, *, days: int = 5) -> None:
    base = _utc() - _dt.timedelta(days=days)
    for d in range(days + 1):
        equity = 10000.0 + 50.0 * d
        journal.insert_account_snapshot(
            AccountSnapshotRow(
                snapshot_at=base + _dt.timedelta(days=d),
                equity=equity,
                cash=equity * 0.6,
                buying_power=equity * 1.2,
                long_market_value=equity * 0.4,
                daypl=12.5 if d == days else 0.0,
            )
        )


def seed_kill_switch(journal: JournalRepo, *, halted: bool = False) -> None:
    journal.insert_kill_switch_state(
        KillSwitchStateRow(
            set_at=_dt.datetime.now(_dt.UTC),
            state="halted" if halted else "active",
            reason="manual:test" if halted else "resume",
            set_by="operator",
            notes="seed",
        )
    )


def seed_llm_calls(journal: JournalRepo, *, n: int = 3) -> None:
    from journal.repo import insert_llm_call

    seed_prompt(journal.db_path, "sonnet:v1#abc")
    for i in range(n):
        insert_llm_call(
            journal.db_path,
            LlmCallRow(
                decision_id=f"d-{i:03d}",
                purpose="analysis",
                model_id="claude-sonnet-4-6" if i % 2 == 0 else "claude-opus-4-7",
                prompt_version="sonnet:v1#abc",
                input_tokens=1000,
                output_tokens=200,
                cache_read_tokens=400,
                cache_creation_tokens=600,
                latency_ms=2500 + 100 * i,
                cost_usd=0.012 + 0.005 * i,
                error_class=None,
            ),
        )

"""S9.1 — JournalRepo helpers added for the operator dashboard (D-063).

Each new helper is exercised here per the dev-note rule that any new query
added to `src/journal/repo.py` requires a unit test.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pytest

from journal.migrate import apply_migrations
from journal.models import (
    AccountSnapshotRow,
    ExecutionRow,
    FilingRow,
    LlmCallRow,
    OrderRow,
    PromptRow,
    ProposalRow,
)
from journal.repo import (
    JournalRepo,
    insert_filing,
    insert_llm_call,
    insert_prompt,
    insert_proposal,
)


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    p = tmp_path / "journal.db"
    apply_migrations(str(p))
    return str(p)


@pytest.fixture
def journal(db_path: str) -> JournalRepo:
    return JournalRepo(db_path)


def _now() -> _dt.datetime:
    return _dt.datetime(2026, 4, 28, 12, 0, 0, tzinfo=_dt.UTC)


def test_count_filings_since(db_path: str, journal: JournalRepo) -> None:
    base = _now()
    for i in range(3):
        insert_filing(
            db_path,
            FilingRow(
                accession_number=f"0000000000-{i:02d}-000001",
                cik=320193 + i,
                form_type="8-K",
                filed_at=base - _dt.timedelta(hours=i + 1),
                fetched_at=base - _dt.timedelta(hours=i + 1),
                raw_text_path=f"/tmp/x{i}.txt",
                content_hash=f"h{i}",
            ),
        )
    assert journal.count_filings_since(base - _dt.timedelta(hours=4)) == 3
    assert journal.count_filings_since(base - _dt.timedelta(hours=2)) == 2
    assert journal.count_filings_since(base) == 0


def test_count_proposals_and_executions_today(
    db_path: str, journal: JournalRepo
) -> None:
    insert_prompt(
        db_path,
        PromptRow(
            prompt_version="sonnet:v1#abc",
            name="sonnet_filing_analysis_v1",
            file_path="src/prompts/files/sonnet.md",
            content_hash="abc",
        ),
    )
    fid = insert_filing(
        db_path,
        FilingRow(
            accession_number="0000000000-99-000001",
            cik=320193,
            form_type="8-K",
            filed_at=_now(),
            fetched_at=_now(),
            raw_text_path="/tmp/x.txt",
            content_hash="h",
        ),
    )
    pid = insert_proposal(
        db_path,
        ProposalRow(
            filing_id=fid,
            decision_id="d-1",
            model_id="claude-sonnet-4-6",
            prompt_version="sonnet:v1#abc",
            raw_response="{}",
            kind="trade",
            symbol="AB",
            direction="long",
            size_pct_requested=0.05,
            conviction=8,
            thesis="alpha",
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            latency_ms=1000,
            cost_usd=0.01,
        ),
    )
    journal.insert_execution(
        ExecutionRow(
            proposal_id=pid,
            decision="accepted",
            reject_reason=None,
            realized_size_pct=0.05,
            realized_dollar_size=100.0,
            submitted_orders_json=None,
        )
    )
    today = _dt.datetime(_now().year, _now().month, _now().day, tzinfo=_dt.UTC)
    assert journal.count_proposals_since(today) == 1
    assert journal.count_executions_since(today) == 1
    assert journal.count_executions_since(today, decision="accepted") == 1
    assert journal.count_executions_since(today, decision="rejected") == 0


def test_get_account_snapshots_since(journal: JournalRepo) -> None:
    base = _now()
    for d in range(5):
        journal.insert_account_snapshot(
            AccountSnapshotRow(
                snapshot_at=base - _dt.timedelta(days=4 - d),
                equity=10000 + 50 * d,
                cash=5000,
                buying_power=20000,
                long_market_value=5000,
                daypl=0,
            )
        )
    rows = journal.get_account_snapshots_since(base - _dt.timedelta(days=10))
    assert len(rows) == 5
    # Ordered ASC.
    eqs = [r.equity for r in rows]
    assert eqs == sorted(eqs)


def test_list_recent_proposals_with_filters(
    db_path: str, journal: JournalRepo
) -> None:
    insert_prompt(
        db_path,
        PromptRow(
            prompt_version="sonnet:v1#abc",
            name="sonnet_filing_analysis_v1",
            file_path="src/prompts/files/sonnet.md",
            content_hash="abc",
        ),
    )
    fids: list[int] = []
    for i in range(3):
        fids.append(
            insert_filing(
                db_path,
                FilingRow(
                    accession_number=f"0000-{i:02d}-001",
                    cik=320193 + i,
                    form_type="8-K",
                    filed_at=_now(),
                    fetched_at=_now(),
                    raw_text_path=f"/tmp/{i}.txt",
                    content_hash=f"h{i}",
                ),
            )
        )
    pids: list[int] = []
    for i, fid in enumerate(fids):
        pids.append(
            insert_proposal(
                db_path,
                ProposalRow(
                    filing_id=fid,
                    decision_id=f"d-{i}",
                    model_id="claude-sonnet-4-6",
                    prompt_version="sonnet:v1#abc",
                    raw_response="{}",
                    kind="trade",
                    symbol=f"AB{i}",
                    direction="long",
                    size_pct_requested=0.05,
                    conviction=8,
                    thesis="alpha",
                    input_tokens=100,
                    output_tokens=50,
                    cache_read_tokens=0,
                    cache_creation_tokens=0,
                    latency_ms=1000,
                    cost_usd=0.01,
                ),
            )
        )
    # accepted execution on the FIRST proposal only
    journal.insert_execution(
        ExecutionRow(
            proposal_id=pids[0],
            decision="accepted",
            reject_reason=None,
            realized_size_pct=0.05,
            realized_dollar_size=100.0,
            submitted_orders_json=None,
        )
    )
    all_rows = journal.list_recent_proposals(limit=100)
    assert len(all_rows) == 3
    only_ab1 = journal.list_recent_proposals(limit=100, symbol="AB1")
    assert len(only_ab1) == 1
    assert only_ab1[0].symbol == "AB1"
    accepted = journal.list_recent_proposals(limit=100, decision_status="accepted")
    assert len(accepted) == 1
    assert accepted[0].decision_id == "d-0"


def test_list_recent_executions_with_orders(
    db_path: str, journal: JournalRepo
) -> None:
    insert_prompt(
        db_path,
        PromptRow(
            prompt_version="sonnet:v1#abc",
            name="sonnet_filing_analysis_v1",
            file_path="src/prompts/files/sonnet.md",
            content_hash="abc",
        ),
    )
    fid = insert_filing(
        db_path,
        FilingRow(
            accession_number="0000-99-001",
            cik=1,
            form_type="8-K",
            filed_at=_now(),
            fetched_at=_now(),
            raw_text_path="/tmp/x",
            content_hash="h",
        ),
    )
    pid = insert_proposal(
        db_path,
        ProposalRow(
            filing_id=fid,
            decision_id="d-x",
            model_id="claude-sonnet-4-6",
            prompt_version="sonnet:v1#abc",
            raw_response="{}",
            kind="trade",
            symbol="AB",
            direction="long",
            size_pct_requested=0.05,
            conviction=8,
            thesis="alpha",
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            latency_ms=1000,
            cost_usd=0.01,
        ),
    )
    eid = journal.insert_execution(
        ExecutionRow(
            proposal_id=pid,
            decision="accepted",
            reject_reason=None,
            realized_size_pct=0.05,
            realized_dollar_size=100.0,
            submitted_orders_json=None,
        )
    )
    journal.insert_order(
        OrderRow(
            execution_id=eid,
            role="entry",
            symbol="AB",
            side="buy",
            order_type="limit",
            qty=10,
            limit_price=12.0,
            stop_price=None,
            tif="day",
            broker_order_id=f"alpaca-{eid}",
            submitted_at=_now(),
            pre_submission_bid=11.99,
            pre_submission_ask=12.01,
            pre_submission_last=12.0,
            pre_submission_quote_at=_now(),
            final_status="filled",
            realized_fill_price=12.0,
            realized_fill_qty=10,
            realized_fill_at=_now(),
            realized_fee=0.0,
            notes=None,
        )
    )
    triples = journal.list_recent_executions_with_orders(limit=10)
    assert len(triples) == 1
    erow, orders, prow = triples[0]
    assert erow.decision == "accepted"
    assert len(orders) == 1
    assert orders[0].symbol == "AB"
    assert prow is not None
    assert prow.decision_id == "d-x"


def test_llm_spend_breakdown_mtd_and_cache_hit_pct(
    db_path: str, journal: JournalRepo
) -> None:
    insert_prompt(
        db_path,
        PromptRow(
            prompt_version="sonnet:v1#abc",
            name="sonnet_filing_analysis_v1",
            file_path="src/prompts/files/sonnet.md",
            content_hash="abc",
        ),
    )
    for i in range(4):
        insert_llm_call(
            db_path,
            LlmCallRow(
                decision_id=f"d-{i}",
                purpose="analysis",
                model_id="claude-sonnet-4-6" if i % 2 == 0 else "claude-opus-4-7",
                prompt_version="sonnet:v1#abc",
                input_tokens=1000,
                output_tokens=200,
                cache_read_tokens=400,
                cache_creation_tokens=600,
                latency_ms=1000 + 100 * i,
                cost_usd=0.01 + 0.005 * i,
            ),
        )
    out = journal.llm_spend_breakdown_mtd(_now())
    models = {row["model_id"] for row in out}
    assert models == {"claude-sonnet-4-6", "claude-opus-4-7"}
    for r in out:
        # cache_read=400, denom = 400+600+1000 = 2000, hit pct = 20%
        assert r["cache_hit_pct"] == 20.0
        assert r["calls"] >= 1
        assert r["p50_latency_ms"] >= 1000
        assert r["p95_latency_ms"] >= r["p50_latency_ms"]


def test_llm_spend_breakdown_empty_returns_empty_list(journal: JournalRepo) -> None:
    assert journal.llm_spend_breakdown_mtd(_now()) == []

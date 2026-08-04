"""SYMBOL-HISTORY memory provider — `make_symbol_history_provider`.

Covers: None on no symbol / no history / non-served purpose;
closed-trade rendering (incl. realized % math + rounding); analyzed-
but-not-traded rendering with reject-reason labels; current-engagement
exclusion for thesis_review; postmortem lesson inclusion; the cap at
3; byte-identical determinism across calls.
"""

from __future__ import annotations

import datetime as dt
import itertools
import sqlite3
from pathlib import Path

import pytest

from app.memory_context import MemoryQuery
from app.memory_symbol_history import make_symbol_history_provider
from journal.migrate import apply_migrations
from journal.models import (
    ExecutionRow,
    FilingRow,
    OrderRow,
    PromptRow,
    ProposalRow,
    TradePostmortemRow,
)
from journal.repo import JournalRepo, get_prompt_by_version, insert_prompt

_PV = "pv@aaaaaaaaaaaa"
_seq = itertools.count(1)


@pytest.fixture
def journal(tmp_path: Path) -> JournalRepo:
    p = tmp_path / "journal.db"
    apply_migrations(str(p))
    return JournalRepo(str(p))


def _q(
    symbol: str | None = "GCBC",
    purpose: str = "analyze",
    execution_id: int | None = None,
) -> MemoryQuery:
    return MemoryQuery(
        symbol=symbol,
        purpose=purpose,  # type: ignore[arg-type]
        execution_id=execution_id,
        conviction=None,
    )


def _seed_proposal(
    journal: JournalRepo,
    symbol: str,
    *,
    conviction: int = 7,
    created_at: dt.datetime | None = None,
) -> int:
    from journal.repo import insert_filing, insert_proposal

    db = journal.db_path
    n = next(_seq)
    if get_prompt_by_version(db, _PV) is None:
        insert_prompt(
            db,
            PromptRow(
                prompt_version=_PV,
                name="sonnet_filing_analysis_v2",
                file_path="src/prompts/sonnet_filing_analysis_v2.txt",
                content_hash="a" * 64,
            ),
        )
    fid = insert_filing(
        db,
        FilingRow(
            accession_number=f"acc-p-{n}",
            cik=2000000 + n,
            form_type="8-K",
            filed_at=dt.datetime(2026, 4, 1, 12, 0, 0),
            fetched_at=dt.datetime(2026, 4, 1, 12, 1, 0),
            raw_text_path=f"/raw/p-{n}.txt",
            content_hash=f"h-p-{n}",
        ),
    )
    pid = insert_proposal(
        db,
        ProposalRow(
            filing_id=fid,
            decision_id=f"dec-p-{n}",
            model_id="claude-sonnet-4-6",
            prompt_version=_PV,
            raw_response="{}",
            kind="trade_proposal",
            symbol=symbol,
            direction="long",
            size_pct_requested=0.05,
            conviction=conviction,
            thesis="x",
            input_tokens=10,
            output_tokens=10,
            latency_ms=100,
            cost_usd=0.001,
        ),
    )
    if created_at is not None:
        with sqlite3.connect(db) as conn:
            conn.execute(
                "UPDATE proposals SET created_at = ? WHERE id = ?",
                (created_at.isoformat(sep=" "), pid),
            )
    return pid


def _reject(journal: JournalRepo, pid: int, reason: str) -> None:
    journal.insert_execution(
        ExecutionRow(
            proposal_id=pid,
            decision="rejected",
            reject_reason=reason,
            submitted_orders_json="[]",
        )
    )


def _fill(
    journal: JournalRepo,
    eid: int,
    symbol: str,
    *,
    side: str,
    qty: int,
    price: float,
    at: dt.datetime,
) -> None:
    n = next(_seq)
    journal.insert_order(
        OrderRow(
            execution_id=eid,
            role="entry" if side == "buy" else "stop",
            symbol=symbol,
            side=side,
            order_type="market",
            qty=qty,
            tif="day",
            broker_order_id=f"bo-p-{n}",
            submitted_at=at,
            final_status="filled",
            realized_fill_price=price,
            realized_fill_qty=qty,
            realized_fill_at=at,
        )
    )


def _closed_trade(
    journal: JournalRepo,
    symbol: str,
    *,
    entry_at: dt.datetime,
    entry_px: float,
    exit_at: dt.datetime,
    exit_px: float,
    conviction: int = 7,
) -> int:
    pid = _seed_proposal(journal, symbol, conviction=conviction)
    eid = journal.insert_execution(
        ExecutionRow(proposal_id=pid, decision="accepted", submitted_orders_json="[]")
    )
    _fill(journal, eid, symbol, side="buy", qty=10, price=entry_px, at=entry_at)
    _fill(journal, eid, symbol, side="sell", qty=10, price=exit_px, at=exit_at)
    return eid


def _gcbc_losing_trade(journal: JournalRepo) -> int:
    """The docstring example: bought 32.85 on 2026-07-23, closed at
    31.20 on 2026-08-01 → -5.0%, held 9 calendar days."""
    return _closed_trade(
        journal,
        "GCBC",
        entry_at=dt.datetime(2026, 7, 23, 13, 31),
        entry_px=32.85,
        exit_at=dt.datetime(2026, 8, 1, 19, 55),
        exit_px=31.20,
    )


# ---------------------------------------------------------------------------
# None contract
# ---------------------------------------------------------------------------


def test_no_history_returns_none(journal: JournalRepo) -> None:
    provider = make_symbol_history_provider(journal)
    assert provider(_q("GCBC")) is None


def test_symbol_none_returns_none(journal: JournalRepo) -> None:
    provider = make_symbol_history_provider(journal)
    assert provider(_q(symbol=None)) is None


def test_proposal_review_purpose_not_served(journal: JournalRepo) -> None:
    _gcbc_losing_trade(journal)
    provider = make_symbol_history_provider(journal)
    assert provider(_q("GCBC", purpose="proposal_review")) is None


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def test_closed_trade_rendering(journal: JournalRepo) -> None:
    _gcbc_losing_trade(journal)
    provider = make_symbol_history_provider(journal)
    section = provider(_q("GCBC"))
    assert section is not None
    assert section.provider_name == "symbol_history"
    assert section.title == "Symbol history"
    assert section.body == (
        "Prior engagements with GCBC: "
        "[2026-07-23 bought 32.85, closed 2026-08-01 at 31.20 (-5.0%), held 9d]"
    )


def test_realized_pct_positive_sign_and_rounding(journal: JournalRepo) -> None:
    # (11.13 - 10.00) / 10.00 = +11.3%
    _closed_trade(
        journal,
        "ACME",
        entry_at=dt.datetime(2026, 7, 1, 14, 0),
        entry_px=10.00,
        exit_at=dt.datetime(2026, 7, 4, 14, 0),
        exit_px=11.13,
    )
    provider = make_symbol_history_provider(journal)
    section = provider(_q("ACME"))
    assert section is not None
    assert "(+11.3%)" in section.body
    assert "held 3d" in section.body


def test_not_traded_rendering(journal: JournalRepo) -> None:
    pid = _seed_proposal(
        journal, "GCBC", conviction=4, created_at=dt.datetime(2026, 6, 11, 13, 5)
    )
    _reject(journal, pid, "opus_reject")
    provider = make_symbol_history_provider(journal)
    section = provider(_q("GCBC"))
    assert section is not None
    assert section.body == (
        "Prior engagements with GCBC: [2026-06-11 analyzed cv4, rejected by review]"
    )


def test_reject_reason_labels(journal: JournalRepo) -> None:
    p1 = _seed_proposal(
        journal, "GCBC", conviction=6, created_at=dt.datetime(2026, 6, 1, 13, 0)
    )
    _reject(journal, p1, "insufficient_capital")
    p2 = _seed_proposal(
        journal, "GCBC", conviction=5, created_at=dt.datetime(2026, 6, 2, 13, 0)
    )
    _reject(journal, p2, "price_floor")
    provider = make_symbol_history_provider(journal)
    section = provider(_q("GCBC"))
    assert section is not None
    assert "[2026-06-02 analyzed cv5, rejected by validator]" in section.body
    assert "[2026-06-01 analyzed cv6, rejected on capital]" in section.body


def test_serves_thesis_review_and_excludes_current_engagement(
    journal: JournalRepo,
) -> None:
    _closed_trade(
        journal,
        "GCBC",
        entry_at=dt.datetime(2026, 6, 1, 14, 0),
        entry_px=10.0,
        exit_at=dt.datetime(2026, 6, 9, 14, 0),
        exit_px=11.0,
    )
    current = _gcbc_losing_trade(journal)
    provider = make_symbol_history_provider(journal)
    section = provider(_q("GCBC", purpose="thesis_review", execution_id=current))
    assert section is not None
    # Only the June trade renders; the current engagement is excluded.
    assert "2026-06-01 bought 10.00" in section.body
    assert "32.85" not in section.body


# ---------------------------------------------------------------------------
# postmortem lessons
# ---------------------------------------------------------------------------


def test_postmortem_lesson_appended(journal: JournalRepo) -> None:
    eid = _gcbc_losing_trade(journal)
    journal.insert_trade_postmortem(
        TradePostmortemRow(
            execution_id=eid,
            symbol="GCBC",
            lesson="thin float, exits get slippage — size down",
        )
    )
    provider = make_symbol_history_provider(journal)
    section = provider(_q("GCBC"))
    assert section is not None
    assert section.body.endswith(
        "held 9d — postmortem: thin float, exits get slippage — size down]"
    )


def test_postmortem_without_lesson_is_ignored(journal: JournalRepo) -> None:
    eid = _gcbc_losing_trade(journal)
    journal.insert_trade_postmortem(
        TradePostmortemRow(execution_id=eid, symbol="GCBC", lesson=None)
    )
    provider = make_symbol_history_provider(journal)
    section = provider(_q("GCBC"))
    assert section is not None
    assert "postmortem" not in section.body


# ---------------------------------------------------------------------------
# cap + determinism
# ---------------------------------------------------------------------------


def test_capped_at_three_newest_first(journal: JournalRepo) -> None:
    _closed_trade(  # oldest — must be dropped
        journal,
        "GCBC",
        entry_at=dt.datetime(2026, 2, 2, 14, 0),
        entry_px=8.0,
        exit_at=dt.datetime(2026, 2, 6, 14, 0),
        exit_px=8.4,
    )
    for month, day in ((5, 4), (6, 11), (7, 1)):
        pid = _seed_proposal(
            journal,
            "GCBC",
            conviction=4,
            created_at=dt.datetime(2026, month, day, 13, 0),
        )
        _reject(journal, pid, "opus_reject")
    provider = make_symbol_history_provider(journal)
    section = provider(_q("GCBC"))
    assert section is not None
    assert section.body.count("[") == 3
    assert "2026-02-02" not in section.body
    # newest first
    assert section.body.index("2026-07-01") < section.body.index("2026-06-11")
    assert section.body.index("2026-06-11") < section.body.index("2026-05-04")


def test_two_calls_byte_identical(journal: JournalRepo) -> None:
    eid = _gcbc_losing_trade(journal)
    journal.insert_trade_postmortem(
        TradePostmortemRow(execution_id=eid, symbol="GCBC", lesson="cut faster")
    )
    pid = _seed_proposal(
        journal, "GCBC", conviction=4, created_at=dt.datetime(2026, 6, 11, 13, 5)
    )
    _reject(journal, pid, "opus_reject")
    provider = make_symbol_history_provider(journal)
    first = provider(_q("GCBC"))
    second = provider(_q("GCBC"))
    assert first is not None and second is not None
    assert first == second
    assert first.body.encode() == second.body.encode()

"""Symbol-history repo query — `get_prior_engagements_for_symbol`.

Covers: empty result; closed-trade derivation (entry fill, last-sell
exit price, realized %, calendar days held); still-open and
partially-exited executions excluded; current-engagement exclusion via
`exclude_execution_id`; analyzed-but-rejected proposals with
latest-execution-wins semantics; newest-first ordering and the cap.
"""

from __future__ import annotations

import datetime as dt
import itertools
import sqlite3
from pathlib import Path

import pytest

from journal.migrate import apply_migrations
from journal.models import ExecutionRow, FilingRow, OrderRow, PromptRow, ProposalRow
from journal.repo import (
    JournalRepo,
    get_prior_engagements_for_symbol,
    get_prompt_by_version,
    insert_execution,
    insert_filing,
    insert_order,
    insert_prompt,
    insert_proposal,
)

_PV = "pv@aaaaaaaaaaaa"
_seq = itertools.count(1)


@pytest.fixture
def db(tmp_path: Path) -> str:
    p = tmp_path / "journal.db"
    apply_migrations(str(p))
    return str(p)


def _seed_proposal(
    db_path: str,
    symbol: str,
    *,
    conviction: int = 7,
    created_at: dt.datetime | None = None,
) -> int:
    n = next(_seq)
    if get_prompt_by_version(db_path, _PV) is None:
        insert_prompt(
            db_path,
            PromptRow(
                prompt_version=_PV,
                name="sonnet_filing_analysis_v2",
                file_path="src/prompts/sonnet_filing_analysis_v2.txt",
                content_hash="a" * 64,
            ),
        )
    fid = insert_filing(
        db_path,
        FilingRow(
            accession_number=f"acc-{n}",
            cik=1000000 + n,
            form_type="8-K",
            filed_at=dt.datetime(2026, 4, 1, 12, 0, 0),
            fetched_at=dt.datetime(2026, 4, 1, 12, 1, 0),
            raw_text_path=f"/raw/{n}.txt",
            content_hash=f"h-{n}",
        ),
    )
    pid = insert_proposal(
        db_path,
        ProposalRow(
            filing_id=fid,
            decision_id=f"dec-{n}",
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
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE proposals SET created_at = ? WHERE id = ?",
                (created_at.isoformat(sep=" "), pid),
            )
    return pid


def _seed_execution(
    db_path: str,
    pid: int,
    *,
    decision: str = "accepted",
    reject_reason: str | None = None,
) -> int:
    return insert_execution(
        db_path,
        ExecutionRow(
            proposal_id=pid,
            decision=decision,
            reject_reason=reject_reason,
            submitted_orders_json="[]",
        ),
    )


def _fill(
    db_path: str,
    eid: int,
    symbol: str,
    *,
    side: str,
    qty: int,
    price: float,
    at: dt.datetime,
    role: str | None = None,
) -> None:
    n = next(_seq)
    insert_order(
        db_path,
        OrderRow(
            execution_id=eid,
            role=role or ("entry" if side == "buy" else "stop"),
            symbol=symbol,
            side=side,
            order_type="market",
            qty=qty,
            tif="day",
            broker_order_id=f"bo-{n}",
            submitted_at=at,
            final_status="filled",
            realized_fill_price=price,
            realized_fill_qty=qty,
            realized_fill_at=at,
        ),
    )


def _closed_trade(
    db_path: str,
    symbol: str,
    *,
    entry_at: dt.datetime,
    entry_px: float,
    exit_at: dt.datetime,
    exit_px: float,
    qty: int = 10,
    conviction: int = 7,
) -> int:
    pid = _seed_proposal(db_path, symbol, conviction=conviction)
    eid = _seed_execution(db_path, pid)
    _fill(db_path, eid, symbol, side="buy", qty=qty, price=entry_px, at=entry_at)
    _fill(db_path, eid, symbol, side="sell", qty=qty, price=exit_px, at=exit_at)
    return eid


# ---------------------------------------------------------------------------
# empty / no-history
# ---------------------------------------------------------------------------


def test_no_history_returns_empty(db: str) -> None:
    assert get_prior_engagements_for_symbol(db, "GCBC") == []


def test_other_symbols_do_not_leak(db: str) -> None:
    _closed_trade(
        db,
        "ZZZZ",
        entry_at=dt.datetime(2026, 7, 1, 14, 0),
        entry_px=10.0,
        exit_at=dt.datetime(2026, 7, 5, 14, 0),
        exit_px=11.0,
    )
    assert get_prior_engagements_for_symbol(db, "GCBC") == []


# ---------------------------------------------------------------------------
# closed trades
# ---------------------------------------------------------------------------


def test_closed_trade_fields(db: str) -> None:
    eid = _closed_trade(
        db,
        "GCBC",
        entry_at=dt.datetime(2026, 7, 23, 13, 31),
        entry_px=32.85,
        exit_at=dt.datetime(2026, 8, 1, 19, 55),
        exit_px=31.20,
    )
    got = get_prior_engagements_for_symbol(db, "GCBC")
    assert len(got) == 1
    g = got[0]
    assert g.kind == "closed_trade"
    assert g.execution_id == eid
    assert g.entry_date == dt.date(2026, 7, 23)
    assert g.entry_price == 32.85
    assert g.date == dt.date(2026, 8, 1)  # exit-fill date
    assert g.exit_price == 31.20
    assert g.realized_pct == pytest.approx((31.20 - 32.85) / 32.85 * 100.0)
    assert g.days_held == 9  # calendar days, Jul 23 → Aug 1


def test_exit_price_is_last_sell_fill(db: str) -> None:
    """Scale-out: two sell fills → exit price/date come from the LAST
    sell fill (the one that took the position flat)."""
    pid = _seed_proposal(db, "GCBC")
    eid = _seed_execution(db, pid)
    _fill(db, eid, "GCBC", side="buy", qty=10, price=20.0, at=dt.datetime(2026, 7, 1, 14, 0))
    _fill(
        db, eid, "GCBC", side="sell", qty=5, price=22.0,
        at=dt.datetime(2026, 7, 3, 14, 0), role="take_profit",
    )
    _fill(
        db, eid, "GCBC", side="sell", qty=5, price=23.0,
        at=dt.datetime(2026, 7, 8, 14, 0), role="stop",
    )
    got = get_prior_engagements_for_symbol(db, "GCBC")
    assert len(got) == 1
    assert got[0].exit_price == 23.0
    assert got[0].date == dt.date(2026, 7, 8)
    assert got[0].days_held == 7


def test_open_execution_excluded(db: str) -> None:
    """Entry filled but no sell → still open, not a prior engagement."""
    pid = _seed_proposal(db, "GCBC")
    eid = _seed_execution(db, pid)
    _fill(db, eid, "GCBC", side="buy", qty=10, price=20.0, at=dt.datetime(2026, 7, 1, 14, 0))
    assert get_prior_engagements_for_symbol(db, "GCBC") == []


def test_partial_exit_still_open_excluded(db: str) -> None:
    """Sell fills covering less than the bought qty → not flat → open."""
    pid = _seed_proposal(db, "GCBC")
    eid = _seed_execution(db, pid)
    _fill(db, eid, "GCBC", side="buy", qty=10, price=20.0, at=dt.datetime(2026, 7, 1, 14, 0))
    _fill(db, eid, "GCBC", side="sell", qty=4, price=22.0, at=dt.datetime(2026, 7, 3, 14, 0))
    assert get_prior_engagements_for_symbol(db, "GCBC") == []


def test_exclude_execution_id_drops_current_engagement(db: str) -> None:
    prior = _closed_trade(
        db,
        "GCBC",
        entry_at=dt.datetime(2026, 6, 1, 14, 0),
        entry_px=10.0,
        exit_at=dt.datetime(2026, 6, 9, 14, 0),
        exit_px=11.0,
    )
    current = _closed_trade(
        db,
        "GCBC",
        entry_at=dt.datetime(2026, 7, 23, 14, 0),
        entry_px=32.85,
        exit_at=dt.datetime(2026, 8, 1, 14, 0),
        exit_px=31.20,
    )
    got = get_prior_engagements_for_symbol(db, "GCBC", exclude_execution_id=current)
    assert [g.execution_id for g in got] == [prior]


# ---------------------------------------------------------------------------
# analyzed-but-not-traded
# ---------------------------------------------------------------------------


def test_analyzed_rejected_fields(db: str) -> None:
    pid = _seed_proposal(
        db, "GCBC", conviction=4, created_at=dt.datetime(2026, 6, 11, 13, 5)
    )
    _seed_execution(db, pid, decision="rejected", reject_reason="opus_reject")
    got = get_prior_engagements_for_symbol(db, "GCBC")
    assert len(got) == 1
    g = got[0]
    assert g.kind == "analyzed_no_trade"
    assert g.date == dt.date(2026, 6, 11)
    assert g.conviction == 4
    assert g.reject_reason == "opus_reject"
    assert g.execution_id is None


def test_latest_execution_wins_over_pending_capacity(db: str) -> None:
    """A pending_capacity placeholder superseded by an accepted retro
    row must NOT surface the proposal as analyzed-but-rejected — the
    trade happened; only the closed trade is reported."""
    pid = _seed_proposal(db, "GCBC", conviction=8)
    _seed_execution(db, pid, decision="rejected", reject_reason="pending_capacity")
    eid = _seed_execution(db, pid, decision="accepted")
    _fill(db, eid, "GCBC", side="buy", qty=10, price=20.0, at=dt.datetime(2026, 7, 1, 14, 0))
    _fill(db, eid, "GCBC", side="sell", qty=10, price=21.0, at=dt.datetime(2026, 7, 6, 14, 0))
    got = get_prior_engagements_for_symbol(db, "GCBC")
    assert [g.kind for g in got] == ["closed_trade"]


def test_accepted_but_open_appears_nowhere(db: str) -> None:
    """An accepted, entry-filled, still-open execution is neither a
    closed trade nor an analyzed-no-trade engagement."""
    pid = _seed_proposal(db, "GCBC")
    eid = _seed_execution(db, pid)
    _fill(db, eid, "GCBC", side="buy", qty=10, price=20.0, at=dt.datetime(2026, 7, 1, 14, 0))
    assert get_prior_engagements_for_symbol(db, "GCBC") == []


# ---------------------------------------------------------------------------
# ordering + cap
# ---------------------------------------------------------------------------


def test_newest_first_and_capped(db: str) -> None:
    _closed_trade(  # oldest — must fall off the cap
        db,
        "GCBC",
        entry_at=dt.datetime(2026, 3, 2, 14, 0),
        entry_px=10.0,
        exit_at=dt.datetime(2026, 3, 9, 14, 0),
        exit_px=9.0,
    )
    pid = _seed_proposal(
        db, "GCBC", conviction=4, created_at=dt.datetime(2026, 6, 11, 13, 0)
    )
    _seed_execution(db, pid, decision="rejected", reject_reason="opus_reject")
    _closed_trade(
        db,
        "GCBC",
        entry_at=dt.datetime(2026, 7, 23, 14, 0),
        entry_px=32.85,
        exit_at=dt.datetime(2026, 8, 1, 14, 0),
        exit_px=31.20,
    )
    pid2 = _seed_proposal(
        db, "GCBC", conviction=6, created_at=dt.datetime(2026, 5, 4, 13, 0)
    )
    _seed_execution(db, pid2, decision="rejected", reject_reason="insufficient_capital")

    got = get_prior_engagements_for_symbol(db, "GCBC", limit=3)
    assert len(got) == 3
    assert [g.date for g in got] == [
        dt.date(2026, 8, 1),
        dt.date(2026, 6, 11),
        dt.date(2026, 5, 4),
    ]
    assert [g.kind for g in got] == [
        "closed_trade",
        "analyzed_no_trade",
        "analyzed_no_trade",
    ]


def test_journal_repo_facade_binding(db: str) -> None:
    _closed_trade(
        db,
        "GCBC",
        entry_at=dt.datetime(2026, 7, 23, 14, 0),
        entry_px=32.85,
        exit_at=dt.datetime(2026, 8, 1, 14, 0),
        exit_px=31.20,
    )
    repo = JournalRepo(db)
    got = repo.get_prior_engagements_for_symbol("GCBC")
    assert len(got) == 1
    assert got[0].kind == "closed_trade"

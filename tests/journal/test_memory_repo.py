"""Migrations 009/010 (desk_memory, trade_postmortems) + their repo
helpers — the schema stubs the LLM memory providers build on.

Covers: migrations apply on a fresh (v8-and-beyond) schema and
`verify_schema` passes; desk-memory insert/active-read (version + active
filtering); post-mortem insert/read-by-symbol with the executions FK
satisfied.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

import pytest

from journal.migrate import apply_migrations, verify_schema
from journal.models import (
    DeskMemoryRow,
    ExecutionRow,
    FilingRow,
    PromptRow,
    ProposalRow,
    TradePostmortemRow,
)
from journal.repo import (
    get_active_desk_memory,
    get_postmortems_for_symbol,
    get_prompt_by_version,
    insert_desk_memory,
    insert_execution,
    insert_filing,
    insert_prompt,
    insert_proposal,
    insert_trade_postmortem,
)

_PROMPT_VERSION = "pv@aaaaaaaaaaaa"


@pytest.fixture
def db(tmp_path: Path) -> str:
    p = tmp_path / "journal.db"
    apply_migrations(str(p))
    return str(p)


def _table_names(db_path: str) -> set[str]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    return {r[0] for r in rows}


def _index_names(db_path: str) -> set[str]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    return {r[0] for r in rows}


# ---------------------------------------------------------------------------
# migration application
# ---------------------------------------------------------------------------


def test_migrations_009_010_apply_and_verify(db: str) -> None:
    tables = _table_names(db)
    assert "desk_memory" in tables
    assert "trade_postmortems" in tables

    indexes = _index_names(db)
    assert "idx_desk_memory_active_kind" in indexes
    assert "idx_desk_memory_one_active" in indexes
    assert "idx_trade_postmortems_symbol" in indexes

    verify_schema(db)  # no raise → on-disk set matches meta


def test_desk_memory_kind_check_constraint(db: str) -> None:
    with sqlite3.connect(db) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO desk_memory (kind, content) VALUES ('bogus', 'x')"
            )


# ---------------------------------------------------------------------------
# desk_memory repo helpers
# ---------------------------------------------------------------------------


def test_desk_memory_roundtrip(db: str) -> None:
    new_id = insert_desk_memory(
        db, DeskMemoryRow(kind="doctrine", content="cut losers fast")
    )
    assert new_id > 0
    got = get_active_desk_memory(db, "doctrine")
    assert got is not None
    assert got.content == "cut losers fast"
    assert got.active == 1
    assert got.version == 1


def test_get_active_desk_memory_none_when_absent(db: str) -> None:
    assert get_active_desk_memory(db, "synthesis") is None


def test_get_active_desk_memory_picks_highest_version(db: str) -> None:
    # Retired history plus ONE active row — the 009 partial unique index
    # (idx_desk_memory_one_active) forbids concurrent active versions.
    insert_desk_memory(
        db, DeskMemoryRow(kind="doctrine", content="v1", version=1, active=0)
    )
    insert_desk_memory(db, DeskMemoryRow(kind="doctrine", content="v3", version=3))
    insert_desk_memory(
        db, DeskMemoryRow(kind="doctrine", content="v2", version=2, active=0)
    )
    got = get_active_desk_memory(db, "doctrine")
    assert got is not None
    assert got.content == "v3"


def test_get_active_desk_memory_ignores_inactive(db: str) -> None:
    insert_desk_memory(
        db, DeskMemoryRow(kind="doctrine", content="retired", version=5, active=0)
    )
    insert_desk_memory(
        db, DeskMemoryRow(kind="doctrine", content="live", version=1, active=1)
    )
    got = get_active_desk_memory(db, "doctrine")
    assert got is not None
    assert got.content == "live"


def test_desk_memory_kinds_are_independent(db: str) -> None:
    insert_desk_memory(db, DeskMemoryRow(kind="doctrine", content="D"))
    insert_desk_memory(db, DeskMemoryRow(kind="synthesis", content="S"))
    assert get_active_desk_memory(db, "doctrine").content == "D"  # type: ignore[union-attr]
    assert get_active_desk_memory(db, "synthesis").content == "S"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# trade_postmortems repo helpers
# ---------------------------------------------------------------------------


def _seed_execution(db_path: str, symbol: str = "ACME") -> int:
    if get_prompt_by_version(db_path, _PROMPT_VERSION) is None:
        insert_prompt(
            db_path,
            PromptRow(
                prompt_version=_PROMPT_VERSION,
                name="sonnet_filing_analysis_v2",
                file_path="src/prompts/sonnet_filing_analysis_v2.txt",
                content_hash="a" * 64,
            ),
        )
    fid = insert_filing(
        db_path,
        FilingRow(
            accession_number=f"acc-{symbol}",
            cik=1234567,
            form_type="8-K",
            filed_at=dt.datetime(2026, 4, 28, 14, 30, 0),
            fetched_at=dt.datetime(2026, 4, 28, 14, 31, 0),
            raw_text_path=f"/raw/{symbol}.txt",
            content_hash=f"h-{symbol}",
        ),
    )
    pid = insert_proposal(
        db_path,
        ProposalRow(
            filing_id=fid,
            decision_id=f"dec-{symbol}",
            model_id="claude-sonnet-4-6",
            prompt_version="pv@aaaaaaaaaaaa",
            raw_response="{}",
            kind="trade_proposal",
            symbol=symbol,
            direction="long",
            size_pct_requested=0.05,
            conviction=8,
            thesis="x",
            input_tokens=10,
            output_tokens=10,
            latency_ms=100,
            cost_usd=0.001,
        ),
    )
    return insert_execution(
        db_path,
        ExecutionRow(
            proposal_id=pid, decision="accepted", submitted_orders_json="[]"
        ),
    )


def test_trade_postmortem_roundtrip_by_symbol(db: str) -> None:
    eid = _seed_execution(db, "ACME")
    pm_id = insert_trade_postmortem(
        db,
        TradePostmortemRow(
            execution_id=eid,
            symbol="ACME",
            thesis_summary="8-K item 2.02 beat",
            outcome_summary="+18% in 6 trading days",
            exit_quality_pct=72.5,
            lesson="held through the first pullback — correct",
        ),
    )
    assert pm_id > 0

    rows = get_postmortems_for_symbol(db, "ACME")
    assert len(rows) == 1
    assert rows[0].exit_quality_pct == 72.5
    assert rows[0].lesson == "held through the first pullback — correct"


def test_get_postmortems_for_symbol_empty(db: str) -> None:
    assert get_postmortems_for_symbol(db, "NONE") == []


def test_trade_postmortem_fk_requires_execution(db: str) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        insert_trade_postmortem(
            db,
            TradePostmortemRow(execution_id=999999, symbol="GHOST"),
        )


def test_postmortems_scoped_to_symbol(db: str) -> None:
    e1 = _seed_execution(db, "ACME")
    e2 = _seed_execution(db, "ZZZZ")
    insert_trade_postmortem(db, TradePostmortemRow(execution_id=e1, symbol="ACME"))
    insert_trade_postmortem(db, TradePostmortemRow(execution_id=e2, symbol="ZZZZ"))
    assert len(get_postmortems_for_symbol(db, "ACME")) == 1
    assert get_postmortems_for_symbol(db, "ACME")[0].symbol == "ACME"

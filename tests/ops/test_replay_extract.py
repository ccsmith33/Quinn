"""Tests for journal proposal extraction (`ops/replay/extract.py`).

Covers the three raw_response shapes (direct / `trade`-wrapped /
`proposal`-wrapped — same resolution strategy as ops/scripts/place_stops.py),
kind/direction filtering, skip reasons, and the read-only guarantee.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path

import pytest
from ops.replay.extract import load_proposals

from journal.migrate import apply_migrations
from journal.models import ExecutionRow, FilingRow, PromptRow, ProposalRow
from journal.repo import insert_execution, insert_filing, insert_prompt, insert_proposal


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    p = str(tmp_path / "journal.db")
    apply_migrations(p)
    insert_prompt(
        p,
        PromptRow(
            prompt_version="v1", name="analyzer", file_path="x.md", content_hash="h"
        ),
    )
    return p


def _filing(db: str, n: int) -> int:
    return insert_filing(
        db,
        FilingRow(
            accession_number=f"0000000000-26-{n:06d}",
            cik=1000 + n,
            form_type="8-K",
            filed_at=dt.datetime(2026, 5, 1, tzinfo=dt.UTC),
            fetched_at=dt.datetime(2026, 5, 1, tzinfo=dt.UTC),
            raw_text_path=f"/tmp/f{n}.txt",
            content_hash=f"hash{n}",
        ),
    )


def _proposal(
    db: str,
    n: int,
    raw_response: dict,
    kind: str = "trade_proposal",
    symbol: str | None = "ABCD",
    direction: str | None = "long",
    created_at: str = "2026-05-04 12:00:00",
) -> int:
    pid = insert_proposal(
        db,
        ProposalRow(
            filing_id=_filing(db, n),
            decision_id=f"dec-{n}",
            model_id="claude-opus-4-7",
            prompt_version="v1",
            raw_response=json.dumps(raw_response),
            kind=kind,
            symbol=symbol,
            direction=direction,
            conviction=7,
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
            cost_usd=0.0,
        ),
    )
    conn = sqlite3.connect(db)
    conn.execute("UPDATE proposals SET created_at = ? WHERE id = ?", (created_at, pid))
    conn.commit()
    conn.close()
    return pid


GEO = {"stop_loss_price": 8.0, "take_profit_price": 14.0, "time_horizon_days": 21}


class TestRawResponseShapes:
    def test_direct_shape(self, db_path):
        _proposal(db_path, 1, GEO)
        proposals, skipped = load_proposals(db_path)
        assert not skipped
        (p,) = proposals
        assert p.stop_loss_price == 8.0
        assert p.take_profit_price == 14.0
        assert p.time_horizon_days == 21
        assert p.created_at_utc == dt.datetime(2026, 5, 4, 12, 0, tzinfo=dt.UTC)

    def test_trade_wrapped_shape(self, db_path):
        _proposal(db_path, 1, {"decision": "trade", "trade": GEO})
        proposals, skipped = load_proposals(db_path)
        assert not skipped
        assert proposals[0].stop_loss_price == 8.0

    def test_proposal_wrapped_shape(self, db_path):
        _proposal(db_path, 1, {"proposal": GEO})
        proposals, skipped = load_proposals(db_path)
        assert not skipped
        assert proposals[0].stop_loss_price == 8.0

    def test_missing_tp_and_horizon_are_none(self, db_path):
        _proposal(db_path, 1, {"stop_loss_price": 8.0})
        (p,), skipped = load_proposals(db_path)
        assert not skipped
        assert p.take_profit_price is None
        assert p.time_horizon_days is None


class TestFilteringAndSkips:
    def test_no_trade_kind_excluded_entirely(self, db_path):
        _proposal(db_path, 1, {"decision": "no_trade"}, kind="no_trade", symbol=None)
        proposals, skipped = load_proposals(db_path)
        assert proposals == [] and skipped == []

    def test_missing_stop_is_skipped_with_reason(self, db_path):
        _proposal(db_path, 1, {"take_profit_price": 14.0})
        proposals, skipped = load_proposals(db_path)
        assert proposals == []
        assert "stop_loss_price" in skipped[0].reason

    def test_non_long_direction_is_skipped(self, db_path):
        _proposal(db_path, 1, GEO, direction="short")
        proposals, skipped = load_proposals(db_path)
        assert proposals == []
        assert "direction=short" in skipped[0].reason

    def test_unparseable_raw_response_is_skipped(self, db_path):
        pid = _proposal(db_path, 1, GEO)
        conn = sqlite3.connect(db_path)
        conn.execute("UPDATE proposals SET raw_response = 'not json' WHERE id = ?", (pid,))
        conn.commit()
        conn.close()
        proposals, skipped = load_proposals(db_path)
        assert proposals == [] and len(skipped) == 1

    def test_exec_decision_joined_when_present(self, db_path):
        pid = _proposal(db_path, 1, GEO)
        insert_execution(db_path, ExecutionRow(proposal_id=pid, decision="submitted"))
        (p,), _ = load_proposals(db_path)
        assert p.exec_decision == "submitted"

    def test_exec_decision_none_when_absent(self, db_path):
        _proposal(db_path, 1, GEO)
        (p,), _ = load_proposals(db_path)
        assert p.exec_decision is None

    def test_ordering_by_created_at(self, db_path):
        _proposal(db_path, 1, GEO, created_at="2026-05-06 12:00:00")
        _proposal(db_path, 2, GEO, created_at="2026-05-02 12:00:00")
        proposals, _ = load_proposals(db_path)
        assert [p.created_at_utc.day for p in proposals] == [2, 6]


def test_journal_is_opened_read_only(db_path):
    # Removing write permission must not affect extraction.
    import os

    os.chmod(db_path, 0o444)
    try:
        _ = load_proposals(db_path)
    finally:
        os.chmod(db_path, 0o644)

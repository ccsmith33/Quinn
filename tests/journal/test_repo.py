"""S1.3 — Journal repository tests.

Architecture references: §2.9 (journal owns all writes), §8.1 (tables),
§8.2 (append-only), NFR-16.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
import threading
from pathlib import Path

import pytest

from journal.migrate import apply_migrations
from journal.models import (
    FilingRow,
    KillSwitchStateRow,
    PromptRow,
    ProposalRow,
    UniverseMemberRow,
    UniverseSnapshotRow,
)
from journal.repo import (
    DuplicateAccession,
    connect,
    get_current_universe_member,
    get_filing_by_id,
    get_latest_filing_for_issuer_form,
    get_latest_kill_switch_state,
    get_prompt_by_version,
    get_proposal_by_decision_id,
    insert_filing,
    insert_kill_switch_state,
    insert_prompt,
    insert_proposal,
    insert_universe_member,
    insert_universe_snapshot,
)


@pytest.fixture
def db(tmp_path: Path) -> str:
    db_path = tmp_path / "journal.db"
    apply_migrations(str(db_path))
    return str(db_path)


def _filing(**overrides: object) -> FilingRow:
    base = dict(
        accession_number="0001234567-26-000001",
        cik=1234567,
        form_type="8-K",
        filed_at=dt.datetime(2026, 4, 28, 14, 30, 0),
        fetched_at=dt.datetime(2026, 4, 28, 14, 31, 0),
        raw_text_path="/var/lib/quinn/raw/0001234567-26-000001.txt",
        content_hash="abc123",
        item_codes='["1.01"]',
        issuer_ticker="ACME",
        ingest_state="ok",
        ingest_error=None,
    )
    base.update(overrides)
    return FilingRow(**base)


def test_insert_and_get_filing_roundtrip(db: str) -> None:
    f = _filing()
    new_id = insert_filing(db, f)
    assert isinstance(new_id, int) and new_id > 0
    got = get_filing_by_id(db, new_id)
    assert got is not None
    assert got.accession_number == f.accession_number
    assert got.cik == f.cik
    assert got.form_type == f.form_type
    assert got.content_hash == f.content_hash


def test_duplicate_accession_raises(db: str) -> None:
    f = _filing()
    insert_filing(db, f)
    with pytest.raises(DuplicateAccession) as exc_info:
        insert_filing(db, f)
    assert exc_info.value.accession_number == f.accession_number
    # original row id is retrievable per AC-4
    assert isinstance(exc_info.value.existing_id, int) and exc_info.value.existing_id > 0


def test_get_latest_filing_for_issuer_form(db: str) -> None:
    older = _filing(
        accession_number="0001234567-26-000001",
        filed_at=dt.datetime(2026, 4, 1, 10, 0, 0),
    )
    newer = _filing(
        accession_number="0001234567-26-000002",
        filed_at=dt.datetime(2026, 4, 28, 10, 0, 0),
    )
    insert_filing(db, older)
    insert_filing(db, newer)
    latest = get_latest_filing_for_issuer_form(db, cik=1234567, form_type="8-K")
    assert latest is not None
    assert latest.accession_number == newer.accession_number


def test_get_latest_filing_returns_none_when_absent(db: str) -> None:
    assert get_latest_filing_for_issuer_form(db, cik=999, form_type="10-K") is None


def test_get_latest_kill_switch_state_seeded_active(db: str) -> None:
    """Seed row inserted by 001_init.sql per S1.3 tactical clarification."""
    state = get_latest_kill_switch_state(db)
    assert state is not None
    assert state.state == "active"
    assert state.set_by == "system"


def test_kill_switch_state_append_only_returns_latest(db: str) -> None:
    insert_kill_switch_state(
        db,
        KillSwitchStateRow(
            set_at=dt.datetime(2026, 4, 28, 15, 0, 0),
            state="halted",
            reason="manual:telegram",
            set_by="operator",
            notes=None,
        ),
    )
    latest = get_latest_kill_switch_state(db)
    assert latest.state == "halted"
    assert latest.reason == "manual:telegram"


def test_universe_current_member_lookup(db: str) -> None:
    snap = UniverseSnapshotRow(
        snapshot_date=dt.date(2026, 4, 28),
        sec_tickers_hash="sec_h",
        alpaca_assets_hash="alp_h",
        yfinance_failures=0,
        member_count=1,
        is_degraded=0,
    )
    snap_id = insert_universe_snapshot(db, snap)
    insert_universe_member(
        db,
        UniverseMemberRow(
            snapshot_id=snap_id,
            cik=1234567,
            ticker="ACME",
            exchange="NASDAQ",
            market_cap=500_000_000.0,
            prev_close=12.34,
        ),
    )
    member = get_current_universe_member(db, ticker="ACME")
    assert member is not None
    assert member.ticker == "ACME"
    assert member.cik == 1234567
    assert get_current_universe_member(db, ticker="NOPE") is None


def test_prompt_and_proposal_lookup(db: str) -> None:
    prompt = PromptRow(
        prompt_version="sonnet_filing_analysis@a1b2c3d4e5f6",
        name="sonnet_filing_analysis",
        file_path="src/prompts/sonnet_filing_analysis_v1.txt",
        content_hash="a1b2c3d4e5f6",
    )
    insert_prompt(db, prompt)
    fetched = get_prompt_by_version(db, prompt.prompt_version)
    assert fetched is not None
    assert fetched.name == prompt.name

    f = _filing()
    filing_id = insert_filing(db, f)
    p = ProposalRow(
        filing_id=filing_id,
        decision_id="dec_xyz_001",
        model_id="claude-sonnet-4-6",
        prompt_version=prompt.prompt_version,
        raw_response='{"conviction": 7}',
        kind="trade_proposal",
        symbol="ACME",
        direction="long",
        size_pct_requested=0.05,
        conviction=7,
        thesis="strong 8-K",
        input_tokens=1000,
        output_tokens=200,
        latency_ms=1234,
        cost_usd=0.012,
    )
    insert_proposal(db, p)
    got = get_proposal_by_decision_id(db, "dec_xyz_001")
    assert got is not None
    assert got.symbol == "ACME"
    assert got.conviction == 7
    assert get_proposal_by_decision_id(db, "missing") is None


def test_no_update_or_delete_on_append_only_tables() -> None:
    """AC-5: defense-in-depth — repo.py contains no UPDATE/DELETE against append-only tables."""
    src = (Path(__file__).resolve().parent.parent.parent / "src" / "journal" / "repo.py").read_text(
        encoding="utf-8"
    )
    upper = src.upper()
    append_only_tables = [
        "PROPOSALS",
        "PROPOSAL_REVIEWS",
        "EXECUTIONS",
        "ORDERS",
        "POSITIONS",
        "ACCOUNT_SNAPSHOTS",
        "KILL_SWITCH_STATE",
        "UNIVERSE_SNAPSHOTS",
        "UNIVERSE_MEMBERS",
        "LLM_CALLS",
        "BACKUPS",
    ]
    for tbl in append_only_tables:
        assert f"UPDATE {tbl}" not in upper, f"UPDATE on append-only table {tbl} found in repo.py"
        assert f"DELETE FROM {tbl}" not in upper, (
            f"DELETE on append-only table {tbl} found in repo.py"
        )


def test_concurrent_inserts_persist(db: str) -> None:
    """AC-2: WAL allows concurrent writers (serialized via SQLite); all rows persist."""
    errors: list[Exception] = []

    def worker(idx: int) -> None:
        try:
            insert_filing(
                db,
                _filing(
                    accession_number=f"0009999999-26-{idx:06d}",
                    cik=9999999,
                ),
            )
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    with connect(db) as conn:
        n = conn.execute("SELECT COUNT(*) FROM filings WHERE cik=9999999").fetchone()[0]
    assert n == 5


def test_connect_returns_context_manager(db: str) -> None:
    with connect(db) as conn:
        assert isinstance(conn, sqlite3.Connection)
        conn.execute("SELECT 1")

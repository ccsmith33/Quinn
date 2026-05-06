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


# ---------------------------------------------------------------------------
# Feature C — retro-fill helpers
# ---------------------------------------------------------------------------

def _proposal(
    db: str,
    *,
    prompt_version: str,
    decision_id: str,
    symbol: str = "ACME",
    conviction: int = 9,
    kind: str = "trade_proposal",
    created_at: dt.datetime | None = None,
    seed_pending_capacity: bool = True,
) -> int:
    """Insert a proposal for retro-fill tests. Optionally back-date
    `created_at` (the SQL helper enforces freshness by created_at).

    By default also seeds the `pending_capacity` execution row that
    Feature B writes when its capacity gate fires — that row is the
    deterministic signal `find_retro_candidate` keys on. Pass
    `seed_pending_capacity=False` for tests that simulate other
    lifecycle states (e.g., proposal already reviewed by another path).
    """
    from journal.models import ExecutionRow
    from journal.repo import insert_execution

    f = _filing(accession_number=f"acc-{decision_id}", content_hash=f"h-{decision_id}")
    fid = insert_filing(db, f)
    p = ProposalRow(
        filing_id=fid,
        decision_id=decision_id,
        model_id="claude-sonnet-4-6",
        prompt_version=prompt_version,
        raw_response='{"conviction": ' + str(conviction) + "}",
        kind=kind,
        symbol=symbol if kind == "trade_proposal" else None,
        direction="long" if kind == "trade_proposal" else None,
        size_pct_requested=0.05 if kind == "trade_proposal" else None,
        conviction=conviction if kind == "trade_proposal" else None,
        thesis="x" if kind == "trade_proposal" else None,
        input_tokens=10,
        output_tokens=10,
        latency_ms=100,
        cost_usd=0.001,
    )
    pid = insert_proposal(db, p)
    if created_at is not None:
        with connect(db) as conn:
            conn.execute(
                "UPDATE proposals SET created_at = ? WHERE id = ?",
                (created_at, pid),
            )
            conn.commit()
    if seed_pending_capacity and kind == "trade_proposal":
        insert_execution(
            db,
            ExecutionRow(
                proposal_id=pid,
                decision="rejected",
                reject_reason="pending_capacity",
                submitted_orders_json="[]",
            ),
        )
    return pid


def _seed_prompt(db: str) -> str:
    pv = "sonnet_filing_analysis@deadbeef0001"
    insert_prompt(
        db,
        PromptRow(
            prompt_version=pv,
            name="sonnet_filing_analysis",
            file_path="src/prompts/sonnet_filing_analysis_v1.txt",
            content_hash="deadbeef",
        ),
    )
    return pv


def test_find_retro_candidate_picks_cv9_in_window(db: str) -> None:
    from journal.repo import find_retro_candidate

    pv = _seed_prompt(db)
    now = dt.datetime(2026, 5, 6, 14, 0, 0)
    pid = _proposal(
        db, prompt_version=pv, decision_id="d1", conviction=9,
        created_at=now - dt.timedelta(hours=2),
    )
    got = find_retro_candidate(
        db, min_conviction=9, not_older_than=now - dt.timedelta(hours=6)
    )
    assert got is not None
    assert got.id == pid
    assert got.conviction == 9


def test_find_retro_candidate_excludes_cv8(db: str) -> None:
    from journal.repo import find_retro_candidate

    pv = _seed_prompt(db)
    now = dt.datetime(2026, 5, 6, 14, 0, 0)
    _proposal(
        db, prompt_version=pv, decision_id="d2", conviction=8,
        created_at=now - dt.timedelta(hours=1),
    )
    got = find_retro_candidate(
        db, min_conviction=9, not_older_than=now - dt.timedelta(hours=6)
    )
    assert got is None


def test_find_retro_candidate_excludes_stale(db: str) -> None:
    from journal.repo import find_retro_candidate

    pv = _seed_prompt(db)
    now = dt.datetime(2026, 5, 6, 14, 0, 0)
    _proposal(
        db, prompt_version=pv, decision_id="d3", conviction=10,
        created_at=now - dt.timedelta(hours=7),
    )
    got = find_retro_candidate(
        db, min_conviction=9, not_older_than=now - dt.timedelta(hours=6)
    )
    assert got is None


def test_find_retro_candidate_excludes_already_reviewed(db: str) -> None:
    from journal.repo import find_retro_candidate, insert_proposal_review
    from journal.models import ProposalReviewRow

    pv = _seed_prompt(db)
    now = dt.datetime(2026, 5, 6, 14, 0, 0)
    pid = _proposal(
        db, prompt_version=pv, decision_id="d4", conviction=9,
        created_at=now - dt.timedelta(hours=1),
    )
    # Add a review (e.g., from production review path) → no longer a candidate.
    insert_proposal_review(
        db,
        ProposalReviewRow(
            proposal_id=pid,
            model_id="claude-opus-4-7",
            prompt_version=pv,
            decision="ratify",
            raw_response="{}",
            rationale="x" * 60,
            input_tokens=10,
            output_tokens=10,
            latency_ms=100,
            cost_usd=0.001,
        ),
    )
    got = find_retro_candidate(
        db, min_conviction=9, not_older_than=now - dt.timedelta(hours=6)
    )
    assert got is None


def test_find_retro_candidate_excludes_already_executed(db: str) -> None:
    from journal.repo import find_retro_candidate, insert_execution
    from journal.models import ExecutionRow

    pv = _seed_prompt(db)
    now = dt.datetime(2026, 5, 6, 14, 0, 0)
    pid = _proposal(
        db, prompt_version=pv, decision_id="d5", conviction=9,
        created_at=now - dt.timedelta(hours=1),
    )
    insert_execution(
        db,
        ExecutionRow(
            proposal_id=pid,
            decision="rejected",
            reject_reason="kill_switch",
            submitted_orders_json="[]",
        ),
    )
    got = find_retro_candidate(
        db, min_conviction=9, not_older_than=now - dt.timedelta(hours=6)
    )
    assert got is None


def test_find_retro_candidate_two_cv9_picks_most_recent(db: str) -> None:
    from journal.repo import find_retro_candidate

    pv = _seed_prompt(db)
    now = dt.datetime(2026, 5, 6, 14, 0, 0)
    older = _proposal(
        db, prompt_version=pv, decision_id="d6a", conviction=9,
        created_at=now - dt.timedelta(hours=4),
    )
    newer = _proposal(
        db, prompt_version=pv, decision_id="d6b", conviction=9,
        created_at=now - dt.timedelta(hours=1),
    )
    got = find_retro_candidate(
        db, min_conviction=9, not_older_than=now - dt.timedelta(hours=6)
    )
    assert got is not None
    assert got.id == newer
    assert got.id != older


def test_find_retro_candidate_higher_conviction_wins(db: str) -> None:
    from journal.repo import find_retro_candidate

    pv = _seed_prompt(db)
    now = dt.datetime(2026, 5, 6, 14, 0, 0)
    _proposal(
        db, prompt_version=pv, decision_id="d7a", conviction=9,
        created_at=now - dt.timedelta(hours=1),
    )
    cv10_pid = _proposal(
        db, prompt_version=pv, decision_id="d7b", conviction=10,
        created_at=now - dt.timedelta(hours=2),
    )
    got = find_retro_candidate(
        db, min_conviction=9, not_older_than=now - dt.timedelta(hours=6)
    )
    assert got is not None
    assert got.id == cv10_pid


def test_has_open_position_true_when_qty_nonzero(db: str) -> None:
    from journal.repo import has_open_position, insert_position
    from journal.models import PositionRow

    insert_position(
        db,
        PositionRow(
            snapshot_at=dt.datetime(2026, 5, 6, 14, 0, 0),
            source="reconciler",
            symbol="ACME",
            qty=100,
            avg_entry_price=10.0,
            market_value=1010.0,
            unrealized_pnl=10.0,
        ),
    )
    assert has_open_position(db, "ACME") is True
    assert has_open_position(db, "OTHER") is False


def test_has_open_position_false_when_latest_snapshot_zero_qty(db: str) -> None:
    """A symbol that was held but is now closed (qty=0 latest snapshot)
    must not block a retro candidate."""
    from journal.repo import has_open_position, insert_position
    from journal.models import PositionRow

    insert_position(
        db,
        PositionRow(
            snapshot_at=dt.datetime(2026, 5, 6, 14, 0, 0),
            source="reconciler",
            symbol="ACME",
            qty=100,
            avg_entry_price=10.0,
            market_value=1010.0,
            unrealized_pnl=10.0,
        ),
    )
    insert_position(
        db,
        PositionRow(
            snapshot_at=dt.datetime(2026, 5, 6, 15, 0, 0),
            source="reconciler",
            symbol="ACME",
            qty=0,
            avg_entry_price=10.0,
            market_value=0.0,
            unrealized_pnl=0.0,
        ),
    )
    assert has_open_position(db, "ACME") is False

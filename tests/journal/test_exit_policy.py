"""D-079 §3.5 — exit_policy_state DAO (migration 006).

Operational state for the trailing ratchet: one row per execution,
updated in place. The DAO lives in src/journal/exit_policy.py (not
repo.py) per the delta §1 ownership seam.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from journal.exit_policy import (
    ExitPolicyStateRow,
    get_exit_policy_state,
    set_stop_order_journal_id,
    upsert_exit_policy_state,
)
from journal.migrate import apply_migrations
from journal.models import ExecutionRow, FilingRow, PromptRow, ProposalRow
from journal.repo import (
    insert_execution,
    insert_filing,
    insert_prompt,
    insert_proposal,
)


@pytest.fixture
def db(tmp_path: Path) -> str:
    p = str(tmp_path / "journal.db")
    apply_migrations(p)
    return p


def _execution(db: str) -> int:
    fid = insert_filing(
        db,
        FilingRow(
            accession_number="eps-acc-1",
            cik=320193,
            form_type="8-K",
            filed_at=dt.datetime(2026, 6, 8, 14, 30),
            fetched_at=dt.datetime(2026, 6, 8, 14, 31),
            raw_text_path="/tmp/eps.txt",
            content_hash="h1",
            item_codes='["1.01"]',
            issuer_ticker="ACME",
        ),
    )
    insert_prompt(
        db,
        PromptRow(
            prompt_version="sonnet@epstest",
            name="sonnet",
            file_path="src/prompts/sonnet.txt",
            content_hash="x" * 64,
        ),
    )
    pid = insert_proposal(
        db,
        ProposalRow(
            filing_id=fid,
            decision_id="eps-d-1",
            model_id="claude-sonnet-4-6",
            prompt_version="sonnet@epstest",
            raw_response="{}",
            kind="trade_proposal",
            symbol="ACME",
            direction="long",
            size_pct_requested=0.05,
            conviction=8,
            thesis="thesis",
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
            realized_size_pct=0.05,
            realized_dollar_size=250.0,
            submitted_orders_json="[]",
        ),
    )


def test_migration_006_applies(db: str) -> None:
    from journal.repo import connect

    with connect(db) as conn:
        cols = {
            r["name"]
            for r in conn.execute("PRAGMA table_info(exit_policy_state)")
        }
    assert {
        "execution_id",
        "symbol",
        "trail_distance_pct",
        "trail_engaged",
        "high_water_mark",
        "stop_order_journal_id",
        "updated_at",
    } <= cols


def test_get_state_returns_none_when_absent(db: str) -> None:
    assert get_exit_policy_state(db, execution_id=999) is None


def test_upsert_inserts_then_round_trips(db: str) -> None:
    eid = _execution(db)
    upsert_exit_policy_state(
        db,
        ExitPolicyStateRow(
            execution_id=eid,
            symbol="ACME",
            trail_distance_pct=5.0,
            trail_engaged=False,
            high_water_mark=10.02,
        ),
    )
    row = get_exit_policy_state(db, execution_id=eid)
    assert row is not None
    assert row.symbol == "ACME"
    assert row.trail_distance_pct == 5.0
    assert row.trail_engaged is False
    assert row.high_water_mark == 10.02
    assert row.stop_order_journal_id is None


def test_upsert_updates_in_place(db: str) -> None:
    """Operational state is updated in place — one row per execution
    (the audit trail lives in the append-only orders chain)."""
    eid = _execution(db)
    upsert_exit_policy_state(
        db,
        ExitPolicyStateRow(
            execution_id=eid,
            symbol="ACME",
            trail_distance_pct=5.0,
            trail_engaged=False,
            high_water_mark=10.02,
        ),
    )
    oid = _stop_order(db, eid, broker_order_id="b-stop-upd")
    upsert_exit_policy_state(
        db,
        ExitPolicyStateRow(
            execution_id=eid,
            symbol="ACME",
            trail_distance_pct=5.0,
            trail_engaged=True,
            high_water_mark=11.40,
            stop_order_journal_id=oid,
        ),
    )
    row = get_exit_policy_state(db, execution_id=eid)
    assert row is not None
    assert row.trail_engaged is True
    assert row.high_water_mark == 11.40
    assert row.stop_order_journal_id == oid

    from journal.repo import connect

    with connect(db) as conn:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM exit_policy_state"
        ).fetchone()["n"]
    assert n == 1


def _stop_order(db: str, eid: int, *, broker_order_id: str) -> int:
    from journal.models import OrderRow
    from journal.repo import insert_order

    return insert_order(
        db,
        OrderRow(
            execution_id=eid,
            role="stop",
            symbol="ACME",
            side="sell",
            order_type="stop",
            qty=25,
            tif="gtc",
            stop_price=9.0,
            broker_order_id=broker_order_id,
            submitted_at=dt.datetime(2026, 6, 8, 14, 32),
            final_status="accepted",
        ),
    )


def test_set_stop_order_journal_id_updates_existing_row(db: str) -> None:
    """§3.3 step 4: a thesis adjust_stop rotates the tracked stop order
    so the ratchet follows the replacement. FK on orders(id) is real —
    use journalled order rows, not synthetic ids."""
    eid = _execution(db)
    old_oid = _stop_order(db, eid, broker_order_id="b-stop-1")
    new_oid = _stop_order(db, eid, broker_order_id="b-stop-2")
    upsert_exit_policy_state(
        db,
        ExitPolicyStateRow(
            execution_id=eid,
            symbol="ACME",
            trail_distance_pct=5.0,
            trail_engaged=True,
            high_water_mark=11.40,
            stop_order_journal_id=old_oid,
        ),
    )
    set_stop_order_journal_id(db, execution_id=eid, order_journal_id=new_oid)
    row = get_exit_policy_state(db, execution_id=eid)
    assert row is not None
    assert row.stop_order_journal_id == new_oid


def test_set_stop_order_journal_id_noop_when_absent(db: str) -> None:
    """No trailing state (trail never engaged) → silently a no-op; the
    coordinator calls this unconditionally after a stop replacement."""
    set_stop_order_journal_id(db, execution_id=12345, order_journal_id=9)
    assert get_exit_policy_state(db, execution_id=12345) is None

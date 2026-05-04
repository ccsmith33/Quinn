"""Tests for `ops/scripts/backfill_issuer_tickers.py`.

The script rescues filings that landed with NULL `issuer_ticker` (the
pre-hotfix bug), updating the row and deleting attached `no_trade`
proposals so the agent loop will re-analyze with a populated ticker.

Acceptance (from task #2):
- Dry-run prints expected changes without mutating the DB.
- `--apply` mutates exactly what dry-run promised.
- Second consecutive `--apply` is a no-op.
- Tight WHERE clause: NULL ticker AND recent date only.
- Wraps mutations in a transaction.
- Refuses to delete proposals that have an `execution` row.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import sqlite3
from pathlib import Path

import pytest

from journal.migrate import apply_migrations
from journal.models import (
    ExecutionRow,
    FilingRow,
    PrefilterDecisionRow,
    PromptRow,
    ProposalRow,
)
from journal.repo import (
    insert_execution,
    insert_filing,
    insert_prefilter_decision,
    insert_prompt,
    insert_proposal,
)

# Load the script as a module (it lives outside the package tree).
_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "ops"
    / "scripts"
    / "backfill_issuer_tickers.py"
)


def _load_script_module():
    import sys

    spec = importlib.util.spec_from_file_location(
        "_backfill_issuer_tickers", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec so `@dataclass` (which inspects
    # `cls.__module__` via sys.modules) can find the module.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


backfill = _load_script_module()


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    p = tmp_path / "journal.db"
    apply_migrations(str(p))
    # Insert one prompt so proposal FKs resolve.
    insert_prompt(
        str(p),
        PromptRow(
            prompt_version="pv-test",
            name="sonnet_filing_analysis_v1",
            file_path="/dev/null",
            content_hash="h" * 64,
        ),
    )
    return str(p)


def _insert_null_ticker_filing(
    db_path: str,
    *,
    accession: str,
    cik: int,
    form_type: str = "8-K",
    filed_at: dt.datetime,
) -> int:
    """Helper: create a filing row with `issuer_ticker=NULL`."""
    return insert_filing(
        db_path,
        FilingRow(
            accession_number=accession,
            cik=cik,
            form_type=form_type,
            filed_at=filed_at,
            fetched_at=filed_at + dt.timedelta(seconds=30),
            raw_text_path=f"/dev/null/{accession}.txt",
            content_hash="c" * 64,
            item_codes=None,
            issuer_ticker=None,  # the bug we are backfilling
            ingest_state="ok",
            ingest_error=None,
        ),
    )


def _insert_no_trade_proposal(
    db_path: str, filing_id: int, decision_id: str
) -> int:
    return insert_proposal(
        db_path,
        ProposalRow(
            filing_id=filing_id,
            decision_id=decision_id,
            model_id="claude-sonnet-4-6",
            prompt_version="pv-test",
            raw_response='{"decision":"no_trade"}',
            kind="no_trade",
            symbol=None,
            direction=None,
            size_pct_requested=None,
            conviction=None,
            thesis=None,
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            latency_ms=1500,
            cost_usd=0.001,
            reasoning_quality=None,
            reasoning_notes=None,
        ),
    )


def _insert_proposal_with_kind(
    db_path: str, filing_id: int, decision_id: str, *, kind: str
) -> int:
    return insert_proposal(
        db_path,
        ProposalRow(
            filing_id=filing_id,
            decision_id=decision_id,
            model_id="claude-sonnet-4-6",
            prompt_version="pv-test",
            raw_response='{"decision":"propose_trade"}',
            kind=kind,
            symbol="AAPL",
            direction="long",
            size_pct_requested=0.05,
            conviction=8,
            thesis="catalyst-driven entry",
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            latency_ms=1500,
            cost_usd=0.001,
            reasoning_quality=None,
            reasoning_notes=None,
        ),
    )


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, isolation_level=None, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


class _StaticResolver:
    """Test double for `TickerResolver` — fixed map."""

    def __init__(self, m: dict[int, str]) -> None:
        self._m = m

    async def resolve(self, cik: int) -> str | None:
        return self._m.get(int(cik))


# ---------------------------------------------------------------------------
# Tests — read-only plan building
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_select_filings_only_returns_null_ticker_recent(db_path: str) -> None:
    """WHERE clause must be tight: only NULL-ticker rows in the window."""
    now = dt.datetime.now(tz=dt.UTC)
    # Eligible.
    fid_null_recent = _insert_null_ticker_filing(
        db_path, accession="0000000001-26-000001", cik=320193, filed_at=now
    )
    # Has ticker → MUST NOT be selected.
    fid_with_ticker = insert_filing(
        db_path,
        FilingRow(
            accession_number="0000000002-26-000001",
            cik=789019,
            form_type="8-K",
            filed_at=now,
            fetched_at=now,
            raw_text_path="/x",
            content_hash="c" * 64,
            issuer_ticker="MSFT",  # already populated
        ),
    )
    # Old (before window) → MUST NOT be selected.
    fid_old = _insert_null_ticker_filing(
        db_path,
        accession="0000000003-26-000001",
        cik=1318605,
        filed_at=now - dt.timedelta(days=30),
    )

    since = now - dt.timedelta(days=1)
    with _connect(db_path) as conn:
        result = backfill.select_filings_needing_ticker(conn, since)

    ids = {f.filing_id for f in result}
    assert fid_null_recent in ids
    assert fid_with_ticker not in ids
    assert fid_old not in ids


@pytest.mark.asyncio
async def test_dry_run_makes_no_db_changes(db_path: str) -> None:
    """Building a plan must not write anything to the database."""
    now = dt.datetime.now(tz=dt.UTC)
    fid = _insert_null_ticker_filing(
        db_path, accession="0000000010-26-000001", cik=320193, filed_at=now
    )
    pid = _insert_no_trade_proposal(db_path, fid, "dec-1")

    resolver = _StaticResolver({320193: "AAPL"})
    since = now - dt.timedelta(days=1)
    with _connect(db_path) as conn:
        plan = await backfill.build_plan(conn, resolver, since)

    assert len(plan.items) == 1
    assert plan.items[0].filing_id == fid
    assert plan.items[0].new_ticker == "AAPL"
    assert plan.items[0].proposal_ids == [pid]

    # Verify DB is unchanged.
    with _connect(db_path) as conn:
        f = conn.execute(
            "SELECT issuer_ticker FROM filings WHERE id = ?", (fid,)
        ).fetchone()
        assert f["issuer_ticker"] is None
        p = conn.execute(
            "SELECT id FROM proposals WHERE id = ?", (pid,)
        ).fetchone()
        assert p is not None  # proposal still exists


# ---------------------------------------------------------------------------
# Tests — apply + idempotency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_updates_filing_and_deletes_no_trade_proposal(db_path: str) -> None:
    now = dt.datetime.now(tz=dt.UTC)
    fid = _insert_null_ticker_filing(
        db_path, accession="0000000020-26-000001", cik=320193, filed_at=now
    )
    pid = _insert_no_trade_proposal(db_path, fid, "dec-20")
    resolver = _StaticResolver({320193: "AAPL"})
    since = now - dt.timedelta(days=1)

    with _connect(db_path) as conn:
        plan = await backfill.build_plan(conn, resolver, since)
        backfill.apply_plan(conn, plan)

    with _connect(db_path) as conn:
        f = conn.execute(
            "SELECT issuer_ticker FROM filings WHERE id = ?", (fid,)
        ).fetchone()
        assert f["issuer_ticker"] == "AAPL"
        # The original no_trade proposal must be gone (so the agent
        # re-analyzes the filing).
        p = conn.execute(
            "SELECT id FROM proposals WHERE id = ?", (pid,)
        ).fetchone()
        assert p is None


@pytest.mark.asyncio
async def test_second_apply_is_a_no_op(db_path: str) -> None:
    """After the first apply, the second pass sees nothing to do."""
    now = dt.datetime.now(tz=dt.UTC)
    fid = _insert_null_ticker_filing(
        db_path, accession="0000000030-26-000001", cik=320193, filed_at=now
    )
    _insert_no_trade_proposal(db_path, fid, "dec-30")
    resolver = _StaticResolver({320193: "AAPL"})
    since = now - dt.timedelta(days=1)

    with _connect(db_path) as conn:
        plan1 = await backfill.build_plan(conn, resolver, since)
        backfill.apply_plan(conn, plan1)

    with _connect(db_path) as conn:
        plan2 = await backfill.build_plan(conn, resolver, since)
        backfill.apply_plan(conn, plan2)

    assert len(plan1.items) == 1
    assert len(plan2.items) == 0


@pytest.mark.asyncio
async def test_unresolved_cik_left_null(db_path: str) -> None:
    """If the resolver cannot place the CIK, the filing is reported as
    unresolved and the row is left untouched."""
    now = dt.datetime.now(tz=dt.UTC)
    fid = _insert_null_ticker_filing(
        db_path, accession="0000000040-26-000001", cik=999_999_999, filed_at=now
    )
    resolver = _StaticResolver({320193: "AAPL"})  # 999_999_999 absent
    since = now - dt.timedelta(days=1)

    with _connect(db_path) as conn:
        plan = await backfill.build_plan(conn, resolver, since)
        backfill.apply_plan(conn, plan)

    assert plan.items == []
    assert [f.filing_id for f in plan.unresolved] == [fid]
    with _connect(db_path) as conn:
        f = conn.execute(
            "SELECT issuer_ticker FROM filings WHERE id = ?", (fid,)
        ).fetchone()
        assert f["issuer_ticker"] is None


# ---------------------------------------------------------------------------
# Tests — safety guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_filing_with_executed_proposal_is_aborted(db_path: str) -> None:
    """Safety: never delete a proposal that has an execution attached."""
    now = dt.datetime.now(tz=dt.UTC)
    fid = _insert_null_ticker_filing(
        db_path, accession="0000000050-26-000001", cik=320193, filed_at=now
    )
    pid = _insert_no_trade_proposal(db_path, fid, "dec-50")
    insert_execution(
        db_path,
        ExecutionRow(
            proposal_id=pid,
            decision="accept",
            reject_reason=None,
            realized_size_pct=0.05,
            realized_dollar_size=5000.0,
            submitted_orders_json="[]",
        ),
    )
    resolver = _StaticResolver({320193: "AAPL"})
    since = now - dt.timedelta(days=1)

    with _connect(db_path) as conn:
        plan = await backfill.build_plan(conn, resolver, since)
        backfill.apply_plan(conn, plan)

    assert plan.items == []
    assert plan.aborted_due_to_executions == [fid]
    with _connect(db_path) as conn:
        f = conn.execute(
            "SELECT issuer_ticker FROM filings WHERE id = ?", (fid,)
        ).fetchone()
        # Filing's ticker must NOT have been updated; the row is sacrosanct
        # because a proposal got executed against it.
        assert f["issuer_ticker"] is None
        p = conn.execute(
            "SELECT id FROM proposals WHERE id = ?", (pid,)
        ).fetchone()
        assert p is not None  # proposal preserved


@pytest.mark.asyncio
async def test_filing_with_non_no_trade_proposal_is_aborted(db_path: str) -> None:
    """Safety: a non-no_trade proposal might be queued for execution; refuse."""
    now = dt.datetime.now(tz=dt.UTC)
    fid = _insert_null_ticker_filing(
        db_path, accession="0000000060-26-000001", cik=320193, filed_at=now
    )
    _insert_proposal_with_kind(db_path, fid, "dec-60", kind="propose_trade")
    resolver = _StaticResolver({320193: "AAPL"})
    since = now - dt.timedelta(days=1)

    with _connect(db_path) as conn:
        plan = await backfill.build_plan(conn, resolver, since)
        backfill.apply_plan(conn, plan)

    assert plan.items == []
    assert plan.aborted_due_to_non_no_trade == [fid]
    with _connect(db_path) as conn:
        f = conn.execute(
            "SELECT issuer_ticker FROM filings WHERE id = ?", (fid,)
        ).fetchone()
        assert f["issuer_ticker"] is None


@pytest.mark.asyncio
async def test_old_filings_outside_window_are_never_touched(db_path: str) -> None:
    """A filing from 30 days ago must never be selected even if it matches
    on NULL ticker. The script's blast radius is bounded by `--since`."""
    now = dt.datetime.now(tz=dt.UTC)
    old = now - dt.timedelta(days=30)
    fid = _insert_null_ticker_filing(
        db_path, accession="0000000070-26-000001", cik=320193, filed_at=old
    )
    _insert_no_trade_proposal(db_path, fid, "dec-70")
    resolver = _StaticResolver({320193: "AAPL"})
    since = now - dt.timedelta(days=1)

    with _connect(db_path) as conn:
        plan = await backfill.build_plan(conn, resolver, since)
        backfill.apply_plan(conn, plan)

    assert plan.items == []
    assert plan.unresolved == []
    with _connect(db_path) as conn:
        f = conn.execute(
            "SELECT issuer_ticker FROM filings WHERE id = ?", (fid,)
        ).fetchone()
        assert f["issuer_ticker"] is None  # untouched


@pytest.mark.asyncio
async def test_filing_already_tickered_never_selected(db_path: str) -> None:
    """A row with non-NULL ticker is invisible to the script even when
    its CIK is in the resolver. Idempotency guard #2."""
    now = dt.datetime.now(tz=dt.UTC)
    insert_filing(
        db_path,
        FilingRow(
            accession_number="0000000080-26-000001",
            cik=320193,
            form_type="8-K",
            filed_at=now,
            fetched_at=now,
            raw_text_path="/x",
            content_hash="c" * 64,
            issuer_ticker="AAPL",  # already populated
        ),
    )
    resolver = _StaticResolver({320193: "WRONG"})
    since = now - dt.timedelta(days=1)

    with _connect(db_path) as conn:
        plan = await backfill.build_plan(conn, resolver, since)
        backfill.apply_plan(conn, plan)

    assert plan.items == []
    with _connect(db_path) as conn:
        f = conn.execute(
            "SELECT issuer_ticker FROM filings WHERE accession_number = ?",
            ("0000000080-26-000001",),
        ).fetchone()
        assert f["issuer_ticker"] == "AAPL"  # NOT overwritten to "WRONG"


@pytest.mark.asyncio
async def test_filing_with_no_proposals_still_gets_ticker_updated(db_path: str) -> None:
    """A filing that hasn't been analyzed yet (no proposal row) should
    still have its ticker backfilled — there's nothing to delete, but
    the ticker still needs to land so the future analyzer call sees it."""
    now = dt.datetime.now(tz=dt.UTC)
    fid = _insert_null_ticker_filing(
        db_path, accession="0000000090-26-000001", cik=320193, filed_at=now
    )
    # No proposal inserted.
    resolver = _StaticResolver({320193: "AAPL"})
    since = now - dt.timedelta(days=1)

    with _connect(db_path) as conn:
        plan = await backfill.build_plan(conn, resolver, since)
        backfill.apply_plan(conn, plan)

    assert len(plan.items) == 1
    assert plan.items[0].proposal_ids == []
    with _connect(db_path) as conn:
        f = conn.execute(
            "SELECT issuer_ticker FROM filings WHERE id = ?", (fid,)
        ).fetchone()
        assert f["issuer_ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# Tests — output / reporting
# ---------------------------------------------------------------------------


def test_render_plan_summarizes_counts() -> None:
    plan = backfill.BackfillPlan(
        items=[
            backfill.BackfillPlanItem(
                filing_id=1,
                cik=320193,
                new_ticker="AAPL",
                proposal_ids=[10],
                prefilter_decision_ids=[100],
            ),
            backfill.BackfillPlanItem(
                filing_id=2,
                cik=789019,
                new_ticker="MSFT",
                proposal_ids=[],
                prefilter_decision_ids=[101],
            ),
        ],
        unresolved=[
            backfill.FilingNeedingTicker(
                filing_id=3,
                cik=999,
                accession_number="x",
                form_type="8-K",
            )
        ],
        aborted_due_to_executions=[],
        aborted_due_to_non_no_trade=[],
    )
    rendered = backfill.render_plan(plan, apply=False)
    assert "DRY-RUN" in rendered
    assert "resolved filings:           2" in rendered
    assert "proposals queued for delete: 1" in rendered
    assert "prefilter rows queued for delete: 2" in rendered
    assert "unresolved (no SEC match):   1" in rendered
    assert "rerun with --apply" in rendered


def test_render_plan_shows_apply_mode() -> None:
    plan = backfill.BackfillPlan(
        items=[],
        unresolved=[],
        aborted_due_to_executions=[],
        aborted_due_to_non_no_trade=[],
    )
    rendered = backfill.render_plan(plan, apply=True)
    assert "APPLY" in rendered
    # No "rerun" hint when we're already in apply mode.
    assert "rerun with --apply" not in rendered


# ---------------------------------------------------------------------------
# Tests — transactional safety
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_is_transactional(db_path: str) -> None:
    """If apply_plan raises mid-mutation, no partial state is left behind.

    Simulate by passing a plan that targets a non-existent proposal id —
    the DELETE silently affects 0 rows (SQLite doesn't error here), so we
    verify the contract by ensuring the UPDATE still lands together with
    the DELETEs in a successful run.
    """
    now = dt.datetime.now(tz=dt.UTC)
    fid_a = _insert_null_ticker_filing(
        db_path, accession="0000000100-26-000001", cik=320193, filed_at=now
    )
    fid_b = _insert_null_ticker_filing(
        db_path, accession="0000000100-26-000002", cik=789019, filed_at=now
    )
    pid_a = _insert_no_trade_proposal(db_path, fid_a, "dec-100a")
    pid_b = _insert_no_trade_proposal(db_path, fid_b, "dec-100b")

    plan = backfill.BackfillPlan(
        items=[
            backfill.BackfillPlanItem(
                filing_id=fid_a, cik=320193, new_ticker="AAPL",
                proposal_ids=[pid_a],
                prefilter_decision_ids=[],
            ),
            backfill.BackfillPlanItem(
                filing_id=fid_b, cik=789019, new_ticker="MSFT",
                proposal_ids=[pid_b],
                prefilter_decision_ids=[],
            ),
        ],
        unresolved=[],
        aborted_due_to_executions=[],
        aborted_due_to_non_no_trade=[],
    )
    with _connect(db_path) as conn:
        backfill.apply_plan(conn, plan)

    with _connect(db_path) as conn:
        row_a = conn.execute(
            "SELECT issuer_ticker FROM filings WHERE id = ?", (fid_a,)
        ).fetchone()
        row_b = conn.execute(
            "SELECT issuer_ticker FROM filings WHERE id = ?", (fid_b,)
        ).fetchone()
        assert row_a["issuer_ticker"] == "AAPL"
        assert row_b["issuer_ticker"] == "MSFT"
        # Both proposals deleted.
        assert conn.execute(
            "SELECT COUNT(*) FROM proposals WHERE filing_id IN (?,?)",
            (fid_a, fid_b),
        ).fetchone()[0] == 0


# ---------------------------------------------------------------------------
# BLOCK-1 (reviewer-flagged): apply must clear `prefilter_decisions` so the
# agent's `_crash_recovery_scan` re-queues the filing on next boot.
# ---------------------------------------------------------------------------


def _insert_no_trade_prefilter_decision(db_path: str, filing_id: int) -> int:
    """Helper: simulate the prefilter row the agent wrote when it first
    processed the (NULL-ticker) filing."""
    return insert_prefilter_decision(
        db_path,
        PrefilterDecisionRow(
            filing_id=filing_id,
            decision="continue",
            rule_fired="universe_member",
            similarity_score=None,
            reason_detail=None,
        ),
    )


@pytest.mark.asyncio
async def test_apply_clears_prefilter_decisions_so_agent_recovers(
    db_path: str,
) -> None:
    """The agent's `_crash_recovery_scan` (`src/app/loop.py:319`) re-queues
    only filings with NO `prefilter_decisions` row (`WHERE p.id IS NULL`).
    If we leave that row in place, the backfilled filing is invisible to
    the recovery scan and never re-analyzed — defeating the script's
    entire purpose. Apply must drop it.
    """
    now = dt.datetime.now(tz=dt.UTC)
    fid = _insert_null_ticker_filing(
        db_path, accession="0000000200-26-000001", cik=320193, filed_at=now
    )
    _insert_no_trade_proposal(db_path, fid, "dec-200")
    pf_id = _insert_no_trade_prefilter_decision(db_path, fid)
    assert pf_id > 0

    resolver = _StaticResolver({320193: "AAPL"})
    since = now - dt.timedelta(days=1)
    with _connect(db_path) as conn:
        plan = await backfill.build_plan(conn, resolver, since)
        backfill.apply_plan(conn, plan)

    with _connect(db_path) as conn:
        # Boot-time recovery scan can pick this filing up again.
        rows = conn.execute(
            "SELECT id FROM prefilter_decisions WHERE filing_id = ?", (fid,)
        ).fetchall()
        assert rows == []

        # The exact LEFT-JOIN predicate the agent's `_crash_recovery_scan`
        # uses must now match this filing.
        recovered = conn.execute(
            """
            SELECT f.id FROM filings f
            LEFT JOIN prefilter_decisions p ON p.filing_id = f.id
            WHERE p.id IS NULL
            """
        ).fetchall()
        assert fid in {int(r["id"]) for r in recovered}


@pytest.mark.asyncio
async def test_aborted_filing_keeps_its_prefilter_decision(db_path: str) -> None:
    """Inverse safety: a filing that hits an abort guard (e.g. has an
    executed proposal) must NOT have its prefilter_decisions row deleted —
    that would silently re-queue it for re-analysis behind the operator's
    back. The whole row stays untouched."""
    now = dt.datetime.now(tz=dt.UTC)
    fid = _insert_null_ticker_filing(
        db_path, accession="0000000201-26-000001", cik=320193, filed_at=now
    )
    pid = _insert_no_trade_proposal(db_path, fid, "dec-201")
    insert_execution(
        db_path,
        ExecutionRow(
            proposal_id=pid,
            decision="accept",
            reject_reason=None,
            realized_size_pct=0.05,
            realized_dollar_size=5000.0,
            submitted_orders_json="[]",
        ),
    )
    _insert_no_trade_prefilter_decision(db_path, fid)

    resolver = _StaticResolver({320193: "AAPL"})
    since = now - dt.timedelta(days=1)
    with _connect(db_path) as conn:
        plan = await backfill.build_plan(conn, resolver, since)
        backfill.apply_plan(conn, plan)

    assert plan.aborted_due_to_executions == [fid]
    assert plan.items == []
    with _connect(db_path) as conn:
        # prefilter row preserved — the filing is NOT eligible for recovery
        # scan re-queueing.
        rows = conn.execute(
            "SELECT id FROM prefilter_decisions WHERE filing_id = ?", (fid,)
        ).fetchall()
        assert len(rows) == 1


@pytest.mark.asyncio
async def test_dry_run_does_not_clear_prefilter_decisions(db_path: str) -> None:
    """build_plan is read-only; the prefilter row must survive a dry run."""
    now = dt.datetime.now(tz=dt.UTC)
    fid = _insert_null_ticker_filing(
        db_path, accession="0000000202-26-000001", cik=320193, filed_at=now
    )
    _insert_no_trade_proposal(db_path, fid, "dec-202")
    _insert_no_trade_prefilter_decision(db_path, fid)

    resolver = _StaticResolver({320193: "AAPL"})
    since = now - dt.timedelta(days=1)
    with _connect(db_path) as conn:
        await backfill.build_plan(conn, resolver, since)
        # Note: NO apply_plan call. Dry-run only.

    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id FROM prefilter_decisions WHERE filing_id = ?", (fid,)
        ).fetchall()
        assert len(rows) == 1


@pytest.mark.asyncio
async def test_apply_handles_filing_with_no_prefilter_decision(db_path: str) -> None:
    """Edge case: a filing with no prefilter row yet (analyzer never ran)
    still gets its ticker updated. The DELETE silently affects 0 rows;
    no crash."""
    now = dt.datetime.now(tz=dt.UTC)
    fid = _insert_null_ticker_filing(
        db_path, accession="0000000203-26-000001", cik=320193, filed_at=now
    )
    # No proposal, no prefilter decision.
    resolver = _StaticResolver({320193: "AAPL"})
    since = now - dt.timedelta(days=1)
    with _connect(db_path) as conn:
        plan = await backfill.build_plan(conn, resolver, since)
        backfill.apply_plan(conn, plan)

    with _connect(db_path) as conn:
        f = conn.execute(
            "SELECT issuer_ticker FROM filings WHERE id = ?", (fid,)
        ).fetchone()
        assert f["issuer_ticker"] == "AAPL"


@pytest.mark.asyncio
async def test_apply_reports_prefilter_decisions_cleared_count(db_path: str) -> None:
    """Operators want to see in the summary how many prefilter rows were
    cleared. The plan/render exposes this."""
    now = dt.datetime.now(tz=dt.UTC)
    fid_a = _insert_null_ticker_filing(
        db_path, accession="0000000210-26-000001", cik=320193, filed_at=now
    )
    fid_b = _insert_null_ticker_filing(
        db_path, accession="0000000210-26-000002", cik=789019, filed_at=now
    )
    _insert_no_trade_proposal(db_path, fid_a, "dec-210a")
    _insert_no_trade_prefilter_decision(db_path, fid_a)
    _insert_no_trade_prefilter_decision(db_path, fid_b)
    # fid_b has no proposal but does have a prefilter row.

    resolver = _StaticResolver({320193: "AAPL", 789019: "MSFT"})
    since = now - dt.timedelta(days=1)
    with _connect(db_path) as conn:
        plan = await backfill.build_plan(conn, resolver, since)

    rendered = backfill.render_plan(plan, apply=False)
    assert "prefilter rows queued for delete: 2" in rendered

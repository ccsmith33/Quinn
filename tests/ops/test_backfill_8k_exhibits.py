"""Tests for `ops/scripts/backfill_8k_exhibits.py`.

The script appends EX-99.x exhibit content to 8-Ks ingested before the
ADR-008 fix, then deletes their attached `no_trade` proposals + prefilter
rows so the agent's `_crash_recovery_scan` re-queues them.

Acceptance (from task #2):
- WHERE clause scope: only 8-K furnish-item filings within the window.
- Aborted filing (executions row present) keeps prefilter and proposal.
- Dry-run is read-only (no DB / file mutations).
- `--apply` mutates and is idempotent (second run is no-op).
- post-`--apply` `prefilter_decisions` for affected filing_ids is empty.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
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

# Load the script as a module.
_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "ops" / "scripts" / "backfill_8k_exhibits.py"
)


def _load_script_module():
    import sys

    spec = importlib.util.spec_from_file_location(
        "_backfill_8k_exhibits", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
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


def _insert_8k_filing(
    db_path: str,
    *,
    raw_root: Path,
    accession: str,
    cik: int,
    filed_at: dt.datetime,
    item_codes: list[str] | None,
    body_text: str = "Item 2.02 Results of Operations.",
) -> tuple[int, Path]:
    """Insert an 8-K filing row + write its `raw_text_path` body to disk.

    Returns `(filing_id, raw_text_path)`.
    """
    raw_root.mkdir(parents=True, exist_ok=True)
    raw_path = raw_root / f"{accession}.txt"
    raw_path.write_text(body_text)
    fid = insert_filing(
        db_path,
        FilingRow(
            accession_number=accession,
            cik=cik,
            form_type="8-K",
            filed_at=filed_at,
            fetched_at=filed_at + dt.timedelta(seconds=30),
            raw_text_path=str(raw_path),
            content_hash="c" * 64,
            item_codes=json.dumps(item_codes) if item_codes is not None else None,
            issuer_ticker="TST",
            ingest_state="ok",
            ingest_error=None,
        ),
    )
    return fid, raw_path


def _insert_no_trade_proposal(db_path: str, filing_id: int, decision_id: str) -> int:
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
            symbol="TST",
            direction="long",
            size_pct_requested=0.05,
            conviction=8,
            thesis="catalyst",
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


def _insert_prefilter(db_path: str, filing_id: int) -> int:
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


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, isolation_level=None, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


class _StaticExhibitFetcher:
    """Test double for `_EdgarExhibitFetcher` — fixed map keyed on
    (cik, accession_number).

    Each entry returns `(index_items, name_to_body_bytes)` exactly as
    the production protocol specifies.
    """

    def __init__(
        self,
        m: dict[tuple[int, str], tuple[list[dict[str, str]], dict[str, bytes]]],
    ) -> None:
        self._m = m
        self.calls: list[tuple[int, str]] = []

    async def fetch(
        self, cik: int, accession_number: str
    ) -> tuple[list[dict[str, str]], dict[str, bytes]]:
        self.calls.append((cik, accession_number))
        return self._m.get((cik, accession_number), ([], {}))


def _index_with_ex99(filename: str = "ex-99-1.htm", size: int = 1000) -> list[dict[str, str]]:
    return [
        {"name": "primary.htm", "type": "text.gif", "size": "5000"},
        {"name": filename, "type": "text.gif", "size": str(size)},
    ]


def _ex99_body(marker: str = "EARNINGS RELEASE") -> bytes:
    return (
        f"<html><body><p>{marker}</p>"
        f"<p>Revenue: $1.234B; EPS: $0.42.</p></body></html>"
    ).encode()


# ---------------------------------------------------------------------------
# Tests — read-only plan building
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_select_filings_only_returns_8k_furnish_items_recent(
    db_path: str, tmp_path: Path
) -> None:
    """WHERE clause: only 8-K + furnish item code + within window."""
    now = dt.datetime.now(tz=dt.UTC)
    raw_root = tmp_path / "raw"

    # Eligible — 8-K, item 2.02, recent.
    fid_eligible, _ = _insert_8k_filing(
        db_path,
        raw_root=raw_root,
        accession="0000000001-26-000001",
        cik=320193,
        filed_at=now,
        item_codes=["2.02"],
    )
    # Wrong form type — 10-Q with item codes (synthetic).
    insert_filing(
        db_path,
        FilingRow(
            accession_number="0000000002-26-000001",
            cik=789019,
            form_type="10-Q",
            filed_at=now,
            fetched_at=now,
            raw_text_path=str(raw_root / "0000000002-26-000001.txt"),
            content_hash="c" * 64,
            item_codes=json.dumps(["2.02"]),
            issuer_ticker="MSFT",
        ),
    )
    # 8-K with non-furnish item — must NOT be selected.
    _insert_8k_filing(
        db_path,
        raw_root=raw_root,
        accession="0000000003-26-000001",
        cik=11111,
        filed_at=now,
        item_codes=["5.02"],
    )
    # Old (before window).
    _insert_8k_filing(
        db_path,
        raw_root=raw_root,
        accession="0000000004-26-000001",
        cik=22222,
        filed_at=now - dt.timedelta(days=30),
        item_codes=["2.02"],
    )

    since = now - dt.timedelta(days=2)
    with _connect(db_path) as conn:
        result = backfill.select_filings_needing_exhibit(conn, since)
    ids = {f.filing_id for f in result}
    assert fid_eligible in ids
    assert len(ids) == 1


@pytest.mark.asyncio
async def test_select_filings_handles_each_furnish_item(
    db_path: str, tmp_path: Path
) -> None:
    """Items 2.02, 7.01, 8.01 each individually qualify."""
    now = dt.datetime.now(tz=dt.UTC)
    raw_root = tmp_path / "raw"
    f202, _ = _insert_8k_filing(
        db_path, raw_root=raw_root, accession="0000202-26-000001",
        cik=10001, filed_at=now, item_codes=["2.02"],
    )
    f701, _ = _insert_8k_filing(
        db_path, raw_root=raw_root, accession="0000701-26-000001",
        cik=10002, filed_at=now, item_codes=["7.01"],
    )
    f801, _ = _insert_8k_filing(
        db_path, raw_root=raw_root, accession="0000801-26-000001",
        cik=10003, filed_at=now, item_codes=["8.01"],
    )

    since = now - dt.timedelta(days=2)
    with _connect(db_path) as conn:
        result = backfill.select_filings_needing_exhibit(conn, since)
    assert {f202, f701, f801} == {f.filing_id for f in result}


@pytest.mark.asyncio
async def test_dry_run_makes_no_db_or_file_changes(
    db_path: str, tmp_path: Path
) -> None:
    """build_plan must not write to DB or to the raw_text file."""
    now = dt.datetime.now(tz=dt.UTC)
    raw_root = tmp_path / "raw"
    fid, raw_path = _insert_8k_filing(
        db_path,
        raw_root=raw_root,
        accession="0000010-26-000001",
        cik=320193,
        filed_at=now,
        item_codes=["2.02"],
    )
    pid = _insert_no_trade_proposal(db_path, fid, "dec-10")
    pf = _insert_prefilter(db_path, fid)
    body_before = raw_path.read_text()

    fetcher = _StaticExhibitFetcher(
        {(320193, "0000010-26-000001"): (
            _index_with_ex99(),
            {"ex-99-1.htm": _ex99_body("DRY_RUN_MARKER")},
        )},
    )
    since = now - dt.timedelta(days=2)
    with _connect(db_path) as conn:
        plan = await backfill.build_plan(conn, fetcher, since)

    assert len(plan.items) == 1
    item = plan.items[0]
    assert item.filing_id == fid
    assert item.proposal_ids == [pid]
    assert item.prefilter_decision_ids == [pf]

    # Verify no mutations.
    assert raw_path.read_text() == body_before
    with _connect(db_path) as conn:
        f = conn.execute(
            "SELECT content_hash FROM filings WHERE id = ?", (fid,)
        ).fetchone()
        assert f["content_hash"] == "c" * 64  # untouched
        assert (
            conn.execute(
                "SELECT id FROM proposals WHERE id = ?", (pid,)
            ).fetchone()
            is not None
        )
        assert (
            conn.execute(
                "SELECT id FROM prefilter_decisions WHERE id = ?", (pf,)
            ).fetchone()
            is not None
        )


# ---------------------------------------------------------------------------
# Tests — apply + idempotency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_appends_exhibit_and_clears_proposal_and_prefilter(
    db_path: str, tmp_path: Path
) -> None:
    """The happy path. Apply must:
    - Append exhibit text to raw_text_path
    - Update content_hash
    - DELETE the no_trade proposal
    - DELETE the prefilter_decision
    """
    now = dt.datetime.now(tz=dt.UTC)
    raw_root = tmp_path / "raw"
    fid, raw_path = _insert_8k_filing(
        db_path,
        raw_root=raw_root,
        accession="0000020-26-000001",
        cik=320193,
        filed_at=now,
        item_codes=["2.02"],
        body_text="Item 2.02 Results of Operations.",
    )
    pid = _insert_no_trade_proposal(db_path, fid, "dec-20")
    pf = _insert_prefilter(db_path, fid)

    fetcher = _StaticExhibitFetcher(
        {(320193, "0000020-26-000001"): (
            _index_with_ex99(),
            {"ex-99-1.htm": _ex99_body("EARNINGS_BODY_MARKER")},
        )},
    )
    since = now - dt.timedelta(days=2)
    with _connect(db_path) as conn:
        plan = await backfill.build_plan(conn, fetcher, since)
        backfill.apply_plan(conn, plan)

    augmented = raw_path.read_text()
    assert "Item 2.02" in augmented
    assert "EARNINGS_BODY_MARKER" in augmented
    assert "--- EXHIBIT 99.1 ---" in augmented

    with _connect(db_path) as conn:
        f = conn.execute(
            "SELECT content_hash FROM filings WHERE id = ?", (fid,)
        ).fetchone()
        assert f["content_hash"] != "c" * 64  # updated
        assert (
            conn.execute(
                "SELECT id FROM proposals WHERE id = ?", (pid,)
            ).fetchone()
            is None
        )
        assert (
            conn.execute(
                "SELECT id FROM prefilter_decisions WHERE id = ?", (pf,)
            ).fetchone()
            is None
        )
        # Recovery-scan predicate matches now.
        rec = conn.execute(
            """
            SELECT f.id FROM filings f
            LEFT JOIN prefilter_decisions p ON p.filing_id = f.id
            WHERE p.id IS NULL
            """
        ).fetchall()
        assert fid in {int(r["id"]) for r in rec}


@pytest.mark.asyncio
async def test_second_apply_is_a_no_op(db_path: str, tmp_path: Path) -> None:
    """Idempotency: after the first apply augments raw_text, the second
    pass sees the EXHIBIT 99 separator and skips."""
    now = dt.datetime.now(tz=dt.UTC)
    raw_root = tmp_path / "raw"
    fid, raw_path = _insert_8k_filing(
        db_path,
        raw_root=raw_root,
        accession="0000030-26-000001",
        cik=320193,
        filed_at=now,
        item_codes=["2.02"],
    )
    _insert_no_trade_proposal(db_path, fid, "dec-30")
    _insert_prefilter(db_path, fid)

    fetcher = _StaticExhibitFetcher(
        {(320193, "0000030-26-000001"): (
            _index_with_ex99(),
            {"ex-99-1.htm": _ex99_body()},
        )},
    )
    since = now - dt.timedelta(days=2)
    with _connect(db_path) as conn:
        plan1 = await backfill.build_plan(conn, fetcher, since)
        backfill.apply_plan(conn, plan1)

    text_after_first = raw_path.read_text()

    with _connect(db_path) as conn:
        plan2 = await backfill.build_plan(conn, fetcher, since)
        backfill.apply_plan(conn, plan2)

    assert len(plan1.items) == 1
    assert len(plan2.items) == 0
    # File is unchanged across the two applies.
    assert raw_path.read_text() == text_after_first
    assert fid in plan2.skipped_already_augmented


@pytest.mark.asyncio
async def test_no_exhibit_in_index_skips_filing(
    db_path: str, tmp_path: Path
) -> None:
    """Filing has a furnish item but no EX-99 in the index → skipped, not
    treated as a failure. Filing remains as-is (body-only)."""
    now = dt.datetime.now(tz=dt.UTC)
    raw_root = tmp_path / "raw"
    fid, raw_path = _insert_8k_filing(
        db_path,
        raw_root=raw_root,
        accession="0000040-26-000001",
        cik=320193,
        filed_at=now,
        item_codes=["2.02"],
    )
    _insert_no_trade_proposal(db_path, fid, "dec-40")
    pf = _insert_prefilter(db_path, fid)
    body_before = raw_path.read_text()

    fetcher = _StaticExhibitFetcher(
        {(320193, "0000040-26-000001"): (
            # Index has primary but no EX-99.
            [{"name": "primary.htm", "type": "text.gif", "size": "5000"}],
            {},
        )},
    )
    since = now - dt.timedelta(days=2)
    with _connect(db_path) as conn:
        plan = await backfill.build_plan(conn, fetcher, since)
        backfill.apply_plan(conn, plan)

    assert plan.items == []
    assert plan.skipped_no_exhibit_in_index == [fid]
    # Filing unchanged.
    assert raw_path.read_text() == body_before
    with _connect(db_path) as conn:
        # Prefilter row preserved → recovery scan does NOT pick it up.
        assert conn.execute(
            "SELECT id FROM prefilter_decisions WHERE id = ?", (pf,)
        ).fetchone() is not None


# ---------------------------------------------------------------------------
# Tests — safety guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_filing_with_executed_proposal_is_aborted(
    db_path: str, tmp_path: Path
) -> None:
    """Never delete a proposal that has an execution attached."""
    now = dt.datetime.now(tz=dt.UTC)
    raw_root = tmp_path / "raw"
    fid, raw_path = _insert_8k_filing(
        db_path,
        raw_root=raw_root,
        accession="0000050-26-000001",
        cik=320193,
        filed_at=now,
        item_codes=["2.02"],
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
    pf = _insert_prefilter(db_path, fid)
    body_before = raw_path.read_text()

    fetcher = _StaticExhibitFetcher(
        {(320193, "0000050-26-000001"): (
            _index_with_ex99(),
            {"ex-99-1.htm": _ex99_body()},
        )},
    )
    since = now - dt.timedelta(days=2)
    with _connect(db_path) as conn:
        plan = await backfill.build_plan(conn, fetcher, since)
        backfill.apply_plan(conn, plan)

    assert plan.items == []
    assert plan.aborted_due_to_executions == [fid]
    # Raw text untouched.
    assert raw_path.read_text() == body_before
    with _connect(db_path) as conn:
        assert (
            conn.execute(
                "SELECT id FROM proposals WHERE id = ?", (pid,)
            ).fetchone()
            is not None
        )
        # Prefilter row preserved.
        assert (
            conn.execute(
                "SELECT id FROM prefilter_decisions WHERE id = ?", (pf,)
            ).fetchone()
            is not None
        )


@pytest.mark.asyncio
async def test_filing_with_non_no_trade_proposal_is_aborted(
    db_path: str, tmp_path: Path
) -> None:
    """A non-no_trade proposal might be queued for execution; refuse."""
    now = dt.datetime.now(tz=dt.UTC)
    raw_root = tmp_path / "raw"
    fid, raw_path = _insert_8k_filing(
        db_path,
        raw_root=raw_root,
        accession="0000060-26-000001",
        cik=320193,
        filed_at=now,
        item_codes=["2.02"],
    )
    _insert_proposal_with_kind(db_path, fid, "dec-60", kind="propose_trade")
    body_before = raw_path.read_text()

    fetcher = _StaticExhibitFetcher(
        {(320193, "0000060-26-000001"): (
            _index_with_ex99(),
            {"ex-99-1.htm": _ex99_body()},
        )},
    )
    since = now - dt.timedelta(days=2)
    with _connect(db_path) as conn:
        plan = await backfill.build_plan(conn, fetcher, since)
        backfill.apply_plan(conn, plan)

    assert plan.items == []
    assert plan.aborted_due_to_non_no_trade == [fid]
    assert raw_path.read_text() == body_before


@pytest.mark.asyncio
async def test_old_filings_outside_window_are_never_touched(
    db_path: str, tmp_path: Path
) -> None:
    """Bounded blast radius: a 30-day-old filing is never selected."""
    now = dt.datetime.now(tz=dt.UTC)
    old = now - dt.timedelta(days=30)
    raw_root = tmp_path / "raw"
    fid, raw_path = _insert_8k_filing(
        db_path,
        raw_root=raw_root,
        accession="0000070-26-000001",
        cik=320193,
        filed_at=old,
        item_codes=["2.02"],
    )
    _insert_no_trade_proposal(db_path, fid, "dec-70")
    body_before = raw_path.read_text()

    fetcher = _StaticExhibitFetcher(
        {(320193, "0000070-26-000001"): (
            _index_with_ex99(),
            {"ex-99-1.htm": _ex99_body()},
        )},
    )
    since = now - dt.timedelta(days=2)
    with _connect(db_path) as conn:
        plan = await backfill.build_plan(conn, fetcher, since)
        backfill.apply_plan(conn, plan)

    assert plan.items == []
    assert raw_path.read_text() == body_before


@pytest.mark.asyncio
async def test_non_furnish_item_filing_never_selected(
    db_path: str, tmp_path: Path
) -> None:
    """An 8-K with only Item 5.02 must be invisible to this script even
    if it's recent and has an EX-99."""
    now = dt.datetime.now(tz=dt.UTC)
    raw_root = tmp_path / "raw"
    _insert_8k_filing(
        db_path,
        raw_root=raw_root,
        accession="0000080-26-000001",
        cik=320193,
        filed_at=now,
        item_codes=["5.02"],
    )
    fetcher = _StaticExhibitFetcher({})
    since = now - dt.timedelta(days=2)
    with _connect(db_path) as conn:
        plan = await backfill.build_plan(conn, fetcher, since)
    assert plan.items == []
    assert plan.skipped_no_exhibit_in_index == []


@pytest.mark.asyncio
async def test_filing_already_augmented_is_skipped(
    db_path: str, tmp_path: Path
) -> None:
    """A raw_text file that already has the EXHIBIT 99 separator is
    skipped via the idempotency check, even on the first run."""
    now = dt.datetime.now(tz=dt.UTC)
    raw_root = tmp_path / "raw"
    fid, raw_path = _insert_8k_filing(
        db_path,
        raw_root=raw_root,
        accession="0000090-26-000001",
        cik=320193,
        filed_at=now,
        item_codes=["2.02"],
        body_text="Item 2.02 Results.\n\n--- EXHIBIT 99.1 ---\n\nAlready augmented.",
    )
    fetcher = _StaticExhibitFetcher(
        {(320193, "0000090-26-000001"): (
            _index_with_ex99(),
            {"ex-99-1.htm": _ex99_body()},
        )},
    )
    since = now - dt.timedelta(days=2)
    with _connect(db_path) as conn:
        plan = await backfill.build_plan(conn, fetcher, since)
    assert plan.items == []
    assert plan.skipped_already_augmented == [fid]


@pytest.mark.asyncio
async def test_filing_with_only_oversized_exhibit_yields_no_augmentation(
    db_path: str, tmp_path: Path
) -> None:
    """A single 6 MB exhibit (over the 5 MB cumulative cap) → recorded as
    a fetch failure; raw_text untouched. The filing is left as body-only."""
    now = dt.datetime.now(tz=dt.UTC)
    raw_root = tmp_path / "raw"
    fid, raw_path = _insert_8k_filing(
        db_path,
        raw_root=raw_root,
        accession="0000100-26-000001",
        cik=320193,
        filed_at=now,
        item_codes=["2.02"],
    )
    _insert_no_trade_proposal(db_path, fid, "dec-100")
    body_before = raw_path.read_text()

    oversized = b"<html><body><p>" + b"X" * (6 * 1024 * 1024) + b"</p></body></html>"
    fetcher = _StaticExhibitFetcher(
        {(320193, "0000100-26-000001"): (
            _index_with_ex99(size=len(oversized)),
            {"ex-99-1.htm": oversized},
        )},
    )
    since = now - dt.timedelta(days=2)
    with _connect(db_path) as conn:
        plan = await backfill.build_plan(conn, fetcher, since)
        backfill.apply_plan(conn, plan)

    assert plan.items == []
    assert any(fid == f for f, _ in plan.fetch_failures)
    assert raw_path.read_text() == body_before


@pytest.mark.asyncio
async def test_apply_appends_all_ex_99_exhibits_in_order(
    db_path: str, tmp_path: Path
) -> None:
    """Multiple-exhibit policy: when 99.1 + 99.2 + 99.3 are all present
    and fit under the cumulative cap, ALL three are appended to
    `raw_text` in numeric order (mirrors the live detail_fetcher
    semantics so backfilled and freshly-ingested rows look identical)."""
    now = dt.datetime.now(tz=dt.UTC)
    raw_root = tmp_path / "raw"
    fid, raw_path = _insert_8k_filing(
        db_path,
        raw_root=raw_root,
        accession="0000150-26-000001",
        cik=320193,
        filed_at=now,
        item_codes=["2.02"],
        body_text="Item 2.02 Results.",
    )
    _insert_no_trade_proposal(db_path, fid, "dec-150")
    _insert_prefilter(db_path, fid)

    body_991 = _ex99_body("EX991_PRESS")
    body_992 = _ex99_body("EX992_SUPPLEMENT")
    body_993 = _ex99_body("EX993_SLIDES")
    fetcher = _StaticExhibitFetcher(
        {(320193, "0000150-26-000001"): (
            [
                {"name": "primary.htm", "type": "text.gif", "size": "5000"},
                # Out-of-order in the index — selector must sort.
                {"name": "ex-99-3.htm", "type": "text.gif", "size": str(len(body_993))},
                {"name": "ex-99-1.htm", "type": "text.gif", "size": str(len(body_991))},
                {"name": "ex-99-2.htm", "type": "text.gif", "size": str(len(body_992))},
            ],
            {
                "ex-99-1.htm": body_991,
                "ex-99-2.htm": body_992,
                "ex-99-3.htm": body_993,
            },
        )},
    )
    since = now - dt.timedelta(days=2)
    with _connect(db_path) as conn:
        plan = await backfill.build_plan(conn, fetcher, since)
        backfill.apply_plan(conn, plan)

    augmented = raw_path.read_text()
    assert "EX991_PRESS" in augmented
    assert "EX992_SUPPLEMENT" in augmented
    assert "EX993_SLIDES" in augmented
    # Numeric ordering preserved.
    assert (
        augmented.find("EX991_PRESS")
        < augmented.find("EX992_SUPPLEMENT")
        < augmented.find("EX993_SLIDES")
    )
    # Per-exhibit separators present.
    assert "--- EXHIBIT 99.1 ---" in augmented
    assert "--- EXHIBIT 99.2 ---" in augmented
    assert "--- EXHIBIT 99.3 ---" in augmented


@pytest.mark.asyncio
async def test_partial_exhibit_set_still_augments(
    db_path: str, tmp_path: Path
) -> None:
    """If 99.2 is missing/undelivered but 99.1 and 99.3 succeed,
    `raw_text` still gets 99.1 + 99.3. The missing exhibit becomes a
    `fetch_failures` entry but does not block augmentation."""
    now = dt.datetime.now(tz=dt.UTC)
    raw_root = tmp_path / "raw"
    fid, raw_path = _insert_8k_filing(
        db_path,
        raw_root=raw_root,
        accession="0000160-26-000001",
        cik=320193,
        filed_at=now,
        item_codes=["7.01"],
        body_text="Item 7.01 Reg FD.",
    )
    body_991 = _ex99_body("EX991_OK")
    body_993 = _ex99_body("EX993_OK")
    fetcher = _StaticExhibitFetcher(
        {(320193, "0000160-26-000001"): (
            [
                {"name": "primary.htm", "type": "text.gif", "size": "5000"},
                {"name": "ex-99-1.htm", "type": "text.gif", "size": str(len(body_991))},
                {"name": "ex-99-2.htm", "type": "text.gif", "size": "1000"},
                {"name": "ex-99-3.htm", "type": "text.gif", "size": str(len(body_993))},
            ],
            # 99.2 absent from delivered bodies — simulates a 4xx on its fetch.
            {"ex-99-1.htm": body_991, "ex-99-3.htm": body_993},
        )},
    )
    since = now - dt.timedelta(days=2)
    with _connect(db_path) as conn:
        plan = await backfill.build_plan(conn, fetcher, since)
        backfill.apply_plan(conn, plan)

    augmented = raw_path.read_text()
    assert "EX991_OK" in augmented
    assert "EX993_OK" in augmented
    # 99.2 reported as a failure but did not block the rest.
    assert any(fid == f and "ex-99-2" in r for f, r in plan.fetch_failures)


@pytest.mark.asyncio
async def test_cumulative_cap_stops_at_overshoot(
    db_path: str, tmp_path: Path
) -> None:
    """Cumulative cap: 99.1 fits (~half cap); 99.1 + 99.2 overshoots →
    99.2 is skipped; 99.3 is also skipped (loop breaks on the first
    overshoot, doesn't keep trying)."""
    from ingestion.detail_fetcher import _EXHIBIT_CUMULATIVE_BYTE_CAP

    now = dt.datetime.now(tz=dt.UTC)
    raw_root = tmp_path / "raw"
    fid, raw_path = _insert_8k_filing(
        db_path,
        raw_root=raw_root,
        accession="0000170-26-000001",
        cik=320193,
        filed_at=now,
        item_codes=["2.02"],
    )
    half = (_EXHIBIT_CUMULATIVE_BYTE_CAP // 2) + 100_000
    body_991 = (
        b"<html><body><p>EX991_FITS</p><p>" + b"x" * half + b"</p></body></html>"
    )
    body_992 = (
        b"<html><body><p>EX992_OVERFLOW</p><p>" + b"y" * half + b"</p></body></html>"
    )
    body_993 = b"<html><body><p>EX993_NEVER</p></body></html>"
    fetcher = _StaticExhibitFetcher(
        {(320193, "0000170-26-000001"): (
            [
                {"name": "primary.htm", "type": "text.gif", "size": "5000"},
                {"name": "ex-99-1.htm", "type": "text.gif", "size": str(len(body_991))},
                {"name": "ex-99-2.htm", "type": "text.gif", "size": str(len(body_992))},
                {"name": "ex-99-3.htm", "type": "text.gif", "size": str(len(body_993))},
            ],
            {
                "ex-99-1.htm": body_991,
                "ex-99-2.htm": body_992,
                "ex-99-3.htm": body_993,
            },
        )},
    )
    since = now - dt.timedelta(days=2)
    with _connect(db_path) as conn:
        plan = await backfill.build_plan(conn, fetcher, since)
        backfill.apply_plan(conn, plan)

    augmented = raw_path.read_text()
    assert "EX991_FITS" in augmented
    assert "EX992_OVERFLOW" not in augmented
    assert "EX993_NEVER" not in augmented


@pytest.mark.asyncio
async def test_filing_with_no_proposal_or_prefilter_still_augments(
    db_path: str, tmp_path: Path
) -> None:
    """A filing that hasn't been analyzed yet (no proposal, no prefilter
    decision) still gets its raw_text augmented. The DELETEs silently
    affect 0 rows; UPDATE still lands."""
    now = dt.datetime.now(tz=dt.UTC)
    raw_root = tmp_path / "raw"
    fid, raw_path = _insert_8k_filing(
        db_path,
        raw_root=raw_root,
        accession="0000110-26-000001",
        cik=320193,
        filed_at=now,
        item_codes=["7.01"],
    )
    fetcher = _StaticExhibitFetcher(
        {(320193, "0000110-26-000001"): (
            _index_with_ex99(),
            {"ex-99-1.htm": _ex99_body("UNANALYZED_BODY_MARKER")},
        )},
    )
    since = now - dt.timedelta(days=2)
    with _connect(db_path) as conn:
        plan = await backfill.build_plan(conn, fetcher, since)
        backfill.apply_plan(conn, plan)

    assert len(plan.items) == 1
    augmented = raw_path.read_text()
    assert "UNANALYZED_BODY_MARKER" in augmented


@pytest.mark.asyncio
async def test_partial_state_filings_are_excluded(
    db_path: str, tmp_path: Path
) -> None:
    """Filings with `ingest_state='partial'` were never successfully
    ingested; the backfill must not touch them — they need a different
    recovery path."""
    now = dt.datetime.now(tz=dt.UTC)
    raw_root = tmp_path / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    raw_path = raw_root / "0000120-26-000001.partial"
    insert_filing(
        db_path,
        FilingRow(
            accession_number="0000120-26-000001",
            cik=320193,
            form_type="8-K",
            filed_at=now,
            fetched_at=now,
            raw_text_path=str(raw_path),
            content_hash="",
            item_codes=json.dumps(["2.02"]),
            issuer_ticker=None,
            ingest_state="partial",
            ingest_error="some error",
        ),
    )
    fetcher = _StaticExhibitFetcher({})
    since = now - dt.timedelta(days=2)
    with _connect(db_path) as conn:
        plan = await backfill.build_plan(conn, fetcher, since)
    assert plan.items == []


@pytest.mark.asyncio
async def test_apply_handles_filing_with_no_prefilter_row(
    db_path: str, tmp_path: Path
) -> None:
    """No prefilter row yet → DELETE silently affects 0 rows; no crash."""
    now = dt.datetime.now(tz=dt.UTC)
    raw_root = tmp_path / "raw"
    fid, _ = _insert_8k_filing(
        db_path,
        raw_root=raw_root,
        accession="0000130-26-000001",
        cik=320193,
        filed_at=now,
        item_codes=["8.01"],
    )
    fetcher = _StaticExhibitFetcher(
        {(320193, "0000130-26-000001"): (
            _index_with_ex99(),
            {"ex-99-1.htm": _ex99_body()},
        )},
    )
    since = now - dt.timedelta(days=2)
    with _connect(db_path) as conn:
        plan = await backfill.build_plan(conn, fetcher, since)
        backfill.apply_plan(conn, plan)
    # No exception; planned augmentation succeeded.
    assert len(plan.items) == 1


# ---------------------------------------------------------------------------
# Tests — output / reporting
# ---------------------------------------------------------------------------


def test_render_plan_summarizes_counts() -> None:
    plan = backfill.BackfillPlan(
        items=[
            backfill.BackfillPlanItem(
                filing_id=1,
                cik=320193,
                accession_number="0000001-26-000001",
                raw_text_path="/x/0000001.txt",
                augmented_text="…",
                new_content_hash="abc",
                proposal_ids=[10],
                prefilter_decision_ids=[100],
            ),
        ],
        skipped_no_exhibit_in_index=[2],
        skipped_already_augmented=[3],
        fetch_failures=[(4, "5xx")],
        aborted_due_to_executions=[],
        aborted_due_to_non_no_trade=[],
    )
    rendered = backfill.render_plan(plan, apply=False)
    assert "DRY-RUN" in rendered
    assert "filings to augment:                1" in rendered
    assert "proposals queued for delete:       1" in rendered
    assert "prefilter rows queued for delete:  1" in rendered
    assert "skipped (no EX-99 in index):       1" in rendered
    assert "skipped (already augmented):       1" in rendered
    assert "fetch failures:                    1" in rendered
    assert "rerun with --apply" in rendered


def test_render_plan_apply_mode_drops_dryrun_hint() -> None:
    plan = backfill.BackfillPlan(
        items=[],
        skipped_no_exhibit_in_index=[],
        skipped_already_augmented=[],
        fetch_failures=[],
        aborted_due_to_executions=[],
        aborted_due_to_non_no_trade=[],
    )
    rendered = backfill.render_plan(plan, apply=True)
    assert "APPLY" in rendered
    assert "rerun with --apply" not in rendered


# ---------------------------------------------------------------------------
# Tests — transactional safety
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_is_transactional_for_db_mutations(
    db_path: str, tmp_path: Path
) -> None:
    """Apply two filings in one plan; both DB mutations must land together."""
    now = dt.datetime.now(tz=dt.UTC)
    raw_root = tmp_path / "raw"
    fid_a, _ = _insert_8k_filing(
        db_path, raw_root=raw_root, accession="0000200-26-000001",
        cik=320193, filed_at=now, item_codes=["2.02"],
    )
    fid_b, _ = _insert_8k_filing(
        db_path, raw_root=raw_root, accession="0000200-26-000002",
        cik=789019, filed_at=now, item_codes=["7.01"],
    )
    pid_a = _insert_no_trade_proposal(db_path, fid_a, "dec-200a")
    pid_b = _insert_no_trade_proposal(db_path, fid_b, "dec-200b")
    pf_a = _insert_prefilter(db_path, fid_a)
    pf_b = _insert_prefilter(db_path, fid_b)

    fetcher = _StaticExhibitFetcher(
        {
            (320193, "0000200-26-000001"): (
                _index_with_ex99(),
                {"ex-99-1.htm": _ex99_body("BODY_A")},
            ),
            (789019, "0000200-26-000002"): (
                _index_with_ex99(),
                {"ex-99-1.htm": _ex99_body("BODY_B")},
            ),
        },
    )
    since = now - dt.timedelta(days=2)
    with _connect(db_path) as conn:
        plan = await backfill.build_plan(conn, fetcher, since)
        backfill.apply_plan(conn, plan)

    with _connect(db_path) as conn:
        # Both proposals + prefilter rows gone.
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM proposals WHERE id IN (?, ?)",
                (pid_a, pid_b),
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM prefilter_decisions WHERE id IN (?, ?)",
                (pf_a, pf_b),
            ).fetchone()[0]
            == 0
        )


@pytest.mark.asyncio
async def test_aborted_filing_keeps_its_prefilter_decision(
    db_path: str, tmp_path: Path
) -> None:
    """Inverse safety: abort guard preserves proposal AND prefilter row,
    so the filing is NOT silently re-queued behind the operator's back."""
    now = dt.datetime.now(tz=dt.UTC)
    raw_root = tmp_path / "raw"
    fid, raw_path = _insert_8k_filing(
        db_path, raw_root=raw_root, accession="0000300-26-000001",
        cik=320193, filed_at=now, item_codes=["2.02"],
    )
    pid = _insert_no_trade_proposal(db_path, fid, "dec-300")
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
    pf = _insert_prefilter(db_path, fid)

    fetcher = _StaticExhibitFetcher(
        {(320193, "0000300-26-000001"): (
            _index_with_ex99(),
            {"ex-99-1.htm": _ex99_body()},
        )},
    )
    since = now - dt.timedelta(days=2)
    with _connect(db_path) as conn:
        plan = await backfill.build_plan(conn, fetcher, since)
        backfill.apply_plan(conn, plan)

    assert plan.aborted_due_to_executions == [fid]
    assert plan.items == []
    with _connect(db_path) as conn:
        assert (
            conn.execute(
                "SELECT id FROM prefilter_decisions WHERE id = ?", (pf,)
            ).fetchone()
            is not None
        )


@pytest.mark.asyncio
async def test_post_apply_prefilter_decisions_for_affected_filings_empty(
    db_path: str, tmp_path: Path
) -> None:
    """Direct acceptance criterion from task #2: after `--apply`, the
    `prefilter_decisions` rows for affected filing_ids are gone."""
    now = dt.datetime.now(tz=dt.UTC)
    raw_root = tmp_path / "raw"
    fid_a, _ = _insert_8k_filing(
        db_path, raw_root=raw_root, accession="0000400-26-000001",
        cik=320193, filed_at=now, item_codes=["2.02"],
    )
    fid_b, _ = _insert_8k_filing(
        db_path, raw_root=raw_root, accession="0000400-26-000002",
        cik=789019, filed_at=now, item_codes=["8.01"],
    )
    _insert_prefilter(db_path, fid_a)
    _insert_prefilter(db_path, fid_b)

    fetcher = _StaticExhibitFetcher(
        {
            (320193, "0000400-26-000001"): (
                _index_with_ex99(),
                {"ex-99-1.htm": _ex99_body()},
            ),
            (789019, "0000400-26-000002"): (
                _index_with_ex99(),
                {"ex-99-1.htm": _ex99_body()},
            ),
        },
    )
    since = now - dt.timedelta(days=2)
    with _connect(db_path) as conn:
        plan = await backfill.build_plan(conn, fetcher, since)
        backfill.apply_plan(conn, plan)

    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id FROM prefilter_decisions "
            "WHERE filing_id IN (?, ?)",
            (fid_a, fid_b),
        ).fetchall()
        assert rows == []


# ---------------------------------------------------------------------------
# Tests — raw_text_already_augmented helper
# ---------------------------------------------------------------------------


def test_raw_text_already_augmented_detects_separator() -> None:
    text = "Item 2.02.\n\n--- EXHIBIT 99.1 ---\n\nEarnings."
    assert backfill.raw_text_already_augmented(text)


def test_raw_text_already_augmented_does_not_false_positive_on_body_mention() -> None:
    """Mere mention of `Exhibit 99` in the body (e.g., the Reg FD
    disclaimer) must NOT trip the idempotency check — only the canonical
    `--- EXHIBIT 99` separator does."""
    text = (
        "Item 2.02 Results of Operations and Financial Condition. "
        "The information furnished pursuant to Exhibit 99.1 shall not be "
        "deemed filed for the purposes of Section 18."
    )
    assert not backfill.raw_text_already_augmented(text)

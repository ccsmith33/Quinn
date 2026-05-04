"""Backfill `filings.issuer_ticker` for rows that were ingested with NULL.

Production hotfix companion (2026-05-04): the detail-fetcher bug fixed in
`src/ingestion/ticker_resolver.py` left today's filings with NULL ticker.
The analyzer's universe-summary builder declared them all "not in
current universe member list", and Sonnet refused 36/36 proposals as
no_trade. This script rescues those filings so the agent loop will
re-analyze them with a populated ticker.

Behavior
--------
1. SELECT filings WHERE issuer_ticker IS NULL AND filed_at >= now-N days
   (default N=1; configurable via --since).
2. For each filing, resolve ticker via the same TickerResolver the
   detail-fetcher uses. Resolution is best-effort; unresolved CIKs are
   left NULL and reported.
3. For filings whose ticker resolved AND that have at least one proposal
   row attached, plan to:
     a) UPDATE filings.issuer_ticker = resolved
     b) DELETE proposal_reviews where proposal_id ∈ those filings'
        proposals
     c) DELETE proposals where filing_id ∈ resolved set
   Plans are printed in dry-run; only `--apply` mutates.
4. Idempotent: on a second `--apply` run, the filings have non-NULL
   ticker → SELECT returns nothing → no work is performed.

Re-analysis trigger choice
--------------------------
For each backfilled filing, DELETE its `proposals` rows AND its
`prefilter_decisions` row, then UPDATE `filings.issuer_ticker`. This
re-arms the agent's `_crash_recovery_scan` (`src/app/loop.py:319`)
which on boot picks up filings via:

    LEFT JOIN prefilter_decisions p ON p.filing_id = f.id
    WHERE p.id IS NULL

Rationale:

- Deleting the proposal alone is necessary but not sufficient: the
  agent loop's idempotency lookup (`_lookup_existing_proposal` in
  `src/app/loop.py`) keys on `decision_id = hash(filing_id, model_id,
  prompt_version)`. Without the proposal row, the lookup misses on
  re-run — but the loop must FIRST be invoked for that filing again,
  and it never will be unless the prefilter row is also gone.
- The "superseded" sentinel approach was considered and rejected: it
  would require schema or analyzer changes (out of scope per the
  hotfix constraints).
- Audit history of the LLM call is still preserved in `llm_calls`
  (the proposals table is a derived projection of those calls; the
  raw call record is what the daily report uses for cost telemetry).
- Re-running the analyzer for the same `decision_id` produces a fresh
  `llm_calls` row but the same `decision_id`; the daily-cost report
  will sum both calls under that key. Acceptable for this hotfix —
  treat it as the cost of correcting a bug-induced wrong refusal.

Operator runbook
----------------
1. Stop the agent: `systemctl stop quinn` (recommended; without this
   the resolver may race against the agent's own ticker fetches, and
   on `--apply` an in-flight analyzer call could create a fresh
   proposal between our DELETE and the recovery scan). The
   `aborted_*` guards prevent silent data loss but stopping the
   agent is the cleanest option.
2. Run `--dry-run` first to inspect the plan.
3. Run `--apply` to mutate.
4. Start the agent: `systemctl start quinn`. On boot,
   `_crash_recovery_scan` re-queues the backfilled filings.

Safety rails
------------
- Default to dry-run; no `--apply` = no writes.
- ABORT if any proposal targeted for deletion has an `executions` row.
  Today the production droplet has zero executions ever, so this
  guard is precautionary; it makes the script future-safe for the
  unlikely case where a proposal got executed before the bug was
  found.
- ABORT if any proposal has a non-`no_trade` kind. Today every
  proposal is no_trade, but we don't want to silently lose a real
  proposal that's been queued for execution.
- All mutations run inside a single SQLite transaction.
- WHERE clause requires BOTH `issuer_ticker IS NULL` AND a recent
  `filed_at`; the script never touches old or already-tickered rows.

Usage
-----
  # Default: dry-run, last 24 hours.
  .venv/bin/python ops/scripts/backfill_issuer_tickers.py

  # Apply the planned changes.
  .venv/bin/python ops/scripts/backfill_issuer_tickers.py --apply

  # Wider window, custom DB path.
  .venv/bin/python ops/scripts/backfill_issuer_tickers.py \\
      --since 7 --db /var/lib/quinn/journal.db
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

# Allow running as a script: add `src/` to path so `ingestion.*` resolves.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


_DEFAULT_DB_PATH = "/var/lib/quinn/journal.db"
_DEFAULT_CACHE_PATH = "/var/lib/quinn/cache/cik_ticker_map.json"
_DEFAULT_USER_AGENT = "Quinn-Backfill/1 ccsmith33@crimson.ua.edu"


# ---------------------------------------------------------------------------
# Pure-logic helpers (testable without SEC or filesystem)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilingNeedingTicker:
    """A filing row eligible for ticker backfill."""

    filing_id: int
    cik: int
    accession_number: str
    form_type: str


@dataclass(frozen=True)
class BackfillPlanItem:
    """One filing's worth of planned mutations."""

    filing_id: int
    cik: int
    new_ticker: str
    proposal_ids: list[int]  # proposals to delete (may be empty)
    prefilter_decision_ids: list[int]  # prefilter rows to delete (may be empty)


@dataclass(frozen=True)
class BackfillPlan:
    """Aggregated plan + diagnostics from the read-only analysis pass."""

    items: list[BackfillPlanItem]
    unresolved: list[FilingNeedingTicker]  # CIKs the resolver could not place
    aborted_due_to_executions: list[int]  # filing_ids
    aborted_due_to_non_no_trade: list[int]  # filing_ids


def select_filings_needing_ticker(
    conn: sqlite3.Connection, since: dt.datetime
) -> list[FilingNeedingTicker]:
    """SELECT NULL-ticker filings filed at-or-after `since`.

    Tight WHERE clause: BOTH `issuer_ticker IS NULL` AND `filed_at >= ?`.
    Old rows and already-tickered rows are never touched.
    """
    rows = conn.execute(
        "SELECT id, cik, accession_number, form_type "
        "FROM filings "
        "WHERE issuer_ticker IS NULL AND filed_at >= ? "
        "ORDER BY id",
        (since,),
    ).fetchall()
    return [
        FilingNeedingTicker(
            filing_id=int(r["id"]),
            cik=int(r["cik"]),
            accession_number=str(r["accession_number"]),
            form_type=str(r["form_type"]),
        )
        for r in rows
    ]


def proposals_for_filing(conn: sqlite3.Connection, filing_id: int) -> list[sqlite3.Row]:
    """Fetch proposal rows attached to `filing_id`."""
    return list(
        conn.execute(
            "SELECT id, kind FROM proposals WHERE filing_id = ?", (filing_id,)
        ).fetchall()
    )


def proposal_has_execution(conn: sqlite3.Connection, proposal_id: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM executions WHERE proposal_id = ? LIMIT 1", (proposal_id,)
    ).fetchone()
    return row is not None


def prefilter_decisions_for_filing(
    conn: sqlite3.Connection, filing_id: int
) -> list[int]:
    """IDs of `prefilter_decisions` rows attached to `filing_id`.

    Used so `apply_plan` can clear them — the agent's
    `_crash_recovery_scan` (`src/app/loop.py:319`) re-queues only filings
    where `LEFT JOIN prefilter_decisions p WHERE p.id IS NULL`. Without
    this clear, a backfilled filing is invisible to the recovery scan
    and never re-analyzed.
    """
    rows = conn.execute(
        "SELECT id FROM prefilter_decisions WHERE filing_id = ?", (filing_id,)
    ).fetchall()
    return [int(r["id"]) for r in rows]


def apply_plan(conn: sqlite3.Connection, plan: BackfillPlan) -> None:
    """Apply mutations in a single transaction.

    Order: DELETE proposal_reviews → DELETE proposals → DELETE
    prefilter_decisions → UPDATE filings. Reverse FK order so referential
    integrity holds at every step (proposal_reviews → proposals →
    filings; prefilter_decisions → filings).

    The `prefilter_decisions` deletion is what re-arms the agent's
    `_crash_recovery_scan` (`src/app/loop.py:319`) for the backfilled
    filings — see `prefilter_decisions_for_filing`. Without it, the
    recovery scan's `WHERE p.id IS NULL` predicate excludes the filing
    and the script's stated purpose ("get re-analyzed") fails silently.
    """
    if not plan.items:
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        for item in plan.items:
            for pid in item.proposal_ids:
                conn.execute(
                    "DELETE FROM proposal_reviews WHERE proposal_id = ?", (pid,)
                )
            for pid in item.proposal_ids:
                conn.execute("DELETE FROM proposals WHERE id = ?", (pid,))
            for pf_id in item.prefilter_decision_ids:
                conn.execute(
                    "DELETE FROM prefilter_decisions WHERE id = ?", (pf_id,)
                )
            conn.execute(
                "UPDATE filings SET issuer_ticker = ? WHERE id = ?",
                (item.new_ticker, item.filing_id),
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


# ---------------------------------------------------------------------------
# Resolver protocol — lets tests substitute a static map
# ---------------------------------------------------------------------------


class _ResolverLike(Protocol):
    async def resolve(self, cik: int) -> str | None: ...


async def build_plan(
    conn: sqlite3.Connection,
    resolver: _ResolverLike,
    since: dt.datetime,
) -> BackfillPlan:
    """Read-only analysis pass — returns the planned mutations.

    Each filing's safety preconditions are checked:
      - Resolver returns non-None ticker → eligible for update
      - All attached proposals are kind='no_trade' → safe to delete
      - No proposal has an execution → safe to delete
    Filings that fail any precondition are reported in the plan's
    `aborted_*` lists and are NOT included in `items`.
    """
    needing = select_filings_needing_ticker(conn, since)
    items: list[BackfillPlanItem] = []
    unresolved: list[FilingNeedingTicker] = []
    aborted_executions: list[int] = []
    aborted_non_no_trade: list[int] = []

    for f in needing:
        resolved = await resolver.resolve(f.cik)
        if resolved is None:
            unresolved.append(f)
            continue

        proposals = proposals_for_filing(conn, f.filing_id)
        # Safety: refuse if any proposal has an execution attached.
        if any(proposal_has_execution(conn, int(p["id"])) for p in proposals):
            aborted_executions.append(f.filing_id)
            continue
        # Safety: refuse if any proposal is something we shouldn't drop.
        if any(str(p["kind"]) != "no_trade" for p in proposals):
            aborted_non_no_trade.append(f.filing_id)
            continue

        items.append(
            BackfillPlanItem(
                filing_id=f.filing_id,
                cik=f.cik,
                new_ticker=resolved,
                proposal_ids=[int(p["id"]) for p in proposals],
                prefilter_decision_ids=prefilter_decisions_for_filing(
                    conn, f.filing_id
                ),
            )
        )

    return BackfillPlan(
        items=items,
        unresolved=unresolved,
        aborted_due_to_executions=aborted_executions,
        aborted_due_to_non_no_trade=aborted_non_no_trade,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_plan(plan: BackfillPlan, *, apply: bool) -> str:
    """Human-readable summary suitable for stdout."""
    lines: list[str] = []
    mode = "APPLY" if apply else "DRY-RUN"
    lines.append(f"=== backfill_issuer_tickers ({mode}) ===")
    lines.append(f"resolved filings:           {len(plan.items)}")
    n_proposals = sum(len(i.proposal_ids) for i in plan.items)
    n_prefilter = sum(len(i.prefilter_decision_ids) for i in plan.items)
    lines.append(f"proposals queued for delete: {n_proposals}")
    lines.append(f"prefilter rows queued for delete: {n_prefilter}")
    lines.append(f"unresolved (no SEC match):   {len(plan.unresolved)}")
    lines.append(
        f"aborted (execution exists):  {len(plan.aborted_due_to_executions)}"
    )
    lines.append(
        f"aborted (non-no_trade kind): {len(plan.aborted_due_to_non_no_trade)}"
    )
    if plan.items:
        lines.append("")
        lines.append("Planned updates:")
        for item in plan.items:
            lines.append(
                f"  filing_id={item.filing_id} cik={item.cik} "
                f"→ ticker={item.new_ticker} "
                f"(proposals: {item.proposal_ids or '[]'}, "
                f"prefilter: {item.prefilter_decision_ids or '[]'})"
            )
    if plan.unresolved:
        lines.append("")
        lines.append("Unresolved (CIK not in SEC company_tickers.json):")
        for f in plan.unresolved:
            lines.append(
                f"  filing_id={f.filing_id} cik={f.cik} "
                f"form={f.form_type} accession={f.accession_number}"
            )
    if plan.aborted_due_to_executions:
        lines.append("")
        lines.append(
            "ABORTED — filing has a proposal with an execution row attached:"
        )
        for fid in plan.aborted_due_to_executions:
            lines.append(f"  filing_id={fid}  (skip; investigate manually)")
    if plan.aborted_due_to_non_no_trade:
        lines.append("")
        lines.append(
            "ABORTED — filing has a non-no_trade proposal kind (potentially queued):"
        )
        for fid in plan.aborted_due_to_non_no_trade:
            lines.append(f"  filing_id={fid}  (skip; investigate manually)")
    if not apply and plan.items:
        lines.append("")
        lines.append("(dry-run; rerun with --apply to mutate the database)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backfill NULL issuer_ticker on recent filings.",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Mutate the database. Without this flag the script is read-only.",
    )
    p.add_argument(
        "--since",
        type=int,
        default=1,
        help="Look back N days (default 1; tight default for the hotfix window).",
    )
    p.add_argument(
        "--db",
        default=os.environ.get("QUINN_DB_PATH", _DEFAULT_DB_PATH),
        help=f"SQLite path (default {_DEFAULT_DB_PATH} or $QUINN_DB_PATH).",
    )
    p.add_argument(
        "--cache",
        default=_DEFAULT_CACHE_PATH,
        help=f"CIK→ticker cache path (default {_DEFAULT_CACHE_PATH}).",
    )
    p.add_argument(
        "--user-agent",
        default=os.environ.get("EDGAR_USER_AGENT", _DEFAULT_USER_AGENT),
        help="User-Agent header for SEC fetch (NFR-17).",
    )
    return p.parse_args(argv)


async def _amain(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    from ingestion.edgar_client import EdgarClient
    from ingestion.ticker_resolver import TickerResolver

    since = dt.datetime.now(tz=dt.UTC) - dt.timedelta(days=args.since)

    edgar = EdgarClient(user_agent=args.user_agent)
    resolver = TickerResolver(edgar=edgar, cache_path=Path(args.cache))

    try:
        with sqlite3.connect(args.db, isolation_level=None, timeout=30.0) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            plan = await build_plan(conn, resolver, since)
            print(render_plan(plan, apply=args.apply))
            if args.apply and plan.items:
                apply_plan(conn, plan)
                n_proposals = sum(len(i.proposal_ids) for i in plan.items)
                n_prefilter = sum(len(i.prefilter_decision_ids) for i in plan.items)
                print(
                    f"\napplied: updated {len(plan.items)} filings, "
                    f"deleted {n_proposals} proposals, "
                    f"deleted {n_prefilter} prefilter_decisions rows"
                )
                print(
                    "\nNEXT STEP: restart the agent (systemctl restart quinn). "
                    "On boot, `_crash_recovery_scan` will re-queue the "
                    "backfilled filings for analysis."
                )
    finally:
        await edgar.aclose()
    return 0


def main() -> int:  # pragma: no cover — thin asyncio shim
    return asyncio.run(_amain())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

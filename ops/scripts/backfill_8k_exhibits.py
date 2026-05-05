"""Backfill EX-99.x exhibit content into 8-K filings ingested before ADR-008.

Production hotfix companion (2026-05-05): the detail-fetcher fix in
`src/ingestion/detail_fetcher.py` adds Exhibit 99.x augmentation for 8-Ks
with furnish item codes (2.02 / 7.01 / 8.01). Mon (2026-05-04) and
Tue (2026-05-05) earnings filings were ingested with body-only `raw_text`,
so Sonnet refused them all citing "no figures available." This script
rescues those filings by appending EX-99.x content to their `raw_text_path`
and re-arming the agent's `_crash_recovery_scan` to re-analyze them.

Behavior
--------
1. SELECT filings WHERE form_type='8-K' AND item_codes JSON contains any
   of {"2.02","7.01","8.01"} AND filed_at >= now - N days (default 2;
   covers Mon+Tue).
2. For each filing, fetch its `index.json`, enumerate ALL EX-99.x
   exhibits in numeric order, and fetch each body in turn under a
   cumulative byte cap (`_EXHIBIT_CUMULATIVE_BYTE_CAP`). Reuses
   `_select_all_ex_99` from `detail_fetcher` (no duplication).
3. Read the existing `raw_text_path`, append every successfully-fetched
   exhibit's text (via the `--- EXHIBIT 99.{n} ---` separator), write
   back, recompute `content_hash`. SKIP filings whose raw_text already
   contains `--- EXHIBIT 99` (idempotency for re-runs after partial
   failures).
4. For each successfully augmented filing: DELETE its `proposal_reviews`
   → `proposals` → `prefilter_decisions` rows so the agent's boot-time
   `_crash_recovery_scan` re-queues it (mirrors `backfill_issuer_tickers.py`
   pattern).
5. Single `BEGIN IMMEDIATE` transaction; ABORT if any proposal has an
   `executions` row attached or a non-`no_trade` kind. Dry-run by default;
   `--apply` opts in. Idempotent: a second `--apply` finds nothing to do
   because `raw_text` already contains the separator.

Operator runbook
----------------
1. Stop the agent: `systemctl stop quinn` (recommended — same race
   considerations as `backfill_issuer_tickers.py`).
2. Run `--dry-run` (the default) to inspect the plan.
3. Run `--apply` to mutate.
4. Start the agent: `systemctl start quinn`. On boot,
   `_crash_recovery_scan` re-queues the backfilled filings.

Safety rails
------------
- Default dry-run; `--apply` required for writes.
- ABORT if any proposal targeted for deletion has an `executions` row.
- ABORT if any proposal has a non-`no_trade` kind (might be queued for
  execution).
- All mutations in a single SQLite transaction.
- WHERE clause restricts to 8-K + furnish-item-code + recent filed_at.
- Idempotent via the `EXHIBIT 99` separator in `raw_text`: a second
  `--apply` skips already-augmented filings.

Usage
-----
  # Default: dry-run, last 2 days.
  .venv/bin/python ops/scripts/backfill_8k_exhibits.py

  # Apply the planned changes.
  .venv/bin/python ops/scripts/backfill_8k_exhibits.py --apply

  # Wider window, custom DB path.
  .venv/bin/python ops/scripts/backfill_8k_exhibits.py \\
      --since 7 --db /var/lib/quinn/journal.db
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
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

# Constants reused from detail_fetcher — single source of truth. Imports
# come after the sys.path manipulation above and are intentionally
# placed here, not at the top of the file (E402 / I001 suppressed).
from ingestion.detail_fetcher import (  # noqa: E402, I001
    _EXHIBIT_CUMULATIVE_BYTE_CAP,
    _EXHIBIT_SEPARATOR_TEMPLATE,
    _FURNISH_ITEM_CODES,
    _select_all_ex_99,
)
from ingestion.normalize import content_hash  # noqa: E402, I001
from ingestion.parsers.html_to_text import html_to_text  # noqa: E402, I001

_DEFAULT_DB_PATH = "/var/lib/quinn/journal.db"
_DEFAULT_USER_AGENT = "Quinn-Backfill/1 ccsmith33@crimson.ua.edu"
_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"


# ---------------------------------------------------------------------------
# Pure-logic helpers (testable without SEC or filesystem)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilingNeedingExhibit:
    """A filing row eligible for exhibit backfill (8-K, furnish items,
    `raw_text` does not yet contain the EXHIBIT 99 separator)."""

    filing_id: int
    cik: int
    accession_number: str
    raw_text_path: str
    item_codes: list[str]


@dataclass(frozen=True)
class BackfillPlanItem:
    """One filing's worth of planned mutations."""

    filing_id: int
    cik: int
    accession_number: str
    raw_text_path: str
    augmented_text: str
    new_content_hash: str
    proposal_ids: list[int]
    prefilter_decision_ids: list[int]


@dataclass(frozen=True)
class BackfillPlan:
    """Aggregated plan + diagnostics from the read-only analysis pass."""

    items: list[BackfillPlanItem]
    skipped_no_exhibit_in_index: list[int]  # filing_ids with no EX-99 found
    skipped_already_augmented: list[int]  # raw_text already had separator
    fetch_failures: list[tuple[int, str]]  # (filing_id, reason)
    aborted_due_to_executions: list[int]
    aborted_due_to_non_no_trade: list[int]


def select_filings_needing_exhibit(
    conn: sqlite3.Connection, since: dt.datetime
) -> list[FilingNeedingExhibit]:
    """SELECT 8-K filings filed at-or-after `since` whose `item_codes` JSON
    contains any furnish item code AND whose `raw_text` doesn't already
    show the EXHIBIT 99 separator.

    The item_codes-intersection check happens in Python (the column is a
    JSON-string array; SQLite has no native JSON-array intersection across
    versions we target). The recency + form_type filtering happens in SQL
    so the row scan is bounded.
    """
    rows = conn.execute(
        "SELECT id, cik, accession_number, raw_text_path, item_codes "
        "FROM filings "
        "WHERE form_type = '8-K' "
        "  AND filed_at >= ? "
        "  AND ingest_state = 'ok' "
        "  AND item_codes IS NOT NULL "
        "ORDER BY id",
        (since,),
    ).fetchall()
    eligible: list[FilingNeedingExhibit] = []
    for r in rows:
        try:
            codes = json.loads(r["item_codes"]) if r["item_codes"] else []
        except json.JSONDecodeError:
            continue
        if not isinstance(codes, list):
            continue
        if not any(c in _FURNISH_ITEM_CODES for c in codes):
            continue
        eligible.append(
            FilingNeedingExhibit(
                filing_id=int(r["id"]),
                cik=int(r["cik"]),
                accession_number=str(r["accession_number"]),
                raw_text_path=str(r["raw_text_path"]),
                item_codes=[str(c) for c in codes],
            )
        )
    return eligible


def proposals_for_filing(conn: sqlite3.Connection, filing_id: int) -> list[sqlite3.Row]:
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
    rows = conn.execute(
        "SELECT id FROM prefilter_decisions WHERE filing_id = ?", (filing_id,)
    ).fetchall()
    return [int(r["id"]) for r in rows]


def raw_text_already_augmented(text: str) -> bool:
    """The exhibit fetch wrote `--- EXHIBIT 99.{n} ---` as a separator;
    presence of `EXHIBIT 99` (with surrounding whitespace) in `raw_text`
    is a reliable idempotency marker — the original 8-K body never
    contains this exact phrasing as a standalone token (the body's
    references to exhibits use `Exhibit 99` capitalization but rarely
    in this all-caps separator form, and never bracketed by the dash
    template the augmentation writes). To be safe we look for the
    canonical separator prefix."""
    return "--- EXHIBIT 99" in text


def apply_plan(conn: sqlite3.Connection, plan: BackfillPlan) -> None:
    """Apply mutations in a single transaction.

    Order: rewrite `raw_text_path` files → DELETE proposal_reviews →
    DELETE proposals → DELETE prefilter_decisions → UPDATE filings
    (content_hash). File writes happen INSIDE the transaction window —
    if any DB mutation fails, the transaction rolls back but the file
    rewrites already happened. This is acceptable: the file content is
    derivable from EDGAR; a partial-state file rewrite without the
    matching DB updates is still valid raw_text (just not yet pointed at
    by an updated content_hash). On the next `--apply`, the idempotency
    check (raw_text already contains separator) skips the file rewrite
    and only the DB mutations re-run.
    """
    if not plan.items:
        return
    # Rewrite files first — outside the SQL transaction. If the write
    # partially fails, raw_text on disk may be replaced without DB
    # updates, but the idempotency check on next run keeps things
    # consistent.
    for item in plan.items:
        Path(item.raw_text_path).write_text(item.augmented_text)
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
                "UPDATE filings SET content_hash = ? WHERE id = ?",
                (item.new_content_hash, item.filing_id),
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


# ---------------------------------------------------------------------------
# Exhibit fetcher protocol — lets tests substitute a static map without a
# real EdgarClient. The production fetcher is a thin wrapper around the
# shared EdgarClient (ADR-002 §6 — never instantiate parallel clients).
# ---------------------------------------------------------------------------


class _ExhibitFetcherLike(Protocol):
    async def fetch(
        self, cik: int, accession_number: str
    ) -> tuple[list[dict[str, str]], dict[str, bytes]]:
        """Return (`index_items`, `name_to_body_bytes`).

        `index_items` is the parsed `directory.item[]` from index.json;
        `name_to_body_bytes` maps every fetched exhibit filename to its
        body bytes (the implementation may be lazy — only fetch the
        chosen EX-99 — but the protocol supports eager pre-fetching for
        test simplicity).
        """
        ...


async def build_plan(
    conn: sqlite3.Connection,
    fetcher: _ExhibitFetcherLike,
    since: dt.datetime,
) -> BackfillPlan:
    """Read-only analysis pass — returns the planned mutations.

    For each candidate filing:
      1. Read the existing `raw_text_path` content. Skip if already
         augmented (idempotency).
      2. Fetch its EDGAR index + every EX-99.x exhibit body. Walk
         exhibits in numeric order under a cumulative byte cap (matching
         `detail_fetcher._maybe_fetch_exhibits_99`). Skip if no EX-99
         found; degrade gracefully (any partial set still augments).
      3. Build the augmented text and new content_hash.
      4. Check safety: refuse if any attached proposal has an execution
         or a non-no_trade kind; collect the abort.
      5. Collect proposal IDs + prefilter IDs to delete.
    """
    needing = select_filings_needing_exhibit(conn, since)
    items: list[BackfillPlanItem] = []
    skipped_no_exhibit: list[int] = []
    skipped_already_augmented: list[int] = []
    fetch_failures: list[tuple[int, str]] = []
    aborted_executions: list[int] = []
    aborted_non_no_trade: list[int] = []

    for f in needing:
        # 1. Idempotency: read existing raw_text, skip if already augmented.
        try:
            existing_text = Path(f.raw_text_path).read_text()
        except OSError as exc:
            fetch_failures.append((f.filing_id, f"raw_text unreadable: {exc}"))
            continue
        if raw_text_already_augmented(existing_text):
            skipped_already_augmented.append(f.filing_id)
            continue

        # 2. Fetch index + every EX-99.x exhibit body.
        try:
            index_items, name_to_body = await fetcher.fetch(f.cik, f.accession_number)
        except Exception as exc:  # noqa: BLE001 — best-effort; bug reports surface here
            fetch_failures.append((f.filing_id, f"fetch error: {exc}"))
            continue
        candidates = _select_all_ex_99(index_items)
        if not candidates:
            skipped_no_exhibit.append(f.filing_id)
            continue

        # Walk exhibits in numeric order under cumulative byte cap.
        # Mirrors `detail_fetcher._maybe_fetch_exhibits_99` semantics:
        # a missing/empty body skips that exhibit; the first body that
        # would push us over the cap stops the loop entirely.
        chunks: list[str] = []
        cumulative_bytes = 0
        for ex_number, exhibit_name in candidates:
            body = name_to_body.get(exhibit_name)
            if body is None:
                fetch_failures.append(
                    (f.filing_id, f"exhibit {exhibit_name} not delivered")
                )
                continue
            if cumulative_bytes + len(body) > _EXHIBIT_CUMULATIVE_BYTE_CAP:
                fetch_failures.append(
                    (
                        f.filing_id,
                        f"exhibit {exhibit_name} would exceed cumulative cap "
                        f"({cumulative_bytes + len(body)} > "
                        f"{_EXHIBIT_CUMULATIVE_BYTE_CAP})",
                    )
                )
                break
            cumulative_bytes += len(body)
            exhibit_text = html_to_text(body)
            if not exhibit_text:
                fetch_failures.append(
                    (f.filing_id, f"exhibit {exhibit_name} yielded empty plaintext")
                )
                continue
            chunks.append(
                _EXHIBIT_SEPARATOR_TEMPLATE.format(n=ex_number) + exhibit_text
            )
        if not chunks:
            # All candidate exhibits failed (no usable text). Treat the
            # filing the same as "no EX-99 in index" — body-only stays.
            skipped_no_exhibit.append(f.filing_id)
            continue

        # 3. Build augmented text + content hash.
        augmented = existing_text + "".join(chunks)
        new_hash = content_hash(augmented)

        # 4. Safety — abort filings whose proposals are dangerous to drop.
        proposals = proposals_for_filing(conn, f.filing_id)
        if any(proposal_has_execution(conn, int(p["id"])) for p in proposals):
            aborted_executions.append(f.filing_id)
            continue
        if any(str(p["kind"]) != "no_trade" for p in proposals):
            aborted_non_no_trade.append(f.filing_id)
            continue

        # 5. Plan mutations.
        items.append(
            BackfillPlanItem(
                filing_id=f.filing_id,
                cik=f.cik,
                accession_number=f.accession_number,
                raw_text_path=f.raw_text_path,
                augmented_text=augmented,
                new_content_hash=new_hash,
                proposal_ids=[int(p["id"]) for p in proposals],
                prefilter_decision_ids=prefilter_decisions_for_filing(
                    conn, f.filing_id
                ),
            )
        )

    return BackfillPlan(
        items=items,
        skipped_no_exhibit_in_index=skipped_no_exhibit,
        skipped_already_augmented=skipped_already_augmented,
        fetch_failures=fetch_failures,
        aborted_due_to_executions=aborted_executions,
        aborted_due_to_non_no_trade=aborted_non_no_trade,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_plan(plan: BackfillPlan, *, apply: bool) -> str:
    """Human-readable summary for stdout."""
    lines: list[str] = []
    mode = "APPLY" if apply else "DRY-RUN"
    lines.append(f"=== backfill_8k_exhibits ({mode}) ===")
    lines.append(f"filings to augment:                {len(plan.items)}")
    n_proposals = sum(len(i.proposal_ids) for i in plan.items)
    n_prefilter = sum(len(i.prefilter_decision_ids) for i in plan.items)
    lines.append(f"proposals queued for delete:       {n_proposals}")
    lines.append(f"prefilter rows queued for delete:  {n_prefilter}")
    lines.append(f"skipped (already augmented):       {len(plan.skipped_already_augmented)}")
    lines.append(f"skipped (no EX-99 in index):       {len(plan.skipped_no_exhibit_in_index)}")
    lines.append(f"fetch failures:                    {len(plan.fetch_failures)}")
    lines.append(
        f"aborted (execution exists):        {len(plan.aborted_due_to_executions)}"
    )
    lines.append(
        f"aborted (non-no_trade kind):       {len(plan.aborted_due_to_non_no_trade)}"
    )
    if plan.items:
        lines.append("")
        lines.append("Planned augmentations:")
        for item in plan.items:
            lines.append(
                f"  filing_id={item.filing_id} cik={item.cik} "
                f"accession={item.accession_number} "
                f"(proposals: {item.proposal_ids or '[]'}, "
                f"prefilter: {item.prefilter_decision_ids or '[]'})"
            )
    if plan.fetch_failures:
        lines.append("")
        lines.append("Fetch failures (degraded — left as body-only):")
        for fid, reason in plan.fetch_failures:
            lines.append(f"  filing_id={fid}: {reason}")
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
            "ABORTED — filing has a non-no_trade proposal kind:"
        )
        for fid in plan.aborted_due_to_non_no_trade:
            lines.append(f"  filing_id={fid}  (skip; investigate manually)")
    if not apply and plan.items:
        lines.append("")
        lines.append("(dry-run; rerun with --apply to mutate the database)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Production fetcher — wraps the shared EdgarClient (ADR-002 §6)
# ---------------------------------------------------------------------------


class _EdgarExhibitFetcher:
    """Adapts `EdgarClient` to the `_ExhibitFetcherLike` protocol.

    Fetches index.json, then walks every EX-99.x exhibit in numeric
    order via `EdgarClient.get_bounded` so each exhibit fetch is
    memory-bounded by the **remaining** cumulative budget. Matches
    `detail_fetcher._maybe_fetch_exhibits_99` semantics 1:1 — same cap,
    same numeric ordering, same per-exhibit-failure-skip / cap-overshoot-
    stop posture. A truncated streaming read (server tried to send more
    than the budget) is treated as cap-reached and stops the loop.
    """

    def __init__(self, edgar) -> None:  # type: ignore[no-untyped-def]
        self._edgar = edgar

    async def fetch(
        self, cik: int, accession_number: str
    ) -> tuple[list[dict[str, str]], dict[str, bytes]]:
        acc_nd = accession_number.replace("-", "")
        index_url = f"{_ARCHIVES_BASE}/{cik}/{acc_nd}/index.json"
        resp = await self._edgar.get(index_url)
        if resp.status_code != 200:
            return [], {}
        try:
            payload = resp.json()
        except json.JSONDecodeError:
            return [], {}
        directory = payload.get("directory") if isinstance(payload, dict) else None
        items_raw = directory.get("item") if isinstance(directory, dict) else None
        if not isinstance(items_raw, list):
            return [], {}
        items = [i for i in items_raw if isinstance(i, dict)]
        candidates = _select_all_ex_99(items)
        bodies: dict[str, bytes] = {}
        cumulative_bytes = 0
        for _ex_number, exhibit_name in candidates:
            remaining = _EXHIBIT_CUMULATIVE_BYTE_CAP - cumulative_bytes
            if remaining <= 0:
                break
            ex_url = f"{_ARCHIVES_BASE}/{cik}/{acc_nd}/{exhibit_name}"
            bounded = await self._edgar.get_bounded(ex_url, max_bytes=remaining)
            if bounded.truncated:
                # Body exceeded remaining budget — stop. Later exhibits
                # would also overshoot.
                break
            if bounded.status_code != 200:
                continue
            cumulative_bytes += len(bounded.body)
            bodies[exhibit_name] = bounded.body
        return items, bodies


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backfill EX-99.x content into 8-K filings missing it.",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Mutate the database. Without this flag the script is read-only.",
    )
    p.add_argument(
        "--since",
        type=int,
        default=2,
        help="Look back N days (default 2; covers Mon+Tue hotfix window).",
    )
    p.add_argument(
        "--db",
        default=os.environ.get("QUINN_DB_PATH", _DEFAULT_DB_PATH),
        help=f"SQLite path (default {_DEFAULT_DB_PATH} or $QUINN_DB_PATH).",
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

    since = dt.datetime.now(tz=dt.UTC) - dt.timedelta(days=args.since)
    edgar = EdgarClient(user_agent=args.user_agent)
    fetcher = _EdgarExhibitFetcher(edgar)

    try:
        with sqlite3.connect(args.db, isolation_level=None, timeout=30.0) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            plan = await build_plan(conn, fetcher, since)
            print(render_plan(plan, apply=args.apply))
            if args.apply and plan.items:
                apply_plan(conn, plan)
                n_proposals = sum(len(i.proposal_ids) for i in plan.items)
                n_prefilter = sum(len(i.prefilter_decision_ids) for i in plan.items)
                print(
                    f"\napplied: augmented {len(plan.items)} filings, "
                    f"deleted {n_proposals} proposals, "
                    f"deleted {n_prefilter} prefilter_decisions rows"
                )
                print(
                    "\nNEXT STEP: restart the agent (systemctl restart quinn). "
                    "On boot, `_crash_recovery_scan` will re-queue the "
                    "augmented filings for analysis."
                )
    finally:
        await edgar.aclose()
    return 0


def main() -> int:  # pragma: no cover — thin asyncio shim
    return asyncio.run(_amain())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

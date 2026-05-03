"""One-shot backfill — drop `partial` filings rows left behind by the
prose primary-doc bug (ADR-007).

The 6-hourly submissions-API reconciler (ADR-002 §"Mechanics" item 4)
will re-discover dropped rows on its next pass, this time exercising
the fixed `_select_prose_primary` path.

Default: dry-run (prints what would be deleted, makes no changes).
Pass `--apply` to actually delete.

ADR-007 §"Backfill posture" specifies the operator action as
`DELETE FROM filings WHERE ingest_state='partial' AND ingest_error LIKE
'no primary document found in index%';`. This script wraps that with
two safety guards:

1. **Dry-run by default.** No deletes happen without `--apply`.
2. **Skip rows that already have a `prefilter_decisions` entry.** The
   normal ingestion pipeline (`src/app/loop.py` post-fetch gate) does
   not invoke the prefilter on partials, so this set should be empty
   in practice. The guard is defense-in-depth against pathological
   states (test contamination, manual SQL edits, mid-migration).

Usage on production (DO droplet):

    python3 scripts/drop_partial_filings.py --db /var/lib/quinn/journal.db
    python3 scripts/drop_partial_filings.py --db /var/lib/quinn/journal.db --apply
"""

from __future__ import annotations

import argparse
import sqlite3
import sys

# Default WHERE clause matches the specific bug fixed by ADR-007 — partials
# whose error message indicates the prose primary-doc selector failed. The
# `--all-non-ok` flag broadens to every non-`ok` row without a downstream
# prefilter_decision row, per task #2 spec.
_DEFAULT_WHERE = (
    "ingest_state != 'ok' "
    "AND ingest_error LIKE 'no primary document found in index%'"
)
_BROAD_WHERE = "ingest_state != 'ok'"

_NO_DECISION_CLAUSE = (
    "AND id NOT IN (SELECT filing_id FROM prefilter_decisions)"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        required=True,
        help="path to journal.db (production: /var/lib/quinn/journal.db)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually delete the rows. Default is dry-run (count only).",
    )
    parser.add_argument(
        "--all-non-ok",
        action="store_true",
        help=(
            "broaden the filter to every ingest_state != 'ok' row (still "
            "skipping rows that already have a prefilter_decisions row). "
            "Default is the narrower ADR-007 prose-primary-bug filter."
        ),
    )
    args = parser.parse_args()

    where = _BROAD_WHERE if args.all_non_ok else _DEFAULT_WHERE
    full_where = f"{where} {_NO_DECISION_CLAUSE}"

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        # Count by form_type for the operator's sanity check.
        rows = conn.execute(
            f"SELECT form_type, COUNT(*) AS n FROM filings "
            f"WHERE {full_where} GROUP BY form_type ORDER BY n DESC"
        ).fetchall()
        total = sum(r["n"] for r in rows)
        print(f"Filter: WHERE {full_where}")
        print(f"Rows matching: {total}")
        for r in rows:
            print(f"  {r['form_type']:>10}  {r['n']}")
        if total == 0:
            print("Nothing to delete.")
            return 0
        if not args.apply:
            print("(dry-run; pass --apply to actually delete)")
            return 0
        cur = conn.execute(f"DELETE FROM filings WHERE {full_where}")
        conn.commit()
        print(f"Deleted {cur.rowcount} rows.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())

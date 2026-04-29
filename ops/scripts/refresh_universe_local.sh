#!/usr/bin/env bash
# Local-only convenience: clear any prior snapshot, run the universe refresh,
# print the resulting row. Use this instead of pasting multi-line fish commands.
#
# Run from project root:
#   bash ops/scripts/refresh_universe_local.sh
#
# Requires .env at the project root with all required secrets and QUINN_DB_PATH set.

set -euo pipefail

# Source .env into the bash subshell environment (handles tokens with leading
# dashes, special chars, etc. — does NOT leak into the parent fish shell).
set -a
# shellcheck disable=SC1091
source .env
set +a

DB_PATH="${QUINN_DB_PATH:-$HOME/quinn-data/journal.db}"
echo "=== Using DB: $DB_PATH ==="

mkdir -p "$(dirname "$DB_PATH")"

echo "=== Clearing prior snapshot (if any) ==="
sqlite3 "$DB_PATH" \
  "DELETE FROM universe_members; DELETE FROM universe_snapshots;" \
  2>/dev/null || echo "  (no prior snapshot — fresh DB)"

echo
echo "=== Running universe refresh (30-90s) ==="
.venv/bin/python -m src.jobs.refresh_universe --db "$DB_PATH"
echo "  exit code: $?"

echo
echo "=== Snapshot result ==="
sqlite3 -header -column "$DB_PATH" \
  "SELECT snapshot_date, member_count, yfinance_failures, is_degraded
   FROM universe_snapshots
   ORDER BY snapshot_id DESC
   LIMIT 1;"

echo
echo "=== Sample members (first 10) ==="
sqlite3 -header -column "$DB_PATH" \
  "SELECT ticker, exchange, market_cap, prev_close
   FROM universe_members
   ORDER BY market_cap DESC
   LIMIT 10;"

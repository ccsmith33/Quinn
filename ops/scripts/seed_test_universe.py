"""Local-only seed: populate universe from SEC ∩ Alpaca intersection without yfinance.

Use this when yfinance throttles (residential IP flagged) but you still need a
realistic universe size to test the agent loop locally. Synthesizes a market_cap
+ prev_close per ticker so the universe filter passes; values aren't real but
satisfy the schema and the agent's downstream logic. Production droplet runs
the real refresh job, NOT this.

Run from project root:
    bash -c 'set -a; source .env; set +a; .venv/bin/python ops/scripts/seed_test_universe.py'
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetStatus

# Add src to path so we can import journal helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from journal.repo import (  # noqa: E402
    connect,
    insert_universe_snapshot,
    insert_universe_member,
)
from journal.models import UniverseSnapshotRow, UniverseMemberRow  # noqa: E402
from universe.sec_tickers import fetch_sec_tickers  # noqa: E402
from universe.alpaca_assets import _extract_str_value  # noqa: E402

DB_PATH = os.environ.get("QUINN_DB_PATH") or os.path.expanduser("~/quinn-data/journal.db")
ALLOWED_EXCHANGES = {"NYSE", "NASDAQ", "ARCA", "AMEX"}

# Synthetic mid-range fundamentals — ALL members get these. NOT real data.
SYNTHETIC_MARKET_CAP = 500_000_000.0  # $500M, dead-center of $50M-$2B band
SYNTHETIC_PREV_CLOSE = 10.00          # above the $5 floor


def main() -> int:
    print(f"DB: {DB_PATH}")

    print("Fetching SEC tickers...")
    secs = fetch_sec_tickers(
        user_agent=os.environ.get(
            "EDGAR_USER_AGENT", "Quinn-Local-Seed/1 ccsmith33@crimson.ua.edu"
        )
    )
    sec_by_ticker = {s.ticker.upper(): s for s in secs}
    print(f"  sec tickers: {len(sec_by_ticker)}")

    print("Fetching Alpaca assets...")
    client = TradingClient(
        api_key=os.environ["ALPACA_API_KEY_ID"],
        secret_key=os.environ["ALPACA_API_SECRET_KEY"],
        paper=True,
    )
    assets = client.get_all_assets(filter=GetAssetsRequest(status=AssetStatus.ACTIVE))
    print(f"  alpaca total: {len(assets)}")

    candidates: list[tuple[str, int, str]] = []  # (ticker, cik, exchange)
    for a in assets:
        ticker = a.symbol.upper()
        if ticker not in sec_by_ticker:
            continue
        if _extract_str_value(a.asset_class) != "us_equity":
            continue
        exchange = _extract_str_value(a.exchange)
        if exchange not in ALLOWED_EXCHANGES:
            continue
        if not getattr(a, "tradable", True):
            continue
        candidates.append((ticker, sec_by_ticker[ticker].cik, exchange))

    print(f"  candidates after intersection + class/exchange/tradable filters: {len(candidates)}")

    snapshot_date = dt.date.today()
    sec_payload = json.dumps([s.ticker for s in secs[:100]], sort_keys=True).encode()
    alpaca_payload = json.dumps(sorted(t for t, _, _ in candidates)[:100]).encode()
    sec_hash = hashlib.sha256(sec_payload).hexdigest()[:32]
    alpaca_hash = hashlib.sha256(alpaca_payload).hexdigest()[:32]

    # Wipe any prior snapshot to avoid idempotent-replay
    with connect(DB_PATH) as conn:
        conn.execute("DELETE FROM universe_members")
        conn.execute("DELETE FROM universe_snapshots")

    snapshot_id = insert_universe_snapshot(
        DB_PATH,
        UniverseSnapshotRow(
            snapshot_date=snapshot_date.isoformat(),
            sec_tickers_hash=sec_hash,
            alpaca_assets_hash=alpaca_hash,
            yfinance_failures=0,
            member_count=len(candidates),
            is_degraded=0,
        ),
    )
    print(f"  inserted snapshot id={snapshot_id}")

    for ticker, cik, exchange in candidates:
        insert_universe_member(
            DB_PATH,
            UniverseMemberRow(
                snapshot_id=snapshot_id,
                cik=cik,
                ticker=ticker,
                exchange=exchange,
                market_cap=SYNTHETIC_MARKET_CAP,
                prev_close=SYNTHETIC_PREV_CLOSE,
            ),
        )

    print(f"\nDone. Universe seeded with {len(candidates)} members (synthetic fundamentals).")
    print("Verify:")
    print(f"  sqlite3 {DB_PATH} 'SELECT count(*) FROM universe_members;'")
    return 0


if __name__ == "__main__":
    sys.exit(main())

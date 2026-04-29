"""S2.2 — daily universe refresh job.

Implements ADR-006 mechanics §1–§2 + §4 (degraded-fallback policy):

1. Pull SEC `company_tickers.json` and Alpaca `/v2/assets`.
2. Intersect SEC ↔ Alpaca by ticker. Apply tradable + class=us_equity
   + exchange ∈ {NYSE, NASDAQ, ARCA, AMEX} filters.
3. For each candidate, fetch fundamentals via the `MarketDataProvider` port.
4. Apply market_cap ∈ [$50M, $2B] and prev_close ≥ $5.00 filters.
5. Persist `universe_snapshots` + `universe_members` rows.

Failure handling:
- Per-ticker yfinance failure → exclude (no retry, story-level decision).
- Aggregate failure rate ≥ 5% → mark snapshot `is_degraded=1` and emit
  a structured WARN (`event="universe.degraded"`).
- ≥ 3 prior consecutive degraded days → on day 4, exit non-zero without
  writing; agent loop falls back to most recent non-degraded snapshot.

Idempotency: re-running on the same calendar date with identical source
content (same SEC + Alpaca hashes) returns the existing snapshot id without
inserting a new row. A different content hash on the same date is a
determinism violation and surfaces as a `sqlite3.IntegrityError` from the
UNIQUE(snapshot_date) constraint — callers must investigate.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass

from journal.models import UniverseMemberRow, UniverseSnapshotRow
from journal.repo import (
    connect,
    insert_universe_member,
    insert_universe_snapshot,
)
from observability.log_port import get_logger
from universe.alpaca_assets import AlpacaAsset
from universe.market_data import MarketDataProvider
from universe.sec_tickers import SecTicker
from universe.snapshot import (
    ALLOWED_EXCHANGES,
    MARKET_CAP_CEILING_USD,
    MARKET_CAP_FLOOR_USD,
    PREV_CLOSE_FLOOR_USD,
    hash_alpaca_assets,
    hash_sec_tickers,
)

log = get_logger(__name__)

DEGRADED_FAILURE_RATE = 0.05
CONSECUTIVE_DEGRADED_LIMIT = 3


@dataclass(frozen=True)
class UniverseSources:
    """Bundle of injectable dependencies the refresh job needs.

    `sec_payload_seed` and `alpaca_payload_seed` are test hooks letting tests
    vary source-content hashes without changing logical fixture data; they
    default to "" in production.
    """

    db_path: str
    fetch_sec_tickers: Callable[[], list[SecTicker]]
    fetch_alpaca_assets: Callable[[], list[AlpacaAsset]]
    market_data: MarketDataProvider
    sec_payload_seed: str = ""
    alpaca_payload_seed: str = ""


@dataclass(frozen=True)
class SnapshotResult:
    snapshot_date: dt.date
    snapshot_id: int | None
    member_count: int
    yfinance_failures: int
    is_degraded: bool
    wrote: bool
    reason: str | None = None


def _consecutive_degraded_prior_days(db_path: str, today: dt.date) -> int:
    """Count consecutive `is_degraded=1` days immediately preceding `today`.

    Walks back day-by-day from `today - 1`; stops at the first clean or missing
    day. Days with no snapshot break the streak (the streak is consecutive
    degraded *records*, not gaps).
    """
    streak = 0
    with connect(db_path) as conn:
        cursor_date = today - dt.timedelta(days=1)
        while True:
            row = conn.execute(
                "SELECT is_degraded FROM universe_snapshots WHERE snapshot_date = ?",
                (cursor_date,),
            ).fetchone()
            if row is None or int(row["is_degraded"]) == 0:
                return streak
            streak += 1
            cursor_date -= dt.timedelta(days=1)


def _existing_snapshot_id(
    db_path: str,
    snapshot_date: dt.date,
    sec_hash: str,
    alpaca_hash: str,
) -> int | None:
    """Return the existing snapshot id iff the date+both-hashes match exactly."""
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT snapshot_id, sec_tickers_hash, alpaca_assets_hash "
            "FROM universe_snapshots WHERE snapshot_date = ?",
            (snapshot_date,),
        ).fetchone()
    if row is None:
        return None
    if row["sec_tickers_hash"] != sec_hash or row["alpaca_assets_hash"] != alpaca_hash:
        return None
    return int(row["snapshot_id"])


def run(now: dt.datetime, sources: UniverseSources) -> SnapshotResult:
    """Build, persist, and return today's universe snapshot."""
    snapshot_date = now.date()

    sec_tickers = sources.fetch_sec_tickers()
    alpaca_assets = sources.fetch_alpaca_assets()
    sec_hash = hash_sec_tickers(sec_tickers, seed=sources.sec_payload_seed)
    alpaca_hash = hash_alpaca_assets(alpaca_assets, seed=sources.alpaca_payload_seed)

    existing_id = _existing_snapshot_id(
        sources.db_path, snapshot_date, sec_hash, alpaca_hash
    )
    if existing_id is not None:
        with connect(sources.db_path) as conn:
            row = conn.execute(
                "SELECT member_count, yfinance_failures, is_degraded "
                "FROM universe_snapshots WHERE snapshot_id = ?",
                (existing_id,),
            ).fetchone()
        return SnapshotResult(
            snapshot_date=snapshot_date,
            snapshot_id=existing_id,
            member_count=int(row["member_count"]),
            yfinance_failures=int(row["yfinance_failures"]),
            is_degraded=bool(row["is_degraded"]),
            wrote=False,
            reason="idempotent_replay",
        )

    # Stage 1: intersect SEC ↔ Alpaca, apply tradable/class/exchange filters.
    sec_by_ticker = {t.ticker: t for t in sec_tickers}
    candidates: list[tuple[SecTicker, AlpacaAsset]] = []
    for asset in alpaca_assets:
        if asset.asset_class != "us_equity":
            continue
        if not asset.tradable:
            continue
        if asset.exchange not in ALLOWED_EXCHANGES:
            continue
        sec = sec_by_ticker.get(asset.symbol)
        if sec is None:
            continue
        candidates.append((sec, asset))

    # Stage 2: fetch fundamentals + apply cap/price filters.
    members: list[UniverseMemberRow] = []
    yfinance_failures = 0
    for sec, asset in candidates:
        funds = sources.market_data.fetch_fundamentals(asset.symbol)
        if funds is None:
            yfinance_failures += 1
            continue
        if not (MARKET_CAP_FLOOR_USD <= funds.market_cap <= MARKET_CAP_CEILING_USD):
            continue
        if funds.prev_close < PREV_CLOSE_FLOOR_USD:
            continue
        members.append(
            UniverseMemberRow(
                snapshot_id=0,  # populated after snapshot insert
                cik=sec.cik,
                ticker=sec.ticker,
                exchange=asset.exchange,
                market_cap=funds.market_cap,
                prev_close=funds.prev_close,
            )
        )

    # Degraded determination (per-candidate failure rate; AC-4).
    failure_rate = (
        yfinance_failures / len(candidates) if candidates else 0.0
    )
    is_degraded = failure_rate >= DEGRADED_FAILURE_RATE

    # Consecutive-degraded threshold: only short-circuits when *this* run is
    # also degraded. A clean day breaks the streak and resumes normal writes.
    if is_degraded:
        prior_streak = _consecutive_degraded_prior_days(sources.db_path, snapshot_date)
        if prior_streak >= CONSECUTIVE_DEGRADED_LIMIT:
            log.warning(
                "universe.degraded.threshold_exceeded",
                extra={
                    "event": "universe.degraded.threshold_exceeded",
                    "consecutive_prior_days": prior_streak,
                    "failure_rate": failure_rate,
                    "snapshot_date": snapshot_date.isoformat(),
                },
            )
            return SnapshotResult(
                snapshot_date=snapshot_date,
                snapshot_id=None,
                member_count=0,
                yfinance_failures=yfinance_failures,
                is_degraded=True,
                wrote=False,
                reason="consecutive_degraded_threshold",
            )
        log.warning(
            "universe.degraded",
            extra={
                "event": "universe.degraded",
                "failure_rate": failure_rate,
                "yfinance_failures": yfinance_failures,
                "candidates": len(candidates),
                "snapshot_date": snapshot_date.isoformat(),
            },
        )

    # Persist snapshot + members. UNIQUE(snapshot_date) is the determinism
    # guard: a same-date row with different hashes here surfaces as
    # IntegrityError to the caller.
    snapshot_id = insert_universe_snapshot(
        sources.db_path,
        UniverseSnapshotRow(
            snapshot_date=snapshot_date,
            sec_tickers_hash=sec_hash,
            alpaca_assets_hash=alpaca_hash,
            yfinance_failures=yfinance_failures,
            member_count=len(members),
            is_degraded=1 if is_degraded else 0,
        ),
    )
    for m in members:
        insert_universe_member(
            sources.db_path,
            UniverseMemberRow(
                snapshot_id=snapshot_id,
                cik=m.cik,
                ticker=m.ticker,
                exchange=m.exchange,
                market_cap=m.market_cap,
                prev_close=m.prev_close,
            ),
        )

    return SnapshotResult(
        snapshot_date=snapshot_date,
        snapshot_id=snapshot_id,
        member_count=len(members),
        yfinance_failures=yfinance_failures,
        is_degraded=is_degraded,
        wrote=True,
        reason=None,
    )



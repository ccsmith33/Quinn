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


# ---------------------------------------------------------------------------
# CLI shim — `python -m jobs.refresh_universe` (S2.2 follow-up; runs from
# `ops/systemd/quinn-universe.service` on the daily timer + invoked manually
# during local rehydration when `Universe.load_latest()` raises
# `NoUniverseSnapshot`).
# ---------------------------------------------------------------------------


_DEFAULT_DB_PATH = "/var/lib/quinn/journal.db"


def cli_main(
    *,
    argv: list[str] | None = None,
    sec_fetcher: Callable[[], list[SecTicker]] | None = None,
    alpaca_fetcher: Callable[[], list[AlpacaAsset]] | None = None,
    market_data: MarketDataProvider | None = None,
    clock: Callable[[], dt.datetime] | None = None,
) -> int:
    """Entrypoint for `python -m jobs.refresh_universe`. Returns the
    process exit code.

    Test seam: `sec_fetcher`, `alpaca_fetcher`, `market_data`, and `clock`
    are injectable. Production calls with all four as None, which lazy-
    builds the real SEC/Alpaca/yfinance dependencies from `Secrets` +
    `AppConfig`. The lazy build is required so module import (and the
    test suite) does not require `alpaca-py` or live secrets.

    Exit codes:
      0 — snapshot written, idempotent replay, or any other healthy run
      2 — `run()` returned wrote=False with reason
          `consecutive_degraded_threshold` (the operator-visible failure
          mode: ≥ 3 prior consecutive degraded days + another degraded
          day, per ADR-006 §4)
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="jobs.refresh_universe",
        description="Run the daily universe refresh (SEC + Alpaca + yfinance).",
    )
    parser.add_argument("--db", default=_DEFAULT_DB_PATH)
    parser.add_argument(
        "--edgar-user-agent",
        default=None,
        help="Override the SEC User-Agent (default: read from quinn.toml).",
    )
    args = parser.parse_args(argv)

    now = (clock or (lambda: dt.datetime.now(dt.UTC)))()
    sec_call, alpaca_call, md = _resolve_sources(
        sec_fetcher=sec_fetcher,
        alpaca_fetcher=alpaca_fetcher,
        market_data=market_data,
        edgar_user_agent_override=args.edgar_user_agent,
    )
    sources = UniverseSources(
        db_path=args.db,
        fetch_sec_tickers=sec_call,
        fetch_alpaca_assets=alpaca_call,
        market_data=md,
    )

    result = run(now=now, sources=sources)
    if not result.wrote and result.reason == "consecutive_degraded_threshold":
        log.error(
            "universe.cli_failed",
            extra={
                "event": "universe.cli_failed",
                "reason": result.reason,
                "yfinance_failures": result.yfinance_failures,
            },
        )
        return 2
    return 0


def _resolve_sources(
    *,
    sec_fetcher: Callable[[], list[SecTicker]] | None,
    alpaca_fetcher: Callable[[], list[AlpacaAsset]] | None,
    market_data: MarketDataProvider | None,
    edgar_user_agent_override: str | None,
) -> tuple[
    Callable[[], list[SecTicker]],
    Callable[[], list[AlpacaAsset]],
    MarketDataProvider,
]:
    """Return (sec_call, alpaca_call, market_data). Test path injects all
    three; production path lazy-builds them from `Secrets` + `AppConfig`.

    `alpaca-py`, `load_secrets`, `load_config`, and `YFinanceProvider`
    are imported inside the production branch so module import stays
    cheap and test runs need neither secrets nor `alpaca-py`."""
    if (
        sec_fetcher is not None
        and alpaca_fetcher is not None
        and market_data is not None
    ):
        return sec_fetcher, alpaca_fetcher, market_data

    from config.loader import load_config
    from config.secrets import load_secrets
    from universe.alpaca_assets import fetch_alpaca_assets as alpaca_fetch
    from universe.sec_tickers import fetch_sec_tickers as sec_fetch
    from universe.yfinance_provider import YFinanceProvider

    secrets = load_secrets()
    cfg = load_config()
    user_agent = edgar_user_agent_override or cfg.ingestion.edgar_user_agent

    real_sec = sec_fetcher or (lambda: sec_fetch(user_agent))
    real_alpaca = alpaca_fetcher or (
        lambda: alpaca_fetch(_build_alpaca_client(secrets, cfg.execution.broker_mode))
    )
    real_md = market_data or YFinanceProvider(
        min_interval_seconds=cfg.universe.yfinance_min_interval_seconds,
    )
    return real_sec, real_alpaca, real_md


def _build_alpaca_client(secrets, broker_mode: str) -> object:  # noqa: ANN001 - Secrets type lazy
    """Lazy-import `alpaca-py` and build a `TradingClient` from secrets.

    Imported inside the function so the test suite (and module import)
    does not require `alpaca-py` to be installed. Mirrors the lazy
    pattern used by `jobs.backup._build_b2sdk_uploader` and the
    construction-time-only mode plumbing in `app.composition`
    (D-007: mode is set at wiring, never branched on at runtime)."""
    from alpaca.trading.client import TradingClient  # type: ignore[import-not-found]

    return TradingClient(
        api_key=secrets.alpaca_api_key_id.get_secret_value(),
        secret_key=secrets.alpaca_api_secret_key.get_secret_value(),
        paper=(broker_mode == "paper"),
    )


if __name__ == "__main__":  # pragma: no cover - exercised via cli_main tests
    import sys

    sys.exit(cli_main())

"""S2.2 — daily universe refresh job tests (TDD).

Architecture references: §2.10, ADR-006, FR-7..FR-10, NFR-16.
Story: artifacts/implementation/stories/story-02-02-universe-refresh-job.md.
"""

from __future__ import annotations

import datetime as dt
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

from jobs.refresh_universe import SnapshotResult, UniverseSources, run
from journal.migrate import apply_migrations
from journal.models import UniverseSnapshotRow
from journal.repo import (
    connect,
    get_current_universe_snapshot,
    insert_universe_snapshot,
)
from universe.alpaca_assets import AlpacaAsset
from universe.market_data import Fundamentals
from universe.sec_tickers import SecTicker

# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


@dataclass
class _Fixture:
    """Encapsulates ticker/exchange/marketcap/prevclose to drive all three sources."""

    cik: int
    ticker: str
    exchange: str
    tradable: bool
    asset_class: str
    market_cap: float | None  # None means yfinance fails for this ticker
    prev_close: float | None


def _make_sources(
    db_path: str,
    fixtures: list[_Fixture],
    *,
    sec_seed: str = "sec-default",
    alpaca_seed: str = "alpaca-default",
    failures: set[str] | None = None,
) -> UniverseSources:
    """Build UniverseSources from a list of fixtures.

    `failures` overrides per-ticker yfinance failures (raises). Otherwise a
    fixture with market_cap=None or prev_close=None implies the provider returns None.
    `sec_seed`/`alpaca_seed` allow tests to vary the source-content hashes
    without changing membership.
    """
    failures = failures or set()
    sec_tickers = [
        SecTicker(cik=f.cik, ticker=f.ticker, name=f"{f.ticker} Inc")
        for f in fixtures
    ]
    alpaca_assets = [
        AlpacaAsset(
            symbol=f.ticker,
            exchange=f.exchange,
            status="active",
            tradable=f.tradable,
            fractionable=True,
            asset_class=f.asset_class,
        )
        for f in fixtures
    ]
    fundamentals: dict[str, Fundamentals | None] = {}
    for f in fixtures:
        if f.market_cap is None or f.prev_close is None:
            fundamentals[f.ticker] = None
        else:
            fundamentals[f.ticker] = Fundamentals(
                market_cap=f.market_cap,
                prev_close=f.prev_close,
                fetched_at=dt.datetime(2026, 4, 28, 7, 0, tzinfo=dt.UTC),
            )

    class _Provider:
        def fetch_fundamentals(self, ticker: str) -> Fundamentals | None:
            if ticker in failures:
                return None
            return fundamentals.get(ticker)

    # The seed strings are mixed into the source payloads so different test
    # configurations produce different hashes; same fixtures with same seed
    # produce identical hashes (idempotency).
    def fetch_sec() -> list[SecTicker]:
        return list(sec_tickers)

    def fetch_alpaca() -> list[AlpacaAsset]:
        return list(alpaca_assets)

    return UniverseSources(
        db_path=db_path,
        fetch_sec_tickers=fetch_sec,
        fetch_alpaca_assets=fetch_alpaca,
        market_data=_Provider(),
        sec_payload_seed=sec_seed,
        alpaca_payload_seed=alpaca_seed,
    )


@pytest.fixture
def db(tmp_path: Path) -> str:
    db_path = tmp_path / "journal.db"
    apply_migrations(str(db_path))
    return str(db_path)


@pytest.fixture
def now() -> dt.datetime:
    return dt.datetime(2026, 4, 28, 11, 0, 0, tzinfo=dt.UTC)


# ---------------------------------------------------------------------------
# AC-2: filter pipeline (ADR-006 §"Test scenarios")
# ---------------------------------------------------------------------------


def _baseline_fixtures() -> list[_Fixture]:
    """A small mixed set covering each filter-rule edge."""
    return [
        # included: meets every constraint
        _Fixture(1, "ACME", "NASDAQ", True, "us_equity", 500_000_000.0, 12.34),
        _Fixture(2, "AAPL", "NASDAQ", True, "us_equity", 1_900_000_000.0, 150.0),
        _Fixture(3, "NYSEN", "NYSE", True, "us_equity", 100_000_000.0, 25.0),
        _Fixture(4, "ARCAN", "ARCA", True, "us_equity", 250_000_000.0, 30.0),
        _Fixture(5, "AMEXN", "AMEX", True, "us_equity", 60_000_000.0, 6.0),
        # excluded: market cap below $50M
        _Fixture(10, "TINY", "NASDAQ", True, "us_equity", 49_000_000.0, 10.0),
        # excluded: market cap above $2B
        _Fixture(11, "BIG", "NASDAQ", True, "us_equity", 2_500_000_000.0, 100.0),
        # excluded: prev close below $5
        _Fixture(12, "PENNY", "NASDAQ", True, "us_equity", 200_000_000.0, 4.99),
        # excluded: OTC exchange
        _Fixture(13, "OTC1", "OTC", True, "us_equity", 200_000_000.0, 12.0),
        # excluded: not tradable
        _Fixture(14, "NTRD", "NASDAQ", False, "us_equity", 200_000_000.0, 12.0),
        # excluded: wrong asset class
        _Fixture(15, "CRYPTO", "NASDAQ", True, "crypto", 200_000_000.0, 12.0),
        # boundary-included: $50M cap exactly
        _Fixture(16, "EDGEMC", "NASDAQ", True, "us_equity", 50_000_000.0, 7.0),
        # boundary-included: $5.00 prev close exactly
        _Fixture(17, "EDGEPX", "NASDAQ", True, "us_equity", 80_000_000.0, 5.00),
    ]


def test_refresh_produces_snapshot_with_filters_applied(db: str, now: dt.datetime) -> None:
    sources = _make_sources(db, _baseline_fixtures())
    result = run(now, sources)

    assert isinstance(result, SnapshotResult)
    assert result.wrote is True
    assert result.is_degraded is False
    # included: ACME, AAPL, NYSEN, ARCAN, AMEXN, EDGEMC, EDGEPX
    assert result.member_count == 7
    assert result.snapshot_date == dt.date(2026, 4, 28)

    snap = get_current_universe_snapshot(db)
    assert snap is not None
    assert snap.member_count == 7
    assert snap.is_degraded == 0

    with connect(db) as conn:
        rows = conn.execute(
            "SELECT ticker FROM universe_members WHERE snapshot_id = ? ORDER BY ticker",
            (snap.snapshot_id,),
        ).fetchall()
    tickers = {r["ticker"] for r in rows}
    assert tickers == {"ACME", "AAPL", "NYSEN", "ARCAN", "AMEXN", "EDGEMC", "EDGEPX"}


def test_market_cap_boundary_50m_included_49m_excluded(db: str, now: dt.datetime) -> None:
    fixtures = [
        _Fixture(1, "INC", "NASDAQ", True, "us_equity", 50_000_000.0, 7.0),
        _Fixture(2, "EXC", "NASDAQ", True, "us_equity", 49_000_000.0, 7.0),
        _Fixture(3, "INCB", "NASDAQ", True, "us_equity", 51_000_000.0, 7.0),
    ]
    run(now, _make_sources(db, fixtures))
    with connect(db) as conn:
        tickers = {
            r["ticker"]
            for r in conn.execute("SELECT ticker FROM universe_members").fetchall()
        }
    assert tickers == {"INC", "INCB"}


def test_otc_excluded_by_exchange_filter(db: str, now: dt.datetime) -> None:
    fixtures = [
        _Fixture(1, "OTCONLY", "OTC", True, "us_equity", 200_000_000.0, 10.0),
        _Fixture(2, "OK", "NASDAQ", True, "us_equity", 200_000_000.0, 10.0),
    ]
    run(now, _make_sources(db, fixtures))
    with connect(db) as conn:
        tickers = {
            r["ticker"]
            for r in conn.execute("SELECT ticker FROM universe_members").fetchall()
        }
    assert tickers == {"OK"}


def test_prev_close_below_floor_excluded(db: str, now: dt.datetime) -> None:
    fixtures = [
        _Fixture(1, "LOW", "NASDAQ", True, "us_equity", 100_000_000.0, 4.99),
        _Fixture(2, "OK", "NASDAQ", True, "us_equity", 100_000_000.0, 5.00),
    ]
    run(now, _make_sources(db, fixtures))
    with connect(db) as conn:
        tickers = {
            r["ticker"]
            for r in conn.execute("SELECT ticker FROM universe_members").fetchall()
        }
    assert tickers == {"OK"}


# ---------------------------------------------------------------------------
# AC-3: snapshot persistence (hashes + counts + append-only)
# ---------------------------------------------------------------------------


def test_snapshot_records_source_hashes_and_counts(db: str, now: dt.datetime) -> None:
    sources = _make_sources(db, _baseline_fixtures())
    result = run(now, sources)
    snap = get_current_universe_snapshot(db)
    assert snap is not None
    # hashes are non-empty hex strings (64 chars for sha256)
    assert isinstance(snap.sec_tickers_hash, str) and len(snap.sec_tickers_hash) == 64
    assert isinstance(snap.alpaca_assets_hash, str) and len(snap.alpaca_assets_hash) == 64
    assert snap.member_count == result.member_count
    assert snap.yfinance_failures == 0


# ---------------------------------------------------------------------------
# AC-4: yfinance failure handling
# ---------------------------------------------------------------------------


def test_low_yfinance_failure_excludes_but_not_degraded(
    db: str, now: dt.datetime, caplog: pytest.LogCaptureFixture
) -> None:
    """1% failure rate: tickers excluded, no degraded flag, no operator alert."""
    fixtures = [
        _Fixture(i, f"T{i:03d}", "NASDAQ", True, "us_equity", 100_000_000.0, 10.0)
        for i in range(100)
    ]
    fixtures[0] = _Fixture(0, "T000", "NASDAQ", True, "us_equity", None, None)
    sources = _make_sources(db, fixtures)
    with caplog.at_level(logging.WARNING):
        result = run(now, sources)
    assert result.is_degraded is False
    assert result.member_count == 99
    assert result.yfinance_failures == 1
    assert "universe.degraded" not in caplog.text


def test_high_yfinance_failure_marks_degraded_and_alerts(
    db: str, now: dt.datetime, caplog: pytest.LogCaptureFixture
) -> None:
    """10% failure rate: alert + is_degraded=1."""
    fixtures = [
        _Fixture(i, f"T{i:03d}", "NASDAQ", True, "us_equity", 100_000_000.0, 10.0)
        for i in range(100)
    ]
    for i in range(10):
        fixtures[i] = _Fixture(i, f"T{i:03d}", "NASDAQ", True, "us_equity", None, None)
    sources = _make_sources(db, fixtures)
    with caplog.at_level(logging.WARNING):
        result = run(now, sources)
    assert result.is_degraded is True
    assert result.yfinance_failures == 10
    assert "universe.degraded" in caplog.text
    snap = get_current_universe_snapshot(db)
    assert snap is not None
    assert snap.is_degraded == 1


def test_exactly_5pct_failure_rate_marks_degraded(db: str, now: dt.datetime) -> None:
    """ADR-006 says >5% triggers alert; story AC-4 says >=5% — story is authoritative."""
    fixtures = [
        _Fixture(i, f"T{i:03d}", "NASDAQ", True, "us_equity", 100_000_000.0, 10.0)
        for i in range(100)
    ]
    for i in range(5):
        fixtures[i] = _Fixture(i, f"T{i:03d}", "NASDAQ", True, "us_equity", None, None)
    sources = _make_sources(db, fixtures)
    result = run(now, sources)
    assert result.is_degraded is True


# ---------------------------------------------------------------------------
# AC-1: idempotency (FR-10 determinism)
# ---------------------------------------------------------------------------


def test_idempotent_on_same_date_and_sources(db: str, now: dt.datetime) -> None:
    sources = _make_sources(db, _baseline_fixtures())
    first = run(now, sources)
    second = run(now, sources)
    assert first.snapshot_id is not None
    assert second.snapshot_id == first.snapshot_id
    assert second.wrote is False  # no new write on re-run
    with connect(db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM universe_snapshots").fetchone()[0]
        members = conn.execute("SELECT COUNT(*) FROM universe_members").fetchone()[0]
    assert count == 1
    assert members == first.member_count


def test_same_date_different_sources_raises(db: str, now: dt.datetime) -> None:
    """UNIQUE(snapshot_date) enforces one row per day; if hashes differ on rerun
    same day, the job must signal a determinism violation (not silently overwrite)."""
    run(now, _make_sources(db, _baseline_fixtures(), sec_seed="seed-A"))
    with pytest.raises(sqlite3.IntegrityError):
        run(now, _make_sources(db, _baseline_fixtures(), sec_seed="seed-B"))


# ---------------------------------------------------------------------------
# AC-4: consecutive-degraded threshold
# ---------------------------------------------------------------------------


def _seed_degraded_history(db: str, today: dt.date, n: int) -> None:
    """Seed `n` consecutive prior days of is_degraded=1 snapshots."""
    for offset in range(1, n + 1):
        d = today - dt.timedelta(days=offset)
        insert_universe_snapshot(
            db,
            UniverseSnapshotRow(
                snapshot_date=d,
                sec_tickers_hash=f"sec-{offset}",
                alpaca_assets_hash=f"alp-{offset}",
                yfinance_failures=10,
                member_count=0,
                is_degraded=1,
            ),
        )


def test_three_consecutive_degraded_then_high_failure_no_write(
    db: str, now: dt.datetime
) -> None:
    """Day 4 with another high failure rate exits without writing."""
    today = now.date()
    _seed_degraded_history(db, today, n=3)
    fixtures = [
        _Fixture(i, f"T{i:03d}", "NASDAQ", True, "us_equity", 100_000_000.0, 10.0)
        for i in range(100)
    ]
    for i in range(10):  # 10% failure → degraded
        fixtures[i] = _Fixture(i, f"T{i:03d}", "NASDAQ", True, "us_equity", None, None)
    result = run(now, _make_sources(db, fixtures))
    assert result.wrote is False
    assert result.reason == "consecutive_degraded_threshold"
    # snapshot row for today is NOT created
    with connect(db) as conn:
        n_today = conn.execute(
            "SELECT COUNT(*) FROM universe_snapshots WHERE snapshot_date = ?",
            (today,),
        ).fetchone()[0]
    assert n_today == 0


def test_three_consecutive_degraded_then_clean_writes_normally(
    db: str, now: dt.datetime
) -> None:
    """Day 4 with healthy data resets — snapshot is written, is_degraded=0."""
    today = now.date()
    _seed_degraded_history(db, today, n=3)
    fixtures = [
        _Fixture(i, f"T{i:03d}", "NASDAQ", True, "us_equity", 100_000_000.0, 10.0)
        for i in range(100)
    ]
    result = run(now, _make_sources(db, fixtures))
    assert result.wrote is True
    assert result.is_degraded is False


def test_two_consecutive_degraded_then_high_failure_still_writes(
    db: str, now: dt.datetime
) -> None:
    """Threshold is 3 consecutive prior days; on day 3 we still write (degraded)."""
    today = now.date()
    _seed_degraded_history(db, today, n=2)
    fixtures = [
        _Fixture(i, f"T{i:03d}", "NASDAQ", True, "us_equity", 100_000_000.0, 10.0)
        for i in range(100)
    ]
    for i in range(10):
        fixtures[i] = _Fixture(i, f"T{i:03d}", "NASDAQ", True, "us_equity", None, None)
    result = run(now, _make_sources(db, fixtures))
    assert result.wrote is True
    assert result.is_degraded is True


def test_non_consecutive_degraded_does_not_trigger_threshold(
    db: str, now: dt.datetime
) -> None:
    """A clean day in the middle resets the streak."""
    today = now.date()
    # Days T-1, T-2 degraded; T-3 clean; T-4 degraded → streak is 2, not 3
    insert_universe_snapshot(
        db,
        UniverseSnapshotRow(
            snapshot_date=today - dt.timedelta(days=1),
            sec_tickers_hash="h1",
            alpaca_assets_hash="h1",
            yfinance_failures=10,
            member_count=0,
            is_degraded=1,
        ),
    )
    insert_universe_snapshot(
        db,
        UniverseSnapshotRow(
            snapshot_date=today - dt.timedelta(days=2),
            sec_tickers_hash="h2",
            alpaca_assets_hash="h2",
            yfinance_failures=10,
            member_count=0,
            is_degraded=1,
        ),
    )
    insert_universe_snapshot(
        db,
        UniverseSnapshotRow(
            snapshot_date=today - dt.timedelta(days=3),
            sec_tickers_hash="h3",
            alpaca_assets_hash="h3",
            yfinance_failures=0,
            member_count=10,
            is_degraded=0,
        ),
    )
    insert_universe_snapshot(
        db,
        UniverseSnapshotRow(
            snapshot_date=today - dt.timedelta(days=4),
            sec_tickers_hash="h4",
            alpaca_assets_hash="h4",
            yfinance_failures=10,
            member_count=0,
            is_degraded=1,
        ),
    )
    fixtures = [
        _Fixture(i, f"T{i:03d}", "NASDAQ", True, "us_equity", 100_000_000.0, 10.0)
        for i in range(100)
    ]
    for i in range(10):
        fixtures[i] = _Fixture(i, f"T{i:03d}", "NASDAQ", True, "us_equity", None, None)
    result = run(now, _make_sources(db, fixtures))
    assert result.wrote is True  # streak only 2, not 3


# ---------------------------------------------------------------------------
# AC-5: scale (representative-size integration)
# ---------------------------------------------------------------------------


def test_handles_large_fixture_set(db: str, now: dt.datetime) -> None:
    """≥1,000 fixture issuers; mix of pass and fail filter rules."""
    fixtures: list[_Fixture] = []
    # 800 pass-all
    for i in range(800):
        fixtures.append(
            _Fixture(i, f"P{i:04d}", "NASDAQ", True, "us_equity", 100_000_000.0, 10.0)
        )
    # 100 cap too small
    for i in range(800, 900):
        fixtures.append(
            _Fixture(i, f"S{i:04d}", "NASDAQ", True, "us_equity", 10_000_000.0, 10.0)
        )
    # 100 OTC
    for i in range(900, 1000):
        fixtures.append(
            _Fixture(i, f"O{i:04d}", "OTC", True, "us_equity", 100_000_000.0, 10.0)
        )
    # 100 prev_close below 5
    for i in range(1000, 1100):
        fixtures.append(
            _Fixture(i, f"L{i:04d}", "NASDAQ", True, "us_equity", 100_000_000.0, 4.50)
        )

    result = run(now, _make_sources(db, fixtures))
    assert result.member_count == 800
    assert result.is_degraded is False

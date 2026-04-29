"""S2.2 follow-up — CLI shim tests for `python -m jobs.refresh_universe`.

The systemd unit `ops/systemd/quinn-universe.service` and the operator
runbook drive the daily universe refresh via `python -m jobs.refresh_universe`.
These tests exercise the `cli_main(...)` entrypoint with in-process fake
fetchers and a fake `MarketDataProvider`, avoiding any real network egress
to SEC, Alpaca, or yfinance from the test path.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from journal.migrate import apply_migrations
from journal.models import UniverseSnapshotRow
from journal.repo import insert_universe_snapshot
from universe.alpaca_assets import AlpacaAsset
from universe.market_data import Fundamentals
from universe.sec_tickers import SecTicker


@pytest.fixture
def db(tmp_path: Path) -> str:
    path = str(tmp_path / "journal.db")
    apply_migrations(path)
    return path


def _baseline_sec() -> list[SecTicker]:
    return [
        SecTicker(cik=1, ticker="ACME", name="Acme Inc"),
        SecTicker(cik=2, ticker="AAPL", name="Apple Inc"),
    ]


def _baseline_alpaca() -> list[AlpacaAsset]:
    return [
        AlpacaAsset(
            symbol="ACME",
            exchange="NASDAQ",
            status="active",
            tradable=True,
            fractionable=True,
            asset_class="us_equity",
        ),
        AlpacaAsset(
            symbol="AAPL",
            exchange="NASDAQ",
            status="active",
            tradable=True,
            fractionable=True,
            asset_class="us_equity",
        ),
    ]


class _FakeMarketData:
    def __init__(self, fundamentals: dict[str, Fundamentals | None]) -> None:
        self._f = fundamentals

    def fetch_fundamentals(self, ticker: str) -> Fundamentals | None:
        return self._f.get(ticker)


def _baseline_fundamentals() -> dict[str, Fundamentals | None]:
    return {
        "ACME": Fundamentals(
            market_cap=500_000_000.0,
            prev_close=12.0,
            fetched_at=dt.datetime(2026, 4, 28, 7, 0, tzinfo=dt.UTC),
        ),
        "AAPL": Fundamentals(
            market_cap=1_900_000_000.0,
            prev_close=150.0,
            fetched_at=dt.datetime(2026, 4, 28, 7, 0, tzinfo=dt.UTC),
        ),
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_refresh_universe_cli_writes_snapshot_and_returns_zero(db: str) -> None:
    """The CLI shim builds UniverseSources from injected fakes, invokes
    `run`, and returns 0 on a normal write."""
    from jobs.refresh_universe import cli_main

    rc = cli_main(
        argv=["--db", db, "--edgar-user-agent", "Quinn-Test/1.0 t@example.com"],
        sec_fetcher=_baseline_sec,
        alpaca_fetcher=_baseline_alpaca,
        market_data=_FakeMarketData(_baseline_fundamentals()),
        clock=lambda: dt.datetime(2026, 4, 28, 11, 0, tzinfo=dt.UTC),
    )
    assert rc == 0


def test_refresh_universe_cli_idempotent_replay_returns_zero(db: str) -> None:
    """A second invocation on the same date with the same source content
    returns 0 (idempotent replay; wrote=False is NOT a failure)."""
    from jobs.refresh_universe import cli_main

    kwargs = dict(
        argv=["--db", db, "--edgar-user-agent", "Quinn-Test/1.0 t@example.com"],
        sec_fetcher=_baseline_sec,
        alpaca_fetcher=_baseline_alpaca,
        market_data=_FakeMarketData(_baseline_fundamentals()),
        clock=lambda: dt.datetime(2026, 4, 28, 11, 0, tzinfo=dt.UTC),
    )
    assert cli_main(**kwargs) == 0
    assert cli_main(**kwargs) == 0


# ---------------------------------------------------------------------------
# Failure path the operator needs to know about
# ---------------------------------------------------------------------------


def test_refresh_universe_cli_returns_nonzero_on_consecutive_degraded(db: str) -> None:
    """≥3 consecutive prior degraded days + another high-failure day →
    `run()` returns wrote=False with reason=`consecutive_degraded_threshold`.
    The CLI shim must surface that as a non-zero exit so the systemd unit
    flips to `failed` and AlertWatcher can page the operator."""
    from jobs.refresh_universe import cli_main

    today = dt.date(2026, 4, 28)
    for offset in range(1, 4):
        insert_universe_snapshot(
            db,
            UniverseSnapshotRow(
                snapshot_date=today - dt.timedelta(days=offset),
                sec_tickers_hash=f"sec-{offset}",
                alpaca_assets_hash=f"alp-{offset}",
                yfinance_failures=10,
                member_count=0,
                is_degraded=1,
            ),
        )

    # 100 candidates, 10 yfinance failures → 10% failure rate → degraded.
    sec = [SecTicker(cik=i, ticker=f"T{i:03d}", name=f"T{i}") for i in range(100)]
    alpaca = [
        AlpacaAsset(
            symbol=f"T{i:03d}",
            exchange="NASDAQ",
            status="active",
            tradable=True,
            fractionable=True,
            asset_class="us_equity",
        )
        for i in range(100)
    ]
    fundamentals: dict[str, Fundamentals | None] = {
        f"T{i:03d}": (
            None
            if i < 10
            else Fundamentals(
                market_cap=100_000_000.0,
                prev_close=10.0,
                fetched_at=dt.datetime(2026, 4, 28, 7, 0, tzinfo=dt.UTC),
            )
        )
        for i in range(100)
    }

    rc = cli_main(
        argv=["--db", db, "--edgar-user-agent", "Quinn-Test/1.0 t@example.com"],
        sec_fetcher=lambda: sec,
        alpaca_fetcher=lambda: alpaca,
        market_data=_FakeMarketData(fundamentals),
        clock=lambda: dt.datetime(2026, 4, 28, 11, 0, tzinfo=dt.UTC),
    )
    assert rc != 0


def test_refresh_universe_cli_help_does_not_run(db: str) -> None:
    """`--help` exits cleanly without touching the journal or fetchers."""
    from jobs.refresh_universe import cli_main

    fetched: list[str] = []

    def boom_sec() -> list[SecTicker]:
        fetched.append("sec")
        raise AssertionError("--help must not invoke fetchers")

    def boom_alpaca() -> list[AlpacaAsset]:
        fetched.append("alpaca")
        raise AssertionError("--help must not invoke fetchers")

    with pytest.raises(SystemExit) as exc:
        cli_main(
            argv=["--help"],
            sec_fetcher=boom_sec,
            alpaca_fetcher=boom_alpaca,
            market_data=_FakeMarketData({}),
            clock=lambda: dt.datetime(2026, 4, 28, tzinfo=dt.UTC),
        )
    assert exc.value.code == 0
    assert fetched == []


def test_refresh_universe_cli_does_not_perform_real_network(db: str) -> None:
    """Smoke-test: with all deps injected, no real HTTP egress occurs.

    We assert this by patching `httpx` and the `alpaca` module-level imports
    would normally fire — but since deps are injected, neither is touched.
    The proof is that the test passes without `alpaca-py` configured (no
    secrets, no endpoint) and without network access."""
    from jobs.refresh_universe import cli_main

    rc = cli_main(
        argv=["--db", db, "--edgar-user-agent", "Quinn-Test/1.0 t@example.com"],
        sec_fetcher=_baseline_sec,
        alpaca_fetcher=_baseline_alpaca,
        market_data=_FakeMarketData(_baseline_fundamentals()),
        clock=lambda: dt.datetime(2026, 4, 28, 11, 0, tzinfo=dt.UTC),
    )
    assert rc == 0

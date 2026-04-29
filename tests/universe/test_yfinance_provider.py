"""S2.1 — yfinance MarketDataProvider implementation."""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from universe.market_data import Fundamentals, MarketDataProvider
from universe.yfinance_provider import YFinanceProvider


class _FakeTicker:
    def __init__(self, info: dict[str, Any]) -> None:
        self.info = info


def _provider_with_info(info: dict[str, Any]) -> YFinanceProvider:
    return YFinanceProvider(
        ticker_factory=lambda symbol: _FakeTicker(info),
        min_interval_seconds=0.0,
    )


def test_returns_fundamentals_when_fields_present() -> None:
    p = _provider_with_info({"marketCap": 500_000_000, "previousClose": 12.34})
    result = p.fetch_fundamentals("ACME")
    assert isinstance(result, Fundamentals)
    assert result.market_cap == 500_000_000.0
    assert result.prev_close == 12.34
    assert isinstance(result.fetched_at, dt.datetime)


def test_returns_none_on_missing_marketcap() -> None:
    p = _provider_with_info({"previousClose": 12.34})
    assert p.fetch_fundamentals("ACME") is None


def test_returns_none_on_missing_prev_close() -> None:
    p = _provider_with_info({"marketCap": 1_000_000})
    assert p.fetch_fundamentals("ACME") is None


def test_returns_none_on_factory_exception() -> None:
    def factory(_symbol: str) -> _FakeTicker:
        raise RuntimeError("scrape failed")

    p = YFinanceProvider(ticker_factory=factory, min_interval_seconds=0.0)
    assert p.fetch_fundamentals("ACME") is None


def test_returns_none_on_empty_info() -> None:
    p = _provider_with_info({})
    assert p.fetch_fundamentals("ACME") is None


def test_satisfies_protocol() -> None:
    """AC-3: YFinanceProvider implements MarketDataProvider."""
    p: MarketDataProvider = _provider_with_info({"marketCap": 1.0, "previousClose": 1.0})
    assert p.fetch_fundamentals("X") is not None


def test_min_interval_enforced() -> None:
    """D-065 — local rate-limiter spaces calls to honour `min_interval_seconds`.

    Two back-to-back calls with a fake monotonic clock advancing 50ms between
    them must invoke `sleep(0.150)` before the second call (200ms target -
    50ms elapsed). The first call sleeps 0 because there is no prior call to
    space against.
    """
    clock = iter([1000.0, 1000.05])
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    info = {"marketCap": 1.0, "previousClose": 1.0}
    p = YFinanceProvider(
        ticker_factory=lambda symbol: _FakeTicker(info),
        min_interval_seconds=0.2,
        monotonic=lambda: next(clock),
        sleep=fake_sleep,
    )

    p.fetch_fundamentals("A")
    p.fetch_fundamentals("B")

    assert len(sleeps) == 2
    assert sleeps[0] == 0.0
    assert sleeps[1] == pytest.approx(0.15, abs=1e-9)


def test_min_interval_zero_never_sleeps() -> None:
    """`min_interval_seconds=0` (the test-default) must skip the sleep path
    entirely so the test suite runs at full speed."""
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    info = {"marketCap": 1.0, "previousClose": 1.0}
    p = YFinanceProvider(
        ticker_factory=lambda symbol: _FakeTicker(info),
        min_interval_seconds=0.0,
        sleep=fake_sleep,
    )
    p.fetch_fundamentals("A")
    p.fetch_fundamentals("B")
    assert sleeps == []

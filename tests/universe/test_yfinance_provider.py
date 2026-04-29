"""S2.1 — yfinance MarketDataProvider implementation."""

from __future__ import annotations

import datetime as dt
from typing import Any

from universe.market_data import Fundamentals, MarketDataProvider
from universe.yfinance_provider import YFinanceProvider


class _FakeTicker:
    def __init__(self, info: dict[str, Any]) -> None:
        self.info = info


def _provider_with_info(info: dict[str, Any]) -> YFinanceProvider:
    return YFinanceProvider(ticker_factory=lambda symbol: _FakeTicker(info))


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

    p = YFinanceProvider(ticker_factory=factory)
    assert p.fetch_fundamentals("ACME") is None


def test_returns_none_on_empty_info() -> None:
    p = _provider_with_info({})
    assert p.fetch_fundamentals("ACME") is None


def test_satisfies_protocol() -> None:
    """AC-3: YFinanceProvider implements MarketDataProvider."""
    p: MarketDataProvider = _provider_with_info({"marketCap": 1.0, "previousClose": 1.0})
    assert p.fetch_fundamentals("X") is not None

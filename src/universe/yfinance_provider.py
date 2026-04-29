"""yfinance implementation of MarketDataProvider.

Egress: `query1.finance.yahoo.com` and `query2.finance.yahoo.com`
(architecture §9.5). Per ADR-006 / D-006, no other module imports yfinance
directly — substitution is local to this file.

On any scrape failure or missing required field returns `None`; callers
(S2.2's snapshot composer) decide whether the absence excludes the ticker.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from typing import Any

from observability.log_port import get_logger

from .market_data import Fundamentals

log = get_logger(__name__)

_TickerFactory = Callable[[str], Any]


def _default_ticker_factory(symbol: str) -> Any:
    import yfinance  # local import per ADR-006

    return yfinance.Ticker(symbol)


class YFinanceProvider:
    """Fetches fundamentals via yfinance. Implements `MarketDataProvider`."""

    def __init__(self, ticker_factory: _TickerFactory | None = None) -> None:
        self._ticker_factory = ticker_factory or _default_ticker_factory

    def fetch_fundamentals(self, ticker: str) -> Fundamentals | None:
        try:
            t = self._ticker_factory(ticker)
            info = getattr(t, "info", None)
            if not isinstance(info, dict):
                return None
            market_cap = info.get("marketCap")
            prev_close = info.get("previousClose")
            if market_cap is None or prev_close is None:
                return None
            return Fundamentals(
                market_cap=float(market_cap),
                prev_close=float(prev_close),
                fetched_at=dt.datetime.now(dt.UTC),
            )
        except Exception as e:
            log.debug("yfinance fetch failed", extra={"ticker": ticker, "error": str(e)})
            return None

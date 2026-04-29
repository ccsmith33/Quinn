"""MarketDataProvider port (architecture §2.10, ADR-006).

Fundamentals come from a swappable provider. v1 ships `YFinanceProvider`; v1.x
can substitute a different source by implementing this Protocol — no other
module imports yfinance directly.
"""

from __future__ import annotations

import datetime as dt
from typing import Protocol

from pydantic import BaseModel, ConfigDict


class Fundamentals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market_cap: float
    prev_close: float
    fetched_at: dt.datetime


class MarketDataProvider(Protocol):
    def fetch_fundamentals(self, ticker: str) -> Fundamentals | None: ...

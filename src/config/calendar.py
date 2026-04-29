"""NYSE market-hours / holiday helpers (D-025).

Holiday calendar comes from the checked-in static JSON list at
`src/config/nyse_holidays.json`; zero runtime calendar dependency. Refreshed
yearly per architecture §11 #6.

Used by S3.2 to choose RSS poll cadence (60s market-hours / 5min off-hours).
S6.x execution paths can reuse the same predicate.
"""

from __future__ import annotations

import datetime as dt
import json
from functools import cache
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
_HOLIDAYS_PATH = Path(__file__).parent / "nyse_holidays.json"

_MARKET_OPEN = dt.time(9, 30)
_MARKET_CLOSE = dt.time(16, 0)


@cache
def load_holidays() -> frozenset[dt.date]:
    """Read `nyse_holidays.json` once and cache. Returns a frozenset of dates."""
    raw = json.loads(_HOLIDAYS_PATH.read_text())
    return frozenset(dt.date.fromisoformat(s) for s in raw["holidays"])


def is_market_open_day(d: dt.date) -> bool:
    """True iff `d` is a weekday and not a NYSE full-day closure."""
    if d.weekday() >= 5:  # 5=Sat, 6=Sun
        return False
    return d not in load_holidays()


def is_market_hours(when: dt.datetime) -> bool:
    """True iff `when` is during a regular NYSE session (09:30–16:00 ET).

    Open is inclusive (09:30 → True), close is exclusive (16:00 → False).
    A naive `datetime` is interpreted as ET — trading-domain code is ET-by-default
    in this codebase (architecture §9.8); requiring tz-aware everywhere just
    invites correctness bugs at the boundaries.
    """
    if when.tzinfo is None:
        when_et = when.replace(tzinfo=ET)
    else:
        when_et = when.astimezone(ET)
    if not is_market_open_day(when_et.date()):
        return False
    t = when_et.time()
    return _MARKET_OPEN <= t < _MARKET_CLOSE

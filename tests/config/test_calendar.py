"""S3.2 — NYSE market-hours / holiday helpers (D-025).

Calendar comes from a checked-in static JSON list of NYSE holidays
(`src/config/nyse_holidays.json`); zero runtime calendar deps.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

from config.calendar import is_market_hours, is_market_open_day, load_holidays

_ET = ZoneInfo("America/New_York")


def _et(year: int, month: int, day: int, hour: int, minute: int) -> dt.datetime:
    return dt.datetime(year, month, day, hour, minute, tzinfo=_ET)


def test_market_hours_weekday_during_session() -> None:
    # Wed 2026-04-29 11:00 ET — full session.
    assert is_market_hours(_et(2026, 4, 29, 11, 0)) is True


def test_market_hours_open_edge() -> None:
    # 09:30 ET is the open — inclusive.
    assert is_market_hours(_et(2026, 4, 29, 9, 30)) is True
    assert is_market_hours(_et(2026, 4, 29, 9, 29)) is False


def test_market_hours_close_edge() -> None:
    # 16:00 ET is the close — exclusive.
    assert is_market_hours(_et(2026, 4, 29, 15, 59)) is True
    assert is_market_hours(_et(2026, 4, 29, 16, 0)) is False


def test_market_hours_weekend() -> None:
    # Sat 2026-05-02 — closed all day.
    assert is_market_hours(_et(2026, 5, 2, 11, 0)) is False
    assert is_market_open_day(dt.date(2026, 5, 2)) is False
    # Sun 2026-05-03 — closed all day.
    assert is_market_hours(_et(2026, 5, 3, 11, 0)) is False


def test_market_hours_holiday_christmas_2026() -> None:
    # Christmas 2026 is Friday — NYSE closed.
    assert is_market_open_day(dt.date(2026, 12, 25)) is False
    assert is_market_hours(_et(2026, 12, 25, 11, 0)) is False


def test_market_hours_holiday_new_years_2027() -> None:
    # New Year's Day 2027 is Friday — NYSE closed.
    assert is_market_open_day(dt.date(2027, 1, 1)) is False


def test_naive_datetime_treated_as_et() -> None:
    # Pragmatic: a caller passing a naive dt is interpreted as ET because
    # all market-hour math is ET; this avoids ambiguous UTC↔ET surprises.
    naive = dt.datetime(2026, 4, 29, 11, 0)
    assert is_market_hours(naive) is True


def test_load_holidays_returns_set_of_dates() -> None:
    holidays = load_holidays()
    assert dt.date(2026, 12, 25) in holidays
    assert dt.date(2027, 1, 1) in holidays
    # Should NOT include random weekdays.
    assert dt.date(2026, 4, 29) not in holidays

"""P3 — `/` today-counters reset at midnight ET, not UTC midnight."""

from __future__ import annotations

import datetime as _dt
from zoneinfo import ZoneInfo

from dashboard.app import _today_et_start

ET = ZoneInfo("America/New_York")


def test_today_et_start_at_18_et_returns_start_of_same_et_day() -> None:
    """At 18:00 ET on day D, today-start is 00:00 ET on day D (UTC ~04:00 of D)."""
    now_et = _dt.datetime(2026, 5, 3, 18, 0, tzinfo=ET)  # EDT: UTC-4
    now_utc = now_et.astimezone(_dt.UTC)
    start = _today_et_start(now_utc)
    assert start.astimezone(ET).date() == _dt.date(2026, 5, 3)
    assert start.astimezone(ET).hour == 0
    assert start.tzinfo is _dt.UTC


def test_today_et_start_just_past_midnight_et_advances_to_new_day() -> None:
    """At 00:00:01 ET on day D+1, today-start is 00:00 ET on D+1 (not D)."""
    now_et = _dt.datetime(2026, 5, 4, 0, 0, 1, tzinfo=ET)
    now_utc = now_et.astimezone(_dt.UTC)
    start = _today_et_start(now_utc)
    assert start.astimezone(ET).date() == _dt.date(2026, 5, 4)


def test_today_et_start_just_before_midnight_et_stays_on_previous_day() -> None:
    """23:59 ET on D still resolves to 00:00 ET on D — counters do NOT roll
    early at UTC midnight (the previous bug)."""
    now_et = _dt.datetime(2026, 5, 3, 23, 59, tzinfo=ET)  # 03:59 UTC on May 4
    now_utc = now_et.astimezone(_dt.UTC)
    assert now_utc.date() == _dt.date(2026, 5, 4)  # confirm UTC has rolled
    start = _today_et_start(now_utc)
    assert start.astimezone(ET).date() == _dt.date(2026, 5, 3)


def test_today_et_start_handles_dst_spring_forward() -> None:
    """Day after DST starts (2026-03-08): start-of-day is 00:00 EDT = 04:00 UTC."""
    now_et = _dt.datetime(2026, 3, 9, 12, 0, tzinfo=ET)  # EDT (UTC-4)
    now_utc = now_et.astimezone(_dt.UTC)
    start = _today_et_start(now_utc)
    assert start == _dt.datetime(2026, 3, 9, 4, 0, tzinfo=_dt.UTC)


def test_today_et_start_handles_dst_fall_back() -> None:
    """Day after DST ends (2026-11-01): start-of-day is 00:00 EST = 05:00 UTC."""
    now_et = _dt.datetime(2026, 11, 2, 12, 0, tzinfo=ET)  # EST (UTC-5)
    now_utc = now_et.astimezone(_dt.UTC)
    start = _today_et_start(now_utc)
    assert start == _dt.datetime(2026, 11, 2, 5, 0, tzinfo=_dt.UTC)

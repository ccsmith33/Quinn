"""Watchlist / deferred entry — pure-helper unit tests (`app.watchlist`).

Covers the trading-day expiry computation, the chase-guard boundary
(exactly AT the ceiling is allowed; strictly above is skipped), the
feature switch, and the exact capital-class reject-reason vocabulary.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

from app.watchlist import (
    CAPITAL_REJECT_REASONS,
    chase_exceeded,
    compute_expiry,
    ensure_utc,
    watchlist_enabled,
)
from config.calendar import ET

# ---------------------------------------------------------------------------
# compute_expiry — trading days, not calendar days.
# ---------------------------------------------------------------------------


def test_expiry_counts_trading_days_midweek() -> None:
    """Monday 2026-07-20 enrollment + 3 trading days → Thursday
    2026-07-23 16:00 ET (20:00 UTC in EDT)."""
    now = dt.datetime(2026, 7, 20, 15, 0, tzinfo=dt.UTC)  # Mon 11:00 ET
    assert compute_expiry(now, 3) == dt.datetime(2026, 7, 23, 20, 0, tzinfo=dt.UTC)


def test_expiry_skips_weekend() -> None:
    """Friday 2026-07-24 enrollment + 2 trading days → Tuesday 2026-07-28
    close, NOT Sunday — weekends don't consume the retry window."""
    now = dt.datetime(2026, 7, 24, 15, 0, tzinfo=dt.UTC)  # Fri
    expiry = compute_expiry(now, 2)
    assert expiry.astimezone(ET).date() == dt.date(2026, 7, 28)
    assert expiry.astimezone(ET).time() == dt.time(16, 0)


def test_expiry_skips_nyse_holiday() -> None:
    """Thursday 2026-07-02 + 1 trading day: Fri 07-03 is the July-4th
    observance (full-day closure) → expiry lands Monday 2026-07-06."""
    now = dt.datetime(2026, 7, 2, 15, 0, tzinfo=dt.UTC)
    expiry = compute_expiry(now, 1)
    assert expiry.astimezone(ET).date() == dt.date(2026, 7, 6)


def test_expiry_uses_et_date_not_utc_date() -> None:
    """23:30 UTC Monday is 19:30 ET Monday — the enrollment day is the ET
    Monday, so +1 trading day expires Tuesday close (not Wednesday, which
    a UTC-date reading of the same instant would produce)."""
    now = dt.datetime(2026, 7, 20, 23, 30, tzinfo=dt.UTC)
    expiry = compute_expiry(now, 1)
    assert expiry.astimezone(ET).date() == dt.date(2026, 7, 21)


# ---------------------------------------------------------------------------
# chase guard — boundary semantics.
# ---------------------------------------------------------------------------


def test_chase_exactly_at_ceiling_is_allowed() -> None:
    # ref 100 + 8% → ceiling 108: paying exactly the ceiling is allowed.
    assert chase_exceeded(108.0, 100.0, 8.0) is False


def test_chase_above_ceiling_is_skipped() -> None:
    assert chase_exceeded(108.02, 100.0, 8.0) is True


def test_chase_below_ceiling_is_allowed() -> None:
    assert chase_exceeded(99.0, 100.0, 8.0) is False
    assert chase_exceeded(104.0, 100.0, 8.0) is False


def test_chase_zero_pct_allows_reference_price_only() -> None:
    """max_chase_pct = 0 → any tick above the reference blocks; the
    reference itself is still payable."""
    assert chase_exceeded(100.0, 100.0, 0.0) is False
    assert chase_exceeded(100.01, 100.0, 0.0) is True


# ---------------------------------------------------------------------------
# feature switch + reason vocabulary.
# ---------------------------------------------------------------------------


def test_watchlist_enabled_switch() -> None:
    assert watchlist_enabled(SimpleNamespace(watchlist_min_conviction=6)) is True
    assert watchlist_enabled(SimpleNamespace(watchlist_min_conviction=0)) is False
    # Stub/legacy configs without the field → off, never a crash.
    assert watchlist_enabled(SimpleNamespace()) is False


def test_capital_reject_reasons_exact_vocabulary() -> None:
    """The enrollment filter is exactly the capital class — the sizer's
    KS-5 (book full) and KS-7 (cash floor) rejects plus the validator's
    buying-power check. Everything else must stay out: non-capital
    validator rejects, Opus rejects, kill-switch halts, KS-6 re-entry,
    conviction floor, and the ambiguous one-share case."""
    assert CAPITAL_REJECT_REASONS == {
        "ks5_concurrent_limit",
        "ks7_cash_reserve",
        "insufficient_capital",
    }
    for excluded in (
        "opus_reject",
        "kill_switch",
        "universe",
        "price_floor",
        "exit_geometry",
        "schema",
        "pending_capacity",
        "ks6_already_held",
        "conviction_too_low",
        "size_too_small_for_one_share",
        "broker_unavailable",
    ):
        assert excluded not in CAPITAL_REJECT_REASONS


def test_ensure_utc_normalizes_naive_sqlite_timestamps() -> None:
    naive = dt.datetime(2026, 7, 20, 15, 0)
    aware = ensure_utc(naive)
    assert aware.tzinfo is dt.UTC
    assert aware.replace(tzinfo=None) == naive
    already = dt.datetime(2026, 7, 20, 11, 0, tzinfo=ET)
    assert ensure_utc(already) is already

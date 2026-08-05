"""Deferred-entry watchlist — pure helpers (feature: watchlist retry).

Business context: Quinn analyzes 8-K catalysts and buys same-day. When
the book is full (KS-5) or cash is at the KS-7 floor, a high-conviction
proposal is rejected and — because filings are analyzed exactly once,
idempotently — never revisited. The 33k-event study
(quinn-catalyst-study/results/big_winner_day1_study.md) shows eventual
big winners are only ~+1% median at day-1 close and ~45% are
flat-or-down, so entering 1–2 trading days late retains ~98% of their
P&L. A bounded retry queue therefore captures nearly all of the day-0
value whenever capital frees within a couple of sessions. The one
danger is CHASING: if the name already ran, late entry at a worse price
loses the edge — hence the hard chase ceiling.

This module is pure logic (no I/O); the enrollment/retry orchestration
lives in `app.loop.AgentLoop` next to the execution chain it reuses.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from config.calendar import ET, is_market_open_day

# The CAPITAL-class reject reasons that qualify a rejected execution for
# watchlist enrollment. Deliberately narrow:
#   - ks5_concurrent_limit  — book full (SizingEngine, KS-5)
#   - ks7_cash_reserve      — cash-reserve floor (SizingEngine, KS-7)
#   - insufficient_capital  — buying-power check (validator step 6)
#   - review_skipped_full_book — the Opus-review cost gate declined to pay
#     for review because the book was full AND cash was at the KS-7 floor
#     (`execution.review_gate`). Capital-class by construction: the ONLY
#     thing standing between this proposal and a normal entry is capital,
#     and the watchlist is how it gets its second chance. Without this
#     entry the gate would silently discard proposals that today reach
#     the queue via `ks5_concurrent_limit`.
# NOT included: opus_reject / kill_switch / universe / price_floor /
# exit_geometry / schema (not capital), ks6_already_held (we never
# pyramid), conviction_too_low (won't change), and
# size_too_small_for_one_share (ambiguous — usually the KS-4 absolute
# cap vs an expensive share price, which freeing capital cannot fix).
CAPITAL_REJECT_REASONS = frozenset(
    {
        "ks5_concurrent_limit",
        "ks7_cash_reserve",
        "insufficient_capital",
        "review_skipped_full_book",
    }
)

_MARKET_CLOSE_ET = dt.time(16, 0)


def watchlist_enabled(execution_cfg: Any) -> bool:
    """Feature switch: `watchlist_min_conviction > 0`. Tolerates stub /
    legacy configs that lack the field (test convenience — mirrors the
    loop's other config probes)."""
    try:
        return int(getattr(execution_cfg, "watchlist_min_conviction", 0)) > 0
    except (TypeError, ValueError):
        return False


def compute_expiry(now: dt.datetime, trading_days: int) -> dt.datetime:
    """Expiry = 16:00 ET (market close) on the Nth market-open day AFTER
    the enrollment's ET date, returned in UTC.

    Trading days, not calendar days (same day-counting convention as
    `execution.displacement.trading_days_held`): a Friday-afternoon
    enrollment with N=3 stays retryable through Wednesday's close.
    """
    d = now.astimezone(ET).date()
    remaining = trading_days
    while remaining > 0:
        d += dt.timedelta(days=1)
        if is_market_open_day(d):
            remaining -= 1
    close_et = dt.datetime.combine(d, _MARKET_CLOSE_ET, tzinfo=ET)
    return close_et.astimezone(dt.UTC)


def chase_exceeded(
    current_last: float, reference_price: float, max_chase_pct: float
) -> bool:
    """True when the current price is STRICTLY above the chase ceiling.

    Exactly at `reference_price * (1 + max_chase_pct/100)` is still
    allowed — the guard rejects paying more than the ceiling, not the
    ceiling itself.
    """
    return current_last > reference_price * (1.0 + max_chase_pct / 100.0)


def ensure_utc(d: dt.datetime) -> dt.datetime:
    """SQLite returns naive datetimes for TIMESTAMP columns; journal
    timestamps are UTC by convention (mirrors execution.displacement)."""
    if d.tzinfo is None:
        return d.replace(tzinfo=dt.UTC)
    return d


__all__ = [
    "CAPITAL_REJECT_REASONS",
    "chase_exceeded",
    "compute_expiry",
    "ensure_utc",
    "watchlist_enabled",
]

"""Simulation core: replay one proposal against daily bars. Pure stdlib.

Every rule here is a documented assumption that `report.py` reproduces in
the report header — if you change a rule, update ASSUMPTIONS there too.

Rules (D-080 task spec):
- Enter at the first regular-session open strictly after the proposal
  timestamp (UTC→ET; a proposal created before 09:30 ET on a trading day
  enters at that day's open).
- GTC stop and take-profit per proposal geometry, active from entry day on.
- Touch detection from daily OHLC: stop fires when low <= stop, TP when
  high >= TP. Same-day both-touch resolves stop-first (conservative).
- Gap fills: sell-stop fills at the open when the open gaps below the stop;
  TP limit fills at the open when the open gaps above the TP. Otherwise
  fills are at exact stop/TP price. No slippage, no fees.
- Time-stop at the close of the last trading bar within
  entry_date + time_horizon_days calendar days (default 30 when absent).
- Trailing mode (H3 counterfactual): TP removed; effective stop is
  max(proposal stop, high-watermark * (1 - trailing_pct)); the watermark is
  the entry open then prior-day highs — today's high never arms today's
  trail level (conservative: no intraday peak-then-reversal credit).
- Splits: only splits whose ex-date is after the proposal's creation date
  (ET) apply — geometry proposed on or after an ex-date already states
  post-split prices. On ex-date, stop/TP/watermark divide by the ratio,
  qty multiplies. A split effective on or before the entry session divides
  stop/TP only: the entry fill (hence qty and the watermark) is already in
  post-split prices.
- Bars exhausted before the horizon → status "open", marked at last close.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from zoneinfo import ZoneInfo

from ops.replay.models import (
    EXIT_END_OF_DATA,
    EXIT_STOP,
    EXIT_TAKE_PROFIT,
    EXIT_TIME_STOP,
    EXIT_TRAILING_STOP,
    STATUS_CLOSED,
    STATUS_NO_DATA,
    STATUS_OPEN,
    DailyBar,
    ReplayProposal,
    SimConfig,
    SplitEvent,
    TradeResult,
)

ET = ZoneInfo("America/New_York")
_MARKET_OPEN = dt.time(9, 30)

IsOpenDay = Callable[[dt.date], bool]


def entry_session_date(proposal_utc: dt.datetime, is_open_day: IsOpenDay) -> dt.date:
    """First regular-session date whose 09:30 ET open is strictly after the
    proposal timestamp."""
    if proposal_utc.tzinfo is None:
        raise ValueError("proposal timestamp must be tz-aware UTC")
    et = proposal_utc.astimezone(ET)
    if is_open_day(et.date()) and et.time() < _MARKET_OPEN:
        return et.date()
    candidate = et.date() + dt.timedelta(days=1)
    while not is_open_day(candidate):
        candidate += dt.timedelta(days=1)
    return candidate


def simulate_trade(
    proposal: ReplayProposal,
    bars: list[DailyBar],
    splits: list[SplitEvent],
    config: SimConfig,
    is_open_day: IsOpenDay,
) -> TradeResult:
    """Replay one proposal against `bars` (sorted ascending, raw prices)."""
    entry_d = entry_session_date(proposal.created_at_utc, is_open_day)
    idx = next((i for i, b in enumerate(bars) if b.date >= entry_d), None)
    if idx is None:
        return TradeResult(proposal=proposal, status=STATUS_NO_DATA)

    entry_bar = bars[idx]
    entry_price = entry_bar.open
    qty = config.notional_usd / entry_price
    stop = proposal.stop_loss_price
    trailing = config.exit_mode == "trailing"
    tp = None if trailing else proposal.take_profit_price
    horizon = proposal.time_horizon_days or config.default_horizon_days
    horizon_end = entry_bar.date + dt.timedelta(days=horizon)
    # A proposal created on or after a split's ex-date already states
    # post-split prices — re-applying that split would corrupt the geometry.
    created_et_date = proposal.created_at_utc.astimezone(ET).date()
    pending_splits = sorted(
        (s for s in splits if s.ex_date > created_et_date), key=lambda s: s.ex_date
    )
    splits_applied = 0
    peak = entry_price  # trailing high watermark (prior-day basis)

    def closed(bar_date: dt.date, exit_price: float, reason: str) -> TradeResult:
        pnl = qty * exit_price - config.notional_usd
        return TradeResult(
            proposal=proposal,
            status=STATUS_CLOSED if reason != EXIT_END_OF_DATA else STATUS_OPEN,
            entry_date=entry_bar.date,
            entry_price=entry_price,
            qty=qty,
            exit_date=bar_date,
            exit_price=exit_price,
            exit_reason=reason,
            holding_days=(bar_date - entry_bar.date).days,
            pnl_usd=pnl,
            pnl_pct=pnl / config.notional_usd,
            splits_applied=splits_applied,
        )

    in_horizon = [b for b in bars[idx:] if b.date <= horizon_end]
    for i, bar in enumerate(in_horizon):
        while pending_splits and pending_splits[0].ex_date <= bar.date:
            ev = pending_splits.pop(0)
            stop /= ev.ratio
            if tp is not None:
                tp /= ev.ratio
            if ev.ex_date > entry_bar.date:
                # held across the ex-date: shares and watermark scale too
                peak /= ev.ratio
                qty *= ev.ratio
            # else: split effective on/before the entry session — the entry
            # fill (hence qty and the watermark) is already post-split, so
            # only the proposed stop/TP geometry needs adjusting
            splits_applied += 1

        eff_stop = max(stop, peak * (1.0 - config.trailing_pct)) if trailing else stop
        if bar.low <= eff_stop:  # stop-first on same-day both-touch
            exit_price = bar.open if bar.open <= eff_stop else eff_stop
            reason = EXIT_TRAILING_STOP if trailing and eff_stop > stop else EXIT_STOP
            return closed(bar.date, exit_price, reason)
        if tp is not None and bar.high >= tp:
            exit_price = bar.open if bar.open >= tp else tp
            return closed(bar.date, exit_price, EXIT_TAKE_PROFIT)

        peak = max(peak, bar.high)
        if i == len(in_horizon) - 1:
            if bar is bars[-1] and bar.date < horizon_end:
                # data ran out before the horizon: still open, mark at close
                return closed(bar.date, bar.close, EXIT_END_OF_DATA)
            return closed(bar.date, bar.close, EXIT_TIME_STOP)

    # bars[idx] exists but nothing fell inside the horizon window — cannot
    # happen since in_horizon always contains entry_bar; defensive only.
    return TradeResult(proposal=proposal, status=STATUS_NO_DATA)

"""Unit tests for the replay simulation core (`ops/replay/sim.py`).

Task #5 acceptance: same-day stop+TP touch resolves stop-first, time-stop
behavior, split handling in the price series — plus entry-session selection,
gap fills, trailing-mode mechanics, and aggregate math.
"""

from __future__ import annotations

import datetime as dt

import pytest
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
from ops.replay.report import compute_aggregates
from ops.replay.sim import entry_session_date, simulate_trade


def weekdays_open(d: dt.date) -> bool:
    return d.weekday() < 5


def make_proposal(
    stop: float = 8.0,
    tp: float | None = 14.0,
    horizon: int | None = 30,
    created_utc: dt.datetime | None = None,
) -> ReplayProposal:
    return ReplayProposal(
        proposal_id=1,
        decision_id="d-1",
        symbol="TEST",
        # Monday 2026-04-06 08:00 ET = 12:00 UTC (EDT) → pre-open, enters same day
        created_at_utc=created_utc or dt.datetime(2026, 4, 6, 12, 0, tzinfo=dt.UTC),
        stop_loss_price=stop,
        take_profit_price=tp,
        time_horizon_days=horizon,
        conviction=7,
        exec_decision=None,
    )


def bar(day: int, o: float, h: float, lo: float, c: float) -> DailyBar:
    """Bar on 2026-04-(day). April 2026: 6th is a Monday."""
    return DailyBar(date=dt.date(2026, 4, day), open=o, high=h, low=lo, close=c)


CFG = SimConfig(notional_usd=1000.0)
CFG_TRAIL = SimConfig(notional_usd=1000.0, exit_mode="trailing", trailing_pct=0.15)


def run(proposal, bars, splits=(), cfg=CFG) -> TradeResult:
    return simulate_trade(proposal, bars, list(splits), cfg, weekdays_open)


# ---------------------------------------------------------------------------
# Entry session selection
# ---------------------------------------------------------------------------


class TestEntrySessionDate:
    def test_pre_open_proposal_enters_same_day(self):
        # 08:00 ET Monday
        ts = dt.datetime(2026, 4, 6, 12, 0, tzinfo=dt.UTC)
        assert entry_session_date(ts, weekdays_open) == dt.date(2026, 4, 6)

    def test_mid_session_proposal_enters_next_day(self):
        # 11:00 ET Monday
        ts = dt.datetime(2026, 4, 6, 15, 0, tzinfo=dt.UTC)
        assert entry_session_date(ts, weekdays_open) == dt.date(2026, 4, 7)

    def test_friday_afternoon_enters_monday(self):
        # 15:00 ET Friday 2026-04-10
        ts = dt.datetime(2026, 4, 10, 19, 0, tzinfo=dt.UTC)
        assert entry_session_date(ts, weekdays_open) == dt.date(2026, 4, 13)

    def test_saturday_enters_monday(self):
        ts = dt.datetime(2026, 4, 11, 12, 0, tzinfo=dt.UTC)
        assert entry_session_date(ts, weekdays_open) == dt.date(2026, 4, 13)

    def test_exactly_at_open_enters_next_day(self):
        # 09:30:00 ET is not strictly before the open
        ts = dt.datetime(2026, 4, 6, 13, 30, tzinfo=dt.UTC)
        assert entry_session_date(ts, weekdays_open) == dt.date(2026, 4, 7)

    def test_naive_timestamp_rejected(self):
        with pytest.raises(ValueError):
            entry_session_date(dt.datetime(2026, 4, 6, 12, 0), weekdays_open)


# ---------------------------------------------------------------------------
# Exit mechanics — proposal-TP mode
# ---------------------------------------------------------------------------


class TestExits:
    def test_take_profit_fills_at_tp_price(self):
        bars = [bar(6, 10, 10.5, 9.5, 10), bar(7, 10.2, 14.5, 10.0, 13.0)]
        r = run(make_proposal(), bars)
        assert r.status == STATUS_CLOSED
        assert r.exit_reason == EXIT_TAKE_PROFIT
        assert r.exit_price == 14.0
        assert r.pnl_usd == pytest.approx(100 * 14.0 - 1000)

    def test_gap_above_tp_fills_at_open(self):
        bars = [bar(6, 10, 10.5, 9.5, 10), bar(7, 15.0, 16.0, 14.5, 15.5)]
        r = run(make_proposal(), bars)
        assert r.exit_reason == EXIT_TAKE_PROFIT
        assert r.exit_price == 15.0

    def test_stop_fills_at_stop_price(self):
        bars = [bar(6, 10, 10.5, 9.5, 10), bar(7, 9.0, 9.2, 7.5, 8.5)]
        r = run(make_proposal(), bars)
        assert r.exit_reason == EXIT_STOP
        assert r.exit_price == 8.0
        assert r.pnl_usd == pytest.approx(100 * 8.0 - 1000)

    def test_gap_below_stop_fills_at_open(self):
        bars = [bar(6, 10, 10.5, 9.5, 10), bar(7, 7.0, 7.4, 6.5, 7.2)]
        r = run(make_proposal(), bars)
        assert r.exit_reason == EXIT_STOP
        assert r.exit_price == 7.0

    def test_same_day_both_touch_resolves_stop_first(self):
        # Day spans both the stop (8) and the TP (14): conservative = stop.
        bars = [bar(6, 10, 10.5, 9.5, 10), bar(7, 10.0, 14.5, 7.9, 12.0)]
        r = run(make_proposal(), bars)
        assert r.exit_reason == EXIT_STOP
        assert r.exit_price == 8.0

    def test_entry_day_touches_count(self):
        bars = [bar(6, 10, 14.5, 9.8, 13.0)]
        r = run(make_proposal(horizon=1), bars)
        assert r.exit_reason == EXIT_TAKE_PROFIT
        assert r.exit_date == dt.date(2026, 4, 6)

    def test_entry_day_open_below_stop_exits_flat_at_open(self):
        bars = [bar(6, 7.5, 8.2, 7.0, 8.0)]
        r = run(make_proposal(), bars)
        assert r.exit_reason == EXIT_STOP
        assert r.exit_price == 7.5  # = entry → zero P&L
        assert r.pnl_usd == pytest.approx(0.0)

    def test_no_tp_proposal_runs_to_time_stop(self):
        bars = [bar(6, 10, 11, 9.5, 10.5), bar(7, 10.5, 12, 10, 11.0)]
        r = run(make_proposal(tp=None, horizon=1), bars)
        assert r.exit_reason == EXIT_TIME_STOP


# ---------------------------------------------------------------------------
# Time-stop
# ---------------------------------------------------------------------------


class TestTimeStop:
    def test_exits_at_close_of_last_bar_within_horizon(self):
        bars = [
            bar(6, 10, 10.5, 9.5, 10),
            bar(7, 10, 10.5, 9.5, 10.2),
            bar(8, 10.2, 10.6, 9.9, 10.4),  # horizon end (entry + 2d)
            bar(9, 10.4, 13.0, 10.0, 12.5),  # beyond horizon — must not count
        ]
        r = run(make_proposal(horizon=2), bars)
        assert r.status == STATUS_CLOSED
        assert r.exit_reason == EXIT_TIME_STOP
        assert r.exit_date == dt.date(2026, 4, 8)
        assert r.exit_price == 10.4
        assert r.holding_days == 2

    def test_horizon_counts_calendar_days_over_weekend(self):
        # Entry Friday 2026-04-10 (proposal Friday pre-open), horizon 3 days
        # → horizon end Monday 4/13; Monday bar is the time-stop bar.
        p = make_proposal(
            horizon=3, created_utc=dt.datetime(2026, 4, 10, 12, 0, tzinfo=dt.UTC)
        )
        bars = [
            bar(10, 10, 10.5, 9.5, 10),
            bar(13, 10, 10.5, 9.5, 10.1),
            bar(14, 10.1, 10.5, 9.5, 10.2),
        ]
        r = run(p, bars)
        assert r.exit_reason == EXIT_TIME_STOP
        assert r.exit_date == dt.date(2026, 4, 13)

    def test_missing_horizon_uses_config_default(self):
        bars = [bar(6, 10, 10.5, 9.5, 10), bar(7, 10, 10.5, 9.5, 10.3)]
        cfg = SimConfig(notional_usd=1000.0, default_horizon_days=1)
        r = run(make_proposal(horizon=None), bars, cfg=cfg)
        assert r.exit_reason == EXIT_TIME_STOP
        assert r.exit_date == dt.date(2026, 4, 7)

    def test_data_exhausted_before_horizon_is_open_marked_at_close(self):
        bars = [bar(6, 10, 10.5, 9.5, 10), bar(7, 10, 10.5, 9.5, 10.8)]
        r = run(make_proposal(horizon=30), bars)
        assert r.status == STATUS_OPEN
        assert r.exit_reason == EXIT_END_OF_DATA
        assert r.exit_price == 10.8

    def test_no_bars_after_entry_is_no_data(self):
        bars = [DailyBar(date=dt.date(2026, 4, 1), open=10, high=11, low=9, close=10)]
        r = run(make_proposal(), bars)
        assert r.status == STATUS_NO_DATA


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


class TestSplits:
    def test_forward_split_scales_geometry_and_qty(self):
        # 2-for-1 on 4/8: stop 8→4, tp 14→7, qty 100→200.
        bars = [
            bar(6, 10, 10.5, 9.5, 10),
            bar(7, 10, 10.5, 9.5, 10),
            bar(8, 5.1, 5.2, 4.9, 5.0),  # post-split prices; 4.9 > new stop 4
            bar(9, 5.0, 7.2, 4.95, 7.0),  # high 7.2 ≥ new tp 7
        ]
        r = run(make_proposal(), bars, splits=[SplitEvent(dt.date(2026, 4, 8), 2.0)])
        assert r.exit_reason == EXIT_TAKE_PROFIT
        assert r.exit_price == pytest.approx(7.0)
        assert r.qty == pytest.approx(200.0)
        assert r.splits_applied == 1
        # economically identical to hitting the original 14 TP unsplit
        assert r.pnl_usd == pytest.approx(400.0)

    def test_forward_split_does_not_false_trigger_stop(self):
        # Without scaling, post-split prices (~5) would breach the raw stop (8).
        bars = [
            bar(6, 10, 10.5, 9.5, 10),
            bar(7, 5.1, 5.2, 4.9, 5.0),
            bar(8, 5.0, 5.3, 4.8, 5.2),
        ]
        r = run(
            make_proposal(horizon=2), bars, splits=[SplitEvent(dt.date(2026, 4, 7), 2.0)]
        )
        assert r.exit_reason == EXIT_TIME_STOP

    def test_proposal_created_after_split_sees_no_adjustment(self):
        # W6-M1: the 4/1 split predates the proposal (created 4/6), so the
        # journaled geometry already states post-split prices — re-applying
        # the split would halve stop/TP and double qty.
        bars = [bar(6, 10, 10.5, 9.5, 10), bar(7, 10.2, 14.5, 10.0, 13.0)]
        r = run(make_proposal(), bars, splits=[SplitEvent(dt.date(2026, 4, 1), 2.0)])
        assert r.exit_reason == EXIT_TAKE_PROFIT
        assert r.exit_price == 14.0
        assert r.qty == pytest.approx(100.0)
        assert r.splits_applied == 0
        assert r.pnl_usd == pytest.approx(400.0)

    def test_split_on_creation_date_not_applied(self):
        # Ex-date == creation date: quotes are post-split from that day's
        # open, so the pre-open proposal geometry is post-split too.
        bars = [bar(6, 10, 10.5, 9.5, 10), bar(7, 10.2, 14.5, 10.0, 13.0)]
        r = run(make_proposal(), bars, splits=[SplitEvent(dt.date(2026, 4, 6), 2.0)])
        assert r.exit_reason == EXIT_TAKE_PROFIT
        assert r.exit_price == 14.0
        assert r.splits_applied == 0

    def test_split_on_entry_session_adjusts_geometry_only(self):
        # W6-M2: created mid-session Mon 4/6 → enters Tue 4/7; 2-for-1 ex 4/7.
        # The entry fill is already post-split, so qty is notional/open with
        # no scaling — but the pre-split stop 8 and TP 14 must still divide
        # (an undivided stop would fire on the entry bar's 4.5 low).
        p = make_proposal(created_utc=dt.datetime(2026, 4, 6, 15, 0, tzinfo=dt.UTC))
        bars = [
            bar(6, 10, 10.5, 9.5, 10),  # pre-split session, before entry
            bar(7, 5.0, 5.2, 4.5, 5.0),  # entry; low 4.5 > new stop 4
            bar(8, 5.0, 7.2, 4.95, 7.0),  # high ≥ new tp 7
        ]
        r = run(p, bars, splits=[SplitEvent(dt.date(2026, 4, 7), 2.0)])
        assert r.entry_price == 5.0
        assert r.qty == pytest.approx(200.0)  # 1000/5, NOT doubled
        assert r.exit_reason == EXIT_TAKE_PROFIT
        assert r.exit_price == pytest.approx(7.0)
        assert r.splits_applied == 1
        assert r.pnl_usd == pytest.approx(400.0)

    def test_reverse_split_on_entry_session_keeps_watermark_unscaled(self):
        # W6-M2, trailing mode: 1-for-4 ex on the entry session. A wrongly
        # scaled watermark (40 → 160) would arm a 136 trail above price and
        # force a spurious exit at the entry open; the unscaled watermark
        # (the post-split entry open itself) lets the trade run to time-stop.
        p = make_proposal(
            horizon=2, created_utc=dt.datetime(2026, 4, 6, 15, 0, tzinfo=dt.UTC)
        )
        bars = [
            bar(6, 10, 10.5, 9.5, 10),
            bar(7, 40.0, 41.0, 39.0, 40.5),  # entry; stop 8→32; eff trail 34
            bar(8, 40.5, 41.0, 39.5, 40.0),  # eff trail max(32, 41*0.85)=34.85
            bar(9, 40.0, 40.5, 39.0, 39.5),  # horizon end → time-stop at close
        ]
        r = run(p, bars, splits=[SplitEvent(dt.date(2026, 4, 7), 0.25)], cfg=CFG_TRAIL)
        assert r.qty == pytest.approx(25.0)  # 1000/40, NOT quartered
        assert r.exit_reason == EXIT_TIME_STOP
        assert r.exit_date == dt.date(2026, 4, 9)
        assert r.splits_applied == 1
        assert r.pnl_usd == pytest.approx(25.0 * 39.5 - 1000.0)

    def test_reverse_split_scales_geometry_and_qty(self):
        # 1-for-4 on 4/7: stop 8→32, tp 14→56, qty 100→25.
        bars = [
            bar(6, 10, 10.5, 9.5, 10),
            bar(7, 40.5, 41.0, 39.0, 40.0),  # low 39 > new stop 32
            bar(8, 40.0, 57.0, 39.5, 56.5),  # high ≥ new tp 56
        ]
        r = run(make_proposal(), bars, splits=[SplitEvent(dt.date(2026, 4, 7), 0.25)])
        assert r.exit_reason == EXIT_TAKE_PROFIT
        assert r.exit_price == pytest.approx(56.0)
        assert r.qty == pytest.approx(25.0)
        assert r.pnl_usd == pytest.approx(400.0)


# ---------------------------------------------------------------------------
# Trailing mode (H3 counterfactual)
# ---------------------------------------------------------------------------


class TestTrailingMode:
    def test_tp_is_ignored_and_trailing_stop_exits(self):
        bars = [
            bar(6, 10, 10.5, 9.5, 10),  # peak 10 → eff stop 8.5
            bar(7, 10, 14.5, 9.9, 11.9),  # high 14.5 ≥ tp 14 but tp removed; peak→14.5
            bar(8, 12.4, 12.5, 12.3, 12.4),  # eff stop 12.325; low 12.3 ≤ → exit
        ]
        r = run(make_proposal(), bars, cfg=CFG_TRAIL)
        assert r.exit_reason == EXIT_TRAILING_STOP
        assert r.exit_price == pytest.approx(14.5 * 0.85)

    def test_watermark_uses_prior_days_only(self):
        # Day 2 spikes to 20 and falls back intraday; today's high must not
        # arm today's trail level, so no exit on day 2.
        bars = [
            bar(6, 10, 10.5, 9.5, 10),
            # day 2: eff stop is still max(8, 10*0.85)=8.5 (prior-day peak),
            # so the intraday 20 → 10.4 round-trip does not exit
            bar(7, 10.5, 20.0, 10.4, 16.0),
            # day 3: eff stop = max(8, 20*0.85=17); opens below it → open fill
            bar(8, 16.5, 16.8, 16.4, 16.6),
        ]
        r = run(make_proposal(tp=None), bars, cfg=CFG_TRAIL)
        assert r.exit_reason == EXIT_TRAILING_STOP
        assert r.exit_date == dt.date(2026, 4, 8)
        assert r.exit_price == pytest.approx(16.5)  # gapped below trail → open fill

    def test_proposal_stop_is_the_floor(self):
        # Trail never sits below the proposal stop; before the peak rises,
        # the proposal stop (9.0 > 10*0.85=8.5) is the binding level.
        bars = [
            bar(6, 10, 10.2, 9.6, 10),
            bar(7, 9.8, 9.9, 8.9, 9.0),  # low 8.9 ≤ floor 9.0 → plain stop
        ]
        r = run(make_proposal(stop=9.0, tp=None), bars, cfg=CFG_TRAIL)
        assert r.exit_reason == EXIT_STOP
        assert r.exit_price == pytest.approx(9.0)

    def test_time_stop_still_applies_in_trailing_mode(self):
        bars = [bar(6, 10, 10.4, 9.8, 10.1), bar(7, 10.1, 10.4, 9.9, 10.2)]
        r = run(make_proposal(tp=None, horizon=1), bars, cfg=CFG_TRAIL)
        assert r.exit_reason == EXIT_TIME_STOP


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


def closed_result(pid: int, exit_day: int, pnl: float) -> TradeResult:
    p = make_proposal()
    return TradeResult(
        proposal=ReplayProposal(
            proposal_id=pid,
            decision_id=f"d-{pid}",
            symbol="TEST",
            created_at_utc=p.created_at_utc,
            stop_loss_price=8.0,
            take_profit_price=14.0,
            time_horizon_days=30,
            conviction=5,
            exec_decision=None,
        ),
        status=STATUS_CLOSED,
        entry_date=dt.date(2026, 4, 6),
        entry_price=10.0,
        qty=100.0,
        exit_date=dt.date(2026, 4, exit_day),
        exit_price=10.0 + pnl / 100.0,
        exit_reason=EXIT_TIME_STOP,
        holding_days=exit_day - 6,
        pnl_usd=pnl,
        pnl_pct=pnl / 1000.0,
    )


class TestAggregates:
    def test_headline_stats(self):
        results = [
            closed_result(1, 7, 100.0),
            closed_result(2, 8, -50.0),
            closed_result(3, 9, 50.0),
        ]
        agg = compute_aggregates(results)
        assert agg.n_closed == 3
        assert agg.n_wins == 2 and agg.n_losses == 1
        assert agg.win_rate == pytest.approx(2 / 3)
        assert agg.avg_win_usd == pytest.approx(75.0)
        assert agg.avg_loss_usd == pytest.approx(-50.0)
        assert agg.payoff_ratio == pytest.approx(1.5)
        assert agg.expectancy_usd == pytest.approx(100.0 / 3)
        assert agg.total_pnl_usd == pytest.approx(100.0)

    def test_max_drawdown_from_equity_curve(self):
        results = [
            closed_result(1, 7, 100.0),  # cum 100 (peak)
            closed_result(2, 8, -80.0),  # cum 20 → dd 80
            closed_result(3, 9, 30.0),  # cum 50
        ]
        agg = compute_aggregates(results)
        assert agg.max_drawdown_usd == pytest.approx(80.0)
        assert agg.max_drawdown_date == dt.date(2026, 4, 8)
        assert agg.equity_curve == [
            (dt.date(2026, 4, 7), pytest.approx(100.0)),
            (dt.date(2026, 4, 8), pytest.approx(20.0)),
            (dt.date(2026, 4, 9), pytest.approx(50.0)),
        ]

    def test_open_and_no_data_excluded_from_stats(self):
        open_r = TradeResult(proposal=make_proposal(), status=STATUS_OPEN, pnl_usd=999.0)
        nodata_r = TradeResult(proposal=make_proposal(), status=STATUS_NO_DATA)
        agg = compute_aggregates([closed_result(1, 7, 10.0), open_r, nodata_r])
        assert agg.n_closed == 1
        assert agg.n_open == 1
        assert agg.n_no_data == 1
        assert agg.total_pnl_usd == pytest.approx(10.0)

    def test_empty_results(self):
        agg = compute_aggregates([])
        assert agg.n_closed == 0
        assert agg.win_rate is None

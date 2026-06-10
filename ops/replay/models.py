"""Shared dataclasses for the replay tool. Pure stdlib, no I/O."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DailyBar:
    """One raw (split-unadjusted) daily OHLCV bar."""

    date: dt.date
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass(frozen=True)
class SplitEvent:
    """A stock split effective on `ex_date`.

    `ratio` is new shares per old share: 2.0 for a 2-for-1 forward split,
    0.25 for a 1-for-4 reverse split. Prices divide by `ratio` on ex-date;
    share counts multiply by it.
    """

    ex_date: dt.date
    ratio: float


@dataclass(frozen=True)
class ReplayProposal:
    """A journaled trade proposal reduced to what the simulator needs."""

    proposal_id: int
    decision_id: str
    symbol: str
    created_at_utc: dt.datetime  # tz-aware UTC
    stop_loss_price: float
    take_profit_price: float | None
    time_horizon_days: int | None
    conviction: int | None
    exec_decision: str | None  # executions.decision for this proposal, if any


@dataclass(frozen=True)
class SkippedProposal:
    """A proposal row the replay could not simulate, with the reason why."""

    proposal_id: int
    symbol: str | None
    reason: str


@dataclass(frozen=True)
class SimConfig:
    """Simulation knobs. Defaults match the D-080 task spec."""

    notional_usd: float = 1000.0
    default_horizon_days: int = 30
    exit_mode: str = "proposal_tp"  # "proposal_tp" | "trailing"
    trailing_pct: float = 0.15


# TradeResult.status values
STATUS_CLOSED = "closed"
STATUS_OPEN = "open"  # horizon not reached before data ended; marked-to-market
STATUS_NO_DATA = "no_data"  # no bars on/after the computed entry session

# TradeResult.exit_reason values
EXIT_STOP = "stop"
EXIT_TAKE_PROFIT = "take_profit"
EXIT_TRAILING_STOP = "trailing_stop"
EXIT_TIME_STOP = "time_stop"
EXIT_END_OF_DATA = "end_of_data"  # status=open mark-to-market pseudo-exit


@dataclass(frozen=True)
class TradeResult:
    """Outcome of simulating one proposal under one SimConfig."""

    proposal: ReplayProposal
    status: str
    entry_date: dt.date | None = None
    entry_price: float | None = None
    qty: float | None = None  # shares at exit (post-split)
    exit_date: dt.date | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    holding_days: int | None = None  # calendar days entry→exit
    pnl_usd: float | None = None
    pnl_pct: float | None = None
    splits_applied: int = 0


@dataclass
class Aggregates:
    """Headline stats over the closed trades of a result set."""

    n_total: int = 0
    n_closed: int = 0
    n_open: int = 0
    n_no_data: int = 0
    n_wins: int = 0
    n_losses: int = 0
    win_rate: float | None = None
    avg_win_usd: float | None = None
    avg_loss_usd: float | None = None
    payoff_ratio: float | None = None
    expectancy_usd: float | None = None
    total_pnl_usd: float = 0.0
    max_drawdown_usd: float = 0.0
    max_drawdown_date: dt.date | None = None
    # cumulative realized P&L by exit date, one point per closed trade
    equity_curve: list[tuple[dt.date, float]] = field(default_factory=list)

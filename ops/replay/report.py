"""Aggregate statistics and report rendering (CSV + markdown).

The ASSUMPTIONS list below is the contract with the reader: every
simulation rule in `sim.py` (and the sizing/data conventions) must appear
here, because the report header is the only place the operator sees them.
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

from ops.replay.extract import REGIME_SPLIT_UTC
from ops.replay.models import (
    STATUS_CLOSED,
    STATUS_NO_DATA,
    STATUS_OPEN,
    Aggregates,
    SimConfig,
    SkippedProposal,
    TradeResult,
)

ASSUMPTIONS = [
    "Entry at the first regular NYSE session open strictly after the proposal's "
    "created_at (stored UTC, converted to ET). A proposal created before 09:30 ET "
    "on a trading day enters at that same day's open.",
    "Entry style and limit prices in the proposal are ignored — every entry is "
    "simulated as market-on-open. This removes the entry-style confound and is "
    "slightly pessimistic for limit entries that would have filled lower.",
    "Position size is a flat configurable notional per trade (default $1,000), "
    "fractional shares, every proposal independently funded. No capital "
    "constraint, no position-count limit, no overlap or PDT constraint.",
    "GTC stop and take-profit at the proposal's exact geometry, live from entry "
    "day onward (entry-day touches count).",
    "Touch detection uses daily OHLC only: stop fires if low <= stop, TP fires "
    "if high >= TP. When both touch on the same day, the stop is assumed to "
    "fire first (conservative).",
    "Gap fills: a sell-stop fills at the open when the session opens below the "
    "stop; a TP limit fills at the open when the session opens above the TP. "
    "Otherwise fills are at the exact stop/TP price. Zero slippage, zero "
    "commission/fees, full fill of the notional assumed regardless of liquidity "
    "(optimistic for microcaps).",
    "An entry session that opens at or below the stop enters at the open and "
    "is stopped out at that same open — a zero-P&L scratch (entry and exit at "
    "the identical price).",
    "Time-stop: exit at the close of the last trading bar within entry date + "
    "time_horizon_days calendar days; proposals without a horizon use the "
    "configurable default (30 days).",
    "Counterfactual mode (H3): take-profit removed entirely; exit is "
    "max(proposal stop, trailing stop at the configured percent below the high "
    "watermark). The watermark starts at the entry open and advances on prior "
    "days' highs only — today's high never arms today's trail level. The "
    "time-stop is unchanged.",
    "Bars are raw (split-unadjusted) daily OHLC from the Alpaca market-data "
    "API. Forward/reverse splits adjust geometry and share count on ex-date — "
    "but only splits whose ex-date falls after the proposal's creation date "
    "(ET): geometry proposed on or after an ex-date already states post-split "
    "prices. A split effective on or before the entry session adjusts stop/TP "
    "only — the entry fill (and so the share count and trailing watermark) is "
    "already in post-split prices. Cash dividends are ignored (long-only: "
    "slightly understates returns).",
    "Proposals whose horizon extends past the last available bar are reported "
    "as OPEN, marked to market at the last close, and excluded from "
    "closed-trade statistics.",
    "Pre/post regime split at 2026-05-08 00:00 UTC (PDT-layer merge) by "
    "proposal created_at.",
    "Equity curve is cumulative realized P&L ordered by exit date; max "
    "drawdown is the largest peak-to-trough decline of that curve in dollars. "
    "Sequencing ignores capital reuse — trades are independent $-notional bets.",
]


def compute_aggregates(results: list[TradeResult]) -> Aggregates:
    agg = Aggregates(n_total=len(results))
    closed = [r for r in results if r.status == STATUS_CLOSED]
    agg.n_closed = len(closed)
    agg.n_open = sum(1 for r in results if r.status == STATUS_OPEN)
    agg.n_no_data = sum(1 for r in results if r.status == STATUS_NO_DATA)
    if not closed:
        return agg

    pnls = [r.pnl_usd for r in closed if r.pnl_usd is not None]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    agg.n_wins = len(wins)
    agg.n_losses = len(losses)
    agg.win_rate = len(wins) / len(pnls)
    agg.avg_win_usd = sum(wins) / len(wins) if wins else None
    agg.avg_loss_usd = sum(losses) / len(losses) if losses else None
    if agg.avg_win_usd is not None and agg.avg_loss_usd is not None and agg.avg_loss_usd != 0:
        agg.payoff_ratio = agg.avg_win_usd / abs(agg.avg_loss_usd)
    agg.expectancy_usd = sum(pnls) / len(pnls)
    agg.total_pnl_usd = sum(pnls)

    ordered = sorted(closed, key=lambda r: (r.exit_date or dt.date.min, r.proposal.proposal_id))
    cum = 0.0
    peak = 0.0
    for r in ordered:
        cum += r.pnl_usd or 0.0
        agg.equity_curve.append((r.exit_date or dt.date.min, cum))
        peak = max(peak, cum)
        dd = peak - cum
        if dd > agg.max_drawdown_usd:
            agg.max_drawdown_usd = dd
            agg.max_drawdown_date = r.exit_date
    return agg


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

_TRADE_FIELDS = [
    "proposal_id",
    "decision_id",
    "symbol",
    "created_at_utc",
    "conviction",
    "exec_decision",
    "stop_loss_price",
    "take_profit_price",
    "time_horizon_days",
    "status",
    "entry_date",
    "entry_price",
    "qty",
    "exit_date",
    "exit_price",
    "exit_reason",
    "holding_days",
    "pnl_usd",
    "pnl_pct",
    "splits_applied",
]


def write_trades_csv(path: Path, results: list[TradeResult]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_TRADE_FIELDS)
        w.writeheader()
        for r in results:
            p = r.proposal
            w.writerow(
                {
                    "proposal_id": p.proposal_id,
                    "decision_id": p.decision_id,
                    "symbol": p.symbol,
                    "created_at_utc": p.created_at_utc.isoformat(),
                    "conviction": p.conviction,
                    "exec_decision": p.exec_decision,
                    "stop_loss_price": p.stop_loss_price,
                    "take_profit_price": p.take_profit_price,
                    "time_horizon_days": p.time_horizon_days,
                    "status": r.status,
                    "entry_date": r.entry_date.isoformat() if r.entry_date else "",
                    "entry_price": _fmt(r.entry_price),
                    "qty": _fmt(r.qty, 6),
                    "exit_date": r.exit_date.isoformat() if r.exit_date else "",
                    "exit_price": _fmt(r.exit_price),
                    "exit_reason": r.exit_reason or "",
                    "holding_days": r.holding_days if r.holding_days is not None else "",
                    "pnl_usd": _fmt(r.pnl_usd),
                    "pnl_pct": _fmt(r.pnl_pct, 4),
                    "splits_applied": r.splits_applied,
                }
            )


def write_equity_csv(path: Path, agg: Aggregates) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exit_date", "cumulative_pnl_usd"])
        for date, cum in agg.equity_curve:
            w.writerow([date.isoformat(), f"{cum:.2f}"])


def _fmt(v: float | None, nd: int = 2) -> str:
    return "" if v is None else f"{v:.{nd}f}"


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _agg_rows(label: str, agg: Aggregates) -> str:
    def pct(v: float | None) -> str:
        return "-" if v is None else f"{v * 100:.1f}%"

    def usd(v: float | None) -> str:
        return "-" if v is None else f"${v:,.2f}"

    def ratio(v: float | None) -> str:
        return "-" if v is None else f"{v:.2f}"

    dd_date = agg.max_drawdown_date.isoformat() if agg.max_drawdown_date else "-"
    return (
        f"| {label} | {agg.n_closed} | {agg.n_open} | {agg.n_no_data} "
        f"| {pct(agg.win_rate)} | {usd(agg.avg_win_usd)} | {usd(agg.avg_loss_usd)} "
        f"| {ratio(agg.payoff_ratio)} | {usd(agg.expectancy_usd)} "
        f"| {usd(agg.total_pnl_usd)} | {usd(agg.max_drawdown_usd)} ({dd_date}) |"
    )


_AGG_HEADER = (
    "| segment | closed | open | no-data | win rate | avg win | avg loss "
    "| payoff | expectancy | total P&L | max drawdown |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|"
)


def _split_pre_post(
    results: list[TradeResult],
) -> tuple[list[TradeResult], list[TradeResult]]:
    pre = [r for r in results if r.proposal.created_at_utc < REGIME_SPLIT_UTC]
    post = [r for r in results if r.proposal.created_at_utc >= REGIME_SPLIT_UTC]
    return pre, post


def render_markdown(
    journal_path: str,
    config: SimConfig,
    results_tp: list[TradeResult],
    results_trailing: list[TradeResult],
    skipped: list[SkippedProposal],
    bar_failures: dict[str, str],
    bar_source: str,
) -> str:
    now = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []
    lines.append("# Retrospective clean-execution replay (D-080)")
    lines.append("")
    lines.append(f"Generated: {now}  ")
    lines.append(f"Journal: `{journal_path}` (opened read-only)  ")
    lines.append(f"Bar source: {bar_source}  ")
    lines.append(
        f"Config: notional ${config.notional_usd:,.0f}/trade, default horizon "
        f"{config.default_horizon_days}d, trailing {config.trailing_pct * 100:.0f}%"
    )
    lines.append("")
    lines.append("## Simulation assumptions")
    lines.append("")
    lines.extend(f"{i}. {a}" for i, a in enumerate(ASSUMPTIONS, 1))
    lines.append("")

    lines.append("## A. Proposal geometry honored (stop + TP as proposed)")
    lines.append("")
    lines.append(_AGG_HEADER)
    pre, post = _split_pre_post(results_tp)
    lines.append(_agg_rows("overall", compute_aggregates(results_tp)))
    lines.append(_agg_rows("pre 2026-05-08", compute_aggregates(pre)))
    lines.append(_agg_rows("post 2026-05-08", compute_aggregates(post)))
    lines.append("")

    lines.append("## B. H3 counterfactual: TP removed, 15% trailing stop")
    lines.append("")
    lines.append(
        "Same trades, identical entries and stops, but the take-profit is "
        "removed and replaced with a trailing stop. The delta against section "
        "A measures what TP-capping cost (positive delta = the TPs were "
        "leaving money on the table)."
    )
    lines.append("")
    lines.append(_AGG_HEADER)
    pre_t, post_t = _split_pre_post(results_trailing)
    agg_tp = compute_aggregates(results_tp)
    agg_tr = compute_aggregates(results_trailing)
    lines.append(_agg_rows("overall", agg_tr))
    lines.append(_agg_rows("pre 2026-05-08", compute_aggregates(pre_t)))
    lines.append(_agg_rows("post 2026-05-08", compute_aggregates(post_t)))
    lines.append("")
    delta = agg_tr.total_pnl_usd - agg_tp.total_pnl_usd
    lines.append(
        f"**TP-capping cost (trailing minus proposal-TP, closed trades): "
        f"${delta:,.2f}** "
        f"(proposal-TP total ${agg_tp.total_pnl_usd:,.2f} → trailing total "
        f"${agg_tr.total_pnl_usd:,.2f})"
    )
    lines.append("")
    # Trailing mode tends to leave more trades open (winners keep running past
    # the last bar), so also compare with OPEN positions marked to market —
    # otherwise the closed-only delta understates the counterfactual.
    marked_tp = _marked_total(results_tp)
    marked_tr = _marked_total(results_trailing)
    lines.append(
        f"Including OPEN positions marked to market: proposal-TP "
        f"${marked_tp:,.2f} → trailing ${marked_tr:,.2f} "
        f"(delta ${marked_tr - marked_tp:,.2f})."
    )
    lines.append("")

    lines.append("## Per-trade results (proposal-TP mode, trailing comparison)")
    lines.append("")
    lines.append(
        "| id | symbol | proposed (UTC) | entry | stop | tp | exit | reason "
        "| P&L | P&L% | trail exit | trail reason | trail P&L |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    by_id = {r.proposal.proposal_id: r for r in results_trailing}
    for r in sorted(results_tp, key=lambda r: (r.proposal.created_at_utc, r.proposal.proposal_id)):
        p = r.proposal
        t = by_id.get(p.proposal_id)
        tp_cell = f"{p.take_profit_price:.2f}" if p.take_profit_price is not None else "-"
        lines.append(
            f"| {p.proposal_id} | {p.symbol} "
            f"| {p.created_at_utc.strftime('%Y-%m-%d %H:%M')} "
            f"| {_cell_date_price(r.entry_date, r.entry_price)} "
            f"| {p.stop_loss_price:.2f} | {tp_cell} "
            f"| {_cell_date_price(r.exit_date, r.exit_price)} "
            f"| {_reason_cell(r)} | {_pnl_cell(r.pnl_usd)} "
            f"| {_fmt_pct(r.pnl_pct)} "
            f"| {_cell_date_price(t.exit_date, t.exit_price) if t else '-'} "
            f"| {_reason_cell(t) if t else '-'} "
            f"| {_pnl_cell(t.pnl_usd) if t else '-'} |"
        )
    lines.append("")

    if skipped or bar_failures:
        lines.append("## Skipped proposals")
        lines.append("")
        for s in skipped:
            lines.append(f"- proposal {s.proposal_id} ({s.symbol or '?'}): {s.reason}")
        for symbol, reason in sorted(bar_failures.items()):
            lines.append(f"- all proposals for {symbol}: bars unavailable — {reason}")
        lines.append("")

    lines.append("---")
    lines.append(
        "*Caveats: this is a fill-realism-limited estimate (ADR-001) of signal "
        "quality, not of realized system performance. It deliberately excludes "
        "the live exit layer's behavior; the gap between this curve and "
        "broker-reconstructed realized P&L is the mechanical tax (strategic "
        "assessment §5).*"
    )
    lines.append("")
    return "\n".join(lines)


def _marked_total(results: list[TradeResult]) -> float:
    return sum(
        r.pnl_usd or 0.0 for r in results if r.status in (STATUS_CLOSED, STATUS_OPEN)
    )


def _cell_date_price(date: dt.date | None, price: float | None) -> str:
    if date is None or price is None:
        return "-"
    return f"{date.isoformat()} @ {price:.2f}"


def _reason_cell(r: TradeResult) -> str:
    if r.exit_reason is None:
        return r.status
    return r.exit_reason if r.status == STATUS_CLOSED else f"{r.exit_reason} (open)"


def _pnl_cell(v: float | None) -> str:
    return "-" if v is None else f"{v:+,.2f}"


def _fmt_pct(v: float | None) -> str:
    return "-" if v is None else f"{v * 100:+.1f}%"

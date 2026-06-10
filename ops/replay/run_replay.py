"""CLI entry point for the retrospective clean-execution replay (D-080).

Usage (from the repo root — see ops/replay/RUNBOOK.md for the full runbook):

    python3 ops/replay/run_replay.py --journal /var/lib/quinn/journal.db \
        --out /tmp/replay --cache /tmp/replay-bars

This tool is read-only everywhere it matters: the journal is opened with
SQLite mode=ro, the only network egress is the Alpaca market-data API, and
no trading client exists anywhere under ops/replay/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ops.replay.bars import BarsClient, BarsUnavailable  # noqa: E402
from ops.replay.extract import load_proposals  # noqa: E402
from ops.replay.models import (  # noqa: E402
    ReplayProposal,
    SimConfig,
    TradeResult,
)
from ops.replay.report import (  # noqa: E402
    compute_aggregates,
    render_markdown,
    write_equity_csv,
    write_trades_csv,
)
from ops.replay.sim import simulate_trade  # noqa: E402

from config.calendar import is_market_open_day  # noqa: E402  (src/, read-only)

# Generous fetch tail past the last proposal: max horizon (60d per schema)
# plus calendar slack so the time-stop bar always exists when it can.
_FETCH_TAIL_DAYS = 75


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--journal", required=True, help="Path to journal.db (opened read-only)")
    parser.add_argument("--out", default="replay-out", help="Output directory")
    parser.add_argument("--cache", default="replay-bars-cache", help="Bars cache directory")
    parser.add_argument(
        "--notional", type=float, default=1000.0, help="USD per trade (default 1000)"
    )
    parser.add_argument(
        "--default-horizon",
        type=int,
        default=30,
        help=(
            "Days when proposal lacks one (default 30; values above "
            f"{_FETCH_TAIL_DAYS} are clamped — the bars fetch window ends "
            f"{_FETCH_TAIL_DAYS} days past the last proposal)"
        ),
    )
    parser.add_argument(
        "--trailing-pct", type=float, default=0.15, help="Counterfactual trail pct (default 0.15)"
    )
    parser.add_argument(
        "--feed", choices=("sip", "iex"), default="sip", help="Alpaca data feed (default sip)"
    )
    parser.add_argument(
        "--offline", action="store_true", help="Serve bars from cache only; no network"
    )
    parser.add_argument("--symbol", help="Replay only this symbol (debugging aid)")
    args = parser.parse_args(argv)
    if args.default_horizon > _FETCH_TAIL_DAYS:
        print(
            f"WARN --default-horizon {args.default_horizon} exceeds the bars fetch "
            f"window; clamped to {_FETCH_TAIL_DAYS}",
            file=sys.stderr,
        )
        args.default_horizon = _FETCH_TAIL_DAYS
    return args


def _simulate_all(
    proposals: list[ReplayProposal],
    client: BarsClient,
    config_tp: SimConfig,
    config_trailing: SimConfig,
) -> tuple[list[TradeResult], list[TradeResult], dict[str, str]]:
    by_symbol: dict[str, list[ReplayProposal]] = {}
    for p in proposals:
        by_symbol.setdefault(p.symbol, []).append(p)

    results_tp: list[TradeResult] = []
    results_trailing: list[TradeResult] = []
    bar_failures: dict[str, str] = {}
    for symbol, group in sorted(by_symbol.items()):
        start = min(p.created_at_utc for p in group).date() - dt.timedelta(days=1)
        end = max(p.created_at_utc for p in group).date() + dt.timedelta(days=_FETCH_TAIL_DAYS)
        end = min(end, dt.datetime.now(dt.UTC).date())
        try:
            bars, splits = client.get(symbol, start, end)
        except BarsUnavailable as e:
            bar_failures[symbol] = str(e)
            continue
        for p in group:
            results_tp.append(simulate_trade(p, bars, splits, config_tp, is_market_open_day))
            results_trailing.append(
                simulate_trade(p, bars, splits, config_trailing, is_market_open_day)
            )
    return results_tp, results_trailing, bar_failures


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    proposals, skipped = load_proposals(args.journal)
    if args.symbol:
        proposals = [p for p in proposals if p.symbol == args.symbol]
        skipped = [s for s in skipped if s.symbol == args.symbol]
    print(f"Loaded {len(proposals)} replayable proposals ({len(skipped)} skipped) from journal")
    if not proposals and not skipped:
        print("Nothing to replay — journal has no trade proposals.", file=sys.stderr)
        return 1

    config_tp = SimConfig(
        notional_usd=args.notional,
        default_horizon_days=args.default_horizon,
        exit_mode="proposal_tp",
        trailing_pct=args.trailing_pct,
    )
    config_trailing = SimConfig(
        notional_usd=args.notional,
        default_horizon_days=args.default_horizon,
        exit_mode="trailing",
        trailing_pct=args.trailing_pct,
    )
    client = BarsClient(cache_dir=args.cache, feed=args.feed, offline=args.offline)
    results_tp, results_trailing, bar_failures = _simulate_all(
        proposals, client, config_tp, config_trailing
    )
    for symbol, reason in sorted(bar_failures.items()):
        print(f"WARN bars unavailable for {symbol}: {reason}", file=sys.stderr)

    write_trades_csv(out_dir / "trades_proposal_tp.csv", results_tp)
    write_trades_csv(out_dir / "trades_trailing.csv", results_trailing)
    agg_tp = compute_aggregates(results_tp)
    agg_trailing = compute_aggregates(results_trailing)
    write_equity_csv(out_dir / "equity_proposal_tp.csv", agg_tp)
    write_equity_csv(out_dir / "equity_trailing.csv", agg_trailing)

    bar_source = (
        f"local cache only ({args.cache})"
        if args.offline
        else f"Alpaca market-data API (feed={args.feed}, raw adjustment), cached in {args.cache}"
    )
    report_md = render_markdown(
        journal_path=args.journal,
        config=config_tp,
        results_tp=results_tp,
        results_trailing=results_trailing,
        skipped=skipped,
        bar_failures=bar_failures,
        bar_source=bar_source,
    )
    (out_dir / "report.md").write_text(report_md)

    print(
        f"Closed trades: {agg_tp.n_closed}  open: {agg_tp.n_open}  "
        f"no-data: {agg_tp.n_no_data}"
    )
    if agg_tp.win_rate is not None:
        print(
            f"Proposal-TP: win rate {agg_tp.win_rate * 100:.1f}%, "
            f"expectancy ${agg_tp.expectancy_usd:.2f}/trade, "
            f"total ${agg_tp.total_pnl_usd:,.2f}"
        )
    if agg_trailing.win_rate is not None:
        print(
            f"Trailing-15%: win rate {agg_trailing.win_rate * 100:.1f}%, "
            f"expectancy ${agg_trailing.expectancy_usd:.2f}/trade, "
            f"total ${agg_trailing.total_pnl_usd:,.2f}"
        )
    print(f"Report: {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

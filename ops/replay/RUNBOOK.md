# Runbook — retrospective clean-execution replay (D-080)

Replays every journaled trade proposal under honest simulated execution and
reports what the LLM's picks would have returned, uncontaminated by the
broken live exit layer. Offline analysis only: the tool **cannot place
orders** (no trading client exists under `ops/replay/`; the only network
egress is `data.alpaca.markets`), and the journal is opened read-only.

## Prerequisites

- Python 3.11+ and the repo checked out (no extra dependencies — stdlib only,
  plus `src/config/calendar.py` for the NYSE calendar).
- Alpaca **data** credentials in the environment for the first (fetching)
  run. Only these two variables are read — `ALPACA_ENDPOINT` is ignored:

  ```sh
  export ALPACA_API_KEY_ID=...
  export ALPACA_API_SECRET_KEY=...
  ```

- A copy of the journal. Run against a snapshot, not the live file, to avoid
  contending with the agent's SQLite locks:

  ```sh
  # on the droplet
  sqlite3 /var/lib/quinn/journal.db ".backup /tmp/journal-replay.db"
  ```

## Run (on the droplet)

```sh
cd /opt/quinn && python3 ops/replay/run_replay.py --journal /tmp/journal-replay.db --out /tmp/replay-out --cache /tmp/replay-bars
```

Expected output (counts and dollar figures vary with journal contents):

```
Loaded 3 replayable proposals (0 skipped) from journal
Closed trades: 3  open: 0  no-data: 0
Proposal-TP: win rate 66.7%, expectancy $33.33/trade, total $100.00
Trailing-15%: win rate 50.0%, expectancy $0.00/trade, total $0.00
Report: /tmp/replay-out/report.md
```

`WARN bars unavailable for XYZ: ...` lines on stderr are non-fatal — see
Failure modes below.

To run locally instead, `scp` the journal snapshot down and point
`--journal` at it; everything else is identical.

## Outputs (in `--out`)

| file | contents |
|---|---|
| `report.md` | Assumptions header, aggregate tables (overall + pre/post 2026-05-08), H3 trailing-stop counterfactual with TP-capping cost, per-trade table, skip list |
| `trades_proposal_tp.csv` | Per-trade results, proposal geometry honored |
| `trades_trailing.csv` | Per-trade results, TP removed / 15% trailing stop |
| `equity_proposal_tp.csv`, `equity_trailing.csv` | Cumulative-P&L curves by exit date |

## Useful flags

| flag | default | notes |
|---|---|---|
| `--notional` | `1000` | Flat USD per trade |
| `--default-horizon` | `30` | Days, for proposals without `time_horizon_days` (max 75 — larger values are clamped to the bars fetch window, with a warning) |
| `--trailing-pct` | `0.15` | Counterfactual trailing-stop percent |
| `--feed` | `sip` | Use `--feed iex` if the API returns HTTP 403 (free-plan data subscription) |
| `--offline` | off | Serve bars purely from `--cache`; no network, no creds needed |
| `--symbol XYZ` | — | Replay a single symbol (debugging) |

## Re-runs and cache

Bars and split events are cached as one JSON file per symbol under
`--cache`. A second run with the same (or narrower) date range **and the
same `--feed`** does no network I/O; after the first run you can add
`--offline`. A cache written from a different feed is ignored and refetched
(or reported unavailable under `--offline`). Delete the cache directory to
force a refetch.

## Interpreting the report

- Read the **Simulation assumptions** header first — every result is
  conditional on those rules (next-open entry, stop-first same-day touches,
  zero slippage/fees, no liquidity modeling, $1k independent bets).
- Section A is the best available estimate of the signal's edge under the
  proposals' own geometry. Section B (H3 counterfactual) re-runs the same
  trades with TP removed and a trailing stop; the stated **TP-capping cost**
  is section B total minus section A total. Prefer the
  marked-to-market comparison line when the two modes leave different
  numbers of trades open — trailing mode systematically leaves more winners
  running past the last bar.
- `open` rows are proposals whose horizon extends past the last bar; they
  are marked to market and excluded from the closed-trade statistics.
- This estimates *signal* quality, not realized system performance; the gap
  against broker-reconstructed realized P&L is the mechanical tax
  (strategic-assessment §5).

## Failure modes

| symptom | cause / fix |
|---|---|
| `ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY not set` | Export the two data creds, or use `--offline` with a populated cache |
| `HTTP 403` on bars | Data feed not in plan — retry with `--feed iex` |
| `WARN bars unavailable for XYZ` | Symbol delisted/renamed at the data API; its proposals appear in the report's skip section, everything else proceeds |
| `--offline and cache miss` | The cache doesn't cover the symbol/range, or was written from a different `--feed`; run once without `--offline` |
